from __future__ import annotations

import argparse
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, as_completed, wait
from dataclasses import dataclass
from pathlib import Path
import random
import shutil
import sys
import time

from .discovery import find_root_pdfs
from .ollama_client import LLMClient, LLMError
from .pdf_extract import PagePayload, PageTextPayload, extract_page_text_payloads, is_native_text_usable, render_selected_pages_to_png
from .smart_trigger import has_diagram_keywords, has_math_indicators, has_table_keyword, should_call_vision_ocr
from .tui import OcrPipelineTui
from .transform import PageAnalysis
from .writer import write_document_markdown


@dataclass(slots=True)
class ProcessResult:
    output_path: Path
    total_slides: int
    ocr_queue_slides: int


def _is_endpoint_unreachable(error: Exception) -> bool:
    text = str(error).lower()
    markers = (
        "connection refused",
        "connecterror",
        "all connection attempts failed",
        "failed to establish a new connection",
        "name or service not known",
        "temporary failure in name resolution",
        "connection reset",
        "network is unreachable",
    )
    return any(marker in text for marker in markers)


def _ensure_backend_online(client: LLMClient, launch_if_offline: str, *, dry_run: bool) -> bool:
    if dry_run:
        return True

    online, reason = client.is_server_online()
    if online:
        return True

    print(f"  ! OCR backend is offline: {reason}", file=sys.stderr)
    if launch_if_offline == "y":
        print("  - Attempting to launch local backend service...", file=sys.stderr)
        started, launch_error = client.launch_local_server()
        if started:
            online_after_start, _ = client.is_server_online(timeout_seconds=1.0)
            if online_after_start:
                print("  - Backend is online.", file=sys.stderr)
                return True
        if launch_error:
            print(f"  ! Auto-launch failed for configured URL: {launch_error}", file=sys.stderr)

        ollama_fallback_url = "http://localhost:11434"
        if client.base_url.rstrip("/") != ollama_fallback_url:
            fallback_client = LLMClient(base_url=ollama_fallback_url, model=client.model)
            try:
                print(
                    f"  - Trying local Ollama fallback at {ollama_fallback_url}...",
                    file=sys.stderr,
                )
                fallback_started, fallback_error = fallback_client.launch_local_server()
                if fallback_started:
                    fallback_online, _ = fallback_client.is_server_online(timeout_seconds=1.0)
                    if fallback_online:
                        client.base_url = ollama_fallback_url
                        print(
                            f"  - Switched backend URL to {ollama_fallback_url}",
                            file=sys.stderr,
                        )
                        return True
                if fallback_error:
                    print(f"  ! Ollama fallback launch failed: {fallback_error}", file=sys.stderr)
            finally:
                fallback_client.close()

    print(
        "  ! Start your LLM server and retry. "
        "Use --launch-if-offline y to auto-launch local Ollama when possible.",
        file=sys.stderr,
    )
    return False


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="pdf-ocr-md",
        description=(
            "Find PDFs in the current directory root, run OCR/transcription with Ollama, "
            "and write one clean markdown file per PDF."
        ),
    )
    parser.add_argument("--model", default="qwen3.5:9b", help="Model name (default: qwen3.5:9b)")
    parser.add_argument(
        "--llm-url",
        default="http://localhost:1234",
        help="Base URL for LM Studio server",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=150,
        help="Render DPI for page images sent to vision model",
    )
    parser.add_argument(
        "--min-native-chars",
        type=int,
        default=80,
        help="Minimum native text characters before considering it strong context",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Optional output folder for markdown files (default: same as PDF)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Discover and parse PDFs without calling Ollama or writing files",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=2,
        help="Number of parallel page OCR requests per PDF",
    )
    parser.add_argument(
        "--request-timeout-seconds",
        type=float,
        default=240.0,
        help="Maximum seconds to wait for a single OCR/cleanup LLM request",
    )
    parser.add_argument(
        "--page-timeout-seconds",
        type=float,
        default=180.0,
        help="Hard wall-clock timeout per page task in OCR batches",
    )
    parser.add_argument(
        "--native-fast-path",
        action="store_true",
        help="Skip Ollama OCR on pages with strong native PDF text",
    )
    parser.add_argument(
        "--skip-aggregate-cleanup",
        action="store_true",
        help="Skip final aggregate cleanup call to Ollama for faster runs",
    )
    parser.add_argument(
        "--no-tui",
        action="store_true",
        help="Disable terminal progress UI",
    )
    parser.add_argument(
        "--pdf",
        action="append",
        default=[],
        help="Process only selected PDF filename(s) in current root (repeat flag for multiple)",
    )
    parser.add_argument(
        "--max-image-dim",
        type=int,
        default=1536,
        help="Cap longest image dimension before sending to LLM (0 = no cap)",
    )
    parser.add_argument(
        "--launch-if-offline",
        choices=["y", "n"],
        default="n",
        help="If backend is offline, auto-launch local service when set to 'y' (default: n)",
    )
    parser.add_argument(
        "--lauch-if-offline",
        dest="launch_if_offline",
        choices=["y", "n"],
        help=argparse.SUPPRESS,
    )
    return parser.parse_args(argv)


def process_pdf(
    pdf_path: Path,
    client: LLMClient,
    dpi: int,
    min_native_chars: int,
    workers: int,
    page_timeout_seconds: float,
    native_fast_path: bool,
    skip_aggregate_cleanup: bool,
    output_dir: Path | None,
    dry_run: bool,
    tui: OcrPipelineTui | None = None,
    max_image_dim: int = 0,
    prefetched_text_payloads: list[PageTextPayload] | None = None,
) -> ProcessResult | None:
    process_start = time.perf_counter()
    extraction_start = time.perf_counter()
    text_payloads = prefetched_text_payloads or extract_page_text_payloads(pdf_path)
    extraction_seconds = time.perf_counter() - extraction_start

    if tui is not None:
        tui.extend_global_slides(len(text_payloads))

    if dry_run:
        print(f"[DRY RUN] {pdf_path.name}: {len(text_payloads)} pages discovered")
        return None

    print(
        f"  - Extracted {len(text_payloads)} slide(s) in {extraction_seconds:.2f}s"
    )

    analyses_by_page: dict[int, PageAnalysis] = {}
    finalized_all_pages: set[int] = set()
    finalized_ocr_pages: set[int] = set()
    ocr_seconds_total = 0.0
    retry_wait_seconds = 0.0

    def _finalize_page(payload: PageTextPayload | PagePayload, analysis: PageAnalysis, *, from_ocr: bool) -> None:
        analyses_by_page[payload.page_number] = analysis
        if payload.page_number not in finalized_all_pages:
            finalized_all_pages.add(payload.page_number)
            if tui is not None:
                tui.advance_pdf_all(pdf_path.name)
                tui.advance_global_slides()
        if from_ocr and payload.page_number not in finalized_ocr_pages:
            finalized_ocr_pages.add(payload.page_number)
            if tui is not None:
                tui.advance_pdf_ocr(pdf_path.name)

    def _build_native_only_analysis(payload: PageTextPayload | PagePayload) -> PageAnalysis:
        retranscribed = payload.native_text.strip() or "(No native text detected.)"
        return PageAnalysis(
            page_number=payload.page_number,
            retranscribed_text=retranscribed,
            image_descriptions=["Used native PDF text fast path (vision OCR skipped)."],
        )

    def _analyze_payload(payload: PagePayload) -> tuple[PageAnalysis, float]:
        native_context = payload.native_text if is_native_text_usable(payload.native_text, min_native_chars) else ""
        analyze_start = time.perf_counter()
        ocr = client.analyze_page(
            image_png=payload.image_png,
            page_number=payload.page_number,
            total_pages=payload.total_pages,
            native_text=native_context,
        )
        elapsed = time.perf_counter() - analyze_start

        retranscribed = ocr.retranscribed_text.strip() or payload.native_text.strip()
        return (
            PageAnalysis(
                page_number=payload.page_number,
                retranscribed_text=retranscribed,
                math_markdown=ocr.math_markdown,
                image_descriptions=ocr.image_descriptions,
            ),
            elapsed,
        )

    def _process_ocr_batch(batch: list[PagePayload], is_retry: bool) -> list[tuple[PagePayload, Exception]]:
        nonlocal ocr_seconds_total
        failures: list[tuple[PagePayload, Exception]] = []
        unreachable_count = 0
        first_unreachable_error: Exception | None = None
        if not batch:
            return failures

        max_workers = max(1, workers)
        if max_workers == 1:
            for payload in batch:
                try:
                    analysis, elapsed = _analyze_payload(payload)
                    ocr_seconds_total += elapsed
                    _finalize_page(payload, analysis, from_ocr=True)
                    if tui is None and is_retry:
                        print(f"  - Recovered slide {payload.page_number}/{payload.total_pages} on retry")
                except Exception as exc:
                    failures.append((payload, exc))
                    if _is_endpoint_unreachable(exc):
                        unreachable_count += 1
                        if first_unreachable_error is None:
                            first_unreachable_error = exc
                    else:
                        reason = "retry failed" if is_retry else "failed (will retry later)"
                        print(f"  ! Slide {payload.page_number}/{payload.total_pages} {reason}: {exc}", file=sys.stderr)

            if unreachable_count:
                reason = "retry failed" if is_retry else "failed (will retry later)"
                print(
                    f"  ! OCR endpoint unreachable; {unreachable_count} slide(s) {reason}. "
                    f"First error: {first_unreachable_error}",
                    file=sys.stderr,
                )
            return failures

        executor = ThreadPoolExecutor(max_workers=max_workers)
        try:
            # Sliding window
            pending_payloads = list(batch)
            # active maps future → (payload, start_time)
            active: dict = {}

            def _submit_next() -> None:
                if pending_payloads:
                    p = pending_payloads.pop(0)
                    f = executor.submit(_analyze_payload, p)
                    active[f] = (p, time.perf_counter())

            for _ in range(min(max_workers, len(pending_payloads))):
                _submit_next()

            while active:
                done, _ = wait(list(active), timeout=0.5, return_when=FIRST_COMPLETED)

                for future in done:
                    payload, _ = active.pop(future)
                    try:
                        analysis, elapsed = future.result()
                        ocr_seconds_total += elapsed
                        _finalize_page(payload, analysis, from_ocr=True)
                        if tui is None and is_retry:
                            print(f"  - Recovered slide {payload.page_number}/{payload.total_pages} on retry")
                    except Exception as exc:
                        failures.append((payload, exc))
                        if _is_endpoint_unreachable(exc):
                            unreachable_count += 1
                            if first_unreachable_error is None:
                                first_unreachable_error = exc
                        else:
                            reason = "retry failed" if is_retry else "failed (will retry later)"
                            print(f"  ! Slide {payload.page_number}/{payload.total_pages} {reason}: {exc}", file=sys.stderr)
                    # Open up a slot for the next pending task
                    _submit_next()

                now = time.perf_counter()
                for future, (payload, start_time) in list(active.items()):
                    if now - start_time > max(1.0, page_timeout_seconds):
                        active.pop(future)
                        future.cancel()
                        timeout_exc = TimeoutError(
                            f"Page processing exceeded {page_timeout_seconds:.1f}s"
                        )
                        failures.append((payload, timeout_exc))
                        reason = "retry timed out" if is_retry else "timed out (will retry later)"
                        print(
                            f"  ! Slide {payload.page_number}/{payload.total_pages} {reason}: {timeout_exc}",
                            file=sys.stderr,
                        )
                        # Slot freed by timeout — submit next pending task
                        _submit_next()
        finally:
            executor.shutdown(wait=False, cancel_futures=True)

        if unreachable_count:
            reason = "retry failed" if is_retry else "failed (will retry later)"
            print(
                f"  ! OCR endpoint unreachable; {unreachable_count} slide(s) {reason}. "
                f"First error: {first_unreachable_error}",
                file=sys.stderr,
            )

        return failures

    needs_ocr_text_payloads: list[PageTextPayload] = []
    native_fast_path_pages: list[tuple[PageTextPayload, str]] = []
    needs_visual_check: list[PageTextPayload] = []
    render_for_route_seconds = 0.0
    route_start = time.perf_counter()
    for payload in text_payloads:
        if native_fast_path:
            compact = " ".join(payload.native_text.split())
            native_len = len(compact)

            if native_len < min_native_chars:
                needs_ocr_text_payloads.append(payload)
                if tui is None:
                    print(f"  - Slide {payload.page_number}/{payload.total_pages} queued for OCR (weak_native_text)")
            elif has_math_indicators(compact):
                needs_ocr_text_payloads.append(payload)
                if tui is None:
                    print(f"  - Slide {payload.page_number}/{payload.total_pages} queued for OCR (math_detected)")
            elif has_diagram_keywords(compact):
                needs_ocr_text_payloads.append(payload)
                if tui is None:
                    print(f"  - Slide {payload.page_number}/{payload.total_pages} queued for OCR (diagram_keyword_detected)")
            elif has_table_keyword(compact) and native_len < 350:
                # slides with an actual grid/table image will be caught by
                # has_visual_structure() in the visual check stage.
                needs_visual_check.append(payload)
            elif native_len >= 350:
                native_fast_path_pages.append((payload, "native_text_only"))
            else:
                needs_visual_check.append(payload)
        else:
            needs_ocr_text_payloads.append(payload)

    # Route render uses a lower DPI — enough for the visual-structure heuristic
    # but ~4× cheaper than the full OCR render. Pages that skip OCR pay only
    # this cheap render cost; pages that need OCR get a full-quality re-render.
    route_dpi = max(90, dpi // 2)

    if needs_visual_check:
        route_render_start = time.perf_counter()
        route_images = render_selected_pages_to_png(
            pdf_path=pdf_path,
            page_numbers=[payload.page_number for payload in needs_visual_check],
            dpi=route_dpi,
        )
        render_for_route_seconds = time.perf_counter() - route_render_start

        for payload in needs_visual_check:
            image_png = route_images.get(payload.page_number)
            if image_png is None:
                raise RuntimeError(f"Missing rendered page image for route check: {payload.page_number}")
            use_vision, reason = should_call_vision_ocr(
                native_text=payload.native_text,
                image_png=image_png,
                min_native_chars=min_native_chars,
            )
            if use_vision:
                needs_ocr_text_payloads.append(payload)
                if tui is None:
                    print(f"  - Slide {payload.page_number}/{payload.total_pages} queued for OCR ({reason})")
            else:
                native_fast_path_pages.append((payload, reason))

    render_for_ocr_start = time.perf_counter()
    # OCR renders use JPEG + resolution cap for smaller LLM payloads.
    ocr_images = render_selected_pages_to_png(
        pdf_path=pdf_path,
        page_numbers=[payload.page_number for payload in needs_ocr_text_payloads],
        dpi=dpi,
        max_image_dim=max_image_dim,
        use_jpeg=True,
    )
    render_for_ocr_seconds = time.perf_counter() - render_for_ocr_start if needs_ocr_text_payloads else 0.0

    needs_ocr: list[PagePayload] = []
    for payload in needs_ocr_text_payloads:
        image_png = ocr_images.get(payload.page_number)
        if image_png is None:
            raise RuntimeError(f"Missing rendered page image for OCR: {payload.page_number}")
        needs_ocr.append(
            PagePayload(
                page_number=payload.page_number,
                total_pages=payload.total_pages,
                native_text=payload.native_text,
                image_png=image_png,
            )
        )

    route_seconds = time.perf_counter() - route_start

    if tui is not None:
        tui.start_pdf(pdf_path.name, queued_slides=len(needs_ocr), total_slides=len(text_payloads))

    if needs_ocr:
        online, reason = client.is_server_online(timeout_seconds=1.0)
        if not online:
            print(
                f"  ! OCR endpoint unavailable ({reason}). "
                f"Using fallback text for {len(needs_ocr)} slide(s).",
                file=sys.stderr,
            )
            for payload in needs_ocr:
                fallback_text = payload.native_text.strip() or "(Slide OCR failed and no native PDF text was available.)"
                _finalize_page(
                    payload,
                    PageAnalysis(
                        page_number=payload.page_number,
                        retranscribed_text=fallback_text,
                        image_descriptions=[
                            "OCR backend was unavailable; used native fallback text.",
                        ],
                    ),
                    from_ocr=True,
                )
            needs_ocr = []
            skip_aggregate_cleanup = True

    for payload, reason in native_fast_path_pages:
        _finalize_page(payload, _build_native_only_analysis(payload), from_ocr=False)
        if tui is None:
            print(
                f"  - Processed slide {payload.page_number}/{payload.total_pages} via native fast path ({reason})"
            )

    first_failures = _process_ocr_batch(needs_ocr, is_retry=False)
    first_error_by_page = {payload.page_number: error for payload, error in first_failures}

    def _is_transient_error(error: Exception) -> bool:
        text = str(error).lower()
        transient_markers = (
            "timeout",
            "timed out",
            "temporary",
            "connection",
            "refused",
            "reset",
            "overload",
            "busy",
            "rate limit",
            "429",
            "502",
            "503",
            "504",
        )
        return any(marker in text for marker in transient_markers)

    retry_candidates: list[PagePayload] = []
    non_retry_failures: list[tuple[PagePayload, Exception]] = []
    for payload, error in first_failures:
        if _is_transient_error(error):
            retry_candidates.append(payload)
        else:
            non_retry_failures.append((payload, error))

    for payload, first_error in non_retry_failures:
        if payload.page_number not in analyses_by_page:
            fallback_text = payload.native_text.strip() or "(Slide OCR failed and no native PDF text was available.)"
            _finalize_page(
                payload,
                PageAnalysis(
                    page_number=payload.page_number,
                    retranscribed_text=fallback_text,
                    image_descriptions=[
                        f"OCR failed; retry skipped (non-transient). Error: {first_error}",
                    ],
                ),
                from_ocr=True,
            )

    if retry_candidates:
        delay_seconds = min(3.0, 0.8 + random.uniform(0.0, 0.5))
        retry_wait_seconds += delay_seconds
        print(f"  - Retrying {len(retry_candidates)} transient failure(s) after {delay_seconds:.2f}s backoff")
        time.sleep(delay_seconds)

    retry_failures = _process_ocr_batch(retry_candidates, is_retry=True)

    retry_fallback_count = 0
    for payload, retry_error in retry_failures:
        first_error = first_error_by_page.get(payload.page_number)
        if payload.page_number not in analyses_by_page:
            fallback_text = payload.native_text.strip() or "(Slide OCR failed and no native PDF text was available.)"
            _finalize_page(
                payload,
                PageAnalysis(
                    page_number=payload.page_number,
                    retranscribed_text=fallback_text,
                    image_descriptions=[
                        f"OCR failed after retry. First error: {first_error}",
                        f"Retry error: {retry_error}",
                    ],
                ),
                from_ocr=True,
            )
            retry_fallback_count += 1

    if retry_fallback_count:
        print(
            f"  ! OCR retry exhausted; used fallback text for {retry_fallback_count} slide(s).",
            file=sys.stderr,
        )

    analyses = [analyses_by_page[page_number] for page_number in sorted(analyses_by_page)]

    # Auto-skip cleanup when no slides went through OCR at all — the cleanup
    # LLM pass is only useful for normalizing OCR-introduced inconsistencies.
    if not skip_aggregate_cleanup and len(needs_ocr) == 0:
        skip_aggregate_cleanup = True

    if skip_aggregate_cleanup:
        cleaned_aggregate = None
        cleanup_seconds = 0.0
    else:
        cleanup_start = time.perf_counter()
        try:
            cleaned_aggregate = client.clean_aggregate_markdown([item.retranscribed_text for item in analyses])
        except Exception as exc:
            cleaned_aggregate = None
            print(
                f"  ! Aggregate cleanup failed for {pdf_path.name}; using fallback aggregate. Reason: {exc}",
                file=sys.stderr,
            )
        cleanup_seconds = time.perf_counter() - cleanup_start

    if tui is not None:
        tui.finish_pdf(pdf_path.name)

    write_start = time.perf_counter()
    output_path = write_document_markdown(
        pdf_path=pdf_path,
        page_analyses=analyses,
        cleaned_aggregate=cleaned_aggregate,
        output_dir=output_dir,
    )
    write_seconds = time.perf_counter() - write_start
    total_seconds = time.perf_counter() - process_start
    print(
        "  - Timings: "
        f"extract={extraction_seconds:.2f}s, "
        f"route={route_seconds:.2f}s, "
        f"route_render={render_for_route_seconds:.2f}s, "
        f"ocr_render={render_for_ocr_seconds:.2f}s, "
        f"ocr={ocr_seconds_total:.2f}s, "
        f"retry_wait={retry_wait_seconds:.2f}s, "
        f"cleanup={cleanup_seconds:.2f}s, "
        f"write={write_seconds:.2f}s, "
        f"total={total_seconds:.2f}s"
    )
    return ProcessResult(
        output_path=output_path,
        total_slides=len(text_payloads),
        ocr_queue_slides=len(needs_ocr),
    )


def _resolve_unique_target(target_path: Path) -> Path:
    if not target_path.exists():
        return target_path

    stem = target_path.stem
    suffix = target_path.suffix
    parent = target_path.parent

    counter = 1
    while True:
        candidate = parent / f"{stem}-{counter}{suffix}"
        if not candidate.exists():
            return candidate
        counter += 1


def move_pdf_to_processed(pdf_path: Path, processed_dir: Path) -> Path:
    processed_dir.mkdir(parents=True, exist_ok=True)
    target = _resolve_unique_target(processed_dir / pdf_path.name)
    shutil.move(str(pdf_path), str(target))
    return target


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    root = Path.cwd()
    output_dir = args.output_dir if args.output_dir is not None else root / "markdown"
    processed_dir = root / "processed"
    pdf_files = find_root_pdfs(root)

    if not pdf_files:
        print("No PDF files found in the current directory root.")
        return 0

    if args.pdf:
        selected_names = {name.strip() for name in args.pdf if name.strip()}
        pdf_files = [pdf_file for pdf_file in pdf_files if pdf_file.name in selected_names]
        missing = sorted(selected_names - {pdf_file.name for pdf_file in pdf_files})
        for name in missing:
            print(f"Warning: selected PDF not found in root: {name}", file=sys.stderr)
        if not pdf_files:
            print("No selected PDF files found in the current directory root.")
            return 1

    print(f"Found {len(pdf_files)} PDF(s) in {root}")

    client = LLMClient(
        base_url=args.llm_url,
        model=args.model,
        timeout_seconds=max(1.0, float(args.request_timeout_seconds)),
    )
    if not _ensure_backend_online(client, args.launch_if_offline, dry_run=args.dry_run):
        client.close()
        return 2

    model_ok, model_info = client.ensure_model_selected()
    if not model_ok:
        print(
            "  ! Backend requires a model, but none is configured. "
            "Pass --model <name> or configure a default model in your backend.",
            file=sys.stderr,
        )
        if model_info:
            print(f"  ! Details: {model_info}", file=sys.stderr)
        client.close()
        return 2
    if not args.model.strip() and model_info:
        print(f"  - Auto-selected model: {model_info}")

    exit_code = 0
    documents_succeeded = 0
    slides_succeeded = 0
    ocr_queue_succeeded = 0

    try:
        with OcrPipelineTui(enabled=(not args.no_tui and not args.dry_run)) as tui:
            tui.start_documents(len(pdf_files))
            tui.start_global_slides(0)
            # Pipeline: pre-extract text payloads for the next PDF while
            # the current one is processing (CPU extraction overlaps GPU OCR).
            prefetch_executor = ThreadPoolExecutor(max_workers=1)
            prefetched: dict[Path, list[PageTextPayload]] = {}
            for idx, pdf_file in enumerate(pdf_files):
                # Kick off prefetch for the next PDF (if any).
                next_pdf = pdf_files[idx + 1] if idx + 1 < len(pdf_files) else None
                prefetch_future = None
                if next_pdf is not None:
                    prefetch_future = prefetch_executor.submit(extract_page_text_payloads, next_pdf)

                print(f"Processing: {pdf_file.name}")
                try:
                    output = process_pdf(
                        pdf_path=pdf_file,
                        client=client,
                        dpi=args.dpi,
                        min_native_chars=args.min_native_chars,
                        workers=args.workers,
                        page_timeout_seconds=max(1.0, float(args.page_timeout_seconds)),
                        native_fast_path=args.native_fast_path,
                        skip_aggregate_cleanup=args.skip_aggregate_cleanup,
                        output_dir=output_dir,
                        dry_run=args.dry_run,
                        tui=tui,
                        max_image_dim=args.max_image_dim,
                        prefetched_text_payloads=prefetched.pop(pdf_file, None),
                    )
                    if output is not None:
                        documents_succeeded += 1
                        slides_succeeded += output.total_slides
                        ocr_queue_succeeded += output.ocr_queue_slides
                        print(f"  -> Wrote {output.output_path}")
                        moved_to = move_pdf_to_processed(pdf_file, processed_dir)
                        print(f"  -> Moved source PDF to {moved_to}")
                except Exception as exc:
                    exit_code = 1
                    print(f"  ! Failed {pdf_file.name}: {exc}", file=sys.stderr)
                finally:
                    tui.finish_document()
                    # Collect the prefetched result for the next iteration.
                    if prefetch_future is not None and next_pdf is not None:
                        try:
                            prefetched[next_pdf] = prefetch_future.result()
                        except Exception:
                            pass  # Will fall back to extraction inside process_pdf
            prefetch_executor.shutdown(wait=False)
    except LLMError as exc:
        print(f"LLM error: {exc}", file=sys.stderr)
        return 2
    finally:
        client.close()

    if args.no_tui and not args.dry_run:
        print(
            "Summary: "
            f"Documents {documents_succeeded}/{len(pdf_files)} | "
            f"Slides processed {slides_succeeded} | "
            f"OCR queue slides {ocr_queue_succeeded}"
        )

    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())

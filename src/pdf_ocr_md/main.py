from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
import shutil
import sys

from .discovery import find_root_pdfs
from .ollama_client import LLMClient, LLMError
from .pdf_extract import PagePayload, extract_page_payloads, get_pdf_page_count, is_native_text_usable
from .smart_trigger import should_call_vision_ocr
from .tui import OcrPipelineTui
from .transform import PageAnalysis
from .writer import write_document_markdown


@dataclass(slots=True)
class ProcessResult:
    output_path: Path
    total_slides: int
    ocr_queue_slides: int


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="pdf-ocr-md",
        description=(
            "Find PDFs in the current directory root, run OCR/transcription with Ollama, "
            "and write one clean markdown file per PDF."
        ),
    )
    parser.add_argument("--model", default="", help="Model name (leave empty for LM Studio default)")
    parser.add_argument(
        "--llm-url",
        default="http://localhost:1234",
        help="Base URL for LM Studio server",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=180,
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
    return parser.parse_args(argv)


def process_pdf(
    pdf_path: Path,
    client: OllamaClient,
    dpi: int,
    min_native_chars: int,
    workers: int,
    native_fast_path: bool,
    skip_aggregate_cleanup: bool,
    output_dir: Path | None,
    dry_run: bool,
    tui: OcrPipelineTui | None = None,
) -> ProcessResult | None:
    payloads = extract_page_payloads(pdf_path, dpi=dpi)

    if dry_run:
        print(f"[DRY RUN] {pdf_path.name}: {len(payloads)} pages discovered")
        return None

    analyses_by_page: dict[int, PageAnalysis] = {}
    finalized_all_pages: set[int] = set()
    finalized_ocr_pages: set[int] = set()

    def _finalize_page(payload: PagePayload, analysis: PageAnalysis, *, from_ocr: bool) -> None:
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

    def _build_native_only_analysis(payload: PagePayload) -> PageAnalysis:
        retranscribed = payload.native_text.strip() or "(No native text detected.)"
        return PageAnalysis(
            page_number=payload.page_number,
            retranscribed_text=retranscribed,
            image_descriptions=["Used native PDF text fast path (vision OCR skipped)."],
        )

    def _analyze_payload(payload: PagePayload) -> PageAnalysis:
        native_context = payload.native_text if is_native_text_usable(payload.native_text, min_native_chars) else ""
        ocr = client.analyze_page(
            image_png=payload.image_png,
            page_number=payload.page_number,
            total_pages=payload.total_pages,
            native_text=native_context,
        )

        retranscribed = ocr.retranscribed_text.strip() or payload.native_text.strip()
        return PageAnalysis(
            page_number=payload.page_number,
            retranscribed_text=retranscribed,
            math_markdown=ocr.math_markdown,
            image_descriptions=ocr.image_descriptions,
        )

    def _process_ocr_batch(batch: list[PagePayload], is_retry: bool) -> list[tuple[PagePayload, Exception]]:
        failures: list[tuple[PagePayload, Exception]] = []
        if not batch:
            return failures

        max_workers = max(1, workers)
        if max_workers == 1:
            for payload in batch:
                try:
                    _finalize_page(payload, _analyze_payload(payload), from_ocr=True)
                    if is_retry:
                        print(f"  - Recovered slide {payload.page_number}/{payload.total_pages} on retry")
                    else:
                        print(f"  - Processed slide {payload.page_number}/{payload.total_pages}")
                except Exception as exc:
                    failures.append((payload, exc))
                    reason = "retry failed" if is_retry else "failed (will retry later)"
                    print(f"  ! Slide {payload.page_number}/{payload.total_pages} {reason}: {exc}", file=sys.stderr)
            return failures

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_map = {executor.submit(_analyze_payload, payload): payload for payload in batch}
            for future in as_completed(future_map):
                payload = future_map[future]
                try:
                    _finalize_page(payload, future.result(), from_ocr=True)
                    if is_retry:
                        print(f"  - Recovered slide {payload.page_number}/{payload.total_pages} on retry")
                    else:
                        print(f"  - Processed slide {payload.page_number}/{payload.total_pages}")
                except Exception as exc:
                    failures.append((payload, exc))
                    reason = "retry failed" if is_retry else "failed (will retry later)"
                    print(f"  ! Slide {payload.page_number}/{payload.total_pages} {reason}: {exc}", file=sys.stderr)

        return failures

    needs_ocr: list[PagePayload] = []
    native_fast_path_pages: list[tuple[PagePayload, str]] = []
    for payload in payloads:
        if native_fast_path:
            use_vision, reason = should_call_vision_ocr(
                native_text=payload.native_text,
                image_png=payload.image_png,
                min_native_chars=min_native_chars,
            )
            if use_vision:
                needs_ocr.append(payload)
                print(
                    f"  - Slide {payload.page_number}/{payload.total_pages} queued for OCR ({reason})"
                )
            else:
                native_fast_path_pages.append((payload, reason))
        else:
            needs_ocr.append(payload)

    if tui is not None:
        tui.start_pdf(pdf_path.name, queued_slides=len(needs_ocr), total_slides=len(payloads))

    for payload, reason in native_fast_path_pages:
        _finalize_page(payload, _build_native_only_analysis(payload), from_ocr=False)
        print(
            f"  - Processed slide {payload.page_number}/{payload.total_pages} via native fast path ({reason})"
        )

    first_failures = _process_ocr_batch(needs_ocr, is_retry=False)
    first_error_by_page = {payload.page_number: error for payload, error in first_failures}

    if first_failures:
        print(f"  - Retrying {len(first_failures)} failed slide(s) for {pdf_path.name}")

    retry_failures = _process_ocr_batch([payload for payload, _ in first_failures], is_retry=True)

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
        print(
            f"  ! Slide {payload.page_number}/{payload.total_pages} failed again; used fallback text.",
            file=sys.stderr,
        )

    analyses = [analyses_by_page[page_number] for page_number in sorted(analyses_by_page)]

    if skip_aggregate_cleanup:
        cleaned_aggregate = None
    else:
        try:
            cleaned_aggregate = client.clean_aggregate_markdown([item.retranscribed_text for item in analyses])
        except Exception as exc:
            cleaned_aggregate = None
            print(
                f"  ! Aggregate cleanup failed for {pdf_path.name}; using fallback aggregate. Reason: {exc}",
                file=sys.stderr,
            )

    if tui is not None:
        tui.finish_pdf(pdf_path.name)

    output_path = write_document_markdown(
        pdf_path=pdf_path,
        page_analyses=analyses,
        cleaned_aggregate=cleaned_aggregate,
        output_dir=output_dir,
    )
    return ProcessResult(
        output_path=output_path,
        total_slides=len(payloads),
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

    total_slides = 0
    for pdf_file in pdf_files:
        try:
            total_slides += get_pdf_page_count(pdf_file)
        except Exception as exc:
            print(f"Warning: could not pre-count slides in {pdf_file.name}: {exc}", file=sys.stderr)

    client = LLMClient(base_url=args.llm_url, model=args.model)
    exit_code = 0
    documents_succeeded = 0
    slides_succeeded = 0
    ocr_queue_succeeded = 0

    try:
        with OcrPipelineTui(enabled=(not args.no_tui and not args.dry_run)) as tui:
            tui.start_documents(len(pdf_files))
            tui.start_global_slides(total_slides)
            for pdf_file in pdf_files:
                print(f"Processing: {pdf_file.name}")
                try:
                    output = process_pdf(
                        pdf_path=pdf_file,
                        client=client,
                        dpi=args.dpi,
                        min_native_chars=args.min_native_chars,
                        workers=args.workers,
                        native_fast_path=args.native_fast_path,
                        skip_aggregate_cleanup=args.skip_aggregate_cleanup,
                        output_dir=output_dir,
                        dry_run=args.dry_run,
                        tui=tui,
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
    except LLMError as exc:
        print(f"LLM error: {exc}", file=sys.stderr)
        return 2
    finally:
        client.close()

    if args.no_tui and not args.dry_run:
        remaining_slides = max(total_slides - slides_succeeded, 0)
        print(
            "Summary: "
            f"Documents {documents_succeeded}/{len(pdf_files)} | "
            f"Slides {slides_succeeded}/{total_slides} | "
            f"Slides left {remaining_slides} | "
            f"OCR queue slides {ocr_queue_succeeded}"
        )

    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())

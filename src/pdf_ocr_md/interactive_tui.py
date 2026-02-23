from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import httpx
from rich.console import Console
from rich.prompt import Confirm, IntPrompt, Prompt
from rich.table import Table

from .discovery import find_root_pdfs
from .main import main as run_main


@dataclass(slots=True)
class QualityPreset:
    key: str
    label: str
    dpi: int
    workers: int
    native_fast_path: bool
    skip_aggregate_cleanup: bool


QUALITY_PRESETS: list[QualityPreset] = [
    QualityPreset(
        key="1",
        label="Fast (selective OCR, lower image quality)",
        dpi=140,
        workers=3,
        native_fast_path=True,
        skip_aggregate_cleanup=True,
    ),
    QualityPreset(
        key="2",
        label="Balanced (recommended)",
        dpi=180,
        workers=2,
        native_fast_path=True,
        skip_aggregate_cleanup=False,
    ),
    QualityPreset(
        key="3",
        label="High Quality (OCR every needed page with cleanup)",
        dpi=220,
        workers=1,
        native_fast_path=False,
        skip_aggregate_cleanup=False,
    ),
]


def _fetch_llm_models(llm_url: str) -> list[str]:
    """Fetch available models from LM Studio."""
    url = f"{llm_url.rstrip('/')}/v1/models"
    try:
        response = httpx.get(url, timeout=8.0)
        response.raise_for_status()
        payload = response.json()
        models = payload.get("data", [])
        names = [str(item.get("id", "")).strip() for item in models if str(item.get("id", "")).strip()]
        return sorted(set(names))
    except Exception:
        return []


def _render_pdf_table(console: Console, pdfs: list[Path]) -> None:
    table = Table(title="PDF Files in Current Directory")
    table.add_column("#", justify="right")
    table.add_column("Filename", overflow="fold")
    for index, pdf in enumerate(pdfs, start=1):
        table.add_row(str(index), pdf.name)
    console.print(table)


def _parse_index_selection(raw: str, max_index: int) -> list[int]:
    values: set[int] = set()
    for chunk in raw.split(","):
        item = chunk.strip()
        if not item:
            continue
        if "-" in item:
            start_raw, end_raw = item.split("-", 1)
            start = int(start_raw)
            end = int(end_raw)
            if start > end:
                start, end = end, start
            for number in range(start, end + 1):
                if 1 <= number <= max_index:
                    values.add(number)
            continue
        number = int(item)
        if 1 <= number <= max_index:
            values.add(number)
    return sorted(values)


def _ask_pdf_selection(console: Console, pdfs: list[Path]) -> list[Path]:
    _render_pdf_table(console, pdfs)
    console.print("Select PDFs by number, e.g. [bold]1,3-5[/bold], or [bold]all[/bold].")

    while True:
        choice = Prompt.ask("PDF selection", default="all").strip().lower()
        if choice == "all":
            return pdfs
        try:
            indexes = _parse_index_selection(choice, max_index=len(pdfs))
        except ValueError:
            console.print("[red]Invalid selection format.[/red]")
            continue
        if not indexes:
            console.print("[red]No valid PDF indexes selected.[/red]")
            continue
        return [pdfs[index - 1] for index in indexes]


def _ask_model(console: Console, llm_url: str) -> str:
    discovered = _fetch_llm_models(llm_url)
    default_model = ""

    if discovered:
        table = Table(title="Detected LM Studio Models")
        table.add_column("#", justify="right")
        table.add_column("Model")
        for index, model in enumerate(discovered, start=1):
            table.add_row(str(index), model)
        console.print(table)
        console.print("Pick a model number or enter a custom model name (leave empty for default).")

        raw = Prompt.ask("Model", default=discovered[0]).strip()
        if raw.isdigit():
            model_index = int(raw)
            if 1 <= model_index <= len(discovered):
                return discovered[model_index - 1]
        return raw or discovered[0]

    console.print("[yellow]Could not fetch model list from LM Studio; enter model manually (leave empty for default).[/yellow]")
    return Prompt.ask("Model", default=default_model).strip() or default_model


def _ask_quality_preset(console: Console) -> QualityPreset:
    table = Table(title="Quality Presets")
    table.add_column("Key", justify="right")
    table.add_column("Preset")
    table.add_column("DPI", justify="right")
    table.add_column("Workers", justify="right")
    table.add_column("Selective OCR")
    table.add_column("Aggregate Cleanup")

    for preset in QUALITY_PRESETS:
        table.add_row(
            preset.key,
            preset.label,
            str(preset.dpi),
            str(preset.workers),
            "Yes" if preset.native_fast_path else "No",
            "No" if preset.skip_aggregate_cleanup else "Yes",
        )

    console.print(table)
    while True:
        key = Prompt.ask("Choose quality preset", default="2").strip()
        for preset in QUALITY_PRESETS:
            if preset.key == key:
                return preset
        console.print("[red]Invalid preset. Choose 1, 2, or 3.[/red]")


def _build_argv(
    selected_pdfs: Iterable[Path],
    model: str,
    llm_url: str,
    preset: QualityPreset,
    min_native_chars: int,
    output_dir: str,
    no_tui_progress: bool,
    dry_run: bool,
) -> list[str]:
    args: list[str] = [
        "--model",
        model,
        "--llm-url",
        llm_url,
        "--dpi",
        str(preset.dpi),
        "--workers",
        str(preset.workers),
        "--min-native-chars",
        str(min_native_chars),
        "--output-dir",
        output_dir,
    ]

    if preset.native_fast_path:
        args.append("--native-fast-path")
    if preset.skip_aggregate_cleanup:
        args.append("--skip-aggregate-cleanup")
    if no_tui_progress:
        args.append("--no-tui")
    if dry_run:
        args.append("--dry-run")

    for pdf in selected_pdfs:
        args.extend(["--pdf", pdf.name])

    return args


def main() -> int:
    console = Console()
    root = Path.cwd()
    pdfs = find_root_pdfs(root)

    if not pdfs:
        console.print("[yellow]No PDF files found in the current directory root.[/yellow]")
        return 0

    console.print("[bold cyan]PDF OCR Markdown - Interactive TUI[/bold cyan]")
    selected_pdfs = _ask_pdf_selection(console, pdfs)

    llm_url = Prompt.ask("LM Studio URL", default="http://localhost:1234").strip() or "http://localhost:1234"
    model = _ask_model(console, llm_url)
    preset = _ask_quality_preset(console)

    min_native_chars = IntPrompt.ask("Native-text threshold", default=80)
    output_dir = Prompt.ask("Markdown output folder", default="markdown").strip() or "markdown"
    dry_run = Confirm.ask("Dry run only?", default=False)

    show_progress_slider = Confirm.ask("Show OCR queue slider during run?", default=True)

    args = _build_argv(
        selected_pdfs=selected_pdfs,
        model=model,
        llm_url=llm_url,
        preset=preset,
        min_native_chars=min_native_chars,
        output_dir=output_dir,
        no_tui_progress=(not show_progress_slider),
        dry_run=dry_run,
    )

    console.print("\n[bold green]Starting pipeline...[/bold green]")
    return run_main(args)


if __name__ == "__main__":
    raise SystemExit(main())

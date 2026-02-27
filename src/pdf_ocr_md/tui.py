from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(slots=True)
class OcrTaskState:
    ocr_task_id: Any
    all_task_id: Any
    queued: int
    total: int


class OcrPipelineTui:
    def __init__(self, enabled: bool = True) -> None:
        self.enabled = enabled
        self.progress: Any | None = None
        self._task_by_pdf: dict[str, OcrTaskState] = {}
        self._documents_task_id: Any | None = None
        self._global_slides_task_id: Any | None = None

    def __enter__(self) -> "OcrPipelineTui":
        if self.enabled:
            try:
                from rich.console import Console
                from rich.progress import (
                    BarColumn,
                    Progress,
                    SpinnerColumn,
                    TaskProgressColumn,
                    TextColumn,
                    TimeElapsedColumn,
                    TimeRemainingColumn,
                )

                console = Console()
                self.progress = Progress(
                    SpinnerColumn(),
                    TextColumn("[bold cyan]{task.description}"),
                    BarColumn(bar_width=40),
                    TaskProgressColumn(),
                    TextColumn("done {task.fields[done]}/{task.fields[queued]}"),
                    TextColumn("left {task.fields[left]}"),
                    TimeElapsedColumn(),
                    TimeRemainingColumn(),
                    console=console,
                    transient=False,
                )
                self.progress.start()
            except Exception:
                self.progress = None
                self.enabled = False
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        if self.progress is not None:
            self.progress.stop()

    def start_documents(self, total_documents: int) -> None:
        if not self.enabled or self.progress is None:
            return
        total = max(1, total_documents)
        self._documents_task_id = self.progress.add_task(
            "Documents",
            total=total,
            completed=0,
            done=0,
            queued=total_documents,
            left=total_documents,
        )

    def start_global_slides(self, total_slides: int) -> None:
        if not self.enabled or self.progress is None:
            return

        total = max(1, total_slides)
        self._global_slides_task_id = self.progress.add_task(
            "Global slides",
            total=total,
            completed=0,
            done=0,
            queued=total_slides,
            left=total_slides,
        )

    def extend_global_slides(self, additional_slides: int) -> None:
        if not self.enabled or self.progress is None or self._global_slides_task_id is None:
            return
        if additional_slides <= 0:
            return

        task = self.progress.tasks[self._global_slides_task_id]
        queued = int(task.fields.get("queued", 0)) + additional_slides
        done = int(task.fields.get("done", 0))
        left = max(queued - done, 0)
        self.progress.update(
            self._global_slides_task_id,
            total=max(1, queued),
            queued=queued,
            left=left,
        )

    def advance_global_slides(self, step: int = 1) -> None:
        if not self.enabled or self.progress is None or self._global_slides_task_id is None:
            return

        task = self.progress.tasks[self._global_slides_task_id]
        queued = int(task.fields.get("queued", 0))
        done = min(int(task.fields.get("done", 0)) + step, queued)
        left = max(queued - done, 0)
        self.progress.update(
            self._global_slides_task_id,
            advance=step,
            done=done,
            left=left,
        )

    def finish_document(self) -> None:
        if not self.enabled or self.progress is None or self._documents_task_id is None:
            return

        task = self.progress.tasks[self._documents_task_id]
        done = min(int(task.fields.get("done", 0)) + 1, int(task.fields.get("queued", 0)))
        queued = int(task.fields.get("queued", 0))
        left = max(queued - done, 0)
        self.progress.update(
            self._documents_task_id,
            advance=1,
            done=done,
            left=left,
        )

    def start_pdf(self, pdf_name: str, queued_slides: int, total_slides: int) -> None:
        if not self.enabled or self.progress is None:
            return

        ocr_description = f"{pdf_name} OCR queue"
        ocr_task_id = self.progress.add_task(
            ocr_description,
            total=max(1, queued_slides),
            completed=0,
            done=0,
            queued=queued_slides,
            left=queued_slides,
        )
        all_description = f"{pdf_name} all slides"
        all_task_id = self.progress.add_task(
            all_description,
            total=max(1, total_slides),
            completed=0,
            done=0,
            queued=total_slides,
            left=total_slides,
        )
        self._task_by_pdf[pdf_name] = OcrTaskState(
            ocr_task_id=ocr_task_id,
            all_task_id=all_task_id,
            queued=queued_slides,
            total=total_slides,
        )

    def advance_pdf_ocr(self, pdf_name: str, step: int = 1) -> None:
        if not self.enabled or self.progress is None:
            return
        state = self._task_by_pdf.get(pdf_name)
        if not state:
            return
        self.progress.advance(state.ocr_task_id, step)
        task = self.progress.tasks[state.ocr_task_id]
        done = min(int(task.fields.get("done", 0)) + step, state.queued)
        left = max(state.queued - done, 0)
        self.progress.update(state.ocr_task_id, done=done, left=left)

    def advance_pdf_all(self, pdf_name: str, step: int = 1) -> None:
        if not self.enabled or self.progress is None:
            return
        state = self._task_by_pdf.get(pdf_name)
        if not state:
            return
        self.progress.advance(state.all_task_id, step)
        task = self.progress.tasks[state.all_task_id]
        done = min(int(task.fields.get("done", 0)) + step, state.total)
        left = max(state.total - done, 0)
        self.progress.update(state.all_task_id, done=done, left=left)

    def finish_pdf(self, pdf_name: str) -> None:
        if not self.enabled or self.progress is None:
            return
        state = self._task_by_pdf.get(pdf_name)
        if not state:
            return

        self.progress.update(
            state.ocr_task_id,
            completed=max(1, state.queued),
            done=state.queued,
            left=0,
            description=f"{pdf_name} OCR queue complete ({state.queued} slide{'s' if state.queued != 1 else ''})",
        )
        self.progress.update(
            state.all_task_id,
            completed=max(1, state.total),
            done=state.total,
            left=0,
            description=f"{pdf_name} all slides complete ({state.total} slide{'s' if state.total != 1 else ''})",
        )

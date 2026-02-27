from __future__ import annotations

from dataclasses import dataclass
from io import BytesIO
from pathlib import Path

from pypdf import PdfReader
import pypdfium2 as pdfium


@dataclass(slots=True)
class PagePayload:
    page_number: int
    total_pages: int
    native_text: str
    image_png: bytes


@dataclass(slots=True)
class PageTextPayload:
    page_number: int
    total_pages: int
    native_text: str


def _extract_native_text(pdf_path: Path) -> list[str]:
    reader = PdfReader(str(pdf_path))
    texts: list[str] = []
    for page in reader.pages:
        text = page.extract_text() or ""
        texts.append(text.strip())
    return texts


def _render_pages_to_png(pdf_path: Path, dpi: int) -> list[bytes]:
    doc = pdfium.PdfDocument(str(pdf_path))
    scale = max(1, round(dpi / 72))
    images: list[bytes] = []

    try:
        for page_index in range(len(doc)):
            page = doc[page_index]
            bitmap = page.render(scale=scale)
            pil_image = bitmap.to_pil()
            buffer = BytesIO()
            pil_image.save(buffer, format="PNG")
            images.append(buffer.getvalue())
    finally:
        doc.close()

    return images


def render_selected_pages_to_png(pdf_path: Path, page_numbers: list[int], dpi: int = 180) -> dict[int, bytes]:
    if not page_numbers:
        return {}

    unique_pages = sorted(set(page_numbers))
    doc = pdfium.PdfDocument(str(pdf_path))
    scale = max(1, round(dpi / 72))
    images: dict[int, bytes] = {}

    try:
        total_pages = len(doc)
        for page_number in unique_pages:
            page_index = page_number - 1
            if page_index < 0 or page_index >= total_pages:
                raise RuntimeError(
                    f"Invalid page number {page_number} for {pdf_path.name} with {total_pages} pages"
                )

            page = doc[page_index]
            bitmap = page.render(scale=scale)
            pil_image = bitmap.to_pil()
            buffer = BytesIO()
            pil_image.save(buffer, format="PNG")
            images[page_number] = buffer.getvalue()
    finally:
        doc.close()

    return images


def extract_page_text_payloads(pdf_path: Path) -> list[PageTextPayload]:
    native_texts = _extract_native_text(pdf_path)
    total_pages = len(native_texts)
    payloads: list[PageTextPayload] = []
    for index, text in enumerate(native_texts, start=1):
        payloads.append(
            PageTextPayload(
                page_number=index,
                total_pages=total_pages,
                native_text=text,
            )
        )
    return payloads


def extract_page_payloads(pdf_path: Path, dpi: int = 180) -> list[PagePayload]:
    native_texts = _extract_native_text(pdf_path)
    page_images = _render_pages_to_png(pdf_path, dpi=dpi)

    if len(native_texts) != len(page_images):
        raise RuntimeError(
            f"Page count mismatch for {pdf_path.name}: "
            f"text={len(native_texts)}, images={len(page_images)}"
        )

    total_pages = len(native_texts)
    payloads: list[PagePayload] = []
    for index, (text, image) in enumerate(zip(native_texts, page_images), start=1):
        payloads.append(
            PagePayload(
                page_number=index,
                total_pages=total_pages,
                native_text=text,
                image_png=image,
            )
        )

    return payloads


def is_native_text_usable(text: str, min_chars: int) -> bool:
    compact = " ".join(text.split())
    return len(compact) >= min_chars


def get_pdf_page_count(pdf_path: Path) -> int:
    reader = PdfReader(str(pdf_path))
    return len(reader.pages)

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(slots=True)
class PageAnalysis:
    page_number: int
    retranscribed_text: str
    math_markdown: list[str] = field(default_factory=list)
    image_descriptions: list[str] = field(default_factory=list)


def _normalize_multiline(text: str) -> str:
    lines = [line.rstrip() for line in text.splitlines()]
    cleaned = "\n".join(lines).strip()
    return cleaned


def normalize_math_entries(entries: list[str]) -> list[str]:
    normalized: list[str] = []
    for entry in entries:
        cleaned = _normalize_multiline(entry)
        if not cleaned:
            continue
        if cleaned.startswith("$") and cleaned.endswith("$"):
            normalized.append(cleaned)
        else:
            normalized.append(f"$$\n{cleaned}\n$$")
    return normalized


def build_page_markdown(analysis: PageAnalysis) -> str:
    text_block = _normalize_multiline(analysis.retranscribed_text) or "(No text detected)"
    math_entries = normalize_math_entries(analysis.math_markdown)
    image_entries = [d.strip() for d in analysis.image_descriptions if d.strip() and not d.strip().startswith("Used native")]

    parts: list[str] = []
    
    # Extract potential title from first line of text
    text_lines = text_block.split('\n')
    first_line = text_lines[0].strip() if text_lines else ""
    
    # Use first line as title if it's reasonably short and looks like a title
    title = None
    remaining_text = text_block
    if first_line and len(first_line) < 100 and not first_line.startswith("-") and not first_line.startswith("●"):
        # Check if first line doesn't look like regular paragraph text (e.g., has formatting or is short)
        if len(first_line) < 60 or first_line.isupper() or "**" in first_line or any(c.isupper() for c in first_line[:3]):
            title = first_line
            remaining_text = "\n".join(text_lines[1:]).strip()
    
    # Only add title header if we have one
    if title:
        parts.append(f"## {title}")
    
    # Add main text content (skip if it's the same as title)
    if remaining_text and remaining_text != first_line:
        parts.append(remaining_text)
    elif not title and text_block:
        parts.append(text_block)

    # Only add Math section if we have math content
    if math_entries:
        parts.extend(math_entries)

    # Only add Images section if we have meaningful image descriptions
    if image_entries:
        parts.extend([f"- {item}" for item in image_entries])

    return "\n\n".join(parts).strip()


def build_fallback_aggregate(pages: list[PageAnalysis]) -> str:
    chunks: list[str] = []
    for page in pages:
        text = _normalize_multiline(page.retranscribed_text)
        if text:
            chunks.append(text)
    if not chunks:
        return "(No aggregate text available)"
    return "\n\n".join(chunks)

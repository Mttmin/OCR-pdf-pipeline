from __future__ import annotations

from io import BytesIO
import re

from PIL import Image, ImageFilter, ImageStat


_MATH_PATTERNS = [
    re.compile(r"\$[^$]+\$"),
    re.compile(r"\b(sum|prod|lim|sin|cos|tan|log|ln|sqrt|frac|integral|derivative)\b", re.IGNORECASE),
    re.compile(r"[=≈≠≤≥∑∫∞√α-ωΑ-Ω]"),
]

_DIAGRAM_KEYWORDS = re.compile(
    r"\b(figure|diagram|chart|graph|plot|workflow|architecture|pipeline|schema|illustration|photo|image)\b",
    re.IGNORECASE,
)

# "table" is checked separately: it routes to visual check rather than directly to
# OCR, because many slides merely *mention* a table without containing one.
_TABLE_KEYWORD = re.compile(r"\btable\b", re.IGNORECASE)


def has_math_indicators(text: str) -> bool:
    if not text.strip():
        return False
    return any(pattern.search(text) for pattern in _MATH_PATTERNS)


def has_diagram_keywords(text: str) -> bool:
    if not text.strip():
        return False
    return _DIAGRAM_KEYWORDS.search(text) is not None


def has_table_keyword(text: str) -> bool:
    if not text.strip():
        return False
    return _TABLE_KEYWORD.search(text) is not None


def has_visual_structure(image_png: bytes) -> bool:
    """Cheap image heuristic to flag likely non-trivial visuals/diagrams."""
    with Image.open(BytesIO(image_png)) as image:
        rgb = image.convert("RGB")
        gray = rgb.convert("L")

        width, height = gray.size
        if width * height < 40000:
            return False

        entropy = gray.entropy()
        if entropy < 2.0:
            return False

        edge_image = gray.filter(ImageFilter.FIND_EDGES)
        edge_stat = ImageStat.Stat(edge_image)
        edge_mean = edge_stat.mean[0]

        gray_stat = ImageStat.Stat(gray)
        gray_std = gray_stat.stddev[0]

        rgb_stat = ImageStat.Stat(rgb)
        color_std_avg = sum(rgb_stat.stddev) / 3.0

        return (edge_mean >= 10.0 and gray_std >= 28.0) or color_std_avg >= 35.0


def should_call_vision_ocr(
    native_text: str,
    image_png: bytes,
    min_native_chars: int,
) -> tuple[bool, str]:
    compact = " ".join(native_text.split())
    native_len = len(compact)

    if native_len < min_native_chars:
        return True, "weak_native_text"

    if has_math_indicators(compact):
        return True, "math_detected"

    if has_diagram_keywords(compact):
        return True, "diagram_keyword_detected"

    if native_len < 450 and has_visual_structure(image_png):
        return True, "visual_structure_detected"

    return False, "native_text_only"

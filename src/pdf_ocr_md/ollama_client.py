from __future__ import annotations

import base64
import json
import re
from dataclasses import dataclass

import httpx


class LLMError(RuntimeError):
    pass


@dataclass(slots=True)
class OCRResponse:
    retranscribed_text: str
    math_markdown: list[str]
    image_descriptions: list[str]


_JSON_RE = re.compile(r"\{.*\}", re.DOTALL)


class LLMClient:
    """Client for LM Studio's OpenAI-compatible API."""

    def __init__(self, base_url: str, model: str, timeout_seconds: float = 240.0) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self._client = httpx.Client(timeout=timeout_seconds)

    def close(self) -> None:
        self._client.close()

    def _extract_json(self, text: str) -> dict:
        text = text.strip()
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            match = _JSON_RE.search(text)
            if not match:
                raise LLMError("Model response was not valid JSON")
            return json.loads(match.group(0))

    def _chat(self, messages: list[dict]) -> str:
        url = f"{self.base_url}/v1/chat/completions"
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": 0.0,
        }
        response = self._client.post(url, json=payload)
        if response.status_code >= 400:
            raise LLMError(f"LLM call failed ({response.status_code}): {response.text}")

        data = response.json()
        choices = data.get("choices", [])
        if not choices:
            raise LLMError("LLM returned no choices")
        content = choices[0].get("message", {}).get("content", "")
        if not content:
            raise LLMError("LLM returned empty content")
        return content

    def analyze_page(
        self,
        image_png: bytes,
        page_number: int,
        total_pages: int,
        native_text: str,
    ) -> OCRResponse:
        image_b64 = base64.b64encode(image_png).decode("ascii")

        system_prompt = (
            "You are an OCR and document-transcription assistant. "
            "Extract slide text exactly, preserve meaning, and improve readability. "
            "Capture math in LaTeX-compatible markdown. "
            "Describe meaningful visual content and figures succinctly. "
            "Return strict JSON only."
        )

        user_prompt = (
            f"Analyze slide/page {page_number} of {total_pages}.\n"
            "Return JSON with keys: retranscribed_text (string), "
            "math_markdown (array of strings), image_descriptions (array of strings).\n"
            "Rules:\n"
            "1) retranscribed_text: clean and complete transcript of textual content on page.\n"
            "2) math_markdown: include each distinct equation as markdown-ready LaTeX strings.\n"
            "3) image_descriptions: bullet-ready short descriptions of charts, diagrams, photos, and key visual signals.\n"
            "4) Do not include explanations outside JSON.\n"
            f"Native extracted text (may be partial/noisy):\n{native_text if native_text.strip() else '(none)'}"
        )

        content = self._chat(
            [
                {"role": "system", "content": system_prompt},
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": user_prompt},
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/png;base64,{image_b64}"},
                        },
                    ],
                },
            ]
        )

        payload = self._extract_json(content)

        text = str(payload.get("retranscribed_text", "")).strip()
        math_items = payload.get("math_markdown", [])
        image_items = payload.get("image_descriptions", [])

        if not isinstance(math_items, list):
            math_items = []
        if not isinstance(image_items, list):
            image_items = []

        return OCRResponse(
            retranscribed_text=text,
            math_markdown=[str(item).strip() for item in math_items if str(item).strip()],
            image_descriptions=[str(item).strip() for item in image_items if str(item).strip()],
        )

    def clean_aggregate_markdown(self, page_text_blocks: list[str]) -> str:
        joined = "\n\n".join(text.strip() for text in page_text_blocks if text.strip())
        if not joined:
            return ""

        system_prompt = (
            "You are a markdown editor for OCR transcripts. "
            "Output clean markdown only, preserving technical meaning and equations."
        )
        user_prompt = (
            "Rewrite the following slide transcript into one clean markdown narrative. "
            "Preserve equations in LaTeX markdown and keep all important technical content. "
            "Do not add facts not present in text.\n\n"
            f"{joined}"
        )

        content = self._chat([
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ])
        return content.strip()


# Backward compatibility aliases
OllamaClient = LLMClient
OllamaError = LLMError

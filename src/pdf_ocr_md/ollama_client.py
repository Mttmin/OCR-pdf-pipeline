from __future__ import annotations

import base64
import json
import re
import shutil
import subprocess
from dataclasses import dataclass
import time
from urllib.parse import urlparse

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

    def list_available_models(self, timeout_seconds: float = 3.0) -> list[str]:
        base = self.base_url.rstrip("/")
        candidates: list[str] = []

        try:
            response = httpx.get(f"{base}/v1/models", timeout=timeout_seconds)
            if response.status_code < 500:
                payload = response.json()
                data = payload.get("data", []) if isinstance(payload, dict) else []
                for item in data:
                    if isinstance(item, dict):
                        model_id = str(item.get("id", "")).strip()
                        if model_id:
                            candidates.append(model_id)
        except Exception:
            pass

        try:
            response = httpx.get(f"{base}/api/tags", timeout=timeout_seconds)
            if response.status_code < 500:
                payload = response.json()
                models = payload.get("models", []) if isinstance(payload, dict) else []
                for item in models:
                    if isinstance(item, dict):
                        model_name = str(item.get("name", "")).strip()
                        if model_name:
                            candidates.append(model_name)
        except Exception:
            pass

        return sorted(set(candidates))

    def _preferred_model(self, models: list[str]) -> str | None:
        if not models:
            return None

        scored: list[tuple[int, str]] = []
        for model in models:
            normalized = model.lower().replace("-", "").replace("_", "").replace(" ", "")
            score = 0

            if "qwen" in normalized:
                score += 40
            if "vl" in normalized:
                score += 35
            if "3" in normalized:
                score += 20
            if "8b" in normalized:
                score += 30

            if "qwen" in normalized and "vl" in normalized and "8b" in normalized:
                score += 25

            scored.append((score, model))

        scored.sort(key=lambda item: (item[0], item[1]), reverse=True)
        best_score, best_model = scored[0]
        if best_score <= 0:
            return None
        return best_model

    def ensure_model_selected(self) -> tuple[bool, str | None]:
        if self.model.strip():
            return True, None

        available = self.list_available_models()
        if not available:
            return False, "No model configured and no models discovered from backend"

        preferred = self._preferred_model(available)
        self.model = preferred if preferred else available[0]
        return True, self.model

    def is_server_online(self, timeout_seconds: float = 1.5) -> tuple[bool, str | None]:
        base = self.base_url.rstrip("/")
        probe_urls = [f"{base}/v1/models", f"{base}/api/tags"]
        last_error: str | None = None

        for probe_url in probe_urls:
            try:
                response = httpx.get(probe_url, timeout=timeout_seconds)
                if response.status_code < 500:
                    return True, None
            except Exception as exc:
                last_error = str(exc)
                continue

        return False, last_error or "server probe failed"

    def _is_likely_local_ollama_endpoint(self) -> bool:
        parsed = urlparse(self.base_url)
        host = (parsed.hostname or "").lower()
        port = parsed.port
        if host not in {"127.0.0.1", "localhost"}:
            return False
        return port in {None, 11434}

    def launch_local_server(self, wait_seconds: float = 12.0) -> tuple[bool, str | None]:
        if not self._is_likely_local_ollama_endpoint():
            return (
                False,
                "Auto-launch is only supported for local Ollama endpoints "
                "(http://localhost:11434). Start your configured backend manually.",
            )

        if shutil.which("ollama") is None:
            return False, "Could not find 'ollama' in PATH."

        try:
            subprocess.Popen(
                ["ollama", "serve"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                start_new_session=True,
            )
        except Exception as exc:
            return False, f"Failed to start 'ollama serve': {exc}"

        deadline = time.monotonic() + max(0.0, wait_seconds)
        while time.monotonic() < deadline:
            online, _ = self.is_server_online(timeout_seconds=1.0)
            if online:
                return True, None
            time.sleep(0.4)

        return False, f"Ollama did not become ready within {wait_seconds:.0f}s."

    def _extract_json(self, text: str) -> dict:
        text = text.strip()
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            match = _JSON_RE.search(text)
            if not match:
                raise LLMError("Model response was not valid JSON")
            return json.loads(match.group(0))

    def _chat(self, messages: list[dict], max_tokens: int = 1200) -> str:
        url = f"{self.base_url}/v1/chat/completions"
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": 0.0,
            "max_tokens": max_tokens,
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
        mime = "image/jpeg" if image_png[:3] == b"\xff\xd8\xff" else "image/png"

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
                            "image_url": {"url": f"data:{mime};base64,{image_b64}"},
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

        content = self._chat(
            [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            max_tokens=4096,
        )
        return content.strip()


# Backward compatibility aliases
OllamaClient = LLMClient
OllamaError = LLMError

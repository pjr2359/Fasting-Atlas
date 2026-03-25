from __future__ import annotations

import json
import time
from dataclasses import dataclass
from typing import Any

import requests


class LLMError(RuntimeError):
    pass


@dataclass
class OllamaClient:
    base_url: str = "http://localhost:11434"
    model: str = "mistral:7b"
    timeout_seconds: int = 120
    debug: bool = False

    def extract_json(self, prompt: str, schema_hint: str, temperature: float = 0.1, retries: int = 2) -> dict[str, Any]:
        request_prompt = (
            "Return only valid JSON matching the schema hint.\n"
            f"Schema hint:\n{schema_hint}\n\n"
            f"Task input:\n{prompt}"
        )
        last_error: Exception | None = None
        for attempt in range(retries + 1):
            try:
                response_text = self._generate(request_prompt, temperature=temperature)
                return _extract_first_json_object(response_text)
            except Exception as exc:  # noqa: BLE001
                last_error = exc
                if self.debug:
                    print(
                        f"[llm] request failed attempt={attempt + 1}/{retries + 1} error={type(exc).__name__}: {exc}",
                        flush=True,
                    )
                continue
        raise LLMError(f"Failed to parse JSON from Ollama response: {last_error}") from last_error

    def _generate(self, prompt: str, temperature: float = 0.1) -> str:
        url = f"{self.base_url}/api/generate"
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": temperature},
        }
        started = time.perf_counter()
        if self.debug:
            print(
                f"[llm] request start model={self.model} temp={temperature} timeout={self.timeout_seconds}s",
                flush=True,
            )
        response = requests.post(url, json=payload, timeout=self.timeout_seconds)
        if self.debug:
            elapsed = time.perf_counter() - started
            print(f"[llm] request end status={response.status_code} elapsed={elapsed:.2f}s", flush=True)
        if response.status_code >= 400:
            raise LLMError(f"Ollama request failed: HTTP {response.status_code} - {response.text[:300]}")
        body = response.json()
        text = body.get("response")
        if not isinstance(text, str) or not text.strip():
            raise LLMError("Ollama response did not include text.")
        return text.strip()


def _extract_first_json_object(text: str) -> dict[str, Any]:
    start = text.find("{")
    end = text.rfind("}")
    if start < 0 or end <= start:
        raise ValueError("No JSON object found.")
    candidate = text[start : end + 1]
    parsed = json.loads(candidate)
    if not isinstance(parsed, dict):
        raise ValueError("Expected JSON object.")
    return parsed

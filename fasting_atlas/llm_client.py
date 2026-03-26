from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

import requests


class LLMError(RuntimeError):
    pass


@runtime_checkable
class JsonLLM(Protocol):
    """Any backend that can return a JSON object from a schema-guided extraction."""

    def extract_json(
        self,
        prompt: str,
        schema_hint: str,
        temperature: float = 0.1,
        retries: int = 2,
        num_predict: int | None = None,
    ) -> dict[str, Any]: ...


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


def _is_model_not_found(exc: LLMError) -> bool:
    message = str(exc).lower()
    return "404" in message and "not found" in message


def _is_anthropic_model_not_found(exc: LLMError) -> bool:
    message = str(exc).lower()
    return "not_found_error" in message or ("404" in message and "model" in message)


@dataclass
class OllamaClient:
    base_url: str = "http://localhost:11434"
    model: str = "llama3.2:1b"
    timeout_seconds: int = 600
    debug: bool = False
    default_num_predict: int = 1536

    def extract_json(
        self,
        prompt: str,
        schema_hint: str,
        temperature: float = 0.1,
        retries: int = 2,
        num_predict: int | None = None,
    ) -> dict[str, Any]:
        request_prompt = (
            "Return only valid JSON matching the schema hint. No markdown, no prose.\n"
            f"Schema hint:\n{schema_hint}\n\n"
            f"Task input:\n{prompt}"
        )
        cap = num_predict if num_predict is not None else self.default_num_predict
        last_error: Exception | None = None
        for attempt in range(retries + 1):
            try:
                response_text = self._generate(request_prompt, temperature=temperature, num_predict=cap)
                return _extract_first_json_object(response_text)
            except Exception as exc:  # noqa: BLE001
                last_error = exc
                if self.debug:
                    print(
                        f"[llm] request failed attempt={attempt + 1}/{retries + 1} error={type(exc).__name__}: {exc}",
                        flush=True,
                    )
                if isinstance(exc, LLMError) and _is_model_not_found(exc):
                    raise LLMError(
                        f"{exc} — Install the model with: ollama pull {self.model}"
                    ) from exc
                continue
        raise LLMError(f"Failed to parse JSON from Ollama response: {last_error}") from last_error

    def _generate(self, prompt: str, temperature: float = 0.1, num_predict: int | None = None) -> str:
        url = f"{self.base_url}/api/generate"
        options: dict[str, Any] = {
            "temperature": temperature,
            "num_predict": num_predict if num_predict is not None else self.default_num_predict,
        }
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "options": options,
        }
        started = time.perf_counter()
        if self.debug:
            print(
                f"[llm/ollama] start model={self.model} num_predict={options['num_predict']} "
                f"timeout={self.timeout_seconds}s",
                flush=True,
            )
        response = requests.post(url, json=payload, timeout=self.timeout_seconds)
        if self.debug:
            print(f"[llm/ollama] end status={response.status_code} elapsed={time.perf_counter() - started:.2f}s", flush=True)
        if response.status_code >= 400:
            raise LLMError(
                f"Ollama request failed: HTTP {response.status_code} - {response.text[:500]}"
            )
        body = response.json()
        text = body.get("response")
        if not isinstance(text, str) or not text.strip():
            raise LLMError("Ollama response did not include text.")
        return text.strip()


CLAUDE_API_URL = "https://api.anthropic.com/v1/messages"
# Fast / economical snapshot; see https://docs.anthropic.com/en/docs/about-claude/models
DEFAULT_CLAUDE_MODEL = "claude-haiku-4-5-20251001"


@dataclass
class ClaudeClient:
    """Anthropic Messages API — set API key via constructor or env `ANTHROPIC_API_KEY`."""

    api_key: str
    model: str = DEFAULT_CLAUDE_MODEL
    timeout_seconds: int = 120
    debug: bool = False
    anthropic_version: str = "2023-06-01"

    def __post_init__(self) -> None:
        key = (self.api_key or "").strip()
        if not key:
            key = os.environ.get("ANTHROPIC_API_KEY", "").strip()
        if not key:
            raise LLMError("ClaudeClient requires an API key (argument or ANTHROPIC_API_KEY).")
        self.api_key = key

    def extract_json(
        self,
        prompt: str,
        schema_hint: str,
        temperature: float = 0.1,
        retries: int = 2,
        num_predict: int | None = None,
    ) -> dict[str, Any]:
        user_content = (
            "Return only valid JSON matching the schema hint. No markdown, no prose, no code fences.\n"
            f"Schema hint:\n{schema_hint}\n\n"
            f"Task input:\n{prompt}"
        )
        max_tokens = num_predict if num_predict is not None else 4096
        max_tokens = max(256, min(int(max_tokens), 8192))

        last_error: Exception | None = None
        for attempt in range(retries + 1):
            try:
                response_text = self._messages(user_content, temperature, max_tokens)
                return _extract_first_json_object(response_text)
            except Exception as exc:  # noqa: BLE001
                last_error = exc
                if self.debug:
                    print(
                        f"[llm/claude] request failed attempt={attempt + 1}/{retries + 1} "
                        f"error={type(exc).__name__}: {exc}",
                        flush=True,
                    )
                if isinstance(exc, LLMError) and _is_anthropic_model_not_found(exc):
                    raise LLMError(
                        f"{exc} — Check model id (e.g. --model {DEFAULT_CLAUDE_MODEL}) or "
                        "https://docs.anthropic.com/en/docs/about-claude/models"
                    ) from exc
                continue
        raise LLMError(f"Failed to parse JSON from Claude response: {last_error}") from last_error

    def _messages(self, user_text: str, temperature: float, max_tokens: int) -> str:
        headers = {
            "x-api-key": self.api_key,
            "anthropic-version": self.anthropic_version,
            "content-type": "application/json",
        }
        payload: dict[str, Any] = {
            "model": self.model,
            "max_tokens": max_tokens,
            "messages": [{"role": "user", "content": user_text}],
        }
        if temperature is not None:
            payload["temperature"] = min(1.0, max(0.0, float(temperature)))

        started = time.perf_counter()
        if self.debug:
            print(
                f"[llm/claude] start model={self.model} max_tokens={max_tokens} "
                f"timeout={self.timeout_seconds}s",
                flush=True,
            )
        response = requests.post(
            CLAUDE_API_URL,
            headers=headers,
            json=payload,
            timeout=self.timeout_seconds,
        )
        if self.debug:
            print(
                f"[llm/claude] end status={response.status_code} elapsed={time.perf_counter() - started:.2f}s",
                flush=True,
            )

        if response.status_code >= 400:
            raise LLMError(
                f"Anthropic request failed: HTTP {response.status_code} - {response.text[:800]}"
            )

        body = response.json()
        blocks = body.get("content") or []
        parts: list[str] = []
        for block in blocks:
            if isinstance(block, dict) and block.get("type") == "text":
                parts.append(str(block.get("text", "")))
        text = "".join(parts).strip()
        if not text:
            raise LLMError("Anthropic response had no text content.")
        return text

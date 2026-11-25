from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Any

import httpx

log = logging.getLogger(__name__)


class OllamaError(RuntimeError):
    """A single request failed – the document is skipped."""


class OllamaUnavailable(OllamaError):
    """Ollama is unreachable or the model is missing – the whole step stops."""


@dataclass
class ChatResult:
    content: str
    input_tokens: int | None
    output_tokens: int | None


class Ollama:
    def __init__(self, url: str, keep_alive: str = "30m", timeout: float = 1800,
                 transport: httpx.BaseTransport | None = None):
        self.url = url.rstrip("/")
        self.keep_alive = keep_alive
        self.http = httpx.Client(base_url=self.url, timeout=httpx.Timeout(timeout, connect=10.0), transport=transport)
        self._capabilities: dict[str, set[str]] = {}

    def close(self) -> None:
        self.http.close()

    def version(self) -> str:
        return str(self._request("GET", "/api/version").get("version", "?"))

    def models(self) -> list[str]:
        return [m.get("name", "") for m in self._request("GET", "/api/tags").get("models", [])]

    def has_model(self, name: str) -> bool:
        names = set(self.models())
        return name in names or f"{name}:latest" in names

    def require(self, *models: str) -> None:
        missing = [m for m in models if m and not self.has_model(m)]
        if missing:
            raise OllamaUnavailable("Model missing in Ollama: " + ", ".join(missing)
                                    + " – install with: " + "; ".join(f"ollama pull {m}" for m in missing))

    def capabilities(self, model: str) -> set[str]:
        if model not in self._capabilities:
            data = self._request("POST", "/api/show", {"model": model})
            self._capabilities[model] = set(data.get("capabilities") or [])
        return self._capabilities[model]

    def systemone(self, model: str, state: str, questions: dict, images: list[str] | None = None) -> dict:
        body: dict[str, Any] = {"model": model, "state": state, "questions": questions,
                                "keep_alive": self.keep_alive}
        if images:
            body["images"] = images
        return self._request("POST", "/v1/systemone", body)

    def chat(self, model: str, messages: list[dict], schema: dict | None = None, think: bool | str = False,
             options: dict | None = None) -> ChatResult:
        body: dict[str, Any] = {"model": model, "messages": messages, "stream": False,
                                "keep_alive": self.keep_alive, "options": options or {}}
        if schema is not None:
            body["format"] = schema
        if "thinking" in self.capabilities(model):
            body["think"] = think
        data = self._request("POST", "/api/chat", body)
        message = data.get("message") or {}
        return ChatResult(message.get("content") or "", data.get("prompt_eval_count"), data.get("eval_count"))

    def _request(self, method: str, path: str, body: dict | None = None, retries: int = 3) -> dict:
        model = (body or {}).get("model", "")
        for attempt in range(retries + 1):
            try:
                r = self.http.request(method, path, json=body)
            except httpx.TimeoutException:
                raise OllamaError(f"Timeout ({path}, model {model}) – consider raising [ollama] timeout") from None
            except httpx.TransportError as e:
                if attempt == retries:
                    raise OllamaUnavailable(f"Ollama unreachable at {self.url}: {e}") from None
                log.warning("Ollama unreachable (%s) – retrying in %ds", e, 10 * (attempt + 1))
                time.sleep(10 * (attempt + 1))
                continue
            if r.status_code == 404 and "not found" in r.text:
                raise OllamaUnavailable(f"Ollama: {r.text.strip()[:300]} – maybe `ollama pull {model}`")
            if r.status_code >= 500 and attempt < retries:
                log.warning("Ollama HTTP %d: %s – retrying", r.status_code, r.text.strip()[:300])
                time.sleep(10 * (attempt + 1))
                continue
            if r.status_code >= 400:
                raise OllamaError(f"Ollama {path}: HTTP {r.status_code}: {r.text.strip()[:500]}")
            return r.json()
        raise OllamaError("unreachable")

"""Transports: a callable that takes a request body and returns the raw reply text."""

from __future__ import annotations

import json
import sys
import time
import urllib.error
import urllib.request
from typing import Any, Callable

Transport = Callable[[dict[str, Any]], str]

OLLAMA_URL = "http://localhost:11434"


class OllamaTransport:
    """POST to /api/chat. Counts replies whose prompt filled the context window."""

    def __init__(self, url: str = OLLAMA_URL, timeout: float = 900.0, retries: int = 2):
        self.url = url.rstrip("/")
        self.timeout = timeout
        self.retries = retries
        self.truncated = 0

    def __call__(self, body: dict[str, Any]) -> str:
        data = json.dumps(body).encode("utf-8")
        last: Exception | None = None
        for attempt in range(self.retries + 1):
            req = urllib.request.Request(
                self.url + "/api/chat", data=data, headers={"Content-Type": "application/json"}
            )
            try:
                with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                    out = json.loads(resp.read().decode("utf-8"))
                break
            except (urllib.error.URLError, TimeoutError, ConnectionError) as exc:
                last = exc
                if isinstance(exc, urllib.error.HTTPError):
                    try:
                        last = f"{exc} {exc.read().decode('utf-8', 'replace')[:300]}"
                    except Exception:
                        pass
                time.sleep(2.0 * (attempt + 1))
        else:
            raise RuntimeError(f"ollama request failed after {self.retries + 1} tries: {last}")
        num_ctx = body.get("options", {}).get("num_ctx")
        if num_ctx and out.get("prompt_eval_count", 0) >= num_ctx:
            self.truncated += 1
            print("warning: prompt filled num_ctx, the model may have lost the start of it", file=sys.stderr)
        return out["message"]["content"]

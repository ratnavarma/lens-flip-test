"""Plain-text generation and embeddings through Ollama, both cached in the call cache.

OllamaText backs the summariser and the handoff note-writer. CachedEmbedder backs
the rag-embed system. Both take their transport as an argument, so tests pass a fake.
"""

from __future__ import annotations

import hashlib
import json
import re
import time
import urllib.error
import urllib.request
from typing import Any, Callable, Sequence

from .backends import OLLAMA_URL, OllamaTransport, Transport
from .cache import CallCache, make_key
from .decider import CallStats

TextGen = Callable[[str, int], str]  # (prompt, num_predict) -> text
EmbedTransport = Callable[[str, Sequence[str]], "list[list[float]]"]  # (model, texts) -> vectors

TEXT_PROMPT_VERSION = "text-v2"
EMBED_PROMPT_VERSION = "embed-v1"
SUMMARY_MODEL = "qwen3:30b-a3b"
EMBED_MODEL = "nomic-embed-text"
EMBED_BATCH = 64

_THINK = re.compile(r"<think>.*?</think>", re.DOTALL)
_TEXT_FIELD = re.compile(r'"text"\s*:\s*"(.*)', re.DOTALL)

# qwen3:30b-a3b writes its reasoning into free text even with think off (checked live,
# 2026-10-20). A one-field JSON schema keeps the reply to the text itself, as the decider's
# schema does for actions. The callable still returns plain text.
TEXT_FORMAT = {
    "type": "object",
    "properties": {"text": {"type": "string"}},
    "required": ["text"],
}


def extract_text(raw: str) -> str:
    """The "text" field of a reply shaped like TEXT_FORMAT.

    A reply cut off by num_predict is not valid JSON; then the string after "text": is
    used as far as it goes. A reply that is not JSON at all is returned as it is.
    """
    raw = _THINK.sub("", raw).strip()
    try:
        obj = json.loads(raw)
        if isinstance(obj, dict) and isinstance(obj.get("text"), str):
            return obj["text"].strip()
    except ValueError:
        pass
    m = _TEXT_FIELD.search(raw)
    if not m:
        return raw
    body = m.group(1)
    end = re.search(r'(?<!\\)"\s*}?\s*$', body)
    if end:
        body = body[: end.start()]
    try:
        return json.loads('"' + body + '"').strip()
    except ValueError:
        return body.replace("\\n", "\n").replace('\\"', '"').strip()


class OllamaText:
    """Text completion. temperature 0, fixed seed, think off, one user message, reply in a one-field schema."""

    def __init__(self, model: str = SUMMARY_MODEL, num_ctx: int = 8192, *, transport: Transport | None = None,
                 cache: CallCache | None = None, stats: CallStats | None = None, url: str = OLLAMA_URL):
        self.model = model
        self.num_ctx = num_ctx
        self.transport = transport or OllamaTransport(url)
        self.cache = cache
        self.stats = stats if stats is not None else CallStats()

    def __call__(self, prompt: str, num_predict: int) -> str:
        messages = [{"role": "user", "content": prompt}]
        options = {"temperature": 0, "seed": 0, "num_ctx": self.num_ctx, "num_predict": int(num_predict)}
        key = make_key(self.model, TEXT_PROMPT_VERSION, messages, options, {"think": False, "format": TEXT_FORMAT})
        if self.cache is not None:
            hit = self.cache.get(key)
            if hit is not None:
                self.stats.cache_hits += 1
                return hit
        body = {"model": self.model, "messages": messages, "stream": False, "think": False,
                "format": TEXT_FORMAT, "options": options}
        raw = extract_text(self.transport(body))
        self.stats.real_calls += 1
        if self.cache is not None:
            self.cache.put(key, self.model, TEXT_PROMPT_VERSION, raw)
        return raw


class OllamaEmbedTransport:
    """POST /api/embed with {"model", "input": [...]}. Reads "embeddings" or the older "embedding"."""

    def __init__(self, url: str = OLLAMA_URL, timeout: float = 300.0, retries: int = 2):
        self.url = url.rstrip("/")
        self.timeout = timeout
        self.retries = retries

    def __call__(self, model: str, texts: Sequence[str]) -> list[list[float]]:
        data = json.dumps({"model": model, "input": list(texts)}).encode("utf-8")
        last: Any = None
        for attempt in range(self.retries + 1):
            req = urllib.request.Request(self.url + "/api/embed", data=data,
                                         headers={"Content-Type": "application/json"})
            try:
                with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                    out = json.loads(resp.read().decode("utf-8"))
                break
            except (urllib.error.URLError, TimeoutError, ConnectionError) as exc:
                last = exc
                time.sleep(2.0 * (attempt + 1))
        else:
            raise RuntimeError(f"ollama embed failed after {self.retries + 1} tries: {last}")
        return parse_embeddings(out, len(texts))


def parse_embeddings(out: dict[str, Any], n: int) -> list[list[float]]:
    """Vectors from an /api/embed reply. Handles "embeddings" (list of vectors) and "embedding" (one vector)."""
    if "embeddings" in out:
        vecs = out["embeddings"]
    elif "embedding" in out:
        vecs = out["embedding"]
        if vecs and not isinstance(vecs[0], list):
            vecs = [vecs]
    else:
        raise RuntimeError(f"embed reply has no embeddings: keys {sorted(out)}")
    if len(vecs) != n:
        raise RuntimeError(f"embed reply has {len(vecs)} vectors for {n} inputs")
    return [[float(x) for x in v] for v in vecs]


def embed_key(model: str, text: str) -> str:
    return hashlib.sha256(f"{model}\n{text}".encode("utf-8")).hexdigest()


class CachedEmbedder:
    """texts -> vectors. Each vector is cached by sha256 of the model and the text."""

    def __init__(self, model: str = EMBED_MODEL, *, transport: EmbedTransport | None = None,
                 cache: CallCache | None = None, stats: CallStats | None = None, url: str = OLLAMA_URL,
                 batch: int = EMBED_BATCH):
        self.model = model
        self.transport = transport or OllamaEmbedTransport(url)
        self.cache = cache
        self.stats = stats if stats is not None else CallStats()
        self.batch = batch
        self._memo: dict[str, list[float]] = {}

    def __call__(self, texts: Sequence[str]) -> list[list[float]]:
        keys = [embed_key(self.model, t) for t in texts]
        missing: list[int] = []
        seen: set[str] = set()
        for i, k in enumerate(keys):
            if k in self._memo:
                continue
            hit = self.cache.get(k) if self.cache is not None else None
            if hit is not None:
                self._memo[k] = json.loads(hit)
                self.stats.embed_hits += 1
            elif k not in seen:
                seen.add(k)
                missing.append(i)
        for lo in range(0, len(missing), self.batch):
            idx = missing[lo : lo + self.batch]
            vecs = self.transport(self.model, [texts[i] for i in idx])
            self.stats.embed_requests += 1
            for i, v in zip(idx, vecs):
                self._memo[keys[i]] = v
                if self.cache is not None:
                    self.cache.put(keys[i], self.model, EMBED_PROMPT_VERSION, json.dumps(v, separators=(",", ":")))
        return [self._memo[k] for k in keys]

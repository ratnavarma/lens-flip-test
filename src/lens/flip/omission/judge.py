"""The local judge (qwen3:30b-a3b through Ollama) and the priced plan for a Haiku Batch judge.

The judge answers a one-field JSON schema, {"conveyed": true|false}. think=false does not stop free-text
reasoning on this model, so the schema is what keeps the reply short (as with the summariser).

The Haiku plan is arithmetic only. It prices the borderline pairs at 2.2 characters per token and Batch rates.
Nothing here imports an API client or submits a batch.
"""

from __future__ import annotations

import json
import math
import re
from typing import Any, Callable, Sequence

from ..backends import OLLAMA_URL, OllamaTransport, Transport
from ..cache import CallCache, make_key
from ..decider import CallStats
from ..headline import requests as rq
from ..systems.base import CHARS_PER_TOKEN

JUDGE_MODEL = "qwen3:30b-a3b"
JUDGE_PROMPT_VERSION = "omission-judge-v1"
JUDGE_NUM_PREDICT = 40
JUDGE_NUM_CTX = 4096
JUDGE_FORMAT = {
    "type": "object",
    "properties": {"conveyed": {"type": "boolean"}},
    "required": ["conveyed"],
}
HAIKU_OUTPUT_TOKENS = 10          # per judgement
HAIKU_RATES = {"input": 0.05, "output": 0.25}   # USD per million tokens, Batch

Judge = Callable[[str, str], "bool | None"]      # (note, unit text) -> conveyed, or None on a parse failure

_THINK = re.compile(r"<think>.*?</think>", re.DOTALL)


def build_prompt(note: str, unit: str) -> str:
    return (
        "A colleague wrote this handoff note.\n"
        f"Note:\n{note.strip()}\n\n"
        f"Record:\n{unit.strip()}\n\n"
        "Does the note convey this record's content well enough that a reader would know it? "
        'Reply as JSON: {"conveyed": true or false}.'
    )


def parse_answer(raw: str) -> bool | None:
    """True or False when the reply is JSON with a boolean "conveyed". Anything else is a parse failure."""
    try:
        obj = json.loads(_THINK.sub("", raw).strip())
    except ValueError:
        return None
    if isinstance(obj, dict) and isinstance(obj.get("conveyed"), bool):
        return obj["conveyed"]
    return None


class OllamaJudge:
    """temperature 0, fixed seed, think off, reply in the one-field schema. Cached by request."""

    def __init__(self, model: str = JUDGE_MODEL, *, transport: Transport | None = None, cache: CallCache | None = None,
                 stats: CallStats | None = None, url: str = OLLAMA_URL):
        self.model = model
        self.transport = transport or OllamaTransport(url)
        self.cache = cache
        self.stats = stats if stats is not None else CallStats()

    def __call__(self, note: str, unit: str) -> bool | None:
        messages = [{"role": "user", "content": build_prompt(note, unit)}]
        options = {"temperature": 0, "seed": 0, "num_ctx": JUDGE_NUM_CTX, "num_predict": JUDGE_NUM_PREDICT}
        key = make_key(self.model, JUDGE_PROMPT_VERSION, messages, options, {"think": False, "format": JUDGE_FORMAT})
        if self.cache is not None:
            hit = self.cache.get(key)
            if hit is not None:
                self.stats.cache_hits += 1
                return parse_answer(hit)
        body = {"model": self.model, "messages": messages, "stream": False, "think": False,
                "format": JUDGE_FORMAT, "options": options}
        raw = self.transport(body)
        self.stats.real_calls += 1
        if self.cache is not None:
            self.cache.put(key, self.model, JUDGE_PROMPT_VERSION, raw)
        return parse_answer(raw)


def haiku_plan(pairs: Sequence[tuple[str, str]], rates: dict[str, float] = HAIKU_RATES) -> dict[str, Any]:
    """Priced plan for the Haiku Batch judge on (note, unit) pairs. No API call, nothing submitted.

    Input tokens per pair = ceil(prompt characters / 2.2). Output = 10 tokens per pair."""
    tin = [math.ceil(len(build_prompt(n, u)) / CHARS_PER_TOKEN) for n, u in pairs]
    in_tokens, out_tokens = sum(tin), HAIKU_OUTPUT_TOKENS * len(tin)
    usd = sum(rq.request_price(t, HAIKU_OUTPUT_TOKENS, rates) for t in tin)
    return {"judge": "claude-haiku-5-5 via the Message Batches API, thinking disabled", "n_pairs": len(tin),
            "chars_per_token": CHARS_PER_TOKEN, "rates_usd_per_million": dict(rates),
            "input_tokens": in_tokens, "output_tokens": out_tokens, "output_tokens_per_pair": HAIKU_OUTPUT_TOKENS,
            "estimated_usd": round(usd, 6), "count_tokens_calls": 0, "api_calls": 0, "submitted": False,
            "note": "Estimate only. Needs Varma's yes in the session before any batch is submitted."}

"""Request bodies, request keys, reply parsing and Batch prices for the headline run.

Every request is a Messages API body. Its key is the sha256 of the canonical JSON of the body,
so identical requests share one key, one cache row and one batch entry. The custom_id is
derived from the key.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
import subprocess
from typing import Any

from .. import batch
from ..decider import PARSE_ERROR, parse_reply
from ..llm import TEXT_FORMAT, extract_text
from ..residue import NOTE_SCHEMA, NOTE_TOKENS, build_residue_note_prompt, parse_note_reply
from ..schema import Case, Unit, View
from ..systems.base import truncate_tokens
from ..systems.summary import build_summary_prompt

MODEL = batch.MODEL  # claude-haiku-5-5

DECIDER: dict[str, Any] = {"model": MODEL, "arm": "adaptive-high"}
SUMMARIZER: dict[str, Any] = {"model": MODEL, "effort": "medium", "thinking": {"type": "disabled"},
                              "max_tokens_factor": 1.5, "chunk_events": 40}
NOTE_WRITER: dict[str, Any] = {"model": MODEL, "effort": "high", "thinking": None, "max_tokens": 6000,
                               "note_tokens": NOTE_TOKENS}

SUMMARY_SCHEMA: dict[str, Any] = copy.deepcopy(TEXT_FORMAT)
SUMMARY_SCHEMA["additionalProperties"] = False

RATES = {"input": batch.PRICE_IN, "output": batch.PRICE_OUT}  # USD per million tokens, Batch
LONG_PROMPT_TOKENS = 100_000  # Haiku 5.5 prices prompts above this at 5x
LONG_PROMPT_FACTOR = 5.0

KINDS = ("decide", "summary", "note")


def canonical(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def request_key(params: dict[str, Any]) -> str:
    return hashlib.sha256(canonical(params).encode("ascii")).hexdigest()


def custom_id_for(key: str) -> str:
    cid = "q" + key[:56]
    assert batch.CUSTOM_ID_RE.match(cid)
    return cid


def request_chars(params: dict[str, Any]) -> int:
    """Characters of the system prompt and every message (what count_tokens sees)."""
    n = len(params.get("system") or "")
    for m in params.get("messages", []):
        c = m.get("content")
        n += len(c) if isinstance(c, str) else sum(len(b.get("text", "")) for b in c)
    return n


# ---------------------------------------------------------------------------
# Builders
# ---------------------------------------------------------------------------

def decide_request(view: View, inserted: list[Unit], case: Case, version: str,
                   cfg: dict[str, Any] = DECIDER) -> dict[str, Any]:
    params = batch.decide_params(view, inserted, case, version, cfg["arm"])
    params["model"] = cfg["model"]
    return params


def summary_max_tokens(budget: int, cfg: dict[str, Any] = SUMMARIZER) -> int:
    return int(cfg["max_tokens_factor"] * budget)


def summary_request(summary: str, chunk: list[str], budget: int, cfg: dict[str, Any] = SUMMARIZER) -> dict[str, Any]:
    params: dict[str, Any] = {
        "model": cfg["model"],
        "max_tokens": summary_max_tokens(budget, cfg),
        "messages": [{"role": "user", "content": build_summary_prompt(summary, chunk, budget)}],
        "output_config": {"effort": cfg["effort"],
                          "format": {"type": "json_schema", "schema": copy.deepcopy(SUMMARY_SCHEMA)}},
    }
    if cfg.get("thinking") is not None:
        params["thinking"] = copy.deepcopy(cfg["thinking"])
    return params


def note_request(view: View, case: Case, cfg: dict[str, Any] = NOTE_WRITER) -> dict[str, Any]:
    params: dict[str, Any] = {
        "model": cfg["model"],
        "max_tokens": cfg["max_tokens"],
        "messages": [{"role": "user", "content": build_residue_note_prompt(view, case)}],
        "output_config": {"effort": cfg["effort"],
                          "format": {"type": "json_schema", "schema": copy.deepcopy(NOTE_SCHEMA)}},
    }
    if cfg.get("thinking") is not None:
        params["thinking"] = copy.deepcopy(cfg["thinking"])
    return params


# ---------------------------------------------------------------------------
# Replies
# ---------------------------------------------------------------------------

def raw_outcome(line: dict[str, Any]) -> dict[str, Any]:
    """A batch results line, reduced to what the cache keeps: type, stop reason, text, usage."""
    res = line.get("result") or {}
    rtype = res.get("type", "missing")
    if rtype != "succeeded":
        err = res.get("error") or {}
        if isinstance(err.get("error"), dict):
            err = err["error"]
        return {"result_type": rtype, "stop_reason": None, "text": None, "usage": {},
                "error": f"{err.get('type', rtype)}"}
    msg = res.get("message") or {}
    text = next((b.get("text", "") for b in msg.get("content") or [] if b.get("type") == "text"), None)
    return {"result_type": rtype, "stop_reason": msg.get("stop_reason"), "text": text,
            "usage": msg.get("usage") or {}}


def parse_answer(kind: str, raw: dict[str, Any], meta: dict[str, Any]) -> dict[str, Any]:
    """The task-level answer for a cached reply. meta carries "budget" for summary steps."""
    stop = raw.get("stop_reason")
    text = raw.get("text")
    if kind == "decide":
        if stop in ("refusal", "max_tokens"):
            return {"action": PARSE_ERROR, "status": stop}
        parsed = parse_reply(text) if text is not None else None
        return {"action": parsed["action"], "status": "ok"} if parsed else {"action": PARSE_ERROR, "status": "parse"}
    if kind == "summary":
        budget = int(meta["budget"])
        if stop == "refusal" or text is None:
            return {"text": None, "status": stop or "empty"}
        return {"text": truncate_tokens(extract_text(text), budget), "status": "ok" if stop != "max_tokens" else stop}
    if kind == "note":
        if stop == "refusal" or text is None:
            return {"note": "", "p_residue": None, "status": stop or "empty"}
        note, p = parse_note_reply(text)
        status = "ok" if p is not None else "p_parse"
        if stop == "max_tokens":
            status = "max_tokens"
        return {"note": note, "p_residue": p, "status": status}
    raise ValueError(f"unknown kind {kind!r}")


def failure_answer(kind: str, reason: str) -> dict[str, Any]:
    """The answer for a request that errored or expired twice."""
    status = f"failed:{reason}"
    if kind == "decide":
        return {"action": PARSE_ERROR, "status": status}
    if kind == "summary":
        return {"text": None, "status": status}
    return {"note": "", "p_residue": None, "status": status}


# ---------------------------------------------------------------------------
# Prices and token estimates
# ---------------------------------------------------------------------------

def price(input_tokens: float, output_tokens: float, rates: dict[str, float] = RATES) -> float:
    return input_tokens * rates["input"] / 1e6 + output_tokens * rates["output"] / 1e6


def request_price(input_tokens: float, output_tokens: float, rates: dict[str, float] = RATES) -> float:
    factor = LONG_PROMPT_FACTOR if input_tokens > LONG_PROMPT_TOKENS else 1.0
    return factor * price(input_tokens, output_tokens, rates)


def usage_tokens(usage: dict[str, Any]) -> tuple[int, int]:
    tin = sum(int(usage.get(k, 0) or 0) for k in ("input_tokens", "cache_creation_input_tokens",
                                                  "cache_read_input_tokens"))
    return tin, int(usage.get("output_tokens", 0) or 0)


# Fallback estimate before any calibration: dense records run about 2.1 characters per token
# (count_tokens on results/validity-haiku-v4), and the JSON output format adds 331 tokens.
DEFAULT_ESTIMATE = {"chars_per_token_low": 2.0, "overhead_tokens": 400}


def estimate_input_tokens(chars: int, kind: str, calib: dict[str, Any] | None, *, bound: bool) -> int:
    """Input tokens from characters. bound=True uses the densest ratio seen (for the spend guard)."""
    if calib and kind in calib.get("kinds", {}):
        k = calib["kinds"][kind]
        if bound:
            return math.ceil(chars / k["chars_per_token_low"]) + k["overhead_tokens"]
        return math.ceil(chars * k["slope"] + k["intercept"]) + k["overhead_tokens"]
    return math.ceil(chars / DEFAULT_ESTIMATE["chars_per_token_low"]) + DEFAULT_ESTIMATE["overhead_tokens"]


# ---------------------------------------------------------------------------
# Wrapper
# ---------------------------------------------------------------------------

def wrapper_runner(method: str, path: str, body: str | None = None, timeout: int = 120) -> tuple[int, str]:
    """~/.local/bin/anthropic-api as an argument list, with ANTHROPIC_BASE_URL unset. Body on stdin."""
    cmd = ["env", "-u", "ANTHROPIC_BASE_URL", batch.WRAPPER, method, path, "--timeout", str(int(timeout))]
    if body is not None:
        cmd += ["--body", "-"]
    proc = subprocess.run(cmd, input=body, capture_output=True, text=True, check=False)
    if proc.returncode == 2:
        raise RuntimeError(f"anthropic-api usage or environment error: {proc.stderr.strip()[:300]}")
    return proc.returncode, proc.stdout

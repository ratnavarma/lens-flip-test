"""Deciders: functions from (view, inserted units, case) to one action name.

The prompt shows the view text under "Records:". Inserted units, if any, follow in a
block headed "Additional records:", in the order given (history order for events).
oracle_decider is the mock (rule oracle). ollama_decider asks a local model.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Callable

from .backends import OLLAMA_URL, OllamaTransport, Transport
from .cache import CallCache, make_key
from .generator import oracle
from .schema import ACTIONS, Case, Unit, View

Decider = Callable[[View, "list[Unit]", Case], str]
ADDITIONAL = "Additional records:"

PARSE_ERROR = "PARSE_ERROR"
PROMPT_VERSIONS = ("v1", "v2")

SYSTEM = {
    "v1": "You are reviewing an account's records to choose one action.",
    "v2": "You read an account's records and pick the single best action for the case.",
}

POLICY = {
    "v1": (
        "Action policy:\n"
        "- EXPLAIN: the higher amount is explained by normal seasonal activity.\n"
        "- OPEN_REVIEW: the statement rests on a figure that later records contradict.\n"
        "- CHECK_DEVICE: a device change during the period may have distorted the figures.\n"
        "- APPLY_CREDIT: an approved credit was never applied.\n"
        "- REQUEST_INFO: the records do not settle the cause, or two or more different problems are present "
        "for the period, because each would call for a different action and a reviewer must decide first. "
        "Normal seasonal activity is never a problem or a cause for this rule."
    ),
    "v2": (
        "How to choose:\n"
        "- EXPLAIN means ordinary seasonal activity accounts for the higher amount.\n"
        "- OPEN_REVIEW means the statement relies on a figure that later records contradict.\n"
        "- CHECK_DEVICE means a change of device within the period may have distorted the figures.\n"
        "- APPLY_CREDIT means a credit was approved and never applied.\n"
        "- REQUEST_INFO means the records do not settle the cause, or two or more different problems are present "
        "for the period, because each would call for a different action and a reviewer must decide first. "
        "Normal seasonal activity is never a problem or a cause for this rule."
    ),
}

# One neutral context line, shared by v1, v2 and the batch path (batch.py calls build_messages).
CONTEXT = (
    "Context: each reading is the activity figure for its own period. A reset reading after a device change "
    "restarts the count. Statements list activity charges, other charges and credits."
)

REPLY = {
    "v1": 'Reply only with JSON: {"action": one of the five names, "reason": one sentence, '
          '"event_ids": ids you relied on}.',
    "v2": 'Answer with a JSON object only, shaped like {"action": one of the five names, '
          '"reason": one sentence, "event_ids": the ids you used}.',
}

# Sent as the "format" field. Plain "json" made qwen3:30b-a3b loop until Ollama aborted
# ("token repeat limit reached") on full-length prompts, so the reply shape is given as a schema.
REPLY_FORMAT = {
    "type": "object",
    "properties": {
        "action": {"type": "string", "enum": list(ACTIONS)},
        "reason": {"type": "string"},
        "event_ids": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["action", "reason", "event_ids"],
}

REMINDER = (
    'Your last reply was not valid. Reply with one JSON object only. The "action" value must be exactly one of: '
    + ", ".join(ACTIONS) + "."
)


def oracle_decider(view: View, inserted: list[Unit], case: Case) -> str:
    """Mock decider: the rule oracle applied to the view's events plus the inserted events.

    Text that is not an event (a summary, a note, a sentence) is invisible to it.
    """
    ids = set(view.event_ids) | {u.event.id for u in inserted if u.event is not None}
    return oracle([e for e in case.events if e.id in ids], case.query_period)


def records_block(view: View, inserted: list[Unit]) -> str:
    """The view text, then the inserted units under "Additional records:" when there are any."""
    records = view.text if view.text.strip() else "(no records)"
    if inserted:
        records += f"\n\n{ADDITIONAL}\n" + "\n".join(u.text for u in inserted)
    return records


def build_messages(view: View, inserted: list[Unit], case: Case, version: str) -> list[dict[str, str]]:
    records = records_block(view, inserted)
    user = f"{CONTEXT}\n\nRecords:\n{records}\n\n{case.query_text}\n\n{POLICY[version]}\n\n{REPLY[version]}"
    return [{"role": "system", "content": SYSTEM[version]}, {"role": "user", "content": user}]


def parse_reply(raw: str) -> dict | None:
    """The reply as a dict with a valid action, or None."""
    candidates = [raw]
    lo, hi = raw.find("{"), raw.rfind("}")
    if 0 <= lo < hi:
        candidates.append(raw[lo : hi + 1])
    for text in candidates:
        try:
            obj = json.loads(text)
        except ValueError:
            continue
        if isinstance(obj, dict) and isinstance(obj.get("action"), str):
            action = obj["action"].strip().upper()
            if action in ACTIONS:
                obj["action"] = action
                return obj
    return None


@dataclass
class CallStats:
    real_calls: int = 0
    cache_hits: int = 0
    embed_requests: int = 0  # HTTP calls to the embedding endpoint
    embed_hits: int = 0  # texts whose vector came from the cache


class LLMDecider:
    """Chat-model decider with a call cache and one stricter retry on a bad reply."""

    def __init__(self, model: str, prompt_version: str, num_ctx: int, transport: Transport,
                 cache: CallCache | None = None, stats: CallStats | None = None):
        if prompt_version not in PROMPT_VERSIONS:
            raise ValueError(f"prompt_version must be one of {PROMPT_VERSIONS}")
        self.model = model
        self.prompt_version = prompt_version
        self.options = {"temperature": 0, "seed": 0, "num_ctx": num_ctx}
        self.transport = transport
        self.cache = cache
        self.stats = stats if stats is not None else CallStats()
        self.last_raw = ""  # the full raw reply behind the last action
        self.last_parsed: dict | None = None

    def _ask(self, messages: list[dict[str, str]]) -> str:
        key = make_key(self.model, self.prompt_version, messages, self.options, {"think": False, "format": REPLY_FORMAT})
        if self.cache is not None:
            hit = self.cache.get(key)
            if hit is not None:
                self.stats.cache_hits += 1
                return hit
        body = {
            "model": self.model, "messages": messages, "stream": False, "think": False,
            "format": REPLY_FORMAT, "options": dict(self.options),
        }
        raw = self.transport(body)
        self.stats.real_calls += 1
        if self.cache is not None:
            self.cache.put(key, self.model, self.prompt_version, raw)
        return raw

    def __call__(self, view: View, inserted: list[Unit], case: Case) -> str:
        messages = build_messages(view, inserted, case, self.prompt_version)
        raw = self._ask(messages)
        parsed = parse_reply(raw)
        if parsed is None:
            retry = [dict(m) for m in messages]
            retry[-1]["content"] += "\n\n" + REMINDER
            raw = self._ask(retry)
            parsed = parse_reply(raw)
        self.last_raw, self.last_parsed = raw, parsed
        return parsed["action"] if parsed else PARSE_ERROR


def ollama_decider(model: str, prompt_version: str = "v1", num_ctx: int = 16384, *,
                   transport: Transport | None = None, cache: CallCache | None = None,
                   stats: CallStats | None = None, url: str = OLLAMA_URL) -> LLMDecider:
    """Decider backed by a local Ollama model. temperature 0, seed 0, think off, JSON schema format."""
    return LLMDecider(model, prompt_version, num_ctx, transport or OllamaTransport(url), cache, stats)

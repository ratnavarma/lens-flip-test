"""Memory system interface and the shared token count.

A system sees the whole history and the query, and returns a View: the text it
hands to the decider within a token budget, plus the ids of the events quoted
verbatim and the text units the view is made of.
"""

from __future__ import annotations

import math
import re
from typing import Iterable, Protocol, Sequence

from ..render import render_events, render_lines
from ..schema import Case, View

# Characters per Claude token on these rendered records, measured with
# count_tokens on Haiku 5.5 (2.16 to 2.24 by request kind, 2026-10-20).
# Budgets B are therefore close to real model tokens.
CHARS_PER_TOKEN = 2.2


class MemorySystem(Protocol):
    name: str
    reference: bool  # True for a reference system that ignores the budget

    def view(self, case: Case, budget_tokens: int) -> View:
        ...


def count_tokens(lines: Sequence[str]) -> int:
    """Token count of a rendered view: ceil(characters / CHARS_PER_TOKEN), lines joined by newlines."""
    return _tokens_for_chars(_chars(lines))


def _chars(lines: Sequence[str]) -> int:
    return sum(len(x) for x in lines) + max(len(lines) - 1, 0)


def _tokens_for_chars(chars: int) -> int:
    return math.ceil(chars / CHARS_PER_TOKEN)


def rendered(case: Case) -> list[tuple[str, str]]:
    """(event id, rendered line) for every event, in history order."""
    return [(e.id, line) for e, line in zip(case.events, render_events(case, case.events))]


def take_within_budget(ranked: Iterable[tuple[str, str]], budget_tokens: int) -> list[str]:
    """Add (id, line) pairs in the given priority order until the next one would not fit.

    The line that does not fit is dropped and the walk stops. Returned ids are
    not sorted; callers sort them into history order.
    """
    chosen: list[str] = []
    chars = 0
    for eid, line in ranked:
        extra = len(line) + (1 if chosen else 0)
        if _tokens_for_chars(chars + extra) > budget_tokens:
            break
        chars += extra
        chosen.append(eid)
    return chosen


def history_order(case: Case, ids: Iterable[str]) -> list[str]:
    wanted = set(ids)
    return [e.id for e in case.events if e.id in wanted]


def event_view(case: Case, ids: Iterable[str], system: str, budget: int | None = None) -> View:
    """A view made of whole events: the text is their rendered lines, in history order."""
    ordered = history_order(case, ids)
    lines = render_lines(case, ordered)
    return View(system=system, budget=budget, text="\n".join(lines), event_ids=ordered, units=lines)


def truncate_tokens(text: str, budget_tokens: int) -> str:
    """Cut text to at most budget_tokens (the shared ceil(chars / CHARS_PER_TOKEN) count).

    Cuts back to the last whitespace when that splits a word. Whitespace at the ends is stripped.
    """
    text = text.strip()
    if _tokens_for_chars(len(text)) <= budget_tokens:
        return text
    limit = int(max(budget_tokens, 0) * CHARS_PER_TOKEN)
    cut = text[:limit]
    if limit < len(text) and not text[limit].isspace():
        head, sep, _ = cut.rpartition(" ")
        if sep:
            cut = head
    return cut.rstrip()


_BULLET = re.compile(r"^\s*(?:[-*\u2022]\s+)")
_SENTENCE_END = re.compile(r"(?<=[.!?])\s+")


def split_sentences(text: str) -> list[str]:
    """Sentences of a plain-text summary: split at line breaks, then after . ! or ? plus space.

    A leading bullet mark is dropped. Empty pieces are skipped.
    """
    units: list[str] = []
    for line in text.splitlines():
        line = _BULLET.sub("", line).strip()
        if not line:
            continue
        units += [p.strip() for p in _SENTENCE_END.split(line) if p.strip()]
    return units

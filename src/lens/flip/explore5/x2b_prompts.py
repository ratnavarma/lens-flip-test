"""Prompt, reply schema, renderer and parser of arm X2b (docs/flip-protocol-v5b-x2-rerun.md).

X2b is the X2 rerun with a form that fits the 150-token note budget: three short fields with hard word caps,
one-word labels, whitespace collapsed. The budget is the shared helper's (ceil(chars / 2.2)), as for every note.
The v5 prompts in prompts.py are not touched; X2b reuses their p_residue sentence and the JSON parser.
"""

from __future__ import annotations

from typing import Any, Mapping

from ..schema import Case, View
from ..systems.base import CHARS_PER_TOKEN, count_tokens, truncate_tokens
from . import prompts as P

NOTE_BUDGET = P.NOTE_BUDGET  # 150 tokens, as in v3 and X2
CHARS_PER_WORD = 7.0         # X2 forms ran 6.9 characters per word, space included; rounded up

FIELDS = ("facts", "changed", "open")
LABELS = {"facts": "Facts", "changed": "Changed", "open": "Open"}
HELP = {
    "facts": "the facts the colleague needs for this decision.",
    "changed": "what changed or what conflicts in the records.",
    "open": "what is uncertain or still open.",
}
CAPS_FIRST = {"facts": 20, "changed": 10, "open": 8}
CAPS_TIGHT = {"facts": 14, "changed": 7, "open": 5}
EMPTY = "none"


def budget_chars(budget: int = NOTE_BUDGET) -> int:
    return int(budget * CHARS_PER_TOKEN)


def budget_words(budget: int = NOTE_BUDGET) -> int:
    return int(budget_chars(budget) / CHARS_PER_WORD)


def build_x2b_prompt(view: View, case: Case, caps: Mapping[str, int] = CAPS_FIRST, budget: int = NOTE_BUDGET) -> str:
    fields = "\n".join(f"- {k}: {HELP[k]} At most {caps[k]} words." for k in FIELDS)
    shape = ", ".join(f'"{k}": text' for k in FIELDS)
    lines = ", ".join(f'"{LABELS[k]}: ..."' for k in FIELDS)
    return (
        "You are handing an account case to a colleague who will choose the action. "
        "Read the records and the request, then write a short handoff note for the colleague.\n"
        f"Write the note as three fields, in this order:\n{fields}\n"
        f"The note is shown as three lines: {lines}. "
        f"The whole note, labels included, must fit in {budget} tokens: about {budget_chars(budget)} characters "
        f"or about {budget_words(budget)} words. Text past that is cut. "
        f"Keep each field under its word cap, {sum(caps[k] for k in FIELDS)} words in all. "
        "Use short phrases. Numbers and dates count as words.\n"
        "Include what the colleague needs to choose well.\n"
        "Also give p_residue: your probability, from 0 to 1, that something you left out of the note "
        "would change the colleague's action.\n"
        f'Reply only with JSON: {{{shape}, "p_residue": a number from 0 to 1}}.\n\n'
        f"Records:\n{view.text if view.text.strip() else '(no records)'}\n\n"
        f"Request: {case.query_text}"
    )


X2B_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {**{k: {"type": "string"} for k in FIELDS}, "p_residue": {"type": "number"}},
    "required": list(FIELDS) + ["p_residue"],
    "additionalProperties": False,
}

RENDER_RULE = {"labels": LABELS, "order": list(FIELDS), "empty": EMPTY, "join": "newline",
               "whitespace": "collapsed to single spaces inside a field"}


def _compact(text: str) -> str:
    return " ".join((text or "").split())


def render_x2b(fields: Mapping[str, str]) -> str:
    """Three labelled lines, in field order. Whitespace inside a field is collapsed; an empty field reads "none"."""
    return "\n".join(f"{LABELS[k]}: {_compact(fields.get(k) or '') or EMPTY}" for k in FIELDS)


def fits(text: str, budget: int = NOTE_BUDGET) -> bool:
    return count_tokens([text.strip()]) <= budget


def parse_x2b_reply(raw: str | None, budget: int = NOTE_BUDGET) -> dict[str, Any]:
    """{fields, rendered_tokens, over, note, p_residue, cut, complete}.

    rendered_tokens is the shared count of the full rendered form; over says it passes `budget`. note is the
    rendered form cut at `budget` by the shared helper (unchanged when it fits). A reply that is not JSON gives
    its text as the form, and complete False."""
    obj = P._json_obj(raw)
    if obj is None:
        full = (raw or "").strip()
        fields = None
        complete = False
        p = None
    else:
        fields = {k: (obj[k] if isinstance(obj.get(k), str) else "") for k in FIELDS}
        full = render_x2b(fields)
        complete = all(isinstance(obj.get(k), str) for k in FIELDS)
        p = P._p(obj)
    note = truncate_tokens(full, budget)
    return {"fields": fields, "rendered_tokens": count_tokens([full]) if full else 0, "over": not fits(full, budget),
            "note": note, "p_residue": p, "cut": len(note) < len(full), "complete": complete}

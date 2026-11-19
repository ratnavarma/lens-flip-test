"""Residue stage (design section 7): what a handoff note leaves out.

1. Agent A sees the view V and the query and writes a note N of at most 150 tokens
   for a colleague who will choose the action.
2. Agent B decides from N and the query, with the standard decider prompt and N as
   the view text.
3. A residue flip: a_A == target and a_B != target, where target is a_H on cases
   where a_H == a_star.
4. On a residue flip, the flip search runs with the view text = N. Its candidates are
   V's units that N does not contain verbatim: event lines for an event view, sentences
   for a summary. The record carries stage="residue".

The note-writer is an injected callable (view, case) -> text. `ollama_note_writer` is
the default.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from .decider import Decider
from .flip import group_key, flip_search
from .llm import TextGen
from .schema import Case, DistinctionRecord, Unit, View
from .systems.base import CHARS_PER_TOKEN, count_tokens, split_sentences, truncate_tokens

NOTE_TOKENS = 150
NOTE_NUM_PREDICT = 200  # a little over the cap; the text is cut to NOTE_TOKENS afterwards

NoteWriter = Callable[[View, Case], str]


def build_note_prompt(view: View, case: Case) -> str:
    return (
        "You are handing an account case to a colleague who will choose the action. "
        "Read the records and the request, then write a short handoff note for the colleague.\n"
        f"Write at most {NOTE_TOKENS} tokens (about {int(NOTE_TOKENS * CHARS_PER_TOKEN)} characters) of plain text. "
        "Include what the colleague needs to choose well. Reply with the note only.\n\n"
        f"Records:\n{view.text if view.text.strip() else '(no records)'}\n\n"
        f"Request: {case.query_text}"
    )


def ollama_note_writer(text_gen: TextGen) -> NoteWriter:
    def write(view: View, case: Case) -> str:
        return text_gen(build_note_prompt(view, case), NOTE_NUM_PREDICT)

    return write


def note_view(note: str) -> View:
    return View(system="handoff", budget=NOTE_TOKENS, text=note, event_ids=[], units=split_sentences(note))


def residue_candidates(case: Case, view: View, note: str) -> list[Unit]:
    """V's units that the note does not contain verbatim, in view order."""
    if view.event_ids and len(view.event_ids) == len(view.units):
        by_id = {e.id: e for e in case.events}
        units = [Unit(id=i, text=t, group=group_key(by_id[i]), event=by_id[i])
                 for i, t in zip(view.event_ids, view.units)]
    else:
        units = [Unit(id=f"u{k:03d}", text=t, group="sentence") for k, t in enumerate(view.units)]
    return [u for u in units if u.text.strip() not in note]


@dataclass
class ResidueOutcome:
    note: str
    note_tokens: int
    a_B: str
    flipped: bool
    record: DistinctionRecord | None


def run_residue(
    case: Case,
    view: View,
    note_writer: NoteWriter,
    decide: Decider,
    target: str,
    a_A: str,
    *,
    system: str,
    budget: int | None,
    model: str,
    max_calls: int = 48,
    verify: Decider | None = None,
    strategy: str = "group",
) -> ResidueOutcome:
    """Write the note, let B decide, and search for the left-out units on a residue flip."""
    note = truncate_tokens(note_writer(view, case), NOTE_TOKENS)
    nview = note_view(note)
    a_b = decide(nview, [], case)
    flipped = a_A == target and a_b != target
    record = None
    if flipped:
        record = flip_search(
            case, nview, decide, target, system=system, budget=budget, model=model, max_calls=max_calls,
            verify=verify, strategy=strategy, candidates=residue_candidates(case, view, note), stage="residue",
        )
    return ResidueOutcome(note=note, note_tokens=count_tokens([note]), a_B=a_b, flipped=flipped, record=record)


# ---------------------------------------------------------------------------
# Section 7.2: the note with p_residue (headline prompt) and the truncation control
# ---------------------------------------------------------------------------

TRUNC_TOKENS = NOTE_TOKENS

# The note-writer's reply shape for headline runs (section 7.2, H2c).
NOTE_SCHEMA = {
    "type": "object",
    "properties": {"note": {"type": "string"}, "p_residue": {"type": "number"}},
    "required": ["note", "p_residue"],
    "additionalProperties": False,
}


def build_residue_note_prompt(view: View, case: Case) -> str:
    """The note prompt that also asks for p_residue. All headline residue rates use this prompt.

    build_note_prompt (plain note) stays for development runs only (section 7.2, item 7).
    """
    return (
        "You are handing an account case to a colleague who will choose the action. "
        "Read the records and the request, then write a short handoff note for the colleague.\n"
        f"Write at most {NOTE_TOKENS} tokens (about {int(NOTE_TOKENS * CHARS_PER_TOKEN)} characters) of plain text. "
        "Include what the colleague needs to choose well.\n"
        "Also give p_residue: your probability, from 0 to 1, that something you left out of the note "
        "would change the colleague's action.\n"
        'Reply only with JSON: {"note": the note, "p_residue": a number from 0 to 1}.\n\n'
        f"Records:\n{view.text if view.text.strip() else '(no records)'}\n\n"
        f"Request: {case.query_text}"
    )


def parse_note_reply(raw: str | None) -> tuple[str, float | None]:
    """(note cut to NOTE_TOKENS, p_residue or None). p_residue is None unless it is a number in [0, 1].

    A reply that is not JSON gives its text as the note and no p_residue. Failures are never imputed.
    """
    import json
    import math

    if raw is None:
        return "", None
    text = raw.strip()
    obj = None
    for cand in (text, text[text.find("{"): text.rfind("}") + 1] if "{" in text else ""):
        try:
            obj = json.loads(cand)
            break
        except ValueError:
            continue
    if not isinstance(obj, dict):
        return truncate_tokens(text, NOTE_TOKENS), None
    note = obj.get("note")
    note = truncate_tokens(note, NOTE_TOKENS) if isinstance(note, str) else ""
    p = obj.get("p_residue")
    if isinstance(p, bool) or not isinstance(p, (int, float)) or not math.isfinite(p) or not 0.0 <= p <= 1.0:
        return note, None
    return note, float(p)


def tail_tokens(text: str, budget_tokens: int = TRUNC_TOKENS) -> str:
    """The last budget_tokens of text (shared ceil(chars / CHARS_PER_TOKEN) count), cut forward to a word start."""
    text = text.strip()
    limit = int(max(budget_tokens, 0) * CHARS_PER_TOKEN)
    if len(text) <= limit:
        return text
    start = len(text) - limit
    cut = text[start:]
    if not text[start - 1].isspace():
        head, sep, rest = cut.partition(" ")
        if sep:
            cut = rest
    return cut.strip()


def omission_fraction(case: Case, view: View, note: str) -> float | None:
    """Units of V not contained verbatim in N, over units of V (the H2c baseline)."""
    total = len(view.units)
    if not total:
        return None
    return len(residue_candidates(case, view, note)) / total

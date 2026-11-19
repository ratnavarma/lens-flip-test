"""Prompts, reply schemas and parsers of exploratory protocol v5.

X1 adds one paragraph to the frozen v3 note prompt (tested: removing it gives the v3 prompt back byte for
byte). X2 asks for five fields and renders them as five labelled lines. C2 is a wording-only rewrite.
C4 reuses the judge prompt of src/lens/flip/omission/judge.py, unchanged.
"""

from __future__ import annotations

import copy
import json
import math
from typing import Any

from ..decider import POLICY
from ..omission import judge as omission_judge
from ..residue import NOTE_SCHEMA, NOTE_TOKENS, build_residue_note_prompt
from ..schema import ACTIONS, Case, View
from ..systems.base import CHARS_PER_TOKEN, truncate_tokens

NOTE_BUDGET = NOTE_TOKENS  # 150, as in v3

# ---------------------------------------------------------------------------
# X1: reader-conditioned estimate
# ---------------------------------------------------------------------------

READER_PARAGRAPH = (
    "The colleague reads only your note and the request. They do not see the records. "
    "They choose one of " + ", ".join(ACTIONS[:-1]) + " or " + ACTIONS[-1] + " with the policy below, "
    "and they apply it to your note as written. "
    "Judge p_residue for that colleague: the chance that, from your note alone, they choose a different action "
    "from the one you would choose from the records.\n"
    + POLICY["v1"]
)
_X1_ANCHOR = "\n\nRecords:\n"


def build_x1_prompt(view: View, case: Case) -> str:
    """The v3 note prompt with READER_PARAGRAPH between the reply instruction and the records."""
    base = build_residue_note_prompt(view, case)
    head, sep, tail = base.partition(_X1_ANCHOR)
    if not sep:
        raise ValueError("the v3 note prompt has no Records block")
    return head + "\n\n" + READER_PARAGRAPH + sep + tail


X1_SCHEMA = copy.deepcopy(NOTE_SCHEMA)

# ---------------------------------------------------------------------------
# X2: structured note
# ---------------------------------------------------------------------------

X2_FIELDS = ("leading_explanation", "live_alternatives", "missing_evidence", "next_action", "reversibility")
X2_LABELS = {"leading_explanation": "Leading explanation", "live_alternatives": "Live alternatives",
             "missing_evidence": "Missing evidence", "next_action": "Next action", "reversibility": "Reversibility"}
_X2_HELP = {
    "leading_explanation": "the explanation you think is most likely.",
    "live_alternatives": "the other explanations that are still possible.",
    "missing_evidence": "what the records do not show and what would settle the question.",
    "next_action": "the action you would choose.",
    "reversibility": "whether that action can be undone, and at what cost.",
}


def build_x2_prompt(view: View, case: Case, budget: int = NOTE_BUDGET) -> str:
    fields = "\n".join(f"- {k}: {_X2_HELP[k]}" for k in X2_FIELDS)
    shape = ", ".join(f'"{k}": text' for k in X2_FIELDS)
    return (
        "You are handing an account case to a colleague who will choose the action. "
        "Read the records and the request, then write a short handoff note for the colleague.\n"
        f"Write the note as five fields, in this order:\n{fields}\n"
        f"The five fields together, with their labels, must fit in at most {budget} tokens "
        f"(about {int(budget * CHARS_PER_TOKEN)} characters) of plain text. "
        "Include what the colleague needs to choose well.\n"
        "Also give p_residue: your probability, from 0 to 1, that something you left out of the note "
        "would change the colleague's action.\n"
        f'Reply only with JSON: {{{shape}, "p_residue": a number from 0 to 1}}.\n\n'
        f"Records:\n{view.text if view.text.strip() else '(no records)'}\n\n"
        f"Request: {case.query_text}"
    )


X2_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {**{k: {"type": "string"} for k in X2_FIELDS}, "p_residue": {"type": "number"}},
    "required": list(X2_FIELDS) + ["p_residue"],
    "additionalProperties": False,
}


def render_x2(fields: dict[str, str]) -> str:
    """Five labelled lines, in field order. An empty field reads "none"."""
    return "\n".join(f"{X2_LABELS[k]}: {(fields.get(k) or '').strip() or 'none'}" for k in X2_FIELDS)


def _json_obj(raw: str | None) -> dict | None:
    if raw is None:
        return None
    text = raw.strip()
    for cand in (text, text[text.find("{"): text.rfind("}") + 1] if "{" in text else ""):
        try:
            obj = json.loads(cand)
        except ValueError:
            continue
        if isinstance(obj, dict):
            return obj
    return None


def _p(obj: dict | None) -> float | None:
    if obj is None:
        return None
    p = obj.get("p_residue")
    if isinstance(p, bool) or not isinstance(p, (int, float)) or not math.isfinite(p) or not 0.0 <= p <= 1.0:
        return None
    return float(p)


def parse_x2_reply(raw: str | None, budget: int = NOTE_BUDGET) -> dict[str, Any]:
    """{fields, note, p_residue, cut, complete}. note is the rendered five lines cut at `budget` tokens by the
    shared helper (as the v3 free-text note is). cut says the cut removed text. complete says all five fields
    came back as strings; a reply that is not JSON gives its text as the note, cut, and complete False."""
    obj = _json_obj(raw)
    if obj is None:
        text = truncate_tokens(raw or "", budget)
        return {"fields": None, "note": text, "p_residue": None, "cut": len((raw or "").strip()) > len(text),
                "complete": False}
    fields = {k: (obj[k] if isinstance(obj.get(k), str) else "") for k in X2_FIELDS}
    rendered = render_x2(fields)
    note = truncate_tokens(rendered, budget)
    return {"fields": fields, "note": note, "p_residue": _p(obj), "cut": len(note) < len(rendered.strip()),
            "complete": all(isinstance(obj.get(k), str) for k in X2_FIELDS)}


# ---------------------------------------------------------------------------
# C2: content-preserving rewrite
# ---------------------------------------------------------------------------

def build_rewrite_prompt(note: str, budget: int = NOTE_BUDGET) -> str:
    return (
        "Rewrite this handoff note. Keep every fact, number, date and recommendation. "
        "Change the wording only. Do not add, drop or soften anything.\n"
        f"Write at most {budget} tokens (about {int(budget * CHARS_PER_TOKEN)} characters) of plain text.\n"
        'Reply only with JSON: {"note": the rewritten note}.\n\n'
        f"Note:\n{note}"
    )


REWRITE_SCHEMA: dict[str, Any] = {"type": "object", "properties": {"note": {"type": "string"}},
                                  "required": ["note"], "additionalProperties": False}


def parse_rewrite_reply(raw: str | None, budget: int = NOTE_BUDGET) -> str | None:
    obj = _json_obj(raw)
    if obj is None or not isinstance(obj.get("note"), str):
        return None
    return truncate_tokens(obj["note"], budget)


# ---------------------------------------------------------------------------
# C4: semantic absence
# ---------------------------------------------------------------------------

def build_judge_prompt(note: str, unit: str) -> str:
    """The prompt of the local judge in src/lens/flip/omission/judge.py, unchanged."""
    return omission_judge.build_prompt(note, unit)


JUDGE_SCHEMA: dict[str, Any] = {"type": "object", "properties": {"conveyed": {"type": "boolean"}},
                                "required": ["conveyed"], "additionalProperties": False}


def parse_judge_reply(raw: str | None) -> bool | None:
    return None if raw is None else omission_judge.parse_answer(raw)

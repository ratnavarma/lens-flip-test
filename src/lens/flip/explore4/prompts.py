"""Prompts, reply schemas, note assembly and the advice-strip rule of exploratory protocol v4.

Nothing here changes a v3 prompt. The budget note prompt renders byte-identical to the frozen v3
note prompt at 150 tokens (tested), so only the two numbers move at 300 and 600.
"""

from __future__ import annotations

import copy
import json
import math
import re
from typing import Any

from ..residue import NOTE_SCHEMA
from ..schema import Case, View
from ..systems.base import CHARS_PER_TOKEN, count_tokens, split_sentences, truncate_tokens

EXTRACT_TOKENS = 150
NOTE_BUDGETS = (300, 600)

# ---------------------------------------------------------------------------
# Extractive note (Study A)
# ---------------------------------------------------------------------------


def unit_labels(n: int) -> list[str]:
    width = max(2, len(str(n)))
    return [f"R{i:0{width}d}" for i in range(1, n + 1)]


def build_extract_prompt(view: View, case: Case, budget: int = EXTRACT_TOKENS) -> str:
    labels = unit_labels(len(view.units))
    lines = "\n".join(f"[{lab}] {u}" for lab, u in zip(labels, view.units))
    return (
        "You are handing an account case to a colleague who will choose the action. "
        "Read the records and the request, then pick the records the colleague needs to choose well.\n"
        f"The colleague will see only the records you pick, copied word for word in their original order, "
        f"at most {budget} tokens (about {int(budget * CHARS_PER_TOKEN)} characters) in total. "
        "Records that do not fit are left out. "
        "You cannot add words of your own: no summary, no comment and no recommendation.\n"
        "List the labels of the records you pick, most important first.\n"
        "Also give p_residue: your probability, from 0 to 1, that something you left out "
        "would change the colleague's action.\n"
        'Reply only with JSON: {"labels": the labels you pick, "p_residue": a number from 0 to 1}.\n\n'
        f"Records:\n{lines if lines.strip() else '(no records)'}\n\n"
        f"Request: {case.query_text}"
    )


def extract_schema(labels: list[str]) -> dict[str, Any]:
    return {
        "type": "object",
        "properties": {
            "labels": {"type": "array", "items": {"type": "string", "enum": list(labels)}},
            "p_residue": {"type": "number"},
        },
        "required": ["labels", "p_residue"],
        "additionalProperties": False,
    }


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


def parse_extract_reply(raw: str | None) -> tuple[list[str] | None, float | None]:
    """(labels in A's order, p_residue). labels is None when the reply is not JSON with a label list."""
    obj = _json_obj(raw)
    if obj is None or not isinstance(obj.get("labels"), list):
        return None, _p(obj)
    return [x for x in obj["labels"] if isinstance(x, str)], _p(obj)


def assemble_extract(view: View, labels: list[str] | None, budget: int = EXTRACT_TOKENS) -> tuple[str, dict]:
    """The extractive note: A's units, walked in A's order, each kept if it still fits in `budget`
    tokens (shared ceil(chars / 2.2) helper, lines joined by newlines) and skipped otherwise; the kept
    units are then printed verbatim in view order, one per line."""
    lab = unit_labels(len(view.units))
    pos = {x: i for i, x in enumerate(lab)}
    chosen: list[int] = []
    chars = 0
    unknown = skipped = dup = 0
    seen: set[str] = set()
    for x in labels or []:
        if x in seen:
            dup += 1
            continue
        seen.add(x)
        if x not in pos:
            unknown += 1
            continue
        line = view.units[pos[x]]
        new = chars + len(line) + (1 if chosen else 0)
        if math.ceil(new / CHARS_PER_TOKEN) > budget:
            skipped += 1
            continue
        chosen.append(pos[x])
        chars = new
    note = "\n".join(view.units[i] for i in sorted(chosen))
    return note, {"n_listed": len(labels or []), "n_kept": len(chosen), "n_unknown": unknown,
                  "n_skipped_budget": skipped, "n_duplicate": dup, "kept_index": sorted(chosen)}


# ---------------------------------------------------------------------------
# Budget note (Study C)
# ---------------------------------------------------------------------------


def build_budget_note_prompt(view: View, case: Case, budget: int) -> str:
    """residue.build_residue_note_prompt with the note budget as an argument (byte-identical at 150)."""
    return (
        "You are handing an account case to a colleague who will choose the action. "
        "Read the records and the request, then write a short handoff note for the colleague.\n"
        f"Write at most {budget} tokens (about {int(budget * CHARS_PER_TOKEN)} characters) of plain text. "
        "Include what the colleague needs to choose well.\n"
        "Also give p_residue: your probability, from 0 to 1, that something you left out of the note "
        "would change the colleague's action.\n"
        'Reply only with JSON: {"note": the note, "p_residue": a number from 0 to 1}.\n\n'
        f"Records:\n{view.text if view.text.strip() else '(no records)'}\n\n"
        f"Request: {case.query_text}"
    )


BUDGET_NOTE_SCHEMA = copy.deepcopy(NOTE_SCHEMA)


def parse_budget_note_reply(raw: str | None, budget: int) -> tuple[str, float | None]:
    """residue.parse_note_reply with the cut at `budget` tokens instead of 150."""
    if raw is None:
        return "", None
    obj = _json_obj(raw)
    if obj is None:
        return truncate_tokens(raw.strip(), budget), None
    note = obj.get("note")
    note = truncate_tokens(note, budget) if isinstance(note, str) else ""
    return note, _p(obj)


# ---------------------------------------------------------------------------
# Advice strip (Study A, second arm)
# ---------------------------------------------------------------------------

# R1: a canonical action name, upper case, with an underscore or a space.
STRIP_R1 = r"\b(EXPLAIN|OPEN[ _]REVIEW|CHECK[ _]DEVICE|APPLY[ _]CREDIT|REQUEST[ _]INFO)\b"
# R2: a multi-word action name in any case, with a space, underscore or hyphen.
STRIP_R2 = r"\b(open[ _-]review|check[ _-]device|apply[ _-]credit|request[ _-]info)\b"
# R3: the sentence opens with a directive word (letters of the first token, lower case) ...
STRIP_R3_OPENERS = ("action", "advise", "apply", "ask", "best", "check", "choose", "confirm", "default", "do",
                    "escalate", "get", "hold", "lean", "next", "obtain", "open", "proceed", "recommend",
                    "recommended", "recommendation", "request", "review", "route", "suggest", "suggested",
                    "treat", "use", "verify")
# R4: ... or holds a recommending word anywhere.
STRIP_R4 = r"\b(recommend\w*|suggest\w*|advis\w*)\b"

_R1, _R2, _R4 = re.compile(STRIP_R1), re.compile(STRIP_R2, re.I), re.compile(STRIP_R4, re.I)


def strip_rule_text() -> str:
    return json.dumps({"splitter": "systems.base.split_sentences", "R1": STRIP_R1, "R2": STRIP_R2 + " (case-insensitive)",
                       "R3_openers": list(STRIP_R3_OPENERS), "R4": STRIP_R4 + " (case-insensitive)",
                       "join": "kept sentences joined by one space, in order"}, sort_keys=True)


def _opener(s: str) -> str:
    toks = s.split()
    return re.sub(r"[^a-z]", "", toks[0].lower()) if toks else ""


def strip_advice(note: str) -> tuple[str, list[dict[str, Any]]]:
    """Drop every sentence that names an action or gives advice. Returns (stripped note, dropped sentences
    with the rules that fired). An unchanged note comes back byte-identical."""
    sents = split_sentences(note)
    kept, dropped = [], []
    for s in sents:
        rules = [name for name, hit in (("R1", _R1.search(s)), ("R2", _R2.search(s)),
                                        ("R3", _opener(s) in STRIP_R3_OPENERS), ("R4", _R4.search(s))) if hit]
        (dropped if rules else kept).append({"text": s, "rules": rules} if rules else s)
    if not dropped:
        return note, []
    return " ".join(kept), dropped


def note_tokens(note: str) -> int:
    return count_tokens([note])

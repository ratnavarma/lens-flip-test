"""Prompts and fixed text rules of exploratory protocol v7.

R1. The v1 reader prompt with the input framed as a note. Two edits express that one change: the system line
    says the text is a colleague's handoff note and not the records themselves, and the block label reads
    "Handoff note:" in place of "Records:". The context line, the request, the action policy, the reply
    instruction and the reply schema are the v1 ones, byte for byte.
R2. Hedge strip. A sentence of the note is dropped when a hedge marker fires. Same shape as the v4 advice strip.
R3. Trailing fragment. A note that ends mid-sentence loses the text after its last sentence-ending mark.
X1p, X1r. The frozen X1 paragraph of v5 (explore5.prompts.READER_PARAGRAPH), split into its two halves by
    slicing, so no word is retyped. Each half is placed where X1 places the whole paragraph.
"""

from __future__ import annotations

import json
import re
from typing import Any

from ..decider import CONTEXT, POLICY, REPLY, SYSTEM
from ..explore5 import prompts as P5
from ..residue import build_residue_note_prompt
from ..schema import Case, View
from ..systems.base import split_sentences

# ---------------------------------------------------------------------------
# R1: the note-framed reader
# ---------------------------------------------------------------------------

R1_SYSTEM = ("You are reviewing a colleague's handoff note about an account to choose one action. "
             "The note is not the records themselves.")
R1_LABEL = "Handoff note:"
V1_LABEL = "Records:"
R1_EMPTY = "(no note)"


def build_r1_user(note: str, case: Case) -> str:
    """The v1 user message with the block labelled as a note. Nothing else differs from decider.build_messages."""
    body = note if note.strip() else R1_EMPTY
    return f"{CONTEXT}\n\n{R1_LABEL}\n{body}\n\n{case.query_text}\n\n{POLICY['v1']}\n\n{REPLY['v1']}"


def r1_diff_from_v1() -> dict[str, Any]:
    return {"system": {"v1": SYSTEM["v1"], "r1": R1_SYSTEM}, "label": {"v1": V1_LABEL, "r1": R1_LABEL},
            "empty": {"v1": "(no records)", "r1": R1_EMPTY},
            "unchanged": ["context line", "request", "action policy v1", "reply instruction v1", "action schema"]}


# ---------------------------------------------------------------------------
# R2: hedge strip
# ---------------------------------------------------------------------------

# H1: epistemic modals and adverbs. "may" is matched in lower case only, so the month name is not a marker.
HEDGE_H1_MAY = r"\bmay\b"
HEDGE_H1 = (r"\b(might|could|possibly|possible|perhaps|probably|likely|unlikely|presumably|potential|potentially|"
            r"apparently|seemingly)\b")
# H2: appearance and inference verbs.
HEDGE_H2 = r"\b(appears? to|seems?|seemed|suggests)\b"
# H3: stated uncertainty.
HEDGE_H3 = (r"\b(unclear|uncertain|uncertainty|unknown|unconfirmed|unverified|unresolved|inconclusive|ambiguous|"
            r"not (?:clear|certain|confirmed|verified|known|sure)|cannot|can ?not|can't|caveats?)\b")
# H4: conditions and open questions.
HEDGE_H4 = r"\b(if|unless|whether|pending)\b"
HEDGE_NOT_MARKERS = ("about", "roughly", "approximately", "~", "estimate", "estimated", "only", "but", "confirm",
                     "verify", "missing", "no record")

_H1M = re.compile(HEDGE_H1_MAY)
_H1, _H2, _H3, _H4 = (re.compile(p, re.I) for p in (HEDGE_H1, HEDGE_H2, HEDGE_H3, HEDGE_H4))


def hedge_rule_text() -> str:
    return json.dumps({"splitter": "systems.base.split_sentences", "H1_may": HEDGE_H1_MAY + " (case-sensitive)",
                       "H1": HEDGE_H1 + " (case-insensitive)", "H2": HEDGE_H2 + " (case-insensitive)",
                       "H3": HEDGE_H3 + " (case-insensitive)", "H4": HEDGE_H4 + " (case-insensitive)",
                       "drop": "a sentence where any rule fires", "join": "kept sentences joined by one space, in order",
                       "not_markers": list(HEDGE_NOT_MARKERS)}, sort_keys=True)


def hedge_rules(sentence: str) -> list[str]:
    return [name for name, hit in (("H1", _H1M.search(sentence) or _H1.search(sentence)), ("H2", _H2.search(sentence)),
                                   ("H3", _H3.search(sentence)), ("H4", _H4.search(sentence))) if hit]


def strip_hedges(note: str) -> tuple[str, list[dict[str, Any]]]:
    """Drop every sentence that holds a hedge marker. Returns (stripped note, dropped sentences with the rules
    that fired). An unchanged note comes back byte-identical."""
    kept, dropped = [], []
    for s in split_sentences(note):
        rules = hedge_rules(s)
        if rules:
            dropped.append({"text": s, "rules": rules})
        else:
            kept.append(s)
    if not dropped:
        return note, []
    return " ".join(kept), dropped


# ---------------------------------------------------------------------------
# R3: trailing fragment
# ---------------------------------------------------------------------------

SENTENCE_END = ".!?"
FRAGMENT_MARK = r"[.!?](?=\s)"
_MARK = re.compile(FRAGMENT_MARK)


def fragment_rule_text() -> str:
    return json.dumps({"applies_to": "a note whose last character, after stripping white space, is not one of . ! ?",
                       "mark": FRAGMENT_MARK + " (a sentence-ending mark followed by white space)",
                       "keep": "the text up to and including the last mark; the rest is dropped",
                       "no_mark": "nothing remains; the note is counted and not sent",
                       "other_notes": "unchanged, byte for byte"}, sort_keys=True)


def ends_mid_sentence(note: str) -> bool:
    return note.rstrip()[-1:] not in tuple(SENTENCE_END)


def drop_trailing_fragment(note: str) -> tuple[str, str]:
    """(note without its trailing fragment, the fragment). A note that ends a sentence comes back unchanged with
    an empty fragment. A note with no sentence-ending mark comes back empty."""
    if not ends_mid_sentence(note):
        return note, ""
    body = note.rstrip()
    last = None
    for m in _MARK.finditer(body):
        last = m
    if last is None:
        return "", body
    return body[: last.end()], body[last.end():].strip()


# ---------------------------------------------------------------------------
# X1p and X1r: the X1 paragraph split at its seam
# ---------------------------------------------------------------------------

_PARA = P5.READER_PARAGRAPH
_S2_START = _PARA.index("They choose one of ")
_S3_START = _PARA.index("Judge p_residue for that colleague")
_POLICY_START = _PARA.index(POLICY["v1"])

X1_S1 = _PARA[:_S2_START]                  # who reads the note and what they see
X1_S2 = _PARA[_S2_START:_S3_START]         # the reader's choice, by the policy below
X1_S3 = _PARA[_S3_START:_POLICY_START]     # p_residue for that reader
X1_POLICY = _PARA[_POLICY_START:]          # the action policy v1

READER_HALF = (X1_S1 + X1_S3).rstrip()     # X1r: S1 and S3, no policy
POLICY_HALF = X1_S2.rstrip() + "\n" + X1_POLICY   # X1p: S2 and the policy, no reader description


def split_text() -> str:
    return json.dumps({"source": "explore5.prompts.READER_PARAGRAPH", "S1": X1_S1, "S2": X1_S2, "S3": X1_S3,
                       "policy": "decider.POLICY['v1']", "X1r": "S1 + S3", "X1p": "S2 + newline + policy",
                       "placement": "between the reply instruction and the Records block, as in X1"}, sort_keys=True)


def _with_paragraph(view: View, case: Case, paragraph: str) -> str:
    base = build_residue_note_prompt(view, case)
    head, sep, tail = base.partition(P5._X1_ANCHOR)
    if not sep:
        raise ValueError("the v3 note prompt has no Records block")
    return head + "\n\n" + paragraph + sep + tail


def build_x1p_prompt(view: View, case: Case) -> str:
    """The v3 note prompt plus the policy half. p_residue is asked as the v3 prompt asks it."""
    return _with_paragraph(view, case, POLICY_HALF)


def build_x1r_prompt(view: View, case: Case) -> str:
    """The v3 note prompt plus the reader half. p_residue is asked for that reader, as in X1."""
    return _with_paragraph(view, case, READER_HALF)


X1_SPLIT_SCHEMA = P5.X1_SCHEMA

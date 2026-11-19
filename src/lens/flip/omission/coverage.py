"""Embedding coverage: for every (note, unit) pair, the maximum cosine between the unit and the note's sentences.

The embedder is an injected callable (texts in, vectors out), as in the rag-embed system. Note sentences come
from the shared sentence splitter. The same texts are embedded once: the call cache holds them by sha256.
"""

from __future__ import annotations

import math
import re
from operator import mul
from typing import Callable, Sequence

from ..systems.base import split_sentences
from .frame import Frame, NoteRec

Embedder = Callable[[Sequence[str]], "list[list[float]]"]

_DATE = r"\d{4}-\d{2}(?:-\d{2})?"
_NUM = r"\d+(?:\.\d+)?"
_TOKEN = re.compile(rf"(?<![\w.\-]){_DATE}(?![\d])|(?<![\w.]){_NUM}(?![\d]|-\d)")
_LEAD_ID = re.compile(r"^e\d{3,}\s+")


def salient_tokens(text: str) -> set[str]:
    """Dates (YYYY-MM or YYYY-MM-DD) and numbers worth matching: not zero, with a decimal point or at least 3 digits.

    The leading event id is dropped. "0.00" and bare small integers match too often to say anything."""
    out: set[str] = set()
    for m in _TOKEN.finditer(_LEAD_ID.sub("", text)):
        t = m.group(0)
        if "-" in t:
            out.add(t)
            continue
        try:
            v = float(t)
        except ValueError:
            continue
        if v != 0 and ("." in t or len(t) >= 3):
            out.add(t)
    return out


def number_overlap(unit_text: str, note: str) -> bool:
    """True when a salient number or date of the unit appears among the note's salient tokens."""
    u = salient_tokens(unit_text)
    return bool(u) and bool(u & salient_tokens(note))


def note_sentences(note: NoteRec) -> list[str]:
    return split_sentences(note.note)


def _unit(vec: Sequence[float]) -> tuple[float, ...]:
    n = math.sqrt(sum(x * x for x in vec))
    return tuple(x / n for x in vec) if n else tuple(0.0 for _ in vec)


def coverage(frame: Frame, embedder: Embedder, *, chunk_notes: int = 25) -> list[list[float]]:
    """Per note, the max cosine of each unit to the note's sentences (0.0 when the note has no sentence).

    Same order as frame.notes and each note's units."""
    out: list[list[float]] = []
    for lo in range(0, len(frame.notes), chunk_notes):
        cache: dict[str, tuple[float, ...]] = {}
        part = frame.notes[lo: lo + chunk_notes]
        sents = [note_sentences(n) for n in part]
        texts = sorted({t for n, ss in zip(part, sents) for t in ss + [u.text for u in n.units] })
        if texts:
            for t, v in zip(texts, embedder(texts)):
                cache[t] = _unit(v)
        for n, ss in zip(part, sents):
            svs = [cache[s] for s in ss]
            row = []
            for u in n.units:
                uv = cache[u.text]
                row.append(max((sum(map(mul, uv, sv)) for sv in svs), default=0.0))
            out.append(row)
    return out

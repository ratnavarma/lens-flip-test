"""The calibration set: 200 (note, unit) pairs for Varma to label by hand.

Frame: every (note, unit) pair of the eligible notes. Local signals only, no model beyond the free embedder:

  verbatim      a salient number or date of the unit appears in the note (coverage.number_overlap)
  paraphrased   no overlap and the max cosine is at or above the median over the system's non-overlap pairs
  absent        no overlap and the max cosine is below that median

The cut is per system because cosine levels differ by system (the pooled median left rag-embed with no absent
pair). Strata are therefore "likely" labels, and "high" and "low" are relative to the system. Within a system the sample takes 50 pairs, about a third from each stratum. Within a
stratum the picks are spread over the cosine range (equal-count bins, one random pick per bin), so the
borderline range is covered.

To keep the labelling short, a system's pairs come from a small cluster of notes (NOTES_PER_SYSTEM). The cluster
grows when a stratum cannot be filled. Units with a D* event are picked first, up to DSTAR_CAP per system, and
count toward their stratum. The frame holds far more D* units than the sample can take, so the cap is a limit,
not a promise to include every one.

Deterministic: random.Random seeded from the seed, the system and the purpose of the draw.
"""

from __future__ import annotations

import json
import random
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

from . import SEED
from .coverage import number_overlap
from .frame import SYSTEMS, Frame

STRATA = ("verbatim", "paraphrased", "absent")
PER_SYSTEM = 50
NOTES_PER_SYSTEM = 12
DSTAR_CAP = 8


@dataclass
class PairInfo:
    note_idx: int
    unit_idx: int
    case_id: str
    system: str
    unit_id: str
    unit_text: str
    cosine: float
    overlap: bool
    dstar: bool
    stratum: str = ""


def classify(frame: Frame, cos: Sequence[Sequence[float]]) -> list[PairInfo]:
    """All pairs of the frame with their cosine, overlap flag and stratum."""
    pairs: list[PairInfo] = []
    for ni, (note, row) in enumerate(zip(frame.notes, cos)):
        for u, c in zip(note.units, row):
            pairs.append(PairInfo(ni, u.index, note.case_id, note.system, u.unit_id, u.text, float(c),
                                  number_overlap(u.text, note.note), u.dstar))
    for system in {p.system for p in pairs}:
        sp = [p for p in pairs if p.system == system]
        rest = [p.cosine for p in sp if not p.overlap]
        cut = statistics.median(rest) if rest else 0.0
        for p in sp:
            p.stratum = "verbatim" if p.overlap else ("paraphrased" if p.cosine >= cut else "absent")
    return pairs


def quotas(system_index: int, total: int = PER_SYSTEM) -> dict[str, int]:
    """About a third each. The remainder goes to different strata for different systems."""
    base, extra = divmod(total, len(STRATA))
    return {s: base + (1 if (i - system_index) % len(STRATA) < extra else 0) for i, s in enumerate(STRATA)}


def spread_pick(cands: list[PairInfo], q: int, rng: random.Random) -> list[PairInfo]:
    """q picks spread over the cosine range: sort, cut into q equal-count bins, one random pick per bin."""
    if q <= 0:
        return []
    if q > len(cands):
        raise ValueError("fewer candidates than picks")
    ordered = sorted(cands, key=lambda p: (p.cosine, p.note_idx, p.unit_idx))
    n = len(ordered)
    out = []
    for b in range(q):
        lo, hi = (b * n) // q, ((b + 1) * n) // q
        out.append(ordered[rng.randrange(lo, hi)])
    return out


def _shift(need: dict[str, int], avail: dict[str, int]) -> dict[str, int] | None:
    """Move what a stratum cannot supply to strata with spare candidates. None when the total does not fit."""
    final = {s: min(need[s], avail[s]) for s in STRATA}
    deficit = sum(need.values()) - sum(final.values())
    while deficit > 0:
        spare = sorted(((avail[s] - final[s], s) for s in STRATA if avail[s] > final[s]), reverse=True)
        if not spare:
            return None
        final[spare[0][1]] += 1
        deficit -= 1
    return final


def draw_system(system: str, system_index: int, pairs: list[PairInfo], seed: int = SEED, *,
                per_system: int = PER_SYSTEM, notes_per_system: int = NOTES_PER_SYSTEM,
                dstar_cap: int = DSTAR_CAP) -> list[PairInfo]:
    rng = random.Random(f"omission|{seed}|{system}|draw")
    note_ids = sorted({p.note_idx for p in pairs})
    order = note_ids[:]
    rng.shuffle(order)
    quota = quotas(system_index, per_system)
    by_note: dict[int, list[PairInfo]] = {}
    for p in pairs:
        by_note.setdefault(p.note_idx, []).append(p)
    d_notes = [n for n in order if any(p.dstar for p in by_note[n])]
    want_d = min(dstar_cap, sum(1 for p in pairs if p.dstar))
    k = min(notes_per_system, len(order))
    while True:
        cluster = order[:k]
        d_in = [p for n in cluster for p in by_note[n] if p.dstar]
        if len(d_in) < want_d:
            nxt = next((n for n in d_notes if n not in cluster), None)
            if nxt is not None:
                order.remove(nxt)
                order.insert(k, nxt)
                k += 1
                continue
        cpairs = [p for n in cluster for p in by_note[n]]
        # D* picks first, spread over strata, never more than the stratum quota
        dpicks: list[PairInfo] = []
        by_st = {s: sorted([p for p in d_in if p.stratum == s], key=lambda p: (p.note_idx, p.unit_idx))
                 for s in STRATA}
        for s in STRATA:
            rng_s = random.Random(f"omission|{seed}|{system}|dstar|{s}")
            rng_s.shuffle(by_st[s])
        taken = {s: 0 for s in STRATA}
        while len(dpicks) < want_d:
            moved = False
            for s in STRATA:
                if len(dpicks) < want_d and taken[s] < len(by_st[s]) and taken[s] < quota[s]:
                    dpicks.append(by_st[s][taken[s]])
                    taken[s] += 1
                    moved = True
            if not moved:
                break
        chosen = {(p.note_idx, p.unit_idx) for p in dpicks}
        pool = {s: [p for p in cpairs if p.stratum == s and not p.dstar and (p.note_idx, p.unit_idx) not in chosen]
                for s in STRATA}
        need = {s: quota[s] - taken[s] for s in STRATA}
        final = _shift(need, {s: len(pool[s]) for s in STRATA})
        if final is None and k < len(order):
            k += 1
            continue
        if final is None:
            raise RuntimeError(f"{system}: the frame has fewer than {per_system} pairs to sample")
        picks = list(dpicks)
        for s in STRATA:
            picks += spread_pick(pool[s], final[s], random.Random(f"omission|{seed}|{system}|{s}"))
        return sorted(picks, key=lambda p: (p.note_idx, p.unit_idx))


@dataclass
class SampleRow:
    unit_label: str     # u001..
    note_label: str     # N1..
    pair: PairInfo


def draw_sample(frame: Frame, pairs: list[PairInfo], seed: int = SEED, **kw) -> tuple[list[SampleRow], list[str]]:
    """The sampled pairs with labelling ids, and the note labels' order. Notes of all systems are shuffled together."""
    picked: list[PairInfo] = []
    for si, system in enumerate(SYSTEMS):
        sp = [p for p in pairs if p.system == system]
        if sp:
            picked += draw_system(system, si, sp, seed, **kw)
    notes = sorted({p.note_idx for p in picked})
    random.Random(f"omission|{seed}|note-order").shuffle(notes)
    label = {n: f"N{i + 1}" for i, n in enumerate(notes)}
    rows: list[SampleRow] = []
    k = 0
    for n in notes:
        for p in sorted((p for p in picked if p.note_idx == n), key=lambda p: p.unit_idx):
            k += 1
            rows.append(SampleRow(f"u{k:03d}", label[n], p))
    return rows, [label[n] for n in notes]


INSTRUCTIONS = """# Omission calibration 2026-10-21

You label 200 records. Each record belongs to a handoff note. The note is shown once above its records.

The question for every record: does the note convey this record's content well enough that a reader would know it?

- Write y if the reader would know it.
- Write n if the reader would not.
- Write ? if you are unsure.

Write the answer after "label:". Upper or lower case both work. Change nothing else in the file.
"""


def vault_markdown(frame: Frame, rows: list[SampleRow]) -> str:
    """Obsidian Markdown for the labeller. No system, stratum, case id or score appears."""
    out = [INSTRUCTIONS]
    seen: set[str] = set()
    for r in rows:
        if r.note_label not in seen:
            seen.add(r.note_label)
            note = frame.notes[r.pair.note_idx].note.strip()
            out.append(f"\n### {r.note_label}\n")
            out.append("\n".join("> " + ln if ln.strip() else ">" for ln in note.splitlines()) + "\n")
        out.append(f"- [{r.unit_label}] label:   | {r.pair.unit_text}")
    return "\n".join(out) + "\n"


def key_rows(rows: list[SampleRow], seed: int = SEED) -> list[dict]:
    return [{"unit_label": r.unit_label, "note_label": r.note_label, "case_id": r.pair.case_id,
             "system": r.pair.system, "stratum": r.pair.stratum, "cosine": round(r.pair.cosine, 6),
             "overlap": r.pair.overlap, "dstar": r.pair.dstar, "unit_id": r.pair.unit_id,
             "unit_index": r.pair.unit_idx, "unit_text": r.pair.unit_text, "seed": seed}
            for r in rows]


def write_key(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as fh:
        for r in rows:
            fh.write(json.dumps(r, sort_keys=True, separators=(",", ":"), ensure_ascii=True) + "\n")

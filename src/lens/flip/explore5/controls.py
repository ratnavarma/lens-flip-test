"""C1 control draws and the C3 request salt.

C1. For each v3 residue flip, K control sets of V's units that are NOT the v3 minimal restoring set (for a
non-recoverable flip: any candidate), drawn at random without replacement, whose rendered length matches the
minimal set's within +-10 percent. The rendered length of a set is len("\\n".join(texts)), the length of the
"Additional records" block the decider sees. A non-recoverable flip has no minimal set, so its target length
is the median length of the minimal sets of the recoverable flips.

Draw rule (fixed before any call). The pool is shuffled by random.Random(f"x5-c1|<case>|<system>|<k>"),
units are walked in that order, a unit is added when it keeps the length at or below 1.1 x target, and the walk
stops at 0.9 x target. A walk that stops short is repeated with the next shuffle of the same stream, up to
MAX_ATTEMPTS. A set equal to an earlier control of the same flip counts as a failed attempt. The first walk
that lands inside the band is the control. If none does, the closest walk is kept and flagged matched=False.
Units are returned in view order.

A control is sent (send=True) when it holds at least one unit and is not a copy of an earlier control of the
same flip: an empty set is the bare note (that is C3's repeat) and a copy would be the same request. The
analysis reads only matched controls for the primary comparison and all sent controls as a sensitivity check.
"""

from __future__ import annotations

import hashlib
import json
import random
import statistics
from typing import Any, Sequence

from ..schema import Unit

K_CONTROLS = 5
TOLERANCE = 0.10
MAX_ATTEMPTS = 200
SEED_TAG = "x5-c1"


def block_chars(texts: Sequence[str]) -> int:
    return len("\n".join(texts))


def target_length(rec: dict[str, Any], median_chars: float) -> tuple[int, str]:
    """(target characters, source) for one v3 flip."""
    if rec["status"] == "restored":
        return int(rec["chars"]), "minimal_set"
    return int(round(median_chars)), "median_of_recoverable"


def median_recoverable_chars(min_sets: Sequence[dict[str, Any]]) -> float:
    vals = [r["chars"] for r in min_sets if r["status"] == "restored"]
    if not vals:
        raise ValueError("no recoverable flip: the median minimal-set length is undefined")
    return float(statistics.median(vals))


def draw_control(pool: Sequence[Unit], target: int, seed: str, *, avoid: Sequence[frozenset] = (),
                 tol: float = TOLERANCE, max_attempts: int = MAX_ATTEMPTS) -> dict[str, Any]:
    """One control set. Returns unit ids (view order), chars, ratio to target, matched, duplicate, attempts."""
    lo, hi = (1 - tol) * target, (1 + tol) * target
    rng = random.Random(seed)
    order_of = {u.id: i for i, u in enumerate(pool)}
    best: dict[str, Any] | None = None
    for attempt in range(1, max_attempts + 1):
        order = list(pool)
        rng.shuffle(order)
        chosen: list[Unit] = []
        total = 0
        for u in order:
            new = len(u.text) if not chosen else total + 1 + len(u.text)
            if new > hi:
                continue
            chosen.append(u)
            total = new
            if total >= lo:
                break
        ids = frozenset(u.id for u in chosen)
        cand = {"ids": ids, "chars": total, "attempts": attempt, "matched": bool(chosen) and lo <= total <= hi,
                "duplicate": ids in set(avoid)}
        if cand["matched"] and not cand["duplicate"]:
            best = cand
            break
        # fallback order: inside the band, then distinct from earlier controls, then closest to the target
        key = (not cand["matched"], cand["duplicate"], abs(total - target))
        if best is None or key < (not best["matched"], best["duplicate"], abs(best["chars"] - target)):
            best = cand
    assert best is not None
    ordered = sorted(best["ids"], key=order_of.__getitem__)
    return {"unit_ids": ordered, "chars": best["chars"], "target_chars": target,
            "ratio": round(best["chars"] / target, 6) if target else None, "matched": best["matched"],
            "duplicate": best["duplicate"], "attempts": best["attempts"], "seed": seed, "pool_size": len(pool),
            "send": bool(ordered) and not best["duplicate"]}


def control_seed(case_id: str, system: str, k: int) -> str:
    return f"{SEED_TAG}|{case_id}|{system}|{k}"


def build_controls(case_id: str, system: str, candidates: Sequence[Unit], rec: dict[str, Any], median_chars: float,
                   k_controls: int = K_CONTROLS, tol: float = TOLERANCE) -> list[dict[str, Any]]:
    """The K control draws of one flip. The pool excludes the minimal set of a recoverable flip."""
    target, source = target_length(rec, median_chars)
    minimal = set(rec["unit_ids"]) if rec["status"] == "restored" else set()
    unknown = minimal - {u.id for u in candidates}
    if unknown:
        raise ValueError(f"{case_id} {system}: minimal-set ids {sorted(unknown)} are not candidates of V")
    pool = [u for u in candidates if u.id not in minimal]
    out, seen = [], []
    for k in range(k_controls):
        c = draw_control(pool, target, control_seed(case_id, system, k), avoid=seen, tol=tol)
        seen.append(frozenset(c["unit_ids"]))
        out.append({"case_id": case_id, "system": system, "k": k, "target_source": source, **c})
    return out


def controls_jsonl(rows: Sequence[dict[str, Any]]) -> str:
    return "".join(json.dumps(r, sort_keys=True, separators=(",", ":")) + "\n" for r in rows)


def controls_sha(rows: Sequence[dict[str, Any]]) -> str:
    return hashlib.sha256(controls_jsonl(rows).encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------
# C3: the request salt
# ---------------------------------------------------------------------------

def salted_key(params_key_source: str, salt: str) -> str:
    """The cache and batch key of a C3 repeat: the sha256 of the canonical body plus the repeat tag.

    The body sent to the API is unchanged, so the five repeats are identical requests with five keys and five
    custom_ids. Every other arm keys by body alone, so identical bodies there share one request."""
    return hashlib.sha256((params_key_source + "\x00repeat=" + salt).encode("ascii")).hexdigest()

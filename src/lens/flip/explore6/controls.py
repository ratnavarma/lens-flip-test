"""E1 control draws: the v5 C1 draw rule with the seed tag x6-c1.

For each extractive flip, K control sets of units of V. The pool is the search's candidate list (the units of V
the extractive note does not contain, in view order) minus the flip's minimal restoring set. A flip the search
did not restore has no minimal set, so its pool is every candidate and its target length is the median
minimal-set length of the restored extractive flips. Length, tolerance, attempts, the walk and the send rule are
`explore5.controls.draw_control`, unchanged. Only the seed differs: random.Random(f"x6-c1|<case>|<system>|<k>").
"""

from __future__ import annotations

from typing import Any, Sequence

from ..explore5 import controls as CT
from ..schema import Unit

K_CONTROLS = CT.K_CONTROLS
TOLERANCE = CT.TOLERANCE
MAX_ATTEMPTS = CT.MAX_ATTEMPTS
SEED_TAG = "x6-c1"
REREADS = 5

controls_jsonl = CT.controls_jsonl
controls_sha = CT.controls_sha
median_recoverable_chars = CT.median_recoverable_chars
salted_key = CT.salted_key


def control_seed(case_id: str, system: str, k: int) -> str:
    return f"{SEED_TAG}|{case_id}|{system}|{k}"


def reread_salt(i: int) -> str:
    """The repeat tag of the i-th fresh re-read. The key is explore5.controls.salted_key(body, tag)."""
    return f"t{i}"


def build_controls(case_id: str, system: str, candidates: Sequence[Unit], rec: dict[str, Any], median_chars: float,
                   k_controls: int = K_CONTROLS, tol: float = TOLERANCE) -> list[dict[str, Any]]:
    """The K control draws of one flip. `rec` carries status, unit_ids and chars of the flip's minimal set."""
    target, source = CT.target_length(rec, median_chars)
    minimal = set(rec["unit_ids"]) if rec["status"] == "restored" else set()
    unknown = minimal - {u.id for u in candidates}
    if unknown:
        raise ValueError(f"{case_id} {system}: minimal-set ids {sorted(unknown)} are not candidates of V")
    pool = [u for u in candidates if u.id not in minimal]
    out, seen = [], []
    for k in range(k_controls):
        c = CT.draw_control(pool, target, control_seed(case_id, system, k), avoid=seen, tol=tol)
        seen.append(frozenset(c["unit_ids"]))
        out.append({"case_id": case_id, "system": system, "k": k, "target_source": source, **c})
    return out

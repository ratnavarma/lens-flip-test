"""Shared gate and rate functions for the flip test.

Validity gate (Varma, 2026-10-20): overall accuracy of a_H at least 0.90 and every
mechanism at least 0.80. runner.py and batch.py both call validity_gate.
"""

from __future__ import annotations

from typing import Any, Iterable, Mapping

from .decider import PARSE_ERROR
from .schema import REQUEST_INFO

GATE_OVERALL = 0.90
GATE_PER_MECHANISM = 0.80
_EPS = 1e-9


def validity_gate(overall: float | None, per_mechanism: Mapping[str, float | None]) -> bool:
    """True when overall >= 0.90 and every mechanism with cases is >= 0.80.

    A mechanism with no cases has accuracy None and is skipped. No overall accuracy fails.
    """
    if overall is None or overall < GATE_OVERALL - _EPS:
        return False
    return all(a >= GATE_PER_MECHANISM - _EPS for a in per_mechanism.values() if a is not None)


def gate_report(overall: float | None, per_mechanism: Mapping[str, float | None]) -> dict[str, Any]:
    return {"overall_min": GATE_OVERALL, "per_mechanism_min": GATE_PER_MECHANISM,
            "passes": validity_gate(overall, per_mechanism)}


def is_committed(action: str) -> bool:
    return action not in (REQUEST_INFO, PARSE_ERROR)


def rate(num: int, den: int) -> float | None:
    return num / den if den else None


def fcr_raw(rows: Iterable[dict]) -> dict[str, Any]:
    """Raw False Closure Rate: over all committed a_V, the share that is not a_star."""
    rows = list(rows)
    committed = [d for d in rows if is_committed(d["a_V"])]
    wrong = sum(d["a_V"] != d["a_star"] for d in committed)
    return {"n": len(rows), "committed": len(committed), "false_closures": wrong, "fcr": rate(wrong, len(committed))}


def fcr_gated(rows: Iterable[dict]) -> dict[str, Any]:
    """Gated False Closure Rate: only cases where a_H == a_star count.

    Among those, committed a_V decisions that are not a_star, over committed a_V decisions.
    """
    rows = [d for d in rows if d["a_H"] == d["a_star"]]
    committed = [d for d in rows if is_committed(d["a_V"])]
    wrong = sum(d["a_V"] != d["a_star"] for d in committed)
    return {"n": len(rows), "committed": len(committed), "false_closures": wrong, "fcr": rate(wrong, len(committed))}


def harmful_flips(rows: Iterable[dict]) -> dict[str, Any]:
    """Harmful flip: a_H == a_star and a_V != a_star. rate is over all rows, rate_gated over gated rows."""
    rows = list(rows)
    gated = sum(d["a_H"] == d["a_star"] for d in rows)
    k = sum(d["a_H"] == d["a_star"] and d["a_V"] != d["a_star"] for d in rows)
    return {"n": len(rows), "count": k, "rate": rate(k, len(rows)), "gated_n": gated, "rate_gated": rate(k, gated)}

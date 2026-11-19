"""Settings for the mock model and the prices, read from a finished run (`dryrun --calibrate-from-run`).

The v4 guesses of the first dry run priced the pilot at $2.88 against $5.76 actual. A finished run
holds what those guesses lacked, so this module reads it and returns one JSON-friendly dict:

- a_H accuracy per mechanism and the wrong answer each a* draws;
- per cell (system and budget): the flip rate (a_V differs from a_H), the share of flip searches that
  end "not recoverable by insertion" (2 calls), the cap-hit rate and the call distribution;
- the sizes of the sets the pilot's searches returned (the mock asks for a set of such a size to restore a_H) and
  a probe noise level per cell. The mock reverses a probe's pass with this probability. The level is fitted so
  that the real search code (group_search), run on the pilot's own candidate lists and found sets against that
  noisy oracle, reproduces the cell's mean calls and cap-hit rate;
- the verify rate; the residue stage (eligibility, residue flip rate per system, floor and truncation
  rates, residue searches);
- output tokens per request stage (quantiles and mean), taken from the usage of every request;
- input tokens against request characters per kind (decide, summary, note), by least squares on usage.

Nothing here calls an API. It reads RUN/state, RUN/rounds and RUN/spend.json.
"""

from __future__ import annotations

import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Sequence

from ..decider import PARSE_ERROR
from ..flip import missing_units
from ..pddmin import group_search
from ..residue import residue_candidates
from ..schema import MECHANISMS
from . import requests as rq
from .metrics import analysis_sizes, collect
from .mock import h01
from .tasks import cell_name, rebuild_view

OVERHEAD_TOKENS = 331  # output-format overhead in usage input (constant on the v4 requests)
NOISE_LEVELS = (0.0, 0.01, 0.02, 0.04, 0.06, 0.08, 0.11, 0.15, 0.2, 0.3)
FIT_REPS = 3
MIN_RESIDUE_FIT = 3  # residue searches needed for a residue-specific noise level


def quantiles(vals: Sequence[float], k: int = 101) -> list[int]:
    v = sorted(vals)
    if not v:
        return []
    return [int(round(v[int(round(i * (len(v) - 1) / (k - 1)))])) for i in range(k)]


# ---------------------------------------------------------------------------
# Usage of every request
# ---------------------------------------------------------------------------

def pilot_requests(run_dir: str | Path) -> list[dict[str, Any]]:
    """One row per succeeded request: stage, kind, cell, budget, characters, input and output tokens."""
    d = Path(run_dir)
    rows = []
    for rp in sorted((d / "rounds").glob("*/round.json")):
        metas = json.loads(rp.read_text(encoding="utf-8"))["requests"]
        for rf in sorted(rp.parent.glob("results-*.jsonl")):
            for ln in rf.read_text(encoding="utf-8").splitlines():
                if not ln.strip():
                    continue
                line = json.loads(ln)
                m = metas.get(line.get("custom_id"))
                if not m:
                    continue
                raw = rq.raw_outcome(line)
                if raw["result_type"] != "succeeded":
                    continue
                tin, tout = rq.usage_tokens(raw["usage"])
                rows.append({"stage": m["stage"], "kind": m["kind"], "cell": m.get("cell"), "budget": m.get("budget"),
                             "chars": int(m["chars"]), "tin": tin, "tout": tout})
    return rows


def input_fit(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Per kind: tokens - 331 = slope x chars + intercept, in the shape of dryrun.fit."""
    kinds: dict[str, Any] = {}
    for kind in sorted({r["kind"] for r in rows}):
        rs = [r for r in rows if r["kind"] == kind and r["tin"] > OVERHEAD_TOKENS + 50]
        if not rs:
            continue
        xs = [r["chars"] for r in rs]
        ys = [r["tin"] - OVERHEAD_TOKENS for r in rs]
        n = len(rs)
        mx, my = sum(xs) / n, sum(ys) / n
        sxx = sum((x - mx) ** 2 for x in xs)
        slope = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / sxx if sxx else my / mx
        intercept = my - slope * mx
        ratios = [x / y for x, y in zip(xs, ys)]
        err = sum(abs(slope * x + intercept - y) / y for x, y in zip(xs, ys)) / n
        kinds[kind] = {"n": n, "slope": slope, "intercept": intercept, "chars_per_token_mean": sum(xs) / sum(ys),
                       "chars_per_token_low": min(ratios) * 0.97, "chars_per_token_high": max(ratios),
                       "overhead_tokens": OVERHEAD_TOKENS, "mean_abs_rel_error": err,
                       "usage_chars_per_token": sum(xs) / sum(r["tin"] for r in rs)}
    return {"kinds": kinds}


def output_tables(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by: dict[str, list[int]] = defaultdict(list)
    for r in rows:
        by[r["stage"]].append(r["tout"])
        if r["stage"] == "summary" and r["budget"] is not None:
            by[f"summary@{int(r['budget'])}"].append(r["tout"])
    return {"mean": {k: sum(v) / len(v) for k, v in sorted(by.items())},
            "n": {k: len(v) for k, v in sorted(by.items())},
            "quantiles": {k: quantiles(v) for k, v in sorted(by.items())}}


# ---------------------------------------------------------------------------
# Probe noise fit
# ---------------------------------------------------------------------------

def simulate_search(units: list, need: frozenset, eps: float, salt: str, max_calls: int) -> tuple[int, bool, str]:
    """group_search against the oracle "a set passes iff it holds every unit of `need`", reversed with
    probability eps (hash of the set). The empty set never passes: it is the view's own answer."""

    def ask_many(sets: list[frozenset]) -> list[str]:
        out = []
        for t in sets:
            ok = bool(t) and need <= t
            if eps and t and h01(salt, ",".join(sorted(t))) < eps:
                ok = not ok
            out.append("T" if ok else "X")
        return out

    res = group_search(units, "T", ask_many, max_calls)
    return int(res["calls_used"]), bool(res["hit_cap"]), res["status"]


def fit_noise(items: list[tuple[str, list, frozenset]], max_calls: int, target_mean: float, target_cap: float,
              tag: str) -> dict[str, Any]:
    """Noise level whose simulated mean calls and cap-hit rate sit closest to the pilot's."""
    best: dict[str, Any] | None = None
    for eps in NOISE_LEVELS:
        calls, caps, n = 0.0, 0.0, 0
        for cid, units, need in items:
            for rep in range(FIT_REPS):
                c, cap, _ = simulate_search(units, need, eps, f"{tag}|{cid}|{rep}", max_calls)
                calls += c
                caps += cap
                n += 1
        mean, cap_rate = calls / n, caps / n
        loss = ((mean - target_mean) / max(target_mean, 1.0)) ** 2 + (cap_rate - target_cap) ** 2
        if best is None or loss < best["loss"] - 1e-12:
            best = {"noise": eps, "sim_mean_calls": mean, "sim_cap_rate": cap_rate, "loss": loss}
    assert best is not None
    return {**best, "target_mean_calls": target_mean, "target_cap_rate": target_cap, "n_searches": len(items)}


# ---------------------------------------------------------------------------
# The settings
# ---------------------------------------------------------------------------

def _rate(num: int, den: int) -> float | None:
    return num / den if den else None


def _weights(c: Counter) -> dict[str, float]:
    tot = sum(c.values())
    return {k: v / tot for k, v in sorted(c.items())} if tot else {}


def _search_stats(recs: list[dict]) -> dict[str, Any]:
    n = len(recs)
    restored = [r for r in recs if r["status"] == "restored"]
    return {"n": n, "p_not_recoverable": _rate(sum(r["status"] == "not_recoverable_by_insertion" for r in recs), n),
            "cap_rate": _rate(sum(r["hit_cap"] for r in recs), n),
            "mean_calls": sum(r["calls_used"] for r in recs) / n if n else None,
            "calls_quantiles": quantiles([r["calls_used"] for r in recs], 11),
            "verify_rate": _rate(sum(r.get("verified") is True for r in restored), len(restored)),
            "restored": len(restored)}


def from_run(run_dir: str | Path, log=lambda m: None) -> dict[str, Any]:
    d = Path(run_dir)
    data = collect(d)
    cfg, cases = data["config"], data["cases"]
    by_id = {c.case_id: c for c in cases}
    systems, budgets = cfg["systems"], [int(b) for b in cfg["budgets"]]
    cells = [cell_name(s, b) for s in systems for b in budgets]
    max_calls = int(cfg["max_calls"])
    dec, recs = data["decisions"], data["records"]
    rrows, rrecs = data["residue_rows"], data["residue_records"]

    # ----- a_H and a_V -----
    a_h = {r["case_id"]: r["a_H"] for r in dec}
    acc = {}
    for m in MECHANISMS:
        cs = [c for c in cases if c.mechanism == m and c.case_id in a_h]
        acc[m] = _rate(sum(a_h[c.case_id] == c.a_star for c in cs), len(cs))
    overall_acc = _rate(sum(a_h[c.case_id] == c.a_star for c in cases if c.case_id in a_h), len(a_h))
    wrong_for: dict[str, Counter] = defaultdict(Counter)
    for c in cases:
        if c.case_id in a_h and a_h[c.case_id] != c.a_star:
            wrong_for[c.a_star][a_h[c.case_id]] += 1
    ok = [r for r in dec if r["a_H"] != PARSE_ERROR]
    flip_rate, n_dec = {}, {}
    for cell in cells:
        rs = [r for r in ok if cell_name(r["system"], r["budget"]) == cell]
        n_dec[cell] = len(rs)
        flip_rate[cell] = _rate(sum(r["a_V"] != r["a_H"] for r in rs), len(rs))
    pooled_flip = _rate(sum(r["a_V"] != r["a_H"] for r in ok), len(ok)) or 0.0
    flipped_to = _weights(Counter(r["a_V"] for r in ok if r["a_V"] != r["a_H"]))

    # ----- flip searches -----
    states: dict[str, dict] = {}

    def state(cid: str) -> dict:
        if cid not in states:
            states[cid] = json.loads((d / "state" / "cases" / f"{cid}.json").read_text(encoding="utf-8"))
        return states[cid]

    per_cell: dict[str, list[dict]] = defaultdict(list)
    for r in recs:
        per_cell[cell_name(r["system"], int(r["budget"]))].append(r)
    fit_items: dict[str, list] = defaultdict(list)
    for cell, rs in per_cell.items():
        for r in rs:
            if r["status"] != "restored":
                continue
            case = by_id[r["case_id"]]
            cs = state(r["case_id"])["cells"][cell]
            units = missing_units(case, rebuild_view(case, r["system"], int(r["budget"]), cs["view"]))
            ids = {u.id for u in units}
            need = frozenset(set(r["event_ids"]) & ids)
            if need:
                fit_items[cell].append((r["case_id"], units, need))
    pooled_search = _search_stats(recs)
    search: dict[str, Any] = {"pooled": pooled_search, "cells": {}, "noise": {}, "fit": {}}
    all_items = [it for v in fit_items.values() for it in v]
    all_rec_ok = [r for r in recs if r["status"] == "restored"]
    pooled_fit = fit_noise(all_items, max_calls, _mean([r["calls_used"] for r in all_rec_ok]),
                           _mean([r["hit_cap"] for r in all_rec_ok]), "pooled") if all_items else None
    for cell in cells:
        st = _search_stats(per_cell.get(cell, []))
        search["cells"][cell] = st
        rs = [r for r in per_cell.get(cell, []) if r["status"] == "restored"]
        items = fit_items.get(cell, [])
        if len(items) >= 3:
            f = fit_noise(items, max_calls, _mean([r["calls_used"] for r in rs]), _mean([r["hit_cap"] for r in rs]),
                          cell)
            search["noise"][cell], search["fit"][cell] = f["noise"], f
            log(f"noise fit {cell}: eps {f['noise']} mean calls {f['sim_mean_calls']:.1f} (pilot "
                f"{f['target_mean_calls']:.1f}), cap {f['sim_cap_rate']:.2f} (pilot {f['target_cap_rate']:.2f})")
    search["pooled_noise"] = pooled_fit["noise"] if pooled_fit else 0.0
    search["pooled_fit"] = pooled_fit
    search["pooled_not_recoverable"] = pooled_search["p_not_recoverable"] or 0.0
    search["pooled_verify_rate"] = pooled_search["verify_rate"] if pooled_search["verify_rate"] is not None else 0.75
    search["p_not_recoverable"] = {c: st["p_not_recoverable"] for c, st in search["cells"].items()
                                   if st["p_not_recoverable"] is not None}
    # Sizes of the sets the pilot's searches returned: the mock asks for a set of this size to restore a_H.
    search["found_sizes"] = {c: sorted(len(r["event_ids"]) for r in per_cell.get(c, []) if r["status"] == "restored")
                             for c in cells}
    search["found_sizes_pooled"] = sorted(len(r["event_ids"]) for r in recs if r["status"] == "restored")

    # ----- residue stage -----
    n_pairs = len(cases) * len(systems)
    res: dict[str, Any] = {"eligibility_rate": _rate(len(rrows), n_pairs), "n_eligible": len(rrows),
                           "floor_rate": _rate(sum(r["a_floor"] != r["a_A"] for r in rrows), len(rrows)) or 0.0,
                           "trunc_rate": _rate(sum(r["a_trunc"] != r["a_star"] for r in rrows), len(rrows)) or 0.0,
                           "flip_rate": {}, "n_eligible_by_system": {}}
    res["pooled_flip_rate"] = _rate(sum(r["residue_flip"] for r in rrows), len(rrows)) or 0.0
    for s in systems:
        rs = [r for r in rrows if r["system"] == s]
        res["n_eligible_by_system"][s] = len(rs)
        if rs:
            res["flip_rate"][s] = sum(r["residue_flip"] for r in rs) / len(rs)
    res["wrong_to"] = {
        "residue_B": _weights(Counter(r["a_B"] for r in rrows if r["residue_flip"])),
        "floor": _weights(Counter(r["a_floor"] for r in rrows if r["a_floor"] != r["a_A"])),
        "trunc": _weights(Counter(r["a_trunc"] for r in rrows if r["a_trunc"] != r["a_star"]))}
    rsearch = _search_stats(rrecs)
    r_items = []
    for r in rrecs:
        if r["status"] != "restored":
            continue
        case = by_id[r["case_id"]]
        cs = state(r["case_id"])["cells"][cell_name(r["system"], int(r["budget"]))]
        view = rebuild_view(case, r["system"], int(r["budget"]), cs["view"])
        note = cs["residue"]["out"]["note"]
        units = residue_candidates(case, view, note)
        ids = {u.id for u in units}
        need = frozenset(set(case.d_star) & ids) or frozenset(set(r["event_ids"]) & ids)
        if need:
            r_items.append((r["case_id"], units, need))
    r_ok = [r for r in rrecs if r["status"] == "restored"]
    if len(r_items) >= MIN_RESIDUE_FIT:
        rf = fit_noise(r_items, max_calls, _mean([r["calls_used"] for r in r_ok]), _mean([r["hit_cap"] for r in r_ok]),
                       "residue")
    else:
        rf = None
    rsearch["noise"] = rf["noise"] if rf else search["pooled_noise"]
    rsearch["fit"] = rf
    rsearch["pooled_not_recoverable"] = rsearch["p_not_recoverable"] or 0.0
    rsearch["verify_rate"] = (rsearch["verify_rate"] if rsearch["restored"] >= 5 and rsearch["verify_rate"] is not None
                              else search["pooled_verify_rate"])

    # ----- tokens, time, sizes -----
    reqs = pilot_requests(d)
    spend = json.loads((d / "spend.json").read_text(encoding="utf-8")) if (d / "spend.json").exists() else {}
    secs = sorted(r["ended_at"] - r["submitted_at"] for r in spend.get("rounds", []) if "ended_at" in r)
    rrmap = {(r["case_id"], r["system"]): r for r in rrecs}
    sizes = analysis_sizes(cases, rrows, rrmap, cfg)
    return {
        "source": str(run_dir), "n_cases": len(cases), "systems": list(systems), "budgets": budgets,
        "max_calls": max_calls,
        "a_H": {"accuracy_by_mechanism": acc, "overall": overall_acc,
                "wrong_answer_for": {a: _weights(c) for a, c in sorted(wrong_for.items())}},
        "a_V": {"flip_rate": {k: v for k, v in flip_rate.items() if v is not None}, "n": n_dec,
                "pooled_flip_rate": pooled_flip, "flipped_to": flipped_to},
        "search": search, "residue": res, "residue_search": rsearch,
        "output": output_tables(reqs), "input": input_fit(reqs),
        "requests": len(reqs),
        "minutes_per_round": {"median": secs[len(secs) // 2] / 60 if secs else None,
                              "p90": secs[min(len(secs) - 1, int(0.9 * len(secs)))] / 60 if secs else None,
                              "rounds": len(secs)},
        "analysis_sizes": sizes, "pilot_rounds": len(spend.get("rounds", [])),
        "pilot_usd": spend.get("actual_usd"), "pilot_by_stage": spend.get("by_stage", {}),
    }


def _mean(xs: Sequence[float]) -> float:
    return float(sum(xs) / len(xs)) if xs else 0.0


def poisson_bounds(k: int, level: float = 0.95) -> tuple[float, float]:
    """Exact bounds for a Poisson mean given a count k (bisection on the cdf)."""
    a = (1 - level) / 2

    def cdf(lam: float, kk: int) -> float:
        return sum(math.exp(-lam) * lam**i / math.factorial(i) for i in range(kk + 1))

    def solve(f, lo: float, hi: float, increasing: bool) -> float:
        for _ in range(80):
            mid = (lo + hi) / 2
            if (f(mid) > 0) == increasing:
                hi = mid
            else:
                lo = mid
        return (lo + hi) / 2

    lower = 0.0 if k == 0 else solve(lambda lam: (1 - cdf(lam, k - 1)) - a, 0.0, 10.0 * k + 20, True)
    upper = solve(lambda lam: cdf(lam, k) - a, 0.0, 10.0 * k + 20, False)
    return lower, upper

"""Price every v5 arm without a model call. Arithmetic on stored prompt sizes and stored usage.

Inputs (read only): the v5 frame and C1 draws, the usage of the finished v3 and v4 runs. No network, no
count_tokens call, no Batch submission.

Input tokens. likely = ceil(chars / 2.2) + 331 per request. Upper bound = ceil(chars / 1.9020) + 331 (the densest
ratio seen on decide requests). 331 is the structured-output overhead that usage showed on every v3 and v4
request. The rule is checked against the real input tokens of the v4 stages in `calibration_check`.

Output tokens. likely = the mean of the matching v3 or v4 stage (ASSUMED_OUT in engine.py, compared with the
observed means in `observed_usage`). Upper bound = max_tokens on every request.

Flip search (X2 only). likely = FLIP_LIKELY x 490 searches at the pooled calls per search of v3 and v4 (stored
records), plus the pooled verify calls. Upper bound = FLIP_UPPER x 490 searches, every one at 48 calls plus one
verify, every call at max_tokens, every call carrying the note and all of V's candidate units.
"""

from __future__ import annotations

import json
import math
import time
from pathlib import Path
from typing import Any, Sequence

from ..headline import requests as rq
from ..headline.calibrate import pilot_requests
from ..residue import note_view, residue_candidates
from . import engine as E
from .frame import (CHARS_PER_TOKEN_BOUND, CHARS_PER_TOKEN_LIKELY, OUT, OVERHEAD_TOKENS, RESULTS, load_frame,
                    load_min_sets, sha256_file)

RATES = E.RATES
CAP_CHECK_USD = 10.0
V4_SEARCH_ARM = "A-extractive"
# Flip rates of the v3 note and the v4 note-writer arms (k of 490): v3 142, c300 162, c600 167; the highest seen
# on any v4 arm is the extractive note at 193.
FLIP_LIKELY = (142 + 162 + 167) / (3 * 490)
FLIP_UPPER = 193 / 490


def tokens_likely(chars: int) -> int:
    return math.ceil(chars / CHARS_PER_TOKEN_LIKELY) + OVERHEAD_TOKENS


def tokens_bound(chars: int) -> int:
    return math.ceil(chars / CHARS_PER_TOKEN_BOUND) + OVERHEAD_TOKENS


def stage_price(name: str, chars: Sequence[int], out_likely: float, max_tokens: int, *, scale: float = 1.0,
                note: str = "") -> dict[str, Any]:
    """One request stage. `scale` multiplies the request count (expected flips for a search stage)."""
    n = len(chars) * scale
    tin_l = sum(tokens_likely(c) for c in chars) * scale
    tin_u = sum(tokens_bound(c) for c in chars) * scale
    return {"stage": name, "requests": round(n, 1), "input_tokens_likely": round(tin_l),
            "output_tokens_likely": round(n * out_likely), "likely_usd": round(rq.price(tin_l, n * out_likely, RATES), 6),
            "input_tokens_upper": round(tin_u), "output_tokens_upper": round(n * max_tokens),
            "upper_usd": round(rq.price(tin_u, n * max_tokens, RATES), 6), "out_per_request_likely": out_likely,
            "max_tokens": max_tokens, "note": note}


def search_stage(name: str, n_searches: float, calls_each: float, in_tokens_each: float, out_each: float,
                 max_tokens: int, in_upper_each: float, upper_calls_each: float, n_upper: float,
                 note: str = "") -> dict[str, Any]:
    """A flip-search stage priced from observed per-call usage (likely) and from 48 calls at max_tokens (upper)."""
    n_l, n_u = n_searches * calls_each, n_upper * upper_calls_each
    return {"stage": name, "requests": round(n_l, 1), "input_tokens_likely": round(n_l * in_tokens_each),
            "output_tokens_likely": round(n_l * out_each),
            "likely_usd": round(rq.price(n_l * in_tokens_each, n_l * out_each, RATES), 6),
            "input_tokens_upper": round(n_u * in_upper_each), "output_tokens_upper": round(n_u * max_tokens),
            "upper_usd": round(rq.price(n_u * in_upper_each, n_u * max_tokens, RATES), 6),
            "out_per_request_likely": out_each, "max_tokens": max_tokens, "requests_upper": round(n_u, 1), "note": note}


# ---------------------------------------------------------------------------
# Stored usage
# ---------------------------------------------------------------------------

def observed_usage(results: Path = RESULTS) -> dict[str, Any]:
    """Mean input and output tokens per request of the stages the estimates lean on, from spend.json of the runs."""
    def stages(p: Path) -> dict[str, Any]:
        if not p.exists():
            return {}
        sp = json.loads(p.read_text())
        by = sp.get("by_stage") or {}
        if "arms" in sp:  # the v4 ledger: arms -> by_stage
            by = {k: v for a in sp["arms"].values() for k, v in a["by_stage"].items()}
        return by

    def row(by, stage):
        x = by.get(stage)
        if not x or not x["requests"]:
            return None
        return {"requests": x["requests"], "in_mean": round(x["input_tokens"] / x["requests"], 1),
                "out_mean": round(x["output_tokens"] / x["requests"], 1)}

    v4 = stages(results / "exploratory-v4" / "spend.json")
    v3a, v3b = stages(results / "headline-300" / "spend.json"), stages(results / "headline-u2-v3" / "spend.json")

    def pool(stage):
        rs = [x[stage] for x in (v3a, v3b) if stage in x]
        if not rs:
            return None
        n = sum(r["requests"] for r in rs)
        return {"requests": n, "in_mean": round(sum(r["input_tokens"] for r in rs) / n, 1),
                "out_mean": round(sum(r["output_tokens"] for r in rs) / n, 1)}

    return {"v4": {s: row(v4, s) for s in ("x_A", "x_B", "c_A300", "c_A600", "c_B300", "c_B600", "residue_flip",
                                            "residue_flip_verify", "a_B_stripped")},
            "v3": {s: pool(s) for s in ("residue_A", "residue_B", "residue_flip", "residue_flip_verify", "summary")},
            "note": "v3 rows pool headline-300 and headline-u2-v3, including the 30 replaced U2 cases of the base run"}


def search_profile(root: Path, results: Path = RESULTS) -> dict[str, Any]:
    """Calls, verify calls and rounds per flip search, from the v3 residue records of the 142 frozen flips and the
    193 v4 extractive searches; per-call usage from the stages above."""
    ms = list(load_min_sets(root).values())
    calls = [r["calls_used"] for r in ms]
    rounds = [r["rounds"] for r in ms]
    verifies = sum(r["status"] == "restored" for r in ms)
    src = {"v3_flips": len(ms), "v3_calls_mean": sum(calls) / len(calls), "v3_rounds_max": max(rounds),
           "v3_verify": verifies}
    d = results / "exploratory-v4" / V4_SEARCH_ARM / "state" / "cases"
    c4, r4, v4 = [], [], 0
    if d.exists():
        for p in sorted(d.glob("*.json")):
            cs = json.loads(p.read_text())
            for k, v in cs.items():
                if isinstance(v, dict) and "record" in v:
                    rec = v["record"]
                    c4.append(rec["calls_used"])
                    r4.append(rec["rounds"])
                    v4 += rec["status"] == "restored"
    src.update({"v4_flips": len(c4), "v4_calls_mean": (sum(c4) / len(c4)) if c4 else None,
                "v4_rounds_max": max(r4) if r4 else None, "v4_verify": v4})
    n = len(calls) + len(c4)
    return {**src, "calls_per_search": (sum(calls) + sum(c4)) / n, "verify_per_search": (verifies + v4) / n,
            "rounds_max": max(rounds + r4), "source": "v3 residue records of the frame's flips; v4 A-extractive state"}


def calibration_check(results: Path = RESULTS) -> dict[str, Any]:
    """The 2.2-characters-plus-331 rule against the real input tokens of the v4 stages that have stored batches."""
    out: dict[str, Any] = {}
    for arm in ("A-extractive", "C-budget", "A-stripped"):
        d = results / "exploratory-v4" / arm
        if not (d / "rounds").exists():
            continue
        by: dict[str, list[tuple[int, int]]] = {}
        for r in pilot_requests(d):
            by.setdefault(r["stage"], []).append((r["chars"], r["tin"]))
        for stage, rows in sorted(by.items()):
            pred = [tokens_likely(c) for c, _ in rows]
            up = [tokens_bound(c) for c, _ in rows]
            act = [t for _, t in rows]
            out[f"{arm}/{stage}"] = {
                "n": len(rows), "actual_mean": round(sum(act) / len(act), 1), "predicted_mean": round(sum(pred) / len(pred), 1),
                "ratio_predicted_to_actual": round(sum(pred) / sum(act), 4),
                "mean_abs_rel_error": round(sum(abs(p - a) / a for p, a in zip(pred, act)) / len(act), 4),
                "bound_covers_actual": sum(u >= a for u, a in zip(up, act)) / len(act)}
    return out


# ---------------------------------------------------------------------------
# The arms
# ---------------------------------------------------------------------------

def _chars(params: Sequence[dict[str, Any]]) -> list[int]:
    return [rq.request_chars(p) for p in params]


def arm_stages(root: Path, results: Path = RESULTS) -> dict[str, dict[str, Any]]:
    hs = load_frame(root)
    n = len(hs)
    cfg = {a: E.arm_config(a) for a in E.ARMS}
    dec = cfg["X1"]["decider"]
    nw = cfg["X1"]["note_writer"]
    ao = E.ASSUMED_OUT
    v3_notes = [h.row["v3"]["note"] for h in hs]
    b_proxy = _chars([rq.decide_request(note_view(x), [], h.case, "v1", dec) for h, x in zip(hs, v3_notes)])
    prof = search_profile(root, results)
    use = observed_usage(results)
    arms: dict[str, dict[str, Any]] = {}

    arms["X1"] = {"stages": [
        stage_price("x1_A note writer", _chars([E.x1_request(h.view, h.case, nw) for h in hs]), ao["x1_A"], nw["max_tokens"]),
        stage_price("x1_B decide on the note", b_proxy, ao["x1_B"], dec["max_tokens"],
                    note="note length proxied by the v3 note of the handoff")],
        "rounds_likely": 2, "rounds_upper": 2, "depth": "note, then decision"}

    # X2: writer, B, then the flip search
    v4_in = (use["v4"]["residue_flip"] or {}).get("in_mean") or 1200.0
    v4_out = (use["v4"]["residue_flip"] or {}).get("out_mean") or ao["residue_flip"]
    ver = use["v4"]["residue_flip_verify"] or {}
    all_cands = _chars([rq.decide_request(note_view(h.row["v3"]["note"]),
                                          residue_candidates(h.case, h.view, h.row["v3"]["note"]), h.case, "v1", dec)
                        for h in hs])
    n_l, n_u = FLIP_LIKELY * n, FLIP_UPPER * n
    mean_bound_in = sum(tokens_bound(c) for c in all_cands) / len(all_cands)
    arms["X2"] = {"stages": [
        stage_price("x2_A structured note writer", _chars([E.x2_request(h.view, h.case, nw) for h in hs]), ao["x2_A"],
                    nw["max_tokens"]),
        stage_price("x2_B decide on the note", b_proxy, ao["x2_B"], dec["max_tokens"],
                    note="note length proxied by the v3 note"),
        search_stage("flip search calls", n_l, prof["calls_per_search"],
                     v4_in, v4_out, dec["max_tokens"], mean_bound_in, 48, n_u,
                     note=(f"{n_l:.0f} searches likely ({FLIP_LIKELY:.1%} of 490), {prof['calls_per_search']:.2f} calls each "
                           f"(pooled v3 and v4); upper {n_u:.0f} searches ({FLIP_UPPER:.1%}), 48 calls each at max_tokens, "
                           "every call carrying the note and all candidate units")),
        search_stage("v2 verify calls", n_l, prof["verify_per_search"], ver.get("in_mean") or 1000.0,
                     ver.get("out_mean") or ao["residue_flip_verify"], dec["max_tokens"], mean_bound_in, 1, n_u,
                     note=f"{prof['verify_per_search']:.2f} verifies per search likely; upper one per search")],
        "rounds_likely": 2 + prof["rounds_max"] + 1, "rounds_upper": 2 + 24 + 1,
        "depth": (f"note, decision, then the search: {prof['rounds_max']} search rounds at most in the stored v3 and v4 "
                  "searches, plus one verify round; under the 48-call cap a search round asks at least 2 new sets, "
                  "so at most 24 search rounds"),
        "search": {**prof, "flip_rate_likely": FLIP_LIKELY, "flip_rate_upper": FLIP_UPPER,
                   "absolute_worst_case_usd": round(rq.price(
                       n * 49 * mean_bound_in, n * 49 * dec["max_tokens"], RATES), 4),
                   "absolute_worst_case_note": "every one of the 490 handoffs flips and every search uses 48 calls + 1 verify"}}

    # C1: the exact requests of the sent controls
    ctl = [json.loads(x) for x in (root / "c1_controls.jsonl").read_text().splitlines() if x.strip()]
    sent = [c for c in ctl if c["send"]]
    by_h = {(h.case.case_id, h.system): h for h in hs}
    c1_params = []
    for c in sent:
        h = by_h[(c["case_id"], c["system"])]
        by_id = {u.id: u for u in residue_candidates(h.case, h.view, h.row["v3"]["note"])}
        c1_params.append(rq.decide_request(note_view(h.row["v3"]["note"]), [by_id[i] for i in c["unit_ids"]], h.case,
                                           "v1", dec))
    arms["C1"] = {"stages": [stage_price("c1_B decide on note + irrelevant units", _chars(c1_params), ao["c1_B"],
                                         dec["max_tokens"],
                                         note=f"{len(sent)} sent of {len(ctl)} draws; {sum(not c['send'] for c in ctl)} are not "
                                              f"sent (an empty set or a copy of an earlier control of the same flip); "
                                              f"{sum(not c['matched'] for c in ctl)} draws miss the 10% band (flagged)")],
                  "rounds_likely": 1, "rounds_upper": 1, "depth": "one round", "n_draws": len(ctl)}

    # C2: rewrite, then B
    rw = cfg["C2"]["rewriter"]
    arms["C2"] = {"stages": [
        stage_price("c2_A rewrite", _chars([E.rewrite_request(x, rw) for x in v3_notes]), ao["c2_A"], rw["max_tokens"]),
        stage_price("c2_B decide on the rewrite", b_proxy, ao["c2_B"], dec["max_tokens"],
                    note="rewrite length proxied by the v3 note")],
        "rounds_likely": 2, "rounds_upper": 2, "depth": "rewrite, then decision"}

    # C3: R identical requests per handoff
    rep = cfg["C3"]["repeats"]
    arms["C3"] = {"stages": [stage_price("c3_B decide on the identical v3 note", b_proxy, ao["c3_B"], dec["max_tokens"],
                                         scale=rep, note=f"{rep} repeats per handoff, cache bypassed, distinct custom_ids")],
                  "rounds_likely": 1, "rounds_upper": 1, "depth": "one round"}

    # C4: every (note, unit) pair
    jd = cfg["C4"]["judge"]
    pairs = [rq.request_chars(E.judge_request(h.row["v3"]["note"], u, jd)) for h in hs for u in h.view.units]
    arms["C4"] = {"stages": [stage_price("c4_judge (note, unit) pairs", pairs, ao["c4_judge"], jd["max_tokens"],
                                         note=f"{len(pairs)} pairs; prompt of omission/judge.py")],
                  "rounds_likely": 1, "rounds_upper": 1, "depth": "one round", "n_pairs": len(pairs)}
    return arms


def dryrun(root: Path = OUT, results: Path = RESULTS, write: bool = True) -> dict[str, Any]:
    arms = arm_stages(root, results)
    tot = {"requests": 0.0, "likely_usd": 0.0, "upper_usd": 0.0}
    for a in E.ARMS:
        st = arms[a]["stages"]
        arms[a].update({"requests": round(sum(s["requests"] for s in st)),
                        "likely_usd": round(sum(s["likely_usd"] for s in st), 4),
                        "upper_usd": round(sum(s["upper_usd"] for s in st), 4),
                        "requests_upper": round(sum(s.get("requests_upper", s["requests"]) for s in st))})
        for k in tot:
            tot[k] += arms[a][k]
    st2 = arms["X2"]["stages"]
    srch = [x for x in st2 if x["stage"].startswith(("flip search", "v2 verify"))]
    arms["X2"]["search_priced_separately"] = {
        "requests": round(sum(x["requests"] for x in srch)), "likely_usd": round(sum(x["likely_usd"] for x in srch), 4),
        "upper_usd": round(sum(x["upper_usd"] for x in srch), 4),
        "requests_upper": round(sum(x.get("requests_upper", x["requests"]) for x in srch)),
        "without_search": {"requests": round(sum(x["requests"] for x in st2 if x not in srch)),
                           "likely_usd": round(sum(x["likely_usd"] for x in st2 if x not in srch), 4),
                           "upper_usd": round(sum(x["upper_usd"] for x in st2 if x not in srch), 4)}}
    plus50 = 0.0
    for a in E.ARMS:
        for s in arms[a]["stages"]:
            plus50 += rq.price(s["input_tokens_likely"], s["output_tokens_likely"] * 1.5, RATES)
    res = {
        "schema": "lens.flip.exploratory/1", "kind": "dryrun", "version": "5-exploratory",
        "at": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()),
        "model": "claude-haiku-5-5", "api": "Message Batches (50% rates)",
        "rates_usd_per_million": dict(RATES),
        "rule": {"input_likely": "ceil(chars / 2.2) + 331", "input_upper": "ceil(chars / 1.9020) + 331",
                 "output_likely": "stage means from v3 and v4 usage (engine.ASSUMED_OUT)",
                 "output_upper": "max_tokens on every request"},
        "max_tokens": {"decide": 6000, "note_writers": 6000, "rewriter": E.REWRITER["max_tokens"],
                       "judge": E.JUDGE["max_tokens"]},
        "assumed_output_tokens": dict(E.ASSUMED_OUT),
        "observed_usage": observed_usage(results),
        "calibration_check": calibration_check(results),
        "arms": arms,
        "total": {"requests": round(tot["requests"]), "likely_usd": round(tot["likely_usd"], 4),
                  "upper_usd": round(tot["upper_usd"], 4),
                  "likely_usd_if_outputs_x1_5": round(plus50, 4)},
        "cap_check_usd": CAP_CHECK_USD, "likely_at_or_under_cap_check": tot["likely_usd"] <= CAP_CHECK_USD,
        "api_calls": 0, "count_tokens_calls": 0, "submitted": False,
        "inputs_sha256": {n: sha256_file(root / n) for n in ("frame.jsonl", "cases.jsonl", "v3_min_sets.jsonl",
                                                             "c1_controls.jsonl")},
    }
    if write:
        (root / "dryrun_v5.json").write_text(json.dumps(res, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    return res


def table(res: dict[str, Any]) -> str:
    lines = ["| Arm | Requests | Likely USD | Upper USD | Rounds (likely / upper) |", "| --- | --- | --- | --- | --- |"]
    for a in E.ARMS:
        x = res["arms"][a]
        lines.append(f"| {a} | {x['requests']:,} | {x['likely_usd']:.2f} | {x['upper_usd']:.2f} | "
                     f"{x['rounds_likely']} / {x['rounds_upper']} |")
        if a == "X2":
            sp, ns = x["search_priced_separately"], x["search_priced_separately"]["without_search"]
            lines.append(f"|   of X2: notes and B | {ns['requests']:,} | {ns['likely_usd']:.2f} | {ns['upper_usd']:.2f} | 2 / 2 |")
            lines.append(f"|   of X2: flip search | {sp['requests']:,} | {sp['likely_usd']:.2f} | {sp['upper_usd']:.2f} | "
                         f"{x['rounds_likely'] - 2} / {x['rounds_upper'] - 2} |")
    t = res["total"]
    lines.append(f"| Total | {t['requests']:,} | {t['likely_usd']:.2f} | {t['upper_usd']:.2f} | |")
    return "\n".join(lines)

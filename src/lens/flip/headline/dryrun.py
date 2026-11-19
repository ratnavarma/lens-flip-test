"""Dry run: simulate the whole headline run with the mock model, calibrate input tokens with
count_tokens, price it at Batch rates.

1. Simulation. A copy of the run (config and cases) goes to RUN/sim-<stamp>/ and the real engine
   runs it to the end against FakeBatchAPI and MockModel (mock.py). rag-embed uses the local
   hash embedder there (nomic-embed-text is not called). Every request the real run would send is
   counted by stage, from the engine's round files. No Anthropic call is made.
2. Calibration. A stratified sample of the simulated request bodies (at most 200) goes to
   POST /v1/messages/count_tokens through the wrapper, one at a time. Per request kind
   (decide, summary, note) a line tokens = slope x chars + intercept is fitted. The JSON output
   format adds a constant overhead (331 tokens on every v4 request; assumed for every schema).
   Reuse a calibration with --calibration PATH so a second run dir spends no calls.
3. Prices. Likely: estimated input plus assumed output (v4 mean per decision, 0.8 x B per summary
   step, 300 per note). Cap-limited upper bound: every flip search spends its 48 calls plus the
   verification, and every request reaches max_tokens.

With `calibrate_from_run=PILOT` (calibrate.py) the mock, the output tokens and the input tokens come
from a finished run instead of the v4 guesses, and no count_tokens call is made. The result also holds
a scaling cross-check (the pilot's actual cost per case times N, by stage) and the projected size of
each hypothesis's analysis set against the power rule.
"""

from __future__ import annotations

import json
import math
import shutil
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable

from .. import batch
from . import requests as rq
from .engine import Engine, _read_json, _write_json
from .metrics import power_thresholds, write_report
from .mock import (CalibratedMockModel, FakeBatchAPI, MockModel, default_noise, engine_meta_fn, h01,
                   v4_calibration)

MAX_COUNT_CALLS = 200
STAGE_ORDER = ["a_H", "summary", "a_V", "flip", "flip_verify", "residue_A", "residue_B", "floor", "trunc",
               "residue_flip", "residue_flip_verify"]
TABLE_ROWS = [("a_H", ["a_H"]), ("views (summary steps)", ["summary"]), ("a_V", ["a_V"]),
              ("flip search", ["flip", "flip_verify"]), ("residue A (note)", ["residue_A"]),
              ("residue B", ["residue_B"]), ("floor", ["floor"]), ("trunc", ["trunc"]),
              ("residue flip search", ["residue_flip", "residue_flip_verify"])]
SEARCH_STAGES = {"flip": "flip_verify", "residue_flip": "residue_flip_verify"}
MINUTES_PER_ROUND = (4, 10)


def no_api_runner(method: str, path: str, body: str | None = None, timeout: int = 120) -> tuple[int, str]:
    raise RuntimeError(f"a pilot-calibrated dry run makes no API call ({method} {path})")


def stratum(meta: dict[str, Any], length: int | None) -> str:
    cell = meta.get("cell", "")
    view = "full" if cell == "full" else ("summary" if cell.startswith("summary") else "event")
    return f"{meta['stage']}|{view}|{length}"


class Sampler:
    """Keeps, per stratum, the requests with the smallest hash of their key (deterministic)."""

    def __init__(self, meta_fn: Callable[[], dict], lengths: dict[str, int], per_stratum: int):
        self.meta_fn = meta_fn
        self.lengths = lengths
        self.per = per_stratum
        self.keep: dict[str, list[tuple[float, str, dict, dict]]] = defaultdict(list)

    def add(self, reqs: list[dict]) -> None:
        metas = self.meta_fn()
        for r in reqs:
            m = metas.get(r["custom_id"])
            if not m:
                continue
            s = stratum(m, self.lengths.get(m["case_id"]))
            pri = h01("calib", m["key"])
            lst = self.keep[s]
            if len(lst) < self.per or pri < lst[-1][0]:
                lst.append((pri, r["custom_id"], r["params"], m))
                lst.sort(key=lambda x: x[0])
                del lst[self.per:]

    def picks(self, k: int) -> list[tuple[str, dict, dict, str]]:
        """Round-robin over strata up to k requests."""
        out = []
        strata = sorted(self.keep)
        depth = 0
        while len(out) < k and any(depth < len(self.keep[s]) for s in strata):
            for s in strata:
                if depth < len(self.keep[s]) and len(out) < k:
                    _, cid, params, m = self.keep[s][depth]
                    out.append((cid, params, m, s))
            depth += 1
        return out


def simulate(run_dir: Path, log: Callable[[str], None], per_stratum: int = 12,
             settings: dict[str, Any] | None = None) -> tuple[Path, Sampler, dict]:
    stamp = time.strftime("%Y%m%d-%H%M%S", time.gmtime())
    sim = run_dir / f"sim-{stamp}"
    sim.mkdir(parents=True)
    shutil.copyfile(run_dir / "cases.jsonl", sim / "cases.jsonl")
    cfg = json.loads((run_dir / "config.json").read_text())
    cfg["embedder"] = {"kind": "hash", "model": "hash-bow-256", "note": "dry-run stand-in for nomic-embed-text"}
    cfg["cache"] = "state/calls.sqlite"
    _write_json(sim / "config.json", cfg)
    eng = Engine(sim, check_protocol=False)
    if settings is not None:
        model: MockModel = CalibratedMockModel(eng.cases, settings, seed=int(cfg["seed"]))
    else:
        calib = v4_calibration()
        noise = default_noise(calib)
        model = MockModel(eng.cases, calib, noise, seed=int(cfg["seed"]))
    meta_fn = engine_meta_fn(sim)
    sampler = Sampler(meta_fn, {c.case_id: c.knobs.get("length") for c in eng.cases}, per_stratum)
    eng.runner = FakeBatchAPI(model.answer, meta_fn, on_create=sampler.add)
    t0 = time.time()
    steps = 0
    while True:
        r = eng.step(float("inf"))
        steps += 1
        if r == "complete":
            break
        if r == "submitted":
            rs = eng.round_state()
            log(f"sim round {rs['round']}: {rs['n_requests']} requests")
    eng.close()
    log(f"simulation done in {time.time() - t0:.0f} s")
    if settings is not None:
        return sim, sampler, {"calibrated_from_run": settings["source"]}
    return sim, sampler, {"v4": {k: v for k, v in calib.items() if k != "output_tokens"}, "noise": noise}


def sim_requests(sim: Path) -> list[dict[str, Any]]:
    out = []
    for p in sorted((sim / "rounds").glob("*/round.json")):
        rs = json.loads(p.read_text())
        for cid, m in rs["requests"].items():
            out.append({**m, "custom_id": cid, "round": rs["round"]})
    return out


def calibrate(sampler: Sampler, k: int, runner: Callable, log: Callable[[str], None]) -> dict[str, Any]:
    if k > MAX_COUNT_CALLS:
        raise ValueError(f"at most {MAX_COUNT_CALLS} count_tokens calls")
    picks = sampler.picks(k)
    rows, drop, notes = [], set(), []
    for i, (cid, params, m, s) in enumerate(picks):
        n = batch.count_one(runner, params, drop, notes)
        rows.append({"stratum": s, "kind": m["kind"], "stage": m["stage"], "chars": rq.request_chars(params),
                     "tokens": n})
        if (i + 1) % 20 == 0 or i + 1 == len(picks):
            log(f"count_tokens {i + 1}/{len(picks)}")
    return fit(rows, notes, drop)


def fit(rows: list[dict], notes: list[str] | None = None, drop: set | None = None) -> dict[str, Any]:
    kinds: dict[str, Any] = {}
    for kind in sorted({r["kind"] for r in rows}):
        rs = [r for r in rows if r["kind"] == kind]
        xs, ys = [r["chars"] for r in rs], [r["tokens"] for r in rs]
        n = len(rs)
        mx, my = sum(xs) / n, sum(ys) / n
        sxx = sum((x - mx) ** 2 for x in xs)
        slope = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / sxx if sxx else my / mx
        intercept = my - slope * mx
        ratios = [x / y for x, y in zip(xs, ys) if y]
        kinds[kind] = {"n": n, "slope": slope, "intercept": intercept, "chars_per_token_mean": sum(xs) / sum(ys),
                       "chars_per_token_low": min(ratios) * 0.97, "chars_per_token_high": max(ratios),
                       "overhead_tokens": 331}
    return {"calls": len(rows), "kinds": kinds, "rows": rows, "notes": notes or [],
            "count_tokens_fields_dropped": sorted(drop or []),
            "overhead_note": "usage input minus count_tokens was 331 on all 60 v4 requests (output format "
                             "schema); assumed for the summary and note schemas too"}


def price_requests(reqs: list[dict], calib: dict[str, Any], v4_mean_out: float, rates: dict[str, float],
                   cfg: dict[str, Any], out_by_stage: dict[str, float] | None = None) -> dict[str, Any]:
    """out_by_stage: mean output tokens per stage (and "summary@B"), from a finished run. Without it the
    v4 guesses apply: v4_mean_out per decision, 0.8 x B per summary step, 300 per note."""
    note_out = 300
    by_stage: dict[str, dict[str, float]] = {s: {"requests": 0, "input_tokens": 0, "likely_output_tokens": 0,
                                                 "likely_usd": 0.0, "max_output_usd": 0.0}
                                             for s in STAGE_ORDER}
    searches: dict[tuple, dict[str, Any]] = {}
    for r in reqs:
        est = rq.estimate_input_tokens(r["chars"], r["kind"], calib, bound=False)
        known = None
        if out_by_stage:
            sk = f"summary@{int(r['budget'])}" if r["kind"] == "summary" and r.get("budget") is not None else r["stage"]
            known = out_by_stage.get(sk, out_by_stage.get(r["stage"]))
        if known is not None:
            out = known
        elif r["kind"] == "decide":
            out = v4_mean_out
        elif r["kind"] == "summary":
            out = 0.8 * r["max_tokens"] / cfg["summarizer"]["max_tokens_factor"]
        else:
            out = note_out
        b = by_stage.setdefault(r["stage"], {"requests": 0, "input_tokens": 0, "likely_output_tokens": 0,
                                             "likely_usd": 0.0, "max_output_usd": 0.0})
        b["requests"] += 1
        b["input_tokens"] += est
        b["likely_output_tokens"] += out
        b["likely_usd"] += rq.request_price(est, out, rates)
        b["max_output_usd"] += rq.request_price(est, r["max_tokens"], rates)
        if r["stage"] in SEARCH_STAGES or r["stage"] in SEARCH_STAGES.values():
            base = r["stage"].replace("_verify", "")
            s = searches.setdefault((r["case_id"], r["cell"], base), {"n": 0, "in": 0})
            s["n"] += 1
            s["in"] += est
    return {"by_stage": by_stage, "searches": searches}


def cap_limited(priced: dict[str, Any], searches: list[tuple[str, str, str]], max_calls: int, max_tokens: int,
                rates: dict[str, float]) -> dict[str, float]:
    """Every search spends max_calls calls plus one verification, every request reaches max_tokens.

    searches: (case_id, cell, "flip" or "residue_flip") for every search the simulation ran. A search's
    input size is the mean of its simulated requests (the global search mean when none was sent).
    """
    out = {s: v["max_output_usd"] for s, v in priced["by_stage"].items()}
    found = priced["searches"]
    n_all = sum(v["n"] for v in found.values())
    mean_all = sum(v["in"] for v in found.values()) / n_all if n_all else 0.0
    for base, verify in SEARCH_STAGES.items():
        total = 0.0
        for key in searches:
            if key[2] != base:
                continue
            s = found.get(key)
            mean_in = s["in"] / s["n"] if s else mean_all
            total += (max_calls + 1) * rq.request_price(mean_in, max_tokens, rates)
        out[base] = total
        out[verify] = 0.0
    return out


def scaling_estimate(settings: dict[str, Any], n_target: int) -> dict[str, Any]:
    """Cross-check: the finished run's actual cost by stage, scaled by n_target / its n_cases."""
    by = settings.get("pilot_by_stage", {})
    k = n_target / settings["n_cases"]
    rows = []
    for label, stages in TABLE_ROWS:
        usd = sum(by.get(s, {}).get("usd", 0.0) for s in stages)
        rows.append({"stage": label, "pilot_actual_usd": round(usd, 4), "scaled_usd": round(usd * k, 4),
                     "pilot_requests": sum(by.get(s, {}).get("requests", 0) for s in stages),
                     "scaled_requests": round(sum(by.get(s, {}).get("requests", 0) for s in stages) * k)})
    return {"method": f"pilot actual cost by stage x {n_target}/{settings['n_cases']}",
            "pilot_actual_usd": settings.get("pilot_usd"), "scaled_total_usd": round(sum(r["scaled_usd"] for r in rows), 4),
            "by_stage": rows}


def project_sets(settings: dict[str, Any], n_target: int, thr: dict[str, int],
                 sim_sizes: dict[str, Any] | None) -> dict[str, Any]:
    """Analysis-set sizes at n_target from the pilot's per-case rates, against the power rule, and the N
    each hypothesis needs. H2b also gets exact Poisson bounds on the pilot count, because it is small."""
    from .calibrate import poisson_bounds

    p, nc = settings["analysis_sizes"], settings["n_cases"]

    def row(count: int, need: int) -> dict[str, Any]:
        per = count / nc
        return {"pilot_n": count, "projected_n": round(per * n_target, 1), "min": need,
                "meets_at_target": per * n_target >= need,
                "n_cases_needed": math.ceil(need / per) if per > 0 else None}

    h2a, h2b = row(p["H2a_n"], thr["H2a_min_cases"]), row(p["H2b_n"], thr["H2b_min_flips"])
    pos, neg = row(p["H2c_pos"], thr["H2c_min_positives"]), row(p["H2c_neg"], thr["H2c_min_negatives"])
    lo, hi = poisson_bounds(p["H2b_n"])
    h2b["pilot_count_95_bounds"] = [round(lo, 2), round(hi, 2)]
    h2b["n_cases_needed_range"] = [math.ceil(thr["H2b_min_flips"] / (hi / nc)),
                                   math.ceil(thr["H2b_min_flips"] / (lo / nc)) if lo > 0 else None]
    h2b["funnel_pilot"] = p["H2b_funnel"]
    h2c_need = [x["n_cases_needed"] for x in (pos, neg)]
    out = {"n_target": n_target, "pilot_cases": nc, "H2a": h2a, "H2b": h2b,
           "H2c": {"positives": pos, "negatives": neg,
                   "meets_at_target": pos["meets_at_target"] and neg["meets_at_target"],
                   "n_cases_needed": max(h2c_need) if None not in h2c_need else None}}
    if sim_sizes:
        out["simulated_at_target"] = {k: sim_sizes[k] for k in ("H2a_n", "H2b_n", "H2c_pos", "H2c_neg")}
    return out


def run_dryrun(run_dir: str | Path, *, calibrate_k: int = 120, calibration: str | Path | None = None,
               runner: Callable | None = None, log: Callable[[str], None] = print,
               calibrate_from_run: str | Path | None = None) -> dict[str, Any]:
    from . import calibrate as cal_mod

    d = Path(run_dir)
    cfg = json.loads((d / "config.json").read_text())
    rates = cfg.get("rates", rq.RATES)
    settings = cal_mod.from_run(calibrate_from_run, log) if calibrate_from_run else None
    sim, sampler, mock_info = simulate(d, log, settings=settings)
    reqs = sim_requests(sim)
    if settings is not None:
        calib = {"calls": 0, "calls_this_run": 0, "kinds": settings["input"]["kinds"], "notes": [],
                 "count_tokens_fields_dropped": [], "calibrated_from_run": str(calibrate_from_run),
                 "overhead_note": "input tokens = usage of the finished run; tokens minus 331 fitted on characters"}
    elif calibration:
        calib = json.loads(Path(calibration).read_text())
        calib = {**calib, "reused_from": str(calibration), "calls_this_run": 0}
    else:
        calib = calibrate(sampler, calibrate_k, runner or rq.wrapper_runner, log)
        calib["calls_this_run"] = calib["calls"]
    _write_json(d / "calibration.json", calib)
    summary = write_report(sim, 2)
    spend = _read_json(sim / "spend.json", {})
    n_rounds = len(spend.get("rounds", []))

    v4_mean = mock_info["v4"]["mean_output_tokens"] if settings is None else 0.0
    priced = price_requests(reqs, calib, v4_mean, rates, cfg, settings["output"]["mean"] if settings else None)
    recs = [json.loads(x) for x in (sim / "records.jsonl").read_text().splitlines() if x.strip()]
    rrecs = [json.loads(x) for x in (sim / "residue_records.jsonl").read_text().splitlines() if x.strip()]
    search_keys = ([(r["case_id"], f"{r['system']}@{r['budget']}", "flip") for r in recs]
                   + [(r["case_id"], f"{r['system']}@{r['budget']}", "residue_flip") for r in rrecs])
    cl = cap_limited(priced, search_keys, int(cfg["max_calls"]), int(cfg["decider"]["max_tokens"]), rates)

    table = []
    for label, stages in TABLE_ROWS:
        row = {"stage": label, "requests": sum(priced["by_stage"][s]["requests"] for s in stages),
               "input_tokens": round(sum(priced["by_stage"][s]["input_tokens"] for s in stages)),
               "likely_output_tokens": round(sum(priced["by_stage"][s]["likely_output_tokens"] for s in stages)),
               "likely_usd": round(sum(priced["by_stage"][s]["likely_usd"] for s in stages), 4),
               "cap_limited_usd": round(sum(cl[s] for s in stages), 4)}
        table.append(row)
    likely = sum(r["likely_usd"] for r in table)
    upper = sum(r["cap_limited_usd"] for r in table)

    # Ceiling if every cell flipped and every (case, system) at the residue budget had a residue flip.
    n = int(cfg["n"])
    cells = n * len(cfg["systems"]) * len(cfg["budgets"])
    mean_search_in = (sum(s["in"] for s in priced["searches"].values()) /
                      max(1, sum(s["n"] for s in priced["searches"].values())))
    per_search = (int(cfg["max_calls"]) + 1) * rq.request_price(mean_search_in, int(cfg["decider"]["max_tokens"]), rates)
    non_search = sum(cl[s] for s in ("a_H", "summary", "a_V"))
    res_pairs = n * len(cfg["systems"])
    per_res = sum(priced["by_stage"][s]["max_output_usd"] / max(1, priced["by_stage"][s]["requests"])
                  for s in ("residue_A", "residue_B", "floor", "trunc"))
    ceiling = non_search + cells * per_search + res_pairs * (per_res + per_search)

    pooled = summary["residue"]["pooled"]
    n_flips = pooled.get("n_residue_flips", 0)
    n_elig = pooled.get("n_eligible", 0)
    thr = power_thresholds(cfg)
    sizes = pooled.get("analysis_sizes") or {}
    fs = {k: v["flip_search"] for k, v in summary["cells"].items()}
    searches = sum(v["searches"] for v in fs.values())
    caps = sum(v["cap_hits"] for v in fs.values())
    calls = [r["calls_used"] for r in recs]
    rcalls = [r["calls_used"] for r in rrecs]
    if settings is not None:
        mpr = [settings["minutes_per_round"]["median"], settings["minutes_per_round"]["p90"]]
        mpr_note = "pilot rounds: median and 90th percentile of batch time"
    else:
        mpr, mpr_note = list(MINUTES_PER_ROUND), "assumed"
    out_cost_likely = sum(priced["by_stage"][s]["likely_output_tokens"] for s in priced["by_stage"]) * rates["output"] / 1e6
    flip_cost = sum(priced["by_stage"][s]["likely_usd"] for s in ("flip", "flip_verify"))
    if settings is None:
        sensitivity = {
            "note_output_1000_tokens_usd": round(likely + priced["by_stage"]["residue_A"]["requests"] * 700
                                                 * rates["output"] / 1e6, 4),
            "decide_output_x1_5_usd": round(likely + sum(
                priced["by_stage"][s]["requests"] for s in STAGE_ORDER if s not in ("summary", "residue_A"))
                * 0.5 * v4_mean * rates["output"] / 1e6, 4)}
    else:
        sensitivity = {"output_tokens_x1_25_usd": round(likely + 0.25 * out_cost_likely, 4),
                       "flip_search_requests_x1_25_usd": round(likely + 0.25 * flip_cost, 4)}
    result = {
        "run_dir": str(d), "n": cfg["n"], "seed": cfg["seed"], "sim_dir": str(sim),
        "rates_usd_per_million": rates, "model": cfg["decider"]["model"],
        "assumptions": {
            "decide_output_tokens": round(v4_mean, 1),
            "decide_output_source": "mean usage output_tokens of the 60 adaptive-high requests in results/validity-haiku-v4",
            "summary_output_tokens": "0.8 x B per step",
            "note_output_tokens": "300 per note, thinking included (assumption; see sensitivity)",
            "input_tokens": "chars x slope + intercept per kind from count_tokens, plus 331 for the output format",
            "mock": mock_info,
            "rag_embed_in_sim": "hash bag-of-words stand-in (local); the real run uses nomic-embed-text",
        } if settings is None else {
            "calibrated_from_run": str(calibrate_from_run),
            "output_tokens": "mean output tokens per stage of the finished run (summary steps per budget)",
            "input_tokens": "chars x slope + intercept per kind, fitted on the finished run's usage, plus 331",
            "mock": mock_info,
            "rag_embed_in_sim": "hash bag-of-words stand-in (local); the real run uses nomic-embed-text",
        },
        "calibration": {k: v for k, v in calib.items() if k != "rows"},
        "requests_total": len(reqs),
        "by_stage": table,
        "likely_usd": round(likely, 4),
        "cap_limited_upper_usd": round(upper, 4),
        "ceiling_every_cell_flips_usd": round(ceiling, 4),
        "sensitivity": sensitivity,
        "rounds": n_rounds,
        "minutes_per_round": [round(m, 1) for m in mpr], "minutes_per_round_source": mpr_note,
        "wall_clock_hours": [round(n_rounds * m / 60, 1) for m in mpr],
        "flip_searches": {"n": searches, "cap_hits": caps, "mean_calls": round(sum(calls) / len(calls), 1) if calls else None,
                          "by_cell": {k: v["searches"] for k, v in fs.items()},
                          "residue_searches": len(rrecs), "residue_cap_hits": sum(r["hit_cap"] for r in rrecs),
                          "residue_mean_calls": round(sum(rcalls) / len(rcalls), 1) if rcalls else None},
        "residue": {"n_eligible": n_elig, "n_residue_flips": n_flips, "power_rule": thr,
                    "analysis_sets": sizes,
                    "meets_power_rule": {
                        "H2a": sizes.get("H2a_n", 0) >= thr["H2a_min_cases"],
                        "H2b": sizes.get("H2b_n", 0) >= thr["H2b_min_flips"],
                        "H2c": sizes.get("H2c_pos", 0) >= thr["H2c_min_positives"]
                        and sizes.get("H2c_neg", 0) >= thr["H2c_min_negatives"]},
                    "projected_flips_at_rates": {f"{int(r * 100)}%": round(n_elig * r, 1) for r in (0.05, 0.1, 0.2, 0.3)},
                    "sim_residue_rate": (pooled.get("residue_rate") or {}).get("value"),
                    "sim_floor_rate": (pooled.get("floor_rate") or {}).get("value")},
        "sim_validity": summary["validity"]["overall"],
        "no_batch_submitted": True,
    }
    if settings is not None:
        result["calibrated_settings"] = settings
        result["projection_at_n"] = project_sets(settings, int(cfg["n"]), thr, sizes or None)
        result["cross_check"] = scaling_estimate(settings, int(cfg["n"]))
        cmp: dict[str, Any] = {}
        for cell, st in settings["search"]["cells"].items():
            sim_fs = fs.get(cell, {})
            nrows = summary["cells"].get(cell, {}).get("n") or 0
            cmp[cell] = {"pilot_flip_rate": settings["a_V"]["flip_rate"].get(cell),
                         "sim_flip_rate": sim_fs.get("searches", 0) / nrows if nrows else None,
                         "pilot_mean_calls": st["mean_calls"],
                         "sim_mean_calls": (sim_fs.get("mean_calls") or {}).get("value"),
                         "pilot_cap_rate": st["cap_rate"],
                         "sim_cap_rate": sim_fs.get("cap_hits", 0) / sim_fs["searches"] if sim_fs.get("searches") else None}
        result["sim_vs_pilot_by_cell"] = cmp
    _write_json(d / "dryrun.json", result)
    return result


def format_tables(res: dict[str, Any]) -> str:
    L = [f"dry run for {res['run_dir']} (N = {res['n']}, seed {res['seed']}); simulated requests {res['requests_total']}",
         f"{'stage':<24}{'requests':>9}{'input tok':>13}{'likely out':>12}{'likely $':>10}{'cap-limited $':>15}"]
    for r in res["by_stage"]:
        L.append(f"{r['stage']:<24}{r['requests']:>9}{r['input_tokens']:>13}{r['likely_output_tokens']:>12}"
                 f"{r['likely_usd']:>10.2f}{r['cap_limited_usd']:>15.2f}")
    L.append(f"{'total':<24}{res['requests_total']:>9}{'':>13}{'':>12}{res['likely_usd']:>10.2f}"
             f"{res['cap_limited_upper_usd']:>15.2f}")
    L.append(f"ceiling if every cell flips and every residue pair flips: ${res['ceiling_every_cell_flips_usd']:.2f}")
    lo, hi = res.get("minutes_per_round", MINUTES_PER_ROUND)[:2]
    L.append(f"rounds {res['rounds']}; wall clock {res['wall_clock_hours'][0]} to {res['wall_clock_hours'][1]} hours "
             f"at {lo} to {hi} minutes per round ({res.get('minutes_per_round_source', 'assumed')})")
    fs = res["flip_searches"]
    L.append(f"flip searches {fs['n']}, cap hits {fs['cap_hits']}, mean calls {fs['mean_calls']}")
    r = res["residue"]
    m = r["meets_power_rule"]
    sz = r.get("analysis_sets") or {}
    L.append(f"residue: eligible {r['n_eligible']}, simulated flips {r['n_residue_flips']}; power rule per hypothesis: "
             f"H2a n={sz.get('H2a_n')} (min {r['power_rule']['H2a_min_cases']}, {'met' if m['H2a'] else 'NOT met'}); "
             f"H2b n={sz.get('H2b_n')} (min {r['power_rule']['H2b_min_flips']}, {'met' if m['H2b'] else 'NOT met'}); "
             f"H2c {sz.get('H2c_pos')} flips and {sz.get('H2c_neg')} non-flips (min {r['power_rule']['H2c_min_positives']} "
             f"each, {'met' if m['H2c'] else 'NOT met'})")
    c = res["calibration"]
    if "kinds" in c:
        L.append("calibration: " + "; ".join(f"{k} n={v['n']} {v['chars_per_token_mean']:.2f} chars/token "
                                              f"(low {v['chars_per_token_low']:.2f})" for k, v in c["kinds"].items())
                 + f"; count_tokens calls this run {c.get('calls_this_run', 0)}")
    L.append("sensitivity: " + "; ".join(f"{k.replace('_usd', '').replace('_', ' ')} ${v:.2f}"
                                         for k, v in res["sensitivity"].items()))
    if "cross_check" in res:
        x = res["cross_check"]
        L.append("")
        L.append(f"cross-check: {x['method']} (pilot actual ${x['pilot_actual_usd']:.2f}); scaled total "
                 f"${x['scaled_total_usd']:.2f}")
        for row in x["by_stage"]:
            L.append(f"  {row['stage']:<24}{row['pilot_actual_usd']:>8.2f}{row['scaled_usd']:>10.2f}")
        pj = res["projection_at_n"]
        L.append("")
        L.append(f"analysis sets at N = {pj['n_target']} from the pilot rates ({pj['pilot_cases']} cases)")
        for name in ("H2a", "H2b"):
            h = pj[name]
            L.append(f"  {name}: pilot {h['pilot_n']}, projected {h['projected_n']}, min {h['min']}, "
                     f"cases needed {h['n_cases_needed']}")
        h = pj["H2c"]
        L.append(f"  H2c: positives pilot {h['positives']['pilot_n']} -> {h['positives']['projected_n']}, negatives "
                 f"pilot {h['negatives']['pilot_n']} -> {h['negatives']['projected_n']}, cases needed {h['n_cases_needed']}")
    L.append("no batch was submitted")
    return "\n".join(L)

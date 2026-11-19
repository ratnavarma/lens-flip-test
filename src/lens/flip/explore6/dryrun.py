"""Price both v6 arms without a model call. Arithmetic on the exact request bodies and stored usage.

Inputs (read only): the v6 frame, the extractive flips, the control draws, the v3 minimal sets, and the usage of
the finished v4 and v5 runs. No network, no count_tokens call, no Batch submission.

Input tokens: the v5 rule. likely = ceil(chars / 2.2) + 331; upper bound = ceil(chars / 1.9020) + 331. Output
tokens: likely = the stored mean of the calls each stage repeats (engine.ASSUMED_OUT); upper bound = max_tokens.
Every request body is built here exactly as the engine builds it, so the request counts are exact and the upper
bound of an arm equals the upper bound the engine's guard computes for the arm's one round.

Guard preview. The run goes E1 then P1 with one round each. E1's bound must fit the cap. P1's bound must fit the
cap minus what E1 spent; the preview uses E1's likely price and E1's likely price with outputs 50 percent longer.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

from ..explore5 import dryrun as D
from ..headline import requests as rq
from ..residue import residue_candidates
from . import SCHEMA, VERSION
from . import engine as E
from . import frame as F
from . import protocol as PR
from .frame import OUT, RESULTS, load_frame, sha256_file

NAME = PR.DRYRUN_NAME


def arm_requests(root: Path = OUT) -> dict[str, dict[str, list[dict[str, Any]]]]:
    """Every request body of both arms, by arm and stage. A re-read body appears once per re-read."""
    hs = load_frame(root)
    e1, p1 = F.load_e1_flips(root), F.load_p1_sets(root)
    ctl = F.load_controls(root / F.E1_CONTROLS)
    dec = {a: E.arm_config(a)["decider"] for a in E.ARMS}
    reps = E.arm_config("E1")["rereads"]
    out: dict[str, dict[str, list[dict[str, Any]]]] = {"E1": {"e1_ctl_B": [], "e1_fresh_B": []}, "P1": {"p1_fresh_B": []}}
    for h in hs:
        k = (h.case.case_id, h.system)
        rec = e1.get(k)
        if rec is not None:
            cands = residue_candidates(h.case, h.view, rec["note"])
            for c in ctl.get(k, []):
                if c["send"]:
                    out["E1"]["e1_ctl_B"].append(E.restoring_request(rec["note"], F.units_in_order(cands, c["unit_ids"]),
                                                                     h, dec["E1"]))
            if rec["status"] == "restored":
                body = E.restoring_request(rec["note"], F.units_in_order(cands, rec["unit_ids"]), h, dec["E1"])
                out["E1"]["e1_fresh_B"] += [body] * reps
        rec = p1.get(k)
        if rec is not None and rec["status"] == "restored":
            note = h.row["v3"]["note"]
            cands = residue_candidates(h.case, h.view, note)
            body = E.restoring_request(note, F.units_in_order(cands, rec["unit_ids"]), h, dec["P1"])
            out["P1"]["p1_fresh_B"] += [body] * reps
    return out


def stored_usage(results: Path = RESULTS) -> dict[str, Any]:
    """Mean tokens per request of the stored stages the estimates lean on."""
    def stage(path: Path, arm: str, name: str) -> dict[str, Any] | None:
        if not path.exists():
            return None
        x = ((json.loads(path.read_text()).get("arms") or {}).get(arm) or {}).get("by_stage", {}).get(name)
        if not x or not x["requests"]:
            return None
        return {"requests": x["requests"], "in_mean": round(x["input_tokens"] / x["requests"], 1),
                "out_mean": round(x["output_tokens"] / x["requests"], 1)}
    v3 = D.observed_usage(results)["v3"].get("residue_flip")
    return {"v4 A-extractive residue_flip": stage(results / "exploratory-v4" / "spend.json", "A-extractive", "residue_flip"),
            "v5 C1 c1_B": stage(results / "exploratory-v5" / "spend.json", "C1", "c1_B"),
            "v3 residue_flip": v3}


def dryrun(root: Path = OUT, results: Path = RESULTS, write: bool = True, cap_usd: float = PR.RUN_CAP_USD) -> dict[str, Any]:
    reqs = arm_requests(root)
    ctl = F._read_jsonl(root / F.E1_CONTROLS)
    arms: dict[str, Any] = {}
    plus50 = 0.0
    for a in E.ARMS:
        cfg = E.arm_config(a)
        mt = cfg["decider"]["max_tokens"]
        stages = [D.stage_price(st, D._chars(bodies), E.ASSUMED_OUT[st], mt) for st, bodies in reqs[a].items()]
        arms[a] = {"stages": stages, "requests": round(sum(s["requests"] for s in stages)),
                   "likely_usd": round(sum(s["likely_usd"] for s in stages), 4),
                   "upper_usd": round(sum(s["upper_usd"] for s in stages), 4),
                   "likely_usd_if_outputs_x1_5": round(sum(rq.price(s["input_tokens_likely"], s["output_tokens_likely"] * 1.5,
                                                                    E.RATES) for s in stages), 4),
                   "rounds_likely": 1, "rounds_upper": 2, "depth": "one round; a second only to retry errored requests"}
        plus50 += arms[a]["likely_usd_if_outputs_x1_5"]
    e, p = arms["E1"], arms["P1"]
    guard = {"cap_usd": cap_usd, "order": list(E.RUN_ORDER),
             "E1_round_upper_usd": e["upper_usd"], "E1_fits": e["upper_usd"] <= cap_usd,
             "P1_round_upper_usd": p["upper_usd"],
             "P1_fits_after_E1_likely": e["likely_usd"] + p["upper_usd"] <= cap_usd,
             "P1_fits_after_E1_outputs_x1_5": e["likely_usd_if_outputs_x1_5"] + p["upper_usd"] <= cap_usd}
    guard["fits"] = bool(guard["E1_fits"] and guard["P1_fits_after_E1_outputs_x1_5"])
    tot_l = round(e["likely_usd"] + p["likely_usd"], 4)
    res = {
        "schema": SCHEMA, "kind": "dryrun", "version": VERSION,
        "at": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()),
        "model": E.HAIKU, "api": "Message Batches (50% rates)", "rates_usd_per_million": dict(E.RATES),
        "rule": {"input_likely": "ceil(chars / 2.2) + 331", "input_upper": "ceil(chars / 1.9020) + 331",
                 "output_likely": "stored stage means (engine.ASSUMED_OUT)", "output_upper": "max_tokens on every request"},
        "assumed_output_tokens": dict(E.ASSUMED_OUT), "stored_usage": stored_usage(results),
        "arms": arms,
        "e1_controls": {"drawn": len(ctl), "sent": sum(c["send"] for c in ctl), "matched": sum(c["matched"] for c in ctl),
                        "matched_and_sent": sum(c["send"] and c["matched"] for c in ctl),
                        "copies": sum(c["duplicate"] for c in ctl)},
        "total": {"requests": e["requests"] + p["requests"], "likely_usd": tot_l,
                  "upper_usd": round(e["upper_usd"] + p["upper_usd"], 4), "likely_usd_if_outputs_x1_5": round(plus50, 4)},
        "price_gate_usd": PR.PRICE_GATE_USD, "likely_at_or_under_price_gate": tot_l <= PR.PRICE_GATE_USD,
        "guard": guard,
        "api_calls": 0, "count_tokens_calls": 0, "submitted": False,
        "inputs_sha256": {name: sha256_file(root / name) for name in PR.RUN_FILES.values()},
    }
    if write:
        (root / NAME).write_text(json.dumps(res, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    return res


def table(res: dict[str, Any]) -> str:
    lines = ["| Arm | Stage | Requests | Likely USD | Upper USD |", "| --- | --- | --- | --- | --- |"]
    for a in E.ARMS:
        for s in res["arms"][a]["stages"]:
            lines.append(f"| {a} | {s['stage']} | {s['requests']:,.0f} | {s['likely_usd']:.3f} | {s['upper_usd']:.3f} |")
    t = res["total"]
    lines.append(f"| Total | | {t['requests']:,} | {t['likely_usd']:.2f} | {t['upper_usd']:.2f} |")
    return "\n".join(lines)

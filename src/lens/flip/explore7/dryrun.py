"""Price both v7 studies without a model call. Arithmetic on request bodies and stored usage.

Inputs (read only): the v7 frame, the A-wrong frame, the text-rule notes, and the usage of the finished v3 and
v5 runs. No network, no count_tokens call, no Batch submission.

Exact bodies: every R request (the notes are known), and every note-writer request of X. Stand-ins: a B request
of X reads a note that does not exist yet, so its body is built with a stand-in note. For X1p and X1r that is
the v3 note of the handoff. For XW it is the longest v3 note of the frame (every note is cut at 150 tokens, so
no real note is longer).

Input tokens: the v5 rule. likely = ceil(chars / 2.2) + 331; upper bound = ceil(chars / 1.9020) + 331. Output
tokens: likely = the stored mean of the calls each stage repeats (engine.ASSUMED_OUT); upper bound = max_tokens.

Guard preview. A round holds at most engine.MAX_REQUESTS_PER_ROUND requests. The preview is deliberately crude
and safe: the largest possible round bound (the costliest requests of a study, up to the round size) must fit
the cap after every request of the run has been paid at the likely price with outputs 50 percent longer.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

from ..explore5 import dryrun as D5
from ..headline import requests as rq
from . import SCHEMA, VERSION
from . import engine as E
from . import frame as F
from . import protocol as PR
from .frame import OUT, RESULTS, load_frame, sha256_file

NAME = PR.DRYRUN_NAME


def arm_requests(root: Path = OUT) -> dict[str, dict[str, list[dict[str, Any]]]]:
    """Request bodies by run dir and stage. R bodies are exact and deduplicated by key, as the engine does."""
    hs = load_frame(root)
    wrong = F.load_wrong(root)
    notes = F.by_key(F._read_jsonl(root / F.R_NOTES))
    dec = E.arm_config("R")["decider"]
    nw = E.arm_config("X")["note_writer"]
    out: dict[str, dict[str, list[dict[str, Any]]]] = {a: {s: [] for sub in E.SUB_ARMS if sub[0] == a for s in E.STAGES[sub]}
                                                       for a in E.ARMS}
    seen: set[str] = set()
    longest = max((h.row["v3"]["note"] for h in hs), key=len)
    for h in hs:
        rn = notes[(h.case.case_id, h.system)]
        v3 = h.row["v3"]["note"]
        bodies = [("r1_B", E.r1_request(v3, h, dec))]
        bodies += [(f"{arm}_B", E.v1_request(rn[f"{arm}_note"], h, dec)) for arm in ("r2", "r3")
                   if rn[f"{arm}_changed"] and not rn[f"{arm}_empty"]]
        for stage, body in bodies:
            k = rq.request_key(body)
            if k not in seen:
                seen.add(k)
                out["R"][stage].append(body)
        out["X"]["x1p_A"].append(E.x1p_note_request(h, nw))
        out["X"]["x1r_A"].append(E.x1r_note_request(h, nw))
        out["X"]["x1p_B"].append(E.v1_request(v3, h, dec))
        out["X"]["x1r_B"].append(E.v1_request(v3, h, dec))
    for h in wrong:
        out["X"]["xw_v3_A"].append(E.v3_note_request(h, nw))
        out["X"]["xw_x1_A"].append(E.x1_note_request(h, nw))
        out["X"]["xw_v3_B"].append(E.v1_request(longest, h, dec))
        out["X"]["xw_x1_B"].append(E.v1_request(longest, h, dec))
    return out


def stored_usage(results: Path = RESULTS) -> dict[str, Any]:
    use = D5.observed_usage(results)

    def v5(arm: str, name: str) -> dict[str, Any] | None:
        p = results / "exploratory-v5" / "spend.json"
        if not p.exists():
            return None
        x = ((json.loads(p.read_text()).get("arms") or {}).get(arm) or {}).get("by_stage", {}).get(name)
        if not x or not x["requests"]:
            return None
        return {"requests": x["requests"], "in_mean": round(x["input_tokens"] / x["requests"], 1),
                "out_mean": round(x["output_tokens"] / x["requests"], 1)}
    return {"v3 residue_A": use["v3"].get("residue_A"), "v3 residue_B": use["v3"].get("residue_B"),
            "v5 X1 x1_A": v5("X1", "x1_A"), "v5 X1 x1_B": v5("X1", "x1_B"), "v5 C3 c3_B": v5("C3", "c3_B")}


def dryrun(root: Path = OUT, results: Path = RESULTS, write: bool = True, cap_usd: float = PR.RUN_CAP_USD) -> dict[str, Any]:
    reqs = arm_requests(root)
    limit = E.MAX_REQUESTS_PER_ROUND
    arms: dict[str, Any] = {}
    worst_round = 0.0
    for a in E.ARMS:
        cfg = E.arm_config(a)
        stages, bounds = [], []
        for st, bodies in reqs[a].items():
            mt = bodies[0]["max_tokens"] if bodies else cfg["decider"]["max_tokens"]
            stages.append(D5.stage_price(st, D5._chars(bodies), E.ASSUMED_OUT[st], mt))
            bounds += [rq.price(D5.tokens_bound(c), mt, E.RATES) for c in D5._chars(bodies)]
        n = sum(len(b) for b in reqs[a].values())
        round_bound = sum(sorted(bounds, reverse=True)[:limit])
        worst_round = max(worst_round, round_bound)
        arms[a] = {"stages": stages, "requests": n,
                   "likely_usd": round(sum(s["likely_usd"] for s in stages), 4),
                   "upper_usd": round(sum(s["upper_usd"] for s in stages), 4),
                   "likely_usd_if_outputs_x1_5": round(sum(rq.price(s["input_tokens_likely"], s["output_tokens_likely"] * 1.5,
                                                                    E.RATES) for s in stages), 4),
                   "largest_round_upper_usd": round(round_bound, 4),
                   "rounds_at_least": -(-n // limit) if a == "R" else max(2, -(-n // limit)),
                   "depth": "one step" if a == "R" else "note, then decision"}
    by_sub = {}
    for sub in E.SUB_ARMS:
        st = [s for a in E.ARMS for s in arms[a]["stages"] if s["stage"] in E.STAGES[sub]]
        by_sub[sub] = {"requests": round(sum(s["requests"] for s in st)), "likely_usd": round(sum(s["likely_usd"] for s in st), 4),
                       "upper_usd": round(sum(s["upper_usd"] for s in st), 4)}
    tot_l = round(sum(arms[a]["likely_usd"] for a in E.ARMS), 4)
    tot_15 = round(sum(arms[a]["likely_usd_if_outputs_x1_5"] for a in E.ARMS), 4)
    guard = {"cap_usd": cap_usd, "order": list(E.RUN_ORDER), "max_requests_per_round": limit,
             "largest_round_upper_usd": round(worst_round, 4), "all_paid_at_outputs_x1_5_usd": tot_15,
             "fits": tot_15 + worst_round <= cap_usd,
             "rule": "all requests paid at the likely price with outputs x1.5, plus the largest possible round bound, at or under the cap"}
    res = {
        "schema": SCHEMA, "kind": "dryrun", "version": VERSION,
        "at": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()),
        "model": E.HAIKU, "api": "Message Batches (50% rates)", "rates_usd_per_million": dict(E.RATES),
        "rule": {"input_likely": "ceil(chars / 2.2) + 331", "input_upper": "ceil(chars / 1.9020) + 331",
                 "output_likely": "stored stage means (engine.ASSUMED_OUT)", "output_upper": "max_tokens on every request",
                 "stand_ins": "B requests of X are priced with a stand-in note (see the module text)"},
        "assumed_output_tokens": dict(E.ASSUMED_OUT), "stored_usage": stored_usage(results),
        "arms": arms, "by_sub_arm": by_sub,
        "total": {"requests": sum(arms[a]["requests"] for a in E.ARMS), "likely_usd": tot_l,
                  "upper_usd": round(sum(arms[a]["upper_usd"] for a in E.ARMS), 4), "likely_usd_if_outputs_x1_5": tot_15},
        "price_gate_usd": PR.PRICE_GATE_USD, "likely_at_or_under_price_gate": tot_l <= PR.PRICE_GATE_USD,
        "guard": guard,
        "api_calls": 0, "count_tokens_calls": 0, "submitted": False,
        "inputs_sha256": {name: sha256_file(root / name) for name in PR.RUN_FILES.values()},
    }
    if write:
        (root / NAME).write_text(json.dumps(res, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    return res


def table(res: dict[str, Any]) -> str:
    lines = ["| Arm | Requests | Likely USD | Upper USD |", "| --- | --- | --- | --- |"]
    for sub, x in res["by_sub_arm"].items():
        lines.append(f"| {sub} | {x['requests']:,} | {x['likely_usd']:.3f} | {x['upper_usd']:.3f} |")
    t = res["total"]
    lines.append(f"| Total | {t['requests']:,} | {t['likely_usd']:.2f} | {t['upper_usd']:.2f} |")
    return "\n".join(lines)

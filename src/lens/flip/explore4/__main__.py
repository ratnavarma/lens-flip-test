"""CLI: uv run python -m lens.flip.explore4 <command>

frame | freeze | estimate ARM | init ARM | run ARM | probe-sonnet | local-gate | local-run | report
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from ..headline import requests as rq
from ..headline.engine import Refused, _read_json, _write_json
from ..residue import note_view
from ..schema import View
from . import engine as E
from . import prompts as P
from .frame import OUT, build_frame, load_frame


def estimate(arm: str, root: Path = OUT, search_share: float = 0.30, search_calls: float = 8.0) -> dict:
    """Pre-study estimate at Batch rates: calls, input and output tokens, likely and upper dollars."""
    cfg = E.arm_config(arm, _assumed(root, arm))
    calib = json.loads(Path("results/headline-300/calibration.json").read_text())
    rates = cfg["rates"]
    hs = load_frame(root)
    parts: list[dict] = []

    def add(name: str, reqs: list[dict], kind: str, n_scale: float = 1.0, note: str = ""):
        tin_l = sum(rq.estimate_input_tokens(rq.request_chars(p), kind, calib, bound=False) for p in reqs) * n_scale
        tin_u = sum(rq.estimate_input_tokens(rq.request_chars(p), kind, calib, bound=True) for p in reqs) * n_scale
        n = len(reqs) * n_scale
        out_l = n * cfg["assumed_output_tokens"].get(kind, 1000)
        out_u = sum(p["max_tokens"] for p in reqs) * n_scale
        parts.append({"stage": name, "calls": round(n, 1), "input_tokens_likely": round(tin_l),
                      "output_tokens_likely": round(out_l), "input_tokens_upper": round(tin_u),
                      "output_tokens_upper": round(out_u), "likely_usd": round(rq.price(tin_l, out_l, rates), 4),
                      "upper_usd": round(rq.price(tin_u, out_u, rates), 4), "note": note})

    dummy = lambda k: "x" * int(k * 2.2)  # a note of k tokens
    dec = cfg["decider"]
    if arm == "A-extractive":
        add("extract (A)", [E.extract_request(h.view, h.case, cfg["note_writer"]) for h in hs], "extract")
        add("B on extractive note", [rq.decide_request(note_view(dummy(150)), [], h.case, "v1", dec) for h in hs],
            "decide", note="note length assumed at the 150-token cap")
        add("flip searches", [rq.decide_request(h.view, [], h.case, "v1", dec) for h in hs], "decide",
            n_scale=search_share * search_calls,
            note=f"likely: {search_share:.0%} of handoffs flip, {search_calls} calls per search (note + inserted units "
                 "priced as V). Worst case is 490 x 49 calls; the per-round guard bounds it.")
    elif arm == "B-sonnet":
        add("Sonnet B on v3 note", [rq.decide_request(note_view(h.row["v3"]["note"]), [], h.case, "v1", dec) for h in hs],
            "decide")
    elif arm == "B-sonnet-view":
        add("Sonnet on V (v2)", [rq.decide_request(h.view, [], h.case, "v2", dec) for h in hs], "decide")
    elif arm == "C-budget":
        for b in P.NOTE_BUDGETS:
            add(f"note A at {b}", [E.budget_note_request(h.view, h.case, cfg["note_writer"], b) for h in hs], "note_budget")
            add(f"B on {b}-token note", [rq.decide_request(note_view(dummy(b)), [], h.case, "v1", dec) for h in hs], "decide",
                note="note length assumed at the cap")
    elif arm == "A-stripped":
        reqs = []
        for h in hs:
            note, dropped = P.strip_advice(h.row["v3"]["note"])
            if dropped:
                reqs.append(rq.decide_request(note_view(note), [], h.case, "v1", dec))
        add("B on stripped note (changed notes only)", reqs, "decide",
            note=f"{len(reqs)} of {len(hs)} notes change; unchanged notes keep the v3 answer")
    tot = {k: round(sum(p[k] for p in parts), 4) for k in ("calls", "input_tokens_likely", "output_tokens_likely",
                                                            "input_tokens_upper", "output_tokens_upper",
                                                            "likely_usd", "upper_usd")}
    est = {"arm": arm, "model": dec["model"], "rates_usd_per_million": rates, "parts": parts, "total": tot,
           "assumed_output_tokens": cfg["assumed_output_tokens"],
           "at": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime())}
    path = root / "cost_estimate.json"
    allest = _read_json(path, {"cap_usd": E.STUDY_CAP_USD, "studies": {}})
    allest["studies"][arm] = est
    led = E.update_ledger(root)
    allest["ledger_total_at_last_estimate_usd"] = led["total_usd"]
    _write_json(path, allest)
    return est


def _assumed(root: Path, arm: str):
    if arm.startswith("B-sonnet"):
        probe = _read_json(root / "sonnet_probe.json", None)
        if probe:
            return {"decide": int(probe["assumed_decide_output_tokens"])}
    return None


def probe_sonnet(root: Path = OUT) -> dict:
    """One synchronous Sonnet decision on one frozen note, with the arm's exact request settings, to confirm
    the API accepts them and to measure output length. Logged in the ledger as a probe."""
    if (root / "sonnet_probe.json").exists():
        return json.loads((root / "sonnet_probe.json").read_text())
    from .protocol import check
    if check(root):
        raise SystemExit("v4 protocol check failed")
    led = E.update_ledger(root)
    if led["total_usd"] + 0.07 > E.STUDY_CAP_USD:
        raise SystemExit("no room for the probe")
    h = load_frame(root)[0]
    cfg = E.arm_config("B-sonnet")
    params = rq.decide_request(note_view(h.row["v3"]["note"]), [], h.case, "v1", cfg["decider"])
    rc, out = rq.wrapper_runner("POST", "/v1/messages", rq.canonical(params), 300)
    resp = json.loads(out)
    if rc != 0:
        raise SystemExit(f"probe failed: {json.dumps(resp.get('error'))[:300]}")
    usage = resp.get("usage", {})
    tin, tout = rq.usage_tokens(usage)
    usd = round(rq.price(tin, tout, {"input": 2.0, "output": 10.0}), 6)  # list price: the probe is not batched
    text = next((b.get("text") for b in resp.get("content", []) if b.get("type") == "text"), None)
    rec = {"case_id": h.case.case_id, "system": h.system, "stop_reason": resp.get("stop_reason"),
           "input_tokens": tin, "output_tokens": tout, "usd_list_price": usd, "reply": text,
           "assumed_decide_output_tokens": max(1500, int(round(tout * 2.0, -2)))}
    (root / "sonnet_probe.json").write_text(json.dumps(rec, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    E.add_probe(root, {"what": "sonnet sync probe (1 call, list price)", "usd": usd, "input_tokens": tin,
                       "output_tokens": tout})
    return rec


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m lens.flip.explore4")
    ap.add_argument("command")
    ap.add_argument("arm", nargs="?")
    ap.add_argument("--poll", type=float, default=60.0)
    a = ap.parse_args(argv)
    root = OUT
    if a.command == "frame":
        print(json.dumps(build_frame(root), indent=1))
    elif a.command == "freeze":
        from .protocol import freeze
        print(json.dumps(freeze(root), indent=1))
    elif a.command == "estimate":
        print(json.dumps(estimate(a.arm, root)["total"], indent=1))
    elif a.command == "init":
        print(E.init_arm(a.arm, root, _assumed(root, a.arm)))
    elif a.command == "run":
        try:
            r = E.run_arm(root / a.arm, poll=a.poll, log=lambda m: print(m, flush=True), root=root)
        except Refused as exc:
            print(f"REFUSED: {exc}", flush=True)
            E.update_ledger(root)
            return 3
        print(f"{a.arm}: {r}; ledger ${E.update_ledger(root)['total_usd']:.4f}", flush=True)
    elif a.command == "probe-sonnet":
        print(json.dumps(probe_sonnet(root), indent=1))
    elif a.command == "local-gate":
        from .local import run_gate
        g = run_gate(root, log=lambda m: print(m, flush=True))
        print(f"gate {g['correct']}/{g['n']} passes={g['passes']}")
    elif a.command == "local-run":
        from .local import run_reader
        print(run_reader(root, log=lambda m: print(m, flush=True)))
    elif a.command == "report":
        from .analysis import write
        s = write(root)
        print(json.dumps(s["spend"], indent=1))
    else:
        ap.error(f"unknown command {a.command}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

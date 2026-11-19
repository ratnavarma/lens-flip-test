"""Arm X2b: the X2 rerun with a form that fits the 150-token budget (docs/flip-protocol-v5b-x2-rerun.md).

An addendum to exploratory protocol v5. Nothing in the v5 modules or the six v5 arm dirs changes. X2b reuses the
v5 engine (XEngine, XAsker), the headline Engine (rounds, call cache, retries, Batch plumbing), the v3 flip search
and the v5 dry-run and analysis helpers by import. Its own files:

    results/exploratory-v5/X2b/           run dir (config.json, cases.jsonl, frame.jsonl, calibration.json, state/)
    results/exploratory-v5/dryrun_v5b.json
    results/exploratory-v5/protocol_v5b.json
    results/exploratory-v5/summary_v5b.json

The ledger is the shared results/exploratory-v5/spend.json. X2b is added to it as its own arm after every fetch,
and both v5 guards apply: ledger total + the round's likely cost under the cap, and the round's upper bound inside
the cap minus the other arms' spend.

CLI: uv run python -m lens.flip.explore5.x2b {dryrun,freeze,run,report}
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any, Mapping

from ..explore4.analysis import _complete, _states
from ..headline import requests as rq
from ..headline.engine import Engine, Refused, _read_json, _write_json
from ..headline.protocol import _sentinel_case
from ..headline.tasks import Asker, Pending, run_search
from ..residue import note_view, residue_candidates
from ..schema import View
from ..systems.base import count_tokens
from . import SCHEMA
from . import analysis as A
from . import dryrun as D
from . import engine as E
from . import frame as F
from . import protocol as PR
from . import x2b_prompts as XP
from .frame import Handoff, load_frame

ARM = "X2b"
VERSION_B = "5b-exploratory"
KIND = "note_x2b"
PROTOCOL_NAME = "protocol_v5b.json"
DRYRUN_NAME = "dryrun_v5b.json"
SUMMARY_NAME = "summary_v5b.json"
DOC = "docs/flip-protocol-v5b-x2-rerun.md"

CUT_GATE = 0.10          # the arm is readable only if at most 10 in 100 notes are cut
PRICE_GATE_USD = 1.50    # freeze --price-confirmed refuses above this dry-run likely price
REASK_LIKELY, REASK_UPPER = 0.5, 1.0

# Likely output tokens per request. X2b stages from arm X2's real usage (x2_A 906,729 / 490; x2_B 212,603 / 490).
ASSUMED_OUT: dict[str, int] = {**E.ASSUMED_OUT, "x2b_A": 1851, "x2b_A2": 1851, "x2b_B": 434}


# ---------------------------------------------------------------------------
# Config, calibration, requests
# ---------------------------------------------------------------------------

def arm_config(assumed_out: Mapping[str, int] | None = None) -> dict[str, Any]:
    """X2's request settings (writer, decider, search) with the X2b form and re-ask rule."""
    cfg = E.arm_config("X2")
    cfg.update({"version": "exploratory-v5b", "arm": ARM, "assumed_output_tokens": dict(assumed_out or ASSUMED_OUT)})
    cfg["x2b"] = {"fields": list(XP.FIELDS), "labels": dict(XP.LABELS), "caps_first": dict(XP.CAPS_FIRST),
                  "caps_tight": dict(XP.CAPS_TIGHT), "budget_tokens": XP.NOTE_BUDGET,
                  "reask": "once, when the rendered form passes the budget; the second call is priced and paid",
                  "cut_gate_max_share": CUT_GATE}
    return cfg


def calibration_b() -> dict[str, Any]:
    """The v5 input-token rule, with the X2b reply kind added (same rule as every other kind)."""
    c = F.calibration_v5()
    c["kinds"][KIND] = dict(c["kinds"]["note_x2"])
    return c


def calibration_text() -> str:
    return json.dumps(calibration_b(), indent=1, sort_keys=True) + "\n"


def x2b_request(view, case, cfg: Mapping[str, Any], caps: Mapping[str, int] = XP.CAPS_FIRST) -> dict[str, Any]:
    return E._json_schema_params(dict(cfg), XP.build_x2b_prompt(view, case, caps), XP.X2B_SCHEMA)


# ---------------------------------------------------------------------------
# Replies
# ---------------------------------------------------------------------------

def _empty_note(status: str) -> dict[str, Any]:
    return {"note": "", "fields": None, "rendered_tokens": 0, "over": False, "p_residue": None, "cut": False,
            "complete": False, "status": status}


def parse_b(kind: str, raw: dict[str, Any], meta: dict[str, Any]) -> dict[str, Any]:
    if kind != KIND:
        return E.parse_any(kind, raw, meta)
    stop, text = raw.get("stop_reason"), raw.get("text")
    if stop == "refusal" or text is None:
        return _empty_note(stop or "empty")
    r = XP.parse_x2b_reply(text)
    status = "max_tokens" if stop == "max_tokens" else ("ok" if r["complete"] and r["p_residue"] is not None
                                                        else ("fields_parse" if not r["complete"] else "p_parse"))
    return {**r, "status": status}


def failure_b(kind: str, reason: str) -> dict[str, Any]:
    return _empty_note(f"failed:{reason}") if kind == KIND else E.failure_any(kind, reason)


class BAsker(Asker):
    """The v5 XAsker's lookup and registration, with the X2b reply kind. No salted keys (cache on)."""

    def ask_many(self, items):
        answers = self.ns.setdefault("answers", {})
        out, pending = [], False
        for tag, kind, build, stage, extra in items:
            if tag in answers:
                out.append(answers[tag])
                continue
            params = build()
            key = rq.request_key(params)
            meta = {**self.meta, "stage": stage, "kind": kind, **(extra or {})}
            raw = self.ctx.lookup(key)
            if raw is not None:
                answers[tag] = parse_b(kind, raw, meta)
            else:
                reason = self.ctx.failed(key)
                if reason is None:
                    self.ctx.register(key, params, meta)
                    pending = True
                    out.append(None)
                    continue
                answers[tag] = failure_b(kind, reason)
            self.ctx.touch(self.meta["case_id"])
            out.append(answers[tag])
        if pending:
            raise Pending
        return out


# ---------------------------------------------------------------------------
# The task
# ---------------------------------------------------------------------------

def choose(first: dict[str, Any], second: dict[str, Any] | None) -> tuple[dict[str, Any], str]:
    """The answer whose note goes to B. The re-ask replaces the first form when it parsed as JSON; otherwise the
    first form is used, cut. With no re-ask the first form is used as it is."""
    if second is None:
        return first, "first"
    if second.get("fields") is not None:
        return second, "second"
    return first, "first"


def task_x2b(h: Handoff, ns: dict, ctx) -> bool:
    """A writes the three-field form; one re-ask with tighter caps if it passes the budget; B decides; on a flip
    the v3 flip search runs (group-first, 48 calls, v2 verify), exactly as in X2."""
    cfg = ctx.config
    target = h.row["a_A"]
    a = BAsker(ns, ctx, {"case_id": h.case.case_id, "cell": f"{h.system}@1000"})
    if "out" not in ns:
        xc = cfg["x2b"]
        n1 = a.ask("n", KIND, lambda: x2b_request(h.view, h.case, cfg["note_writer"], xc["caps_first"]), "x2b_A")
        n2 = None
        if n1["over"]:
            n2 = a.ask("n2", KIND, lambda: x2b_request(h.view, h.case, cfg["note_writer"], xc["caps_tight"]), "x2b_A2")
        n, used = choose(n1, n2)
        b = E._decide(a, "B", n["note"], h, ctx, "x2b_B")
        cands = residue_candidates(h.case, h.view, n["note"])
        ns["out"] = {"note": n["note"], "note_tokens": count_tokens([n["note"]]), "note_status": n["status"],
                     "fields": n["fields"], "cut": n["cut"], "complete": n["complete"], "p_residue": n["p_residue"],
                     "reasked": n2 is not None, "used": used,
                     "first_tokens": n1["rendered_tokens"], "first_status": n1["status"],
                     "second_tokens": None if n2 is None else n2["rendered_tokens"],
                     "second_status": None if n2 is None else n2["status"],
                     "a_B": b["action"], "residue_flip": b["action"] != target, "n_candidates": len(cands)}
        ctx.touch(h.case.case_id)
    out = ns["out"]
    if not out["residue_flip"]:
        return True
    if "record" not in ns:
        note = out["note"]
        cands = residue_candidates(h.case, h.view, note)
        sub = ns.setdefault("search", {})
        run_search(h.case, note_view(note), cands, target, sub, ctx,
                   meta={"case_id": h.case.case_id, "cell": f"{h.system}@1000"}, prefix="r",
                   stage="residue_flip", system=h.system, budget=1000)
        ns["record"] = sub["record"]
        ctx.touch(h.case.case_id)
    return True


# ---------------------------------------------------------------------------
# Ledger: the shared v5 ledger plus X2b
# ---------------------------------------------------------------------------

def _arm_entry(sp: dict[str, Any]) -> dict[str, Any]:
    return {"actual_usd": sp.get("actual_usd", 0.0), "input_tokens": sp.get("input_tokens", 0),
            "output_tokens": sp.get("output_tokens", 0),
            "rounds": [{k: r.get(k) for k in ("round", "n_requests", "upper_bound_usd", "actual_usd", "likely_usd",
                                               "by_stage")} for r in sp.get("rounds", [])],
            "by_stage": sp.get("by_stage", {})}


def update_ledger(root: Path, cap_usd: float | None = None) -> dict[str, Any]:
    """The v5 ledger (every v5 arm's spend.json), with X2b's spend.json added as arm X2b."""
    led = E.update_ledger(root, cap_usd)
    sp = _read_json(root / ARM / "spend.json", None)
    if sp is not None:
        led["arms"][ARM] = _arm_entry(sp)
    led["total_usd"] = round(sum(a["actual_usd"] for a in led["arms"].values()), 6)
    led["remaining_usd"] = None if led.get("cap_usd") is None else round(led["cap_usd"] - led["total_usd"], 6)
    led["updated_at"] = time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime())
    _write_json(E.ledger_path(root), led)
    return led


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------

def init_x2b(root: Path) -> Path:
    d = root / ARM
    if (d / "config.json").exists():
        raise Refused(f"{d} already holds a run")
    (d / "state" / "cases").mkdir(parents=True, exist_ok=True)
    for name in ("cases.jsonl", "frame.jsonl"):
        shutil.copyfile(root / name, d / name)
    (d / "calibration.json").write_text(calibration_text(), encoding="utf-8")
    _write_json(d / "config.json", arm_config())
    return d


class X2bEngine(E.XEngine):
    """XEngine with the X2b task, the v5b protocol check and the ledger that carries X2b."""

    def __init__(self, run_dir, runner=None, *, root: Path, cap_usd: float, log=None, now=time.time,
                 check_protocol: bool = True):
        Engine.__init__(self, run_dir, runner, log=log, now=now, check_protocol=check_protocol)
        self.root = Path(root)
        self.cap = float(cap_usd)
        self.arm = self.config["arm"]
        if self.arm != ARM:
            raise Refused(f"{self.dir} is arm {self.arm}, not {ARM}")
        self.task = task_x2b
        self.handoffs: dict[str, list[Handoff]] = {}
        for h in load_frame(self.dir, self.cases):
            self.handoffs.setdefault(h.case.case_id, []).append(h)
        self.cases = [c for c in self.cases if c.case_id in self.handoffs]
        self.controls = {}
        self.last_estimate = None

    def _check_protocol(self) -> None:
        if not self.check_protocol:
            return
        bad = check(self.root, self.dir)
        if bad:
            raise Refused("v5b protocol check failed: " + ", ".join(bad))

    def plan_round(self, pending):
        plan = Engine.plan_round(self, pending)
        led = update_ledger(self.root, self.cap)
        est = self.likely_estimate(pending)
        est["upper_bound_usd"] = round(plan["upper_bound_usd"], 6)
        est["ledger_total_before_usd"] = led["total_usd"]
        self.last_estimate = est
        log = _read_json(self.dir / "round_estimates.json", [])
        log.append({**est, "at": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime())})
        _write_json(self.dir / "round_estimates.json", log)
        if led["total_usd"] + est["likely_usd"] > self.cap + 1e-12:
            raise Refused(f"cap: spent ${led['total_usd']:.4f} + this round's likely ${est['likely_usd']:.4f} "
                          f"passes ${self.cap:.2f}; round not submitted")
        return plan

    def submit_round(self, cap_usd: float) -> str:
        led = update_ledger(self.root, self.cap)
        mine = float(self.spend()["actual_usd"])
        room = self.cap - (led["total_usd"] - mine)
        r = Engine.submit_round(self, min(cap_usd, room))
        if r == "submitted" and self.last_estimate is not None:
            sp = self.spend()
            sp["rounds"][-1]["likely_usd"] = self.last_estimate["likely_usd"]
            self._write_spend(sp)
        return r


def run_x2b(root: Path, runner=None, *, cap_usd: float, poll: float = 60.0, log=print, sleep=time.sleep,
            check_protocol: bool = True, max_steps: int | None = None) -> str:
    """Create the run dir if needed and run it to the end (or max_steps). Resumes any round in flight."""
    root = Path(root)
    if check_protocol:
        bad = check(root)
        if bad:
            raise Refused("v5b protocol check failed: " + ", ".join(bad))
    d = root / ARM
    if not (d / "config.json").exists():
        init_x2b(root)
    eng = X2bEngine(d, runner, root=root, cap_usd=cap_usd, log=log, check_protocol=check_protocol)
    try:
        steps = 0
        while True:
            r = eng.step(cap_usd)
            update_ledger(root, cap_usd)
            steps += 1
            if r == "complete" or (max_steps is not None and steps >= max_steps):
                return r
            if r == "waiting":
                sleep(poll)
    finally:
        eng.close()


# ---------------------------------------------------------------------------
# Freeze and check
# ---------------------------------------------------------------------------

def prompt_texts() -> dict[str, str]:
    case = _sentinel_case()
    view = View(system="{SYSTEM}", budget=None, text="{RECORDS}", units=["{UNIT}"])
    return {
        "x5b.x2b.prompt_first": XP.build_x2b_prompt(view, case, XP.CAPS_FIRST),
        "x5b.x2b.prompt_tight": XP.build_x2b_prompt(view, case, XP.CAPS_TIGHT),
        "x5b.x2b.schema": rq.canonical(XP.X2B_SCHEMA),
        "x5b.x2b.render": rq.canonical(XP.RENDER_RULE),
        "x5b.x2b.rule": rq.canonical({"budget_tokens": XP.NOTE_BUDGET, "count": "ceil(chars / 2.2)",
                                      "reask": "once, caps_tight, when the rendered form passes the budget",
                                      "use": "the re-ask when it parses as JSON, else the first form",
                                      "cut": "shared truncate_tokens, flagged", "gate": f"cut share <= {CUT_GATE}"}),
    }


def current() -> dict[str, Any]:
    return {"prompts": {k: PR._sha(v) for k, v in prompt_texts().items()},
            "reused_v3": PR.current()["reused_v3"],
            "arm_config": PR._sha(rq.canonical(PR.config_for_hash(arm_config())))}


def _file_hashes(root: Path) -> dict[str, str]:
    return {"frame_sha256": PR._sha((root / "frame.jsonl").read_bytes()),
            "cases_sha256": PR._sha((root / "cases.jsonl").read_bytes()),
            "calibration_sha256": PR._sha(calibration_text()),
            "protocol_v5_sha256": PR._sha((root / PR.NAME).read_bytes())}


def freeze(root: Path, price_confirmed: bool = False, now: float | None = None) -> dict[str, Any]:
    """Write protocol_v5b.json as DRAFT; with price_confirmed, set FINAL (refused above PRICE_GATE_USD)."""
    path = root / PROTOCOL_NAME
    v5bad = PR.check(root)
    if v5bad:
        raise SystemExit("the v5 protocol check fails: " + ", ".join(v5bad))
    dr = _read_json(root / DRYRUN_NAME, None)
    if dr is None:
        raise SystemExit(f"no {DRYRUN_NAME}; run dryrun first")
    stamp = time.strftime("%Y-%m-%d %H:%M UTC", time.gmtime(now if now is not None else time.time()))
    if path.exists():
        rec = json.loads(path.read_text())
        if rec.get("status") == "FINAL":
            raise SystemExit(f"{path} is FINAL; v5b is frozen")
        if not price_confirmed:
            raise SystemExit(f"{path} exists as DRAFT; freeze --price-confirmed sets FINAL")
        bad = check(root, allow_draft=True)
        if bad:
            raise SystemExit("cannot finalize; changed since the draft: " + ", ".join(bad))
    else:
        rec = {"schema": SCHEMA, "version": VERSION_B, "status": "DRAFT", "doc": DOC, "addendum_to": PR.NAME,
               "arm": ARM, "frozen_at": stamp, **current(), **_file_hashes(root)}
    rec["dryrun"] = {"likely_usd": dr["total"]["likely_usd"], "upper_usd": dr["total"]["upper_usd"], "at": dr["at"]}
    if price_confirmed:
        if dr["total"]["likely_usd"] > PRICE_GATE_USD:
            raise SystemExit(f"dry-run likely ${dr['total']['likely_usd']:.2f} is above ${PRICE_GATE_USD:.2f}; "
                             "ask Varma before finalizing")
        rec["status"] = "FINAL"
        rec["finalized_at"] = stamp
        rec["approval"] = "Varma, 2026-10-29: rerun X2 with a form that fits the budget, about $1, inside the $30 cap"
    path.write_text(json.dumps(rec, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return rec


def check(root: Path, run_dir: Path | None = None, *, allow_draft: bool = False) -> list[str]:
    path = root / PROTOCOL_NAME
    if not path.exists():
        return [f"{PROTOCOL_NAME} missing"]
    bad = [f"v5:{x}" for x in PR.check(root)]
    frozen = json.loads(path.read_text())
    cur = current()
    if frozen.get("status") != "FINAL" and not allow_draft:
        bad.append(f"status:{frozen.get('status')}")
    bad += [f"prompt:{k}" for k in set(frozen["prompts"]) | set(cur["prompts"])
            if frozen["prompts"].get(k) != cur["prompts"].get(k)]
    bad += [f"v3:{k}" for k in frozen["reused_v3"] if frozen["reused_v3"][k] != cur["reused_v3"].get(k)]
    if frozen.get("arm_config") != cur["arm_config"]:
        bad.append(f"arm:{ARM}")
    bad += [f"root:{k}" for k, v in _file_hashes(root).items() if frozen.get(k) != v]
    if run_dir is not None:
        cfg = json.loads((run_dir / "config.json").read_text())
        if PR._sha(rq.canonical(PR.config_for_hash(cfg))) != frozen.get("arm_config"):
            bad.append(f"config:{cfg.get('arm')}")
        for name, key in (("frame.jsonl", "frame_sha256"), ("cases.jsonl", "cases_sha256"),
                          ("calibration.json", "calibration_sha256")):
            if PR._sha((run_dir / name).read_bytes()) != frozen.get(key):
                bad.append(f"run:{name}")
    return sorted(bad)


# ---------------------------------------------------------------------------
# Dry run
# ---------------------------------------------------------------------------

def _observed_flip_rate(root: Path, arm: str, n: int) -> float | None:
    s = _states(root, arm)
    if not _complete(s, n):
        return None
    return sum(bool(v["out"]["residue_flip"]) for v in s.values()) / n


def _search_stages(n: int, flip_rate: float, prof: dict, use: dict, dec: dict, mean_bound_in: float,
                   ao: Mapping[str, int]) -> list[dict[str, Any]]:
    """The v5 dry-run's X2 search pricing at a given likely flip rate (upper stays at D.FLIP_UPPER)."""
    v4_in = (use["v4"]["residue_flip"] or {}).get("in_mean") or 1200.0
    v4_out = (use["v4"]["residue_flip"] or {}).get("out_mean") or ao["residue_flip"]
    ver = use["v4"]["residue_flip_verify"] or {}
    n_l, n_u = flip_rate * n, D.FLIP_UPPER * n
    return [
        D.search_stage("flip search calls", n_l, prof["calls_per_search"], v4_in, v4_out, dec["max_tokens"],
                       mean_bound_in, 48, n_u,
                       note=f"{n_l:.0f} searches likely ({flip_rate:.1%}), {prof['calls_per_search']:.2f} calls each; "
                            f"upper {n_u:.0f} searches at 48 calls, max_tokens, all candidate units"),
        D.search_stage("v2 verify calls", n_l, prof["verify_per_search"], ver.get("in_mean") or 1000.0,
                       ver.get("out_mean") or ao["residue_flip_verify"], dec["max_tokens"], mean_bound_in, 1, n_u,
                       note=f"{prof['verify_per_search']:.2f} verifies per search likely; upper one per search")]


def _sum(stages: list[dict[str, Any]]) -> dict[str, float]:
    return {"requests": round(sum(s["requests"] for s in stages)),
            "requests_upper": round(sum(s.get("requests_upper", s["requests"]) for s in stages)),
            "likely_usd": round(sum(s["likely_usd"] for s in stages), 4),
            "upper_usd": round(sum(s["upper_usd"] for s in stages), 4)}


def dryrun_x2b(root: Path, results: Path = F.RESULTS, write: bool = True) -> dict[str, Any]:
    """Price X2b without a model call: exact writer prompts (both caps), the v3 note as B's length stand-in, and
    the v5 search pricing. The re-ask stage is priced at REASK_LIKELY of the handoffs (upper: every handoff)."""
    hs = load_frame(root)
    n = len(hs)
    cfg = arm_config()
    nw, dec, ao = cfg["note_writer"], cfg["decider"], ASSUMED_OUT
    first = D._chars([x2b_request(h.view, h.case, nw, XP.CAPS_FIRST) for h in hs])
    tight = D._chars([x2b_request(h.view, h.case, nw, XP.CAPS_TIGHT) for h in hs])
    b_proxy = D._chars([rq.decide_request(note_view(h.row["v3"]["note"]), [], h.case, "v1", dec) for h in hs])
    re_l = D.stage_price("x2b_A2 re-ask, tighter caps", tight, ao["x2b_A2"], nw["max_tokens"], scale=REASK_LIKELY,
                         note=f"likely {REASK_LIKELY:.0%} of handoffs re-asked; upper every handoff")
    re_u = D.stage_price("x2b_A2 re-ask, tighter caps", tight, ao["x2b_A2"], nw["max_tokens"], scale=REASK_UPPER)
    re_l.update({"upper_usd": re_u["upper_usd"], "input_tokens_upper": re_u["input_tokens_upper"],
                 "output_tokens_upper": re_u["output_tokens_upper"], "requests_upper": re_u["requests"]})
    notes = [D.stage_price("x2b_A form writer", first, ao["x2b_A"], nw["max_tokens"]), re_l,
             D.stage_price("x2b_B decide on the note", b_proxy, ao["x2b_B"], dec["max_tokens"],
                           note="note length proxied by the v3 note")]
    prof = D.search_profile(root, results)
    use = D.observed_usage(results)
    all_cands = D._chars([rq.decide_request(note_view(h.row["v3"]["note"]),
                                            residue_candidates(h.case, h.view, h.row["v3"]["note"]), h.case, "v1", dec)
                          for h in hs])
    mean_bound_in = sum(D.tokens_bound(c) for c in all_cands) / len(all_cands)
    search = _search_stages(n, D.FLIP_LIKELY, prof, use, dec, mean_bound_in, ao)
    stages = notes + search
    tot = _sum(stages)
    x2_rate = _observed_flip_rate(root, "X2", n)
    sens = None
    if x2_rate is not None:
        s2 = _search_stages(n, x2_rate, prof, use, dec, mean_bound_in, ao)
        sens = {"flip_rate": x2_rate, "search_likely_usd": _sum(s2)["likely_usd"],
                "likely_usd": round(_sum(notes)["likely_usd"] + _sum(s2)["likely_usd"], 4),
                "note": "the search priced at arm X2's observed flip rate"}
    plus50 = sum(rq.price(s["input_tokens_likely"], s["output_tokens_likely"] * 1.5, D.RATES) for s in stages)
    res = {
        "schema": SCHEMA, "kind": "dryrun", "version": VERSION_B, "arm": ARM,
        "at": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()),
        "model": E.HAIKU, "api": "Message Batches (50% rates)", "rates_usd_per_million": dict(D.RATES),
        "assumed_output_tokens": {k: ao[k] for k in ("x2b_A", "x2b_A2", "x2b_B", "residue_flip", "residue_flip_verify")},
        "reask_share": {"likely": REASK_LIKELY, "upper": REASK_UPPER},
        "stages": stages, "notes_and_B": _sum(notes), "search": {**_sum(search), "profile": prof,
                                                                 "flip_rate_likely": D.FLIP_LIKELY,
                                                                 "flip_rate_upper": D.FLIP_UPPER},
        "total": {**tot, "likely_usd_if_outputs_x1_5": round(plus50, 4)},
        "sensitivity_x2_flip_rate": sens,
        "rounds_likely": 3 + prof["rounds_max"] + 1, "rounds_upper": 3 + 24 + 1,
        "price_gate_usd": PRICE_GATE_USD, "likely_at_or_under_price_gate": tot["likely_usd"] <= PRICE_GATE_USD,
        "api_calls": 0, "count_tokens_calls": 0, "submitted": False,
        "inputs_sha256": {k: F.sha256_file(root / k) for k in ("frame.jsonl", "cases.jsonl")},
    }
    if write:
        (root / DRYRUN_NAME).write_text(json.dumps(res, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    return res


def dryrun_table(res: dict[str, Any]) -> str:
    lines = ["| Stage | Requests | Likely USD | Upper USD |", "| --- | --- | --- | --- |"]
    for s in res["stages"]:
        lines.append(f"| {s['stage']} | {s['requests']:,.0f} | {s['likely_usd']:.3f} | {s['upper_usd']:.3f} |")
    t = res["total"]
    lines.append(f"| Total | {t['requests']:,} | {t['likely_usd']:.2f} | {t['upper_usd']:.2f} |")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

def cut_gate(n_cut: int, n: int, max_share: float = CUT_GATE) -> dict[str, Any]:
    share = (n_cut / n) if n else None
    return {"n_cut": n_cut, "n": n, "cut_share": share, "max_share": max_share,
            "readable": share is not None and share <= max_share + 1e-12,
            "rule": "X2b is readable only if at most 10 in 100 rendered notes are cut at 150 tokens"}


def gate_line(g: dict[str, Any]) -> str:
    verdict = "PASS, the arm is readable" if g["readable"] else "FAIL, the arm is not readable"
    share = "n/a" if g["cut_share"] is None else f"{100 * g['cut_share']:.1f} in 100"
    return f"cut gate: {g['n_cut']} of {g['n']} notes cut ({share}); gate at or under {100 * g['max_share']:.0f} in 100: {verdict}"


def build(root: Path) -> dict[str, Any]:
    cases, rows = A.base_rows(root)
    n = len(rows)
    st = A.NStats(cases, rows)
    sb = _states(root, ARM)
    if not _complete(sb, n):
        raise SystemExit(f"{ARM} is not complete ({0 if sb is None else sum(bool(v.get('done')) for v in sb.values())} of {n})")
    x = A.study_x(st, rows, "x2b", ARM, sb, True)
    s2 = _states(root, "X2")
    if _complete(s2, n):
        x2 = A.study_x(st, rows, "x2", "X2", s2, True)
        x["residue_minus_x2"] = st.diff("x2b_flip", "x2_flip")
        x["auroc_minus_x2_auroc"] = st.auroc_vs_auroc(("x2b_p", "x2b_flip"), ("x2_p", "x2_flip"))
        x["recoverable_minus_x2_recoverable"] = st.diff("x2b_recov", "x2_recov")
        x["x2_reference"] = {"residue_rate": x2["residue_rate"], "h2c_auroc_p": x2["h2c_auroc_p"],
                             "recoverable_rate": x2["recoverable_rate"], "mean_tokens": x2["notes"]["mean_tokens"],
                             "notes_cut_at_150": sum(bool(s2[(r["case_id"], r["system"])]["out"]["cut"]) for r in rows)}
    outs = [sb[(r["case_id"], r["system"])]["out"] for r in rows]
    toks = sorted(o["note_tokens"] for o in outs)
    firsts = sorted(o["first_tokens"] for o in outs)
    re = [o for o in outs if o["reasked"]]
    g = cut_gate(sum(bool(o["cut"]) for o in outs), n)
    x["structure"] = {
        "notes_cut_at_150": g["n_cut"], "cut_gate": g,
        "first_form_fits": sum(not o["reasked"] for o in outs), "reasked": len(re),
        "reask_fits": sum(not o["cut"] for o in re), "used_second": sum(o["used"] == "second" for o in outs),
        "replies_with_all_three_fields": sum(bool(o["complete"]) for o in outs),
        "notes_with_an_empty_field": sum(any(not (v or "").strip() for v in (o["fields"] or {}).values()) for o in outs),
        "note_tokens": {"mean": sum(toks) / n, "median": toks[n // 2], "p90": toks[int(0.9 * (n - 1))], "max": toks[-1]},
        "first_form_tokens": {"mean": sum(firsts) / n, "median": firsts[n // 2], "max": firsts[-1]},
        "first_status": dict(Counter(o["first_status"] for o in outs)),
        "second_status": dict(Counter(o["second_status"] for o in re)),
    }
    frozen = _read_json(root / PROTOCOL_NAME, {})
    led = _read_json(root / "spend.json", {})
    sp = _read_json(root / ARM / "spend.json", {})
    return {
        "schema": SCHEMA, "version": VERSION_B, "exploratory": True,
        "note": "Exploratory addendum to v5. No result here changes a v3, v4 or v5 number.",
        "protocol": {"doc": DOC, "addendum_to": PR.NAME, "status": frozen.get("status"),
                     "frozen_at": frozen.get("frozen_at"), "finalized_at": frozen.get("finalized_at"),
                     "prompt_sha256": frozen.get("prompts"), "arm_config_sha256": frozen.get("arm_config"),
                     "frame_sha256": frozen.get("frame_sha256"), "cases_sha256": frozen.get("cases_sha256")},
        "denominators": {"cases": len(cases), "eligible_handoffs": n,
                         "by_system": dict(Counter(r["system"] for r in rows)),
                         "v3_residue_flips": sum(r["v3_flip"] for r in rows)},
        "bootstrap": {"reps": 1000, "seed": 11, "unit": "case (paired; every handoff of a resampled case)"},
        "v3_reference": {"residue_rate": st.rate("v3_flip"), "floor_rate": st.rate("v3_floor"),
                         "recoverable_rate": st.rate("v3_recov"), "h2c_auroc": st.auroc("v3_p", "v3_flip")},
        "X2b_structured_fit": x,
        "spend": {"x2b_usd": sp.get("actual_usd"), "x2b_by_stage": sp.get("by_stage"),
                  "x2b_requests": sum(v["requests"] for v in (sp.get("by_stage") or {}).values()),
                  "ledger_cap_usd": led.get("cap_usd"), "ledger_total_usd": led.get("total_usd"),
                  "ledger_remaining_usd": led.get("remaining_usd"),
                  "ledger_by_arm": {a: v.get("actual_usd") for a, v in (led.get("arms") or {}).items()},
                  "dryrun_likely_usd": (frozen.get("dryrun") or {}).get("likely_usd")},
    }


def write(root: Path) -> dict[str, Any]:
    s = build(root)
    (root / SUMMARY_NAME).write_text(json.dumps(s, indent=1, sort_keys=True, default=str) + "\n", encoding="utf-8")
    return s


def headline(s: dict[str, Any]) -> str:
    x = s["X2b_structured_fit"]

    def pct(v):
        return "n/a" if v is None else f"{100 * v:.1f}"

    def ci(c):
        return "n/a" if not c else f"[{100 * c[0]:.1f}, {100 * c[1]:.1f}]"
    lines = [gate_line(x["structure"]["cut_gate"]),
             f"residue {pct(x['residue_rate']['value'])} in 100 ({x['residue_rate']['k']} of {x['residue_rate']['n']}); "
             f"minus v3 {pct(x['residue_minus_v3']['diff'])} {ci(x['residue_minus_v3']['ci95'])}"]
    if "residue_minus_x2" in x:
        lines.append(f"minus X2 {pct(x['residue_minus_x2']['diff'])} {ci(x['residue_minus_x2']['ci95'])}")
    au = x["h2c_auroc_p"]
    lines.append(f"AUROC p_residue {au.get('auroc')} (CI {au.get('ci95')}); minus v3 AUROC "
                 f"{x['auroc_minus_v3_auroc']['diff']} {x['auroc_minus_v3_auroc']['ci95']}")
    lines.append(f"recoverable {pct(x['recoverable_rate']['value'])} in 100 ({x['recoverable_rate']['k']}); "
                 f"mean tokens {x['structure']['note_tokens']['mean']:.1f}; reasked {x['structure']['reasked']}")
    lines.append(f"spend X2b ${s['spend']['x2b_usd']}; ledger ${s['spend']['ledger_total_usd']} of ${s['spend']['ledger_cap_usd']}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m lens.flip.explore5.x2b")
    ap.add_argument("command", choices=["dryrun", "freeze", "run", "report"])
    ap.add_argument("--results", default=None, help="results dir (default $LENS_RESULTS or results)")
    ap.add_argument("--cap-usd", type=float, default=None, help="spend cap for the shared ledger; required by run")
    ap.add_argument("--price-confirmed", action="store_true", help="Varma has confirmed the dry-run price")
    ap.add_argument("--poll", type=float, default=60.0)
    a = ap.parse_args(argv)
    results = Path(a.results) if a.results else F.RESULTS
    root = results / "exploratory-v5"
    if a.command == "dryrun":
        res = dryrun_x2b(root, results)
        print(dryrun_table(res))
        sens = res["sensitivity_x2_flip_rate"]
        print(f"\nlikely ${res['total']['likely_usd']:.2f}, upper ${res['total']['upper_usd']:.2f}"
              + (f", likely at X2's flip rate {sens['flip_rate']:.1%}: ${sens['likely_usd']:.2f}" if sens else "")
              + f"; at or under ${PRICE_GATE_USD:.2f}: {res['likely_at_or_under_price_gate']}; written to {root / DRYRUN_NAME}")
    elif a.command == "freeze":
        rec = freeze(root, a.price_confirmed)
        print(json.dumps({k: rec[k] for k in ("status", "frozen_at", "dryrun", "prompts", "arm_config")}, indent=1))
    elif a.command == "run":
        if not a.price_confirmed:
            raise SystemExit("run needs --price-confirmed (the run happens only after Varma confirms the price)")
        if a.cap_usd is None or a.cap_usd <= 0:
            raise SystemExit("run needs --cap-usd (a positive number)")
        try:
            out = run_x2b(root, cap_usd=a.cap_usd, poll=a.poll, log=lambda m: print(m, flush=True))
        except Refused as exc:
            print(f"REFUSED: {exc}", flush=True)
            update_ledger(root, a.cap_usd)
            return 3
        led = update_ledger(root, a.cap_usd)
        print(f"{ARM}: {out}; ledger ${led['total_usd']:.4f} of ${a.cap_usd:.2f}", flush=True)
    elif a.command == "report":
        s = write(root)
        print(headline(s))
    return 0


if __name__ == "__main__":
    sys.exit(main())

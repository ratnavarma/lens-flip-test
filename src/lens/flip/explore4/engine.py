"""Batch-round engine for the exploratory v4 arms, built on the headline Engine.

Each arm is its own run dir under results/exploratory-v4/<arm>/ with config.json, cases.jsonl
(the v3 combined cases, copied) and frame.jsonl (copied). Requests, keys, the call cache, rounds,
retries and per-run spend.json are the headline Engine's. Two guards sit on top of it:

1. The study ledger (results/exploratory-v4/spend.json). Before a round is submitted, the round's
   likely cost (calibrated input plus assumed output per request) is added to the ledger total. If
   the sum passes the study cap ($30), the round is refused and nothing is sent.
2. The engine's own cap is set to the cap minus what the other arms and probes have spent, so the
   round's upper bound (max_tokens on every request) must also fit inside what is left.

The protocol check compares the v4 prompt hashes and the frame and cases hashes with
results/exploratory-v4/protocol_v4.json, written by `freeze` before any model call.
"""

from __future__ import annotations

import json
import shutil
import time
from collections import Counter
from pathlib import Path
from typing import Any, Callable

from ..decider import PARSE_ERROR
from ..headline import requests as rq
from ..headline.engine import Engine, Refused, _read_json, _write_json
from ..headline.tasks import Asker, Pending, run_search
from ..residue import note_view, omission_fraction, residue_candidates
from ..schema import Case
from ..systems.base import count_tokens
from . import prompts as P
from .frame import OUT, Handoff, load_frame

STUDY_CAP_USD = 30.0
HAIKU = "claude-haiku-5-5"
SONNET = "claude-sonnet-5-5"
RATES = {HAIKU: {"input": 0.05, "output": 0.25}, SONNET: {"input": 1.00, "output": 5.00}}

# Arms in the order the brief fixes: A extractive, B Sonnet, C budgets, A advice-stripped,
# then the optional reader-view arm for Sonnet (declared in the protocol, run only if the money allows).
ARMS = ("A-extractive", "B-sonnet", "C-budget", "A-stripped", "B-sonnet-view")

DECIDER_HAIKU = {"model": HAIKU, "arm": "adaptive-high"}
DECIDER_SONNET = {"model": SONNET, "arm": "adaptive-high"}
NOTE_WRITER = {"model": HAIKU, "effort": "high", "thinking": None, "max_tokens": 6000}


def arm_config(arm: str, assumed_out: dict[str, int] | None = None) -> dict[str, Any]:
    if arm not in ARMS:
        raise ValueError(f"unknown arm {arm!r}")
    sonnet = arm.startswith("B-sonnet")
    dec = DECIDER_SONNET if sonnet else DECIDER_HAIKU
    from .. import batch
    arm_cfg = dict(batch.ARM_CONFIG[dec["arm"]])
    arm_cfg.pop("assumed_out", None)
    return {
        "version": "exploratory-v4", "arm": arm, "residue_budget": 1000, "max_calls": 48, "strategy": "group",
        "decider": {**dec, **arm_cfg}, "note_writer": dict(NOTE_WRITER),
        "extract_tokens": P.EXTRACT_TOKENS, "note_budgets": list(P.NOTE_BUDGETS),
        "rates": dict(RATES[SONNET if sonnet else HAIKU]),
        "assumed_output_tokens": assumed_out or ({"decide": 3000} if sonnet else
                                                  {"decide": 1000, "extract": 1300, "note_budget": 1500}),
        "cache": "state/calls.sqlite",
        "batch": {"max_requests": 100_000, "max_bytes": 128 * 1024 * 1024},
        "bootstrap": {"reps": 1000, "seed": 11},
    }


# ---------------------------------------------------------------------------
# Requests and answers
# ---------------------------------------------------------------------------

def extract_request(view, case: Case, cfg: dict[str, Any], budget: int = P.EXTRACT_TOKENS) -> dict[str, Any]:
    params: dict[str, Any] = {
        "model": cfg["model"], "max_tokens": cfg["max_tokens"],
        "messages": [{"role": "user", "content": P.build_extract_prompt(view, case, budget)}],
        "output_config": {"effort": cfg["effort"],
                          "format": {"type": "json_schema", "schema": P.extract_schema(P.unit_labels(len(view.units)))}},
    }
    if cfg.get("thinking") is not None:
        params["thinking"] = dict(cfg["thinking"])
    return params


def budget_note_request(view, case: Case, cfg: dict[str, Any], budget: int) -> dict[str, Any]:
    params: dict[str, Any] = {
        "model": cfg["model"], "max_tokens": cfg["max_tokens"],
        "messages": [{"role": "user", "content": P.build_budget_note_prompt(view, case, budget)}],
        "output_config": {"effort": cfg["effort"],
                          "format": {"type": "json_schema", "schema": json.loads(json.dumps(P.BUDGET_NOTE_SCHEMA))}},
    }
    if cfg.get("thinking") is not None:
        params["thinking"] = dict(cfg["thinking"])
    return params


def parse_any(kind: str, raw: dict[str, Any], meta: dict[str, Any]) -> dict[str, Any]:
    stop, text = raw.get("stop_reason"), raw.get("text")
    if kind in ("decide", "note", "summary"):
        return rq.parse_answer(kind, raw, meta)
    if kind == "extract":
        if stop == "refusal" or text is None:
            return {"labels": None, "p_residue": None, "status": stop or "empty"}
        labels, p = P.parse_extract_reply(text)
        status = "max_tokens" if stop == "max_tokens" else ("ok" if labels is not None and p is not None
                                                            else ("labels_parse" if labels is None else "p_parse"))
        return {"labels": labels, "p_residue": p, "status": status}
    if kind == "note_budget":
        budget = int(meta["note_budget"])
        if stop == "refusal" or text is None:
            return {"note": "", "p_residue": None, "status": stop or "empty"}
        note, p = P.parse_budget_note_reply(text, budget)
        status = "max_tokens" if stop == "max_tokens" else ("ok" if p is not None else "p_parse")
        return {"note": note, "p_residue": p, "status": status}
    raise ValueError(f"unknown kind {kind!r}")


def failure_any(kind: str, reason: str) -> dict[str, Any]:
    status = f"failed:{reason}"
    if kind == "decide":
        return {"action": PARSE_ERROR, "status": status}
    if kind == "extract":
        return {"labels": None, "p_residue": None, "status": status}
    return {"note": "", "p_residue": None, "status": status}


class XAsker(Asker):
    """tasks.Asker with the v4 reply kinds (extract, note_budget)."""

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
                answers[tag] = parse_any(kind, raw, meta)
            else:
                reason = self.ctx.failed(key)
                if reason is None:
                    self.ctx.register(key, params, meta)
                    pending = True
                    out.append(None)
                    continue
                answers[tag] = failure_any(kind, reason)
            self.ctx.touch(self.meta["case_id"])
            out.append(answers[tag])
        if pending:
            raise Pending
        return out


# ---------------------------------------------------------------------------
# Arm tasks. Each returns True when the handoff is done. ns is the handoff's own state.
# ---------------------------------------------------------------------------

def _asker(h: Handoff, ns: dict, ctx) -> XAsker:
    return XAsker(ns, ctx, {"case_id": h.case.case_id, "cell": f"{h.system}@1000"})


def task_extractive(h: Handoff, ns: dict, ctx) -> bool:
    cfg = ctx.config
    target = h.row["a_A"]
    a = _asker(h, ns, ctx)
    if "out" not in ns:
        x = a.ask("x", "extract", lambda: extract_request(h.view, h.case, cfg["note_writer"], cfg["extract_tokens"]),
                  "x_A", {"note_budget": cfg["extract_tokens"]})
        note, info = P.assemble_extract(h.view, x["labels"], int(cfg["extract_tokens"]))
        nview = note_view(note)
        b = a.ask("B", "decide", lambda: rq.decide_request(nview, [], h.case, "v1", cfg["decider"]), "x_B")
        cands = residue_candidates(h.case, h.view, note)
        ns["out"] = {"note": note, "note_tokens": count_tokens([note]), "x_status": x["status"],
                     "p_residue": x["p_residue"], "assembly": info, "a_B": b["action"],
                     "residue_flip": b["action"] != target, "omission": omission_fraction(h.case, h.view, note),
                     "n_candidates": len(cands), "candidate_ids": [u.id for u in cands]}
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


def task_reader(h: Handoff, ns: dict, ctx) -> bool:
    """B-sonnet: the second reader on the frozen v3 note, v1 prompt."""
    if "out" in ns:
        return True
    cfg = ctx.config
    note = h.row["v3"]["note"]
    b = _asker(h, ns, ctx).ask("B", "decide", lambda: rq.decide_request(note_view(note), [], h.case, "v1",
                                                                          cfg["decider"]), "s_B")
    ns["out"] = {"a_B": b["action"], "status": b["status"], "residue_flip": b["action"] != h.row["a_A"]}
    ctx.touch(h.case.case_id)
    return True


def task_reader_view(h: Handoff, ns: dict, ctx) -> bool:
    """B-sonnet-view: the second reader on V itself with the v2 prompt (the reader's own view baseline)."""
    if "out" in ns:
        return True
    cfg = ctx.config
    v = _asker(h, ns, ctx).ask("V2", "decide", lambda: rq.decide_request(h.view, [], h.case, "v2", cfg["decider"]),
                               "s_view")
    ns["out"] = {"a_view_v2": v["action"], "status": v["status"], "view_miss": v["action"] != h.row["a_A"]}
    ctx.touch(h.case.case_id)
    return True


def task_budget(h: Handoff, ns: dict, ctx) -> bool:
    cfg = ctx.config
    a = _asker(h, ns, ctx)
    outs = ns.setdefault("out", {})
    pending = False
    for budget in cfg["note_budgets"]:
        k = str(budget)
        if k in outs:
            continue
        try:
            n = a.ask(f"n{budget}", "note_budget", lambda b=budget: budget_note_request(h.view, h.case, cfg["note_writer"], b),
                      f"c_A{budget}", {"note_budget": budget})
            nview = note_view(n["note"])
            b = a.ask(f"B{budget}", "decide", lambda nv=nview: rq.decide_request(nv, [], h.case, "v1", cfg["decider"]),
                      f"c_B{budget}")
        except Pending:
            pending = True
            continue
        outs[k] = {"note": n["note"], "note_tokens": count_tokens([n["note"]]), "note_status": n["status"],
                   "p_residue": n["p_residue"], "a_B": b["action"], "residue_flip": b["action"] != h.row["a_A"]}
        ctx.touch(h.case.case_id)
    if pending:
        raise Pending
    return True


def task_stripped(h: Handoff, ns: dict, ctx) -> bool:
    if "out" in ns:
        return True
    cfg = ctx.config
    v3 = h.row["v3"]
    note, dropped = P.strip_advice(v3["note"])
    if not dropped:  # byte-identical to the v3 note: the identical request, so the v3 answer stands
        ns["out"] = {"note": note, "changed": False, "dropped": [], "a_B": v3["a_B"], "source": "v3",
                     "residue_flip": v3["a_B"] != h.row["a_A"], "note_tokens": count_tokens([note])}
        ctx.touch(h.case.case_id)
        return True
    b = _asker(h, ns, ctx).ask("B", "decide", lambda: rq.decide_request(note_view(note), [], h.case, "v1",
                                                                          cfg["decider"]), "a_B_stripped")
    ns["out"] = {"note": note, "changed": True, "dropped": dropped, "a_B": b["action"], "source": "new",
                 "residue_flip": b["action"] != h.row["a_A"], "note_tokens": count_tokens([note]),
                 "empty": not note.strip()}
    ctx.touch(h.case.case_id)
    return True


TASKS: dict[str, Callable[[Handoff, dict, Any], bool]] = {
    "A-extractive": task_extractive, "B-sonnet": task_reader, "C-budget": task_budget,
    "A-stripped": task_stripped, "B-sonnet-view": task_reader_view,
}


# ---------------------------------------------------------------------------
# Ledger
# ---------------------------------------------------------------------------

def ledger_path(root: Path = OUT) -> Path:
    return root / "spend.json"


def update_ledger(root: Path = OUT) -> dict[str, Any]:
    """Sum actual spend from each arm's spend.json plus recorded probes. Written after every fetch."""
    path = ledger_path(root)
    led = _read_json(path, {"cap_usd": STUDY_CAP_USD, "probes": [], "arms": {}})
    arms = {}
    for arm in ARMS:
        sp = _read_json(root / arm / "spend.json", None)
        if sp is None:
            continue
        arms[arm] = {"actual_usd": sp.get("actual_usd", 0.0), "input_tokens": sp.get("input_tokens", 0),
                     "output_tokens": sp.get("output_tokens", 0),
                     "rounds": [{k: r.get(k) for k in ("round", "n_requests", "upper_bound_usd", "actual_usd",
                                                         "likely_usd", "by_stage")} for r in sp.get("rounds", [])],
                     "by_stage": sp.get("by_stage", {})}
    led["arms"] = arms
    led["probes_usd"] = round(sum(p["usd"] for p in led.get("probes", [])), 6)
    led["total_usd"] = round(sum(a["actual_usd"] for a in arms.values()) + led["probes_usd"], 6)
    led["remaining_usd"] = round(led["cap_usd"] - led["total_usd"], 6)
    led["updated_at"] = time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime())
    _write_json(path, led)
    return led


def add_probe(root: Path, entry: dict[str, Any]) -> dict[str, Any]:
    path = ledger_path(root)
    led = _read_json(path, {"cap_usd": STUDY_CAP_USD, "probes": [], "arms": {}})
    led.setdefault("probes", []).append(entry)
    _write_json(path, led)
    return update_ledger(root)


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------

def init_arm(arm: str, root: Path = OUT, assumed_out: dict[str, int] | None = None) -> Path:
    d = root / arm
    if (d / "config.json").exists():
        raise Refused(f"{d} already holds a run")
    (d / "state" / "cases").mkdir(parents=True, exist_ok=True)
    shutil.copyfile(root / "cases.jsonl", d / "cases.jsonl")
    shutil.copyfile(root / "frame.jsonl", d / "frame.jsonl")
    calib = Path("results/headline-300/calibration.json")
    if calib.exists():
        shutil.copyfile(calib, d / "calibration.json")
    _write_json(d / "config.json", arm_config(arm, assumed_out))
    return d


class XEngine(Engine):
    def __init__(self, run_dir, runner=None, *, root: Path = OUT, cap_usd: float = STUDY_CAP_USD,
                 log=None, now=time.time, check_protocol: bool = True):
        super().__init__(run_dir, runner, log=log, now=now, check_protocol=check_protocol)
        self.root = Path(root)
        self.study_cap = cap_usd
        self.arm = self.config["arm"]
        self.task = TASKS[self.arm]
        hs = load_frame(self.dir, self.cases)
        self.handoffs: dict[str, list[Handoff]] = {}
        for h in hs:
            self.handoffs.setdefault(h.case.case_id, []).append(h)
        self.cases = [c for c in self.cases if c.case_id in self.handoffs]
        self.last_estimate: dict[str, Any] | None = None

    def advance(self):
        self.pending, self.waiters = {}, Counter()
        for case in self.cases:
            cs = self.states[case.case_id]
            if cs.get("done"):
                continue
            done = True
            for h in self.handoffs[case.case_id]:
                ns = cs.setdefault(h.system, {})
                if ns.get("done"):
                    continue
                try:
                    ok = self.task(h, ns, self)
                except Pending:
                    ok = False
                if ok:
                    ns["done"] = True
                    self.touch(case.case_id)
                done &= ok
            if done:
                cs["done"] = True
                self.touch(case.case_id)
        self.save()
        return self.pending

    def _check_protocol(self) -> None:
        if not self.check_protocol:
            return
        from .protocol import check
        bad = check(self.root, self.dir)
        if bad:
            raise Refused("v4 prompts, frame or config changed since the freeze: " + ", ".join(bad))

    def likely_estimate(self, pending) -> dict[str, Any]:
        calib = self.calibration()
        rates = self.config["rates"]
        assumed = self.config["assumed_output_tokens"]
        tin = tout = 0
        by_kind: Counter = Counter()
        for key, (params, meta) in pending.items():
            kind = meta["kind"]
            tin += rq.estimate_input_tokens(rq.request_chars(params), kind, calib, bound=False)
            tout += int(assumed.get(kind, params["max_tokens"]))
            by_kind[kind] += 1
        return {"n_requests": len(pending), "by_kind": dict(by_kind), "input_tokens": tin, "output_tokens": tout,
                "likely_usd": round(rq.price(tin, tout, rates), 6)}

    def plan_round(self, pending):
        plan = super().plan_round(pending)
        led = update_ledger(self.root)
        est = self.likely_estimate(pending)
        est["upper_bound_usd"] = round(plan["upper_bound_usd"], 6)
        est["ledger_total_before_usd"] = led["total_usd"]
        self.last_estimate = est
        log = _read_json(self.dir / "round_estimates.json", [])
        log.append({**est, "at": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime())})
        _write_json(self.dir / "round_estimates.json", log)
        if led["total_usd"] + est["likely_usd"] > self.study_cap + 1e-12:
            raise Refused(f"study cap: spent ${led['total_usd']:.4f} + this round's likely ${est['likely_usd']:.4f} "
                          f"passes ${self.study_cap:.2f}; round not submitted")
        return plan

    def submit_round(self, cap_usd: float) -> str:
        # The engine's own guard: this run's spend + the round's upper bound must fit in what the other arms left.
        led = update_ledger(self.root)
        mine = float(self.spend()["actual_usd"])
        room = self.study_cap - (led["total_usd"] - mine)
        r = super().submit_round(min(cap_usd, room))
        if r == "submitted" and self.last_estimate is not None:
            sp = self.spend()
            sp["rounds"][-1]["likely_usd"] = self.last_estimate["likely_usd"]
            self._write_spend(sp)
        return r


def run_arm(run_dir: Path, runner=None, *, poll: float = 60.0, log=print, root: Path = OUT,
            sleep=time.sleep, check_protocol: bool = True, max_steps: int | None = None) -> str:
    eng = XEngine(run_dir, runner, root=root, log=log, check_protocol=check_protocol)
    try:
        steps = 0
        while True:
            r = eng.step(STUDY_CAP_USD)
            update_ledger(root)
            steps += 1
            if r == "complete" or (max_steps is not None and steps >= max_steps):
                return r
            if r == "waiting":
                sleep(poll)
    finally:
        eng.close()

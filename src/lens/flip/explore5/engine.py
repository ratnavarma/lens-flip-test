"""Batch-round engine for the exploratory v5 arms, built on the headline Engine.

Each arm is its own run dir under results/exploratory-v5/<arm>/ with config.json, cases.jsonl, frame.jsonl,
calibration.json (and c1_controls.jsonl for C1). Requests, keys, the call cache, rounds, retries and the
per-run spend.json are the headline Engine's, as in explore4. Two guards sit on top of it, exactly as in v4:

1. The ledger (results/exploratory-v5/spend.json). Before a round is submitted, the round's likely cost is
   added to the ledger total. If the sum passes the cap (`--cap-usd`), the round is refused, nothing is sent.
2. The engine's own cap is the cap minus what the other arms have spent, so the round's upper bound
   (max_tokens on every request) must also fit inside what is left.

C3 is the one arm that bypasses the call cache: its five repeats of an identical request body carry five
different keys (the body key salted with the repeat tag), so they are five requests with five custom_ids.
The arm's config says `cache_bypass: true` and `cache_bypass.json` records the scheme.
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
from ..residue import note_view, residue_candidates
from ..schema import Case
from ..systems.base import count_tokens
from . import controls as CT
from . import prompts as P
from .frame import CALIBRATION_NAME, OUT, Handoff, load_frame

HAIKU = "claude-haiku-5-5"
RATES = {"input": 0.05, "output": 0.25}   # USD per million tokens, Batch, claude-haiku-5-5

# Arms in the protocol's order. RUN_ORDER is the order `run` takes them in: cheap and shallow first, X2's long
# search last, so a cap stop leaves the most arms finished.
ARMS = ("X1", "X2", "C1", "C2", "C3", "C4")
RUN_ORDER = ("X1", "C2", "C3", "C1", "C4", "X2")

DECIDER = {"model": HAIKU, "arm": "adaptive-high"}
NOTE_WRITER = {"model": HAIKU, "effort": "high", "thinking": None, "max_tokens": 6000}
REWRITER = {"model": HAIKU, "effort": "medium", "thinking": {"type": "disabled"}, "max_tokens": 512}
JUDGE = {"model": HAIKU, "effort": "medium", "thinking": {"type": "disabled"}, "max_tokens": 64}
REPEATS = 5

# Likely output tokens per request, by stage. Sources (real usage of the finished runs):
#   note writers   v4 C-budget c_A300: 1,373 per request (v3 residue_A at 150 tokens: 1,214; v4 c_A600: 1,489)
#   B on a note    v4 A-extractive x_B: 594 (v3 residue_B: 548)
#   B with units   v4 residue_flip: 882 (v3: 906); verify: v4 729 (v3 875)
#   rewriter       thinking off, JSON of at most 150 tokens: 175 (summary-kind outputs track their budget)
#   judge          omission/judge.py HAIKU_OUTPUT_TOKENS: 10
ASSUMED_OUT: dict[str, int] = {
    "x1_A": 1373, "x1_B": 594, "x2_A": 1373, "x2_B": 594, "residue_flip": 882, "residue_flip_verify": 729,
    "c1_B": 882, "c2_A": 175, "c2_B": 594, "c3_B": 594, "c4_judge": 10,
}


def arm_config(arm: str, assumed_out: dict[str, int] | None = None) -> dict[str, Any]:
    if arm not in ARMS:
        raise ValueError(f"unknown arm {arm!r}")
    from .. import batch
    arm_cfg = dict(batch.ARM_CONFIG[DECIDER["arm"]])
    arm_cfg.pop("assumed_out", None)
    cfg: dict[str, Any] = {
        "version": "exploratory-v5", "arm": arm, "residue_budget": 1000, "max_calls": 48, "strategy": "group",
        "decider": {**DECIDER, **arm_cfg}, "note_writer": dict(NOTE_WRITER), "note_tokens": P.NOTE_BUDGET,
        "rates": dict(RATES), "assumed_output_tokens": dict(assumed_out or ASSUMED_OUT),
        "cache": "state/calls.sqlite", "cache_bypass": False,
        "batch": {"max_requests": 100_000, "max_bytes": 128 * 1024 * 1024},
        "bootstrap": {"reps": 1000, "seed": 11},
    }
    if arm == "C1":
        cfg["c1"] = {"k": CT.K_CONTROLS, "tolerance": CT.TOLERANCE, "seed_tag": CT.SEED_TAG,
                     "max_attempts": CT.MAX_ATTEMPTS, "controls_file": "c1_controls.jsonl"}
    if arm == "C2":
        cfg["rewriter"] = {**REWRITER, "thinking": dict(REWRITER["thinking"])}
    if arm == "C3":
        cfg["repeats"] = REPEATS
        cfg["cache_bypass"] = True
        cfg["cache_bypass_scheme"] = "key = sha256(canonical body + NUL + 'repeat=' + tag); body sent unchanged"
    if arm == "C4":
        cfg["judge"] = {**JUDGE, "thinking": dict(JUDGE["thinking"]), "prompt": "omission/judge.build_prompt"}
    return cfg


# ---------------------------------------------------------------------------
# Requests and answers
# ---------------------------------------------------------------------------

def _json_schema_params(cfg: dict[str, Any], prompt: str, schema: dict[str, Any]) -> dict[str, Any]:
    params: dict[str, Any] = {
        "model": cfg["model"], "max_tokens": cfg["max_tokens"],
        "messages": [{"role": "user", "content": prompt}],
        "output_config": {"effort": cfg["effort"], "format": {"type": "json_schema", "schema": json.loads(json.dumps(schema))}},
    }
    if cfg.get("thinking") is not None:
        params["thinking"] = dict(cfg["thinking"])
    return params


def x1_request(view, case: Case, cfg: dict[str, Any]) -> dict[str, Any]:
    return _json_schema_params(cfg, P.build_x1_prompt(view, case), P.X1_SCHEMA)


def x2_request(view, case: Case, cfg: dict[str, Any]) -> dict[str, Any]:
    return _json_schema_params(cfg, P.build_x2_prompt(view, case), P.X2_SCHEMA)


def rewrite_request(note: str, cfg: dict[str, Any]) -> dict[str, Any]:
    return _json_schema_params(cfg, P.build_rewrite_prompt(note), P.REWRITE_SCHEMA)


def judge_request(note: str, unit: str, cfg: dict[str, Any]) -> dict[str, Any]:
    return _json_schema_params(cfg, P.build_judge_prompt(note, unit), P.JUDGE_SCHEMA)


def parse_any(kind: str, raw: dict[str, Any], meta: dict[str, Any]) -> dict[str, Any]:
    stop, text = raw.get("stop_reason"), raw.get("text")
    if kind in ("decide", "note", "summary"):
        return rq.parse_answer(kind, raw, meta)
    if kind == "note_x2":
        if stop == "refusal" or text is None:
            return {"note": "", "fields": None, "p_residue": None, "cut": False, "complete": False,
                    "status": stop or "empty"}
        r = P.parse_x2_reply(text)
        status = "max_tokens" if stop == "max_tokens" else ("ok" if r["complete"] and r["p_residue"] is not None
                                                            else ("fields_parse" if not r["complete"] else "p_parse"))
        return {**r, "status": status}
    if kind == "rewrite":
        if stop == "refusal" or text is None:
            return {"note": None, "status": stop or "empty"}
        note = P.parse_rewrite_reply(text)
        return {"note": note, "status": "max_tokens" if stop == "max_tokens" else ("ok" if note is not None else "parse")}
    if kind == "judge":
        if stop == "refusal" or text is None:
            return {"conveyed": None, "status": stop or "empty"}
        v = P.parse_judge_reply(text)
        return {"conveyed": v, "status": "max_tokens" if stop == "max_tokens" else ("ok" if v is not None else "parse")}
    raise ValueError(f"unknown kind {kind!r}")


def failure_any(kind: str, reason: str) -> dict[str, Any]:
    status = f"failed:{reason}"
    if kind == "decide":
        return {"action": PARSE_ERROR, "status": status}
    if kind == "note_x2":
        return {"note": "", "fields": None, "p_residue": None, "cut": False, "complete": False, "status": status}
    if kind == "rewrite":
        return {"note": None, "status": status}
    if kind == "judge":
        return {"conveyed": None, "status": status}
    return {"note": "", "p_residue": None, "status": status}


class XAsker(Asker):
    """tasks.Asker with the v5 reply kinds and the C3 salt. An item's extra may carry {"salt": tag}; that is
    allowed only when the arm's config says cache_bypass."""

    def ask_many(self, items):
        answers = self.ns.setdefault("answers", {})
        out, pending = [], False
        for tag, kind, build, stage, extra in items:
            if tag in answers:
                out.append(answers[tag])
                continue
            params = build()
            salt = (extra or {}).get("salt")
            if salt is None:
                key = rq.request_key(params)
            else:
                if not self.ctx.config.get("cache_bypass"):
                    raise RuntimeError("a salted request key needs cache_bypass in the arm config (C3 only)")
                key = CT.salted_key(rq.canonical(params), salt)
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


def _decide(a: XAsker, tag: str, note: str, h: Handoff, ctx, stage: str, inserted=None, extra=None) -> dict:
    return a.ask(tag, "decide", lambda: rq.decide_request(note_view(note), inserted or [], h.case, "v1",
                                                          ctx.config["decider"]), stage, extra)


def task_x1(h: Handoff, ns: dict, ctx) -> bool:
    """A writes a reader-conditioned note and p_residue; B decides. No flip search."""
    if "out" in ns:
        return True
    cfg = ctx.config
    a = _asker(h, ns, ctx)
    n = a.ask("n", "note", lambda: x1_request(h.view, h.case, cfg["note_writer"]), "x1_A")
    b = _decide(a, "B", n["note"], h, ctx, "x1_B")
    ns["out"] = {"note": n["note"], "note_tokens": count_tokens([n["note"]]), "note_status": n["status"],
                 "p_residue": n["p_residue"], "a_B": b["action"], "residue_flip": b["action"] != h.row["a_A"]}
    ctx.touch(h.case.case_id)
    return True


def task_x2(h: Handoff, ns: dict, ctx) -> bool:
    """A writes the structured note; B decides; on a flip the v3 flip search runs (group-first, 48 calls, v2 verify)."""
    cfg = ctx.config
    target = h.row["a_A"]
    a = _asker(h, ns, ctx)
    if "out" not in ns:
        n = a.ask("n", "note_x2", lambda: x2_request(h.view, h.case, cfg["note_writer"]), "x2_A")
        b = _decide(a, "B", n["note"], h, ctx, "x2_B")
        cands = residue_candidates(h.case, h.view, n["note"])
        ns["out"] = {"note": n["note"], "note_tokens": count_tokens([n["note"]]), "note_status": n["status"],
                     "fields": n["fields"], "cut": n["cut"], "complete": n["complete"], "p_residue": n["p_residue"],
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


def task_c1(h: Handoff, ns: dict, ctx) -> bool:
    """B decides on the v3 note plus each of K irrelevant insertions of the length of the minimal set."""
    if "out" in ns:
        return True
    rows = [c for c in ctx.controls.get((h.case.case_id, h.system), []) if c["send"]]
    if not rows:
        ns["out"] = {"controls": [], "restore_share": None}
        ctx.touch(h.case.case_id)
        return True
    note = h.row["v3"]["note"]
    by_id = {u.id: u for u in residue_candidates(h.case, h.view, note)}
    items = []
    for c in rows:
        missing = [i for i in c["unit_ids"] if i not in by_id]
        if missing:
            raise ValueError(f"{h.hid}: control {c['k']} names units {missing} that are not candidates of V")
        units = [by_id[i] for i in c["unit_ids"]]
        items.append((f"k{c['k']}", "decide",
                      lambda u=units: rq.decide_request(note_view(note), u, h.case, "v1", ctx.config["decider"]),
                      "c1_B", None))
    got = _asker(h, ns, ctx).ask_many(items)
    target = h.row["a_A"]
    ctl = [{"k": c["k"], "a_B": g["action"], "restored": g["action"] == target, "chars": c["chars"],
            "ratio": c["ratio"], "matched": c["matched"], "n_units": len(c["unit_ids"])} for c, g in zip(rows, got)]
    ns["out"] = {"controls": ctl, "restore_share": sum(c["restored"] for c in ctl) / len(ctl)}
    ctx.touch(h.case.case_id)
    return True


def task_c2(h: Handoff, ns: dict, ctx) -> bool:
    """The rewriter rewrites the v3 note (wording only); B decides on the rewrite."""
    if "out" in ns:
        return True
    cfg = ctx.config
    note = h.row["v3"]["note"]
    a = _asker(h, ns, ctx)
    r = a.ask("rw", "rewrite", lambda: rewrite_request(note, cfg["rewriter"]), "c2_A")
    if r["note"] is None:
        ns["out"] = {"failed": True, "rewrite_status": r["status"], "residue_flip": None}
        ctx.touch(h.case.case_id)
        return True
    b = _decide(a, "B", r["note"], h, ctx, "c2_B")
    ns["out"] = {"failed": False, "rewrite": r["note"], "rewrite_status": r["status"],
                 "rewrite_tokens": count_tokens([r["note"]]), "identical": r["note"] == note,
                 "a_B": b["action"], "residue_flip": b["action"] != h.row["a_A"]}
    ctx.touch(h.case.case_id)
    return True


def task_c3(h: Handoff, ns: dict, ctx) -> bool:
    """B decides on the identical v3 note R more times. Same body each time, a distinct key per repeat."""
    if "out" in ns:
        return True
    cfg = ctx.config
    note = h.row["v3"]["note"]
    items = [(f"rep{i}", "decide",
              lambda: rq.decide_request(note_view(note), [], h.case, "v1", cfg["decider"]), "c3_B",
              {"salt": f"rep{i}"}) for i in range(int(cfg["repeats"]))]
    got = _asker(h, ns, ctx).ask_many(items)
    ns["out"] = {"a_B": [g["action"] for g in got], "status": [g["status"] for g in got]}
    ctx.touch(h.case.case_id)
    return True


def task_c4(h: Handoff, ns: dict, ctx) -> bool:
    """The judge decides, for every unit of V, whether the v3 note conveys it."""
    if "out" in ns:
        return True
    cfg = ctx.config
    note = h.row["v3"]["note"]
    items = [(f"j{j:03d}", "judge", lambda u=u: judge_request(note, u, cfg["judge"]), "c4_judge", None)
             for j, u in enumerate(h.view.units)]
    got = _asker(h, ns, ctx).ask_many(items)
    ns["out"] = {"conveyed": [g["conveyed"] for g in got], "status": [g["status"] for g in got],
                 "n_units": len(got)}
    ctx.touch(h.case.case_id)
    return True


TASKS: dict[str, Callable[[Handoff, dict, Any], bool]] = {
    "X1": task_x1, "X2": task_x2, "C1": task_c1, "C2": task_c2, "C3": task_c3, "C4": task_c4,
}


# ---------------------------------------------------------------------------
# Ledger
# ---------------------------------------------------------------------------

def ledger_path(root: Path) -> Path:
    return root / "spend.json"


def update_ledger(root: Path, cap_usd: float | None = None) -> dict[str, Any]:
    """Sum actual spend from each arm's spend.json. Written after every fetch."""
    path = ledger_path(root)
    led = _read_json(path, {"cap_usd": cap_usd, "arms": {}})
    if cap_usd is not None:
        led["cap_usd"] = cap_usd
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
    led["total_usd"] = round(sum(a["actual_usd"] for a in arms.values()), 6)
    led["remaining_usd"] = None if led.get("cap_usd") is None else round(led["cap_usd"] - led["total_usd"], 6)
    led["updated_at"] = time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime())
    _write_json(path, led)
    return led


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------

def init_arm(arm: str, root: Path = OUT, assumed_out: dict[str, int] | None = None) -> Path:
    d = root / arm
    if (d / "config.json").exists():
        raise Refused(f"{d} already holds a run")
    (d / "state" / "cases").mkdir(parents=True, exist_ok=True)
    for name in ("cases.jsonl", "frame.jsonl"):
        shutil.copyfile(root / name, d / name)
    shutil.copyfile(root / CALIBRATION_NAME, d / "calibration.json")
    if arm == "C1":
        shutil.copyfile(root / "c1_controls.jsonl", d / "c1_controls.jsonl")
    cfg = arm_config(arm, assumed_out)
    if cfg.get("cache_bypass"):
        _write_json(d / "cache_bypass.json", {
            "arm": arm, "cache_bypass": True, "scheme": cfg["cache_bypass_scheme"], "repeats": cfg["repeats"],
            "why": "five identical request bodies must be five requests; the call cache would merge them into one",
            "other_arms": "keyed by body alone, cache on"})
    _write_json(d / "config.json", cfg)
    return d


class XEngine(Engine):
    def __init__(self, run_dir, runner=None, *, root: Path = OUT, cap_usd: float, log=None, now=time.time,
                 check_protocol: bool = True):
        super().__init__(run_dir, runner, log=log, now=now, check_protocol=check_protocol)
        self.root = Path(root)
        self.cap = float(cap_usd)
        self.arm = self.config["arm"]
        self.task = TASKS[self.arm]
        hs = load_frame(self.dir, self.cases)
        self.handoffs: dict[str, list[Handoff]] = {}
        for h in hs:
            self.handoffs.setdefault(h.case.case_id, []).append(h)
        self.cases = [c for c in self.cases if c.case_id in self.handoffs]
        self.controls: dict[tuple[str, str], list[dict]] = {}
        cf = self.config.get("c1", {}).get("controls_file")
        if cf:
            for line in (self.dir / cf).read_text(encoding="utf-8").splitlines():
                if line.strip():
                    r = json.loads(line)
                    self.controls.setdefault((r["case_id"], r["system"]), []).append(r)
            for v in self.controls.values():
                v.sort(key=lambda r: r["k"])
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
            raise Refused("v5 protocol check failed: " + ", ".join(bad))

    def likely_estimate(self, pending) -> dict[str, Any]:
        calib = self.calibration()
        rates = self.config["rates"]
        assumed = self.config["assumed_output_tokens"]
        tin = tout = 0
        by_kind: Counter = Counter()
        for key, (params, meta) in pending.items():
            kind = meta["kind"]
            tin += rq.estimate_input_tokens(rq.request_chars(params), kind, calib, bound=False)
            tout += int(assumed.get(meta["stage"], assumed.get(kind, params["max_tokens"])))
            by_kind[kind] += 1
        return {"n_requests": len(pending), "by_kind": dict(by_kind), "input_tokens": tin, "output_tokens": tout,
                "likely_usd": round(rq.price(tin, tout, rates), 6)}

    def plan_round(self, pending):
        plan = super().plan_round(pending)
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
        # The engine's own guard: this run's spend + the round's upper bound must fit in what the other arms left.
        led = update_ledger(self.root, self.cap)
        mine = float(self.spend()["actual_usd"])
        room = self.cap - (led["total_usd"] - mine)
        r = super().submit_round(min(cap_usd, room))
        if r == "submitted" and self.last_estimate is not None:
            sp = self.spend()
            sp["rounds"][-1]["likely_usd"] = self.last_estimate["likely_usd"]
            self._write_spend(sp)
        return r


def run_arm(run_dir: Path, runner=None, *, cap_usd: float, poll: float = 60.0, log=print, root: Path = OUT,
            sleep=time.sleep, check_protocol: bool = True, max_steps: int | None = None) -> str:
    eng = XEngine(run_dir, runner, root=root, cap_usd=cap_usd, log=log, check_protocol=check_protocol)
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


def run_arms(root: Path, arms: list[str], *, cap_usd: float, runner=None, poll: float = 60.0, log=print,
             sleep=time.sleep, check_protocol: bool = True, max_steps: int | None = None) -> dict[str, str]:
    """Run the chosen arms one after another in RUN_ORDER, resuming any arm whose dir exists. A refusal by either
    guard stops the whole run (the stop rule); it is never worked around."""
    bad = [a for a in arms if a not in ARMS]
    if bad or len(set(arms)) != len(arms):
        raise ValueError(f"--arms needs distinct names from {', '.join(ARMS)}; got {', '.join(arms)}")
    out: dict[str, str] = {}
    for arm in [a for a in RUN_ORDER if a in arms]:
        d = root / arm
        if not (d / "config.json").exists():
            init_arm(arm, root)
        out[arm] = run_arm(d, runner, cap_usd=cap_usd, poll=poll, log=log, root=root, sleep=sleep,
                           check_protocol=check_protocol, max_steps=max_steps)
    return out

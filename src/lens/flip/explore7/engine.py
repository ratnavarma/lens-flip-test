"""Batch-round engine for the two v7 studies, built on the v5 XEngine and the headline Engine.

Two run dirs under results/exploratory-v7: R (arms R1, R2, R3 on the 490 eligible handoffs) and X (arms X1p and
X1r on the 490 eligible handoffs, arm XW on the 670 A-wrong handoffs). A stage name carries its arm, so the
spend per arm is read from spend.json by stage. Requests, keys, the call cache, rounds, retries and the per-run
spend.json are the headline Engine's. The two guards of v4 to v6 apply unchanged, on the v7 ledger:

1. Before a round is submitted, the ledger total plus the round's likely cost must stay at or under the cap.
2. The round's upper bound (max_tokens on every request) must fit inside the cap minus the other study's spend.

Round size. A round holds at most `max_requests_per_round` requests (the first ones by request key). The rest
wait for the next round. This keeps each round's upper bound inside the cap; it changes no request and no key.
A note is always answered before the decision that reads it, because the decision's request cannot be built
without the note.

Reader B: `decide_request(note_view(note), [], case, "v1", decider)` with the v3 decider settings. R1 sends the
same body with the note-framed system line and user message of prompts.py. Writer A: the v3 note-writer settings.
Every request is keyed by its body; identical bodies share one request.
"""

from __future__ import annotations

import shutil
import time
from pathlib import Path
from typing import Any, Callable

from ..explore5 import engine as E5
from ..headline import requests as rq
from ..headline.engine import Engine, Refused, _read_json, _write_json
from ..residue import note_view
from ..systems.base import count_tokens
from . import frame as F
from . import prompts as P
from .frame import OUT, Handoff, load_frame

HAIKU = E5.HAIKU
RATES = E5.RATES
ARMS = ("R", "X")                 # run dirs (studies)
RUN_ORDER = ("R", "X")
SUB_ARMS = ("R1", "R2", "R3", "X1p", "X1r", "XW")
STAGES: dict[str, tuple[str, ...]] = {
    "R1": ("r1_B",), "R2": ("r2_B",), "R3": ("r3_B",),
    "X1p": ("x1p_A", "x1p_B"), "X1r": ("x1r_A", "x1r_B"),
    "XW": ("xw_v3_A", "xw_v3_B", "xw_x1_A", "xw_x1_B"),
}
PROMPT_VERSION = "v1"
MAX_REQUESTS_PER_ROUND = 1000

# Likely output tokens per request, from the stored usage of the calls each stage repeats:
#   B on a 150-token note   548 per request, the figure v5 used for this stage (stored: v5 C3 c3_B 546, v3 residue_B 535)
#   v3 note writer          v3 residue_A 1,214
#   X1-type note writer     v5 X1 x1_A 1,231
ASSUMED_OUT: dict[str, int] = {
    "r1_B": 548, "r2_B": 548, "r3_B": 548, "x1p_A": 1231, "x1p_B": 548, "x1r_A": 1231, "x1r_B": 548,
    "xw_v3_A": 1214, "xw_v3_B": 548, "xw_x1_A": 1231, "xw_x1_B": 548,
}


def arm_config(arm: str, assumed_out: dict[str, int] | None = None) -> dict[str, Any]:
    if arm not in ARMS:
        raise ValueError(f"unknown arm {arm!r}")
    base = E5.arm_config("X1")  # the v5 settings of A and B, the rates, the batch limits and the bootstrap
    stages = [s for sub in SUB_ARMS if sub[0] == arm for s in STAGES[sub]]
    ao = dict(assumed_out or ASSUMED_OUT)
    cfg: dict[str, Any] = {
        "version": "exploratory-v7", "arm": arm, "sub_arms": [s for s in SUB_ARMS if s[0] == arm],
        "residue_budget": base["residue_budget"], "decider": dict(base["decider"]), "prompt_version": PROMPT_VERSION,
        "rates": dict(base["rates"]), "assumed_output_tokens": {s: ao[s] for s in stages},
        "cache": "state/calls.sqlite", "cache_bypass": False, "max_requests_per_round": MAX_REQUESTS_PER_ROUND,
        "batch": dict(base["batch"]), "bootstrap": dict(base["bootstrap"]),
    }
    if arm == "R":
        cfg["notes_file"] = F.R_NOTES
        cfg["unchanged_notes"] = "the v3 decision stands; no request"
        cfg["emptied_notes"] = "counted; no request"
    else:
        cfg["note_writer"] = dict(base["note_writer"])
        cfg["note_tokens"] = base["note_tokens"]
        cfg["wrong_frame"] = F.WRONG_FRAME
    return cfg


# ---------------------------------------------------------------------------
# Requests
# ---------------------------------------------------------------------------

def v1_request(note: str, h: Handoff, decider: dict[str, Any]) -> dict[str, Any]:
    """B on a note with the v1 prompt: the v3 residue_B request."""
    return rq.decide_request(note_view(note), [], h.case, PROMPT_VERSION, decider)


def r1_request(note: str, h: Handoff, decider: dict[str, Any]) -> dict[str, Any]:
    """The v1 request with the note-framed system line and user message. Settings and schema are untouched."""
    p = v1_request(note, h, decider)
    p["system"] = P.R1_SYSTEM
    p["messages"] = [{"role": "user", "content": P.build_r1_user(note, h.case)}]
    return p


def v3_note_request(h: Handoff, writer: dict[str, Any]) -> dict[str, Any]:
    """The frozen v3 note request (residue_A)."""
    return rq.note_request(h.view, h.case, writer)


def x1_note_request(h: Handoff, writer: dict[str, Any]) -> dict[str, Any]:
    """The frozen X1 note request of v5."""
    return E5.x1_request(h.view, h.case, writer)


def x1p_note_request(h: Handoff, writer: dict[str, Any]) -> dict[str, Any]:
    return E5._json_schema_params(writer, P.build_x1p_prompt(h.view, h.case), P.X1_SPLIT_SCHEMA)


def x1r_note_request(h: Handoff, writer: dict[str, Any]) -> dict[str, Any]:
    return E5._json_schema_params(writer, P.build_x1r_prompt(h.view, h.case), P.X1_SPLIT_SCHEMA)


# ---------------------------------------------------------------------------
# Tasks
# ---------------------------------------------------------------------------

def task_r(h: Handoff, ns: dict, ctx) -> bool:
    """R1 on every v3 note; R2 and R3 on the notes their rule changed and did not empty."""
    if "out" in ns:
        return True
    rn = ctx.r_notes[(h.case.case_id, h.system)]
    dec = ctx.config["decider"]
    v3 = h.row["v3"]
    items: list[tuple] = [("r1", "decide", lambda: r1_request(v3["note"], h, dec), "r1_B", None)]
    for arm in ("r2", "r3"):
        if rn[f"{arm}_changed"] and not rn[f"{arm}_empty"]:
            items.append((arm, "decide", lambda n=rn[f"{arm}_note"]: v1_request(n, h, dec), f"{arm}_B", None))
    got = dict(zip([i[0] for i in items], E5._asker(h, ns, ctx).ask_many(items)))
    out: dict[str, Any] = {"r1": {"a_B": got["r1"]["action"], "status": got["r1"]["status"], "source": "new"}}
    for arm in ("r2", "r3"):
        if arm in got:
            out[arm] = {"a_B": got[arm]["action"], "status": got[arm]["status"], "source": "new", "changed": True, "empty": False}
        elif rn[f"{arm}_empty"]:
            out[arm] = {"a_B": None, "status": None, "source": "none", "changed": True, "empty": True}
        else:
            out[arm] = {"a_B": v3["a_B"], "status": "v3", "source": "v3", "changed": False, "empty": False}
    ns["out"] = out
    ctx.touch(h.case.case_id)
    return True


def task_x(h: Handoff, ns: dict, ctx) -> bool:
    """Two notes per handoff, then B on each. Eligible: the X1p and X1r notes. A-wrong: a v3 note and an X1 note."""
    if "out" in ns:
        return True
    kind = ctx.kind[(h.case.case_id, h.system)]
    nw, dec = ctx.config["note_writer"], ctx.config["decider"]
    if kind == "eligible":
        arms = (("x1p", lambda: x1p_note_request(h, nw), "x1p"), ("x1r", lambda: x1r_note_request(h, nw), "x1r"))
    else:
        arms = (("v3", lambda: v3_note_request(h, nw), "xw_v3"), ("x1", lambda: x1_note_request(h, nw), "xw_x1"))
    a = E5._asker(h, ns, ctx)
    notes = a.ask_many([(f"n_{t}", "note", build, f"{st}_A", None) for t, build, st in arms])
    decs = a.ask_many([(f"B_{t}", "decide", lambda n=n["note"]: v1_request(n, h, dec), f"{st}_B", None)
                       for (t, _, st), n in zip(arms, notes)])
    ns["out"] = {"set": kind, **{t: {"note": n["note"], "note_tokens": count_tokens([n["note"]]), "note_status": n["status"],
                                     "p_residue": n["p_residue"], "a_B": d["action"], "status": d["status"]}
                                 for (t, _, _), n, d in zip(arms, notes, decs)}}
    ctx.touch(h.case.case_id)
    return True


TASKS: dict[str, Callable[[Handoff, dict, Any], bool]] = {"R": task_r, "X": task_x}


# ---------------------------------------------------------------------------
# Ledger
# ---------------------------------------------------------------------------

def ledger_path(root: Path) -> Path:
    return root / "spend.json"


def update_ledger(root: Path, cap_usd: float | None = None) -> dict[str, Any]:
    """Sum actual spend from each v7 run dir's spend.json. Written after every fetch. Same shape as the v5 ledger."""
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
                                                         "likely_usd", "by_stage", "failed_or_retried")}
                                for r in sp.get("rounds", [])],
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
    cfg = arm_config(arm, assumed_out)
    (d / "state" / "cases").mkdir(parents=True, exist_ok=True)
    for name in ("cases.jsonl", "frame.jsonl"):
        shutil.copyfile(root / name, d / name)
    shutil.copyfile(root / F.CALIBRATION_NAME, d / "calibration.json")
    name = F.R_NOTES if arm == "R" else F.WRONG_FRAME
    shutil.copyfile(root / name, d / name)
    _write_json(d / "config.json", cfg)
    return d


class V7Engine(E5.XEngine):
    """The v5 XEngine with the v7 tasks, the v7 protocol check, the v7 ledger and the round size limit."""

    def __init__(self, run_dir, runner=None, *, root: Path = OUT, cap_usd: float, log=None, now=time.time,
                 check_protocol: bool = True):
        Engine.__init__(self, run_dir, runner, log=log, now=now, check_protocol=check_protocol)
        self.root = Path(root)
        self.cap = float(cap_usd)
        self.arm = self.config["arm"]
        if self.arm not in ARMS:
            raise Refused(f"{self.dir} is arm {self.arm}, not one of {', '.join(ARMS)}")
        self.task = TASKS[self.arm]
        hs = load_frame(self.dir, self.cases)
        self.kind: dict[tuple[str, str], str] = {(h.case.case_id, h.system): "eligible" for h in hs}
        self.r_notes: dict[tuple[str, str], dict] = {}
        if self.arm == "R":
            self.r_notes = F.by_key(F._read_jsonl(self.dir / self.config["notes_file"]))
        else:
            wrong = F.load_wrong(self.dir, self.cases)
            clash = [h.hid for h in wrong if (h.case.case_id, h.system) in self.kind]
            if clash:
                raise Refused(f"A-wrong handoffs that are also eligible: {', '.join(clash[:3])}")
            self.kind.update({(h.case.case_id, h.system): "a_wrong" for h in wrong})
            hs = hs + wrong
        self.handoffs: dict[str, list[Handoff]] = {}
        for h in hs:
            self.handoffs.setdefault(h.case.case_id, []).append(h)
        self.cases = [c for c in self.cases if c.case_id in self.handoffs]
        self.controls = {}
        self.last_estimate: dict[str, Any] | None = None

    def advance(self):
        pending = E5.XEngine.advance(self)
        limit = int(self.config.get("max_requests_per_round") or 0)
        if limit and len(pending) > limit:
            self.pending = {k: pending[k] for k in sorted(pending)[:limit]}
        return self.pending

    def _check_protocol(self) -> None:
        if not self.check_protocol:
            return
        from .protocol import check
        bad = check(self.root, self.dir)
        if bad:
            raise Refused("v7 protocol check failed: " + ", ".join(bad))

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
        # The engine's own guard: this run's spend + the round's upper bound must fit in what the other study left.
        led = update_ledger(self.root, self.cap)
        mine = float(self.spend()["actual_usd"])
        room = self.cap - (led["total_usd"] - mine)
        r = Engine.submit_round(self, min(cap_usd, room))
        if r == "submitted" and self.last_estimate is not None:
            sp = self.spend()
            sp["rounds"][-1]["likely_usd"] = self.last_estimate["likely_usd"]
            self._write_spend(sp)
        return r


def run_arm(run_dir: Path, runner=None, *, cap_usd: float, poll: float = 60.0, log=print, root: Path = OUT,
            sleep=time.sleep, check_protocol: bool = True, max_steps: int | None = None) -> str:
    eng = V7Engine(run_dir, runner, root=root, cap_usd=cap_usd, log=log, check_protocol=check_protocol)
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
    """Run the chosen studies one after another in RUN_ORDER, resuming any whose dir exists. A refusal by either
    guard stops the whole run (the stop rule); it is never worked around."""
    bad = [a for a in arms if a not in ARMS]
    if bad or len(set(arms)) != len(arms):
        raise ValueError(f"--arms needs distinct names from {', '.join(ARMS)}; got {', '.join(arms)}")
    if check_protocol:
        from .protocol import check
        failed = check(root)
        if failed:
            raise Refused("v7 protocol check failed: " + ", ".join(failed))
    out: dict[str, str] = {}
    for arm in [a for a in RUN_ORDER if a in arms]:
        d = root / arm
        if not (d / "config.json").exists():
            init_arm(arm, root)
        out[arm] = run_arm(d, runner, cap_usd=cap_usd, poll=poll, log=log, root=root, sleep=sleep,
                           check_protocol=check_protocol, max_steps=max_steps)
    return out

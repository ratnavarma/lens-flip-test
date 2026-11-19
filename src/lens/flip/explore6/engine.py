"""Batch-round engine for the two v6 arms, built on the v5 XEngine and the headline Engine.

Each arm is its own run dir under results/exploratory-v6/<arm>/ with config.json, cases.jsonl, frame.jsonl,
calibration.json and the arm's frozen inputs (E1: e1_flips.jsonl and e1_controls.jsonl; P1: p1_min_sets.jsonl).
Requests, keys, the call cache, rounds, retries and the per-run spend.json are the headline Engine's. The two v5
guards apply unchanged, on the v6 ledger results/exploratory-v6/spend.json:

1. Before a round is submitted, the ledger total plus the round's likely cost must stay at or under the cap.
2. The round's upper bound (max_tokens on every request) must fit inside the cap minus the other arm's spend.

Reader B. Every request is `headline.requests.decide_request(note_view(note), units, case, "v1", decider)` with
the decider settings of v3, v4 and v5 (`explore5.engine.DECIDER` plus `batch.ARM_CONFIG["adaptive-high"]`).

Keys. A control is keyed by its request body, as in v5 C1; the arm's call cache starts empty, so every control
is a new sample. A fresh re-read is keyed by the v5 C3 salt rule (`explore5.controls.salted_key`): the body sent
is the search's restoring call, unchanged, and the five re-reads carry five keys and five custom_ids.
"""

from __future__ import annotations

import shutil
import time
from pathlib import Path
from typing import Any, Callable

from ..explore5 import engine as E
from ..headline import requests as rq
from ..headline.engine import Engine, Refused, _read_json, _write_json
from ..residue import note_view, residue_candidates
from . import controls as CT
from . import frame as F
from .frame import OUT, Handoff, load_frame

HAIKU = E.HAIKU
RATES = E.RATES
ARMS = ("E1", "P1")
RUN_ORDER = ("E1", "P1")
DECIDER = E.DECIDER
PROMPT_VERSION = "v1"

# Likely output tokens per request, from the stored usage of the calls these arms repeat:
#   E1  v4 A-extractive residue_flip (the extractive search's own calls): 882.1 per request over 1,839 requests
#   P1  v5 C1 c1_B (B on a v3 note plus inserted units): 779.4 over 626; the v3 searches ran 906, so 906 is kept
ASSUMED_OUT: dict[str, int] = {"e1_ctl_B": 882, "e1_fresh_B": 882, "p1_fresh_B": 906}
BYPASS_SCHEME = "key = sha256(canonical body + NUL + 'repeat=' + tag); body sent unchanged"


def arm_config(arm: str, assumed_out: dict[str, int] | None = None) -> dict[str, Any]:
    if arm not in ARMS:
        raise ValueError(f"unknown arm {arm!r}")
    base = E.arm_config("C3")  # the v5 settings of B, the rates, the batch limits and the bootstrap
    cfg: dict[str, Any] = {
        "version": "exploratory-v6", "arm": arm, "residue_budget": base["residue_budget"],
        "decider": dict(base["decider"]), "prompt_version": PROMPT_VERSION,
        "rates": dict(base["rates"]), "assumed_output_tokens": dict(assumed_out or ASSUMED_OUT),
        "cache": "state/calls.sqlite", "cache_bypass": True, "cache_bypass_scheme": BYPASS_SCHEME,
        "rereads": CT.REREADS, "reread_tags": [CT.reread_salt(i) for i in range(CT.REREADS)],
        "batch": dict(base["batch"]), "bootstrap": dict(base["bootstrap"]),
    }
    if arm == "E1":
        cfg["flips_file"] = F.E1_FLIPS
        cfg["controls"] = {"k": CT.K_CONTROLS, "tolerance": CT.TOLERANCE, "seed_tag": CT.SEED_TAG,
                           "max_attempts": CT.MAX_ATTEMPTS, "controls_file": F.E1_CONTROLS,
                           "key": "request body; the arm's cache starts empty"}
    else:
        cfg["min_sets_file"] = F.P1_MIN_SETS
    return cfg


# ---------------------------------------------------------------------------
# Tasks
# ---------------------------------------------------------------------------

def restoring_request(note: str, units: list, h: Handoff, decider: dict[str, Any]) -> dict[str, Any]:
    """The search's call for one set of inserted units: the note as the view, the units under the inserted block."""
    return rq.decide_request(note_view(note), list(units), h.case, PROMPT_VERSION, decider)


def _reread_items(note: str, units: list, h: Handoff, ctx, stage: str) -> list[tuple]:
    dec = ctx.config["decider"]
    return [(f"t{i}", "decide", lambda: restoring_request(note, units, h, dec), stage, {"salt": CT.reread_salt(i)})
            for i in range(int(ctx.config["rereads"]))]


def _fresh(got: list[dict], target: str) -> list[dict[str, Any]]:
    return [{"i": i, "a_B": g["action"], "restored": g["action"] == target, "status": g["status"]}
            for i, g in enumerate(got)]


def task_e1(h: Handoff, ns: dict, ctx) -> bool:
    """On an extractive flip: B on the extractive note plus each sent control set, and, when the search restored
    the flip, five fresh re-reads of the note plus the minimal restoring set. Other handoffs have nothing to ask."""
    if "out" in ns:
        return True
    key = (h.case.case_id, h.system)
    rec = ctx.e1.get(key)
    if rec is None:
        ns["out"] = {"flip": False}
        ctx.touch(h.case.case_id)
        return True
    note = rec["note"]
    cands = residue_candidates(h.case, h.view, note)
    dec = ctx.config["decider"]
    rows = [c for c in ctx.controls.get(key, []) if c["send"]]
    items: list[tuple] = []
    for c in rows:
        units = F.units_in_order(cands, c["unit_ids"])
        items.append((f"k{c['k']}", "decide", lambda u=units: restoring_request(note, u, h, dec), "e1_ctl_B", None))
    restored = rec["status"] == "restored"
    if restored:
        items += _reread_items(note, F.units_in_order(cands, rec["unit_ids"]), h, ctx, "e1_fresh_B")
    got = E._asker(h, ns, ctx).ask_many(items) if items else []
    target = h.row["a_A"]
    ctl = [{"k": c["k"], "a_B": g["action"], "restored": g["action"] == target, "status": g["status"],
            "chars": c["chars"], "ratio": c["ratio"], "matched": c["matched"], "n_units": len(c["unit_ids"])}
           for c, g in zip(rows, got)]
    ns["out"] = {"flip": True, "search_restored": restored, "controls": ctl, "fresh": _fresh(got[len(rows):], target)}
    ctx.touch(h.case.case_id)
    return True


def task_p1(h: Handoff, ns: dict, ctx) -> bool:
    """On a v3 flip the v3 search restored: five fresh re-reads of the v3 note plus the v3 minimal restoring set."""
    if "out" in ns:
        return True
    rec = ctx.p1.get((h.case.case_id, h.system))
    if rec is None or rec["status"] != "restored":
        ns["out"] = {"flip": rec is not None, "search_restored": False, "fresh": []}
        ctx.touch(h.case.case_id)
        return True
    note = h.row["v3"]["note"]
    cands = residue_candidates(h.case, h.view, note)
    got = E._asker(h, ns, ctx).ask_many(_reread_items(note, F.units_in_order(cands, rec["unit_ids"]), h, ctx,
                                                     "p1_fresh_B"))
    ns["out"] = {"flip": True, "search_restored": True, "fresh": _fresh(got, h.row["a_A"])}
    ctx.touch(h.case.case_id)
    return True


TASKS: dict[str, Callable[[Handoff, dict, Any], bool]] = {"E1": task_e1, "P1": task_p1}


# ---------------------------------------------------------------------------
# Ledger
# ---------------------------------------------------------------------------

def ledger_path(root: Path) -> Path:
    return root / "spend.json"


def update_ledger(root: Path, cap_usd: float | None = None) -> dict[str, Any]:
    """Sum actual spend from each v6 arm's spend.json. Written after every fetch. Same shape as the v5 ledger."""
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
    for name in ((F.E1_FLIPS, F.E1_CONTROLS) if arm == "E1" else (F.P1_MIN_SETS,)):
        shutil.copyfile(root / name, d / name)
    _write_json(d / "cache_bypass.json", {
        "arm": arm, "cache_bypass": True, "scheme": cfg["cache_bypass_scheme"], "rereads": cfg["rereads"],
        "tags": cfg["reread_tags"], "applies_to": "the fresh re-reads of a minimal restoring set",
        "why": "five identical request bodies must be five requests; the call cache would merge them into one",
        "controls": "keyed by body alone (E1 only); the arm's cache starts empty, so each is a new sample"})
    _write_json(d / "config.json", cfg)
    return d


class V6Engine(E.XEngine):
    """The v5 XEngine with the v6 tasks, the v6 protocol check and the v6 ledger."""

    def __init__(self, run_dir, runner=None, *, root: Path = OUT, cap_usd: float, log=None, now=time.time,
                 check_protocol: bool = True):
        Engine.__init__(self, run_dir, runner, log=log, now=now, check_protocol=check_protocol)
        self.root = Path(root)
        self.cap = float(cap_usd)
        self.arm = self.config["arm"]
        if self.arm not in ARMS:
            raise Refused(f"{self.dir} is arm {self.arm}, not one of {', '.join(ARMS)}")
        self.task = TASKS[self.arm]
        self.handoffs: dict[str, list[Handoff]] = {}
        for h in load_frame(self.dir, self.cases):
            self.handoffs.setdefault(h.case.case_id, []).append(h)
        self.cases = [c for c in self.cases if c.case_id in self.handoffs]
        self.e1: dict[tuple[str, str], dict] = {}
        self.p1: dict[tuple[str, str], dict] = {}
        self.controls: dict[tuple[str, str], list[dict]] = {}
        if self.arm == "E1":
            self.e1 = F.load_e1_flips(self.dir)
            self.controls = F.load_controls(self.dir / self.config["controls"]["controls_file"])
        else:
            self.p1 = F.load_p1_sets(self.dir)
        self.last_estimate: dict[str, Any] | None = None

    def _check_protocol(self) -> None:
        if not self.check_protocol:
            return
        from .protocol import check
        bad = check(self.root, self.dir)
        if bad:
            raise Refused("v6 protocol check failed: " + ", ".join(bad))

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
        # The engine's own guard: this run's spend + the round's upper bound must fit in what the other arm left.
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
    eng = V6Engine(run_dir, runner, root=root, cap_usd=cap_usd, log=log, check_protocol=check_protocol)
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
    if check_protocol:
        from .protocol import check
        failed = check(root)
        if failed:
            raise Refused("v6 protocol check failed: " + ", ".join(failed))
    out: dict[str, str] = {}
    for arm in [a for a in RUN_ORDER if a in arms]:
        d = root / arm
        if not (d / "config.json").exists():
            init_arm(arm, root)
        out[arm] = run_arm(d, runner, cap_usd=cap_usd, poll=poll, log=log, root=root, sleep=sleep,
                           check_protocol=check_protocol, max_steps=max_steps)
    return out

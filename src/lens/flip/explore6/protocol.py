"""Freeze and check for exploratory protocol v6.

`freeze` writes results/exploratory-v6/protocol_v6.json once, before any model call, with status DRAFT: sha256 of
the control rule, the re-read rule, the draw code, the pre-registered analysis, the v3 prompts B reads, each
arm's request settings, and the run files (frame, cases, extractive flips, control draws, v3 minimal sets,
input-token rule). It also records the dry-run price. `finalize` turns DRAFT into FINAL and refuses when the
dry-run likely price is above PRICE_GATE_USD or a round's upper bound would not fit the run cap. The engine
refuses to run before FINAL and whenever a hash differs. `check` is run by the engine before every step.
"""

from __future__ import annotations

import hashlib
import inspect
import json
import time
from pathlib import Path
from typing import Any

from ..explore4.protocol import REUSED_V3
from ..explore5 import controls as CT5
from ..headline import requests as rq
from ..headline.protocol import V2_HASHES, current_hashes
from . import SCHEMA, VERSION
from . import controls as CT
from . import frame as F
from .frame import OUT

NAME = "protocol_v6.json"
DRYRUN_NAME = "dryrun_v6.json"
DOC = "docs/flip-protocol-v6-exploratory.md"
PRICE_GATE_USD = 1.50   # finalize refuses above this dry-run likely price
RUN_CAP_USD = 3.00      # the cap the run is started with; finalize checks that each round's upper bound fits it
RUN_FILES = {"frame_sha256": "frame.jsonl", "cases_sha256": "cases.jsonl", "e1_flips_sha256": F.E1_FLIPS,
             "e1_controls_sha256": F.E1_CONTROLS, "p1_min_sets_sha256": F.P1_MIN_SETS,
             "calibration_sha256": F.CALIBRATION_NAME}
ARM_FILES = {"E1": ("e1_flips_sha256", "e1_controls_sha256"), "P1": ("p1_min_sets_sha256",)}


def _sha(text: str | bytes) -> str:
    return hashlib.sha256(text.encode("utf-8") if isinstance(text, str) else text).hexdigest()


def rule_texts() -> dict[str, str]:
    from .analysis import ANALYSIS_SPEC
    from .engine import BYPASS_SCHEME, PROMPT_VERSION
    return {
        "x6.e1.control_rule": rq.canonical({
            "k": CT.K_CONTROLS, "tolerance": CT.TOLERANCE, "max_attempts": CT.MAX_ATTEMPTS,
            "seed": CT.SEED_TAG + "|<case>|<system>|<k>", "length": "len('\\n'.join(texts))",
            "pool": "residue_candidates(case, V, extractive note) minus the minimal restoring set",
            "non_recoverable_target": "median minimal-set length of the restored extractive flips",
            "draw": "explore5.controls.draw_control", "send": "not empty and not a copy of an earlier control of the flip",
            "key": "request body", "shown": "note as the view, units as inserted units, view order"}),
        "x6.draw_source": inspect.getsource(CT5.draw_control),
        "x6.reread_rule": rq.canonical({
            "rereads": CT.REREADS, "tags": [CT.reread_salt(i) for i in range(CT.REREADS)],
            "request": "decide_request(note_view(note), minimal set in view order, case, prompt, decider)",
            "prompt": PROMPT_VERSION, "key": BYPASS_SCHEME, "salt_source": inspect.getsource(CT5.salted_key),
            "e1": "the extractive note and the minimal set of the v4 extractive search, restored flips only",
            "p1": "the v3 note and the minimal set of the v3 search, restored flips only"}),
        "x6.analysis": rq.canonical(ANALYSIS_SPEC),
    }


def config_for_hash(cfg: dict[str, Any]) -> dict[str, Any]:
    return {k: v for k, v in cfg.items() if k != "assumed_output_tokens"}


def current() -> dict[str, Any]:
    from .engine import ARMS, arm_config
    v3 = current_hashes()
    return {"rules": {k: _sha(v) for k, v in rule_texts().items()},
            "reused_v3": {k: v3[k] for k in REUSED_V3},
            "arm_configs": {a: _sha(rq.canonical(config_for_hash(arm_config(a)))) for a in ARMS}}


def _file_hashes(root: Path) -> dict[str, str]:
    return {k: _sha((root / name).read_bytes()) for k, name in RUN_FILES.items()}


def _stamp(now: float | None = None) -> str:
    return time.strftime("%Y-%m-%d %H:%M UTC", time.gmtime(now if now is not None else time.time()))


def freeze(root: Path = OUT, now: float | None = None) -> dict[str, Any]:
    path = root / NAME
    if path.exists():
        raise SystemExit(f"{path} exists; v6 is frozen")
    dr_path = root / DRYRUN_NAME
    if not dr_path.exists():
        raise SystemExit(f"no {DRYRUN_NAME}; run dryrun first")
    dr = json.loads(dr_path.read_text())
    cur = current()
    bad = [k for k, v in cur["reused_v3"].items() if k in V2_HASHES and V2_HASHES[k] != v]
    if bad:
        raise SystemExit("reused v3 prompts drifted from the frozen hashes: " + ", ".join(bad))
    files = _file_hashes(root)
    if dr["inputs_sha256"] != {name: files[k] for k, name in RUN_FILES.items()}:
        raise SystemExit("the dry run priced other run files; run dryrun again")
    rec = {"schema": SCHEMA, "version": VERSION, "status": "DRAFT", "doc": DOC, "frozen_at": _stamp(now), **cur, **files,
           "dryrun": {"likely_usd": dr["total"]["likely_usd"], "upper_usd": dr["total"]["upper_usd"],
                      "requests": dr["total"]["requests"], "at": dr["at"], "guard": dr["guard"]}}
    path.write_text(json.dumps(rec, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return rec


def finalize(root: Path = OUT, now: float | None = None) -> dict[str, Any]:
    """DRAFT to FINAL after the hashes are re-checked. Refuses above the price gate or when the guard would refuse."""
    bad = check(root, allow_draft=True)
    if bad:
        raise SystemExit("cannot finalize; changed since the draft: " + ", ".join(bad))
    path = root / NAME
    rec = json.loads(path.read_text())
    if rec["status"] == "FINAL":
        raise SystemExit(f"{path} is FINAL already")
    dr = rec["dryrun"]
    if dr["likely_usd"] > PRICE_GATE_USD:
        raise SystemExit(f"dry-run likely ${dr['likely_usd']:.2f} is above ${PRICE_GATE_USD:.2f}; stop and report")
    if not dr["guard"]["fits"]:
        raise SystemExit(f"a round's upper bound does not fit the ${RUN_CAP_USD:.2f} cap; stop and report")
    rec["status"] = "FINAL"
    rec["finalized_at"] = _stamp(now)
    rec["price_gate_usd"] = PRICE_GATE_USD
    rec["run_cap_usd"] = RUN_CAP_USD
    path.write_text(json.dumps(rec, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return rec


def check(root: Path, run_dir: Path | None = None, *, allow_draft: bool = False) -> list[str]:
    path = root / NAME
    if not path.exists():
        return [f"{NAME} missing"]
    frozen = json.loads(path.read_text())
    cur = current()
    bad: list[str] = []
    if frozen.get("status") != "FINAL" and not allow_draft:
        bad.append(f"status:{frozen.get('status')}")
    bad += [f"rule:{k}" for k in set(frozen["rules"]) | set(cur["rules"]) if frozen["rules"].get(k) != cur["rules"].get(k)]
    bad += [f"v3:{k}" for k in frozen["reused_v3"] if frozen["reused_v3"][k] != cur["reused_v3"].get(k)]
    bad += [f"arm:{a}" for a in cur["arm_configs"] if frozen["arm_configs"].get(a) != cur["arm_configs"][a]]
    bad += [f"root:{k}" for k, v in _file_hashes(root).items() if frozen.get(k) != v]
    if run_dir is not None:
        cfg = json.loads((run_dir / "config.json").read_text())
        arm = cfg.get("arm")
        if _sha(rq.canonical(config_for_hash(cfg))) != frozen["arm_configs"].get(arm):
            bad.append(f"config:{arm}")
        names = [("frame.jsonl", "frame_sha256"), ("cases.jsonl", "cases_sha256"), ("calibration.json", "calibration_sha256")]
        names += [(RUN_FILES[k], k) for k in ARM_FILES.get(arm, ())]
        for name, key in names:
            if not (run_dir / name).exists() or _sha((run_dir / name).read_bytes()) != frozen[key]:
                bad.append(f"run:{name}")
    return sorted(bad)

"""Freeze and check for exploratory protocol v7.

`freeze` writes results/exploratory-v7/protocol_v7.json once, before any model call, with status DRAFT: sha256 of
the R1 prompt, the two text rules, the X1 split and its two prompts, the set rule, the round rule, the
pre-registered analysis, the v3 and v5 prompts that are reused, each study's request settings, and the run files.
It also records the dry-run price. `finalize` turns DRAFT into FINAL and refuses when the dry-run likely price is
above PRICE_GATE_USD or a round's upper bound would not fit the run cap. The engine refuses to run before FINAL
and whenever a hash differs. `check` is run by the engine before every step.
"""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path
from typing import Any

from ..explore4.protocol import REUSED_V3
from ..explore5 import protocol as PR5
from ..headline import requests as rq
from ..headline.protocol import V2_HASHES, _sentinel_case, current_hashes
from ..schema import View
from . import SCHEMA, VERSION
from . import frame as F
from . import prompts as P
from .frame import OUT

NAME = "protocol_v7.json"
DRYRUN_NAME = "dryrun_v7.json"
DOC = "docs/flip-protocol-v7-exploratory.md"
TRACK_CAP_USD = 5.00          # actual spend of the whole track, v6 included
V6_ACTUAL_USD = 0.432525      # results/exploratory-v6/spend.json
RUN_CAP_USD = 4.56            # the track cap minus the v6 actual, rounded down to the cent
PRICE_GATE_USD = 2.00         # finalize refuses above this dry-run likely price
REUSED_V5 = ("x5.x1.prompt", "x5.x1.reader_paragraph", "x5.x1.schema")
RUN_FILES = {"frame_sha256": "frame.jsonl", "cases_sha256": "cases.jsonl", "wrong_frame_sha256": F.WRONG_FRAME,
             "r_notes_sha256": F.R_NOTES, "v5_refs_sha256": F.V5_REFS, "calibration_sha256": F.CALIBRATION_NAME}
ARM_FILES = {"R": ("r_notes_sha256",), "X": ("wrong_frame_sha256",)}


def _sha(text: str | bytes) -> str:
    return hashlib.sha256(text.encode("utf-8") if isinstance(text, str) else text).hexdigest()


def rule_texts() -> dict[str, str]:
    from .analysis import ANALYSIS_SPEC
    from .engine import MAX_REQUESTS_PER_ROUND
    case = _sentinel_case()
    view = View(system="{SYSTEM}", budget=None, text="{RECORDS}", units=["{UNIT}"])
    return {
        "x7.r1.system": P.R1_SYSTEM,
        "x7.r1.prompt": P.build_r1_user("{NOTE}", case),
        "x7.r1.diff": rq.canonical(P.r1_diff_from_v1()),
        "x7.r2.rule": P.hedge_rule_text(),
        "x7.r3.rule": P.fragment_rule_text(),
        "x7.x1.split": P.split_text(),
        "x7.x1p.prompt": P.build_x1p_prompt(view, case),
        "x7.x1r.prompt": P.build_x1r_prompt(view, case),
        "x7.x1.schema": rq.canonical(P.X1_SPLIT_SCHEMA),
        "x7.sets.rule": rq.canonical({"budget": F.RESIDUE_BUDGET, "gated": "a_H == a_star", "eligible": "gated and a_V == a_H",
                                      "a_wrong": "gated and a_V != a_H, parse errors included",
                                      "source": "headline.merge.merge_data(headline-300, headline-u2-v3, U2)"}),
        "x7.rounds.rule": rq.canonical({"max_requests_per_round": MAX_REQUESTS_PER_ROUND, "order": "request key, ascending",
                                        "unchanged_notes": "the v3 decision stands", "emptied_notes": "not sent",
                                        "flip_search": "none"}),
        "x7.analysis": rq.canonical(ANALYSIS_SPEC),
    }


def config_for_hash(cfg: dict[str, Any]) -> dict[str, Any]:
    return {k: v for k, v in cfg.items() if k != "assumed_output_tokens"}


def current() -> dict[str, Any]:
    from .engine import ARMS, arm_config
    v3 = current_hashes()
    v5 = PR5.current()["prompts"]
    return {"rules": {k: _sha(v) for k, v in rule_texts().items()},
            "reused": {**{k: v3[k] for k in REUSED_V3}, **{k: v5[k] for k in REUSED_V5}},
            "arm_configs": {a: _sha(rq.canonical(config_for_hash(arm_config(a)))) for a in ARMS}}


def _file_hashes(root: Path) -> dict[str, str]:
    return {k: _sha((root / name).read_bytes()) for k, name in RUN_FILES.items()}


def _stamp(now: float | None = None) -> str:
    return time.strftime("%Y-%m-%d %H:%M UTC", time.gmtime(now if now is not None else time.time()))


def freeze(root: Path = OUT, now: float | None = None) -> dict[str, Any]:
    path = root / NAME
    if path.exists():
        raise SystemExit(f"{path} exists; v7 is frozen")
    dr_path = root / DRYRUN_NAME
    if not dr_path.exists():
        raise SystemExit(f"no {DRYRUN_NAME}; run dryrun first")
    dr = json.loads(dr_path.read_text())
    cur = current()
    bad = [k for k, v in cur["reused"].items() if k in V2_HASHES and V2_HASHES[k] != v]
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
    rec.update({"status": "FINAL", "finalized_at": _stamp(now), "price_gate_usd": PRICE_GATE_USD, "run_cap_usd": RUN_CAP_USD,
                "track_cap_usd": TRACK_CAP_USD, "v6_actual_usd": V6_ACTUAL_USD})
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
    bad += [f"reused:{k}" for k in frozen["reused"] if frozen["reused"][k] != cur["reused"].get(k)]
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

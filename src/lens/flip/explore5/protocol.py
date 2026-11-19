"""Freeze and check for exploratory protocol v5.

`freeze` writes results/exploratory-v5/protocol_v5.json once, before any model call: sha256 of every v5 prompt
(rendered with sentinel inputs), of the v3 prompts it reuses, of the frame, the cases, the v3 minimal sets, the
C1 control draws and the input-token rule, and of each arm's request settings. It is written with status DRAFT.
`finalize` turns DRAFT into FINAL (Varma's yes on the price); the engine refuses to run before that.
`check` is run by the engine before every step.
"""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path
from typing import Any

from ..explore4.protocol import REUSED_V3
from ..headline import requests as rq
from ..headline.protocol import V2_HASHES, _sentinel_case, current_hashes
from ..omission import judge as omission_judge
from ..schema import View
from . import SCHEMA, VERSION
from . import controls as CT
from . import prompts as P
from .frame import CALIBRATION_NAME, MIN_SETS, OUT

NAME = "protocol_v5.json"
CONTROLS = "c1_controls.jsonl"


def _sha(text: str | bytes) -> str:
    return hashlib.sha256(text.encode("utf-8") if isinstance(text, str) else text).hexdigest()


def prompt_texts() -> dict[str, str]:
    case = _sentinel_case()
    view = View(system="{SYSTEM}", budget=None, text="{RECORDS}", units=["{UNIT}"])
    return {
        "x5.x1.prompt": P.build_x1_prompt(view, case),
        "x5.x1.reader_paragraph": P.READER_PARAGRAPH,
        "x5.x1.schema": rq.canonical(P.X1_SCHEMA),
        "x5.x2.prompt": P.build_x2_prompt(view, case),
        "x5.x2.schema": rq.canonical(P.X2_SCHEMA),
        "x5.x2.render": rq.canonical({"labels": P.X2_LABELS, "order": list(P.X2_FIELDS),
                                      "empty": "none", "join": "newline"}),
        "x5.c1.rule": rq.canonical({"k": CT.K_CONTROLS, "tolerance": CT.TOLERANCE, "max_attempts": CT.MAX_ATTEMPTS,
                                    "seed": CT.SEED_TAG + "|<case>|<system>|<k>", "length": "len('\\n'.join(texts))",
                                    "non_recoverable_target": "median minimal-set length of recoverable flips"}),
        "x5.c2.prompt": P.build_rewrite_prompt("{NOTE}"),
        "x5.c2.schema": rq.canonical(P.REWRITE_SCHEMA),
        "x5.c3.rule": rq.canonical({"repeats": 5, "key": "sha256(canonical body + NUL + 'repeat=' + tag)",
                                    "cache": "bypassed in this arm only"}),
        "x5.c4.prompt": P.build_judge_prompt("{NOTE}", "{UNIT}"),
        "x5.c4.schema": rq.canonical(P.JUDGE_SCHEMA),
        "x5.c4.prompt_source": rq.canonical({"module": "lens.flip.omission.judge",
                                             "version": omission_judge.JUDGE_PROMPT_VERSION}),
    }


def config_for_hash(cfg: dict[str, Any]) -> dict[str, Any]:
    return {k: v for k, v in cfg.items() if k != "assumed_output_tokens"}


def current() -> dict[str, Any]:
    from .engine import ARMS, arm_config
    v3 = current_hashes()
    return {"prompts": {k: _sha(v) for k, v in prompt_texts().items()},
            "reused_v3": {k: v3[k] for k in REUSED_V3},
            "arm_configs": {a: _sha(rq.canonical(config_for_hash(arm_config(a)))) for a in ARMS}}


def _file_hashes(root: Path) -> dict[str, str]:
    return {"frame_sha256": _sha((root / "frame.jsonl").read_bytes()),
            "cases_sha256": _sha((root / "cases.jsonl").read_bytes()),
            "min_sets_sha256": _sha((root / MIN_SETS).read_bytes()),
            "c1_controls_sha256": _sha((root / CONTROLS).read_bytes()),
            "calibration_sha256": _sha((root / CALIBRATION_NAME).read_bytes())}


def freeze(root: Path = OUT, now: float | None = None) -> dict[str, Any]:
    path = root / NAME
    if path.exists():
        raise SystemExit(f"{path} exists; v5 is frozen")
    cur = current()
    bad = [k for k, v in cur["reused_v3"].items() if k in V2_HASHES and V2_HASHES[k] != v]
    if bad:
        raise SystemExit("reused v3 prompts drifted from the frozen hashes: " + ", ".join(bad))
    rec = {"schema": SCHEMA, "version": VERSION, "status": "DRAFT",
           "frozen_at": time.strftime("%Y-%m-%d %H:%M UTC", time.gmtime(now if now is not None else time.time())),
           **cur, **_file_hashes(root)}
    path.write_text(json.dumps(rec, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return rec


def finalize(root: Path = OUT) -> dict[str, Any]:
    """DRAFT to FINAL, after the hashes are re-checked. Run only on Varma's yes to the price."""
    bad = check(root, allow_draft=True)
    if bad:
        raise SystemExit("cannot finalize; changed since the draft: " + ", ".join(bad))
    path = root / NAME
    rec = json.loads(path.read_text())
    rec["status"] = "FINAL"
    rec["finalized_at"] = time.strftime("%Y-%m-%d %H:%M UTC", time.gmtime())
    path.write_text(json.dumps(rec, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return rec


def check(root: Path, run_dir: Path | None = None, *, allow_draft: bool = False) -> list[str]:
    path = root / NAME
    if not path.exists():
        return ["protocol_v5.json missing"]
    frozen = json.loads(path.read_text())
    cur = current()
    bad: list[str] = []
    if frozen.get("status") != "FINAL" and not allow_draft:
        bad.append(f"status:{frozen.get('status')}")
    bad += [f"prompt:{k}" for k in set(frozen["prompts"]) | set(cur["prompts"])
            if frozen["prompts"].get(k) != cur["prompts"].get(k)]
    bad += [f"v3:{k}" for k in frozen["reused_v3"] if frozen["reused_v3"][k] != cur["reused_v3"].get(k)]
    bad += [f"arm:{a}" for a in cur["arm_configs"] if frozen["arm_configs"].get(a) != cur["arm_configs"][a]]
    for k, v in _file_hashes(root).items():
        if frozen.get(k) != v:
            bad.append(f"root:{k}")
    if run_dir is not None:
        cfg = json.loads((run_dir / "config.json").read_text())
        if _sha(rq.canonical(config_for_hash(cfg))) != frozen["arm_configs"].get(cfg["arm"]):
            bad.append(f"config:{cfg['arm']}")
        for name, key in (("frame.jsonl", "frame_sha256"), ("cases.jsonl", "cases_sha256"),
                          ("calibration.json", "calibration_sha256")):
            if _sha((run_dir / name).read_bytes()) != frozen[key]:
                bad.append(f"run:{name}")
        if cfg["arm"] == "C1" and _sha((run_dir / CONTROLS).read_bytes()) != frozen["c1_controls_sha256"]:
            bad.append(f"run:{CONTROLS}")
    return sorted(bad)

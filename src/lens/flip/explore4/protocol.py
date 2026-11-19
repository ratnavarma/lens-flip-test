"""Freeze and check for exploratory protocol v4.

`freeze` writes results/exploratory-v4/protocol_v4.json once, before any model call: sha256 of every
v4 prompt (rendered with sentinel inputs), of the v3 prompts it reuses, of the frame and the cases,
and of each arm's request settings. `check` is run by the engine before every step.
"""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path
from typing import Any

from ..headline import requests as rq
from ..headline.protocol import V2_HASHES, _sentinel_case, current_hashes
from ..schema import View
from . import prompts as P
from .frame import OUT

REUSED_V3 = ("decider.v1", "decider.v1.inserted", "decider.v2", "decider.action_schema", "note.prompt_p_residue",
             "note.schema", "event_templates", "file:render.py")
LOCAL = {"model": "qwen3:30b-a3b", "think": False, "temperature": 0, "seed": 0, "num_ctx": 32768,
         "format": "decider.REPLY_FORMAT", "retry": "one retry with decider.REMINDER on an unparsable reply",
         "gate": {"n": 30, "per_mechanism": 5, "seed": 41, "prompt": "v1", "view": "full history",
                  "pass_at": 24}}


def _sha(text: str | bytes) -> str:
    return hashlib.sha256(text.encode("utf-8") if isinstance(text, str) else text).hexdigest()


def prompt_texts() -> dict[str, str]:
    case = _sentinel_case()
    view = View(system="{SYSTEM}", budget=None, text="{RECORDS}", units=["{UNIT}"])
    out = {
        "x4.extract.prompt": P.build_extract_prompt(view, case),
        "x4.extract.schema": rq.canonical(P.extract_schema(["{LABEL}"])),
        "x4.note_budget.schema": rq.canonical(P.BUDGET_NOTE_SCHEMA),
        "x4.strip.rule": P.strip_rule_text(),
    }
    for b in (150,) + tuple(P.NOTE_BUDGETS):
        out[f"x4.note_budget.prompt.{b}"] = P.build_budget_note_prompt(view, case, b)
    return out


def config_for_hash(cfg: dict[str, Any]) -> dict[str, Any]:
    return {k: v for k, v in cfg.items() if k != "assumed_output_tokens"}


def current() -> dict[str, Any]:
    from .engine import ARMS, arm_config
    h = {k: _sha(v) for k, v in prompt_texts().items()}
    v3 = current_hashes()
    reused = {k: v3[k] for k in REUSED_V3}
    return {"prompts": h, "reused_v3": reused,
            "arm_configs": {a: _sha(rq.canonical(config_for_hash(arm_config(a)))) for a in ARMS},
            "local": _sha(rq.canonical(LOCAL))}


def freeze(root: Path = OUT) -> dict[str, Any]:
    path = root / "protocol_v4.json"
    if path.exists():
        raise SystemExit(f"{path} exists; v4 is frozen")
    cur = current()
    bad = [k for k, v in cur["reused_v3"].items() if k in V2_HASHES and V2_HASHES[k] != v]
    if bad:
        raise SystemExit("reused v3 prompts drifted from the frozen hashes: " + ", ".join(bad))
    if cur["prompts"]["x4.note_budget.prompt.150"] != cur["reused_v3"]["note.prompt_p_residue"]:
        raise SystemExit("the budget note prompt at 150 is not the frozen v3 note prompt")
    rec = {"version": "4-exploratory", "frozen_at": time.strftime("%Y-%m-%d %H:%M UTC", time.gmtime()), **cur,
           "frame_sha256": _sha((root / "frame.jsonl").read_bytes()),
           "cases_sha256": _sha((root / "cases.jsonl").read_bytes())}
    path.write_text(json.dumps(rec, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return rec


def check(root: Path, run_dir: Path | None = None) -> list[str]:
    frozen = json.loads((root / "protocol_v4.json").read_text())
    cur = current()
    bad = [f"prompt:{k}" for k in set(frozen["prompts"]) | set(cur["prompts"])
           if frozen["prompts"].get(k) != cur["prompts"].get(k)]
    bad += [f"v3:{k}" for k in frozen["reused_v3"] if frozen["reused_v3"][k] != cur["reused_v3"].get(k)]
    bad += [f"local" for _ in [0] if frozen["local"] != cur["local"]]
    if run_dir is not None:
        for name in ("frame", "cases"):
            if _sha((run_dir / f"{name}.jsonl").read_bytes()) != frozen[f"{name}_sha256"]:
                bad.append(f"run:{name}.jsonl")
        cfg = json.loads((run_dir / "config.json").read_text())
        if _sha(rq.canonical(config_for_hash(cfg))) != frozen["arm_configs"].get(cfg["arm"]):
            bad.append(f"config:{cfg['arm']}")
    return sorted(bad)

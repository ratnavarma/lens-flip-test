"""Study B, local reader: qwen3:30b-a3b through Ollama (free), behind a 30-case validity gate.

Gate: 5 cases per mechanism from the 300 combined v3 cases (seed 41), full history, v1 prompt.
Pass at 24 of 30 (80 percent). Only on a pass does the reader decide from the 490 frozen notes
(v1 prompt) and from V itself (v2 prompt, the reader's own view baseline).
"""

from __future__ import annotations

import json
import random
import time
from pathlib import Path
from typing import Any

from ..cache import CallCache
from ..decider import ollama_decider
from ..residue import note_view
from ..schema import MECHANISMS
from ..systems.base import event_view
from .frame import OUT, load_cases, load_frame
from .protocol import LOCAL, check


def _decider(version: str, cache: CallCache):
    return ollama_decider(LOCAL["model"], version, int(LOCAL["num_ctx"]), cache=cache)


def gate_sample(cases, seed: int = 41, per: int = 5):
    rng = random.Random(f"x4-gate|{seed}")
    out = []
    for m in MECHANISMS:
        pool = [c for c in cases if c.mechanism == m]
        out += rng.sample(pool, per)
    return out


def run_gate(root: Path = OUT, log=print) -> dict[str, Any]:
    if check(root):
        raise SystemExit("v4 protocol check failed")
    d = root / "B-local"
    d.mkdir(parents=True, exist_ok=True)
    cache = CallCache(d / "calls.sqlite")
    dec = _decider("v1", cache)
    rows = []
    for i, c in enumerate(gate_sample(load_cases(root))):
        t0 = time.time()
        a = dec(event_view(c, (e.id for e in c.events), "full"), [], c)
        rows.append({"case_id": c.case_id, "mechanism": c.mechanism, "length": c.knobs.get("length"),
                     "a_star": c.a_star, "a_local": a, "ok": a == c.a_star, "seconds": round(time.time() - t0, 2)})
        log(f"gate [{i + 1}/30] {c.case_id} {c.mechanism} a*={c.a_star} local={a} {rows[-1]['seconds']}s")
    ok = sum(r["ok"] for r in rows)
    res = {"model": LOCAL["model"], "n": len(rows), "correct": ok, "pass_at": LOCAL["gate"]["pass_at"],
           "passes": ok >= LOCAL["gate"]["pass_at"], "rows": rows,
           "per_mechanism": {m: sum(r["ok"] for r in rows if r["mechanism"] == m) for m in MECHANISMS},
           "mean_seconds": round(sum(r["seconds"] for r in rows) / len(rows), 2)}
    (d / "gate.json").write_text(json.dumps(res, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return res


def run_reader(root: Path = OUT, log=print) -> dict[str, Any]:
    d = root / "B-local"
    gate = json.loads((d / "gate.json").read_text())
    if not gate["passes"]:
        raise SystemExit(f"local gate failed ({gate['correct']}/{gate['n']}); the local reader does not run")
    if check(root):
        raise SystemExit("v4 protocol check failed")
    cache = CallCache(d / "calls.sqlite")
    d1, d2 = _decider("v1", cache), _decider("v2", cache)
    out = []
    hs = load_frame(root)
    t0 = time.time()
    for i, h in enumerate(hs):
        a_b = d1(note_view(h.row["v3"]["note"]), [], h.case)
        a_v = d2(h.view, [], h.case)
        out.append({"case_id": h.case.case_id, "system": h.system, "a_star": h.row["a_star"], "a_A": h.row["a_A"],
                    "a_B": a_b, "a_view_v2": a_v, "residue_flip": a_b != h.row["a_A"],
                    "view_miss": a_v != h.row["a_A"]})
        if (i + 1) % 25 == 0:
            log(f"local reader {i + 1}/{len(hs)} {time.time() - t0:.0f}s")
    with (d / "results.jsonl").open("w", encoding="utf-8") as fh:
        for r in out:
            fh.write(json.dumps(r, sort_keys=True) + "\n")
    return {"n": len(out), "flips": sum(r["residue_flip"] for r in out), "seconds": round(time.time() - t0)}

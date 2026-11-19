"""The 490 eligible handoffs of results/headline-300-v3, read once from the frozen runs.

The merge is the v3 merge (headline-300 without U2, plus headline-u2-v3), done by the same
`merge_data`, so the case order and the bootstrap match summary_v3.json. The frozen dirs are only
read. frame.jsonl and cases.jsonl under results/exploratory-v4 carry everything the arms need:
the view (event ids or summary text), a*, a_A, and the v3 note, p_residue, a_B, a_floor and a_trunc.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..headline.merge import merge_data
from ..headline.tasks import rebuild_view
from ..schema import Case, View, read_jsonl, write_jsonl

BASE = Path("results/headline-300")
U2RUN = Path("results/headline-u2-v3")
OUT = Path("results/exploratory-v4")
RESIDUE_BUDGET = 1000


@dataclass
class Handoff:
    case: Case
    system: str
    view: View
    row: dict[str, Any]

    @property
    def hid(self) -> str:
        return f"{self.case.case_id}__{self.system}"


def _sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def build_frame(out: Path = OUT, base: Path = BASE, u2: Path = U2RUN) -> dict[str, Any]:
    out.mkdir(parents=True, exist_ok=True)
    if (out / "frame.jsonl").exists():
        raise SystemExit(f"{out / 'frame.jsonl'} exists; the frame is built once")
    data, _prov = merge_data(base, u2, "U2")
    cases = data["cases"]
    rows = data["residue_rows"]
    frame = []
    for r in rows:
        src = u2 if any(c.case_id == r["case_id"] for c in data["cases"][-30:]) and r["case_id"].startswith("u3") else base
        st = json.loads((src / "state" / "cases" / f"{r['case_id']}.json").read_text())
        cell = st["cells"][f"{r['system']}@{RESIDUE_BUDGET}"]
        frame.append({
            "case_id": r["case_id"], "system": r["system"], "mechanism": r["mechanism"], "a_star": r["a_star"],
            "a_H": r["a_H"], "a_A": r["a_A"], "view": cell["view"], "source": str(src),
            "v3": {k: r[k] for k in ("a_B", "a_floor", "a_trunc", "note", "note_tokens", "note_status", "p_residue",
                                     "omission", "residue_flip", "search_status", "view_units", "n_candidates",
                                     "dstar_in_view")},
        })
    write_jsonl(out / "cases.jsonl", cases)
    with (out / "frame.jsonl").open("w", encoding="utf-8", newline="\n") as fh:
        for f in frame:
            fh.write(json.dumps(f, sort_keys=True, separators=(",", ":")) + "\n")
    src_hashes = {str(d): {n: _sha(d / n) for n in ("cases.jsonl", "config.json", "residue.jsonl")} for d in (base, u2)}
    meta = {"n_cases": len(cases), "n_handoffs": len(frame), "n_v3_flips": sum(f["v3"]["residue_flip"] for f in frame),
            "frame_sha256": _sha(out / "frame.jsonl"), "cases_sha256": _sha(out / "cases.jsonl"),
            "sources_sha256": src_hashes}
    (out / "frame_meta.json").write_text(json.dumps(meta, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return meta


def load_cases(out: Path = OUT) -> list[Case]:
    return list(read_jsonl(out / "cases.jsonl"))


def load_frame(out: Path = OUT, cases: list[Case] | None = None) -> list[Handoff]:
    cases = cases or load_cases(out)
    by = {c.case_id: c for c in cases}
    hs = []
    for line in (out / "frame.jsonl").read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        f = json.loads(line)
        case = by[f["case_id"]]
        hs.append(Handoff(case=case, system=f["system"], view=rebuild_view(case, f["system"], RESIDUE_BUDGET, f["view"]),
                          row=f))
    return hs

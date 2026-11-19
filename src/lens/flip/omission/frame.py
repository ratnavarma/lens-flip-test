"""The frame: the 490 eligible residue notes of the merged v3 report, each with the units of its view V.

The merged v3 report takes the cases of `merge.json` (the base run's kept cases plus the replacement run's).
The eligible rows are the ones `headline.merge.merge_data` returns as residue rows. A row's view is rebuilt from
the run state exactly as the report does. Units are the lines (event views) or sentences (summary) of V.
Everything is read, nothing is written.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ..headline import merge as merge_mod
from ..headline import provenance
from ..headline.metrics import power_thresholds
from ..headline.tasks import rebuild_view
from ..residue import omission_fraction
from ..schema import read_jsonl

ROOT = provenance.ROOT
SYSTEMS = ("recent", "rag-bm25", "rag-embed", "summary")
MERGE_DIR = "results/headline-300-v3"


@dataclass(frozen=True)
class UnitRec:
    index: int          # position in V
    unit_id: str        # event id for an event view, s000.. for a summary
    text: str
    dstar: bool


@dataclass
class NoteRec:
    case_id: str
    system: str
    note: str
    p_residue: float | None
    residue_flip: bool
    omission_frozen: float | None
    units: list[UnitRec]


@dataclass
class Frame:
    notes: list[NoteRec]
    case_ids: list[str]                      # all cases of the merged report, in report order (bootstrap units)
    bootstrap: dict[str, int]                # reps, seed
    power: dict[str, int]                    # H2c_min_positives, H2c_min_negatives
    inputs: dict[str, Any] = field(default_factory=dict)       # name -> sha256 (provenance)
    v3_reference: dict[str, Any] | None = None                 # h2c_auroc of summary_v3.json, read only
    watch: dict[str, str] = field(default_factory=dict)        # absolute path -> sha256 that must not change

    @property
    def n_pairs(self) -> int:
        return sum(len(n.units) for n in self.notes)


def load_frame(root: Path | None = None, merge_dir: str = MERGE_DIR) -> Frame:
    root = Path(root) if root else ROOT
    md = root / merge_dir
    mj = json.loads((md / "merge.json").read_text())
    base, with_ = root / mj["base"]["dir"], root / mj["with"]["dir"]
    mechanism = mj["replaced_mechanism"]
    data, _ = merge_mod.merge_data(base, with_, mechanism)
    cases = {c.case_id: c for c in data["cases"]}
    with_ids = {c.case_id for c in read_jsonl(with_ / "cases.jsonl")}
    budget = int(data["config"]["residue_budget"])
    notes: list[NoteRec] = []
    for r in sorted(data["residue_rows"], key=lambda r: (SYSTEMS.index(r["system"]), r["case_id"])):
        case = cases[r["case_id"]]
        d = with_ if r["case_id"] in with_ids else base
        cs = json.loads((d / "state" / "cases" / f"{r['case_id']}.json").read_text())
        cell = cs["cells"][f"{r['system']}@{budget}"]
        view = rebuild_view(case, r["system"], budget, cell["view"])
        if len(view.units) != r["view_units"]:
            raise RuntimeError(f"{r['case_id']} {r['system']}: rebuilt view has {len(view.units)} units, "
                               f"the report row says {r['view_units']}")
        om = omission_fraction(case, view, r["note"])
        if (om is None) != (r.get("omission") is None) or (om is not None and abs(om - r["omission"]) > 1e-9):
            raise RuntimeError(f"{r['case_id']} {r['system']}: frozen omission does not reproduce")
        ds = set(case.d_star)
        ids = view.event_ids if view.event_ids and len(view.event_ids) == len(view.units) else None
        units = [UnitRec(k, ids[k] if ids else f"s{k:03d}", t, bool(ids) and ids[k] in ds)
                 for k, t in enumerate(view.units)]
        notes.append(NoteRec(r["case_id"], r["system"], r["note"], r["p_residue"], bool(r["residue_flip"]),
                             r.get("omission"), units))
    cfg = data["config"]
    pw = power_thresholds(cfg)
    inputs: dict[str, Any] = {
        "results/headline-300-v3/merge.json": provenance.sha256_file(md / "merge.json"),
        "results/headline-300-v3/cases.jsonl": provenance.sha256_file(md / "cases.jsonl"),
        provenance.rel(base): provenance.run_inputs(base),
        provenance.rel(with_): provenance.run_inputs(with_),
    }
    ref = None
    watch: dict[str, str] = {}
    s3 = md / "summary_v3.json"
    if s3.exists():
        inputs["results/headline-300-v3/summary_v3.json"] = provenance.sha256_file(s3)
        watch[str(s3.resolve())] = inputs["results/headline-300-v3/summary_v3.json"]
        s = json.loads(s3.read_text())
        pooled = s["residue"]["pooled"]
        ref = {"pooled": pooled["h2c_auroc"]["value"],
               "by_system": {k: v["h2c_auroc"]["value"] for k, v in s["residue"]["by_system"].items()},
               "n_eligible": pooled["n_eligible"], "n_residue_flips": pooled["n_residue_flips"]}
    return Frame(notes=notes, case_ids=[c.case_id for c in data["cases"]],
                 bootstrap={"reps": int(cfg["bootstrap"]["reps"]), "seed": int(cfg["bootstrap"]["seed"])},
                 power={"H2c_min_positives": pw["H2c_min_positives"], "H2c_min_negatives": pw["H2c_min_negatives"]},
                 inputs=inputs, v3_reference=ref, watch=watch)

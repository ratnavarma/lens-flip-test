"""`lens flip omission run`: embedding coverage, calibration on Varma's labels, omission fraction, H2c AUROC.

Exploratory, outside protocol v3. It reads the v3 data and never writes into a v3 run directory. It does not
touch summary_v3.json: the file's hash is recorded and checked again at the end.

Modes
  embedding         (default) one tau, chosen to maximise Cohen's kappa on the labels. A unit is conveyed when its
                    max cosine to a note sentence is >= tau. No judge is called.
  embedding+judge   two thresholds (90% precision each side). Pairs between them are borderline and go to the
                    local judge. The judge must first pass the fixed rule on the labelled borderline pairs
                    (kappa >= 0.6, accuracy >= 80%). If it fails, the run stops before using it on any other pair
                    and writes judge_calibration.json with "passed": false and a priced Haiku Batch plan.
                    Nothing is submitted.

H2c: AUROC of p_residue for residue flips against the new omission measure (units of V not conveyed over units
of V), pooled and per system, on the same eligible notes and the same bootstrap as the v3 summary.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Callable

from ..headline import provenance
from ..headline.metrics import Boot, auroc
from . import SCHEMA, SUMMARY_JUDGE_NAME, SUMMARY_NAME, VERSION
from . import calibrate as cal
from .coverage import Embedder, coverage
from .frame import SYSTEMS, Frame
from .judge import JUDGE_MODEL, Judge, build_prompt, haiku_plan

MODES = ("embedding", "embedding+judge")
EXPLORATORY_NOTE = ("Exploratory. Outside frozen protocol v3. summary_v3.json is not changed and the frozen H2c "
                    "verdict stands as computed.")


class RunRefused(Exception):
    pass


def _jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as fh:
        for r in rows:
            fh.write(json.dumps(r, sort_keys=True, separators=(",", ":"), ensure_ascii=True) + "\n")


def _write_json(path: Path, obj: Any) -> None:
    path.write_text(json.dumps(obj, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _round(x: float | None, k: int = 6) -> float | None:
    return None if x is None else round(x, k)


# ---------------------------------------------------------------------------
# H2c
# ---------------------------------------------------------------------------

def h2c_block(boot: Boot, idx: dict[str, int], rows: list[dict], power: dict[str, int]) -> dict[str, Any]:
    """rows: {case_id, p_residue, om_nv, om_frozen, flip}. Same construction as the v3 residue block."""
    parsed = [r for r in rows if r["p_residue"] is not None]
    items = [(idx[r["case_id"]], (r["p_residue"], r["om_nv"] if r["om_nv"] is not None else 0.0,
                                  r["om_frozen"] if r["om_frozen"] is not None else 0.0, r["flip"]))
             for r in parsed]

    def a_p(ws):
        return auroc((((p, y), w) for (p, _, _, y), w in ws))

    def a_nv(ws):
        return auroc((((o, y), w) for (_, o, _, y), w in ws))

    def a_fr(ws):
        return auroc((((o, y), w) for (_, _, o, y), w in ws))

    def d_nv(ws):
        a, b = a_p(ws), a_nv(ws)
        return None if a is None or b is None else a - b

    def d_half(ws):
        a = a_p(ws)
        return None if a is None else a - 0.5

    n_pos = sum(r["flip"] for r in parsed)
    n_neg = len(parsed) - n_pos
    powered = n_pos >= power["H2c_min_positives"] and n_neg >= power["H2c_min_negatives"]

    def pack(res: dict[str, Any]) -> dict[str, Any]:
        return {"value": _round(res["value"]), "ci95": res["ci95"]}

    out: dict[str, Any] = {
        "n_eligible": len(rows), "n_scored": len(parsed), "p_residue_parse_errors": len(rows) - len(parsed),
        "n_residue_flips": n_pos, "n_non_flips": n_neg, "powered_by_v3_rule": powered,
        "auroc_p_residue": pack(boot.stat_items(items, a_p)),
        "auroc_omission_nv": pack(boot.stat_items(items, a_nv)),
        "auroc_omission_frozen": pack(boot.stat_items(items, a_fr)),
        "diff_p_residue_vs_omission_nv": pack(boot.stat_items(items, d_nv)),
        "diff_p_residue_vs_half": pack(boot.stat_items(items, d_half)),
    }
    ci = out["diff_p_residue_vs_omission_nv"]["ci95"]
    out["reading"] = ("descriptive: below the v3 minimum of residue flips or non-flips" if not powered else
                      "CI not available" if ci is None else
                      "p_residue beats omission_nv: the 95% CI of the difference is above zero" if ci[0] > 0 else
                      "p_residue does not clearly beat omission_nv: the 95% CI of the difference includes zero"
                      if ci[1] >= 0 else "omission_nv beats p_residue: the 95% CI of the difference is below zero")
    return out


def _distribution(vals: list[float]) -> dict[str, Any]:
    if not vals:
        return {"n": 0}
    s = sorted(vals)
    return {"n": len(s), "mean": _round(sum(s) / len(s)), "min": _round(s[0]), "median": _round(s[len(s) // 2]),
            "max": _round(s[-1]), "share_at_one": _round(sum(v >= 1 - 1e-12 for v in s) / len(s))}


# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------

def run(frame: Frame, labels: list[dict[str, Any]], out_dir: Path, *, embedder: Embedder, mode: str = "embedding",
        judge: Judge | None = None, embed_model: str = "nomic-embed-text", judge_model: str = JUDGE_MODEL,
        labels_path: Path | None = None, log: Callable[[str], None] = lambda m: None) -> dict[str, Any]:
    if mode not in MODES:
        raise RunRefused(f"unknown mode {mode!r}")
    out_dir = Path(out_dir)
    if (out_dir / "summary_v3.json").exists():
        raise RunRefused(f"{out_dir} holds a summary_v3.json: refusing to write into a v3 run directory")
    if mode == "embedding+judge" and judge is None:
        raise RunRefused("mode embedding+judge needs a judge")
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. embedding coverage on every pair
    log(f"embedding coverage for {len(frame.notes)} notes and {frame.n_pairs} pairs")
    cos = coverage(frame, embedder)
    at: dict[tuple[str, str, int], float] = {}
    cov_rows = []
    for note, row in zip(frame.notes, cos):
        for u, c in zip(note.units, row):
            at[(note.case_id, note.system, u.index)] = c
            cov_rows.append({"case_id": note.case_id, "system": note.system, "unit_index": u.index,
                             "unit_id": u.unit_id, "cosine": _round(c)})
    _jsonl(out_dir / "coverage.jsonl", cov_rows)

    # 2. labels
    items: list[cal.Item] = []
    labelled: list[dict[str, Any]] = []
    n_q = 0
    for lab in labels:
        k = (lab["case_id"], lab["system"], lab["unit_index"])
        if k not in at:
            raise RunRefused(f"label {lab.get('unit_label')} points at a pair that is not in the frame: {k}")
        if lab["label"] == "?":
            n_q += 1
        elif lab["label"] in ("y", "n"):
            items.append((at[k], lab["label"] == "y"))
            labelled.append({**lab, "cosine": at[k]})
    label_counts = {"rows": len(labels), "y": sum(t for _, t in items), "n": sum(not t for _, t in items),
                    "?": n_q, "used": len(items)}
    if not items or label_counts["y"] == 0 or label_counts["n"] == 0:
        raise RunRefused(f"calibration needs labels y and n; got {label_counts}")

    # 3. calibration
    tau_k = cal.choose_tau_kappa(items)
    band = cal.choose_band(items)
    tau = tau_k["tau"]
    share_band = None
    if frame.n_pairs:
        nb = sum(cal.in_band(c, band["tau_low"], band["tau_high"]) for row in cos for c in row)
        share_band = nb / frame.n_pairs
    calibration = {"labels": label_counts, "embedding_only": tau_k,
                   "precision_band_context": {**band, "borderline_share_of_all_pairs": _round(share_band)},
                   "note": "tau is chosen on the same labels it is scored on (in-sample)."}

    # 4. the judge, in judge mode
    judge_block: dict[str, Any] | None = None
    judged: dict[tuple[str, str, int], bool | None] = {}
    stopped = False
    if mode == "embedding+judge":
        lo, hi = band["tau_low"], band["tau_high"]
        bl = [r for r in labelled if cal.in_band(r["cosine"], lo, hi)]
        notes_by = {(n.case_id, n.system): n for n in frame.notes}
        log(f"local judge ({judge_model}) on {len(bl)} labelled borderline pairs")
        preds = []
        for r in bl:
            n = notes_by[(r["case_id"], r["system"])]
            preds.append(judge(n.note, n.units[r["unit_index"]].text))
        jc = cal.judge_passes(preds, [r["label"] == "y" for r in bl])
        judge_block = {"model": judge_model, "calibration": jc}
        all_border = [(n, u) for n, row in zip(frame.notes, cos) for u, c in zip(n.units, row)
                      if cal.in_band(c, lo, hi)]
        plan = haiku_plan([(n.note, u.text) for n, u in all_border])
        if not jc["passed"]:
            stopped = True
            jc_out = {**jc, "model": judge_model, "thresholds": {"tau_low": lo, "tau_high": hi},
                      "n_borderline_pairs_all": len(all_border), "haiku_plan": plan,
                      "action": "stopped before the judge was used on all borderline pairs"}
            _write_json(out_dir / "judge_calibration.json", jc_out)
            judge_block["stopped"] = True
            judge_block["haiku_plan"] = plan
            log(f"judge failed calibration: {'; '.join(jc['reasons'])}")
        else:
            log(f"judge passed calibration; judging {len(all_border)} borderline pairs")
            for i, (n, u) in enumerate(all_border):
                judged[(n.case_id, n.system, u.index)] = judge(n.note, u.text)
                if (i + 1) % 500 == 0:
                    log(f"  judged {i + 1}/{len(all_border)}")
            _write_json(out_dir / "judge_calibration.json",
                        {**jc, "model": judge_model, "thresholds": {"tau_low": lo, "tau_high": hi},
                         "n_borderline_pairs_all": len(all_border), "haiku_plan": None,
                         "action": "judge used on all borderline pairs"})
            _jsonl(out_dir / "judge_decisions.jsonl",
                   [{"case_id": k[0], "system": k[1], "unit_index": k[2], "conveyed": v}
                    for k, v in sorted(judged.items())])
            judge_block["borderline_pairs_judged"] = len(judged)
            judge_block["parse_failures"] = sum(v is None for v in judged.values())

    # 5. omission fraction per note and H2c
    result_h2c = None
    omission_by_scope = None
    if not stopped:
        rows = []
        note_rows = []
        for note, row in zip(frame.notes, cos):
            conveyed = unknown = 0
            for u, c in zip(note.units, row):
                if mode == "embedding":
                    conveyed += c >= tau
                else:
                    if band["tau_high"] is not None and c >= band["tau_high"]:
                        conveyed += 1
                    elif band["tau_low"] is not None and c <= band["tau_low"]:
                        pass
                    else:
                        v = judged.get((note.case_id, note.system, u.index))
                        conveyed += bool(v)
                        unknown += v is None
            decided = len(note.units) - unknown
            om = (decided - conveyed) / decided if decided else None
            note_rows.append({"case_id": note.case_id, "system": note.system, "n_units": len(note.units),
                              "n_conveyed": conveyed, "n_unknown": unknown, "omission_nv": _round(om),
                              "omission_frozen": note.omission_frozen, "p_residue": note.p_residue,
                              "residue_flip": note.residue_flip})
            rows.append({"case_id": note.case_id, "system": note.system, "p_residue": note.p_residue,
                         "om_nv": om, "om_frozen": note.omission_frozen, "flip": note.residue_flip})
        _jsonl(out_dir / "notes_omission.jsonl", note_rows)
        boot = Boot(len(frame.case_ids), frame.bootstrap["reps"], frame.bootstrap["seed"])
        idx = {c: i for i, c in enumerate(frame.case_ids)}
        result_h2c = {"pooled": h2c_block(boot, idx, rows, frame.power),
                      "by_system": {s: h2c_block(boot, idx, [r for r in rows if r["system"] == s], frame.power)
                                    for s in SYSTEMS if any(r["system"] == s for r in rows)}}
        omission_by_scope = {"pooled": _distribution([r["om_nv"] for r in rows if r["om_nv"] is not None]),
                             "by_system": {s: _distribution([r["om_nv"] for r in rows
                                                             if r["system"] == s and r["om_nv"] is not None])
                                           for s in SYSTEMS if any(r["system"] == s for r in rows)}}

    # 6. checks, provenance, summary
    checks: dict[str, Any] = {}
    if frame.v3_reference and result_h2c:
        ref = frame.v3_reference
        checks["v3_p_residue_auroc_reproduced"] = {
            "pooled": ref["pooled"] is not None and abs(ref["pooled"] - result_h2c["pooled"]["auroc_p_residue"]["value"]) < 1e-5,
            "by_system": {s: abs(v - result_h2c["by_system"][s]["auroc_p_residue"]["value"]) < 1e-5
                          for s, v in ref["by_system"].items() if s in result_h2c["by_system"]},
            "n_eligible_matches": ref["n_eligible"] == len(frame.notes)}
    moved = sorted(p for p, h in frame.watch.items() if not Path(p).exists() or _sha(Path(p)) != h)
    if moved:
        raise RuntimeError(f"a watched v3 file changed during the run: {moved}")
    inputs = dict(frame.inputs)
    if labels_path is not None and Path(labels_path).exists():
        inputs[provenance.rel(labels_path)] = _sha(Path(labels_path))
    keyp = out_dir / "calibration_key.jsonl"
    if keyp.exists():
        inputs[provenance.rel(keyp)] = _sha(keyp)
    summary: dict[str, Any] = {
        "schema": SCHEMA, "exploratory": True, "version": VERSION, "mode": mode,
        "status": "stopped_judge_failed_calibration" if stopped else "complete",
        "note": EXPLORATORY_NOTE,
        "provenance": {
            "inputs_sha256": inputs,
            "models": {"embedder": embed_model, "judge": judge_model if mode == "embedding+judge" else None},
            "thresholds": {"tau_kappa": tau, "tau_low": band["tau_low"], "tau_high": band["tau_high"],
                           "precision_target": cal.PRECISION_TARGET, "min_support": cal.MIN_SUPPORT,
                           "judge_rule": {"kappa_min": cal.JUDGE_MIN_KAPPA, "accuracy_min": cal.JUDGE_MIN_ACCURACY}},
            "seeds": {"sample": 21, "bootstrap": frame.bootstrap["seed"]},
            "bootstrap": {"reps": frame.bootstrap["reps"], "level": 0.95,
                          "unit": "case (the same Boot as the v3 summary)"},
            "summary_v3_untouched": True, "watched_files": sorted(frame.watch),
            "chars_per_token": 2.2,
        },
        "frame": {"n_notes": len(frame.notes), "n_pairs": frame.n_pairs,
                  "by_system": {s: {"notes": sum(n.system == s for n in frame.notes),
                                    "pairs": sum(len(n.units) for n in frame.notes if n.system == s)}
                                for s in SYSTEMS}},
        "calibration": calibration,
        "judge": judge_block,
        "omission_nv": omission_by_scope,
        "h2c": result_h2c,
        "checks": checks,
    }
    name = SUMMARY_JUDGE_NAME if mode == "embedding+judge" else SUMMARY_NAME
    _write_json(out_dir / name, summary)
    summary["_path"] = str(out_dir / name)
    return summary

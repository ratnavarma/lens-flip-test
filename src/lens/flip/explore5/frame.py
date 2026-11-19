"""The v5 frame: the 490 frozen handoffs of exploratory v4, plus the v3 minimal restoring sets.

frame.jsonl and cases.jsonl are read from results/exploratory-v4 (built once by explore4) and checked
against the sha256 frozen in docs/flip-protocol-v4-exploratory.md before anything is copied. Nothing in a
frozen results dir is written. `stage_frame` copies the two files into the v5 results dir and writes
v3_min_sets.jsonl: for each of the 142 v3 residue flips, the units the v3 search restored with (or none).

Paths. RESULTS is `$LENS_RESULTS` when set, else `results` (relative to the working directory, as in v3
and v4). Run from the repository with LENS_RESULTS pointing at the results dir that holds the frozen runs.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any

from ..explore4.frame import Handoff, load_cases, load_frame  # noqa: F401  (re-exported: the same frame code)

RESULTS = Path(os.environ.get("LENS_RESULTS", "results"))
V4 = RESULTS / "exploratory-v4"
OUT = RESULTS / "exploratory-v5"

# docs/flip-protocol-v4-exploratory.md, section 1.
FRAME_SHA256 = "bed6b1d00288354439f4a0459dc7d512cfc83245b369a141e9e4f1a88f6eb69f"
CASES_SHA256 = "432e55ca579b74766e9097428fe513bd5a8c96496892dec1b023365ba1a4ed3c"
N_HANDOFFS, N_CASES, N_FLIPS, N_RECOVERABLE, N_PAIRS = 490, 300, 142, 41, 11_240
MIN_SETS = "v3_min_sets.jsonl"


def sha256_file(p: Path) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def verify_v4_frame(v4: Path = V4) -> None:
    for name, want in (("frame.jsonl", FRAME_SHA256), ("cases.jsonl", CASES_SHA256)):
        got = sha256_file(v4 / name)
        if got != want:
            raise SystemExit(f"{v4 / name}: sha256 {got} is not the frozen {want}")


def _source_dir(source: str, results: Path) -> Path:
    return results / Path(source).name


def min_set_rows(frame_rows: list[dict[str, Any]], results: Path) -> list[dict[str, Any]]:
    """One row per v3 residue flip: the restoring set of the v3 search (ids and texts), or none.

    Read from each handoff's source run (residue_records.jsonl, stage "residue"). Statuses must agree with the
    frozen frame row, or the build stops."""
    recs: dict[tuple[str, str, str], dict[str, Any]] = {}
    for src in sorted({f["source"] for f in frame_rows}):
        d = _source_dir(src, results)
        for line in (d / "residue_records.jsonl").read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            r = json.loads(line)
            if r.get("stage") == "residue":
                recs[(src, r["case_id"], r["system"])] = r
    out = []
    for f in frame_rows:
        v3 = f["v3"]
        if not v3["residue_flip"]:
            continue
        r = recs.get((f["source"], f["case_id"], f["system"]))
        if r is None:
            raise SystemExit(f"no v3 residue record for {f['case_id']} {f['system']}")
        if r["status"] != v3["search_status"]:
            raise SystemExit(f"{f['case_id']} {f['system']}: record status {r['status']} != frame {v3['search_status']}")
        restored = r["status"] == "restored"
        texts = list(r["units_inserted"]) if restored else []
        ids = list(r["event_ids"]) if restored else []
        if len(texts) != len(ids):
            raise SystemExit(f"{f['case_id']} {f['system']}: {len(ids)} ids but {len(texts)} texts")
        out.append({"case_id": f["case_id"], "system": f["system"], "status": r["status"], "hit_cap": bool(r["hit_cap"]),
                    "verified": r.get("verified"), "calls_used": int(r["calls_used"]), "rounds": int(r["rounds"]),
                    "unit_ids": ids, "unit_texts": texts,
                    "chars": len("\n".join(texts)) if restored else None})
    return out


def stage_frame(root: Path = OUT, v4: Path = V4, results: Path = RESULTS) -> dict[str, Any]:
    """Verify the v4 frame, copy it to `root`, write the v3 minimal sets. Safe to repeat (identical bytes only)."""
    verify_v4_frame(v4)
    root.mkdir(parents=True, exist_ok=True)
    for name in ("frame.jsonl", "cases.jsonl"):
        dst = root / name
        if dst.exists() and sha256_file(dst) != sha256_file(v4 / name):
            raise SystemExit(f"{dst} exists and differs from the frozen v4 file")
        if not dst.exists():
            dst.write_bytes((v4 / name).read_bytes())
    rows = [json.loads(x) for x in (root / "frame.jsonl").read_text(encoding="utf-8").splitlines() if x.strip()]
    ms = min_set_rows(rows, results)
    text = "".join(json.dumps(r, sort_keys=True, separators=(",", ":")) + "\n" for r in ms)
    p = root / MIN_SETS
    if p.exists() and p.read_text(encoding="utf-8") != text:
        raise SystemExit(f"{p} exists and differs from the rebuilt minimal sets")
    p.write_text(text, encoding="utf-8")
    n_rec = sum(r["status"] == "restored" for r in ms)
    write_calibration(root)
    ctl = stage_controls(root)
    meta = {"n_handoffs": len(rows), "n_v3_flips": len(ms), "n_recoverable": n_rec,
            "n_not_recoverable": len(ms) - n_rec, "n_pairs": sum(f["v3"]["view_units"] for f in rows),
            "frame_sha256": sha256_file(root / "frame.jsonl"), "cases_sha256": sha256_file(root / "cases.jsonl"),
            "min_sets_sha256": sha256_file(p),
            "calibration_sha256": sha256_file(root / CALIBRATION_NAME), "c1": ctl}
    (root / "frame_meta_v5.json").write_text(json.dumps(meta, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return meta


def build_controls_rows(root: Path = OUT) -> list[dict[str, Any]]:
    """The C1 control draws for every v3 residue flip, in frame order. Pure function of the frame, the minimal
    sets and the seeds in controls.py."""
    from ..residue import residue_candidates
    from . import controls as CT
    ms = load_min_sets(root)
    med = CT.median_recoverable_chars(list(ms.values()))
    rows: list[dict[str, Any]] = []
    for h in load_frame(root):
        rec = ms.get((h.case.case_id, h.system))
        if rec is None:
            continue
        cands = residue_candidates(h.case, h.view, h.row["v3"]["note"])
        rows += CT.build_controls(h.case.case_id, h.system, cands, rec, med)
    return rows


def stage_controls(root: Path = OUT) -> dict[str, Any]:
    from . import controls as CT
    rows = build_controls_rows(root)
    text = CT.controls_jsonl(rows)
    p = root / "c1_controls.jsonl"
    if p.exists() and p.read_text(encoding="utf-8") != text:
        raise SystemExit(f"{p} exists and differs from the rebuilt draws")
    p.write_text(text, encoding="utf-8")
    return {"n_controls": len(rows), "n_flips": len({(r["case_id"], r["system"]) for r in rows}),
            "n_sent": sum(r["send"] for r in rows), "n_unmatched": sum(not r["matched"] for r in rows), "n_duplicate": sum(r["duplicate"] for r in rows),
            "c1_controls_sha256": sha256_file(p)}


def load_min_sets(root: Path = OUT) -> dict[tuple[str, str], dict[str, Any]]:
    out = {}
    for line in (root / MIN_SETS).read_text(encoding="utf-8").splitlines():
        if line.strip():
            r = json.loads(line)
            out[(r["case_id"], r["system"])] = r
    return out


# ---------------------------------------------------------------------------
# Input-token rule (the brief's: 2.2 characters per token plus the 331-token output-format overhead)
# ---------------------------------------------------------------------------

CHARS_PER_TOKEN_LIKELY = 2.2
CHARS_PER_TOKEN_BOUND = 1.9019805757891086   # the densest ratio seen on decide requests (v3 calibration, "low")
OVERHEAD_TOKENS = 331                        # usage input minus text tokens on every structured request of v3 and v4
CALIBRATION_NAME = "calibration_v5.json"
CALIBRATED_KINDS = ("decide", "note", "note_x2", "rewrite", "judge", "summary")


def calibration_v5() -> dict[str, Any]:
    """The `calibration.json` of every v5 arm: one rule for every request kind (see CHARS_PER_TOKEN_*)."""
    one = {"slope": 1 / CHARS_PER_TOKEN_LIKELY, "intercept": 0.0, "overhead_tokens": OVERHEAD_TOKENS,
           "chars_per_token_low": CHARS_PER_TOKEN_BOUND, "chars_per_token_mean": CHARS_PER_TOKEN_LIKELY}
    return {"rule": "input tokens = ceil(chars / 2.2) + 331 (likely); ceil(chars / 1.9020) + 331 (upper bound)",
            "kinds": {k: dict(one) for k in CALIBRATED_KINDS}}


def write_calibration(root: Path = OUT) -> Path:
    p = root / CALIBRATION_NAME
    text = json.dumps(calibration_v5(), indent=1, sort_keys=True) + "\n"
    if p.exists() and p.read_text(encoding="utf-8") != text:
        raise SystemExit(f"{p} exists and differs from the v5 rule")
    p.write_text(text, encoding="utf-8")
    return p

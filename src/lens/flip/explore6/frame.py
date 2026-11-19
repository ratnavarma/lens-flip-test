"""The v6 frame: the 490 frozen handoffs, the extractive flips of v4 with their minimal sets, the v3 minimal sets.

Everything is read from frozen result dirs and nothing there is written (the call caches are opened read only).
`stage_frame` writes into results/exploratory-v6:

    frame.jsonl, cases.jsonl   copied from exploratory-v4 after the sha256 check of protocol v4
    e1_flips.jsonl             one row per extractive flip of v4 arm A-extractive: the note B saw, the search
                               status, the minimal restoring set (unit ids, view order) and its rendered length
    p1_min_sets.jsonl          the v3 minimal sets, rebuilt as in v5 and checked against the v5 sha256
    e1_controls.jsonl          the E1 control draws (controls.py)
    calibration_v6.json        the v5 input-token rule, unchanged
    frame_meta_v6.json         counts, hashes and the restoring-call check

Mirror of the search. The v4 extractive search and the v3 search both run `headline.tasks.run_search` on
`note_view(note)` with candidates `residue_candidates(case, V, note)`: the units of V whose text the note does
not contain. For an extractive note that is the units A did not select. Each search call is
`decide_request(note_view(note), <inserted units in view order>, case, "v1", decider)`. v6 builds controls and
re-reads with the same call. `restoring_call_check` proves it: the body rebuilt here for each minimal set has the
request key of a stored search call whose answer is a_A.
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
from pathlib import Path
from typing import Any

from ..explore4.analysis import _states
from ..explore5 import frame as F
from ..explore5.frame import Handoff, load_cases, load_frame, sha256_file  # noqa: F401  (the same frame code)
from ..headline import requests as rq
from ..residue import note_view, residue_candidates
from ..schema import Unit
from . import controls as CT

RESULTS = F.RESULTS
V4 = RESULTS / "exploratory-v4"
OUT = RESULTS / "exploratory-v6"
X_ARM = "A-extractive"

E1_FLIPS = "e1_flips.jsonl"
E1_CONTROLS = "e1_controls.jsonl"
P1_MIN_SETS = "p1_min_sets.jsonl"
CALIBRATION_NAME = "calibration_v6.json"
META_NAME = "frame_meta_v6.json"

# Counts of the frozen runs (docs/flip-protocol-v4-exploratory.md results, docs/flip-protocol-v5-exploratory.md).
N_HANDOFFS, N_CASES = F.N_HANDOFFS, F.N_CASES
N_E1_FLIPS, N_E1_RESTORED, N_E1_VERIFIED = 193, 144, 113
N_V3_FLIPS, N_P1 = F.N_FLIPS, F.N_RECOVERABLE
N_V3_FLOOR = 51
# docs/flip-protocol-v5-exploratory.md, section 1: v3_min_sets.jsonl.
V3_MIN_SETS_SHA256 = "434b26b8589ae65262445cfb3c7c79bc16f3179f7ada054b7fdc61ea2f713996"


def _jsonl(rows: list[dict[str, Any]]) -> str:
    return "".join(json.dumps(r, sort_keys=True, separators=(",", ":")) + "\n" for r in rows)


def _read_jsonl(p: Path) -> list[dict[str, Any]]:
    return [json.loads(x) for x in p.read_text(encoding="utf-8").splitlines() if x.strip()]


def _write_once(p: Path, text: str, what: str) -> None:
    if p.exists() and p.read_text(encoding="utf-8") != text:
        raise SystemExit(f"{p} exists and differs from the rebuilt {what}")
    p.write_text(text, encoding="utf-8")


def units_in_order(cands: list[Unit], ids: list[str]) -> list[Unit]:
    """The units named by `ids`, in candidate (view) order: what the search's `units_for` inserts."""
    wanted = set(ids)
    missing = wanted - {u.id for u in cands}
    if missing:
        raise ValueError(f"units {sorted(missing)} are not candidates of V")
    return [u for u in cands if u.id in wanted]


# ---------------------------------------------------------------------------
# E1: the extractive flips of v4
# ---------------------------------------------------------------------------

def e1_flip_rows(hs: list[Handoff], v4: Path = V4) -> list[dict[str, Any]]:
    """One row per extractive flip, in frame order, from the frozen state of v4 arm A-extractive."""
    sx = _states(v4, X_ARM)
    if sx is None:
        raise SystemExit(f"no state for {X_ARM} under {v4}")
    out = []
    for h in hs:
        ns = sx.get((h.case.case_id, h.system))
        if ns is None or "out" not in ns:
            raise SystemExit(f"{h.hid}: no extractive result in {v4 / X_ARM}")
        o = ns["out"]
        if not o["residue_flip"]:
            continue
        rec = ns.get("record")
        if rec is None:
            raise SystemExit(f"{h.hid}: an extractive flip without a search record")
        note = o["note"]
        cands = residue_candidates(h.case, h.view, note)
        if [u.id for u in cands] != o["candidate_ids"]:
            raise SystemExit(f"{h.hid}: rebuilt candidates differ from the candidates the v4 search used")
        restored = rec["status"] == "restored"
        ids = list(rec["event_ids"]) if restored else []
        units = units_in_order(cands, ids)
        if [u.id for u in units] != ids or (restored and [u.text for u in units] != list(rec["units_inserted"])):
            raise SystemExit(f"{h.hid}: the minimal set is not in view order or its text changed")
        out.append({"case_id": h.case.case_id, "system": h.system, "note": note, "a_B": o["a_B"],
                    "status": rec["status"], "hit_cap": bool(rec["hit_cap"]), "verified": rec.get("verified"),
                    "calls_used": int(rec["calls_used"]), "rounds": int(rec["rounds"]),
                    "n_candidates": len(cands), "unit_ids": ids,
                    "chars": len("\n".join(u.text for u in units)) if restored else None})
    return out


def load_e1_flips(root: Path = OUT) -> dict[tuple[str, str], dict[str, Any]]:
    return {(r["case_id"], r["system"]): r for r in _read_jsonl(root / E1_FLIPS)}


def load_p1_sets(root: Path = OUT) -> dict[tuple[str, str], dict[str, Any]]:
    return {(r["case_id"], r["system"]): r for r in _read_jsonl(root / P1_MIN_SETS)}


def load_controls(path: Path) -> dict[tuple[str, str], list[dict[str, Any]]]:
    out: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for r in _read_jsonl(path):
        out.setdefault((r["case_id"], r["system"]), []).append(r)
    for v in out.values():
        v.sort(key=lambda r: r["k"])
    return out


def build_controls_rows(root: Path = OUT) -> list[dict[str, Any]]:
    """The E1 control draws for every extractive flip, in frame order. A pure function of the frame, the flips
    file and the seeds in controls.py."""
    e1 = load_e1_flips(root)
    med = CT.median_recoverable_chars(list(e1.values()))
    rows: list[dict[str, Any]] = []
    for h in load_frame(root):
        rec = e1.get((h.case.case_id, h.system))
        if rec is None:
            continue
        cands = residue_candidates(h.case, h.view, rec["note"])
        rows += CT.build_controls(h.case.case_id, h.system, cands, rec, med)
    return rows


def stage_controls(root: Path = OUT) -> dict[str, Any]:
    rows = build_controls_rows(root)
    p = root / E1_CONTROLS
    _write_once(p, CT.controls_jsonl(rows), "draws")
    flips = {(r["case_id"], r["system"]) for r in rows}
    with_matched = {(r["case_id"], r["system"]) for r in rows if r["send"] and r["matched"]}
    with_sent = {(r["case_id"], r["system"]) for r in rows if r["send"]}
    e1 = load_e1_flips(root)
    return {"n_controls": len(rows), "n_flips": len(flips), "n_sent": sum(r["send"] for r in rows),
            "n_matched": sum(r["matched"] for r in rows), "n_matched_and_sent": sum(r["send"] and r["matched"] for r in rows),
            "n_unmatched": sum(not r["matched"] for r in rows), "n_duplicate": sum(r["duplicate"] for r in rows),
            "n_flips_with_matched_control": len(with_matched), "n_flips_with_sent_control": len(with_sent),
            "median_minimal_chars": CT.median_recoverable_chars(list(e1.values())),
            "e1_controls_sha256": sha256_file(p)}


# ---------------------------------------------------------------------------
# Calibration: the v5 input-token rule
# ---------------------------------------------------------------------------

def calibration_text() -> str:
    return json.dumps(F.calibration_v5(), indent=1, sort_keys=True) + "\n"


def write_calibration(root: Path = OUT) -> Path:
    p = root / CALIBRATION_NAME
    _write_once(p, calibration_text(), "input-token rule")
    return p


# ---------------------------------------------------------------------------
# The check that a re-read is the search's restoring call
# ---------------------------------------------------------------------------

def _stored_action(db: sqlite3.Connection, params: dict[str, Any]) -> str | None:
    row = db.execute("SELECT reply FROM calls WHERE key = ?", (rq.request_key(params),)).fetchone()
    if row is None:
        return None
    return rq.parse_answer("decide", json.loads(row[0]), {})["action"]


def _ro(path: Path) -> sqlite3.Connection:
    return sqlite3.connect(f"file:{path}?mode=ro", uri=True)


def restoring_call_check(root: Path = OUT, v4: Path = V4, results: Path = RESULTS,
                         decider: dict[str, Any] | None = None) -> dict[str, Any]:
    """Rebuild the restoring call of every minimal set and look its request key up in the frozen call caches
    (read only). `found` counts the keys present, `answer_is_a_A` the stored answers equal to A's action."""
    if decider is None:
        from .engine import arm_config
        decider = arm_config("E1")["decider"]
    hs = load_frame(root)
    e1, p1 = load_e1_flips(root), load_p1_sets(root)
    out = {"E1": {"n": 0, "found": 0, "answer_is_a_A": 0}, "P1": {"n": 0, "found": 0, "answer_is_a_A": 0}}
    dbx = _ro(v4 / X_ARM / "state" / "calls.sqlite")
    dbs: dict[Path, sqlite3.Connection] = {}
    try:
        for h in hs:
            k = (h.case.case_id, h.system)
            rec = e1.get(k)
            if rec is not None and rec["status"] == "restored":
                cands = residue_candidates(h.case, h.view, rec["note"])
                a = _stored_action(dbx, rq.decide_request(note_view(rec["note"]), units_in_order(cands, rec["unit_ids"]),
                                                          h.case, "v1", decider))
                out["E1"]["n"] += 1
                out["E1"]["found"] += a is not None
                out["E1"]["answer_is_a_A"] += a == h.row["a_A"]
            rec = p1.get(k)
            if rec is not None and rec["status"] == "restored":
                src = results / Path(h.row["source"]).name / "state" / "calls.sqlite"
                if src not in dbs:
                    dbs[src] = _ro(src)
                note = h.row["v3"]["note"]
                cands = residue_candidates(h.case, h.view, note)
                a = _stored_action(dbs[src], rq.decide_request(note_view(note), units_in_order(cands, rec["unit_ids"]),
                                                               h.case, "v1", decider))
                out["P1"]["n"] += 1
                out["P1"]["found"] += a is not None
                out["P1"]["answer_is_a_A"] += a == h.row["a_A"]
    finally:
        dbx.close()
        for d in dbs.values():
            d.close()
    return out


# ---------------------------------------------------------------------------
# Stage
# ---------------------------------------------------------------------------

def stage_frame(root: Path = OUT, v4: Path = V4, results: Path = RESULTS) -> dict[str, Any]:
    """Verify the frozen inputs and write the v6 run files. Safe to repeat (identical bytes only)."""
    F.verify_v4_frame(v4)
    root.mkdir(parents=True, exist_ok=True)
    for name in ("frame.jsonl", "cases.jsonl"):
        dst = root / name
        if dst.exists() and sha256_file(dst) != sha256_file(v4 / name):
            raise SystemExit(f"{dst} exists and differs from the frozen v4 file")
        if not dst.exists():
            dst.write_bytes((v4 / name).read_bytes())
    frame_rows = _read_jsonl(root / "frame.jsonl")
    hs = load_frame(root)

    e1 = e1_flip_rows(hs, v4)
    n_rest = sum(r["status"] == "restored" for r in e1)
    n_ver = sum(r["status"] == "restored" and bool(r["verified"]) for r in e1)
    if (len(e1), n_rest, n_ver) != (N_E1_FLIPS, N_E1_RESTORED, N_E1_VERIFIED):
        raise SystemExit(f"extractive flips {len(e1)}, restored {n_rest}, verified {n_ver}: not the v4 counts "
                         f"{N_E1_FLIPS}, {N_E1_RESTORED}, {N_E1_VERIFIED}")
    _write_once(root / E1_FLIPS, _jsonl(e1), "extractive flips")

    ms_text = _jsonl(F.min_set_rows(frame_rows, results))
    got = hashlib.sha256(ms_text.encode("utf-8")).hexdigest()
    if got != V3_MIN_SETS_SHA256:
        raise SystemExit(f"rebuilt v3 minimal sets have sha256 {got}, not the v5 file's {V3_MIN_SETS_SHA256}")
    _write_once(root / P1_MIN_SETS, ms_text, "v3 minimal sets")
    p1 = load_p1_sets(root)
    n_p1 = sum(r["status"] == "restored" for r in p1.values())
    if (len(p1), n_p1) != (N_V3_FLIPS, N_P1):
        raise SystemExit(f"v3 flips {len(p1)}, restored {n_p1}: not {N_V3_FLIPS}, {N_P1}")

    write_calibration(root)
    ctl = stage_controls(root)
    chk = restoring_call_check(root, v4, results)
    for arm, want in (("E1", N_E1_RESTORED), ("P1", N_P1)):
        c = chk[arm]
        if not (c["n"] == c["found"] == c["answer_is_a_A"] == want):
            raise SystemExit(f"{arm}: restoring-call check {c} does not cover all {want} minimal sets")
    by_sys: dict[str, dict[str, int]] = {}
    for r in e1:
        b = by_sys.setdefault(r["system"], {"flips": 0, "restored": 0})
        b["flips"] += 1
        b["restored"] += r["status"] == "restored"
    meta = {"n_handoffs": len(frame_rows), "n_e1_flips": len(e1), "n_e1_restored": n_rest, "n_e1_verified": n_ver,
            "e1_by_system": dict(sorted(by_sys.items())),
            "n_v3_flips": len(p1), "n_p1_restored": n_p1,
            "n_v3_floor": sum(f["v3"]["a_floor"] != f["a_A"] for f in frame_rows),
            "frame_sha256": sha256_file(root / "frame.jsonl"), "cases_sha256": sha256_file(root / "cases.jsonl"),
            "e1_flips_sha256": sha256_file(root / E1_FLIPS), "p1_min_sets_sha256": sha256_file(root / P1_MIN_SETS),
            "calibration_sha256": sha256_file(root / CALIBRATION_NAME), "e1_controls": ctl,
            "restoring_call_check": chk}
    (root / META_NAME).write_text(json.dumps(meta, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return meta

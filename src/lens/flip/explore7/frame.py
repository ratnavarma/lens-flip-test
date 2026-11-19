"""The v7 frame: the 490 eligible handoffs, the 670 gated handoffs where A was wrong, and the text-rule notes.

Everything is read from frozen result dirs and nothing there is written (call caches are opened read only).
`stage_frame` writes into results/exploratory-v7:

    frame.jsonl, cases.jsonl   the 490 eligible handoffs, copied from exploratory-v4 after the sha256 check
    wrong_frame.jsonl          the gated handoffs at B = 1,000 where the full-history action is right and A's
                               action from the view is wrong, with the stored view of each
    r_notes.jsonl              per eligible handoff: whether the cap cut the v3 note, whether it ends mid-sentence,
                               the hedge-stripped note (R2) and the note without its trailing fragment (R3)
    v5_refs.jsonl              per eligible handoff: B's action on the frozen X1 note of v5 and B's five repeat
                               actions of v5 arm C3 (action names only)
    calibration_v7.json        the v5 input-token rule, unchanged
    frame_meta_v7.json         counts, hashes, the request-shape checks and the R0 table

Sets. gated: a_H equals a_star (the full-history decision is right). eligible: gated and a_V equals a_H (A's
action from the view is right; the 490 of v3). A-wrong: gated and a_V differs from a_H, parse errors included.
All at B = 1,000, from the v3 merge (`headline.merge.merge_data`).
"""

from __future__ import annotations

import json
import sqlite3
from collections import Counter
from pathlib import Path
from typing import Any

from ..explore4.analysis import _states
from ..explore5 import engine as E5
from ..explore5 import frame as F5
from ..explore5.frame import Handoff, load_cases, load_frame, sha256_file  # noqa: F401  (the same frame code)
from ..headline import requests as rq
from ..headline.merge import merge_data
from ..headline.tasks import rebuild_view
from ..schema import ACTIONS, Case
from . import prompts as P

RESULTS = F5.RESULTS
V4 = RESULTS / "exploratory-v4"
V5 = RESULTS / "exploratory-v5"
BASE = RESULTS / "headline-300"
U2RUN = RESULTS / "headline-u2-v3"
OUT = RESULTS / "exploratory-v7"
RESIDUE_BUDGET = 1000

WRONG_FRAME = "wrong_frame.jsonl"
R_NOTES = "r_notes.jsonl"
V5_REFS = "v5_refs.jsonl"
CALIBRATION_NAME = "calibration_v7.json"
META_NAME = "frame_meta_v7.json"

N_GATED, N_ELIGIBLE, N_WRONG, N_CASES = 1160, 490, 670, 300
N_CUT, N_MID, N_V3_FLIPS, N_X1_FLIPS = 349, 331, 142, 71


def _jsonl(rows: list[dict[str, Any]]) -> str:
    return "".join(json.dumps(r, sort_keys=True, separators=(",", ":")) + "\n" for r in rows)


def _read_jsonl(p: Path) -> list[dict[str, Any]]:
    return [json.loads(x) for x in Path(p).read_text(encoding="utf-8").splitlines() if x.strip()]


def _write_once(p: Path, text: str, what: str) -> None:
    if p.exists() and p.read_text(encoding="utf-8") != text:
        raise SystemExit(f"{p} exists and differs from the rebuilt {what}")
    p.write_text(text, encoding="utf-8")


def by_key(rows: list[dict[str, Any]]) -> dict[tuple[str, str], dict[str, Any]]:
    return {(r["case_id"], r["system"]): r for r in rows}


# ---------------------------------------------------------------------------
# The A-wrong handoffs
# ---------------------------------------------------------------------------

def wrong_rows(decisions: list[dict[str, Any]], view_of) -> list[dict[str, Any]]:
    """The A-wrong rows among stage-1 decision rows, in their order. `view_of(case_id, system)` gives the stored
    view dict and its source dir name."""
    out = []
    for r in decisions:
        if int(r["budget"]) != RESIDUE_BUDGET or r["a_H"] != r["a_star"] or r["a_V"] == r["a_H"]:
            continue
        view, source = view_of(r["case_id"], r["system"])
        out.append({"case_id": r["case_id"], "system": r["system"], "mechanism": r["mechanism"], "a_star": r["a_star"],
                    "a_H": r["a_H"], "a_V": r["a_V"], "view": view, "source": source})
    return out


def _stored_view(base: Path, u2: Path):
    memo: dict[str, tuple[dict, str]] = {}

    def view_of(case_id: str, system: str) -> tuple[dict, str]:
        if case_id not in memo:
            src = u2 if (u2 / "state" / "cases" / f"{case_id}.json").exists() else base
            memo[case_id] = (json.loads((src / "state" / "cases" / f"{case_id}.json").read_text()), src.name)
        st, name = memo[case_id]
        return st["cells"][f"{system}@{RESIDUE_BUDGET}"]["view"], name
    return view_of


def load_wrong(root: Path, cases: list[Case] | None = None) -> list[Handoff]:
    cases = cases or load_cases(root)
    by = {c.case_id: c for c in cases}
    return [Handoff(case=by[f["case_id"]], system=f["system"],
                    view=rebuild_view(by[f["case_id"]], f["system"], RESIDUE_BUDGET, f["view"]), row=f)
            for f in _read_jsonl(root / WRONG_FRAME)]


# ---------------------------------------------------------------------------
# The v3 notes: cut by the cap, and the two text rules
# ---------------------------------------------------------------------------

def raw_note_chars(run: Path) -> dict[tuple[str, str], int]:
    """Characters of each v3 note reply before the 150-token cut, by (case, system). Lengths only."""
    out: dict[tuple[str, str], int] = {}
    for rd in sorted((run / "rounds").glob("*/round.json")):
        req = {cid: q for cid, q in json.loads(rd.read_text())["requests"].items() if q.get("kind") == "note"}
        if not req:
            continue
        for res in sorted(rd.parent.glob("results-*.jsonl")):
            for line in res.read_text(encoding="utf-8").splitlines():
                d = json.loads(line) if line.strip() else {}
                q = req.get(d.get("custom_id"))
                if q is None or (d.get("result") or {}).get("type") != "succeeded":
                    continue
                for c in d["result"]["message"]["content"]:
                    if c.get("type") != "text":
                        continue
                    try:
                        note = json.loads(c["text"]).get("note")
                    except (ValueError, AttributeError):
                        note = None
                    if isinstance(note, str):
                        out[(q["case_id"], q["cell"].split("@")[0])] = len(note.strip())
    return out


def _names(text: str, a_A: str) -> tuple[int, int]:
    return int(any(a in text for a in ACTIONS)), int(a_A in text)


def r_note_row(case_id: str, system: str, note: str, a_A: str, raw_chars: int | None) -> dict[str, Any]:
    """The text-rule results for one v3 note. `cut` is None when the reply length is not known."""
    r2, dropped = P.strip_hedges(note)
    r3, frag = P.drop_trailing_fragment(note)
    f_any, f_own = _names(frag, a_A)
    n_any, n_own = _names(note, a_A)
    return {"case_id": case_id, "system": system, "note_chars": len(note), "raw_note_chars": raw_chars,
            "cut": None if raw_chars is None else int(raw_chars > len(note)), "mid": int(P.ends_mid_sentence(note)),
            "names_action": n_any, "names_a_action": n_own,
            "r2_note": r2, "r2_changed": int(bool(dropped)), "r2_empty": int(bool(dropped) and not r2.strip()),
            "r2_dropped": len(dropped), "r2_rules": sorted({x for d in dropped for x in d["rules"]}),
            "r3_note": r3, "r3_changed": int(r3 != note), "r3_empty": int(r3 != note and not r3.strip()),
            "r3_fragment_chars": len(frag), "r3_fragment_names_action": f_any, "r3_fragment_names_a_action": f_own}


def r_note_rows(hs: list[Handoff], raw: dict[tuple[str, str], int]) -> list[dict[str, Any]]:
    return [r_note_row(h.case.case_id, h.system, h.row["v3"]["note"], h.row["a_A"], raw.get((h.case.case_id, h.system)))
            for h in hs]


def r_note_counts(rows: list[dict[str, Any]]) -> dict[str, Any]:
    rules = Counter(x for r in rows for x in r["r2_rules"])
    return {"notes": len(rows), "cut": sum(bool(r["cut"]) for r in rows), "cut_unknown": sum(r["cut"] is None for r in rows),
            "mid_sentence": sum(r["mid"] for r in rows), "cut_and_mid_sentence": sum(bool(r["cut"]) and r["mid"] for r in rows),
            "r2": {"changed": sum(r["r2_changed"] for r in rows), "empty": sum(r["r2_empty"] for r in rows),
                   "sentences_dropped": sum(r["r2_dropped"] for r in rows), "notes_by_rule": dict(sorted(rules.items()))},
            "r3": {"changed": sum(r["r3_changed"] for r in rows), "empty": sum(r["r3_empty"] for r in rows),
                   "fragment_names_an_action": sum(r["r3_fragment_names_action"] for r in rows),
                   "fragment_names_a_action": sum(r["r3_fragment_names_a_action"] for r in rows),
                   "note_names_an_action": sum(r["names_action"] for r in rows)},
            "r2_and_r3_same_text": sum(r["r2_changed"] and r["r3_changed"] and r["r2_note"] == r["r3_note"] for r in rows)}


# ---------------------------------------------------------------------------
# The frozen v5 references: X1 and C3, action names only
# ---------------------------------------------------------------------------

def v5_ref_rows(hs: list[Handoff], v5: Path = V5) -> list[dict[str, Any]]:
    sx, sc = _states(v5, "X1"), _states(v5, "C3")
    if not sx or not sc:
        raise SystemExit(f"no finished X1 and C3 arms under {v5}")
    out = []
    for h in hs:
        k = (h.case.case_id, h.system)
        x, c = sx[k]["out"], sc[k]["out"]
        out.append({"case_id": k[0], "system": k[1], "x1_a_B": x["a_B"], "x1_note_tokens": x["note_tokens"],
                    "x1_p_residue": x["p_residue"], "c3_a_B": list(c["a_B"])})
    return out


# ---------------------------------------------------------------------------
# Calibration and request-shape checks
# ---------------------------------------------------------------------------

def calibration_text() -> str:
    return json.dumps(F5.calibration_v5(), indent=1, sort_keys=True) + "\n"


def _has_key(db: sqlite3.Connection, params: dict[str, Any]) -> bool:
    return db.execute("SELECT 1 FROM calls WHERE key = ?", (rq.request_key(params),)).fetchone() is not None


def _ro(path: Path) -> sqlite3.Connection:
    return sqlite3.connect(f"file:{path}?mode=ro", uri=True)


def request_shape_check(root: Path, results: Path = RESULTS, v5: Path = V5) -> dict[str, Any]:
    """On the 490 eligible handoffs, the note requests v7 builds for XW are the stored requests of the frozen runs:
    the v3 note request is in the v3 call caches and the X1 note request is in the v5 X1 call cache (read only)."""
    from .engine import v3_note_request, x1_note_request
    hs = load_frame(root)
    out = {"v3_note_request": {"n": 0, "found": 0}, "x1_note_request": {"n": 0, "found": 0}}
    dbs: dict[Path, sqlite3.Connection] = {}
    dbx = _ro(v5 / "X1" / "state" / "calls.sqlite")
    try:
        for h in hs:
            src = results / Path(h.row["source"]).name / "state" / "calls.sqlite"
            if src not in dbs:
                dbs[src] = _ro(src)
            out["v3_note_request"]["n"] += 1
            out["v3_note_request"]["found"] += _has_key(dbs[src], v3_note_request(h, E5.NOTE_WRITER))
            out["x1_note_request"]["n"] += 1
            out["x1_note_request"]["found"] += _has_key(dbx, x1_note_request(h, E5.NOTE_WRITER))
    finally:
        dbx.close()
        for d in dbs.values():
            d.close()
    return out


# ---------------------------------------------------------------------------
# Stage
# ---------------------------------------------------------------------------

def stage_frame(root: Path = OUT, v4: Path = V4, results: Path = RESULTS, v5: Path = V5) -> dict[str, Any]:
    """Verify the frozen inputs and write the v7 run files. Safe to repeat (identical bytes only)."""
    F5.verify_v4_frame(v4)
    root.mkdir(parents=True, exist_ok=True)
    for name in ("frame.jsonl", "cases.jsonl"):
        dst = root / name
        if dst.exists() and sha256_file(dst) != sha256_file(v4 / name):
            raise SystemExit(f"{dst} exists and differs from the frozen v4 file")
        if not dst.exists():
            dst.write_bytes((v4 / name).read_bytes())
    hs = load_frame(root)
    base, u2 = results / BASE.name, results / U2RUN.name
    data, _ = merge_data(base, u2, "U2")
    dec = [r for r in data["decisions"] if int(r["budget"]) == RESIDUE_BUDGET]
    gated = [r for r in dec if r["a_H"] == r["a_star"]]
    elig = {(r["case_id"], r["system"]) for r in gated if r["a_V"] == r["a_H"]}
    if elig != {(h.case.case_id, h.system) for h in hs}:
        raise SystemExit("the eligible handoffs of the merge are not the 490 of the frame")
    wrong = wrong_rows(data["decisions"], _stored_view(base, u2))
    if (len(gated), len(elig), len(wrong)) != (N_GATED, N_ELIGIBLE, N_WRONG):
        raise SystemExit(f"gated {len(gated)}, eligible {len(elig)}, A-wrong {len(wrong)}: not {N_GATED}, {N_ELIGIBLE}, {N_WRONG}")
    if [c.case_id for c in data["cases"]] != [c.case_id for c in load_cases(root)]:
        raise SystemExit("the merged cases are not the cases of the frame, in order")
    _write_once(root / WRONG_FRAME, _jsonl(wrong), "A-wrong handoffs")

    raw = raw_note_chars(base) | raw_note_chars(u2)
    notes = r_note_rows(hs, raw)
    counts = r_note_counts(notes)
    if (counts["cut"], counts["mid_sentence"], counts["cut_unknown"]) != (N_CUT, N_MID, 0):
        raise SystemExit(f"cut {counts['cut']}, mid-sentence {counts['mid_sentence']}, unknown {counts['cut_unknown']}: "
                         f"not {N_CUT}, {N_MID}, 0")
    _write_once(root / R_NOTES, _jsonl(notes), "text-rule notes")

    refs = v5_ref_rows(hs, v5)
    x1_flips = sum(r["x1_a_B"] != h.row["a_A"] for r, h in zip(refs, hs))
    if x1_flips != N_X1_FLIPS or any(len(r["c3_a_B"]) != 5 for r in refs):
        raise SystemExit(f"v5 references: X1 flips {x1_flips}, not {N_X1_FLIPS}, or a C3 row without five repeats")
    _write_once(root / V5_REFS, _jsonl(refs), "v5 references")
    _write_once(root / CALIBRATION_NAME, calibration_text(), "input-token rule")

    chk = request_shape_check(root, results, v5)
    if any(c["n"] != c["found"] for c in chk.values()):
        raise SystemExit(f"request-shape check failed: {chk}")
    from .analysis import r0_table, staging_rows
    meta = {"n_cases": len(data["cases"]), "n_gated": len(gated), "n_eligible": len(elig), "n_a_wrong": len(wrong),
            "a_wrong_by_system": dict(sorted(Counter(r["system"] for r in wrong).items())),
            "a_wrong_a_V": dict(sorted(Counter(r["a_V"] for r in wrong).items())),
            "a_wrong_a_star": dict(sorted(Counter(r["a_star"] for r in wrong).items())),
            "notes": counts, "v5_refs": {"x1_flips": x1_flips},
            "request_shape_check": chk, "R0": r0_table(staging_rows(root)),
            **{k: sha256_file(root / name) for k, name in (("frame_sha256", "frame.jsonl"), ("cases_sha256", "cases.jsonl"),
                                                           ("wrong_frame_sha256", WRONG_FRAME), ("r_notes_sha256", R_NOTES),
                                                           ("v5_refs_sha256", V5_REFS), ("calibration_sha256", CALIBRATION_NAME))}}
    (root / META_NAME).write_text(json.dumps(meta, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return meta

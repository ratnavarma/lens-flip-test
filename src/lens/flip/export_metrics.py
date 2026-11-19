"""Export the non-text metric files behind the paper's tables.

    uv run python -m lens.flip.export_metrics --raw RAW_RESULTS --out results --banned WORDS_FILE

RAW_RESULTS is the private results directory that holds the raw runs (headline-300, headline-u2-v3,
exploratory-v4, exploratory-v5). The raw runs hold model text and are not released. This script reads
them and writes only ids, enumerated names, flags, counts, token counts and scores:

    headline-300-v3/stage1_metrics_v3.jsonl            one row per case, system and view budget (Table 1)
    headline-300-v3/handoff_metrics_v3.jsonl           one row per eligible handoff (Tables 2 to 3b)
    exploratory-v4/handoff_metrics_v4.jsonl            one row per handoff and v4 arm (Tables 4 to 6)
    exploratory-v5/handoff_metrics_v5.jsonl            one row per handoff and v5 or v5b arm (Tables 7, 9 to 12)
    exploratory-v5/c1_control_metrics_v5.jsonl         one row per v3 flip and C1 control draw (Tables 8 and 9)
    exploratory-v5/judge_calibration_metrics_v5.jsonl  one row per calibration pair (Table 10, first column)
    exploratory-v5/judge2_frozen_v5.json               prompt hash, cosine threshold and settings of judge v2

No note, view, unit, reason or prompt text leaves this script. `check` refuses a value that is nested, a
string longer than 32 characters, or a string that is not a single plain token. It also refuses any word of
a private word list. The list path comes from LENS_BANNED_WORDS or --banned, so the list is never in code.
"""

from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path
from typing import Any

from .headline.merge import merge_data
from .headline.metrics import h2b_items

ACTIONS = ("EXPLAIN", "OPEN_REVIEW", "CHECK_DEVICE", "APPLY_CREDIT", "REQUEST_INFO")
MAX_STR = 32
TOKEN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:\-]*$")

V3_NAME = "headline-300-v3/handoff_metrics_v3.jsonl"
S1_NAME = "headline-300-v3/stage1_metrics_v3.jsonl"
V4_NAME = "exploratory-v4/handoff_metrics_v4.jsonl"
V5_NAME = "exploratory-v5/handoff_metrics_v5.jsonl"
C1_NAME = "exploratory-v5/c1_control_metrics_v5.jsonl"
CAL_NAME = "exploratory-v5/judge_calibration_metrics_v5.jsonl"
JUDGE_NAME = "exploratory-v5/judge2_frozen_v5.json"
C4_ARMS = {"haiku_v1": "C4-judge1", "qwen30": "C4-judge2", "embedding": "C4-embed"}


def group(m: str) -> str:
    return "M0" if m == "M0" else "M1-M3" if m in ("M1", "M2", "M3") else "U1-U2"


def _names(note: str | None, a_A: str) -> tuple[bool, bool]:
    note = note or ""
    return any(a in note for a in ACTIONS), a_A in note


def _jsonl(p: Path) -> list[dict]:
    return [json.loads(x) for x in p.read_text(encoding="utf-8").splitlines() if x.strip()]


# ---------------------------------------------------------------------------
# v3: stage 1 decisions and the eligible handoffs
# ---------------------------------------------------------------------------

def _raw_note_chars(run: Path) -> dict[tuple[str, str], int]:
    """Characters of each note reply before the 150-token cut, by (case, system). Lengths only."""
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


def v3_tables(raw: Path) -> tuple[list[dict], list[dict]]:
    data, _ = merge_data(raw / "headline-300", raw / "headline-u2-v3", "U2")
    raw_chars = _raw_note_chars(raw / "headline-300") | _raw_note_chars(raw / "headline-u2-v3")
    stage1 = [{"case_id": r["case_id"], "system": r["system"], "budget": r["budget"], "mechanism": r["mechanism"],
               "mechanism_group": group(r["mechanism"]), "a_star": r["a_star"], "a_H": r["a_H"], "a_V": r["a_V"]}
              for r in data["decisions"]]
    recs = {(r["case_id"], r["system"]): r for r in data["residue_records"]}
    cfg = data["config"]
    h2b = {(it["case_id"], it["system"]): it
           for it in h2b_items(data["cases"], data["residue_rows"], recs, int(cfg.get("h2b_draws", 1000)),
                               int(cfg["bootstrap"]["seed"]))}
    out = []
    for r in data["residue_rows"]:
        rec = recs.get((r["case_id"], r["system"]))
        it = h2b.get((r["case_id"], r["system"]))
        any_, own = _names(r["note"], r["a_A"])
        rc = raw_chars.get((r["case_id"], r["system"]))
        out.append({
            "case_id": r["case_id"], "system": r["system"], "budget": r["budget"], "mechanism": r["mechanism"],
            "mechanism_group": group(r["mechanism"]), "a_star": r["a_star"], "a_H": r["a_H"], "a_A": r["a_A"],
            "a_B": r["a_B"], "floor_action": r["a_floor"], "trunc_action": r["a_trunc"],
            "flip": bool(r["residue_flip"]), "floor_flip": r["a_floor"] != r["a_A"],
            "restored": rec["status"] if rec else None,
            "restored_minimal_size": len(rec["event_ids"]) if rec and rec["status"] == "restored" else None,
            "restored_cap_hit": bool(rec["hit_cap"]) if rec and rec["status"] == "restored" else None,
            "p_residue": r["p_residue"], "note_tokens": r["note_tokens"],
            "names_action": any_, "names_a_action": own, "omission_fraction": r["omission"],
            "view_units": r["view_units"], "dstar_in_view": bool(r["dstar_in_view"]),
            "note_chars": len(r["note"]), "note_ends_sentence": r["note"].rstrip()[-1:] in tuple(".!?"),
            "raw_note_chars": rc, "note_cut_at_cap": None if rc is None else rc > len(r["note"]),
            "h2b_recall": it["recall"] if it else None,
            "h2b_random_recall": round(it["random_recall"], 6) if it else None,
        })
    return stage1, out


# ---------------------------------------------------------------------------
# v4
# ---------------------------------------------------------------------------

def _state(root: Path, arm: str) -> dict:
    out = {}
    for p in (root / arm / "state" / "cases").glob("*.json"):
        for k, v in json.loads(p.read_text()).items():
            if isinstance(v, dict) and "out" in v:
                out[(p.stem, k)] = v
    return out


def v4_rows(raw: Path, v3: list[dict]) -> list[dict]:
    root = raw / "exploratory-v4"
    ext, strp, son, sonv, bud = (_state(root, a) for a in ("A-extractive", "A-stripped", "B-sonnet", "B-sonnet-view",
                                                           "C-budget"))
    rows = []
    for b in v3:
        k = (b["case_id"], b["system"])
        base = {"case_id": b["case_id"], "system": b["system"], "mechanism": b["mechanism"],
                "mechanism_group": b["mechanism_group"], "a_A": b["a_A"]}

        def add(arm, **kw):
            row = {**base, "arm": arm, "a_B": None, "flip": None, "restored": None, "restored_verified": None,
                   "floor_action": None, "floor_flip": None, "p_residue": None, "note_tokens": None,
                   "omission_fraction": None}
            row.update(kw)
            rows.append(row)
        o = ext[k]["out"]
        rec = ext[k].get("record")
        add("extractive", a_B=o["a_B"], flip=bool(o["residue_flip"]), restored=rec["status"] if rec else None,
            restored_verified=bool(rec and rec["status"] == "restored" and rec.get("verified")),
            p_residue=o["p_residue"], note_tokens=o["note_tokens"], omission_fraction=o["omission"])
        o = strp[k]["out"]
        add("stripped", a_B=o["a_B"], flip=bool(o["residue_flip"]), note_tokens=o["note_tokens"])
        o, v = son[k]["out"], sonv[k]["out"]
        # the writer is unchanged, so the frozen v3 p_residue is the score for the reader's flips
        add("sonnet", a_B=o["a_B"], flip=bool(o["residue_flip"]), floor_action=v["a_view_v2"],
            floor_flip=bool(v["view_miss"]), p_residue=b["p_residue"], note_tokens=b["note_tokens"])
        for bg in ("300", "600"):
            o = bud[k]["out"][bg]
            add("budget" + bg, a_B=o["a_B"], flip=bool(o["residue_flip"]), p_residue=o["p_residue"],
                note_tokens=o["note_tokens"])
    return rows


# ---------------------------------------------------------------------------
# v5 and v5b
# ---------------------------------------------------------------------------

V5_KEYS = ("a_B", "flip", "p_residue", "note_tokens", "note_status", "cut", "complete", "n_empty_fields", "reasked",
           "restored", "restored_cap_hit", "rewrite_ok", "rewrite_tokens", "rewrite_identical", "rewrite_lost_fact",
           "rep0", "rep1", "rep2", "rep3", "rep4", "n_units", "n_judged", "n_absent", "d_n", "d_conveyed")


def v5_tables(raw: Path) -> tuple[list[dict], list[dict], list[dict]]:
    """Rows for the v5 arms. The derived flags come from the explore5 analysis code, so they are the ones
    summary_v5.json, summary_v5b.json and summary_v5_judge2.json were built from."""
    from .explore5 import analysis as A

    root = raw / "exploratory-v5"
    cases, rows = A.base_rows(root)
    st = A.NStats(cases, rows)
    S = {a: A._states(root, a) for a in ("X1", "X2", "X2b", "C1", "C2", "C3", "C4")}
    out: list[dict] = []

    def add(r: dict, arm: str, **kw: Any) -> None:
        row = {"case_id": r["case_id"], "system": r["system"], "mechanism": r["mechanism"],
               "mechanism_group": group(r["mechanism"]), "a_A": r["a_A"], "arm": arm, **{k: None for k in V5_KEYS}}
        row.update(kw)
        out.append(row)

    for r in rows:
        k = (r["case_id"], r["system"])
        o = S["X1"][k]["out"]
        add(r, "X1", a_B=o["a_B"], flip=bool(o["residue_flip"]), p_residue=o["p_residue"],
            note_tokens=o["note_tokens"], note_status=o["note_status"])
        for arm in ("X2", "X2b"):
            o, rec = S[arm][k]["out"], S[arm][k].get("record")
            empty = sum(not (v or "").strip() for v in (o["fields"] or {}).values())
            add(r, arm, a_B=o["a_B"], flip=bool(o["residue_flip"]), p_residue=o["p_residue"],
                note_tokens=o["note_tokens"], note_status=o["note_status"], cut=bool(o["cut"]),
                complete=bool(o["complete"]), n_empty_fields=empty,
                reasked=bool(o["reasked"]) if arm == "X2b" else None,
                restored=rec["status"] if rec else None,
                restored_cap_hit=bool(rec["hit_cap"]) if rec and rec["status"] == "restored" else None)
    A.study_c2(st, rows, S["C2"])
    for r in rows:
        o = S["C2"][(r["case_id"], r["system"])]["out"]
        ok = bool(r["rw_has"])
        add(r, "C2", a_B=o["a_B"] if ok else None, flip=bool(r["rw_flip"]) if ok else None, rewrite_ok=ok,
            rewrite_tokens=r.get("rw_tokens"), rewrite_identical=bool(r["rw_identical"]),
            rewrite_lost_fact=bool(r["rw_lost_fact"]) if ok else None)
    for r in rows:
        a = list(S["C3"][(r["case_id"], r["system"])]["out"]["a_B"])
        assert len(a) == 5, "C3 has five repeats"
        add(r, "C3", **{f"rep{i}": x for i, x in enumerate(a)})

    # C4: three judges of the same note and unit pairs
    dec = _jsonl(root / "judge2" / "decisions.jsonl")
    tau = json.loads((root / "judge2" / "calibration_v2.json").read_text())["tau_kappa"]["tau"]
    by: dict[tuple[str, str], list[dict]] = {}
    for d in dec:
        by.setdefault((d["case_id"], d["system"]), []).append(d)
    for v in by.values():
        v.sort(key=lambda d: d["unit_index"])
    judges = {
        "haiku_v1": {k: {"out": {"conveyed": list(v["out"]["conveyed"])}} for k, v in S["C4"].items()},
        "qwen30": {k: {"out": {"conveyed": [d["conveyed"] for d in v]}} for k, v in by.items()},
        "embedding": {k: {"out": {"conveyed": [d["cosine"] >= tau for d in v]}} for k, v in by.items()},
    }
    for name, states in judges.items():
        _, jr = A.base_rows(root)
        A.study_c4(A.NStats(cases, jr), jr, states)
        for r in jr:
            conv = states[(r["case_id"], r["system"])]["out"]["conveyed"]
            ok = [c for c in conv if c is not None]
            add(r, C4_ARMS[name], n_units=len(conv), n_judged=len(ok), n_absent=sum(not c for c in ok),
                d_n=r["sem_d_n"], d_conveyed=r["sem_d_conv"])

    # C1: one row per control draw
    draws = _jsonl(root / "c1_controls.jsonl")
    c1 = []
    for c in draws:
        got = {x["k"]: x for x in S["C1"][(c["case_id"], c["system"])]["out"]["controls"]}
        x = got.get(c["k"])
        c1.append({"case_id": c["case_id"], "system": c["system"], "k": c["k"], "send": bool(c["send"]),
                   "matched": bool(c["matched"]), "duplicate": bool(c["duplicate"]),
                   "restored": bool(x["restored"]) if x is not None else None,
                   "chars": c["chars"], "target_chars": c["target_chars"], "ratio": c["ratio"],
                   "target_source": c["target_source"], "n_units": len(c["unit_ids"])})

    # Judge calibration: truth and each judge's answer per pair
    cset = {r["id"]: r for r in _jsonl(root / "judge2" / "calibration_set.jsonl")}
    cal = []
    for d in _jsonl(root / "judge2" / "calibration_decisions.jsonl"):
        s = cset[d["id"]]
        hv = None
        if not s["note_modified"]:   # judge v1 answered the unmodified real pairs only (the v5 C4 run)
            hv = S["C4"][(s["case_id"], s["system"])]["out"]["conveyed"][s["unit_index"]]
        cal.append({"id": d["id"], "kind": d["kind"], "system": s["system"], "note_modified": bool(s["note_modified"]),
                    "truth": bool(d["truth"]), "cosine": d["cosine"], "embedding": d["embedding"],
                    "qwen30": d["qwen30"], "qwen30_status": d["qwen30_status"], "haiku_v1": hv})
    return out, c1, cal


def judge2_frozen(raw: Path) -> dict[str, Any]:
    """The settings of the local judge v2 as its calibration file froze them. Hashes and numbers only."""
    c = json.loads((raw / "exploratory-v5" / "judge2" / "calibration_v2.json").read_text())
    return {"schema": c["schema"], "prompt_version": c["prompt_version"], "prompt_sha256": c["prompt_sha256"],
            "format": c["format"], "options": c["options"], "pass_bar": c["pass_bar"],
            "embedding_tau": c["tau_kappa"]["tau"], "band": {k: c["band"][k] for k in ("tau_low", "tau_high")},
            "calibration_set": {"n": c["recipe"]["n"], "seed": c["recipe"]["seed"], "sha256": c["recipe"]["sha256"],
                                "by_kind": c["recipe"]["by_kind"]}}


# ---------------------------------------------------------------------------
# Checks and output
# ---------------------------------------------------------------------------

def check(path: Path, banned: list[str]) -> None:
    text = path.read_text(encoding="utf-8")
    low = text.lower()
    hits = [w for w in banned if w in low]
    assert not hits, f"{path}: banned words present ({len(hits)})"
    for line in text.splitlines():
        for key, val in json.loads(line).items():
            assert isinstance(val, (str, int, float, bool, type(None))), f"{path}: nested value in {key}"
            if isinstance(val, str):
                assert len(val) <= MAX_STR, f"{path}: long string in {key}"
                assert TOKEN.match(val), f"{path}: {key} is not a plain token"


def write(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(r, sort_keys=True, separators=(",", ":")) + "\n" for r in rows),
                    encoding="utf-8")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--raw", default="results", help="directory with the raw runs (not released)")
    ap.add_argument("--out", default="results", help="directory the metric files are written under")
    ap.add_argument("--banned", default=os.environ.get("LENS_BANNED_WORDS"), help="private word list (or env var)")
    ap.add_argument("--skip-v5", action="store_true", help="write the v3 and v4 files only")
    args = ap.parse_args()
    if not args.banned:
        raise SystemExit("set LENS_BANNED_WORDS or pass --banned")
    banned = [w.strip().lower() for w in Path(args.banned).read_text().splitlines() if w.strip()]
    assert banned, "empty banned list"
    raw, out = Path(args.raw), Path(args.out)
    s1, r3 = v3_tables(raw)
    files = {S1_NAME: s1, V3_NAME: r3, V4_NAME: v4_rows(raw, r3)}
    if not args.skip_v5:
        r5, c1, cal = v5_tables(raw)
        files |= {V5_NAME: r5, C1_NAME: c1, CAL_NAME: cal}
        (out / JUDGE_NAME).write_text(json.dumps(judge2_frozen(raw), indent=1, sort_keys=True) + "\n", encoding="utf-8")
        print(f"{out / JUDGE_NAME}: written")
    for name, rows in files.items():
        write(out / name, rows)
        check(out / name, banned)
        print(f"{out / name}: {len(rows)} rows")
    print(f"checks passed ({len(banned)} words)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

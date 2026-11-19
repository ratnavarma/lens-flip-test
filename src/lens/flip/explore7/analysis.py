"""summary_v7.json and the text-free metric file for exploratory protocol v7. Offline.

Two steps, as in v6. `metric_rows` reads the finished run dirs and writes one flat row per gated handoff (490
eligible, then 670 A-wrong): ids, system, mechanism group, action names, counts and 0/1 flags. No note, view or
prompt text. `summarize` computes every v7 number from those rows alone (`rebuild` does exactly that).

Bootstrap: the v3 one. 1,000 resamples of the 300 cases, seed 11; every handoff of a resampled case comes
along. A row carries its case's position in cases.jsonl (`case_index`) and the case count. Every difference
between two arms on the same handoffs is paired by case and handoff. A difference between two groups of
handoffs (R0, the split by cut) is the difference of two ratios under the same case resamples.

The definitions and reading rules below are frozen in protocol_v7.json (item x7.analysis) before any model call.
"""

from __future__ import annotations

import json
import os
from collections import Counter
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Sequence

from ..explore4.analysis import _complete, _states
from ..explore5.analysis import NStats
from ..schema import ACTIONS, REQUEST_INFO
from . import SCHEMA, VERSION
from . import frame as F
from .engine import ARMS, STAGES, SUB_ARMS
from .frame import OUT, load_cases, load_frame

SYSTEMS = ("recent", "rag-bm25", "rag-embed", "summary")
METRICS_NAME = "handoff_metrics_v7.jsonl"
SUMMARY_NAME = "summary_v7.json"
MAX_STR = 32
C3_DIFFER_REPORTED = 0.093  # docs/exploratory-v5-results-2026-10-29.md: 9.3 in 100 repeat answers differ from the v3 answer

R_RULE = ("an arm 'reduces deferral' only if the upper 95% bound of (arm deferral-flip rate minus v3 deferral-flip rate) "
          "on the handoffs with an arm decision is below zero; 'does not' otherwise")
X_RULE = ("the gain 'comes with the policy' if X1p minus v3 has its upper 95% bound below zero and X1r minus v3 does not; "
          "'comes with the reader description' in the mirrored case; 'both' if both upper bounds are below zero; "
          "'neither separable' if neither is")

# The pre-registered analysis, as data. Its canonical JSON is hashed into protocol_v7.json (x7.analysis).
ANALYSIS_SPEC: dict[str, Any] = {
    "bootstrap": {"reps": 1000, "seed": 11, "unit": "case", "paired": True},
    "outcomes": {"flip": "B's action differs from A's action (eligible handoffs: A's action is the right action)",
                 "deferral_flip": "a flip where B's action is REQUEST_INFO",
                 "commit_flip": "a flip where B's action is one of the four other actions",
                 "unparsed": "a reply that is refused, cut, unparsed or failed twice; it is a flip and neither deferral nor commit; counted"},
    "R0": "descriptive, no call: v3 flip and deferral-flip rates for cut against uncut notes and for notes that end "
          "mid-sentence against notes that do not, with the difference of the two rates",
    "R": {"arms": {"R1": "every handoff has a new decision", "R2": "changed notes have a new decision, unchanged notes keep the v3 decision",
                   "R3": "as R2"},
          "emptied": "a note the rule empties is not sent, is counted, and is left out of that arm's rows",
          "per_arm_against_v3": ["flip rate difference", "deferral-flip rate difference", "commit-flip rate difference"],
          "also": ["each difference restricted to the notes the arm changed (R1: all notes)",
                   "arm flip rate and deferral-flip rate split by whether the cap cut the v3 note, with the v3 rates and differences",
                   "noise floor: the share of arm answers that differ from the v3 answer on changed notes, beside the same share of the "
                   "five C3 repeats; and each arm rate minus the mean rate of the five C3 repeats",
                   "R3, descriptive: the differences split by whether the dropped fragment names an action"],
          "rule": R_RULE},
    "X": {"1": {"rows": "the 490 eligible handoffs", "rates": ["v3", "X1 (frozen v5)", "X1p", "X1r"],
                "differences": ["X1 minus v3", "X1p minus v3", "X1r minus v3", "X1p minus X1r", "X1p minus X1", "X1r minus X1"],
                "rule": X_RULE},
          "2": {"rows": "the 670 A-wrong handoffs", "rescue": "B's action equals the right action",
                "repeat": "B's action equals A's wrong action", "under": ["v3 notes", "X1 notes"],
                "differences": ["rescue, X1 minus v3", "repeat, X1 minus v3"]},
          "3": {"rows": "all 1,160 gated handoffs", "accuracy": "B's action equals the right action",
                "under": ["v3 notes (eligible: the frozen v3 decision)", "X1 notes (eligible: the frozen v5 X1 decision)"],
                "difference": "X1 notes minus v3 notes: the two-sided score"}},
    "multiple_comparisons": "no correction",
}


def group(m: str) -> str:
    return "M0" if m == "M0" else "M1-M3" if m in ("M1", "M2", "M3") else "U1-U2"


# ---------------------------------------------------------------------------
# Step 1: the text-free rows
# ---------------------------------------------------------------------------

def _c3(acts: Sequence[str], a_A: str, v3_a_B: str) -> dict[str, int]:
    return {"c3_n": len(acts), "c3_flip": sum(a != a_A for a in acts),
            "c3_defer": sum(a != a_A and a == REQUEST_INFO for a in acts),
            "c3_commit": sum(a != a_A and a in ACTIONS and a != REQUEST_INFO for a in acts),
            "c3_differ": sum(a != v3_a_B for a in acts)}


ARM_FIELDS = ("r1_a_B", "r2_a_B", "r3_a_B", "x1p_a_B", "x1r_a_B", "x1p_note_tokens", "x1r_note_tokens", "x1p_p", "x1r_p",
              "x1p_note_status", "x1r_note_status", "wv3_a_B", "wx1_a_B", "wv3_note_tokens", "wx1_note_tokens",
              "wv3_note_status", "wx1_note_status")


def _short(s: Any) -> Any:
    return s[:MAX_STR] if isinstance(s, str) else s


def base_rows(root: Path) -> list[dict[str, Any]]:
    """The rows without any v7 outcome: what is known before a call (used for R0 at staging)."""
    cases = load_cases(root)
    idx = {c.case_id: i for i, c in enumerate(cases)}
    notes, refs = F.by_key(F._read_jsonl(root / F.R_NOTES)), F.by_key(F._read_jsonl(root / F.V5_REFS))
    rows = []

    def common(h, kind: str, a_A: str) -> dict[str, Any]:
        return {"case_id": h.case.case_id, "case_index": idx[h.case.case_id], "n_cases": len(cases), "system": h.system,
                "mechanism": h.case.mechanism, "mechanism_group": group(h.case.mechanism), "set": kind,
                "a_star": h.row["a_star"], "a_A": a_A,
                **{k: None for k in ("v3_a_B", "cut", "mid", "note_names_action", "frag_names_action", "c3_n", "c3_flip",
                                     "c3_defer", "c3_commit", "c3_differ", "x1_a_B", "r2_changed", "r2_empty",
                                     "r3_changed", "r3_empty") + ARM_FIELDS}}
    for h in load_frame(root, cases):
        k = (h.case.case_id, h.system)
        v3, rn, ref = h.row["v3"], notes[k], refs[k]
        r = common(h, "eligible", h.row["a_A"])
        r.update({"v3_a_B": v3["a_B"], "cut": rn["cut"], "mid": rn["mid"], "note_names_action": rn["names_action"],
                  "frag_names_action": rn["r3_fragment_names_action"], **_c3(ref["c3_a_B"], h.row["a_A"], v3["a_B"]),
                  "x1_a_B": ref["x1_a_B"], "r2_changed": rn["r2_changed"], "r2_empty": rn["r2_empty"],
                  "r3_changed": rn["r3_changed"], "r3_empty": rn["r3_empty"]})
        rows.append(r)
    for h in F.load_wrong(root, cases):
        rows.append(common(h, "a_wrong", h.row["a_V"]))
    return rows


staging_rows = base_rows


def metric_rows(root: Path = OUT) -> list[dict[str, Any]]:
    rows = base_rows(root)
    n_e = sum(r["set"] == "eligible" for r in rows)
    s_r, s_x = _states(root, "R"), _states(root, "X")
    if not _complete(s_r, n_e) or not _complete(s_x, len(rows)):
        raise SystemExit("R or X is not complete; run both studies before the report")
    for r in rows:
        k = (r["case_id"], r["system"])
        x = s_x[k]["out"]
        if r["set"] == "eligible":
            o = s_r[k]["out"]
            r.update({"r1_a_B": o["r1"]["a_B"], "r2_a_B": o["r2"]["a_B"], "r3_a_B": o["r3"]["a_B"]})
            if (bool(r["r2_changed"]), bool(r["r2_empty"]), bool(r["r3_changed"]), bool(r["r3_empty"])) != \
                    (o["r2"]["changed"], o["r2"]["empty"], o["r3"]["changed"], o["r3"]["empty"]):
                raise SystemExit(f"{k}: the run's changed and empty flags differ from the frozen notes file")
            for t in ("x1p", "x1r"):
                r.update({f"{t}_a_B": x[t]["a_B"], f"{t}_note_tokens": x[t]["note_tokens"], f"{t}_p": x[t]["p_residue"],
                          f"{t}_note_status": _short(x[t]["note_status"])})
        else:
            for t, name in (("v3", "wv3"), ("x1", "wx1")):
                r.update({f"{name}_a_B": x[t]["a_B"], f"{name}_note_tokens": x[t]["note_tokens"],
                          f"{name}_note_status": _short(x[t]["note_status"])})
    return rows


def check_rows_text_free(text: str, banned: Sequence[str] = ()) -> None:
    """The checks of export_metrics.py: flat rows, no string longer than 32 characters, no word of the private list."""
    low = text.lower()
    hits = [w for w in banned if w in low]
    assert not hits, f"banned words present ({len(hits)})"
    for line in text.splitlines():
        for key, val in json.loads(line).items():
            assert not (isinstance(val, str) and len(val) > MAX_STR), f"long string in {key}"
            assert isinstance(val, (str, int, float, bool, type(None))), f"nested value in {key}"


def _banned(path: str | os.PathLike | None) -> list[str]:
    p = Path(path) if path else None
    if p is None or not p.exists():
        return []
    return [w.strip().lower() for w in p.read_text().splitlines() if w.strip()]


def write_metrics(root: Path, rows: list[dict[str, Any]], banned_path: str | os.PathLike | None = None) -> Path:
    text = "".join(json.dumps(r, sort_keys=True, separators=(",", ":")) + "\n" for r in rows)
    check_rows_text_free(text, _banned(banned_path))
    p = root / METRICS_NAME
    p.write_text(text, encoding="utf-8")
    return p


def read_metrics(path: Path) -> list[dict[str, Any]]:
    return [json.loads(x) for x in Path(path).read_text(encoding="utf-8").splitlines() if x.strip()]


# ---------------------------------------------------------------------------
# Step 2: every number, from the rows alone
# ---------------------------------------------------------------------------

def _stats(rows: list[dict[str, Any]]) -> NStats:
    n_cases = int(rows[0]["n_cases"])
    ids = {int(r["case_index"]): r["case_id"] for r in rows}
    return NStats([SimpleNamespace(case_id=ids.get(i, f"_no_gated_handoff_{i}")) for i in range(n_cases)], rows)


def _rate(st: NStats, flag: str, rows: list[dict] | None = None) -> dict[str, Any]:
    """The share of rows with the flag, with bounds. Safe on an empty set of rows."""
    x = st.ratio(lambda r: r[flag], lambda r: 1, rows)
    return {"value": x["value"], "ci95": x["ci95"], "k": int(x["k"]), "n": int(x["n"])}


def _diff(st: NStats, a: str, b: str, rows: list[dict] | None = None) -> dict[str, Any]:
    """Paired difference of two flags on the same rows (explore4 Stats.diff). Safe on an empty set of rows."""
    if rows is not None and not rows:
        return {"diff": None, "ci95": None, "bounds": None, "n": 0, "k_a": 0, "k_b": 0}
    return st.diff(a, b, rows)


def outcome(a_B: str | None, a_A: str, prefix: str) -> dict[str, int]:
    """flip, deferral flip, commit flip and unparsed for one of B's actions against A's action."""
    flip = a_B is not None and a_B != a_A
    return {f"{prefix}_flip": int(flip), f"{prefix}_defer": int(flip and a_B == REQUEST_INFO),
            f"{prefix}_commit": int(flip and a_B in ACTIONS and a_B != REQUEST_INFO),
            f"{prefix}_unparsed": int(a_B is not None and a_B not in ACTIONS)}


def derive(r: dict[str, Any]) -> dict[str, Any]:
    d = dict(r)
    if r["set"] == "eligible":
        for name in ("v3", "x1", "r1", "r2", "r3", "x1p", "x1r"):
            d.update(outcome(r.get(f"{name}_a_B"), r["a_A"], name))
        d["correct_v3"] = int(r["v3_a_B"] == r["a_star"])
        d["correct_x1"] = int(r["x1_a_B"] == r["a_star"])
    else:
        for name in ("wv3", "wx1"):
            a = r.get(f"{name}_a_B")
            d.update({f"{name}_rescue": int(a == r["a_star"]), f"{name}_repeat": int(a == r["a_A"]),
                      f"{name}_defer": int(a == REQUEST_INFO), f"{name}_unparsed": int(a is not None and a not in ACTIONS)})
        d["correct_v3"] = d["wv3_rescue"]
        d["correct_x1"] = d["wx1_rescue"]
    return d


def _two_groups(st: NStats, rows: list[dict], flag: str, in_a, label_a: str, label_b: str) -> dict[str, Any]:
    """The rate of `flag` in two groups of handoffs and the difference of the two rates (a minus b)."""
    a, b = [r for r in rows if in_a(r)], [r for r in rows if not in_a(r)]
    return {label_a: _rate(st, flag, a), label_b: _rate(st, flag, b),
            f"{label_a}_minus_{label_b}": st.ratio_diff((lambda r: r[flag] * int(in_a(r)), lambda r: int(in_a(r))),
                                                        (lambda r: r[flag] * int(not in_a(r)), lambda r: int(not in_a(r))), rows)}


def r0_table(raw_rows: list[dict[str, Any]]) -> dict[str, Any]:
    """R0, no model call: v3 flips by whether the cap cut the note and by whether the note ends mid-sentence."""
    rows = [{**r, **outcome(r["v3_a_B"], r["a_A"], "v3")} for r in raw_rows if r["set"] == "eligible"]
    st = _stats(rows)
    out: dict[str, Any] = {"n": len(rows), "n_cut": sum(bool(r["cut"]) for r in rows), "n_mid_sentence": sum(bool(r["mid"]) for r in rows),
                           "n_cut_and_mid_sentence": sum(bool(r["cut"]) and bool(r["mid"]) for r in rows),
                           "all": {f: _rate(st, f"v3_{f}") for f in ("flip", "defer", "commit")}}
    for key, a, b in (("cut", "cut", "not_cut"), ("mid", "mid_sentence", "ends_a_sentence")):
        out[f"by_{key}"] = {f: _two_groups(st, rows, f"v3_{f}", lambda r, k=key: bool(r[k]), a, b)
                            for f in ("flip", "defer", "commit")}
    return out


def _vs(st: NStats, rows: list[dict], arm: str, ref: str = "v3") -> dict[str, Any]:
    """Arm against the reference on the same rows: the three rates of each and the three paired differences."""
    return {"n": len(rows),
            **{f"{f}_rate": _rate(st, f"{arm}_{f}", rows) for f in ("flip", "defer", "commit")},
            **{f"{ref}_{f}_rate": _rate(st, f"{ref}_{f}", rows) for f in ("flip", "defer", "commit")},
            **{f"{f}_minus_{ref}": _diff(st, f"{arm}_{f}", f"{ref}_{f}", rows) for f in ("flip", "defer", "commit")}}


def _vs_c3(st: NStats, rows: list[dict], arm: str) -> dict[str, Any]:
    """Arm rate minus the mean rate of the five identical-note repeats of v5 arm C3, paired."""
    return {f"{f}_minus_c3": st.ratio_diff((lambda r, f=f: r[f"{arm}_{f}"], lambda r: 1),
                                           (lambda r, f=f: r[f"c3_{f}"], lambda r: r["c3_n"]), rows)
            for f in ("flip", "defer", "commit")}


def r_verdict(defer_minus_v3: dict[str, Any]) -> str:
    ci = defer_minus_v3.get("ci95")
    return "reduces deferral" if ci and ci[1] < 0 else "does not"


def r_arm(st: NStats, rows: list[dict], arm: str, changed_key: str | None, empty_key: str | None) -> dict[str, Any]:
    emptied = [r for r in rows if empty_key and r[empty_key]]
    dec = [r for r in rows if not (empty_key and r[empty_key])]
    changed = [r for r in dec if changed_key is None or r[changed_key]]
    out: dict[str, Any] = {
        "n_handoffs": len(rows), "n_emptied_not_sent": len(emptied), "n_with_decision": len(dec),
        "n_changed": len(changed) + len(emptied), "n_new_decisions": len(changed), "n_v3_decision_kept": len(dec) - len(changed),
        "unparsed_replies": sum(r[f"{arm}_unparsed"] for r in changed),
        "all": _vs(st, dec, arm), "changed_only": _vs(st, changed, arm),
        "by_cut": {name: _vs(st, [r for r in dec if bool(r["cut"]) == flag], arm) for name, flag in (("cut", True), ("not_cut", False))},
        "noise_floor": {
            "answers_that_differ_from_v3_on_changed": st.ratio(lambda r: int(r[f"{arm}_a_B"] != r["v3_a_B"]), lambda r: 1, changed),
            "c3_repeats_that_differ_from_v3_same_handoffs": st.ratio(lambda r: r["c3_differ"], lambda r: r["c3_n"], changed),
            "c3_reported": C3_DIFFER_REPORTED,
            "all": _vs_c3(st, dec, arm), "changed_only": _vs_c3(st, changed, arm)},
    }
    out["rule"] = R_RULE
    out["verdict"] = r_verdict(out["all"]["defer_minus_v3"])
    out["verdict_changed_only_descriptive"] = r_verdict(out["changed_only"]["defer_minus_v3"])
    return out


def x_verdict(p_minus_v3: dict[str, Any], r_minus_v3: dict[str, Any]) -> str:
    p = bool(p_minus_v3.get("ci95")) and p_minus_v3["ci95"][1] < 0
    r = bool(r_minus_v3.get("ci95")) and r_minus_v3["ci95"][1] < 0
    return "both" if p and r else "comes with the policy" if p else "comes with the reader description" if r else "neither separable"


def _note_block(rows: list[dict], name: str) -> dict[str, Any]:
    toks = [r[f"{name}_note_tokens"] for r in rows]
    out = {"n": len(rows), "mean_tokens": (sum(toks) / len(toks)) if toks else None,
           "status": dict(sorted(Counter(str(r[f"{name}_note_status"]) for r in rows).items()))}
    if rows and f"{name}_p" in rows[0]:
        out["p_residue_missing"] = sum(r[f"{name}_p"] is None for r in rows)
    return out


def summarize(raw_rows: list[dict[str, Any]]) -> dict[str, Any]:
    rows = [derive(r) for r in raw_rows]
    st = _stats(rows)
    el = [r for r in rows if r["set"] == "eligible"]
    wr = [r for r in rows if r["set"] == "a_wrong"]

    # ---- Study R ----
    R: dict[str, Any] = {
        "R0": r0_table(raw_rows),
        "R1_note_framed": r_arm(st, el, "r1", None, None),
        "R2_hedges_stripped": r_arm(st, el, "r2", "r2_changed", "r2_empty"),
        "R3_fragment_removed": r_arm(st, el, "r3", "r3_changed", "r3_empty"),
        "c3_reference": {"repeats_per_handoff": max((r["c3_n"] for r in el), default=0),
                         "answers_that_differ_from_v3": st.ratio(lambda r: r["c3_differ"], lambda r: r["c3_n"], el),
                         **{f"{f}_rate": st.ratio(lambda r, f=f: r[f"c3_{f}"], lambda r: r["c3_n"], el) for f in ("flip", "defer", "commit")},
                         **{f"{f}_minus_v3": st.ratio_diff((lambda r, f=f: r[f"c3_{f}"], lambda r: r["c3_n"]),
                                                           (lambda r, f=f: r[f"v3_{f}"], lambda r: 1), el) for f in ("flip", "defer", "commit")},
                         "note": "identical v3 note, five new reads (v5 arm C3): what reader noise alone does"},
    }
    r3c = [r for r in el if r["r3_changed"] and not r["r3_empty"]]
    R["R3_fragment_removed"]["by_fragment_names_action_descriptive"] = {
        name: _vs(st, [r for r in r3c if bool(r["frag_names_action"]) == flag], "r3")
        for name, flag in (("fragment_names_an_action", True), ("fragment_names_no_action", False))}

    # ---- Study X ----
    one = {"n": len(el), **{f"{a}_flip_rate": _rate(st, f"{a}_flip", el) for a in ("v3", "x1", "x1p", "x1r")},
           **{f"{a}_defer_rate": _rate(st, f"{a}_defer", el) for a in ("v3", "x1", "x1p", "x1r")},
           **{f"{a}_minus_{b}": _diff(st, f"{a}_flip", f"{b}_flip", el)
              for a, b in (("x1", "v3"), ("x1p", "v3"), ("x1r", "v3"), ("x1p", "x1r"), ("x1p", "x1"), ("x1r", "x1"))},
           "unparsed_replies": {a: sum(r[f"{a}_unparsed"] for r in el) for a in ("x1p", "x1r")},
           "notes": {a: _note_block(el, a) for a in ("x1p", "x1r")}, "rule": X_RULE}
    one["verdict"] = x_verdict(one["x1p_minus_v3"], one["x1r_minus_v3"])
    two = {"n": len(wr),
           **{f"{k}_{name}": _rate(st, f"{w}_{k}", wr) for k in ("rescue", "repeat", "defer") for w, name in (("wv3", "v3_notes"), ("wx1", "x1_notes"))},
           **{f"{k}_x1_minus_v3": _diff(st, f"wx1_{k}", f"wv3_{k}", wr) for k in ("rescue", "repeat", "defer")},
           "b_actions": {name: dict(sorted(Counter(str(r[f"{w}_a_B"]) for r in wr).items())) for w, name in (("wv3", "v3_notes"), ("wx1", "x1_notes"))},
           "a_wrong_actions": dict(sorted(Counter(str(r["a_A"]) for r in wr).items())),
           "unparsed_replies": {name: sum(r[f"{w}_unparsed"] for r in wr) for w, name in (("wv3", "v3_notes"), ("wx1", "x1_notes"))},
           "notes": {name: _note_block(wr, w) for w, name in (("wv3", "v3_notes"), ("wx1", "x1_notes"))}}
    three = {"n": len(rows), "accuracy_v3_notes": _rate(st, "correct_v3"), "accuracy_x1_notes": _rate(st, "correct_x1"),
             "x1_minus_v3": _diff(st, "correct_x1", "correct_v3"),
             "by_set": {name: {"n": len(rs), "accuracy_v3_notes": _rate(st, "correct_v3", rs), "accuracy_x1_notes": _rate(st, "correct_x1", rs),
                               "x1_minus_v3": _diff(st, "correct_x1", "correct_v3", rs)} for name, rs in (("eligible", el), ("a_wrong", wr))},
             "by_system_descriptive": {s: {"n": sum(r["system"] == s for r in rows),
                                           "x1_minus_v3": _diff(st, "correct_x1", "correct_v3", [r for r in rows if r["system"] == s])}
                                       for s in SYSTEMS if any(r["system"] == s for r in rows)}}
    return {"denominators": {"cases": int(rows[0]["n_cases"]), "gated_handoffs": len(rows), "eligible": len(el), "a_wrong": len(wr),
                             "eligible_by_system": dict(Counter(r["system"] for r in el)),
                             "a_wrong_by_system": dict(Counter(r["system"] for r in wr))},
            "bootstrap": {"reps": 1000, "seed": 11, "unit": "case (paired; every handoff of a resampled case)"},
            "multiple_comparisons": "no correction",
            "study_R": R,
            "study_X": {"1_split_on_eligible": one, "2_a_wrong": two, "3_two_sided": three}}


def rebuild(path: Path) -> dict[str, Any]:
    """Every v7 number from the metric file alone."""
    return summarize(read_metrics(path))


# ---------------------------------------------------------------------------
# Build and write
# ---------------------------------------------------------------------------

def spend_by_sub_arm(led: dict[str, Any]) -> dict[str, Any]:
    by: dict[str, Any] = {}
    stages = {s: v for a in (led.get("arms") or {}).values() for s, v in (a.get("by_stage") or {}).items()}
    for sub in SUB_ARMS:
        got = [stages[s] for s in STAGES[sub] if s in stages]
        by[sub] = {"requests": int(sum(x["requests"] for x in got)), "actual_usd": round(sum(x["usd"] for x in got), 6),
                   "by_stage": {s: stages[s] for s in STAGES[sub] if s in stages}}
    return by


def build(root: Path = OUT, banned_path: str | os.PathLike | None = None) -> dict[str, Any]:
    mpath = write_metrics(root, metric_rows(root), banned_path)
    body = rebuild(mpath)
    frozen = json.loads((root / "protocol_v7.json").read_text()) if (root / "protocol_v7.json").exists() else {}
    led = json.loads((root / "spend.json").read_text()) if (root / "spend.json").exists() else {}
    meta = json.loads((root / F.META_NAME).read_text()) if (root / F.META_NAME).exists() else {}
    return {
        "schema": SCHEMA, "version": VERSION, "exploratory": True,
        "note": "Exploratory, outside the frozen v3 to v6 protocols. No result here changes an earlier number.",
        "protocol": {"doc": "docs/flip-protocol-v7-exploratory.md", "status": frozen.get("status"),
                     "frozen_at": frozen.get("frozen_at"), "finalized_at": frozen.get("finalized_at"),
                     "rules_sha256": frozen.get("rules"), "arm_config_sha256": frozen.get("arm_configs"),
                     "reused_sha256": frozen.get("reused"),
                     "files_sha256": {k: v for k, v in frozen.items() if k.endswith("_sha256")}},
        **body,
        "text_rules": meta.get("notes"),
        "metrics_file": {"name": METRICS_NAME, "rows": len(read_metrics(mpath)), "sha256": F.sha256_file(mpath),
                         "text_free": True, "sufficient": "summarize(rows) gives every number of study_R and study_X"},
        "spend": {"cap_usd": led.get("cap_usd"), "total_usd": led.get("total_usd"), "remaining_usd": led.get("remaining_usd"),
                  "by_run_dir": {a: (led.get("arms") or {}).get(a, {}).get("actual_usd") for a in ARMS},
                  "rounds": {a: len((led.get("arms") or {}).get(a, {}).get("rounds", [])) for a in ARMS},
                  "by_arm": spend_by_sub_arm(led),
                  "dryrun_likely_usd": (frozen.get("dryrun") or {}).get("likely_usd")},
    }


def write(root: Path = OUT, banned_path: str | os.PathLike | None = None) -> dict[str, Any]:
    s = build(root, banned_path)
    (root / SUMMARY_NAME).write_text(json.dumps(s, indent=1, sort_keys=True, default=str) + "\n", encoding="utf-8")
    return s


# ---------------------------------------------------------------------------
# Compact lines
# ---------------------------------------------------------------------------

def _p(x: float | None) -> str:
    return "n/a" if x is None else f"{100 * x:.1f}"


def _ci(c: list[float] | None) -> str:
    return "n/a" if not c else f"[{100 * c[0]:.1f}, {100 * c[1]:.1f}]"


def _d(x: dict[str, Any]) -> str:
    return f"{_p(x['diff'])} {_ci(x['ci95'])}"


def headline(s: dict[str, Any]) -> str:
    R, X = s["study_R"], s["study_X"]
    L = []
    for name in ("R1_note_framed", "R2_hedges_stripped", "R3_fragment_removed"):
        a = R[name]
        L.append(f"{name}: new decisions {a['n_new_decisions']}, emptied {a['n_emptied_not_sent']}; flip {_d(a['all']['flip_minus_v3'])}, "
                 f"deferral {_d(a['all']['defer_minus_v3'])}, commit {_d(a['all']['commit_minus_v3'])}: {a['verdict']}")
    one, two, three = X["1_split_on_eligible"], X["2_a_wrong"], X["3_two_sided"]
    L.append(f"X(1): flip v3 {_p(one['v3_flip_rate']['value'])}, X1 {_p(one['x1_flip_rate']['value'])}, X1p {_p(one['x1p_flip_rate']['value'])}, "
             f"X1r {_p(one['x1r_flip_rate']['value'])}; X1p-v3 {_d(one['x1p_minus_v3'])}, X1r-v3 {_d(one['x1r_minus_v3'])}, "
             f"X1p-X1r {_d(one['x1p_minus_x1r'])}: {one['verdict']}")
    L.append(f"X(2): rescue v3 {_p(two['rescue_v3_notes']['value'])}, X1 {_p(two['rescue_x1_notes']['value'])}, diff {_d(two['rescue_x1_minus_v3'])}; "
             f"repeat v3 {_p(two['repeat_v3_notes']['value'])}, X1 {_p(two['repeat_x1_notes']['value'])}, diff {_d(two['repeat_x1_minus_v3'])}")
    L.append(f"X(3): accuracy v3 notes {_p(three['accuracy_v3_notes']['value'])}, X1 notes {_p(three['accuracy_x1_notes']['value'])}, "
             f"diff {_d(three['x1_minus_v3'])}")
    sp = s["spend"]
    L.append(f"spend ${sp['total_usd']} of ${sp['cap_usd']}: "
             + ", ".join(f"{a} {v['requests']} requests ${v['actual_usd']}" for a, v in sp["by_arm"].items()))
    return "\n".join(L)

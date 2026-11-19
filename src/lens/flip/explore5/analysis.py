"""summary_v5.json (schema lens.flip.exploratory/1) for exploratory protocol v5. Offline; reads the arm dirs only.

Bootstrap: the v3 one (1,000 resamples of the 300 combined cases, seed 11; every handoff of a resampled case
comes along), so every difference against a v3 quantity is paired by case and handoff. Everything here is
exploratory: no result changes a v3 or v4 number.

Decomposition (exploratory), in percentage points of the 490 eligible handoffs:

    r_q = omission + wording + unattributed

    r_q          the v3 residue rate, flips / handoffs
    omission     sum over assessed v3 flips of (targeted restore - irrelevant restore) / handoffs, where targeted
                 restore is 1 when the v3 search restored the flip by inserting its minimal set and irrelevant
                 restore is the share of that flip's matched equal-length control sets that restored it (C1)
    wording      mean over handoffs with a rewrite of (rewrite flip - mean flip of the R identical repeats), (C2 net of C3)
    unattributed what is left: r_q - omission - wording. It holds the reader-noise floor and anything neither
                 control separates. It is reported, never dropped.
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any, Callable, Sequence

from ..explore4.analysis import Stats, _complete, _states
from ..headline.metrics import Boot, _features, auroc
from ..omission.coverage import salient_tokens
from . import SCHEMA, VERSION
from .engine import ARMS
from .frame import OUT, load_cases, load_frame, load_min_sets

SYSTEMS = ("recent", "rag-bm25", "rag-embed", "summary")


class NStats(Stats):
    """explore4's Stats plus numeric ratios and sums with paired bounds."""

    def ratio(self, num: Callable[[dict], float], den: Callable[[dict], float], rows: list[dict] | None = None) -> dict:
        rows = self.rows if rows is None else rows
        if not rows:
            return {"value": None, "ci95": None, "k": 0.0, "n": 0.0}
        f = _features(self.cases, rows, lambda r: {"k": num(r), "n": den(r)})
        s = self.boot.stat(f, lambda x: x["k"] / x["n"] if x["n"] else None)
        return {"value": s["value"], "ci95": s["ci95"], "k": s["sums"].get("k", 0.0), "n": s["sums"].get("n", 0.0)}

    def ratio_diff(self, a: tuple[Callable, Callable], b: tuple[Callable, Callable], rows: list[dict] | None = None) -> dict:
        """(sum a_num / sum a_den) - (sum b_num / sum b_den), paired."""
        rows = self.rows if rows is None else rows
        if not rows:
            return {"diff": None, "ci95": None, "bounds": None, "sums": {}}
        f = _features(self.cases, rows, lambda r: {"an": a[0](r), "ad": a[1](r), "bn": b[0](r), "bd": b[1](r)})
        s = self.boot.stat(f, lambda x: (x["an"] / x["ad"] - x["bn"] / x["bd"]) if x["ad"] and x["bd"] else None)
        lo, hi = (s["ci95"] or [None, None])
        sign = None if lo is None else ("above zero" if lo > 0 else "below zero" if hi < 0 else "includes zero")
        return {"diff": s["value"], "ci95": s["ci95"], "bounds": sign, "sums": s["sums"]}

    def auroc_vs_auroc(self, a: tuple[str, str], b: tuple[str, str], rows: list[dict] | None = None) -> dict:
        """AUROC(score a0 for flag a1) - AUROC(score b0 for flag b1), each on its own flips, paired by handoff."""
        rows = [r for r in (self.rows if rows is None else rows) if r.get(a[0]) is not None and r.get(b[0]) is not None]
        items = [(self.idx[r["case_id"]], (float(r[a[0]]), bool(r[a[1]]), float(r[b[0]]), bool(r[b[1]]))) for r in rows]

        def f(ws):
            x = auroc((((p, fl), w) for (p, fl, _, _), w in ws))
            y = auroc((((q, gl), w) for (_, _, q, gl), w in ws))
            return None if x is None or y is None else x - y
        s = self.boot.stat_items(items, f)
        lo, hi = (s["ci95"] or [None, None])
        return {"diff": s["value"], "ci95": s["ci95"], "n": len(rows),
                "bounds": None if lo is None else ("above zero" if lo > 0 else "below zero" if hi < 0 else "includes zero")}


# ---------------------------------------------------------------------------
# Decomposition (pure arithmetic on per-handoff rows)
# ---------------------------------------------------------------------------

def decomposition_row(r: dict) -> dict[str, float]:
    """One handoff's contribution to the decomposition sums. Per-handoff keys:
    v3_flip (0/1); c1_assessed (0/1: a flip with at least one matched control); c1_t (targeted restore, 0/1);
    c1_c (share of matched controls that restored); rw_has (0/1: a rewrite exists); rw_flip; rep_share (mean flip
    of the repeats). Missing keys count as zero."""
    a, h = r.get("c1_assessed", 0), r.get("rw_has", 0)
    return {"n": 1, "flips": r["v3_flip"], "assessed": a, "t": a * r.get("c1_t", 0), "c": a * r.get("c1_c", 0.0),
            "n_rw": h, "rw": h * r.get("rw_flip", 0), "rep": h * r.get("rep_share", 0.0)}


def decomposition_sums(rows: Sequence[dict]) -> dict[str, float]:
    z = {k: 0.0 for k in ("n", "flips", "assessed", "t", "c", "n_rw", "rw", "rep")}
    for r in rows:
        for k, v in decomposition_row(r).items():
            z[k] += v
    return z


def decompose(s: dict[str, float]) -> dict[str, float | None]:
    """r_q = omission + wording + unattributed, in rate units (multiply by 100 for points)."""
    if not s["n"]:
        return {"r_q": None, "omission": None, "wording": None, "unattributed": None}
    r_q = s["flips"] / s["n"]
    omission = (s["t"] - s["c"]) / s["n"]
    wording = (s["rw"] - s["rep"]) / s["n_rw"] if s["n_rw"] else None
    un = None if wording is None else r_q - omission - wording
    return {"r_q": r_q, "omission": omission, "wording": wording, "unattributed": un}


def decomposition_block(st: NStats, rows: list[dict]) -> dict[str, Any]:
    f = _features(st.cases, rows, decomposition_row)
    out: dict[str, Any] = {"unit": "share of the eligible handoffs", "exploratory": True,
                           "identity": "r_q = omission + wording + unattributed",
                           "omission_def": "sum over assessed v3 flips of (targeted restore - irrelevant restore) / handoffs (C1)",
                           "wording_def": "mean over handoffs with a rewrite of (rewrite flip - repeat flip share) (C2 net of C3)",
                           "n_assessed_flips": int(sum(f["assessed"])) if f else 0}
    for name in ("r_q", "omission", "wording", "unattributed"):
        s = st.boot.stat(f, lambda x, nm=name: decompose(x)[nm])
        out[name] = {"value": s["value"], "ci95": s["ci95"]}
    return out


# ---------------------------------------------------------------------------
# Rows
# ---------------------------------------------------------------------------

def _pair_stats(a: list[Any]) -> tuple[float, int]:
    """(pairwise disagreement share, pairs) among answers."""
    n = len(a)
    pairs = n * (n - 1) // 2
    if not pairs:
        return 0.0, 0
    return sum(a[i] != a[j] for i in range(n) for j in range(i + 1, n)) / pairs, pairs


def base_rows(root: Path) -> tuple[list, list[dict]]:
    cases = load_cases(root)
    rows: list[dict] = []
    for h in load_frame(root, cases):
        v3 = h.row["v3"]
        rows.append({"case_id": h.case.case_id, "system": h.system, "mechanism": h.case.mechanism, "a_A": h.row["a_A"],
                     "v3_flip": int(v3["residue_flip"]), "v3_floor": int(v3["a_floor"] != h.row["a_A"]),
                     "v3_recov": int(v3["search_status"] == "restored"), "v3_p": v3["p_residue"], "v3_a_B": v3["a_B"],
                     "v3_note": v3["note"], "_h": h})
    return cases, rows


def _by_system(st: NStats, flag: str, rows: list[dict] | None = None) -> dict[str, Any]:
    rows = st.rows if rows is None else rows
    return {s: st.rate(flag, [r for r in rows if r["system"] == s]) for s in SYSTEMS
            if any(r["system"] == s for r in rows)}


def study_x(st: NStats, rows: list[dict], key: str, arm: str, states: dict, with_search: bool) -> dict[str, Any]:
    for r in rows:
        o = states[(r["case_id"], r["system"])]["out"]
        r.update({f"{key}_flip": int(o["residue_flip"]), f"{key}_p": o["p_residue"], f"{key}_tokens": o["note_tokens"],
                  f"{key}_status": o["note_status"]})
        if with_search:
            rec = states[(r["case_id"], r["system"])].get("record")
            r.update({f"{key}_recov": int(bool(rec and rec["status"] == "restored")),
                      f"{key}_search": rec["status"] if rec else None})
    flag, p = f"{key}_flip", f"{key}_p"
    out: dict[str, Any] = {
        "arm": arm, "n": len(rows),
        "residue_rate": st.rate(flag), "v3_residue_rate": st.rate("v3_flip"),
        "residue_minus_v3": st.diff(flag, "v3_flip"),
        "h2c_auroc_p": st.auroc(p, flag), "h2c_auroc_p_v3": st.auroc("v3_p", "v3_flip"),
        "auroc_minus_v3_auroc": st.auroc_vs_auroc((p, flag), ("v3_p", "v3_flip")),
        "by_system": _by_system(st, flag),
        "notes": {"mean_tokens": sum(r[f"{key}_tokens"] for r in rows) / len(rows),
                  "status": dict(Counter(r[f"{key}_status"] for r in rows)),
                  "p_missing": sum(r[p] is None for r in rows)},
    }
    if with_search:
        out["recoverable_rate"] = st.rate(f"{key}_recov")
        out["v3_recoverable_rate"] = st.rate("v3_recov")
        out["recoverable_minus_v3_recoverable"] = st.diff(f"{key}_recov", "v3_recov")
        out["recoverable_minus_floor_v3"] = st.diff(f"{key}_recov", "v3_floor")
        out["search_outcomes"] = dict(Counter(r[f"{key}_search"] for r in rows if r[flag]))
    return out


def study_c1(st: NStats, rows: list[dict], states: dict, ctl_rows: list[dict], ms: dict) -> dict[str, Any]:
    by_flip: dict[tuple[str, str], list[dict]] = {}
    for c in ctl_rows:
        by_flip.setdefault((c["case_id"], c["system"]), []).append(c)
    flips = [r for r in rows if r["v3_flip"]]
    for r in flips:
        k = (r["case_id"], r["system"])
        got = {c["k"]: c for c in states[k]["out"]["controls"]}
        matched = [got[c["k"]] for c in by_flip[k] if c["send"] and c["matched"] and c["k"] in got]
        sent = list(got.values())
        t = int(ms[k]["status"] == "restored")
        r.update({"c1_t": t, "c1_assessed": int(bool(matched)),
                  "c1_c": (sum(c["restored"] for c in matched) / len(matched)) if matched else 0.0,
                  "c1_any": int(any(c["restored"] for c in matched)),
                  "c1_n_matched": len(matched), "c1_n_sent": len(sent),
                  "c1_c_all": (sum(c["restored"] for c in sent) / len(sent)) if sent else 0.0,
                  "c1_has_sent": int(bool(sent)), "c1_one": 1})
    ass = [r for r in flips if r["c1_assessed"]]
    sens = [r for r in flips if r["c1_has_sent"]]
    return {
        "arm": "C1", "n_v3_flips": len(flips), "n_assessed": len(ass),
        "controls": {"drawn": len(ctl_rows), "sent": sum(c["send"] for c in ctl_rows),
                     "matched": sum(c["matched"] for c in ctl_rows), "copies": sum(c["duplicate"] for c in ctl_rows),
                     "flips_without_matched_control": len(flips) - len(ass)},
        "targeted_restore_rate": st.ratio(lambda r: r["c1_t"], lambda r: 1, ass),
        "irrelevant_restore_rate": st.ratio(lambda r: r["c1_c"], lambda r: 1, ass),
        "targeted_minus_irrelevant": st.ratio_diff((lambda r: r["c1_t"], lambda r: 1), (lambda r: r["c1_c"], lambda r: 1), ass),
        "any_matched_control_restores": st.ratio(lambda r: r["c1_any"], lambda r: 1, ass),
        "sensitivity_all_sent_controls": {
            "n_flips": len(sens),
            "irrelevant_restore_rate": st.ratio(lambda r: r["c1_c_all"], lambda r: 1, sens),
            "targeted_restore_rate": st.ratio(lambda r: r["c1_t"], lambda r: 1, sens),
            "targeted_minus_irrelevant": st.ratio_diff((lambda r: r["c1_t"], lambda r: 1),
                                                       (lambda r: r["c1_c_all"], lambda r: 1), sens)},
        "by_v3_recoverability": {
            "recoverable": st.ratio(lambda r: r["c1_c"], lambda r: 1, [r for r in ass if r["c1_t"]]),
            "not_recoverable": st.ratio(lambda r: r["c1_c"], lambda r: 1, [r for r in ass if not r["c1_t"]])},
        "v3_restored_rate_all_flips": st.ratio(lambda r: r["v3_recov"], lambda r: 1, flips),
    }


def study_c2(st: NStats, rows: list[dict], states: dict) -> dict[str, Any]:
    for r in rows:
        o = states[(r["case_id"], r["system"])]["out"]
        ok = not o["failed"]
        r.update({"rw_has": int(ok), "rw_flip": int(bool(o["residue_flip"])) if ok else 0, "rw_one": 1,
                  "rw_identical": int(ok and o["identical"])})
        if ok:
            drop = salient_tokens(r["v3_note"]) - salient_tokens(o["rewrite"])
            r.update({"rw_lost_fact": int(bool(drop)), "rw_tokens": o["rewrite_tokens"]})
    ok_rows = [r for r in rows if r["rw_has"]]
    return {"arm": "C2", "n": len(rows), "n_rewrites": len(ok_rows), "n_failed": len(rows) - len(ok_rows),
            "rewrite_flip_rate": st.rate("rw_flip", ok_rows), "v3_residue_rate_same_rows": st.rate("v3_flip", ok_rows),
            "rewrite_minus_v3": st.diff("rw_flip", "v3_flip", ok_rows), "by_system": _by_system(st, "rw_flip", ok_rows),
            "rewrite_identical_to_note": sum(r["rw_identical"] for r in ok_rows),
            "rewrite_loses_a_salient_number_or_date": sum(r.get("rw_lost_fact", 0) for r in ok_rows),
            "mean_rewrite_tokens": sum(r["rw_tokens"] for r in ok_rows) / max(len(ok_rows), 1),
            "flip_transitions": {"v3_flip_to_no_flip": sum(r["v3_flip"] and not r["rw_flip"] for r in ok_rows),
                                 "no_flip_to_flip": sum((not r["v3_flip"]) and r["rw_flip"] for r in ok_rows)}}


def study_c3(st: NStats, rows: list[dict], states: dict) -> dict[str, Any]:
    reps = 0
    for r in rows:
        o = states[(r["case_id"], r["system"])]["out"]
        a = o["a_B"]
        reps = len(a)
        allv = [r["v3_a_B"]] + list(a)
        dis, _ = _pair_stats(allv)
        r.update({"rep_n": len(a), "rep_flips": sum(x != r["a_A"] for x in a), "rep_share": sum(x != r["a_A"] for x in a) / len(a),
                  "rep_dis_v3": sum(x != r["v3_a_B"] for x in a), "rep_pair_dis": dis, "rep_unan": int(len(set(allv)) == 1),
                  "rep_reverse_any": int(r["v3_flip"] and any(x == r["a_A"] for x in a)),
                  "rep_reverse_n": sum(x == r["a_A"] for x in a) if r["v3_flip"] else 0,
                  "rep_new_any": int((not r["v3_flip"]) and any(x != r["a_A"] for x in a)), "one": 1})
    flips = [r for r in rows if r["v3_flip"]]
    non = [r for r in rows if not r["v3_flip"]]
    return {"arm": "C3", "n": len(rows), "repeats_per_handoff": reps, "cache_bypass": True,
            "repeat_flip_rate": st.ratio(lambda r: r["rep_flips"], lambda r: r["rep_n"]),
            "v3_residue_rate": st.rate("v3_flip"),
            "repeat_minus_v3": st.ratio_diff((lambda r: r["rep_flips"], lambda r: r["rep_n"]),
                                             (lambda r: r["v3_flip"], lambda r: 1)),
            "self_disagreement_vs_v3_answer": st.ratio(lambda r: r["rep_dis_v3"], lambda r: r["rep_n"]),
            "pairwise_disagreement_six_answers": st.ratio(lambda r: r["rep_pair_dis"], lambda r: 1),
            "all_six_answers_identical": st.ratio(lambda r: r["rep_unan"], lambda r: 1),
            "v3_flips_reversed_by_at_least_one_repeat": st.ratio(lambda r: r["rep_reverse_any"], lambda r: 1, flips),
            "v3_flip_reversed_share_of_repeats": st.ratio(lambda r: r["rep_reverse_n"], lambda r: r["rep_n"], flips),
            "v3_nonflips_flipped_by_at_least_one_repeat": st.ratio(lambda r: r["rep_new_any"], lambda r: 1, non),
            "by_system": {s: st.ratio(lambda r: r["rep_flips"], lambda r: r["rep_n"], [r for r in rows if r["system"] == s])
                          for s in SYSTEMS if any(r["system"] == s for r in rows)}}


def study_c4(st: NStats, rows: list[dict], states: dict) -> dict[str, Any]:
    n_pairs = n_bad = 0
    for r in rows:
        h = r["_h"]
        o = states[(r["case_id"], r["system"])]["out"]
        conv = o["conveyed"]
        ok = [c for c in conv if c is not None]
        n_pairs += len(conv)
        n_bad += len(conv) - len(ok)
        ds = set(h.case.d_star)
        dflags = [bool(h.view.event_ids) and len(h.view.event_ids) == len(conv) and h.view.event_ids[j] in ds
                  for j in range(len(conv))]
        dj = [c for c, d in zip(conv, dflags) if d and c is not None]
        r.update({"sem_om": (sum(not c for c in ok) / len(ok)) if ok else None,
                  "sem_n_judged": len(ok), "sem_d_n": len(dj), "sem_d_conv": sum(bool(c) for c in dj),
                  "sem_has_d": int(bool(dj))})
    flips = [r for r in rows if r["v3_flip"] and r["sem_has_d"]]
    non = [r for r in rows if (not r["v3_flip"]) and r["sem_has_d"]]
    return {"arm": "C4", "n_pairs": n_pairs, "n_unparsed": n_bad,
            "mean_semantic_omission": st.ratio(lambda r: r["sem_om"] or 0.0, lambda r: 1 if r["sem_om"] is not None else 0),
            "h2c_auroc_semantic_omission": st.auroc("sem_om", "v3_flip"),
            "h2c_auroc_p_v3": st.auroc("v3_p", "v3_flip"),
            "p_minus_semantic_omission": st.auroc_diff("v3_p", "sem_om", "v3_flip"),
            "dstar_semantic_recall_on_flips": {"n_flips": len(flips),
                                               **st.ratio(lambda r: r["sem_d_conv"], lambda r: r["sem_d_n"], flips)},
            "dstar_semantic_recall_on_nonflips": {"n": len(non),
                                                  **st.ratio(lambda r: r["sem_d_conv"], lambda r: r["sem_d_n"], non)},
            "note": "event views only (D* is a set of event ids); M0 cases have no D*"}


# ---------------------------------------------------------------------------
# Build and write
# ---------------------------------------------------------------------------

def build(root: Path = OUT) -> dict[str, Any]:
    cases, rows = base_rows(root)
    n = len(rows)
    st = NStats(cases, rows)
    frozen = json.loads((root / "protocol_v5.json").read_text()) if (root / "protocol_v5.json").exists() else {}
    led = json.loads((root / "spend.json").read_text()) if (root / "spend.json").exists() else {}
    out: dict[str, Any] = {
        "schema": SCHEMA, "version": VERSION, "exploratory": True,
        "note": "Exploratory, outside the frozen v3 and v4 protocols. No result here changes a v3 or v4 verdict.",
        "protocol": {"doc": "docs/flip-protocol-v5-exploratory.md", "status": frozen.get("status"),
                     "frozen_at": frozen.get("frozen_at"), "prompt_sha256": frozen.get("prompts"),
                     "arm_config_sha256": frozen.get("arm_configs"), "frame_sha256": frozen.get("frame_sha256"),
                     "cases_sha256": frozen.get("cases_sha256")},
        "denominators": {"cases": len(cases), "eligible_handoffs": n, "by_system": dict(Counter(r["system"] for r in rows)),
                         "v3_residue_flips": sum(r["v3_flip"] for r in rows)},
        "bootstrap": {"reps": 1000, "seed": 11, "unit": "case (paired; every handoff of a resampled case)"},
        "v3_reference": {"residue_rate": st.rate("v3_flip"), "floor_rate": st.rate("v3_floor"),
                         "recoverable_rate": st.rate("v3_recov"), "h2c_auroc": st.auroc("v3_p", "v3_flip")},
        "spend": {k: led.get(k) for k in ("cap_usd", "total_usd", "remaining_usd")} | {
            "by_arm": {a: v.get("actual_usd") for a, v in led.get("arms", {}).items()}},
        "skipped": [],
    }
    ms = load_min_sets(root) if (root / "v3_min_sets.jsonl").exists() else {}
    ctl = [json.loads(x) for x in (root / "c1_controls.jsonl").read_text().splitlines() if x.strip()] \
        if (root / "c1_controls.jsonl").exists() else []
    done: dict[str, dict] = {}
    for arm in ARMS:
        s = _states(root, arm)
        if _complete(s, n):
            done[arm] = s
        else:
            out["skipped"].append({"arm": arm, "reason": "not run or not complete"})
    if "X1" in done:
        out["X1_reader_conditioned"] = study_x(st, rows, "x1", "X1", done["X1"], False)
    if "X2" in done:
        out["X2_structured"] = study_x(st, rows, "x2", "X2", done["X2"], True)
        s2 = done["X2"]
        out["X2_structured"]["structure"] = {
            "notes_cut_at_150": sum(bool(s2[(r["case_id"], r["system"])]["out"]["cut"]) for r in rows),
            "replies_with_all_five_fields": sum(bool(s2[(r["case_id"], r["system"])]["out"]["complete"]) for r in rows)}
    if "C1" in done:
        out["C1_irrelevant_insertion"] = study_c1(st, rows, done["C1"], ctl, ms)
    if "C2" in done:
        out["C2_rewrite"] = study_c2(st, rows, done["C2"])
    if "C3" in done:
        out["C3_repeats"] = study_c3(st, rows, done["C3"])
    if "C4" in done:
        out["C4_semantic_absence"] = study_c4(st, rows, done["C4"])
    if {"C1", "C2", "C3"} <= set(done):
        out["decomposition"] = decomposition_block(st, rows)
        if "C4" in done:
            out["decomposition"]["c4_check"] = {
                "dstar_semantic_recall_on_flips": out["C4_semantic_absence"]["dstar_semantic_recall_on_flips"],
                "note": "C4 reads the omission term: low semantic recall of D* on flips supports omission; high recall "
                        "means the note conveyed D* and the flip is not omission in the semantic sense."}
    return out


def write(root: Path = OUT) -> dict[str, Any]:
    s = build(root)
    (root / "summary_v5.json").write_text(json.dumps(s, indent=1, sort_keys=True, default=str) + "\n", encoding="utf-8")
    return s

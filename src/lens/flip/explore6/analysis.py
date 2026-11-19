"""summary_v6.json and the text-free metric file for exploratory protocol v6. Offline.

Two steps. `metric_rows` reads the finished arm dirs and writes one flat row per eligible handoff: ids, system,
mechanism group, counts and 0/1 flags. No note, unit or prompt text. `summarize` computes every v6 number from
those rows alone, so the metric file is sufficient to rebuild the summary (`rebuild` does exactly that).

Bootstrap: the v3 one. 1,000 resamples of the 300 cases, seed 11; every handoff of a resampled case comes along.
A row carries its case's position in cases.jsonl (`case_index`) and the case count, which is all the bootstrap
needs. Every difference is paired by case and handoff.

The definitions below are frozen in protocol_v6.json (item x6.analysis) before any model call.
"""

from __future__ import annotations

import json
import os
from collections import Counter
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable, Sequence

from ..explore4.analysis import _complete, _states
from ..explore5.analysis import NStats
from ..headline.metrics import _features
from . import SCHEMA, VERSION
from . import frame as F
from .engine import ARMS
from .frame import OUT, RESULTS, load_cases, load_frame

SYSTEMS = ("recent", "rag-bm25", "rag-embed", "summary")
METRICS_NAME = "handoff_metrics_v6.jsonl"
SUMMARY_NAME = "summary_v6.json"
MAX_STR = 32
FLOOR_RULE = "supported only if the lower 95% bound of (content-specific recoverable rate minus the v3 rewording floor) is above zero; otherwise not supported"

# The pre-registered analysis, as data. Its canonical JSON is hashed into protocol_v6.json (x6.analysis).
ANALYSIS_SPEC: dict[str, Any] = {
    "unit": "one extractive flip of v4 (193); denominators of rates over handoffs are the 490 eligible handoffs",
    "restored": "B's action equals A's action; a reply that is refused, cut, unparsed or failed twice is not restored",
    "targeted_fresh": "share of the 5 fresh re-reads of the minimal restoring set that are restored; 0 for a flip the search did not restore",
    "targeted_v5": "1 if the search restored the flip, else 0",
    "irrelevant": "share of the flip's matched and sent controls that are restored",
    "assessed": "the flip has at least one matched and sent control",
    "bootstrap": {"reps": 1000, "seed": 11, "unit": "case", "paired": True},
    "primary_1": "mean over assessed extractive flips of (targeted_fresh - irrelevant), paired by flip",
    "primary_2": {"rate": "sum over assessed extractive flips of (targeted_fresh - irrelevant) / 490",
                  "difference": "rate minus the v3 rewording floor (handoffs whose v3 floor action differs from A's action, 51 of 490), paired as in explore4 (difference of sums over the same handoffs, by case)",
                  "rule": FLOOR_RULE},
    "sensitivity": ["every sent control instead of matched controls only (assessed = at least one sent control)",
                    "upper bracket: flips without a matched control count with their full targeted_fresh share"],
    "secondary": ["primary_1 and primary_2 with targeted_v5",
                  "irrelevant restore rate split by the search outcome (restored, not restored)",
                  "share of assessed flips where at least one matched control restores",
                  "fresh re-read rate of the minimal restoring sets, E1 and P1",
                  "the v5 omission term (v5 C1 controls, v3 flips assessed in v5) with targeted_fresh from P1, beside the v5 definition",
                  "per-system rows, descriptive"],
    "multiple_comparisons": "no correction",
}


def group(m: str) -> str:
    """The mechanism group of export_metrics.py."""
    return "M0" if m == "M0" else "M1-M3" if m in ("M1", "M2", "M3") else "U1-U2"


# ---------------------------------------------------------------------------
# Step 1: the text-free rows
# ---------------------------------------------------------------------------

def _count_fresh(out: dict | None) -> tuple[int, int, int]:
    fresh = (out or {}).get("fresh") or []
    return len(fresh), sum(bool(x["restored"]) for x in fresh), sum(x["status"] != "ok" for x in fresh)


def _v5_c1(v5: Path | None) -> tuple[dict | None, dict]:
    """(states of the frozen v5 arm C1, its control draws by flip), or (None, {}) when v5 is not there."""
    if v5 is None or not (v5 / "c1_controls.jsonl").exists():
        return None, {}
    st = _states(v5, "C1")
    return (st if st else None), F.load_controls(v5 / "c1_controls.jsonl")


def metric_rows(root: Path = OUT, v5: Path | None = None) -> list[dict[str, Any]]:
    cases = load_cases(root)
    idx = {c.case_id: i for i, c in enumerate(cases)}
    hs = load_frame(root, cases)
    n = len(hs)
    s_e1, s_p1 = _states(root, "E1"), _states(root, "P1")
    if not _complete(s_e1, n) or not _complete(s_p1, n):
        raise SystemExit("E1 or P1 is not complete; run both arms before the report")
    e1, p1 = F.load_e1_flips(root), F.load_p1_sets(root)
    ctl = F.load_controls(root / F.E1_CONTROLS)
    s_c1, ctl5 = _v5_c1(v5)
    rows = []
    for h in hs:
        k = (h.case.case_id, h.system)
        v3 = h.row["v3"]
        r: dict[str, Any] = {
            "case_id": h.case.case_id, "case_index": idx[h.case.case_id], "n_cases": len(cases), "system": h.system,
            "mechanism": h.case.mechanism, "mechanism_group": group(h.case.mechanism),
            "v3_floor_flip": int(v3["a_floor"] != h.row["a_A"]), "v3_flip": int(bool(v3["residue_flip"])),
            "v3_search_restored": int(v3["search_status"] == "restored"),
        }
        rec = e1.get(k)
        o = s_e1[k]["out"]
        draws = ctl.get(k, [])
        got = {c["k"]: c for c in o.get("controls", [])}
        sent = [got[c["k"]] for c in draws if c["send"] and c["k"] in got]
        matched = [got[c["k"]] for c in draws if c["send"] and c["matched"] and c["k"] in got]
        fn, fk, fu = _count_fresh(o)
        r.update({
            "e1_flip": int(rec is not None),
            "e1_search_restored": int(rec is not None and rec["status"] == "restored"),
            "e1_search_verified": int(rec is not None and rec["status"] == "restored" and bool(rec["verified"])),
            "e1_min_units": len(rec["unit_ids"]) if rec and rec["status"] == "restored" else None,
            "e1_min_chars": rec["chars"] if rec else None,
            "e1_target_chars": draws[0]["target_chars"] if draws else None,
            "e1_pool_size": draws[0]["pool_size"] if draws else None,
            "e1_ctl_drawn": len(draws), "e1_ctl_sent": len(sent), "e1_ctl_sent_restored": sum(bool(c["restored"]) for c in sent),
            "e1_ctl_matched": len(matched), "e1_ctl_matched_restored": sum(bool(c["restored"]) for c in matched),
            "e1_ctl_unparsed": sum(c.get("status", "ok") != "ok" for c in sent),
            "e1_fresh_n": fn, "e1_fresh_restored": fk, "e1_fresh_unparsed": fu,
        })
        fn, fk, fu = _count_fresh(s_p1[k]["out"])
        r.update({"p1_fresh_n": fn, "p1_fresh_restored": fk, "p1_fresh_unparsed": fu})
        c5 = {"c1v5_sent": None, "c1v5_sent_restored": None, "c1v5_matched": None, "c1v5_matched_restored": None}
        if s_c1 is not None:
            got5 = {c["k"]: c for c in (s_c1.get(k, {}).get("out") or {}).get("controls", [])}
            d5 = ctl5.get(k, [])
            sent5 = [got5[c["k"]] for c in d5 if c["send"] and c["k"] in got5]
            m5 = [got5[c["k"]] for c in d5 if c["send"] and c["matched"] and c["k"] in got5]
            c5 = {"c1v5_sent": len(sent5), "c1v5_sent_restored": sum(bool(c["restored"]) for c in sent5),
                  "c1v5_matched": len(m5), "c1v5_matched_restored": sum(bool(c["restored"]) for c in m5)}
        r.update(c5)
        rows.append(r)
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

def _share(k: int | None, n: int | None) -> float:
    return (k / n) if n else 0.0


def derive(r: dict[str, Any]) -> dict[str, Any]:
    """The per-handoff quantities of ANALYSIS_SPEC. Everything is zero on a handoff that is not a flip."""
    f = int(r["e1_flip"])
    ass = int(bool(f and r["e1_ctl_matched"]))
    has_sent = int(bool(f and r["e1_ctl_sent"]))
    t_fresh = _share(r["e1_fresh_restored"], r["e1_fresh_n"]) if f and r["e1_search_restored"] else 0.0
    v3f = int(r["v3_flip"])
    p_ass = int(bool(v3f and r.get("c1v5_matched")))
    return {**r, "one": 1,
            "t_fresh": t_fresh, "t_v5": int(bool(f and r["e1_search_restored"])),
            "assessed": ass, "c": _share(r["e1_ctl_matched_restored"], r["e1_ctl_matched"]) if ass else 0.0,
            "has_sent": has_sent, "c_all": _share(r["e1_ctl_sent_restored"], r["e1_ctl_sent"]) if has_sent else 0.0,
            "any": int(bool(ass and r["e1_ctl_matched_restored"])),
            "p_t_fresh": _share(r["p1_fresh_restored"], r["p1_fresh_n"]) if v3f and r["v3_search_restored"] else 0.0,
            "p_t_v5": int(bool(v3f and r["v3_search_restored"])),
            "p_assessed": p_ass, "p_c": _share(r.get("c1v5_matched_restored"), r.get("c1v5_matched")) if p_ass else 0.0}


def _stats(rows: list[dict[str, Any]]) -> NStats:
    n_cases = int(rows[0]["n_cases"])
    ids = {int(r["case_index"]): r["case_id"] for r in rows}
    cases = [SimpleNamespace(case_id=ids.get(i, f"_no_eligible_handoff_{i}")) for i in range(n_cases)]
    return NStats(cases, rows)


def _sign(ci: list[float] | None) -> str | None:
    if not ci:
        return None
    return "above zero" if ci[0] > 0 else "below zero" if ci[1] < 0 else "includes zero"


def _mean_pair(st: NStats, rows: list[dict], t: str, c: str) -> dict[str, Any]:
    """Targeted rate, irrelevant rate and their paired difference over `rows` (the assessed flips)."""
    return {"n_flips": len(rows),
            "targeted_restore_rate": st.ratio(lambda r: r[t], lambda r: 1, rows),
            "irrelevant_restore_rate": st.ratio(lambda r: r[c], lambda r: 1, rows),
            "targeted_minus_irrelevant": st.ratio_diff((lambda r: r[t], lambda r: 1), (lambda r: r[c], lambda r: 1), rows)}


def _over_handoffs(st: NStats, rows: list[dict], term: Callable[[dict], float], floor: str | None = "v3_floor_flip") -> dict[str, Any]:
    """sum(term) / handoffs with its 95% bounds, and (when `floor` is given) its paired difference from the floor."""
    f = _features(st.cases, rows, lambda r: {"s": term(r), "floor": r[floor] if floor else 0, "n": 1})
    rate = st.boot.stat(f, lambda x: x["s"] / x["n"] if x["n"] else None)
    out: dict[str, Any] = {"rate": {"value": rate["value"], "ci95": rate["ci95"], "sum": rate["sums"].get("s"),
                                    "n": int(rate["sums"].get("n", 0))}}
    if floor:
        d = st.boot.stat(f, lambda x: (x["s"] - x["floor"]) / x["n"] if x["n"] else None)
        out["floor_rate_v3"] = st.rate(floor, rows)
        out["minus_floor"] = {"diff": d["value"], "ci95": d["ci95"], "bounds": _sign(d["ci95"])}
    return out


def verdict(minus_floor: dict[str, Any]) -> str:
    ci = minus_floor.get("ci95")
    return "supported" if ci and ci[0] > 0 else "not supported"


def _reread(st: NStats, rows: list[dict], n_key: str, k_key: str) -> dict[str, Any]:
    """How often a minimal restoring set restores again. `rows` are the flips the search restored."""
    dist = Counter(int(r[k_key]) for r in rows)
    reps = max((int(r[n_key]) for r in rows), default=0)
    return {"n_sets": len(rows), "rereads_per_set": reps,
            "rate": st.ratio(lambda r: r[k_key], lambda r: r[n_key], rows),
            "sets_by_restores": {str(i): dist.get(i, 0) for i in range(reps + 1)},
            "all_rereads_restore": st.ratio(lambda r: int(r[n_key] > 0 and r[k_key] == r[n_key]), lambda r: 1, rows),
            "majority_restores": st.ratio(lambda r: int(2 * r[k_key] > r[n_key]), lambda r: 1, rows),
            "at_least_one_restores": st.ratio(lambda r: int(r[k_key] > 0), lambda r: 1, rows),
            "no_reread_restores": st.ratio(lambda r: int(r[n_key] > 0 and r[k_key] == 0), lambda r: 1, rows)}


def summarize(raw_rows: list[dict[str, Any]]) -> dict[str, Any]:
    rows = [derive(r) for r in raw_rows]
    st = _stats(rows)
    n = len(rows)
    flips = [r for r in rows if r["e1_flip"]]
    ass = [r for r in flips if r["assessed"]]
    sens = [r for r in flips if r["has_sent"]]
    restored = [r for r in flips if r["e1_search_restored"]]

    p1 = _mean_pair(st, ass, "t_fresh", "c")
    p2 = _over_handoffs(st, rows, lambda r: r["assessed"] * (r["t_fresh"] - r["c"]))
    p2.update({"rule": FLOOR_RULE, "verdict": verdict(p2["minus_floor"]),
               "earlier_claim": "survives its own falsification test (v4: recoverable minus floor)"})
    s_all = {"specificity": _mean_pair(st, sens, "t_fresh", "c_all"),
             "content_specific": _over_handoffs(st, rows, lambda r: r["has_sent"] * (r["t_fresh"] - r["c_all"]))}
    s_all["content_specific"]["verdict_under_rule"] = verdict(s_all["content_specific"]["minus_floor"])
    s_up = {"content_specific": _over_handoffs(st, rows, lambda r: r["e1_flip"] * (r["t_fresh"] - r["assessed"] * r["c"])),
            "n_flips_without_matched_control": len(flips) - len(ass)}
    s_up["content_specific"]["verdict_under_rule"] = verdict(s_up["content_specific"]["minus_floor"])

    v5def = {"specificity": _mean_pair(st, ass, "t_v5", "c"),
             "content_specific": _over_handoffs(st, rows, lambda r: r["assessed"] * (r["t_v5"] - r["c"]))}
    v5def["content_specific"]["verdict_under_rule"] = verdict(v5def["content_specific"]["minus_floor"])

    def sys_row(s: str) -> dict[str, Any]:
        rs = [r for r in rows if r["system"] == s]
        a = [r for r in rs if r["assessed"]]
        return {"n_handoffs": len(rs), "n_flips": sum(r["e1_flip"] for r in rs),
                "n_search_restored": sum(r["t_v5"] for r in rs), "n_assessed": len(a),
                "specificity": _mean_pair(st, a, "t_fresh", "c"),
                "content_specific": _over_handoffs(st, rs, lambda r: r["assessed"] * (r["t_fresh"] - r["c"])),
                "fresh_reread_rate": st.ratio(lambda r: r["e1_fresh_restored"], lambda r: r["e1_fresh_n"],
                                              [r for r in rs if r["e1_search_restored"]])}

    e1: dict[str, Any] = {
        "arm": "E1", "n_handoffs": n, "n_flips": len(flips), "n_search_restored": len(restored),
        "n_assessed": len(ass), "n_assessed_search_restored": sum(r["t_v5"] for r in ass),
        "controls": {"drawn": sum(r["e1_ctl_drawn"] for r in flips), "sent": sum(r["e1_ctl_sent"] for r in flips),
                     "matched": sum(r["e1_ctl_matched"] for r in flips),
                     "flips_without_matched_control": len(flips) - len(ass),
                     "flips_without_sent_control": len(flips) - len(sens),
                     "unparsed_replies": sum(r["e1_ctl_unparsed"] for r in flips)},
        "rereads": {"sent": sum(r["e1_fresh_n"] for r in flips), "unparsed_replies": sum(r["e1_fresh_unparsed"] for r in flips)},
        "primary_1_specificity": p1,
        "primary_2_content_specific": p2,
        "sensitivity_all_sent_controls": s_all,
        "sensitivity_upper_bracket": s_up,
        "secondary_v5_definition_of_targeted": v5def,
        "irrelevant_by_search_outcome": {
            "restored": st.ratio(lambda r: r["c"], lambda r: 1, [r for r in ass if r["t_v5"]]),
            "not_restored": st.ratio(lambda r: r["c"], lambda r: 1, [r for r in ass if not r["t_v5"]])},
        "any_matched_control_restores": st.ratio(lambda r: r["any"], lambda r: 1, ass),
        "fresh_reread": {**_reread(st, restored, "e1_fresh_n", "e1_fresh_restored"),
                         "by_v2_verified": {
                             "verified": st.ratio(lambda r: r["e1_fresh_restored"], lambda r: r["e1_fresh_n"],
                                                  [r for r in restored if r["e1_search_verified"]]),
                             "not_verified": st.ratio(lambda r: r["e1_fresh_restored"], lambda r: r["e1_fresh_n"],
                                                      [r for r in restored if not r["e1_search_verified"]])}},
        "by_system_descriptive": {s: sys_row(s) for s in SYSTEMS if any(r["system"] == s for r in rows)},
        "v4_reference": {"recoverable_rate": st.rate("t_v5"), "floor_rate_v3": st.rate("v3_floor_flip"),
                         "recoverable_minus_floor": st.diff("t_v5", "v3_floor_flip"),
                         "note": "the v4 claim, recomputed from these rows with no specificity control"},
    }

    v3flips = [r for r in rows if r["v3_flip"]]
    p_rest = [r for r in v3flips if r["v3_search_restored"]]
    pb: dict[str, Any] = {"arm": "P1", "n_v3_flips": len(v3flips), "n_search_restored": len(p_rest),
                          "rereads": {"sent": sum(r["p1_fresh_n"] for r in rows),
                                      "unparsed_replies": sum(r["p1_fresh_unparsed"] for r in rows)},
                          "fresh_reread": _reread(st, p_rest, "p1_fresh_n", "p1_fresh_restored")}
    if any(r.get("c1v5_matched") is not None for r in rows):
        p_ass = [r for r in v3flips if r["p_assessed"]]
        pb["v5_c1_recomputed"] = {
            "n_assessed": len(p_ass), "n_assessed_search_restored": sum(r["p_t_v5"] for r in p_ass),
            "v5_definition": _mean_pair(st, p_ass, "p_t_v5", "p_c"),
            "fresh_targeted": _mean_pair(st, p_ass, "p_t_fresh", "p_c"),
            "omission_term_v5_definition": _over_handoffs(st, rows, lambda r: r["p_assessed"] * (r["p_t_v5"] - r["p_c"]), None)["rate"],
            "omission_term_fresh_targeted": _over_handoffs(st, rows, lambda r: r["p_assessed"] * (r["p_t_fresh"] - r["p_c"]), None)["rate"],
            "note": "controls are the frozen v5 C1 results; only the targeted side changes"}
    else:
        pb["v5_c1_recomputed"] = None
    return {"denominators": {"cases": int(rows[0]["n_cases"]), "eligible_handoffs": n,
                             "by_system": dict(Counter(r["system"] for r in rows)),
                             "extractive_flips": len(flips), "v3_flips": len(v3flips),
                             "v3_floor_flips": sum(r["v3_floor_flip"] for r in rows)},
            "bootstrap": {"reps": 1000, "seed": 11, "unit": "case (paired; every handoff of a resampled case)"},
            "multiple_comparisons": "no correction",
            "E1_extractive": e1, "P1_paraphrased": pb}


def rebuild(path: Path) -> dict[str, Any]:
    """Every v6 number from the metric file alone."""
    return summarize(read_metrics(path))


# ---------------------------------------------------------------------------
# Build and write
# ---------------------------------------------------------------------------

def _requests(sp: dict[str, Any]) -> int:
    return int(sum(v["requests"] for v in (sp.get("by_stage") or {}).values()))


def build(root: Path = OUT, v5: Path | None = None, banned_path: str | os.PathLike | None = None) -> dict[str, Any]:
    v5 = (RESULTS / "exploratory-v5") if v5 is None else v5
    mpath = write_metrics(root, metric_rows(root, v5), banned_path)
    body = rebuild(mpath)
    frozen = json.loads((root / "protocol_v6.json").read_text()) if (root / "protocol_v6.json").exists() else {}
    led = json.loads((root / "spend.json").read_text()) if (root / "spend.json").exists() else {}
    arms = led.get("arms", {})
    v5s = v5 / "summary_v5.json"
    ref = None
    if v5s.exists():
        dec = json.loads(v5s.read_text()).get("decomposition") or {}
        ref = {"omission_term_reported_by_v5": dec.get("omission"), "source": "results/exploratory-v5/summary_v5.json"}
    return {
        "schema": SCHEMA, "version": VERSION, "exploratory": True,
        "note": "Exploratory, outside the frozen v3, v4 and v5 protocols. No result here changes a v3, v4 or v5 number.",
        "protocol": {"doc": "docs/flip-protocol-v6-exploratory.md", "status": frozen.get("status"),
                     "frozen_at": frozen.get("frozen_at"), "finalized_at": frozen.get("finalized_at"),
                     "rules_sha256": frozen.get("rules"), "arm_config_sha256": frozen.get("arm_configs"),
                     "reused_v3_sha256": frozen.get("reused_v3"),
                     "files_sha256": {k: v for k, v in frozen.items() if k.endswith("_sha256")}},
        **body,
        "v5_reference": ref,
        "metrics_file": {"name": METRICS_NAME, "rows": len(read_metrics(mpath)), "sha256": F.sha256_file(mpath),
                         "text_free": True, "sufficient": "summarize(rows) gives every number of this file"},
        "spend": {"cap_usd": led.get("cap_usd"), "total_usd": led.get("total_usd"), "remaining_usd": led.get("remaining_usd"),
                  "by_arm": {a: {"actual_usd": arms[a].get("actual_usd"), "requests": _requests(arms[a]),
                                 "by_stage": arms[a].get("by_stage")} for a in ARMS if a in arms},
                  "dryrun_likely_usd": (frozen.get("dryrun") or {}).get("likely_usd")},
    }


def write(root: Path = OUT, v5: Path | None = None, banned_path: str | os.PathLike | None = None) -> dict[str, Any]:
    s = build(root, v5, banned_path)
    (root / SUMMARY_NAME).write_text(json.dumps(s, indent=1, sort_keys=True, default=str) + "\n", encoding="utf-8")
    return s


# ---------------------------------------------------------------------------
# A compact table of the pre-registered numbers
# ---------------------------------------------------------------------------

def _p(x: float | None) -> str:
    return "n/a" if x is None else f"{100 * x:.1f}"


def _ci(c: list[float] | None) -> str:
    return "n/a" if not c else f"[{100 * c[0]:.1f}, {100 * c[1]:.1f}]"


def headline(s: dict[str, Any]) -> str:
    e, p = s["E1_extractive"], s["P1_paraphrased"]
    p1, p2 = e["primary_1_specificity"], e["primary_2_content_specific"]
    d = p1["targeted_minus_irrelevant"]
    v5d = e["secondary_v5_definition_of_targeted"]
    L = [f"flips {e['n_flips']}, assessed {e['n_assessed']}; controls drawn {e['controls']['drawn']}, sent "
         f"{e['controls']['sent']}, matched {e['controls']['matched']}; re-reads E1 {e['rereads']['sent']}, P1 {p['rereads']['sent']}",
         f"primary 1: targeted {_p(p1['targeted_restore_rate']['value'])}, irrelevant {_p(p1['irrelevant_restore_rate']['value'])}, "
         f"difference {_p(d['diff'])} {_ci(d['ci95'])}",
         f"primary 2: content-specific rate {_p(p2['rate']['value'])} {_ci(p2['rate']['ci95'])}; minus floor "
         f"{_p(p2['minus_floor']['diff'])} {_ci(p2['minus_floor']['ci95'])}: {p2['verdict']}",
         f"v5 definition: difference {_p(v5d['specificity']['targeted_minus_irrelevant']['diff'])} "
         f"{_ci(v5d['specificity']['targeted_minus_irrelevant']['ci95'])}; rate {_p(v5d['content_specific']['rate']['value'])} "
         f"{_ci(v5d['content_specific']['rate']['ci95'])}; minus floor {_p(v5d['content_specific']['minus_floor']['diff'])} "
         f"{_ci(v5d['content_specific']['minus_floor']['ci95'])}",
         f"fresh re-read rate: E1 {_p(e['fresh_reread']['rate']['value'])} {_ci(e['fresh_reread']['rate']['ci95'])}, "
         f"P1 {_p(p['fresh_reread']['rate']['value'])} {_ci(p['fresh_reread']['rate']['ci95'])}"]
    c = p.get("v5_c1_recomputed")
    if c:
        L.append(f"omission term: v5 definition {_p(c['omission_term_v5_definition']['value'])} "
                 f"{_ci(c['omission_term_v5_definition']['ci95'])}; fresh targeted {_p(c['omission_term_fresh_targeted']['value'])} "
                 f"{_ci(c['omission_term_fresh_targeted']['ci95'])}")
    sp = s["spend"]
    L.append(f"spend ${sp['total_usd']} of ${sp['cap_usd']}: "
             + ", ".join(f"{a} {v['requests']} requests ${v['actual_usd']}" for a, v in sp["by_arm"].items()))
    return "\n".join(L)

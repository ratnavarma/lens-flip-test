"""summary.json for the headline run: section 9 metrics per cell and the section 7.2 residue block.

Every rate carries a 95% paired bootstrap CI over cases: one resample of case indices is shared
by every metric, and a case brings all its cells and systems along (paired, clustered by case).
A metric is a function of sums of per-case features, so one resample costs one weighted sum.

Gated metrics count only cases where a_H == a_star. The other cases are excluded and counted.

Protocol v2 (Varma, 2026-10-20) differs from v1 in two places:

- Power rule per hypothesis. H2a needs its own eligible cases, H2b its own eligible residue flips,
  H2c residue flips and non-flips among the eligible cases. Below its threshold a hypothesis is
  reported as descriptive, with bounds and no label. v1 pooled one count of residue flips over H2b and H2c.
- The harmful gated rate is split into wrong commit, abstention and parse error. The three sum to it.

`summarise(..., protocol=1)` keeps the v1 behaviour.
"""

from __future__ import annotations

import json
import random
from collections import Counter
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence

from ..decider import PARSE_ERROR
from ..metrics import gate_report, is_committed
from ..schema import MECHANISMS, REQUEST_INFO, Case, read_jsonl
from ..systems.base import count_tokens
from . import provenance
from .tasks import EVENT_SYSTEMS, cell_name, rebuild_view

RARE_SHARE_REPORT = 0.01

# Protocol v2 power rule (Varma, 2026-10-20). A run config may override these under "power_rule".
POWER_RULE_V2 = {"H2a_min_cases": 20, "H2b_min_flips": 20, "H2c_min_positives": 20, "H2c_min_negatives": 20}


def power_thresholds(cfg: dict[str, Any]) -> dict[str, int]:
    return {**POWER_RULE_V2, **{k: int(v) for k, v in (cfg.get("power_rule") or {}).items()}}


# ---------------------------------------------------------------------------
# Bootstrap
# ---------------------------------------------------------------------------

class Boot:
    def __init__(self, n: int, reps: int = 1000, seed: int = 0):
        self.n = n
        rng = random.Random(f"boot|{seed}")
        self.samples: list[list[tuple[int, int]]] = []
        for _ in range(reps if n else 0):
            c = Counter(rng.randrange(n) for _ in range(n))
            self.samples.append(sorted(c.items()))

    def stat(self, feats: dict[str, Sequence[float]], f: Callable[[dict[str, float]], float | None]) -> dict:
        full = {k: float(sum(v)) for k, v in feats.items()}
        vals = []
        for s in self.samples:
            sums = {k: float(sum(v[i] * w for i, w in s)) for k, v in feats.items()}
            x = f(sums)
            if x is not None:
                vals.append(x)
        return {"value": f(full), "ci95": ci(vals), "sums": {k: round(v, 6) for k, v in full.items()}}

    def stat_items(self, items: Sequence[tuple[int, Any]], f: Callable[[list[tuple[Any, float]]], float | None]) -> dict:
        """For metrics that need the items themselves (AUROC): items are (case index, payload)."""
        full = f([(p, 1.0) for _, p in items])
        by_case: dict[int, list[Any]] = {}
        for i, p in items:
            by_case.setdefault(i, []).append(p)
        vals = []
        for s in self.samples:
            weighted = [(p, float(w)) for i, w in s for p in by_case.get(i, [])]
            x = f(weighted)
            if x is not None:
                vals.append(x)
        return {"value": full, "ci95": ci(vals), "n": len(items)}


def ci(vals: list[float], level: float = 0.95) -> list[float] | None:
    if len(vals) < 2:
        return None
    v = sorted(vals)

    def q(p: float) -> float:
        k = p * (len(v) - 1)
        lo = int(k)
        hi = min(lo + 1, len(v) - 1)
        return v[lo] + (v[hi] - v[lo]) * (k - lo)

    a = (1 - level) / 2
    return [round(q(a), 6), round(q(1 - a), 6)]


def ratio(num: str, den: str) -> Callable[[dict[str, float]], float | None]:
    return lambda s: s[num] / s[den] if s[den] else None


def auroc(items: Iterable[tuple[tuple[float, bool], float]]) -> float | None:
    """Weighted Mann-Whitney AUROC. items: ((score, positive), weight). Ties count one half."""
    rows = sorted(((sc, pos, w) for (sc, pos), w in items), key=lambda r: r[0])
    pos_w = sum(w for _, p, w in rows if p)
    neg_w = sum(w for _, p, w in rows if not p)
    if not pos_w or not neg_w:
        return None
    auc = 0.0
    neg_below = 0.0
    i = 0
    while i < len(rows):
        j = i
        wp = wn = 0.0
        while j < len(rows) and rows[j][0] == rows[i][0]:
            if rows[j][1]:
                wp += rows[j][2]
            else:
                wn += rows[j][2]
            j += 1
        auc += wp * (neg_below + 0.5 * wn)
        neg_below += wn
        i = j
    return auc / (pos_w * neg_w)


def descriptive(n: int, need: int, what: str = "n") -> str:
    return f"descriptive ({what} {n} below the minimum {need}; bounds only, no claim)"


def verdict(ci95: list[float] | None, powered: bool = True) -> str:
    if not powered:
        return "descriptive (power rule: fewer than the minimum residue flips)"
    if ci95 is None:
        return "not testable"
    return "supported" if ci95[0] > 0 else "fails"


# ---------------------------------------------------------------------------
# Rows from the run state
# ---------------------------------------------------------------------------

def collect(run_dir: str | Path) -> dict[str, Any]:
    d = Path(run_dir)
    cfg = json.loads((d / "config.json").read_text())
    cases = list(read_jsonl(d / "cases.jsonl"))
    decisions, records, residue_rows, residue_records = [], [], [], []
    complete = True
    for case in cases:
        p = d / "state" / "cases" / f"{case.case_id}.json"
        cs = json.loads(p.read_text()) if p.exists() else {}
        complete &= bool(cs.get("done"))
        a_h = cs.get("case", {}).get("a_H")
        for system in cfg["systems"]:
            for budget in cfg["budgets"]:
                cell = cs.get("cells", {}).get(cell_name(system, int(budget)), {})
                if "a_V" not in cell or a_h is None:
                    continue
                view = rebuild_view(case, system, int(budget), cell["view"])
                rec = cell.get("flip", {}).get("record")
                decisions.append({
                    "case_id": case.case_id, "mechanism": case.mechanism, "cls": case.cls, "a_star": case.a_star,
                    "length": case.knobs.get("length"), "distance": case.knobs.get("distance"),
                    "system": system, "budget": int(budget), "a_H": a_h, "a_V": cell["a_V"],
                    "view_events": len(view.event_ids), "view_units": len(view.units),
                    "view_tokens": count_tokens([view.text]), "flip_status": rec["status"] if rec else None,
                })
                if rec:
                    records.append(rec)
                res = cell.get("residue", {})
                if "out" in res:
                    out = res["out"]
                    rrec = res.get("record")
                    residue_rows.append({"case_id": case.case_id, "mechanism": case.mechanism, "system": system,
                                         "budget": int(budget), "a_star": case.a_star, "a_H": a_h, **out,
                                         "search_status": rrec["status"] if rrec else None})
                    if rrec:
                        residue_records.append(rrec)
    return {"config": cfg, "cases": cases, "decisions": decisions, "records": records,
            "residue_rows": residue_rows, "residue_records": residue_records, "complete": complete}


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

def _features(cases: list[Case], rows: list[dict], fn: Callable[[dict], dict[str, float]]) -> dict[str, list[float]]:
    idx = {c.case_id: i for i, c in enumerate(cases)}
    feats: dict[str, list[float]] = {}
    for r in rows:
        for k, v in fn(r).items():
            feats.setdefault(k, [0.0] * len(cases))[idx[r["case_id"]]] += float(v)
    return feats


def _fcr_feats(r: dict, gated_only: bool) -> dict[str, float]:
    g = (r["a_H"] == r["a_star"]) or not gated_only
    com = is_committed(r["a_V"]) and g
    fc = com and r["a_V"] != r["a_star"]
    rare = r["mechanism"] != "M0"
    return {"c_r": com and rare, "f_r": fc and rare, "n_r": g and rare,
            "c_0": com and not rare, "f_0": fc and not rare, "n_0": g and not rare}


def _reweighted(s: dict[str, float]) -> float | None:
    if not s["n_r"] or not s["n_0"]:
        return None
    w_r, w_0 = RARE_SHARE_REPORT / s["n_r"], (1 - RARE_SHARE_REPORT) / s["n_0"]
    den = w_r * s["c_r"] + w_0 * s["c_0"]
    return (w_r * s["f_r"] + w_0 * s["f_0"]) / den if den else None


def _fcr_block(boot: Boot, cases, rows, gated_only: bool) -> dict[str, Any]:
    f = _features(cases, rows, lambda r: _fcr_feats(r, gated_only))
    if not f:
        return {}
    return {
        "all": boot.stat(f, lambda s: (s["f_r"] + s["f_0"]) / (s["c_r"] + s["c_0"]) if s["c_r"] + s["c_0"] else None),
        "rare": boot.stat(f, ratio("f_r", "c_r")),
        "rare_reweighted_1pct": boot.stat(f, _reweighted),
    }


def _dstar_scores(case: Case, rec: dict) -> tuple[float | None, float | None]:
    truth = set(case.d_star) & {e.id for e in case.events}
    if not truth:
        return None, None
    found = set(rec["event_ids"])
    hit = len(found & truth)
    prec = hit / len(found) if rec["status"] == "restored" and found else None
    return prec, hit / len(truth)


def harm_part(r: dict) -> str | None:
    """Which part of the harmful flip a decision belongs to, or None when it is not a harmful flip.

    wrong_commit: a_H == a_star, a_V != a_star and a_V is a committing action.
    abstention:   a_H == a_star, a_V is REQUEST_INFO and a_star is not.
    parse_error:  a_H == a_star and a_V is PARSE_ERROR.
    """
    if r["a_H"] != r["a_star"] or r["a_V"] == r["a_star"]:
        return None
    if r["a_V"] == PARSE_ERROR:
        return "parse_error"
    if r["a_V"] == REQUEST_INFO:
        return "abstention"
    return "wrong_commit"


def _denominators(hf: dict) -> dict[str, int]:
    return {"n_all": int(sum(hf.get("one", []))), "n_gated": int(sum(hf.get("g", [])))}


def _part(boot: Boot, hf: dict, name: str) -> dict[str, Any]:
    """One harmful-flip part with both rates and both denominators.

    rate_all divides by n_all (every case of the cell), rate_gated by n_gated (cases with a_H == a_star).
    `rate` is kept as an alias of rate_all for readers written before the explicit keys existed.
    """
    rate_all = boot.stat(hf, ratio(name, "one"))
    return {"count": int(sum(hf.get(name, []))), **_denominators(hf), "rate_all": rate_all, "rate": rate_all,
            "rate_gated": boot.stat(hf, ratio(name, "g"))}


def cell_metrics(boot: Boot, cases: list[Case], rows: list[dict], records: list[dict]) -> dict[str, Any]:
    by_case = {c.case_id: c for c in cases}
    out: dict[str, Any] = {"n": len(rows)}
    out["fcr_raw"] = _fcr_block(boot, cases, rows, False)
    out["fcr_gated"] = _fcr_block(boot, cases, rows, True)
    hf = _features(cases, rows, lambda r: {
        "h": r["a_H"] == r["a_star"] and r["a_V"] != r["a_star"], "one": 1, "g": r["a_H"] == r["a_star"],
        "ri_u": r["mechanism"] in ("U1", "U2") and r["a_V"] == REQUEST_INFO, "u": r["mechanism"] in ("U1", "U2"),
        "ri_ug": r["mechanism"] in ("U1", "U2") and r["a_V"] == REQUEST_INFO and r["a_H"] == r["a_star"],
        "ug": r["mechanism"] in ("U1", "U2") and r["a_H"] == r["a_star"],
        "ri_0": r["mechanism"] == "M0" and r["a_V"] == REQUEST_INFO, "m0": r["mechanism"] == "M0",
        "pe": r["a_V"] == PARSE_ERROR,
        # harmful split: a_H right, a_V wrong. The three parts partition the harmful flips.
        "wc": harm_part(r) == "wrong_commit", "ab": harm_part(r) == "abstention",
        "hpe": harm_part(r) == "parse_error"})
    out["harmful_flip"] = {**_part(boot, hf, "h"), "wrong_commit": _part(boot, hf, "wc"),
                           "abstention": _part(boot, hf, "ab"), "parse_error": _part(boot, hf, "hpe")}
    out["right_restraint"] = {"U1_U2": boot.stat(hf, ratio("ri_u", "u")),
                              "U1_U2_gated": boot.stat(hf, ratio("ri_ug", "ug"))}
    out["over_caution_M0"] = boot.stat(hf, ratio("ri_0", "m0"))
    out["parse_errors_a_V"] = int(sum(hf["pe"]))

    def rec_feats(rec: dict) -> dict[str, float]:
        prec, recall = _dstar_scores(by_case[rec["case_id"]], rec)
        return {"calls": rec["calls_used"], "searches": 1, "cap": rec["hit_cap"],
                "restored": rec["status"] == "restored", "verified": rec.get("verified") is True,
                "p_sum": prec or 0.0, "p_n": prec is not None, "r_sum": recall or 0.0, "r_n": recall is not None,
                "exact": rec["status"] == "restored" and set(rec["event_ids"]) == set(by_case[rec["case_id"]].d_star)
                and bool(by_case[rec["case_id"]].d_star)}

    rf = _features(cases, [{**r, "case_id": r["case_id"]} for r in records], rec_feats)
    out["flip_search"] = {
        "searches": len(records), "status": dict(Counter(r["status"] for r in records)),
        "cap_hits": sum(r["hit_cap"] for r in records),
        "granularity": dict(Counter(r["granularity"] for r in records if r["status"] == "restored")),
        "mean_calls": boot.stat(rf, ratio("calls", "searches")) if records else None,
        "verified_rate": boot.stat(rf, ratio("verified", "restored")) if records else None,
        "precision_vs_dstar": boot.stat(rf, ratio("p_sum", "p_n")) if records else None,
        "recall_vs_dstar": boot.stat(rf, ratio("r_sum", "r_n")) if records else None,
        "exact_dstar": int(sum(rf.get("exact", []))) if records else 0,
    }
    out["view"] = {"mean_tokens": sum(r["view_tokens"] for r in rows) / len(rows) if rows else None,
                   "mean_units": sum(r["view_units"] for r in rows) / len(rows) if rows else None}
    return out


def h2b_items(cases: list[Case], rows: list[dict], records: dict[tuple[str, str], dict], draws: int,
              seed: int) -> list[dict]:
    """Residue flips on event-view systems where every D* event is in V and the search restored."""
    by_case = {c.case_id: c for c in cases}
    items = []
    for r in rows:
        if not (r["residue_flip"] and r["system"] in EVENT_SYSTEMS and r["dstar_in_view"]):
            continue
        rec = records.get((r["case_id"], r["system"]))
        if not rec or rec["status"] != "restored":
            continue
        truth = set(by_case[r["case_id"]].d_star)
        found = set(rec["event_ids"])
        cands = list(r["candidate_ids"])
        rng = random.Random(f"h2b|{seed}|{r['case_id']}|{r['system']}")
        k = min(len(found), len(cands))
        rand = sum(len(truth & set(rng.sample(cands, k))) / len(truth) for _ in range(draws)) / draws
        items.append({"case_id": r["case_id"], "system": r["system"], "recall": len(found & truth) / len(truth),
                      "random_recall": rand, "k": k, "n_candidates": len(cands)})
    return items


def analysis_sizes(cases: list[Case], rows: list[dict], records: dict, cfg: dict[str, Any]) -> dict[str, Any]:
    """Size of each hypothesis's own analysis set (protocol v2), and the H2b funnel.

    H2a: eligible cases. H2b: residue flips on event-view systems with every D* event in V and a restored
    search. H2c: residue flips (positives) and non-flips (negatives) among eligible cases with a parsed p_residue.
    """
    items = h2b_items(cases, rows, records, 1, int(cfg["bootstrap"]["seed"]))
    parsed = [r for r in rows if r.get("p_residue") is not None]
    ev = [r for r in rows if r["residue_flip"] and r["system"] in EVENT_SYSTEMS]
    inview = [r for r in ev if r["dstar_in_view"]]
    return {"H2a_n": len(rows), "H2b_n": len(items),
            "H2c_pos": sum(bool(r["residue_flip"]) for r in parsed),
            "H2c_neg": sum(not r["residue_flip"] for r in parsed),
            "H2b_funnel": {"residue_flips_event_views": len(ev), "dstar_in_view": len(inview),
                           "search_restored": len(items)}}


# Version of the summary_v*.json layout, read by a private dashboard. Bump it when a field changes meaning.
SUMMARY_SCHEMA = "lens.flip.summary/1"


FIELD_NOTES = {
    "rate_all": "harmful_flip, wrong_commit, abstention, parse_error: count / n_all, n_all = every case of the cell.",
    "rate_gated": "the same count / n_gated, n_gated = cases of the cell with a_H == a_star.",
    "rate": "alias of rate_all, kept so readers written before the explicit keys keep working.",
    "recoverability": "residue.<pooled|by_system>.recoverability: how the flip searches of the residue flips ended. "
                      "The four counts sum to n_residue_flips. The not_recoverable_diagnostic is descriptive only.",
    "provenance": "protocol number, protocol file and its commit, source runs, and sha256 of the input files.",
    "recoverability.restored": "restored_1_minimal + restored_cap_hit; its share is restored.share. "
                               "shares.restored_1_minimal is the 1-minimal part only. Do not confuse the two.",
    "omission_degeneracy": "residue.<pooled|by_system>.omission_degeneracy: notes where the frozen verbatim omission "
                           "comparator is 1.0, and the exceptions with case, system and value. Descriptive only.",
}


def power_report(sizes: dict[str, Any], thr: dict[str, int], protocol: int) -> dict[str, Any]:
    """The per-hypothesis power rule. `protocol` is the protocol of the report (2 or 3), never a default."""
    return {
        "protocol": protocol,
        "H2a": {"n": sizes["H2a_n"], "min": thr["H2a_min_cases"], "powered": sizes["H2a_n"] >= thr["H2a_min_cases"]},
        "H2b": {"n": sizes["H2b_n"], "min": thr["H2b_min_flips"], "powered": sizes["H2b_n"] >= thr["H2b_min_flips"]},
        "H2c": {"n_pos": sizes["H2c_pos"], "n_neg": sizes["H2c_neg"], "min_pos": thr["H2c_min_positives"],
                "min_neg": thr["H2c_min_negatives"],
                "powered": sizes["H2c_pos"] >= thr["H2c_min_positives"] and sizes["H2c_neg"] >= thr["H2c_min_negatives"]},
    }


def _share(k: int, n: int) -> float | None:
    return k / n if n else None


def recoverability(rows: list[dict], records: dict) -> dict[str, Any]:
    """How the flip searches of the residue flips ended, with a descriptive look at the unrecoverable ones.

    restored_1_minimal: the search restored the target and stopped at a 1-minimal set (no cap hit).
    restored_cap_hit:   the search restored the target but stopped at the call cap, so the set may not be minimal.
    not_recoverable:    adding back every omitted unit does not restore the target ("not_recoverable_by_insertion").
    other:              a flip with no search record or another status (zero in the stored runs).
    The four counts sum to the number of residue flips.

    The diagnostic for the not-recoverable flips is descriptive only. It counts how often note B's action equals
    the action on the truncation view (a_trunc) and how often B chose REQUEST_INFO. It carries no claim.
    """
    flips = [r for r in rows if r["residue_flip"]]
    counts = {"restored_1_minimal": 0, "restored_cap_hit": 0, "not_recoverable": 0, "other": 0}
    unrecoverable: list[dict] = []
    for r in flips:
        rec = records.get((r["case_id"], r["system"]))
        if rec and rec["status"] == "restored":
            counts["restored_cap_hit" if rec["hit_cap"] else "restored_1_minimal"] += 1
        elif (rec["status"] if rec else r.get("search_status")) == "not_recoverable_by_insertion":
            counts["not_recoverable"] += 1
            unrecoverable.append(r)
        else:
            counts["other"] += 1
    n = len(flips)
    restored = counts["restored_1_minimal"] + counts["restored_cap_hit"]
    same_trunc = sum(r["a_B"] == r["a_trunc"] for r in unrecoverable)
    ask = sum(r["a_B"] == REQUEST_INFO for r in unrecoverable)
    m = len(unrecoverable)
    return {
        "n_flips": n,
        "counts": counts,
        "shares": {k: _share(v, n) for k, v in counts.items()},
        "restored": {"count": restored, "share": _share(restored, n)},
        "not_recoverable_diagnostic": {
            "descriptive_only": True, "n": m,
            "b_equals_trunc": {"count": same_trunc, "share": _share(same_trunc, m)},
            "b_request_info": {"count": ask, "share": _share(ask, m)},
            "b_actions": dict(sorted(Counter(r["a_B"] for r in unrecoverable).items())),
        },
    }


def omission_degeneracy(rows: list[dict]) -> dict[str, Any]:
    """How often the frozen omission comparator (units of V not quoted verbatim in the note, over units of V) is 1.0.

    Notes paraphrase and rarely quote a unit word for word, so the value is 1.0 on almost every note. That makes
    "H2c against omission" uninformative. Descriptive only; it changes no test."""
    known = [r for r in rows if r.get("omission") is not None]
    exc = [{"case_id": r["case_id"], "system": r.get("system"), "value": round(float(r["omission"]), 4)}
           for r in known if float(r["omission"]) < 1.0 - 1e-12]
    return {
        "n_notes": len(rows), "n_known": len(known), "n_missing": len(rows) - len(known),
        "n_at_one": len(known) - len(exc), "share_at_one": _share(len(known) - len(exc), len(known)),
        "exceptions": sorted(exc, key=lambda e: (str(e["system"]), e["case_id"])),
        "descriptive_only": True,
    }


def residue_block(boot: Boot, cases: list[Case], rows: list[dict], records: dict, cfg: dict[str, Any], *,
                  n_flips_run: int, h2b_systems: bool = True, protocol: int = 1) -> dict[str, Any]:
    min_flips = int(cfg.get("power_min_flips", 20))
    powered = n_flips_run >= min_flips
    v2 = protocol >= 2
    thr = power_thresholds(cfg)
    sizes = analysis_sizes(cases, rows, records, cfg) if v2 else {}
    pw = power_report(sizes, thr, protocol) if v2 else {}
    if v2 and not h2b_systems:
        pw["H2b"]["applicable"] = False  # H2b is defined for event-view systems only
    f = _features(cases, rows, lambda r: {"e": 1, "bw": r["a_B"] != r["a_star"], "fl": r["a_floor"] != r["a_A"],
                                          "tw": r["a_trunc"] != r["a_star"]})
    out: dict[str, Any] = {"n_eligible": len(rows), "n_residue_flips": sum(r["residue_flip"] for r in rows),
                           "power_rule": pw if v2 else {"min_residue_flips": min_flips,
                                                        "n_residue_flips_run": n_flips_run, "powered": powered}}
    if v2:
        out["analysis_sizes"] = sizes
    out["recoverability"] = recoverability(rows, records)
    out["omission_degeneracy"] = omission_degeneracy(rows)
    if not rows:
        return out
    out["residue_rate"] = boot.stat(f, ratio("bw", "e"))
    out["floor_rate"] = boot.stat(f, ratio("fl", "e"))
    out["trunc_rate"] = boot.stat(f, ratio("tw", "e"))
    h2a = boot.stat(f, lambda s: (s["bw"] - s["fl"]) / s["e"] if s["e"] else None)
    out["h2a_diff"], out["h2a_ci95"] = h2a["value"], h2a["ci95"]
    out["h2a_verdict"] = verdict(h2a["ci95"]) if not v2 else (
        verdict(h2a["ci95"]) if pw["H2a"]["powered"] else descriptive(pw["H2a"]["n"], pw["H2a"]["min"], "eligible cases"))
    if h2b_systems:
        items = h2b_items(cases, rows, records, int(cfg.get("h2b_draws", 1000)), int(cfg["bootstrap"]["seed"]))
        hb = _features(cases, items, lambda it: {"r": it["recall"], "q": it["random_recall"], "n": 1})
        out["h2b_n"] = len(items)
        if items:
            out["h2b_recall"] = boot.stat(hb, ratio("r", "n"))
            out["h2b_random_recall"] = boot.stat(hb, ratio("q", "n"))
            d = boot.stat(hb, lambda s: (s["r"] - s["q"]) / s["n"] if s["n"] else None)
            out["h2b_diff"], out["h2b_ci95"] = d["value"], d["ci95"]
        else:
            out["h2b_recall"] = out["h2b_random_recall"] = out["h2b_diff"] = out["h2b_ci95"] = None
        if v2:
            out["h2b_verdict"] = (verdict(out.get("h2b_ci95")) if pw["H2b"]["powered"]
                                  else descriptive(pw["H2b"]["n"], pw["H2b"]["min"], "residue flips"))
        else:
            out["h2b_verdict"] = verdict(out.get("h2b_ci95"), powered) if items else "not testable"
    idx = {c.case_id: i for i, c in enumerate(cases)}
    parsed = [r for r in rows if r["p_residue"] is not None]
    out["p_residue_parse_errors"] = len(rows) - len(parsed)
    out["note_status"] = dict(Counter(r["note_status"] for r in rows))
    items_p = [(idx[r["case_id"]], (r["p_residue"], r["omission"] if r["omission"] is not None else 0.0,
                                    r["residue_flip"])) for r in parsed]

    def a_p(ws):
        return auroc((((p, y), w) for (p, _, y), w in ws))

    def a_o(ws):
        return auroc((((o, y), w) for (_, o, y), w in ws))

    def d_half(ws):
        a = a_p(ws)
        return None if a is None else a - 0.5

    def d_om(ws):
        a, b = a_p(ws), a_o(ws)
        return None if a is None or b is None else a - b

    out["h2c_n"] = len(items_p)
    out["h2c_auroc"] = boot.stat_items(items_p, a_p)
    out["h2c_auroc_omission"] = boot.stat_items(items_p, a_o)
    dh, do = boot.stat_items(items_p, d_half), boot.stat_items(items_p, d_om)
    out["h2c_diff_vs_half"], out["h2c_ci95_vs_half"] = dh["value"], dh["ci95"]
    out["h2c_diff_vs_omission"], out["h2c_ci95_vs_omission"] = do["value"], do["ci95"]
    if v2 and not pw["H2c"]["powered"]:
        c = pw["H2c"]
        out["h2c_verdict"] = (f"descriptive (residue flips {c['n_pos']} of {c['min_pos']} needed, "
                              f"non-flips {c['n_neg']} of {c['min_neg']} needed; bounds only, no claim)")
    elif not v2 and not powered:
        out["h2c_verdict"] = verdict(None, False)
    elif dh["ci95"] is None or do["ci95"] is None:
        out["h2c_verdict"] = "not testable"
    else:
        out["h2c_verdict"] = "supported" if dh["ci95"][0] > 0 and do["ci95"][0] > 0 else "fails"
    return out


def summarise(data: dict[str, Any], spend: dict[str, Any] | None = None, protocol: int = 1) -> dict[str, Any]:
    cfg, cases = data["config"], data["cases"]
    boot = Boot(len(cases), int(cfg["bootstrap"]["reps"]), int(cfg["bootstrap"]["seed"]))
    dec = data["decisions"]
    a_h = {}
    for r in dec:
        a_h[r["case_id"]] = r["a_H"]
    by_mech = {m: [c for c in cases if c.mechanism == m] for m in MECHANISMS}

    def acc(cs: list[Case]) -> dict[str, Any]:
        known = [c for c in cs if c.case_id in a_h]
        ok = sum(a_h[c.case_id] == c.a_star for c in known)
        return {"n": len(known), "correct": ok, "accuracy": ok / len(known) if known else None}

    overall = acc(cases)
    per_mech = {m: acc(by_mech[m]) for m in MECHANISMS}
    excluded = [c for c in cases if c.case_id in a_h and a_h[c.case_id] != c.a_star]
    # From protocol 2 on, the echoed config names the protocol of the report. The run's own config.json is not
    # changed: its "version" is "headline-v2" for every run made by the engine (see provenance.runs[].config_version).
    cfg_out = {**cfg, "version": f"headline-v{protocol}"} if protocol >= 2 else cfg
    out: dict[str, Any] = {
        "schema": SUMMARY_SCHEMA,
        "protocol": protocol,
        "complete": data["complete"],
        "field_notes": FIELD_NOTES,
        "config": cfg_out,
        "n_cases": len(cases),
        "mechanism_counts": {m: len(by_mech[m]) for m in MECHANISMS},
        "validity": {"overall": overall, "per_mechanism": per_mech,
                     "gate": gate_report(overall["accuracy"], {m: a["accuracy"] for m, a in per_mech.items()})},
        "excluded_from_gated": {"count": len(excluded), "by_mechanism": dict(Counter(c.mechanism for c in excluded)),
                                "a_H_parse_errors": sum(a_h[c.case_id] == PARSE_ERROR for c in excluded)},
        "bootstrap": {"reps": int(cfg["bootstrap"]["reps"]), "seed": int(cfg["bootstrap"]["seed"]), "level": 0.95,
                      "unit": "case (paired: every cell and system of a resampled case comes along)"},
        "cells": {},
    }
    for system in cfg["systems"]:
        for budget in cfg["budgets"]:
            rows = [r for r in dec if r["system"] == system and r["budget"] == int(budget)]
            recs = [r for r in data["records"] if r["system"] == system and r["budget"] == int(budget)]
            out["cells"][cell_name(system, int(budget))] = cell_metrics(boot, cases, rows, recs)
    rrows = data["residue_rows"]
    rrecs = {(r["case_id"], r["system"]): r for r in data["residue_records"]}
    n_flips = sum(r["residue_flip"] for r in rrows)
    out["residue"] = {
        "budget": int(cfg["residue_budget"]),
        "definition": ("eligible: a_H == a_star and a_A = a_V == a_H at the residue budget; residue_rate = "
                       "P(a_B != target), floor_rate = P(v2 decision on V != a_A), trunc_rate = P(B on the last "
                       f"{cfg['trunc_tokens']} tokens of V != target); H2b on event-view systems with every D* "
                       "event in V and a restored search; H2c AUROC of p_residue for residue flips"),
        "pooled": residue_block(boot, cases, rrows, rrecs, cfg, n_flips_run=n_flips, protocol=protocol),
        "by_system": {s: residue_block(boot, cases, [r for r in rrows if r["system"] == s], rrecs, cfg,
                                       n_flips_run=n_flips, h2b_systems=s in EVENT_SYSTEMS, protocol=protocol)
                      for s in cfg["systems"]},
        "searches": {"n": len(rrecs), "status": dict(Counter(r["status"] for r in rrecs.values())),
                     "cap_hits": sum(r["hit_cap"] for r in rrecs.values())},
    }
    if protocol >= 2:
        out["residue"]["power_rule_v2"] = power_thresholds(cfg)
    if spend is not None:
        rounds = spend.get("rounds", [])
        t0 = min((r["submitted_at"] for r in rounds), default=None)
        t1 = max((r.get("ended_at", 0) for r in rounds), default=None)
        out["cost"] = {"actual_usd": spend.get("actual_usd"), "input_tokens": spend.get("input_tokens"),
                       "output_tokens": spend.get("output_tokens"), "by_stage": spend.get("by_stage"),
                       "rounds": len(rounds),
                       "usd_per_case": (spend.get("actual_usd") or 0) / len(cases) if cases else None,
                       "wall_seconds": (t1 - t0) if t0 is not None and t1 else None,
                       "wall_seconds_per_case": ((t1 - t0) / len(cases)) if t0 is not None and t1 and cases else None}
    return out


# ---------------------------------------------------------------------------
# Tables
# ---------------------------------------------------------------------------

def _f(x: Any, pct: bool = True) -> str:
    if x is None:
        return "-"
    if isinstance(x, dict):
        v, c = x.get("value"), x.get("ci95")
        if v is None:
            return "-"
        s = f"{v * 100:.1f}" if pct else f"{v:.2f}"
        if c:
            s += f" [{c[0] * 100:.1f}, {c[1] * 100:.1f}]" if pct else f" [{c[0]:.2f}, {c[1]:.2f}]"
        return s
    return f"{x * 100:.1f}" if pct else f"{x:.3f}"


def tables(summary: dict[str, Any]) -> str:
    if summary.get("protocol", 1) >= 2:
        return tables_v2(summary)
    lines = [f"cases {summary['n_cases']}, complete {summary['complete']}, "
             f"a_H accuracy {_f(summary['validity']['overall']['accuracy'])}%, gate "
             f"{'PASS' if summary['validity']['gate']['passes'] else 'FAIL'}, excluded from gated "
             f"{summary['excluded_from_gated']['count']}", ""]
    lines.append(f"{'cell':<16}{'FCR gated rare %':<24}{'FCR gated 1% %':<24}{'harmful gated %':<24}"
                 f"{'restraint %':<12}{'caution %':<11}{'searches':>9}{'cap':>5}{'calls':>7}{'recall':>8}")
    for name, c in summary["cells"].items():
        fs = c["flip_search"]
        lines.append(
            f"{name:<16}{_f(c['fcr_gated'].get('rare')):<24}{_f(c['fcr_gated'].get('rare_reweighted_1pct')):<24}"
            f"{_f(c['harmful_flip']['rate_gated']):<24}{_f(c['right_restraint']['U1_U2']['value']):<12}"
            f"{_f(c['over_caution_M0']['value']):<11}{fs['searches']:>9}{fs['cap_hits']:>5}"
            f"{(fs['mean_calls'] or {}).get('value') or 0:>7.1f}"
            f"{_f((fs['recall_vs_dstar'] or {}).get('value')):>8}")
    lines.append("")
    r = summary["residue"]
    lines.append(f"residue at B = {r['budget']}")
    lines.append(f"{'system':<12}{'eligible':>9}{'flips':>7}  {'residue %':<22}{'floor %':<22}{'trunc %':<22}"
                 f"{'H2a diff %':<22}")
    for name, b in [("pooled", r["pooled"]), *r["by_system"].items()]:
        if not b.get("n_eligible"):
            lines.append(f"{name:<12}{0:>9}")
            continue
        h2a = {"value": b["h2a_diff"], "ci95": b["h2a_ci95"]}
        lines.append(f"{name:<12}{b['n_eligible']:>9}{b['n_residue_flips']:>7}  {_f(b['residue_rate']):<22}"
                     f"{_f(b['floor_rate']):<22}{_f(b['trunc_rate']):<22}{_f(h2a):<22}")
    p = r["pooled"]
    if p.get("n_eligible"):
        lines.append("")
        lines.append(f"H2a: {p['h2a_verdict']}")
        lines.append(f"H2b: n={p.get('h2b_n')} recall {_f(p.get('h2b_recall'))} random {_f(p.get('h2b_random_recall'))}"
                     f" diff {_f({'value': p.get('h2b_diff'), 'ci95': p.get('h2b_ci95')})} -> {p.get('h2b_verdict')}")
        lines.append(f"H2c: n={p['h2c_n']} AUROC {_f(p['h2c_auroc'], False)} omission "
                     f"{_f(p['h2c_auroc_omission'], False)} parse errors {p['p_residue_parse_errors']} -> "
                     f"{p['h2c_verdict']}")
        lines.append(f"power rule: {p['power_rule']['n_residue_flips_run']} residue flips, minimum "
                     f"{p['power_rule']['min_residue_flips']}")
    if "cost" in summary:
        c = summary["cost"]
        lines.append("")
        lines.append(f"cost ${c['actual_usd'] or 0:.4f} over {c['rounds']} rounds")
    return "\n".join(lines)


def _hyp_lines(name: str, b: dict[str, Any]) -> list[str]:
    """One line per hypothesis with its own n, its threshold and its verdict (protocol v2)."""
    pw = b["power_rule"]
    a, hb, hc = pw["H2a"], pw["H2b"], pw["H2c"]
    out = [f"{name} H2a: n={a['n']} eligible, needs {a['min']} -> {b['h2a_verdict']}; diff "
           f"{_f({'value': b['h2a_diff'], 'ci95': b['h2a_ci95']})}"]
    if hb.get("applicable", True):
        out.append(f"{name} H2b: n={hb['n']} residue flips, needs {hb['min']} -> {b['h2b_verdict']}; recall "
                   f"{_f(b.get('h2b_recall'))} random {_f(b.get('h2b_random_recall'))} diff "
                   f"{_f({'value': b.get('h2b_diff'), 'ci95': b.get('h2b_ci95')})}")
    else:
        out.append(f"{name} H2b: not defined for text views")
    out.append(f"{name} H2c: positives {hc['n_pos']} (needs {hc['min_pos']}), negatives {hc['n_neg']} "
               f"(needs {hc['min_neg']}) -> {b['h2c_verdict']}; AUROC {_f(b['h2c_auroc'], False)} vs omission "
               f"{_f(b['h2c_auroc_omission'], False)}, p_residue parse errors {b['p_residue_parse_errors']}")
    return out


def _den(hf: dict[str, Any]) -> tuple[int, int]:
    sums = hf["rate"]["sums"]
    return hf.get("n_all", int(sums["one"])), hf.get("n_gated", int(sums["g"]))


def _denominator_line(cells: dict[str, Any]) -> str:
    """One line with n_all and n_gated per cell (a range when the cells differ)."""
    dens = [_den(c["harmful_flip"]) for c in cells.values()]
    if not dens:
        return "denominators: no cells"

    def span(xs: list[int]) -> str:
        return str(xs[0]) if min(xs) == max(xs) else f"{min(xs)} to {max(xs)}"

    return (f"denominators per cell: n_all {span([d[0] for d in dens])} (every case), n_gated "
            f"{span([d[1] for d in dens])} (a_H == a_star). The rates below divide by n_gated. "
            "In the JSON rate_all divides by n_all and rate is its alias.")


def _pc(k: int, share: float | None) -> str:
    return f"{k} ({share * 100:.1f}%)" if share is not None else f"{k} (-)"


def recoverability_lines(r: dict[str, Any]) -> list[str]:
    """How the residue flips ended: restored 1-minimal, restored at the cap, not recoverable (protocol 2 and 3)."""
    blocks = [("pooled", r["pooled"]), *r["by_system"].items()]
    if not all("recoverability" in b for _, b in blocks):
        return []
    L = ["", "recoverability of residue flips (the flip search over the units of V missing from the note)",
         f"{'system':<12}{'flips':>6}  {'restored 1-minimal':<22}{'restored cap hit':<20}{'not recoverable':<20}"]
    for name, b in blocks:
        rc = b["recoverability"]
        c, sh = rc["counts"], rc["shares"]
        L.append(f"{name:<12}{rc['n_flips']:>6}  {_pc(c['restored_1_minimal'], sh['restored_1_minimal']):<22}"
                 f"{_pc(c['restored_cap_hit'], sh['restored_cap_hit']):<20}"
                 f"{_pc(c['not_recoverable'], sh['not_recoverable']):<20}"
                 + (f"other {c['other']}" if c["other"] else ""))
    L += ["restored, 1-minimal: putting back the omitted units restores the right action, and no unit of the set can be dropped.",
          "restored, cap hit: restored, but the search stopped at the call cap, so the set may not be minimal.",
          "not recoverable: adding back every omitted unit does not restore the right action, so the flip is not "
          "explained by omission alone, and the note's own wording or framing may cause it."]
    L += ["", "not recoverable flips, descriptive only, no claim: how often B's action matches the truncation action "
          "and how often B chose REQUEST_INFO",
          f"{'system':<12}{'n':>6}  {'B = truncation action':<24}{'B = REQUEST_INFO':<20}"]
    for name, b in blocks:
        d = b["recoverability"]["not_recoverable_diagnostic"]
        L.append(f"{name:<12}{d['n']:>6}  {_pc(d['b_equals_trunc']['count'], d['b_equals_trunc']['share']):<24}"
                 f"{_pc(d['b_request_info']['count'], d['b_request_info']['share']):<20}")
    return L


def tables_v2(summary: dict[str, Any]) -> str:
    """Protocol v2 tables: the harmful gated rate with its split, and one power-rule line per hypothesis."""
    lines = [f"protocol {summary.get('protocol', 2)}. cases {summary['n_cases']}, complete {summary['complete']}, "
             f"a_H accuracy {_f(summary['validity']['overall']['accuracy'])}%, gate "
             f"{'PASS' if summary['validity']['gate']['passes'] else 'FAIL'}, excluded from gated "
             f"{summary['excluded_from_gated']['count']}", ""]
    lines.append("harmful gated = wrong commit + abstention + parse error (a_H right, a_V wrong)")
    lines.append(_denominator_line(summary["cells"]))
    lines.append(f"{'cell':<16}{'FCR gated rare %':<24}{'FCR gated 1% %':<24}{'harmful % (of gated n)':<26}"
                 f"{'wrong commit % (of gated n)':<30}{'abstention % (of gated n)':<28}{'pe':>3}  "
                 f"{'restraint %':<12}{'caution %':<11}{'searches':>9}{'cap':>5}{'calls':>7}{'recall':>8}")
    for name, c in summary["cells"].items():
        fs, hf = c["flip_search"], c["harmful_flip"]
        lines.append(
            f"{name:<16}{_f(c['fcr_gated'].get('rare')):<24}{_f(c['fcr_gated'].get('rare_reweighted_1pct')):<24}"
            f"{_f(hf['rate_gated']):<26}{_f(hf['wrong_commit']['rate_gated']):<30}"
            f"{_f(hf['abstention']['rate_gated']):<28}{hf['parse_error']['count']:>3}  "
            f"{_f(c['right_restraint']['U1_U2']['value']):<12}{_f(c['over_caution_M0']['value']):<11}"
            f"{fs['searches']:>9}{fs['cap_hits']:>5}{(fs['mean_calls'] or {}).get('value') or 0:>7.1f}"
            f"{_f((fs['recall_vs_dstar'] or {}).get('value')):>8}")
    lines.append("")
    r = summary["residue"]
    thr = r.get("power_rule_v2", POWER_RULE_V2)
    lines.append(f"residue at B = {r['budget']}")
    lines.append(f"power rule per hypothesis: H2a at least {thr['H2a_min_cases']} eligible cases; H2b at least "
                 f"{thr['H2b_min_flips']} of its own residue flips; H2c at least {thr['H2c_min_positives']} residue "
                 f"flips and {thr['H2c_min_negatives']} non-flips. Below the minimum: descriptive, bounds only.")
    lines.append(f"{'system':<12}{'eligible':>9}{'flips':>7}  {'residue %':<22}{'floor %':<22}{'trunc %':<22}"
                 f"{'H2a diff %':<22}")
    for name, b in [("pooled", r["pooled"]), *r["by_system"].items()]:
        if not b.get("n_eligible"):
            lines.append(f"{name:<12}{0:>9}")
            continue
        h2a = {"value": b["h2a_diff"], "ci95": b["h2a_ci95"]}
        lines.append(f"{name:<12}{b['n_eligible']:>9}{b['n_residue_flips']:>7}  {_f(b['residue_rate']):<22}"
                     f"{_f(b['floor_rate']):<22}{_f(b['trunc_rate']):<22}{_f(h2a):<22}")
    for name, b in [("pooled", r["pooled"]), *r["by_system"].items()]:
        lines.append("")
        if not b.get("n_eligible"):
            lines.append(f"{name}: no eligible cases")
            continue
        lines += _hyp_lines(name, b)
    p = r["pooled"]
    if p.get("analysis_sizes"):
        f = p["analysis_sizes"]["H2b_funnel"]
        lines.append("")
        lines.append(f"H2b funnel (pooled): {f['residue_flips_event_views']} residue flips on event views, "
                     f"{f['dstar_in_view']} with every D* event in V, {f['search_restored']} with a restored search")
    lines += recoverability_lines(r)
    if "cost" in summary:
        c = summary["cost"]
        lines.append("")
        lines.append(f"cost ${c['actual_usd'] or 0:.4f} over {c['rounds']} rounds")
    return "\n".join(lines)


def write_report(run_dir: str | Path, protocol: int = 1) -> dict[str, Any]:
    """Protocol 1 writes the original files. Protocol 2 writes summary_v2.json and tables_v2.txt only
    (the row files are written only when absent), so a run stays readable under both protocols.
    Protocol 3 computes the v2 metric set and writes summary_v3.json and tables_v3.txt the same way."""
    d = Path(run_dir)
    data = collect(d)
    spend = json.loads((d / "spend.json").read_text()) if (d / "spend.json").exists() else None
    summary = summarise(data, spend, protocol)
    summary["provenance"] = provenance.build(protocol, [("run", d)])

    def lines(path: Path, rows: list[dict]) -> None:
        if protocol >= 2 and path.exists():
            return
        path.write_text("".join(json.dumps(r, sort_keys=True) + "\n" for r in rows), encoding="utf-8")

    lines(d / "decisions.jsonl", data["decisions"])
    lines(d / "records.jsonl", data["records"])
    lines(d / "residue.jsonl", data["residue_rows"])
    lines(d / "residue_records.jsonl", data["residue_records"])
    suffix = f"_v{protocol}" if protocol >= 2 else ""
    (d / f"summary{suffix}.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (d / f"tables{suffix}.txt").write_text(tables(summary) + "\n", encoding="utf-8")
    return summary

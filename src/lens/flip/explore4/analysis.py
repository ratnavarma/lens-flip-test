"""summary_v4.json and REPORT.md for exploratory protocol v4. Offline; reads the arm dirs only.

Bootstrap: the v3 one (1,000 resamples of the 300 combined cases, seed 11; every handoff of a resampled
case comes along), so every difference against a v3 quantity is paired by case and handoff.
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any, Callable

from ..headline.metrics import Boot, _features, auroc, ratio
from ..schema import REQUEST_INFO
from .engine import ARMS
from .frame import OUT, load_cases, load_frame

SYSTEMS = ("recent", "rag-bm25", "rag-embed", "summary")
EVENT_SYSTEMS = ("recent", "rag-bm25", "rag-embed")


def _states(root: Path, arm: str) -> dict[tuple[str, str], dict] | None:
    d = root / arm / "state" / "cases"
    if not d.exists():
        return None
    out = {}
    for p in d.glob("*.json"):
        cs = json.loads(p.read_text())
        for k, v in cs.items():
            if isinstance(v, dict) and k in SYSTEMS:
                out[(p.stem, k)] = v
    return out


def _complete(st: dict | None, n: int) -> bool:
    return st is not None and sum(bool(v.get("done")) for v in st.values()) == n


class Stats:
    def __init__(self, cases, rows: list[dict]):
        self.cases = cases
        self.rows = rows
        self.boot = Boot(len(cases), 1000, 11)
        self.idx = {c.case_id: i for i, c in enumerate(cases)}

    def rate(self, flag: str, rows: list[dict] | None = None) -> dict[str, Any]:
        rows = self.rows if rows is None else rows
        f = _features(self.cases, rows, lambda r: {"k": bool(r[flag]), "n": 1})
        s = self.boot.stat(f, ratio("k", "n"))
        return {"value": s["value"], "ci95": s["ci95"], "k": int(s["sums"]["k"]), "n": int(s["sums"]["n"])}

    def diff(self, a: str, b: str, rows: list[dict] | None = None) -> dict[str, Any]:
        rows = self.rows if rows is None else rows
        f = _features(self.cases, rows, lambda r: {"a": bool(r[a]), "b": bool(r[b]), "n": 1})
        s = self.boot.stat(f, lambda x: (x["a"] - x["b"]) / x["n"] if x["n"] else None)
        lo = s["ci95"][0] if s["ci95"] else None
        hi = s["ci95"][1] if s["ci95"] else None
        sign = None if lo is None else ("above zero" if lo > 0 else "below zero" if hi < 0 else "includes zero")
        return {"diff": s["value"], "ci95": s["ci95"], "bounds": sign, "n": int(s["sums"]["n"]),
                "k_a": int(s["sums"]["a"]), "k_b": int(s["sums"]["b"])}

    def auroc(self, score: str, flag: str, rows: list[dict] | None = None) -> dict[str, Any]:
        rows = [r for r in (self.rows if rows is None else rows) if r.get(score) is not None]
        items = [(self.idx[r["case_id"]], (float(r[score]), bool(r[flag]))) for r in rows]

        def f(ws):
            return auroc(((p, w) for p, w in ws))

        def d(ws):
            a = f(ws)
            return None if a is None else a - 0.5
        s, dh = self.boot.stat_items(items, f), self.boot.stat_items(items, d)
        pos = sum(bool(r[flag]) for r in rows)
        return {"auroc": s["value"], "ci95": s["ci95"], "diff_vs_half_ci95": dh["ci95"], "n": len(rows),
                "n_pos": pos, "n_neg": len(rows) - pos,
                "powered_20_20": pos >= 20 and len(rows) - pos >= 20}

    def auroc_diff(self, s1: str, s2: str, flag: str, rows: list[dict] | None = None) -> dict[str, Any]:
        rows = [r for r in (self.rows if rows is None else rows) if r.get(s1) is not None and r.get(s2) is not None]
        items = [(self.idx[r["case_id"]], (float(r[s1]), float(r[s2]), bool(r[flag]))) for r in rows]

        def f(ws):
            a = auroc((((x, y), w) for (x, _, y), w in ws))
            b = auroc((((z, y), w) for (_, z, y), w in ws))
            return None if a is None or b is None else a - b
        s = self.boot.stat_items(items, f)
        return {"diff": s["value"], "ci95": s["ci95"], "n": len(rows)}


def _by_system(st: Stats, flag: str) -> dict[str, Any]:
    return {s: st.rate(flag, [r for r in st.rows if r["system"] == s]) for s in SYSTEMS}


def build(root: Path = OUT) -> dict[str, Any]:
    cases = load_cases(root)
    hs = load_frame(root, cases)
    rows: list[dict] = []
    for h in hs:
        v3 = h.row["v3"]
        rows.append({"case_id": h.case.case_id, "system": h.system, "mechanism": h.case.mechanism,
                     "a_A": h.row["a_A"], "v3_flip": v3["residue_flip"], "v3_floor": v3["a_floor"] != h.row["a_A"],
                     "v3_recov": v3["search_status"] == "restored", "v3_p": v3["p_residue"],
                     "v3_note_tokens": v3["note_tokens"], "v3_a_B": v3["a_B"], "_h": h})
    n = len(rows)
    st = Stats(cases, rows)
    frozen = json.loads((root / "protocol_v4.json").read_text())
    led = json.loads((root / "spend.json").read_text()) if (root / "spend.json").exists() else {}
    out: dict[str, Any] = {
        "schema": "lens.flip.explore/4", "exploratory": True,
        "note": "Exploratory, outside the frozen v3 protocol. No result here changes a v3 verdict.",
        "protocol": {"doc": "docs/flip-protocol-v4-exploratory.md", "frozen_at": frozen["frozen_at"],
                     "prompt_sha256": frozen["prompts"], "reused_v3_sha256": frozen["reused_v3"],
                     "arm_config_sha256": frozen["arm_configs"], "local_sha256": frozen["local"],
                     "frame_sha256": frozen["frame_sha256"], "cases_sha256": frozen["cases_sha256"]},
        "denominators": {"cases": len(cases), "eligible_handoffs": n,
                         "by_system": dict(Counter(r["system"] for r in rows))},
        "bootstrap": {"reps": 1000, "seed": 11, "unit": "case (paired; every handoff of a resampled case)"},
        "v3_reference": {"residue_rate": st.rate("v3_flip"), "floor_rate": st.rate("v3_floor"),
                         "recoverable_rate": st.rate("v3_recov"),
                         "recoverable_minus_floor": st.diff("v3_recov", "v3_floor"),
                         "h2c_auroc": st.auroc("v3_p", "v3_flip")},
        "spend": {k: led.get(k) for k in ("cap_usd", "total_usd", "remaining_usd", "probes_usd")} | {
            "by_arm": {a: v.get("actual_usd") for a, v in led.get("arms", {}).items()}},
        "skipped": [],
    }

    # ---- Study A, extractive ----
    sx = _states(root, "A-extractive")
    if _complete(sx, n):
        for r in rows:
            ns = sx[(r["case_id"], r["system"])]
            o, rec = ns["out"], ns.get("record")
            r.update({"x_flip": o["residue_flip"], "x_p": o["p_residue"], "x_om": o["omission"],
                      "x_recov": bool(rec and rec["status"] == "restored"),
                      "x_recov_ver": bool(rec and rec["status"] == "restored" and rec.get("verified")),
                      "x_status": rec["status"] if rec else None, "x_cap": bool(rec and rec.get("hit_cap")),
                      "x_a_B": o["a_B"], "x_tokens": o["note_tokens"], "x_asm": o["assembly"],
                      "x_parse": o["x_status"]})
            h = r["_h"]
            if h.view.event_ids:
                kept = {h.view.event_ids[i] for i in o["assembly"]["kept_index"]}
                ds = set(h.case.d_star)
                r["x_dstar_kept"] = bool(ds) and ds <= kept if ds else None
        flips = [r for r in rows if r["x_flip"]]
        rec_counts = Counter("restored_1_minimal" if r["x_recov"] and not r["x_cap"] else
                             "restored_cap_hit" if r["x_recov"] else
                             "not_recoverable" if r["x_status"] == "not_recoverable_by_insertion" else
                             str(r["x_status"]) for r in flips)
        out["study_A_extractive"] = {
            "n": n,
            "residue_rate": st.rate("x_flip"),
            "recoverable_rate": st.rate("x_recov"),
            "recoverable_verified_rate": st.rate("x_recov_ver"),
            "floor_rate_v3": st.rate("v3_floor"),
            "primary_recoverable_minus_floor": st.diff("x_recov", "v3_floor"),
            "recoverable_verified_minus_floor": st.diff("x_recov_ver", "v3_floor"),
            "recoverable_minus_v3_recoverable": st.diff("x_recov", "v3_recov"),
            "residue_minus_v3_residue": st.diff("x_flip", "v3_flip"),
            "residue_minus_floor": st.diff("x_flip", "v3_floor"),
            "recoverability": {"n_flips": len(flips), "counts": dict(rec_counts),
                               "b_actions_on_flips": dict(Counter(r["x_a_B"] for r in flips)),
                               "b_request_info_on_flips": sum(r["x_a_B"] == REQUEST_INFO for r in flips)},
            "by_system": {s: {"residue_rate": st.rate("x_flip", [r for r in rows if r["system"] == s]),
                              "recoverable_rate": st.rate("x_recov", [r for r in rows if r["system"] == s]),
                              "floor_rate_v3": st.rate("v3_floor", [r for r in rows if r["system"] == s])}
                          for s in SYSTEMS},
            "h2c_auroc_p": st.auroc("x_p", "x_flip"),
            "h2c_auroc_omission": st.auroc("x_om", "x_flip"),
            "h2c_p_minus_omission": st.auroc_diff("x_p", "x_om", "x_flip"),
            "notes": {"mean_tokens": sum(r["x_tokens"] for r in rows) / n,
                      "empty": sum(r["x_tokens"] == 0 for r in rows),
                      "mean_units_kept": sum(r["x_asm"]["n_kept"] for r in rows) / n,
                      "units_skipped_budget": sum(r["x_asm"]["n_skipped_budget"] for r in rows),
                      "unknown_labels": sum(r["x_asm"]["n_unknown"] for r in rows),
                      "reply_status": dict(Counter(r["x_parse"] for r in rows)),
                      "dstar_all_kept_event_views_rare": _share([r.get("x_dstar_kept") for r in rows
                                                                 if r["system"] in EVENT_SYSTEMS
                                                                 and r["mechanism"] != "M0"
                                                                 and r.get("x_dstar_kept") is not None])},
        }
    else:
        out["skipped"].append({"arm": "A-extractive", "reason": "not run or not complete"})

    # ---- Study A, advice-stripped ----
    ss = _states(root, "A-stripped")
    if _complete(ss, n):
        for r in rows:
            o = ss[(r["case_id"], r["system"])]["out"]
            r.update({"s_flip": o["residue_flip"], "s_changed": o["changed"], "s_empty": o.get("empty", False),
                      "s_rules": [x for d in o["dropped"] for x in d["rules"]], "s_dropped": len(o["dropped"])})
        ch = [r for r in rows if r["s_changed"]]
        out["study_A_stripped"] = {
            "n": n, "n_changed": len(ch), "n_emptied": sum(r["s_empty"] for r in rows),
            "sentences_dropped": sum(r["s_dropped"] for r in rows),
            "rule_hits": dict(Counter(x for r in rows for x in r["s_rules"])),
            "residue_rate": st.rate("s_flip"), "residue_rate_v3": st.rate("v3_flip"),
            "stripped_minus_v3": st.diff("s_flip", "v3_flip"),
            "changed_only": {"residue_rate": st.rate("s_flip", ch), "residue_rate_v3": st.rate("v3_flip", ch),
                             "stripped_minus_v3": st.diff("s_flip", "v3_flip", ch)},
            "stripped_minus_floor": st.diff("s_flip", "v3_floor"),
            "by_system": {s: {"residue_rate": st.rate("s_flip", [r for r in rows if r["system"] == s]),
                              "residue_rate_v3": st.rate("v3_flip", [r for r in rows if r["system"] == s])}
                          for s in SYSTEMS},
            "transitions": {"v3_flip_to_no_flip": sum(r["v3_flip"] and not r["s_flip"] for r in rows),
                            "no_flip_to_flip": sum(not r["v3_flip"] and r["s_flip"] for r in rows)},
        }
    else:
        out["skipped"].append({"arm": "A-stripped", "reason": "not run or not complete"})

    # ---- Study B, readers ----
    readers: dict[str, Any] = {"haiku-5-5 (v3)": _reader_block(st, rows, "v3_flip", None)}
    sb = _states(root, "B-sonnet")
    if _complete(sb, n):
        for r in rows:
            o = sb[(r["case_id"], r["system"])]["out"]
            r.update({"sb_flip": o["residue_flip"], "sb_a_B": o["a_B"], "sb_status": o["status"]})
        sv = _states(root, "B-sonnet-view")
        has_view = _complete(sv, n)
        if has_view:
            for r in rows:
                r["sv_miss"] = sv[(r["case_id"], r["system"])]["out"]["view_miss"]
        readers["sonnet-5-5"] = _reader_block(st, rows, "sb_flip", "sv_miss" if has_view else None)
        readers["sonnet-5-5"]["parse_status"] = dict(Counter(r["sb_status"] for r in rows))
        readers["sonnet-5-5"]["agree_with_haiku_a_B"] = sum(r["sb_a_B"] == r["v3_a_B"] for r in rows)
        if not has_view:
            out["skipped"].append({"arm": "B-sonnet-view", "reason": "not run or not complete"})
    else:
        out["skipped"].append({"arm": "B-sonnet", "reason": "not run or not complete"})
    gate_p = root / "B-local" / "gate.json"
    loc_p = root / "B-local" / "results.jsonl"
    if gate_p.exists():
        gate = json.loads(gate_p.read_text())
        out["study_B_local_gate"] = {k: gate[k] for k in ("model", "n", "correct", "pass_at", "passes",
                                                          "per_mechanism", "mean_seconds")}
        if loc_p.exists():
            lr = {(x["case_id"], x["system"]): x for x in map(json.loads, loc_p.read_text().splitlines())}
            for r in rows:
                x = lr[(r["case_id"], r["system"])]
                r.update({"lb_flip": x["residue_flip"], "lv_miss": x["view_miss"], "lb_a_B": x["a_B"]})
            readers[gate["model"] + " (local)"] = _reader_block(st, rows, "lb_flip", "lv_miss")
            readers[gate["model"] + " (local)"]["parse_errors"] = sum(r["lb_a_B"] == "PARSE_ERROR" for r in rows)
        else:
            out["skipped"].append({"arm": "B-local", "reason": "gate failed" if not gate["passes"] else "not run"})
    else:
        out["skipped"].append({"arm": "B-local", "reason": "gate not run"})
    out["study_B_readers"] = readers

    # ---- Study C, budgets ----
    sc = _states(root, "C-budget")
    if _complete(sc, n):
        budgets = {"150": {"flag": "v3_flip", "p": "v3_p", "tok": "v3_note_tokens"}}
        for r in rows:
            o = sc[(r["case_id"], r["system"])]["out"]
            for b, v in o.items():
                r.update({f"c{b}_flip": v["residue_flip"], f"c{b}_p": v["p_residue"], f"c{b}_tok": v["note_tokens"],
                          f"c{b}_status": v["note_status"]})
        for b in sorted(sc[next(iter(sc))]["out"], key=int):
            budgets[b] = {"flag": f"c{b}_flip", "p": f"c{b}_p", "tok": f"c{b}_tok"}
        blk: dict[str, Any] = {"n": n, "by_budget": {}, "paired": {}}
        for b, k in budgets.items():
            for r in rows:
                r[f"cap_{b}"] = r[k["tok"]] >= int(b) - 2
            blk["by_budget"][b] = {"residue_rate": st.rate(k["flag"]), "h2c_auroc": st.auroc(k["p"], k["flag"]),
                                   "mean_note_tokens": sum(r[k["tok"]] for r in rows) / n,
                                   "at_cap": st.rate(f"cap_{b}"),
                                   "by_system": {s: st.rate(k["flag"], [r for r in rows if r["system"] == s])
                                                 for s in SYSTEMS},
                                   **({"reply_status": dict(Counter(r[f"c{b}_status"] for r in rows))}
                                      if b != "150" else {})}
        keys = list(budgets)
        for i in range(len(keys)):
            for j in range(i + 1, len(keys)):
                a, b = keys[j], keys[i]
                blk["paired"][f"{a}_minus_{b}"] = st.diff(budgets[a]["flag"], budgets[b]["flag"])
        rates = [blk["by_budget"][b]["residue_rate"]["value"] for b in keys]
        blk["monotone_decreasing"] = all(x > y for x, y in zip(rates, rates[1:]))
        out["study_C_budget"] = blk
    else:
        out["skipped"].append({"arm": "C-budget", "reason": "not run or not complete"})
    return out


def _share(xs: list) -> dict[str, Any]:
    k = sum(bool(x) for x in xs)
    return {"k": k, "n": len(xs), "share": k / len(xs) if xs else None}


def _reader_block(st: Stats, rows: list[dict], flag: str, view_flag: str | None) -> dict[str, Any]:
    b = {"residue_rate": st.rate(flag), "minus_haiku": st.diff(flag, "v3_flip") if flag != "v3_flip" else None,
         "h2c_auroc_writer_p": st.auroc("v3_p", flag),
         "h2c_by_system": {s: st.auroc("v3_p", flag, [r for r in rows if r["system"] == s]) for s in SYSTEMS},
         "residue_by_system": {s: st.rate(flag, [r for r in rows if r["system"] == s]) for s in SYSTEMS}}
    if flag == "v3_flip":
        b["view_baseline"] = {"definition": "v3 floor: Haiku on V with the v2 prompt differs from a_A",
                              "rate": st.rate("v3_floor"), "residue_minus_view": st.diff("v3_flip", "v3_floor")}
    elif view_flag:
        b["view_baseline"] = {"definition": "the reader on V with the v2 prompt differs from a_A",
                              "rate": st.rate(view_flag), "residue_minus_view": st.diff(flag, view_flag)}
    return b


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

def _pc(x: dict | None) -> str:
    if not x or x.get("value") is None:
        return "-"
    ci = x.get("ci95")
    s = f"{100 * x['value']:.1f}"
    return s + (f" [{100 * ci[0]:.1f}, {100 * ci[1]:.1f}]" if ci else "")


def _kn(x: dict) -> str:
    return f"{x['k']}/{x['n']}"


def _d(x: dict | None) -> str:
    if not x or x.get("diff") is None:
        return "-"
    ci = x["ci95"]
    return f"{100 * x['diff']:+.1f} [{100 * ci[0]:+.1f}, {100 * ci[1]:+.1f}]" if ci else f"{100 * x['diff']:+.1f}"


def _au(x: dict | None) -> str:
    if not x or x.get("auroc") is None:
        return "-"
    ci = x["ci95"]
    return f"{x['auroc']:.2f} [{ci[0]:.2f}, {ci[1]:.2f}]" if ci else f"{x['auroc']:.2f}"


def report(s: dict[str, Any]) -> str:
    L = ["# LENS flip test: exploratory v4 report", "",
         "Exploratory. Outside the frozen v3 protocol. Protocol: `docs/flip-protocol-v4-exploratory.md`. "
         "All bounds are 95% paired bootstrap bounds over the 300 cases (1,000 resamples, seed 11). "
         f"Denominator: {s['denominators']['eligible_handoffs']} eligible handoffs at B = 1,000 unless a row says otherwise. "
         "Rates are in percent.", ""]
    v3 = s["v3_reference"]
    L += ["## v3 reference (recomputed from the frozen rows)", "",
          "| Quantity | k/n | Rate [95% CI] |", "| --- | --- | --- |",
          f"| Residue rate | {_kn(v3['residue_rate'])} | {_pc(v3['residue_rate'])} |",
          f"| Rewording floor | {_kn(v3['floor_rate'])} | {_pc(v3['floor_rate'])} |",
          f"| Insertion-recoverable flips | {_kn(v3['recoverable_rate'])} | {_pc(v3['recoverable_rate'])} |",
          f"| Recoverable minus floor | | {_d(v3['recoverable_minus_floor'])} |", ""]
    x = s.get("study_A_extractive")
    if x:
        L += ["## Study A1: extractive notes (A picks verbatim units, at most 150 tokens)", "",
              "| Quantity | k/n | Rate or difference [95% CI] |", "| --- | --- | --- |",
              f"| Residue rate, extractive | {_kn(x['residue_rate'])} | {_pc(x['residue_rate'])} |",
              f"| Insertion-recoverable flips, extractive | {_kn(x['recoverable_rate'])} | {_pc(x['recoverable_rate'])} |",
              f"| ... of which verified with the v2 prompt | {_kn(x['recoverable_verified_rate'])} | {_pc(x['recoverable_verified_rate'])} |",
              f"| Rewording floor (v3) | {_kn(x['floor_rate_v3'])} | {_pc(x['floor_rate_v3'])} |",
              f"| **Primary: recoverable minus floor** | | **{_d(x['primary_recoverable_minus_floor'])}** ({x['primary_recoverable_minus_floor']["bounds"]}) |",
              f"| Verified recoverable minus floor | | {_d(x['recoverable_verified_minus_floor'])} |",
              f"| Recoverable, extractive minus v3 | | {_d(x['recoverable_minus_v3_recoverable'])} |",
              f"| Residue, extractive minus v3 | | {_d(x['residue_minus_v3_residue'])} |",
              f"| Residue, extractive minus floor | | {_d(x['residue_minus_floor'])} |", "",
              f"Search outcomes on the {x['recoverability']['n_flips']} extractive flips: "
              + ", ".join(f"{k} {v}" for k, v in sorted(x["recoverability"]["counts"].items()))
              + f". B chose REQUEST_INFO on {x['recoverability']['b_request_info_on_flips']} of them.", "",
              "| System | n | Residue, extractive | Recoverable, extractive | Floor (v3) |", "| --- | --- | --- | --- | --- |"]
        for sy, b in x["by_system"].items():
            L.append(f"| {sy} | {b['residue_rate']['n']} | {_pc(b['residue_rate'])} | {_pc(b['recoverable_rate'])} | {_pc(b['floor_rate_v3'])} |")
        nt = x["notes"]
        L += ["", f"H2c on extractive notes (exploratory): AUROC of A's p_residue {_au(x['h2c_auroc_p'])}; "
                  f"AUROC of the omission fraction (now non-degenerate) {_au(x['h2c_auroc_omission'])}; "
                  f"p minus omission {x['h2c_p_minus_omission']['diff']:+.2f} "
                  f"[{x['h2c_p_minus_omission']['ci95'][0]:+.2f}, {x['h2c_p_minus_omission']['ci95'][1]:+.2f}].", "",
              f"Notes: mean {nt['mean_tokens']:.1f} tokens, {nt['mean_units_kept']:.2f} units kept, {nt['empty']} empty, "
              f"{nt['units_skipped_budget']} listed units skipped for the budget, {nt['unknown_labels']} unknown labels, "
              f"reply status {nt['reply_status']}. Rare cases on event views with every D* event kept: "
              f"{nt['dstar_all_kept_event_views_rare']['k']}/{nt['dstar_all_kept_event_views_rare']['n']}.", ""]
    a = s.get("study_A_stripped")
    if a:
        L += ["## Study A2: advice-stripped v3 notes", "",
              f"{a['n_changed']} of {a['n']} notes changed; {a['sentences_dropped']} sentences dropped; "
              f"{a['n_emptied']} notes emptied. Rule hits: {a['rule_hits']}.", "",
              "| Set | n | Residue, stripped | Residue, v3 | Stripped minus v3 |", "| --- | --- | --- | --- | --- |",
              f"| All handoffs | {a['n']} | {_pc(a['residue_rate'])} | {_pc(a['residue_rate_v3'])} | {_d(a['stripped_minus_v3'])} |",
              f"| Changed notes only | {a['changed_only']['residue_rate']['n']} | {_pc(a['changed_only']['residue_rate'])} | "
              f"{_pc(a['changed_only']['residue_rate_v3'])} | {_d(a['changed_only']['stripped_minus_v3'])} |", "",
              f"Stripped minus floor: {_d(a['stripped_minus_floor'])}. Flip to no flip: {a['transitions']['v3_flip_to_no_flip']}; "
              f"no flip to flip: {a['transitions']['no_flip_to_flip']}.", "",
              "| System | n | Residue, stripped | Residue, v3 |", "| --- | --- | --- | --- |"]
        for sy, b in a["by_system"].items():
            L.append(f"| {sy} | {b['residue_rate']['n']} | {_pc(b['residue_rate'])} | {_pc(b['residue_rate_v3'])} |")
        L.append("")
    rd = s.get("study_B_readers", {})
    L += ["## Study B: second readers on the frozen v3 notes", ""]
    if "study_B_local_gate" in s:
        g = s["study_B_local_gate"]
        L += [f"Local gate ({g['model']}, full history, v1 prompt): {g['correct']}/{g['n']}, pass at {g['pass_at']}: "
              f"{'PASS' if g['passes'] else 'FAIL'}. Per mechanism {g['per_mechanism']}. Mean {g['mean_seconds']} s per call.", ""]
    L += ["| Reader | Residue rate | Minus Haiku | H2c AUROC of the writer's p_residue | Reader on V (v2) | Residue minus reader on V |",
          "| --- | --- | --- | --- | --- | --- |"]
    for name, b in rd.items():
        vb = b.get("view_baseline") or {}
        L.append(f"| {name} | {_kn(b['residue_rate'])} {_pc(b['residue_rate'])} | {_d(b['minus_haiku'])} | "
                 f"{_au(b['h2c_auroc_writer_p'])} | {_pc(vb.get('rate'))} | {_d(vb.get('residue_minus_view'))} |")
    L += ["", "| Reader | " + " | ".join(f"{sy} residue | {sy} AUROC" for sy in SYSTEMS) + " |",
          "| --- |" + " --- | --- |" * len(SYSTEMS)]
    for name, b in rd.items():
        L.append(f"| {name} | " + " | ".join(f"{_pc(b['residue_by_system'][sy])} | {_au(b['h2c_by_system'][sy])}"
                                            for sy in SYSTEMS) + " |")
    L.append("")
    c = s.get("study_C_budget")
    if c:
        L += ["## Study C: note-budget sweep", "",
              "| Budget | Residue rate | Mean note tokens | Notes at cap | H2c AUROC |", "| --- | --- | --- | --- | --- |"]
        for b, v in c["by_budget"].items():
            L.append(f"| {b} | {_kn(v['residue_rate'])} {_pc(v['residue_rate'])} | {v['mean_note_tokens']:.1f} | "
                     f"{_pc(v['at_cap'])} | {_au(v['h2c_auroc'])} |")
        L += ["", "| Paired difference | Residue rate difference [95% CI] |", "| --- | --- |"]
        for k, v in c["paired"].items():
            L.append(f"| {k.replace('_', ' ')} | {_d(v)} ({v["bounds"]}) |")
        L += ["", "| Budget | " + " | ".join(SYSTEMS) + " |", "| --- |" + " --- |" * len(SYSTEMS)]
        for b, v in c["by_budget"].items():
            L.append(f"| {b} | " + " | ".join(_pc(v["by_system"][sy]) for sy in SYSTEMS) + " |")
        L += ["", f"Monotone decrease across budgets: {c['monotone_decreasing']}.", ""]
    sp = s["spend"]
    L += ["## Spend", "", f"Total ${sp.get('total_usd') or 0:.4f} of the ${sp.get('cap_usd') or 30:.2f} cap "
          f"(probes ${sp.get('probes_usd') or 0:.4f}).", "", "| Arm | Actual USD |", "| --- | --- |"]
    for k, v in (sp.get("by_arm") or {}).items():
        L.append(f"| {k} | {v:.4f} |")
    if s["skipped"]:
        L += ["", "## Skipped", ""] + [f"- {x['arm']}: {x['reason']}" for x in s["skipped"]]
    L += ["", "## Prompt hashes (sha256)", "", "| Item | sha256 |", "| --- | --- |"]
    for k, v in sorted({**s["protocol"]["prompt_sha256"], **s["protocol"]["reused_v3_sha256"]}.items()):
        L.append(f"| {k} | `{v}` |")
    return "\n".join(L) + "\n"


def write(root: Path = OUT) -> dict[str, Any]:
    s = build(root)
    (root / "summary_v4.json").write_text(json.dumps(s, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
    (root / "REPORT.md").write_text(report(s), encoding="utf-8")
    return s

"""Rebuild every table of the paper (1 to 15), the Figure 2 numbers and the paper facts from the text-free metric files.

    uv run python scripts/rebuild_paper.py                 # print every table and the paper facts
    uv run python scripts/rebuild_paper.py --check         # compare each cell with results/paper_expected.json
    uv run python scripts/rebuild_paper.py --write-facts   # rewrite results/paper_facts.json

Inputs (no note, view, unit, reason or prompt text in any of them):

    results/headline-300-v3/stage1_metrics_v3.jsonl            Table 1
    results/headline-300-v3/handoff_metrics_v3.jsonl           Tables 2, 2a, 2b, 2c, 2d, 3, 3b
    results/exploratory-v4/handoff_metrics_v4.jsonl            Tables 4, 5, 6
    results/exploratory-v5/handoff_metrics_v5.jsonl            Tables 7, 9, 10, 11, 12
    results/exploratory-v5/c1_control_metrics_v5.jsonl         Tables 8, 9
    results/exploratory-v5/judge_calibration_metrics_v5.jsonl  Table 10, first column
    results/exploratory-v6/handoff_metrics_v6.jsonl            Table 13 and the v6 numbers in the text
    results/exploratory-v7/handoff_metrics_v7.jsonl            Tables 14 and 15 and the v7 numbers in the text
    results/exploratory-v7/spend.json                          the v7 spend and request count in the text
    results/headline-300-v3/cases.jsonl                        the generated cases (paper facts on histories)

Bootstrap: the frozen one. Seed 11, 1,000 resamples of the 300 cases, paired by case: every row of a
resampled case comes along. No model call is made and no API key is needed.

Check rule. results/paper_expected.json holds each cell as the paper prints it. A cell matches when its text
around the numbers is the same and each rebuilt number, before rounding, is within half a unit of the last
printed digit. Counts and words must be equal. Expected values are never edited to force a match.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from collections import Counter
from pathlib import Path
from statistics import NormalDist

from lens.flip.headline.metrics import Boot, auroc, ci

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"
EXPECTED = RES / "paper_expected.json"
FACTS = RES / "paper_facts.json"
SYSTEMS = ("recent", "rag-bm25", "rag-embed", "summary")
EVENT_SYSTEMS = ("recent", "rag-bm25", "rag-embed")
GROUPS = ("M0", "M1-M3", "U1-U2")
DEFER, PARSE_ERROR = "REQUEST_INFO", "PARSE_ERROR"
NOTE_CAP = 150
RARE_SHARE = 0.01
NUM = re.compile(r"(?<![\w.])-?\d{1,3}(?:,\d{3})+(?:\.\d+)?|(?<![\w.])-?\d+(?:\.\d+)?")


def _jsonl(p: Path) -> list[dict]:
    return [json.loads(x) for x in p.read_text(encoding="utf-8").splitlines() if x.strip()]


# ---------------------------------------------------------------------------
# Cells
# ---------------------------------------------------------------------------

class Cell:
    """A printed cell: a format and the numbers before rounding."""

    def __init__(self, fmt: str, *nums: float):
        self.fmt, self.nums = fmt, nums

    def __str__(self) -> str:
        return self.fmt.format(*self.nums)


def S(text: str) -> Cell:
    return Cell(text.replace("{", "{{").replace("}", "}}"))


def N(k: int) -> Cell:
    return Cell("{:,d}", int(k))


def OF(k: int, n: int) -> Cell:
    return Cell("{:,d} of {:,d}", int(k), int(n))


def P(r: dict) -> Cell:
    """A rate or a difference in percent with its bounds."""
    return Cell("{:.1f} [{:.1f}, {:.1f}]", 100 * r["v"], 100 * r["ci"][0], 100 * r["ci"][1])


def P1(x: float) -> Cell:
    return Cell("{:.1f}", 100 * x)


def AU(r: dict, shift: float = 0.0) -> Cell:
    return Cell("{:.2f} [{:.2f}, {:.2f}]", r["v"] - shift, r["ci"][0] - shift, r["ci"][1] - shift)


DASH = S("-")


def matches(cell: Cell, expected: str) -> bool:
    got = str(cell)
    if not cell.nums:
        return got == expected
    toks = NUM.findall(expected)
    if len(toks) != len(cell.nums) or NUM.sub("#", expected) != NUM.sub("#", got):
        return False
    for t, x in zip(toks, cell.nums):
        t = t.replace(",", "")
        d = len(t.split(".")[1]) if "." in t else 0
        if abs(float(t) - x) > 0.5 * 10 ** -d + 1e-9:
            return False
    return True


# ---------------------------------------------------------------------------
# Bootstrap statistics, paired by case
# ---------------------------------------------------------------------------

class Metrics:
    def __init__(self, case_ids: list[str]):
        self.idx = {c: i for i, c in enumerate(case_ids)}
        self.n = len(case_ids)
        self.boot = Boot(self.n, 1000, 11)

    def stat(self, rows, feats: dict, fun) -> dict:
        f = {k: [0.0] * self.n for k in feats}
        for r in rows:
            i = self.idx[r["case_id"]]
            for k, fn in feats.items():
                f[k][i] += float(fn(r))
        s = self.boot.stat(f, fun)
        return {"v": s["value"], "ci": s["ci95"], "sums": s["sums"]}

    def ratio(self, rows, num, den=lambda r: 1) -> dict:
        s = self.stat(rows, {"k": num, "n": den}, lambda x: x["k"] / x["n"] if x["n"] else None)
        return {**s, "k": s["sums"]["k"], "n": s["sums"]["n"]}

    def diff(self, rows, a, b, den=lambda r: 1) -> dict:
        return self.stat(rows, {"a": a, "b": b, "n": den}, lambda x: (x["a"] - x["b"]) / x["n"] if x["n"] else None)

    def ratio_diff(self, rows, an, ad, bn, bd) -> dict:
        return self.stat(rows, {"an": an, "ad": ad, "bn": bn, "bd": bd},
                         lambda x: (x["an"] / x["ad"] - x["bn"] / x["bd"]) if x["ad"] and x["bd"] else None)

    def items(self, rows, payload, f) -> dict:
        its = [(self.idx[r["case_id"]], payload(r)) for r in rows]
        by: dict[int, list] = {}
        for i, p in its:
            by.setdefault(i, []).append(p)
        vals = []
        for s in self.boot.samples:
            x = f([(p, float(w)) for i, w in s for p in by.get(i, [])])
            if x is not None:
                vals.append(x)
        return {"v": f([(p, 1.0) for _, p in its]), "ci": ci(vals), "vals": vals}

    def auc(self, rows, score, flag) -> dict:
        rows = [r for r in rows if score(r) is not None]
        out = self.items(rows, lambda r: (float(score(r)), bool(flag(r))), lambda ws: auroc(ws))
        pos = sum(bool(flag(r)) for r in rows)
        return {**out, "pos": pos, "neg": len(rows) - pos}

    def auc_diff(self, rows, s1, f1, s2, f2) -> dict:
        """AUROC(s1 for f1) minus AUROC(s2 for f2) on the same rows."""
        rows = [r for r in rows if s1(r) is not None and s2(r) is not None]

        def f(ws):
            a = auroc((((p[0], p[1]), w) for p, w in ws))
            b = auroc((((p[2], p[3]), w) for p, w in ws))
            return None if a is None or b is None else a - b
        return self.items(rows, lambda r: (float(s1(r)), bool(f1(r)), float(s2(r)), bool(f2(r))), f)


def _holm(ps: dict[str, float]) -> dict[str, float]:
    order = sorted(ps, key=lambda k: ps[k])
    m, run, adj = len(ps), 0.0, {}
    for j, k in enumerate(order):
        run = max(run, min(1.0, (m - j) * ps[k]))
        adj[k] = run
    return adj


def _pfmt(x: float) -> str:
    return "< 0.001" if x < 0.001 else f"{x:.3f}" if x < 0.1 else f"{x:.2f}"


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

class Data:
    def __init__(self, res: Path = RES):
        self.s1 = _jsonl(res / "headline-300-v3" / "stage1_metrics_v3.jsonl")
        self.v3 = _jsonl(res / "headline-300-v3" / "handoff_metrics_v3.jsonl")
        self.v4 = _jsonl(res / "exploratory-v4" / "handoff_metrics_v4.jsonl")
        self.v5 = _jsonl(res / "exploratory-v5" / "handoff_metrics_v5.jsonl")
        self.c1 = _jsonl(res / "exploratory-v5" / "c1_control_metrics_v5.jsonl")
        self.cal = _jsonl(res / "exploratory-v5" / "judge_calibration_metrics_v5.jsonl")
        self.v6 = _jsonl(res / "exploratory-v6" / "handoff_metrics_v6.jsonl")
        self.v7 = _jsonl(res / "exploratory-v7" / "handoff_metrics_v7.jsonl")
        self.v7_spend = json.loads((res / "exploratory-v7" / "spend.json").read_text(encoding="utf-8"))
        self.cases_path = res / "headline-300-v3" / "cases.jsonl"
        self.case_ids = list(dict.fromkeys(r["case_id"] for r in self.s1))   # case order of the frozen run
        self.M = Metrics(self.case_ids)

    def arm(self, rows: list[dict], arm: str) -> list[dict]:
        """The v3 handoffs with one arm's fields added under the prefix x_."""
        by = {(r["case_id"], r["system"]): r for r in rows if r["arm"] == arm}
        return [{**r, **{"x_" + k: v for k, v in by[(r["case_id"], r["system"])].items()}} for r in self.v3]


flip = lambda r: bool(r["flip"])
floor = lambda r: bool(r["floor_flip"])
rec = lambda r: bool(r["flip"] and r["restored"] == "restored")
xflip = lambda r: bool(r["x_flip"])
xrec = lambda r: bool(r["x_flip"] and r["x_restored"] == "restored")
pres = lambda r: r["p_residue"]
xpres = lambda r: r["x_p_residue"]
one = lambda r: 1


# ---------------------------------------------------------------------------
# Tables
# ---------------------------------------------------------------------------

def table1(D: Data) -> dict:
    M, rows = D.M, {}
    for s in SYSTEMS:
        for b in (500, 1000, 2000):
            rs = [r for r in D.s1 if r["system"] == s and r["budget"] == b]
            g = lambda r: r["a_H"] == r["a_star"]
            com = lambda r: g(r) and r["a_V"] not in (DEFER, PARSE_ERROR)
            rare = lambda r: r["mechanism"] != "M0"
            feats = {"c_r": lambda r: com(r) and rare(r), "f_r": lambda r: com(r) and r["a_V"] != r["a_star"] and rare(r),
                     "n_r": lambda r: g(r) and rare(r), "c_0": lambda r: com(r) and not rare(r),
                     "f_0": lambda r: com(r) and r["a_V"] != r["a_star"] and not rare(r),
                     "n_0": lambda r: g(r) and not rare(r)}

            def rew(x):
                if not x["n_r"] or not x["n_0"]:
                    return None
                wr, w0 = RARE_SHARE / x["n_r"], (1 - RARE_SHARE) / x["n_0"]
                den = wr * x["c_r"] + w0 * x["c_0"]
                return (wr * x["f_r"] + w0 * x["f_0"]) / den if den else None
            harm = lambda r: g(r) and r["a_V"] != r["a_star"]
            ab = lambda r: harm(r) and r["a_V"] == DEFER
            wc = lambda r: harm(r) and r["a_V"] not in (DEFER, PARSE_ERROR)
            u = lambda r: r["mechanism"] in ("U1", "U2") and g(r)
            m0 = lambda r: r["mechanism"] == "M0"
            rows[f"{s}@{b}"] = [
                P(M.stat(rs, feats, lambda x: x["f_r"] / x["c_r"] if x["c_r"] else None)), P(M.stat(rs, feats, rew)),
                P1(M.ratio(rs, harm, g)["v"]), P1(M.ratio(rs, wc, g)["v"]), P1(M.ratio(rs, ab, g)["v"]),
                P1(M.ratio(rs, lambda r: u(r) and r["a_V"] == DEFER, u)["v"]),
                P1(M.ratio(rs, lambda r: m0(r) and r["a_V"] == DEFER, m0)["v"])]
    return {"Table 1": rows}


def tables_v3(D: Data) -> dict:
    M, v3 = D.M, D.v3
    sets = {"pooled": v3} | {s: [r for r in v3 if r["system"] == s] for s in SYSTEMS}
    gsets = {"pooled": v3} | {g: [r for r in v3 if r["mechanism_group"] == g] for g in GROUPS}
    bw = lambda r: r["a_B"] != r["a_star"]
    tw = lambda r: r["trunc_action"] != r["a_star"]
    out: dict = {}
    out["Table 2"] = {k: [N(len(rs)), N(sum(map(flip, rs))), P(M.ratio(rs, bw)), P(M.ratio(rs, floor)),
                          P(M.ratio(rs, tw)), P(M.diff(rs, bw, floor))] for k, rs in sets.items()}
    out["Table 2a"] = {k: [N(sum(map(rec, rs))), P(M.ratio(rs, rec)), P(M.diff(rs, rec, floor))]
                       for k, rs in sets.items()}
    out["Table 2b"] = {k: [N(len(rs)), N(sum(map(flip, rs))), P(M.ratio(rs, flip)), P(M.ratio(rs, floor)),
                           P(M.ratio(rs, rec)), P(M.diff(rs, rec, floor))] for k, rs in gsets.items()}
    defer = lambda r: bool(r["flip"] and r["a_B"] == DEFER)
    commit = lambda r: bool(r["flip"] and r["a_B"] != DEFER)
    t2c = {}
    for k, rs in gsets.items():
        d = M.ratio(rs, defer)
        t2c[k] = [P(d) if d["k"] else P1(d["v"]), P(M.ratio(rs, commit)), P(M.diff(rs, defer, floor)),
                  P(M.diff(rs, commit, floor))]
    out["Table 2c"] = t2c
    t2d = {}
    for k, rs in sets.items():
        fl = [r for r in rs if r["flip"]]
        rest = [r for r in fl if r["restored"] == "restored"]
        cap = sum(bool(r["restored_cap_hit"]) for r in rest)
        nr = [r for r in fl if r["restored"] != "restored"]
        t_ = sum(r["a_B"] == r["trunc_action"] for r in nr)
        d_ = sum(r["a_B"] == DEFER for r in nr)
        share = lambda a, b: Cell("{:,d} ({:.1f})", a, 100 * a / b)
        t2d[k] = [N(len(fl)), N(len(rest) - cap), N(cap), share(len(nr), len(fl)), share(t_, len(nr)), share(d_, len(nr))]
    out["Table 2d"] = t2d

    # Table 3: the frozen verdict rule and the post-hoc Holm column
    om = lambda r: r["omission_fraction"] if r["omission_fraction"] is not None else 0.0
    A = {k: M.auc(rs, pres, flip) for k, rs in sets.items()}
    O = {k: M.auc(rs, om, flip) for k, rs in sets.items()}
    p_norm, p_boot = {}, {}
    for s in SYSTEMS:
        vals = A[s]["vals"]
        mean = sum(vals) / len(vals)
        se = (sum((x - mean) ** 2 for x in vals) / (len(vals) - 1)) ** 0.5
        z = (A[s]["v"] - 0.5) / se if se else float("inf")
        p_norm[s] = math.erfc(abs(z) / math.sqrt(2))
        below, above = sum(x <= 0.5 for x in vals) / len(vals), sum(x >= 0.5 for x in vals) / len(vals)
        p_boot[s] = max(min(1.0, 2 * min(below, above)), 1 / len(M.boot.samples))
    hn, hb = _holm(p_norm), _holm(p_boot)
    t3 = {}
    for k, rs in sets.items():
        a = A[k]
        d_om = M.auc_diff([r for r in rs if r["p_residue"] is not None], pres, flip, om, flip)
        powered = a["pos"] >= 20 and a["neg"] >= 20
        ok = powered and a["ci"][0] - 0.5 > 0 and d_om["ci"][0] > 0
        t3[k] = [Cell("{:,d} / {:,d}", a["pos"], a["neg"]), AU(a), AU(a, 0.5),
                 S("not applied") if k == "pooled" else S(f"{_pfmt(hn[k])} / {_pfmt(hb[k])}"),
                 Cell("{:.2f}", O[k]["v"]), S("supported" if ok else "fails")]
    out["Table 3"] = t3
    nm = lambda r: 0.0 if r["names_a_action"] else 1.0
    out["Table 3b"] = {k: [AU(M.auc(rs, lambda r: r["note_tokens"], flip)), AU(M.auc(rs, nm, flip))]
                       for k, rs in sets.items()}
    return out


def tables_v4(D: Data) -> dict:
    M, v3 = D.M, D.v3
    ex, st, sn = D.arm(D.v4, "extractive"), D.arm(D.v4, "stripped"), D.arm(D.v4, "sonnet")
    b3, b6 = D.arm(D.v4, "budget300"), D.arm(D.v4, "budget600")
    xver = lambda r: bool(r["x_restored_verified"])
    xfloor = lambda r: bool(r["x_floor_flip"])
    out: dict = {}
    t4 = {}
    for name, rows, fn in (("Residue rate, extractive", ex, xflip), ("Recoverable, extractive", ex, xrec),
                           ("Recoverable, extractive, verified", ex, xver),
                           ("Residue rate, advice-stripped", st, xflip), ("Recoverable, paraphrased (v3)", v3, rec)):
        t4[name] = [N(sum(map(fn, rows))), P(M.ratio(rows, fn)), P(M.diff(rows, fn, floor))]
    t4["Rewording floor (v3)"] = [N(sum(map(floor, v3))), P(M.ratio(v3, floor)), DASH]
    out["Table 4"] = t4
    a3 = M.auc(v3, pres, flip)
    out["Table 5"] = {
        "Haiku 5.5 (v3)": [P(M.ratio(v3, flip)), P(M.ratio(v3, floor)), P(M.diff(v3, flip, floor)), AU(a3)],
        "Sonnet 5.5": [P(M.ratio(sn, xflip)), P(M.ratio(sn, xfloor)), P(M.diff(sn, xflip, xfloor)),
                       AU(M.auc(sn, pres, xflip))]}
    mean = lambda rows, f: Cell("{:.1f}", sum(f(r) for r in rows) / len(rows))
    out["Table 6"] = {"150": [P(M.ratio(v3, flip)), DASH, mean(v3, lambda r: r["note_tokens"]), AU(a3)]} | {
        b: [P(M.ratio(rows, xflip)), P(M.diff(rows, xflip, flip)), mean(rows, lambda r: r["x_note_tokens"]),
            AU(M.auc(rows, xpres, xflip))] for b, rows in (("300", b3), ("600", b6))}
    return out


def _c3(r: dict) -> dict:
    reps = [r[f"x_rep{i}"] for i in range(5)]
    six = [r["a_B"]] + reps
    pairs = [(i, j) for i in range(6) for j in range(i + 1, 6)]
    return {"n": 5, "flips": sum(x != r["a_A"] for x in reps), "dis_v3": sum(x != r["a_B"] for x in reps),
            "pair": sum(six[i] != six[j] for i, j in pairs) / len(pairs), "unan": len(set(six)) == 1,
            "reverse": sum(x == r["a_A"] for x in reps) if r["flip"] else 0,
            "new": bool(not r["flip"] and any(x != r["a_A"] for x in reps))}


def _c1(D: Data) -> list[dict]:
    """The v3 flips with their C1 control outcomes. Matched: sent, inside the length band and answered."""
    by: dict[tuple[str, str], list[dict]] = {}
    for c in D.c1:
        by.setdefault((c["case_id"], c["system"]), []).append(c)
    out = []
    for r in D.v3:
        if not r["flip"]:
            continue
        cs = by.get((r["case_id"], r["system"]), [])
        m = [c for c in cs if c["send"] and c["matched"] and c["restored"] is not None]
        sent = [c for c in cs if c["restored"] is not None]
        out.append({**r, "t": int(r["restored"] == "restored"), "assessed": bool(m), "n_matched": len(m),
                    "c": sum(c["restored"] for c in m) / len(m) if m else 0.0,
                    "any": any(c["restored"] for c in m), "has_sent": bool(sent),
                    "c_all": sum(c["restored"] for c in sent) / len(sent) if sent else 0.0})
    return out


def tables_v5(D: Data) -> dict:
    M, v3 = D.M, D.v3
    out: dict = {}
    c2, c3 = D.arm(D.v5, "C2"), [dict(r, **_c3(r)) for r in D.arm(D.v5, "C3")]
    ok2 = [r for r in c2 if r["x_rewrite_ok"]]
    fl3, non3 = [r for r in c3 if r["flip"]], [r for r in c3 if not r["flip"]]
    r_v3 = M.ratio(v3, flip)
    rw = M.ratio(ok2, xflip)
    rep = M.ratio(c3, lambda r: r["flips"], lambda r: r["n"])
    dis = M.ratio(c3, lambda r: r["dis_v3"], lambda r: r["n"])
    un = M.ratio(c3, lambda r: r["unan"])
    rv = M.ratio(fl3, lambda r: r["reverse"] > 0)
    nw = M.ratio(non3, lambda r: r["new"])
    out["Table 7"] = {
        "Residue rate, v3 notes": [OF(r_v3["k"], r_v3["n"]), P(r_v3), DASH],
        "Flip rate, rewritten notes (C2)": [OF(rw["k"], rw["n"]), P(rw), P(M.diff(ok2, xflip, flip))],
        "Flip rate, five repeats (C3)": [OF(rep["k"], rep["n"]), P(rep),
                                         P(M.ratio_diff(c3, lambda r: r["flips"], lambda r: r["n"], flip, one))],
        "Repeat answer differs from the v3 answer": [OF(dis["k"], dis["n"]), P(dis), DASH],
        "Pairwise disagreement among six answers": [DASH, P(M.ratio(c3, lambda r: r["pair"])), DASH],
        "All six answers identical": [OF(un["k"], un["n"]), P(un), DASH],
        "v3 flips reversed by at least one repeat": [OF(rv["k"], rv["n"]), P(rv), DASH],
        "v3 non-flips flipped by at least one repeat": [OF(nw["k"], nw["n"]), P(nw), DASH]}

    fl = _c1(D)
    ass = [r for r in fl if r["assessed"]]
    t, c = (lambda r: r["t"]), (lambda r: r["c"])
    row = lambda rows, f, restored: [N(len(rows)), restored, P(M.ratio(rows, f))]
    a_rec, a_not = [r for r in ass if r["t"]], [r for r in ass if not r["t"]]
    f1 = lambda rows: Cell("{:.1f}", sum(map(c, rows)))
    out["Table 8"] = {
        "Targeted, v3 search, all flips": row(fl, t, N(sum(map(t, fl)))),
        "Targeted, assessed flips": row(ass, t, N(sum(map(t, ass)))),
        "Irrelevant, assessed flips": row(ass, c, f1(ass)),
        "Targeted minus irrelevant": [N(len(ass)), DASH, P(M.ratio_diff(ass, t, one, c, one))],
        "Irrelevant, flips v3 called recoverable": row(a_rec, c, f1(a_rec)),
        "Irrelevant, flips v3 called not recoverable": row(a_not, c, f1(a_not))}

    # Table 9: r_q = omission + wording + unattributed, in points of the eligible handoffs
    c1by = {(r["case_id"], r["system"]): r for r in fl}
    c2by = {(r["case_id"], r["system"]): r for r in c2}
    dec = []
    for r in c3:
        k = (r["case_id"], r["system"])
        a, h = c1by.get(k), bool(c2by[k]["x_rewrite_ok"])
        ass_ = bool(a and a["assessed"])
        dec.append({"case_id": r["case_id"], "flips": int(r["flip"]), "t": a["t"] if ass_ else 0,
                    "c": a["c"] if ass_ else 0.0, "n_rw": int(h), "rw": int(bool(c2by[k]["x_flip"])) if h else 0,
                    "rep": r["flips"] / r["n"] if h else 0.0})
    feats = {k: (lambda r, k=k: r[k]) for k in ("flips", "t", "c", "n_rw", "rw", "rep")} | {"n": one}

    def term(name):
        def f(x):
            if not x["n"] or not x["n_rw"]:
                return None
            rq, om, wd = x["flips"] / x["n"], (x["t"] - x["c"]) / x["n"], (x["rw"] - x["rep"]) / x["n_rw"]
            return {"r_q": rq, "omission": om, "wording": wd, "unattributed": rq - om - wd}[name]
        return P(M.stat(dec, feats, f))
    out["Table 9"] = {"Residue r_q": [S("v3 residue rate"), term("r_q")],
                      "Omission": [S("C1, targeted minus irrelevant"), term("omission")],
                      "Wording": [S("C2 net of C3"), term("wording")],
                      "Unattributed": [S("remainder"), term("unattributed")]}

    # Table 10: three judges of semantic absence
    def bal_acc(name):
        # judge v1 answered the unmodified real pairs only; the other two answered every pair
        rs = [r for r in D.cal if name != "haiku_v1" or not r["note_modified"]]
        pred = [(not r["truth"]) if r[name] is None else bool(r[name]) for r in rs]   # no answer counts as wrong
        pos, neg = [p for p, r in zip(pred, rs) if r["truth"]], [p for p, r in zip(pred, rs) if not r["truth"]]
        return Cell("{:.2f}", (sum(pos) / len(pos) + sum(not p for p in neg) / len(neg)) / 2)
    t10 = {}
    for label, arm, cal in (("Haiku 5.5, judge v1", "C4-judge1", "haiku_v1"),
                            ("qwen3:30b-a3b, judge v2", "C4-judge2", "qwen30"),
                            ("nomic-embed-text, cosine", "C4-embed", "embedding")):
        rs = D.arm(D.v5, arm)
        so = lambda r: r["x_n_absent"] / r["x_n_judged"] if r["x_n_judged"] else None
        hasd = [r for r in rs if r["x_d_n"]]
        dcell = lambda rows: OF(sum(r["x_d_conveyed"] for r in rows), sum(r["x_d_n"] for r in rows))
        mean = M.ratio(rs, lambda r: so(r) or 0.0, lambda r: so(r) is not None)
        t10[label] = [bal_acc(cal), Cell("{:.2f} [{:.2f}, {:.2f}]", mean["v"], *mean["ci"]), AU(M.auc(rs, so, flip)),
                      dcell([r for r in hasd if r["flip"]]), dcell([r for r in hasd if not r["flip"]])]
    t10["v3 writer p_residue, same rows"] = [DASH, DASH, AU(M.auc(v3, pres, flip)), DASH, DASH]
    out["Table 10"] = t10

    x1, x2, x2b = D.arm(D.v5, "X1"), D.arm(D.v5, "X2"), D.arm(D.v5, "X2b")
    a3 = M.auc(v3, pres, flip)
    mt = lambda rows, key: Cell("{:.1f}", sum(r[key] for r in rows) / len(rows))
    d_auc = M.auc_diff(x1, xpres, xflip, pres, flip)
    out["Table 11"] = {
        "v3 note prompt": [N(sum(map(flip, v3))), P(r_v3), DASH, mt(v3, "note_tokens"), AU(a3), DASH],
        "Reader-conditioned (X1)": [N(sum(map(xflip, x1))), P(M.ratio(x1, xflip)), P(M.diff(x1, xflip, flip)),
                                    mt(x1, "x_note_tokens"), AU(M.auc(x1, xpres, xflip)),
                                    Cell("{:.2f} [{:.3f}, {:.2f}]", d_auc["v"], *d_auc["ci"])]}

    from lens.flip.explore5 import prompts as P5, x2b_prompts as P5B
    asks = lambda fields: S("yes" if any("action" in f for f in fields) else "no")
    r2, r2b = M.ratio(x2, xflip), M.ratio(x2b, xflip)
    tok_b = [r["x_note_tokens"] for r in x2b]
    out["Table 12"] = {
        "Flips": [OF(r_v3["k"], r_v3["n"]), OF(r2["k"], r2["n"]), OF(r2b["k"], r2b["n"])],
        "Residue rate": [P(r_v3), P(r2), P(r2b)],
        "Minus v3": [DASH, P(M.diff(x2, xflip, flip)), P(M.diff(x2b, xflip, flip))],
        "Notes cut at 150 tokens": [Cell("{:,d} at the cap", sum(r["note_tokens"] == NOTE_CAP for r in v3)),
                                    N(sum(bool(r["x_cut"]) for r in x2)), N(sum(bool(r["x_cut"]) for r in x2b))],
        "Mean note tokens": [mt(v3, "note_tokens"), mt(x2, "x_note_tokens"),
                             Cell("{:.1f} (max {:,d})", sum(tok_b) / len(tok_b), max(tok_b))],
        "Asks for an action": [asks(()), asks(P5.X2_FIELDS), asks(P5B.FIELDS)],
        "Writer AUROC": [AU(a3), AU(M.auc(x2, xpres, xflip)), AU(M.auc(x2b, xpres, xflip))],
        "Recoverable": [P(M.ratio(v3, rec)), P(M.ratio(x2, xrec)), P(M.ratio(x2b, xrec))]}
    return out


def _v6(r: dict) -> dict:
    """The per-handoff quantities of protocol v6. Everything is zero on a handoff that is not a flip.

    t_fresh: share of the five fresh re-reads of the minimal restoring set that restore (0 when the search did not
    restore). t_search: 1 when the search restored. c: share of the matched and sent controls that restore.
    The p_ fields are the same for the paraphrased v3 notes, with the frozen v5 C1 controls."""
    share = lambda k, n: k / n if n else 0.0
    f = bool(r["e1_flip"])
    ass, sent = bool(f and r["e1_ctl_matched"]), bool(f and r["e1_ctl_sent"])
    rest = bool(f and r["e1_search_restored"])
    v3f = bool(r["v3_flip"])
    p_ass, p_rest = bool(v3f and r["c1v5_matched"]), bool(v3f and r["v3_search_restored"])
    return {**r, "flip": f, "assessed": ass, "has_sent": sent, "t_search": int(rest),
            "t_fresh": share(r["e1_fresh_restored"], r["e1_fresh_n"]) if rest else 0.0,
            "c": share(r["e1_ctl_matched_restored"], r["e1_ctl_matched"]) if ass else 0.0,
            "c_all": share(r["e1_ctl_sent_restored"], r["e1_ctl_sent"]) if sent else 0.0,
            "any": bool(ass and r["e1_ctl_matched_restored"]),
            "p_assessed": p_ass, "p_t_search": int(p_rest),
            "p_t_fresh": share(r["p1_fresh_restored"], r["p1_fresh_n"]) if p_rest else 0.0,
            "p_c": share(r["c1v5_matched_restored"], r["c1v5_matched"]) if p_ass else 0.0}


def table13(D: Data) -> dict:
    M = D.M
    rows = [_v6(r) for r in D.v6]
    assert all(D.case_ids[r["case_index"]] == r["case_id"] for r in rows), "v6 case order is not the frozen one"
    flips = [r for r in rows if r["flip"]]
    ass = [r for r in flips if r["assessed"]]
    rest = [r for r in flips if r["t_search"]]
    p_rest = [r for r in rows if r["p_t_search"]]
    p_ass = [r for r in rows if r["p_assessed"]]
    fl = lambda r: r["v3_floor_flip"]
    fsum = lambda rows_, f: Cell("{:.1f} of {:,d}", sum(map(f, rows_)), len(rows_))
    tf, ts, c = (lambda r: r["t_fresh"]), (lambda r: r["t_search"]), (lambda r: r["c"])
    spec = lambda r: r["assessed"] * (r["t_fresh"] - r["c"])
    spec_s = lambda r: r["assessed"] * (r["t_search"] - r["c"])
    over = lambda f: [Cell("{:.1f} of {:,d}", sum(map(f, rows)), len(rows)), P(M.ratio(rows, f)), P(M.diff(rows, f, fl))]
    r_search, r_fresh = M.ratio(rows, ts), M.ratio(rest, lambda r: r["e1_fresh_restored"], lambda r: r["e1_fresh_n"])
    r_any, r_floor = M.ratio(ass, lambda r: r["any"]), M.ratio(rows, fl)
    r_p = M.ratio(p_rest, lambda r: r["p1_fresh_restored"], lambda r: r["p1_fresh_n"])
    t13 = {
        "Recoverable, search outcome (v4)": [OF(r_search["k"], r_search["n"]), P(r_search), P(M.diff(rows, ts, fl))],
        "Fresh re-reads that restore": [OF(r_fresh["k"], r_fresh["n"]), P(r_fresh), DASH],
        "Targeted restore, fresh, assessed flips": [fsum(ass, tf), P(M.ratio(ass, tf)), DASH],
        "Irrelevant restore, assessed flips": [fsum(ass, c), P(M.ratio(ass, c)), DASH],
        "Targeted minus irrelevant": [N(len(ass)), P(M.ratio_diff(ass, tf, one, c, one)), DASH],
        "At least one control restores": [OF(r_any["k"], r_any["n"]), P(r_any), DASH],
        "Content-specific recoverable rate": over(spec),
        "Same, search outcome as targeted": over(spec_s),
        "Rewording floor (v3)": [OF(r_floor["k"], r_floor["n"]), P(r_floor), DASH],
        "Paraphrased notes: fresh re-reads that restore": [OF(r_p["k"], r_p["n"]), P(r_p), DASH],
        "Paraphrased notes: omission term, search outcome (v5)": [
            DASH, P(M.ratio(rows, lambda r: r["p_assessed"] * (r["p_t_search"] - r["p_c"]))), DASH],
        "Paraphrased notes: omission term, fresh re-reads": [
            DASH, P(M.ratio(rows, lambda r: r["p_assessed"] * (r["p_t_fresh"] - r["p_c"]))), DASH]}
    emb = [r for r in ass if r["system"] == "rag-embed"]
    all5 = sum(r["e1_fresh_n"] > 0 and r["e1_fresh_restored"] == r["e1_fresh_n"] for r in rest)
    none = sum(r["e1_fresh_n"] > 0 and r["e1_fresh_restored"] == 0 for r in rest)
    text = {
        "Minimal sets that restore on all five re-reads": [OF(all5, len(rest))],
        "Minimal sets that restore on no re-read": [N(none)],
        "Content-specific rate, every sent control": [P1(M.ratio(rows, lambda r: r["has_sent"] * (r["t_fresh"] - r["c_all"]))["v"])],
        "rag-embed, irrelevant restore, assessed flips": [P1(M.ratio(emb, c)["v"])],
        "rag-embed, targeted restore, fresh, assessed flips": [P1(M.ratio(emb, tf)["v"])],
        "Paraphrased notes, targeted restore, fresh, assessed flips": [P1(M.ratio(p_ass, lambda r: r["p_t_fresh"])["v"])],
        "Paraphrased notes, irrelevant restore, assessed flips": [P1(M.ratio(p_ass, lambda r: r["p_c"])["v"])]}
    return {"Table 13": t13, "Table 13 text": text}


def _v7(r: dict) -> dict:
    """Outcomes of protocol v7 for one gated handoff. A flip: B's action differs from A's action. A deferral
    flip: a flip to REQUEST_INFO. A commit flip: a flip to one of the four other actions. On an A-wrong
    handoff, rescue: B takes the right action; repeat: B takes A's wrong action."""
    commits = ("EXPLAIN", "OPEN_REVIEW", "CHECK_DEVICE", "APPLY_CREDIT")
    d = dict(r)
    if r["set"] == "eligible":
        for name in ("v3", "x1", "r1", "r2", "r3", "x1p", "x1r"):
            a = r[f"{name}_a_B"]
            fl = a is not None and a != r["a_A"]
            d.update({f"{name}_flip": int(fl), f"{name}_defer": int(fl and a == DEFER),
                      f"{name}_commit": int(fl and a in commits)})
        d["correct_v3"], d["correct_x1"] = int(r["v3_a_B"] == r["a_star"]), int(r["x1_a_B"] == r["a_star"])
    else:
        for name in ("wv3", "wx1"):
            a = r[f"{name}_a_B"]
            d.update({f"{name}_rescue": int(a == r["a_star"]), f"{name}_repeat": int(a == r["a_A"])})
        d["correct_v3"], d["correct_x1"] = d["wv3_rescue"], d["wx1_rescue"]
    return d


def tables_v7(D: Data) -> dict:
    M = D.M
    rows = [_v7(r) for r in D.v7]
    assert all(D.case_ids[r["case_index"]] == r["case_id"] for r in rows), "v7 case order is not the frozen one"
    el = [r for r in rows if r["set"] == "eligible"]
    wr = [r for r in rows if r["set"] == "a_wrong"]
    g = lambda key: (lambda r: r[key])
    c3n = g("c3_n")

    def arm(name, n_changed):
        return [n_changed, P(M.ratio(el, g(f"{name}_flip"))), P(M.diff(el, g(f"{name}_flip"), g("v3_flip"))),
                P(M.ratio(el, g(f"{name}_defer"))), P(M.diff(el, g(f"{name}_defer"), g("v3_defer")))]
    t14 = {
        "v3 notes, v1 reader prompt": [DASH, P(M.ratio(el, g("v3_flip"))), DASH, P(M.ratio(el, g("v3_defer"))), DASH],
        "R1, reader told the text is a note": arm("r1", N(len(el))),
        "R2, hedging sentences removed": arm("r2", N(sum(bool(r["r2_changed"]) for r in el))),
        "R3, trailing fragment removed": arm("r3", N(sum(bool(r["r3_changed"]) for r in el))),
        "Five repeats of the v3 note (C3)": [
            DASH, P(M.ratio(el, g("c3_flip"), c3n)), P(M.ratio_diff(el, g("c3_flip"), c3n, g("v3_flip"), one)),
            P(M.ratio(el, g("c3_defer"), c3n)), P(M.ratio_diff(el, g("c3_defer"), c3n, g("v3_defer"), one))]}
    xrow = lambda name: [P(M.ratio(el, g(f"{name}_flip"))), P(M.diff(el, g(f"{name}_flip"), g("v3_flip"))), DASH, DASH]
    t15 = {
        "v3 note prompt": [P(M.ratio(el, g("v3_flip"))), DASH, P(M.ratio(wr, g("wv3_rescue"))), P(M.ratio(rows, g("correct_v3")))],
        "Policy only (X1p)": xrow("x1p"),
        "Reader description only (X1r)": xrow("x1r"),
        "Both (X1)": [P(M.ratio(el, g("x1_flip"))), P(M.diff(el, g("x1_flip"), g("v3_flip"))),
                      P(M.ratio(wr, g("wx1_rescue"))), P(M.ratio(rows, g("correct_x1")))],
        "X1 minus v3": [DASH, DASH, P(M.diff(wr, g("wx1_rescue"), g("wv3_rescue"))),
                        P(M.diff(rows, g("correct_x1"), g("correct_v3")))]}
    cut, unc = [r for r in el if r["cut"]], [r for r in el if not r["cut"]]
    grp = lambda rs: Cell("{:.1f} ({:,d} of {:,d})", 100 * sum(r["v3_flip"] for r in rs) / len(rs),
                          sum(r["v3_flip"] for r in rs), len(rs))
    ic, iu = (lambda r: int(bool(r["cut"]))), (lambda r: int(not r["cut"]))
    requests = sum(v["requests"] for a in D.v7_spend["arms"].values() for v in a["by_stage"].values())
    text = {
        "v3 flip rate, notes cut by the cap": [grp(cut)],
        "v3 flip rate, notes not cut": [grp(unc)],
        "Cut minus not cut": [P(M.ratio_diff(el, lambda r: r["v3_flip"] * ic(r), ic, lambda r: r["v3_flip"] * iu(r), iu))],
        "R1 commit flips minus v3": [P(M.diff(el, g("r1_commit"), g("v3_commit")))],
        "R2, notes changed": [N(sum(bool(r["r2_changed"]) for r in el))],
        "R3, notes changed": [N(sum(bool(r["r3_changed"]) for r in el))],
        "X1p minus X1": [P(M.diff(el, g("x1p_flip"), g("x1_flip")))],
        "X1r minus X1": [P(M.diff(el, g("x1r_flip"), g("x1_flip")))],
        "X1p minus X1r": [P(M.diff(el, g("x1p_flip"), g("x1r_flip")))],
        "Reader repeats A's wrong action, v3 notes": [P1(M.ratio(wr, g("wv3_repeat"))["v"])],
        "Reader repeats A's wrong action, X1 notes": [P1(M.ratio(wr, g("wx1_repeat"))["v"])],
        "A-wrong handoffs": [N(len(wr))],
        "Gated handoffs": [N(len(rows))],
        "v7 spend in USD": [Cell("{:.2f}", D.v7_spend["total_usd"])],
        "v7 requests": [N(requests)]}
    return {"Table 14": t14, "Table 15": t15, "Tables 14 and 15 text": text}


def figure2(D: Data) -> dict:
    M, v3 = D.M, D.v3
    ex, st, sn = D.arm(D.v4, "extractive"), D.arm(D.v4, "stripped"), D.arm(D.v4, "sonnet")
    by = lambda rows, s: [r for r in rows if r["system"] == s]
    nm = lambda r: 0.0 if r["names_a_action"] else 1.0
    rows = {
        "A: rewording floor": [P(M.ratio(v3, floor))],
        "A: paraphrased, flips": [P(M.ratio(v3, flip))],
        "A: paraphrased, recoverable": [P(M.ratio(v3, rec))],
        "A: advice-stripped, flips": [P(M.ratio(st, xflip))],
        "A: extractive, flips": [P(M.ratio(ex, xflip))],
        "A: extractive, recoverable": [P(M.ratio(ex, xrec))],
        "B: Haiku, extractive": [AU(M.auc(ex, xpres, xflip))],
        "B: Sonnet, recent, flips": [N(sum(map(xflip, by(sn, "recent"))))],
        "B: names-action flag, pooled": [Cell("{:.2f}", M.auc(v3, nm, flip)["v"])],
    }
    for s in SYSTEMS:
        rows[f"B: Haiku, {s}"] = [AU(M.auc(by(v3, s), pres, flip))]
        rows[f"B: Sonnet, {s}"] = [AU(M.auc(by(sn, s), pres, xflip))]
    for b, rs, f, p in (("150", v3, flip, pres), ("300", D.arm(D.v4, "budget300"), xflip, xpres),
                        ("600", D.arm(D.v4, "budget600"), xflip, xpres)):
        rows[f"C: {b} tokens, flips"] = [P(M.ratio(rs, f))]
        rows[f"C: {b} tokens, writer AUROC"] = [AU(M.auc(rs, p, f))]
    return {"Figure 2": rows}


def rebuild(D: Data) -> dict:
    return table1(D) | tables_v3(D) | tables_v4(D) | tables_v5(D) | table13(D) | tables_v7(D) | figure2(D)


# ---------------------------------------------------------------------------
# Paper facts
# ---------------------------------------------------------------------------

def _pt(r: dict, scale: float = 100.0, nd: int = 1) -> dict:
    return {"value": round(scale * r["v"], nd), "bounds95": [round(scale * r["ci"][0], nd), round(scale * r["ci"][1], nd)]}


def contrasts(D: Data, expected: dict) -> dict:
    """Exploratory contrasts of Sections 6.5.1 to 6.5.8 that carry bounds: differences between two measured
    quantities. Table cells are counted from the expected-values file. Text-only contrasts are rebuilt here."""
    M, v3 = D.M, D.v3
    in_tables = []
    for t in ("Table 4", "Table 5", "Table 6", "Table 7", "Table 8", "Table 9", "Table 11", "Table 12"):
        cols = expected[t]["columns"]
        for row, cells in expected[t]["rows"].items():
            for col, cell in zip(cols, cells):
                is_diff = "minus" in (row + " " + col).lower() or (t == "Table 9" and row != "Residue r_q" and col == "Points")
                if is_diff and "[" in cell:
                    in_tables.append(f"{t} | {row} | {col}")
    restated_v3 = [x for x in in_tables if "(v3)" in x]   # frozen-run contrasts shown again for reference
    ex, st, sn = D.arm(D.v4, "extractive"), D.arm(D.v4, "stripped"), D.arm(D.v4, "sonnet")
    b3, b6 = D.arm(D.v4, "budget300"), D.arm(D.v4, "budget600")
    x2, x2b = D.arm(D.v5, "X2"), D.arm(D.v5, "X2b")
    by2 = {(r["case_id"], r["system"]): r for r in x2}
    by6 = {(r["case_id"], r["system"]): r for r in b6}
    k = lambda r: (r["case_id"], r["system"])
    changed = [r for r in st if r["x_note_tokens"] != r["note_tokens"]]
    sens = [r for r in _c1(D) if r["has_sent"]]
    j2 = D.arm(D.v5, "C4-judge2")
    so = lambda r: r["x_n_absent"] / r["x_n_judged"] if r["x_n_judged"] else None
    text = {
        "6.5.1 advice-stripped minus v3 residue rate": _pt(M.diff(st, xflip, flip)),
        "6.5.1 the same on the notes the rule changed": {**_pt(M.diff(changed, xflip, flip)), "n": len(changed)},
        "6.5.2 Sonnet minus Haiku residue rate": _pt(M.diff(sn, xflip, flip)),
        "6.5.3 600 minus 300 tokens": _pt(M.diff(b3, lambda r: bool(by6[k(r)]["x_flip"]), xflip)),
        "6.5.5 targeted minus irrelevant, every sent control": {
            **_pt(M.ratio_diff(sens, lambda r: r["t"], one, lambda r: r["c_all"], one)), "n": len(sens)},
        "6.5.6 writer AUROC minus semantic-omission AUROC, judge v2": _pt(M.auc_diff(j2, pres, flip, so, flip), 1.0, 2),
        "6.5.8 X2b minus X2 residue rate": _pt(M.diff(x2b, xflip, lambda r: bool(by2[k(r)]["x_flip"]))),
        "6.5.8 X2b recoverable minus the v3 floor": _pt(M.diff(x2b, xrec, floor)),
    }
    n = len(in_tables) - len(restated_v3) + len(text)
    return {"in_tables": in_tables, "in_tables_restating_v3": restated_v3, "in_text_only": text,
            "n_in_tables": len(in_tables), "n_in_text_only": len(text), "n_distinct_exploratory": n,
            "bonferroni_two_sided_z_at_0.05": round(NormalDist().inv_cdf(1 - 0.025 / n), 2)}


def facts(D: Data, expected: dict) -> dict:
    M, v3 = D.M, D.v3
    F: dict = {}
    cases = _jsonl(D.cases_path)
    assert [c["case_id"] for c in cases] == D.case_ids, "cases.jsonl and the stage 1 file disagree on case order"
    types = [{e["type"] for e in c["events"]} for c in cases]
    all_types = sorted(set().union(*types))
    F["histories_with_all_event_types"] = {
        "value": sum(len(t) == len(all_types) for t in types), "of": len(cases), "event_types": all_types,
        "method": "cases.jsonl: histories whose events cover every event type that occurs in the sample"}
    F["histories_with_a_correction_event"] = {
        "value": sum("correction" in t for t in types), "of": len(cases),
        "method": "cases.jsonl: histories with at least one event of type correction"}
    F["histories_without_plan_notice"] = {
        "value": sum("plan_notice" not in t for t in types), "of": len(cases),
        "method": "cases.jsonl: histories with no event of type plan_notice"}

    from lens.flip.headline.engine import make_headline_sample
    regen = {11: {c.case_id: c.to_dict() for c in make_headline_sample(300, 11, generator_version=2)},
             12: {c.case_id: c.to_dict() for c in make_headline_sample(30, 12, mechanisms=("U2",), prefix="u3",
                                                                       generator_version=3)}}
    split = {s: [c for c in cases if g.get(c["case_id"]) == c] for s, g in regen.items()}
    F["seed_split"] = {
        "value": {"seed_11": len(split[11]), "seed_12": len(split[12])}, "of": len(cases),
        "seed_12_mechanisms": dict(Counter(c["mechanism"] for c in split[12])),
        "method": "each case regenerated from seed 11 (generator version 2) and seed 12 (U2 only, version 3) and "
                  "compared field by field with cases.jsonl"}

    from lens.flip.generator import oracle
    from lens.flip.schema import Case
    objs = [Case.from_dict(c) for c in cases]
    agree = sum(oracle(c.events, c.query_period) == c.a_star for c in objs)
    rare = [c for c in objs if c.mechanism in ("M1", "M2", "M3")]
    changed = sum(oracle([e for e in c.events if e.id not in set(c.d_star)], c.query_period) != c.a_star for c in rare)
    F["rule_oracle_on_the_headline_cases"] = {
        "value": {"agree": agree, "disagree": len(objs) - agree, "of": len(objs),
                  "rare_cases_where_deleting_the_planted_events_changes_the_action": changed, "rare_cases": len(rare)},
        "tests_on_generated_samples": {
            "tests/flip/test_generator.py::test_oracle_matches_a_star": "2,280 generated cases (seeds 1000 to 1011 and 77)",
            "tests/flip/test_protocol_v3.py::test_oracle_agrees_on_every_v3_case": "2,160 generated cases (seeds 5000 to 5011)"},
        "test_on_the_headline_cases": "tests/flip/test_rebuild_paper.py::test_oracle_agrees_on_the_300_headline_cases",
        "method": "generator.oracle applied to the events of each of the 300 cases in cases.jsonl and compared with "
                  "a_star; for M1 to M3 the planted events are deleted first and the action must change"}

    one_note = [r for r in v3 if r["case_id"] == "h11-0184" and r["system"] == "rag-embed"][0]
    tok = lambda chars: math.ceil(chars / 2.2)
    F["note_h11_0184_rag_embed"] = {
        "value": {"note_tokens": one_note["note_tokens"], "ends_mid_sentence": not one_note["note_ends_sentence"],
                  "cut_by_the_cap": one_note["note_cut_at_cap"], "reply_tokens_before_the_cut": tok(one_note["raw_note_chars"])},
        "cap": NOTE_CAP,
        "method": "handoff_metrics_v3.jsonl: tokens are ceil(characters / 2.2); cut when the writer's reply was "
                  "longer than the stored note; the cut backs up to the last whole word, so a cut note can be under 150"}
    cut = [r for r in v3 if r["note_cut_at_cap"]]
    F["v3_notes_cut_by_the_cap"] = {
        "value": {"cut": len(cut), "cut_and_exactly_at_cap": sum(r["note_tokens"] == NOTE_CAP for r in cut),
                  "cut_and_within_2_tokens_of_cap": sum(r["note_tokens"] >= NOTE_CAP - 2 for r in cut),
                  "end_mid_sentence": sum(not r["note_ends_sentence"] for r in v3),
                  "cut_and_end_mid_sentence": sum(not r["note_ends_sentence"] for r in cut)}, "of": len(v3),
        "method": "v3 notes whose reply was longer than the stored note (note_cut_at_cap); a note ends mid-sentence "
                  "when its last character is not a full stop, question mark or exclamation mark"}
    F["v3_notes_near_the_cap"] = {
        "value": {"within_2_tokens_of_cap": sum(r["note_tokens"] >= NOTE_CAP - 2 for r in v3),
                  "exactly_at_cap": sum(r["note_tokens"] == NOTE_CAP for r in v3)}, "of": len(v3),
        "method": "v3 notes with note_tokens of 148 to 150, and with exactly 150"}

    j2 = {(r["case_id"], r["system"]): r for r in D.v5 if r["arm"] == "C4-judge2"}
    ev = [r for r in v3 if r["flip"] and r["system"] in EVENT_SYSTEMS]
    h2b = [r for r in ev if r["dstar_in_view"]]
    dchk = [r for r in v3 if r["flip"] and j2[(r["case_id"], r["system"])]["d_n"]]
    key = lambda r: (r["case_id"], r["system"])
    extra = [r for r in dchk if key(r) not in set(map(key, h2b))]
    F["h2b_flips_against_dstar_check_flips"] = {
        "value": {"event_view_flips": len(ev), "h2b_every_planted_event_in_view": len(h2b),
                  "h2b_with_a_restored_search": sum(r["restored"] == "restored" for r in h2b),
                  "dstar_check_at_least_one_planted_unit_in_view": len(dchk),
                  "in_both": len(dchk) - len(extra), "dstar_check_only": len(extra),
                  "dstar_check_only_detail": [{"case_id": r["case_id"], "system": r["system"],
                                               "mechanism": r["mechanism"],
                                               "planted_units_in_view": j2[key(r)]["d_n"]} for r in extra],
                  "h2b_only": len(h2b) - (len(dchk) - len(extra)),
                  "planted_units_in_dstar_check": sum(j2[key(r)]["d_n"] for r in dchk)},
        "method": "H2b keeps event-view flips whose view holds every planted event; the D* check keeps flips whose "
                  "view holds at least one planted unit. The difference is the flips with only part of D* in view"}

    fd = lambda r: bool(r["floor_flip"] and r["floor_action"] == DEFER)
    fc = lambda r: bool(r["floor_flip"] and r["floor_action"] != DEFER)
    defer = lambda r: bool(r["flip"] and r["a_B"] == DEFER)
    commit = lambda r: bool(r["flip"] and r["a_B"] != DEFER)
    F["floor_split_by_action"] = {
        "value": {"floor_deferral_part": {**_pt(M.ratio(v3, fd)), "k": sum(map(fd, v3)), "n": len(v3)},
                  "floor_commit_part": {**_pt(M.ratio(v3, fc)), "k": sum(map(fc, v3)), "n": len(v3)},
                  "deferral_flips": {**_pt(M.ratio(v3, defer)), "k": sum(map(defer, v3)), "n": len(v3)},
                  "commit_flips": {**_pt(M.ratio(v3, commit)), "k": sum(map(commit, v3)), "n": len(v3)},
                  "deferral_flips_minus_floor_deferral_part": _pt(M.diff(v3, defer, fd)),
                  "commit_flips_minus_floor_commit_part": _pt(M.diff(v3, commit, fc))},
        "unit": "points of the 490 eligible handoffs",
        "method": "the rewording floor split by the floor decision's own action (REQUEST_INFO or not); paired "
                  "bootstrap of flips with that action minus that part of the floor"}

    ex_sum = [r for r in D.arm(D.v4, "extractive") if r["system"] == "summary"]
    F["extractive_recoverable_minus_floor_on_summary_views"] = {
        **_pt(M.diff(ex_sum, xrec, floor)), "n": len(ex_sum), "unit": "points of the eligible summary-view handoffs",
        "recoverable_rate": _pt(M.ratio(ex_sum, xrec)), "floor_rate": _pt(M.ratio(ex_sum, floor)),
        "method": "extractive arm (protocol v4), summary views only: recoverable flips minus the v3 rewording "
                  "floor on the same handoffs, paired bootstrap by case"}

    from lens.flip.explore5 import prompts as P5, x2b_prompts as P5B
    F["structured_form_fields"] = {
        "value": {"X2_protocol_v5": list(P5.X2_FIELDS), "X2b_protocol_v5b": list(P5B.FIELDS)},
        "schema_required": {"X2": list(P5.X2_SCHEMA["required"]), "X2b": list(P5B.X2B_SCHEMA["required"])},
        "method": "field names in order from the frozen reply schemas in src/lens/flip/explore5 (prompts.py and "
                  "x2b_prompts.py); every reply also carries p_residue"}

    fl = _c1(D)
    ass = [r for r in fl if r["assessed"]]
    anyr = M.ratio(ass, lambda r: r["any"])
    F["c1_any_irrelevant_control_restores"] = {
        **_pt(anyr), "k": int(anyr["k"]), "n": len(ass), "unit": "percent of assessed v3 flips",
        "method": "assessed v3 flips where at least one matched irrelevant control restored the action"}
    c3 = [dict(r, **_c3(r)) for r in D.arm(D.v5, "C3") if r["flip"]]
    never = M.ratio(c3, lambda r: r["reverse"] == 0)
    F["v3_flips_never_reversed_in_five_repeats"] = {
        "value": int(never["k"]), "of": len(c3), "share": _pt(never),
        "method": "v3 flips where none of the five C3 repeat decisions equals A's action"}

    nm = lambda r: 0.0 if r["names_a_action"] else 1.0
    F["writer_auroc_minus_names_action_flag_auroc"] = {
        **_pt(M.auc_diff(v3, pres, flip, nm, flip), 1.0, 3), "unit": "AUROC, pooled, v3",
        "method": "AUROC of p_residue minus AUROC of the flag (1 when the note does not name A's action), same "
                  "490 handoffs, paired bootstrap by case"}
    c = contrasts(D, expected)
    F["exploratory_contrasts_with_bounds"] = {
        "value": c["n_distinct_exploratory"], "detail": c,
        "method": "difference cells with bounds in Tables 4 to 12, less the frozen v3 contrasts shown again, plus "
                  "the differences with bounds that Sections 6.5.1 to 6.5.8 give in the text only"}
    return {"schema": "lens.flip.paper_facts/1",
            "bootstrap": {"reps": 1000, "seed": 11, "unit": "case (paired)"}, "facts": F}


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------

def print_tables(tables: dict, expected: dict) -> None:
    for name, rows in tables.items():
        e = expected.get(name, {})
        print(f"\n{name}. {e.get('caption', '')}".rstrip())
        print(" | ".join([""] + e.get("columns", [])))
        for label, cells in rows.items():
            print(" | ".join([label] + [str(c) for c in cells]))


def run_check(tables: dict, expected: dict, quiet: bool) -> tuple[int, int]:
    n_ok = n_bad = 0
    for name, e in expected.items():
        t_ok = t_bad = 0
        for label, exp_cells in e["rows"].items():
            got = tables.get(name, {}).get(label)
            for j, exp in enumerate(exp_cells):
                cell = got[j] if got is not None and j < len(got) else None
                ok = cell is not None and matches(cell, exp)
                t_ok, t_bad = t_ok + ok, t_bad + (not ok)
                if not ok or not quiet:
                    col = e["columns"][j] if j < len(e["columns"]) else str(j)
                    print(f"{'match   ' if ok else 'MISMATCH'} | {name} | {label} | {col} | paper: {exp} | rebuilt: {cell}")
        extra = [r for r in tables.get(name, {}) if r not in e["rows"]]
        if extra:
            print(f"MISMATCH | {name} | rebuilt rows the paper does not have: {extra}")
            t_bad += len(extra)
        print(f"{name}: {t_ok} of {t_ok + t_bad} cells match")
        n_ok, n_bad = n_ok + t_ok, n_bad + t_bad
    return n_ok, n_bad


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--check", action="store_true", help="compare every cell with results/paper_expected.json")
    ap.add_argument("--quiet", action="store_true", help="with --check, print mismatches and totals only")
    ap.add_argument("--write-facts", action="store_true", help="rewrite results/paper_facts.json")
    args = ap.parse_args()
    D = Data()
    expected = json.loads(EXPECTED.read_text(encoding="utf-8"))
    tables = rebuild(D)
    F = facts(D, expected)
    text = json.dumps(F, indent=1, sort_keys=True, ensure_ascii=False) + "\n"
    if args.write_facts:
        FACTS.write_text(text, encoding="utf-8")
        print(f"wrote {FACTS.relative_to(ROOT)}")
    if not args.check:
        print_tables(tables, expected)
        print("\nPaper facts")
        print(text)
        return 0
    n_ok, n_bad = run_check(tables, expected, args.quiet)
    facts_ok = FACTS.exists() and FACTS.read_text(encoding="utf-8") == text
    print(f"paper facts: {'results/paper_facts.json matches the recomputed facts' if facts_ok else 'MISMATCH with results/paper_facts.json'}")
    print(f"TOTAL: {n_ok} of {n_ok + n_bad} cells match, {n_bad} mismatch")
    return 0 if n_bad == 0 and facts_ok else 1


if __name__ == "__main__":
    sys.exit(main())

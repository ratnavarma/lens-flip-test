"""Rebuild the paper's frozen and exploratory tables from the released summary files.

    uv run python scripts/reproduce_tables.py            # print both reports and check them
    uv run python scripts/reproduce_tables.py --quiet    # check only

Part 1. Tables 1 to 4 of the frozen run come from results/headline-300-v3/summary_v3.json (protocol v3).
Tables 4 to 6 of the exploratory run (Tables 5 to 7 in REPORT.md) come from results/exploratory-v4/summary_v4.json (protocol v4).
The check compares the rebuilt text with the stored tables_v3.txt and REPORT.md line by line. Row order
and the order inside inline dicts can differ, because the JSON files store their keys sorted.

Part 2. The paper's own tables 2a, 2b, 2c, 2d, 3 (with the Holm column), 3b and 4 to 6 are rebuilt from the
per-handoff metric files handoff_metrics_v3.jsonl and handoff_metrics_v4.jsonl. Those files hold no
note, view or summary text. The bootstrap is the paper's: seed 11, 1,000 resamples, paired by case.
Each table prints PASS or FAIL against the paper's headline numbers (0.1 point on rates, 0.01 on AUROC).

No model call is made.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter
from pathlib import Path

from lens.flip.explore4 import analysis as explore4
from lens.flip.headline import merge
from lens.flip.headline.metrics import Boot, auroc, ci

ROOT = Path(__file__).resolve().parents[1]
V3 = ROOT / "results" / "headline-300-v3"
V4 = ROOT / "results" / "exploratory-v4"


def rebuild_v3() -> str:
    s = json.loads((V3 / "summary_v3.json").read_text(encoding="utf-8"))
    return merge.tables(s) + "\n" + merge._merge_lines(s["merge"]) + "\n"


def rebuild_v4() -> str:
    s = json.loads((V4 / "summary_v4.json").read_text(encoding="utf-8"))
    return explore4.report(s)


def _norm(line: str) -> tuple[str, ...]:
    # Inline dicts print in key order, which the sorted JSON changes; compare their items as a set.
    return tuple(sorted(x.rstrip(".") for x in line.replace("{", "").replace("}", "").split(", ")))


def same_lines(a: str, b: str) -> bool:
    return Counter(map(_norm, a.splitlines())) == Counter(map(_norm, b.splitlines()))


# ---------------------------------------------------------------------------
# Part 2: tables rebuilt from the per-handoff metric files
# ---------------------------------------------------------------------------

SYSTEMS = ("recent", "rag-bm25", "rag-embed", "summary")
GROUPS = ("M0", "M1-M3", "U1-U2")
DEFER = "REQUEST_INFO"
CHECKS: list[tuple[str, str, bool]] = []


def _jsonl(p: Path) -> list[dict]:
    return [json.loads(x) for x in p.read_text(encoding="utf-8").splitlines() if x.strip()]


class Metrics:
    """Bootstrap statistics over handoff rows, paired by case (seed 11, 1,000 resamples)."""

    def __init__(self, case_ids: list[str]):
        self.idx = {c: i for i, c in enumerate(case_ids)}
        self.n = len(case_ids)
        self.boot = Boot(self.n, 1000, 11)

    def _feats(self, rows: list[dict], fns: dict) -> dict[str, list[float]]:
        f = {k: [0.0] * self.n for k in fns}
        for r in rows:
            for k, fn in fns.items():
                f[k][self.idx[r["case_id"]]] += float(fn(r))
        return f

    def _stat(self, rows, fns, fun):
        f = self._feats(rows, fns)
        s = self.boot.stat(f, fun)
        return s["value"], s["ci95"], s["sums"]

    def rate(self, rows, pred):
        v, c, sums = self._stat(rows, {"k": pred, "n": lambda r: 1}, lambda s: s["k"] / s["n"] if s["n"] else None)
        return {"k": int(sums["k"]), "n": int(sums["n"]), "v": v, "ci": c}

    def diff(self, rows, pa, pb):
        v, c, _ = self._stat(rows, {"a": pa, "b": pb, "n": lambda r: 1},
                             lambda s: (s["a"] - s["b"]) / s["n"] if s["n"] else None)
        return {"v": v, "ci": c}

    def auc_fn(self, rows, score, flag):
        items = [(self.idx[r["case_id"]], (float(score(r)), bool(r[flag]))) for r in rows]
        return self._auc_items(items)

    def _auc_items(self, items):
        by: dict[int, list] = {}
        for i, p in items:
            by.setdefault(i, []).append(p)
        full = auroc(((p, 1.0) for _, p in items))
        vals = []
        for s in self.boot.samples:
            x = auroc(((p, float(w)) for i, w in s for p in by.get(i, [])))
            if x is not None:
                vals.append(x)
        return {"v": full, "ci": ci(vals), "vals": vals, "by": by}


def _p(x, d=1):
    return f"{100 * x:.{d}f}"


def _pc(r):
    if r["v"] is None:
        return "-"
    return f"{_p(r['v'])} [{_p(r['ci'][0])}, {_p(r['ci'][1])}]" if r["ci"] else _p(r["v"])


def _sd(r):
    if r["v"] is None:
        return "-"
    return f"{100 * r['v']:+.1f} [{100 * r['ci'][0]:+.1f}, {100 * r['ci'][1]:+.1f}]"


def _au(a, d=2):
    return f"{a['v']:.{d}f} [{a['ci'][0]:.2f}, {a['ci'][1]:.2f}]"


def _holm(ps: dict[str, float]) -> dict[str, float]:
    order = sorted(ps, key=lambda k: ps[k])
    m, run, adj = len(ps), 0.0, {}
    for j, k in enumerate(order):
        run = max(run, min(1.0, (m - j) * ps[k]))
        adj[k] = run
    return adj


def _near(x, target, tol):
    return x is not None and abs(x - target) <= tol + 1e-9


def _rate_ok(r, k=None, n=None, v=None, lo=None, hi=None, tol=0.1):
    """True when a rate or difference (in points) and its bounds sit within tol of the targets."""
    ok = (k is None or r["k"] == k) and (n is None or r["n"] == n)
    ok &= v is None or _near(100 * r["v"], v, tol)
    ok &= lo is None or _near(100 * r["ci"][0], lo, tol)
    ok &= hi is None or _near(100 * r["ci"][1], hi, tol)
    return bool(ok)


def _verdict(title: str, ok: bool, out: list[str]) -> None:
    CHECKS.append((title, "PASS" if ok else "FAIL", ok))


def rebuild_metrics(quiet: bool) -> bool:
    v3 = _jsonl(V3 / "handoff_metrics_v3.jsonl")
    v4 = _jsonl(V4 / "handoff_metrics_v4.jsonl")
    case_ids = [json.loads(x)["case_id"] for x in (V3 / "cases.jsonl").read_text(encoding="utf-8").splitlines() if x.strip()]
    M = Metrics(case_ids)
    out: list[str] = []
    P = out.append
    rec = lambda r: bool(r["flip"] and r["restored"] == "restored")
    floor = lambda r: bool(r["floor_flip"])
    flip = lambda r: bool(r["flip"])
    sets = {"pooled": v3} | {s: [r for r in v3 if r["system"] == s] for s in SYSTEMS}
    gsets = {"pooled": v3} | {g: [r for r in v3 if r["mechanism_group"] == g] for g in GROUPS}

    # ---- Table 2a
    P("Table 2a. Recoverable flips at B = 1,000, in percent of eligible handoffs.")
    P("System | Recoverable flips | Recoverable rate | Recoverable minus floor")
    t2a = {}
    for k, rs in sets.items():
        t2a[k] = (M.rate(rs, rec), M.diff(rs, rec, floor))
        P(f"{k} | {t2a[k][0]['k']} | {_pc(t2a[k][0])} | {_sd(t2a[k][1])}")
    _verdict("Table 2a", _rate_ok(t2a["pooled"][0], 41, 490, 8.4, 5.9, 11.1)
             and _rate_ok(t2a["pooled"][1], v=-2.0, lo=-5.4, hi=1.8), out)

    # ---- Table 2b
    P("\nTable 2b. Stage 2 by mechanism group at B = 1,000, in percent.")
    P("Group | Eligible | Flips | Residue rate | Rewording floor | Recoverable | Recoverable minus floor")
    t2b = {}
    for k, rs in gsets.items():
        t2b[k] = (M.rate(rs, flip), M.rate(rs, floor), M.rate(rs, rec), M.diff(rs, rec, floor))
        P(f"{k} | {len(rs)} | {t2b[k][0]['k']} | " + " | ".join([_pc(t2b[k][0]), _pc(t2b[k][1]), _pc(t2b[k][2]), _sd(t2b[k][3])]))
    _verdict("Table 2b", t2b["M0"][0]["k"] == 88 and t2b["M0"][0]["n"] == 210
             and _rate_ok(t2b["pooled"][0], 142, 490, 29.0, 24.8, 33.2), out)

    # ---- Table 2c
    P("\nTable 2c. Flips by B's action at B = 1,000, in percent.")
    P("Group | B defers | B commits | Defers minus floor | Commits minus floor")
    defer = lambda r: bool(r["flip"] and r["a_B"] == DEFER)
    commit = lambda r: bool(r["flip"] and r["a_B"] != DEFER)
    t2c = {}
    for k, rs in gsets.items():
        t2c[k] = (M.rate(rs, defer), M.rate(rs, commit), M.diff(rs, defer, floor), M.diff(rs, commit, floor))
        P(f"{k} | " + " | ".join([_pc(t2c[k][0]), _pc(t2c[k][1]), _sd(t2c[k][2]), _sd(t2c[k][3])]))
    P(f"B deferred in {t2c['pooled'][0]['k']} of {t2b['pooled'][0]['k']} flips; "
      f"M0 deferrals {t2c['M0'][0]['k']} of {t2b['M0'][0]['k']}.")
    _verdict("Table 2c", _rate_ok(t2c["pooled"][2], v=11.0, lo=6.5, hi=15.5)
             and _rate_ok(t2c["pooled"][3], v=-2.9, lo=-6.1, hi=0.4), out)

    # ---- Table 2d
    P("\nTable 2d. Recoverability of residue flips at B = 1,000 (counts).")
    P("System | Flips | Restored, 1-minimal | Restored at cap | Not recoverable | B as on truncation | B chose REQUEST_INFO")
    t2d = {}
    for k, rs in sets.items():
        fl = [r for r in rs if r["flip"]]
        rest = [r for r in fl if r["restored"] == "restored"]
        cap = sum(bool(r["restored_cap_hit"]) for r in rest)
        nr = [r for r in fl if r["restored"] != "restored"]
        t_ = sum(r["a_B"] == r["trunc_action"] for r in nr)
        d_ = sum(r["a_B"] == DEFER for r in nr)
        t2d[k] = (len(fl), len(rest) - cap, cap, len(nr), t_, d_)
        P(f"{k} | {len(fl)} | {len(rest) - cap} | {cap} | {len(nr)} ({100 * len(nr) / len(fl):.1f}) | "
          f"{t_} ({100 * t_ / len(nr):.1f}) | {d_} ({100 * d_ / len(nr):.1f})")
    _verdict("Table 2d", t2d["pooled"] == (142, 40, 1, 101, 66, 75), out)

    # ---- Table 3 and 3b
    pr = lambda r: r["p_residue"]
    A = {s: M.auc_fn(rs, pr, "flip") for s, rs in sets.items()}
    holm_in = {}
    P("\nTable 3. H2c at B = 1,000: AUROC of the writer's probability for residue flips.")
    P("System | Flips / non-flips | AUROC of p_residue | Minus 0.5 | Holm p vs 0.5 (normal / bootstrap) | Omission AUROC")
    p_norm, p_boot = {}, {}
    for s in SYSTEMS:
        vals = A[s]["vals"]
        mean = sum(vals) / len(vals)
        se = (sum((x - mean) ** 2 for x in vals) / (len(vals) - 1)) ** 0.5
        z = (A[s]["v"] - 0.5) / se if se else float("inf")
        p_norm[s] = math.erfc(abs(z) / math.sqrt(2))
        below = sum(x <= 0.5 for x in vals) / len(vals)
        above = sum(x >= 0.5 for x in vals) / len(vals)
        p_boot[s] = max(min(1.0, 2 * min(below, above)), 1 / len(M.boot.samples))
    hn, hb = _holm(p_norm), _holm(p_boot)
    fmt = lambda x: "< 0.001" if x < 0.001 else f"{x:.3f}" if x < 0.1 else f"{x:.2f}"
    om = {s: M.auc_fn(rs, lambda r: r["omission_fraction"], "flip") for s, rs in sets.items()}
    for k, rs in sets.items():
        nf = sum(bool(r["flip"]) for r in rs)
        a = A[k]
        d = f"{a['v'] - 0.5:.2f} [{a['ci'][0] - 0.5:.2f}, {a['ci'][1] - 0.5:.2f}]"
        h = "not applied" if k == "pooled" else f"{fmt(hn[k])} / {fmt(hb[k])}"
        P(f"{k} | {nf} / {len(rs) - nf} | {_au(a)} | {d} | {h} | {om[k]['v']:.2f}")
    # size-weighted stratified AUROC
    by_sys = {s: M.auc_fn(sets[s], pr, "flip")["by"] for s in SYSTEMS}
    def strat(ws):
        num = den = 0.0
        for s, w in ws.items():
            a = auroc(w)
            if a is None:
                return None
            n = sum(x for _, x in w)
            num, den = num + n * a, den + n
        return num / den
    full = strat({s: [(p, 1.0) for ps in by_sys[s].values() for p in ps] for s in SYSTEMS})
    sv = []
    for smp in M.boot.samples:
        x = strat({s: [(p, float(w)) for i, w in smp for p in by_sys[s].get(i, [])] for s in SYSTEMS})
        if x is not None:
            sv.append(x)
    sc = ci(sv)
    P(f"Size-weighted mean of the four per-system AUROCs: {full:.3f} [{sc[0]:.2f}, {sc[1]:.2f}]")
    _verdict("Table 3 (AUROC, Holm column)", _near(A["pooled"]["v"], 0.64, 0.01) and _near(full, 0.625, 0.01)
             and _near(hn["summary"], 0.045, 0.001) and _near(hb["summary"], 0.064, 0.001)
             and _near(om["pooled"]["v"], 0.49, 0.01), out)

    P("\nTable 3b. Post-hoc H2c baselines at B = 1,000: AUROC for residue flips.")
    P("System | Note length | Names-action flag")
    nm = lambda r: 0.0 if r["names_a_action"] else 1.0
    ln, na = {}, {}
    for k, rs in sets.items():
        ln[k] = M.auc_fn(rs, lambda r: r["note_tokens"], "flip")
        na[k] = M.auc_fn(rs, nm, "flip")
        P(f"{k} | {_au(ln[k])} | {_au(na[k])}")
    anyn = sum(bool(r["names_action"]) for r in v3)
    own = [r for r in v3 if r["names_a_action"]]
    of = sum(bool(r["flip"]) for r in own)
    nflip = sum(bool(r["flip"]) for r in v3)
    P(f"Of {len(v3)} notes, {anyn} name some action and {len(own)} name A's action. "
      f"Those {len(own)} split {of} of {nflip} flip notes against {len(own) - of} of {len(v3) - nflip} non-flip notes.")
    _verdict("Table 3b", _near(na["pooled"]["v"], 0.60, 0.01) and _near(na["pooled"]["ci"][0], 0.55, 0.01)
             and _near(na["pooled"]["ci"][1], 0.65, 0.01) and _near(ln["pooled"]["v"], 0.49, 0.01)
             and (anyn, len(own), of) == (335, 234, 48), out)

    # ---- v4 tables
    arm = {a: {(r["case_id"], r["system"]): r for r in v4 if r["arm"] == a}
           for a in ("extractive", "stripped", "sonnet", "budget300", "budget600")}
    def join(a):  # v3 handoffs with the arm's fields prefixed
        return [{**r, **{"x_" + k: v for k, v in arm[a][(r["case_id"], r["system"])].items()}} for r in v3]
    ex, st = join("extractive"), join("stripped")
    xflip = lambda r: bool(r["x_flip"])
    xrec = lambda r: bool(r["x_flip"] and r["x_restored"] == "restored")
    xver = lambda r: bool(r["x_restored_verified"])
    sflip = lambda r: bool(r["x_flip"])
    P("\nTable 4. Notes without paraphrase or advice at B = 1,000, in percent of 490 eligible handoffs. Exploratory.")
    P("Quantity | Count | Rate | Minus floor")
    t4 = {}
    for name, rows, fn in (("Residue rate, extractive", ex, xflip), ("Recoverable, extractive", ex, xrec),
                           ("Recoverable, extractive, verified", ex, xver),
                           ("Residue rate, advice-stripped", st, sflip),
                           ("Recoverable, paraphrased (v3)", v3, rec)):
        t4[name] = (M.rate(rows, fn), M.diff(rows, fn, floor))
        P(f"{name} | {t4[name][0]['k']} | {_pc(t4[name][0])} | {_sd(t4[name][1])}")
    fr = M.rate(v3, floor)
    P(f"Rewording floor (v3) | {fr['k']} | {_pc(fr)} | -")
    xa = M.auc_fn([r for r in ex if r["x_p_residue"] is not None], lambda r: r["x_p_residue"], "x_flip")
    xo = M.auc_fn(ex, lambda r: r["x_omission_fraction"], "x_flip")
    P(f"Extractive notes: writer p_residue AUROC {_au(xa)}; omission fraction AUROC {_au(xo)}.")
    _verdict("Table 4", _rate_ok(t4["Recoverable, extractive"][0], 144, 490, 29.4)
             and _rate_ok(t4["Recoverable, extractive"][1], v=19.0, lo=14.5, hi=24.0)
             and _rate_ok(t4["Residue rate, extractive"][0], 193, 490, 39.4)
             and _rate_ok(t4["Residue rate, advice-stripped"][0], 126, 490, 25.7), out)

    P("\nTable 5. Two readers of the same paraphrased notes at B = 1,000, in percent. Exploratory.")
    P("Reader | Residue rate | Reader's floor | Residue minus floor | Writer AUROC")
    sn = join("sonnet")
    sflr = lambda r: bool(r["x_floor_flip"])
    hk = (M.rate(v3, flip), M.rate(v3, floor), M.diff(v3, flip, floor), A["pooled"])
    P(f"Haiku 5.5 (v3) | {_pc(hk[0])} | {_pc(hk[1])} | {_sd(hk[2])} | {_au(hk[3])}")
    sx = (M.rate(sn, sflip), M.rate(sn, sflr), M.diff(sn, sflip, sflr), M.auc_fn(sn, pr, "x_flip"))
    P(f"Sonnet 5.5 | {_pc(sx[0])} | {_pc(sx[1])} | {_sd(sx[2])} | {_au(sx[3])}")
    sdiff = M.diff(sn, sflip, flip)
    P(f"Sonnet minus Haiku residue rate: {_sd(sdiff)}")
    for s in SYSTEMS:
        a = M.auc_fn([r for r in sn if r["system"] == s], pr, "x_flip")
        P(f"  Sonnet flips by system, writer AUROC, {s}: {_au(a)}")
    _verdict("Table 5", _rate_ok(sx[0], v=24.3) and _rate_ok(sdiff, v=-4.7, lo=-7.9, hi=-1.7), out)

    P("\nTable 6. Residue rate by note budget at B = 1,000, in percent of 490 eligible handoffs. Exploratory.")
    P("Note budget | Residue rate | Minus 150 | Mean note tokens | Writer AUROC")
    b3, b6 = join("budget300"), join("budget600")
    P(f"150 | {_pc(hk[0])} | - | {sum(r['note_tokens'] for r in v3) / len(v3):.1f} | {_au(A['pooled'])}")
    t6 = {}
    for b, rows in (("300", b3), ("600", b6)):
        t6[b] = (M.rate(rows, sflip), M.diff(rows, sflip, flip), M.auc_fn(rows, lambda r: r["x_p_residue"], "x_flip"))
        mt = sum(r["x_note_tokens"] for r in rows) / len(rows)
        P(f"{b} | {_pc(t6[b][0])} | {_sd(t6[b][1])} | {mt:.1f} | {_au(t6[b][2])}")
    # 600 minus 300, paired
    cross = {r["case_id"] + r["system"]: r for r in b6}
    d63 = M.diff([{**r, "y": cross[r["case_id"] + r["system"]]["x_flip"]} for r in b3], lambda r: r["y"], sflip)
    P(f"600 minus 300: {_sd(d63)}")
    _verdict("Table 6", _rate_ok(t6["300"][0], v=33.1) and _rate_ok(t6["600"][0], v=34.1)
             and _rate_ok(t6["600"][1], v=5.1, lo=0.8, hi=9.5), out)

    if not quiet:
        print("===== Tables 2a-6 from the per-handoff metric files =====")
        print("\n".join(out))
        print()
    for title, word, _ in CHECKS:
        print(f"{title}: {word}")
    return all(ok for _, _, ok in CHECKS)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--quiet", action="store_true", help="print only the check results")
    args = ap.parse_args()
    ok = True
    for name, text, stored in (("Tables 1-4 (v3)", rebuild_v3(), V3 / "tables_v3.txt"),
                               ("Tables 5-7 (v4, exploratory)", rebuild_v4(), V4 / "REPORT.md")):
        if not args.quiet:
            print(f"===== {name} =====\n{text}")
        match = same_lines(text, stored.read_text(encoding="utf-8"))
        ok &= match
        print(f"{name}: {'matches' if match else 'DIFFERS FROM'} {stored.relative_to(ROOT)}")
    ok &= rebuild_metrics(args.quiet)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())

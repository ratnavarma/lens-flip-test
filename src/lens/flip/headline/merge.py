"""`lens flip headline merge-report` (protocol v3).

Takes every case of one mechanism out of a finished base run and puts the cases of a second, finished run
in their place. The v2 metric set and the validity gate are then computed on the combined cases, with the
bootstrap over the combined cases. Both source runs stay untouched. The output directory gets:

  cases.jsonl            the combined cases: the base cases that were kept, then the replacement run's cases
  merge.json             provenance: both sources with sha256 of their files, counts, the replaced cases' own record
  summary_v<P>.json      the report, protocol P (3 for v3), with its provenance block
  tables_v<P>.txt        the tables

Nothing is called over the network. A merge refuses when the runs are unfinished, when their configs differ in
anything that changes a result, when the replacement run holds another mechanism, or when case ids collide.
"""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any

from ..schema import MECHANISMS, read_jsonl, write_jsonl
from . import provenance
from .metrics import collect, summarise, tables

# Config entries that must match between the two runs. n, seed, sample and the bootstrap seed may differ.
SAME_CONFIG = ("systems", "budgets", "residue_budget", "max_calls", "strategy", "trunc_tokens", "prompt_versions",
               "decider", "summarizer", "note_writer", "embedder", "rates", "h2b_draws", "power_rule")


class MergeRefused(Exception):
    pass


def _sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def _spend(d: Path) -> dict[str, Any] | None:
    p = d / "spend.json"
    if not p.exists():
        return None
    sp = json.loads(p.read_text())
    return {"actual_usd": sp.get("actual_usd"), "rounds": len(sp.get("rounds", [])),
            "input_tokens": sp.get("input_tokens"), "output_tokens": sp.get("output_tokens")}


def _accuracy(cases: list, a_h: dict[str, str]) -> dict[str, Any]:
    known = [c for c in cases if c.case_id in a_h]
    ok = sum(a_h[c.case_id] == c.a_star for c in known)
    return {"n": len(known), "correct": ok, "accuracy": ok / len(known) if known else None}


def merge_data(base: Path, with_: Path, mechanism: str) -> tuple[dict[str, Any], dict[str, Any]]:
    """The combined `collect` data, and the provenance block."""
    if mechanism not in MECHANISMS:
        raise MergeRefused(f"unknown mechanism {mechanism!r}")
    if base.resolve() == with_.resolve():
        raise MergeRefused("--base and --with are the same directory")
    b_cases, w_cases = list(read_jsonl(base / "cases.jsonl")), list(read_jsonl(with_ / "cases.jsonl"))
    off = [c.case_id for c in w_cases if c.mechanism != mechanism]
    if off:
        raise MergeRefused(f"the replacement run holds cases that are not {mechanism}: {', '.join(off[:3])}")
    if not w_cases:
        raise MergeRefused("the replacement run has no cases")
    if not any(c.mechanism == mechanism for c in b_cases):
        raise MergeRefused(f"the base run has no {mechanism} cases to replace")
    kept = {c.case_id for c in b_cases if c.mechanism != mechanism}
    clash = sorted(kept & {c.case_id for c in w_cases})
    if clash:
        raise MergeRefused(f"case ids collide with kept base cases: {', '.join(clash[:3])}")
    if len({c.case_id for c in w_cases}) != len(w_cases):
        raise MergeRefused("the replacement run repeats a case id")
    b, w = collect(base), collect(with_)
    for name, data in (("base", b), ("replacement", w)):
        if not data["complete"]:
            raise MergeRefused(f"the {name} run is not complete")
    differ = [k for k in SAME_CONFIG if b["config"].get(k) != w["config"].get(k)]
    if differ:
        raise MergeRefused("the runs differ in config entries that change results: " + ", ".join(differ))
    if int(b["config"]["bootstrap"]["reps"]) != int(w["config"]["bootstrap"]["reps"]):
        raise MergeRefused("the runs use different bootstrap sizes")
    keep_cases = [c for c in b["cases"] if c.mechanism != mechanism]
    gone_cases = [c for c in b["cases"] if c.mechanism == mechanism]

    def pick(rows_b: list[dict], rows_w: list[dict]) -> list[dict]:
        return [r for r in rows_b if r["case_id"] in kept] + list(rows_w)

    data = {
        "config": {**b["config"], "n": len(keep_cases) + len(w["cases"]),
                   "merged": {"base": base.name, "replaced_mechanism": mechanism, "with": with_.name}},
        "cases": keep_cases + w["cases"],
        "decisions": pick(b["decisions"], w["decisions"]),
        "records": pick(b["records"], w["records"]),
        "residue_rows": pick(b["residue_rows"], w["residue_rows"]),
        "residue_records": pick(b["residue_records"], w["residue_records"]),
        "complete": True,
    }
    a_h_base = {r["case_id"]: r["a_H"] for r in b["decisions"]}
    a_h_with = {r["case_id"]: r["a_H"] for r in w["decisions"]}
    prov = {
        "replaced_mechanism": mechanism,
        "base": {"dir": str(base), "n": len(b["cases"]), "cases_sha256": _sha(base / "cases.jsonl"),
                 "config_sha256": _sha(base / "config.json"), "spend": _spend(base),
                 "kept": len(keep_cases), "replaced": len(gone_cases),
                 "replaced_cases_a_H": _accuracy(gone_cases, a_h_base)},
        "with": {"dir": str(with_), "n": len(w["cases"]), "cases_sha256": _sha(with_ / "cases.jsonl"),
                 "config_sha256": _sha(with_ / "config.json"), "spend": _spend(with_),
                 "a_H": _accuracy(w["cases"], a_h_with)},
        "combined": {"n": len(data["cases"]),
                     "mechanisms": dict(Counter(c.mechanism for c in data["cases"]))},
    }
    return data, prov


def merge_report(base: str | Path, with_: str | Path, mechanism: str, out: str | Path,
                 protocol: int = 3) -> dict[str, Any]:
    base_d, with_d, out_d = Path(base), Path(with_), Path(out)
    if out_d.resolve() in (base_d.resolve(), with_d.resolve()):
        raise MergeRefused("--out must be a new directory, not one of the source runs")
    data, prov = merge_data(base_d, with_d, mechanism)
    summary = summarise(data, None, protocol)
    summary["merge"] = prov
    summary["provenance"] = provenance.build(
        protocol, [("base", base_d), ("replacement", with_d)],
        merge={"base_dir": provenance.rel(base_d), "replacement_dir": provenance.rel(with_d),
               "replaced_mechanism": mechanism})
    out_d.mkdir(parents=True, exist_ok=True)
    write_jsonl(out_d / "cases.jsonl", data["cases"])
    (out_d / "merge.json").write_text(json.dumps(prov, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (out_d / f"summary_v{protocol}.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n",
                                                      encoding="utf-8")
    (out_d / f"tables_v{protocol}.txt").write_text(tables(summary) + "\n" + _merge_lines(prov) + "\n",
                                                    encoding="utf-8")
    return summary


def _merge_lines(prov: dict[str, Any]) -> str:
    m, b, w = prov["replaced_mechanism"], prov["base"], prov["with"]
    old, new = b["replaced_cases_a_H"], w["a_H"]
    L = ["", f"merge: {b['kept']} cases kept from {b['dir']}, {w['n']} {m} cases from {w['dir']} in place of "
             f"{b['replaced']} {m} cases",
         f"{m} a_H on the full history: replaced cases {old['correct']}/{old['n']}, replacement cases "
         f"{new['correct']}/{new['n']}"]
    for tag, src in (("base", b), ("replacement", w)):
        sp = src["spend"]
        if sp:
            L.append(f"{tag} run actual spend ${sp['actual_usd'] or 0:.4f} over {sp['rounds']} rounds "
                     "(the base figure includes the replaced cases)")
    return "\n".join(L)

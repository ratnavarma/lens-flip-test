"""Run one memory system through the flip test and write results to a directory.

Files: cases.jsonl, decisions.jsonl, records.jsonl, summary.json. With residue on, also
residue.jsonl (one row per handoff note) and residue_records.jsonl (flip searches from the note).
"""

from __future__ import annotations

import json
import time
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable

from .decider import PARSE_ERROR, Decider
from .flip import flip_search
from .generator import Knobs, generate_cases
from .metrics import fcr_gated, fcr_raw, gate_report, harmful_flips, is_committed, rate
from .residue import NoteWriter, run_residue
from .schema import MECHANISMS, REQUEST_INFO, Case, DistinctionRecord, write_jsonl
from .systems import get_system
from .systems.base import MemorySystem, count_tokens, event_view


@dataclass
class RunConfig:
    system: str
    budget: int
    n: int
    seed: int = 0
    length: int = 400
    model: str = "qwen3:30b-a3b"
    max_calls: int = 48
    strategy: str = "group"
    verify: bool = False
    distance: str = "middle"
    distractors: int = 2
    rare_share: float = 0.5
    prompt_version: str = "v1"
    residue: bool = False
    # Backends, recorded for provenance. The runner itself uses the objects passed to run_flip.
    summary_model: str = ""
    embed_model: str = ""
    note_model: str = ""


def _dump(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def _write_lines(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as fh:
        for r in rows:
            fh.write(_dump(r) + "\n")


_rate = rate
_is_committed = is_committed


def residue_metrics(rows: list[dict], records: list[DistinctionRecord]) -> dict[str, Any]:
    """Residue rate = P(a_B wrong | a_A right) on gated cases (a_H == a_star), plus counts.

    Rows are the cases where a note was written: gated cases with a_A == a_star. "Wrong" is
    any a_B that is not a_star, parse errors included.
    """
    wrong = [r for r in rows if r["a_B"] != r["a_star"]]
    searched = list(records)
    restored = [r for r in searched if r.status == "restored"]
    return {
        "definition": "residue_rate = P(a_B != a_star | a_A == a_star) on gated cases (a_H == a_star)",
        "a_A_right": len(rows),
        "a_B_wrong": len(wrong),
        "residue_rate": _rate(len(wrong), len(rows)),
        "residue_flips": len(wrong),
        "searches": len(searched),
        "status": dict(Counter(r.status for r in searched)),
        "cap_hits": sum(r.hit_cap for r in searched),
        "mean_units_inserted": (sum(len(r.units_inserted) for r in restored) / len(restored)) if restored else None,
        "mean_note_tokens": sum(r["note_tokens"] for r in rows) / len(rows) if rows else None,
    }


def summarise(config: RunConfig, decisions: list[dict], records: list[DistinctionRecord],
              residue_rows: list[dict] | None = None,
              residue_records: list[DistinctionRecord] | None = None) -> dict[str, Any]:
    """Metrics from the decision rows and flip-search records (no cost fields)."""
    n = len(decisions)
    by_mech = {m: [d for d in decisions if d["mechanism"] == m] for m in MECHANISMS}

    def acc(rows: list[dict]) -> dict[str, Any]:
        ok = sum(d["a_H"] == d["a_star"] for d in rows)
        return {"n": len(rows), "correct": ok, "accuracy": _rate(ok, len(rows))}

    def info_rate(rows: list[dict]) -> dict[str, Any]:
        k = sum(d["a_V"] == REQUEST_INFO for d in rows)
        return {"n": len(rows), "request_info": k, "rate": _rate(k, len(rows))}

    rare = [d for d in decisions if d["mechanism"] != "M0"]
    restraint_rows = by_mech["U1"] + by_mech["U2"]
    calls = [r.calls_used for r in records]
    overall = acc(decisions)
    per_mech = {m: acc(by_mech[m]) for m in MECHANISMS}
    out = {
        "config": asdict(config),
        "n_cases": n,
        "mechanism_counts": {m: len(by_mech[m]) for m in MECHANISMS},
        "validity": {
            "overall": overall,
            "per_mechanism": per_mech,
            "gate": gate_report(overall["accuracy"], {m: a["accuracy"] for m, a in per_mech.items()}),
        },
        # Raw FCR: every committed a_V counts, whatever a_H said.
        "fcr_raw": {"all": fcr_raw(decisions), "rare": fcr_raw(rare)},
        # Gated FCR: only cases with a_H == a_star count (the full-history decision is right).
        "fcr_gated": {"all": fcr_gated(decisions), "rare": fcr_gated(rare)},
        "harmful_flip": {"all": harmful_flips(decisions), "rare": harmful_flips(rare)},
        "right_restraint": {"U1": info_rate(by_mech["U1"]), "U2": info_rate(by_mech["U2"]),
                            "U1_U2": info_rate(restraint_rows)},
        "over_caution_M0": info_rate(by_mech["M0"]),
        "parse_errors": {"a_H": sum(d["a_H"] == PARSE_ERROR for d in decisions),
                         "a_V": sum(d["a_V"] == PARSE_ERROR for d in decisions)},
        "view": {
            "mean_events": sum(d["view_events"] for d in decisions) / n if n else None,
            "mean_tokens": sum(d["view_tokens"] for d in decisions) / n if n else None,
        },
        "flip_search": {
            "searches": len(records),
            "status": dict(Counter(r.status for r in records)),
            "cap_hits": sum(r.hit_cap for r in records),
            "mean_calls": sum(calls) / len(calls) if calls else None,
            "granularity": dict(Counter(r.granularity for r in records if r.status == "restored")),
            "verified": dict(Counter(str(r.verified) for r in records if r.status == "restored")),
        },
    }
    if config.residue:
        out["residue"] = residue_metrics(residue_rows or [], residue_records or [])
    return out


def run_flip(
    config: RunConfig,
    out_dir: str | Path,
    decide: Decider,
    verify_decide: Decider | None = None,
    *,
    system: MemorySystem | None = None,
    note_writer: NoteWriter | None = None,
    stats: Any = None,
    log: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    """Run the test. `stats` is an object with real_calls and cache_hits, if any.

    system defaults to the plain system named in the config (summary and rag-embed need one
    built with their backends). note_writer is required when config.residue is on.
    """
    t0 = time.time()
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    if system is None:
        system = get_system(config.system)
    if config.residue and note_writer is None:
        raise ValueError("residue needs a note_writer")
    knobs = Knobs(length=config.length, distance=config.distance, distractors=config.distractors,
                  rare_share=config.rare_share)
    cases: list[Case] = generate_cases(config.n, config.seed, knobs)
    write_jsonl(out / "cases.jsonl", cases)

    decisions: list[dict] = []
    records: list[DistinctionRecord] = []
    residue_rows: list[dict] = []
    residue_records: list[DistinctionRecord] = []
    for i, case in enumerate(cases):
        full = event_view(case, (e.id for e in case.events), "full", None)
        a_h = decide(full, [], case)
        view = system.view(case, config.budget)
        a_v = decide(view, [], case)
        row = {
            "case_id": case.case_id, "mechanism": case.mechanism, "cls": case.cls, "a_star": case.a_star,
            "a_H": a_h, "a_V": a_v, "view_events": len(view.event_ids), "view_units": len(view.units),
            "view_tokens": count_tokens([view.text]), "full_tokens": count_tokens([full.text]),
        }
        if not view.event_ids and view.text:  # a text view (summary): keep what the decider saw
            row["view_text"] = view.text
        decisions.append(row)
        rec = None
        if a_v != a_h and a_h != PARSE_ERROR:
            rec = flip_search(
                case, view, decide, a_h, system=config.system, budget=config.budget,
                model=config.model, max_calls=config.max_calls,
                verify=verify_decide if config.verify else None, strategy=config.strategy,
            )
            records.append(rec)
        res = None
        if config.residue and a_h == case.a_star and a_v == a_h:  # gated case, agent A right
            res = run_residue(
                case, view, note_writer, decide, a_h, a_v, system=config.system, budget=config.budget,
                model=config.model, max_calls=config.max_calls,
                verify=verify_decide if config.verify else None, strategy=config.strategy,
            )
            residue_rows.append({
                "case_id": case.case_id, "mechanism": case.mechanism, "a_star": case.a_star, "a_H": a_h,
                "a_A": a_v, "a_B": res.a_B, "residue_flip": res.flipped, "note": res.note,
                "note_tokens": res.note_tokens, "status": res.record.status if res.record else None,
            })
            if res.record:
                residue_records.append(res.record)
        if log:
            tail = f" flip={rec.status} calls={rec.calls_used}" if rec else ""
            if res:
                tail += f" a_B={res.a_B}" + (f" residue={res.record.status} calls={res.record.calls_used}"
                                            if res.record else "")
            log(f"[{i + 1}/{len(cases)}] {case.case_id} {case.mechanism} a*={case.a_star} a_H={a_h} a_V={a_v}{tail}")

    _write_lines(out / "decisions.jsonl", decisions)
    _write_lines(out / "records.jsonl", [r.to_dict() for r in records])
    if config.residue:
        _write_lines(out / "residue.jsonl", residue_rows)
        _write_lines(out / "residue_records.jsonl", [r.to_dict() for r in residue_records])
    summary = summarise(config, decisions, records, residue_rows, residue_records)
    summary["cost"] = {
        "real_calls": getattr(stats, "real_calls", None),
        "cache_hits": getattr(stats, "cache_hits", None),
        "embed_requests": getattr(stats, "embed_requests", None),
        "embed_cache_hits": getattr(stats, "embed_hits", None),
        "wall_seconds": round(time.time() - t0, 2),
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return summary

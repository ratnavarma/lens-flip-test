"""Step 3D: runner and CLI wiring with the oracle mock. No Ollama."""

from __future__ import annotations

import json

import pytest

from lens.flip.decider import PARSE_ERROR, oracle_decider
from lens.flip.runner import RunConfig, run_flip, summarise
from lens.flip.schema import DistinctionRecord, read_jsonl


def _read(path):
    return [json.loads(x) for x in path.read_text().splitlines()]


def test_run_with_oracle_decider(tmp_path):
    cfg = RunConfig(system="recent", budget=500, n=30, seed=5, length=100, model="oracle", max_calls=24, verify=True)
    s = run_flip(cfg, tmp_path, oracle_decider, oracle_decider)
    assert len(list(read_jsonl(tmp_path / "cases.jsonl"))) == 30
    dec = _read(tmp_path / "decisions.jsonl")
    assert len(dec) == 30
    assert set(dec[0]) >= {"case_id", "mechanism", "cls", "a_star", "a_H", "a_V", "view_events", "view_tokens"}
    assert all(d["view_tokens"] <= 500 for d in dec)
    recs = [DistinctionRecord.from_dict(r) for r in _read(tmp_path / "records.jsonl")]
    assert len(recs) == sum(d["a_H"] != d["a_V"] for d in dec) == s["flip_search"]["searches"]
    assert s["validity"]["overall"]["accuracy"] == 1.0  # the oracle is the generator's rule
    assert all(r.calls_used <= 24 for r in recs)
    assert json.loads((tmp_path / "summary.json").read_text())["n_cases"] == 30
    assert s["cost"]["wall_seconds"] >= 0


def test_full_system_has_no_flips(tmp_path):
    cfg = RunConfig(system="full", budget=500, n=12, seed=2, length=100, model="oracle")
    s = run_flip(cfg, tmp_path, oracle_decider)
    assert s["flip_search"]["searches"] == 0
    assert s["harmful_flip"]["all"]["count"] == 0
    assert s["fcr_raw"]["all"]["false_closures"] == 0 and s["fcr_gated"]["all"]["false_closures"] == 0


def test_run_is_deterministic(tmp_path):
    cfg = RunConfig(system="rag-bm25", budget=500, n=12, seed=2, length=100, model="oracle")
    run_flip(cfg, tmp_path / "a", oracle_decider)
    run_flip(cfg, tmp_path / "b", oracle_decider)
    for f in ("cases.jsonl", "decisions.jsonl", "records.jsonl"):
        assert (tmp_path / "a" / f).read_bytes() == (tmp_path / "b" / f).read_bytes()


def _row(mech, a_star, a_h, a_v):
    return {"mechanism": mech, "a_star": a_star, "a_H": a_h, "a_V": a_v, "view_events": 1, "view_tokens": 1}


def test_metrics_on_a_toy_table():
    cfg = RunConfig(system="recent", budget=1, n=7)
    rows = [
        _row("M0", "EXPLAIN", "EXPLAIN", "EXPLAIN"),
        _row("M0", "EXPLAIN", "EXPLAIN", "REQUEST_INFO"),   # over-caution, harmful flip
        _row("M1", "OPEN_REVIEW", "OPEN_REVIEW", "EXPLAIN"),  # false closure, harmful flip
        _row("M2", "CHECK_DEVICE", "CHECK_DEVICE", "CHECK_DEVICE"),
        _row("M3", "APPLY_CREDIT", "EXPLAIN", PARSE_ERROR),  # not committed
        _row("U1", "REQUEST_INFO", "REQUEST_INFO", "REQUEST_INFO"),
        _row("U2", "REQUEST_INFO", "REQUEST_INFO", "OPEN_REVIEW"),  # false closure, harmful flip
    ]
    s = summarise(cfg, rows, [])
    assert s["validity"]["overall"]["correct"] == 6 and s["validity"]["per_mechanism"]["M3"]["accuracy"] == 0.0
    assert s["fcr_raw"]["all"] == {"n": 7, "committed": 4, "false_closures": 2, "fcr": 0.5}
    assert s["fcr_raw"]["rare"]["committed"] == 3 and s["fcr_raw"]["rare"]["false_closures"] == 2
    assert s["harmful_flip"]["all"]["count"] == 3
    assert s["right_restraint"]["U1_U2"]["rate"] == 0.5
    assert s["over_caution_M0"]["rate"] == 0.5
    assert s["parse_errors"]["a_V"] == 1

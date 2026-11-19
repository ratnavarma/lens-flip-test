"""Protocol v2: the power rule per hypothesis, the harmful flip split, pilot-calibrated dry run, the v2 protocol file."""

from __future__ import annotations

import hashlib
import json
import random
from pathlib import Path

import pytest

from lens.flip.decider import PARSE_ERROR
from lens.flip.headline import calibrate as C
from lens.flip.headline import dryrun as D
from lens.flip.headline import engine as E
from lens.flip.headline import metrics as HM
from lens.flip.headline import mock as M
from lens.flip.headline import protocol as P
from lens.flip.schema import Case, REQUEST_INFO

CFG = {"power_min_flips": 20, "h2b_draws": 50, "bootstrap": {"seed": 1}}


# ---------------------------------------------------------------------------
# power rule per hypothesis
# ---------------------------------------------------------------------------

def _rows(n_elig: int, n_flip: int, *, event: bool = True, seed: int = 0):
    """Toy residue rows. The first n_flip rows are residue flips with D* in V and a restored search."""
    rng = random.Random(seed)
    cases, rows, recs = [], [], {}
    for i in range(n_elig):
        cid = f"t{i:03d}"
        cases.append(Case(case_id=cid, seed=i, mechanism="M1", events=[], query_period="2020-01", query_text="q",
                          actions=[], a_star="OPEN_REVIEW", d_star=["e1"], deleted_ids=[], cls="rare", knobs={}))
        flip = i < n_flip
        system = "recent" if event else "summary"
        rows.append({"case_id": cid, "system": system, "a_star": "OPEN_REVIEW", "a_A": "OPEN_REVIEW",
                     "a_B": "EXPLAIN" if flip else "OPEN_REVIEW",
                     "a_floor": "OPEN_REVIEW", "a_trunc": "EXPLAIN" if i % 2 else "OPEN_REVIEW",
                     "residue_flip": flip, "dstar_in_view": True,
                     "candidate_ids": ["e1", "e2", "e3", "e4"] if flip else [], "note_status": "ok",
                     "p_residue": 0.6 + 0.4 * rng.random() if flip else 0.5 * rng.random(), "omission": 0.5})
        if flip:
            recs[(cid, system)] = {"status": "restored", "event_ids": ["e1"], "hit_cap": False}
    return cases, rows, recs


def _block(n_elig, n_flip, **kw):
    cases, rows, recs = _rows(n_elig, n_flip, **kw)
    return HM.residue_block(HM.Boot(len(cases), 200, 1), cases, rows, recs, CFG, n_flips_run=n_flip, protocol=2)


def _labelled(verdict: str) -> bool:
    return verdict in ("supported", "fails")


def test_power_rule_h2a_needs_twenty_eligible_cases():
    low, ok = _block(19, 0), _block(20, 0)
    assert low["power_rule"]["H2a"] == {"n": 19, "min": 20, "powered": False}
    assert low["h2a_verdict"].startswith("descriptive") and not _labelled(low["h2a_verdict"])
    assert low["h2a_ci95"] is not None  # bounds are still reported
    assert ok["power_rule"]["H2a"]["powered"] and _labelled(ok["h2a_verdict"])


def test_power_rule_h2b_counts_its_own_flips():
    low, ok = _block(40, 19), _block(40, 20)
    assert low["power_rule"]["H2b"] == {"n": 19, "min": 20, "powered": False}
    assert low["h2b_verdict"].startswith("descriptive") and low["h2b_ci95"] is not None
    assert ok["power_rule"]["H2b"]["powered"] and _labelled(ok["h2b_verdict"])
    # 25 residue flips that H2b cannot use (text view): H2b is out, H2c is in.
    summ = _block(60, 25, event=False)
    assert summ["power_rule"]["H2b"]["n"] == 0 and summ["h2b_verdict"].startswith("descriptive")
    assert summ["power_rule"]["H2c"]["powered"] and _labelled(summ["h2c_verdict"])


def test_power_rule_h2c_needs_flips_and_non_flips():
    cases = {
        "pos 19": (_block(60, 19), False),   # 19 positives, 41 negatives
        "neg 19": (_block(39, 20), False),   # 20 positives, 19 negatives
        "both 20": (_block(40, 20), True),   # 20 and 20
    }
    for name, (b, powered) in cases.items():
        assert b["power_rule"]["H2c"]["powered"] is powered, name
        assert _labelled(b["h2c_verdict"]) is powered, name
        if not powered:
            assert b["h2c_verdict"].startswith("descriptive") and b["h2c_ci95_vs_half"] is not None
    assert cases["pos 19"][0]["power_rule"]["H2c"]["n_pos"] == 19
    assert cases["neg 19"][0]["power_rule"]["H2c"]["n_neg"] == 19


def test_v1_power_rule_is_unchanged():
    cases, rows, recs = _rows(40, 25)
    b = HM.residue_block(HM.Boot(len(cases), 100, 1), cases, rows, recs, CFG, n_flips_run=19)
    assert b["h2b_verdict"].startswith("descriptive") and b["h2c_verdict"].startswith("descriptive")
    assert b["power_rule"] == {"min_residue_flips": 20, "n_residue_flips_run": 19, "powered": False}


def test_thresholds_can_come_from_the_config():
    assert HM.power_thresholds({}) == HM.POWER_RULE_V2
    assert HM.power_thresholds({"power_rule": {"H2b_min_flips": 5}})["H2b_min_flips"] == 5
    assert E.default_config(10, 1)["power_rule"] == HM.POWER_RULE_V2


# ---------------------------------------------------------------------------
# harmful split
# ---------------------------------------------------------------------------

def _decision(case_id, a_star, a_h, a_v, mech="M1"):
    return {"case_id": case_id, "mechanism": mech, "cls": "rare", "a_star": a_star, "length": 100, "distance": "near",
            "system": "recent", "budget": 500, "a_H": a_h, "a_V": a_v, "view_events": 3, "view_units": 3,
            "view_tokens": 100, "flip_status": None}


def test_harm_part_partition():
    f = HM.harm_part
    assert f(_decision("c", "EXPLAIN", "EXPLAIN", "OPEN_REVIEW")) == "wrong_commit"
    assert f(_decision("c", "EXPLAIN", "EXPLAIN", REQUEST_INFO)) == "abstention"
    assert f(_decision("c", "EXPLAIN", "EXPLAIN", PARSE_ERROR)) == "parse_error"
    assert f(_decision("c", REQUEST_INFO, REQUEST_INFO, "EXPLAIN")) == "wrong_commit"  # a* itself is an abstention
    assert f(_decision("c", REQUEST_INFO, REQUEST_INFO, REQUEST_INFO)) is None         # not a flip
    assert f(_decision("c", "EXPLAIN", "EXPLAIN", "EXPLAIN")) is None
    assert f(_decision("c", "EXPLAIN", "OPEN_REVIEW", REQUEST_INFO)) is None           # a_H wrong: not gated


def test_wrong_commit_plus_abstention_plus_parse_is_harmful_on_a_toy_table():
    table = [  # (a_star, a_H, a_V)
        ("EXPLAIN", "EXPLAIN", "OPEN_REVIEW"),       # wrong commit
        ("EXPLAIN", "EXPLAIN", "APPLY_CREDIT"),      # wrong commit
        ("EXPLAIN", "EXPLAIN", REQUEST_INFO),        # abstention
        ("OPEN_REVIEW", "OPEN_REVIEW", REQUEST_INFO),  # abstention
        ("EXPLAIN", "EXPLAIN", REQUEST_INFO),        # abstention
        ("EXPLAIN", "EXPLAIN", PARSE_ERROR),         # parse error
        (REQUEST_INFO, REQUEST_INFO, "EXPLAIN"),     # wrong commit (a* is an abstention)
        (REQUEST_INFO, REQUEST_INFO, REQUEST_INFO),  # right restraint, no flip
        ("EXPLAIN", "EXPLAIN", "EXPLAIN"),           # no flip
        ("EXPLAIN", "OPEN_REVIEW", REQUEST_INFO),    # a_H wrong: not gated, not counted
    ]
    cases, rows = [], []
    for i, (a_star, a_h, a_v) in enumerate(table):
        cid = f"c{i}"
        cases.append(Case(case_id=cid, seed=i, mechanism="M1", events=[], query_period="2020-01", query_text="q",
                          actions=[], a_star=a_star, d_star=[], deleted_ids=[], cls="rare", knobs={}))
        rows.append(_decision(cid, a_star, a_h, a_v))
    m = HM.cell_metrics(HM.Boot(len(cases), 100, 1), cases, rows, [])
    h = m["harmful_flip"]
    assert h["count"] == 7
    parts = {k: h[k]["count"] for k in ("wrong_commit", "abstention", "parse_error")}
    assert parts == {"wrong_commit": 3, "abstention": 3, "parse_error": 1}
    assert sum(parts.values()) == h["count"]
    # rates over gated cases (9 rows have a_H right) and over all cases (10) sum to the harmful rates
    assert h["rate_gated"]["value"] == pytest.approx(7 / 9)
    assert sum(h[k]["rate_gated"]["value"] for k in parts) == pytest.approx(h["rate_gated"]["value"])
    assert sum(h[k]["rate"]["value"] for k in parts) == pytest.approx(h["rate"]["value"])
    assert m["parse_errors_a_V"] == 1


# ---------------------------------------------------------------------------
# a tiny finished run
# ---------------------------------------------------------------------------

def _run(tmp: Path, n: int, seed: int) -> Path:
    d = tmp / f"run{n}-{seed}"
    E.init_run(d, n, seed)
    cfg = json.loads((d / "config.json").read_text())
    cfg["embedder"] = {"kind": "hash", "model": "hash-bow-256"}
    cfg["bootstrap"]["reps"] = 100
    cfg["h2b_draws"] = 50
    (d / "config.json").write_text(json.dumps(cfg))
    return d


@pytest.fixture(scope="module")
def tiny(tmp_path_factory):
    d = _run(tmp_path_factory.mktemp("v2"), 12, 5)
    eng = E.Engine(d, check_protocol=False)
    model = M.MockModel(eng.cases, M.v4_calibration(), seed=1)
    eng.runner = M.FakeBatchAPI(model.answer, M.engine_meta_fn(d))
    for _ in range(500):
        if eng.step(1e9) == "complete":
            break
    else:
        raise AssertionError("did not finish")
    eng.close()
    HM.write_report(d)
    return d


def test_calibrate_from_run_reads_the_finished_run(tiny):
    s = C.from_run(tiny)
    data = HM.collect(tiny)
    spend = json.loads((tiny / "spend.json").read_text())
    # flip rate per cell, from the decisions
    for cell, rate in s["a_V"]["flip_rate"].items():
        rs = [r for r in data["decisions"] if f"{r['system']}@{r['budget']}" == cell]
        assert rate == pytest.approx(sum(r["a_V"] != r["a_H"] for r in rs) / len(rs))
    assert set(s["a_V"]["flip_rate"]) == {f"{sy}@{b}" for sy in data["config"]["systems"] for b in data["config"]["budgets"]}
    # mean output tokens per stage, from usage
    for stage, v in spend["by_stage"].items():
        assert s["output"]["mean"][stage] == pytest.approx(v["output_tokens"] / v["requests"], rel=1e-9)
        assert s["output"]["n"][stage] == v["requests"]
    # input tokens against characters, per kind
    assert {"decide", "summary"} <= set(s["input"]["kinds"])
    k = s["input"]["kinds"]["decide"]
    assert k["slope"] > 0 and k["overhead_tokens"] == 331 and k["chars_per_token_low"] > 0
    # a_H accuracy per mechanism, and the search and residue blocks
    for m, acc in s["a_H"]["accuracy_by_mechanism"].items():
        cs = [c for c in data["cases"] if c.mechanism == m]
        ok = sum(next(r["a_H"] for r in data["decisions"] if r["case_id"] == c.case_id) == c.a_star for c in cs)
        assert acc == pytest.approx(ok / len(cs))
    assert s["search"]["pooled"]["n"] == len(data["records"])
    assert s["residue"]["n_eligible"] == len(data["residue_rows"])
    assert s["analysis_sizes"]["H2a_n"] == len(data["residue_rows"])
    assert json.loads(json.dumps(s)) == s  # plain JSON
    assert s["pilot_usd"] == spend["actual_usd"] and s["pilot_rounds"] == len(spend["rounds"])


def test_calibrated_mock_takes_its_rates_from_the_settings(tiny):
    s = C.from_run(tiny)
    cases = list(E.read_jsonl(tiny / "cases.jsonl"))
    case = cases[0]
    params = {"messages": [{"role": "user", "content": "x"}]}

    def ask(model, stage, cell="recent@500", key="k1"):
        meta = {"kind": "decide", "stage": stage, "case_id": case.case_id, "cell": cell, "key": key}
        line = model.answer("q1", params, meta)
        return json.loads(line["result"]["message"]["content"][0]["text"])["action"], line

    s0 = json.loads(json.dumps(s))
    s0["a_V"]["flip_rate"] = {c: 0.0 for c in s0["a_V"]["flip_rate"]}
    s0["a_H"]["accuracy_by_mechanism"][case.mechanism] = 1.0
    keep = M.CalibratedMockModel(cases, s0, seed=3)
    a_h, _ = ask(keep, "a_H")
    assert a_h == keep.a_h(case) and a_h == case.a_star
    assert ask(keep, "a_V")[0] == a_h
    s1 = json.loads(json.dumps(s0))
    s1["a_V"]["flip_rate"] = {c: 1.0 for c in s1["a_V"]["flip_rate"]}
    s1["a_V"]["flipped_to"] = {REQUEST_INFO: 1.0}
    flip = M.CalibratedMockModel(cases, s1, seed=3)
    a_v, line = ask(flip, "a_V")
    assert a_v != flip.a_h(case)
    # usage comes from the pilot's tables: the input from the fitted line, the output from the stage quantiles
    usage = line["result"]["message"]["usage"]
    assert usage["output_tokens"] in s1["output"]["quantiles"]["a_V"]
    assert usage["input_tokens"] == E.rq.estimate_input_tokens(E.rq.request_chars(params), "decide", s1["input"], bound=False)
    # a wrong a_H follows the accuracy
    s2 = json.loads(json.dumps(s0))
    s2["a_H"]["accuracy_by_mechanism"][case.mechanism] = 0.0
    wrong = M.CalibratedMockModel(cases, s2, seed=3)
    assert wrong.a_h(case) != case.a_star


def test_dryrun_calibrated_from_a_finished_run_makes_no_api_call(tiny, tmp_path):
    d = _run(tmp_path, 8, 21)
    logs: list[str] = []
    res = D.run_dryrun(d, calibrate_k=0, runner=D.no_api_runner, log=logs.append, calibrate_from_run=tiny)
    assert res["no_batch_submitted"] and res["calibration"]["calls_this_run"] == 0
    assert res["assumptions"]["calibrated_from_run"] == str(tiny)
    assert res["cap_limited_upper_usd"] >= res["likely_usd"] > 0 and res["rounds"] > 3
    x = res["cross_check"]
    spend = json.loads((tiny / "spend.json").read_text())
    assert x["pilot_actual_usd"] == spend["actual_usd"]
    assert x["scaled_total_usd"] == pytest.approx(sum(r["scaled_usd"] for r in x["by_stage"]))
    assert x["scaled_total_usd"] == pytest.approx(spend["actual_usd"] * 8 / 12, rel=1e-3)
    p = res["projection_at_n"]
    assert p["n_target"] == 8 and p["pilot_cases"] == 12 and {"H2a", "H2b", "H2c"} <= set(p)
    assert set(res["residue"]["meets_power_rule"]) == {"H2a", "H2b", "H2c"}
    assert "cross-check" in D.format_tables(res)
    with pytest.raises(RuntimeError, match="no API call"):
        D.no_api_runner("POST", "/v1/messages/count_tokens")


def test_cli_dryrun_flag_exists():
    import argparse

    from lens.flip import cli as flip_cli

    parser = argparse.ArgumentParser(prog="lens")
    flip_cli.register(parser.add_subparsers(dest="command", required=True))
    ns = parser.parse_args(["flip", "headline", "dryrun", "--dir", "x", "--calibrate-from-run", "y"])
    assert ns.calibrate_from_run == "y"
    rep = parser.parse_args(["flip", "headline", "report", "--dir", "x", "--protocol", "2"])
    assert rep.protocol == 2
    assert parser.parse_args(["flip", "headline", "freeze", "--dir", "x"]).final is False


def test_poisson_bounds():
    lo, hi = C.poisson_bounds(0)
    assert lo == 0.0 and hi == pytest.approx(3.689, abs=1e-3)
    lo, hi = C.poisson_bounds(3)
    assert lo == pytest.approx(0.619, abs=1e-3) and hi == pytest.approx(8.767, abs=1e-3)


# ---------------------------------------------------------------------------
# report under both protocols, and the protocol file
# ---------------------------------------------------------------------------

def _sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def test_report_v2_writes_new_files_and_leaves_v1_alone(tiny):
    names = ("summary.json", "tables.txt", "decisions.jsonl", "records.jsonl", "residue.jsonl", "residue_records.jsonl")
    before = {n: _sha(tiny / n) for n in names}
    s = HM.write_report(tiny, 2)
    assert {n: _sha(tiny / n) for n in names} == before
    assert s["protocol"] == 2 and (tiny / "summary_v2.json").exists() and (tiny / "tables_v2.txt").exists()
    on_disk = json.loads((tiny / "summary_v2.json").read_text())
    for c in on_disk["cells"].values():
        h = c["harmful_flip"]
        assert {"wrong_commit", "abstention", "parse_error"} <= set(h)
        assert sum(h[k]["count"] for k in ("wrong_commit", "abstention", "parse_error")) == h["count"]
    pr = on_disk["residue"]["pooled"]["power_rule"]
    assert {"H2a", "H2b", "H2c"} <= set(pr) and pr["H2a"]["min"] == 20
    text = (tiny / "tables_v2.txt").read_text()
    assert "wrong commit" in text and "abstention" in text
    assert "H2a: n=" in text and "needs 20" in text and "H2c: positives" in text


def test_v2_protocol_file_lists_the_changes_and_is_a_draft(tmp_path):
    d = _run(tmp_path, 6, 31)
    doc = tmp_path / "flip-protocol-v2.md"
    res = P.freeze([d], doc, now=0)
    text = doc.read_text()
    assert text.startswith("# Decision Flip Test: protocol v2")
    assert "Version: 2" in text and "Status: Draft" in text and "Frozen" not in text.split("## Changes")[0]
    changes = text.split("## Changes from v1")[1].split("## 1.")[0]
    assert "Power rule per hypothesis" in changes and "Harmful flips split" in changes
    assert "Token helper" in changes and "2.2" in changes
    assert "descriptive" in text and "wrong commit plus abstention plus parse error" in changes
    pj = json.loads((d / "protocol.json").read_text())
    assert pj["version"] == 2 and pj["status"] == "draft" and pj["hashes"] == res["hashes"]
    assert all(h in text for h in res["hashes"].values())
    # --final is what marks it frozen; the default and the v2 doc path stay a draft
    P.freeze([d], tmp_path / "final.md", draft=False, now=0)
    assert "Status: Frozen" in (tmp_path / "final.md").read_text()
    assert P.DOC.name == "flip-protocol-v2.md"
    # v1 rendering still works and has no changes section
    P.freeze([d], tmp_path / "v1.md", now=0, version=1)
    assert "Changes from v1" not in (tmp_path / "v1.md").read_text()

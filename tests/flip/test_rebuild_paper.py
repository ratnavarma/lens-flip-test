"""The released metric files hold no text, and every paper cell rebuilds from them. No network, no model call."""

from __future__ import annotations

import copy
import importlib.util
import json
from pathlib import Path

import pytest

from lens.flip import export_metrics as X

ROOT = Path(__file__).resolve().parents[2]
RES = ROOT / "results"
V6_NAME = "exploratory-v6/handoff_metrics_v6.jsonl"
V7_NAME = "exploratory-v7/handoff_metrics_v7.jsonl"
METRIC_FILES = (X.S1_NAME, X.V3_NAME, X.V4_NAME, X.V5_NAME, X.C1_NAME, X.CAL_NAME, V6_NAME, V7_NAME)

_spec = importlib.util.spec_from_file_location("rebuild_paper", ROOT / "scripts" / "rebuild_paper.py")
R = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(R)


@pytest.fixture(scope="module")
def built():
    D = R.Data()
    expected = json.loads(R.EXPECTED.read_text(encoding="utf-8"))
    return D, expected, R.rebuild(D)


def test_match_rule():
    assert R.matches(R.Cell("{:.1f} [{:.1f}, {:.1f}]", 28.98, 24.80, 33.19), "29.0 [24.8, 33.2]")
    assert not R.matches(R.Cell("{:.1f} [{:.1f}, {:.1f}]", 28.98, 24.80, 33.19), "29.1 [24.8, 33.2]")
    assert not R.matches(R.Cell("{:.1f} [{:.1f}, {:.1f}]", 28.98, 24.80, 33.19), "29.0 [24.8, 33.3]")
    assert R.matches(R.Cell("{:.1f}", 6.25), "6.3") and R.matches(R.Cell("{:.1f}", 6.25), "6.2")   # a tie rounds either way
    assert R.matches(R.OF(697, 2450), "697 of 2,450") and not R.matches(R.OF(697, 2450), "698 of 2,450")
    assert not R.matches(R.N(142), "142 of 490") and not R.matches(R.OF(142, 490), "142")
    assert R.matches(R.S("supported"), "supported") and not R.matches(R.S("supported"), "fails")
    assert R.matches(R.Cell("{:.2f} [{:.3f}, {:.2f}]", 0.0712, -0.0049, 0.142), "0.07 [-0.005, 0.14]")
    assert not R.matches(R.Cell("{:.1f}", -2.0), "2.0")


@pytest.mark.parametrize("name", METRIC_FILES)
def test_metric_files_hold_no_text(name):
    X.check(RES / name, [])
    for line in (RES / name).read_text(encoding="utf-8").splitlines():
        for key in json.loads(line):
            assert key not in ("note", "rewrite", "reason", "quote", "unit", "text", "fields", "prompt"), key


def test_case_order_is_the_frozen_one(built):
    D, _, _ = built
    cases = [json.loads(x)["case_id"] for x in D.cases_path.read_text(encoding="utf-8").splitlines() if x.strip()]
    assert D.case_ids == cases and len(cases) == 300
    assert len(D.v3) == 490 and len(D.s1) == 3600 and len(D.v5) == 8 * 490 and len(D.v6) == 490 and len(D.v7) == 1160
    assert all(D.case_ids[r["case_index"]] == r["case_id"] and r["n_cases"] == 300 for r in D.v6 + D.v7)


def test_every_paper_cell_matches(built, capsys):
    _, expected, tables = built
    n_ok, n_bad = R.run_check(tables, expected, quiet=True)
    assert n_bad == 0, capsys.readouterr().out
    assert n_ok == sum(len(c) for t in expected.values() for c in t["rows"].values()) == 509
    assert set(expected) == set(tables)


def test_check_reports_a_wrong_cell(built, capsys):
    _, expected, tables = built
    bad = copy.deepcopy(expected)
    bad["Table 9"]["rows"]["Omission"][1] = "2.9 [1.2, 4.7]"
    bad["Table 2d"]["rows"]["pooled"][0] = "143"
    assert R.run_check(tables, bad, quiet=True)[1] == 2
    assert capsys.readouterr().out.count("MISMATCH") == 2


def test_paper_facts_file_is_current(built):
    D, expected, _ = built
    stored = json.loads(R.FACTS.read_text(encoding="utf-8"))
    assert stored == json.loads(json.dumps(R.facts(D, expected)))
    f = stored["facts"]
    assert f["seed_split"]["value"] == {"seed_11": 270, "seed_12": 30}
    assert f["exploratory_contrasts_with_bounds"]["value"] == f["exploratory_contrasts_with_bounds"]["detail"]["n_distinct_exploratory"]


def test_oracle_agrees_on_the_300_headline_cases(built):
    from lens.flip.generator import oracle
    from lens.flip.schema import Case

    D, _, _ = built
    cases = [Case.from_dict(json.loads(x)) for x in D.cases_path.read_text(encoding="utf-8").splitlines() if x.strip()]
    assert len(cases) == 300
    assert [c.case_id for c in cases if oracle(c.events, c.query_period) != c.a_star] == []
    rare = [c for c in cases if c.mechanism in ("M1", "M2", "M3")]
    assert len(rare) == 90
    for c in rare:
        rest = [e for e in c.events if e.id not in set(c.d_star)]
        assert oracle(rest, c.query_period) != c.a_star, c.case_id


def test_frozen_prompt_hashes_match_the_released_protocol_files():
    import hashlib

    from lens.flip.explore4 import protocol as P4
    from lens.flip.explore5 import judge2, protocol as P5, x2b

    sha = lambda text: hashlib.sha256(text.encode("utf-8")).hexdigest()
    f4 = json.loads((RES / "exploratory-v4" / "protocol_v4.json").read_text())
    f5 = json.loads((RES / "exploratory-v5" / "protocol_v5.json").read_text())
    f5b = json.loads((RES / "exploratory-v5" / "protocol_v5b.json").read_text())
    fj = json.loads((RES / X.JUDGE_NAME).read_text())
    assert {k: sha(v) for k, v in P4.prompt_texts().items()} == f4["prompts"]
    assert {k: sha(v) for k, v in P5.prompt_texts().items()} == f5["prompts"]
    assert {k: sha(v) for k, v in x2b.prompt_texts().items()} == f5b["prompts"]
    assert P5.current()["reused_v3"] == f5["reused_v3"] == f5b["reused_v3"]
    assert sha(judge2.build_prompt("{NOTE}", "{UNIT}")) == fj["prompt_sha256"]
    assert judge2.JUDGE_FORMAT == fj["format"]


def test_v6_summary_rebuilds_from_its_metric_file():
    import hashlib

    from lens.flip.explore6 import analysis as A6

    stored = json.loads((RES / "exploratory-v6" / "summary_v6.json").read_text(encoding="utf-8"))
    body = json.loads(json.dumps(A6.rebuild(RES / V6_NAME)))
    assert body and all(stored[k] == v for k, v in body.items())
    assert hashlib.sha256((RES / V6_NAME).read_bytes()).hexdigest() == stored["metrics_file"]["sha256"]
    assert stored["E1_extractive"]["primary_2_content_specific"]["verdict"] == "not supported"


def test_v7_summary_rebuilds_from_its_metric_file():
    import hashlib

    from lens.flip.explore7 import analysis as A7

    stored = json.loads((RES / "exploratory-v7" / "summary_v7.json").read_text(encoding="utf-8"))
    body = json.loads(json.dumps(A7.rebuild(RES / V7_NAME)))
    assert body and all(stored[k] == v for k, v in body.items())
    assert hashlib.sha256((RES / V7_NAME).read_bytes()).hexdigest() == stored["metrics_file"]["sha256"]
    assert stored["denominators"]["a_wrong"] == 670 and stored["denominators"]["gated_handoffs"] == 1160


def test_v6_and_v7_rules_hash_to_the_released_protocol_files():
    import hashlib

    from lens.flip.explore6 import protocol as P6
    from lens.flip.explore7 import protocol as P7

    sha = lambda text: hashlib.sha256(text.encode("utf-8")).hexdigest()
    f6 = json.loads((RES / "exploratory-v6" / "protocol_v6.json").read_text())
    f7 = json.loads((RES / "exploratory-v7" / "protocol_v7.json").read_text())
    assert {k: sha(v) for k, v in P6.rule_texts().items()} == f6["rules"]
    assert {k: sha(v) for k, v in P7.rule_texts().items()} == f7["rules"]

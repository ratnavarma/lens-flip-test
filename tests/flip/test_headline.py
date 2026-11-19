"""Headline driver: tasks, rounds, routing, spend guard, round-parallel ddmin, residue metrics,
protocol freeze and an end-to-end mock run. No network: every runner is a fake."""

from __future__ import annotations

import json
import random
import subprocess
from pathlib import Path

import pytest

from lens.flip import decider
from lens.flip.cli import build_view
from lens.flip.decider import PARSE_ERROR, oracle_decider
from lens.flip.flip import flip_search
from lens.flip.generator import Knobs, generate_cases
from lens.flip.headline import engine as E
from lens.flip.headline import metrics as HM
from lens.flip.headline import mock as M
from lens.flip.headline import protocol as P
from lens.flip.headline import requests as rq
from lens.flip.pddmin import Pending, Searcher, group_search, pddmin, round_flip_search
from lens.flip.residue import parse_note_reply, tail_tokens
from lens.flip.systems.base import event_view

# The banned domain words are private. They live in an optional, gitignored file at the repo root,
# one word per line. Without it, the word checks skip.
_BANNED_FILE = Path(__file__).resolve().parents[2] / "banned_words.local.txt"
FORBIDDEN = ([w.strip().lower() for w in _BANNED_FILE.read_text().splitlines() if w.strip()]
             if _BANNED_FILE.exists() else [])


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _init(tmp: Path, n: int = 12, seed: int = 5, **cfg_over) -> Path:
    d = tmp / f"run{n}-{seed}"
    E.init_run(d, n, seed)
    cfg = json.loads((d / "config.json").read_text())
    cfg["embedder"] = {"kind": "hash", "model": "hash-bow-256"}
    cfg["bootstrap"]["reps"] = 200
    cfg["h2b_draws"] = 200
    cfg.update(cfg_over)
    (d / "config.json").write_text(json.dumps(cfg))
    return d


def _engine(d: Path, **fake_kw) -> tuple[E.Engine, M.FakeBatchAPI]:
    eng = E.Engine(d, check_protocol=False)
    model = M.MockModel(eng.cases, M.v4_calibration(), seed=1)
    fake = M.FakeBatchAPI(model.answer, M.engine_meta_fn(d), **fake_kw)
    eng.runner = fake
    return eng, fake


def _finish(eng: E.Engine, max_steps: int = 500) -> int:
    for i in range(max_steps):
        if eng.step(1e9) == "complete":
            return i
    raise AssertionError("run did not complete")


def _state_blob(d: Path) -> dict:
    return {p.name: json.loads(p.read_text()) for p in sorted((d / "state" / "cases").glob("*.json"))}


@pytest.fixture(scope="module")
def full_run(tmp_path_factory):
    d = _init(tmp_path_factory.mktemp("hl"), 12, 5)
    eng, fake = _engine(d)
    _finish(eng)
    eng.close()
    summary = HM.write_report(d)
    return d, fake, summary


# ---------------------------------------------------------------------------
# sample and requests
# ---------------------------------------------------------------------------

def test_sample_mix_lengths_and_determinism():
    cs = E.make_headline_sample(60, 3)
    from collections import Counter
    assert Counter(c.mechanism for c in cs) == {"M0": 30, "M1": 6, "M2": 6, "M3": 6, "U1": 6, "U2": 6}
    for m in ("M0", "M1", "U2"):
        assert Counter(c.knobs["length"] for c in cs if c.mechanism == m) in ({100: 15, 400: 15}, {100: 3, 400: 3})
    assert len({c.knobs["distance"] for c in cs}) == 3
    again = E.make_headline_sample(60, 3)
    assert [c.case_id for c in cs] == [c.case_id for c in again]
    assert [c.seed for c in cs] == [c.seed for c in again]


def test_request_shapes_and_keys():
    case = generate_cases(2, 1, Knobs(length=100))[0]
    v = event_view(case, [e.id for e in case.events[:10]], "recent", 500)
    p = rq.decide_request(v, [], case, "v1")
    assert p["model"] == "claude-haiku-5-5" and p["max_tokens"] == 6000 and "thinking" not in p
    assert p["output_config"]["effort"] == "high"
    assert p["output_config"]["format"]["schema"]["additionalProperties"] is False
    s = rq.summary_request("", ["e0001 x"], 1000)
    assert s["max_tokens"] == 1500 and s["thinking"] == {"type": "disabled"} and s["output_config"]["effort"] == "medium"
    n = rq.note_request(v, case)
    assert n["output_config"]["format"]["schema"]["required"] == ["note", "p_residue"]
    assert "p_residue" in n["messages"][0]["content"] and "thinking" not in n
    k = rq.request_key(p)
    assert k == rq.request_key(json.loads(json.dumps(p))) and len(rq.custom_id_for(k)) <= 64
    assert rq.request_key(rq.decide_request(v, [], case, "v2")) != k


def test_wrapper_runner_unsets_base_url(monkeypatch):
    seen = {}

    def fake_run(cmd, **kw):
        seen["cmd"] = cmd
        return subprocess.CompletedProcess(cmd, 0, stdout="{}", stderr="")

    monkeypatch.setattr(rq.subprocess, "run", fake_run)
    rq.wrapper_runner("POST", "/v1/messages/count_tokens", "{}")
    assert seen["cmd"][:3] == ["env", "-u", "ANTHROPIC_BASE_URL"]
    assert seen["cmd"][3].endswith("anthropic-api") and "--body" in seen["cmd"]


def test_parse_answers():
    ok = {"result_type": "succeeded", "stop_reason": "end_turn", "text": '{"action":"EXPLAIN","reason":"r","event_ids":[]}',
          "usage": {}}
    assert rq.parse_answer("decide", ok, {})["action"] == "EXPLAIN"
    for stop in ("refusal", "max_tokens"):
        assert rq.parse_answer("decide", {**ok, "stop_reason": stop}, {})["action"] == PARSE_ERROR
    note = {**ok, "text": '{"note": "abc", "p_residue": 1.5}'}
    a = rq.parse_answer("note", note, {})
    assert a["note"] == "abc" and a["p_residue"] is None and a["status"] == "p_parse"
    s = rq.parse_answer("summary", {**ok, "text": '{"text": "' + "x " * 3000 + '"}', "stop_reason": "max_tokens"},
                        {"budget": 500})
    assert s["status"] == "max_tokens" and len(s["text"]) <= 2000
    assert rq.failure_answer("decide", "errored")["action"] == PARSE_ERROR


# ---------------------------------------------------------------------------
# round-parallel ddmin vs sequential
# ---------------------------------------------------------------------------

def _seq(case, view_ids):
    return flip_search(case, event_view(case, view_ids, "t"), oracle_decider, case.a_star, system="t", budget=None,
                       model="oracle")


def _par(case, view_ids):
    return round_flip_search(case, event_view(case, view_ids, "t"), oracle_decider, case.a_star, system="t",
                             budget=None, model="oracle")


def _passes(case, view_ids, ids):
    return oracle_decider(event_view(case, set(view_ids) | set(ids), "t"), [], case) == case.a_star


def test_round_parallel_recovers_dstar_like_sequential():
    mix = {"M0": 0.0, "M1": 1.0, "M2": 1.0, "M3": 1.0, "U1": 0.0, "U2": 0.0}
    cases = generate_cases(90, 11, Knobs(length=400), mix=mix) + generate_cases(60, 12, Knobs(length=100), mix=mix)
    for c in cases:
        view = build_view(c, "drop-dstar", 0.3)
        s, p = _seq(c, view), _par(c, view)
        assert s.status == p.status == "restored", c.case_id
        assert s.event_ids == p.event_ids == c.d_star, c.case_id
        assert p.one_minimal and not p.hit_cap and p.granularity == "event"
        assert p.calls_used <= 48 and sum(p.phase_calls) == p.calls_used
        assert set(p.groups) == set(s.groups)


def test_round_parallel_results_pass_and_match_when_uncapped():
    cases = generate_cases(200, 3, Knobs(length=400))
    flipped = same = 0
    for c in cases:
        view = build_view(c, "random", 0.3)
        s, p = _seq(c, view), _par(c, view)
        assert s.status == p.status
        if p.status != "restored":
            continue
        flipped += 1
        assert _passes(c, view, p.event_ids), c.case_id
        assert p.calls_used <= 48
        if not s.hit_cap and not p.hit_cap:
            assert p.event_ids == s.event_ids, c.case_id
            same += 1
    assert flipped > 50 and same > 20


def test_pddmin_cap_returns_passing_subset_and_prefix_order():
    asked = []

    def ask_many(sets):
        asked.append(list(sets))
        return ["T" if {"a", "h"} <= s else "F" for s in sets]

    s = Searcher(ask_many, "T", 6)
    s.memo[frozenset("abcdefgh")] = "T"
    s.memo[frozenset()] = "F"
    r = pddmin(list("abcdefgh"), frozenset, s)
    assert r.hit_cap and {"a", "h"} <= set(r.minimal) and s.calls <= 6
    unc = Searcher(ask_many, "T", 100)
    unc.memo[frozenset("abcdefgh")] = "T"
    r2 = pddmin(list("abcdefgh"), frozenset, unc)
    assert sorted(r2.minimal) == ["a", "h"] and r2.one_minimal


def test_group_search_replays_from_answers():
    """A replay with partial answers registers the same frontier and ends with the same result."""
    c = next(c for c in generate_cases(40, 4, Knobs(length=100)) if c.mechanism == "M1")
    view = event_view(c, build_view(c, "drop-dstar", 0.3), "t")
    from lens.flip.flip import missing_units
    units = missing_units(c, view)
    answers: dict[frozenset, str] = {}
    rounds = 0
    while True:
        frontier = []

        def ask_many(sets):
            miss = [s for s in sets if s not in answers]
            if miss:
                frontier.extend(miss)
                raise Pending
            return [answers[s] for s in sets]

        try:
            res = group_search(units, c.a_star, ask_many)
            break
        except Pending:
            rounds += 1
            for s in frontier:
                answers[s] = oracle_decider(view, [u for u in units if u.id in s], c)
    assert res["status"] == "restored" and res["ids"] == c.d_star and rounds == res["rounds"]


# ---------------------------------------------------------------------------
# engine: resume, routing, dedupe, spend, retries
# ---------------------------------------------------------------------------

def test_resume_mid_run_matches_uninterrupted(tmp_path, full_run):
    d0, _, _ = full_run
    d = _init(tmp_path, 12, 5)
    eng, _ = _engine(d)
    for _ in range(7):
        eng.step(1e9)
    assert eng.round_state() is not None or not eng.done()
    eng.close()
    # a fresh process: new Engine, new fake that knows nothing about earlier batches except the round in flight
    eng2 = E.Engine(d, check_protocol=False)
    model = M.MockModel(eng2.cases, M.v4_calibration(), seed=1)
    if eng2.round_state() is not None:
        old = eng.runner
        eng2.runner = old  # the batch in flight lives on the (fake) server
    else:
        eng2.runner = M.FakeBatchAPI(model.answer, M.engine_meta_fn(d))
    _finish(eng2)
    eng2.close()
    assert _state_blob(d) == _state_blob(d0)


def test_crash_between_advance_and_submit_loses_nothing(tmp_path):
    d = _init(tmp_path, 6, 2)
    eng, fake = _engine(d)
    eng.step(1e9)
    eng.step(1e9)  # fetched round 1
    pend = eng.advance()  # advance, then "crash" before submit
    assert pend
    eng.close()
    eng2, fake2 = _engine(d)
    assert set(eng2.advance()) == set(pend)
    _finish(eng2)
    assert eng2.done()


def test_routing_by_custom_id_with_shuffled_results(tmp_path, full_run):
    d0, _, _ = full_run
    d = _init(tmp_path, 12, 5)
    eng, fake = _engine(d)
    inner = fake.__call__

    def shuffled(method, path, body=None, timeout=120):
        rc, out = inner(method, path, body, timeout)
        if path.endswith("/results"):
            lines = out.splitlines()
            random.Random(len(lines)).shuffle(lines)
            out = "\n".join(lines) + "\n"
        return rc, out

    eng.runner = shuffled
    _finish(eng)
    assert _state_blob(d) == _state_blob(d0)


def test_identical_requests_are_sent_once(full_run):
    d, fake, _ = full_run
    assert len(fake.sent_custom_ids) == len(set(fake.sent_custom_ids))
    assert all(p[1].startswith("/v1/messages/batches") for p in fake.calls)
    # the flip search's first call (view alone) is the a_V request: deduped through the cache, never re-sent
    rounds = [json.loads(p.read_text()) for p in sorted((d / "rounds").glob("*/round.json"))]
    keys = [m["key"] for r in rounds for m in r["requests"].values()]
    assert len(keys) == len(set(keys))


def test_cached_replies_are_not_sent_again(tmp_path, full_run):
    d0, fake0, _ = full_run
    d = _init(tmp_path, 12, 5)
    import shutil
    shutil.copyfile(d0 / "state" / "calls.sqlite", d / "state" / "calls.sqlite")
    eng, fake = _engine(d)
    assert eng.step(1e9) == "complete"
    assert fake.requests_sent == 0 and _state_blob(d) == _state_blob(d0)


def test_batches_split_by_request_count(tmp_path):
    d = _init(tmp_path, 6, 3, batch={"max_requests": 7, "max_bytes": 10**9})
    eng, fake = _engine(d)
    eng.step(1e9)
    rs = eng.round_state()
    assert len(rs["batches"]) == -(-rs["n_requests"] // 7) and len(rs["batches"]) > 1
    assert fake.created == len(rs["batches"])
    _finish(eng)
    assert eng.done()


def test_spend_guard_refuses_and_records_actual(tmp_path):
    d = _init(tmp_path, 6, 4)
    eng, fake = _engine(d)
    with pytest.raises(E.Refused):
        eng.step(0.001)
    assert fake.created == 0 and eng.round_state() is None
    assert eng.step(10.0) == "submitted"
    bound = eng.round_state()["upper_bound_usd"]
    assert eng.step(10.0) == "fetched"
    sp = json.loads((d / "spend.json").read_text())
    lines = [json.loads(x) for p in (d / "rounds" / "001").glob("results-*.jsonl") for x in p.read_text().splitlines()]
    tin = sum(ln["result"]["message"]["usage"]["input_tokens"] for ln in lines)
    tout = sum(ln["result"]["message"]["usage"]["output_tokens"] for ln in lines)
    assert sp["input_tokens"] == tin and sp["output_tokens"] == tout
    assert sp["actual_usd"] == pytest.approx(rq.price(tin, tout), abs=1e-6)
    assert sp["actual_usd"] <= bound
    # next round: spent + bound must fit under the cap
    eng.advance()
    nxt = eng.plan_round(eng.pending)["upper_bound_usd"]
    with pytest.raises(E.Refused):
        eng.step(sp["actual_usd"] + nxt * 0.5)
    assert eng.step(sp["actual_usd"] + nxt + 0.01) == "submitted"


def test_errored_requests_retry_once_then_fail(tmp_path):
    d = _init(tmp_path, 6, 6)
    seen: dict[str, int] = {}

    def inject(cid, params, meta):
        if meta.get("stage") == "a_H" and meta.get("case_id", "").endswith("0000"):
            seen[cid] = seen.get(cid, 0) + 1
            return {"custom_id": cid, "result": {"type": "errored", "error": {"type": "error",
                                                                               "error": {"type": "overloaded_error"}}}}
        return None

    eng, fake = _engine(d, inject=inject)
    _finish(eng)
    assert list(seen.values()) == [2]
    st = json.loads((d / "state" / "cases" / f"{eng.cases[0].case_id}.json").read_text())
    assert st["case"]["a_H"] == PARSE_ERROR
    assert st["case"]["answers"]["aH"]["status"].startswith("failed:")
    assert len(json.loads((d / "state" / "failed.json").read_text())) == 1


def test_waiting_batches_poll_without_blocking(tmp_path):
    d = _init(tmp_path, 6, 7)
    eng, fake = _engine(d, polls_before_end=2)
    assert eng.step(1e9) == "submitted"
    assert eng.step(1e9) == "waiting"
    assert eng.step(1e9) == "waiting"
    assert eng.step(1e9) == "fetched"
    sleeps = []
    assert eng.run(1e9, 5, sleep=sleeps.append) == "complete"
    assert sleeps and set(sleeps) == {5}


# ---------------------------------------------------------------------------
# residue metrics
# ---------------------------------------------------------------------------

def test_tail_and_note_parsing():
    text = " ".join(f"w{i:03d}" for i in range(400))
    t = tail_tokens(text, 150)
    assert len(t) <= 600 and text.endswith(t) and t.split()[0] in text.split()
    assert parse_note_reply('{"note": "n", "p_residue": 0.25}') == ("n", 0.25)
    assert parse_note_reply('{"note": "n", "p_residue": "high"}') == ("n", None)
    assert parse_note_reply('{"note": "n", "p_residue": true}') == ("n", None)


def test_auroc_known_values():
    w = lambda xs: [((s, y), 1.0) for s, y in xs]
    assert HM.auroc(w([(0.9, True), (0.8, True), (0.1, False), (0.2, False)])) == 1.0
    assert HM.auroc(w([(0.1, True), (0.9, False)])) == 0.0
    assert HM.auroc(w([(0.5, True), (0.5, False)])) == 0.5
    assert HM.auroc(w([(0.5, True)])) is None
    assert HM.auroc([((0.9, True), 2.0), ((0.5, False), 1.0), ((0.95, False), 1.0)]) == 0.5


def _toy(n_cases=40, flips=25, seed=0):
    """Toy residue rows: notes drop D* on `flips` cases; p_residue tracks the flips."""
    from lens.flip.schema import Case
    rng = random.Random(seed)
    cases, rows, recs = [], [], {}
    for i in range(n_cases):
        cid = f"t{i:03d}"
        cases.append(Case(case_id=cid, seed=i, mechanism="M1", events=[], query_period="2020-01", query_text="q",
                          actions=[], a_star="OPEN_REVIEW", d_star=["e1"], deleted_ids=[], cls="rare", knobs={}))
        flip = i < flips
        cands = ["e1"] + [f"e{k}" for k in range(2, 12)]
        rows.append({"case_id": cid, "system": "recent", "a_star": "OPEN_REVIEW", "a_A": "OPEN_REVIEW",
                     "a_B": "EXPLAIN" if flip else "OPEN_REVIEW", "a_floor": "OPEN_REVIEW" if i % 10 else "EXPLAIN",
                     "a_trunc": "EXPLAIN" if i % 2 else "OPEN_REVIEW", "residue_flip": flip, "dstar_in_view": True,
                     "candidate_ids": cands if flip else [], "note_status": "ok" if i % 13 else "p_parse",
                     "p_residue": None if i % 13 == 0 else (0.7 + rng.random() * 0.3 if flip else rng.random() * 0.6),
                     "omission": 0.5})
        if flip:
            recs[(cid, "recent")] = {"status": "restored", "event_ids": ["e1"], "hit_cap": False}
    return cases, rows, recs


def test_residue_block_floor_trunc_h2a_h2b_h2c():
    cases, rows, recs = _toy()
    cfg = {"power_min_flips": 20, "h2b_draws": 1000, "bootstrap": {"seed": 1}}
    b = HM.residue_block(HM.Boot(len(cases), 300, 1), cases, rows, recs, cfg, n_flips_run=25)
    assert b["n_eligible"] == 40 and b["n_residue_flips"] == 25
    assert b["residue_rate"]["value"] == 25 / 40
    assert b["floor_rate"]["value"] == 4 / 40
    assert b["trunc_rate"]["value"] == 20 / 40
    assert b["h2a_diff"] == pytest.approx(21 / 40) and b["h2a_ci95"][0] > 0 and b["h2a_verdict"] == "supported"
    assert b["h2b_n"] == 25 and b["h2b_recall"]["value"] == 1.0
    assert b["h2b_random_recall"]["value"] == pytest.approx(1 / 11, abs=0.03)  # one of 11 candidates
    assert b["h2b_verdict"] == "supported"
    assert b["p_residue_parse_errors"] == 4  # i = 0, 13, 26, 39: counted, never imputed
    assert b["h2c_n"] == 36 and b["h2c_auroc"]["value"] == 1.0
    assert b["h2c_auroc_omission"]["value"] == 0.5  # constant omission: ties
    assert b["h2c_verdict"] == "supported"
    weak = HM.residue_block(HM.Boot(len(cases), 100, 1), cases, rows, recs, cfg, n_flips_run=19)
    assert weak["power_rule"]["powered"] is False
    assert weak["h2b_verdict"].startswith("descriptive") and weak["h2c_verdict"].startswith("descriptive")
    assert weak["h2a_verdict"] == "supported"  # the power rule covers H2b and H2c only


def test_h2b_random_baseline_uses_same_size_and_candidates():
    cases, rows, recs = _toy(5, 5)
    rows[0]["candidate_ids"] = ["e1", "e2"]
    recs[("t000", "recent")]["event_ids"] = ["e1", "e2"]
    items = HM.h2b_items(cases, rows, recs, 500, 3)
    first = next(i for i in items if i["case_id"] == "t000")
    assert first["k"] == 2 and first["random_recall"] == 1.0 and first["recall"] == 1.0


# ---------------------------------------------------------------------------
# protocol freeze
# ---------------------------------------------------------------------------

def test_freeze_hashes_change_with_a_prompt(tmp_path, monkeypatch):
    d = _init(tmp_path, 6, 8)
    doc = tmp_path / "proto.md"
    res = P.freeze([d], doc)
    before = res["hashes"]
    assert {"decider.v1", "decider.v2", "summary.prompt", "note.prompt_p_residue", "note.schema",
            "file:generator.py", "event_templates"} <= set(before)
    text = doc.read_text()
    assert "H2a" in text and "H2b" in text and "H2c" in text and before["file:generator.py"] in text
    monkeypatch.setitem(decider.POLICY, "v1", decider.POLICY["v1"] + " Changed.")
    after = P.current_hashes()
    assert after["decider.v1"] != before["decider.v1"] and after["decider.v2"] == before["decider.v2"]
    eng = E.Engine(d, M.FakeBatchAPI(lambda *a: None))
    with pytest.raises(E.Refused, match="decider.v1"):
        eng.step(1.0)
    eng.close()


def test_step_needs_a_freeze_and_unchanged_run_files(tmp_path):
    d = _init(tmp_path, 6, 9)
    eng = E.Engine(d, M.FakeBatchAPI(lambda *a: None))
    with pytest.raises(E.Refused, match="protocol.json"):
        eng.step(1.0)
    eng.close()
    P.freeze([d], tmp_path / "p.md")
    cfg = json.loads((d / "config.json").read_text())
    cfg["max_calls"] = 24
    (d / "config.json").write_text(json.dumps(cfg))
    eng = E.Engine(d, M.FakeBatchAPI(lambda *a: None))
    with pytest.raises(E.Refused, match="config.json"):
        eng.step(1.0)
    eng.close()


# ---------------------------------------------------------------------------
# end to end
# ---------------------------------------------------------------------------

REQUIRED_RESIDUE = ["n_eligible", "residue_rate", "floor_rate", "trunc_rate", "h2a_diff", "h2a_ci95", "h2b_recall",
                    "h2b_random_recall", "h2b_diff", "h2b_ci95", "h2c_auroc", "h2c_auroc_omission",
                    "h2c_diff_vs_half", "h2c_ci95_vs_half", "h2c_diff_vs_omission", "h2c_ci95_vs_omission",
                    "n_residue_flips", "p_residue_parse_errors", "power_rule"]


def test_end_to_end_summary_has_every_field(full_run):
    d, fake, s = full_run
    assert s["complete"] and s["n_cases"] == 12
    assert {"validity", "excluded_from_gated", "cells", "residue", "cost", "bootstrap"} <= set(s)
    assert len(s["cells"]) == 12
    for c in s["cells"].values():
        for block in ("fcr_raw", "fcr_gated"):
            assert {"all", "rare", "rare_reweighted_1pct"} <= set(c[block])
        assert {"count", "rate", "rate_gated"} <= set(c["harmful_flip"])
        assert {"U1_U2", "U1_U2_gated"} <= set(c["right_restraint"])
        assert "value" in c["over_caution_M0"]
        assert {"searches", "cap_hits", "mean_calls", "precision_vs_dstar", "recall_vs_dstar"} <= set(c["flip_search"])
    pooled = s["residue"]["pooled"]
    assert set(REQUIRED_RESIDUE) <= set(pooled), set(REQUIRED_RESIDUE) - set(pooled)
    assert set(s["residue"]["by_system"]) == {"recent", "rag-bm25", "rag-embed", "summary"}
    assert s["cost"]["rounds"] > 3 and s["cost"]["actual_usd"] > 0
    for f in ("decisions.jsonl", "records.jsonl", "residue.jsonl", "residue_records.jsonl", "tables.txt",
              "summary.json", "spend.json"):
        assert (d / f).exists(), f
    assert "H2a" in (d / "tables.txt").read_text()


def test_dryrun_end_to_end_offline(tmp_path):
    from lens.flip.headline import dryrun as D
    d = _init(tmp_path, 6, 10)
    fake = M.FakeBatchAPI(lambda *a: None)
    res = D.run_dryrun(d, calibrate_k=12, runner=fake, log=lambda m: None)
    calls = [c for c in fake.calls if c[1] == "/v1/messages/count_tokens"]
    assert len(calls) == res["calibration"]["calls"] == 12 and len(fake.calls) == 12
    assert res["no_batch_submitted"] and fake.created == 0
    assert [r["stage"] for r in res["by_stage"]][0] == "a_H" and res["rounds"] > 3
    assert res["cap_limited_upper_usd"] >= res["likely_usd"] > 0
    assert (d / "dryrun.json").exists() and (d / "calibration.json").exists()
    fake2 = M.FakeBatchAPI(lambda *a: None)
    d2 = _init(tmp_path, 6, 11)
    D.run_dryrun(d2, calibration=d / "calibration.json", runner=fake2, log=lambda m: None)
    assert fake2.calls == []
    assert "no batch was submitted" in D.format_tables(res)


def test_no_banned_words_in_headline_code_prompts_and_protocol(tmp_path):
    if not FORBIDDEN:
        pytest.skip("banned_words.local.txt not present")
    root = Path(P.__file__).parent
    files = list(root.glob("*.py")) + [root.parent / "pddmin.py"]
    blob = "\n".join(f.read_text() for f in files).lower()
    blob += "\n".join(P.prompt_texts().values()).lower()
    d = _init(tmp_path, 6, 12)
    doc = tmp_path / "proto.md"
    P.freeze([d], doc)
    blob += doc.read_text().lower()
    assert [w for w in FORBIDDEN if w in blob] == []

"""Batch validity check: request shapes, pricing, submit guards, fetch scoring. No network."""

from __future__ import annotations

import json
from pathlib import Path
from collections import Counter

import pytest

from lens.flip import batch
from lens.flip.decider import PARSE_ERROR
from lens.flip.schema import ACTIONS, MECHANISMS, read_jsonl


class FakeRunner:
    def __init__(self, replies):
        self.replies = list(replies)  # callable(method, path, body) or (rc, out)
        self.calls = []

    def __call__(self, method, path, body=None, timeout=120):
        self.calls.append((method, path, body))
        r = self.replies.pop(0) if len(self.replies) > 1 else self.replies[0]
        return r(method, path, body) if callable(r) else r


@pytest.fixture(scope="module")
def sample_dir(tmp_path_factory):
    d = tmp_path_factory.mktemp("validity")
    cases = batch.make_sample(60, 11)
    from lens.flip.schema import write_jsonl
    write_jsonl(d / "cases.jsonl", cases)
    # The legacy arms keep the old v1 and v2 result dirs covered. The new default arms have their own tests below.
    batch._write_lines(d / "requests.jsonl", batch.build_requests(cases, batch.LEGACY_ARMS))
    return d


def _fresh(sample_dir, tmp_path):
    for f in ("cases.jsonl", "requests.jsonl"):
        (tmp_path / f).write_bytes((sample_dir / f).read_bytes())
    return tmp_path


def test_sample_balance(sample_dir):
    cases = list(read_jsonl(sample_dir / "cases.jsonl"))
    assert len(cases) == 60
    assert Counter(c.mechanism for c in cases) == {m: 10 for m in MECHANISMS}
    for m in MECHANISMS:
        assert Counter(c.knobs["length"] for c in cases if c.mechanism == m) == {100: 5, 400: 5}
    assert len({c.case_id for c in cases}) == 60


def test_sample_rejects_unbalanced_n():
    with pytest.raises(ValueError):
        batch.make_sample(50, 1)


def test_request_shapes(sample_dir):
    reqs = batch._read_lines(sample_dir / "requests.jsonl")
    assert len(reqs) == 120
    ids = [r["custom_id"] for r in reqs]
    assert len(set(ids)) == 120
    assert all(batch.CUSTOM_ID_RE.match(i) for i in ids)
    cases = {c.case_id: c for c in read_jsonl(sample_dir / "cases.jsonl")}
    for cid in cases:
        assert f"{cid}__off" in ids and f"{cid}__adaptive" in ids
    for r in reqs:
        p = r["params"]
        arm = batch.arm_of(r["custom_id"])
        assert p["model"] == "claude-haiku-5-5"
        assert isinstance(p["system"], str) and p["system"]
        assert [m["role"] for m in p["messages"]] == ["user"]
        assert "Records:" in p["messages"][0]["content"]
        oc = p["output_config"]
        assert oc["effort"] == "medium"
        assert oc["format"]["type"] == "json_schema"
        sch = oc["format"]["schema"]
        assert sch["additionalProperties"] is False
        assert sch["properties"]["action"]["enum"] == list(ACTIONS) and len(ACTIONS) == 5
        assert set(sch["required"]) == set(sch["properties"])
        if arm == "off":
            assert p["thinking"] == {"type": "disabled"} and p["max_tokens"] == 1024
        else:
            assert "thinking" not in p and p["max_tokens"] == 8000


def test_requests_have_all_events(sample_dir):
    cases = {c.case_id: c for c in read_jsonl(sample_dir / "cases.jsonl")}
    r = batch._read_lines(sample_dir / "requests.jsonl")[0]
    case = cases[r["custom_id"].rsplit("__", 1)[0]]
    text = r["params"]["messages"][0]["content"]
    assert text.count("\n") >= len(case.events)


def test_banned_words_absent(sample_dir):
    # The word list is private. It lives in a gitignored file, one word per line.
    banned_file = Path(__file__).resolve().parents[2] / "banned_words.local.txt"
    if not banned_file.exists():
        pytest.skip("banned_words.local.txt not present")
    banned = [w.strip().lower() for w in banned_file.read_text().splitlines() if w.strip()]
    blob = (sample_dir / "requests.jsonl").read_text().lower()
    assert [w for w in banned if w in blob] == []


def test_dryrun_arithmetic(sample_dir, tmp_path):
    d = _fresh(sample_dir, tmp_path)

    def reply(method, path, body):
        assert path == "/v1/messages/count_tokens"
        b = json.loads(body)
        assert set(b) <= {"model", "system", "messages", "thinking"}
        return 0, json.dumps({"input_tokens": 1000 if "thinking" in b else 3000})

    runner = FakeRunner([reply])
    res = batch.run_dryrun_impl(d, None, runner)
    assert len(runner.calls) == 120
    assert all(c[0] == "POST" for c in runner.calls)
    off, ada = res["arms"]["off"], res["arms"]["adaptive"]
    assert off["input_tokens"] == 60_000 and ada["input_tokens"] == 180_000
    assert off["likely_usd"] == pytest.approx((60_000 * 0.05 + 150 * 60 * 0.25) / 1e6)
    assert ada["likely_usd"] == pytest.approx((180_000 * 0.05 + 1500 * 60 * 0.25) / 1e6)
    assert off["upper_bound_usd"] == pytest.approx((60_000 * 0.05 + 1024 * 60 * 0.25) / 1e6)
    assert ada["upper_bound_usd"] == pytest.approx((180_000 * 0.05 + 8000 * 60 * 0.25) / 1e6)
    assert res["upper_bound_usd"] == pytest.approx(off["upper_bound_usd"] + ada["upper_bound_usd"], abs=1e-5)
    assert res["complete"] is True
    assert json.loads((d / "dryrun.json").read_text())["upper_bound_usd"] == res["upper_bound_usd"]


def test_dryrun_sample_extrapolates(sample_dir, tmp_path):
    d = _fresh(sample_dir, tmp_path)
    runner = FakeRunner([(0, json.dumps({"input_tokens": 500}))])
    res = batch.run_dryrun_impl(d, 10, runner)
    assert len(runner.calls) == 10
    assert res["complete"] is False and res["sample"] == 10
    assert res["arms"]["off"]["n_counted"] == 5
    assert res["arms"]["off"]["input_tokens"] == 500 * 60


def test_dryrun_backoff_and_drop_thinking(sample_dir, tmp_path):
    d = _fresh(sample_dir, tmp_path)
    sleeps = []
    err429 = (1, json.dumps({"error": {"type": "rate_limit_error", "message": "slow"}}))
    err400 = (1, json.dumps({"error": {"type": "invalid_request_error", "message": "thinking: not allowed"}}))
    ok = (0, json.dumps({"input_tokens": 7}))

    def reply(method, path, body):
        b = json.loads(body)
        if "thinking" in b:
            return err400
        return ok

    runner = FakeRunner([err429, reply])
    notes: list[str] = []
    drop: set[str] = set()
    off = batch._read_lines(d / "requests.jsonl")
    off_req = next(r for r in off if r["custom_id"].endswith("__off"))
    n = batch.count_one(runner, off_req["params"], drop, notes, sleep=sleeps.append)
    assert n == 7 and drop == {"thinking"} and len(sleeps) == 1 and notes


def test_dryrun_hard_failure(sample_dir, tmp_path):
    d = _fresh(sample_dir, tmp_path)
    runner = FakeRunner([(1, json.dumps({"error": {"type": "invalid_request_error", "message": "bad"}}))])
    with pytest.raises(RuntimeError):
        batch.run_dryrun_impl(d, None, runner)


def _dryrun_file(d, upper, complete=True):
    (d / "dryrun.json").write_text(json.dumps({
        "upper_bound_usd": upper, "likely_usd": upper / 2, "complete": complete,
        "requests_sha256": batch._sha(d / "requests.jsonl"),
    }))


def test_submit_refuses_above_cap(sample_dir, tmp_path):
    d = _fresh(sample_dir, tmp_path)
    _dryrun_file(d, 0.50)
    runner = FakeRunner([(0, "{}")])
    with pytest.raises(batch.Refused):
        batch.submit_impl(d, 0.25, runner)
    assert runner.calls == [] and not (d / "batch.json").exists()


def test_submit_refuses_existing_batch(sample_dir, tmp_path):
    d = _fresh(sample_dir, tmp_path)
    _dryrun_file(d, 0.10)
    (d / "batch.json").write_text("{}")
    runner = FakeRunner([(0, "{}")])
    with pytest.raises(batch.Refused):
        batch.submit_impl(d, 1.0, runner)
    assert runner.calls == []


def test_submit_refuses_without_full_dryrun(sample_dir, tmp_path):
    d = _fresh(sample_dir, tmp_path)
    runner = FakeRunner([(0, "{}")])
    with pytest.raises(batch.Refused):
        batch.submit_impl(d, 1.0, runner)
    _dryrun_file(d, 0.10, complete=False)
    with pytest.raises(batch.Refused):
        batch.submit_impl(d, 1.0, runner)
    _dryrun_file(d, 0.10)
    (d / "requests.jsonl").write_text((d / "requests.jsonl").read_text() + "\n")
    with pytest.raises(batch.Refused):
        batch.submit_impl(d, 1.0, runner)
    assert runner.calls == []


def test_submit_ok(sample_dir, tmp_path):
    d = _fresh(sample_dir, tmp_path)
    _dryrun_file(d, 0.10)
    resp = {"id": "msgbatch_abc", "created_at": "2026-10-20T00:00:00Z", "processing_status": "in_progress"}
    runner = FakeRunner([(0, json.dumps(resp))])
    info = batch.submit_impl(d, 0.25, runner)
    assert len(runner.calls) == 1
    method, path, body = runner.calls[0]
    assert (method, path) == ("POST", "/v1/messages/batches")
    assert len(json.loads(body)["requests"]) == 120
    saved = json.loads((d / "batch.json").read_text())
    assert saved["id"] == "msgbatch_abc" and saved["n_requests"] == 120 and info["id"] == saved["id"]
    with pytest.raises(batch.Refused):
        batch.submit_impl(d, 0.25, runner)
    assert len(runner.calls) == 1


# ---------------------------------------------------------------------------
# fetch
# ---------------------------------------------------------------------------

def _ok_line(cid, action, blocks_before=(), usage=None, stop="end_turn"):
    content = list(blocks_before) + [{"type": "text", "text": json.dumps(
        {"action": action, "reason": "r", "event_ids": []})}]
    return {"custom_id": cid, "result": {"type": "succeeded", "message": {
        "content": content, "stop_reason": stop,
        "usage": usage or {"input_tokens": 1_000_000, "output_tokens": 1_000_000}}}}


def test_parse_result_cases():
    thinking = [{"type": "thinking", "thinking": "hmm {\"action\": \"EXPLAIN\"}", "signature": "s"}]
    r = batch.parse_result(_ok_line("x__adaptive", "OPEN_REVIEW", thinking))
    assert r["a_H"] == "OPEN_REVIEW" and r["stop_reason"] == "end_turn"
    for stop in ("max_tokens", "refusal"):
        r = batch.parse_result(_ok_line("x__off", "EXPLAIN", stop=stop))
        assert r["a_H"] == PARSE_ERROR and r["stop_reason"] == stop
    r = batch.parse_result({"custom_id": "x", "result": {"type": "errored", "error": {}}})
    assert r["a_H"] == PARSE_ERROR and r["stop_reason"] == "errored"
    r = batch.parse_result({"custom_id": "x", "result": {"type": "succeeded", "message": {
        "content": [{"type": "text", "text": "not json"}], "stop_reason": "end_turn", "usage": {}}}})
    assert r["a_H"] == PARSE_ERROR
    assert batch.parse_result({})["stop_reason"] == "missing"


def test_fetch_not_ended(sample_dir, tmp_path, capsys):
    d = _fresh(sample_dir, tmp_path)
    (d / "batch.json").write_text(json.dumps({"id": "msgbatch_abc"}))
    status = {"processing_status": "in_progress", "request_counts": {"processing": 120, "succeeded": 0}}
    runner = FakeRunner([(0, json.dumps(status))])
    assert batch.fetch_impl(d, runner) is None
    assert [c[:2] for c in runner.calls] == [("GET", "/v1/messages/batches/msgbatch_abc")]
    assert not (d / "results.jsonl").exists()
    assert "in_progress" in capsys.readouterr().out


def test_fetch_scores(sample_dir, tmp_path):
    d = _fresh(sample_dir, tmp_path)
    (d / "batch.json").write_text(json.dumps({"id": "msgbatch_abc"}))
    cases = list(read_jsonl(d / "cases.jsonl"))
    lines = []
    thinking = [{"type": "thinking", "thinking": "t", "signature": "s"}]
    # Off arm: first 3 cases correct, 4th wrong, 5th max_tokens, 6th refusal, rest correct.
    # Adaptive arm: first case errored, second wrong, third correct after a thinking block, rest correct.
    for i, c in enumerate(cases):
        off, ada = f"{c.case_id}__off", f"{c.case_id}__adaptive"
        wrong = "REQUEST_INFO" if c.a_star != "REQUEST_INFO" else "EXPLAIN"
        if i == 3:
            lines.append(_ok_line(off, wrong))
        elif i == 4:
            lines.append(_ok_line(off, c.a_star, stop="max_tokens"))
        elif i == 5:
            lines.append(_ok_line(off, c.a_star, stop="refusal"))
        else:
            lines.append(_ok_line(off, c.a_star))
        if i == 0:
            lines.append({"custom_id": ada, "result": {"type": "errored", "error": {"type": "api_error"}}})
        elif i == 1:
            lines.append(_ok_line(ada, wrong, thinking))
        else:
            lines.append(_ok_line(ada, c.a_star, thinking))
    lines.reverse()  # results come back in any order
    status = {"processing_status": "ended", "request_counts": {"succeeded": 119, "errored": 1}}
    runner = FakeRunner([(0, json.dumps(status)),
                         (0, "\n".join(json.dumps(x) for x in lines) + "\n")])
    v = batch.fetch_impl(d, runner)
    assert [c[:2] for c in runner.calls] == [
        ("GET", "/v1/messages/batches/msgbatch_abc"), ("GET", "/v1/messages/batches/msgbatch_abc/results")]
    off, ada = v["arms"]["off"], v["arms"]["adaptive"]
    assert off["overall"] == {"n": 60, "correct": 57, "accuracy": 57 / 60}
    assert off["parse_errors"] == {"total": 2, "by_reason": {"max_tokens": 1, "refusal": 1}}
    assert ada["overall"] == {"n": 60, "correct": 58, "accuracy": 58 / 60}
    assert ada["parse_errors"] == {"total": 1, "by_reason": {"errored": 1}}
    assert off["passes_gate"] is True  # 57/60 = 0.95 meets the gate
    assert sum(m["n"] for m in off["per_mechanism"].values()) == 60
    assert sum(sum(row.values()) for row in off["confusion"].values()) == 60
    assert off["confusion"]["EXPLAIN"].get("PARSE_ERROR", 0) + sum(
        row.get("PARSE_ERROR", 0) for k, row in off["confusion"].items() if k != "EXPLAIN") == 2
    # 60 requests at 1M in / 1M out each: 60 * (0.05 + 0.25) dollars per arm.
    assert off["cost"]["usd"] == pytest.approx(60 * 0.30)
    assert ada["cost"]["usd"] == pytest.approx(59 * 0.30)  # the errored request has no usage
    assert v["total_cost_usd"] == pytest.approx(119 * 0.30)
    rows = batch._read_lines(d / "decisions.jsonl")
    assert len(rows) == 120
    assert set(rows[0]) >= {"case_id", "mechanism", "cls", "length", "arm", "a_star", "a_H", "stop_reason", "usage"}
    assert (d / "results.jsonl").exists()
    assert json.loads((d / "validity.json").read_text())["arms"]["off"]["overall"]["correct"] == 57


def test_cli_registered():
    import argparse
    from lens.flip import cli
    p = argparse.ArgumentParser()
    cli.register(p.add_subparsers(dest="cmd"))
    a = p.parse_args(["flip", "validity", "submit", "--dir", "x", "--cap-usd", "1"])
    assert a.func is batch.run_submit and a.cap_usd == 1.0


# ---------------------------------------------------------------------------
# adaptive-medium and adaptive-high arms (the default for `build`)
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def new_dir(tmp_path_factory):
    d = tmp_path_factory.mktemp("validity-new")
    from lens.flip.schema import write_jsonl
    cases = batch.make_sample(60, 11)
    write_jsonl(d / "cases.jsonl", cases)
    batch._write_lines(d / "requests.jsonl", batch.build_requests(cases))
    return d


def test_default_arms_are_adaptive_medium_and_high():
    assert batch.DEFAULT_ARMS == ("adaptive-medium", "adaptive-high")
    assert batch.parse_arms("adaptive-medium,adaptive-high") == batch.DEFAULT_ARMS
    assert batch.parse_arms("off,adaptive") == batch.LEGACY_ARMS
    for bad in ("", "adaptive-low", "off,off"):
        with pytest.raises(ValueError):
            batch.parse_arms(bad)


def test_new_arms_two_requests_per_case(new_dir):
    reqs = batch._read_lines(new_dir / "requests.jsonl")
    assert len(reqs) == 120
    cases = [c.case_id for c in read_jsonl(new_dir / "cases.jsonl")]
    ids = {r["custom_id"] for r in reqs}
    for cid in cases:
        assert f"{cid}__adaptive-medium" in ids and f"{cid}__adaptive-high" in ids
    assert all(batch.CUSTOM_ID_RE.match(i) for i in ids)
    for r in reqs:
        p, arm = r["params"], batch.arm_of(r["custom_id"])
        assert "thinking" not in p  # adaptive is the default on Haiku 5.5
        assert p["max_tokens"] == 6000
        assert p["output_config"]["effort"] == {"adaptive-medium": "medium", "adaptive-high": "high"}[arm]
        assert p["output_config"]["format"]["type"] == "json_schema"
        assert p["output_config"]["format"]["schema"]["additionalProperties"] is False
        assert p["model"] == "claude-haiku-5-5"


def test_arm_pairs_share_prompt(new_dir):
    reqs = batch._read_lines(new_dir / "requests.jsonl")
    by_case: dict[str, dict] = {}
    for r in reqs:
        case_id, arm = r["custom_id"].rsplit("__", 1)
        by_case.setdefault(case_id, {})[arm] = r["params"]
    for pair in by_case.values():
        a, b = pair["adaptive-medium"], pair["adaptive-high"]
        assert a["system"] == b["system"] and a["messages"] == b["messages"]


def test_legacy_arms_still_build():
    cases = batch.make_sample(12, 3)
    reqs = batch.build_requests(cases, batch.LEGACY_ARMS)
    by = {batch.arm_of(r["custom_id"]): r["params"] for r in reqs}
    assert by["off"]["thinking"] == {"type": "disabled"} and by["off"]["max_tokens"] == 1024
    assert "thinking" not in by["adaptive"] and by["adaptive"]["max_tokens"] == 8000


def test_build_cli_arms_option(tmp_path):
    import argparse
    from lens.flip import cli
    p = argparse.ArgumentParser()
    cli.register(p.add_subparsers(dest="cmd"))
    a = p.parse_args(["flip", "validity", "build", "--n", "12", "--seed", "3", "--out", str(tmp_path / "d")])
    assert a.arms == "adaptive-medium,adaptive-high"
    assert a.func(a) == 0
    reqs = batch._read_lines(tmp_path / "d" / "requests.jsonl")
    assert len(reqs) == 24 and {batch.arm_of(r["custom_id"]) for r in reqs} == set(batch.DEFAULT_ARMS)
    b = p.parse_args(["flip", "validity", "build", "--n", "12", "--seed", "3", "--arms", "off,adaptive",
                      "--out", str(tmp_path / "e")])
    assert b.func(b) == 0
    assert {batch.arm_of(r["custom_id"]) for r in batch._read_lines(tmp_path / "e" / "requests.jsonl")} == {"off", "adaptive"}
    c = p.parse_args(["flip", "validity", "build", "--n", "12", "--arms", "bogus", "--out", str(tmp_path / "f")])
    assert c.func(c) == 1 and not (tmp_path / "f").exists()


def test_dryrun_pricing_new_arms(new_dir, tmp_path):
    d = _fresh(new_dir, tmp_path)
    runner = FakeRunner([(0, json.dumps({"input_tokens": 5000}))])
    res = batch.run_dryrun_impl(d, None, runner)
    assert len(runner.calls) == 120
    assert all("thinking" not in json.loads(c[2]) for c in runner.calls)
    assert list(res["arms"]) == ["adaptive-medium", "adaptive-high"]
    med, high = res["arms"]["adaptive-medium"], res["arms"]["adaptive-high"]
    assert med["input_tokens"] == high["input_tokens"] == 300_000
    assert med["likely_usd"] == pytest.approx((300_000 * 0.05 + 1500 * 60 * 0.25) / 1e6)
    assert high["likely_usd"] == pytest.approx((300_000 * 0.05 + 3000 * 60 * 0.25) / 1e6)
    for a in (med, high):
        assert a["upper_bound_usd"] == pytest.approx((300_000 * 0.05 + 6000 * 60 * 0.25) / 1e6)
    assert res["upper_bound_usd"] == pytest.approx(med["upper_bound_usd"] + high["upper_bound_usd"], abs=1e-5)
    table = batch.format_table(res)
    assert "adaptive-medium" in table and "adaptive-high" in table and "total" in table


def test_dryrun_sample_split_new_arms(new_dir, tmp_path):
    d = _fresh(new_dir, tmp_path)
    runner = FakeRunner([(0, json.dumps({"input_tokens": 400}))])
    res = batch.run_dryrun_impl(d, 10, runner)
    assert len(runner.calls) == 10 and res["complete"] is False
    assert res["arms"]["adaptive-high"]["n_counted"] == 5
    assert res["arms"]["adaptive-high"]["input_tokens"] == 400 * 60


def test_fetch_scores_new_arms(new_dir, tmp_path):
    d = _fresh(new_dir, tmp_path)
    (d / "batch.json").write_text(json.dumps({"id": "msgbatch_new"}))
    cases = list(read_jsonl(d / "cases.jsonl"))
    thinking = [{"type": "thinking", "thinking": "t", "signature": "s"}]
    lines = []
    for i, c in enumerate(cases):
        wrong = "REQUEST_INFO" if c.a_star != "REQUEST_INFO" else "EXPLAIN"
        lines.append(_ok_line(f"{c.case_id}__adaptive-medium", wrong if i < 6 else c.a_star, thinking))
        lines.append(_ok_line(f"{c.case_id}__adaptive-high", c.a_star, thinking,
                              stop="max_tokens" if i == 0 else "end_turn"))
    status = {"processing_status": "ended", "request_counts": {"succeeded": 120}}
    runner = FakeRunner([(0, json.dumps(status)), (0, "\n".join(json.dumps(x) for x in lines) + "\n")])
    v = batch.fetch_impl(d, runner)
    assert list(v["arms"]) == ["adaptive-medium", "adaptive-high"]
    assert v["arms"]["adaptive-medium"]["overall"]["correct"] == 54
    assert v["arms"]["adaptive-high"]["overall"]["correct"] == 59
    assert v["arms"]["adaptive-high"]["parse_errors"] == {"total": 1, "by_reason": {"max_tokens": 1}}
    assert len(batch._read_lines(d / "decisions.jsonl")) == 120


@pytest.mark.parametrize("version", ("v1", "v2"))
def test_old_result_dirs_still_score(version):
    """fetch on the committed v1 and v2 dirs: the arms come from requests.jsonl."""
    from pathlib import Path
    d = Path(__file__).resolve().parents[2] / "results" / f"validity-haiku-{version}"
    if not (d / "results.jsonl").exists():
        pytest.skip("result dir not present")
    cases = list(read_jsonl(d / "cases.jsonl"))
    req_ids = [r["custom_id"] for r in batch._read_lines(d / "requests.jsonl")]
    lines = batch._read_lines(d / "results.jsonl")
    rows, validity = batch.score(cases, req_ids, lines)
    assert list(validity["arms"]) == ["off", "adaptive"]
    saved = json.loads((d / "validity.json").read_text())
    for arm in ("off", "adaptive"):
        assert validity["arms"][arm]["overall"] == saved["arms"][arm]["overall"]

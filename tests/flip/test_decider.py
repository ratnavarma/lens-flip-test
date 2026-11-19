"""Step 3C: LLM decider with fake transports. No test here calls Ollama."""

from __future__ import annotations

import json

import pytest

from lens.flip.cache import CallCache, make_key
from lens.flip.decider import (
    PARSE_ERROR, CallStats, build_messages, ollama_decider, parse_reply,
)
from lens.flip.generator import Knobs, generate_cases
from lens.flip.schema import ACTIONS, Unit
from lens.flip.systems.base import event_view


@pytest.fixture(scope="module")
def case():
    return generate_cases(4, 1, Knobs(length=100))[0]


def _view(case, events=None):
    ids = [e.id for e in (case.events if events is None else events)]
    return event_view(case, ids, "t")


class Fake:
    def __init__(self, replies):
        self.replies = list(replies)
        self.bodies = []

    def __call__(self, body):
        self.bodies.append(body)
        return self.replies[min(len(self.bodies) - 1, len(self.replies) - 1)]


GOOD = json.dumps({"action": "APPLY_CREDIT", "reason": "x.", "event_ids": ["e0001"]})


def test_fixed_reply_and_request_shape(case):
    t = Fake([GOOD])
    d = ollama_decider("m", transport=t)
    assert d(_view(case), [], case) == "APPLY_CREDIT"
    body = t.bodies[0]
    assert body["stream"] is False and body["think"] is False and body["format"]["properties"]["action"]["enum"] == list(ACTIONS)
    assert body["options"] == {"temperature": 0, "seed": 0, "num_ctx": 16384}
    assert [m["role"] for m in body["messages"]] == ["system", "user"]
    assert d.last_raw == GOOD and d.last_parsed["event_ids"] == ["e0001"]


def test_prompt_layout_and_versions(case):
    v1 = build_messages(_view(case, case.events[:3]), [], case, "v1")
    v2 = build_messages(_view(case, case.events[:3]), [], case, "v2")
    assert v1 != v2
    for msgs in (v1, v2):
        user = msgs[1]["content"]
        assert user.index(case.events[0].id) < user.index(case.query_text) < user.index("EXPLAIN:" if "EXPLAIN:" in user else "EXPLAIN means")
        assert all(a in user for a in ACTIONS)
        assert user.rstrip().endswith("}.")
    assert "reviewing an account's records" in v1[0]["content"]
    with pytest.raises(ValueError):
        ollama_decider("m", prompt_version="v9", transport=Fake([GOOD]))


def test_bad_then_good_retries_once(case):
    t = Fake(["not json", GOOD])
    d = ollama_decider("m", transport=t)
    assert d(_view(case), [], case) == "APPLY_CREDIT"
    assert len(t.bodies) == 2
    assert "not valid" in t.bodies[1]["messages"][-1]["content"]
    assert "not valid" not in t.bodies[0]["messages"][-1]["content"]


def test_always_bad_is_parse_error(case):
    t = Fake(["nope"])
    d = ollama_decider("m", transport=t)
    assert d(_view(case), [], case) == PARSE_ERROR
    assert len(t.bodies) == 2 and d.last_raw == "nope"


def test_unknown_action_counts_as_bad(case):
    t = Fake([json.dumps({"action": "SHRUG"}), json.dumps({"action": "explain"})])
    assert ollama_decider("m", transport=t)(_view(case), [], case) == "EXPLAIN"


def test_parse_reply_variants():
    assert parse_reply('```json\n{"action": "CHECK_DEVICE"}\n```')["action"] == "CHECK_DEVICE"
    assert parse_reply('[1]') is None and parse_reply('{"action": 3}') is None


def test_cache_makes_second_call_free(case, tmp_path):
    cache = CallCache(tmp_path / "c.sqlite")
    stats = CallStats()
    t = Fake([GOOD])
    d = ollama_decider("m", transport=t, cache=cache, stats=stats)
    assert d(_view(case), [], case) == "APPLY_CREDIT"
    assert (stats.real_calls, stats.cache_hits, len(t.bodies)) == (1, 0, 1)
    assert d(_view(case), [], case) == "APPLY_CREDIT"
    assert (stats.real_calls, stats.cache_hits, len(t.bodies)) == (1, 1, 1)
    # a new decider on the same file, and a different prompt version
    t2 = Fake(["unused"])
    d2 = ollama_decider("m", transport=t2, cache=CallCache(tmp_path / "c.sqlite"), stats=CallStats())
    assert d2(_view(case), [], case) == "APPLY_CREDIT" and t2.bodies == []
    d3 = ollama_decider("m", "v2", transport=Fake([GOOD]), cache=cache, stats=stats)
    d3(_view(case), [], case)
    assert stats.real_calls == 2


def test_cache_covers_retry_replies(case, tmp_path):
    cache = CallCache(tmp_path / "c.sqlite")
    t = Fake(["bad", GOOD])
    d = ollama_decider("m", transport=t, cache=cache)
    d(_view(case), [], case)
    t2 = Fake(["x"])
    d2 = ollama_decider("m", transport=t2, cache=cache)
    assert d2(_view(case), [], case) == "APPLY_CREDIT" and t2.bodies == []


def test_key_depends_on_each_input():
    base = make_key("m", "v1", [{"role": "user", "content": "a"}], {"seed": 0})
    assert base == make_key("m", "v1", [{"role": "user", "content": "a"}], {"seed": 0})
    for other in (
        make_key("n", "v1", [{"role": "user", "content": "a"}], {"seed": 0}),
        make_key("m", "v2", [{"role": "user", "content": "a"}], {"seed": 0}),
        make_key("m", "v1", [{"role": "user", "content": "b"}], {"seed": 0}),
        make_key("m", "v1", [{"role": "user", "content": "a"}], {"seed": 1}),
    ):
        assert other != base


# Two-causes rule in the action policy
SEASONAL_EXCLUSION = "Normal seasonal activity is never a problem or a cause for this rule."
TWO_CAUSES = ("the records do not settle the cause, or two or more different problems are present for the period, "
              "because each would call for a different action and a reviewer must decide first. " + SEASONAL_EXCLUSION)


def test_policy_has_two_causes_rule_and_keeps_other_lines():
    from lens.flip.decider import POLICY
    for version, lead in (("v1", "- REQUEST_INFO: "), ("v2", "- REQUEST_INFO means ")):
        lines = POLICY[version].split("\n")
        assert lines[-1] == lead + TWO_CAUSES
        assert len(lines) == 6
    # The other four meanings are unchanged.
    assert "- EXPLAIN: the higher amount is explained by normal seasonal activity." in POLICY["v1"]
    assert "- OPEN_REVIEW: the statement rests on a figure that later records contradict." in POLICY["v1"]
    assert "- CHECK_DEVICE: a device change during the period may have distorted the figures." in POLICY["v1"]
    assert "- APPLY_CREDIT: an approved credit was never applied." in POLICY["v1"]
    assert "- EXPLAIN means ordinary seasonal activity accounts for the higher amount." in POLICY["v2"]
    assert "- APPLY_CREDIT means a credit was approved and never applied." in POLICY["v2"]


@pytest.mark.parametrize("version", ("v1", "v2"))
def test_policy_excludes_seasonal_activity_from_problems_and_causes(version):
    from lens.flip.decider import POLICY
    assert SEASONAL_EXCLUSION in POLICY[version]
    assert "two or more different problems" in POLICY[version]
    assert "different causes" not in POLICY[version]


@pytest.mark.parametrize("version", ("v1", "v2"))
def test_policy_identical_in_ollama_and_batch_requests(case, version):
    from lens.flip import batch
    from lens.flip.decider import POLICY
    t = Fake([GOOD])
    ollama_decider("m", version, transport=t)(_view(case), [], case)
    ollama_user = t.bodies[0]["messages"][1]["content"]
    assert POLICY[version].encode() in ollama_user.encode()
    assert TWO_CAUSES in ollama_user
    if version == "v1":  # the batch path uses the v1 prompt
        params = batch.request_params(case, "adaptive-medium")
        batch_user = params["messages"][0]["content"]
        assert batch_user == ollama_user
        assert params["system"] == t.bodies[0]["messages"][0]["content"]
        assert POLICY["v1"].encode() in batch_user.encode()

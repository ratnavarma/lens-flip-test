"""Step 3B: memory systems."""

from __future__ import annotations

import math

import pytest

from lens.flip.generator import Knobs, generate_cases
from lens.flip.render import render_lines
from lens.flip.systems import SYSTEMS, FullSystem, RagBm25System, RecentSystem, get_system
from lens.flip.systems.base import count_tokens
from lens.flip.systems.rag_bm25 import bm25_scores, tokenise


@pytest.fixture(scope="module")
def cases():
    return generate_cases(12, 4, Knobs(length=400))


def test_count_tokens_is_ceil_chars_over_chars_per_token():
    from lens.flip.systems.base import CHARS_PER_TOKEN
    assert CHARS_PER_TOKEN == 2.2
    assert count_tokens([]) == 0
    assert count_tokens(["ab"]) == 1
    assert count_tokens(["abc"]) == 2
    assert count_tokens(["abcd", "efg"]) == math.ceil(8 / 2.2)  # 4 + 1 newline + 3


@pytest.mark.parametrize("cls", [RecentSystem, RagBm25System])
@pytest.mark.parametrize("budget", [50, 500, 1000, 2000])
def test_budget_respected_and_history_order(cases, cls, budget):
    sysm = cls()
    for c in cases:
        v = sysm.view(c, budget)
        ids = v.event_ids
        assert v.text == "\n".join(render_lines(c, ids)) and v.units == render_lines(c, ids)
        assert v.system == sysm.name and v.budget == budget
        order = {e.id: i for i, e in enumerate(c.events)}
        assert ids == sorted(ids, key=order.__getitem__) and len(set(ids)) == len(ids)
        assert count_tokens(render_lines(c, ids)) <= budget
        assert len(ids) < len(c.events)


def test_budget_fills_most_of_the_space(cases):
    c = cases[0]
    for sysm in (RecentSystem(), RagBm25System()):
        used = count_tokens(render_lines(c, sysm.view(c, 1000).event_ids))
        assert 900 <= used <= 1000


def test_full_ignores_budget(cases):
    c = cases[0]
    assert FullSystem().view(c, 10).event_ids == [e.id for e in c.events]
    assert FullSystem.reference is True and RecentSystem.reference is False


def test_recent_is_a_suffix(cases):
    for c in cases:
        ids = RecentSystem().view(c, 500).event_ids
        assert ids == [e.id for e in c.events][-len(ids):]


def test_tokeniser_keeps_period_whole():
    assert tokenise("Statement for 2015-10 sent, E0012 on 2015-10-04.") == [
        "statement", "for", "2015-10", "sent", "e0012", "on", "2015-10-04"]


def test_bm25_ranks_query_period_above_other_lines():
    docs = [
        "e1 2015-03-04 Statement issued for period 2015-03. Amount due: 80.00.",
        "e2 2015-11-04 Statement issued for period 2015-10. Amount due: 80.00.",
        "e3 2015-11-05 Payment received: 80.00.",
    ]
    s = bm25_scores(docs, "The statement for period 2015-10 is higher than usual.")
    assert s[1] > s[0] > s[2] or (s[1] > s[0] and s[1] > s[2])
    assert s[1] > max(s[0], s[2])


def test_bm25_view_contains_query_period_lines(cases):
    for c in cases[:6]:
        ids = set(RagBm25System().view(c, 1000).event_ids)
        tagged = [e.id for e in c.events if e.type == "statement" and e.fields.get("period") == c.query_period]
        assert tagged and all(i in ids for i in tagged)


def test_deterministic(cases):
    for name in ("full", "recent", "rag-bm25"):
        for c in cases[:4]:
            assert get_system(name).view(c, 500) == get_system(name).view(c, 500)


def test_unknown_system():
    with pytest.raises(ValueError):
        get_system("nope")

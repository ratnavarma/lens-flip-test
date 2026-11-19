"""Step 2 checks: ddmin, flip search with the oracle mock, Distinction Record."""

from __future__ import annotations

import itertools
import random

import pytest

from lens.flip.ddmin import ddmin
from lens.flip.decider import oracle_decider
from lens.flip.flip import flip_search
from lens.flip.generator import DISTANCES, Knobs, detect_patterns, generate_cases
from lens.flip.systems.base import event_view
from lens.flip.schema import (
    MECHANISMS,
    REQUEST_INFO,
    Case,
    DistinctionRecord,
)

IDS = [f"e{i:03d}" for i in range(100)]
UNIFORM = {m: 1.0 for m in MECHANISMS}


# ----- pure ddmin -----


def test_one_culprit():
    for culprit in (IDS[0], IDS[37], IDS[99]):
        r = ddmin(IDS, lambda s: culprit in s)
        assert r.minimal == [culprit]
        assert r.calls <= 20
        assert r.one_minimal and not r.hit_cap


def test_required_pair():
    pair = [IDS[12], IDS[80]]
    r = ddmin(IDS, lambda s: all(p in s for p in pair))
    assert r.minimal == pair
    assert r.one_minimal


def test_two_alternative_culprits():
    alts = {IDS[5], IDS[70]}
    r = ddmin(IDS, lambda s: bool(alts & set(s)))
    assert len(r.minimal) == 1 and r.minimal[0] in alts


def test_cap_hit_returns_passing_subset():
    need = set(IDS[::10])  # 10 specific ids
    assert len(need) == 10
    test = lambda s: need <= set(s)
    r = ddmin(IDS, test, max_calls=12)
    assert r.hit_cap and not r.one_minimal
    assert r.calls <= 12
    assert test(r.minimal)


def test_preconditions():
    with pytest.raises(ValueError):
        ddmin(IDS, lambda s: False)  # full set fails
    with pytest.raises(ValueError):
        ddmin(IDS, lambda s: True)  # empty set passes
    with pytest.raises(ValueError):
        ddmin([], lambda s: True)


def test_cache_and_order():
    seen: list[list[str]] = []
    count = 0

    def test(s):
        nonlocal count
        count += 1
        seen.append(list(s))
        return IDS[3] in s and IDS[60] in s

    r = ddmin(IDS, test)
    assert count == r.calls
    assert len({frozenset(s) for s in seen}) == len(seen)  # no repeated subset
    pos = {x: i for i, x in enumerate(IDS)}
    for s in seen:
        assert s == sorted(s, key=pos.__getitem__)


# ----- flip search with the oracle mock -----


def _sample(per_combo=30, seed=2000):
    cases: list[Case] = []
    for i, (length, dist, k) in enumerate(itertools.product((100, 400), DISTANCES, (2, 8))):
        cases += generate_cases(per_combo, seed + i, Knobs(length=length, distance=dist, distractors=k), mix=UNIFORM)
    return cases


@pytest.fixture(scope="module")
def sample():
    cases = _sample()
    assert len(cases) >= 300
    sizes = [len(c.events) for c in cases]
    assert min(sizes) < 150 and max(sizes) > 350  # lengths 100 and 400 both present
    return cases


def _search(case, view, **kw):
    return flip_search(case, event_view(case, view, "t"), oracle_decider, case.a_star, system="t", budget=None,
                       model="oracle", **kw)


def test_drop_dstar_restores_exactly(sample):
    checked = 0
    for c in sample:
        if c.mechanism not in ("M1", "M2", "M3"):
            continue
        gone = set(c.d_star)
        view = [e.id for e in c.events if e.id not in gone]
        r = _search(c, view)
        assert r.status == "restored", c.case_id
        assert r.event_ids == c.d_star, c.case_id
        assert r.one_minimal and not r.hit_cap
        checked += 1
    assert checked >= 150


def _split_dstar(case: Case) -> dict[str, list[str]]:
    """U2: group d_star ids by the mechanism they plant, via the event type."""
    kind = {"reading": "M1", "device": "M2", "adjustment": "M3"}
    by = {e.id: kind[e.type] for e in case.events if e.id in set(case.d_star)}
    out: dict[str, list[str]] = {}
    for i in case.d_star:
        out.setdefault(by[i], []).append(i)
    return out


def test_u2_drop_one_mechanism(sample):
    checked = 0
    for c in sample:
        if c.mechanism != "U2":
            continue
        groups = _split_dstar(c)
        assert len(groups) == 2
        for keep_m, drop_m in itertools.permutations(groups, 2):
            drop = set(groups[drop_m])
            view = [e.id for e in c.events if e.id not in drop]
            r = _search(c, view)
            assert detect_patterns([e for e in c.events if e.id in set(view)], c.query_period) == [keep_m]
            assert r.action_from_view not in (REQUEST_INFO, "EXPLAIN")
            assert r.status == "restored" and r.action_restored == REQUEST_INFO
            assert r.event_ids == groups[drop_m]
            checked += 1
    assert checked >= 50


def test_random_views_are_minimal(sample):
    restored = hit_cap = flipped = 0
    for c in sample:
        if c.cls == "none":
            continue
        rng = random.Random(f"view|{c.case_id}")
        view = [e.id for e in c.events if rng.random() < 0.3]
        r = _search(c, view)
        if r.status == "no_flip":
            continue
        flipped += 1
        if r.status != "restored":
            continue
        restored += 1
        hit_cap += r.hit_cap
        vset = set(view)
        full = lambda extra: oracle_decider(event_view(c, vset | set(extra), "t"), [], c) == c.a_star
        assert full(r.event_ids)
        if r.one_minimal:
            for x in r.event_ids:
                assert not full([y for y in r.event_ids if y != x]), (c.case_id, x)
    print(f"random views: flipped={flipped} restored={restored} cap_hits={hit_cap}")
    assert restored > 0


def test_never_target_is_not_recoverable(sample):
    c = next(c for c in sample if c.mechanism == "M2")
    r = flip_search(c, event_view(c, [], "t"),
                    lambda view, ins, case: "EXPLAIN" if case.a_star != "EXPLAIN" else "OPEN_REVIEW",
                    c.a_star, system="t", budget=None, model="never")
    assert r.status == "not_recoverable_by_insertion"
    assert r.calls_used == 2 and r.event_ids == []


def test_no_flip_is_one_call(sample):
    c = next(c for c in sample if c.mechanism == "M0")
    r = _search(c, [e.id for e in c.events])
    assert r.status == "no_flip" and r.calls_used == 1


def test_verify_and_call_budget(sample):
    c = next(c for c in sample if c.mechanism == "M1")
    view = [e.id for e in c.events if e.id not in set(c.d_star)]
    ok = _search(c, view, verify=oracle_decider)
    assert ok.verified is True
    bad = _search(c, view, verify=lambda view, ins, case: "EXPLAIN")
    assert bad.verified is False
    assert _search(c, view).verified is None
    capped = _search(c, view, max_calls=4)
    assert capped.calls_used <= 4


def test_record_round_trip(sample):
    c = next(c for c in sample if c.mechanism == "M1")
    view = [e.id for e in c.events if e.id not in set(c.d_star)]
    r = _search(c, view, verify=oracle_decider)
    assert len(r.event_texts) == len(r.event_ids) == 2
    assert all(t.startswith(i) for t, i in zip(r.event_texts, r.event_ids))
    assert DistinctionRecord.from_dict(r.to_dict()) == r

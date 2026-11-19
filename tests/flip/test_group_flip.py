"""Step 3A: group-first flip search against the oracle mock."""

from __future__ import annotations

import json

from lens.flip.cli import build_view
from lens.flip.decider import oracle_decider
from lens.flip.flip import flip_search, group_key
from lens.flip.generator import Knobs, generate_cases
from lens.flip.schema import DistinctionRecord
from lens.flip.systems.base import event_view


def _search(case, view, **kw):
    return flip_search(case, event_view(case, view, "t"), oracle_decider, case.a_star, system="t", budget=None,
                       model="oracle", **kw)


def _passes(case, view, ids):
    vset = set(view) | set(ids)
    return oracle_decider(event_view(case, vset, "t"), [], case) == case.a_star


def test_group_first_beats_plain_on_random_views():
    cases = generate_cases(300, 3, Knobs(length=400))
    assert len(cases) == 300
    stats = {}
    for strategy in ("plain", "group"):
        flipped = cap = restored = 0
        for c in cases:
            view = build_view(c, "random", 0.3)
            r = _search(c, view, strategy=strategy)
            if r.status == "no_flip":
                continue
            flipped += 1
            cap += r.hit_cap
            if r.status == "restored":
                restored += 1
                assert _passes(c, view, r.event_ids), (strategy, c.case_id)
                assert r.calls_used <= 48
        stats[strategy] = (flipped, cap, restored)
    print(f"cap-hit rate plain={stats['plain'][1] / stats['plain'][0]:.3f} group={stats['group'][1] / stats['group'][0]:.3f} "
          f"(flipped={stats['plain'][0]})")
    assert stats["plain"][0] == stats["group"][0] > 100
    assert stats["group"][1] / stats["group"][0] < stats["plain"][1] / stats["plain"][0]


def test_group_first_recovers_dstar_on_drop_dstar_views():
    mix = {"M0": 0.0, "M1": 1.0, "M2": 1.0, "M3": 1.0, "U1": 0.0, "U2": 0.0}
    cases = generate_cases(150, 11, Knobs(length=400), mix=mix)
    assert len(cases) == 150 and all(c.cls == "rare" for c in cases)
    for c in cases:
        view = build_view(c, "drop-dstar", 0.3)
        r = _search(c, view)
        assert r.status == "restored" and r.event_ids == c.d_star, c.case_id
        assert r.granularity == "event" and r.one_minimal and not r.hit_cap
        assert sum(r.phase_calls) == r.calls_used
        assert set(r.groups) == {group_key(e) for e in c.events if e.id in set(c.d_star)}


def test_event_granularity_results_are_one_minimal_and_valid():
    cases = generate_cases(120, 7, Knobs(length=100))
    checked = 0
    for c in cases:
        view = build_view(c, "random", 0.3)
        r = _search(c, view)
        if r.status != "restored":
            continue
        assert _passes(c, view, r.event_ids)
        if r.granularity == "event":
            for x in r.event_ids:
                assert not _passes(c, view, [y for y in r.event_ids if y != x]), (c.case_id, x)
            checked += 1
    assert checked > 20


def test_cap_keeps_groups_whole_and_result_passes():
    cases = generate_cases(100, 3, Knobs(length=400))
    seen_group_gran = 0
    for c in cases:
        view = build_view(c, "random", 0.3)
        r = _search(c, view, max_calls=10)
        assert r.calls_used <= 10
        if r.status == "restored":
            assert _passes(c, view, r.event_ids)
            if r.hit_cap:
                assert r.granularity == "group" and not r.one_minimal
                seen_group_gran += 1
    assert seen_group_gran > 0


def test_plain_strategy_still_works_and_old_record_loads():
    c = next(c for c in generate_cases(40, 2, Knobs(length=100)) if c.mechanism == "M1")
    view = [e.id for e in c.events if e.id not in set(c.d_star)]
    r = _search(c, view, strategy="plain")
    assert r.status == "restored" and r.event_ids == c.d_star
    assert r.phase_calls[0] == 2 and sum(r.phase_calls) == r.calls_used
    old = {k: v for k, v in r.to_dict().items() if k not in ("granularity", "groups", "phase_calls")}
    loaded = DistinctionRecord.from_dict(json.loads(json.dumps(old)))
    assert loaded.granularity == "event" and loaded.groups == [] and loaded.phase_calls == [0, 0]
    assert DistinctionRecord.from_dict(r.to_dict()) == r

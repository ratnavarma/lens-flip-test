"""Minimal flip search: the smallest set of missing events that restores an action."""

from __future__ import annotations

from typing import Iterable

from .ddmin import CapReached, ddmin
from .decider import Decider
from .render import render_events, render_lines  # noqa: F401  (render_lines re-exported)
from .schema import Case, DistinctionRecord, Event, STAGES, Unit, View

STRATEGIES = ("group", "plain")
_UNBOUNDED = 10**9  # budget is enforced by decide_on, not by ddmin's own counter


def group_key(event: Event) -> str:
    """Group name: "type/subtype", or just "type" when there is no subtype."""
    return f"{event.type}/{event.subtype}" if event.subtype else event.type


def event_units(case: Case, events: Iterable[Event]) -> list[Unit]:
    """Units for events: id, rendered line, "type/subtype" group, in the order given."""
    events = list(events)
    return [Unit(id=e.id, text=t, group=group_key(e), event=e) for e, t in zip(events, render_events(case, events))]


def missing_units(case: Case, view: View) -> list[Unit]:
    """The case events not quoted verbatim in the view, in history order (all of them for a summary)."""
    quoted = set(view.event_ids)
    return event_units(case, (e for e in case.events if e.id not in quoted))


def flip_search(
    case: Case,
    view: View,
    decide: Decider,
    target_action: str,
    *,
    system: str,
    budget: int | None,
    model: str,
    max_calls: int = 48,
    verify: Decider | None = None,
    strategy: str = "group",
    candidates: list[Unit] | None = None,
    stage: str = "flip",
) -> DistinctionRecord:
    """Search for the smallest set of units outside the view whose insertion restores target_action.

    candidates: the units that may be inserted, in insertion order. The default is the
    case events not in view.event_ids, in history order (every event for a summary).
    Inserted units go to the decider in an "Additional records:" block after the view text.
    stage labels the record ("flip", or "residue" when the view is a handoff note).

    strategy "group" (default) is hierarchical delta debugging (Misherghi and Su, 2006):

    1. group the missing events by (type, subtype), in order of first occurrence;
    2. phase 1: ddmin over groups, each group inserted whole;
    3. phase 2: for each selected group in turn, ddmin over its events while the
       other selected groups stay at their current sets.

    One budget of max_calls decider calls covers steps 1 and 2 of the search and
    both phases. If phase 2 runs out of budget on a group, that group stays whole.
    Strategy "plain" runs ddmin over all missing events at once.

    calls_used counts every call to `decide`, including the two opening calls
    (view alone, view plus every missing event). Calls to `verify` are not counted.
    """
    if strategy not in STRATEGIES:
        raise ValueError(f"strategy must be one of {STRATEGIES}")
    if stage not in STAGES:
        raise ValueError(f"stage must be one of {STAGES}")
    units = missing_units(case, view) if candidates is None else list(candidates)
    if len({u.id for u in units}) != len(units):
        raise ValueError("candidate unit ids must be unique")
    order = {u.id: i for i, u in enumerate(units)}
    by_id = {u.id: u for u in units}
    missing = [u.id for u in units]
    key_of = {u.id: u.group for u in units}

    calls = 0
    memo: dict[frozenset[str], str] = {}
    enforce_cap = False

    def units_for(ids: Iterable[str]) -> list[Unit]:
        wanted = set(ids)
        return [u for u in units if u.id in wanted]

    def decide_on(extra: Iterable[str]) -> str:
        nonlocal calls
        extra_set = frozenset(extra)
        if extra_set not in memo:
            if enforce_cap and calls >= max_calls:
                raise CapReached
            calls += 1
            memo[extra_set] = decide(view, units_for(extra_set), case)
        return memo[extra_set]

    def record(status: str, ids: list[str], restored: str | None, **kw) -> DistinctionRecord:
        phase1 = kw.get("phase1_calls", calls)
        seen: list[str] = []
        for i in ids:
            if key_of[i] not in seen:
                seen.append(key_of[i])
        return DistinctionRecord(
            case_id=case.case_id,
            system=system,
            budget=budget,
            model=model,
            status=status,
            event_ids=ids,
            event_texts=[by_id[i].text for i in ids],
            action_from_view=action_from_view,
            action_restored=restored,
            cls=case.cls,
            calls_used=calls,
            hit_cap=kw.get("hit_cap", False),
            one_minimal=kw.get("one_minimal", False),
            verified=kw.get("verified"),
            granularity=kw.get("granularity", "event"),
            groups=kw.get("groups", seen if status == "restored" else []),
            phase_calls=[phase1, calls - phase1],
            stage=stage,
            units_inserted=[by_id[i].text for i in ids],
        )

    action_from_view = decide_on([])
    if action_from_view == target_action:
        return record("no_flip", [], action_from_view)

    full_action = decide_on(missing)
    if full_action != target_action:
        return record("not_recoverable_by_insertion", [], full_action)

    def finish(ids: list[str], *, hit_cap: bool, one_minimal: bool, granularity: str, phase1_calls: int,
               groups: list[str] | None = None) -> DistinctionRecord:
        verified = None
        if verify is not None:
            verified = verify(view, units_for(ids), case) == target_action
        extra = {} if groups is None else {"groups": groups}
        return record(
            "restored", ids, target_action, hit_cap=hit_cap, one_minimal=one_minimal, verified=verified,
            granularity=granularity, phase1_calls=phase1_calls, **extra,
        )

    if strategy == "plain":
        enforce_cap = False
        res = ddmin(missing, lambda s: decide_on(s) == target_action, max_calls=max_calls)
        return finish(
            res.minimal, hit_cap=res.hit_cap, one_minimal=res.one_minimal,
            granularity="event" if res.one_minimal else "group", phase1_calls=2,
        )

    # ----- group-first -----
    enforce_cap = True
    members: dict[str, list[str]] = {}
    for i in missing:
        members.setdefault(key_of[i], []).append(i)
    keys = list(members)  # order of first occurrence in the history

    def ids_of(selected: Iterable[str]) -> set[str]:
        return {i for k in selected for i in members[k]}

    def in_history_order(ids: Iterable[str]) -> list[str]:
        return sorted(ids, key=order.__getitem__)

    # Phase 1: both ends are known from the two opening calls.
    r1 = ddmin(keys, lambda ks: decide_on(ids_of(ks)) == target_action,
               max_calls=_UNBOUNDED, assume_preconditions=True)
    selected = list(r1.minimal)
    phase1_calls = calls
    if r1.hit_cap:
        return finish(in_history_order(ids_of(selected)), hit_cap=True, one_minimal=False,
                      granularity="group", phase1_calls=phase1_calls, groups=selected)

    # Phase 2: reduce inside each selected group, the others held at their current sets.
    state = {k: list(members[k]) for k in selected}
    hit_cap = False
    for g in selected:
        if len(state[g]) < 2:
            continue
        others = {i for k in selected if k != g for i in state[k]}
        r2 = ddmin(state[g], lambda sub: decide_on(others | set(sub)) == target_action,
                   max_calls=_UNBOUNDED, assume_preconditions=True)
        if r2.hit_cap:  # keep this group whole and stop
            hit_cap = True
            break
        state[g] = r2.minimal
    ids = in_history_order(i for k in selected for i in state[k])
    return finish(ids, hit_cap=hit_cap, one_minimal=not hit_cap,
                  granularity="group" if hit_cap else "event", phase1_calls=phase1_calls, groups=selected)

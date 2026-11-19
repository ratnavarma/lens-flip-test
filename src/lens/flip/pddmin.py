"""Round-parallel delta debugging and the group-first flip search built on it.

The sequential ddmin in ddmin.py asks one subset at a time. A Batch round costs minutes, so
here one granularity level asks every candidate subset and complement at once (one round),
then reduces exactly as sequential ddmin would: the first passing subset, else the first
passing complement, else a finer split. On a deterministic test and without the cap the
result is the same set sequential ddmin returns. It spends more calls per level, because
it asks the complements even when a subset passes.

Cap. One budget of max_calls covers the whole search, as in flip.flip_search. A level that
needs more calls than are left asks only a prefix (subsets first, then complements) and the
search stops there with hit_cap. The returned set always passed the test.

Replay. The search is a pure function of its answers. `ask_many(sets)` returns one action
per set of inserted unit ids, or raises Pending after it has registered the sets it lacks.
The headline driver reruns the search from the start each round with the answers so far,
so its state is only the answer table.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable, Sequence

from .ddmin import _split
from .decider import Decider
from .schema import Case, DistinctionRecord, STAGES, Unit, View


class Pending(Exception):
    """Some asked sets have no answer yet. They have been registered for the next round."""


AskMany = Callable[[list[frozenset]], list[str]]


class Searcher:
    """Memo and shared call budget over sets of inserted unit ids."""

    def __init__(self, ask_many: AskMany, target: str, max_calls: int):
        self.ask_many = ask_many
        self.target = target
        self.max_calls = max_calls
        self.memo: dict[frozenset, str] = {}
        self.calls = 0
        self.rounds = 0

    def actions(self, sets: Sequence[frozenset], *, capped: bool = True) -> tuple[list[str | None], bool]:
        """Actions for the sets, in order. None where the cap left a set unasked."""
        new = [s for s in dict.fromkeys(sets) if s not in self.memo]
        room = self.max_calls - self.calls if capped else len(new)
        short = len(new) > room
        asked = new[: max(room, 0)]
        if asked:
            got = self.ask_many(asked)  # may raise Pending
            for s, a in zip(asked, got):
                self.memo[s] = a
            self.calls += len(asked)
            self.rounds += 1
        return [self.memo.get(s) for s in sets], short

    def passes(self, sets: Sequence[frozenset]) -> tuple[list[bool | None], bool]:
        acts, short = self.actions(sets)
        return [None if a is None else a == self.target for a in acts], short


@dataclass
class PddminResult:
    minimal: list[str]
    hit_cap: bool
    one_minimal: bool


def pddmin(candidates: Sequence[str], to_set: Callable[[list[str]], frozenset], s: Searcher) -> PddminResult:
    """Round-parallel ddmin. Preconditions (all passes, none fails) are the caller's, as with
    ddmin(assume_preconditions=True)."""
    current = list(candidates)
    if not current:
        raise ValueError("pddmin needs at least one candidate")
    best = list(current)
    n = 2
    while len(current) >= 2:
        chunks = _split(current, n)
        comps = [[x for j, c in enumerate(chunks) if j != i for x in c] for i in range(len(chunks))]
        tests = chunks + comps
        res, short = s.passes([to_set(t) for t in tests])
        for t, ok in zip(tests, res):
            if ok and len(t) < len(best):
                best = list(t)
        i = next((k for k in range(len(chunks)) if res[k]), None)
        if i is not None:
            current, n = chunks[i], 2
            continue
        j = next((k for k in range(len(comps)) if res[len(chunks) + k]), None)
        if j is not None:
            current, n = comps[j], max(n - 1, 2)
            continue
        if short:
            return PddminResult(best, hit_cap=True, one_minimal=False)
        if n >= len(current):
            break
        n = min(len(current), 2 * n)
    return PddminResult(current, hit_cap=False, one_minimal=True)


def group_search(units: Sequence[Unit], target: str, ask_many: AskMany, max_calls: int = 48) -> dict:
    """Group-first flip search (flip.flip_search, strategy "group") on round-parallel ddmin.

    Returns a dict with status, ids (history order of `units`), hit_cap, one_minimal,
    granularity, groups, phase_calls, calls_used, action_from_view, action_restored and rounds.
    calls_used counts the two opening calls (view alone, view plus every unit), as before.
    """
    order = {u.id: i for i, u in enumerate(units)}
    if len(order) != len(units):
        raise ValueError("candidate unit ids must be unique")
    key_of = {u.id: u.group for u in units}
    every = frozenset(order)
    s = Searcher(ask_many, target, max_calls)

    def out(status: str, ids: Iterable[str], restored: str | None, *, hit_cap=False, one_minimal=False,
            granularity="event", groups=None, phase1=None) -> dict:
        ids = sorted(ids, key=order.__getitem__)
        if groups is None:
            groups = list(dict.fromkeys(key_of[i] for i in ids)) if status == "restored" else []
        p1 = s.calls if phase1 is None else phase1
        return {"status": status, "ids": ids, "action_from_view": a0, "action_restored": restored,
                "hit_cap": hit_cap, "one_minimal": one_minimal, "granularity": granularity, "groups": groups,
                "calls_used": s.calls, "phase_calls": [p1, s.calls - p1], "rounds": s.rounds}

    (a0, afull), _ = s.actions([frozenset(), every])
    if a0 == target:
        return out("no_flip", [], a0)
    if afull != target:
        return out("not_recoverable_by_insertion", [], afull)

    members: dict[str, list[str]] = {}
    for u in units:
        members.setdefault(u.group, []).append(u.id)
    keys = list(members)

    def ids_of(ks: Iterable[str]) -> frozenset:
        return frozenset(i for k in ks for i in members[k])

    r1 = pddmin(keys, ids_of, s) if len(keys) >= 2 else PddminResult(keys, False, True)
    selected = list(r1.minimal)
    phase1 = s.calls
    if r1.hit_cap:
        return out("restored", ids_of(selected), target, hit_cap=True, granularity="group", groups=selected,
                   phase1=phase1)

    state = {k: list(members[k]) for k in selected}
    hit_cap = False
    for g in selected:
        if len(state[g]) < 2:
            continue
        others = frozenset(i for k in selected if k != g for i in state[k])
        r2 = pddmin(state[g], lambda sub, o=others: o | frozenset(sub), s)
        if r2.hit_cap:  # keep this group whole and stop
            hit_cap = True
            break
        state[g] = r2.minimal
    ids = [i for k in selected for i in state[k]]
    return out("restored", ids, target, hit_cap=hit_cap, one_minimal=not hit_cap,
               granularity="group" if hit_cap else "event", groups=selected, phase1=phase1)


def to_record(res: dict, case: Case, units: Sequence[Unit], *, system: str, budget: int | None, model: str,
              verified: bool | None, stage: str = "flip") -> DistinctionRecord:
    if stage not in STAGES:
        raise ValueError(f"stage must be one of {STAGES}")
    by_id = {u.id: u for u in units}
    return DistinctionRecord(
        case_id=case.case_id, system=system, budget=budget, model=model, status=res["status"],
        event_ids=list(res["ids"]), event_texts=[by_id[i].text for i in res["ids"]],
        action_from_view=res["action_from_view"], action_restored=res["action_restored"], cls=case.cls,
        calls_used=res["calls_used"], hit_cap=res["hit_cap"], one_minimal=res["one_minimal"], verified=verified,
        granularity=res["granularity"], groups=list(res["groups"]), phase_calls=list(res["phase_calls"]),
        stage=stage, units_inserted=[by_id[i].text for i in res["ids"]],
    )


def round_flip_search(case: Case, view: View, decide: Decider, target: str, *, system: str, budget: int | None,
                      model: str, max_calls: int = 48, verify: Decider | None = None,
                      candidates: list[Unit] | None = None, stage: str = "flip") -> DistinctionRecord:
    """Synchronous wrapper: the round-parallel search with a plain decider. For tests and the mock."""
    from .flip import missing_units

    units = missing_units(case, view) if candidates is None else list(candidates)
    by_id = {u.id: u for u in units}

    def ask_many(sets: list[frozenset]) -> list[str]:
        return [decide(view, [u for u in units if u.id in s], case) for s in sets]

    res = group_search(units, target, ask_many, max_calls)
    verified = None
    if verify is not None and res["status"] == "restored":
        verified = verify(view, [by_id[i] for i in res["ids"]], case) == target
    return to_record(res, case, units, system=system, budget=budget, model=model, verified=verified, stage=stage)

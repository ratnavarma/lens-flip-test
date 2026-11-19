"""Delta debugging (Zeller and Hildebrandt, 2002) over an ordered list of event ids.

`test(subset)` is True when inserting that subset restores the target action.
Preconditions: test(all candidates) is True and test([]) is False.
Both calls count toward the budget. Results are cached by frozenset, so a
repeated subset costs no call. Every subset handed to `test` keeps the order
of `candidates`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Sequence


@dataclass
class DdminResult:
    minimal: list[str]  # ids in original order
    calls: int  # real calls to `test`
    hit_cap: bool
    one_minimal: bool  # True only if the search finished without hitting the cap


class CapReached(Exception):
    """Raised by a `test` function (or inside ddmin) when the call budget is spent."""


_CapReached = CapReached


def _split(items: list[str], n: int) -> list[list[str]]:
    """Split into n non-empty chunks of near-equal size, keeping order."""
    size, extra = divmod(len(items), n)
    chunks, start = [], 0
    for i in range(n):
        end = start + size + (1 if i < extra else 0)
        chunks.append(items[start:end])
        start = end
    return chunks


def ddmin(
    candidates: Sequence[str],
    test: Callable[[list[str]], bool],
    max_calls: int = 48,
    *,
    assume_preconditions: bool = False,
) -> DdminResult:
    """Minimise `candidates`.

    A `test` that raises CapReached stops the search like the internal cap does.
    With assume_preconditions the caller guarantees test(candidates) is True and
    test([]) is False, and ddmin does not spend calls on them.
    """
    candidates = list(candidates)
    if not candidates:
        raise ValueError("ddmin needs at least one candidate")
    if max_calls < 2:
        raise ValueError("max_calls must be at least 2 (the two precondition calls)")

    cache: dict[frozenset[str], bool] = {}
    calls = 0
    best = list(candidates)  # smallest passing subset seen so far

    def check(subset: list[str]) -> bool:
        nonlocal calls, best
        key = frozenset(subset)
        if key in cache:
            return cache[key]
        if calls >= max_calls:
            raise _CapReached
        calls += 1
        result = bool(test(list(subset)))
        cache[key] = result
        if result and len(subset) < len(best):
            best = list(subset)
        return result

    current = list(candidates)
    n = 2
    hit_cap = False
    try:
        if assume_preconditions:
            cache[frozenset(candidates)] = True
            cache[frozenset()] = False
        else:
            if not check(candidates):
                raise ValueError("precondition failed: test(candidates) is False")
            if check([]):
                raise ValueError("precondition failed: test([]) is True")
        while len(current) >= 2:
            chunks = _split(current, n)
            reduced = False
            for chunk in chunks:  # reduce to subset
                if check(chunk):
                    current, n, reduced = chunk, 2, True
                    break
            if not reduced:
                for i in range(len(chunks)):  # reduce to complement
                    comp = [x for j, c in enumerate(chunks) if j != i for x in c]
                    if check(comp):
                        current, n, reduced = comp, max(n - 1, 2), True
                        break
            if not reduced:
                if n >= len(current):
                    break
                n = min(len(current), 2 * n)
    except _CapReached:
        hit_cap = True

    minimal = best if hit_cap else current
    return DdminResult(minimal=list(minimal), calls=calls, hit_cap=hit_cap, one_minimal=not hit_cap)

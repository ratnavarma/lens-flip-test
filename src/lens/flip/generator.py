"""Case generator and rule oracle for the Decision Flip Test.

Randomness comes from three separate streams, all `random.Random` with string
seeds, so results do not depend on the clock or on hash randomisation:

- plant:  the mechanism, the query period and every planted or decoy event
- fill:   calendar, values and the ordinary events (contacts, plan notices ...)
- dist:   the distractors only (seed can be replaced to re-draw them)

Count balance. A rare mechanism is a common event type in an uncommon place.
So that event-type counts never reveal the mechanism, each case carries three
"slots", one per pattern (M1, M2, M3). A slot holds either the planted
pattern or a decoy with the same (type, subtype) count profile:

- M1 slot: planted = estimated reading for P, well above the month's norm and
  never superseded in time, plus a later actual reading in line with the norm.
  Decoy = one extra estimated reading in another period, superseded by that
  period's actual reading before its statement.
- M2 slot: planted = device replaced inside P with no reset reading, so P's
  figure carries over the old device's figure and is well above the norm.
  Decoy = a device replacement in another period followed by a reset reading
  and a normal figure.
- M3 slot: planted = a one-off charge on P's statement plus a credit approved
  90+ days earlier for exactly that charge, never applied (credits line 0).
  Plus a matched pair elsewhere: a one-off charge, its credit and a
  credit_applied event, with the credits line netting the charge to zero.
  Decoy = the matched pair plus one plain credit approved after P's statement
  that nothing has applied yet (the oracle ignores it by age).

Seasonality is real: every high-demand month, in every year, is raised by one
account-wide factor (1.6 to 1.8, small noise). So M0's P matches the same
month in earlier years. Each rare mechanism raises P's amount in a way the
records show. Statements carry a line-item breakdown (activity_charge,
other_charges, credits) that adds up to amount.

U1 plants one slot, deletes its D* events, then re-adds benign events of the
deleted (type, subtype) pairs elsewhere. U2 plants two slots.

Generator version (protocol v3). Version 3 draws the U2 pair only from M1+M3 and M2+M3, so one cause
sits on the activity charge (M1 or M2) and the other on the other charges (M3). M1+M2 puts both causes on
the same reading of P and its later actual reading points to the estimate alone, so its ground truth is
contested. Version 2 drew any pair of the three slots and stays available (`generator_version=2`) to
regenerate the headline-300 cases byte for byte. Only U2 differs between the versions. The plant stream
is seeded per mechanism, so the U2 draw changes nothing for M0 to M3 and U1.

Clean window. P, the calendar month before P and the month after P hold no device
event and no plan notice, except the device replacement a mechanism plants in P (M2, and U2 with M2).
A plan notice counts as inside the window when its date or its effective date is.
Background, decoy and distractor device and plan events are moved out of the window
to the next allowed month. They are never dropped, so (type, subtype) counts match
across mechanisms.

Length is approximate: the history has about `length` events.
"""

from __future__ import annotations

import datetime as _dt
import random
from collections import Counter
from dataclasses import asdict, dataclass
from typing import Any, Sequence

from .schema import (
    ACTIONS,
    APPLY_CREDIT,
    CHECK_DEVICE,
    EXPLAIN,
    MECHANISMS,
    OPEN_REVIEW,
    REQUEST_INFO,
    Case,
    Event,
)

# Jan, Feb, Jul, Aug. Under half of all months, so the account median is a low-demand level.
HIGH_MONTHS = frozenset({1, 2, 7, 8})
BASE_VALUE = 400.0
SEASON_LO, SEASON_HI = 1.6, 1.8  # one factor per account, applied to every high-demand month
NOISE = 0.03  # per-month noise, plus or minus
M1_EXCESS = (0.6, 0.9)  # estimate = norm * (1 + excess)
M2_EXCESS = (0.8, 1.2)  # carried-over figure of the old device, as a share of the norm
M3_CHARGE = (0.55, 0.9)  # one-off charge as a share of the month's activity charge
MIN_YEARS_BEFORE_P = 12  # P has the same calendar month in an earlier year
RATE = 0.2  # activity charge = reading value * RATE
CREDIT_MIN_AGE_DAYS = 90
SLOTS = ("M1", "M2", "M3")
GENERATOR_VERSION = 3
U2_PAIRS = {"M1+M3": ("M1", "M3"), "M2+M3": ("M2", "M3")}  # version 3; version 2 sampled any two slots
ACTION_FOR = {"M1": OPEN_REVIEW, "M2": CHECK_DEVICE, "M3": APPLY_CREDIT}
CLS_FOR = {"M0": "none", "M1": "rare", "M2": "rare", "M3": "rare", "U1": "unobserved", "U2": "unresolved"}
DISTANCES = ("near", "middle", "far")

TOPICS = ("question about statement", "address update", "payment arrangement", "general complaint")
CHANNELS = ("phone", "email", "chat", "mail")
PLANS = ("Standard", "Flex", "Basic", "Plus")
TYPE_RANK = {
    "device": 0,
    "reading": 1,
    "correction": 2,
    "statement": 3,
    "adjustment": 4,
    "payment": 5,
    "plan_notice": 6,
    "contact": 7,
}


@dataclass
class Knobs:
    length: int = 400
    distance: str = "middle"
    distractors: int = 2
    rare_share: float = 0.5
    mix: dict[str, float] | None = None

    def resolved_mix(self) -> dict[str, float]:
        return dict(self.mix) if self.mix else default_mix(self.rare_share)

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["mix"] = self.resolved_mix()
        return d


def default_mix(rare_share: float = 0.5) -> dict[str, float]:
    """M0 gets 1 - rare_share. The five rare mechanisms split the rest evenly."""
    rare = rare_share / 5.0
    return {"M0": 1.0 - rare_share, "M1": rare, "M2": rare, "M3": rare, "U1": rare, "U2": rare}


# ---------------------------------------------------------------------------
# Rule oracle (works from events and the query period only)
# ---------------------------------------------------------------------------


def is_high_demand(period: str) -> bool:
    return int(period[5:7]) in HIGH_MONTHS


def _next_period(period: str) -> str:
    y, m = int(period[:4]), int(period[5:7])
    y, m = (y + 1, 1) if m == 12 else (y, m + 1)
    return f"{y:04d}-{m:02d}"


def detect_patterns(events: Sequence[Event], query_period: str) -> list[str]:
    """Return the patterns (subset of M1, M2, M3) that hold for the query period."""
    stmt_dates = [e.date for e in events if e.type == "statement" and e.fields.get("period") == query_period]
    stmt_date = min(stmt_dates) if stmt_dates else _next_period(query_period) + "-04"
    stmt_day = _dt.date.fromisoformat(stmt_date)
    corrected = {e.fields.get("refers_to") for e in events if e.type == "correction"}
    applied = {e.fields.get("credit_id") for e in events if e.type == "adjustment" and e.subtype == "credit_applied"}
    p_readings = [e for e in events if e.type == "reading" and e.fields.get("period") == query_period]
    found: list[str] = []

    # M1: an estimate for P that nothing superseded in time, and a later lower actual reading.
    actual_before = any(r.subtype == "actual" and r.date <= stmt_date for r in p_readings)
    if not actual_before:
        for est in (r for r in p_readings if r.subtype == "estimated"):
            if est.id in corrected:
                continue
            if any(r.subtype == "actual" and r.date > stmt_date and r.fields["value"] < est.fields["value"] for r in p_readings):
                found.append("M1")
                break

    # M2: a device replaced inside P with no reset reading after it inside P.
    for dev in (e for e in events if e.type == "device" and e.subtype == "replaced" and e.date[:7] == query_period):
        if not any(
            r.type == "reading" and r.fields.get("reset") and r.date[:7] == query_period and r.date >= dev.date for r in events
        ):
            found.append("M2")
            break

    # M3: a credit approved 90+ days before P's statement and never applied.
    for adj in (e for e in events if e.type == "adjustment" and e.subtype == "credit_approved"):
        if adj.fields.get("credit_id") in applied:
            continue
        if (stmt_day - _dt.date.fromisoformat(adj.date)).days >= CREDIT_MIN_AGE_DAYS:
            found.append("M3")
            break
    return found


def oracle(events: Sequence[Event], query_period: str) -> str:
    patterns = detect_patterns(events, query_period)
    if len(patterns) >= 2:
        return REQUEST_INFO
    if len(patterns) == 1:
        return ACTION_FOR[patterns[0]]
    return EXPLAIN if is_high_demand(query_period) else REQUEST_INFO


# ---------------------------------------------------------------------------
# Builder
# ---------------------------------------------------------------------------


class _Build:
    def __init__(self, n_months: int, start_abs: int) -> None:
        self.N = n_months
        self.start = start_abs
        self.events: list[Event] = []
        self.primary: dict[int, Event] = {}  # month index -> its monthly reading
        self.used: set[int] = set()  # periods already holding a planted, decoy or distractor edit
        self.P = -1
        self.deleted: list[Event] = []  # U1: events removed from the history
        self.distance = "middle"
        self.val: dict[int, float] = {}  # month index -> activity figure the statement is computed from
        self.norm: dict[int, float] = {}  # month index -> the seasonal norm for that month
        self.stmt: dict[int, Event] = {}
        self.pay: dict[int, Event] = {}
        # month index -> {"ref": str | None, "amount": one-off charge, "credit": credit line}
        self.charge: dict[int, dict[str, Any]] = {}
        self._n = 0
        self._c = 0

    def ym(self, k: int) -> tuple[int, int]:
        a = self.start + k
        return a // 12, a % 12 + 1

    def period(self, k: int) -> str:
        y, m = self.ym(k)
        return f"{y:04d}-{m:02d}"

    def high(self, k: int) -> bool:
        return self.ym(k)[1] in HIGH_MONTHS

    def date(self, k: int, day: int) -> str:
        y, m = self.ym(k)
        return f"{y:04d}-{m:02d}-{day:02d}"

    def add(self, k: int, day: int, type_: str, subtype: str = "", **fields: Any) -> Event:
        self._n += 1
        ev = Event(id=f"t{self._n}", date=self.date(k, day), type=type_, subtype=subtype, fields=dict(fields))
        self.events.append(ev)
        return ev

    def new_credit(self) -> str:
        self._c += 1
        return f"~{self._c}"

    def free_period(self, rng: random.Random, clean: bool = False) -> int | None:
        """A period with no planted, decoy or distractor edit. clean=True also skips the months either side of P."""
        cands = [
            k for k in range(self.N)
            if k != self.P and k not in self.used and not (clean and k in (self.P - 1, self.P + 1))
        ]
        if not cands:
            return None
        k = rng.choice(cands)
        self.used.add(k)
        return k

    def value(self, k: int) -> float:
        return float(self.primary[k].fields["value"])

    def outside_window(self, k: int, plan: bool = False) -> int:
        """Nearest month at or after k (wrapping) that keeps a device or plan event out of the clean window.

        The window is the month before P, P and the month after P. A plan notice dated in month k takes
        effect on the first of month k+1, so for plan notices the month two before P is out as well,
        and the month after P is out whether the notice is dated or effective in it. No RNG.
        """
        banned = {self.P - 2, self.P - 1, self.P, self.P + 1} if plan else {self.P - 1, self.P, self.P + 1}
        for _ in range(self.N):
            if k not in banned:
                return k
            k = (k + 1) % self.N
        raise RuntimeError("no month outside the clean window")


def _choose_period(prng: random.Random, b: _Build, distance: str, want_high: bool) -> int:
    N = b.N
    lo_all, hi_all = MIN_YEARS_BEFORE_P, N - 3
    if distance == "near":
        lo, hi = N - 5, N - 3
    elif distance == "middle":
        lo, hi = N // 2 - 1, N // 2 + 1
    elif distance == "far":
        lo, hi = MIN_YEARS_BEFORE_P, max(MIN_YEARS_BEFORE_P, N // 4)
    else:
        raise ValueError(f"distance must be one of {DISTANCES}, got {distance!r}")
    for widen in range(N):
        cands = [k for k in range(max(lo_all, lo - widen), min(hi_all, hi + widen) + 1) if b.high(k) == want_high]
        if cands:
            return prng.choice(cands)
    raise RuntimeError("no query period available")


# ----- planted patterns -----


def _raise(b: _Build, k: int, excess: float) -> float:
    """Raise month k's figure by excess * norm and return the new value."""
    b.val[k] = round(b.val[k] + b.norm[k] * excess, 1)
    return b.val[k]


def _plant_m1(b: _Build, rng: random.Random) -> list[str]:
    P = b.P
    prim = b.primary[P]
    prim.subtype = "estimated"
    prim.fields["value"] = _raise(b, P, rng.uniform(*M1_EXCESS))
    late = b.add(
        P + 1, rng.randrange(10, 28), "reading", "actual",
        period=b.period(P), value=round(b.norm[P] * rng.uniform(0.95, 1.05), 1),
    )
    b.used.add(P)
    return [prim.id, late.id]


def _plant_m2(b: _Build, rng: random.Random) -> list[str]:
    # The primary reading of P has no reset: it carries over the old device's figure.
    b.primary[b.P].fields["value"] = _raise(b, b.P, rng.uniform(*M2_EXCESS))
    dev = b.add(b.P, rng.randrange(3, 10), "device", "replaced")
    b.used.add(b.P)
    return [dev.id]


def _approval_month(b: _Build, rng: random.Random) -> int:
    """Month of the approval, 3 or more months before P, placed by the distance knob."""
    P = b.P
    hi_cap = P - 3
    if b.distance == "near":
        lo, hi = max(0, P - 6), hi_cap
    elif b.distance == "middle":
        lo, hi = max(0, P // 2 - 1), min(hi_cap, P // 2 + 1)
    else:
        lo, hi = 0, min(hi_cap, max(0, P // 4))
    return rng.randint(lo, max(lo, hi))


def _matched_pair(b: _Build, rng: random.Random, q: int) -> None:
    """A one-off charge on period q's statement, its credit and the event that applies it.

    The credit is approved in month q-1 or q. A credit_applied event follows on the first
    of month q+1, before the statement for q is issued, and the credits line nets the charge.
    """
    x = round(b.val[q] * RATE * rng.uniform(0.3, 0.8), 2)
    ref = f"CH-{b.period(q)}"
    cid = b.new_credit()
    b.add(max(0, q - rng.randrange(0, 2)), rng.randrange(1, 29), "adjustment", "credit_approved",
          credit_id=cid, amount=x, charge_ref=ref, period=b.period(q))
    b.add(q + 1, 1, "adjustment", "credit_applied", credit_id=cid)
    b.charge[q] = {"ref": ref, "amount": x, "credit": x}


def _shifted_free(b: _Build, q: int) -> int:
    """Nearest period at or after q (wrapping) that is not P and has no one-off charge. No RNG."""
    for _ in range(b.N):
        if q != b.P and q not in b.charge:
            return q
        q = (q + 1) % b.N
    raise RuntimeError("no period free for a one-off charge")


def _plant_m3(b: _Build, rng: random.Random) -> list[str]:
    P = b.P
    x = round(b.norm[P] * RATE * rng.uniform(*M3_CHARGE), 2)
    ref = f"CH-{b.period(P)}"
    b.charge[P] = {"ref": ref, "amount": x, "credit": 0.0}
    adj = b.add(
        _approval_month(b, rng), rng.randrange(1, 29), "adjustment", "credit_approved",
        credit_id=b.new_credit(), amount=x, charge_ref=ref, period=b.period(P),
    )
    q = b.free_period(rng)
    if q is not None:
        _matched_pair(b, rng, q)
    return [adj.id]


# ----- decoys (same count profile, benign) and benign re-adds -----


def _decoy_m1(b: _Build, rng: random.Random) -> None:
    x = b.free_period(rng)
    if x is not None:
        b.add(x, rng.randrange(5, 12), "reading", "estimated",
              period=b.period(x), value=round(b.value(x) * rng.uniform(0.97, 1.03), 1))


def _decoy_m2(b: _Build, rng: random.Random) -> None:
    x = b.free_period(rng, clean=True)
    if x is not None:
        b.add(x, rng.randrange(3, 10), "device", "replaced")
        b.primary[x].fields["reset"] = 1


def _plain_credit(b: _Build, rng: random.Random) -> None:
    """A credit approved after P's statement. Nothing has applied it yet, and it cannot concern P."""
    b.add(rng.randrange(b.P + 2, b.N), rng.randrange(1, 29), "adjustment", "credit_approved",
          credit_id=b.new_credit(), amount=round(rng.uniform(10, 60), 2))


def _decoy_m3(b: _Build, rng: random.Random) -> None:
    x = b.free_period(rng)
    if x is not None:
        _matched_pair(b, rng, x)
    _plain_credit(b, rng)


def _extra_actual(b: _Build, rng: random.Random) -> None:
    x = b.free_period(rng)
    if x is not None:
        b.add(x, rng.randrange(25, 29), "reading", "actual",
              period=b.period(x), value=round(b.value(x) * rng.uniform(0.97, 1.03), 1))


_PLANTERS = {"M1": _plant_m1, "M2": _plant_m2, "M3": _plant_m3}
_DECOYS = {"M1": _decoy_m1, "M2": _decoy_m2, "M3": _decoy_m3}
_READD = {
    ("reading", "actual"): _extra_actual,
    ("reading", "estimated"): _decoy_m1,
    ("device", "replaced"): _decoy_m2,
    ("adjustment", "credit_approved"): _plain_credit,
}


def _profile(events: Sequence[Event]) -> Counter:
    return Counter((e.type, e.subtype) for e in events)


# ----- ordinary events -----


def _add_filler(b: _Build, rng: random.Random, count: int) -> None:
    kinds = ("contact", "plan", "serviced", "installed", "credit")
    weights = (0.38, 0.10, 0.17, 0.05, 0.30)
    made = 0
    while made < count:
        kind = rng.choices(kinds, weights)[0]
        if kind == "contact":
            b.add(rng.randrange(b.N), rng.randrange(1, 29), "contact",
                  topic=rng.choice(TOPICS), channel=rng.choice(CHANNELS))
            made += 1
        elif kind == "plan":
            k = b.outside_window(rng.randrange(b.N), plan=True)
            y, m = b.ym(k + 1)
            b.add(k, rng.randrange(1, 29), "plan_notice",
                  new_plan=rng.choice(PLANS), effective=f"{y:04d}-{m:02d}-01")
            made += 1
        elif kind in ("serviced", "installed"):
            b.add(b.outside_window(rng.randrange(b.N)), rng.randrange(1, 29), "device", kind)
            made += 1
        else:
            _matched_pair(b, rng, _shifted_free(b, rng.randrange(b.N)))
            made += 2


# ----- distractors -----


def _add_distractors(b: _Build, rng: random.Random, count: int) -> None:
    kinds = ("est_corrected", "replaced_reset", "credit_applied", "complaint")
    for _ in range(count):
        kind = rng.choice(kinds)
        if kind in ("est_corrected", "replaced_reset", "credit_applied"):
            x = b.free_period(rng, clean=kind == "replaced_reset")
            if x is None:
                kind = "complaint"
        if kind == "est_corrected":
            prim = b.primary[x]
            prim.subtype = "estimated"
            if rng.random() < 0.5:
                b.add(x, rng.randrange(25, 29), "correction",
                      refers_to=prim.id, new_value=round(b.value(x) * rng.uniform(0.92, 1.08), 1))
            else:
                b.add(x, rng.randrange(25, 29), "reading", "actual",
                      period=b.period(x), value=round(b.value(x) * rng.uniform(0.95, 1.05), 1))
        elif kind == "replaced_reset":
            b.add(x, rng.randrange(3, 10), "device", "replaced")
            b.primary[x].fields["reset"] = 1
        elif kind == "credit_applied":
            _matched_pair(b, rng, _shifted_free(b, x))
        else:
            k = rng.choice([m for m in range(b.N) if m != b.P])
            b.add(k, rng.randrange(1, 29), "contact", topic="general complaint", channel=rng.choice(CHANNELS))


# ---------------------------------------------------------------------------
# Case generation
# ---------------------------------------------------------------------------


def generate_case(
    seed: int,
    mechanism: str,
    knobs: Knobs | None = None,
    distractor_seed: int | None = None,
    case_id: str | None = None,
    generator_version: int = GENERATOR_VERSION,
    u2_pair: str | None = None,
) -> Case:
    """One case. `generator_version` 2 reproduces the headline-300 cases. `u2_pair` (version 3, U2 only)
    names the pair ("M1+M3" or "M2+M3"); without it the seed decides, half and half."""
    if mechanism not in MECHANISMS:
        raise ValueError(f"unknown mechanism {mechanism!r}")
    if generator_version not in (2, 3):
        raise ValueError(f"generator_version must be 2 or 3, got {generator_version!r}")
    if u2_pair is not None and (generator_version < 3 or mechanism != "U2" or u2_pair not in U2_PAIRS):
        raise ValueError(f"u2_pair {u2_pair!r} needs generator_version 3, mechanism U2 and one of {sorted(U2_PAIRS)}")
    knobs = knobs or Knobs()
    dseed = seed if distractor_seed is None else distractor_seed
    prng = random.Random(f"plant|{seed}|{mechanism}")
    frng = random.Random(f"fill|{seed}")
    drng = random.Random(f"dist|{dseed}")

    N = max(20, round(knobs.length / 5))
    b = _Build(N, (2010 + frng.randrange(8)) * 12 + frng.randrange(12))
    b.distance = knobs.distance

    if mechanism == "U1":
        planted = [prng.choice(SLOTS)]
    elif mechanism == "U2":
        if generator_version >= 3:
            coin = prng.random()  # always drawn, so an explicit pair leaves the stream where the seed would
            planted = list(U2_PAIRS[u2_pair or ("M1+M3" if coin < 0.5 else "M2+M3")])
        else:
            planted = sorted(prng.sample(SLOTS, 2))
    elif mechanism == "M0":
        planted = []
    else:
        planted = [mechanism]
    want_high = True if mechanism == "M0" else False if mechanism == "U1" else prng.random() < 0.5
    b.P = P = _choose_period(prng, b, knobs.distance, want_high)

    # Monthly cycle: reading (month k), statement and payment (month k+1).
    noise = [frng.uniform(1 - NOISE, 1 + NOISE) for _ in range(N)]
    season = frng.uniform(SEASON_LO, SEASON_HI)
    for k in range(N):
        value = round(BASE_VALUE * (season if b.high(k) else 1.0) * noise[k], 1)
        b.norm[k] = b.val[k] = value
        b.primary[k] = b.add(k, frng.randrange(14, 25), "reading", "actual", period=b.period(k), value=value)
        b.stmt[k] = b.add(k + 1, frng.randrange(2, 7), "statement", period=b.period(k))
        b.pay[k] = b.add(k + 1, frng.randrange(10, 25), "payment")

    # Plant, then fill the other slots with decoys.
    d_tmp: list[str] = []
    for slot in SLOTS:
        if slot in planted:
            d_tmp += _PLANTERS[slot](b, prng)
        else:
            _DECOYS[slot](b, prng)

    if mechanism == "U1":
        before = _profile(b.events)
        gone = set(d_tmp)
        b.deleted = [e for e in b.events if e.id in gone]
        b.events = [e for e in b.events if e.id not in gone]
        d_tmp = []
        for pair, n in sorted((before - _profile(b.events)).items()):
            for _ in range(n):
                _READD[pair](b, prng)
        # Nothing left explains P's amount: the figure stays high, the one-off charge loses its reference.
        if P in b.charge and b.charge[P]["credit"] == 0.0:
            b.charge[P]["ref"] = None

    # Ordinary events, then distractors (own stream).
    K = knobs.distractors
    _add_filler(b, frng, max(0, knobs.length - 3 * N - round(1.5 * K) - 6))
    _add_distractors(b, drng, K)

    _settle(b)
    return _finalise(b, seed, mechanism, knobs, case_id, d_tmp)


def _settle(b: _Build) -> None:
    """Fill every statement's line items and its payment from the final figures."""
    for k in range(b.N):
        activity = round(b.val[k] * RATE, 2)
        ch = b.charge.get(k)
        other = ch["amount"] if ch else 0.0
        credits = ch["credit"] if ch else 0.0
        amount = round(activity + other - credits, 2)
        st = b.stmt[k]
        st.fields.update(activity_charge=activity, other_charges=other, credits=credits, amount=amount)
        if ch and ch["ref"]:
            st.fields["charge_ref"] = ch["ref"]
        b.pay[k].fields["amount"] = amount


def _finalise(b: _Build, seed: int, mechanism: str, knobs: Knobs, case_id: str | None, d_tmp: list[str]) -> Case:
    evs = sorted(b.events, key=lambda e: (e.date, TYPE_RANK[e.type], int(e.id[1:])))
    idmap = {e.id: f"e{i:04d}" for i, e in enumerate(evs)}
    cmap: dict[str, str] = {}
    for e in evs:
        cid = e.fields.get("credit_id")
        if isinstance(cid, str) and cid.startswith("~") and cid not in cmap:
            cmap[cid] = f"cr{len(cmap) + 1:04d}"
    for e in evs:
        if "credit_id" in e.fields:
            e.fields["credit_id"] = cmap[e.fields["credit_id"]]
        if "refers_to" in e.fields:
            e.fields["refers_to"] = idmap[e.fields["refers_to"]]
    order = {e.id: i for i, e in enumerate(evs)}
    d_star = [idmap[t] for t in sorted(d_tmp, key=lambda t: order[t])]
    for e in evs:
        e.id = idmap[e.id]
    deleted = sorted(b.deleted, key=lambda e: (e.date, int(e.id[1:])))
    deleted_ids = [f"x{i + 1:04d}" for i in range(len(deleted))]

    P = b.period(b.P)
    if mechanism == "M0":
        a_star = EXPLAIN
    elif mechanism in ("U1", "U2"):
        a_star = REQUEST_INFO
    else:
        a_star = ACTION_FOR[mechanism]
    query = (
        f"The statement for period {P} is higher than usual. "
        f"Choose one action: {', '.join(ACTIONS)}."
    )
    return Case(
        case_id=case_id or f"{mechanism}-{seed}",
        seed=seed,
        mechanism=mechanism,
        events=evs,
        query_period=P,
        query_text=query,
        actions=list(ACTIONS),
        a_star=a_star,
        d_star=d_star,
        deleted_ids=deleted_ids,
        cls=CLS_FOR[mechanism],
        knobs=knobs.to_dict(),
    )


def _quotas(n: int, mix: dict[str, float]) -> dict[str, int]:
    total = sum(mix.values())
    raw = {m: n * mix.get(m, 0.0) / total for m in MECHANISMS}
    out = {m: int(raw[m]) for m in MECHANISMS}
    left = n - sum(out.values())
    for m in sorted(MECHANISMS, key=lambda m: (-(raw[m] - out[m]), MECHANISMS.index(m)))[:left]:
        out[m] += 1
    return out


def generate_cases(n: int, seed: int, knobs: Knobs | None = None, mix: dict[str, float] | None = None,
                   generator_version: int = GENERATOR_VERSION) -> list[Case]:
    knobs = knobs or Knobs()
    if mix is not None:
        knobs = Knobs(knobs.length, knobs.distance, knobs.distractors, knobs.rare_share, dict(mix))
    mix = knobs.resolved_mix()
    knobs = Knobs(knobs.length, knobs.distance, knobs.distractors, knobs.rare_share, mix)
    mechs = [m for m, q in _quotas(n, mix).items() for _ in range(q)]
    rng = random.Random(f"cases|{seed}")
    rng.shuffle(mechs)
    seeds = [rng.randrange(2**31) for _ in range(n)]
    return [
        generate_case(s, m, knobs, case_id=f"c{seed}-{i:05d}", generator_version=generator_version)
        for i, (m, s) in enumerate(zip(mechs, seeds))
    ]

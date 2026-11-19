"""Event and Case records for the Decision Flip Test, plus JSONL helpers.

Event types and subtypes (fields in brackets):

- statement (period, activity_charge, other_charges, credits, amount, optional charge_ref);
  amount = activity_charge + other_charges - credits
- payment (amount)
- reading, subtype actual or estimated (period, value = activity figure for that period, optional reset=1)
- correction (refers_to, new_value)
- device, subtype installed, replaced or serviced
- plan_notice (new_plan, effective)
- adjustment, subtype credit_approved (credit_id, amount, optional charge_ref and period) or credit_applied (credit_id)
- contact (topic, channel)

Design choice: a reading taken after a device change is an ordinary
reading event with the field reset=1. There is no separate reset type, so the
count of (type, subtype) pairs does not change when a device change is reset.
"""

from __future__ import annotations

import copy
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Iterator

EXPLAIN = "EXPLAIN"
OPEN_REVIEW = "OPEN_REVIEW"
CHECK_DEVICE = "CHECK_DEVICE"
APPLY_CREDIT = "APPLY_CREDIT"
REQUEST_INFO = "REQUEST_INFO"
ACTIONS = (EXPLAIN, OPEN_REVIEW, CHECK_DEVICE, APPLY_CREDIT, REQUEST_INFO)

MECHANISMS = ("M0", "M1", "M2", "M3", "U1", "U2")
CLASSES = ("none", "rare", "unobserved", "unresolved")

STATEMENT = "statement"
PAYMENT = "payment"
READING = "reading"
CORRECTION = "correction"
DEVICE = "device"
PLAN_NOTICE = "plan_notice"
ADJUSTMENT = "adjustment"
CONTACT = "contact"

EVENT_KINDS = (
    (STATEMENT, ""),
    (PAYMENT, ""),
    (READING, "actual"),
    (READING, "estimated"),
    (CORRECTION, ""),
    (DEVICE, "installed"),
    (DEVICE, "replaced"),
    (DEVICE, "serviced"),
    (PLAN_NOTICE, ""),
    (ADJUSTMENT, "credit_approved"),
    (ADJUSTMENT, "credit_applied"),
    (CONTACT, ""),
)


@dataclass
class Event:
    id: str
    date: str  # ISO YYYY-MM-DD
    type: str
    subtype: str = ""
    fields: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "date": self.date,
            "type": self.type,
            "subtype": self.subtype,
            "fields": copy.deepcopy(self.fields),
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "Event":
        return cls(
            id=d["id"],
            date=d["date"],
            type=d["type"],
            subtype=d.get("subtype", ""),
            fields=copy.deepcopy(d.get("fields", {})),
        )


@dataclass
class Case:
    case_id: str
    seed: int
    mechanism: str
    events: list[Event]
    query_period: str  # YYYY-MM
    query_text: str
    actions: list[str]
    a_star: str
    d_star: list[str]
    deleted_ids: list[str]
    cls: str
    knobs: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "case_id": self.case_id,
            "seed": self.seed,
            "mechanism": self.mechanism,
            "events": [e.to_dict() for e in self.events],
            "query_period": self.query_period,
            "query_text": self.query_text,
            "actions": list(self.actions),
            "a_star": self.a_star,
            "d_star": list(self.d_star),
            "deleted_ids": list(self.deleted_ids),
            "cls": self.cls,
            "knobs": copy.deepcopy(self.knobs),
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "Case":
        return cls(
            case_id=d["case_id"],
            seed=d["seed"],
            mechanism=d["mechanism"],
            events=[Event.from_dict(e) for e in d["events"]],
            query_period=d["query_period"],
            query_text=d["query_text"],
            actions=list(d["actions"]),
            a_star=d["a_star"],
            d_star=list(d["d_star"]),
            deleted_ids=list(d["deleted_ids"]),
            cls=d["cls"],
            knobs=copy.deepcopy(d["knobs"]),
        )


def dumps_case(case: Case) -> str:
    """One canonical JSON line: sorted keys, fixed separators, ASCII only."""
    return json.dumps(case.to_dict(), sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def write_jsonl(path: str | Path, cases: Iterable[Case]) -> int:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with p.open("w", encoding="utf-8", newline="\n") as fh:
        for c in cases:
            fh.write(dumps_case(c))
            fh.write("\n")
            n += 1
    return n


def read_jsonl(path: str | Path) -> Iterator[Case]:
    with Path(path).open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                yield Case.from_dict(json.loads(line))


@dataclass
class View:
    """What a memory system hands the decider.

    text: the exact text the decider sees as its records.
    event_ids: ids of the events quoted verbatim in the text (empty for a summary).
    units: the text units the view is made of, in order: event lines for an event
    view, sentences for a summary. The residue stage inserts units, not events.
    """

    system: str
    budget: int | None
    text: str
    event_ids: list[str] = field(default_factory=list)
    units: list[str] = field(default_factory=list)


@dataclass
class Unit:
    """One insertable piece of text for the flip search.

    id: the event id, or a unit id such as "u003" for a sentence.
    group: the group name for group-first search ("type/subtype" for events).
    event: the source event when the unit is an event line, else None.
    """

    id: str
    text: str
    group: str = ""
    event: Event | None = None


STATUSES = ("restored", "not_recoverable_by_insertion", "no_flip")
STAGES = ("flip", "residue")


@dataclass
class DistinctionRecord:
    """Result of one flip search (Distinction Record v0).

    action_restored: the target action for "restored" and "no_flip"; for
    "not_recoverable_by_insertion" it is the action returned with every missing
    event inserted. one_minimal is only meaningful when status is "restored".
    """

    case_id: str
    system: str
    budget: int | None
    model: str
    status: str
    event_ids: list[str]
    event_texts: list[str]
    action_from_view: str
    action_restored: str | None
    cls: str
    calls_used: int
    hit_cap: bool
    one_minimal: bool
    verified: bool | None
    # Group-first search (added in step 3). Defaults keep old records loadable.
    # granularity: "event" when the result is 1-minimal at event level, else "group".
    # groups: "type/subtype" (or just "type") of each selected group.
    # phase_calls: [group-level calls including the two opening calls, event-level calls].
    granularity: str = "event"
    groups: list[str] = field(default_factory=list)
    phase_calls: list[int] = field(default_factory=lambda: [0, 0])
    # Added in step 5. stage "flip" searches from the view V; "residue" searches from the
    # handoff note N. units_inserted: the texts of the minimal set (event lines or sentences).
    # For a residue record, event_ids holds unit ids (event ids for event views).
    stage: str = "flip"
    units_inserted: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "case_id": self.case_id,
            "system": self.system,
            "budget": self.budget,
            "model": self.model,
            "status": self.status,
            "event_ids": list(self.event_ids),
            "event_texts": list(self.event_texts),
            "action_from_view": self.action_from_view,
            "action_restored": self.action_restored,
            "cls": self.cls,
            "calls_used": self.calls_used,
            "hit_cap": self.hit_cap,
            "one_minimal": self.one_minimal,
            "verified": self.verified,
            "granularity": self.granularity,
            "groups": list(self.groups),
            "phase_calls": list(self.phase_calls),
            "stage": self.stage,
            "units_inserted": list(self.units_inserted),
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "DistinctionRecord":
        return cls(
            case_id=d["case_id"],
            system=d["system"],
            budget=d["budget"],
            model=d["model"],
            status=d["status"],
            event_ids=list(d["event_ids"]),
            event_texts=list(d["event_texts"]),
            action_from_view=d["action_from_view"],
            action_restored=d["action_restored"],
            cls=d["cls"],
            calls_used=d["calls_used"],
            hit_cap=d["hit_cap"],
            one_minimal=d["one_minimal"],
            verified=d["verified"],
            granularity=d.get("granularity", "event"),
            groups=list(d.get("groups", [])),
            phase_calls=list(d.get("phase_calls", [0, 0])),
            stage=d.get("stage", "flip"),
            units_inserted=list(d.get("units_inserted", [])),
        )

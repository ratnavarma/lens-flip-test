"""Event wording. Three templates per (type, subtype), chosen by a seeded RNG."""

from __future__ import annotations

import random
from typing import Iterable

from .schema import Case, Event

TEMPLATES: dict[tuple[str, str], tuple[str, str, str]] = {
    ("statement", ""): (
        "Statement for period {period}: activity charges {activity_charge:.2f}, other charges {other_charges:.2f}, "
        "credits {credits:.2f}, due {amount:.2f}.",
        "Account statement {period}: activity charges {activity_charge:.2f}, other charges {other_charges:.2f}, "
        "credits {credits:.2f}, total {amount:.2f}.",
        "Statement {period} sent. Activity {activity_charge:.2f}, other charges {other_charges:.2f}, "
        "credits {credits:.2f}. Charge {amount:.2f}.",
    ),
    ("payment", ""): (
        "Payment received: {amount:.2f}.",
        "Customer paid {amount:.2f}.",
        "A payment of {amount:.2f} was posted.",
    ),
    ("reading", "actual"): (
        "Actual reading for {period}: {value:.1f}. Activity in that month alone.",
        "Reading taken on site for {period}. Month activity {value:.1f}.",
        "Recorded reading ({period}, actual): {value:.1f}, activity of the month only.",
    ),
    ("reading", "estimated"): (
        "Estimated reading for {period}: {value:.1f}. Estimated activity in that month alone.",
        "No access on the visit. Reading for {period} estimated at {value:.1f}, month activity.",
        "Estimate recorded for {period}: {value:.1f}, estimated activity of the month only.",
    ),
    ("correction", ""): (
        "Correction to {refers_to}: new value {new_value:.1f}.",
        "Record {refers_to} corrected. Value is now {new_value:.1f}.",
        "Amended {refers_to} with the value {new_value:.1f}.",
    ),
    ("device", "installed"): (
        "Device installed.",
        "A new device was installed at the account.",
        "Installation of a device completed.",
    ),
    ("device", "replaced"): (
        "Device replaced.",
        "The device was swapped for a new unit.",
        "Replacement of the device completed.",
    ),
    ("device", "serviced"): (
        "Device serviced.",
        "Routine service visit on the device.",
        "Technician serviced the device.",
    ),
    ("plan_notice", ""): (
        "Plan change notice: moves to the {new_plan} plan effective {effective}.",
        "Notice sent. From {effective} the account is on the {new_plan} plan.",
        "The {new_plan} plan takes effect on {effective}.",
    ),
    ("adjustment", "credit_approved"): (
        "Credit {credit_id} approved for {amount:.2f}.",
        "Approved a credit ({credit_id}) of {amount:.2f}.",
        "Credit approval {credit_id}: {amount:.2f}.",
    ),
    ("adjustment", "credit_applied"): (
        "Credit {credit_id} applied to the account.",
        "Applied credit {credit_id}.",
        "Credit {credit_id} was posted against the balance.",
    ),
    ("contact", ""): (
        "Customer contact by {channel}: {topic}.",
        "Contact logged ({channel}). Topic: {topic}.",
        "The customer reached out by {channel} about: {topic}.",
    ),
}

RESET_SUFFIX = " The device counter restarted, so this is a reset reading."
CHARGE_SUFFIX = " One-off {charge_ref}."
OFFSET_SUFFIX = " Offsets one-off charge {charge_ref} charged in {period}."


def render_event(event: Event, rng: random.Random) -> str:
    """One line: id, date, wording. Consumes exactly one draw from rng."""
    templates = TEMPLATES[(event.type, event.subtype)]
    text = templates[rng.randrange(3)].format(**event.fields)
    if event.type == "reading" and event.fields.get("reset"):
        text += RESET_SUFFIX
    if event.type == "statement" and event.fields.get("charge_ref"):
        text += CHARGE_SUFFIX.format(**event.fields)
    if event.subtype == "credit_approved" and event.fields.get("charge_ref"):
        text += OFFSET_SUFFIX.format(**event.fields)
    return f"{event.id} {event.date} {text}"


def render_history(events: Iterable[Event], seed: int) -> str:
    """Render events in the given order, one per line, wording chosen by seed."""
    rng = random.Random(f"render|{seed}")
    return "\n".join(render_event(e, rng) for e in events)


def render_events(case: Case, events: Iterable[Event]) -> list[str]:
    """Rendered lines for the given events, in the given order.

    Wording depends on (case seed, event id) only, so an event reads the same in
    every view of the case.
    """
    return [render_event(e, random.Random(f"render|{case.seed}|{e.id}")) for e in events]


def render_lines(case: Case, ids: Iterable[str]) -> list[str]:
    """Rendered lines for ids, in history order."""
    wanted = set(ids)
    return render_events(case, (e for e in case.events if e.id in wanted))

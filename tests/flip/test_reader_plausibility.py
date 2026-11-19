"""Reader-plausibility checks: every mechanism visibly causes P's high amount, seasonality is real."""

from __future__ import annotations

import datetime as dt
import itertools
import statistics

import pytest

from lens.flip.batch import request_params
from lens.flip.decider import CONTEXT, PROMPT_VERSIONS, build_messages
from lens.flip.systems.base import event_view
from lens.flip.generator import DISTANCES, Knobs, detect_patterns, generate_cases, is_high_demand
from lens.flip.render import render_history
from lens.flip.schema import MECHANISMS, Case

UNIFORM = {m: 1.0 for m in MECHANISMS}


@pytest.fixture(scope="module")
def sample():
    cases: list[Case] = []
    for i, (length, dist, k) in enumerate(itertools.product((100, 400), DISTANCES, (2, 8))):
        cases += generate_cases(90, 3000 + i, Knobs(length=length, distance=dist, distractors=k), mix=UNIFORM)
    cases += generate_cases(60, 3100, Knobs(length=1600, distance="middle", distractors=8), mix=UNIFORM)
    return cases


def _stmts(case: Case) -> dict[str, dict]:
    return {e.fields["period"]: e.fields for e in case.events if e.type == "statement"}


def _earlier_same_month(periods, p: str) -> list[str]:
    return [q for q in periods if q[5:7] == p[5:7] and q[:4] < p[:4]]


def _norm(case: Case) -> float:
    """Seasonal norm for P: mean activity charge of the same calendar month in earlier years."""
    st = _stmts(case)
    earlier = _earlier_same_month(st, case.query_period)
    assert earlier, case.case_id
    return statistics.mean(st[q]["activity_charge"] for q in earlier)


def _readings(case: Case, period: str, subtype: str | None = None):
    return [e for e in case.events if e.type == "reading" and e.fields["period"] == period
            and (subtype is None or e.subtype == subtype)]


def test_statement_lines_add_up(sample):
    for c in sample:
        for e in c.events:
            if e.type == "statement":
                f = e.fields
                assert abs(f["activity_charge"] + f["other_charges"] - f["credits"] - f["amount"]) < 0.011, e.id


def test_every_p_has_an_earlier_year(sample):
    for c in sample:
        assert _earlier_same_month(_stmts(c), c.query_period), c.case_id


def test_m0_matches_same_month_of_earlier_years(sample):
    n = 0
    for c in sample:
        if c.mechanism != "M0":
            continue
        st = _stmts(c)
        earlier = [st[q]["activity_charge"] for q in _earlier_same_month(st, c.query_period)]
        mean = statistics.mean(earlier)
        assert abs(st[c.query_period]["activity_charge"] / mean - 1) <= 0.10, c.case_id
        assert st[c.query_period]["other_charges"] == 0 and st[c.query_period]["credits"] == 0
        n += 1
    assert n > 100


def test_high_demand_months_are_at_least_1_5x_low_demand_months(sample):
    # Per account, from the activity charges of every period except P (P carries the planted anomaly).
    for c in sample:
        st = _stmts(c)
        hi = [v["activity_charge"] for p, v in st.items() if p != c.query_period and is_high_demand(p)]
        lo = [v["activity_charge"] for p, v in st.items() if p != c.query_period and not is_high_demand(p)]
        assert hi and lo, c.case_id
        assert statistics.mean(hi) >= 1.5 * statistics.mean(lo), c.case_id


def test_high_demand_months_share_one_factor(sample):
    for c in sample:
        st = _stmts(c)
        hi = [v["activity_charge"] for p, v in st.items() if p != c.query_period and is_high_demand(p)]
        assert max(hi) / min(hi) <= 1.08, c.case_id  # same factor, 3 percent noise each way


def test_rare_mechanisms_raise_p_visibly(sample):
    n = 0
    for c in sample:
        if c.mechanism not in ("M1", "M2", "M3", "U1", "U2"):
            continue
        amount = _stmts(c)[c.query_period]["amount"]
        assert amount >= 1.3 * _norm(c), (c.case_id, c.mechanism, amount, _norm(c))
        n += 1
    assert n > 400


def test_m1_estimate_is_high_and_actual_is_in_line(sample):
    n = 0
    for c in sample:
        if c.mechanism != "M1":
            continue
        st = _stmts(c)
        prior = [e.fields["value"] for e in c.events if e.type == "reading" and e.fields["period"] in
                 _earlier_same_month(st, c.query_period)]
        mean = statistics.mean(prior)
        est = _readings(c, c.query_period, "estimated")
        assert len(est) == 1 and est[0].fields["value"] >= 1.5 * mean, c.case_id
        late = [r for r in _readings(c, c.query_period, "actual")]
        assert len(late) == 1 and late[0].date > next(
            e.date for e in c.events if e.type == "statement" and e.fields["period"] == c.query_period)
        assert abs(late[0].fields["value"] / mean - 1) <= 0.12, c.case_id
        assert not any(e.type == "correction" and e.fields["refers_to"] == est[0].id for e in c.events)
        # the statement is computed from the estimate
        assert abs(st[c.query_period]["activity_charge"] - est[0].fields["value"] * 0.2) < 0.011
        n += 1
    assert n > 50


def test_m2_replaced_in_p_without_reset_and_decoys_reset(sample):
    n = 0
    for c in sample:
        if c.mechanism != "M2":
            continue
        p = c.query_period
        assert any(e.type == "device" and e.subtype == "replaced" and e.date[:7] == p for e in c.events)
        prim = _readings(c, p, "actual")
        assert len(prim) == 1 and not prim[0].fields.get("reset")
        prior = [e.fields["value"] for e in c.events if e.type == "reading" and e.fields["period"] in
                 _earlier_same_month(_stmts(c), p)]
        assert prim[0].fields["value"] >= 1.6 * statistics.mean(prior), c.case_id
        # every other replacement is followed by a reset reading in its own period
        for dev in (e for e in c.events if e.type == "device" and e.subtype == "replaced" and e.date[:7] != p):
            rs = [r for r in _readings(c, dev.date[:7]) if r.fields.get("reset")]
            assert rs and rs[0].date >= dev.date, (c.case_id, dev.id)
        n += 1
    assert n > 50


def test_m3_charge_credit_and_never_applied(sample):
    n = 0
    for c in sample:
        if c.mechanism != "M3":
            continue
        p = c.query_period
        f = _stmts(c)[p]
        assert f["other_charges"] > 0 and f["credits"] == 0 and f["charge_ref"].endswith(p)
        assert abs(f["amount"] - f["activity_charge"] - f["other_charges"]) < 0.011
        assert abs(f["activity_charge"] - _norm(c)) <= 0.10 * _norm(c)  # the whole excess is the charge
        mine = [e for e in c.events if e.subtype == "credit_approved" and e.fields.get("charge_ref") == f["charge_ref"]]
        assert len(mine) == 1 and mine[0].fields["amount"] == f["other_charges"], c.case_id
        assert mine[0].fields["period"] == p
        applied = {e.fields["credit_id"] for e in c.events if e.subtype == "credit_applied"}
        assert mine[0].fields["credit_id"] not in applied
        stmt_date = next(e.date for e in c.events if e.type == "statement" and e.fields["period"] == p)
        assert (dt.date.fromisoformat(stmt_date) - dt.date.fromisoformat(mine[0].date)).days >= 90
        assert mine[0].id in c.d_star
        n += 1
    assert n > 50


def test_applied_credits_show_in_the_credits_line(sample):
    seen = 0
    for c in sample:
        st = _stmts(c)
        approved = {e.fields["credit_id"]: e for e in c.events if e.subtype == "credit_approved"}
        for e in c.events:
            if e.subtype != "credit_applied":
                continue
            adj = approved[e.fields["credit_id"]]
            q = adj.fields["period"]
            assert q != c.query_period, c.case_id  # decoys never sit in P
            assert st[q]["credits"] == adj.fields["amount"] > 0
            assert st[q]["charge_ref"] == adj.fields["charge_ref"]
            assert abs(st[q]["amount"] - st[q]["activity_charge"]) < 0.011  # nets to zero
            assert adj.date < e.date <= next(
                s.date for s in c.events if s.type == "statement" and s.fields["period"] == q)
            seen += 1
    assert seen > 500


def test_decoys_and_distractors_leave_p_alone(sample):
    for c in sample:
        p = c.query_period
        f = _stmts(c)[p]
        if c.mechanism in ("M0", "M1", "M2"):
            assert f["other_charges"] == 0 and f["credits"] == 0 and "charge_ref" not in f, c.case_id
        in_p = [e for e in c.events if e.date[:7] == p]
        assert not any(e.type == "correction" or e.fields.get("reset") for e in in_p)
        # no credit refers to a charge in P except M3's own (and U2's M3 when planted)
        refs = [e for e in c.events if e.subtype == "credit_approved" and e.fields.get("period") == p]
        wants = "M3" in detect_patterns(c.events, p)
        assert len(refs) == (1 if wants else 0), c.case_id


def test_u1_leaves_no_recorded_cause(sample):
    n = 0
    for c in sample:
        if c.mechanism != "U1":
            continue
        p = c.query_period
        f = _stmts(c)[p]
        assert not is_high_demand(p)
        assert not _readings(c, p, "estimated")
        assert not any(e.type == "device" and e.subtype == "replaced" and e.date[:7] == p for e in c.events)
        assert "charge_ref" not in f and f["credits"] == 0
        assert not any(e.subtype == "credit_approved" and e.fields.get("period") == p for e in c.events)
        assert detect_patterns(c.events, p) == []
        assert f["amount"] >= 1.3 * _norm(c)
        n += 1
    assert n > 50


def test_u2_both_mechanisms_visible(sample):
    n = 0
    for c in sample:
        if c.mechanism != "U2":
            continue
        pats = detect_patterns(c.events, c.query_period)
        assert len(pats) == 2
        f = _stmts(c)[c.query_period]
        norm = _norm(c)
        if "M3" in pats:
            assert f["other_charges"] > 0 and f.get("charge_ref")
        if "M1" in pats or "M2" in pats:
            assert f["activity_charge"] >= 1.5 * norm
        assert f["amount"] >= 1.3 * norm
        n += 1
    assert n > 50


def _full(case):
    return event_view(case, (e.id for e in case.events), "full")


def test_context_line_is_shared_by_every_prompt_path(sample):
    c = sample[0]
    for v in PROMPT_VERSIONS:
        user = build_messages(_full(c), [], c, v)[1]["content"]
        assert user.startswith(CONTEXT)
        assert "activity figure for its own period" in CONTEXT and "restarts the count" in CONTEXT
        assert "activity charges, other charges and credits" in CONTEXT
    batch_user = request_params(c, "off")["messages"][0]["content"]
    assert batch_user == build_messages(_full(c), [], c, "v1")[1]["content"]


def test_readings_are_worded_as_per_period_figures(sample):
    text = render_history([e for e in sample[0].events if e.type == "reading"], 1)
    lines = [ln for ln in text.splitlines() if "reading" in ln.lower() or "Estimate" in ln]
    assert lines and all("month" in ln.lower() for ln in lines)

"""Step 1 checks: schema, rendering, generator and rule oracle."""

from __future__ import annotations

import itertools
import re
import statistics
from collections import Counter
from pathlib import Path

import pytest

from lens import main as lens_main
from lens.flip.generator import (
    DISTANCES,
    Knobs,
    default_mix,
    detect_patterns,
    generate_case,
    generate_cases,
    is_high_demand,
    oracle,
)
from lens.flip.render import render_history
from lens.flip.schema import ACTIONS, EVENT_KINDS, MECHANISMS, Case, Event, read_jsonl, write_jsonl

UNIFORM = {m: 1.0 for m in MECHANISMS}

# The banned domain words are private. They live in an optional, gitignored file at the repo root,
# one word per line. Without it, the word checks skip.
_BANNED_FILE = Path(__file__).resolve().parents[2] / "banned_words.local.txt"
FORBIDDEN = ([w.strip().lower() for w in _BANNED_FILE.read_text().splitlines() if w.strip()]
             if _BANNED_FILE.exists() else [])


def _sample(lengths=(100, 400), per_combo=180, seed=1000):
    """Class-balanced cases across lengths, distances and distractor counts."""
    cases: list[Case] = []
    for i, (length, dist, k) in enumerate(itertools.product(lengths, DISTANCES, (2, 8))):
        knobs = Knobs(length=length, distance=dist, distractors=k)
        cases += generate_cases(per_combo, seed + i, knobs, mix=UNIFORM)
    return cases


@pytest.fixture(scope="module")
def sample():
    return _sample()


@pytest.fixture(scope="module")
def long_sample():
    return generate_cases(120, 77, Knobs(length=1600, distance="middle", distractors=8), mix=UNIFORM)


def test_sample_size(sample, long_sample):
    assert len(sample) >= 2000
    assert len(long_sample) == 120


# Check 1
def test_oracle_matches_a_star(sample, long_sample):
    bad = [c.case_id for c in sample + long_sample if oracle(c.events, c.query_period) != c.a_star]
    assert not bad, bad[:5]


# Check 2
def test_deleting_d_star_changes_oracle(sample, long_sample):
    checked = 0
    for c in sample + long_sample:
        if c.mechanism not in ("M1", "M2", "M3"):
            continue
        assert c.d_star, c.case_id
        gone = set(c.d_star)
        rest = [e for e in c.events if e.id not in gone]
        assert len(rest) == len(c.events) - len(gone)
        assert oracle(rest, c.query_period) != c.a_star, c.case_id
        checked += 1
    assert checked > 500


# Check 3
_ID_FIELDS = {"credit_id", "refers_to"}


def _sig(e: Event):
    return (e.date, e.type, e.subtype, tuple(sorted((k, v) for k, v in e.fields.items() if k not in _ID_FIELDS)))


def test_swapping_distractors_never_changes_a_star(sample):
    differ = 0
    for i, c in enumerate(sample[::7]):
        knobs = Knobs(**{k: v for k, v in c.knobs.items()})
        other = generate_case(c.seed, c.mechanism, knobs, distractor_seed=c.seed + 12345)
        assert other.a_star == c.a_star
        assert other.query_period == c.query_period
        assert oracle(other.events, other.query_period) == c.a_star
        by_id = {e.id: e for e in c.events}
        by_id2 = {e.id: e for e in other.events}
        assert sorted(_sig(by_id[x]) for x in c.d_star) == sorted(_sig(by_id2[x]) for x in other.d_star)
        differ += other.events != c.events
    assert differ > 0  # the distractors really were re-drawn


def test_distractors_only_in_other_periods(sample):
    # P itself holds no look-alike: at most the planted pattern, never a corrected estimate or a reset.
    for c in sample[::5]:
        in_p = [e for e in c.events if e.date[:7] == c.query_period]
        assert not any(e.type == "correction" for e in in_p)
        assert not any(e.fields.get("reset") for e in in_p)


# Check 4
def _features(case: Case):
    cnt = Counter((e.type, e.subtype) for e in case.events)
    return [cnt[k] for k in EVENT_KINDS]


def _centroid_accuracy(cases):
    by_class: dict[str, list[Case]] = {m: [] for m in MECHANISMS}
    for c in cases:
        by_class[c.mechanism].append(c)
    n = min(len(v) for v in by_class.values())
    train, test = [], []
    for m in MECHANISMS:
        half = n // 2
        train += by_class[m][:half]
        test += by_class[m][half:n]
    cent = {}
    for m in MECHANISMS:
        rows = [_features(c) for c in train if c.mechanism == m]
        cent[m] = [sum(col) / len(rows) for col in zip(*rows)]
    hits = 0
    for c in test:
        f = _features(c)
        pred = min(MECHANISMS, key=lambda m: sum((a - b) ** 2 for a, b in zip(f, cent[m])))
        hits += pred == c.mechanism
    return hits / len(test)


def test_type_count_classifier_is_near_chance(sample):
    acc = _centroid_accuracy(sample)
    print(f"\nclassifier accuracy (mixed knobs): {acc:.3f}")
    assert acc <= 0.30


def test_type_count_classifier_fixed_knobs():
    cases = generate_cases(1200, 4242, Knobs(length=100, distance="middle", distractors=2), mix=UNIFORM)
    acc = _centroid_accuracy(cases)
    print(f"\nclassifier accuracy (length 100, 2 distractors): {acc:.3f}")
    assert acc <= 0.30


# Check 5
def test_same_seed_same_bytes(tmp_path):
    knobs = Knobs(length=100, distance="far", distractors=8)
    a, b = tmp_path / "a.jsonl", tmp_path / "b.jsonl"
    write_jsonl(a, generate_cases(60, 5, knobs))
    write_jsonl(b, generate_cases(60, 5, knobs))
    assert a.read_bytes() == b.read_bytes()
    c = tmp_path / "c.jsonl"
    write_jsonl(c, generate_cases(60, 6, knobs))
    assert c.read_bytes() != a.read_bytes()


def test_cli_same_seed_same_bytes(tmp_path):
    args = ["flip", "gen", "--n", "12", "--seed", "3", "--length", "100", "--distance", "near", "--distractors", "8"]
    assert lens_main(args + ["--out", str(tmp_path / "x.jsonl")]) == 0
    assert lens_main(args + ["--out", str(tmp_path / "y.jsonl")]) == 0
    assert (tmp_path / "x.jsonl").read_bytes() == (tmp_path / "y.jsonl").read_bytes()
    assert len(list(read_jsonl(tmp_path / "x.jsonl"))) == 12


# Forbidden words
def test_no_forbidden_words_in_rendered_histories():
    if not FORBIDDEN:
        pytest.skip("banned_words.local.txt not present")
    cases = []
    for i, (length, dist, k) in enumerate(itertools.product((100, 400), DISTANCES, (2, 8))):
        cases += generate_cases(25, 900 + i, Knobs(length=length, distance=dist, distractors=k), mix=UNIFORM)
    cases = cases[:300]
    assert len(cases) == 300
    pattern = re.compile("|".join(FORBIDDEN), re.I)
    for c in cases:
        text = render_history(c.events, c.seed) + "\n" + c.query_text + "\n" + " ".join(c.knobs.keys())
        m = pattern.search(text)
        assert m is None, (c.case_id, m and m.group(0))


def test_no_forbidden_words_in_sources():
    if not FORBIDDEN:
        pytest.skip("banned_words.local.txt not present")
    root = Path(__file__).resolve().parents[2]
    pattern = re.compile("|".join(FORBIDDEN), re.I)
    for path in list((root / "src").rglob("*.py")) + [Path(__file__)]:
        assert pattern.search(path.read_text()) is None, path


# Round trip
def test_round_trip(sample, tmp_path):
    for c in sample[::25]:
        assert Case.from_dict(c.to_dict()) == c
    p = tmp_path / "rt.jsonl"
    write_jsonl(p, sample[:50])
    assert list(read_jsonl(p)) == sample[:50]


# Extra structure checks
def test_structure(sample):
    for c in sample[::3]:
        dates = [e.date for e in c.events]
        assert dates == sorted(dates)
        assert len({e.id for e in c.events}) == len(c.events)
        assert c.actions == list(ACTIONS)
        assert c.query_text.startswith(f"The statement for period {c.query_period} is higher than usual.")
        high = is_high_demand(c.query_period)
        if c.mechanism == "M0":
            assert high and c.d_star == [] and c.cls == "none"
        if c.mechanism == "U1":
            assert not high and c.d_star == [] and c.cls == "unobserved" and 1 <= len(c.deleted_ids) <= 2
            assert not set(c.deleted_ids) & {e.id for e in c.events}
        else:
            assert c.deleted_ids == []
        if c.mechanism == "U2":
            assert c.cls == "unresolved" and len(c.d_star) >= 2
            assert len(detect_patterns(c.events, c.query_period)) == 2
        # P's statement is clearly above the typical amount
        stmts = {e.fields["period"]: e.fields["amount"] for e in c.events if e.type == "statement"}
        typical = statistics.median(v for p, v in stmts.items() if p != c.query_period)
        assert stmts[c.query_period] >= 1.25 * typical
        # P has a later period in the history
        assert max(stmts) > c.query_period


def test_high_demand_share_for_rare_mechanisms(sample):
    for m in ("M1", "M2", "M3", "U2"):
        flags = [is_high_demand(c.query_period) for c in sample if c.mechanism == m]
        assert 0.35 < sum(flags) / len(flags) < 0.65, m


def test_lengths_and_mix():
    for length in (100, 400, 1600):
        cs = generate_cases(10, 3, Knobs(length=length, distractors=2))
        sizes = [len(c.events) for c in cs]
        assert all(0.9 * length <= s <= 1.15 * length for s in sizes), (length, sizes)
    cs = generate_cases(100, 3, Knobs(rare_share=0.5))
    counts = Counter(c.mechanism for c in cs)
    assert counts["M0"] == 50 and all(counts[m] == 10 for m in MECHANISMS if m != "M0")
    assert cs[0].knobs["mix"] == default_mix(0.5)


def test_count_profile_identical_across_mechanisms_for_one_seed():
    # With no distractors the ordinary events are the same for a seed, so the
    # (type, subtype) counts must match exactly across all six mechanisms.
    for length in (100, 400):
        for dist in DISTANCES:
            knobs = Knobs(length=length, distance=dist, distractors=0)
            for seed in range(40):
                profiles = {
                    m: Counter((e.type, e.subtype) for e in generate_case(seed, m, knobs).events) for m in MECHANISMS
                }
                assert all(p == profiles["M0"] for p in profiles.values()), (length, dist, seed, profiles)


# Clean window
def _window_events(c: Case):
    """Device events and plan notices in the month before P, P or the month after P (by date or effective date)."""
    y, m = int(c.query_period[:4]), int(c.query_period[5:7])
    prev = f"{y - 1:04d}-12" if m == 1 else f"{y:04d}-{m - 1:02d}"
    nxt = f"{y + 1:04d}-01" if m == 12 else f"{y:04d}-{m + 1:02d}"
    window = {c.query_period, prev, nxt}
    out = []
    for e in c.events:
        if e.type == "device" and e.date[:7] in window:
            out.append(e)
        elif e.type == "plan_notice" and (e.date[:7] in window or e.fields["effective"][:7] in window):
            out.append(e)
    return out


def test_clean_window_has_no_unplanted_device_or_plan_events(sample):
    assert len(sample) >= 2000
    planted_seen = 0
    for c in sample:
        planted = {e for e in c.d_star}
        for e in _window_events(c):
            # The only allowed event is the replacement a mechanism planted in P.
            assert e.id in planted and e.type == "device" and e.subtype == "replaced" and e.date[:7] == c.query_period, (
                c.case_id, c.mechanism, e,
            )
            planted_seen += 1
    assert planted_seen > 300  # M2 and U2 with M2 do plant one


def test_clean_window_long_histories(long_sample):
    for c in long_sample:
        planted = set(c.d_star)
        assert all(e.id in planted for e in _window_events(c)), c.case_id

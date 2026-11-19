"""Protocol v3: the U2 pair rule, generator versions, subset runs, the merge report, the pre-check and the protocol file.
No network: every runner is a fake."""

from __future__ import annotations

import hashlib
import itertools
import json
import re
from collections import Counter
from pathlib import Path

import pytest

from lens import main as lens_main
from lens.flip import batch, decider
from lens.flip.generator import (
    DISTANCES,
    GENERATOR_VERSION,
    U2_PAIRS,
    Knobs,
    detect_patterns,
    generate_case,
    generate_cases,
    oracle,
)
from lens.flip.headline import engine as E
from lens.flip.headline import merge as MG
from lens.flip.headline import metrics as HM
from lens.flip.headline import mock as M
from lens.flip.headline import protocol as P
from lens.flip.schema import MECHANISMS, REQUEST_INFO, Case, read_jsonl, write_jsonl
from lens.flip.systems.base import event_view

ROOT = Path(__file__).resolve().parents[2]
UNIFORM = {m: 1.0 for m in MECHANISMS}
OTHER = ("M0", "M1", "M2", "M3", "U1")
# The banned domain words are private. They live in an optional, gitignored file at the repo root,
# one word per line. Without it, the word checks skip.
_BANNED_FILE = Path(__file__).resolve().parents[2] / "banned_words.local.txt"
FORBIDDEN = ([w.strip().lower() for w in _BANNED_FILE.read_text().splitlines() if w.strip()]
             if _BANNED_FILE.exists() else [])
BANNED_STYLE = ("delve", "leverage", "robust", "crucial", "moreover", "notably")


def _sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def _pair(c: Case) -> str:
    return "+".join(detect_patterns(c.events, c.query_period))


# ---------------------------------------------------------------------------
# the generator
# ---------------------------------------------------------------------------

def test_default_generator_version_is_three():
    assert GENERATOR_VERSION == 3 and set(U2_PAIRS) == {"M1+M3", "M2+M3"}


def test_u2_never_plants_m1_m2_in_v3_mode():
    pairs = Counter()
    for i, (length, dist) in enumerate(itertools.product((100, 400), DISTANCES)):
        for c in generate_cases(150, 300 + i, Knobs(length=length, distance=dist), mix={"U2": 1.0}):
            assert c.mechanism == "U2" and c.cls == "unresolved" and c.a_star == REQUEST_INFO
            pairs[_pair(c)] += 1
    assert set(pairs) == {"M1+M3", "M2+M3"}, pairs
    total = sum(pairs.values())
    assert 0.4 < pairs["M1+M3"] / total < 0.6  # half each, by seed


def test_v2_mode_still_plants_all_three_pairs():
    pairs = Counter(_pair(c) for c in generate_cases(300, 5, Knobs(length=100), mix={"U2": 1.0}, generator_version=2))
    assert set(pairs) == {"M1+M2", "M1+M3", "M2+M3"}


def test_each_pair_puts_one_cause_on_each_charge_line():
    # M1 and M2 raise the activity charge of P. M3 puts a one-off charge on the other charges and no credit.
    for c in generate_cases(120, 41, Knobs(length=100), mix={"U2": 1.0}):
        st = next(e for e in c.events if e.type == "statement" and e.fields["period"] == c.query_period)
        assert st.fields["other_charges"] > 0 and st.fields["credits"] == 0
        assert _pair(c) in ("M1+M3", "M2+M3")


def test_explicit_pair_and_bad_arguments():
    for pair in U2_PAIRS:
        for seed in range(6):
            assert _pair(generate_case(seed, "U2", Knobs(length=100), u2_pair=pair)) == pair
    # an explicit pair leaves everything else of the seed's case where the seed would have put it
    a = generate_case(9, "U2", Knobs(length=100))
    b = generate_case(9, "U2", Knobs(length=100), u2_pair=_pair(a))
    assert a == b
    with pytest.raises(ValueError):
        generate_case(1, "U2", Knobs(length=100), u2_pair="M1+M2")
    with pytest.raises(ValueError):
        generate_case(1, "M1", Knobs(length=100), u2_pair="M1+M3")
    with pytest.raises(ValueError):
        generate_case(1, "U2", Knobs(length=100), generator_version=2, u2_pair="M1+M3")
    with pytest.raises(ValueError):
        generate_case(1, "U2", Knobs(length=100), generator_version=4)


def test_other_mechanisms_are_identical_in_both_versions():
    for length, dist in itertools.product((100, 400), DISTANCES):
        knobs = Knobs(length=length, distance=dist)
        for seed in range(25):
            for m in OTHER:
                a = generate_case(seed, m, knobs, generator_version=2)
                b = generate_case(seed, m, knobs, generator_version=3)
                assert a.to_dict() == b.to_dict(), (m, seed)
    # and U2 is the one that differs
    assert any(generate_case(s, "U2", Knobs(length=100), generator_version=2).to_dict()
               != generate_case(s, "U2", Knobs(length=100)).to_dict() for s in range(10))


def test_decider_prompt_text_is_byte_identical_across_versions():
    for m in MECHANISMS:
        for seed in range(6):
            texts = {}
            for v in (2, 3):
                case = generate_case(seed, m, Knobs(length=100), generator_version=v)
                view = event_view(case, (e.id for e in case.events), "full")
                texts[v] = {pv: decider.build_messages(view, [], case, pv) for pv in decider.PROMPT_VERSIONS}
            for pv in decider.PROMPT_VERSIONS:
                sys2 = next(x["content"] for x in texts[2][pv] if x["role"] == "system")
                sys3 = next(x["content"] for x in texts[3][pv] if x["role"] == "system")
                assert sys2 == sys3  # the policy and the action list
                if m != "U2":
                    assert texts[2][pv] == texts[3][pv]


def test_v3_hashes_equal_v2_except_the_generator():
    now = P.current_hashes()
    assert set(now) == set(P.V2_HASHES)
    assert [k for k in now if now[k] != P.V2_HASHES[k]] == ["file:generator.py"]


def test_v2_hash_constants_match_the_frozen_v2_file():
    text = (ROOT / "docs" / "flip-protocol-v2.md").read_text()
    rows = dict(re.findall(r"^\| ([\w.:]+) \| `([0-9a-f]{64})` \|$", text, re.M))
    assert rows == P.V2_HASHES
    assert P.V2_BASE_RUN["cases.jsonl"] in text and P.V2_BASE_RUN["config.json"] in text


def test_v2_mode_regenerates_headline_300_byte_for_byte(tmp_path):
    cases = E.make_headline_sample(300, 11, generator_version=2)
    out = tmp_path / "cases.jsonl"
    write_jsonl(out, cases)
    assert _sha(out) == P.V2_BASE_RUN["cases.jsonl"]  # the hash frozen in protocol v2
    real = ROOT / "results" / "headline-300" / "cases.jsonl"
    if real.exists():
        assert out.read_bytes() == real.read_bytes()
        mine, theirs = out.read_text().splitlines(), real.read_text().splitlines()
        assert mine[::13] == theirs[::13]
    # version 3 differs from it in U2 cases only
    v3 = E.make_headline_sample(300, 11)
    changed = [a.mechanism for a, b in zip(cases, v3) if a.to_dict() != b.to_dict()]
    assert changed and set(changed) == {"U2"}
    assert [c.case_id for c in cases] == [c.case_id for c in v3]


# ---------------------------------------------------------------------------
# generator checks, on the v3 default
# ---------------------------------------------------------------------------

def _features(case: Case):
    from lens.flip.schema import EVENT_KINDS

    cnt = Counter((e.type, e.subtype) for e in case.events)
    return [cnt[k] for k in EVENT_KINDS]


def _centroid_accuracy(cases):
    by_class: dict[str, list[Case]] = {m: [] for m in MECHANISMS}
    for c in cases:
        by_class[c.mechanism].append(c)
    n = min(len(v) for v in by_class.values())
    train, test = [], []
    for m in MECHANISMS:
        train += by_class[m][: n // 2]
        test += by_class[m][n // 2: n]
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


@pytest.fixture(scope="module")
def v3_sample():
    cases: list[Case] = []
    for i, (length, dist, k) in enumerate(itertools.product((100, 400), DISTANCES, (2, 8))):
        cases += generate_cases(180, 5000 + i, Knobs(length=length, distance=dist, distractors=k), mix=UNIFORM)
    return cases


def test_type_count_classifier_is_at_most_030_in_v3(v3_sample):
    acc = _centroid_accuracy(v3_sample)
    print(f"\nclassifier accuracy (v3, mixed knobs): {acc:.3f}")
    assert acc <= 0.30
    fixed = generate_cases(1200, 6161, Knobs(length=100, distance="middle", distractors=2), mix=UNIFORM)
    acc2 = _centroid_accuracy(fixed)
    print(f"classifier accuracy (v3, length 100, 2 distractors): {acc2:.3f}")
    assert acc2 <= 0.30


def test_type_counts_match_across_mechanisms_for_each_u2_pair():
    for length, dist in itertools.product((100, 400), DISTANCES):
        knobs = Knobs(length=length, distance=dist, distractors=0)
        for seed in range(30):
            base = Counter((e.type, e.subtype) for e in generate_case(seed, "M0", knobs).events)
            for pair in U2_PAIRS:
                u2 = generate_case(seed, "U2", knobs, u2_pair=pair)
                assert Counter((e.type, e.subtype) for e in u2.events) == base, (length, dist, seed, pair)


def test_oracle_agrees_on_every_v3_case(v3_sample):
    bad = [c.case_id for c in v3_sample if oracle(c.events, c.query_period) != c.a_star]
    assert not bad, bad[:5]
    assert sum(c.mechanism == "U2" for c in v3_sample) > 300


def test_d_star_deletion_flips_the_oracle_on_rare_cases(v3_sample):
    checked = 0
    for c in v3_sample:
        if c.mechanism not in ("M1", "M2", "M3"):
            continue
        rest = [e for e in c.events if e.id not in set(c.d_star)]
        assert oracle(rest, c.query_period) != c.a_star, c.case_id
        checked += 1
    assert checked > 500


def test_deleting_d_star_leaves_no_cause_in_u2(v3_sample):
    for c in v3_sample:
        if c.mechanism != "U2":
            continue
        assert len(c.d_star) >= 2
        rest = [e for e in c.events if e.id not in set(c.d_star)]
        assert detect_patterns(rest, c.query_period) == [], c.case_id


# ---------------------------------------------------------------------------
# subset runs
# ---------------------------------------------------------------------------

def test_subset_init_u2_only_with_ids_that_cannot_collide(tmp_path):
    d = tmp_path / "u2"
    info = E.init_run(d, 30, 12, ("U2",), "u3")
    assert info["mechanisms"] == {"U2": 30} and info["lengths"] == {100: 15, 400: 15}
    cases = list(read_jsonl(d / "cases.jsonl"))
    assert [c.case_id for c in cases] == [f"u3-{i:04d}" for i in range(30)]
    full_ids = {c.case_id for c in E.make_headline_sample(300, 11)}
    assert not full_ids & {c.case_id for c in cases}
    assert Counter(_pair(c) for c in cases) == {"M1+M3": 15, "M2+M3": 15}
    for length in (100, 400):  # crossed with the length: about half each within a length
        by = Counter(_pair(c) for c in cases if c.knobs["length"] == length)
        assert by["M1+M3"] in (7, 8) and by["M2+M3"] in (7, 8)
    assert len({c.knobs["distance"] for c in cases}) >= 2
    cfg = json.loads((d / "config.json").read_text())
    assert cfg["sample"]["mechanisms"] == ["U2"] and cfg["sample"]["case_ids"] == "u3-NNNN"
    assert cfg["sample"]["generator_version"] == 3 and cfg["n"] == 30 and cfg["seed"] == 12
    # the seed alone makes a different sample than headline-300's U2 cases
    old = {c.seed for c in E.make_headline_sample(300, 11) if c.mechanism == "U2"}
    assert not old & {c.seed for c in cases}


def test_subset_init_needs_a_prefix_and_known_mechanisms(tmp_path):
    with pytest.raises(E.Refused, match="prefix"):
        E.init_run(tmp_path / "a", 30, 12, ("U2",), None)
    with pytest.raises(E.Refused, match="mechanisms"):
        E.init_run(tmp_path / "b", 30, 12, ("X9",), "x")
    assert not (tmp_path / "b").exists()


def test_full_sample_is_unchanged_by_the_new_arguments():
    a = E.make_headline_sample(60, 3)
    b = E.make_headline_sample(60, 3, mechanisms=None, prefix=None, generator_version=3)
    assert [c.to_dict() for c in a] == [c.to_dict() for c in b]
    assert all(c.case_id.startswith("h3-") for c in a)


def test_validity_build_makes_the_same_cases_as_the_subset_init(tmp_path):
    out = tmp_path / "val"
    rc = lens_main(["flip", "validity", "build", "--n", "30", "--seed", "12", "--mechanisms", "U2", "--prefix", "u3",
                    "--arms", "adaptive-high", "--out", str(out)])
    assert rc == 0
    E.init_run(tmp_path / "hl", 30, 12, ("U2",), "u3")
    assert (out / "cases.jsonl").read_bytes() == (tmp_path / "hl" / "cases.jsonl").read_bytes()
    reqs = batch._read_lines(out / "requests.jsonl")
    assert len(reqs) == 30 and {batch.arm_of(r["custom_id"]) for r in reqs} == {"adaptive-high"}
    assert reqs[0]["custom_id"] == "u3-0000__adaptive-high"
    real = ROOT / "results" / "headline-u2-v3" / "cases.jsonl"
    if real.exists():
        assert real.read_bytes() == (out / "cases.jsonl").read_bytes()
    # without --prefix the subset build refuses
    assert lens_main(["flip", "validity", "build", "--n", "30", "--seed", "12", "--mechanisms", "U2",
                      "--out", str(tmp_path / "nope")]) == 1


def test_precheck_dryrun_makes_one_count_call_per_case(tmp_path):
    out = tmp_path / "val"
    assert lens_main(["flip", "validity", "build", "--n", "30", "--seed", "12", "--mechanisms", "U2", "--prefix", "u3",
                      "--arms", "adaptive-high", "--out", str(out)]) == 0
    calls = []

    def runner(method, path, body=None, timeout=120):
        calls.append(path)
        return 0, json.dumps({"input_tokens": 4000})

    res = batch.run_dryrun_impl(out, None, runner)
    assert calls == ["/v1/messages/count_tokens"] * 30 and res["complete"]
    arm = res["arms"]["adaptive-high"]
    assert arm["n_requests"] == 30 and res["likely_usd"] == pytest.approx(30 * (4000 * 0.05 + 3000 * 0.25) / 1e6)
    assert res["upper_bound_usd"] == pytest.approx(30 * (4000 * 0.05 + 6000 * 0.25) / 1e6)


# ---------------------------------------------------------------------------
# merge report on fakes
# ---------------------------------------------------------------------------

def _finish(d: Path, seed: int = 1) -> None:
    cfg = json.loads((d / "config.json").read_text())
    cfg["embedder"] = {"kind": "hash", "model": "hash-bow-256"}
    cfg["bootstrap"]["reps"] = 100
    cfg["h2b_draws"] = 50
    (d / "config.json").write_text(json.dumps(cfg))
    eng = E.Engine(d, check_protocol=False)
    model = M.MockModel(eng.cases, M.v4_calibration(), seed=seed)
    eng.runner = M.FakeBatchAPI(model.answer, M.engine_meta_fn(d))
    for _ in range(600):
        if eng.step(1e9) == "complete":
            break
    else:
        raise AssertionError("did not finish")
    eng.close()
    HM.write_report(d, 2)


@pytest.fixture(scope="module")
def fake_pair(tmp_path_factory):
    root = tmp_path_factory.mktemp("merge")
    base, u2 = root / "base", root / "u2"
    E.init_run(base, 24, 5)
    E.init_run(u2, 4, 6, ("U2",), "u3")
    _finish(base)
    _finish(u2, seed=2)
    return root, base, u2


NAMES = ("cases.jsonl", "config.json", "decisions.jsonl", "records.jsonl", "residue.jsonl", "residue_records.jsonl",
         "summary_v2.json", "tables_v2.txt", "spend.json")


def test_merge_report_keeps_non_u2_from_base_and_all_of_the_u2_run(fake_pair):
    root, base, u2 = fake_pair
    before = {d.name: {n: _sha(d / n) for n in NAMES} for d in (base, u2)}
    out = root / "merged"
    s = MG.merge_report(base, u2, "U2", out, 3)
    base_cases = list(read_jsonl(base / "cases.jsonl"))
    n_base_u2 = sum(c.mechanism == "U2" for c in base_cases)
    assert n_base_u2 >= 2
    n = len(base_cases) - n_base_u2 + 4
    assert s["protocol"] == 3 and s["n_cases"] == n and s["config"]["n"] == n and s["complete"]
    mech = Counter(c.mechanism for c in base_cases)
    assert s["mechanism_counts"] == {**{m: mech[m] for m in MECHANISMS}, "U2": 4}
    merged = list(read_jsonl(out / "cases.jsonl"))
    assert [c.case_id for c in merged] == ([c.case_id for c in base_cases if c.mechanism != "U2"]
                                           + [c.case_id for c in read_jsonl(u2 / "cases.jsonl")])
    assert not {c.case_id for c in base_cases if c.mechanism == "U2"} & {c.case_id for c in merged}
    # files, tables and the validity gate on the combined cases
    assert (out / "summary_v3.json").exists() and (out / "tables_v3.txt").exists() and (out / "merge.json").exists()
    text = (out / "tables_v3.txt").read_text()
    assert text.startswith(f"protocol 3. cases {n}") and "merge:" in text
    assert s["validity"]["gate"] is not None and s["validity"]["per_mechanism"]["U2"]["n"] == 4
    on_disk = json.loads((out / "summary_v3.json").read_text())
    assert on_disk["n_cases"] == n and on_disk["merge"]["base"]["replaced"] == n_base_u2
    assert on_disk["merge"]["with"]["n"] == 4 and on_disk["merge"]["combined"]["n"] == n
    # the bootstrap runs over the combined cases: every cell counts only combined case ids
    ids = {c.case_id for c in merged}
    dec = [r for r in HM.collect(base)["decisions"] if r["case_id"] in ids] + HM.collect(u2)["decisions"]
    for name, cell in s["cells"].items():
        system, budget = name.split("@")
        assert cell["n"] == sum(r["system"] == system and r["budget"] == int(budget) for r in dec)
    # base and U2 run files are untouched
    assert {d.name: {n_: _sha(d / n_) for n_ in NAMES} for d in (base, u2)} == before
    assert not list(base.glob("*_v3*")) and not list(u2.glob("*_v3*"))


def test_merge_report_is_deterministic_and_matches_a_direct_summary(fake_pair):
    root, base, u2 = fake_pair
    a, b = root / "m1", root / "m2"
    MG.merge_report(base, u2, "U2", a, 3)
    MG.merge_report(base, u2, "U2", b, 3)
    for f in ("summary_v3.json", "tables_v3.txt", "cases.jsonl", "merge.json"):
        assert (a / f).read_bytes() == (b / f).read_bytes()
    data, _ = MG.merge_data(base, u2, "U2")
    direct = HM.summarise(data, None, 3)
    on_disk = json.loads((a / "summary_v3.json").read_text())
    assert json.loads(json.dumps(direct["cells"])) == on_disk["cells"]
    assert json.loads(json.dumps(direct["residue"])) == on_disk["residue"]


def test_merge_report_cli(fake_pair, capsys):
    root, base, u2 = fake_pair
    out = root / "cli"
    rc = lens_main(["flip", "headline", "merge-report", "--base", str(base), "--replace-mechanism", "U2",
                    "--with", str(u2), "--protocol", "3", "--out", str(out)])
    assert rc == 0 and (out / "summary_v3.json").exists()
    assert "protocol 3. cases" in capsys.readouterr().out


def test_merge_report_refusals(fake_pair, tmp_path):
    root, base, u2 = fake_pair
    with pytest.raises(MG.MergeRefused, match="same directory"):
        MG.merge_report(base, base, "U2", tmp_path / "x")
    with pytest.raises(MG.MergeRefused, match="not complete"):
        unfinished = tmp_path / "unfinished"
        E.init_run(unfinished, 4, 7, ("U2",), "u4")
        MG.merge_report(base, unfinished, "U2", tmp_path / "x")
    with pytest.raises(MG.MergeRefused, match="not M1"):
        MG.merge_report(base, u2, "M1", tmp_path / "x")
    with pytest.raises(MG.MergeRefused, match="must be a new directory"):
        MG.merge_report(base, u2, "U2", base)
    with pytest.raises(MG.MergeRefused, match="unknown mechanism"):
        MG.merge_report(base, u2, "Z", tmp_path / "x")
    assert lens_main(["flip", "headline", "merge-report", "--base", str(base), "--replace-mechanism", "M1",
                      "--with", str(u2), "--out", str(tmp_path / "y")]) == 1


def test_merge_report_refuses_changed_config_and_id_clash(fake_pair, tmp_path):
    root, base, u2 = fake_pair
    odd = tmp_path / "odd"
    for f in ("cases.jsonl", "config.json"):
        odd.mkdir(exist_ok=True)
        (odd / f).write_bytes((u2 / f).read_bytes())
    import shutil
    shutil.copytree(u2 / "state", odd / "state")
    cfg = json.loads((odd / "config.json").read_text())
    cfg["max_calls"] = 12
    (odd / "config.json").write_text(json.dumps(cfg))
    with pytest.raises(MG.MergeRefused, match="max_calls"):
        MG.merge_report(base, odd, "U2", tmp_path / "x")
    # a replacement run whose case ids reuse a kept base id
    clash = tmp_path / "clash"
    clash.mkdir()
    base_non_u2 = next(c for c in read_jsonl(base / "cases.jsonl") if c.mechanism != "U2")
    rows = list(read_jsonl(u2 / "cases.jsonl"))
    rows[0].case_id = base_non_u2.case_id
    write_jsonl(clash / "cases.jsonl", rows)
    (clash / "config.json").write_bytes((u2 / "config.json").read_bytes())
    with pytest.raises(MG.MergeRefused, match="collide"):
        MG.merge_report(base, clash, "U2", tmp_path / "x")


def test_report_cli_accepts_protocol_three_and_writes_v3_files(fake_pair):
    root, base, u2 = fake_pair
    s = HM.write_report(u2, 3)
    assert s["protocol"] == 3 and (u2 / "summary_v3.json").exists() and (u2 / "tables_v3.txt").exists()
    assert "protocol 3. cases 4" in (u2 / "tables_v3.txt").read_text()


# ---------------------------------------------------------------------------
# the protocol file
# ---------------------------------------------------------------------------

def test_v3_protocol_file_is_a_draft_with_the_changes_and_the_hashes(tmp_path):
    d = tmp_path / "u2"
    E.init_run(d, 30, 12, ("U2",), "u3")
    doc = tmp_path / "flip-protocol-v3.md"
    rc = lens_main(["flip", "headline", "freeze", "--dir", str(d), "--doc", str(doc)])
    assert rc == 0
    text = doc.read_text()
    assert text.startswith("# Decision Flip Test: protocol v3") and "Version: 3" in text and "Status: Draft" in text
    assert "Frozen" not in text.split("## Changes")[0]
    changes = text.split("## Changes from v2")[1].split("## Changes from v1")[0]
    for needle in ("21 of 30", "4 of 13", "11 of 11", "6 of 6", "M1+M3 and M2+M3", "M1+M2 is never planted",
                   "contested", "at least 24 of 30", "Otherwise stop and report", "merge-report", "headline-300-v3",
                   "summary_v3.json", "tables_v3.txt", "Base files stay untouched", "identical to v2",
                   "declared before any v3 model call"):
        assert needle in changes, needle
    pj = json.loads((d / "protocol.json").read_text())
    assert pj["version"] == 3 and pj["status"] == "draft" and pj["n"] == 30 and pj["seed"] == 12
    for v in P.V2_HASHES.values():
        assert v in changes  # v2's hashes repeated
    assert pj["hashes"]["file:generator.py"] in changes and pj["hashes"]["file:generator.py"] != P.V2_HASHES["file:generator.py"]
    rows = [ln for ln in changes.splitlines() if ln.startswith("| ") and "sha256" not in ln and "---" not in ln]
    assert len(rows) == len(P.V2_HASHES) and [ln for ln in rows if ln.endswith("| NO |")] == [
        ln for ln in rows if ln.startswith("| file:generator.py")]
    assert "Mechanisms: U2 only" in text and "results/u2 | 30 | 12 |" in text
    # an explicit --protocol-version still wins, and v2 rendering is unchanged by the new code
    P.freeze([d], tmp_path / "v2.md", now=0, version=2)
    assert "Changes from v2" not in (tmp_path / "v2.md").read_text()


def test_v3_protocol_text_and_new_sources_obey_the_word_rules(tmp_path):
    if not FORBIDDEN:
        pytest.skip("banned_words.local.txt not present")
    d = tmp_path / "u2"
    E.init_run(d, 30, 12, ("U2",), "u3")
    doc = tmp_path / "flip-protocol-v3.md"
    P.freeze([d], doc, now=0, version=3)
    text = doc.read_text()
    blob = (text + (ROOT / "src/lens/flip/headline/merge.py").read_text() + Path(__file__).read_text()).lower()
    assert [w for w in FORBIDDEN if w in blob] == []
    assert [w for w in BANNED_STYLE if w in text.lower()] == []
    assert "—" not in text and ", and " not in text.split("## Changes from v2")[1].split("## Changes from v1")[0]
    sentences = [s for s in re.split(r"(?<=[.])\s+", text.split("## Changes from v2")[1].split("| Item")[0]) if s.strip()]
    assert max(len(s.split()) for s in sentences) <= 60  # long lines are the enumerations, not prose


def test_summary_carries_a_schema_version(fake_pair):
    root, base, u2 = fake_pair
    out = root / "schema"
    MG.merge_report(base, u2, "U2", out, 3)
    on_disk = json.loads((out / "summary_v3.json").read_text())
    assert on_disk["schema"] == HM.SUMMARY_SCHEMA == "lens.flip.summary/1"

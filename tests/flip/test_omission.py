"""Exploratory omission comparator (design 7.3): sampler, vault file, key, label parser, thresholds, judge rule,
the runner and its stop on a failed calibration. No network: the embedder and the judge are fakes, and the
stop test makes urllib and subprocess fail so any outside call would show.

The stored-data check skips when results/ is absent (it is git-ignored)."""

from __future__ import annotations

import hashlib
import json
import random
import re
import subprocess
import urllib.request
from pathlib import Path

import pytest

from lens import main as lens_main
from lens.flip.omission import SEED, SUMMARY_JUDGE_NAME, SUMMARY_NAME
from lens.flip.omission import calibrate as cal
from lens.flip.omission import labels as L
from lens.flip.omission import sample as S
from lens.flip.omission.coverage import coverage, number_overlap, salient_tokens
from lens.flip.omission.frame import SYSTEMS, Frame, NoteRec, UnitRec
from lens.flip.omission.judge import haiku_plan, parse_answer
from lens.flip.omission.runner import RunRefused, run

ROOT = Path(__file__).resolve().parents[2]
VOCAB = ["device", "statement", "reading", "credit", "payment", "plan", "notice", "contact", "technician", "visit",
         "charge", "balance", "request", "account", "posted", "approved", "estimated", "actual", "review", "phone"]


# ---------------------------------------------------------------------------
# A synthetic frame and fakes
# ---------------------------------------------------------------------------

def fake_embedder(texts):
    """Bag-of-words hash vectors: texts that share words have a high cosine."""
    out = []
    for t in texts:
        v = [0.0] * 48
        for w in re.findall(r"[a-z]+", t.lower()):
            h = int(hashlib.sha256(w.encode()).hexdigest(), 16)
            v[h % 48] += 1.0
        out.append(v)
    return out


def make_frame(n_per_system: int = 30, units: int = 18, seed: int = 5, reps: int = 60) -> Frame:
    rng = random.Random(seed)
    notes: list[NoteRec] = []
    case_ids: list[str] = []
    for system in SYSTEMS:
        for i in range(n_per_system):
            cid = f"{system[:3]}-{i:03d}"
            case_ids.append(cid)
            us = []
            for k in range(units):
                words = " ".join(rng.sample(VOCAB, 4))
                amt = f"{rng.randint(100, 900)}.{rng.randint(10, 99)}"
                us.append(UnitRec(k, f"e{k:04d}" if system != "summary" else f"s{k:03d}",
                                  f"e{k:04d} 2020-{rng.randint(1, 12):02d}-{rng.randint(10, 28)} {words} {amt}.",
                                  dstar=(system != "summary" and k % 9 == 0)))
            sents = []
            for u in rng.sample(us, 4):
                if rng.random() < 0.5:
                    sents.append(f"Seen {u.text.split()[-1]} on the account.")      # number overlap
                else:
                    sents.append("The " + " ".join(u.text.split()[2:6]) + " matter is noted.")   # paraphrase
            sents.append("Recommend a review.")
            notes.append(NoteRec(cid, system, " ".join(sents), round(rng.random(), 3), rng.random() < 0.3,
                                 1.0, us))
    return Frame(notes=notes, case_ids=case_ids, bootstrap={"reps": reps, "seed": 11},
                 power={"H2c_min_positives": 20, "H2c_min_negatives": 20}, inputs={"toy": "0" * 64})


@pytest.fixture(scope="module")
def frame() -> Frame:
    return make_frame()


@pytest.fixture(scope="module")
def cosines(frame):
    return coverage(frame, fake_embedder)


@pytest.fixture(scope="module")
def drawn(frame, cosines):
    pairs = S.classify(frame, cosines)
    rows, _ = S.draw_sample(frame, pairs, SEED)
    return pairs, rows


def truth_labels(keys):
    """Labels from the key by cosine, with a few ? rows, as a labelled-pairs list for the runner."""
    out = []
    for i, k in enumerate(keys):
        noise = (int(hashlib.sha256(k["unit_label"].encode()).hexdigest(), 16) % 1000) / 1000 * 0.2 - 0.1
        lab = "?" if i % 25 == 0 else ("y" if k["cosine"] + noise >= 0.35 else "n")
        out.append({"unit_label": k["unit_label"], "note_label": k["note_label"], "case_id": k["case_id"],
                    "system": k["system"], "unit_id": k["unit_id"], "unit_index": k["unit_index"], "label": lab})
    return out


# ---------------------------------------------------------------------------
# Overlap signal
# ---------------------------------------------------------------------------

def test_salient_tokens_skip_ids_zero_and_small_numbers():
    t = salient_tokens("e0072 2013-01-15 Statement for period 2013-01: charges 144.12, other 0.00, 7 visits cr0001.")
    assert t == {"2013-01-15", "2013-01", "144.12"}
    assert number_overlap("e0001 2020-03-04 Paid 52.10.", "A payment of 52.10 posted.")
    assert not number_overlap("e0001 2020-03-04 Paid 52.10.", "A payment of 52.11 posted.")


# ---------------------------------------------------------------------------
# Sampler
# ---------------------------------------------------------------------------

def test_sampler_gives_200_pairs_50_per_system(drawn):
    _, rows = drawn
    assert len(rows) == 200
    keys = S.key_rows(rows)
    by_sys = {s: sum(k["system"] == s for k in keys) for s in SYSTEMS}
    assert by_sys == {s: 50 for s in SYSTEMS}
    assert len({k["unit_label"] for k in keys}) == 200
    assert len({(k["case_id"], k["system"], k["unit_index"]) for k in keys}) == 200


def test_sampler_strata_are_about_a_third_each(drawn):
    _, rows = drawn
    keys = S.key_rows(rows)
    for s in SYSTEMS:
        for st in S.STRATA:
            n = sum(k["system"] == s and k["stratum"] == st for k in keys)
            assert 14 <= n <= 20, (s, st, n)


def test_sampler_is_deterministic(frame, cosines):
    a = S.key_rows(S.draw_sample(frame, S.classify(frame, cosines), SEED)[0])
    b = S.key_rows(S.draw_sample(frame, S.classify(frame, cosines), SEED)[0])
    c = S.key_rows(S.draw_sample(frame, S.classify(frame, cosines), SEED + 1)[0])
    assert a == b
    assert a != c


def test_sampler_dstar_units_marked_and_capped(drawn):
    _, rows = drawn
    keys = S.key_rows(rows)
    for s in SYSTEMS:
        d = sum(k["dstar"] for k in keys if k["system"] == s)
        assert d == (0 if s == "summary" else S.DSTAR_CAP)


def test_sampler_clusters_notes(drawn):
    _, rows = drawn
    per_sys = {s: {r.note_label for r in rows if r.pair.system == s} for s in SYSTEMS}
    for s, notes in per_sys.items():
        assert len(notes) <= S.NOTES_PER_SYSTEM + S.DSTAR_CAP
    assert all(per_sys[a].isdisjoint(per_sys[b]) for a in SYSTEMS for b in SYSTEMS if a < b)


def test_quotas_sum_to_50_and_rotate():
    qs = [S.quotas(i) for i in range(4)]
    assert all(sum(q.values()) == 50 for q in qs)
    assert len({tuple(q.values()) for q in qs}) > 1


# ---------------------------------------------------------------------------
# Vault file and key
# ---------------------------------------------------------------------------

def test_vault_file_hides_system_stratum_case_and_scores(frame, drawn):
    _, rows = drawn
    md = S.vault_markdown(frame, rows)
    keys = S.key_rows(rows)
    lines = md.splitlines()
    unit_lines = [ln for ln in lines if re.match(r"^- \[u\d{3}\] label:", ln)]
    assert len(unit_lines) == 200
    head = "\n".join(lines[:15])
    assert "does the note convey this record's content well enough that a reader would know it?" in head
    assert "—" not in md.split("### N1")[0]
    note_texts = {frame.notes[r.pair.note_idx].note for r in rows}
    stripped = md
    for t in note_texts:
        stripped = stripped.replace(t, "")
    for u in (k["unit_text"] for k in keys):
        stripped = stripped.replace(u, "")
    for word in (*SYSTEMS, *S.STRATA, "rag", "dstar", "D*", "cosine", "stratum", "overlap"):
        assert word not in stripped, word
    assert not any(k["case_id"] in md for k in keys)
    assert not re.search(r"0\.\d{4}", stripped)         # no score
    # every note once, in a section named N<k>
    heads = re.findall(r"^### (N\d+)$", md, re.M)
    assert len(heads) == len(set(heads)) == len({k["note_label"] for k in keys})


def test_key_matches_vault(frame, drawn):
    _, rows = drawn
    md = S.vault_markdown(frame, rows)
    keys = S.key_rows(rows)
    section = None
    seen = {}
    for ln in md.splitlines():
        m = re.match(r"^### (N\d+)$", ln)
        if m:
            section = m.group(1)
        m = re.match(r"^- \[(u\d{3})\] label:\s*\| (.*)$", ln)
        if m:
            seen[m.group(1)] = (section, m.group(2))
    assert len(seen) == 200
    for k in keys:
        assert seen[k["unit_label"]] == (k["note_label"], k["unit_text"])
    for k in keys:
        assert {"unit_label", "note_label", "case_id", "system", "stratum", "cosine", "overlap", "dstar",
                "unit_id"} <= set(k)


def test_sample_stratum_matches_signals(drawn):
    _, rows = drawn
    for r in rows:
        assert (r.pair.stratum == "verbatim") == r.pair.overlap


# ---------------------------------------------------------------------------
# Parser
# ---------------------------------------------------------------------------

KEY = [{"unit_label": f"u{i:03d}", "note_label": "N1", "case_id": "c1", "system": "recent", "unit_id": f"e{i:04d}",
        "unit_index": i, "unit_text": f"text {i}"} for i in range(1, 9)]

FILLED = """# Title
Some text.

### N1

> a note

- [u001] label: y | text 1
- [u002] label: N | text 2
- [u003] label: ?| text 3
- [u004] label:   | text 4
- [u005] label: yes | text 5
- [u006] label: n text 6
- [u007] label: Y | text 7
- [u007] label: n | text 7 again
- [u099] label: y | no such unit
- [u008] label: n | text 8
- [u1] not a record line
"""


def test_parser_handles_y_n_question_blank_and_malformed():
    res = L.parse_text(FILLED, KEY)
    by = {r["unit_label"]: r["label"] for r in res["labels"]}
    assert by == {"u001": "y", "u002": "n", "u003": "?", "u007": "y", "u008": "n"}
    assert [r["unit_label"] for r in res["unlabelled"]] == ["u004"]
    reasons = {r.get("unit_label"): r["reason"] for r in res["invalid"]}
    assert "is not y, n or ?" in reasons["u005"]
    assert "missing" in reasons["u006"]
    assert reasons["u007"] == "unit id appears twice"
    assert reasons["u099"] == "unit id not in the key"
    assert res["missing"] == []
    assert any(r["reason"].startswith("record line without") for r in res["invalid"])
    c = res["counts"]
    assert (c["y"], c["n"], c["?"], c["labelled"]) == (2, 2, 1, 5)


def test_parser_never_edits_the_file_and_writes_labels(tmp_path, capsys):
    md = tmp_path / "filled.md"
    md.write_text(FILLED, encoding="utf-8")
    key = tmp_path / "key.jsonl"
    S.write_key(key, KEY)
    before = hashlib.sha256(md.read_bytes()).hexdigest()
    mtime = md.stat().st_mtime_ns
    rc = lens_main(["flip", "omission", "parse-labels", "--file", str(md), "--key", str(key)])
    assert rc == 1                                   # problems found
    assert hashlib.sha256(md.read_bytes()).hexdigest() == before
    assert md.stat().st_mtime_ns == mtime
    out = tmp_path / "labels.jsonl"
    rows = [json.loads(ln) for ln in out.read_text().splitlines()]
    assert {r["unit_label"]: r["label"] for r in rows} == {"u001": "y", "u002": "n", "u003": "?", "u007": "y",
                                                           "u008": "n"}
    text = capsys.readouterr().out
    assert "unlabelled 1" in text and "invalid 5" in text


def test_parser_clean_file_returns_zero(tmp_path):
    key = tmp_path / "key.jsonl"
    S.write_key(key, KEY)
    md = tmp_path / "f.md"
    md.write_text("\n".join(f"- [{k['unit_label']}] label: {'y' if i % 2 else 'N'} | {k['unit_text']}"
                            for i, k in enumerate(KEY)), encoding="utf-8")
    assert lens_main(["flip", "omission", "parse-labels", "--file", str(md), "--key", str(key)]) == 0
    assert len(L.read_labels(tmp_path / "labels.jsonl")) == 8


def test_parser_roundtrip_on_generated_file(frame, drawn, tmp_path):
    _, rows = drawn
    keys = S.key_rows(rows)
    md = S.vault_markdown(frame, rows)
    res = L.parse_text(md, keys)
    assert res["counts"]["unlabelled"] == 200 and res["counts"]["invalid"] == 0 and res["counts"]["missing"] == 0
    filled = re.sub(r"label:   \|", "label: y |", md)
    assert L.parse_text(filled, keys)["counts"]["y"] == 200


# ---------------------------------------------------------------------------
# Thresholds
# ---------------------------------------------------------------------------

def test_kappa_and_metrics_known_values():
    assert cal.kappa([True, True, False, False], [True, True, False, False]) == 1.0
    assert cal.kappa([True, False, True, False], [True, True, False, False]) == 0.0
    m = cal.metrics([True, True, False, False, True], [True, False, False, True, True])
    assert (m["tp"], m["fp"], m["fn"], m["tn"]) == (2, 1, 1, 1)
    assert m["agreement"] == 0.6 and m["precision"] == pytest.approx(2 / 3) and m["recall"] == pytest.approx(2 / 3)


def toy_items():
    # cosines 0.0 to 0.99; conveyed above 0.5, except two noisy points
    items = [(i / 100, i >= 50) for i in range(100)]
    items[55] = (0.55, False)
    items[45] = (0.45, True)
    return items


def test_threshold_choice_on_toy_data():
    items = toy_items()
    t = cal.choose_tau_kappa(items)
    assert 0.46 <= t["tau"] <= 0.56
    assert t["kappa"] > 0.9 and t["agreement"] >= 0.96
    band = cal.choose_band(items)
    assert band["tau_high"] is not None and band["tau_low"] is not None
    assert band["tau_low"] < band["tau_high"]
    hi = [y for c, y in items if c >= band["tau_high"]]
    lo = [y for c, y in items if c <= band["tau_low"]]
    assert sum(hi) / len(hi) >= 0.9
    assert (len(lo) - sum(lo)) / len(lo) >= 0.9
    # the lowest qualifying tau_high and the highest qualifying tau_low: just one step further breaks it
    assert band["high_support"]["n"] >= cal.MIN_SUPPORT and band["low_support"]["n"] >= cal.MIN_SUPPORT


def test_threshold_choice_no_side_when_labels_are_mixed():
    items = [(i / 40, i % 2 == 0) for i in range(40)]          # cosine says nothing
    band = cal.choose_band(items)
    assert band["tau_high"] is None and band["tau_low"] is None
    assert cal.in_band(0.3, None, None)
    assert cal.choose_tau_kappa([])["tau"] is None


def test_in_band():
    assert not cal.in_band(0.9, 0.4, 0.8)
    assert not cal.in_band(0.3, 0.4, 0.8)
    assert cal.in_band(0.6, 0.4, 0.8)
    assert not cal.in_band(0.4, 0.4, 0.8)           # at tau_low is decided


# ---------------------------------------------------------------------------
# Judge pass rule
# ---------------------------------------------------------------------------

def test_judge_pass_fail_rule():
    truth = [True] * 15 + [False] * 15
    perfect = list(truth)
    assert cal.judge_passes(perfect, truth)["passed"]
    # 5 wrong of 30: accuracy 0.833, kappa 0.667 -> pass
    five = [not t if i in (0, 1, 2, 20, 21) else t for i, t in enumerate(truth)]
    r = cal.judge_passes(five, truth)
    assert r["passed"] and r["accuracy"] == pytest.approx(25 / 30) and r["kappa"] >= 0.6
    # 7 wrong: accuracy 0.767 -> fail on accuracy
    seven = [not t if i in (0, 1, 2, 3, 20, 21, 22) else t for i, t in enumerate(truth)]
    r = cal.judge_passes(seven, truth)
    assert not r["passed"] and any("accuracy" in x for x in r["reasons"])
    # always conveyed: accuracy 0.5, kappa 0
    r = cal.judge_passes([True] * 30, truth)
    assert not r["passed"] and r["kappa"] == 0.0
    # kappa under 0.6 with accuracy over 80% (skewed labels)
    t2 = [True] * 27 + [False] * 3
    p2 = [True] * 27 + [True, True, False]
    r = cal.judge_passes(p2, t2)
    assert r["accuracy"] >= 0.8 and r["kappa"] < 0.6 and not r["passed"]
    # too few labelled pairs
    assert not cal.judge_passes([True] * 10, [True] * 10)["passed"]
    # a parse failure counts as wrong
    r = cal.judge_passes([None] + perfect[1:], truth)
    assert r["parse_failures"] == 1 and r["accuracy"] == pytest.approx(29 / 30)


def test_judge_answer_parser():
    assert parse_answer('{"conveyed": true}') is True
    assert parse_answer('{"conveyed": false}') is False
    assert parse_answer('<think>x</think>{"conveyed": false}') is False
    assert parse_answer('{"conveyed": "yes"}') is None
    assert parse_answer("The note conveys it.") is None


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

def keys_and_labels(frame, cosines):
    pairs = S.classify(frame, cosines)
    rows, _ = S.draw_sample(frame, pairs, SEED)
    keys = S.key_rows(rows)
    return keys, truth_labels(keys)


class CountingEmbedder:
    def __init__(self):
        self.calls = 0

    def __call__(self, texts):
        self.calls += 1
        return fake_embedder(texts)


def test_runner_embedding_mode_default_calls_no_judge(frame, cosines, tmp_path):
    keys, labels = keys_and_labels(frame, cosines)

    def boom(note, unit):
        raise AssertionError("the judge must not be called in embedding mode")

    s = run(frame, labels, tmp_path / "out", embedder=fake_embedder, judge=boom)
    assert s["schema"] == "lens.flip.omission/1" and s["exploratory"] is True and s["mode"] == "embedding"
    assert s["status"] == "complete"
    out = tmp_path / "out"
    assert (out / SUMMARY_NAME).exists() and not (out / "judge_calibration.json").exists()
    disk = json.loads((out / SUMMARY_NAME).read_text())
    prov = disk["provenance"]
    assert prov["models"] == {"embedder": "nomic-embed-text", "judge": None}
    assert prov["seeds"]["sample"] == 21 and prov["bootstrap"]["reps"] == 60
    assert prov["inputs_sha256"]["toy"] == "0" * 64
    e = disk["calibration"]["embedding_only"]
    assert e["tau"] is not None and {"agreement", "precision", "recall", "kappa"} <= set(e)
    assert disk["calibration"]["labels"]["?"] == 8                         # i % 25 == 0 over 200 rows
    assert "tau_low" in disk["calibration"]["precision_band_context"]
    pooled = disk["h2c"]["pooled"]
    assert pooled["n_eligible"] == len(frame.notes)
    for key in ("auroc_p_residue", "auroc_omission_nv", "auroc_omission_frozen", "diff_p_residue_vs_omission_nv"):
        assert set(pooled[key]) == {"value", "ci95"}
    assert set(disk["h2c"]["by_system"]) == set(SYSTEMS)
    rows = [json.loads(ln) for ln in (out / "notes_omission.jsonl").read_text().splitlines()]
    assert len(rows) == len(frame.notes)
    assert all(0.0 <= r["omission_nv"] <= 1.0 for r in rows)
    # the frozen comparator is 1.0 everywhere here; the new one is not constant
    assert len({r["omission_nv"] for r in rows}) > 3
    assert len((out / "coverage.jsonl").read_text().splitlines()) == frame.n_pairs


def test_runner_matches_hand_computed_omission(frame, cosines, tmp_path):
    keys, labels = keys_and_labels(frame, cosines)
    s = run(frame, labels, tmp_path / "o", embedder=fake_embedder)
    tau = s["calibration"]["embedding_only"]["tau"]
    rows = {(r["case_id"], r["system"]): r
            for r in map(json.loads, (tmp_path / "o" / "notes_omission.jsonl").read_text().splitlines())}
    for note, row in list(zip(frame.notes, cosines))[:10]:
        want = sum(c < tau for c in row) / len(row)
        assert rows[(note.case_id, note.system)]["omission_nv"] == pytest.approx(want, abs=1e-6)


def test_runner_refuses_without_both_label_classes(frame, cosines, tmp_path):
    _, labels = keys_and_labels(frame, cosines)
    only_y = [{**r, "label": "y"} for r in labels]
    with pytest.raises(RunRefused):
        run(frame, only_y, tmp_path / "o", embedder=fake_embedder)


def test_runner_judge_mode_stops_on_failed_calibration_and_writes_haiku_plan(frame, cosines, tmp_path, monkeypatch):
    keys, labels = keys_and_labels(frame, cosines)

    def no_network(*a, **k):
        raise AssertionError("an outside call was attempted")

    monkeypatch.setattr(urllib.request, "urlopen", no_network)
    monkeypatch.setattr(subprocess, "run", no_network)
    monkeypatch.setattr(subprocess, "Popen", no_network)
    calls = []

    def bad_judge(note, unit):
        calls.append(unit)
        return True                        # always conveyed: fails the rule

    out = tmp_path / "o"
    s = run(frame, labels, out, embedder=fake_embedder, mode="embedding+judge", judge=bad_judge)
    assert s["status"] == "stopped_judge_failed_calibration"
    jc = json.loads((out / "judge_calibration.json").read_text())
    assert jc["passed"] is False and jc["reasons"]
    plan = jc["haiku_plan"]
    assert plan["submitted"] is False and plan["api_calls"] == 0 and plan["count_tokens_calls"] == 0
    assert plan["n_pairs"] == jc["n_borderline_pairs_all"] > 0
    assert plan["rates_usd_per_million"] == {"input": 0.05, "output": 0.25}
    assert plan["chars_per_token"] == 2.2 and plan["output_tokens"] == 10 * plan["n_pairs"]
    expect = plan["input_tokens"] * 0.05 / 1e6 + plan["output_tokens"] * 0.25 / 1e6
    assert plan["estimated_usd"] == pytest.approx(expect, abs=1e-5)
    # the judge saw only the labelled borderline pairs, not the whole frame
    assert cal.JUDGE_MIN_LABELLED <= len(calls) < frame.n_pairs / 10
    disk = json.loads((out / SUMMARY_JUDGE_NAME).read_text())
    assert disk["h2c"] is None and disk["judge"]["stopped"] is True
    assert not (out / "judge_decisions.jsonl").exists() and not (out / "notes_omission.jsonl").exists()
    assert not (out / SUMMARY_NAME).exists()


def test_runner_judge_mode_passes_with_a_good_fake_judge(frame, cosines, tmp_path):
    keys, labels = keys_and_labels(frame, cosines)
    cos_of = {(n.note, u.text): c for n, row in zip(frame.notes, cosines) for u, c in zip(n.units, row)}
    note_of = {(n.case_id, n.system): n for n in frame.notes}
    said = {}
    for lab in labels:
        n = note_of[(lab["case_id"], lab["system"])]
        if lab["label"] != "?":
            said[(n.note, n.units[lab["unit_index"]].text)] = lab["label"] == "y"

    def good_judge(note, unit):
        return said.get((note, unit), cos_of[(note, unit)] >= 0.35)      # agrees with every label

    out = tmp_path / "o"
    s = run(frame, labels, out, embedder=fake_embedder, mode="embedding+judge", judge=good_judge)
    assert s["status"] == "complete" and s["mode"] == "embedding+judge"
    jc = json.loads((out / "judge_calibration.json").read_text())
    assert jc["passed"] is True and jc["haiku_plan"] is None
    assert (out / "judge_decisions.jsonl").exists() and (out / SUMMARY_JUDGE_NAME).exists()
    assert not (out / SUMMARY_NAME).exists()
    assert s["h2c"]["pooled"]["n_eligible"] == len(frame.notes)
    assert s["provenance"]["models"]["judge"] == "qwen3:30b-a3b"


def test_runner_judge_mode_needs_a_judge(frame, cosines, tmp_path):
    _, labels = keys_and_labels(frame, cosines)
    with pytest.raises(RunRefused):
        run(frame, labels, tmp_path / "o", embedder=fake_embedder, mode="embedding+judge")


def test_haiku_plan_is_arithmetic_only():
    plan = haiku_plan([("a note " * 10, "a unit"), ("b note", "b unit")])
    assert plan["n_pairs"] == 2 and plan["submitted"] is False
    assert plan["estimated_usd"] > 0


# ---------------------------------------------------------------------------
# summary_v3.json stays untouched
# ---------------------------------------------------------------------------

def test_summary_v3_untouched(frame, cosines, tmp_path):
    _, labels = keys_and_labels(frame, cosines)
    v3 = tmp_path / "v3" / "summary_v3.json"
    v3.parent.mkdir()
    v3.write_text('{"keep": "me"}\n')
    sha = hashlib.sha256(v3.read_bytes()).hexdigest()
    f = make_frame()
    f.watch = {str(v3.resolve()): sha}
    run(f, labels, tmp_path / "out", embedder=fake_embedder)
    assert hashlib.sha256(v3.read_bytes()).hexdigest() == sha
    assert not (tmp_path / "out" / "summary_v3.json").exists()


def test_runner_refuses_a_v3_directory_and_catches_a_changed_v3_file(frame, cosines, tmp_path):
    _, labels = keys_and_labels(frame, cosines)
    d = tmp_path / "v3run"
    d.mkdir()
    (d / "summary_v3.json").write_text("{}\n")
    with pytest.raises(RunRefused):
        run(frame, labels, d, embedder=fake_embedder)
    f = make_frame()
    f.watch = {str((d / "summary_v3.json").resolve()): "0" * 64}
    with pytest.raises(RuntimeError):
        run(f, labels, tmp_path / "out2", embedder=fake_embedder)


def test_run_cli_defaults_to_embedding_mode():
    from lens.flip import cli as flip_cli
    import argparse

    p = argparse.ArgumentParser()
    sub = p.add_subparsers()
    flip_cli.register(sub)
    args = p.parse_args(["flip", "omission", "run", "--labels", "x.jsonl"])
    assert args.mode == "embedding"
    assert p.parse_args(["flip", "omission", "run", "--labels", "x", "--mode", "embedding+judge"]).mode == \
        "embedding+judge"


def test_run_cli_refuses_missing_labels(tmp_path, capsys):
    rc = lens_main(["flip", "omission", "run", "--labels", str(tmp_path / "none.jsonl"), "--out", str(tmp_path)])
    assert rc == 1
    assert "no labels file" in capsys.readouterr().err


def test_sample_cli_refuses_to_overwrite_the_vault_file(tmp_path, capsys):
    vault = tmp_path / "v.md"
    vault.write_text("filled in by hand\n")
    rc = lens_main(["flip", "omission", "sample", "--vault", str(vault), "--out", str(tmp_path / "o")])
    assert rc == 1 and vault.read_text() == "filled in by hand\n"


# ---------------------------------------------------------------------------
# Stored data (skips without results/)
# ---------------------------------------------------------------------------

def test_stored_frame_has_490_notes_and_11240_pairs():
    if not (ROOT / "results" / "headline-300-v3" / "merge.json").exists():
        pytest.skip("results/ is absent")
    from lens.flip.omission.frame import load_frame

    f = load_frame(ROOT)
    assert len(f.notes) == 490 and f.n_pairs == 11240 and len(f.case_ids) == 300
    counts = {s: (sum(n.system == s for n in f.notes), sum(len(n.units) for n in f.notes if n.system == s))
              for s in SYSTEMS}
    assert counts == {"recent": (98, 2771), "rag-bm25": (163, 4397), "rag-embed": (144, 2494),
                      "summary": (85, 1578)}
    assert f.v3_reference and f.v3_reference["n_eligible"] == 490
    assert any(u.dstar for n in f.notes for u in n.units)
    assert str(ROOT / "results/headline-300-v3/summary_v3.json") in f.watch

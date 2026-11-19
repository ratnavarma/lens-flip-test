"""C4 judge v2: anchors, the synthetic truth set, the paraphrase, the quote-checked parser, the metrics, the cache.
No network: the judge runs on a fake transport and the embedder is a fake. Checks on the real set skip when
results/ is absent (it is git-ignored)."""

from __future__ import annotations

import json
import random
from pathlib import Path

import pytest

from lens.flip.cache import CallCache
from lens.flip.explore5 import judge2 as J
from lens.flip.explore5.frame import OUT

NOTE = ("Stmt 2016-02 = 139.60 (paid 2016-03-10) vs about 80 typical. No reading, credit or device event is "
        "recorded for Feb 2016. Next actual reading is 403.6 (2016-11). Suggest OPEN_REVIEW of the charge.")


# ---------------------------------------------------------------------------
# Anchors
# ---------------------------------------------------------------------------

def test_mentions_numbers_ids_and_dates_in_any_form():
    assert J.mentions(NOTE, "e0122 2016-12-01 Actual reading for 2016-11: 403.6.")          # number
    assert J.mentions(NOTE, "e0077 2016-03-02 Statement 2016-02 sent.")                     # YYYY-MM
    assert J.mentions("The Feb 2016 charge was high.", "e0077 2016-02-03 Notice sent.")        # month name + year
    assert J.mentions("Device replaced 09-08, read later.", "e0382 2022-09-08 Device replaced.")  # MM-DD
    assert J.mentions("Credit CR0010 was applied.", "e0092 2012-04-01 Applied credit cr0010.")    # id, any case
    assert not J.mentions(NOTE, "e0052 2012-08-26 Technician serviced the device.")
    assert not J.mentions(NOTE, "e0010 2016-05-03 Statement for period 2016-04: activity charges 81.12.")


def test_month_name_needs_the_year_somewhere_in_the_note():
    assert not J.mentions("Jul/Aug are high months.", "e0001 2022-08-02 Statement for period 2022-07: due 144.72.")
    assert J.mentions("In 2022, Jul/Aug are high months.", "e0001 2022-08-02 Statement for period 2022-07: due 144.72.")


def test_natural_truth_rules():
    assert J.naturally_present(NOTE, "e0122 2016-12-01 Actual reading for 2016-11: 403.6.")
    assert not J.naturally_present(NOTE, "e0052 2012-08-26 Technician serviced the device.")    # no number
    # one of three distinguishing numbers is not half
    assert not J.naturally_present("Charge 229.18.", "e1 2018-02-03 Activity 229.18, fee 53.66, total 282.84.")
    assert J.strictly_absent(NOTE, "e0052 2012-08-26 Technician serviced the device.")
    assert not J.strictly_absent(NOTE, "e0077 2016-03-02 Statement 2016-02 sent.")
    assert not J.strictly_absent(NOTE, "No date or number here.")                               # nothing to check


def test_a_number_counts_only_in_a_sentence_about_the_same_kind_of_record():
    note = "Jan 2014 statement (138.42) matches the actual reading. Customer has paid on time."
    assert not J.naturally_present(note, "e0078 2014-02-22 Payment received: 138.42.")
    assert J.naturally_present(note, "e0076 2014-02-02 Statement 2014-01 sent. Activity 138.42. Charge 138.42.")
    assert J.naturally_present("The 138.42 payment was posted.", "e0078 2014-02-22 Payment received: 138.42.")
    assert not J.naturally_present("One-off charge of 114.37 was added.", "e1 2021-08-01 Credit approval: 114.37.")


def test_mechanical_class():
    assert J.mechanical_class(NOTE, "e0122 2016-12-01 Actual reading for 2016-11: 403.6.") == "present_rule"
    assert J.mechanical_class(NOTE, "e0052 2012-08-26 Technician serviced the device.") == "strict_absent"
    assert J.mechanical_class(NOTE, "e0077 2016-03-02 Statement 2016-02 sent.") == "ambiguous"


def test_distinguishing_numbers_drop_dates_ids_and_small_numbers():
    assert J.distinguishing_numbers("e0062 2018-02-03 Statement 2018-01: activity 229.18, other 0.00, 12 days") \
        == {"229.18"}


def test_kind_of():
    assert J.kind_of("e1 2013-02-03 Statement for period 2013-01: activity charges 144.12.") == "statement"
    assert J.kind_of("e1 2013-01-15 Actual reading for 2013-01: 720.6.") == "reading"
    assert J.kind_of("e1 2012-08-26 Technician serviced the device.") == "device"
    assert J.kind_of("e1 2012-04-01 Applied credit cr0010.") == "credit"
    assert J.kind_of("e1 2022-11-01 The customer reached out by phone about: payment arrangement.") == "contact"
    assert J.kind_of("e1 2016-08-22 Customer paid 127.10.") == "payment"


# ---------------------------------------------------------------------------
# Paraphrase, insertion, deletion
# ---------------------------------------------------------------------------

UNITS = [
    "e0076 2013-02-03 Statement for period 2013-01: activity charges 144.12, other charges 0.00, credits 0.00, due 144.12.",
    "e0092 2012-04-01 Applied credit cr0010.",
    "e0065 2018-02-18 Reading taken on site for 2018-01. Month activity 658.4.",
    "e0078 2012-09-06 Statement 2012-08 sent. Activity 129.82, other charges 64.96, credits 64.96. One-off CH-2012-08.",
    "Statement 2016-01 was 146.34 paid 2016-02-19 (e0073, e0076).",
]


@pytest.mark.parametrize("u", UNITS)
def test_paraphrase_keeps_every_number_and_id_and_changes_the_words(u):
    p = J.paraphrase(u)
    assert J.distinguishing_numbers(u) <= J.distinguishing_numbers(p)
    assert {i.lower() for i in J._ID.findall(u)} <= {i.lower() for i in J._ID.findall(p)}
    assert p != J.verbatim_sentence(u)
    assert not p.startswith("e0") and "(e00" not in p
    assert p.endswith(".")


def test_paraphrase_rewords_dates():
    p = J.paraphrase(UNITS[0])
    assert "January 2013" in p and "Feb 3, 2013" in p and "2013-01" not in p
    assert "no other charges" in p and "no credits" in p


def test_insert_sentence_keeps_the_note_and_never_appends_after_the_cut_piece():
    s = "Inserted fact 999.99."
    for k in range(20):
        out = J.insert_sentence(NOTE, s, random.Random(k))
        ss = J.split_sentences(out)
        assert s in ss and len(ss) == len(J.split_sentences(NOTE)) + 1
        assert ss[-1] != s
        assert [x for x in ss if x != s] == J.split_sentences(NOTE)


def test_delete_carrying_removes_the_anchor_sentences():
    u = "e0122 2016-12-01 Actual reading for 2016-11: 403.6."
    note = NOTE + " The account had steady usage for many months before the spike, with nothing unusual on file."
    out = J.delete_carrying(note, u)
    assert out is not None and "403.6" not in out and J.strictly_absent(out, u)
    assert J.delete_carrying("Reading 403.6 for 2016-11.", u) is None          # nothing left
    assert J.delete_carrying(NOTE, "e0052 2012-08-26 Technician serviced the device.") is None  # nothing removed


# ---------------------------------------------------------------------------
# Prompt and parser
# ---------------------------------------------------------------------------

def test_prompt_puts_the_note_first_and_asks_for_a_quote_from_the_note():
    p = J.build_prompt("THE NOTE", "THE RECORD")
    assert p.index("THE NOTE") < p.index("THE RECORD")
    assert "stated in the note, or clearly implied" in p
    assert "copy the words from the NOTE (not from the record)" in p
    assert J.JUDGE_FORMAT["required"] == ["quote", "conveyed"]


def test_parse_reply_checks_the_quote_against_the_note():
    ok = J.parse_reply(json.dumps({"quote": "next actual reading is 403.6", "conveyed": True}), NOTE)
    assert ok == {"conveyed": True, "raw_conveyed": True, "quote": "next actual reading is 403.6", "status": "ok"}
    rec = J.parse_reply(json.dumps({"quote": "Actual reading for 2016-11: 403.6", "conveyed": True}), NOTE)
    assert rec["conveyed"] is False and rec["raw_conveyed"] is True and rec["status"] == "unsupported"
    gap = J.parse_reply(json.dumps({"quote": "Stmt 2016-02 ... Next actual reading", "conveyed": True}), NOTE)
    assert gap["conveyed"] is True
    empty = J.parse_reply(json.dumps({"quote": "", "conveyed": True}), NOTE)
    assert empty["status"] == "unsupported"
    no = J.parse_reply(json.dumps({"quote": "", "conveyed": False}), NOTE)
    assert no["conveyed"] is False and no["status"] == "ok"
    assert J.parse_reply("not json", NOTE)["conveyed"] is None
    assert J.parse_reply('{"conveyed": "yes"}', NOTE)["status"] == "parse"
    assert J.parse_reply(None, NOTE)["status"] == "parse"
    assert J.parse_reply('<think>x</think>{"quote": "", "conveyed": false}', NOTE)["conveyed"] is False


def test_quote_matching_ignores_case_spacing_and_edge_punctuation():
    assert J.quote_in_note('"NEXT  actual reading is 403.6."', NOTE)
    assert not J.quote_in_note("ab", NOTE)
    assert not J.quote_in_note("reading is 999.9", NOTE)


class FakeTransport:
    def __init__(self, reply):
        self.reply = reply
        self.bodies = []

    def __call__(self, body):
        self.bodies.append(body)
        return self.reply(body) if callable(self.reply) else self.reply


def test_judge_caches_every_call_and_sends_the_schema(tmp_path):
    t = FakeTransport(json.dumps({"quote": "", "conveyed": False}))
    cache = CallCache(tmp_path / "c.sqlite")
    j = J.OllamaJudge2("m", transport=t, cache=cache, workers=3)
    pairs = [(NOTE, u) for u in UNITS]
    out = j.many(pairs)
    assert [o["conveyed"] for o in out] == [False] * len(UNITS) and j.real_calls == len(UNITS)
    b = t.bodies[0]
    assert b["think"] is False and b["format"] == J.JUDGE_FORMAT and b["options"]["temperature"] == 0
    j2 = J.OllamaJudge2("m", transport=t, cache=cache)
    out2 = j2.many(pairs)
    assert out2 == out and j2.real_calls == 0 and j2.cache_hits == len(UNITS) and len(t.bodies) == len(UNITS)
    j3 = J.OllamaJudge2("other", transport=t, cache=cache)          # the model is part of the key
    j3.many(pairs[:1])
    assert j3.real_calls == 1


def test_judge_keeps_the_order_of_the_pairs_under_threads():
    def reply(body):
        rec = body["messages"][0]["content"].split("Record:\n<<<\n")[1].split("\n>>>")[0]
        return json.dumps({"quote": "Stmt 2016-02", "conveyed": "403.6" in rec})
    j = J.OllamaJudge2("m", transport=FakeTransport(reply), cache=CallCache(":memory:"), workers=4)
    us = [f"e{i:04d} 2016-12-01 reading {'403.6' if i % 3 == 0 else '111.1'}." for i in range(30)]
    got = j.many([(NOTE, u) for u in us])
    assert [g["conveyed"] for g in got] == [i % 3 == 0 for i in range(30)]


# ---------------------------------------------------------------------------
# Metrics and the pick
# ---------------------------------------------------------------------------

def test_score_balanced_accuracy_and_parse_failures():
    truth = [True, True, True, True, False, False]
    s = J.score([True, True, True, False, False, True], truth)
    assert s["balanced_accuracy"] == pytest.approx((3 / 4 + 1 / 2) / 2)
    s2 = J.score([None, True, True, True, False, False], truth)            # None is wrong
    assert s2["tp"] == 3 and s2["fn"] == 1 and s2["parse_failures"] == 1


def test_absent_rate_and_hybrid_and_pass_bar():
    assert J.absent_rate([True, False, False, None]) == {"value": 2 / 3, "n": 3, "unparsed": 1}
    assert J.hybrid([0.9, 0.5, 0.7, 0.7], [False, True, True, None], 0.6, 0.8) == [True, False, True, None]
    assert J.hybrid([0.9], [False], None, None) == [False]                  # no thresholds: the judge decides
    assert J.passes(0.85, 0.7499) and not J.passes(0.849, 0.5) and not J.passes(0.9, 0.75) and not J.passes(None, 0.1)


def test_haiku_plan_is_arithmetic_only():
    p = J.haiku_plan_v2([(NOTE, UNITS[0])] * 10)
    assert p["submitted"] is False and p["n_pairs"] == 10 and p["output_tokens"] == 400
    assert p["estimated_usd"] > 0


# ---------------------------------------------------------------------------
# The calibration set on a synthetic frame
# ---------------------------------------------------------------------------

def _synthetic_pairs() -> tuple[list[dict], list[float]]:
    rp, cos = [], []
    rng = random.Random(5)
    for system in J.SYSTEMS:
        for n in range(6):
            y = 2010 + n
            units = [f"e{n}{k:02d} {y}-{k + 1:02d}-03 Statement for period {y}-{k + 1:02d}: activity charges "
                     f"{100 + 10 * n + k}.{k:02d}, credits 0.00." for k in range(8)]
            units += [f"e{n}{k + 8:02d} {y}-{k + 1:02d}-20 Technician serviced the device." for k in range(4)]
            note = (f"Charge {y}-01 was {100 + 10 * n}.00 and {y}-02 was {101 + 10 * n}.01. Usage is steady and the "
                    f"account has no open dispute on file. Recommend EXPLAIN for the {y}-01 charge and close the review.")
            for j, u in enumerate(units):
                rp.append({"case_id": f"k{n}", "system": system, "unit_index": j, "note": note, "unit": u})
                cos.append(rng.random())
    return rp, cos


def test_calibration_set_truth_and_construction():
    rp, cos = _synthetic_pairs()
    quota = {k: 2 for k in J.QUOTA}
    rows = J.build_calibration_set(rp, cos, quota=quota)
    assert rows == J.build_calibration_set(rp, cos, quota=quota)            # deterministic
    assert rows != J.build_calibration_set(rp, cos, seed="other", quota=quota)
    by = {k: [r for r in rows if r["kind"] == k] for k in J.QUOTA}
    assert all(by[k] for k in J.QUOTA)
    for r in rows:
        assert r["truth"] is (r["kind"] in J.KINDS_YES)
        if r["kind"] == "natural_present":
            assert r["note"] == r["real_note"] and J.naturally_present(r["note"], r["unit"])
        if r["kind"] == "natural_absent":
            assert r["note"] == r["real_note"] and J.strictly_absent(r["note"], r["unit"])
        if r["kind"] in ("inserted_verbatim", "inserted_paraphrase"):
            assert r["inserted"] in r["note"] and J.strictly_absent(r["real_note"], r["unit"])
        if r["kind"] == "same_topic_distractor":
            assert r["inserted"] in r["note"] and J.strictly_absent(r["note"], r["unit"])
            d = next(p for p in rp if (p["case_id"], p["system"], p["unit_index"])
                     == (r["case_id"], r["system"], r["distractor_unit_index"]))
            assert J.kind_of(d["unit"]) == J.kind_of(r["unit"]) and d["unit"] != r["unit"]
        if r["kind"] == "deleted_sentence":
            assert J.strictly_absent(r["note"], r["unit"]) and not J.strictly_absent(r["real_note"], r["unit"])
    keys = {}
    for r in rows:
        keys[(r["case_id"], r["system"])] = keys.get((r["case_id"], r["system"]), 0) + 1
    assert max(keys.values()) <= J.PER_NOTE_CAP
    assert len({r["id"] for r in rows}) == len(rows)


def test_pair_cosines_groups_by_note():
    def emb(texts):
        return [[1.0, 0.0] if "a" in t else [0.0, 1.0] for t in texts]
    got = J.pair_cosines([("a here. b there.", "a"), ("b only.", "a"), ("a here. b there.", "b")], emb)
    assert got == pytest.approx([1.0, 0.0, 1.0])


# ---------------------------------------------------------------------------
# The real set, when results/ is present
# ---------------------------------------------------------------------------

REAL_SET = Path(OUT) / J.OUT2_NAME / "calibration_set.jsonl"


@pytest.mark.skipif(not REAL_SET.exists(), reason="results/ not present")
def test_real_calibration_set_is_balanced_and_true_by_construction():
    rows = [json.loads(x) for x in REAL_SET.read_text().splitlines() if x.strip()]
    assert len(rows) == 400 and sum(r["truth"] for r in rows) == 200
    assert {r["system"] for r in rows} == set(J.SYSTEMS)
    for r in rows:
        if r["kind"] in ("natural_absent", "deleted_sentence", "same_topic_distractor"):
            assert J.strictly_absent(r["note"], r["unit"])
        if r["kind"] == "natural_present":
            assert J.naturally_present(r["note"], r["unit"])


def test_run_refuses_a_failed_calibration_without_an_explicit_accept(tmp_path):
    out = tmp_path / J.OUT2_NAME
    out.mkdir()
    calj = {"choice": {"status": "no_local_judge_passed"}, "judges": {"qwen30": {"passes": False}},
            "tau_kappa": {"tau": 0.5}, "band": {"tau_low": 0.4, "tau_high": 0.9}}
    (out / "calibration_v2.json").write_text(json.dumps(calj))
    assert J.main(["--root", str(tmp_path), "run"]) == 1
    assert J.main(["--root", str(tmp_path), "run", "--accept", "qwen30"]) == 1           # no reason
    assert J.main(["--root", str(tmp_path), "run", "--accept", "haiku_v1", "--accept-reason", "x"]) == 1
    assert J.main(["--root", str(tmp_path), "run", "--accept", "nope", "--accept-reason", "x"]) == 1

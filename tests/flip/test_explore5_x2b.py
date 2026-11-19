"""Arm X2b (docs/flip-protocol-v5b-x2-rerun.md): the renderer fits the budget, the re-ask path, the cut gate, the
dry-run arithmetic, the freeze and the spend guard. No network: a fake Batches API on the synthetic v5 frame.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

import test_explore5 as T5
from lens.flip.explore5 import engine as E
from lens.flip.explore5 import frame as F
from lens.flip.explore5 import protocol as PR
from lens.flip.explore5 import x2b as X
from lens.flip.explore5 import x2b_prompts as XP
from lens.flip.headline.engine import Refused, _read_json
from lens.flip.headline.mock import FakeBatchAPI, engine_meta_fn
from lens.flip.headline.protocol import _sentinel_case
from lens.flip.schema import View
from lens.flip.systems.base import count_tokens

REPO = Path(__file__).resolve().parents[2]


# ---------------------------------------------------------------------------
# Prompt and renderer
# ---------------------------------------------------------------------------

def _words(n: int, word: str = "2015-07") -> str:
    return " ".join([word] * n)


SAMPLE_FORMS = [
    {"facts": "Recommend EXPLAIN. 2015-07 charge 131.26 matches 2014-02, 2014-08, 2015-01 highs near 130; "
              "no credits or one-off items.",
     "changed": "Charge up 60% on the 2015-06 statement of 81.40.",
     "open": "No usage detail for 2015-07."},
    {"facts": "Account 4471: three late fees since 2016-03, payment plan ended 2016-09, balance 412.18 overdue.",
     "changed": "Plan lapsed; autopay removed 2016-10-02.",
     "open": "Whether the customer asked to stop autopay."},
    {"facts": "", "changed": "  none\n\nreally  ", "open": "Unclear."},
]


def test_prompt_states_caps_budget_in_tokens_and_words():
    v, c = View(system="s", budget=None, text="{RECORDS}"), _sentinel_case()
    p1 = XP.build_x2b_prompt(v, c, XP.CAPS_FIRST)
    p2 = XP.build_x2b_prompt(v, c, XP.CAPS_TIGHT)
    for p, caps in ((p1, XP.CAPS_FIRST), (p2, XP.CAPS_TIGHT)):
        for k in XP.FIELDS:
            assert f"- {k}: {XP.HELP[k]} At most {caps[k]} words." in p
        assert "150 tokens" in p and "about 330 characters" in p and "about 47 words" in p
        assert f"{sum(caps.values())} words in all" in p and "p_residue" in p
        assert '{"facts": text, "changed": text, "open": text, "p_residue": a number from 0 to 1}' in p
    assert p1 != p2 and p1.replace("20 words", "").replace("38 words", "") != p1
    assert XP.X2B_SCHEMA["required"] == ["facts", "changed", "open", "p_residue"]
    assert XP.X2B_SCHEMA["additionalProperties"] is False


def test_labels_are_short_and_render_is_compact():
    assert all(len(lbl.split()) <= 2 for lbl in XP.LABELS.values())
    r = XP.render_x2b(SAMPLE_FORMS[2])
    assert r == "Facts: none\nChanged: none really\nOpen: Unclear."
    assert XP.render_x2b({}) == "Facts: none\nChanged: none\nOpen: none"


@pytest.mark.parametrize("form", SAMPLE_FORMS)
def test_sample_forms_fit_the_budget(form):
    r = XP.render_x2b(form)
    assert count_tokens([r]) <= XP.NOTE_BUDGET and XP.fits(r)
    got = XP.parse_x2b_reply(json.dumps({**form, "p_residue": 0.3}))
    assert got["note"] == r and not got["cut"] and not got["over"] and got["complete"] and got["p_residue"] == 0.3


@pytest.mark.parametrize("caps", [XP.CAPS_FIRST, XP.CAPS_TIGHT])
def test_a_form_at_its_caps_fits_even_with_long_words(caps):
    # 7-character words ("2015-07") plus a space: denser than the 6.9 characters per word seen on X2 forms
    r = XP.render_x2b({k: _words(caps[k]) for k in XP.FIELDS})
    assert XP.fits(r)
    assert sum(caps.values()) <= XP.budget_words()


def test_over_budget_form_is_flagged_and_cut_by_the_shared_helper():
    big = {k: _words(30) for k in XP.FIELDS}
    got = XP.parse_x2b_reply(json.dumps({**big, "p_residue": 0.5}))
    assert got["over"] and got["cut"] and got["rendered_tokens"] > XP.NOTE_BUDGET
    assert count_tokens([got["note"]]) <= XP.NOTE_BUDGET and got["note"].startswith("Facts: 2015-07")
    bad = XP.parse_x2b_reply("not json " * 80)
    assert bad["fields"] is None and not bad["complete"] and bad["over"] and bad["cut"] and bad["p_residue"] is None


def test_choose_prefers_a_parsed_reask():
    first = {"fields": {"facts": "a"}, "note": "first", "cut": True}
    second_ok = {"fields": {"facts": "b"}, "note": "second", "cut": False}
    second_bad = {"fields": None, "note": "x", "cut": True}
    assert X.choose(first, None) == (first, "first")
    assert X.choose(first, second_ok) == (second_ok, "second")
    assert X.choose(first, second_bad) == (first, "first")


# ---------------------------------------------------------------------------
# The re-ask path end to end on the fake API
# ---------------------------------------------------------------------------

def _mode(case_id: str) -> int:
    """0: the first form fits; 1: over, the re-ask fits; 2: over twice (cut); 3: over, the re-ask is not JSON."""
    return T5._h(case_id) % 4


def _answers_b(cases):
    base = T5._answers(cases)
    by = {c.case_id: c for c in cases}

    def answer(cid, params, meta):
        if meta["kind"] != X.KIND:
            return base(cid, params, meta)
        case = by[meta["case_id"]]
        user = params["messages"][0]["content"]
        tight = f"At most {XP.CAPS_TIGHT['facts']} words." in user
        m = _mode(case.case_id)
        short = {"facts": f"Recommend {case.a_star}. Records cover {case.query_period}.", "changed": "nothing",
                 "open": "none open"}
        long = {k: _words(40) for k in XP.FIELDS}
        if not tight:
            text = json.dumps({**(short if m == 0 else long), "p_residue": 0.11})
        elif m == 1:
            text = json.dumps({**short, "p_residue": 0.22})
        elif m == 2:
            text = json.dumps({**long, "p_residue": 0.33})
        else:
            text = "I cannot fit this. " * 40
        return {"custom_id": cid, "result": {"type": "succeeded", "message": {
            "content": [{"type": "text", "text": text}], "stop_reason": "end_turn",
            "usage": {"input_tokens": len(user) // 2 + 331, "output_tokens": 100}}}}
    return answer


def _run_b(root, cap=100.0, check=False, **kw):
    cases = F.load_cases(root)
    api = FakeBatchAPI(_answers_b(cases), engine_meta_fn(root / X.ARM), **kw)
    r = X.run_x2b(root, api, cap_usd=cap, log=lambda m: None, sleep=lambda s: None, check_protocol=check,
                  max_steps=400)
    return r, api


def _stage_counts(root):
    sp = _read_json(root / X.ARM / "spend.json", {})
    return {k: v["requests"] for k, v in sp["by_stage"].items()}


def test_reask_path_end_to_end(tmp_path):
    root, hs = T5._make_root(tmp_path, n_cases=16)
    r, api = _run_b(root)
    assert r == "complete"
    st = _read_json(root / X.ARM / "state" / "cases" / f"{hs[0].case.case_id}.json", {})
    assert "recent" in st
    modes = {h.case.case_id: _mode(h.case.case_id) for h in hs}
    assert set(modes.values()) >= {0, 1, 2}
    counts = _stage_counts(root)
    n_over = sum(m != 0 for m in modes.values())
    assert counts["x2b_A"] == len(hs) and counts["x2b_B"] == len(hs)
    assert counts.get("x2b_A2", 0) == n_over          # one re-ask per over-budget form, paid and counted
    for h in hs:
        o = _read_json(root / X.ARM / "state" / "cases" / f"{h.case.case_id}.json", {})["recent"]["out"]
        m = modes[h.case.case_id]
        assert o["note_tokens"] <= XP.NOTE_BUDGET
        assert o["reasked"] == (m != 0)
        if m == 0:
            assert not o["cut"] and o["used"] == "first" and o["p_residue"] == 0.11
        elif m == 1:
            assert not o["cut"] and o["used"] == "second" and o["p_residue"] == 0.22 and o["second_tokens"] <= 150
        elif m == 2:
            assert o["cut"] and o["used"] == "second" and o["p_residue"] == 0.33 and o["second_tokens"] > 150
        else:
            assert o["cut"] and o["used"] == "first" and o["p_residue"] == 0.11 and o["second_status"] == "fields_parse"
    # the shared ledger carries X2b; resuming a finished arm submits nothing
    led = _read_json(root / "spend.json", {})
    assert X.ARM in led["arms"] and led["total_usd"] == pytest.approx(sum(a["actual_usd"] for a in led["arms"].values()))
    created = api.created
    r2, api2 = _run_b(root)
    assert r2 == "complete" and api2.created == 0 and created > 0


def test_report_gate_and_pairing_against_v3_and_x2(tmp_path, capsys):
    root, hs = T5._make_root(tmp_path, n_cases=16)
    assert T5._run(root, "X2")[0] == "complete"
    assert _run_b(root)[0] == "complete"
    s = X.write(root)
    x = s["X2b_structured_fit"]
    n = len(hs)
    n_cut = sum(_mode(h.case.case_id) in (2, 3) for h in hs)
    g = x["structure"]["cut_gate"]
    assert g["n_cut"] == n_cut and g["n"] == n and g["readable"] == (n_cut / n <= 0.10)
    assert x["structure"]["reasked"] == sum(_mode(h.case.case_id) != 0 for h in hs)
    assert x["structure"]["reask_fits"] == sum(_mode(h.case.case_id) == 1 for h in hs)
    assert x["residue_rate"]["n"] == n and "residue_minus_v3" in x and "residue_minus_x2" in x
    assert "auroc_minus_v3_auroc" in x and "auroc_minus_x2_auroc" in x and "recoverable_rate" in x
    assert x["x2_reference"]["residue_rate"]["n"] == n
    assert s["spend"]["x2b_usd"] > 0 and s["spend"]["ledger_by_arm"].keys() >= {"X2", "X2b"}
    assert (root / X.SUMMARY_NAME).exists()
    print(X.headline(s))
    assert "cut gate:" in capsys.readouterr().out


def test_cut_gate():
    assert X.cut_gate(10, 100)["readable"] and not X.cut_gate(11, 100)["readable"]
    assert X.cut_gate(49, 490)["readable"] and not X.cut_gate(50, 490)["readable"]
    assert not X.cut_gate(0, 0)["readable"]
    assert "PASS" in X.gate_line(X.cut_gate(3, 490)) and "FAIL" in X.gate_line(X.cut_gate(477, 490))
    assert "477 of 490" in X.gate_line(X.cut_gate(477, 490))


# ---------------------------------------------------------------------------
# Dry run
# ---------------------------------------------------------------------------

def test_dryrun_arithmetic(tmp_path):
    from lens.flip.explore5 import dryrun as D
    root, hs = T5._make_root(tmp_path)
    res = X.dryrun_x2b(root, tmp_path / "no-results", write=True)
    assert res["api_calls"] == 0 and res["count_tokens_calls"] == 0 and res["submitted"] is False
    n = len(hs)
    by = {s["stage"].split()[0]: s for s in res["stages"]}
    assert by["x2b_A"]["requests"] == n and by["x2b_B"]["requests"] == n
    assert by["x2b_A2"]["requests"] == pytest.approx(X.REASK_LIKELY * n) and by["x2b_A2"]["requests_upper"] == n
    full = D.stage_price("t", [D.rq.request_chars(X.x2b_request(h.view, h.case, E.NOTE_WRITER, XP.CAPS_TIGHT))
                               for h in hs], X.ASSUMED_OUT["x2b_A2"], 6000)
    assert by["x2b_A2"]["likely_usd"] == pytest.approx(X.REASK_LIKELY * full["likely_usd"], rel=1e-6)
    assert by["x2b_A2"]["upper_usd"] == pytest.approx(full["upper_usd"], rel=1e-6)
    one = by["x2b_A"]
    tin = sum(D.tokens_likely(D.rq.request_chars(X.x2b_request(h.view, h.case, E.NOTE_WRITER))) for h in hs)
    assert one["likely_usd"] == pytest.approx((tin * 0.05 + n * 1851 * 0.25) / 1e6, abs=1e-6)
    t = res["total"]
    assert t["likely_usd"] == pytest.approx(sum(s["likely_usd"] for s in res["stages"]), abs=1e-3)
    assert t["upper_usd"] == pytest.approx(sum(s["upper_usd"] for s in res["stages"]), abs=1e-3)
    assert 0 < t["likely_usd"] < t["upper_usd"] and res["rounds_likely"] <= res["rounds_upper"]
    assert res["notes_and_B"]["likely_usd"] + res["search"]["likely_usd"] == pytest.approx(t["likely_usd"], abs=1e-3)
    assert res["likely_at_or_under_price_gate"] is (t["likely_usd"] <= 1.50)
    assert res["sensitivity_x2_flip_rate"] is None          # no finished X2 in this root
    assert (root / X.DRYRUN_NAME).exists()


# ---------------------------------------------------------------------------
# Freeze, check, guards
# ---------------------------------------------------------------------------

def _frozen_v5(root):
    PR.freeze(root)
    PR.finalize(root)


def test_freeze_draft_refuses_then_final_runs(tmp_path, monkeypatch):
    root, hs = T5._make_root(tmp_path)
    with pytest.raises(SystemExit, match="v5 protocol"):
        X.freeze(root)
    _frozen_v5(root)
    with pytest.raises(SystemExit, match="dryrun"):
        X.freeze(root)
    X.dryrun_x2b(root, tmp_path / "no-results")
    rec = X.freeze(root, now=0)
    assert rec["status"] == "DRAFT" and rec["version"] == "5b-exploratory" and len(rec["prompts"]) == 5
    assert X.check(root) == ["status:DRAFT"]
    cases = F.load_cases(root)
    api = FakeBatchAPI(_answers_b(cases), engine_meta_fn(root / X.ARM))
    with pytest.raises(Refused, match="status:DRAFT"):
        X.run_x2b(root, api, cap_usd=10, log=lambda m: None, sleep=lambda s: None, max_steps=3)
    assert api.created == 0
    # the price gate: finalizing refuses when the dry run's likely price is above $1.50
    p = root / X.DRYRUN_NAME
    good = p.read_text()
    d = json.loads(good)
    d["total"]["likely_usd"] = 2.0
    p.write_text(json.dumps(d))
    with pytest.raises(SystemExit, match="above"):
        X.freeze(root, price_confirmed=True)
    p.write_text(good)
    assert X.freeze(root, price_confirmed=True)["status"] == "FINAL"
    with pytest.raises(SystemExit, match="FINAL"):
        X.freeze(root, price_confirmed=True)
    assert X.check(root) == []
    r, api = _run_b(root, check=True)
    assert r == "complete" and X.check(root, root / X.ARM) == []
    # a change to the form, to the v5 prompts or to protocol_v5.json is caught
    monkeypatch.setattr(XP, "CAPS_TIGHT", {"facts": 15, "changed": 7, "open": 5})
    assert "prompt:x5b.x2b.prompt_tight" in X.check(root) and "arm:X2b" in X.check(root)
    monkeypatch.undo()
    from lens.flip.explore5 import prompts as P
    monkeypatch.setattr(P, "READER_PARAGRAPH", P.READER_PARAGRAPH + " Extra.")
    assert "v5:prompt:x5.x1.prompt" in X.check(root)
    monkeypatch.undo()
    v5 = root / PR.NAME
    v5.write_text(v5.read_text() + "\n")
    assert "root:protocol_v5_sha256" in X.check(root)


def test_spend_guard_refuses_before_any_request(tmp_path):
    root, hs = T5._make_root(tmp_path)
    cases = F.load_cases(root)
    api = FakeBatchAPI(_answers_b(cases), engine_meta_fn(root / X.ARM))
    with pytest.raises(Refused, match="cap|upper bound"):
        X.run_x2b(root, api, cap_usd=0.0001, log=lambda m: None, sleep=lambda s: None, check_protocol=False,
                  max_steps=5)
    assert api.created == 0


def test_spend_guard_counts_the_v5_arms(tmp_path):
    root, hs = T5._make_root(tmp_path)
    assert T5._run(root, "X1")[0] == "complete"
    spent = E.update_ledger(root, 100.0)["total_usd"]
    assert spent > 0
    cases = F.load_cases(root)
    api = FakeBatchAPI(_answers_b(cases), engine_meta_fn(root / X.ARM))
    with pytest.raises(Refused):
        X.run_x2b(root, api, cap_usd=spent + 1e-6, log=lambda m: None, sleep=lambda s: None, check_protocol=False,
                  max_steps=5)
    assert api.created == 0
    led = X.update_ledger(root, 100.0)
    assert set(led["arms"]) >= {"X1"} and led["total_usd"] == pytest.approx(spent)


def test_cli_guards(tmp_path):
    with pytest.raises(SystemExit, match="price-confirmed"):
        X.main(["run", "--results", str(tmp_path), "--cap-usd", "5"])
    with pytest.raises(SystemExit, match="cap-usd"):
        X.main(["run", "--results", str(tmp_path), "--price-confirmed"])


def test_private_word_list_absent_from_x2b_files():
    banned_file = REPO / "banned_words.local.txt"
    if not banned_file.exists():
        pytest.skip("banned_words.local.txt not present")
    banned = [w.strip().lower() for w in banned_file.read_text().splitlines() if w.strip()]
    files = [REPO / "src" / "lens" / "flip" / "explore5" / n for n in ("x2b.py", "x2b_prompts.py")]
    files += [Path(__file__), REPO / "docs" / "flip-protocol-v5b-x2-rerun.md"]
    blob = "\n".join(f.read_text().lower() for f in files if f.exists())
    blob += "\n".join(X.prompt_texts().values()).lower()
    hits = [w for w in banned if w in blob]
    assert not hits, hits

"""Exploratory protocol v4: prompts, strip rule, assembly, and every arm end to end on a fake Batches API."""

from __future__ import annotations

import json
import re
import shutil
from pathlib import Path

import pytest

from lens.flip.explore4 import prompts as P
from lens.flip.residue import build_residue_note_prompt, parse_note_reply
from lens.flip.schema import View

ROOT = Path("results/exploratory-v4")


def _view(units):
    return View(system="recent", budget=1000, text="\n".join(units), event_ids=[], units=list(units))


def test_budget_prompt_is_v3_prompt_at_150():
    from lens.flip.headline.protocol import _sentinel_case
    v, c = View(system="s", budget=None, text="{RECORDS}"), _sentinel_case()
    assert P.build_budget_note_prompt(v, c, 150) == build_residue_note_prompt(v, c)
    raw = json.dumps({"note": "word " * 400, "p_residue": 0.3})
    assert P.parse_budget_note_reply(raw, 150) == parse_note_reply(raw)
    long_note, p = P.parse_budget_note_reply(raw, 600)
    assert p == 0.3 and len(long_note) > len(parse_note_reply(raw)[0])


def test_strip_rule():
    note = "Recommend EXPLAIN. Similar winter charges appear in Feb 2016. Choose apply credit now. Seasonal use explains it."
    out, dropped = P.strip_advice(note)
    assert out == "Similar winter charges appear in Feb 2016. Seasonal use explains it."
    assert [d["rules"] for d in dropped] == [["R1", "R3", "R4"], ["R2", "R3"]]
    same = "No device event. Activity is seasonal."
    assert P.strip_advice(same) == (same, [])
    assert P.strip_advice("Status: OPEN_REVIEW fits.")[0] == ""


def test_assemble_extract_budget_and_order():
    units = ["a" * 100, "b" * 100, "c" * 200, "d" * 50]
    v = _view(units)
    note, info = P.assemble_extract(v, ["R03", "R01", "R02", "R04", "R01", "R99"], budget=150)
    # 330 chars: R03 (200) + R01 (101) fit; R02 would pass; R04 (51) passes 352 -> skipped too.
    assert note == "a" * 100 + "\n" + "c" * 200
    assert info["n_kept"] == 2 and info["n_skipped_budget"] == 2 and info["n_duplicate"] == 1 and info["n_unknown"] == 1
    assert P.assemble_extract(v, None)[0] == ""


def test_parse_extract():
    assert P.parse_extract_reply('{"labels": ["R01", 3], "p_residue": 0.2}') == (["R01"], 0.2)
    assert P.parse_extract_reply('{"p_residue": 2}') == (None, None)
    assert P.parse_extract_reply(None) == (None, None)


# ---------------------------------------------------------------------------
# End to end on the fake Batches endpoint
# ---------------------------------------------------------------------------

_LINE_ID = re.compile(r"\be(\d{4})\b")


def _answer_factory(cases):
    from lens.flip.generator import oracle
    by = {c.case_id: c for c in cases}

    def answer(cid, params, meta):
        case = by[meta["case_id"]]
        user = params["messages"][0]["content"]
        kind = meta["kind"]
        if kind == "decide":
            ids = {f"e{m}" for m in _LINE_ID.findall(user)}
            act = oracle([e for e in case.events if e.id in ids], case.query_period)
            text = json.dumps({"action": act, "reason": "mock", "event_ids": []})
        elif kind == "extract":
            labels = re.findall(r"^\[(R\d+)\]", user, re.M)
            text = json.dumps({"labels": labels[::-1][:4], "p_residue": 0.3})
        elif kind == "note_budget":
            recs = user.split("Records:\n", 1)[1].split("\n\nRequest:")[0]
            text = json.dumps({"note": recs[: int(meta["note_budget"] * 2.2)], "p_residue": 0.4})
        else:
            raise AssertionError(kind)
        return {"custom_id": cid, "result": {"type": "succeeded", "message": {
            "content": [{"type": "text", "text": text}], "stop_reason": "end_turn",
            "usage": {"input_tokens": len(user) // 2, "output_tokens": 500}}}}
    return answer


@pytest.mark.skipif(not (ROOT / "frame.jsonl").exists(), reason="needs the built v4 frame")
def test_all_arms_end_to_end(tmp_path):
    from lens.flip.explore4 import engine as E
    from lens.flip.explore4.analysis import write
    from lens.flip.explore4.frame import load_cases
    from lens.flip.explore4.protocol import freeze
    from lens.flip.headline.mock import FakeBatchAPI, engine_meta_fn

    root = tmp_path / "x4"
    root.mkdir()
    lines = (ROOT / "frame.jsonl").read_text().splitlines()
    (root / "frame.jsonl").write_text("\n".join(lines[:40]) + "\n")
    shutil.copyfile(ROOT / "cases.jsonl", root / "cases.jsonl")
    freeze(root)
    cases = load_cases(root)
    for arm in E.ARMS:
        d = E.init_arm(arm, root)
        api = FakeBatchAPI(_answer_factory(cases), engine_meta_fn(d))
        r = E.run_arm(d, api, root=root, log=lambda m: None, sleep=lambda s: None, max_steps=200)
        assert r == "complete", arm
    led = json.loads((root / "spend.json").read_text())
    assert set(led["arms"]) == set(E.ARMS) and led["total_usd"] > 0
    s = write(root)
    assert s["study_A_extractive"]["residue_rate"]["n"] == 40
    assert s["study_C_budget"]["by_budget"]["600"]["residue_rate"]["n"] == 40
    assert "sonnet-5-5" in s["study_B_readers"]
    assert s["study_B_readers"]["sonnet-5-5"]["view_baseline"]["rate"]["n"] == 40
    assert (root / "REPORT.md").read_text().startswith("# LENS flip test")


@pytest.mark.skipif(not (ROOT / "frame.jsonl").exists(), reason="needs the built v4 frame")
def test_study_cap_refuses(tmp_path):
    from lens.flip.explore4 import engine as E
    from lens.flip.explore4.protocol import freeze
    from lens.flip.headline.engine import Refused
    from lens.flip.headline.mock import FakeBatchAPI, engine_meta_fn

    root = tmp_path / "x4"
    root.mkdir()
    lines = (ROOT / "frame.jsonl").read_text().splitlines()
    (root / "frame.jsonl").write_text("\n".join(lines[:10]) + "\n")
    shutil.copyfile(ROOT / "cases.jsonl", root / "cases.jsonl")
    freeze(root)
    E.add_probe(root, {"what": "pretend spend", "usd": 29.999, "input_tokens": 0, "output_tokens": 0})
    d = E.init_arm("B-sonnet", root)
    api = FakeBatchAPI(lambda *a: None, engine_meta_fn(d))
    with pytest.raises(Refused, match="study cap"):
        E.run_arm(d, api, root=root, log=lambda m: None, sleep=lambda s: None, max_steps=5)
    assert api.created == 0


@pytest.mark.skipif(not (ROOT / "frame.jsonl").exists(), reason="needs the built v4 frame")
def test_reproduces_v3_rates():
    from lens.flip.explore4.analysis import Stats
    from lens.flip.explore4.frame import load_cases, load_frame
    cases = load_cases(ROOT)
    rows = [{"case_id": h.case.case_id, "f": h.row["v3"]["residue_flip"], "fl": h.row["v3"]["a_floor"] != h.row["a_A"]}
            for h in load_frame(ROOT, cases)]
    st = Stats(cases, rows)
    r = st.rate("f")
    assert (r["k"], r["n"]) == (142, 490) and r["ci95"] == [0.248031, 0.331933]
    assert st.rate("fl")["ci95"] == [0.077552, 0.132797]

"""Exploratory protocol v7: the note-framed reader prompt, the two text rules, the X1 split, the A-wrong set, both
studies end to end on a fake Batches API, the round size limit, the spend guard, the freeze, the dry run and the
analysis on a mock. No network.

The end-to-end tests run on a small synthetic frame built here from the generator, so they need no stored
results. The checks against the real frame skip when results/ is absent (it is git-ignored).
"""

from __future__ import annotations

import hashlib
import json
import zlib
from pathlib import Path

import pytest

from lens.flip.decider import POLICY, build_messages
from lens.flip.explore5 import engine as E5
from lens.flip.explore5 import prompts as P5
from lens.flip.explore5 import protocol as PR5
from lens.flip.explore7 import analysis as AN
from lens.flip.explore7 import dryrun as D
from lens.flip.explore7 import engine as E
from lens.flip.explore7 import frame as F
from lens.flip.explore7 import prompts as P
from lens.flip.explore7 import protocol as PR
from lens.flip.headline import requests as rq
from lens.flip.headline.engine import Refused, _read_json
from lens.flip.headline.mock import FakeBatchAPI, engine_meta_fn
from lens.flip.headline.protocol import _sentinel_case
from lens.flip.residue import build_residue_note_prompt, note_view
from lens.flip.schema import ACTIONS, View, write_jsonl

REPO = Path(__file__).resolve().parents[2]
REAL = F.V4 / "frame.jsonl"
REAL_V5 = F.V5 / "X1" / "state" / "cases"


# ---------------------------------------------------------------------------
# R1: one change to the v1 reader prompt
# ---------------------------------------------------------------------------

def test_r1_prompt_is_v1_with_the_note_framing_only():
    case = _sentinel_case()
    note = "The figure is high. Recommend EXPLAIN."
    v1 = build_messages(note_view(note), [], case, "v1")
    v1_sys, v1_user = v1[0]["content"], v1[1]["content"]
    r1_user = P.build_r1_user(note, case)
    assert r1_user != v1_user and r1_user.replace("\n\nHandoff note:\n", "\n\nRecords:\n", 1) == v1_user
    assert P.R1_SYSTEM != v1_sys and "handoff note" in P.R1_SYSTEM and "not the records themselves" in P.R1_SYSTEM
    assert P.R1_SYSTEM.endswith("to choose one action. The note is not the records themselves.")
    assert POLICY["v1"] in r1_user and r1_user.count("Records:") == 0
    assert P.build_r1_user("  ", case).split("Handoff note:\n")[1].startswith("(no note)")
    dec = E.arm_config("R")["decider"]
    h = type("H", (), {"case": case})()
    a, b = E.v1_request(note, h, dec), E.r1_request(note, h, dec)
    assert {k: v for k, v in a.items() if k not in ("system", "messages")} == {k: v for k, v in b.items() if k not in ("system", "messages")}
    assert b["system"] == P.R1_SYSTEM and b["messages"] == [{"role": "user", "content": r1_user}]
    assert a["system"] == v1_sys and a["messages"][0]["content"] == v1_user
    assert "thinking" not in b and b["max_tokens"] == 6000 and b["output_config"]["effort"] == "high"


# ---------------------------------------------------------------------------
# R2 and R3: the text rules
# ---------------------------------------------------------------------------

def test_hedge_strip_drops_marked_sentences_only():
    note = ("The March figure is 135.82 against about 80. It may be seasonal. No service is logged. "
            "The cause is unclear. Choose the review only if readings look faulty. The reset appears to explain it.")
    out, dropped = P.strip_hedges(note)
    assert out == "The March figure is 135.82 against about 80. No service is logged."
    assert [d["rules"] for d in dropped] == [["H1"], ["H3"], ["H4"], ["H2"]]
    clean = "In May 2019 the figure was 140.10. The estimate is about 80, roughly. Verify the reading."
    assert P.strip_hedges(clean) == (clean, [])                       # the month, approximators and advice are not markers
    assert P.hedge_rules("This cannot be checked here.") == ["H3"] and P.hedge_rules("Likely a reset.") == ["H1"]
    assert P.hedge_rules("it may differ") == ["H1"] and P.hedge_rules("May figures differ") == []
    assert P.strip_hedges("It might be a fault.") == ("", [{"text": "It might be a fault.", "rules": ["H1"]}])
    rule = json.loads(P.hedge_rule_text())
    assert set(rule) >= {"H1", "H1_may", "H2", "H3", "H4", "join", "splitter", "not_markers"} and "about" in rule["not_markers"]


def test_trailing_fragment_rule():
    note = "The figure is 135.82. No service is logged. Recommend the review to verify the new unit before any"
    assert P.ends_mid_sentence(note)
    assert P.drop_trailing_fragment(note) == ("The figure is 135.82. No service is logged.",
                                              "Recommend the review to verify the new unit before any")
    whole = "The figure is 135.82. No service is logged."
    assert not P.ends_mid_sentence(whole) and P.drop_trailing_fragment(whole) == (whole, "")
    assert P.drop_trailing_fragment("The figure is 135.82 and the prior one is 80.10 so the") == \
        ("", "The figure is 135.82 and the prior one is 80.10 so the")  # a decimal point is not a sentence end
    assert P.drop_trailing_fragment("Done! Is it high? The rest is cut  ") == ("Done! Is it high?", "The rest is cut")
    assert "no_mark" in json.loads(P.fragment_rule_text())


# ---------------------------------------------------------------------------
# X1p and X1r: the X1 paragraph, split by slicing
# ---------------------------------------------------------------------------

def test_x1_paragraph_splits_without_a_changed_word():
    assert P.X1_S1 + P.X1_S2 + P.X1_S3 + P.X1_POLICY == P5.READER_PARAGRAPH
    assert P.X1_POLICY == POLICY["v1"]
    assert P.READER_HALF == (P.X1_S1 + P.X1_S3).rstrip() and P.POLICY_HALF == P.X1_S2.rstrip() + "\n" + POLICY["v1"]
    # every word of the paragraph is in exactly one half
    assert sorted((P.READER_HALF + " " + P.POLICY_HALF).split()) == sorted(P5.READER_PARAGRAPH.split())
    assert "reads only your note" in P.READER_HALF and "do not see the records" in P.READER_HALF
    assert "Judge p_residue for that colleague" in P.READER_HALF
    assert "policy" not in P.READER_HALF.lower() and not any(a in P.READER_HALF for a in ACTIONS)
    assert P.POLICY_HALF.startswith("They choose one of ") and all(a in P.POLICY_HALF for a in ACTIONS)
    assert "reads only" not in P.POLICY_HALF and "p_residue" not in P.POLICY_HALF and "do not see" not in P.POLICY_HALF
    v, c = View(system="s", budget=None, text="{RECORDS}"), _sentinel_case()
    v3 = build_residue_note_prompt(v, c)
    for build, half in ((P.build_x1p_prompt, P.POLICY_HALF), (P.build_x1r_prompt, P.READER_HALF)):
        pr = build(v, c)
        assert pr.replace("\n\n" + half, "", 1) == v3 and pr.index(half) < pr.index("Records:")
    assert P.build_x1p_prompt(v, c) != P.build_x1r_prompt(v, c) != P5.build_x1_prompt(v, c)
    nw = E.arm_config("X")["note_writer"]
    h = type("H", (), {"case": c, "view": v})()
    a, b, x = E.x1p_note_request(h, nw), E.x1r_note_request(h, nw), E.x1_note_request(h, nw)
    strip = lambda p: {k: v for k, v in p.items() if k != "messages"}
    assert strip(a) == strip(b) == strip(x) == strip(E.v3_note_request(h, nw))  # the settings and schema of the v3 writer
    assert x == E5.x1_request(v, c, nw) and E.v3_note_request(h, nw) == rq.note_request(v, c, nw)


def test_settings_are_the_v5_settings_and_v5_is_untouched():
    for arm in E.ARMS:
        cfg = E.arm_config(arm)
        assert cfg["decider"] == E5.arm_config("X1")["decider"] and cfg["rates"] == {"input": 0.05, "output": 0.25}
        assert cfg["bootstrap"] == {"reps": 1000, "seed": 11} and cfg["cache_bypass"] is False
        assert cfg["max_requests_per_round"] == E.MAX_REQUESTS_PER_ROUND == 1000
    assert E.arm_config("X")["note_writer"] == E5.arm_config("X1")["note_writer"] and "note_writer" not in E.arm_config("R")
    assert set(E.arm_config("R")["assumed_output_tokens"]) == {"r1_B", "r2_B", "r3_B"}
    assert E.arm_config("R")["sub_arms"] == ["R1", "R2", "R3"] and E.arm_config("X")["sub_arms"] == ["X1p", "X1r", "XW"]
    with pytest.raises(ValueError):
        E.arm_config("X1p")
    v5 = PR5.current()
    assert v5["prompts"]["x5.x1.prompt"] == "0f922c5080f2b2e969d5e2decc2c0d2378c90c704ba43dd0e05facc4703cad16"
    assert v5["arm_configs"]["X1"] == "5cb9340bbd11f3345ea0ce6ac3c11841f91142d2b9159248fed4f598693baef0"
    assert PR.current()["reused"]["x5.x1.prompt"] == v5["prompts"]["x5.x1.prompt"]
    assert PR.RUN_CAP_USD == 4.56 and PR.PRICE_GATE_USD == 2.00
    assert int((PR.TRACK_CAP_USD - PR.V6_ACTUAL_USD) * 100) / 100 == PR.RUN_CAP_USD


# ---------------------------------------------------------------------------
# The A-wrong set and the note rows
# ---------------------------------------------------------------------------

def test_wrong_rows_rule_and_note_rows():
    dec = [
        {"case_id": "c1", "system": "recent", "budget": 1000, "mechanism": "M1", "a_star": "EXPLAIN", "a_H": "EXPLAIN", "a_V": "REQUEST_INFO"},
        {"case_id": "c1", "system": "summary", "budget": 1000, "mechanism": "M1", "a_star": "EXPLAIN", "a_H": "EXPLAIN", "a_V": "EXPLAIN"},
        {"case_id": "c1", "system": "rag-bm25", "budget": 500, "mechanism": "M1", "a_star": "EXPLAIN", "a_H": "EXPLAIN", "a_V": "REQUEST_INFO"},
        {"case_id": "c2", "system": "recent", "budget": 1000, "mechanism": "M2", "a_star": "EXPLAIN", "a_H": "OPEN_REVIEW", "a_V": "REQUEST_INFO"},
        {"case_id": "c3", "system": "recent", "budget": 1000, "mechanism": "M0", "a_star": "EXPLAIN", "a_H": "EXPLAIN", "a_V": "PARSE_ERROR"},
    ]
    rows = F.wrong_rows(dec, lambda c, s: ({"event_ids": [c + s]}, "src"))
    assert [(r["case_id"], r["system"], r["a_V"]) for r in rows] == [("c1", "recent", "REQUEST_INFO"), ("c3", "recent", "PARSE_ERROR")]
    assert rows[0]["view"] == {"event_ids": ["c1recent"]} and rows[0]["source"] == "src"
    note = "The figure is high. It may be seasonal. Recommend EXPLAIN to the"
    r = F.r_note_row("c", "recent", note, "EXPLAIN", len(note) + 9)
    assert (r["cut"], r["mid"], r["r2_changed"], r["r2_empty"], r["r3_changed"], r["r3_empty"]) == (1, 1, 1, 0, 1, 0)
    assert r["r2_note"] == "The figure is high. Recommend EXPLAIN to the" and r["r3_note"] == "The figure is high. It may be seasonal."
    assert (r["r3_fragment_names_action"], r["r3_fragment_names_a_action"], r["names_action"], r["r2_rules"]) == (1, 1, 1, ["H1"])
    whole = F.r_note_row("c", "recent", "The figure is high.", "EXPLAIN", len("The figure is high."))
    assert (whole["cut"], whole["mid"], whole["r2_changed"], whole["r3_changed"]) == (0, 0, 0, 0)
    assert whole["r2_note"] == whole["r3_note"] == "The figure is high."
    assert F.r_note_row("c", "recent", "x.", "EXPLAIN", None)["cut"] is None
    c = F.r_note_counts([r, whole])
    assert (c["cut"], c["mid_sentence"], c["r2"]["changed"], c["r3"]["changed"], c["r3"]["fragment_names_an_action"]) == (1, 1, 1, 1, 1)


@pytest.mark.skipif(not (REAL.exists() and REAL_V5.exists()), reason="needs the frozen v3, v4 and v5 results")
def test_real_frame_stages(tmp_path):
    root = tmp_path / "r"
    meta = F.stage_frame(root, F.V4, F.RESULTS, F.V5)
    assert (meta["n_gated"], meta["n_eligible"], meta["n_a_wrong"], meta["n_cases"]) == (1160, 490, 670, 300)
    assert (meta["notes"]["cut"], meta["notes"]["mid_sentence"], meta["notes"]["cut_and_mid_sentence"]) == (349, 331, 331)
    assert meta["notes"]["r3"]["changed"] == 331 and meta["notes"]["r3"]["empty"] == 0 and meta["v5_refs"]["x1_flips"] == 71
    assert meta["request_shape_check"] == {"v3_note_request": {"n": 490, "found": 490}, "x1_note_request": {"n": 490, "found": 490}}
    r0 = meta["R0"]
    assert r0["all"]["flip"]["k"] == 142 and r0["all"]["defer"]["k"] == 105 and r0["n_cut"] == 349
    assert r0["by_cut"]["flip"]["cut"]["n"] == 349 and r0["by_cut"]["flip"]["not_cut"]["n"] == 141
    wrong = F.load_wrong(root)
    elig = {(h.case.case_id, h.system) for h in F.load_frame(root)}
    assert len(wrong) == 670 and not ({(h.case.case_id, h.system) for h in wrong} & elig)
    assert all(h.row["a_H"] == h.row["a_star"] != h.row["a_V"] and h.view.units for h in wrong)
    refs = F._read_jsonl(root / F.V5_REFS)
    differ = sum(a != h.row["v3"]["a_B"] for r, h in zip(refs, F.load_frame(root)) for a in r["c3_a_B"])
    assert round(differ / (5 * 490), 3) == AN.C3_DIFFER_REPORTED
    assert F.stage_frame(root, F.V4, F.RESULTS, F.V5) == meta  # idempotent
    res = D.dryrun(root, F.RESULTS, write=False)
    assert res["by_sub_arm"]["R1"]["requests"] == 490 and res["by_sub_arm"]["XW"]["requests"] == 4 * 670
    assert res["by_sub_arm"]["X1p"]["requests"] == res["by_sub_arm"]["X1r"]["requests"] == 980


# ---------------------------------------------------------------------------
# A synthetic frame
# ---------------------------------------------------------------------------

NOTES = [
    "The figure is 135.82 against about 80. No service is logged. Recommend the standard reply.",          # unchanged by both rules
    "The figure is high. It may be seasonal. No credit is on file. Recommend a review to verify the new",  # R2 and R3 change it
    "It might be a fault.",                                                                                # R2 empties it
    "Recommend the standard reply because the figure of 80.10 is in line with the",                        # R3 empties it
    "The figure is high. The cause is unclear",                                                            # R2 and R3 give one text
    "The reset explains it. Check whether the visit was logged. The plan changed later.",                  # R2 only
]


def _make_root(tmp_path, n_cases=12):
    """A v7 root on a synthetic frame: even cases are eligible, odd cases are A-wrong, one system."""
    from lens.flip.headline.engine import make_headline_sample
    from lens.flip.systems import RecentSystem
    root = tmp_path / "x7"
    root.mkdir()
    cases = make_headline_sample(n_cases, 5)
    write_jsonl(root / "cases.jsonl", cases)
    frame, wrong = [], []
    for i, c in enumerate(cases):
        v = RecentSystem().view(c, 1000)
        if i % 2:
            a_v = "REQUEST_INFO" if c.a_star != "REQUEST_INFO" else "EXPLAIN"
            wrong.append({"case_id": c.case_id, "system": "recent", "mechanism": c.mechanism, "a_star": c.a_star,
                          "a_H": c.a_star, "a_V": a_v, "view": {"event_ids": v.event_ids}, "source": "synthetic"})
            continue
        flip = (i // 2) % 3 != 2
        frame.append({"case_id": c.case_id, "system": "recent", "mechanism": c.mechanism, "a_star": c.a_star,
                      "a_H": c.a_star, "a_A": c.a_star, "view": {"event_ids": v.event_ids}, "source": "results/synthetic",
                      "v3": {"a_B": ("REQUEST_INFO" if c.a_star != "REQUEST_INFO" else "EXPLAIN") if flip else c.a_star,
                             "a_floor": c.a_star, "a_trunc": c.a_star, "note": NOTES[(i // 2) % len(NOTES)],
                             "note_tokens": 20, "note_status": "ok", "p_residue": 0.2, "omission": 1.0,
                             "residue_flip": flip, "search_status": None, "view_units": len(v.units),
                             "n_candidates": len(v.units), "dstar_in_view": False}})
    (root / "frame.jsonl").write_text(F._jsonl(frame))
    (root / F.WRONG_FRAME).write_text(F._jsonl(wrong))
    hs = F.load_frame(root)
    raw = {(h.case.case_id, h.system): len(h.row["v3"]["note"]) + (7 if P.ends_mid_sentence(h.row["v3"]["note"]) else 0) for h in hs}
    (root / F.R_NOTES).write_text(F._jsonl(F.r_note_rows(hs, raw)))
    refs = [{"case_id": h.case.case_id, "system": h.system, "x1_a_B": h.row["a_A"] if j % 2 else h.row["v3"]["a_B"],
             "x1_note_tokens": 140, "x1_p_residue": 0.1,
             "c3_a_B": [h.row["v3"]["a_B"]] * 4 + [h.row["a_A"]]} for j, h in enumerate(hs)]
    (root / F.V5_REFS).write_text(F._jsonl(refs))
    (root / F.CALIBRATION_NAME).write_text(F.calibration_text())
    return root, hs, F.load_wrong(root)


def _answers(cases):
    by = {c.case_id: c for c in cases}

    def answer(cid, params, meta):
        case = by[meta["case_id"]]
        user = params["messages"][0]["content"]
        h = zlib.crc32(cid.encode())
        if meta["kind"] == "decide":
            act = case.a_star if h % 3 else ("REQUEST_INFO" if case.a_star != "REQUEST_INFO" else "EXPLAIN")
            text = json.dumps({"action": act, "reason": "mock", "event_ids": []})
        else:
            assert meta["kind"] == "note"
            text = json.dumps({"note": f"Recommend {case.a_star}. Short note {h % 97}.", "p_residue": (h % 100) / 100})
        return {"custom_id": cid, "result": {"type": "succeeded", "message": {
            "content": [{"type": "text", "text": text}], "stop_reason": "end_turn",
            "usage": {"input_tokens": len(user) // 2 + 331, "output_tokens": 100}}}}
    return answer


def _run(root, arm, cap=100.0, check=False, **kw):
    d = root / arm
    if not d.exists():
        d = E.init_arm(arm, root)
    api = FakeBatchAPI(_answers(F.load_cases(root)), engine_meta_fn(d), **kw)
    r = E.run_arm(d, api, cap_usd=cap, root=root, log=lambda m: None, sleep=lambda s: None, check_protocol=check,
                  max_steps=80)
    return r, api, d


def _all_meta(root):
    def read():
        out = {}
        for arm in E.ARMS:
            p = root / arm / "state" / "round.json"
            if p.exists():
                out.update(json.loads(p.read_text())["requests"])
        return out
    return read


def _finalized(root, tmp_path):
    D.dryrun(root, tmp_path / "no-results")
    PR.freeze(root, now=0)
    return PR.finalize(root, now=0)


# ---------------------------------------------------------------------------
# Study R end to end
# ---------------------------------------------------------------------------

def test_r_sends_changed_notes_only_and_keeps_v3_for_the_rest(tmp_path):
    root, hs, _ = _make_root(tmp_path)
    notes = F.by_key(F._read_jsonl(root / F.R_NOTES))
    assert [(notes[(h.case.case_id, h.system)]["r2_changed"], notes[(h.case.case_id, h.system)]["r2_empty"],
             notes[(h.case.case_id, h.system)]["r3_changed"], notes[(h.case.case_id, h.system)]["r3_empty"]) for h in hs] == \
        [(0, 0, 0, 0), (1, 0, 1, 0), (1, 1, 0, 0), (0, 0, 1, 1), (1, 0, 1, 0), (1, 0, 0, 0)]
    d = E.init_arm("R", root)
    seen: list[dict] = []
    api = FakeBatchAPI(_answers(F.load_cases(root)), engine_meta_fn(d), on_create=seen.extend)
    assert E.run_arm(d, api, cap_usd=100.0, root=root, log=lambda m: None, sleep=lambda s: None, check_protocol=False,
                     max_steps=20) == "complete"
    rnd = json.loads((d / "rounds" / "001" / "round.json").read_text())["requests"]
    stages = [m["stage"] for m in rnd.values()]
    # R1 on all six; R2 on three changed and not emptied; R3 on two, one of which is the same text as R2's
    assert api.created == 1 and stages.count("r1_B") == 6 and stages.count("r2_B") == 3 and stages.count("r3_B") == 1
    assert api.requests_sent == 10 == len(set(api.sent_custom_ids))
    body = {r["custom_id"]: r["params"] for r in seen}
    dec = E.arm_config("R")["decider"]
    st = AN._states(root, "R")
    for h in hs:
        k = (h.case.case_id, h.system)
        rn, o, v3 = notes[k], st[k]["out"], h.row["v3"]
        mine = {m["stage"]: body[cid] for cid, m in rnd.items() if m["case_id"] == k[0]}
        assert mine["r1_B"] == E.r1_request(v3["note"], h, dec) and mine["r1_B"]["system"] == P.R1_SYSTEM
        assert o["r1"]["source"] == "new"
        for arm in ("r2", "r3"):
            if rn[f"{arm}_empty"]:
                assert o[arm] == {"a_B": None, "status": None, "source": "none", "changed": True, "empty": True}
            elif not rn[f"{arm}_changed"]:
                assert o[arm] == {"a_B": v3["a_B"], "status": "v3", "source": "v3", "changed": False, "empty": False}
            else:
                assert o[arm]["source"] == "new" and o[arm]["a_B"] in ACTIONS
                want = E.v1_request(rn[f"{arm}_note"], h, dec)
                assert rq.request_key(want) in {m["key"] for m in rnd.values()} and want["system"] != P.R1_SYSTEM
    same = notes[(hs[4].case.case_id, hs[4].system)]
    assert same["r2_note"] == same["r3_note"]  # one body, one request, the same answer for both arms
    o = st[(hs[4].case.case_id, hs[4].system)]["out"]
    assert o["r2"]["a_B"] == o["r3"]["a_B"]


# ---------------------------------------------------------------------------
# Study X end to end, with the round size limit
# ---------------------------------------------------------------------------

def test_x_notes_before_decisions_and_rounds_hold_the_limit(tmp_path, monkeypatch):
    root, hs, wrong = _make_root(tmp_path)
    monkeypatch.setattr(E, "MAX_REQUESTS_PER_ROUND", 7)
    d = E.init_arm("X", root)
    assert json.loads((d / "config.json").read_text())["max_requests_per_round"] == 7
    seen: list[dict] = []
    api = FakeBatchAPI(_answers(F.load_cases(root)), engine_meta_fn(d), on_create=seen.extend)
    assert E.run_arm(d, api, cap_usd=100.0, root=root, log=lambda m: None, sleep=lambda s: None, check_protocol=False,
                     max_steps=80) == "complete"
    n = len(hs) + len(wrong)
    assert api.requests_sent == 4 * n == len(set(api.sent_custom_ids)) and api.created >= -(-4 * n // 7)
    rounds = [json.loads(p.read_text()) for p in sorted((d / "rounds").glob("*/round.json"))]
    assert all(r["n_requests"] <= 7 for r in rounds) and sum(r["n_requests"] for r in rounds) == 4 * n
    first: dict[tuple[str, str], int] = {}
    for r in rounds:
        for m in r["requests"].values():
            first.setdefault((m["case_id"], m["stage"]), r["round"])
    for (cid, stage), rno in first.items():
        if stage.endswith("_B"):
            assert first[(cid, stage[:-2] + "_A")] < rno  # a note is answered before the decision that reads it
    body = {r["custom_id"]: r["params"] for r in seen}
    metas = {cid: m for r in rounds for cid, m in r["requests"].items()}
    nw, dec = E.arm_config("X")["note_writer"], E.arm_config("X")["decider"]
    st = AN._states(root, "X")
    for h, kind in [(h, "eligible") for h in hs] + [(h, "a_wrong") for h in wrong]:
        o = st[(h.case.case_id, h.system)]["out"]
        mine = {m["stage"]: body[cid] for cid, m in metas.items() if m["case_id"] == h.case.case_id}
        assert o["set"] == kind
        if kind == "eligible":
            assert set(mine) == {"x1p_A", "x1p_B", "x1r_A", "x1r_B"} and set(o) == {"set", "x1p", "x1r"}
            assert mine["x1p_A"] == E.x1p_note_request(h, nw) and mine["x1r_A"] == E.x1r_note_request(h, nw)
            assert mine["x1p_B"] == E.v1_request(o["x1p"]["note"], h, dec) and mine["x1r_B"] == E.v1_request(o["x1r"]["note"], h, dec)
        else:
            assert set(mine) == {"xw_v3_A", "xw_v3_B", "xw_x1_A", "xw_x1_B"} and set(o) == {"set", "v3", "x1"}
            assert mine["xw_v3_A"] == rq.note_request(h.view, h.case, nw) and mine["xw_x1_A"] == E5.x1_request(h.view, h.case, nw)
            assert mine["xw_v3_B"] == E.v1_request(o["v3"]["note"], h, dec) and mine["xw_x1_B"] == E.v1_request(o["x1"]["note"], h, dec)
        for t in set(o) - {"set"}:
            assert o[t]["a_B"] in ACTIONS and o[t]["note"].startswith("Recommend") and o[t]["p_residue"] is not None
    # resume: a finished run submits nothing
    assert E.run_arm(d, api, cap_usd=100.0, root=root, log=lambda m: None, sleep=lambda s: None, check_protocol=False) == "complete"
    assert api.requests_sent == 4 * n


def test_resume_pays_once_and_cap_refuses(tmp_path):
    root, hs, wrong = _make_root(tmp_path)
    d = E.init_arm("X", root)
    api = FakeBatchAPI(_answers(F.load_cases(root)), engine_meta_fn(d))
    kw = dict(cap_usd=50, root=root, log=lambda m: None, sleep=lambda s: None, check_protocol=False)
    assert E.run_arm(d, api, max_steps=1, **kw) == "submitted"      # round 1 in flight, the process "dies"
    assert E.run_arm(d, api, max_steps=40, **kw) == "complete"      # a new engine picks the round up
    assert api.created == 2 and api.requests_sent == 4 * (len(hs) + len(wrong))
    spent = E.update_ledger(root, 50.0)["total_usd"]
    assert spent > 0 and set(E.update_ledger(root, 50.0)["arms"]) == {"X"}
    d2 = E.init_arm("R", root)
    api2 = FakeBatchAPI(lambda *a: None, engine_meta_fn(d2))
    with pytest.raises(Refused):  # a cap just above what X spent leaves no room for R's round
        E.run_arm(d2, api2, cap_usd=spent + 1e-6, root=root, log=lambda m: None, sleep=lambda s: None,
                  check_protocol=False, max_steps=5)
    assert api2.created == 0
    with pytest.raises(Refused, match="already holds"):
        E.init_arm("X", root)


# ---------------------------------------------------------------------------
# Freeze, finalize, check, CLI
# ---------------------------------------------------------------------------

def test_freeze_is_draft_then_final_and_check_catches_changes(tmp_path, monkeypatch):
    root, hs, wrong = _make_root(tmp_path)
    with pytest.raises(SystemExit, match="dryrun"):
        PR.freeze(root)
    D.dryrun(root, tmp_path / "no-results")
    rec = PR.freeze(root, now=0)
    assert rec["status"] == "DRAFT" and rec["version"] == "7-exploratory" and set(rec["arm_configs"]) == {"R", "X"}
    assert set(rec["rules"]) == {"x7.r1.system", "x7.r1.prompt", "x7.r1.diff", "x7.r2.rule", "x7.r3.rule", "x7.x1.split",
                                 "x7.x1p.prompt", "x7.x1r.prompt", "x7.x1.schema", "x7.sets.rule", "x7.rounds.rule", "x7.analysis"}
    for key, name in PR.RUN_FILES.items():
        assert rec[key] == hashlib.sha256((root / name).read_bytes()).hexdigest()
    assert PR.check(root) == ["status:DRAFT"]
    with pytest.raises(Refused, match="status:DRAFT"):
        E.run_arms(root, ["R"], cap_usd=10, runner=FakeBatchAPI(lambda *a: None), log=lambda m: None, sleep=lambda s: None)
    assert not (root / "R").exists()
    p = root / PR.NAME
    p.write_text(json.dumps({**rec, "dryrun": {**rec["dryrun"], "likely_usd": PR.PRICE_GATE_USD + 0.01}}))
    with pytest.raises(SystemExit, match="above"):
        PR.finalize(root)
    p.write_text(json.dumps({**rec, "dryrun": {**rec["dryrun"], "guard": {**rec["dryrun"]["guard"], "fits": False}}}))
    with pytest.raises(SystemExit, match="does not fit"):
        PR.finalize(root)
    p.write_text(json.dumps(rec))
    assert PR.finalize(root, now=0)["status"] == "FINAL"
    d = E.init_arm("R", root)
    assert PR.check(root) == [] and PR.check(root, d) == []
    monkeypatch.setattr(P, "R1_SYSTEM", P.R1_SYSTEM + " Extra.")
    assert PR.check(root) == ["rule:x7.r1.diff", "rule:x7.r1.system"]
    monkeypatch.undo()
    monkeypatch.setattr(P, "HEDGE_H4", P.HEDGE_H4 + "|maybe")
    assert PR.check(root) == ["rule:x7.r2.rule"]
    monkeypatch.undo()
    monkeypatch.setitem(AN.ANALYSIS_SPEC, "multiple_comparisons", "something")
    assert PR.check(root) == ["rule:x7.analysis"]
    monkeypatch.undo()
    monkeypatch.setattr(E, "MAX_REQUESTS_PER_ROUND", 5)
    assert {"rule:x7.rounds.rule", "arm:R", "arm:X"} <= set(PR.check(root))
    monkeypatch.undo()
    (d / F.R_NOTES).write_text((d / F.R_NOTES).read_text() + "\n")
    assert PR.check(root, d) == [f"run:{F.R_NOTES}"]
    (root / F.WRONG_FRAME).write_text((root / F.WRONG_FRAME).read_text() + "\n")
    assert "root:wrong_frame_sha256" in PR.check(root)


def test_cli_guards(tmp_path):
    from lens.flip.explore7.__main__ import main, parse_arms
    with pytest.raises(SystemExit, match="price-confirmed"):
        main(["run", "--results", str(tmp_path), "--cap-usd", "3"])
    with pytest.raises(SystemExit, match="cap-usd"):
        main(["run", "--results", str(tmp_path), "--price-confirmed"])
    with pytest.raises(SystemExit, match="may not pass"):
        main(["run", "--results", str(tmp_path), "--price-confirmed", "--cap-usd", "4.57"])
    with pytest.raises(SystemExit, match="price-confirmed"):
        main(["finalize", "--results", str(tmp_path)])
    with pytest.raises(SystemExit):
        parse_arms("R,R1")
    assert parse_arms("X,R") == ["X", "R"] and parse_arms(None) == ["R", "X"]


def test_dryrun_counts_and_guard_preview(tmp_path):
    root, hs, wrong = _make_root(tmp_path)
    res = D.dryrun(root, tmp_path / "no-results")
    assert res["api_calls"] == 0 and res["count_tokens_calls"] == 0 and res["submitted"] is False
    assert res["by_sub_arm"]["R1"]["requests"] == 6 and res["by_sub_arm"]["R2"]["requests"] == 3 and res["by_sub_arm"]["R3"]["requests"] == 1
    assert res["by_sub_arm"]["X1p"]["requests"] == res["by_sub_arm"]["X1r"]["requests"] == 2 * len(hs)
    assert res["by_sub_arm"]["XW"]["requests"] == 4 * len(wrong)
    assert res["total"]["requests"] == sum(x["requests"] for x in res["by_sub_arm"].values())
    assert res["total"]["likely_usd"] == pytest.approx(sum(x["likely_usd"] for x in res["by_sub_arm"].values()), abs=1e-3)
    assert 0 < res["total"]["likely_usd"] < res["total"]["likely_usd_if_outputs_x1_5"] < res["total"]["upper_usd"]
    assert res["guard"]["fits"] and res["guard"]["max_requests_per_round"] == 1000
    assert not D.dryrun(root, tmp_path / "no-results", write=False, cap_usd=1e-6)["guard"]["fits"]
    for arm in E.ARMS:
        r, api, d = _run(root, arm)
        assert r == "complete" and api.requests_sent == res["arms"][arm]["requests"]
    est = json.loads((root / "R" / "round_estimates.json").read_text())
    assert len(est) == 1 and est[0]["upper_bound_usd"] == pytest.approx(res["arms"]["R"]["upper_usd"], abs=1e-3)
    assert est[0]["likely_usd"] == pytest.approx(res["arms"]["R"]["likely_usd"], abs=1e-3)


# ---------------------------------------------------------------------------
# Analysis on a mock
# ---------------------------------------------------------------------------

def _row(case, kind="eligible", **kw):
    r = {"case_id": f"c{case}", "case_index": case, "n_cases": 5, "system": "recent", "mechanism": "M1", "mechanism_group": "M1-M3",
         "set": kind, "a_star": "EXPLAIN", "a_A": "EXPLAIN" if kind == "eligible" else "REQUEST_INFO",
         **{k: None for k in ("v3_a_B", "cut", "mid", "note_names_action", "frag_names_action", "c3_n", "c3_flip", "c3_defer",
                              "c3_commit", "c3_differ", "x1_a_B", "r2_changed", "r2_empty", "r3_changed", "r3_empty") + AN.ARM_FIELDS}}
    if kind == "eligible":
        r.update({"v3_a_B": "EXPLAIN", "cut": 0, "mid": 0, "note_names_action": 1, "frag_names_action": 0, "c3_n": 5, "c3_flip": 0,
                  "c3_defer": 0, "c3_commit": 0, "c3_differ": 0, "x1_a_B": "EXPLAIN", "r1_a_B": "EXPLAIN", "r2_changed": 0,
                  "r2_empty": 0, "r2_a_B": "EXPLAIN", "r3_changed": 0, "r3_empty": 0, "r3_a_B": "EXPLAIN", "x1p_a_B": "EXPLAIN",
                  "x1r_a_B": "EXPLAIN", "x1p_note_tokens": 100, "x1r_note_tokens": 120, "x1p_p": 0.1, "x1r_p": 0.2,
                  "x1p_note_status": "ok", "x1r_note_status": "ok"})
    else:
        r.update({"wv3_a_B": "REQUEST_INFO", "wx1_a_B": "EXPLAIN", "wv3_note_tokens": 110, "wx1_note_tokens": 130,
                  "wv3_note_status": "ok", "wx1_note_status": "ok"})
    r.update(kw)
    return r


def _toy():
    D_, C_ = "REQUEST_INFO", "OPEN_REVIEW"
    return [
        # v3 deferral flip on a cut note; R1 and R3 clear it, R2 unchanged keeps it; X1p clears, X1r keeps
        _row(0, v3_a_B=D_, cut=1, mid=1, c3_flip=4, c3_defer=4, c3_differ=1, r1_a_B="EXPLAIN", r2_a_B=D_, r3_changed=1,
             r3_a_B="EXPLAIN", frag_names_action=1, x1_a_B="EXPLAIN", x1p_a_B="EXPLAIN", x1r_a_B=D_),
        # v3 commit flip, not cut; R1 turns it into a deferral; R2 changed and clears it
        _row(0, v3_a_B=C_, c3_flip=5, c3_commit=5, r1_a_B=D_, r2_changed=1, r2_a_B="EXPLAIN", r3_a_B=C_, x1_a_B=C_,
             x1p_a_B=C_, x1r_a_B=C_),
        # no v3 flip on a cut note; R3 changed and makes a deferral; R2 emptied
        _row(1, cut=1, mid=1, r2_changed=1, r2_empty=1, r2_a_B=None, r3_changed=1, r3_a_B=D_),
        # v3 deferral flip, cut; nothing changes it; the reply of R1 is unparsed
        _row(2, v3_a_B=D_, cut=1, c3_flip=5, c3_defer=5, r1_a_B="PARSE_ERROR", r2_a_B=D_, r3_a_B=D_, x1_a_B=D_, x1p_a_B=D_, x1r_a_B=D_),
        _row(3),
        # A-wrong: v3 note repeats A's deferral, X1 note rescues
        _row(3, "a_wrong"),
        # A-wrong: both rescue
        _row(4, "a_wrong", wv3_a_B="EXPLAIN"),
        # A-wrong: v3 note gives a third action, X1 note repeats A
        _row(4, "a_wrong", wv3_a_B="OPEN_REVIEW", wx1_a_B="REQUEST_INFO"),
    ]


def test_analysis_toy_arithmetic():
    s = AN.summarize(_toy())
    assert s["denominators"] == {"cases": 5, "gated_handoffs": 8, "eligible": 5, "a_wrong": 3,
                                 "eligible_by_system": {"recent": 5}, "a_wrong_by_system": {"recent": 3}}
    R = s["study_R"]
    r0 = R["R0"]
    assert (r0["n"], r0["n_cut"], r0["n_mid_sentence"]) == (5, 3, 2)
    assert r0["all"]["flip"]["k"] == 3 and r0["all"]["defer"]["k"] == 2 and r0["all"]["commit"]["k"] == 1
    assert r0["by_cut"]["flip"]["cut"]["value"] == pytest.approx(2 / 3) and r0["by_cut"]["flip"]["not_cut"]["value"] == pytest.approx(1 / 2)
    assert r0["by_cut"]["flip"]["cut_minus_not_cut"]["diff"] == pytest.approx(2 / 3 - 1 / 2)
    assert r0["by_mid"]["defer"]["mid_sentence_minus_ends_a_sentence"]["diff"] == pytest.approx(1 / 2 - 1 / 3)
    r1 = R["R1_note_framed"]
    assert (r1["n_with_decision"], r1["n_changed"], r1["n_new_decisions"], r1["unparsed_replies"]) == (5, 5, 5, 1)
    assert r1["all"]["flip_rate"]["k"] == 2 and r1["all"]["v3_flip_rate"]["k"] == 3      # the deferral made and the unparsed reply
    assert r1["all"]["flip_minus_v3"]["diff"] == pytest.approx(-1 / 5)
    assert r1["all"]["defer_minus_v3"]["diff"] == pytest.approx((1 - 2) / 5)
    assert r1["all"]["commit_minus_v3"]["diff"] == pytest.approx((0 - 1) / 5)
    assert r1["all"] == r1["changed_only"]
    assert r1["by_cut"]["cut"]["n"] == 3 and r1["by_cut"]["cut"]["flip_minus_v3"]["diff"] == pytest.approx((1 - 2) / 3)
    assert r1["by_cut"]["not_cut"]["flip_minus_v3"]["diff"] == pytest.approx(0.0)
    assert r1["noise_floor"]["answers_that_differ_from_v3_on_changed"]["value"] == pytest.approx(3 / 5)
    assert r1["noise_floor"]["c3_repeats_that_differ_from_v3_same_handoffs"]["value"] == pytest.approx(1 / 25)
    assert r1["noise_floor"]["all"]["flip_minus_c3"]["diff"] == pytest.approx(2 / 5 - 14 / 25)
    assert r1["verdict"] in ("reduces deferral", "does not") and r1["rule"] == AN.R_RULE
    r2 = R["R2_hedges_stripped"]
    assert (r2["n_handoffs"], r2["n_emptied_not_sent"], r2["n_with_decision"], r2["n_changed"], r2["n_new_decisions"],
            r2["n_v3_decision_kept"]) == (5, 1, 4, 2, 1, 3)
    assert r2["all"]["flip_minus_v3"]["diff"] == pytest.approx((2 - 3) / 4) and r2["all"]["defer_minus_v3"]["diff"] == pytest.approx(0.0)
    assert r2["changed_only"]["n"] == 1 and r2["changed_only"]["commit_minus_v3"]["diff"] == pytest.approx(-1.0)
    r3 = R["R3_fragment_removed"]
    assert (r3["n_changed"], r3["n_new_decisions"], r3["n_emptied_not_sent"]) == (2, 2, 0)
    assert r3["all"]["defer_minus_v3"]["diff"] == pytest.approx((2 - 2) / 5) and r3["changed_only"]["flip_minus_v3"]["diff"] == pytest.approx(0.0)
    fr = r3["by_fragment_names_action_descriptive"]
    assert fr["fragment_names_an_action"]["n"] == 1 and fr["fragment_names_an_action"]["flip_minus_v3"]["diff"] == pytest.approx(-1.0)
    assert fr["fragment_names_no_action"]["flip_minus_v3"]["diff"] == pytest.approx(1.0)
    c3 = R["c3_reference"]
    assert c3["flip_rate"]["value"] == pytest.approx(14 / 25) and c3["flip_minus_v3"]["diff"] == pytest.approx(14 / 25 - 3 / 5)
    X = s["study_X"]
    one = X["1_split_on_eligible"]
    assert [one[f"{a}_flip_rate"]["k"] for a in ("v3", "x1", "x1p", "x1r")] == [3, 2, 2, 3]
    assert one["x1p_minus_v3"]["diff"] == pytest.approx(-1 / 5) and one["x1r_minus_v3"]["diff"] == pytest.approx(0.0)
    assert one["x1p_minus_x1r"]["diff"] == pytest.approx(-1 / 5) and one["x1p_minus_x1"]["diff"] == pytest.approx(0.0)
    assert one["verdict"] in ("comes with the policy", "comes with the reader description", "both", "neither separable")
    assert one["notes"]["x1p"] == {"n": 5, "mean_tokens": 100.0, "status": {"ok": 5}, "p_residue_missing": 0}
    two = X["2_a_wrong"]
    assert two["rescue_v3_notes"]["k"] == 1 and two["rescue_x1_notes"]["k"] == 2 and two["rescue_x1_minus_v3"]["diff"] == pytest.approx(1 / 3)
    assert two["repeat_v3_notes"]["k"] == 1 and two["repeat_x1_notes"]["k"] == 1 and two["repeat_x1_minus_v3"]["diff"] == pytest.approx(0.0)
    assert two["b_actions"]["v3_notes"] == {"EXPLAIN": 1, "OPEN_REVIEW": 1, "REQUEST_INFO": 1}
    three = X["3_two_sided"]
    assert three["n"] == 8 and three["accuracy_v3_notes"]["k"] == 2 + 1 and three["accuracy_x1_notes"]["k"] == 3 + 2
    assert three["x1_minus_v3"]["diff"] == pytest.approx(2 / 8)
    assert three["by_set"]["eligible"]["x1_minus_v3"]["diff"] == pytest.approx(1 / 5)
    assert three["by_set"]["a_wrong"]["x1_minus_v3"]["diff"] == pytest.approx(1 / 3)


def test_reading_rules_use_the_fixed_words():
    assert AN.r_verdict({"ci95": [-0.2, -0.01]}) == "reduces deferral"
    assert AN.r_verdict({"ci95": [-0.2, 0.0]}) == "does not" and AN.r_verdict({"ci95": [0.01, 0.2]}) == "does not"
    assert AN.r_verdict({"ci95": None}) == "does not"
    lo, no = {"ci95": [-0.2, -0.01]}, {"ci95": [-0.2, 0.01]}
    assert AN.x_verdict(lo, no) == "comes with the policy" and AN.x_verdict(no, lo) == "comes with the reader description"
    assert AN.x_verdict(lo, lo) == "both" and AN.x_verdict(no, no) == "neither separable"
    assert AN.x_verdict({"ci95": None}, no) == "neither separable"
    assert AN.outcome("REQUEST_INFO", "EXPLAIN", "a") == {"a_flip": 1, "a_defer": 1, "a_commit": 0, "a_unparsed": 0}
    assert AN.outcome("OPEN_REVIEW", "EXPLAIN", "a") == {"a_flip": 1, "a_defer": 0, "a_commit": 1, "a_unparsed": 0}
    assert AN.outcome("PARSE_ERROR", "EXPLAIN", "a") == {"a_flip": 1, "a_defer": 0, "a_commit": 0, "a_unparsed": 1}
    assert AN.outcome("REQUEST_INFO", "REQUEST_INFO", "a")["a_flip"] == 0 and AN.outcome(None, "EXPLAIN", "a")["a_flip"] == 0
    # a clear case for the R rule: every handoff defers under v3 and none under the arm
    rows = [_row(i % 5, v3_a_B="REQUEST_INFO", c3_flip=5, c3_defer=5) for i in range(10)]
    assert AN.summarize(rows)["study_R"]["R1_note_framed"]["verdict"] == "reduces deferral"


def test_metric_rows_are_text_free_and_rebuild_the_summary(tmp_path):
    root, hs, wrong = _make_root(tmp_path)
    _finalized(root, tmp_path)
    with pytest.raises(SystemExit, match="not complete"):
        AN.metric_rows(root)
    out = E.run_arms(root, ["X", "R"], cap_usd=100.0, runner=FakeBatchAPI(_answers(F.load_cases(root)), _all_meta(root)),
                     log=lambda m: None, sleep=lambda s: None)
    assert out == {"R": "complete", "X": "complete"} and list(out) == ["R", "X"]
    s = AN.write(root)
    mpath = root / AN.METRICS_NAME
    rows = AN.read_metrics(mpath)
    assert len(rows) == len(hs) + len(wrong) == s["metrics_file"]["rows"]
    assert [r["set"] for r in rows] == ["eligible"] * len(hs) + ["a_wrong"] * len(wrong)
    text = mpath.read_text()
    assert not any(n[:25] in text for n in NOTES) and "Short note" not in text and "Recommend" not in text
    strings = {k for r in rows for k, v in r.items() if isinstance(v, str)}
    assert strings <= {"case_id", "system", "mechanism", "mechanism_group", "set", "a_star", "a_A", "v3_a_B", "x1_a_B", "r1_a_B",
                       "r2_a_B", "r3_a_B", "x1p_a_B", "x1r_a_B", "wv3_a_B", "wx1_a_B", "x1p_note_status", "x1r_note_status",
                       "wv3_note_status", "wx1_note_status"}
    assert all(len(v) <= 32 for r in rows for v in r.values() if isinstance(v, str))
    body = AN.rebuild(mpath)
    assert body == {k: s[k] for k in body}  # every number comes from the metric file alone
    R = s["study_R"]
    assert R["R2_hedges_stripped"]["n_emptied_not_sent"] == 1 and R["R3_fragment_removed"]["n_emptied_not_sent"] == 1
    assert R["R2_hedges_stripped"]["n_new_decisions"] == 3 and R["R3_fragment_removed"]["n_new_decisions"] == 2
    sp = s["spend"]["by_arm"]
    assert sp["R1"]["requests"] == 6 and sp["R2"]["requests"] == 3 and sp["R3"]["requests"] == 1  # the shared body is paid once
    assert sp["X1p"]["requests"] == sp["X1r"]["requests"] == 2 * len(hs) and sp["XW"]["requests"] == 4 * len(wrong)
    assert s["spend"]["total_usd"] == pytest.approx(sum(v["actual_usd"] for v in sp.values()), abs=1e-5)
    assert s["protocol"]["status"] == "FINAL" and "X(3)" in AN.headline(s)
    with pytest.raises(AssertionError, match="long string"):
        AN.check_rows_text_free(json.dumps({"x": "y" * 33}) + "\n")
    with pytest.raises(AssertionError, match="banned"):
        AN.check_rows_text_free(json.dumps({"x": "Zebra"}) + "\n", ["zebra"])


# ---------------------------------------------------------------------------
# The private word list
# ---------------------------------------------------------------------------

def test_private_word_list_absent_from_v7_code_rules_and_docs():
    banned_file = REPO / "banned_words.local.txt"
    if not banned_file.exists():
        pytest.skip("banned_words.local.txt not present")
    banned = [w.strip().lower() for w in banned_file.read_text().splitlines() if w.strip()]
    files = sorted((REPO / "src" / "lens" / "flip" / "explore7").glob("*.py")) + [Path(__file__)]
    files += [p for p in (REPO / "docs" / "flip-protocol-v7-exploratory.md",
                          *sorted((REPO / "docs").glob("exploratory-v7-results-*.md"))) if p.exists()]
    blob = "\n".join(f.read_text().lower() for f in files)
    blob += "\n".join(PR.rule_texts().values()).lower()
    blob += rq.canonical({a: E.arm_config(a) for a in E.ARMS}).lower()
    assert [w for w in banned if w in blob] == []

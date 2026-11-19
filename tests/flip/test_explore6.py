"""Exploratory protocol v6: the E1 draw rule, frame staging, the re-read salt, both arms end to end on a fake
Batches API, the spend guard, the freeze, the dry run and the analysis on a mock. No network.

The end-to-end tests run on a small synthetic frame built here from the generator, so they need no stored
results. The checks against the real frame skip when results/ is absent (it is git-ignored).
"""

from __future__ import annotations

import hashlib
import json
import zlib
from pathlib import Path

import pytest

from lens.flip.explore5 import controls as CT5
from lens.flip.explore5 import engine as E5
from lens.flip.explore5 import protocol as PR5
from lens.flip.explore6 import analysis as AN
from lens.flip.explore6 import controls as CT
from lens.flip.explore6 import dryrun as D
from lens.flip.explore6 import engine as E
from lens.flip.explore6 import frame as F
from lens.flip.explore6 import protocol as PR
from lens.flip.headline import requests as rq
from lens.flip.headline.engine import Refused, _read_json
from lens.flip.headline.mock import FakeBatchAPI, engine_meta_fn
from lens.flip.residue import note_view, residue_candidates
from lens.flip.schema import Unit, write_jsonl

REPO = Path(__file__).resolve().parents[2]
REAL_V4 = F.V4 / "frame.jsonl"
REAL_X = F.V4 / F.X_ARM / "state" / "cases"


def _units(lengths):
    return [Unit(id=f"e{i:04d}", text="x" * n) for i, n in enumerate(lengths)]


# ---------------------------------------------------------------------------
# The draw rule
# ---------------------------------------------------------------------------

def test_e1_draw_is_the_v5_rule_with_the_x6_seed():
    pool_all = _units([40, 55, 70, 90, 120, 35, 60, 80, 45, 100, 150, 30])
    minimal = {"e0003", "e0004"}  # 90 + 120 characters: 211 with the newline
    rec = {"status": "restored", "unit_ids": sorted(minimal), "chars": 211}
    rows = CT.build_controls("c", "recent", pool_all, rec, median_chars=100.0)
    assert len(rows) == CT.K_CONTROLS == 5 and CT.SEED_TAG == "x6-c1"
    assert CT.control_seed("case9", "recent", 2) == "x6-c1|case9|recent|2"
    assert (CT.TOLERANCE, CT.MAX_ATTEMPTS) == (CT5.TOLERANCE, CT5.MAX_ATTEMPTS) == (0.10, 200)
    by = {u.id: u for u in pool_all}
    pool = [u for u in pool_all if u.id not in minimal]
    seen = []
    for r in rows:
        assert not (set(r["unit_ids"]) & minimal)
        assert r["matched"] and 0.9 * 211 <= r["chars"] <= 1.1 * 211 and r["send"]
        assert r["chars"] == CT5.block_chars([by[i].text for i in r["unit_ids"]])
        assert r["unit_ids"] == sorted(r["unit_ids"]) and r["target_source"] == "minimal_set"
        # the same walk as the v5 function gives with this seed and the earlier controls to avoid
        one = CT5.draw_control(pool, 211, f"x6-c1|c|recent|{r['k']}", avoid=seen)
        assert one["unit_ids"] == r["unit_ids"] and r["seed"] == f"x6-c1|c|recent|{r['k']}"
        seen.append(frozenset(r["unit_ids"]))
    assert rows == CT.build_controls("c", "recent", pool_all, rec, 100.0)             # seeded
    assert rows != CT.build_controls("d", "recent", pool_all, rec, 100.0)             # by case
    v5 = CT5.build_controls("c", "recent", pool_all, rec, 100.0)
    assert [r["unit_ids"] for r in v5] != [r["unit_ids"] for r in rows]               # not the x5 stream


def test_e1_draw_non_recoverable_uses_median_and_whole_pool():
    pool = _units([60] * 8)
    rec = {"status": "not_recoverable_by_insertion", "unit_ids": [], "chars": None}
    med = CT.median_recoverable_chars([{"status": "restored", "chars": c} for c in (100, 200, 400)] + [rec])
    assert med == 200.0
    rows = CT.build_controls("c", "s", pool, rec, med)
    assert all(r["target_source"] == "median_of_recoverable" and r["target_chars"] == 200 and r["pool_size"] == 8
               for r in rows)
    assert all(r["matched"] and 180 <= r["chars"] <= 220 for r in rows)
    # one feasible set only: the first control is sent, the rest are copies and are not
    rows = CT.build_controls("c", "r", _units([100, 100]), {"status": "restored", "unit_ids": ["e0000"], "chars": 100}, 100.0)
    assert [r["send"] for r in rows] == [True, False, False, False, False] and rows[1]["duplicate"]
    # an empty pool: nothing to send, nothing matched
    rows = CT.build_controls("c", "r", _units([100]), {"status": "restored", "unit_ids": ["e0000"], "chars": 100}, 100.0)
    assert all(not r["send"] and not r["matched"] and r["unit_ids"] == [] for r in rows)
    with pytest.raises(ValueError):
        CT.build_controls("c", "r", _units([10]), {"status": "restored", "unit_ids": ["e9999"], "chars": 5}, 5.0)


def test_reread_salt_is_the_v5_c3_rule():
    assert CT.salted_key is CT5.salted_key and CT.REREADS == 5
    assert [CT.reread_salt(i) for i in range(5)] == ["t0", "t1", "t2", "t3", "t4"]
    body = rq.canonical({"a": 1})
    keys = {CT.salted_key(body, CT.reread_salt(i)) for i in range(5)}
    assert len(keys) == 5 and rq.request_key({"a": 1}) not in keys
    assert CT.salted_key(body, "t0") == hashlib.sha256((body + "\x00repeat=t0").encode("ascii")).hexdigest()


# ---------------------------------------------------------------------------
# A synthetic frame
# ---------------------------------------------------------------------------

def _frame_rows(cases):
    from lens.flip.systems import RecentSystem
    frame = []
    for i, c in enumerate(cases):
        v = RecentSystem().view(c, 1000)
        flip = i % 3 != 2
        recov = flip and i % 2 == 0
        frame.append({"case_id": c.case_id, "system": "recent", "mechanism": c.mechanism, "a_star": c.a_star,
                      "a_H": c.a_star, "a_A": c.a_star, "view": {"event_ids": v.event_ids}, "source": "results/synthetic",
                      "v3": {"a_B": "REQUEST_INFO" if flip else c.a_star, "a_floor": c.a_star if i % 4 else "REQUEST_INFO",
                             "a_trunc": c.a_star,
                             "note": f"Recommend {c.a_star}. The records show {len(v.units)} items for {c.query_period}.",
                             "note_tokens": 20, "note_status": "ok", "p_residue": 0.1 + 0.05 * (i % 7), "omission": 1.0,
                             "residue_flip": flip, "search_status": ("restored" if recov else "not_recoverable_by_insertion")
                             if flip else None, "view_units": len(v.units), "n_candidates": len(v.units),
                             "dstar_in_view": False}})
    return frame


def _x_state(h, i):
    """What v4 arm A-extractive stored for one handoff: an extractive note of the first two units, and a search
    record on a flip (i % 4 != 3), restored for i % 4 in (0, 1)."""
    note = "\n".join(h.view.units[:2])
    cands = residue_candidates(h.case, h.view, note)
    flip = i % 4 != 3
    out = {"note": note, "a_B": "REQUEST_INFO" if flip else h.row["a_A"], "residue_flip": flip,
           "candidate_ids": [u.id for u in cands], "n_candidates": len(cands)}
    ns = {"out": out, "done": True}
    if flip:
        restored = i % 4 in (0, 1)
        pick = ([cands[len(cands) // 2]] if i % 4 == 0 else [cands[1], cands[-1]]) if restored else []
        ns["record"] = {"status": "restored" if restored else "not_recoverable_by_insertion", "hit_cap": False,
                        "verified": (i % 8 == 0) if restored else None, "calls_used": 6, "rounds": 3,
                        "event_ids": [u.id for u in pick], "units_inserted": [u.text for u in pick]}
    return ns


def _make_root(tmp_path, n_cases=12):
    """A v6 root on a synthetic frame, staged through the same functions as the real one: frame.jsonl, cases.jsonl,
    e1_flips.jsonl (from a synthetic v4 state), p1_min_sets.jsonl, the input-token rule and the E1 draws."""
    from lens.flip.headline.engine import make_headline_sample
    root, v4 = tmp_path / "x6", tmp_path / "v4"
    root.mkdir()
    cases = make_headline_sample(n_cases, 5)
    write_jsonl(root / "cases.jsonl", cases)
    (root / "frame.jsonl").write_text(F._jsonl(_frame_rows(cases)))
    hs = F.load_frame(root)
    sdir = v4 / F.X_ARM / "state" / "cases"
    sdir.mkdir(parents=True)
    for i, h in enumerate(hs):
        (sdir / f"{h.case.case_id}.json").write_text(json.dumps({h.system: _x_state(h, i)}))
    (root / F.E1_FLIPS).write_text(F._jsonl(F.e1_flip_rows(hs, v4)))
    ms = []
    for h in hs:
        v3 = h.row["v3"]
        if not v3["residue_flip"]:
            continue
        cands = residue_candidates(h.case, h.view, v3["note"])
        if v3["search_status"] == "restored":
            u = cands[len(cands) // 2]
            ms.append({"case_id": h.case.case_id, "system": h.system, "status": "restored", "hit_cap": False,
                       "verified": True, "calls_used": 6, "rounds": 3, "unit_ids": [u.id], "unit_texts": [u.text],
                       "chars": len(u.text)})
        else:
            ms.append({"case_id": h.case.case_id, "system": h.system, "status": "not_recoverable_by_insertion",
                       "hit_cap": False, "verified": None, "calls_used": 2, "rounds": 1, "unit_ids": [],
                       "unit_texts": [], "chars": None})
    (root / F.P1_MIN_SETS).write_text(F._jsonl(ms))
    F.write_calibration(root)
    F.stage_controls(root)
    return root, hs, v4


def _answers(cases):
    by = {c.case_id: c for c in cases}

    def answer(cid, params, meta):
        case = by[meta["case_id"]]
        user = params["messages"][0]["content"]
        assert meta["kind"] == "decide"
        # by custom_id, so the five re-reads of one body can differ, as real samples do
        act = case.a_star if zlib.crc32(cid.encode()) % 3 else "REQUEST_INFO"
        text = json.dumps({"action": act, "reason": "mock", "event_ids": []})
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
                  max_steps=60)
    return r, api, d


def _finalized(root, tmp_path):
    D.dryrun(root, tmp_path / "no-results")
    PR.freeze(root, now=0)
    return PR.finalize(root, now=0)


# ---------------------------------------------------------------------------
# Frame staging
# ---------------------------------------------------------------------------

def test_e1_flip_rows_mirror_the_search(tmp_path):
    root, hs, v4 = _make_root(tmp_path)
    e1 = F.load_e1_flips(root)
    flips = [h for i, h in enumerate(hs) if i % 4 != 3]
    assert set(e1) == {(h.case.case_id, h.system) for h in flips}
    for i, h in enumerate(hs):
        rec = e1.get((h.case.case_id, h.system))
        if rec is None:
            continue
        cands = residue_candidates(h.case, h.view, rec["note"])
        assert rec["n_candidates"] == len(cands) and not any(u.text in rec["note"] for u in cands)
        assert len(cands) == len(h.view.units) - 2  # the two units the note holds are not candidates
        if rec["status"] == "restored":
            units = F.units_in_order(cands, rec["unit_ids"])
            assert [u.id for u in units] == rec["unit_ids"] and rec["chars"] == len("\n".join(u.text for u in units))
            assert len(rec["unit_ids"]) == (1 if i % 4 == 0 else 2)
        else:
            assert rec["unit_ids"] == [] and rec["chars"] is None
    assert "unit_texts" not in next(iter(e1.values()))
    # a search record that is not in view order, or candidates that differ, stop the build
    h = flips[1]
    p = v4 / F.X_ARM / "state" / "cases" / f"{h.case.case_id}.json"
    st = json.loads(p.read_text())
    st[h.system]["record"]["event_ids"].reverse()
    p.write_text(json.dumps(st))
    with pytest.raises(SystemExit, match="view order"):
        F.e1_flip_rows(hs, v4)
    st[h.system]["record"]["event_ids"].reverse()
    st[h.system]["out"]["candidate_ids"] = st[h.system]["out"]["candidate_ids"][1:]
    p.write_text(json.dumps(st))
    with pytest.raises(SystemExit, match="candidates"):
        F.e1_flip_rows(hs, v4)
    with pytest.raises(ValueError):
        F.units_in_order(_units([5, 5]), ["e0007"])


def test_stage_controls_pool_excludes_note_and_minimal_set(tmp_path):
    root, hs, _ = _make_root(tmp_path)
    e1 = F.load_e1_flips(root)
    ctl = F.load_controls(root / F.E1_CONTROLS)
    assert set(ctl) == set(e1) and all(len(v) == 5 and [c["k"] for c in v] == list(range(5)) for v in ctl.values())
    by_h = {(h.case.case_id, h.system): h for h in hs}
    med = CT.median_recoverable_chars(list(e1.values()))
    n_sent = 0
    for k, rows in ctl.items():
        rec, h = e1[k], by_h[k]
        cand = {u.id: u for u in residue_candidates(h.case, h.view, rec["note"])}
        for c in rows:
            assert set(c["unit_ids"]) <= set(cand) and not (set(c["unit_ids"]) & set(rec["unit_ids"]))
            assert not any(cand[i].text in rec["note"] for i in c["unit_ids"])
            assert c["target_chars"] == (rec["chars"] if rec["status"] == "restored" else int(round(med)))
            assert c["pool_size"] == len(cand) - len(rec["unit_ids"])
            if c["matched"]:
                assert 0.9 <= c["ratio"] <= 1.1
            n_sent += c["send"]
    assert n_sent > 0
    again = F.stage_controls(root)  # identical bytes: no error
    assert again["n_controls"] == 5 * len(e1) and again["n_sent"] == n_sent
    assert again["e1_controls_sha256"] == CT.controls_sha(F.build_controls_rows(root))
    (root / F.E1_CONTROLS).write_text("{}\n")
    with pytest.raises(SystemExit, match="differs"):
        F.stage_controls(root)


@pytest.mark.skipif(not (REAL_V4.exists() and REAL_X.exists()), reason="needs the frozen v4 results")
def test_real_frame_stages_and_the_reread_is_the_stored_restoring_call(tmp_path):
    root = tmp_path / "r"
    meta = F.stage_frame(root, F.V4, F.RESULTS)
    assert (meta["n_handoffs"], meta["n_e1_flips"], meta["n_e1_restored"], meta["n_e1_verified"]) == (490, 193, 144, 113)
    assert (meta["n_v3_flips"], meta["n_p1_restored"], meta["n_v3_floor"]) == (142, 41, 51)
    assert meta["p1_min_sets_sha256"] == F.V3_MIN_SETS_SHA256
    assert meta["restoring_call_check"] == {"E1": {"n": 144, "found": 144, "answer_is_a_A": 144},
                                            "P1": {"n": 41, "found": 41, "answer_is_a_A": 41}}
    rows = F._read_jsonl(root / F.E1_CONTROLS)
    assert len(rows) == 193 * 5 and all(0.9 <= r["ratio"] <= 1.1 for r in rows if r["matched"])
    e1 = F.load_e1_flips(root)
    assert not any(set(r["unit_ids"]) & set(e1[(r["case_id"], r["system"])]["unit_ids"]) for r in rows)
    assert CT.controls_sha(F.build_controls_rows(root)) == meta["e1_controls"]["e1_controls_sha256"]
    assert F.stage_frame(root, F.V4, F.RESULTS) == meta  # idempotent
    res = D.dryrun(root, F.RESULTS, write=False)
    assert res["arms"]["E1"]["requests"] == meta["e1_controls"]["n_sent"] + 5 * 144
    assert res["arms"]["P1"]["requests"] == 5 * 41


# ---------------------------------------------------------------------------
# Reader settings: the v5 ones, and v5 itself untouched
# ---------------------------------------------------------------------------

def test_reader_settings_are_the_v5_settings():
    for arm in E.ARMS:
        cfg = E.arm_config(arm)
        assert cfg["decider"] == E5.arm_config("C1")["decider"] == E5.arm_config("C3")["decider"]
        assert cfg["decider"] == {"model": "claude-haiku-5-5", "arm": "adaptive-high", "max_tokens": 6000,
                                  "effort": "high", "thinking": None}
        assert cfg["rates"] == {"input": 0.05, "output": 0.25} and cfg["bootstrap"] == {"reps": 1000, "seed": 11}
        assert cfg["cache_bypass"] is True and cfg["rereads"] == 5 and cfg["prompt_version"] == "v1"
    assert "controls" in E.arm_config("E1") and "controls" not in E.arm_config("P1")
    with pytest.raises(ValueError):
        E.arm_config("C1")
    # the v5 arm settings still hash to the values frozen in docs/flip-protocol-v5-exploratory.md
    v5 = PR5.current()
    assert v5["arm_configs"]["C1"] == "56f4d1148346c2156ebbd392327c9e3d2f16c5b9a3a3680f3ef0065594789df7"
    assert v5["arm_configs"]["C3"] == "f8b4bdd70959e663478b22c4bfe2933737d4b17c17b597ee44d4f56d71843389"
    assert v5["prompts"]["x5.c1.rule"] == "71879188097dc3d218375c3533fc35da007411dc4d58cd548c6a79ec707dafe7"
    assert v5["reused_v3"]["decider.v1.inserted"] == "74be01664a51887455a9445a08edbf8eae8c1f75b9a34db9d924a1b9d9b5a8d0"
    assert PR.current()["reused_v3"] == v5["reused_v3"]


# ---------------------------------------------------------------------------
# E1 and P1 end to end
# ---------------------------------------------------------------------------

def test_e1_requests_controls_by_body_and_rereads_by_salt(tmp_path):
    root, hs, _ = _make_root(tmp_path)
    d = E.init_arm("E1", root)
    seen: list[dict] = []
    api = FakeBatchAPI(_answers(F.load_cases(root)), engine_meta_fn(d), on_create=seen.extend)
    assert E.run_arm(d, api, cap_usd=100.0, root=root, log=lambda m: None, sleep=lambda s: None,
                     check_protocol=False, max_steps=20) == "complete"
    e1, ctl = F.load_e1_flips(root), F.load_controls(root / F.E1_CONTROLS)
    n_sent = sum(c["send"] for v in ctl.values() for c in v)
    n_rest = sum(r["status"] == "restored" for r in e1.values())
    assert api.created == 1 and api.requests_sent == n_sent + 5 * n_rest == len(set(api.sent_custom_ids))
    rnd = json.loads((d / "rounds" / "001" / "round.json").read_text())["requests"]
    assert sum(m["stage"] == "e1_ctl_B" for m in rnd.values()) == n_sent
    assert sum(m["stage"] == "e1_fresh_B" for m in rnd.values()) == 5 * n_rest
    body_of = {r["custom_id"]: r["params"] for r in seen}
    dec = E.arm_config("E1")["decider"]
    by_h = {(h.case.case_id, h.system): h for h in hs}
    for k, rec in e1.items():
        h = by_h[k]
        cands = residue_candidates(h.case, h.view, rec["note"])
        mine = {cid: m for cid, m in rnd.items() if m["case_id"] == k[0]}
        fresh = {cid: m for cid, m in mine.items() if m["stage"] == "e1_fresh_B"}
        plain = {cid: m for cid, m in mine.items() if m["stage"] == "e1_ctl_B"}
        # a control: keyed by its body; the note is the view and the control units are the inserted units
        want = {rq.request_key(rq.decide_request(note_view(rec["note"]), F.units_in_order(cands, c["unit_ids"]), h.case,
                                                 "v1", dec)) for c in ctl[k] if c["send"]}
        assert {m["key"] for m in plain.values()} == want and not any("salt" in m for m in plain.values())
        if rec["status"] != "restored":
            assert not fresh
            continue
        # a re-read: five keys, five salts, one body, and that body is the search's restoring call
        call = rq.decide_request(note_view(rec["note"]), [u for u in cands if u.id in set(rec["unit_ids"])], h.case, "v1", dec)
        assert len(fresh) == 5 and {m["salt"] for m in fresh.values()} == {"t0", "t1", "t2", "t3", "t4"}
        assert {rq.canonical(body_of[cid]) for cid in fresh} == {rq.canonical(call)}
        assert {m["key"] for m in fresh.values()} == {CT.salted_key(rq.canonical(call), f"t{i}") for i in range(5)}
        assert rq.request_key(call) not in {m["key"] for m in mine.values()}
        assert "Additional records" in call["messages"][0]["content"] or len(call["messages"][0]["content"]) > len(rec["note"])
    st = AN._states(root, "E1")
    assert len(st) == len(hs) and all(v["done"] for v in st.values())
    for k, v in st.items():
        o = v["out"]
        if k not in e1:
            assert o == {"flip": False}
            continue
        assert len(o["controls"]) == sum(c["send"] for c in ctl[k])
        assert len(o["fresh"]) == (5 if e1[k]["status"] == "restored" else 0) and o["search_restored"] == (len(o["fresh"]) == 5)
        assert all(x["restored"] == (x["a_B"] == by_h[k].row["a_A"]) for x in o["controls"] + o["fresh"])
    # the mock answers by custom_id, so some set of five re-reads is split: they are separate samples
    assert any(0 < sum(x["restored"] for x in v["out"]["fresh"]) < 5 for v in st.values() if v["out"].get("fresh"))
    assert json.loads((d / "cache_bypass.json").read_text())["tags"] == ["t0", "t1", "t2", "t3", "t4"]


def test_p1_five_rereads_of_each_restored_v3_set(tmp_path):
    root, hs, _ = _make_root(tmp_path)
    r, api, d = _run(root, "P1")
    p1 = F.load_p1_sets(root)
    n_rest = sum(x["status"] == "restored" for x in p1.values())
    assert r == "complete" and n_rest > 0 and api.requests_sent == 5 * n_rest == len(set(api.sent_custom_ids))
    rnd = json.loads((d / "rounds" / "001" / "round.json").read_text())["requests"]
    assert all(m["stage"] == "p1_fresh_B" and m["salt"] in {"t0", "t1", "t2", "t3", "t4"} for m in rnd.values())
    dec = E.arm_config("P1")["decider"]
    keys = {m["key"] for m in rnd.values()}
    for h in hs:
        rec = p1.get((h.case.case_id, h.system))
        o = AN._states(root, "P1")[(h.case.case_id, h.system)]["out"]
        if rec is None or rec["status"] != "restored":
            assert o["fresh"] == [] and o["search_restored"] is False and o["flip"] == (rec is not None)
            continue
        note = h.row["v3"]["note"]
        cands = residue_candidates(h.case, h.view, note)
        call = rq.decide_request(note_view(note), F.units_in_order(cands, rec["unit_ids"]), h.case, "v1", dec)
        assert {CT.salted_key(rq.canonical(call), f"t{i}") for i in range(5)} <= keys and len(o["fresh"]) == 5


def test_resume_pays_once(tmp_path):
    root, hs, _ = _make_root(tmp_path)
    d = E.init_arm("E1", root)
    api = FakeBatchAPI(_answers(F.load_cases(root)), engine_meta_fn(d))
    kw = dict(cap_usd=50, root=root, log=lambda m: None, sleep=lambda s: None, check_protocol=False)
    assert E.run_arm(d, api, max_steps=1, **kw) == "submitted"      # the round is in flight, the process "dies"
    sent = api.requests_sent
    assert E.run_arm(d, api, max_steps=20, **kw) == "complete"      # a new engine picks the round up
    assert api.created == 1 and api.requests_sent == sent
    assert E.run_arm(d, api, max_steps=5, **kw) == "complete" and api.created == 1
    with pytest.raises(Refused, match="already holds"):
        E.init_arm("E1", root)


def test_cap_refuses_and_counts_the_other_arm(tmp_path):
    root, hs, _ = _make_root(tmp_path)
    d = E.init_arm("E1", root)
    api = FakeBatchAPI(lambda *a: None, engine_meta_fn(d))
    with pytest.raises(Refused, match="cap|upper bound"):
        E.run_arm(d, api, cap_usd=0.0001, root=root, log=lambda m: None, sleep=lambda s: None, check_protocol=False,
                  max_steps=5)
    assert api.created == 0
    led = _read_json(root / "spend.json", {})
    assert led["cap_usd"] == 0.0001 and led["total_usd"] == 0
    r, api, d = _run(root, "E1", cap=100.0)
    spent = E.update_ledger(root, 100.0)["total_usd"]
    assert r == "complete" and spent > 0 and set(E.update_ledger(root, 100.0)["arms"]) == {"E1"}
    d2 = E.init_arm("P1", root)
    api2 = FakeBatchAPI(lambda *a: None, engine_meta_fn(d2))
    with pytest.raises(Refused):  # a cap just above what E1 spent leaves no room for P1's round
        E.run_arm(d2, api2, cap_usd=spent + 1e-6, root=root, log=lambda m: None, sleep=lambda s: None,
                  check_protocol=False, max_steps=5)
    assert api2.created == 0


# ---------------------------------------------------------------------------
# Freeze, finalize, check
# ---------------------------------------------------------------------------

def test_freeze_needs_a_dryrun_is_draft_and_engine_refuses_until_final(tmp_path):
    root, hs, _ = _make_root(tmp_path)
    with pytest.raises(SystemExit, match="dryrun"):
        PR.freeze(root)
    D.dryrun(root, tmp_path / "no-results")
    rec = PR.freeze(root, now=0)
    assert rec["status"] == "DRAFT" and rec["version"] == "6-exploratory" and rec["schema"] == "lens.flip.exploratory/1"
    assert set(rec["arm_configs"]) == {"E1", "P1"}
    assert set(rec["rules"]) == {"x6.e1.control_rule", "x6.draw_source", "x6.reread_rule", "x6.analysis"}
    for key, name in PR.RUN_FILES.items():
        assert rec[key] == hashlib.sha256((root / name).read_bytes()).hexdigest()
    assert rec["dryrun"]["likely_usd"] > 0 and rec["dryrun"]["guard"]["fits"]
    with pytest.raises(SystemExit):
        PR.freeze(root)
    assert PR.check(root) == ["status:DRAFT"]
    with pytest.raises(Refused, match="status:DRAFT"):
        E.run_arms(root, ["E1"], cap_usd=10, runner=FakeBatchAPI(lambda *a: None), log=lambda m: None, sleep=lambda s: None)
    assert not (root / "E1").exists()
    d = E.init_arm("P1", root)
    api = FakeBatchAPI(lambda *a: None, engine_meta_fn(d))
    with pytest.raises(Refused, match="status:DRAFT"):
        E.run_arm(d, api, cap_usd=10, root=root, log=lambda m: None, sleep=lambda s: None, check_protocol=True, max_steps=3)
    assert api.created == 0
    assert PR.finalize(root, now=0)["status"] == "FINAL" and PR.check(root, d) == []
    with pytest.raises(SystemExit, match="FINAL already"):
        PR.finalize(root)


def test_finalize_refuses_above_the_price_gate_or_when_the_guard_would_refuse(tmp_path):
    root, hs, _ = _make_root(tmp_path)
    D.dryrun(root, tmp_path / "no-results")
    PR.freeze(root, now=0)
    p = root / PR.NAME
    rec = json.loads(p.read_text())
    p.write_text(json.dumps({**rec, "dryrun": {**rec["dryrun"], "likely_usd": PR.PRICE_GATE_USD + 0.01}}))
    with pytest.raises(SystemExit, match="above"):
        PR.finalize(root)
    p.write_text(json.dumps({**rec, "dryrun": {**rec["dryrun"], "guard": {**rec["dryrun"]["guard"], "fits": False}}}))
    with pytest.raises(SystemExit, match="does not fit"):
        PR.finalize(root)
    assert json.loads(p.read_text())["status"] == "DRAFT"
    # the guard preview itself: a cap under E1's round bound does not fit
    tight = D.dryrun(root, tmp_path / "no-results", write=False, cap_usd=1e-6)
    assert not tight["guard"]["fits"] and not tight["guard"]["E1_fits"]
    assert PR.PRICE_GATE_USD == 1.50 and PR.RUN_CAP_USD == 3.00


def test_check_catches_rule_analysis_file_and_config_changes(tmp_path, monkeypatch):
    root, hs, _ = _make_root(tmp_path)
    _finalized(root, tmp_path)
    d = E.init_arm("E1", root)
    assert PR.check(root, d) == []
    monkeypatch.setattr(CT, "SEED_TAG", "x7-c1")
    assert "rule:x6.e1.control_rule" in PR.check(root)
    monkeypatch.undo()
    monkeypatch.setitem(AN.ANALYSIS_SPEC, "assessed", "something else")
    assert PR.check(root) == ["rule:x6.analysis"]
    monkeypatch.undo()
    monkeypatch.setattr(CT, "REREADS", 3)
    assert {"rule:x6.reread_rule", "arm:E1", "arm:P1"} <= set(PR.check(root))
    monkeypatch.undo()
    assert PR.check(root) == []
    (d / F.E1_CONTROLS).write_text((d / F.E1_CONTROLS).read_text() + "\n")
    assert PR.check(root, d) == [f"run:{F.E1_CONTROLS}"]
    cfg = json.loads((d / "config.json").read_text())
    (d / "config.json").write_text(json.dumps({**cfg, "rereads": 9}))
    assert "config:E1" in PR.check(root, d)
    (root / F.E1_FLIPS).write_text((root / F.E1_FLIPS).read_text() + "\n")
    assert "root:e1_flips_sha256" in PR.check(root)


def test_cli_guards(tmp_path):
    from lens.flip.explore6.__main__ import main, parse_arms
    with pytest.raises(SystemExit, match="price-confirmed"):
        main(["run", "--results", str(tmp_path), "--cap-usd", "3"])
    with pytest.raises(SystemExit, match="cap-usd"):
        main(["run", "--results", str(tmp_path), "--price-confirmed"])
    with pytest.raises(SystemExit, match="price-confirmed"):
        main(["finalize", "--results", str(tmp_path)])
    with pytest.raises(SystemExit):
        parse_arms("E1,C1")
    assert parse_arms("P1,E1") == ["P1", "E1"] and parse_arms(None) == ["E1", "P1"]
    with pytest.raises(ValueError):
        E.run_arms(tmp_path, ["E1", "E1"], cap_usd=1, check_protocol=False)


# ---------------------------------------------------------------------------
# Dry run
# ---------------------------------------------------------------------------

def test_dryrun_counts_the_engine_requests_without_a_call(tmp_path):
    root, hs, _ = _make_root(tmp_path)
    res = D.dryrun(root, tmp_path / "no-results")
    assert res["api_calls"] == 0 and res["count_tokens_calls"] == 0 and res["submitted"] is False
    assert (root / "dryrun_v6.json").exists() and set(res["arms"]) == {"E1", "P1"}
    for a, x in res["arms"].items():
        assert 0 < x["likely_usd"] < x["likely_usd_if_outputs_x1_5"] < x["upper_usd"], a
    assert res["total"]["likely_usd"] == pytest.approx(sum(x["likely_usd"] for x in res["arms"].values()), abs=1e-3)
    apis = {}
    for arm in E.ARMS:
        r, apis[arm], d = _run(root, arm)
        assert r == "complete" and apis[arm].requests_sent == res["arms"][arm]["requests"]
        # the dry run's upper bound is the bound the engine's guard computed for the arm's one round
        est = json.loads((d / "round_estimates.json").read_text())
        assert len(est) == 1 and est[0]["upper_bound_usd"] == pytest.approx(res["arms"][arm]["upper_usd"], abs=1e-3)
        assert est[0]["likely_usd"] == pytest.approx(res["arms"][arm]["likely_usd"], abs=1e-3)
    assert res["e1_controls"]["sent"] + 5 * sum(r["status"] == "restored" for r in F.load_e1_flips(root).values()) \
        == res["arms"]["E1"]["requests"]


# ---------------------------------------------------------------------------
# Analysis on a mock
# ---------------------------------------------------------------------------

def _row(i, case, **kw):
    r = {"case_id": f"c{case}", "case_index": case, "n_cases": 4, "system": "recent", "mechanism": "M1",
         "mechanism_group": "M1-M3", "v3_floor_flip": 0, "v3_flip": 0, "v3_search_restored": 0,
         "e1_flip": 0, "e1_search_restored": 0, "e1_search_verified": 0, "e1_min_units": None, "e1_min_chars": None,
         "e1_target_chars": None, "e1_pool_size": None, "e1_ctl_drawn": 0, "e1_ctl_sent": 0, "e1_ctl_sent_restored": 0,
         "e1_ctl_matched": 0, "e1_ctl_matched_restored": 0, "e1_ctl_unparsed": 0, "e1_fresh_n": 0, "e1_fresh_restored": 0,
         "e1_fresh_unparsed": 0, "p1_fresh_n": 0, "p1_fresh_restored": 0, "p1_fresh_unparsed": 0,
         "c1v5_sent": 0, "c1v5_sent_restored": 0, "c1v5_matched": 0, "c1v5_matched_restored": 0}
    r.update(kw)
    return r


def _toy():
    flip = dict(e1_flip=1, e1_ctl_drawn=5)
    return [
        # targeted 4 of 5, one of five matched controls restores; also a restored v3 flip assessed in v5
        _row(0, 0, **flip, e1_search_restored=1, e1_search_verified=1, e1_fresh_n=5, e1_fresh_restored=4,
             e1_ctl_sent=5, e1_ctl_sent_restored=1, e1_ctl_matched=5, e1_ctl_matched_restored=1,
             v3_flip=1, v3_search_restored=1, p1_fresh_n=5, p1_fresh_restored=3, c1v5_sent=5, c1v5_sent_restored=1,
             c1v5_matched=5, c1v5_matched_restored=1),
        # the search did not restore: targeted 0; two of four matched controls restore; a floor flip
        _row(1, 0, **flip, e1_ctl_sent=5, e1_ctl_sent_restored=3, e1_ctl_matched=4, e1_ctl_matched_restored=2,
             v3_floor_flip=1),
        # restored, 5 of 5, but no matched control: not assessed
        _row(2, 1, **flip, e1_search_restored=1, e1_fresh_n=5, e1_fresh_restored=5, e1_ctl_sent=1),
        # restored, 2 of 5, no control restores
        _row(3, 2, **flip, e1_search_restored=1, e1_fresh_n=5, e1_fresh_restored=2, e1_ctl_sent=5, e1_ctl_matched=5),
        # not an extractive flip; a floor flip; a v3 flip the search did not restore, assessed in v5
        _row(4, 2, v3_floor_flip=1, v3_flip=1, c1v5_sent=2, c1v5_sent_restored=1, c1v5_matched=2, c1v5_matched_restored=1),
        # not an extractive flip; a restored v3 flip without a matched v5 control
        _row(5, 3, v3_flip=1, v3_search_restored=1, p1_fresh_n=5, p1_fresh_restored=5, c1v5_sent=1),
    ]


def test_analysis_toy_arithmetic():
    s = AN.summarize(_toy())
    e, p = s["E1_extractive"], s["P1_paraphrased"]
    assert (e["n_handoffs"], e["n_flips"], e["n_search_restored"], e["n_assessed"]) == (6, 4, 3, 3)
    assert e["controls"] == {"drawn": 20, "sent": 16, "matched": 14, "flips_without_matched_control": 1,
                             "flips_without_sent_control": 0, "unparsed_replies": 0}
    assert e["rereads"]["sent"] == 15
    p1 = e["primary_1_specificity"]
    assert p1["n_flips"] == 3
    assert p1["targeted_restore_rate"]["value"] == pytest.approx((0.8 + 0 + 0.4) / 3)
    assert p1["irrelevant_restore_rate"]["value"] == pytest.approx((0.2 + 0.5 + 0) / 3)
    assert p1["targeted_minus_irrelevant"]["diff"] == pytest.approx(0.5 / 3)
    p2 = e["primary_2_content_specific"]
    assert p2["rate"]["value"] == pytest.approx(0.5 / 6) and p2["rate"]["n"] == 6 and p2["rate"]["sum"] == pytest.approx(0.5)
    assert p2["floor_rate_v3"]["k"] == 2 and p2["minus_floor"]["diff"] == pytest.approx((0.5 - 2) / 6)
    assert p2["verdict"] == "not supported" and "lower 95% bound" in p2["rule"]
    v5d = e["secondary_v5_definition_of_targeted"]
    assert v5d["specificity"]["targeted_restore_rate"]["value"] == pytest.approx(2 / 3)
    assert v5d["content_specific"]["rate"]["value"] == pytest.approx(1.3 / 6)
    assert v5d["content_specific"]["minus_floor"]["diff"] == pytest.approx((1.3 - 2) / 6)
    sa = e["sensitivity_all_sent_controls"]
    assert sa["specificity"]["n_flips"] == 4
    assert sa["content_specific"]["rate"]["value"] == pytest.approx((0.6 - 0.6 + 1.0 + 0.4) / 6)
    assert e["sensitivity_upper_bracket"]["content_specific"]["rate"]["value"] == pytest.approx((0.6 - 0.5 + 1.0 + 0.4) / 6)
    assert e["irrelevant_by_search_outcome"]["restored"]["value"] == pytest.approx(0.1)
    assert e["irrelevant_by_search_outcome"]["not_restored"]["value"] == pytest.approx(0.5)
    assert e["any_matched_control_restores"]["value"] == pytest.approx(2 / 3)
    fr = e["fresh_reread"]
    assert fr["n_sets"] == 3 and fr["rate"]["value"] == pytest.approx(11 / 15)
    assert fr["sets_by_restores"] == {"0": 0, "1": 0, "2": 1, "3": 0, "4": 1, "5": 1}
    assert fr["all_rereads_restore"]["value"] == pytest.approx(1 / 3) and fr["majority_restores"]["value"] == pytest.approx(2 / 3)
    assert fr["at_least_one_restores"]["value"] == 1.0 and fr["no_reread_restores"]["value"] == 0.0
    assert fr["by_v2_verified"]["verified"]["value"] == pytest.approx(0.8)
    assert fr["by_v2_verified"]["not_verified"]["value"] == pytest.approx(0.7)
    assert e["v4_reference"]["recoverable_rate"]["k"] == 3 and e["v4_reference"]["recoverable_minus_floor"]["diff"] == pytest.approx(1 / 6)
    assert e["by_system_descriptive"]["recent"]["n_flips"] == 4
    assert p["fresh_reread"]["rate"]["value"] == pytest.approx(0.8) and p["n_search_restored"] == 2
    c = p["v5_c1_recomputed"]
    assert c["n_assessed"] == 2
    assert c["omission_term_v5_definition"]["value"] == pytest.approx((1 - 0.2 + 0 - 0.5) / 6)
    assert c["omission_term_fresh_targeted"]["value"] == pytest.approx((0.6 - 0.2 + 0 - 0.5) / 6)
    assert c["fresh_targeted"]["targeted_restore_rate"]["value"] == pytest.approx(0.3)
    assert s["denominators"]["cases"] == 4 and s["bootstrap"] == {"reps": 1000, "seed": 11, "unit": s["bootstrap"]["unit"]}
    # the bounds are the paired case bootstrap of the v5 helper on the same sums
    assert p2["minus_floor"]["ci95"] is not None and p2["minus_floor"]["ci95"][0] <= p2["minus_floor"]["diff"] <= p2["minus_floor"]["ci95"][1]


def test_verdict_rule_reads_the_lower_bound_only():
    assert AN.verdict({"diff": 0.1, "ci95": [0.001, 0.2]}) == "supported"
    assert AN.verdict({"diff": 0.1, "ci95": [0.0, 0.2]}) == "not supported"
    assert AN.verdict({"diff": 0.1, "ci95": [-0.01, 0.2]}) == "not supported"
    assert AN.verdict({"diff": 0.1, "ci95": None}) == "not supported"
    # a clear case: every handoff is a flip whose minimal set always restores and no control ever does
    rows = [_row(i, i % 4, e1_flip=1, e1_ctl_drawn=5, e1_search_restored=1, e1_fresh_n=5, e1_fresh_restored=5,
                 e1_ctl_sent=5, e1_ctl_matched=5) for i in range(8)]
    s = AN.summarize(rows)["E1_extractive"]["primary_2_content_specific"]
    assert s["rate"]["value"] == 1.0 and s["minus_floor"]["bounds"] == "above zero" and s["verdict"] == "supported"
    # the same flips with controls that restore as often as the minimal set: nothing content-specific is left
    rows = [{**r, "e1_ctl_matched_restored": 5, "e1_ctl_sent_restored": 5} for r in rows]
    s = AN.summarize(rows)["E1_extractive"]["primary_2_content_specific"]
    assert s["rate"]["value"] == 0.0 and s["verdict"] == "not supported"


def test_metric_rows_are_text_free_and_rebuild_the_summary(tmp_path):
    root, hs, _ = _make_root(tmp_path)
    _finalized(root, tmp_path)
    out = E.run_arms(root, ["P1", "E1"], cap_usd=100.0, runner=FakeBatchAPI(_answers(F.load_cases(root)), _all_meta(root)),
                     log=lambda m: None, sleep=lambda s: None)
    assert out == {"E1": "complete", "P1": "complete"} and list(out) == ["E1", "P1"]  # RUN_ORDER
    with pytest.raises(SystemExit, match="not complete"):
        AN.metric_rows(_empty_copy(root, tmp_path))
    s = AN.write(root, tmp_path / "no-v5")
    mpath = root / AN.METRICS_NAME
    rows = AN.read_metrics(mpath)
    assert len(rows) == len(hs) and s["metrics_file"]["rows"] == len(hs)
    assert s["metrics_file"]["sha256"] == hashlib.sha256(mpath.read_bytes()).hexdigest()
    text = mpath.read_text()
    e1 = F.load_e1_flips(root)
    for rec in e1.values():                       # no note text, no unit text
        assert rec["note"][:40] not in text
    for r in rows:
        assert all(isinstance(v, (str, int, float, bool, type(None))) for v in r.values())
        assert all(len(v) <= 32 for v in r.values() if isinstance(v, str))
        assert {k for k, v in r.items() if isinstance(v, str)} == {"case_id", "system", "mechanism", "mechanism_group"}
    body = AN.rebuild(mpath)
    assert body == {k: s[k] for k in body}        # every number comes from the metric file alone
    e = s["E1_extractive"]
    assert e["n_flips"] == len(e1) and e["n_search_restored"] == sum(r["status"] == "restored" for r in e1.values())
    ctl = F._read_jsonl(root / F.E1_CONTROLS)
    assert e["controls"]["drawn"] == len(ctl) and e["controls"]["sent"] == sum(c["send"] for c in ctl)
    assert e["controls"]["matched"] == sum(c["send"] and c["matched"] for c in ctl)
    assert e["rereads"]["sent"] == 5 * e["n_search_restored"]
    assert s["P1_paraphrased"]["v5_c1_recomputed"] is None and s["v5_reference"] is None
    assert s["protocol"]["status"] == "FINAL" and s["spend"]["total_usd"] > 0
    assert s["spend"]["by_arm"]["E1"]["requests"] == e["controls"]["sent"] + e["rereads"]["sent"]
    assert s["spend"]["by_arm"]["P1"]["requests"] == s["P1_paraphrased"]["rereads"]["sent"]
    assert "primary 1" in AN.headline(s) and "primary 2" in AN.headline(s)
    with pytest.raises(AssertionError, match="long string"):
        AN.check_rows_text_free(json.dumps({"x": "y" * 33}) + "\n")
    with pytest.raises(AssertionError, match="nested"):
        AN.check_rows_text_free(json.dumps({"x": [1]}) + "\n")
    with pytest.raises(AssertionError, match="banned"):
        AN.check_rows_text_free(json.dumps({"x": "Zebra"}) + "\n", ["zebra"])


def _empty_copy(root, tmp_path):
    """A root with the run files but no arm state."""
    d = tmp_path / "x6-empty"
    d.mkdir()
    for name in PR.RUN_FILES.values():
        (d / name).write_bytes((root / name).read_bytes())
    return d


def _all_meta(root):
    def read():
        out = {}
        for arm in E.ARMS:
            p = root / arm / "state" / "round.json"
            if p.exists():
                out.update(json.loads(p.read_text())["requests"])
        return out
    return read


def test_v5_c1_results_feed_the_recomputed_omission_term(tmp_path):
    """With a v5 root beside it, the metric rows carry the frozen C1 counts by flip."""
    root, hs, _ = _make_root(tmp_path)
    for arm in E.ARMS:
        assert _run(root, arm)[0] == "complete"
    v5 = tmp_path / "v5"
    (v5 / "C1" / "state" / "cases").mkdir(parents=True)
    p1 = F.load_p1_sets(root)
    draws, want = [], {}
    for j, (k, rec) in enumerate(sorted(p1.items())):
        ctl = [{"case_id": k[0], "system": k[1], "k": i, "send": i < 4, "matched": i < 3} for i in range(5)]
        draws += ctl
        got = [{"k": i, "restored": (i + j) % 2 == 0} for i in range(4)]
        (v5 / "C1" / "state" / "cases" / f"{k[0]}.json").write_text(json.dumps({k[1]: {"out": {"controls": got}, "done": True}}))
        want[k] = (4, sum(g["restored"] for g in got), 3, sum(g["restored"] for g in got[:3]))
    (v5 / "c1_controls.jsonl").write_text(F._jsonl(draws))
    rows = AN.metric_rows(root, v5)
    for r in rows:
        k = (r["case_id"], r["system"])
        have = (r["c1v5_sent"], r["c1v5_sent_restored"], r["c1v5_matched"], r["c1v5_matched_restored"])
        assert have == want.get(k, (0, 0, 0, 0))
    c = AN.summarize(rows)["P1_paraphrased"]["v5_c1_recomputed"]
    assert c["n_assessed"] == len(p1) and c["omission_term_fresh_targeted"]["value"] is not None
    assert c["omission_term_v5_definition"]["value"] >= c["omission_term_fresh_targeted"]["value"]


# ---------------------------------------------------------------------------
# The private word list
# ---------------------------------------------------------------------------

def test_private_word_list_absent_from_v6_code_rules_and_docs():
    banned_file = REPO / "banned_words.local.txt"
    if not banned_file.exists():
        pytest.skip("banned_words.local.txt not present")
    banned = [w.strip().lower() for w in banned_file.read_text().splitlines() if w.strip()]
    files = sorted((REPO / "src" / "lens" / "flip" / "explore6").glob("*.py")) + [Path(__file__)]
    files += [p for p in (REPO / "docs" / "flip-protocol-v6-exploratory.md",
                          *sorted((REPO / "docs").glob("exploratory-v6-results-*.md"))) if p.exists()]
    blob = "\n".join(f.read_text().lower() for f in files)
    blob += "\n".join(PR.rule_texts().values()).lower()
    blob += rq.canonical({a: E.arm_config(a) for a in E.ARMS}).lower()
    assert [w for w in banned if w in blob] == []

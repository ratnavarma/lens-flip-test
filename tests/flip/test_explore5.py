"""Exploratory protocol v5: prompts, C1 length matching, C3 cache bypass, C4 pair count, the decomposition, the
spend guard, the freeze, and every arm end to end on a fake Batches API. No network.

The end-to-end tests run on a small synthetic frame built here from the generator, so they need no stored
results. The checks against the real frame skip when results/ is absent (it is git-ignored).
"""

from __future__ import annotations

import hashlib
import json
import re
import zlib
from pathlib import Path

import pytest

from lens.flip.explore5 import analysis as AN
from lens.flip.explore5 import controls as CT
from lens.flip.explore5 import engine as E
from lens.flip.explore5 import frame as F
from lens.flip.explore5 import prompts as P
from lens.flip.explore5 import protocol as PR
from lens.flip.headline import requests as rq
from lens.flip.headline.engine import Refused, _read_json
from lens.flip.headline.mock import FakeBatchAPI, engine_meta_fn
from lens.flip.headline.protocol import _sentinel_case
from lens.flip.residue import build_residue_note_prompt, note_view, residue_candidates
from lens.flip.schema import ACTIONS, Unit, View, write_jsonl

REPO = Path(__file__).resolve().parents[2]
REAL = F.RESULTS / "exploratory-v4" / "frame.jsonl"


# ---------------------------------------------------------------------------
# Prompts and parsers
# ---------------------------------------------------------------------------

def test_x1_prompt_is_v3_prompt_plus_one_paragraph():
    v, c = View(system="s", budget=None, text="{RECORDS}"), _sentinel_case()
    x1, v3 = P.build_x1_prompt(v, c), build_residue_note_prompt(v, c)
    assert x1.replace("\n\n" + P.READER_PARAGRAPH, "", 1) == v3
    assert "\n\n" not in P.READER_PARAGRAPH  # one paragraph
    assert "only your note and the request" in P.READER_PARAGRAPH
    assert all(a in P.READER_PARAGRAPH for a in ACTIONS)
    assert x1.index(P.READER_PARAGRAPH) < x1.index("Records:")


def test_x2_prompt_render_and_parse():
    v, c = View(system="s", budget=None, text="{RECORDS}"), _sentinel_case()
    pr = P.build_x2_prompt(v, c)
    assert all(f in pr for f in P.X2_FIELDS) and "at most 150 tokens" in pr and "Records:\n{RECORDS}" in pr
    fields = {k: f"text {i}" for i, k in enumerate(P.X2_FIELDS)}
    lines = P.render_x2(fields).split("\n")
    assert [l.split(":")[0] for l in lines] == ["Leading explanation", "Live alternatives", "Missing evidence",
                                                 "Next action", "Reversibility"]
    assert P.render_x2({})[-4:] == "none"
    raw = json.dumps({**fields, "p_residue": 0.3})
    r = P.parse_x2_reply(raw)
    assert r["complete"] and not r["cut"] and r["p_residue"] == 0.3 and r["note"] == P.render_x2(fields)
    long = {k: "word " * 80 for k in P.X2_FIELDS}
    r = P.parse_x2_reply(json.dumps({**long, "p_residue": 1}))
    assert r["cut"] and len(r["note"]) <= int(150 * 2.2)
    bad = P.parse_x2_reply(json.dumps({"leading_explanation": "x", "p_residue": 2}))
    assert not bad["complete"] and bad["p_residue"] is None
    assert P.X2_SCHEMA["additionalProperties"] is False and set(P.X2_SCHEMA["required"]) == set(P.X2_FIELDS) | {"p_residue"}


def test_rewrite_and_judge_prompts():
    p = P.build_rewrite_prompt("Recommend EXPLAIN.")
    assert p.endswith("Note:\nRecommend EXPLAIN.") and "wording only" in p and "at most 150 tokens" in p
    assert P.parse_rewrite_reply('{"note": "Advise EXPLAIN."}') == "Advise EXPLAIN."
    assert P.parse_rewrite_reply('{"x": 1}') is None and P.parse_rewrite_reply(None) is None
    from lens.flip.omission import judge as J
    assert P.build_judge_prompt("a note", "a unit") == J.build_prompt("a note", "a unit")
    assert P.parse_judge_reply('{"conveyed": true}') is True and P.parse_judge_reply("nope") is None
    assert P.JUDGE_SCHEMA["additionalProperties"] is False


def test_request_settings():
    cfg = E.arm_config("C2")
    r = E.rewrite_request("a note", cfg["rewriter"])
    assert r["model"] == "claude-haiku-5-5" and r["thinking"] == {"type": "disabled"}
    assert r["output_config"]["effort"] == "medium" and r["max_tokens"] == 512
    j = E.judge_request("a note", "a unit", E.arm_config("C4")["judge"])
    assert j["thinking"] == {"type": "disabled"} and j["output_config"]["effort"] == "medium"
    assert j["output_config"]["format"]["schema"] == P.JUDGE_SCHEMA
    x = E.x1_request(View(system="s", budget=None, text="t"), _sentinel_case(), E.arm_config("X1")["note_writer"])
    assert "thinking" not in x and x["max_tokens"] == 6000 and x["output_config"]["effort"] == "high"
    assert E.arm_config("C3")["cache_bypass"] and not any(E.arm_config(a)["cache_bypass"] for a in E.ARMS if a != "C3")


# ---------------------------------------------------------------------------
# C1: length matching and seeding
# ---------------------------------------------------------------------------

def _units(lengths):
    return [Unit(id=f"e{i:04d}", text="x" * n) for i, n in enumerate(lengths)]


def test_c1_matches_length_and_excludes_minimal_set():
    pool_all = _units([40, 55, 70, 90, 120, 35, 60, 80, 45, 100, 150, 30])
    minimal = {"e0003", "e0004"}  # 90 + 120 chars -> 211 with the newline
    rec = {"status": "restored", "unit_ids": sorted(minimal), "chars": 211}
    rows = CT.build_controls("c", "recent", pool_all, rec, median_chars=100.0)
    assert len(rows) == CT.K_CONTROLS == 5
    for r in rows:
        assert not (set(r["unit_ids"]) & minimal)
        assert r["matched"] and 0.9 * 211 <= r["chars"] <= 1.1 * 211
        assert len(set(r["unit_ids"])) == len(r["unit_ids"])  # without replacement
        by = {u.id: u for u in pool_all}
        assert r["chars"] == CT.block_chars([by[i].text for i in r["unit_ids"]])
        assert r["unit_ids"] == sorted(r["unit_ids"])  # view order
        assert r["send"] and r["target_source"] == "minimal_set"


def test_c1_seeded_and_k_distinct():
    pool = _units([40, 55, 70, 90, 120, 35, 60, 80, 45, 100, 150, 30])
    rec = {"status": "restored", "unit_ids": ["e0000"], "chars": 200}
    a = CT.build_controls("case9", "recent", pool, rec, 100.0)
    b = CT.build_controls("case9", "recent", pool, rec, 100.0)
    c = CT.build_controls("case8", "recent", pool, rec, 100.0)
    assert a == b and a != c
    assert CT.control_seed("case9", "recent", 2) == "x5-c1|case9|recent|2"
    assert len({tuple(r["unit_ids"]) for r in a}) == 5 and not any(r["duplicate"] for r in a)
    # the draw does not depend on the other controls' results, only on its own stream and the avoid list
    one = CT.draw_control(pool[1:], 200, CT.control_seed("case9", "recent", 0))
    assert one["unit_ids"] == a[0]["unit_ids"]


def test_c1_non_recoverable_uses_median_and_all_candidates():
    pool = _units([60, 60, 60, 60, 60, 60, 60, 60])
    rec = {"status": "not_recoverable_by_insertion", "unit_ids": [], "chars": None}
    med = CT.median_recoverable_chars([{"status": "restored", "chars": c} for c in (100, 200, 400)]
                                      + [{"status": "not_recoverable_by_insertion", "chars": None}])
    assert med == 200.0
    rows = CT.build_controls("c", "s", pool, rec, med)
    assert all(r["target_source"] == "median_of_recoverable" and r["target_chars"] == 200 for r in rows)
    assert all(r["pool_size"] == 8 for r in rows)  # nothing excluded
    assert all(r["matched"] and 180 <= r["chars"] <= 220 for r in rows)  # 3 units: 182, or 4: 243 no; 3 fits
    with pytest.raises(ValueError):
        CT.median_recoverable_chars([{"status": "not_recoverable_by_insertion", "chars": None}])


def test_c1_flags_infeasible_draws_and_does_not_send_empty_or_copies():
    # the only usable pool units are 5,000 and 10 characters: a 2,109-character target cannot be matched
    rec = {"status": "restored", "unit_ids": ["e0001"], "chars": 2109}
    rows = CT.build_controls("c", "summary", _units([5000, 87, 10]), rec, 100.0)
    assert not any(r["matched"] for r in rows)
    assert rows[0]["unit_ids"] == ["e0002"] and rows[0]["send"]
    assert all(r["duplicate"] and not r["send"] for r in rows[1:])
    # a 33-character target with only longer units leaves the set empty: nothing to send
    rec = {"status": "restored", "unit_ids": ["e0000"], "chars": 33}
    rows = CT.build_controls("c", "recent", _units([33, 80, 90, 100]), rec, 100.0)
    assert all(r["unit_ids"] == [] and not r["send"] and not r["matched"] for r in rows)
    # one feasible set only: the first control is sent, the rest are copies and are not
    rec = {"status": "restored", "unit_ids": ["e0000"], "chars": 100}
    rows = CT.build_controls("c", "recent", _units([100, 100]), rec, 100.0)
    assert [r["send"] for r in rows] == [True, False, False, False, False] and rows[1]["duplicate"] and rows[1]["matched"]
    with pytest.raises(ValueError):
        CT.build_controls("c", "recent", _units([10]), {"status": "restored", "unit_ids": ["e9999"], "chars": 5}, 5.0)


# ---------------------------------------------------------------------------
# A synthetic frame for the engine tests
# ---------------------------------------------------------------------------

def _make_root(tmp_path, n_cases=12):
    """frame.jsonl, cases.jsonl, v3_min_sets.jsonl, calibration and the C1 draws for a small synthetic frame."""
    from lens.flip.headline.engine import make_headline_sample
    from lens.flip.systems import RecentSystem
    root = tmp_path / "x5"
    root.mkdir()
    cases = make_headline_sample(n_cases, 5)
    write_jsonl(root / "cases.jsonl", cases)
    frame, flips = [], 0
    for i, c in enumerate(cases):
        v = RecentSystem().view(c, 1000)
        flip = i % 3 != 2
        flips += flip
        recov = flip and i % 2 == 0
        frame.append({"case_id": c.case_id, "system": "recent", "mechanism": c.mechanism, "a_star": c.a_star,
                      "a_H": c.a_star, "a_A": c.a_star, "view": {"event_ids": v.event_ids}, "source": "results/synthetic",
                      "v3": {"a_B": "REQUEST_INFO" if flip else c.a_star, "a_floor": c.a_star, "a_trunc": c.a_star,
                             "note": f"Recommend {c.a_star}. The records show {len(v.units)} items for {c.query_period}.",
                             "note_tokens": 20, "note_status": "ok", "p_residue": 0.1 + 0.05 * (i % 7), "omission": 1.0,
                             "residue_flip": flip, "search_status": ("restored" if recov else "not_recoverable_by_insertion")
                             if flip else None, "view_units": len(v.units), "n_candidates": len(v.units),
                             "dstar_in_view": False}})
    (root / "frame.jsonl").write_text("".join(json.dumps(f, sort_keys=True, separators=(",", ":")) + "\n" for f in frame))
    hs = F.load_frame(root)
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
    (root / F.MIN_SETS).write_text("".join(json.dumps(r, sort_keys=True, separators=(",", ":")) + "\n" for r in ms))
    F.write_calibration(root)
    F.stage_controls(root)
    return root, hs


def _h(s: str) -> int:
    return zlib.crc32(s.encode())


def _answers(cases):
    from lens.flip.generator import oracle
    by = {c.case_id: c for c in cases}

    def answer(cid, params, meta):
        case = by[meta["case_id"]]
        user = params["messages"][0]["content"]
        kind = meta["kind"]
        if kind == "decide":
            # a_star unless a hash of the whole request says otherwise (so repeats can differ only by salt: they do not)
            act = case.a_star if _h(rq.canonical(params)) % 3 else "REQUEST_INFO"
            text = json.dumps({"action": act, "reason": "mock", "event_ids": []})
        elif kind == "note":
            text = json.dumps({"note": f"Recommend {case.a_star}. Short note.", "p_residue": (_h(user) % 100) / 100})
        elif kind == "note_x2":
            text = json.dumps({**{k: f"{k} text" for k in P.X2_FIELDS}, "p_residue": (_h(user) % 100) / 100})
        elif kind == "rewrite":
            note = user.split("Note:\n", 1)[1]
            text = json.dumps({"note": "Rewritten: " + note})
        elif kind == "judge":
            text = json.dumps({"conveyed": bool(_h(user) % 2)})
        else:
            raise AssertionError(kind)
        return {"custom_id": cid, "result": {"type": "succeeded", "message": {
            "content": [{"type": "text", "text": text}], "stop_reason": "end_turn",
            "usage": {"input_tokens": len(user) // 2 + 331, "output_tokens": 100}}}}
    return answer


def _run(root, arm, cap=100.0, check=False, **kw):
    d = root / arm
    if not d.exists():
        d = E.init_arm(arm, root)
    cases = F.load_cases(root)
    api = FakeBatchAPI(_answers(cases), engine_meta_fn(d), **kw)
    r = E.run_arm(d, api, cap_usd=cap, root=root, log=lambda m: None, sleep=lambda s: None, check_protocol=check,
                  max_steps=400)
    return r, api, d


# ---------------------------------------------------------------------------
# C3: cache bypass
# ---------------------------------------------------------------------------

def test_c3_five_identical_bodies_five_keys(tmp_path):
    root, hs = _make_root(tmp_path)
    d = E.init_arm("C3", root)
    seen: list[dict] = []
    api = FakeBatchAPI(_answers(F.load_cases(root)), engine_meta_fn(d), on_create=seen.extend)
    assert E.run_arm(d, api, cap_usd=100.0, root=root, log=lambda m: None, sleep=lambda s: None,
                     check_protocol=False, max_steps=50) == "complete"
    n = len(hs)
    assert api.requests_sent == 5 * n and len(set(api.sent_custom_ids)) == 5 * n
    # five requests per handoff with one and the same body
    bodies: dict[str, set] = {}
    rnd = json.loads((d / "rounds" / "001" / "round.json").read_text())["requests"]
    for r in seen:
        bodies.setdefault(rnd[r["custom_id"]]["case_id"], set()).add(rq.canonical(r["params"]))
    assert len(bodies) == n and all(len(v) == 1 for v in bodies.values())
    by_case: dict[str, list[dict]] = {}
    for m in rnd.values():
        by_case.setdefault(m["case_id"], []).append(m)
    assert all(len(v) == 5 and len({m["salt"] for m in v}) == 5 and len({m["key"] for m in v}) == 5
               for v in by_case.values())
    cfg = json.loads((d / "config.json").read_text())
    assert cfg["cache_bypass"] is True and (d / "cache_bypass.json").exists()
    st = json.loads((d / "state" / "cases" / f"{hs[0].case.case_id}.json").read_text())
    assert len(st["recent"]["out"]["a_B"]) == 5
    h = hs[0]
    body = rq.decide_request(note_view(h.row["v3"]["note"]), [], h.case, "v1", cfg["decider"])
    assert rq.request_key(body) not in {m["key"] for m in rnd.values()}  # the plain body key is never used
    assert CT.salted_key(rq.canonical(body), "rep0") != CT.salted_key(rq.canonical(body), "rep1")


def test_other_arms_still_dedupe_identical_bodies(tmp_path):
    root, hs = _make_root(tmp_path)
    d = E.init_arm("X1", root)
    eng = E.XEngine(d, root=root, cap_usd=10, check_protocol=False)
    h = hs[0]
    a = E._asker(h, {}, eng)
    build = lambda: rq.decide_request(note_view("same"), [], h.case, "v1", eng.config["decider"])
    from lens.flip.headline.tasks import Pending
    with pytest.raises(Pending):
        a.ask_many([("a", "decide", build, "s", None), ("b", "decide", build, "s", None)])
    assert len(eng.pending) == 1 and sum(eng.waiters.values()) == 2
    with pytest.raises(RuntimeError, match="cache_bypass"):
        a.ask_many([("c", "decide", build, "s", {"salt": "rep0"})])
    eng.close()


# ---------------------------------------------------------------------------
# C4: pair count
# ---------------------------------------------------------------------------

def test_c4_one_request_per_note_unit_pair(tmp_path):
    root, hs = _make_root(tmp_path)
    r, api, d = _run(root, "C4")
    assert r == "complete"
    pairs = sum(len(h.view.units) for h in hs)
    assert api.requests_sent == pairs  # every pair has a different prompt, so nothing is merged
    st = AN._states(root, "C4")
    assert sum(s["out"]["n_units"] for s in st.values()) == pairs


@pytest.mark.skipif(not REAL.exists(), reason="needs the built v4 frame")
def test_real_frame_pairs_and_hashes():
    F.verify_v4_frame(F.V4)
    rows = [json.loads(x) for x in REAL.read_text().splitlines() if x.strip()]
    assert len(rows) == F.N_HANDOFFS == 490
    assert sum(r["v3"]["view_units"] for r in rows) == F.N_PAIRS == 11_240
    assert sum(r["v3"]["residue_flip"] for r in rows) == F.N_FLIPS == 142
    assert sum(r["v3"]["search_status"] == "restored" for r in rows) == F.N_RECOVERABLE == 41
    assert hashlib.sha256(REAL.read_bytes()).hexdigest() == F.FRAME_SHA256


@pytest.mark.skipif(not REAL.exists(), reason="needs the built v4 frame")
def test_real_controls_are_reproducible_and_in_band(tmp_path):
    root = tmp_path / "r"
    meta = F.stage_frame(root, F.V4, F.RESULTS)
    assert meta["n_v3_flips"] == 142 and meta["n_recoverable"] == 41 and meta["n_pairs"] == 11_240
    rows = [json.loads(x) for x in (root / "c1_controls.jsonl").read_text().splitlines()]
    assert len(rows) == 142 * 5
    assert all(0.9 <= r["ratio"] <= 1.1 for r in rows if r["matched"])
    again = F.build_controls_rows(root)
    assert CT.controls_sha(again) == meta["c1"]["c1_controls_sha256"]
    ms = F.load_min_sets(root)
    assert not any(set(r["unit_ids"]) & set(ms[(r["case_id"], r["system"])]["unit_ids"]) for r in rows)
    F.stage_frame(root, F.V4, F.RESULTS)  # idempotent


# ---------------------------------------------------------------------------
# Decomposition arithmetic
# ---------------------------------------------------------------------------

def _toy_rows():
    rows = []
    # 10 handoffs; 4 v3 flips. Flips A, B assessed with targeted restore 1, 0; controls restore 0.2 and 0.4.
    # Flip C has no matched control (not assessed). Flip D assessed, targeted 1, controls 0.
    flips = {0: (1, 1, 0.2), 1: (1, 0, 0.4), 2: (0, None, None), 3: (1, 1, 0.0)}
    for i in range(10):
        r = {"v3_flip": int(i in flips), "rw_has": 1, "rw_flip": int(i in (0, 1, 4, 5)), "rep_share": 0.1 if i < 5 else 0.0}
        if i in flips and flips[i][0]:
            r.update({"c1_assessed": 1, "c1_t": flips[i][1], "c1_c": flips[i][2]})
        rows.append(r)
    return rows


def test_decomposition_toy_arithmetic():
    s = AN.decomposition_sums(_toy_rows())
    assert s["n"] == 10 and s["flips"] == 4 and s["assessed"] == 3
    assert s["t"] == 2 and s["c"] == pytest.approx(0.6)
    d = AN.decompose(s)
    assert d["r_q"] == pytest.approx(0.4)
    assert d["omission"] == pytest.approx((2 - 0.6) / 10)          # 0.14
    assert d["wording"] == pytest.approx((4 - 0.5) / 10)           # 0.35: rewrite flips 4, repeat share sum 0.5
    assert d["unattributed"] == pytest.approx(0.4 - 0.14 - 0.35)   # -0.09, reported not hidden
    assert d["omission"] + d["wording"] + d["unattributed"] == pytest.approx(d["r_q"])
    # no rewrite rows: wording and the remainder are undefined, not zero
    none = AN.decompose(AN.decomposition_sums([{"v3_flip": 1}]))
    assert none["wording"] is None and none["unattributed"] is None and none["r_q"] == 1.0
    assert AN.decompose(AN.decomposition_sums([]))["r_q"] is None


def test_decomposition_bootstrap_identity():
    from lens.flip.headline.metrics import _features
    cases = [type("C", (), {"case_id": f"c{i}"})() for i in range(5)]
    rows = [{**r, "case_id": f"c{i // 2}", "system": "recent"} for i, r in enumerate(_toy_rows())]
    st = AN.NStats(cases, rows)
    blk = AN.decomposition_block(st, rows)
    v = {k: blk[k]["value"] for k in ("r_q", "omission", "wording", "unattributed")}
    assert v["omission"] + v["wording"] + v["unattributed"] == pytest.approx(v["r_q"])
    assert blk["n_assessed_flips"] == 3 and blk["identity"].startswith("r_q =")
    f = _features(cases, rows, AN.decomposition_row)
    assert sum(f["n"]) == 10


def test_auroc_vs_auroc_and_ratio_helpers():
    cases = [type("C", (), {"case_id": f"c{i}"})() for i in range(6)]
    rows = []
    for i in range(12):
        rows.append({"case_id": f"c{i % 6}", "system": "recent", "p1": i / 12, "f1": int(i >= 6), "p3": 0.5, "f3": int(i % 2)})
    st = AN.NStats(cases, rows)
    d = st.auroc_vs_auroc(("p1", "f1"), ("p3", "f3"))
    assert d["diff"] == pytest.approx(1.0 - 0.5)
    r = st.ratio(lambda x: x["f1"], lambda x: 1)
    assert r["value"] == 0.5 and r["k"] == 6 and r["n"] == 12


# ---------------------------------------------------------------------------
# Spend guard and freeze
# ---------------------------------------------------------------------------

def test_cap_refuses_before_any_request(tmp_path):
    root, hs = _make_root(tmp_path)
    d = E.init_arm("C4", root)
    api = FakeBatchAPI(lambda *a: None, engine_meta_fn(d))
    with pytest.raises(Refused, match="cap|upper bound"):
        E.run_arm(d, api, cap_usd=0.0001, root=root, log=lambda m: None, sleep=lambda s: None, check_protocol=False,
                  max_steps=5)
    assert api.created == 0
    led = _read_json(root / "spend.json", {})
    assert led["cap_usd"] == 0.0001 and led["total_usd"] == 0


def test_cap_counts_other_arms_spend(tmp_path):
    root, hs = _make_root(tmp_path)
    r, api, d = _run(root, "X1", cap=100.0)
    assert r == "complete"
    spent = E.update_ledger(root, 100.0)["total_usd"]
    assert spent > 0
    # a cap just above what X1 spent leaves no room for C3's round: refused, nothing sent
    d3 = E.init_arm("C3", root)
    api3 = FakeBatchAPI(lambda *a: None, engine_meta_fn(d3))
    with pytest.raises(Refused):
        E.run_arm(d3, api3, cap_usd=spent + 1e-6, root=root, log=lambda m: None, sleep=lambda s: None,
                  check_protocol=False, max_steps=5)
    assert api3.created == 0


def _all_meta(root):
    def read():
        out = {}
        for arm in E.ARMS:
            p = root / arm / "state" / "round.json"
            if p.exists():
                out.update(json.loads(p.read_text())["requests"])
        return out
    return read


def test_run_arms_selects_a_subset_in_order_and_cap_stops_the_rest(tmp_path):
    root, hs = _make_root(tmp_path)
    with pytest.raises(ValueError):
        E.run_arms(root, ["X1", "X1"], cap_usd=1)
    with pytest.raises(ValueError):
        E.run_arms(root, ["C9"], cap_usd=1)
    cases = F.load_cases(root)
    api = FakeBatchAPI(_answers(cases), _all_meta(root))
    out = E.run_arms(root, ["C1", "C2"], cap_usd=50, runner=api, log=lambda m: None, sleep=lambda s: None,
                     check_protocol=False)
    assert out == {"C2": "complete", "C1": "complete"} and list(out) == ["C2", "C1"]  # RUN_ORDER, not the order given
    assert not (root / "X1").exists() and not (root / "C4").exists() and not (root / "X2").exists()
    assert set(E.update_ledger(root, 50.0)["arms"]) == {"C1", "C2"}
    # a cap below the first round of the next arm stops the whole run: later arms are never created
    api2 = FakeBatchAPI(_answers(cases), _all_meta(root))
    spent = E.update_ledger(root, 50.0)["total_usd"]
    with pytest.raises(Refused):
        E.run_arms(root, ["C3", "C4"], cap_usd=spent + 1e-6, runner=api2, log=lambda m: None, sleep=lambda s: None,
                   check_protocol=False)
    assert api2.created == 0 and not (root / "C4").exists()


def test_freeze_is_draft_and_engine_refuses_until_final(tmp_path):
    root, hs = _make_root(tmp_path)
    rec = PR.freeze(root, now=0)
    assert rec["status"] == "DRAFT" and rec["version"] == "5-exploratory" and rec["schema"] == "lens.flip.exploratory/1"
    assert set(rec["arm_configs"]) == set(E.ARMS) and len(rec["prompts"]) == 13
    assert rec["frame_sha256"] == hashlib.sha256((root / "frame.jsonl").read_bytes()).hexdigest()
    with pytest.raises(SystemExit):
        PR.freeze(root)
    assert PR.check(root) == ["status:DRAFT"]
    d = E.init_arm("C4", root)
    api = FakeBatchAPI(lambda *a: None, engine_meta_fn(d))
    with pytest.raises(Refused, match="status:DRAFT"):
        E.run_arm(d, api, cap_usd=10, root=root, log=lambda m: None, sleep=lambda s: None, check_protocol=True, max_steps=3)
    assert api.created == 0
    assert PR.finalize(root)["status"] == "FINAL" and PR.check(root, d) == []


def test_check_catches_prompt_and_file_changes(tmp_path, monkeypatch):
    root, hs = _make_root(tmp_path)
    PR.freeze(root)
    PR.finalize(root)
    monkeypatch.setattr(P, "READER_PARAGRAPH", P.READER_PARAGRAPH + " Extra.")
    bad = PR.check(root)
    assert "prompt:x5.x1.prompt" in bad and "prompt:x5.x1.reader_paragraph" in bad
    monkeypatch.undo()
    assert PR.check(root) == []
    (root / "frame.jsonl").write_text((root / "frame.jsonl").read_text() + "\n")
    assert "root:frame_sha256" in PR.check(root)


def test_cli_guards(tmp_path, capsys):
    from lens.flip.explore5.__main__ import main, parse_arms
    with pytest.raises(SystemExit, match="price-confirmed"):
        main(["run", "--results", str(tmp_path), "--cap-usd", "5"])
    with pytest.raises(SystemExit, match="cap-usd"):
        main(["run", "--results", str(tmp_path), "--price-confirmed"])
    with pytest.raises(SystemExit, match="price-confirmed"):
        main(["finalize", "--results", str(tmp_path)])
    with pytest.raises(SystemExit):
        parse_arms("X1,Z9")
    assert parse_arms("C4,X1") == ["C4", "X1"] and parse_arms(None) == list(E.ARMS)


# ---------------------------------------------------------------------------
# Every arm end to end, then the report
# ---------------------------------------------------------------------------

def test_all_arms_end_to_end_and_report(tmp_path):
    root, hs = _make_root(tmp_path)
    PR.freeze(root)
    PR.finalize(root)
    cases = F.load_cases(root)
    apis = {}
    for arm in E.RUN_ORDER:
        d = E.init_arm(arm, root)
        api = apis[arm] = FakeBatchAPI(_answers(cases), engine_meta_fn(d))
        r = E.run_arm(d, api, cap_usd=100.0, root=root, log=lambda m: None, sleep=lambda s: None, max_steps=400)
        assert r == "complete", arm
    led = _read_json(root / "spend.json", {})
    assert set(led["arms"]) == set(E.ARMS) and led["total_usd"] > 0 and led["cap_usd"] == 100.0
    n = len(hs)
    # request counts per arm
    assert apis["X1"].requests_sent == 2 * n and apis["C2"].requests_sent == 2 * n
    assert apis["C3"].requests_sent == 5 * n and apis["C4"].requests_sent == sum(len(h.view.units) for h in hs)
    sent = sum(json.loads(x)["send"] for x in (root / "c1_controls.jsonl").read_text().splitlines())
    assert 0 < apis["C1"].requests_sent <= sent
    s = AN.write(root)
    assert s["schema"] == "lens.flip.exploratory/1" and s["skipped"] == []
    assert s["X1_reader_conditioned"]["residue_rate"]["n"] == n
    assert s["X2_structured"]["structure"]["replies_with_all_five_fields"] == n
    assert s["C3_repeats"]["cache_bypass"] and s["C3_repeats"]["repeats_per_handoff"] == 5
    assert s["C4_semantic_absence"]["n_pairs"] == sum(len(h.view.units) for h in hs)
    assert s["C2_rewrite"]["n_rewrites"] == n
    dec = s["decomposition"]
    assert dec["omission"]["value"] + dec["wording"]["value"] + dec["unattributed"]["value"] == pytest.approx(dec["r_q"]["value"])
    assert s["C1_irrelevant_insertion"]["n_v3_flips"] == sum(h.row["v3"]["residue_flip"] for h in hs)
    # resume: a second run of a finished arm submits nothing
    d = root / "X1"
    api = FakeBatchAPI(_answers(cases), engine_meta_fn(d))
    assert E.run_arm(d, api, cap_usd=100.0, root=root, log=lambda m: None, sleep=lambda s: None) == "complete"
    assert api.created == 0


def test_resume_after_stop(tmp_path):
    root, hs = _make_root(tmp_path)
    d = E.init_arm("C2", root)
    api = FakeBatchAPI(_answers(F.load_cases(root)), engine_meta_fn(d))
    kw = dict(cap_usd=50, root=root, log=lambda m: None, sleep=lambda s: None, check_protocol=False)
    assert E.run_arm(d, api, max_steps=1, **kw) == "submitted"       # round 1 in flight, process "dies"
    assert E.run_arm(d, api, max_steps=400, **kw) == "complete"      # a new engine picks the round up
    assert api.created == 2                                           # one batch per round, none re-created
    assert E.run_arm(d, api, max_steps=5, **kw) == "complete" and api.created == 2


# ---------------------------------------------------------------------------
# Dry run and the private word list
# ---------------------------------------------------------------------------

def test_dryrun_prices_every_arm_without_a_call(tmp_path):
    from lens.flip.explore5 import dryrun as D
    root, hs = _make_root(tmp_path)
    res = D.dryrun(root, tmp_path / "no-results", write=True)
    assert res["api_calls"] == 0 and res["count_tokens_calls"] == 0 and res["submitted"] is False
    assert set(res["arms"]) == set(E.ARMS)
    for a, x in res["arms"].items():
        assert 0 < x["likely_usd"] < x["upper_usd"], a
        assert x["rounds_likely"] <= x["rounds_upper"]
    assert res["arms"]["C4"]["requests"] == sum(len(h.view.units) for h in hs)
    assert res["arms"]["C3"]["requests"] == 5 * len(hs)
    assert res["total"]["likely_usd"] == pytest.approx(sum(x["likely_usd"] for x in res["arms"].values()), abs=1e-3)
    assert D.tokens_likely(2200) == 1000 + 331 and D.tokens_bound(1902) > 1000 + 331
    one = D.stage_price("s", [2200], 100, 6000)
    assert one["likely_usd"] == pytest.approx((1331 * 0.05 + 100 * 0.25) / 1e6, abs=1e-6)
    assert one["upper_usd"] == pytest.approx((D.tokens_bound(2200) * 0.05 + 6000 * 0.25) / 1e6, abs=1e-6)
    assert (root / "dryrun_v5.json").exists()


def test_private_word_list_absent_from_new_code_prompts_and_doc(tmp_path):
    banned_file = REPO / "banned_words.local.txt"
    if not banned_file.exists():
        pytest.skip("banned_words.local.txt not present")
    banned = [w.strip().lower() for w in banned_file.read_text().splitlines() if w.strip()]
    files = sorted((REPO / "src" / "lens" / "flip" / "explore5").glob("*.py")) + [Path(__file__)]
    doc = REPO / "docs" / "flip-protocol-v5-exploratory.md"
    if doc.exists():
        files.append(doc)
    blob = "\n".join(f.read_text().lower() for f in files)
    blob += "\n".join(PR.prompt_texts().values()).lower()
    root, hs = _make_root(tmp_path)
    # requests built from the synthetic frame carry only the v5 templates (the notes there are written here)
    h = hs[0]
    for params in (E.x1_request(h.view, h.case, E.NOTE_WRITER), E.x2_request(h.view, h.case, E.NOTE_WRITER),
                   E.rewrite_request("a note", E.REWRITER), E.judge_request("a note", "a unit", E.JUDGE)):
        blob += rq.canonical(params).lower()
    assert [w for w in banned if w in blob] == []

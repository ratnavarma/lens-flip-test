"""Step 5: gate and metrics, views, summary, embedding retrieval, residue stage. No test calls Ollama."""

from __future__ import annotations

import dataclasses
import json
import math
import re
from pathlib import Path

import pytest

from lens.flip import batch, residue
from lens.flip.cache import CallCache
from lens.flip.decider import ADDITIONAL, PARSE_ERROR, CallStats, build_messages, oracle_decider
from lens.flip.flip import flip_search
from lens.flip.generator import Knobs, generate_cases
from lens.flip.llm import (
    TEXT_FORMAT, CachedEmbedder, OllamaText, embed_key, extract_text, parse_embeddings,
)
from lens.flip.metrics import (
    GATE_OVERALL, GATE_PER_MECHANISM, fcr_gated, fcr_raw, gate_report, harmful_flips, validity_gate,
)
from lens.flip.generator import oracle
from lens.flip.render import render_lines
from lens.flip.runner import RunConfig, run_flip, summarise
from lens.flip.schema import MECHANISMS, DistinctionRecord, Unit, View
from lens.flip.systems import RagEmbedSystem, RecentSystem, SummarySystem, get_system
from lens.flip.systems.base import count_tokens, event_view, rendered, split_sentences, truncate_tokens
from lens.flip.systems.rag_embed import cosine
from lens.flip.systems.summary import build_summary_prompt, ollama_summarizer

# The banned domain words are private. They live in an optional, gitignored file at the repo root,
# one word per line. Without it, the word checks skip.
_BANNED_FILE = Path(__file__).resolve().parents[2] / "banned_words.local.txt"
FORBIDDEN = ([w.strip().lower() for w in _BANNED_FILE.read_text().splitlines() if w.strip()]
             if _BANNED_FILE.exists() else [])

_ID = re.compile(r"^(e\d+) ", re.M)


def text_oracle(view: View, inserted: list[Unit], case) -> str:
    """Rule oracle over the event ids that start a line of the view text or an inserted unit."""
    ids = set(_ID.findall(view.text))
    for u in inserted:
        ids |= set(_ID.findall(u.text))
    return oracle([e for e in case.events if e.id in ids], case.query_period)


@pytest.fixture(scope="module")
def cases():
    return generate_cases(30, 8, Knobs(length=100))


@pytest.fixture(scope="module")
def case(cases):
    return cases[0]


# ----- A. gate and metrics -----


def test_gate_thresholds():
    assert (GATE_OVERALL, GATE_PER_MECHANISM) == (0.90, 0.80)
    ok = {m: 0.9 for m in MECHANISMS}
    assert validity_gate(0.90, ok)
    assert validity_gate(0.90, {**ok, "U2": 0.80})  # exactly 0.80 passes
    assert not validity_gate(0.8999, ok)
    assert not validity_gate(0.95, {**ok, "U2": 0.7999})
    assert not validity_gate(None, ok)
    assert validity_gate(0.95, {**ok, "M3": None})  # a mechanism with no cases is skipped
    assert 56 / 60 >= GATE_OVERALL and validity_gate(56 / 60, {**ok, "U2": 8 / 10})  # the v4 numbers
    assert gate_report(0.95, ok) == {"overall_min": 0.9, "per_mechanism_min": 0.8, "passes": True}


def _line(cid, action):
    return {"custom_id": cid, "result": {"type": "succeeded", "message": {
        "content": [{"type": "text", "text": json.dumps({"action": action, "reason": "r", "event_ids": []})}],
        "stop_reason": "end_turn", "usage": {"input_tokens": 1, "output_tokens": 1}}}}


@pytest.mark.parametrize("wrong_in_m1, expect", [(2, True), (3, False)])
def test_batch_score_uses_the_shared_gate(wrong_in_m1, expect):
    cs = batch.make_sample(60, 11)
    req_ids = [batch.custom_id(c.case_id, "adaptive-high") for c in cs]
    lines, left = [], wrong_in_m1
    for c, cid in zip(cs, req_ids):
        wrong = c.mechanism == "M1" and left > 0
        left -= wrong
        a = ("EXPLAIN" if c.a_star != "EXPLAIN" else "OPEN_REVIEW") if wrong else c.a_star
        lines.append(_line(cid, a))
    _, validity = batch.score(cs, req_ids, lines)
    arm = validity["arms"]["adaptive-high"]
    assert arm["overall"]["accuracy"] >= 0.9  # 57/60 or 58/60 either way
    assert arm["per_mechanism"]["M1"]["accuracy"] == (10 - wrong_in_m1) / 10
    assert arm["passes_gate"] is expect
    assert validity["gate"] == {"overall_min": 0.9, "per_mechanism_min": 0.8}


def _row(mech, a_star, a_h, a_v):
    return {"mechanism": mech, "a_star": a_star, "a_H": a_h, "a_V": a_v, "view_events": 1, "view_tokens": 1}


TOY = [
    _row("M0", "EXPLAIN", "EXPLAIN", "EXPLAIN"),
    _row("M0", "EXPLAIN", "EXPLAIN", "REQUEST_INFO"),      # gated, harmful flip, not committed
    _row("M1", "OPEN_REVIEW", "OPEN_REVIEW", "EXPLAIN"),   # gated, committed, false closure
    _row("M2", "CHECK_DEVICE", "EXPLAIN", "OPEN_REVIEW"),  # a_H wrong: raw false closure, not gated
    _row("M3", "APPLY_CREDIT", "APPLY_CREDIT", "APPLY_CREDIT"),
    _row("U1", "REQUEST_INFO", "REQUEST_INFO", "OPEN_REVIEW"),  # gated, committed, false closure
    _row("U2", "REQUEST_INFO", "EXPLAIN", PARSE_ERROR),    # a_H wrong, not committed
]


def test_gated_fcr_on_a_toy_table():
    raw = fcr_raw(TOY)
    assert raw == {"n": 7, "committed": 5, "false_closures": 3, "fcr": 3 / 5}
    gated = fcr_gated(TOY)
    assert gated == {"n": 5, "committed": 4, "false_closures": 2, "fcr": 2 / 4}
    h = harmful_flips(TOY)
    assert h["count"] == 3 and h["rate"] == 3 / 7 and h["gated_n"] == 5 and h["rate_gated"] == 3 / 5
    assert fcr_gated([_row("M0", "EXPLAIN", "OPEN_REVIEW", "OPEN_REVIEW")])["fcr"] is None


def test_summary_json_labels_both_rates():
    s = summarise(RunConfig(system="recent", budget=1, n=7), TOY, [])
    assert s["fcr_raw"]["all"]["fcr"] == 3 / 5 and s["fcr_gated"]["all"]["fcr"] == 2 / 4
    assert s["fcr_gated"]["rare"] == fcr_gated([r for r in TOY if r["mechanism"] != "M0"])
    assert "fcr" not in s
    assert s["validity"]["gate"]["passes"] is False and s["validity"]["gate"]["per_mechanism_min"] == 0.8
    assert "residue" not in s


# ----- B. View abstraction -----


def test_event_view_fields(case):
    ids = [case.events[5].id, case.events[2].id]
    v = event_view(case, ids, "t", 99)
    assert v.event_ids == [case.events[2].id, case.events[5].id]  # history order
    assert v.units == render_lines(case, v.event_ids) and v.text == "\n".join(v.units)
    assert (v.system, v.budget) == ("t", 99)


def test_additional_records_block_follows_the_view_in_given_order(case):
    v = event_view(case, [e.id for e in case.events[:3]], "t")
    extra = [Unit(id=e.id, text=line, event=e) for e, line in zip(case.events[10:12], render_lines(
        case, [e.id for e in case.events[10:12]]))]
    user = build_messages(v, extra, case, "v1")[1]["content"]
    pos = [user.index(x) for x in (v.text, ADDITIONAL, extra[0].text, extra[1].text, case.query_text)]
    assert pos == sorted(pos) and ADDITIONAL == "Additional records:"
    assert ADDITIONAL not in build_messages(v, [], case, "v1")[1]["content"]
    # no view text at all
    empty = build_messages(View("t", 1, ""), extra[:1], case, "v1")[1]["content"]
    assert "(no records)" in empty and empty.index("(no records)") < empty.index(ADDITIONAL)


def test_flip_search_inserts_candidates_outside_the_view_in_history_order(cases):
    c = next(c for c in cases if c.mechanism == "M1")
    view = event_view(c, [e.id for e in c.events if e.id not in set(c.d_star)], "t")
    seen: list[tuple[View, list[Unit]]] = []

    def decide(v, ins, case):
        seen.append((v, list(ins)))
        return oracle_decider(v, ins, case)

    r = flip_search(c, view, decide, c.a_star, system="t", budget=None, model="oracle")
    assert r.status == "restored" and r.event_ids == c.d_star
    order = {e.id: i for i, e in enumerate(c.events)}
    for v, ins in seen:
        assert v is view
        assert all(u.id not in set(view.event_ids) for u in ins)
        assert [order[u.id] for u in ins] == sorted(order[u.id] for u in ins)
    assert [u.id for u in seen[1][1]] == c.d_star  # the "insert every missing event" call
    assert r.stage == "flip" and r.units_inserted == r.event_texts


def test_summary_view_makes_every_event_a_candidate(case):
    sview = View(system="summary", budget=500, text="Some summary.", event_ids=[], units=["Some summary."])
    seen = []

    def decide(v, ins, c):
        seen.append(list(ins))
        return "EXPLAIN" if not ins else "OPEN_REVIEW"

    r = flip_search(case, sview, decide, "OPEN_REVIEW", system="summary", budget=500, model="fake", max_calls=6)
    assert seen[0] == [] and [u.id for u in seen[1]] == [e.id for e in case.events]
    assert r.status == "restored"


def test_old_record_loads_without_the_new_fields(case):
    v = event_view(case, [], "t")
    r = flip_search(case, v, oracle_decider, case.a_star, system="t", budget=None, model="oracle")
    old = {k: x for k, x in r.to_dict().items() if k not in ("stage", "units_inserted")}
    loaded = DistinctionRecord.from_dict(json.loads(json.dumps(old)))
    assert loaded.stage == "flip" and loaded.units_inserted == []
    assert DistinctionRecord.from_dict(r.to_dict()) == r


# ----- text helpers -----


def test_truncate_tokens_uses_the_shared_count():
    text = "alpha beta gamma delta " * 50
    for b in (1, 7, 50, 200):
        out = truncate_tokens(text, b)
        assert count_tokens([out]) <= b
    assert truncate_tokens("short text", 50) == "short text"
    assert truncate_tokens("abcdefghij klmno", 5) == "abcdefghij"  # cut back to a word edge
    assert truncate_tokens("abcdefghijklmno", 3) == "abcdef"  # no space to cut back to (int(3 * 2.2) = 6 characters)


def test_split_sentences():
    text = "Reading e0001 was 80.5. Credit applied!\n- Device replaced? Yes.\n\n* Plan moved."
    assert split_sentences(text) == ["Reading e0001 was 80.5.", "Credit applied!", "Device replaced?", "Yes.",
                                     "Plan moved."]
    assert split_sentences("") == []


# ----- C. summary system -----


def test_summary_chunks_chain_and_units(case):
    calls = []

    def summarize(prev, chunk, budget):
        calls.append((prev, list(chunk), budget))
        return f"Pass {len(calls)} done. Next idea."

    n = len(case.events)
    v = SummarySystem(summarize).view(case, 700)
    sizes = [40] * (n // 40) + ([n % 40] if n % 40 else [])
    assert [len(x[1]) for x in calls] == sizes and len(sizes) == math.ceil(n / 40)
    assert [x[2] for x in calls] == [700] * len(calls)
    assert calls[0][0] == "" and [x[0] for x in calls[1:]] == [f"Pass {k} done. Next idea." for k in range(1, len(calls))]
    assert [ln for x in calls for ln in x[1]] == [line for _, line in rendered(case)]
    assert v.text == f"Pass {len(calls)} done. Next idea."
    assert v.event_ids == [] and v.units == [f"Pass {len(calls)} done.", "Next idea."]
    assert v.system == "summary" and v.budget == 700


def test_summary_is_cut_to_the_budget(case):
    v = SummarySystem(lambda prev, chunk, b: "word " * 3000).view(case, 60)
    assert count_tokens([v.text]) <= 60 and len(v.text) > 100
    assert v.units == split_sentences(v.text)


def test_summary_prompt_never_contains_the_query(case):
    prompts: list[tuple[str, int]] = []

    def gen(prompt, num_predict):
        prompts.append((prompt, num_predict))
        return "A fact. Another fact."

    other = dataclasses.replace(case, query_text="A totally different request about period 1999-01.")
    v1 = SummarySystem(ollama_summarizer(gen)).view(case, 500)
    n1 = len(prompts)
    v2 = SummarySystem(ollama_summarizer(gen)).view(other, 500)
    assert n1 == math.ceil(len(case.events) / 40) and len(prompts) == 2 * n1
    for prompt, num_predict in prompts:
        assert case.query_text not in prompt and other.query_text not in prompt
        assert "higher than usual" not in prompt and "Choose one action" not in prompt
        assert num_predict == 750
    assert [p for p, _ in prompts[:n1]] == [p for p, _ in prompts[n1:]]  # same prompts, whatever the query
    assert v1 == v2
    p0 = build_summary_prompt("", ["e1 line"], 500)
    assert "(empty)" in p0 and "e1 line" in p0 and "500 tokens" in p0


def test_summary_needs_a_summarizer_and_rag_embed_an_embedder():
    with pytest.raises(ValueError):
        get_system("summary")
    with pytest.raises(ValueError):
        get_system("rag-embed")
    assert isinstance(get_system("summary", summarizer=lambda p, c, b: "x"), SummarySystem)
    assert isinstance(get_system("rag-embed", embedder=lambda t: [[1.0]] * len(t)), RagEmbedSystem)


# ----- D. embedding retrieval -----


def test_cosine():
    assert cosine([1, 0], [1, 0]) == pytest.approx(1.0)
    assert cosine([1, 0], [0, 1]) == 0.0
    assert cosine([1, 1], [-1, -1]) == pytest.approx(-1.0)
    assert cosine([0, 0], [1, 0]) == 0.0


def _planted_embedder(case, chosen):
    """Lines of the chosen events point at the query; the rest are orthogonal to it."""
    starts = tuple(f"{i} " for i in chosen)

    def embed(texts):
        out = []
        for t in texts:
            if t == case.query_text or t.startswith(starts):
                out.append([1.0, 0.0])
            else:
                out.append([0.0, 1.0])
        return out

    return embed


def test_embed_top_k_order_and_budget(case):
    chosen = [case.events[i].id for i in (12, 40, 41, 77, 90)]
    sysm = RagEmbedSystem(_planted_embedder(case, chosen))
    big = sysm.view(case, 600)
    order = {e.id: i for i, e in enumerate(case.events)}
    assert set(chosen) <= set(big.event_ids)
    assert big.event_ids == sorted(big.event_ids, key=order.__getitem__)
    assert 540 <= count_tokens([big.text]) <= 600
    assert big.text == "\n".join(render_lines(case, big.event_ids)) and big.system == "rag-embed"
    # the fill after the chosen lines goes to the latest events (ties to the later one)
    rest = [i for i in big.event_ids if i not in chosen]
    assert rest == [e.id for e in case.events if e.id not in chosen][-len(rest):]
    # a small budget keeps the latest chosen lines only
    small = sysm.view(case, 60)
    assert count_tokens([small.text]) <= 60
    assert small.event_ids and set(small.event_ids) <= set(chosen)
    assert small.event_ids == [i for i in chosen if order[i] >= order[small.event_ids[0]]]


def test_embed_asks_for_lines_and_query_together(case):
    seen = []

    def embed(texts):
        seen.append(list(texts))
        return [[1.0, 0.0]] * len(texts)

    RagEmbedSystem(embed).view(case, 100)
    assert len(seen) == 1 and seen[0][-1] == case.query_text
    assert seen[0][:-1] == [line for _, line in rendered(case)]


def test_parse_embeddings_shapes():
    assert parse_embeddings({"embeddings": [[1, 2], [3, 4]]}, 2) == [[1.0, 2.0], [3.0, 4.0]]
    assert parse_embeddings({"embedding": [1, 2]}, 1) == [[1.0, 2.0]]
    assert parse_embeddings({"embedding": [[1, 2]]}, 1) == [[1.0, 2.0]]
    with pytest.raises(RuntimeError):
        parse_embeddings({"embeddings": [[1]]}, 2)
    with pytest.raises(RuntimeError):
        parse_embeddings({"nope": 1}, 1)


def test_embedder_caches_vectors_by_model_and_text(tmp_path):
    calls = []

    def transport(model, texts):
        calls.append((model, list(texts)))
        return [[float(len(t)), 0.5] for t in texts]

    cache = CallCache(tmp_path / "c.sqlite")
    stats = CallStats()
    emb = CachedEmbedder("m1", transport=transport, cache=cache, stats=stats)
    assert emb(["a", "bb", "a"]) == [[1.0, 0.5], [2.0, 0.5], [1.0, 0.5]]
    assert calls == [("m1", ["a", "bb"])] and stats.embed_requests == 1
    assert json.loads(cache.get(embed_key("m1", "a"))) == [1.0, 0.5]
    assert embed_key("m1", "a") != embed_key("m2", "a") != embed_key("m1", "b")
    # a fresh embedder on the same file makes no request
    stats2 = CallStats()
    emb2 = CachedEmbedder("m1", transport=transport, cache=CallCache(tmp_path / "c.sqlite"), stats=stats2)
    assert emb2(["bb", "a"]) == [[2.0, 0.5], [1.0, 0.5]]
    assert len(calls) == 1 and stats2.embed_requests == 0 and stats2.embed_hits == 2
    # a different model misses
    CachedEmbedder("m2", transport=transport, cache=cache)(["a"])
    assert calls[-1][0] == "m2"
    # batching
    big = CachedEmbedder("m3", transport=transport, batch=2)
    big(["a", "b", "c", "d", "e"])
    assert [len(t) for m, t in calls if m == "m3"] == [2, 2, 1]


def test_ollama_text_request_shape_and_cache(tmp_path):
    bodies = []

    def transport(body):
        bodies.append(body)
        return json.dumps({"text": "  Some text.  "})

    cache, stats = CallCache(tmp_path / "t.sqlite"), CallStats()
    gen = OllamaText("qwen3:30b-a3b", transport=transport, cache=cache, stats=stats)
    assert gen("write it", 750) == "Some text."
    b = bodies[0]
    assert b["think"] is False and b["stream"] is False and b["format"] == TEXT_FORMAT
    assert b["options"] == {"temperature": 0, "seed": 0, "num_ctx": 8192, "num_predict": 750}
    assert b["messages"] == [{"role": "user", "content": "write it"}]
    assert gen("write it", 750) == "Some text." and len(bodies) == 1
    assert (stats.real_calls, stats.cache_hits) == (1, 1)
    gen("write it", 751)
    assert len(bodies) == 2


def test_extract_text_variants():
    assert extract_text('{"text": "A. B."}') == "A. B."
    assert extract_text('{\n  "text": "Line one.\\nLine two \\"q\\""\n}') == 'Line one.\nLine two "q"'
    assert extract_text('{"text": "cut off here and') == "cut off here and"  # reply stopped by num_predict
    assert extract_text("plain words") == "plain words"
    assert extract_text("<think>x</think>{\"text\": \"y\"}") == "y"


# ----- E. residue stage -----


def _single_dropped(case, view_ids, x, *, verbatim):
    """A fake note-writer: the view's lines without the one for x. Others stay verbatim or are lower-cased."""
    def write(view, c):
        keep = [ln for i, ln in zip(view.event_ids, view.units) if i != x]
        return "\n".join(keep if verbatim else [ln.lower() for ln in keep])

    return write


def _tight_cases(cs):
    out = []
    for c in cs:
        if c.cls != "rare" or not c.d_star or c.mechanism == "U1":
            continue
        view = event_view(c, c.d_star, "t", 500)
        if text_oracle(view, [], c) == c.a_star and count_tokens([view.text]) <= 150:
            out.append((c, view))
    return out


@pytest.fixture(scope="module")
def rare_cases():
    mix = {"M0": 0.0, "M1": 1.0, "M2": 1.0, "M3": 1.0, "U1": 0.0, "U2": 1.0}
    return generate_cases(120, 21, Knobs(length=100), mix=mix)


@pytest.mark.parametrize("verbatim", [True, False])
def test_residue_search_recovers_exactly_the_dropped_line(rare_cases, verbatim):
    tested = 0
    for c, view in _tight_cases(rare_cases):
        for x in view.event_ids:
            writer = _single_dropped(c, view.event_ids, x, verbatim=verbatim)
            out = residue.run_residue(c, view, writer, text_oracle, c.a_star, c.a_star, system="t",
                                      budget=500, model="fake", max_calls=24)
            note_ids = set(_ID.findall(out.note))
            if text_oracle(residue.note_view(out.note), [], c) == c.a_star:
                assert not out.flipped and out.record is None  # losing the line changed nothing
                continue
            assert out.flipped and out.a_B != c.a_star
            r = out.record
            assert r.stage == "residue" and r.status == "restored"
            assert r.event_ids == [x] and r.units_inserted == r.event_texts == view.units[view.event_ids.index(x):][:1]
            assert x not in note_ids and r.system == "t" and r.budget == 500
            assert DistinctionRecord.from_dict(r.to_dict()) == r
            tested += 1
    assert tested >= 15


def test_residue_candidates_are_units_not_verbatim_in_the_note(rare_cases):
    c, view = _tight_cases(rare_cases)[0]
    note = "\n".join(view.units[:-1])
    cands = residue.residue_candidates(c, view, note)
    assert [u.text for u in cands] == [view.units[-1]] and cands[0].id == view.event_ids[-1]
    assert cands[0].event is not None and cands[0].group
    assert [u.id for u in residue.residue_candidates(c, view, "nothing here")] == view.event_ids


def test_residue_on_a_summary_view_uses_sentences(rare_cases):
    c, ev = _tight_cases(rare_cases)[0]
    sview = View("summary", 500, ev.text, event_ids=[], units=ev.units)
    x = ev.event_ids[-1]
    writer = lambda v, case: "\n".join(v.units[:-1])
    out = residue.run_residue(c, sview, writer, text_oracle, c.a_star, c.a_star, system="summary", budget=500,
                              model="fake")
    if len(ev.units) > 1 and text_oracle(residue.note_view(out.note), [], c) == c.a_star:
        pytest.skip("this case does not depend on its last line")
    assert out.flipped and out.record.stage == "residue"
    assert out.record.event_ids == [f"u{len(ev.units) - 1:03d}"] and out.record.units_inserted == [ev.units[-1]]
    assert out.record.groups == ["sentence"]
    assert x in out.record.units_inserted[0]


def test_residue_no_flip_when_a_is_wrong_or_b_is_right(rare_cases):
    c, view = _tight_cases(rare_cases)[0]
    lossless = lambda v, case: "\n".join(v.units)
    out = residue.run_residue(c, view, lossless, text_oracle, c.a_star, c.a_star, system="t", budget=500, model="f")
    assert out.a_B == c.a_star and not out.flipped and out.record is None
    empty = lambda v, case: "nothing"
    wrong_a = residue.run_residue(c, view, empty, text_oracle, c.a_star, "EXPLAIN" if c.a_star != "EXPLAIN" else "x",
                                  system="t", budget=500, model="f")
    assert not wrong_a.flipped and wrong_a.record is None


def test_note_is_cut_to_150_tokens(rare_cases):
    c, view = _tight_cases(rare_cases)[0]
    out = residue.run_residue(c, view, lambda v, case: "word " * 500, text_oracle, c.a_star, c.a_star,
                              system="t", budget=500, model="f", max_calls=4)
    assert out.note_tokens <= 150 == residue.NOTE_TOKENS


def test_note_prompt_has_view_and_query(case):
    v = event_view(case, [case.events[0].id], "t")
    p = residue.build_note_prompt(v, case)
    assert v.text in p and case.query_text in p and "150 tokens" in p
    seen = []
    w = residue.ollama_note_writer(lambda prompt, n: seen.append((prompt, n)) or "note")
    assert w(v, case) == "note" and seen[0][0] == p and seen[0][1] == 200


def _runner_residue(tmp_path, name):
    cfg = RunConfig(system="recent", budget=1000, n=30, seed=5, length=100, model="fake", max_calls=24,
                    residue=True)
    return cfg, run_flip(cfg, tmp_path / name, text_oracle, note_writer=lambda v, c: "nothing noted")


def test_runner_residue_outputs_and_rate(tmp_path):
    cfg, s = _runner_residue(tmp_path, "a")
    out = tmp_path / "a"
    dec = [json.loads(x) for x in (out / "decisions.jsonl").read_text().splitlines()]
    res = [json.loads(x) for x in (out / "residue.jsonl").read_text().splitlines()]
    recs = [DistinctionRecord.from_dict(json.loads(x)) for x in (out / "residue_records.jsonl").read_text().splitlines()]
    plain = [json.loads(x) for x in (out / "records.jsonl").read_text().splitlines()]
    assert all(r["stage"] == "flip" for r in plain) and all(r.stage == "residue" for r in recs)
    gated = [d for d in dec if d["a_H"] == d["a_star"]]
    a_right = [d for d in gated if d["a_V"] == d["a_star"]]
    assert [r["case_id"] for r in res] == [d["case_id"] for d in a_right]
    wrong = [r for r in res if r["a_B"] != r["a_star"]]
    m = s["residue"]
    assert m["a_A_right"] == len(a_right) and m["a_B_wrong"] == len(wrong) == m["residue_flips"] == len(recs)
    assert m["residue_rate"] == len(wrong) / len(a_right) and len(a_right) > 5 and wrong
    assert all(r["note"] == "nothing noted" and r["residue_flip"] == (r["a_B"] != r["a_star"]) for r in res)
    assert sum(m["status"].values()) == m["searches"] == len(recs)
    assert s["validity"]["gate"]["passes"] in (True, False)
    # same seed, same bytes
    _runner_residue(tmp_path, "b")
    for f in ("decisions.jsonl", "residue.jsonl", "residue_records.jsonl", "records.jsonl"):
        assert (tmp_path / "a" / f).read_bytes() == (tmp_path / "b" / f).read_bytes()


def test_runner_needs_a_note_writer_for_residue(tmp_path):
    with pytest.raises(ValueError):
        run_flip(RunConfig(system="recent", budget=500, n=2, length=100, residue=True), tmp_path, oracle_decider)


def test_runner_with_a_summary_system_keeps_the_summary_text(tmp_path):
    cfg = RunConfig(system="summary", budget=200, n=6, seed=5, length=100, model="fake", max_calls=12)
    sysm = SummarySystem(lambda prev, chunk, b: "Account has records. Several periods are covered.")
    s = run_flip(cfg, tmp_path, text_oracle, system=sysm)
    dec = [json.loads(x) for x in (tmp_path / "decisions.jsonl").read_text().splitlines()]
    assert all(d["view_events"] == 0 and d["view_units"] == 2 and d["view_text"].startswith("Account") for d in dec)
    assert s["config"]["system"] == "summary" and s["n_cases"] == 6


# ----- F. CLI and hygiene -----


def test_cli_accepts_the_new_systems_and_residue():
    import argparse
    from lens.flip import cli

    p = argparse.ArgumentParser()
    cli.register(p.add_subparsers(dest="cmd"))
    a = p.parse_args(["flip", "run", "--system", "summary", "--budget", "500", "--n", "4", "--model", "m",
                      "--residue", "--out", "x"])
    assert a.system == "summary" and a.residue is True and a.summary_model == "qwen3:30b-a3b"
    b = p.parse_args(["flip", "run", "--system", "rag-embed", "--budget", "500", "--n", "4", "--model", "m",
                      "--out", "x"])
    assert b.system == "rag-embed" and b.residue is False and b.embed_model == "nomic-embed-text"
    c = p.parse_args(["flip", "validity", "rescore", "--dir", "x"])
    assert c.func is batch.run_rescore and c.write is False


def test_rescore_offline_matches_score(tmp_path):
    cs = batch.make_sample(60, 11)
    d = tmp_path / "v"
    d.mkdir()
    from lens.flip.schema import write_jsonl
    write_jsonl(d / "cases.jsonl", cs)
    reqs = batch.build_requests(cs, ("adaptive-high",))
    batch._write_lines(d / "requests.jsonl", reqs)
    lines = [_line(r["custom_id"], c.a_star) for r, c in zip(reqs, cs)]
    (d / "results.jsonl").write_text("\n".join(json.dumps(x) for x in lines) + "\n")
    v = batch.rescore_impl(d)
    assert v["arms"]["adaptive-high"]["passes_gate"] is True and not (d / "validity.json").exists()
    batch.rescore_impl(d, write=True)
    assert (d / "validity.json").exists()


def test_no_banned_words_in_new_code_or_prompts(case):
    if not FORBIDDEN:
        pytest.skip("banned_words.local.txt not present")
    root = Path(batch.__file__).parent
    files = [root / n for n in ("llm.py", "residue.py", "metrics.py", "flip.py", "runner.py", "cli.py")]
    files += [root / "systems" / n for n in ("summary.py", "rag_embed.py", "base.py")]
    blob = "\n".join(f.read_text() for f in files).lower()
    blob += build_summary_prompt("s", ["e1 x"], 500).lower()
    blob += residue.build_note_prompt(event_view(case, [case.events[0].id], "t"), case).lower()
    assert [w for w in FORBIDDEN if w in blob] == []

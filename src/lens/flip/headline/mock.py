"""Fake Batches endpoint and the mock model for the dry-run simulation and the tests.

FakeBatchAPI is a runner (method, path, body, timeout) -> (exit code, stdout) like the
anthropic-api wrapper. It serves POST /v1/messages/batches, GET a batch, GET its results and
POST /v1/messages/count_tokens. Any other path, /v1/messages included, raises.

MockModel answers every request from its body and the engine's request meta:

- decide: the rule oracle over the events whose id starts a line of the prompt, plus seeded
  noise. On the full history (stage a_H) the decider is right with the v4 accuracy of the case's
  mechanism, and wrong answers follow the v4 confusion table. On any other prompt the answer
  departs from the oracle with probability eps_small (prompt under 25,000 characters, the size
  of a 100-event history) or eps_large (larger prompts, the 400-event error rate of v4).
- summary: keeps whole record lines (id first), the most informative first, up to 0.8 x B
  tokens, so the summary cites ids as the prompt asks.
- note: keeps each line of V that holds a D* event with probability note_keep_dstar, fills up
  to 150 tokens with the most informative other lines, and gives p_residue from the share of
  V it left out plus noise.

Noise draws are a hash of the request key, so an identical request gets the identical answer.
"""

from __future__ import annotations

import hashlib
import json
import math

from ..systems.base import CHARS_PER_TOKEN
import re
from collections import Counter
from pathlib import Path
from typing import Any, Callable, Sequence

from ..decider import ADDITIONAL
from ..generator import oracle
from ..schema import ACTIONS, Case
from . import requests as rq

_LINE_ID = re.compile(r"(?m)^(e\d+) ")
_TOKEN = re.compile(r"[a-z0-9]+")

V4_DIR = Path("results/validity-haiku-v4")


def h01(*parts: Any) -> float:
    d = hashlib.sha256("|".join(str(p) for p in parts).encode("utf-8")).digest()
    return int.from_bytes(d[:8], "big") / 2**64


class HashEmbedder:
    """Local stand-in for nomic-embed-text in the simulation: hashed bag of words, 256 dimensions."""

    name = "hash-bow-256"

    def __call__(self, texts: Sequence[str]) -> list[list[float]]:
        out = []
        for t in texts:
            v = [0.0] * 256
            for tok in _TOKEN.findall(t.lower()):
                v[int(hashlib.md5(tok.encode()).hexdigest()[:8], 16) % 256] += 1.0
            out.append(v)
        return out


# ---------------------------------------------------------------------------
# Calibration from the v4 validity run
# ---------------------------------------------------------------------------

def v4_calibration(v4_dir: Path = V4_DIR) -> dict[str, Any]:
    """Per-mechanism accuracy, wrong-answer table and output-token sample from results/validity-haiku-v4."""
    val = json.loads((v4_dir / "validity.json").read_text())["arms"]["adaptive-high"]
    rows = [json.loads(x) for x in (v4_dir / "decisions.jsonl").read_text().splitlines() if x.strip()]
    outs = sorted(int(r["usage"].get("output_tokens", 0)) for r in rows if r.get("usage"))
    wrong: dict[str, str] = {}
    for a_star, row in val["confusion"].items():
        bad = {k: v for k, v in row.items() if k != a_star}
        if bad:
            wrong[a_star] = max(sorted(bad), key=lambda k: bad[k])
    per_len = {k: v["accuracy"] for k, v in val["per_length"].items()}
    return {
        "source": str(v4_dir),
        "accuracy_by_mechanism": {m: a["accuracy"] for m, a in val["per_mechanism"].items()},
        "wrong_answer_for": wrong,
        "accuracy_by_length": per_len,
        "output_tokens": outs,
        "mean_output_tokens": sum(outs) / len(outs),
        "input_overhead_tokens": 331,  # usage input minus count_tokens, constant on all 60 v4 requests
    }


def default_noise(calib: dict[str, Any]) -> dict[str, float]:
    """View error rates. v4 had 0 of 30 wrong at 100 events; 0.05 sits between that and the
    rule-of-three bound (3/30). Large prompts take the 400-event error rate."""
    return {"eps_small": 0.05, "eps_large": round(1 - calib["accuracy_by_length"]["400"], 4),
            "small_chars": 25_000, "note_keep_dstar": 0.8, "note_output_tokens": 300,
            "summary_fill": 0.8}


# ---------------------------------------------------------------------------
# Mock model
# ---------------------------------------------------------------------------

def _rank(case: Case) -> dict[str, int]:
    """Lower is kept first: anomalies and their context, then statements and readings, then the rest."""
    out = {}
    for e in case.events:
        if e.type in ("correction", "device", "adjustment") or (e.type == "reading" and e.subtype == "estimated"):
            r = 0
        elif e.type == "plan_notice" or (e.type == "statement" and e.fields.get("charge_ref")):
            r = 1
        elif e.type in ("statement", "reading"):
            r = 2
        else:
            r = 3
        out[e.id] = r
    return out


def _section(text: str, start: str, end: str | None) -> str:
    i = text.find(start)
    if i < 0:
        return ""
    i += len(start)
    j = text.find(end, i) if end else -1
    return text[i: j if j >= 0 else len(text)]


class MockModel:
    def __init__(self, cases: Sequence[Case], calib: dict[str, Any], noise: dict[str, float] | None = None,
                 seed: int = 0):
        self.cases = {c.case_id: c for c in cases}
        self.calib = calib
        self.noise = noise or default_noise(calib)
        self.seed = seed
        self.ranks: dict[str, dict[str, int]] = {}
        self.events: dict[str, dict] = {}
        self.counts: Counter = Counter()

    def _case(self, cid: str) -> Case:
        return self.cases[cid]

    def _rank_of(self, case: Case) -> dict[str, int]:
        if case.case_id not in self.ranks:
            self.ranks[case.case_id] = _rank(case)
            self.events[case.case_id] = {e.id: e for e in case.events}
        return self.ranks[case.case_id]

    def _keep(self, case: Case, lines: list[str], limit_chars: int, must: Sequence[str] = ()) -> list[str]:
        rank = self._rank_of(case)
        by_id = {}
        for ln in lines:
            m = _LINE_ID.match(ln)
            if m:
                by_id.setdefault(m.group(1), ln)
        order = sorted(by_id, key=lambda i: (i not in must, rank.get(i, 3), -int(i[1:])))
        kept, size = [], 0
        for i in order:
            ln = by_id[i]
            if size + len(ln) + 1 > limit_chars:
                if i in must:
                    continue
                break
            kept.append(i)
            size += len(ln) + 1
        return [by_id[i] for i in sorted(kept, key=lambda i: int(i[1:]))]

    def answer(self, cid: str, params: dict[str, Any], meta: dict[str, Any]) -> dict[str, Any]:
        kind, case = meta["kind"], self._case(meta["case_id"])
        key = meta["key"]
        user = params["messages"][0]["content"]
        chars = rq.request_chars(params)
        tin = math.ceil(chars / 2.15) + int(self.calib["input_overhead_tokens"])
        self.counts[kind] += 1
        if kind == "decide":
            self._rank_of(case)
            evs = self.events[case.case_id]
            seen = [evs[i] for i in dict.fromkeys(_LINE_ID.findall(user)) if i in evs]
            base = oracle(sorted(seen, key=lambda e: int(e.id[1:])), case.query_period)
            u = h01(self.seed, key, "noise")
            if meta["stage"] == "a_H":
                err = 1.0 - self.calib["accuracy_by_mechanism"][case.mechanism]
                alt = self.calib["wrong_answer_for"].get(base)
            else:
                err = self.noise["eps_small"] if chars < self.noise["small_chars"] else self.noise["eps_large"]
                alt = None
            action = base
            if u < err:
                others = [a for a in ACTIONS if a != base]
                action = alt or others[int(h01(self.seed, key, "alt") * len(others))]
            text = json.dumps({"action": action, "reason": "mock", "event_ids": []})
            outs = self.calib["output_tokens"]
            tout = outs[int(h01(self.seed, key, "out") * len(outs))]
        elif kind == "summary":
            budget = int(meta["budget"])
            cur = _section(user, "Current summary:\n", "\n\nNew records:\n")
            new = user.split("\n\nNew records:\n", 1)[1] if "\n\nNew records:\n" in user else ""
            lines = [x for x in (cur + "\n" + new).splitlines() if x.strip()]
            limit = int(self.noise["summary_fill"] * budget * CHARS_PER_TOKEN)
            body = "\n".join(self._keep(case, lines, limit))
            text = json.dumps({"text": body})
            tout = math.ceil(len(text) / CHARS_PER_TOKEN)
        elif kind == "note":
            recs = _section(user, "Records:\n", "\n\nRequest:")
            lines = [x for x in recs.splitlines() if x.strip()]
            must = [i for i in case.d_star if h01(self.seed, key, "keep", i) < self.noise["note_keep_dstar"]]
            kept = self._keep(case, lines, int(150 * CHARS_PER_TOKEN), must)
            omitted = 1 - len(kept) / len(lines) if lines else 0.0
            p = min(1.0, max(0.0, 0.1 + 0.6 * omitted + (h01(self.seed, key, "p") - 0.5) * 0.3))
            text = json.dumps({"note": "\n".join(kept), "p_residue": round(p, 3)})
            tout = int(self.noise["note_output_tokens"])
        else:
            raise ValueError(kind)
        return {"custom_id": cid, "result": {"type": "succeeded", "message": {
            "content": [{"type": "text", "text": text}], "stop_reason": "end_turn",
            "usage": {"input_tokens": tin, "output_tokens": tout}}}}


# ---------------------------------------------------------------------------
# Fake Batches endpoint
# ---------------------------------------------------------------------------

class FakeBatchAPI:
    """A runner that fakes the Batches API. `answer(cid, params, meta)` gives one results line.

    meta_fn() returns the engine's request meta by custom_id (state/round.json), read at results time.
    polls_before_end: GET status answers in_progress that many times per batch first.
    inject(cid, params, meta) may return a replacement results line (errors, refusals) or None.
    """

    def __init__(self, answer: Callable[[str, dict, dict], dict], meta_fn: Callable[[], dict] | None = None,
                 *, polls_before_end: int = 0, inject: Callable[[str, dict, dict], dict | None] | None = None,
                 on_create: Callable[[list[dict]], None] | None = None):
        self.answer = answer
        self.meta_fn = meta_fn
        self.polls_before_end = polls_before_end
        self.inject = inject
        self.on_create = on_create
        self.batches: dict[str, dict] = {}
        self.calls: list[tuple[str, str]] = []
        self.created = 0
        self.requests_sent = 0
        self.sent_custom_ids: list[str] = []

    def __call__(self, method: str, path: str, body: str | None = None, timeout: int = 120) -> tuple[int, str]:
        self.calls.append((method, path))
        if method == "POST" and path == "/v1/messages/count_tokens":
            p = json.loads(body or "{}")
            return 0, json.dumps({"input_tokens": math.ceil(rq.request_chars(p) / 2.15)})
        if method == "POST" and path == "/v1/messages/batches":
            reqs = json.loads(body or "{}")["requests"]
            if len(reqs) > 100_000:
                return 1, json.dumps({"error": {"type": "invalid_request_error", "message": "too many requests"}})
            if self.on_create:
                self.on_create(reqs)
            self.created += 1
            bid = f"msgbatch_fake{self.created:04d}"
            self.batches[bid] = {"requests": reqs, "polls": 0}
            self.requests_sent += len(reqs)
            self.sent_custom_ids += [r["custom_id"] for r in reqs]
            return 0, json.dumps({"id": bid, "processing_status": "in_progress", "created_at": "fake"})
        m = re.fullmatch(r"/v1/messages/batches/([A-Za-z0-9_]+)(/results)?", path)
        if method == "GET" and m and m.group(1) in self.batches:
            b = self.batches[m.group(1)]
            if not m.group(2):
                b["polls"] += 1
                ended = b["polls"] > self.polls_before_end
                n = len(b["requests"])
                return 0, json.dumps({"id": m.group(1), "processing_status": "ended" if ended else "in_progress",
                                      "request_counts": {"succeeded": n if ended else 0,
                                                         "processing": 0 if ended else n}})
            metas = self.meta_fn() if self.meta_fn else {}
            lines = []
            for r in b["requests"]:
                meta = metas.get(r["custom_id"], {})
                line = self.inject(r["custom_id"], r["params"], meta) if self.inject else None
                lines.append(line or self.answer(r["custom_id"], r["params"], meta))
            b["requests"] = [{"custom_id": r["custom_id"], "params": {}} for r in b["requests"]]
            return 0, "\n".join(json.dumps(x) for x in lines) + "\n"
        raise AssertionError(f"the fake endpoint does not serve {method} {path}")


def engine_meta_fn(run_dir: Path) -> Callable[[], dict]:
    def read() -> dict:
        p = Path(run_dir) / "state" / "round.json"
        return json.loads(p.read_text())["requests"] if p.exists() else {}
    return read



# ---------------------------------------------------------------------------
# Mock model with settings read from a finished run
# ---------------------------------------------------------------------------

_CAL_NOISE = {"eps_small": 0.0, "eps_large": 0.0, "small_chars": 25_000, "note_keep_dstar": 0.8,
              "note_output_tokens": 300, "summary_fill": 0.8}


def _pick(avoid: str, weights: dict[str, float], u: float) -> str:
    """A weighted draw among the actions other than `avoid` (uniform when the weights name none)."""
    items = [(a, w) for a, w in sorted(weights.items()) if a != avoid and w > 0]
    if not items:
        items = [(a, 1.0) for a in ACTIONS if a != avoid]
    tot = sum(w for _, w in items)
    acc = 0.0
    for a, w in items:
        acc += w / tot
        if u < acc:
            return a
    return items[-1][0]


class CalibratedMockModel(MockModel):
    """Answers decisions with the rates of a finished run (calibrate.from_run) instead of the v4 guesses.

    - a_H: right with the pilot's accuracy for the mechanism (a hash of the seed and the case).
    - a_V: equals a_H, except with the cell's pilot flip rate, then a draw from the pilot's flipped answers.
    - flip probe: with the cell's "not recoverable" share every probe fails. Otherwise a probe passes when it
      inserts every event of a needed set (D* first, of a size the pilot's searches returned), and the cell's
      noise level reverses the pass with its probability (a hash of the request key). A pass answers a_H.
      Residue probes use the rule oracle on the events shown instead.
    - verify: passes with the pilot's verify rate.
    - residue B, floor and truncation: depart from a_H with the pilot's rates (B per system).
    - summary steps and notes: the mock above. Output and input tokens come from the pilot's usage.
    """

    def __init__(self, cases: Sequence[Case], settings: dict[str, Any], seed: int = 0):
        calib = {"input_overhead_tokens": settings["input"]["kinds"]["decide"]["overhead_tokens"],
                 "output_tokens": [0], "accuracy_by_length": {"400": 1.0}}
        super().__init__(cases, calib, dict(_CAL_NOISE), seed)
        self.cal = settings
        self.input_calib = settings["input"]
        self._a_h: dict[str, str] = {}
        self._full: dict[str, str] = {}
        self._need_memo: dict[tuple[str, str], frozenset] = {}

    # ----- helpers -----

    def _full_answer(self, case: Case) -> str:
        if case.case_id not in self._full:
            self._full[case.case_id] = oracle(list(case.events), case.query_period)
        return self._full[case.case_id]

    def a_h(self, case: Case) -> str:
        if case.case_id not in self._a_h:
            base = self._full_answer(case)
            acc = self.cal["a_H"]["accuracy_by_mechanism"].get(case.mechanism)
            acc = self.cal["a_H"]["overall"] if acc is None else acc
            wrong = h01(self.seed, "aH", case.case_id) >= (1.0 if acc is None else acc)
            self._a_h[case.case_id] = (_pick(base, self.cal["a_H"]["wrong_answer_for"].get(base, {}),
                                             h01(self.seed, "aHalt", case.case_id)) if wrong else base)
        return self._a_h[case.case_id]

    def _other(self, a_h: str, weights: dict[str, float], *key: Any) -> str:
        return _pick(a_h, weights, h01(self.seed, "alt", *key))

    def _need(self, case: Case, cell: str, view_ids: set[str]) -> frozenset:
        """The events a flip probe must show to restore a_H: a set of the size a pilot search returned, D* first."""
        k = (case.case_id, cell)
        if k not in self._need_memo:
            s = self.cal["search"]
            sizes = s["found_sizes"].get(cell) or s["found_sizes_pooled"] or [1]
            size = sizes[int(h01(self.seed, "size", *k) * len(sizes))]
            cands = [e.id for e in case.events if cell.startswith("summary@") or e.id not in view_ids]
            dstar = set(case.d_star)
            ranked = sorted(cands, key=lambda i: (i not in dstar, h01(self.seed, "need", case.case_id, cell, i)))
            self._need_memo[k] = frozenset(ranked[: min(size, len(ranked))])
        return self._need_memo[k]

    def _probe(self, case: Case, params: dict[str, Any], meta: dict[str, Any], a_h: str) -> str:
        stage, cell, key = meta["stage"], meta.get("cell", ""), meta["key"]
        s = self.cal["search"] if stage == "flip" else self.cal["residue_search"]
        p_nr = (s["p_not_recoverable"].get(cell, s["pooled_not_recoverable"]) if stage == "flip"
                else s["pooled_not_recoverable"])
        eps = s["noise"].get(cell, s["pooled_noise"]) if stage == "flip" else s["noise"]
        flipped_to = self.cal["a_V"]["flipped_to"]
        if h01(self.seed, "nr", stage, case.case_id, cell) < p_nr:
            return self._other(a_h, flipped_to, key)
        user = params["messages"][0]["content"]
        if stage == "flip":
            head, _, tail = user.partition(ADDITIONAL)
            need = self._need(case, cell, set(_LINE_ID.findall(head)))
            ok = bool(need) and need <= set(_LINE_ID.findall(tail))
        else:  # residue probes: the rule oracle on the events the prompt shows
            self._rank_of(case)
            evs = self.events[case.case_id]
            seen = [evs[i] for i in dict.fromkeys(_LINE_ID.findall(user)) if i in evs]
            ok = oracle(sorted(seen, key=lambda e: int(e.id[1:])), case.query_period) == self._full_answer(case)
        if h01(self.seed, key, "probe") < eps:
            ok = not ok
        return a_h if ok else self._other(a_h, flipped_to, key)

    def _decide(self, case: Case, params: dict[str, Any], meta: dict[str, Any]) -> str:
        stage, cell, key = meta["stage"], meta.get("cell", ""), meta["key"]
        a_h = self.a_h(case)
        cal = self.cal
        if stage == "a_H":
            return a_h
        if stage == "a_V":
            rate = cal["a_V"]["flip_rate"].get(cell, cal["a_V"]["pooled_flip_rate"])
            if h01(self.seed, "aV", case.case_id, cell) < rate:
                return self._other(a_h, cal["a_V"]["flipped_to"], case.case_id, cell)
            return a_h
        if stage in ("flip", "residue_flip"):
            return self._probe(case, params, meta, a_h)
        if stage in ("flip_verify", "residue_flip_verify"):
            rate = (cal["search"]["pooled_verify_rate"] if stage == "flip_verify"
                    else cal["residue_search"]["verify_rate"])
            if h01(self.seed, key, "verify") < rate:
                return a_h
            return self._other(a_h, cal["a_V"]["flipped_to"], key)
        res = cal["residue"]
        if stage == "residue_B":
            system = cell.split("@")[0]
            rate = res["flip_rate"].get(system, res["pooled_flip_rate"])
        elif stage == "floor":
            rate = res["floor_rate"]
        elif stage == "trunc":
            rate = res["trunc_rate"]
        else:
            raise ValueError(f"unknown decide stage {stage!r}")
        if h01(self.seed, stage, case.case_id, cell) < rate:
            return self._other(a_h, res["wrong_to"].get(stage) or cal["a_V"]["flipped_to"], stage, case.case_id, cell)
        return a_h

    def _tokens(self, kind: str, stage: str, meta: dict[str, Any], chars: int, key: str) -> tuple[int, int]:
        tin = rq.estimate_input_tokens(chars, kind, self.input_calib, bound=False)
        out = self.cal["output"]["quantiles"]
        q = None
        if stage == "summary" and meta.get("budget") is not None:
            q = out.get(f"summary@{int(meta['budget'])}")
        q = q or out.get(stage) or out.get("flip") or [0]
        return tin, int(q[int(h01(self.seed, key, "out") * len(q))])

    def answer(self, cid: str, params: dict[str, Any], meta: dict[str, Any]) -> dict[str, Any]:
        kind = meta["kind"]
        chars = rq.request_chars(params)
        if kind != "decide":
            line = super().answer(cid, params, meta)
        else:
            case = self._case(meta["case_id"])
            self.counts[kind] += 1
            text = json.dumps({"action": self._decide(case, params, meta), "reason": "mock", "event_ids": []})
            line = {"custom_id": cid, "result": {"type": "succeeded", "message": {
                "content": [{"type": "text", "text": text}], "stop_reason": "end_turn", "usage": {}}}}
        tin, tout = self._tokens(kind, meta["stage"], meta, chars, meta["key"])
        line["result"]["message"]["usage"] = {"input_tokens": tin, "output_tokens": tout}
        return line

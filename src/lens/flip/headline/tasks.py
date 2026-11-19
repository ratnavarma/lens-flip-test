"""Resumable task state machines for the headline run.

Every task keeps a JSON state with an "answers" table (tag -> parsed answer) and reruns its
deterministic logic from the start each round (replay). An ask looks the tag up in the table.
When the tag has no answer yet, it builds the request body, hashes it and looks the key up in
the call cache. A cached reply is parsed and stored under the tag. A key that failed twice gives
a failure answer. Anything else is registered as pending for the next round, and the task raises
Pending. Crash or restart: reload the JSON and replay. Nothing else is needed.

Task types:

- CaseTask: a_H, the decider on the full history (shared by every cell of the case).
- SummaryChain: the rolling summary of one case at one budget, one chunk step per round.
- FlipSearch: the group-first search on round-parallel ddmin, then the v2 verification.
- ResidueBundle: note A with p_residue, the v2 floor and the truncation control in one round,
  then B on the note, then on a residue flip a search over V's units missing from N.
- CellTask: one (case, system, budget). It builds the view (local, or a SummaryChain), asks
  a_V and runs FlipSearch and ResidueBundle when they apply.
"""

from __future__ import annotations

import hashlib
from typing import Any, Callable, Iterable, Protocol

from ..decider import PARSE_ERROR
from ..flip import missing_units
from ..pddmin import Pending, group_search, to_record
from ..residue import note_view, omission_fraction, residue_candidates, tail_tokens
from ..schema import Case, Unit, View
from ..systems import RagBm25System, RagEmbedSystem, RecentSystem
from ..systems.base import count_tokens, event_view, rendered, split_sentences
from . import requests as rq

__all__ = ["Pending", "Ctx", "Asker", "run_case", "cell_name", "rebuild_view"]

EVENT_SYSTEMS = ("recent", "rag-bm25", "rag-embed")
TEXT_SYSTEMS = ("summary",)


class Ctx(Protocol):
    config: dict[str, Any]
    memo: dict[Any, Any]

    def lookup(self, key: str) -> dict[str, Any] | None: ...
    def failed(self, key: str) -> str | None: ...
    def register(self, key: str, params: dict[str, Any], meta: dict[str, Any]) -> None: ...
    def touch(self, case_id: str) -> None: ...
    def embedder(self, case_id: str) -> Callable | None: ...


Item = tuple  # (tag, kind, build, stage, extra_meta)


class Asker:
    def __init__(self, ns: dict[str, Any], ctx: Ctx, meta: dict[str, Any]):
        self.ns = ns
        self.ctx = ctx
        self.meta = meta

    def ask_many(self, items: Iterable[Item]) -> list[dict[str, Any]]:
        answers = self.ns.setdefault("answers", {})
        out: list[dict[str, Any] | None] = []
        pending = False
        for tag, kind, build, stage, extra in items:
            if tag in answers:
                out.append(answers[tag])
                continue
            params = build()
            key = rq.request_key(params)
            meta = {**self.meta, "stage": stage, "kind": kind, **(extra or {})}
            raw = self.ctx.lookup(key)
            if raw is not None:
                answers[tag] = rq.parse_answer(kind, raw, meta)
            else:
                reason = self.ctx.failed(key)
                if reason is None:
                    self.ctx.register(key, params, meta)
                    pending = True
                    out.append(None)
                    continue
                answers[tag] = rq.failure_answer(kind, reason)
            self.ctx.touch(self.meta["case_id"])
            out.append(answers[tag])
        if pending:
            raise Pending
        return out  # type: ignore[return-value]

    def ask(self, tag: str, kind: str, build: Callable[[], dict], stage: str, extra: dict | None = None) -> dict:
        return self.ask_many([(tag, kind, build, stage, extra)])[0]


def cell_name(system: str, budget: int) -> str:
    return f"{system}@{budget}"


def set_tag(prefix: str, ids: Iterable[str]) -> str:
    return prefix + hashlib.sha256("\n".join(sorted(ids)).encode("ascii")).hexdigest()[:24]


# ---------------------------------------------------------------------------
# a_H
# ---------------------------------------------------------------------------

def full_view(case: Case, ctx: Ctx) -> View:
    k = ("full", case.case_id)
    if k not in ctx.memo:
        ctx.memo[k] = event_view(case, (e.id for e in case.events), "full", None)
    return ctx.memo[k]


def run_case_task(case: Case, st: dict[str, Any], ctx: Ctx) -> str | None:
    """a_H, or None while it is pending."""
    if "a_H" in st:
        return st["a_H"]
    a = Asker(st, ctx, {"case_id": case.case_id, "cell": "full"})
    try:
        ans = a.ask("aH", "decide", lambda: rq.decide_request(full_view(case, ctx), [], case, "v1",
                                                            ctx.config["decider"]), "a_H")
    except Pending:
        return None
    st["a_H"] = ans["action"]
    ctx.touch(case.case_id)
    return st["a_H"]


# ---------------------------------------------------------------------------
# Views
# ---------------------------------------------------------------------------

def rebuild_view(case: Case, system: str, budget: int, v: dict[str, Any]) -> View:
    if "event_ids" in v:
        return event_view(case, v["event_ids"], system, budget)
    return View(system=system, budget=budget, text=v["text"], event_ids=[], units=split_sentences(v["text"]))


def summary_chain(case: Case, budget: int, ns: dict[str, Any], ctx: Ctx) -> str:
    """The final summary text. Raises Pending until every chunk step is answered."""
    cfg = ctx.config["summarizer"]
    lines = [line for _, line in rendered(case)]
    step = int(cfg["chunk_events"])
    a = Asker(ns, ctx, {"case_id": case.case_id, "cell": cell_name("summary", budget)})
    summary = ""
    for k, lo in enumerate(range(0, len(lines), step)):
        chunk = lines[lo: lo + step]
        ans = a.ask(f"sum{k:03d}", "summary", lambda s=summary, ch=chunk: rq.summary_request(s, ch, budget, cfg),
                    "summary", {"budget": budget, "step": k})
        if ans["text"] is not None:  # a refused or failed step keeps the previous summary
            summary = ans["text"]
    return summary


def cell_view(case: Case, system: str, budget: int, st: dict[str, Any], ctx: Ctx) -> View:
    mk = ("view", case.case_id, system, budget)
    if mk in ctx.memo:
        return ctx.memo[mk]
    if "view" not in st:
        if system == "summary":
            text = summary_chain(case, budget, st.setdefault("summary", {}), ctx)
            st["view"] = {"text": text}
        else:
            if system == "recent":
                sysobj: Any = RecentSystem()
            elif system == "rag-bm25":
                sysobj = RagBm25System()
            elif system == "rag-embed":
                sysobj = RagEmbedSystem(ctx.embedder(case.case_id))
            else:
                raise ValueError(f"unknown system {system!r}")
            st["view"] = {"event_ids": sysobj.view(case, budget).event_ids}
        ctx.touch(case.case_id)
    view = rebuild_view(case, system, budget, st["view"])
    ctx.memo[mk] = view
    return view


# ---------------------------------------------------------------------------
# Flip searches
# ---------------------------------------------------------------------------

def run_search(case: Case, view: View, units: list[Unit], target: str, ns: dict[str, Any], ctx: Ctx, *,
               meta: dict[str, Any], prefix: str, stage: str, system: str, budget: int) -> dict[str, Any]:
    """One FlipSearch. Stores and returns the Distinction Record dict. Raises Pending."""
    if "record" in ns:
        return ns["record"]
    cfg = ctx.config
    a = Asker(ns, ctx, meta)
    by_id = {u.id: u for u in units}

    def units_for(ids: Iterable[str]) -> list[Unit]:
        wanted = set(ids)
        return [u for u in units if u.id in wanted]

    def ask_many(sets: list[frozenset]) -> list[str]:
        items = [(set_tag(prefix, s), "decide",
                  lambda s=s: rq.decide_request(view, units_for(s), case, "v1", cfg["decider"]), stage, None)
                 for s in sets]
        return [x["action"] for x in a.ask_many(items)]

    res = group_search(units, target, ask_many, int(cfg["max_calls"]))
    verified = None
    if res["status"] == "restored":
        v = a.ask(prefix + "verify", "decide",
                  lambda: rq.decide_request(view, [by_id[i] for i in res["ids"]], case, "v2", cfg["decider"]),
                  stage + "_verify")
        verified = v["action"] == target
    rec = to_record(res, case, units, system=system, budget=budget, model=cfg["decider"]["model"],
                    verified=verified, stage="residue" if stage.startswith("residue") else "flip")
    ns["record"] = {**rec.to_dict(), "rounds": res["rounds"]}
    ctx.touch(case.case_id)
    return ns["record"]


def search_units(case: Case, view: View, ctx: Ctx) -> list[Unit]:
    mk = ("units", case.case_id, view.system, view.budget)
    if mk not in ctx.memo:
        ctx.memo[mk] = missing_units(case, view)
    return ctx.memo[mk]


# ---------------------------------------------------------------------------
# Residue bundle
# ---------------------------------------------------------------------------

def run_residue(case: Case, view: View, target: str, ns: dict[str, Any], ctx: Ctx, *, system: str,
                budget: int) -> dict[str, Any]:
    if "out" in ns and ("record" in ns or not ns["out"]["residue_flip"]):
        return ns["out"]
    cfg = ctx.config
    meta = {"case_id": case.case_id, "cell": cell_name(system, budget)}
    a = Asker(ns, ctx, meta)
    trunc_text = tail_tokens(view.text, int(cfg["trunc_tokens"]))
    note_a, floor_a, trunc_a = a.ask_many([
        ("note", "note", lambda: rq.note_request(view, case, cfg["note_writer"]), "residue_A", None),
        ("floor", "decide", lambda: rq.decide_request(view, [], case, "v2", cfg["decider"]), "floor", None),
        ("trunc", "decide", lambda: rq.decide_request(note_view(trunc_text), [], case, "v1", cfg["decider"]),
         "trunc", None),
    ])
    note = note_a["note"]
    nview = note_view(note)
    b = a.ask("B", "decide", lambda: rq.decide_request(nview, [], case, "v1", cfg["decider"]), "residue_B")
    flipped = b["action"] != target
    cands = residue_candidates(case, view, note)
    dstar = set(case.d_star)
    out = {
        "a_A": target, "a_B": b["action"], "a_floor": floor_a["action"], "a_trunc": trunc_a["action"],
        "note": note, "note_tokens": count_tokens([note]), "note_status": note_a["status"],
        "p_residue": note_a["p_residue"], "omission": omission_fraction(case, view, note),
        "view_units": len(view.units), "n_candidates": len(cands), "residue_flip": flipped,
        "candidate_ids": [u.id for u in cands] if flipped else [],
        "dstar_in_view": bool(dstar) and bool(view.event_ids) and dstar <= set(view.event_ids),
    }
    ns["out"] = out
    ctx.touch(case.case_id)
    if flipped:
        sub = ns.setdefault("search", {})
        run_search(case, nview, cands, target, sub, ctx, meta=meta, prefix="r", stage="residue_flip",
                   system=system, budget=budget)
        ns["record"] = sub["record"]
    return out


# ---------------------------------------------------------------------------
# Cells and cases
# ---------------------------------------------------------------------------

def run_cell(case: Case, system: str, budget: int, st: dict[str, Any], a_h: str | None, ctx: Ctx) -> bool:
    """Advance one cell as far as answers allow. True when it is done."""
    if st.get("done"):
        return True
    cfg = ctx.config
    try:
        view = cell_view(case, system, budget, st, ctx)
        a = Asker(st, ctx, {"case_id": case.case_id, "cell": cell_name(system, budget)})
        a_v = a.ask("aV", "decide", lambda: rq.decide_request(view, [], case, "v1", cfg["decider"]), "a_V")["action"]
    except Pending:
        return False
    if st.get("a_V") != a_v:
        st["a_V"] = a_v
        ctx.touch(case.case_id)
    if a_h is None:
        return False
    blocked = False
    if a_v != a_h and a_h != PARSE_ERROR:
        try:
            run_search(case, view, search_units(case, view, ctx), a_h, st.setdefault("flip", {}), ctx,
                       meta={"case_id": case.case_id, "cell": cell_name(system, budget)}, prefix="f",
                       stage="flip", system=system, budget=budget)
        except Pending:
            blocked = True
    if budget == int(cfg["residue_budget"]) and a_h == case.a_star and a_v == a_h:
        try:
            run_residue(case, view, a_h, st.setdefault("residue", {}), ctx, system=system, budget=budget)
        except Pending:
            blocked = True
    if not blocked:
        st["done"] = True
        ctx.touch(case.case_id)
    return not blocked


def run_case(case: Case, cs: dict[str, Any], ctx: Ctx) -> bool:
    """Advance every task of one case. True when the case is done."""
    if cs.get("done"):
        return True
    a_h = run_case_task(case, cs.setdefault("case", {}), ctx)
    cells = cs.setdefault("cells", {})
    done = a_h is not None
    for system in ctx.config["systems"]:
        for budget in ctx.config["budgets"]:
            done &= run_cell(case, system, int(budget), cells.setdefault(cell_name(system, int(budget)), {}), a_h, ctx)
    if done:
        cs["done"] = True
        ctx.touch(case.case_id)
    return done

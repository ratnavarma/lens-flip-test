"""Round engine for the headline run.

One round: advance every task, collect every pending request across tasks (deduped by key),
price the round's upper bound, refuse it when spend so far plus that bound passes the cap,
split it into batches (100,000 requests or the byte limit), submit. The next step polls; when
every batch of the round has ended it fetches the results, puts each succeeded reply into the
call cache under its key and adds actual cost from usage. Routing by custom_id is then a
replay: each task finds its answer in the cache by key.

Errored, expired or canceled requests are retried once in the next round, then marked failed.

Files in RUN/:
  config.json, cases.jsonl, protocol.json (freeze), calibration.json (dry run), spend.json
  state/cases/<case_id>.json   task state per case (JSON)
  state/round.json             the round in flight
  state/failed.json, state/attempts.json
  state/calls.sqlite           the call cache (replies by request key)
  rounds/NNN/                  round.json and the raw results of each finished round
"""

from __future__ import annotations

import fcntl
import json
import os
import random
import time
from collections import Counter
from pathlib import Path
from typing import Any, Callable

from ..cache import CallCache
from ..generator import DISTANCES, GENERATOR_VERSION, U2_PAIRS, Knobs, _quotas, default_mix, generate_case
from ..schema import MECHANISMS, Case, read_jsonl, write_jsonl
from . import requests as rq
from .metrics import POWER_RULE_V2
from .tasks import run_case

SYSTEMS = ("recent", "rag-bm25", "rag-embed", "summary")
BUDGETS = (500, 1000, 2000)
LENGTHS = (100, 400)
MAX_BATCH_REQUESTS = 100_000
MAX_BATCH_BYTES = 128 * 1024 * 1024  # API limit 256 MB; half of it, for the wrapper's stdin path
MAX_ATTEMPTS = 2

Runner = Callable[[str, str, "str | None", int], "tuple[int, str]"]


class Refused(Exception):
    pass


def _dump(obj: Any) -> str:
    return rq.canonical(obj)


def _write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def _read_json(path: Path, default: Any) -> Any:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else default


# ---------------------------------------------------------------------------
# Sample and init
# ---------------------------------------------------------------------------

def sample_mix(mechanisms: tuple[str, ...] | None = None) -> dict[str, float]:
    """The default power mix, or (for a subset run) the chosen mechanisms alone."""
    mix = default_mix(0.5)
    if not mechanisms:
        return mix
    bad = [m for m in mechanisms if m not in MECHANISMS]
    if bad or len(set(mechanisms)) != len(mechanisms):
        raise ValueError(f"--mechanisms needs distinct names from {', '.join(MECHANISMS)}; got {', '.join(mechanisms)}")
    return {m: mix[m] for m in MECHANISMS if m in mechanisms}


def u2_pair_for(i: int) -> str:
    """Protocol v3: the pair of the i-th U2 case of a run. Exactly half and half over any even count, and
    crossed with the length, which alternates with i % 2."""
    return tuple(U2_PAIRS)[(i // 2 + i % 2) % 2]


def make_headline_sample(n: int, seed: int, distractors: int = 2, mechanisms: tuple[str, ...] | None = None,
                         prefix: str | None = None, generator_version: int = GENERATOR_VERSION) -> list[Case]:
    """n cases in the generator's default power mix (rare share 0.5). Within each mechanism the
    lengths alternate 100 and 400. The distance knob is drawn per case from the seeded stream.

    `mechanisms` limits the run to those mechanisms (their default shares, renormalised). `prefix` replaces
    the "h<seed>" in the case ids. `generator_version` 2 regenerates the headline-300 sample; version 3
    differs from it in U2 cases only, and plants their pairs half and half by position."""
    quotas = _quotas(n, sample_mix(mechanisms))
    rng = random.Random(f"headline|{seed}")
    plan = [(m, i) for m in MECHANISMS for i in range(quotas[m])]
    rng.shuffle(plan)
    cases = []
    for idx, (m, i) in enumerate(plan):
        knobs = Knobs(length=LENGTHS[i % 2], distance=rng.choice(DISTANCES), distractors=distractors)
        pair = u2_pair_for(i) if m == "U2" and generator_version >= 3 else None
        cases.append(generate_case(rng.randrange(2**31), m, knobs, case_id=f"{prefix or f'h{seed}'}-{idx:04d}",
                                   generator_version=generator_version, u2_pair=pair))
    return cases


def default_config(n: int, seed: int, mechanisms: tuple[str, ...] | None = None, prefix: str | None = None,
                   generator_version: int = GENERATOR_VERSION) -> dict[str, Any]:
    from .. import batch

    arm = dict(batch.ARM_CONFIG[rq.DECIDER["arm"]])
    arm.pop("assumed_out", None)
    return {
        "version": "headline-v2",
        "n": n, "seed": seed,
        "sample": {"mix": sample_mix(mechanisms), "lengths": list(LENGTHS), "distances": list(DISTANCES),
                   "distractors": 2, "case_ids": f"{prefix or f'h{seed}'}-NNNN", "generator_version": generator_version,
                   **({"mechanisms": list(mechanisms)} if mechanisms else {})},
        "systems": list(SYSTEMS), "budgets": list(BUDGETS), "residue_budget": 1000,
        "max_calls": 48, "strategy": "group", "trunc_tokens": 150,
        "prompt_versions": {"decide": "v1", "verify": "v2", "floor": "v2"},
        "decider": {**rq.DECIDER, **arm},
        "summarizer": dict(rq.SUMMARIZER),
        "note_writer": dict(rq.NOTE_WRITER),
        "embedder": {"kind": "ollama", "model": "nomic-embed-text", "cache": "results/cache/calls.sqlite"},
        "cache": "state/calls.sqlite",
        "batch": {"max_requests": MAX_BATCH_REQUESTS, "max_bytes": MAX_BATCH_BYTES},
        "rates": dict(rq.RATES),
        "bootstrap": {"reps": 1000, "seed": seed}, "h2b_draws": 1000,
        "power_min_flips": 20,  # protocol v1 only; v2 reads power_rule
        "power_rule": dict(POWER_RULE_V2),
    }


def init_run(out: str | Path, n: int, seed: int, mechanisms: tuple[str, ...] | None = None,
             prefix: str | None = None, generator_version: int = GENERATOR_VERSION) -> dict[str, Any]:
    d = Path(out)
    if (d / "config.json").exists() or (d / "state").exists():
        raise Refused(f"{d} already holds a run; choose a new --out")
    if mechanisms and not prefix:
        raise Refused("a subset run needs --prefix, so its case ids cannot collide with a full run's")
    try:
        cases = make_headline_sample(n, seed, mechanisms=mechanisms, prefix=prefix,
                                     generator_version=generator_version)
    except ValueError as exc:
        raise Refused(str(exc)) from exc
    d.mkdir(parents=True, exist_ok=True)
    write_jsonl(d / "cases.jsonl", cases)
    cfg = default_config(n, seed, mechanisms, prefix, generator_version)
    _write_json(d / "config.json", cfg)
    (d / "state" / "cases").mkdir(parents=True, exist_ok=True)
    return {"cases": len(cases), "mechanisms": dict(Counter(c.mechanism for c in cases)),
            "lengths": dict(Counter(c.knobs["length"] for c in cases)),
            "distances": dict(Counter(c.knobs["distance"] for c in cases))}


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------

class Engine:
    def __init__(self, run_dir: str | Path, runner: Runner | None = None, *, embedder: Callable | None = None,
                 log: Callable[[str], None] | None = None, now: Callable[[], float] = time.time,
                 check_protocol: bool = True):
        self.dir = Path(run_dir)
        self.config: dict[str, Any] = _read_json(self.dir / "config.json", None)
        if self.config is None:
            raise Refused(f"no config.json in {self.dir}; run init first")
        self.cases = list(read_jsonl(self.dir / "cases.jsonl"))
        self.by_id = {c.case_id: c for c in self.cases}
        self.runner = runner or rq.wrapper_runner
        self.log = log or (lambda m: None)
        self.now = now
        self.check_protocol = check_protocol
        self.sdir = self.dir / "state"
        (self.sdir / "cases").mkdir(parents=True, exist_ok=True)
        cache_path = Path(self.config.get("cache") or "state/calls.sqlite")
        self.cache = CallCache(cache_path if cache_path.is_absolute() else self.dir / cache_path)
        self.failed_map: dict[str, str] = _read_json(self.sdir / "failed.json", {})
        self.attempts: dict[str, list[str]] = _read_json(self.sdir / "attempts.json", {})
        self.states: dict[str, dict] = {c.case_id: _read_json(self.sdir / "cases" / f"{c.case_id}.json", {})
                                        for c in self.cases}
        self.memo: dict[Any, Any] = {}
        self.pending: dict[str, tuple[dict, dict]] = {}
        self.waiters: Counter = Counter()
        self._touched: set[str] = set()
        self._embedder_override = embedder
        self._embed_case: tuple[str, Any] | None = None
        self._lock_fh = None

    # ----- Ctx protocol (used by tasks) -----

    def lookup(self, key: str) -> dict[str, Any] | None:
        hit = self.cache.get(key)
        return json.loads(hit) if hit is not None else None

    def failed(self, key: str) -> str | None:
        return self.failed_map.get(key)

    def register(self, key: str, params: dict, meta: dict) -> None:
        self.waiters[key] += 1
        if key not in self.pending:
            self.pending[key] = (params, meta)

    def touch(self, case_id: str) -> None:
        self._touched.add(case_id)

    def embedder(self, case_id: str) -> Callable:
        """A fresh embedder per case (its in-memory memo stays small); vectors stay in the sqlite cache."""
        if self._embed_case and self._embed_case[0] == case_id:
            return self._embed_case[1]
        if self._embedder_override is not None:
            emb = self._embedder_override
        else:
            ec = self.config["embedder"]
            if ec["kind"] == "hash":
                from .mock import HashEmbedder
                emb = HashEmbedder()
            else:
                from ..llm import CachedEmbedder, OllamaEmbedTransport
                cpath = Path(ec.get("cache") or "results/cache/calls.sqlite")
                if not hasattr(self, "_embed_cache"):
                    self._embed_cache = CallCache(cpath)
                emb = CachedEmbedder(ec["model"], transport=OllamaEmbedTransport(ec.get("url", "http://localhost:11434")),
                                     cache=self._embed_cache)
        self._embed_case = (case_id, emb)
        return emb

    # ----- persistence -----

    def _lock(self) -> None:
        if self._lock_fh is None:
            self._lock_fh = open(self.sdir / "lock", "a")
            try:
                fcntl.flock(self._lock_fh, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                self._lock_fh.close()
                self._lock_fh = None
                raise Refused("another step holds the lock on this run")

    def save(self) -> None:
        for cid in sorted(self._touched):
            _write_json(self.sdir / "cases" / f"{cid}.json", self.states[cid])
        self._touched.clear()
        _write_json(self.sdir / "failed.json", self.failed_map)
        _write_json(self.sdir / "attempts.json", self.attempts)

    def spend(self) -> dict[str, Any]:
        return _read_json(self.dir / "spend.json", {"actual_usd": 0.0, "input_tokens": 0, "output_tokens": 0,
                                                    "by_stage": {}, "batches": {}, "rounds": []})

    def _write_spend(self, sp: dict[str, Any]) -> None:
        tot_in = sum(b["input_tokens"] for b in sp["batches"].values())
        tot_out = sum(b["output_tokens"] for b in sp["batches"].values())
        sp["input_tokens"], sp["output_tokens"] = tot_in, tot_out
        sp["actual_usd"] = round(sum(b["usd"] for b in sp["batches"].values()), 6)
        by: dict[str, dict[str, float]] = {}
        for b in sp["batches"].values():
            for st, v in b["by_stage"].items():
                acc = by.setdefault(st, {"requests": 0, "input_tokens": 0, "output_tokens": 0, "usd": 0.0})
                for k in acc:
                    acc[k] += v[k]
        sp["by_stage"] = {k: {**v, "usd": round(v["usd"], 6)} for k, v in sorted(by.items())}
        _write_json(self.dir / "spend.json", sp)

    def calibration(self) -> dict[str, Any] | None:
        return _read_json(self.dir / "calibration.json", None)

    # ----- advance -----

    def advance(self) -> dict[str, tuple[dict, dict]]:
        """Replay every task. Returns the pending requests (key -> (body, meta)), deduped."""
        self.pending, self.waiters = {}, Counter()
        for case in self.cases:
            run_case(case, self.states[case.case_id], self)
        self.save()
        return self.pending

    def done(self) -> bool:
        return all(self.states[c.case_id].get("done") for c in self.cases)

    # ----- protocol -----

    def _check_protocol(self) -> None:
        if not self.check_protocol:
            return
        from .protocol import current_hashes, run_file_hashes

        frozen = _read_json(self.dir / "protocol.json", None)
        if frozen is None:
            raise Refused("no protocol.json; run `lens flip headline freeze --dir RUN` first")
        now = {**current_hashes(), **{f"run:{k}": v for k, v in run_file_hashes(self.dir).items()}}
        was = {**frozen["hashes"], **{f"run:{k}": v for k, v in frozen["run_files"].items()}}
        changed = sorted(k for k in set(now) | set(was) if now.get(k) != was.get(k))
        if changed:
            raise Refused("prompts or generator changed since the freeze: " + ", ".join(changed))

    # ----- rounds -----

    def round_state(self) -> dict[str, Any] | None:
        return _read_json(self.sdir / "round.json", None)

    def plan_round(self, pending: dict[str, tuple[dict, dict]]) -> dict[str, Any]:
        """Upper bound and batch split for the pending requests."""
        calib = self.calibration()
        rates = self.config.get("rates", rq.RATES)
        bcfg = self.config.get("batch", {})
        max_n = int(bcfg.get("max_requests", MAX_BATCH_REQUESTS))
        max_b = int(bcfg.get("max_bytes", MAX_BATCH_BYTES))
        reqs, metas, bound = [], {}, 0.0
        for key, (params, meta) in sorted(pending.items()):
            cid = rq.custom_id_for(key)
            chars = rq.request_chars(params)
            est = rq.estimate_input_tokens(chars, meta["kind"], calib, bound=True)
            ub = rq.request_price(est, params["max_tokens"], rates)
            bound += ub
            line = _dump({"custom_id": cid, "params": params})
            reqs.append((cid, line))
            metas[cid] = {**meta, "key": key, "chars": chars, "est_input_tokens_bound": est,
                          "max_tokens": params["max_tokens"], "upper_usd": round(ub, 8)}
        batches: list[list[tuple[str, str]]] = [[]]
        size = 0
        for cid, line in reqs:
            if batches[-1] and (len(batches[-1]) >= max_n or size + len(line) + 1 > max_b - 64):
                batches.append([])
                size = 0
            batches[-1].append((cid, line))
            size += len(line) + 1
        return {"requests": metas, "batches": [b for b in batches if b], "upper_bound_usd": bound}

    def submit_round(self, cap_usd: float) -> str:
        pending = self.advance()
        if not pending:
            if self.done():
                return "complete"
            raise RuntimeError("no pending requests but the run is not done")
        plan = self.plan_round(pending)
        sp = self.spend()
        spent = float(sp["actual_usd"])
        bound = plan["upper_bound_usd"]
        if spent + bound > cap_usd + 1e-12:
            raise Refused(f"round of {len(plan['requests'])} requests has an upper bound of ${bound:.4f}; "
                          f"spent ${spent:.4f} + bound passes the cap ${cap_usd:.4f}")
        rnd = len(sp["rounds"]) + 1
        rs = {"round": rnd, "submitted_at": self.now(), "upper_bound_usd": round(bound, 6), "cap_usd": cap_usd,
              "spent_before_usd": spent, "n_requests": len(plan["requests"]), "requests": plan["requests"],
              "batches": [], "by_stage": dict(Counter(m["stage"] for m in plan["requests"].values()))}
        for i, chunk in enumerate(plan["batches"]):
            for cid, _ in chunk:
                rs["requests"][cid]["batch"] = i
        _write_json(self.sdir / "round.json", rs)
        sp["rounds"].append({k: rs[k] for k in ("round", "submitted_at", "upper_bound_usd", "cap_usd",
                                                "n_requests", "by_stage")})
        self._write_spend(sp)
        for i, chunk in enumerate(plan["batches"]):
            body = '{"requests":[' + ",".join(line for _, line in chunk) + "]}"
            rc, out = self.runner("POST", "/v1/messages/batches", body, 600)
            if rc != 0:
                _write_json(self.sdir / "round.json", rs)
                raise RuntimeError(f"batch create failed on batch {i} of round {rnd}: {out[:300]}")
            resp = json.loads(out)
            rs["batches"].append({"index": i, "id": resp["id"], "n": len(chunk), "ended": False})
            _write_json(self.sdir / "round.json", rs)
        self.log(f"round {rnd}: submitted {len(plan['requests'])} requests in {len(plan['batches'])} batch(es); "
                 f"upper bound ${bound:.4f}, spent ${spent:.4f}, cap ${cap_usd:.2f}; "
                 + ", ".join(f"{k}={v}" for k, v in sorted(rs["by_stage"].items())))
        return "submitted"

    def fetch_round(self) -> str:
        rs = self.round_state()
        assert rs is not None
        rdir = self.dir / "rounds" / f"{rs['round']:03d}"
        rdir.mkdir(parents=True, exist_ok=True)
        waiting = []
        sp = self.spend()
        for b in rs["batches"]:
            if b["ended"]:
                continue
            rc, out = self.runner("GET", f"/v1/messages/batches/{b['id']}", None, 120)
            if rc != 0:
                raise RuntimeError(f"batch status failed for {b['id']}: {out[:300]}")
            status = json.loads(out)
            if status.get("processing_status") != "ended":
                waiting.append(f"{b['id']} {status.get('processing_status')} "
                               f"{json.dumps(status.get('request_counts'), sort_keys=True)}")
                continue
            rc, out = self.runner("GET", f"/v1/messages/batches/{b['id']}/results", None, 600)
            if rc != 0:
                raise RuntimeError(f"batch results failed for {b['id']}: {out[:300]}")
            (rdir / f"results-{b['index']:03d}.jsonl").write_text(out if out.endswith("\n") else out + "\n",
                                                                  encoding="utf-8")
            lines = [json.loads(x) for x in out.splitlines() if x.strip()]
            sp["batches"][b["id"]] = self._route(rs, b, lines)
            b["ended"] = True
            b["request_counts"] = status.get("request_counts")
            self.save()
            self._write_spend(sp)
            _write_json(self.sdir / "round.json", rs)
        if waiting:
            self.log(f"round {rs['round']}: waiting on " + "; ".join(waiting))
            return "waiting"
        rs["ended_at"] = self.now()
        for r in sp["rounds"]:
            if r["round"] == rs["round"]:
                r["ended_at"] = rs["ended_at"]
                r["actual_usd"] = round(sum(sp["batches"][b["id"]]["usd"] for b in rs["batches"]), 6)
                r["failed_or_retried"] = sum(sp["batches"][b["id"]]["not_succeeded"] for b in rs["batches"])
        self._write_spend(sp)
        _write_json(self.sdir / "round.json", rs)
        os.replace(self.sdir / "round.json", rdir / "round.json")
        self.log(f"round {rs['round']}: fetched; actual spend now ${sp['actual_usd']:.4f}")
        return "fetched"

    def _route(self, rs: dict[str, Any], b: dict[str, Any], lines: list[dict]) -> dict[str, Any]:
        """Cache succeeded replies by key, count failures, return the batch's cost from usage."""
        mine = {cid for cid, m in rs["requests"].items() if m.get("batch") == b["index"]}
        rates = self.config.get("rates", rq.RATES)
        seen: set[str] = set()
        cost = {"usd": 0.0, "input_tokens": 0, "output_tokens": 0, "requests": 0, "not_succeeded": 0,
                "by_stage": {}}
        for line in lines:
            cid = line.get("custom_id")
            if cid not in mine:
                continue
            seen.add(cid)
            meta = rs["requests"][cid]
            raw = rq.raw_outcome(line)
            tin, tout = rq.usage_tokens(raw["usage"])
            usd = rq.request_price(tin, tout, rates)
            st = cost["by_stage"].setdefault(meta["stage"], {"requests": 0, "input_tokens": 0, "output_tokens": 0,
                                                             "usd": 0.0})
            for acc in (cost, st):
                acc["requests"] += 1
                acc["input_tokens"] += tin
                acc["output_tokens"] += tout
                acc["usd"] += usd
            if raw["result_type"] == "succeeded":
                self.cache.put(meta["key"], self.config["decider"]["model"], meta["kind"], _dump(raw))
            else:
                cost["not_succeeded"] += 1
                self._fail(meta["key"], b["id"], raw.get("error") or raw["result_type"])
        for cid in sorted(mine - seen):  # no line came back for it
            cost["not_succeeded"] += 1
            self._fail(rs["requests"][cid]["key"], b["id"], "missing")
        cost["usd"] = round(cost["usd"], 8)
        return cost

    def _fail(self, key: str, batch_id: str, reason: str) -> None:
        seen = self.attempts.setdefault(key, [])
        if batch_id not in seen:
            seen.append(batch_id)
        if len(seen) >= MAX_ATTEMPTS:
            self.failed_map[key] = reason

    # ----- step and run -----

    def step(self, cap_usd: float) -> str:
        """Fetch and route the round in flight, or submit the next round. Never sleeps."""
        self._lock()
        self._check_protocol()
        rs = self.round_state()
        if rs is not None:
            if not rs["batches"]:  # nothing was created; the requests are still pending
                os.replace(self.sdir / "round.json", self.sdir / f"round.empty-{rs['round']:03d}.json")
                sp = self.spend()
                sp["rounds"] = [r for r in sp["rounds"] if r["round"] != rs["round"]]
                self._write_spend(sp)
            else:
                return self.fetch_round()
        return self.submit_round(cap_usd)

    def run(self, cap_usd: float, poll_seconds: float = 120, *, sleep: Callable[[float], None] = time.sleep,
            max_steps: int | None = None) -> str:
        steps = 0
        while True:
            r = self.step(cap_usd)
            steps += 1
            if r == "complete":
                return r
            if max_steps is not None and steps >= max_steps:
                return r
            if r in ("waiting", "submitted"):
                sleep(poll_seconds)

    def close(self) -> None:
        self.cache.close()
        if self._lock_fh is not None:
            self._lock_fh.close()
            self._lock_fh = None


def rounds_used(run_dir: str | Path) -> int:
    return len(_read_json(Path(run_dir) / "spend.json", {"rounds": []})["rounds"])


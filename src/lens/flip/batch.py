"""Batch API validity check for the Decision Flip Test (design section 5).

Question: does Claude Haiku pick a* from the FULL history (a_H vs a_star)? Arms are chosen with
`build --arms`. The default is adaptive-medium and adaptive-high (adaptive thinking is the default
on Haiku 5.5, so those arms send no `thinking` field and differ in output_config.effort). The old
`off` and `adaptive` arms still build and score, so v1 and v2 result dirs keep working. The arms
of a dir are read from its requests.jsonl. All HTTP goes through the anthropic-api wrapper, which holds the key. This
module never sees a key. Commands: build, dryrun, submit, fetch.

A runner is a callable (method, path, body_text_or_None, timeout) -> (exit_code, stdout). Tests pass
a fake. The real one shells out to the wrapper with an argument list.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import random
import re
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any, Callable

from .decider import PARSE_ERROR, REPLY_FORMAT, build_messages, parse_reply
from .generator import Knobs, generate_case
from .metrics import GATE_OVERALL, GATE_PER_MECHANISM, validity_gate
from .schema import ACTIONS, MECHANISMS, Case, read_jsonl, write_jsonl
from .systems.base import event_view

MODEL = "claude-haiku-5-5"
PROMPT_VERSION = "v1"
WRAPPER = str(Path.home() / ".local" / "bin" / "anthropic-api")

# Per-arm request settings. `thinking` None means the field is omitted (adaptive on Haiku 5.5).
ARM_CONFIG: dict[str, dict[str, Any]] = {
    "off": {"max_tokens": 1024, "effort": "medium", "thinking": {"type": "disabled"}, "assumed_out": 150},
    "adaptive": {"max_tokens": 8000, "effort": "medium", "thinking": None, "assumed_out": 1500},
    "adaptive-medium": {"max_tokens": 6000, "effort": "medium", "thinking": None, "assumed_out": 1500},
    "adaptive-high": {"max_tokens": 6000, "effort": "high", "thinking": None, "assumed_out": 3000},
}
ARMS = tuple(ARM_CONFIG)  # every known arm, in display order
LEGACY_ARMS = ("off", "adaptive")
DEFAULT_ARMS = ("adaptive-medium", "adaptive-high")
MAX_TOKENS = {a: c["max_tokens"] for a, c in ARM_CONFIG.items()}
LENGTHS = (100, 400)

# Batch prices, USD per million tokens.
PRICE_IN = 0.05
PRICE_OUT = 0.25
# Assumed output tokens per request for the likely cost line.
ASSUMED_OUT = {a: c["assumed_out"] for a, c in ARM_CONFIG.items()}

CUSTOM_ID_RE = re.compile(r"^[a-zA-Z0-9_-]{1,64}$")

Runner = Callable[[str, str, "str | None", int], "tuple[int, str]"]

ACTION_SCHEMA: dict[str, Any] = copy.deepcopy(REPLY_FORMAT)
ACTION_SCHEMA["additionalProperties"] = False
ACTION_SCHEMA["properties"]["action"]["enum"] = list(ACTIONS)


# ---------------------------------------------------------------------------
# Wrapper runner
# ---------------------------------------------------------------------------

def wrapper_runner(method: str, path: str, body: str | None = None, timeout: int = 120) -> tuple[int, str]:
    """Call ~/.local/bin/anthropic-api with an argument list. Body goes in on stdin."""
    cmd = [WRAPPER, method, path, "--timeout", str(int(timeout))]
    if body is not None:
        cmd += ["--body", "-"]
    proc = subprocess.run(cmd, input=body, capture_output=True, text=True, check=False)
    if proc.returncode == 2:
        raise RuntimeError(f"anthropic-api usage or environment error: {proc.stderr.strip()[:300]}")
    return proc.returncode, proc.stdout


def _dump(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def _write_lines(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as fh:
        for r in rows:
            fh.write(_dump(r) + "\n")


def _read_lines(path: Path) -> list[dict]:
    return [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x.strip()]


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


# ---------------------------------------------------------------------------
# Build
# ---------------------------------------------------------------------------

def make_sample(n: int, seed: int) -> list[Case]:
    """n cases, n/6 per mechanism, half at length 100 and half at 400, in shuffled order."""
    k = len(MECHANISMS)
    if n % k:
        raise ValueError(f"n must be a multiple of {k} for a balanced sample")
    per = n // k
    if per % 2:
        raise ValueError("cases per mechanism must be even so lengths split evenly")
    rng = random.Random(f"validity|{seed}")
    plan = [(m, i) for m in MECHANISMS for i in range(per)]
    seeds = rng.sample(range(2**31), len(plan))
    cases = []
    for (m, i), s in zip(plan, seeds):
        knobs = Knobs(length=LENGTHS[i % 2])
        cases.append(generate_case(s, m, knobs, case_id=f"v{seed}-{m}-{i:02d}"))
    rng.shuffle(cases)
    return cases


def custom_id(case_id: str, arm: str) -> str:
    return f"{case_id}__{arm}"


def parse_arms(text: str) -> tuple[str, ...]:
    """Split a comma list of arm names. Rejects unknown or repeated names."""
    arms = tuple(a.strip() for a in text.split(",") if a.strip())
    if not arms:
        raise ValueError("--arms needs at least one arm")
    bad = [a for a in arms if a not in ARM_CONFIG]
    if bad:
        raise ValueError(f"unknown arm {bad[0]!r}; known arms: {', '.join(ARMS)}")
    if len(set(arms)) != len(arms):
        raise ValueError("--arms lists an arm twice")
    return arms


def dir_arms(reqs: list[dict]) -> tuple[str, ...]:
    """The arms present in a requests list, in the order of ARMS."""
    seen = {arm_of(r["custom_id"]) for r in reqs}
    unknown = seen - set(ARM_CONFIG)
    if unknown:
        raise ValueError(f"unknown arm in requests: {sorted(unknown)[0]!r}")
    return tuple(a for a in ARMS if a in seen)


def request_params(case: Case, arm: str) -> dict[str, Any]:
    """The Messages API body for one case and arm: full history, v1 prompt."""
    return decide_params(event_view(case, (e.id for e in case.events), "full"), [], case, PROMPT_VERSION, arm)


def decide_params(view: Any, inserted: list, case: Case, version: str = PROMPT_VERSION,
                  arm: str = "adaptive-high") -> dict[str, Any]:
    """The Messages API body for one decision on any view, with inserted units, in prompt version v1 or v2."""
    cfg = ARM_CONFIG[arm]
    msgs = build_messages(view, inserted, case, version)
    system = next(m["content"] for m in msgs if m["role"] == "system")
    user = next(m["content"] for m in msgs if m["role"] == "user")
    params: dict[str, Any] = {
        "model": MODEL,
        "max_tokens": cfg["max_tokens"],
        "system": system,
        "messages": [{"role": "user", "content": user}],
        "output_config": {
            "effort": cfg["effort"],
            "format": {"type": "json_schema", "schema": copy.deepcopy(ACTION_SCHEMA)},
        },
    }
    if cfg["thinking"] is not None:
        params["thinking"] = copy.deepcopy(cfg["thinking"])
    return params


def build_requests(cases: list[Case], arms: tuple[str, ...] = DEFAULT_ARMS) -> list[dict[str, Any]]:
    out = []
    for case in cases:
        for arm in arms:
            cid = custom_id(case.case_id, arm)
            if not CUSTOM_ID_RE.match(cid):
                raise ValueError(f"bad custom_id {cid!r}")
            out.append({"custom_id": cid, "params": request_params(case, arm)})
    ids = [r["custom_id"] for r in out]
    if len(set(ids)) != len(ids):
        raise ValueError("duplicate custom_id")
    return out


def make_subset_sample(n: int, seed: int, mechanisms: tuple[str, ...], prefix: str) -> list[Case]:
    """The cases of `lens flip headline init --n N --seed S --mechanisms ... --prefix P`, built by the same code,
    so a pre-check and its headline run share every case and every id."""
    from .headline.engine import make_headline_sample  # headline imports this module, so import late

    return make_headline_sample(n, seed, mechanisms=mechanisms, prefix=prefix)


def run_build(args: argparse.Namespace) -> int:
    try:
        arms = parse_arms(args.arms)
    except ValueError as exc:
        print(f"refused: {exc}", file=sys.stderr)
        return 1
    mechs = tuple(m.strip() for m in args.mechanisms.split(",") if m.strip()) if args.mechanisms else None
    if mechs:
        if not args.prefix:
            print("refused: --mechanisms needs --prefix", file=sys.stderr)
            return 1
        try:
            cases = make_subset_sample(args.n, args.seed, mechs, args.prefix)
        except ValueError as exc:
            print(f"refused: {exc}", file=sys.stderr)
            return 1
    else:
        cases = make_sample(args.n, args.seed)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    write_jsonl(out / "cases.jsonl", cases)
    reqs = build_requests(cases, arms)
    _write_lines(out / "requests.jsonl", reqs)
    by = Counter(c.mechanism for c in cases)
    lens = Counter(c.knobs["length"] for c in cases)
    print(f"wrote {len(cases)} cases and {len(reqs)} requests to {out} (arms: {', '.join(arms)})")
    print("mechanisms: " + ", ".join(f"{m}={by[m]}" for m in MECHANISMS if by[m] or not mechs))
    print("lengths: " + ", ".join(f"{k}={lens[k]}" for k in sorted(lens)))
    return 0


# ---------------------------------------------------------------------------
# Pricing
# ---------------------------------------------------------------------------

def price(input_tokens: float, output_tokens: float) -> float:
    return input_tokens * PRICE_IN / 1e6 + output_tokens * PRICE_OUT / 1e6


def price_dryrun(per_arm: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """per_arm[arm] has n_requests and input_tokens (total, extrapolated if sampled)."""
    arms: dict[str, Any] = {}
    for arm in per_arm:
        n = per_arm[arm]["n_requests"]
        tin = per_arm[arm]["input_tokens"]
        likely = price(tin, ASSUMED_OUT[arm] * n)
        upper = price(tin, MAX_TOKENS[arm] * n)
        arms[arm] = {
            **per_arm[arm],
            "assumed_output_tokens": ASSUMED_OUT[arm] * n,
            "max_output_tokens": MAX_TOKENS[arm] * n,
            "likely_usd": round(likely, 6),
            "upper_bound_usd": round(upper, 6),
        }
    return {
        "arms": arms,
        "likely_usd": round(sum(a["likely_usd"] for a in arms.values()), 6),
        "upper_bound_usd": round(sum(a["upper_bound_usd"] for a in arms.values()), 6),
    }


def _error_text(out: str) -> str:
    try:
        err = json.loads(out).get("error", {})
        return f"{err.get('type', '?')}: {err.get('message', '')}"[:300]
    except Exception:
        return out[:300]


def _is_rate_limit(out: str) -> bool:
    try:
        return json.loads(out).get("error", {}).get("type") == "rate_limit_error"
    except Exception:
        return False


def count_body(params: dict[str, Any], drop: set[str]) -> dict[str, Any]:
    body = {k: params[k] for k in ("model", "system", "messages", "thinking") if k in params and k not in drop}
    return body


def count_one(runner: Runner, params: dict[str, Any], drop: set[str], notes: list[str],
              sleep: Callable[[float], None] = time.sleep, max_retries: int = 6) -> int:
    """Input tokens for one request. Backs off on 429. Drops `thinking` if the API rejects it."""
    delay = 2.0
    tries = 0
    while True:
        rc, out = runner("POST", "/v1/messages/count_tokens", _dump(count_body(params, drop)), 120)
        if rc == 0:
            return int(json.loads(out)["input_tokens"])
        if _is_rate_limit(out) and tries < max_retries:
            tries += 1
            sleep(delay)
            delay = min(delay * 2, 60.0)
            continue
        if "thinking" in params and "thinking" not in drop and "thinking" in out.lower():
            drop.add("thinking")
            notes.append(f"count_tokens rejected the thinking field: {_error_text(out)}")
            continue
        raise RuntimeError(f"count_tokens failed: {_error_text(out)}")


def pick_sample(reqs: list[dict], k: int | None) -> list[dict]:
    """K requests, split evenly across arms and evenly spaced within each arm."""
    if k is None or k >= len(reqs):
        return list(reqs)
    if k <= 0:
        raise ValueError("--sample must be positive")
    chosen = []
    arms = dir_arms(reqs)
    for j, arm in enumerate(arms):
        arm_reqs = [r for r in reqs if arm_of(r["custom_id"]) == arm]
        kk = k // len(arms) + (1 if j < k % len(arms) else 0)
        kk = min(kk, len(arm_reqs))
        step = len(arm_reqs) / kk if kk else 0
        chosen += [arm_reqs[int(i * step)] for i in range(kk)]
    return chosen


def run_dryrun_impl(dir_: Path, sample: int | None, runner: Runner,
                    log: Callable[[str], None] | None = None,
                    sleep: Callable[[float], None] = time.sleep) -> dict[str, Any]:
    reqs = _read_lines(dir_ / "requests.jsonl")
    arms = dir_arms(reqs)
    chosen = pick_sample(reqs, sample)
    drop: set[str] = set()
    notes: list[str] = []
    counts: dict[str, int] = {}
    for i, r in enumerate(chosen):
        counts[r["custom_id"]] = count_one(runner, r["params"], drop, notes, sleep)
        if log and ((i + 1) % 10 == 0 or i + 1 == len(chosen)):
            log(f"counted {i + 1}/{len(chosen)}")
    per_arm: dict[str, dict[str, Any]] = {}
    for arm in arms:
        arm_reqs = [r for r in reqs if arm_of(r["custom_id"]) == arm]
        arm_counted = [counts[r["custom_id"]] for r in arm_reqs if r["custom_id"] in counts]
        n = len(arm_reqs)
        if not arm_counted:
            raise ValueError(f"no counted requests for arm {arm}; raise --sample")
        total = sum(arm_counted) if len(arm_counted) == n else sum(arm_counted) / len(arm_counted) * n
        per_arm[arm] = {
            "n_requests": n, "n_counted": len(arm_counted), "input_tokens": round(total),
            "mean_input_tokens": round(sum(arm_counted) / len(arm_counted), 1),
            "max_input_tokens": max(arm_counted),
        }
    result = price_dryrun(per_arm)
    result.update({
        "model": MODEL,
        "arm_names": list(arms),
        "requests_sha256": _sha(dir_ / "requests.jsonl"),
        "complete": len(chosen) == len(reqs),
        "sample": None if len(chosen) == len(reqs) else len(chosen),
        "rates_usd_per_million": {"input": PRICE_IN, "output": PRICE_OUT},
        "count_tokens_fields_dropped": sorted(drop),
        "notes": notes,
        "per_request": counts,
    })
    (dir_ / "dryrun.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return result


def format_table(res: dict[str, Any]) -> str:
    rows = [f"{'arm':<17}{'requests':>9}{'input tok':>12}{'likely $':>11}{'upper $':>11}"]
    arms = list(res["arms"])
    for arm in arms:
        a = res["arms"][arm]
        rows.append(f"{arm:<17}{a['n_requests']:>9}{a['input_tokens']:>12}{a['likely_usd']:>11.4f}"
                    f"{a['upper_bound_usd']:>11.4f}")
    tin = sum(res["arms"][a]["input_tokens"] for a in arms)
    rows.append(f"{'total':<17}{sum(res['arms'][a]['n_requests'] for a in arms):>9}{tin:>12}"
                f"{res['likely_usd']:>11.4f}{res['upper_bound_usd']:>11.4f}")
    rows.append("likely: input + assumed output per request (" + ", ".join(
        f"{a} {res['arms'][a]['assumed_output_tokens'] // res['arms'][a]['n_requests']}" for a in arms) + ")")
    rows.append("upper bound: input + max_tokens (" + ", ".join(f"{a} {MAX_TOKENS[a]}" for a in arms)
                + ") for every request")
    if not res["complete"]:
        rows.append(f"EXTRAPOLATED from a sample of {res['sample']} requests")
    for note in res["notes"]:
        rows.append("note: " + note)
    return "\n".join(rows)


def run_dryrun(args: argparse.Namespace, runner: Runner = wrapper_runner) -> int:
    res = run_dryrun_impl(Path(args.dir), args.sample, runner,
                          log=lambda m: print(m, file=sys.stderr, flush=True))
    print(format_table(res))
    return 0


# ---------------------------------------------------------------------------
# Submit
# ---------------------------------------------------------------------------

class Refused(Exception):
    pass


def submit_impl(dir_: Path, cap_usd: float, runner: Runner) -> dict[str, Any]:
    if (dir_ / "batch.json").exists():
        raise Refused("batch.json already exists in this directory; refusing to submit again")
    dry_path = dir_ / "dryrun.json"
    if not dry_path.exists():
        raise Refused("no dryrun.json; run dryrun first")
    dry = json.loads(dry_path.read_text(encoding="utf-8"))
    if not dry.get("complete"):
        raise Refused("dryrun was a sample; run a full dryrun before submitting")
    if dry.get("requests_sha256") != _sha(dir_ / "requests.jsonl"):
        raise Refused("requests.jsonl changed since the dryrun; run dryrun again")
    if dry["upper_bound_usd"] > cap_usd:
        raise Refused(f"upper bound ${dry['upper_bound_usd']:.4f} is above the cap ${cap_usd:.4f}")
    reqs = _read_lines(dir_ / "requests.jsonl")
    rc, out = runner("POST", "/v1/messages/batches", _dump({"requests": reqs}), 600)
    if rc != 0:
        raise RuntimeError(f"batch create failed: {_error_text(out)}")
    resp = json.loads(out)
    info = {
        "id": resp["id"], "created_at": resp.get("created_at"), "n_requests": len(reqs),
        "processing_status": resp.get("processing_status"), "cap_usd": cap_usd,
        "upper_bound_usd": dry["upper_bound_usd"], "likely_usd": dry["likely_usd"],
        "requests_sha256": dry["requests_sha256"],
    }
    (dir_ / "batch.json").write_text(json.dumps(info, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return info


def run_submit(args: argparse.Namespace, runner: Runner = wrapper_runner) -> int:
    try:
        info = submit_impl(Path(args.dir), args.cap_usd, runner)
    except Refused as exc:
        print(f"refused: {exc}", file=sys.stderr)
        return 1
    print(f"submitted batch {info['id']} with {info['n_requests']} requests")
    return 0


# ---------------------------------------------------------------------------
# Fetch and score
# ---------------------------------------------------------------------------

def parse_result(line: dict[str, Any]) -> dict[str, Any]:
    """One results line to {a_H, stop_reason, usage, result_type}."""
    res = line.get("result") or {}
    rtype = res.get("type", "missing")
    if rtype != "succeeded":
        return {"a_H": PARSE_ERROR, "stop_reason": rtype, "usage": {}, "result_type": rtype}
    msg = res.get("message") or {}
    stop = msg.get("stop_reason")
    usage = msg.get("usage") or {}
    base = {"stop_reason": stop, "usage": usage, "result_type": rtype}
    if stop in ("refusal", "max_tokens"):
        return {**base, "a_H": PARSE_ERROR}
    text = next((b.get("text", "") for b in msg.get("content") or [] if b.get("type") == "text"), None)
    parsed = parse_reply(text) if text is not None else None
    return {**base, "a_H": parsed["action"] if parsed else PARSE_ERROR}


def arm_of(cid: str) -> str:
    return cid.rsplit("__", 1)[1]


def usage_cost(usages: list[dict[str, Any]]) -> dict[str, Any]:
    tin = sum(u.get("input_tokens", 0) or 0 for u in usages)
    tin += sum(u.get("cache_creation_input_tokens", 0) or 0 for u in usages)
    tin += sum(u.get("cache_read_input_tokens", 0) or 0 for u in usages)
    tout = sum(u.get("output_tokens", 0) or 0 for u in usages)
    return {"input_tokens": tin, "output_tokens": tout, "usd": round(price(tin, tout), 6)}


def score(cases: list[Case], req_ids: list[str], lines: list[dict]) -> tuple[list[dict], dict[str, Any]]:
    by_case = {c.case_id: c for c in cases}
    by_id = {ln["custom_id"]: ln for ln in lines}
    rows = []
    for cid in req_ids:
        case_id = cid.rsplit("__", 1)[0]
        case = by_case[case_id]
        parsed = parse_result(by_id.get(cid, {}))
        rows.append({
            "custom_id": cid, "case_id": case_id, "mechanism": case.mechanism, "cls": case.cls,
            "length": case.knobs.get("length"), "arm": arm_of(cid), "a_star": case.a_star, **parsed,
        })

    def acc(rs: list[dict]) -> dict[str, Any]:
        ok = sum(r["a_H"] == r["a_star"] for r in rs)
        return {"n": len(rs), "correct": ok, "accuracy": ok / len(rs) if rs else None}

    validity: dict[str, Any] = {
        "model": MODEL, "gate": {"overall_min": GATE_OVERALL, "per_mechanism_min": GATE_PER_MECHANISM}, "arms": {}}
    for arm in dict.fromkeys(arm_of(cid) for cid in req_ids):
        rs = [r for r in rows if r["arm"] == arm]
        confusion: dict[str, dict[str, int]] = {}
        for r in rs:
            confusion.setdefault(r["a_star"], {}).setdefault(r["a_H"], 0)
            confusion[r["a_star"]][r["a_H"]] += 1
        errs = [r for r in rs if r["a_H"] == PARSE_ERROR]
        overall = acc(rs)
        per_mech = {m: acc([r for r in rs if r["mechanism"] == m]) for m in MECHANISMS}
        validity["arms"][arm] = {
            "overall": overall,
            "passes_gate": validity_gate(overall["accuracy"], {m: a["accuracy"] for m, a in per_mech.items()}),
            "per_mechanism": per_mech,
            "per_length": {str(L): acc([r for r in rs if r["length"] == L])
                           for L in sorted({r["length"] for r in rs})},
            "confusion": {k: dict(sorted(v.items())) for k, v in sorted(confusion.items())},
            "parse_errors": {"total": len(errs), "by_reason": dict(Counter(str(r["stop_reason"]) for r in errs))},
            "cost": usage_cost([r["usage"] for r in rs]),
        }
    validity["total_cost_usd"] = round(sum(a["cost"]["usd"] for a in validity["arms"].values()), 6)
    return rows, validity


def rescore_impl(dir_: Path, write: bool = False) -> dict[str, Any]:
    """Score a fetched dir again from its results.jsonl, offline, under the current gate.

    Writes validity.json and decisions.jsonl only with write=True.
    """
    results = dir_ / "results.jsonl"
    if not results.exists():
        raise Refused("no results.jsonl; fetch the batch first")
    lines = [json.loads(x) for x in results.read_text(encoding="utf-8").splitlines() if x.strip()]
    cases = list(read_jsonl(dir_ / "cases.jsonl"))
    req_ids = [r["custom_id"] for r in _read_lines(dir_ / "requests.jsonl")]
    rows, validity = score(cases, req_ids, lines)
    if write:
        _write_lines(dir_ / "decisions.jsonl", rows)
        (dir_ / "validity.json").write_text(json.dumps(validity, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return validity


def fetch_impl(dir_: Path, runner: Runner) -> dict[str, Any] | None:
    """Returns the validity dict when the batch has ended, else None (counts are printed)."""
    batch_path = dir_ / "batch.json"
    if not batch_path.exists():
        raise Refused("no batch.json; nothing was submitted from this directory")
    bid = json.loads(batch_path.read_text(encoding="utf-8"))["id"]
    rc, out = runner("GET", f"/v1/messages/batches/{bid}", None, 120)
    if rc != 0:
        raise RuntimeError(f"batch status failed: {_error_text(out)}")
    status = json.loads(out)
    if status.get("processing_status") != "ended":
        print(f"batch {bid}: {status.get('processing_status')} {json.dumps(status.get('request_counts'), sort_keys=True)}")
        return None
    rc, out = runner("GET", f"/v1/messages/batches/{bid}/results", None, 600)
    if rc != 0:
        raise RuntimeError(f"batch results failed: {_error_text(out)}")
    (dir_ / "results.jsonl").write_text(out if out.endswith("\n") else out + "\n", encoding="utf-8")
    lines = [json.loads(x) for x in out.splitlines() if x.strip()]
    cases = list(read_jsonl(dir_ / "cases.jsonl"))
    req_ids = [r["custom_id"] for r in _read_lines(dir_ / "requests.jsonl")]
    rows, validity = score(cases, req_ids, lines)
    _write_lines(dir_ / "decisions.jsonl", rows)
    (dir_ / "validity.json").write_text(json.dumps(validity, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return validity


def run_fetch(args: argparse.Namespace, runner: Runner = wrapper_runner) -> int:
    try:
        validity = fetch_impl(Path(args.dir), runner)
    except Refused as exc:
        print(f"refused: {exc}", file=sys.stderr)
        return 1
    if validity is not None:
        for arm, a in validity["arms"].items():
            print(f"{arm}: accuracy {a['overall']['correct']}/{a['overall']['n']} "
                  f"parse_errors={a['parse_errors']['total']} cost=${a['cost']['usd']:.4f}")
        print(f"total actual cost ${validity['total_cost_usd']:.4f}")
    return 0


def run_rescore(args: argparse.Namespace) -> int:
    try:
        validity = rescore_impl(Path(args.dir), write=args.write)
    except Refused as exc:
        print(f"refused: {exc}", file=sys.stderr)
        return 1
    g = validity["gate"]
    print(f"gate: overall >= {g['overall_min']}, every mechanism >= {g['per_mechanism_min']}")
    for arm, a in validity["arms"].items():
        worst = min((m for m in a["per_mechanism"] if a["per_mechanism"][m]["accuracy"] is not None),
                    key=lambda m: a["per_mechanism"][m]["accuracy"])
        print(f"{arm}: accuracy {a['overall']['correct']}/{a['overall']['n']} "
              f"lowest mechanism {worst} {a['per_mechanism'][worst]['accuracy']:.2f} "
              f"gate {'PASS' if a['passes_gate'] else 'FAIL'}")
    return 0


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def register(flip_sub: argparse._SubParsersAction) -> None:
    val = flip_sub.add_parser("validity", help="Batch API validity check: a_H vs a_star on the full history")
    sub = val.add_subparsers(dest="validity_command", required=True)

    b = sub.add_parser("build", help="write a mechanism-balanced sample and its batch requests")
    b.add_argument("--n", type=int, default=60)
    b.add_argument("--seed", type=int, default=11)
    b.add_argument("--arms", default=",".join(DEFAULT_ARMS),
                   help=f"comma list of arms, any of: {', '.join(ARMS)} (default: %(default)s)")
    b.add_argument("--mechanisms", default=None,
                   help="comma list: build the headline subset sample of these mechanisms (needs --prefix); "
                        "same seed, prefix and n as `headline init` give the same cases")
    b.add_argument("--prefix", default=None, help="case id prefix for --mechanisms (ids are <prefix>-NNNN)")
    b.add_argument("--out", required=True, help="output directory")
    b.set_defaults(func=run_build)

    d = sub.add_parser("dryrun", help="count input tokens (free) and price the batch")
    d.add_argument("--dir", required=True)
    d.add_argument("--sample", type=int, default=None, help="count only K requests and extrapolate")
    d.set_defaults(func=run_dryrun)

    s = sub.add_parser("submit", help="create the batch (spends money; needs a full dryrun and a cap)")
    s.add_argument("--dir", required=True)
    s.add_argument("--cap-usd", type=float, required=True)
    s.set_defaults(func=run_submit)

    f = sub.add_parser("fetch", help="check the batch and score the results when it has ended")
    f.add_argument("--dir", required=True)
    f.set_defaults(func=run_fetch)

    r = sub.add_parser("rescore", help="score a fetched dir again offline under the current gate")
    r.add_argument("--dir", required=True)
    r.add_argument("--write", action="store_true", help="rewrite validity.json and decisions.jsonl in the dir")
    r.set_defaults(func=run_rescore)

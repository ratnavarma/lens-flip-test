"""`lens flip headline init | freeze | dryrun | step | run | report | merge-report`."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from . import requests as rq
from .engine import Engine, Refused, init_run


def _log(msg: str) -> None:
    print(msg, file=sys.stderr, flush=True)


def cmd_init(args: argparse.Namespace) -> int:
    mechs = tuple(m.strip() for m in args.mechanisms.split(",") if m.strip()) if args.mechanisms else None
    try:
        info = init_run(args.out, args.n, args.seed, mechs, args.prefix, args.generator_version)
    except Refused as exc:
        print(f"refused: {exc}", file=sys.stderr)
        return 1
    print(f"wrote {info['cases']} cases and config.json to {args.out}")
    print("mechanisms: " + ", ".join(f"{k}={v}" for k, v in sorted(info["mechanisms"].items())))
    print("lengths: " + ", ".join(f"{k}={v}" for k, v in sorted(info["lengths"].items()))
          + "; distances: " + ", ".join(f"{k}={v}" for k, v in sorted(info["distances"].items())))
    return 0


def cmd_freeze(args: argparse.Namespace) -> int:
    from .protocol import DOC, VERSION, freeze, version_from_doc

    doc = Path(args.doc) if args.doc else DOC
    res = freeze([Path(d) for d in args.dir], doc, draft=not args.final,
                 version=args.protocol_version or version_from_doc(doc) or VERSION)
    print(f"wrote {res['doc']} and protocol.json in {len(res['runs'])} run dir(s); {len(res['hashes'])} hashes")
    return 0


def cmd_dryrun(args: argparse.Namespace) -> int:
    from .dryrun import MAX_COUNT_CALLS, format_tables, no_api_runner, run_dryrun

    if args.calibrate > MAX_COUNT_CALLS:
        print(f"refused: at most {MAX_COUNT_CALLS} count_tokens calls", file=sys.stderr)
        return 1
    if args.calibrate_from_run:  # zero API calls: a runner that refuses every call
        res = run_dryrun(args.dir, calibrate_k=0, runner=no_api_runner, log=_log,
                         calibrate_from_run=args.calibrate_from_run)
    else:
        res = run_dryrun(args.dir, calibrate_k=args.calibrate, calibration=args.calibration, log=_log)
    print(format_tables(res))
    return 0


def _engine(args: argparse.Namespace) -> Engine:
    return Engine(args.dir, rq.wrapper_runner, log=_log)


def cmd_step(args: argparse.Namespace) -> int:
    eng = _engine(args)
    try:
        r = eng.step(args.cap_usd)
    except Refused as exc:
        print(f"refused: {exc}", file=sys.stderr)
        return 1
    finally:
        eng.close()
    print(r)
    return 0


def cmd_run(args: argparse.Namespace) -> int:
    eng = _engine(args)
    try:
        r = eng.run(args.cap_usd, args.poll_seconds)
    except Refused as exc:
        print(f"refused: {exc}", file=sys.stderr)
        return 1
    finally:
        eng.close()
    print(r)
    return 0


def cmd_report(args: argparse.Namespace) -> int:
    from .metrics import tables, write_report

    protocol = args.protocol
    if protocol is None:  # the run's own protocol file decides; a run frozen before v2 has no version
        pj = Path(args.dir) / "protocol.json"
        protocol = int(json.loads(pj.read_text()).get("version", 1)) if pj.exists() else 1
    s = write_report(args.dir, protocol)
    print(tables(s))
    if args.json:
        print(json.dumps(s, indent=2, sort_keys=True))
    return 0


def cmd_merge_report(args: argparse.Namespace) -> int:
    from .merge import MergeRefused, merge_report

    try:
        s = merge_report(args.base, args.with_run, args.replace_mechanism, args.out, args.protocol)
    except MergeRefused as exc:
        print(f"refused: {exc}", file=sys.stderr)
        return 1
    from .metrics import tables

    print(tables(s))
    print(f"wrote summary_v{args.protocol}.json, tables_v{args.protocol}.txt, cases.jsonl and merge.json to {args.out}")
    return 0


def register(flip_sub: argparse._SubParsersAction) -> None:
    hl = flip_sub.add_parser("headline", help="headline run on Haiku through the Message Batches API, in rounds")
    sub = hl.add_subparsers(dest="headline_command", required=True)

    p = sub.add_parser("init", help="write the case sample and config.json to a new run dir")
    p.add_argument("--n", type=int, required=True)
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--mechanisms", default=None,
                   help="comma list: generate only these mechanisms (needs --prefix); default all six")
    p.add_argument("--prefix", default=None, help="case id prefix (default h<seed>); ids are <prefix>-NNNN")
    p.add_argument("--generator-version", type=int, choices=(2, 3), default=3,
                   help="2 regenerates the headline-300 cases; 3 plants U2 pairs M1+M3 and M2+M3 only (default 3)")
    p.set_defaults(func=cmd_init)

    p = sub.add_parser("freeze", help="hash prompts and the generator; write docs/flip-protocol-v2.md")
    p.add_argument("--dir", action="append", required=True, help="run dir (repeat for several)")
    p.add_argument("--doc", default=None, help="protocol file (default docs/flip-protocol-v2.md)")
    p.add_argument("--protocol-version", type=int, choices=(1, 2, 3), default=None,
                   help="default: the version in the --doc file name (flip-protocol-v3.md is 3), else 2")
    p.add_argument("--final", action="store_true", help="mark the protocol frozen, not draft")
    p.set_defaults(func=cmd_freeze)

    p = sub.add_parser("dryrun", help="simulate with the mock model, calibrate with count_tokens, price")
    p.add_argument("--dir", required=True)
    p.add_argument("--calibrate", type=int, default=120, help="count_tokens calls (at most 200)")
    p.add_argument("--calibration", default=None, help="reuse a calibration.json instead of new calls")
    p.add_argument("--calibrate-from-run", default=None, metavar="PILOT",
                   help="set the mock and the prices from a finished run; makes no API call")
    p.set_defaults(func=cmd_dryrun)

    p = sub.add_parser("step", help="submit one round, or fetch and route the round in flight (spends money)")
    p.add_argument("--dir", required=True)
    p.add_argument("--cap-usd", type=float, required=True)
    p.set_defaults(func=cmd_step)

    p = sub.add_parser("run", help="loop of step until the run is complete (spends money)")
    p.add_argument("--dir", required=True)
    p.add_argument("--cap-usd", type=float, required=True)
    p.add_argument("--poll-seconds", type=float, default=120)
    p.set_defaults(func=cmd_run)

    p = sub.add_parser("report", help="summary.json and tables from the run state (protocol 2: summary_v2.json)")
    p.add_argument("--dir", required=True)
    p.add_argument("--protocol", type=int, choices=(1, 2, 3), default=None,
                   help="metrics protocol; default is the version in the run's protocol.json, else 1")
    p.add_argument("--json", action="store_true")
    p.set_defaults(func=cmd_report)

    p = sub.add_parser("merge-report", help="replace one mechanism's cases of a finished run with a second run's, "
                                           "then report the combined cases (protocol 3)")
    p.add_argument("--base", required=True)
    p.add_argument("--replace-mechanism", required=True)
    p.add_argument("--with", dest="with_run", required=True)
    p.add_argument("--protocol", type=int, choices=(2, 3), default=3)
    p.add_argument("--out", required=True)
    p.set_defaults(func=cmd_merge_report)

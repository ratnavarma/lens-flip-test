"""CLI: uv run python -m lens.flip.explore6 <command>

    frame      verify the frozen inputs; write the frame, the extractive flips, the v3 minimal sets, the E1 draws
    dryrun     price both arms from the exact requests and stored usage; no model call
    freeze     write protocol_v6.json as DRAFT with the hashes and the dry-run price
    finalize   DRAFT to FINAL (needs --price-confirmed; refuses above the price gate or when the guard would refuse)
    run        run the arms in --arms (needs --cap-usd and --price-confirmed; the protocol must be FINAL)
    report     handoff_metrics_v6.jsonl (text free) and summary_v6.json from the finished arms

Paths: --results DIR, else $LENS_RESULTS, else ./results. Output goes to DIR/exploratory-v6.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from ..headline.engine import Refused
from . import dryrun as D
from . import engine as E
from . import frame as F
from . import protocol as PR


def parse_arms(text: str | None) -> list[str]:
    arms = [a.strip() for a in (text or ",".join(E.ARMS)).split(",") if a.strip()]
    bad = [a for a in arms if a not in E.ARMS]
    if not arms or bad or len(set(arms)) != len(arms):
        raise SystemExit(f"--arms needs distinct names from {', '.join(E.ARMS)}; got {text!r}")
    return arms


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m lens.flip.explore6")
    ap.add_argument("command", choices=["frame", "dryrun", "freeze", "finalize", "run", "report"])
    ap.add_argument("--results", default=None, help="results dir holding the frozen runs (default $LENS_RESULTS or results)")
    ap.add_argument("--arms", default=None, help="comma list of arms to run, any subset of " + ",".join(E.ARMS))
    ap.add_argument("--cap-usd", type=float, default=None, help="spend cap for the run; required by run")
    ap.add_argument("--price-confirmed", action="store_true", help="the dry-run price is inside the approved spend")
    ap.add_argument("--banned", default=os.environ.get("LENS_BANNED_WORDS"), help="private word list for the metric file check")
    ap.add_argument("--poll", type=float, default=60.0)
    a = ap.parse_args(argv)
    results = Path(a.results) if a.results else F.RESULTS
    root = results / "exploratory-v6"
    if a.command == "frame":
        print(json.dumps(F.stage_frame(root, results / "exploratory-v4", results), indent=1))
    elif a.command == "dryrun":
        res = D.dryrun(root, results)
        print(D.table(res))
        print(f"\nlikely ${res['total']['likely_usd']:.2f}, upper ${res['total']['upper_usd']:.2f}, likely with outputs x1.5 "
              f"${res['total']['likely_usd_if_outputs_x1_5']:.2f}; at or under ${res['price_gate_usd']:.2f}: "
              f"{res['likely_at_or_under_price_gate']}; guard at ${res['guard']['cap_usd']:.2f} fits: {res['guard']['fits']}; "
              f"written to {root / D.NAME}")
    elif a.command == "freeze":
        rec = PR.freeze(root)
        print(json.dumps({k: rec[k] for k in ("status", "frozen_at", "dryrun", "rules", "arm_configs")}, indent=1))
    elif a.command == "finalize":
        if not a.price_confirmed:
            raise SystemExit("finalize needs --price-confirmed (the dry-run price is inside the approved spend)")
        rec = PR.finalize(root)
        print(json.dumps({k: rec[k] for k in ("status", "frozen_at", "finalized_at", "dryrun")}, indent=1))
    elif a.command == "run":
        if not a.price_confirmed:
            raise SystemExit("run needs --price-confirmed (the run happens only inside the approved spend)")
        if a.cap_usd is None or a.cap_usd <= 0:
            raise SystemExit("run needs --cap-usd (a positive number)")
        try:
            out = E.run_arms(root, parse_arms(a.arms), cap_usd=a.cap_usd, poll=a.poll, log=lambda m: print(m, flush=True))
        except Refused as exc:
            print(f"REFUSED: {exc}", flush=True)
            E.update_ledger(root, a.cap_usd)
            return 3
        led = E.update_ledger(root, a.cap_usd)
        print(f"{out}; ledger ${led['total_usd']:.4f} of ${a.cap_usd:.2f}", flush=True)
    elif a.command == "report":
        from .analysis import headline, write
        banned = a.banned or ("banned_words.local.txt" if Path("banned_words.local.txt").exists() else None)
        s = write(root, results / "exploratory-v5", banned)
        print(headline(s))
        print(f"metric file check: flat rows, strings at most 32 characters, private word list "
              f"{'checked' if banned else 'NOT checked (no list given)'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

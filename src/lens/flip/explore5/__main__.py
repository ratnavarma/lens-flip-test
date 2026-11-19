"""CLI: uv run python -m lens.flip.explore5 <command>

    frame      verify the v4 frame, copy it, write the v3 minimal sets, the C1 draws and the input-token rule
    dryrun     price every arm from stored usage; no model call, no count_tokens call
    freeze     write protocol_v5.json as DRAFT with the hashes
    finalize   DRAFT to FINAL (only on Varma's yes to the price; needs --price-confirmed)
    init ARM   create one arm's run dir (needs --price-confirmed)
    run        run the arms in --arms (needs --cap-usd and --price-confirmed; the protocol must be FINAL)
    report     summary_v5.json from the finished arms

Paths: --results DIR, else $LENS_RESULTS, else ./results. Output goes to DIR/exploratory-v5.
"""

from __future__ import annotations

import argparse
import json
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
    ap = argparse.ArgumentParser(prog="python -m lens.flip.explore5")
    ap.add_argument("command", choices=["frame", "dryrun", "freeze", "finalize", "init", "run", "report"])
    ap.add_argument("arm", nargs="?")
    ap.add_argument("--results", default=None, help="results dir holding the frozen runs (default $LENS_RESULTS or results)")
    ap.add_argument("--arms", default=None, help="comma list of arms to run, any subset of " + ",".join(E.ARMS))
    ap.add_argument("--cap-usd", type=float, default=None, help="spend cap for the run; required by run")
    ap.add_argument("--price-confirmed", action="store_true", help="Varma has confirmed the dry-run price")
    ap.add_argument("--poll", type=float, default=60.0)
    a = ap.parse_args(argv)
    results = Path(a.results) if a.results else F.RESULTS
    root = results / "exploratory-v5"
    v4 = results / "exploratory-v4"
    if a.command == "frame":
        print(json.dumps(F.stage_frame(root, v4, results), indent=1))
    elif a.command == "dryrun":
        res = D.dryrun(root, results)
        print(D.table(res))
        print(f"\nlikely total ${res['total']['likely_usd']:.2f}, cap-limited upper ${res['total']['upper_usd']:.2f}; "
              f"at or under ${res['cap_check_usd']:.0f}: {res['likely_at_or_under_cap_check']}; written to {root / 'dryrun_v5.json'}")
    elif a.command == "freeze":
        print(json.dumps(PR.freeze(root), indent=1))
    elif a.command == "finalize":
        if not a.price_confirmed:
            raise SystemExit("finalize needs --price-confirmed (Varma's yes to the dry-run price)")
        print(json.dumps(PR.finalize(root), indent=1))
    elif a.command in ("init", "run"):
        if not a.price_confirmed:
            raise SystemExit(f"{a.command} needs --price-confirmed (the run happens only after Varma confirms the price)")
        if a.command == "init":
            if a.arm not in E.ARMS:
                raise SystemExit(f"init needs one arm of {', '.join(E.ARMS)}")
            print(E.init_arm(a.arm, root))
        else:
            if a.cap_usd is None or a.cap_usd <= 0:
                raise SystemExit("run needs --cap-usd (a positive number)")
            try:
                out = E.run_arms(root, parse_arms(a.arms), cap_usd=a.cap_usd, poll=a.poll,
                                 log=lambda m: print(m, flush=True))
            except Refused as exc:
                print(f"REFUSED: {exc}", flush=True)
                E.update_ledger(root, a.cap_usd)
                return 3
            led = E.update_ledger(root, a.cap_usd)
            print(f"{out}; ledger ${led['total_usd']:.4f} of ${a.cap_usd:.2f}", flush=True)
    elif a.command == "report":
        from .analysis import write
        s = write(root)
        print(json.dumps(s["spend"], indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())

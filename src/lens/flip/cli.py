"""`lens flip gen`: write generated cases to JSONL."""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter
from pathlib import Path

from . import batch
from .backends import OLLAMA_URL, OllamaTransport
from .cache import DEFAULT_PATH as DEFAULT_CACHE, CallCache
from .decider import CallStats, oracle_decider, ollama_decider
from .flip import STRATEGIES, flip_search
from .generator import DISTANCES, Knobs, generate_cases
from .llm import EMBED_MODEL, SUMMARY_MODEL, CachedEmbedder, OllamaEmbedTransport, OllamaText
from .residue import ollama_note_writer
from .runner import RunConfig, run_flip
from .schema import Case, write_jsonl
from .systems import SYSTEMS, get_system
from .systems.base import event_view
from .systems.summary import ollama_summarizer


VIEWS = ("drop-dstar", "random")


def register(subparsers: argparse._SubParsersAction) -> None:
    flip = subparsers.add_parser("flip", help="Decision Flip Test")
    flip_sub = flip.add_subparsers(dest="flip_command", required=True)
    gen = flip_sub.add_parser("gen", help="generate cases to a JSONL file")
    gen.add_argument("--n", type=int, required=True, help="number of cases")
    gen.add_argument("--seed", type=int, default=0)
    gen.add_argument("--out", required=True, help="output JSONL path")
    gen.add_argument("--length", type=int, default=400, help="events per history (100, 400, 1600)")
    gen.add_argument("--distance", choices=DISTANCES, default="middle")
    gen.add_argument("--distractors", type=int, default=2, help="distractors per history")
    gen.add_argument("--rare-share", type=float, default=0.5, help="share of rare mechanisms (M1-M3, U1, U2)")
    gen.set_defaults(func=run_gen)

    mock = flip_sub.add_parser("mock", help="run the flip search with the rule-oracle mock decider")
    mock.add_argument("--n", type=int, required=True, help="number of cases")
    mock.add_argument("--seed", type=int, default=0)
    mock.add_argument("--view", choices=VIEWS, default="drop-dstar")
    mock.add_argument("--keep", type=float, default=0.3, help="keep probability for --view random")
    mock.add_argument("--max-calls", type=int, default=48)
    mock.add_argument("--strategy", choices=STRATEGIES, default="group")
    mock.add_argument("--length", type=int, default=400, help="events per history (100, 400, 1600)")
    mock.add_argument("--distance", choices=DISTANCES, default="middle")
    mock.add_argument("--distractors", type=int, default=2, help="distractors per history")
    mock.add_argument("--rare-share", type=float, default=0.5, help="share of rare mechanisms")
    mock.add_argument("--out", required=True, help="output JSONL path for Distinction Records")
    mock.set_defaults(func=run_mock)

    run = flip_sub.add_parser("run", help="run a memory system through the flip test with an Ollama decider")
    run.add_argument("--system", choices=sorted(SYSTEMS), required=True)
    run.add_argument("--budget", type=int, required=True, help="view budget in tokens (ignored by full)")
    run.add_argument("--n", type=int, required=True, help="number of cases")
    run.add_argument("--seed", type=int, default=0)
    run.add_argument("--length", type=int, default=400, help="events per history (100, 400, 1600)")
    run.add_argument("--model", required=True, help="Ollama model name")
    run.add_argument("--max-calls", type=int, default=48, help="decider call cap per flip search")
    run.add_argument("--strategy", choices=STRATEGIES, default="group")
    run.add_argument("--verify", action="store_true", help="re-test each restored set with the v2 prompt")
    run.add_argument("--out", required=True, help="output directory")
    run.add_argument("--distance", choices=DISTANCES, default="middle")
    run.add_argument("--distractors", type=int, default=2)
    run.add_argument("--rare-share", type=float, default=0.5)
    run.add_argument("--num-ctx", type=int, default=16384)
    run.add_argument("--cache", default=DEFAULT_CACHE, help="sqlite call cache path")
    run.add_argument("--ollama-url", default=OLLAMA_URL)
    run.add_argument("--residue", action="store_true",
                     help="add the handoff stage: a note from the view, a second decider, a search over what it left out")
    run.add_argument("--summary-model", default=SUMMARY_MODEL, help="Ollama model for the summary system")
    run.add_argument("--embed-model", default=EMBED_MODEL, help="Ollama embedding model for rag-embed")
    run.add_argument("--note-model", default=None, help="Ollama model for the handoff note (default: --model)")
    run.set_defaults(func=run_run)

    batch.register(flip_sub)
    from .headline import cli as headline_cli

    headline_cli.register(flip_sub)
    from .omission import cli as omission_cli

    omission_cli.register(flip_sub)


def run_gen(args: argparse.Namespace) -> int:
    knobs = Knobs(length=args.length, distance=args.distance, distractors=args.distractors, rare_share=args.rare_share)
    cases = generate_cases(args.n, args.seed, knobs)
    write_jsonl(args.out, cases)
    counts = Counter(c.mechanism for c in cases)
    print(f"wrote {len(cases)} cases to {args.out}: " + ", ".join(f"{m}={counts[m]}" for m in sorted(counts)))
    return 0


def run_run(args: argparse.Namespace) -> int:
    note_model = args.note_model or args.model
    config = RunConfig(
        system=args.system, budget=args.budget, n=args.n, seed=args.seed, length=args.length,
        model=args.model, max_calls=args.max_calls, strategy=args.strategy, verify=args.verify,
        distance=args.distance, distractors=args.distractors, rare_share=args.rare_share,
        residue=args.residue,
        summary_model=args.summary_model if args.system == "summary" else "",
        embed_model=args.embed_model if args.system == "rag-embed" else "",
        note_model=note_model if args.residue else "",
    )
    cache = CallCache(args.cache)
    stats = CallStats()
    transport = OllamaTransport(args.ollama_url)
    decide = ollama_decider(args.model, "v1", args.num_ctx, transport=transport, cache=cache, stats=stats)
    verify = ollama_decider(args.model, "v2", args.num_ctx, transport=transport, cache=cache, stats=stats)
    system = get_system(
        args.system,
        summarizer=ollama_summarizer(
            OllamaText(args.summary_model, transport=transport, cache=cache, stats=stats)),
        embedder=CachedEmbedder(args.embed_model, transport=OllamaEmbedTransport(args.ollama_url),
                                cache=cache, stats=stats),
    )
    note_writer = ollama_note_writer(
        OllamaText(note_model, transport=transport, cache=cache, stats=stats)) if args.residue else None
    summary = run_flip(config, args.out, decide, verify, system=system, note_writer=note_writer, stats=stats,
                       log=lambda m: print(m, file=sys.stderr, flush=True))
    cache.close()
    summary["cost"]["truncated_prompts"] = transport.truncated
    (Path(args.out) / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


def build_view(case: Case, kind: str, keep: float) -> list[str]:
    """Ids of the events the mock agent sees."""
    if kind == "drop-dstar":
        gone = set(case.d_star)
        return [e.id for e in case.events if e.id not in gone]
    rng = random.Random(f"view|{case.case_id}|{case.seed}")
    return [e.id for e in case.events if rng.random() < keep]


def run_mock(args: argparse.Namespace) -> int:
    knobs = Knobs(length=args.length, distance=args.distance, distractors=args.distractors, rare_share=args.rare_share)
    cases = generate_cases(args.n, args.seed, knobs)
    records = []
    for case in cases:
        view = event_view(case, build_view(case, args.view, args.keep), f"mock-{args.view}")
        records.append(
            flip_search(
                case, view, oracle_decider, case.a_star,
                system=f"mock-{args.view}", budget=None, model="oracle", max_calls=args.max_calls,
                strategy=args.strategy,
            )
        )
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8", newline="\n") as fh:
        for r in records:
            fh.write(json.dumps(r.to_dict(), sort_keys=True, separators=(",", ":"), ensure_ascii=True) + "\n")
    print(summarise(cases, records, args.out))
    return 0


def summarise(cases: list[Case], records: list, out: str) -> str:
    by_status = Counter(r.status for r in records)
    calls = [r.calls_used for r in records]
    lines = [f"wrote {len(records)} records to {out}"]
    lines.append("status: " + ", ".join(f"{k}={by_status[k]}" for k in sorted(by_status)))
    lines.append(f"calls: mean={sum(calls) / len(calls):.2f} max={max(calls)}")
    restored = [r for r in records if r.status == "restored"]
    lines.append(f"cap hits: {sum(r.hit_cap for r in records)}")
    if restored:
        rc = [r.calls_used for r in restored]
        lines.append(f"restored calls: mean={sum(rc) / len(rc):.2f} max={max(rc)}")
    # Precision and recall against d_star on rare cases with a planted set where the view flipped.
    by_id = {c.case_id: c for c in cases}
    scored = [(r, by_id[r.case_id]) for r in records if r.status != "no_flip" and by_id[r.case_id].cls != "none" and by_id[r.case_id].d_star]
    prec, rec, exact = [], [], 0
    for r, c in scored:
        found, truth = set(r.event_ids), set(c.d_star)
        hit = len(found & truth)
        rec.append(hit / len(truth))
        if r.status == "restored":
            prec.append(hit / len(found) if found else 0.0)
        exact += r.status == "restored" and found == truth
    if scored:
        p = f"{sum(prec) / len(prec):.3f}" if prec else "n/a"
        lines.append(
            f"vs d_star on {len(scored)} flipped rare cases with a planted set: "
            f"restored={len(prec)} precision(restored)={p} recall(all)={sum(rec) / len(rec):.3f} exact={exact}"
        )
    else:
        lines.append("vs d_star: no flipped rare cases with a planted set")
    return "\n".join(lines)

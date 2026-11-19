"""`lens flip omission sample | parse-labels | run` (exploratory, outside protocol v3)."""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter
from pathlib import Path

from ..backends import OLLAMA_URL
from ..cache import CallCache
from ..headline.provenance import ROOT
from ..llm import EMBED_MODEL, CachedEmbedder, OllamaEmbedTransport
from . import KEY_NAME, LABELS_NAME, OUT_DIR, SEED, VAULT_FILE


def _log(msg: str) -> None:
    print(msg, file=sys.stderr, flush=True)


def _path(p: str) -> Path:
    q = Path(p).expanduser()
    return q if q.is_absolute() else Path.cwd() / q


def _embedder(out: Path, url: str, model: str) -> CachedEmbedder:
    cache = CallCache(out / "cache.sqlite")
    return CachedEmbedder(model, transport=OllamaEmbedTransport(url), cache=cache)


def cmd_sample(args: argparse.Namespace) -> int:
    from . import sample as S
    from .coverage import coverage
    from .frame import load_frame

    out = _path(args.out)
    vault = _path(args.vault)
    if vault.exists() and not args.force:
        print(f"refused: {vault} exists (it may hold labels). Use --force to overwrite.", file=sys.stderr)
        return 1
    t0 = time.monotonic()
    frame = load_frame(ROOT)
    out.mkdir(parents=True, exist_ok=True)
    cos = coverage(frame, _embedder(out, args.ollama_url, args.embed_model))
    pairs = S.classify(frame, cos)
    rows, _ = S.draw_sample(frame, pairs, args.seed)
    keys = S.key_rows(rows, args.seed)
    S.write_key(out / KEY_NAME, keys)
    vault.parent.mkdir(parents=True, exist_ok=True)
    vault.write_text(S.vault_markdown(frame, rows), encoding="utf-8")
    dt = time.monotonic() - t0
    print(f"wrote {len(rows)} units in {len({r.note_label for r in rows})} notes to {vault}")
    print(f"wrote key to {out / KEY_NAME}")
    print("by system: " + ", ".join(f"{s}={n}" for s, n in sorted(Counter(k['system'] for k in keys).items())))
    print("by stratum: " + ", ".join(f"{s}={n}" for s, n in sorted(Counter(k['stratum'] for k in keys).items())))
    print(f"D* units: {sum(k['dstar'] for k in keys)}; sampling took {dt:.1f} s")
    return 0


def cmd_parse(args: argparse.Namespace) -> int:
    from . import labels as L

    md, key = _path(args.file), _path(args.key)
    res = L.parse_file(md, key)
    out = _path(args.out) if args.out else key.parent / LABELS_NAME
    L.write_labels(out, res["labels"])
    c = res["counts"]
    print(f"read {md}")
    print(f"labelled {c['labelled']} of {c['key_units']}: y={c['y']} n={c['n']} ?={c['?']}")
    print(f"unlabelled {c['unlabelled']}, invalid {c['invalid']}, missing from the file {c['missing']}")
    for r in res["unlabelled"][:20]:
        print(f"  unlabelled: line {r['line']} {r['unit_label']}")
    for r in res["invalid"][:20]:
        print(f"  invalid: line {r['line']} {r.get('unit_label', '')} {r['reason']}")
    for u in res["missing"][:20]:
        print(f"  missing: {u}")
    print(f"wrote {len(res['labels'])} labels to {out}")
    return 0 if not (c["unlabelled"] or c["invalid"] or c["missing"]) else 1


def cmd_run(args: argparse.Namespace) -> int:
    from . import labels as L
    from .frame import load_frame
    from .judge import OllamaJudge
    from .runner import RunRefused, run

    out = _path(args.out)
    labels_path = _path(args.labels)
    if not labels_path.exists():
        print(f"refused: no labels file at {labels_path}. Run parse-labels first.", file=sys.stderr)
        return 1
    emb = _embedder(out, args.ollama_url, args.embed_model)
    judge = None
    if args.mode == "embedding+judge":
        judge = OllamaJudge(args.judge_model, cache=emb.cache, url=args.ollama_url)
    try:
        s = run(load_frame(ROOT), L.read_labels(labels_path), out, embedder=emb, mode=args.mode, judge=judge,
                embed_model=args.embed_model, judge_model=args.judge_model, labels_path=labels_path, log=_log)
    except RunRefused as exc:
        print(f"refused: {exc}", file=sys.stderr)
        return 1
    print(f"wrote {s['_path']}")
    if s["status"] != "complete":
        print(f"stopped: {s['status']}. See {out / 'judge_calibration.json'}", file=sys.stderr)
        return 2
    e = s["calibration"]["embedding_only"]
    print(f"tau {e['tau']:.4f}: agreement {e['agreement']:.3f} precision {e['precision']:.3f} "
          f"recall {e['recall']:.3f} kappa {e['kappa']:.3f}")
    p = s["h2c"]["pooled"]
    print(f"pooled H2c AUROC p_residue {p['auroc_p_residue']['value']} {p['auroc_p_residue']['ci95']}; "
          f"omission_nv {p['auroc_omission_nv']['value']} {p['auroc_omission_nv']['ci95']}")
    return 0


def register(flip_sub: argparse._SubParsersAction) -> None:
    om = flip_sub.add_parser("omission", help="exploratory omission comparator for H2c (outside protocol v3)")
    sub = om.add_subparsers(dest="omission_command", required=True)

    p = sub.add_parser("sample", help="draw the 200-unit calibration set; write the vault file and the key")
    p.add_argument("--out", default=str(ROOT / OUT_DIR))
    p.add_argument("--vault", default=VAULT_FILE)
    p.add_argument("--seed", type=int, default=SEED)
    p.add_argument("--force", action="store_true", help="overwrite an existing vault file")
    p.add_argument("--ollama-url", default=OLLAMA_URL)
    p.add_argument("--embed-model", default=EMBED_MODEL)
    p.set_defaults(func=cmd_sample)

    p = sub.add_parser("parse-labels", help="read the filled Markdown file; write labels.jsonl (never edits the file)")
    p.add_argument("--file", required=True)
    p.add_argument("--key", required=True)
    p.add_argument("--out", default=None, help="default: labels.jsonl beside the key")
    p.set_defaults(func=cmd_parse)

    p = sub.add_parser("run", help="calibrate on the labels, apply to the 490 notes, compute H2c")
    p.add_argument("--labels", required=True)
    p.add_argument("--out", default=str(ROOT / OUT_DIR))
    p.add_argument("--mode", choices=("embedding", "embedding+judge"), default="embedding")
    p.add_argument("--judge-model", default="qwen3:30b-a3b")
    p.add_argument("--ollama-url", default=OLLAMA_URL)
    p.add_argument("--embed-model", default=EMBED_MODEL)
    p.set_defaults(func=cmd_run)

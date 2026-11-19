"""Protocol freeze (design section 7.2, item 8): hashes of every prompt template and of the
generator, written to RUN/protocol.json and to docs/flip-protocol-v2.md (v1 stays as frozen on 2026-10-20).

`step` refuses to run when a prompt, the generator, config.json or cases.jsonl no longer
matches its frozen hash.
"""

from __future__ import annotations

import hashlib
import json
import re
import time
from pathlib import Path
from typing import Any

from .. import batch, decider, render, residue
from ..schema import Case, Unit, View
from ..systems import summary as summary_mod
from . import requests as rq
from .metrics import POWER_RULE_V2

ROOT = Path(__file__).resolve().parents[4]
VERSION = 2
DOC = ROOT / "docs" / f"flip-protocol-v{VERSION}.md"
SRC = Path(__file__).resolve().parents[1]


# The hashes frozen in docs/flip-protocol-v2.md (its section 6), and the run files of results/headline-300.
V2_HASHES = {
    "decider.action_schema": "b5c809762c595781e481572d7b6ead5fb9ef9a4afa1e2afcae093ca81a69efbb",
    "decider.v1": "f0828f379ef244779cec7c9e8fd2260b23b53f9da81cfd35bf0a8c4dd3673f04",
    "decider.v1.inserted": "74be01664a51887455a9445a08edbf8eae8c1f75b9a34db9d924a1b9d9b5a8d0",
    "decider.v2": "afa9c09a90ae59bfc1f92e858bc5d9d34125298c17c0bd74c1bfa70c9b9d718b",
    "decider.v2.inserted": "5628b0bf90fc428bd722e17d4860046d4f0fa69f2692f025a17b6e85d717de2b",
    "event_templates": "38f5302ef4a810a19679a838828c909cc0e1c753fd2676f6df5776c5af3ac919",
    "file:generator.py": "df562bd8f38fb7827c061825c4c07fb2af3d5c6762f9afe1a96fbe5c716b5dbd",
    "file:render.py": "46567cebd52980297247f58f1c46ad315eac94ff3736427fe092bcb85dc6dfaf",
    "note.prompt_p_residue": "d06b8aafacbcc2e16280c4e6ebd5069a2e5ec8baeb8e23a2e0829d435a08b6cd",
    "note.prompt_plain_dev": "0f5d2b2200c03ed60eca4fcaf3f9798adfbba5d1b13626819c2c6be06a2f9aa0",
    "note.schema": "23b194ce2295f649c6a2a17cdaedf6bccbc5ae05eda412d1d6a65c1af2ae88d0",
    "summary.prompt": "c9e5b1c1517b4b78dfd8117282b94fd21a6fb746183b4c2b4f2051e8f155da77",
    "summary.schema": "54669b5e6a1bfebb4d15788d41b5cd5fb8e51fc2d982eb2383a42262c748c90a",
}
V2_BASE_RUN = {"dir": "results/headline-300", "n": 300, "seed": 11,
               "cases.jsonl": "74a1e3a421da9e658cf930f5017a92bfa92d775af6dfb0f14b2cdcb7c0731374",
               "config.json": "cc00fc06f4d799ea0b5ba4cfb40c4253e7b4848674f53061f4f0dca399f155ab"}


def version_from_doc(doc: Path) -> int | None:
    """3 for docs/flip-protocol-v3.md, and so on; None when the file name does not say."""
    m = re.fullmatch(r"flip-protocol-v(\d+)\.md", doc.name)
    return int(m.group(1)) if m else None


def _sha(text: str | bytes) -> str:
    return hashlib.sha256(text.encode("utf-8") if isinstance(text, str) else text).hexdigest()


def _sentinel_case() -> Case:
    return Case(case_id="{CASE}", seed=0, mechanism="M0", events=[], query_period="{P}", query_text="{QUERY}",
                actions=[], a_star="", d_star=[], deleted_ids=[], cls="none", knobs={})


def prompt_texts() -> dict[str, str]:
    """Every template rendered with sentinel inputs, plus the reply schemas."""
    case = _sentinel_case()
    view = View(system="{SYSTEM}", budget=None, text="{RECORDS}")
    unit = [Unit(id="{ID}", text="{UNIT}")]
    out: dict[str, str] = {}
    for v in decider.PROMPT_VERSIONS:
        out[f"decider.{v}"] = rq.canonical(decider.build_messages(view, [], case, v))
        out[f"decider.{v}.inserted"] = rq.canonical(decider.build_messages(view, unit, case, v))
    out["decider.action_schema"] = rq.canonical(batch.ACTION_SCHEMA)
    out["summary.prompt"] = summary_mod.build_summary_prompt("{SUMMARY}", ["{RECORD}"], 1000)
    out["summary.schema"] = rq.canonical(rq.SUMMARY_SCHEMA)
    out["note.prompt_p_residue"] = residue.build_residue_note_prompt(view, case)
    out["note.schema"] = rq.canonical(residue.NOTE_SCHEMA)
    out["note.prompt_plain_dev"] = residue.build_note_prompt(view, case)
    out["event_templates"] = rq.canonical({
        "templates": {f"{t}/{s}": list(v) for (t, s), v in sorted(render.TEMPLATES.items())},
        "suffixes": [render.RESET_SUFFIX, render.CHARGE_SUFFIX, render.OFFSET_SUFFIX]})
    return out


def current_hashes() -> dict[str, str]:
    h = {k: _sha(v) for k, v in prompt_texts().items()}
    for name in ("generator.py", "render.py"):
        h[f"file:{name}"] = _sha((SRC / name).read_bytes())
    return h


def run_file_hashes(run_dir: Path) -> dict[str, str]:
    return {name: _sha((run_dir / name).read_bytes()) for name in ("config.json", "cases.jsonl")}


def freeze(dirs: list[Path], doc: Path = DOC, *, draft: bool = True, now: float | None = None,
           version: int = VERSION) -> dict[str, Any]:
    hashes = current_hashes()
    runs = []
    stamp = time.strftime("%Y-%m-%d %H:%M UTC", time.gmtime(now if now is not None else time.time()))
    for d in dirs:
        cfg = json.loads((d / "config.json").read_text())
        rec = {"version": version, "status": "draft" if draft else "frozen", "frozen_at": stamp, "hashes": hashes,
               "run_files": run_file_hashes(d), "n": cfg["n"], "seed": cfg["seed"], "dir": d.name}
        (d / "protocol.json").write_text(json.dumps(rec, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        runs.append((d, cfg, rec))
    doc.parent.mkdir(parents=True, exist_ok=True)
    doc.write_text(render_doc(runs, hashes, draft, stamp, version), encoding="utf-8")
    return {"doc": str(doc), "runs": [r[2] for r in runs], "hashes": hashes}


CHANGES_FROM_V1 = [
    "Power rule per hypothesis. Each hypothesis is tested only when its own analysis set is large enough. "
    "H2a needs at least 20 eligible cases. H2b needs at least 20 of its own eligible residue flips "
    "(event-view systems, restored search, every D* event in V). H2c needs at least 20 residue flips "
    "and at least 20 non-flips among the eligible cases. Below its minimum a hypothesis is descriptive: "
    "bounds and n, no claim, no label of fails or supported. v1 pooled one count of 20 residue flips over H2b and H2c. "
    "The pilot had 3 flips for H2b and 26 flips for H2c.",
    "Harmful flips split. The harmful gated rate stays for continuity. It is reported with two parts that sum to it. "
    "Wrong commit: a_H equals a*, a_V differs from a* and a_V is a committing action (not REQUEST_INFO, not "
    "PARSE_ERROR). Abstention: a_H equals a*, a_V is REQUEST_INFO and a* is not. PARSE_ERROR flips are reported "
    "separately, so wrong commit plus abstention plus parse error equals the harmful rate.",
    "Token helper, note only. The helper is unchanged from v1: ceil(characters / 2.2). The v1 pilot ran with it, "
    "and its usage gave 2.19 characters per input token on decision requests, net of the 331-token output-format "
    "overhead. Dry runs made before the 2026-10-20 calibration used 4 characters per token and are not comparable "
    "(their directories end in -v0-chars4). Run costs in v2 are priced from the pilot's real usage.",
]


def changes_from_v2(hashes: dict[str, str], cfg: dict[str, Any]) -> list[str]:
    """Protocol v3 (2026-10-21): the U2 fix. Declared before any v3 model call."""
    n_u2 = cfg["n"]
    changed = sorted(k for k in set(hashes) | set(V2_HASHES) if hashes.get(k) != V2_HASHES.get(k))
    L = [
        "v2 is `docs/flip-protocol-v2.md`. It stays as frozen. v3 changes the U2 cases only.",
        "",
        "1. Diagnosis, declared before any v3 model call. In the headline-300 run U2 scored 21 of 30 on the full "
        "history. That is under the 80% gate. Every miss is an M1+M2 case: 4 of 13 right, against 11 of 11 for "
        "M1+M3 and 6 of 6 for M2+M3. In M1+M2 both causes act on the same reading of P. The planted later actual "
        "reading for P is normal. That points to the estimate alone. So two causes is not the only sensible reading "
        "of the full history. The ground truth is contested. This is a flaw in the case design.",
        "2. U2 pair rule. U2 draws its pair only from M1+M3 and M2+M3, half each. A run plants the two pairs in turn "
        "by position. Any even count splits exactly half and half, crossed with the length. Each pair puts one cause "
        "on the activity charge (M1 or M2) and one on the other charges (M3). Both stay visible on their own. "
        "M1+M2 is never planted. The (type, subtype) counts still match across mechanisms. Each case carries a "
        "decoy for the slot it does not plant.",
        "3. Nothing else in the generator moves. Prompts, the action policy and every other mechanism are "
        "byte-identical to v2. The plant stream is seeded per mechanism, so M0, M1, M2, M3 and U1 come out the same "
        "for the same seed. `generator_version` 2 regenerates the headline-300 cases byte for byte.",
        f"4. Rerun scope. Only U2 is rerun, in `results/headline-u2-v3`: {n_u2} cases, seed {cfg['seed']}, case ids "
        f"`{cfg['sample']['case_ids']}`. Half the cases are at length 100 and half at length 400. Distances are "
        "drawn from the seeded stream. The ids cannot collide with headline-300 (`h11-NNNN`).",
        "5. Pre-check gate. First run `results/validity-u2-v3`. It holds the same cases (same seed and ids). It uses "
        "the adaptive-high arm only, with one full-history decision per case. The full U2 rerun runs only if U2 "
        "scores at least 24 of 30 (80%). Otherwise stop and report. Varma approves the rerun's cap after the "
        "pre-check, as for any spend.",
        "6. Merge rule. `lens flip headline merge-report --base results/headline-300 --replace-mechanism U2 "
        "--with results/headline-u2-v3 --protocol 3 --out results/headline-300-v3` takes every non-U2 case from "
        "the base run. It takes every case from the U2 run. It computes the full v2 metric set and the validity gate "
        "on the combined 300 cases. The bootstrap is over the combined cases. It writes `summary_v3.json` and "
        "`tables_v3.txt`. Base files stay untouched. The 30 old U2 cases and their 21 of 30 stay on record in the "
        "base run. They led to this change. They are not in the combined report.",
        "7. Every other item is identical to v2: hypotheses, power rule, systems, budgets, stages, models, "
        "request settings, metrics, validity gate and known limits. The hashes in section 6 equal the v2 "
        "hashes in every row except "
        + (", ".join(f"`{k}`" for k in changed) if changed else "none") + ". The rows are repeated here.",
        "",
        "| Item | v2 sha256 | v3 sha256 | same |",
        "| --- | --- | --- | --- |",
    ]
    for k in sorted(set(hashes) | set(V2_HASHES)):
        a, b = V2_HASHES.get(k), hashes.get(k)
        L.append(f"| {k} | `{a}` | `{b}` | {'yes' if a == b else 'NO'} |")
    L += ["", f"Base run kept as is: `{V2_BASE_RUN['dir']}`, N {V2_BASE_RUN['n']}, seed {V2_BASE_RUN['seed']}, "
          f"cases.jsonl `{V2_BASE_RUN['cases.jsonl']}`, config.json `{V2_BASE_RUN['config.json']}`."]
    return L


def render_doc(runs: list[tuple[Path, dict, dict]], hashes: dict[str, str], draft: bool, stamp: str,
               version: int = 1) -> str:
    cfg = runs[0][1]
    dec, summ, note = cfg["decider"], cfg["summarizer"], cfg["note_writer"]
    status = ("Draft for Varma's approval. Nothing has run under it. It binds once he approves it before phase 1c."
              if draft else "Frozen. Any change to a hashed item needs a new protocol version.")
    if version >= 3 and draft:
        status = ("Draft for Varma's approval. No v3 model call has run. It binds once he approves it, "
                  "before the validity pre-check.")
    v2 = version >= 2
    v3 = version >= 3
    L = [
        f"# Decision Flip Test: protocol v{version} ({'U2 rerun' if v3 else 'headline run'})",
        "",
        f"Status: {status}",
        f"Version: {version}",
        f"Written: {stamp} by `lens flip headline freeze`.",
        "Design: `docs/flip-test-design.md`, sections 4 to 10 and 7.2.",
        "",
    ]
    if v3:
        L += ["## Changes from v2", "", *changes_from_v2(hashes, cfg), ""]
    if v2:
        L += ["## Changes from v1", "",
              "v1 is `docs/flip-protocol-v1.md`. It stays as frozen. The pilot is readable under both protocols.", ""]
        L += [f"{i}. {c}" for i, c in enumerate(CHANGES_FROM_V1, 1)]
        L += [""]
    L += [
        "## 1. Hypotheses",
        "",
        "Eligible cases: a_H equals a* and a_A = a_V equals a_H at the residue budget. "
        "All bounds are 95% paired bootstrap bounds over cases.",
        "",
        "- H2a: residue_rate minus floor_rate is above zero. It fails if the bounds of the paired difference include zero.",
        "- H2b: on event-view systems, on residue flips where every D* event is in V, mean recall of D* in the found set "
        "beats recall in random subsets of the same size from the same candidates (1,000 draws per case). "
        "It fails if the bounds of the difference include zero.",
        "- H2c: the AUROC of p_residue for residue flips beats 0.5 and beats the omission fraction. "
        "It fails if the bounds of either difference include zero. Parse failures are counted, never imputed.",
    ]
    if v2:
        pr = {**POWER_RULE_V2, **(cfg.get("power_rule") or {})}
        L += [
            f"- Power rule, per hypothesis. H2a is tested with at least {pr['H2a_min_cases']} eligible cases. "
            f"H2b is tested with at least {pr['H2b_min_flips']} eligible residue flips of its own (event-view systems, "
            f"restored search, every D* event in V). H2c is tested with at least {pr['H2c_min_positives']} residue flips "
            f"and at least {pr['H2c_min_negatives']} non-flips among the eligible cases with a parsed p_residue.",
            "- Below its minimum a hypothesis is descriptive. It carries n and bounds, no claim, and neither the label "
            "fails nor the label supported. The rule applies to the pooled set and to each system's own set. "
            "The report prints every hypothesis with its own n and its minimum.",
        ]
    else:
        L += [
            f"- Power rule: H2b and H2c are tested only with at least {cfg['power_min_flips']} residue flips in the run. "
            "With fewer they are reported as descriptive. They carry bounds but no claim.",
        ]
    L += [
        "- Prompt note: asking for p_residue may change the notes. Every residue rate is reported for the p_residue prompt.",
        "",
        "## 2. Systems, budgets and stages",
        "",
        f"- Systems: {', '.join(cfg['systems'])}. Budgets: {', '.join(str(b) for b in cfg['budgets'])} tokens.",
        "- a_H: the decider on the full history, once per case.",
        "- a_V: the decider on each view. Flip search where a_V differs from a_H and a_H parsed: target a_H, "
        f"group-first, {cfg['max_calls']} calls per search, round-parallel ddmin, verified with the v2 prompt.",
        f"- Residue stage at B = {cfg['residue_budget']} for all systems on eligible cases: note A with p_residue, "
        f"B on the note, the floor (v2 prompt on V), truncation (B on the last {cfg['trunc_tokens']} tokens of V). "
        "A residue flip runs a flip search over the units of V missing from the note.",
        "- Gated metrics count only cases where a_H equals a*. The other cases are excluded and counted.",
        "- Validity gate: a_H right on at least 90% of cases and at least 80% of every mechanism."
        + (" In v3 it is applied to the combined 300 cases of the merge report. The pre-check gate is separate: "
           "U2 at least 24 of 30." if v3 else ""),
        "",
        "## 3. Models and request settings",
        "",
        "| Role | Model | Thinking | Effort | max_tokens | Reply format |",
        "| --- | --- | --- | --- | --- | --- |",
        f"| decider | {dec['model']} | adaptive (field omitted) | {dec['effort']} | {dec['max_tokens']} | action schema |",
        f"| summarizer | {summ['model']} | {json.dumps(summ['thinking'])} | {summ['effort']} | "
        f"{summ['max_tokens_factor']} x B | one-field text schema |",
        f"| note-writer | {note['model']} | adaptive (field omitted) | {note['effort']} | {note['max_tokens']} | "
        "note and p_residue schema |",
        f"| rag-embed | {cfg['embedder']['model']} (local Ollama) | - | - | - | vectors, cached |",
        "",
        f"Summary chunks: {summ['chunk_events']} events per step. Each step is cut to B tokens. "
        f"The note is cut to {note['note_tokens']} tokens. Every request goes through the Message Batches API "
        "in rounds. Identical requests share one key and are sent once.",
        "",
        "## 4. Sample and seeds",
        "",
        *([f"Mechanisms: {', '.join(cfg['sample']['mechanisms'])} only. Lengths 100 and 400 alternate. The distance "
           "knob is drawn per case from the seeded stream. U2 pairs M1+M3 and M2+M3 alternate by position, half each. "
           f"Generator version {cfg['sample'].get('generator_version', 3)}. "
           f"Distractors: {cfg['sample']['distractors']}. Bootstrap: {cfg['bootstrap']['reps']} resamples."]
          if cfg["sample"].get("mechanisms") else
          ["Mechanism mix: the generator's default power mix (rare share 0.5: M0 0.5, each rare mechanism 0.1). "
           "Lengths 100 and 400 alternate within each mechanism. The distance knob is drawn per case from the seeded "
           f"stream. Distractors: {cfg['sample']['distractors']}. Bootstrap: {cfg['bootstrap']['reps']} resamples."]),
        "",
        "| Run dir | N | Seed | sha256 of cases.jsonl | sha256 of config.json |",
        "| --- | --- | --- | --- | --- |",
    ]
    for d, c, rec in runs:
        L.append(f"| results/{d.name} | {c['n']} | {c['seed']} | `{rec['run_files']['cases.jsonl']}` | "
                 f"`{rec['run_files']['config.json']}` |")
    L += [
        "",
        "## 5. Metrics",
        "",
        "Per cell (system and budget): FCR raw and gated (all, rare only, reweighted to 1% rare), harmful flip rate "
        "(over all cases and over gated cases), right restraint on U1 and U2, over-caution on M0, flip-search "
        "precision and recall against D*, mean calls and cap hits. Residue block per system and pooled: n_eligible, "
        "residue_rate, floor_rate, trunc_rate, H2a, H2b, H2c, n_residue_flips and p_residue parse errors.",
        *(["Protocol v2 adds, for every cell, the harmful flip split into wrong commit, abstention and parse error "
           "(counts, rate over all cases and rate over gated cases), and the analysis-set size and minimum of each hypothesis.",
           "Files: `summary_v2.json` and `tables_v2.txt`."] if v2 else []),
        "",
        "## 6. Hashes (sha256)",
        "",
        "Prompts are rendered with sentinel inputs, so a change of wording changes the hash.",
        "",
        "| Item | sha256 |",
        "| --- | --- |",
    ]
    for k in sorted(hashes):
        L.append(f"| {k} | `{hashes[k]}` |")
    L += [
        "",
        "## 7. Known limits",
        "",
        "- The shared token helper counts ceil(characters / 2.2), calibrated with count_tokens on Haiku 5.5 "
        "(2.16 to 2.24 characters per token by request kind). Budgets B are close to real model tokens.",
        "- A summary step may stop at max_tokens. The partial text is kept and cut to B. The stop is counted.",
        "",
    ]
    return "\n".join(L)

# Decision Flip Test

Code and run outputs for the paper "Position: Agents Must Know When Their Handoff Notes Change the Decision" [arXiv id pending].

## What the test is

The Decision Flip Test asks one question: did a memory view change an agent's decision?

1. A generator writes synthetic account histories with a planted cause and a known right action.
2. A memory system cuts each history down to a view or a handoff note under a token budget.
3. A decider model acts on the full history and then on the view or note.
4. A flip is a change of action between the two.
5. A delta-debugging search finds the fewest left-out events that undo each flip.

The testbed is synthetic. It holds no personal or proprietary records.

## Install

You need Python 3.12 and [uv](https://docs.astral.sh/uv/).

```sh
uv sync
uv run pytest -q
```

The suite needs no network. Live-model tests skip unless `LENS_OLLAMA=1` is set.

## Layout

| Path | What it holds |
| --- | --- |
| `src/lens/flip/generator.py`, `render.py`, `schema.py` | case generator, history renderer, case schema |
| `src/lens/flip/systems/` | memory systems: recent, rag-bm25, rag-embed, summary, full |
| `src/lens/flip/decider.py`, `ddmin.py`, `pddmin.py` | decider prompts, flip search (sequential and round-parallel) |
| `src/lens/flip/batch.py`, `headline/` | the headline pipeline on the Message Batches API |
| `src/lens/flip/explore4/` | the exploratory v4 studies (Tables 4 to 6) |
| `src/lens/flip/explore5/` | the exploratory v5 and v5b studies (Tables 7 to 12) |
| `src/lens/flip/export_metrics.py` | writes the text-free metric files from the raw runs |
| `src/lens/flip/omission/` | an exploratory omission comparator, not used in the paper's tables |
| `docs/` | the design and the frozen protocols v1 to v5b |
| `results/` | released summaries, cases and metric files (see below) |
| `scripts/rebuild_paper.py` | rebuilds every table, the Figure 2 numbers and the paper facts |
| `scripts/reproduce_tables.py` | the earlier check of Tables 1 to 6 against the stored reports |

## Generate cases

```sh
uv run lens flip gen --n 60 --seed 0 --length 400 --out cases.jsonl
```

Each line is one case. It holds the events, the query, the right action `a_star` and the planted event ids.
The paper's 300 cases come back byte for byte:

```sh
uv run lens flip headline init --n 300 --seed 11 --generator-version 2 --out runs/headline-300
uv run lens flip headline init --n 30 --seed 12 --mechanisms U2 --prefix u3 --generator-version 3 --out runs/headline-u2-v3
```

Their `cases.jsonl` files hash to the sha256 values in `docs/flip-protocol-v3.md`.

## Run the headline pipeline

Step 1 is a dry run. A mock decider simulates every round and the run is priced. No API call is made.

```sh
uv run lens flip headline init --n 60 --seed 11 --out runs/pilot
uv run lens flip headline dryrun --dir runs/pilot --calibration results/headline-300/calibration.json
```

Step 2 freezes the prompt and generator hashes. Step 3 is the batch run. It spends money.
It submits one round at a time and stops at `--cap-usd`.

```sh
uv run lens flip headline freeze --dir runs/pilot --doc runs/pilot/protocol.md
uv run lens flip headline run --dir runs/pilot --cap-usd 10
uv run lens flip headline report --dir runs/pilot --protocol 3
```

Step 3 needs two outside pieces:

- An executable at `~/.local/bin/anthropic-api` that holds your API key. The call shape is `anthropic-api METHOD PATH --timeout SECONDS [--body -]`. It reads the JSON body on stdin and prints the response body. It exits 0 on success and 2 on a usage error.
- A local [Ollama](https://ollama.com) server with `nomic-embed-text` for the rag-embed system.

The paper's run merged two runs. Item 6 of `docs/flip-protocol-v3.md` gives the `merge-report` command.

## Rebuild every table in three commands

```sh
uv sync
uv run pytest -q
uv run python scripts/rebuild_paper.py --check
```

The third command rebuilds Tables 1 to 12 and the Figure 2 numbers. It needs no API key and makes no model call.
It reads only the metric files under `results/`.
It compares each cell with `results/paper_expected.json`, which holds the cells as the paper prints them.
It prints `match` or `MISMATCH` for each cell and exits with 1 on any mismatch.
A cell matches when each rebuilt number is within half a unit of the last printed digit.
The bootstrap is the frozen one: seed 11, 1,000 resamples, paired by case.

Without `--check` the script prints the tables and the paper facts.
`results/paper_facts.json` holds the facts with a one-line method each.

`scripts/reproduce_tables.py` is the earlier check. It rebuilds Tables 1 to 6 and compares the text with `tables_v3.txt` and `REPORT.md`.

| Paper table | Metric file |
| --- | --- |
| 1 | `results/headline-300-v3/stage1_metrics_v3.jsonl` |
| 2, 2a, 2b, 2c, 2d, 3, 3b | `results/headline-300-v3/handoff_metrics_v3.jsonl` |
| 4, 5, 6 | `results/exploratory-v4/handoff_metrics_v4.jsonl` |
| 7, 9, 10, 11, 12 | `results/exploratory-v5/handoff_metrics_v5.jsonl` |
| 8, 9 | `results/exploratory-v5/c1_control_metrics_v5.jsonl` |
| 10, first column | `results/exploratory-v5/judge_calibration_metrics_v5.jsonl` |

## What is shipped and what is not

Shipped:

- the generator, the harness and the frozen prompt templates in source
- the 300 generated cases
- the protocol documents and their sha256 hashes
- summaries of each run
- metric files with ids, action names, flags, counts, token counts and scores

Not shipped:

- notes, rewrites, summaries, view text, unit text and model reasons
- prompts rendered with real records
- raw model replies and the call cache

The metric files hold no model text. `uv run pytest -q` checks this on every file.
`src/lens/flip/export_metrics.py` wrote them from the raw runs. The raw runs are private.

## Released results

| File | Content |
| --- | --- |
| `results/headline-300-v3/summary_v3.json` | protocol v3 metrics with provenance hashes |
| `results/headline-300-v3/tables_v3.txt` | the same as text |
| `results/headline-300-v3/cases.jsonl` | the 300 merged cases |
| `results/headline-300-v3/stage1_metrics_v3.jsonl` | one row per case, system and view budget: actions only |
| `results/headline-300-v3/handoff_metrics_v3.jsonl` | one row per eligible handoff: actions, flags, restore status, scores |
| `results/exploratory-v4/summary_v4.json`, `REPORT.md` | exploratory v4 metrics and report |
| `results/exploratory-v4/handoff_metrics_v4.jsonl` | one row per handoff and v4 arm |
| `results/exploratory-v4/protocol_v4.json` | v4 prompt and arm hashes |
| `results/exploratory-v4/cost_estimate.json`, `spend.json` | v4 pre-study estimates and actual spend |
| `results/exploratory-v5/summary_v5.json`, `summary_v5b.json` | exploratory v5 and v5b metrics |
| `results/exploratory-v5/handoff_metrics_v5.jsonl` | one row per handoff and arm: X1, X2, X2b, C2, C3 and three C4 judges |
| `results/exploratory-v5/c1_control_metrics_v5.jsonl` | one row per v3 flip and C1 control draw |
| `results/exploratory-v5/judge_calibration_metrics_v5.jsonl` | one row per calibration pair: truth and each judge's answer |
| `results/exploratory-v5/judge2_frozen_v5.json` | prompt hash, threshold and settings of the local judge v2 |
| `results/exploratory-v5/protocol_v5.json`, `protocol_v5b.json` | v5 and v5b prompt and arm hashes |
| `results/exploratory-v5/spend.json` | v5 and v5b spend by arm |
| `results/paper_expected.json` | every table cell as the paper prints it |
| `results/paper_facts.json` | facts computed from the released files |
| `results/headline-300/calibration.json` | token calibration for dry runs |
| `results/validity-haiku-v4/validity.json`, `decisions.jsonl` | decider accuracy and usage that set the dry-run mock |

`results/exploratory-v4/cases.jsonl` is byte-identical to the v3 cases. It is left out to save space.

## Protocol chain

| Protocol | Document | Frozen commit | Scope |
| --- | --- | --- | --- |
| v1 | `docs/flip-protocol-v1.md` | 6f02358 | 60-case pilot |
| v2 | `docs/flip-protocol-v2.md` | 1654510 | first 300-case run |
| v3 | `docs/flip-protocol-v3.md` | e3a8f7b | headline run: U2 rebuilt, 30 cases rerun |
| v4 | `docs/flip-protocol-v4-exploratory.md` | 706ed36 | exploratory: Tables 4 to 6 |
| v5 | `docs/flip-protocol-v5-exploratory.md` | 52b6ee0 draft, 4b3d047 final | exploratory: Tables 7 to 12 |
| v5b | `docs/flip-protocol-v5b-x2-rerun.md` | 042b810 | exploratory: X2b and judge v2 |

The commit ids belong to the private working repository. They are not in this repository's history.
The documents and the sha256 hashes they freeze are here. The tests check that the prompts in source still hash to them.

## Spend

The paid calls used the Message Batches API. One v4 probe call was synchronous and cost under 0.01.
Local models cost nothing.

| Run | Protocol | Spend in USD |
| --- | --- | --- |
| 60-case pilot | v1 | 5.76 |
| 300-case run | v2 | 27.95 |
| U2 rerun, 30 cases | v3 | 1.66 |
| Headline run as merged | v3 | 29.61 |
| Exploratory v4, five arms | v4 | 4.73 |
| Exploratory v5, six arms | v5 | 2.13 |
| X2b rerun | v5b | 0.60 |
| Judge v2, local model | v5b | 0.00 |

The validity gate runs have no spend file.

## Vocabulary

Event names such as "reading", "device", "plan" and "statement" are generic labels.
They are frozen in the hashed prompts and templates. Changing them would need a new protocol and a rerun.

## Cite

```
FNU Ratnavarma. Position: Agents Must Know When Their Handoff Notes Change the Decision. [arXiv id pending], 2026.
```

## License

Apache-2.0. See `LICENSE`. Copyright 2026 FNU Ratnavarma.

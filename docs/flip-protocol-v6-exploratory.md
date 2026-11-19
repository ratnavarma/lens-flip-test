# Decision Flip Test: protocol v6 (exploratory)

Status: FINAL since 2026-11-19 07:58 UTC. Frozen before any model call. `results/exploratory-v6/protocol_v6.json` sha256 `1818614a6741b9624f6c241e9392832c967ca4effb03374761980d70dac95dbe`.
Version: 6-exploratory
Written: 2026-11-18, branch `exploratory-v5`, on top of 8a56e2b.
Approved: Varma authorised the model spend for this study on 2026-11-18. Run cap $3.00. Price gate $1.50 on the dry-run likely price.

## 0. Status of this protocol

This protocol is exploratory. It sits outside the frozen v3 protocol and the exploratory v4 and v5 protocols. None of those files is edited.

- No result here changes a v3, v4 or v5 number.
- `results/headline-300`, `results/headline-u2-v3`, `results/exploratory-v4` and `results/exploratory-v5` are read only. Their call caches are opened read only.
- Results go to `results/exploratory-v6/`, schema `lens.flip.exploratory/1`.
- Every v6 claim carries the label "exploratory".
- The protocol check refuses to run while the status is DRAFT or any hash differs.

Why v6. A reviewer found two gaps.

1. The v4 extractive margin has no specificity control. v4 read "recoverable flips on extractive notes are 29.4 percent, 19.0 points above the rewording floor of 10.4 percent [14.5, 24.0]". Extractive notes are thin (2.51 units, 115 tokens). Random lines may restore many flips too.
2. The v5 comparison is not symmetric. "Targeted" is the outcome of a search of up to 48 calls, so it carries selection. "Irrelevant" is one fixed draw per control.

## 1. Frame: the same 490 handoffs

- The 490 eligible handoffs of the v3 merge at B = 1,000, as in v4 and v5. No case is regenerated. No note is rewritten.
- `frame.jsonl` sha256 `bed6b1d00288354439f4a0459dc7d512cfc83245b369a141e9e4f1a88f6eb69f`. `cases.jsonl` sha256 `432e55ca579b74766e9097428fe513bd5a8c96496892dec1b023365ba1a4ed3c`. Both are checked against the v4 hashes before the copy.
- Extractive flips: read from the frozen state of v4 arm A-extractive. 193 flips, 144 restored by the search, 113 of those verified with the v2 prompt, 49 not restored. By system (flips / restored): rag-bm25 67 / 41, rag-embed 63 / 54, recent 28 / 26, summary 35 / 23.
- `e1_flips.jsonl` holds, per extractive flip, the note B saw, the search status and the minimal restoring set (unit ids in view order, rendered length). sha256 `582e9da022bb812cf8c5f1945d479d6354ab9d34d3d4f09533bb3308f507fd1e`.
- v3 flips: 142, of which 41 restored. `p1_min_sets.jsonl` is rebuilt as in v5 and equals the v5 file `v3_min_sets.jsonl` byte for byte (sha256 `434b26b8589ae65262445cfb3c7c79bc16f3179f7ada054b7fdc61ea2f713996`).
- v3 rewording floor: 51 of 490 handoffs.

How the search showed units to B (mirrored exactly).

- Both searches run `headline.tasks.run_search` on `note_view(note)`.
- Candidates are `residue_candidates(case, V, note)`: the units of V whose text the note does not contain. For an extractive note these are the units A did not select. The rebuilt candidate list equals the stored v4 list on all 193 flips.
- Each search call is `decide_request(note_view(note), inserted units in view order, case, "v1", decider)`. The inserted units appear in the "Additional records" block.
- Check at staging: the body rebuilt for each minimal set has the request key of a stored search call, and the stored answer is A's action. E1: 144 of 144. P1: 41 of 41.

## 2. The two arms

All requests go to claude-haiku-5-5 through the Batch API. B's settings are in section 4.

### E1: extractive flips, irrelevant controls and fresh re-reads

Controls, for each of the 193 extractive flips.

- K = 5 control sets by the v5 draw rule (`explore5.controls.draw_control`, unchanged). Seeds: `random.Random("x6-c1|<case>|<system>|<k>")`.
- Pool: the search's candidates minus the flip's minimal restoring set. For a flip the search did not restore, the pool is every candidate.
- Length: len of the units joined by newlines, within 10 percent of the minimal set's length. A flip without a minimal set takes the median minimal-set length of the 144 restored extractive flips (137 characters).
- The walk, in short: shuffle the pool, add units that keep the length at or below 1.1 times the target, stop at 0.9 times, repeat up to 200 shuffles, keep the first set inside the band. A copy of an earlier control of the same flip counts as a failed attempt. A draw that cannot reach the band is kept and flagged. A set that is empty or a copy is not sent.
- Shown to B as the search shows inserted units: the extractive note as the view, the control units in view order as inserted units, v1 prompt. Targeted and irrelevant insertion differ in content only.
- Keyed by request body, as in v5 C1. The arm's call cache starts empty, so every control is a new sample.
- On the frozen data: 965 draws, 870 sent, 895 inside the band, 858 inside the band and sent, 88 copies, 70 outside the band. 179 flips have at least one matched and sent control. 14 do not (all on the summary view; 10 restored, 4 not). `e1_controls.jsonl` sha256 `95f2417d96780c290c43329662953bbc5360d71ed1de73874cc22fc860113b11`.

Fresh re-reads, for each of the 144 restored extractive flips.

- 5 new reads of the note plus the minimal restoring set. The request body is the search's restoring call, byte for byte.
- The cache is bypassed with the v5 C3 salt rule (`explore5.controls.salted_key`): key = sha256(canonical body + NUL + "repeat=" + tag), tags t0 to t4. The body sent is unchanged. Five keys, five custom_ids, five independent samples.
- This removes the selection asymmetry: the targeted side is now a fixed set read fresh, like a control.

Requests: 870 controls plus 720 re-reads = 1,590. One round.

### P1: paraphrased v3 flips, fresh re-reads

- For each of the 41 restored v3 flips: 5 new reads of the v3 note plus the v3 minimal restoring set. Same request shape as the v3 search's restoring call. Same salt rule and tags.
- No new controls. The irrelevant side is the frozen v5 C1 result.
- Requests: 205. One round.

## 3. Pre-registered analysis

Fixed here and in `explore6/analysis.py` (`ANALYSIS_SPEC`, hashed as x6.analysis) before any model call.

Definitions.

| Term | Definition |
| --- | --- |
| restored | B's action equals A's action. A reply that is refused, cut, unparsed or failed twice counts as not restored. Counts are reported. |
| targeted, fresh | The share of the 5 fresh re-reads of the minimal set that are restored. 0 for a flip with no restoring set. |
| targeted, v5 | 1 if the search restored the flip, else 0. |
| irrelevant | The share of the flip's matched and sent controls that are restored. |
| assessed | The flip has at least one matched and sent control, as in v5. |
| bootstrap | The v3 one: 1,000 resamples of the 300 cases, seed 11. Every handoff of a resampled case comes along. Bounds are 95 percent. |

Primary.

1. Extractive specificity. The mean over assessed extractive flips of (targeted, fresh minus irrelevant), paired by flip, with paired bootstrap bounds.
2. The falsification quantity. The content-specific recoverable rate on extractive notes = the sum over assessed flips of (targeted, fresh minus irrelevant), divided by 490, with bounds. Then its paired difference from the frozen v3 rewording floor (51 of 490 = 10.4 percent). Pairing is the v4 one (`explore4/analysis.py`, `Stats.diff`): the difference of the two sums over the same 490 handoffs, resampled by case.

Rule for primary 2. The earlier claim "survives its own falsification test" is supported under specificity control only if the lower bound of this difference is above zero. Otherwise it is not supported. The result is reported whichever way it comes out.

Sensitivity (does not change the rule).

- Every sent control in place of matched controls only. Assessed then means at least one sent control.
- Upper bracket. Flips without a matched control count with their full targeted, fresh share and no irrelevant share.

Secondary.

| # | Quantity |
| --- | --- |
| S1 | Primary 1 and primary 2 with targeted, v5. This is the definition behind Table 8 of the paper. |
| S2 | Irrelevant restore rate, split by flips the search restored and flips it did not. |
| S3 | The share of assessed flips where at least one matched control restores. |
| S4 | The fresh re-read rate itself: how often a minimal restoring set restores again. E1 and P1. |
| S5 | The v5 omission term with targeted, fresh from P1, on the v3 flips assessed in v5 with the frozen v5 C1 controls. The paper reports 2.8 [1.2, 4.7] under the v5 definition. Both values are given side by side. |
| S6 | Per-system rows. Descriptive only. |

Also descriptive: the fresh re-read rate split by the search's v2 verification, and the count of sets by number of restoring re-reads.

No correction for multiple comparisons.

## 4. Models and request settings

Read from the v5 arm config (`explore5.engine.arm_config`). Nothing is new.

| Role | Model | Thinking | Effort | max_tokens | Prompt | Reply |
| --- | --- | --- | --- | --- | --- | --- |
| B, both arms | claude-haiku-5-5 | adaptive (field omitted) | high | 6000 | v1 decider | v3 action schema |

Batch rates, USD per million tokens: 0.05 input and 0.25 output. Requests go through the v3 headline engine by way of the v5 engine: one key per request, rounds, one retry of errored requests, the key-holding wrapper as a subprocess.

## 5. Runner, guards and order

- `uv run python -m lens.flip.explore6 <command>`: `frame`, `dryrun`, `freeze`, `finalize`, `run`, `report`.
- `run --arms E1,P1 --cap-usd 3.00 --price-confirmed`. Order E1, then P1. State is resumable: a rerun picks up each arm's saved state, open batch and call cache. No request is paid twice.
- `finalize` refuses when the dry-run likely price is above $1.50 or when a round's upper bound would not fit the $3.00 cap.
- Spend guard, as in v4 and v5. Before each round, the ledger total plus the round's likely cost must stay at or under the cap. The round's upper bound (max_tokens on every request) must also fit inside the cap minus the other arm's spend. A refusal stops the run. It is never worked around.
- The ledger is `results/exploratory-v6/spend.json`.

## 6. Outputs

- `results/exploratory-v6/summary_v6.json`: every number with n and bounds, the hashes and the spend.
- `results/exploratory-v6/handoff_metrics_v6.jsonl`: one flat row per eligible handoff. Ids, system, mechanism group, counts and 0/1 flags only. No note, unit or prompt text. The summary is computed from this file alone.
- Code: `src/lens/flip/explore6/`. Tests: `tests/flip/test_explore6.py`.

## 7. Price (dry run, nothing spent)

Computed by `dryrun` from the exact request bodies and stored usage. No model call, no count_tokens call, no batch.

| Arm | Stage | Requests | Likely USD | Upper USD |
| --- | --- | --- | --- | --- |
| E1 | controls | 870 | 0.237 | 1.355 |
| E1 | fresh re-reads | 720 | 0.197 | 1.122 |
| P1 | fresh re-reads | 205 | 0.058 | 0.320 |
| Total | | 1,795 | 0.49 | 2.80 |

- Likely total $0.49. With every output 50 percent longer, $0.69. Upper bound $2.80 (max_tokens on every request).
- The likely total is under the $1.50 gate.
- Guard preview at the $3.00 cap: E1's round bound is $2.48 and fits. P1's round bound is $0.32 and fits after E1's likely spend, also with outputs 50 percent longer.
- Input tokens per request: the v5 rule. Likely ceil(chars / 2.2) plus 331. Upper bound ceil(chars / 1.902) plus 331.
- Output tokens per request, likely: 882 for E1 (the stored mean of the v4 extractive search calls, 1,839 requests), 906 for P1 (the stored mean of the v3 search calls). Upper bound: max_tokens.

## 8. Hashes (sha256)

| Item | sha256 |
| --- | --- |
| x6.e1.control_rule | `f24e277328dfe745f15744b8722164eb2e8ed597258d30aa74b6a05478b3273f` |
| x6.draw_source | `c60cf2970ce9e5b94dec250451f44f36657c879a319f4bf1c18bedbe7595704b` |
| x6.reread_rule | `7bde0ae00d27461ed914fb63bfd36281c20b8fc7536d0e433426895a07ee9143` |
| x6.analysis | `18b0334951ac96967c3e0f2d3968bf071a7be27e2504e61aa80ae2ff7f36712e` |
| decider.v1 | `f0828f379ef244779cec7c9e8fd2260b23b53f9da81cfd35bf0a8c4dd3673f04` |
| decider.v1.inserted | `74be01664a51887455a9445a08edbf8eae8c1f75b9a34db9d924a1b9d9b5a8d0` |
| decider.v2 | `afa9c09a90ae59bfc1f92e858bc5d9d34125298c17c0bd74c1bfa70c9b9d718b` |
| decider.action_schema | `b5c809762c595781e481572d7b6ead5fb9ef9a4afa1e2afcae093ca81a69efbb` |
| event_templates | `38f5302ef4a810a19679a838828c909cc0e1c753fd2676f6df5776c5af3ac919` |
| file:render.py | `46567cebd52980297247f58f1c46ad315eac94ff3736427fe092bcb85dc6dfaf` |
| note.prompt_p_residue | `d06b8aafacbcc2e16280c4e6ebd5069a2e5ec8baeb8e23a2e0829d435a08b6cd` |
| note.schema | `23b194ce2295f649c6a2a17cdaedf6bccbc5ae05eda412d1d6a65c1af2ae88d0` |

The v3 rows are reused unchanged and equal the v3, v4 and v5 hashes. x6.draw_source is the source text of the v5 draw function. x6.reread_rule includes the source text of the v5 salt function.

Arm request settings, as `config.json` minus the assumed output tokens:

| Arm | sha256 |
| --- | --- |
| E1 | `89494e6c70b3debb0c1a8af5a695b73bd2d2842774833f0ae5ba883777bcdd16` |
| P1 | `ee0d7974f0a4824fd41a320d1d1d98a64ef3820b38a7f83618fdf8a9b051cec0` |

Run files:

| File | sha256 |
| --- | --- |
| frame.jsonl | `bed6b1d00288354439f4a0459dc7d512cfc83245b369a141e9e4f1a88f6eb69f` |
| cases.jsonl | `432e55ca579b74766e9097428fe513bd5a8c96496892dec1b023365ba1a4ed3c` |
| e1_flips.jsonl | `582e9da022bb812cf8c5f1945d479d6354ab9d34d3d4f09533bb3308f507fd1e` |
| e1_controls.jsonl | `95f2417d96780c290c43329662953bbc5360d71ed1de73874cc22fc860113b11` |
| p1_min_sets.jsonl | `434b26b8589ae65262445cfb3c7c79bc16f3179f7ada054b7fdc61ea2f713996` |
| calibration_v6.json (the v5 input-token rule) | `7dfa41aa22ab0772d3fc9e8b428f388ffa85f7e18686f9c958f14ec1f0727225` |
| protocol_v6.json (FINAL) | `1818614a6741b9624f6c241e9392832c967ca4effb03374761980d70dac95dbe` |

`results/` is git-ignored, so the last row is the record of the frozen file. The engine checks every hash before every step.

## 9. Known limits and readings

- One sample per request. The reader with adaptive thinking is not deterministic. Five re-reads estimate a share in steps of 0.2.
- A fresh re-read cannot exceed the search's own call: the search kept the set because one read restored. The fresh rate is expected to fall below 1. That fall is the selection effect this protocol measures. It is not an error.
- "Irrelevant" means outside the minimal restoring set. The minimal set is 1-minimal, not unique. Another unit of V may carry the same content. A control can therefore hold relevant content, which raises the irrelevant rate and lowers the content-specific rate. The primary quantities are conservative in that direction.
- Extractive minimal sets are short: 109 of 144 hold one unit. A one-unit control of equal length is a random single line of V.
- 14 of 193 extractive flips have no matched control, all on the summary view. They drop out of the primary sums and are counted. The two sensitivity rows bracket them.
- A flip the search did not restore has targeted share 0 and can have a positive irrelevant share. Its term is then negative. This is by design, as in v5.
- The rule for primary 2 compares a content-specific rate with a floor that has no such control. The floor is the frozen v3 quantity and is kept as it is, so the comparison is the one v4 made.
- P1 changes the targeted side only. Its controls are the v5 C1 controls: one read each, drawn with the x5 seeds.
- Controls and re-reads are new samples from the same model as the search. A change in the served model between the v4 run and this run would show up in the fresh re-read rate. It cannot be separated from selection here.
- No correction for multiple comparisons. Per-system results are descriptive.
- Prices are estimates. The ledger holds the actual spend after the run.
- The frozen notes are stored text and are used as they are. They sit in `results/`, which is git-ignored. The v6 code, rules and this document carry no word of the private list (test).

# Decision Flip Test: protocol v4 (exploratory)

Status: Frozen before any v4 model call. Any change to a hashed item needs a new protocol version.
Version: 4-exploratory
Written: 2026-10-27 00:05 UTC, branch `exploratory-v4` from `flip-test-v0`.
Approved: Varma, 2026-10-26, three exploratory studies with a hard cap of $30 on the Batch API.

## 0. Status of this protocol

This protocol is exploratory. It sits outside the frozen v3 protocol (`docs/flip-protocol-v3.md`).

- No result here changes a v3 verdict or a v3 number.
- `results/headline-300-v3` is read only. No file in it is touched.
- Every v4 claim carries the label "exploratory".
- The studies answer review changes 2, 3 and 9 of "Between States v2 harsh review 2026-10-26".

## 1. Frame: the frozen handoffs

All three studies reuse the 490 eligible handoffs of the v3 merge at B = 1,000. No case is regenerated.

- Source runs: `results/headline-300` (U2 cases dropped) and `results/headline-u2-v3`, merged by the v3 `merge_data`.
- Eligible: a_H = a* and a_A = a_V = a_H, as in v3. Target = a_A.
- Views: rebuilt from the stored event ids or summary text of each cell. A's decision is the frozen a_V.
- The v3 note, p_residue, a_B, floor action and truncation action come from the frozen residue rows.
- `results/exploratory-v4/frame.jsonl` sha256 `bed6b1d00288354439f4a0459dc7d512cfc83245b369a141e9e4f1a88f6eb69f`.
- `results/exploratory-v4/cases.jsonl` sha256 `432e55ca579b74766e9097428fe513bd5a8c96496892dec1b023365ba1a4ed3c`. It is byte-identical to `results/headline-300-v3/cases.jsonl`.
- By system: rag-bm25 163, rag-embed 144, recent 98, summary 85. v3 residue flips: 142.

Bootstrap: the v3 bootstrap. 1,000 resamples of the 300 cases, seed 11. Every handoff of a resampled case comes along. Every comparison with a v3 quantity is paired by case and handoff. The code reproduces the v3 residue bounds [0.248031, 0.331933] and floor bounds [0.077552, 0.132797] exactly (test `test_reproduces_v3_rates`).

## 2. Study A1: extractive notes (review change 2)

Question. Does omission alone flip B more often than rewording noise, once the note cannot paraphrase or advise?

Arm.

1. A (Haiku 5.5, note-writer settings of v3) sees the units of V, each with a label R01, R02 and so on, and the query.
2. A lists the labels it picks, most important first, and gives p_residue. The reply schema allows only the labels of that view (an enum per request).
3. Assembly rule. Walk A's list in order. Keep a unit if the note still fits in 150 tokens (shared helper, ceil(chars / 2.2), lines joined by newlines). Skip it otherwise and go on. Unknown and repeated labels are skipped and counted. The kept units are printed verbatim, one per line, in view order.
4. B (Haiku 5.5, v1 prompt, decider settings of v3) decides from the note.
5. A flip is a_B differing from a_A.
6. On a flip, the v3 flip search runs: group-first round-parallel ddmin, 48 calls, candidates = the units of V not in the note, v2 verification.

Units: event lines for event views, summary sentences for the summary view.

Metrics.

- Primary (HX1). Insertion-recoverable flip rate on extractive notes minus the v3 rewording floor (51 of 490). Paired bootstrap difference. Recoverable means the search status is "restored", cap hit included, as in the v3 `restored` count.
- HX1 reads "exceeds the floor" when the lower bound is above zero and "does not exceed" otherwise. This is the review's falsification test for Haiku as reader.
- Secondary: the same with v2-verified restores only; the extractive residue rate; differences against the v3 residue and v3 recoverable rates (41 of 490); the search outcome counts; rates by system.
- H2c on extractive notes, exploratory: AUROC of A's p_residue and of the omission fraction. The omission fraction is not degenerate here, because notes are verbatim.

## 3. Study A2: advice-stripped notes (review change 2)

Question. How much of the v3 residue rate rides on the advice in A's note?

Strip rule, applied to each frozen v3 note. Sentences come from the shared splitter (`split_sentences`). A sentence is dropped when any rule fires:

- R1: a canonical action name in upper case, with an underscore or a space (EXPLAIN, OPEN_REVIEW, CHECK_DEVICE, APPLY_CREDIT, REQUEST_INFO).
- R2: open review, check device, apply credit or request info in any case, with a space, underscore or hyphen.
- R3: the sentence opens with a directive word. The list is: action, advise, apply, ask, best, check, choose, confirm, default, do, escalate, get, hold, lean, next, obtain, open, proceed, recommend, recommended, recommendation, request, review, route, suggest, suggested, treat, use, verify.
- R4: the sentence holds recommend, suggest or advis followed by any letters.

Kept sentences are joined by one space. On the frozen notes the rule changes 418 of 490 notes. It was fitted on the frozen note text only, before any v4 call.

A note with no dropped sentence is byte-identical to the v3 note, so its B request is identical. Its v3 answer stands, with no new call. B (Haiku, v1) decides from every changed note. An emptied note is shown as "(no records)" and counted.

Metric (HX2, two-sided). Residue rate of stripped notes minus the v3 rate, paired. Also on the changed notes only, against the floor, by system, and the flip transitions in each direction.

## 4. Study B: second readers on the frozen v3 notes (review change 3)

Readers decide from the 490 frozen v3 notes with the v1 prompt. The writer stays Haiku, so p_residue is the frozen v3 value.

- B-sonnet: claude-sonnet-5-5 through the Batch API, decider settings of v3 (adaptive thinking with the field omitted, effort high, max_tokens 6000, action schema).
- Before the batch, one synchronous Sonnet call on one frozen note checks that the API takes these settings and measures output length. It is charged at list price and logged as a probe. It sets the assumed output tokens for the estimate (twice the probe, at least 1,500). It changes no request.
- B-local: qwen3:30b-a3b through local Ollama. Settings: think off, temperature 0, seed 0, num_ctx 32768, the decider reply schema, one retry with the reminder on an unparsable reply. Free.
- Local validity gate: 30 cases, 5 per mechanism, drawn with seed 41 from the 300 cases. Full history, v1 prompt. The local reader runs only at 24 of 30 or better (80 percent). On a fail it is skipped and reported.
- Reader view baseline, declared optional: each reader also decides from V itself with the v2 prompt. view_miss = that action differs from a_A. For Haiku this is the v3 floor. For the local reader it runs with the gate's pass. For Sonnet it is a separate batch arm (B-sonnet-view) and runs last, only if the money allows.

Metrics (HX3), per reader: residue rate with bounds; the paired difference against Haiku; H2c AUROC of the writer's p_residue for that reader's flips, against 0.5; per-system residue and AUROC (descriptive); residue minus the reader's own view baseline.

## 5. Study C: note-budget sweep (review change 9)

A writes notes at 300 and 600 tokens on the 490 handoffs with the v3 note prompt. Only the two numbers in it change. At 150 the v4 prompt renders byte-identical to the frozen v3 prompt (hash check in the freeze and a test). The reply is cut at the budget with the shared helper. B (Haiku, v1) decides. The 150 arm is the frozen v3 data.

Metrics (HX4): residue rate per budget with bounds; paired differences 300 minus 150, 600 minus 150 and 600 minus 300; mean note tokens; share of notes within 2 tokens of the cap; H2c AUROC per budget; rates by system. Dose-response reads "decreasing" when 600 minus 150 has an upper bound below zero.

No flip searches run in Study C.

## 6. Models and request settings

| Role | Model | Thinking | Effort | max_tokens | Reply |
| --- | --- | --- | --- | --- | --- |
| A, extractive | claude-haiku-5-5 | adaptive (field omitted) | high | 6000 | labels and p_residue, label enum per view |
| A, budget notes | claude-haiku-5-5 | adaptive (field omitted) | high | 6000 | v3 note schema |
| B, Studies A and C | claude-haiku-5-5 | adaptive (field omitted) | high | 6000 | v3 action schema |
| B, Study B | claude-sonnet-5-5 | adaptive (field omitted) | high | 6000 | v3 action schema |
| B, Study B local | qwen3:30b-a3b (Ollama) | think off | - | - | decider reply schema |

Batch rates, USD per million tokens: Haiku 0.05 input and 0.25 output, Sonnet 1.00 and 5.00. Requests go through the v3 headline engine: one key per request body, dedup, rounds, one retry of errored requests, the anthropic-api wrapper.

## 7. Order, budget and stop rule

Order: A1 extractive, then B-sonnet (probe first), then C, then A2 advice-stripped, then the optional B-sonnet-view. B-local is free and may run at any point.

- Before each study, the estimate goes to `results/exploratory-v4/cost_estimate.json`: calls, tokens and dollars at Batch rates, likely and upper.
- The ledger `results/exploratory-v4/spend.json` sums actual usage of every arm and the probe. It is updated after every batch.
- Stop rule (Varma). Before each round, if the ledger total plus the round's likely cost passes $30, the round is not submitted and the work stops there.
- Engine guard. The round's upper bound (max_tokens on every request) must also fit inside $30 minus the other arms' spend. A refusal by this guard also stops the arm. It is never worked around.

Pre-study estimates (likely / upper, USD): A1 0.75 / 3.43, B-sonnet 7.77 / 15.22 before the probe, C 0.76 / 3.10, A2 0.12 / 0.65, B-sonnet-view 8.22 / 15.71.

## 8. Outputs

`results/exploratory-v4/summary_v4.json` (schema lens.flip.explore/4) and `REPORT.md`. They carry every metric with its n, bounds, the prompt hashes and the spend. Code: `src/lens/flip/explore4/`. Tests: `tests/flip/test_explore4.py`.

## 9. Hashes (sha256)

Rendered with sentinel inputs, as in v3. `x4.*` items are new. The other rows are v3 prompts reused unchanged. They equal the v3 hashes.

| Item | sha256 |
| --- | --- |
| x4.extract.prompt | `c8b8697b944b9ebded64e427ea7f408869f83d6d845ae1d8a79bd21cf35f3711` |
| x4.extract.schema | `55976ef045f27a3d7d8f6ddeec88f80a0230c628286e62afef43d28371d10731` |
| x4.note_budget.prompt.150 | `d06b8aafacbcc2e16280c4e6ebd5069a2e5ec8baeb8e23a2e0829d435a08b6cd` |
| x4.note_budget.prompt.300 | `ed78331b70730c9f565ba6108b310b664f090fb557567c7c5dbbb47d0e80744f` |
| x4.note_budget.prompt.600 | `c98fcb27385d7ef144dd8d0675cd0f41c45c9ffd6a8fbddc0f68c403a0e4cc3e` |
| x4.note_budget.schema | `23b194ce2295f649c6a2a17cdaedf6bccbc5ae05eda412d1d6a65c1af2ae88d0` |
| x4.strip.rule | `11258fa7668fb82afb3ef0c4a26068532264300fe0da3b3f874e216598a0edc7` |
| decider.v1 | `f0828f379ef244779cec7c9e8fd2260b23b53f9da81cfd35bf0a8c4dd3673f04` |
| decider.v1.inserted | `74be01664a51887455a9445a08edbf8eae8c1f75b9a34db9d924a1b9d9b5a8d0` |
| decider.v2 | `afa9c09a90ae59bfc1f92e858bc5d9d34125298c17c0bd74c1bfa70c9b9d718b` |
| decider.action_schema | `b5c809762c595781e481572d7b6ead5fb9ef9a4afa1e2afcae093ca81a69efbb` |
| note.prompt_p_residue | `d06b8aafacbcc2e16280c4e6ebd5069a2e5ec8baeb8e23a2e0829d435a08b6cd` |
| note.schema | `23b194ce2295f649c6a2a17cdaedf6bccbc5ae05eda412d1d6a65c1af2ae88d0` |
| event_templates | `38f5302ef4a810a19679a838828c909cc0e1c753fd2676f6df5776c5af3ac919` |
| file:render.py | `46567cebd52980297247f58f1c46ad315eac94ff3736427fe092bcb85dc6dfaf` |

Arm request settings, as `config.json` minus the assumed output tokens:

| Arm | sha256 |
| --- | --- |
| A-extractive | `c0e229e99d4be622167b1a4390fc31aac967986fa89282349f5506ad991d5f12` |
| B-sonnet | `f6d7f078d5156a491d2dd5672ca25d5f5b8f07caf801a6b73ab99bda7e9fbaa3` |
| C-budget | `e7d7a544c1f1975c5d23414730e4f977f101e4d114280c805ef1314b2b1a6738` |
| A-stripped | `17f30d46193bb4ca8b127463c74b8bb23479ff8afbc4f5377329f3e2bb29bcc7` |
| B-sonnet-view | `4b0a2e0d22fb78bc1c78dad62e959cd0e61ef1a81c0c5d728cf0c87414304291` |
| local reader and gate | `9af60c18151ba0a9b9196f765dcdc1cde722b93f63fd9c9e54e3d5d614140595` |

`results/exploratory-v4/protocol_v4.json` holds the same hashes. The engine checks them before every step and refuses on any change.

## 10. Known limits

- One sample per request, as in v3. Haiku with adaptive thinking is not deterministic, so part of any difference is sampling noise. The rewording floor measures that noise for Haiku on V only.
- The extractive note can hold only a few event lines at 150 tokens. Its residue rate mixes omission with the loss of A's synthesis.
- The strip rule is a fixed text rule. It may miss advice in other words and may drop a factual sentence that opens with a listed word.
- The second readers read notes written for Haiku by Haiku. The local reader is a different family and much smaller.
- No correction for multiple comparisons. Per-system results are descriptive.
- Study C shares the 150 arm with v3, run in October. The 300 and 600 notes are new samples.

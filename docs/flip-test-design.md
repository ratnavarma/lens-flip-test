# Decision Flip Test: design v0

Status: design, 2026-10-20. Steps 1 to 5 built on branch flip-test-v0.

Owner: Varma. Design by Claude (Opus). Build by Sonnet agents, runs by Haiku agents.
Parent plan: a private planning note (not released), section 4.

## 1. Purpose

The test asks one question of any memory system. Did the agent act on a memory view that lacked a distinction which would have changed its action?

It returns three things:

- the False Closure Rate of the system at a stated budget
- for each flipped decision, the smallest set of events that restores the right action
- the class of each failure (unobserved, unretained, unresolved or unexpressed).

The test is open code. It measures memory. It contains no repair method.

## 2. Terms

- Event: one dated record in a history, with a stable id.
- History H: the ordered events of one account.
- Query q: a decision request with a closed action set.
- Memory view V: the text a memory system hands the decider, at most B tokens.
- Handoff note N: a short note one agent writes for another from V.
- Ground truth a*: the right action, fixed by the generator.
- Reference a_H: the decider's action given the full history.
- Planted differentiator D*: the event ids that decide a*.

## 3. Case generator

Domain: generic long-horizon account histories with corrections. No industry words appear in code, data or prose.

### 3.1 Event types

All types occur in every history at ordinary rates:

- statement (period, amount)
- payment
- reading (period, value, actual or estimated)
- correction (refers to an earlier event id)
- device event (installed, replaced, serviced)
- plan notice (plan changed, effective date)
- adjustment (credit approved, credit applied)
- contact (topic, channel).

A rare mechanism is never a rare event type. It is a common type in an uncommon position or combination. Surface keywords cannot reveal the answer.

### 3.2 Mechanisms and actions

The query: "The statement for period P is higher than usual. Choose one action."

| Mechanism | Share in 99/1 | Differentiator D* | Ground truth a* |
| --- | --- | --- | --- |
| M0 seasonal | 99% | none | EXPLAIN |
| M1 estimate not corrected | rare | estimated reading for P, a lower actual reading after the statement | OPEN_REVIEW |
| M2 device swap | rare | device replaced inside P, no reset reading | CHECK_DEVICE |
| M3 unapplied credit | rare | credit approved months earlier, never applied | APPLY_CREDIT |
| U1 unobserved | rare | D* deleted from the history. P is a low-demand month, so the seasonal explanation does not fit | REQUEST_INFO |
| U2 unresolved | rare | two mechanisms both planted, no tie-break. Protocol v3: M1+M3 or M2+M3 only, never M1+M2 | REQUEST_INFO |

Committing means choosing any action except REQUEST_INFO.

How the cause shows in the records. Readings are per-period activity figures, never running counts. Statements carry line items (activity charges, other charges, credits) that add up to the amount. Every high-demand month in every year is raised by one account factor (1.6 to 1.8), so a seasonal month matches the same month of earlier years.

- M0: P matches the same calendar month of earlier years within 10%. It is above the account median only because of the season.
- M1: P's reading is an estimate at least 1.5 times that month's norm. P's statement is computed from it. A later actual reading for P is in line with the norm. No correction names the estimate.
- M2: a device is replaced inside P and no reset reading follows. P's reading carries over the old device's figure and is about twice the norm. Decoys show a replacement, a reset reading and a normal figure.
- M3: P's statement has a one-off charge in other charges, with credits 0. An earlier credit approval for the same amount names that charge and is never applied. Decoys show approved credits that a later statement applies in its credits line.
- U1: built from M1, M2 or M3 with D* deleted, P in a low-demand month. The amount stays high and no record explains it.
- U2: two of the M1, M2 and M3 causes are both visible in P. Protocol v3 (2026-10-21) plants only M1+M3 or M2+M3, half each, so one cause sits on the activity charge and one on the other charges. In M1+M2 both causes act on the same reading of P and the later actual reading points to the estimate alone. The headline-300 run scored 4 of 13 on that pairing, against 11 of 11 and 6 of 6. `generator_version=2` keeps the old draw to regenerate headline-300.

### 3.3 Distractors

Every history carries harmless look-alikes:

- estimated readings in other periods that were corrected in time
- device service events with a reset reading
- credits that were applied
- complaint contacts that change nothing.

These measure over-reaction. A system that flags every anomaly fails on them.

Clean window. The query period P, the calendar month before P and the month after P hold no device event and no plan notice, except the replacement a mechanism plants in P (M2, plus U2 when it includes M2). A plan notice counts as inside the window when its date or its effective date is. Background, decoy and distractor device and plan events are moved out to the next allowed month and never deleted. The (type, subtype) counts still match across mechanisms. The v2 Haiku run showed why. A nearby install, service or plan notice read as the cause. The v3 run added the month after P, where a device install was read as the cause.

### 3.4 Knobs

| Knob | Values in v0 |
| --- | --- |
| History length (events) | 100, 400, 1600 |
| Distance of D* from the end | near, middle, far |
| Distractors per history | 2, 8 |
| Memory budget B (tokens) | 500, 1000, 2000 |
| Rare share in the sample | 50% for power, reweighted to 1% in reports |
| Wording | three templates per event type, chosen by seed |

### 3.5 Generator checks

Each must pass before any model run:

1. The rule oracle matches a* on 100% of cases.
2. Deleting D* changes a* on 100% of rare cases.
3. Swapping distractors never changes a*.
4. A keyword classifier on event types scores near chance on mechanism.
5. Same seed, same bytes.

## 4. Memory systems under test

Every system implements one interface: ingest the events, then return a view of at most B tokens for the query.

| System | v0 | Notes |
| --- | --- | --- |
| full | yes | Reference only. Used where the history fits the context. |
| recent | yes | The last B tokens. |
| summary | yes | Rolling LLM summary in chunks, final summary at most B. |
| rag-bm25 | yes | Top events by BM25 until B is full. |
| rag-embed | yes | Top events by local embeddings until B is full. |
| Mem0, Letta, LangMem | phase 2 | Open-source agent memory. Check each licence and local-model support at build time. UNVERIFIED. |

Each system uses its documented defaults. Configs are published with results. Maintainers get a chance to send a better config before release.

## 5. Decider

One prompt for all systems: the view, the query and the six actions. The reply is JSON with the action, a short reason and the event ids it relied on.

- Development: a local model through Ollama (qwen3:30b-a3b), no cost.
- Headline: Haiku 5.5 on all cases and Sonnet 5.5 on a subset, through the Batch API with the anthropic-api wrapper. No session reads the key.
- Temperature 0. Every call is cached by a hash of model, prompt and settings.

REQUEST_INFO means the records do not settle the cause, or two or more different problems are present for the period, because each would call for a different action and a reviewer must decide first. Normal seasonal activity is never a problem or a cause for this rule. The line is in one place (`POLICY` in `decider.py`) and both the Ollama and the batch request builders use it, in the v1 and v2 wordings. The other four action meanings are unchanged. The v2 Haiku run acted on one of the two planted causes in U2 (3 of 10 right), hence the rule. The v3 Haiku run read seasonal activity as a second cause in M3 cases, hence the seasonal exclusion.

Validity gate (Varma, 2026-10-20): on histories that fit the context, a_H must match a* on at least 90% of cases overall and on at least 80% of the cases of every mechanism. Below that the task tests reasoning, not memory. Make the wording simpler and re-run. One function applies it: `validity_gate` in `metrics.py`. `runner.py` and `batch.py` both call it. A mechanism with no cases is skipped. The v4 run (Haiku 5.5, adaptive-high) scores 56 of 60 overall and 8 of 10 on U2, so it passes.

### 5.1 Batch validity check (Haiku 5.5)

Code: `src/lens/flip/batch.py`, CLI `lens flip validity build|dryrun|submit|fetch`.

- Sample: 60 cases, 10 per mechanism, half at length 100 and half at 400 (seed 11).
- Arms on the full history with the v1 prompt, chosen with `build --arms` (comma list). The default is `adaptive-medium,adaptive-high`. Both omit `thinking` (adaptive is the default on Haiku 5.5), set `output_config.effort` to medium or high and use max_tokens 6000. The old arms `off` (thinking disabled, 1024) and `adaptive` (8000, medium) still build. fetch reads the arms of any v1 or v2 dir from its requests.jsonl. The reply is forced to the action schema (enum of five names, additionalProperties false).
- All calls go through the anthropic-api wrapper. No session reads the key.
- Dry run: `POST /v1/messages/count_tokens` for each request. It is free. count_tokens takes model, system, messages and, for the old off arm, thinking. Max_tokens and output_config are not sent.
- Price at Batch rates ($0.05 per million input tokens, $0.25 per million output tokens): a likely cost and a hard upper bound (max_tokens on every request). Assumed output tokens per request: 1,500 for adaptive-medium and 3,000 for adaptive-high. The old arms keep 150 for off and 1,500 for adaptive.
- `submit` needs a full dry run, a `--cap-usd` at or above the upper bound and no existing `batch.json`. Varma approves the price first.
- `fetch` writes results.jsonl, decisions.jsonl and validity.json: accuracy per arm, per mechanism and per length, a confusion table, parse errors by reason and actual cost from usage.
- Gate: an arm passes at 90% or more correct overall and 80% or more on every mechanism (`validity_gate`). A parse error, a refusal and a max_tokens stop count as wrong.
- `rescore --dir D [--write]` scores a fetched dir again from results.jsonl under the current gate, offline.

### 5.2 Headline driver (Batch rounds)

Code: `src/lens/flip/headline/` and `src/lens/flip/pddmin.py`. CLI `lens flip headline init|freeze|dryrun|step|run|report`.

- Every unit of work is a task with JSON state in `RUN/state/cases/`. The tasks are a_H, the summary chain, the flip search and the residue bundle.
- A task replays its logic each round over its stored answers. A missing answer becomes a pending request.
- A request key is the sha256 of its body. Identical requests share one key and one cache row. They are sent once.
- One round sends every pending request in one batch, split at 100,000 requests or 128 MB.
- Results go into the call cache by custom_id and key. The next replay picks them up.
- Errored or expired requests are retried once, then marked failed. A failed decision is a PARSE_ERROR.
- The flip search runs ddmin by levels. One level asks every subset and complement in one round. It then reduces as sequential ddmin does. Without the cap it returns the same set. The 48-call cap and group-first order stay.
- The summarizer is Haiku 5.5 at effort medium with thinking off and max_tokens 1.5 B. The note-writer runs as the decider arm and returns `{"note", "p_residue"}`.
- Spend guard: `--cap-usd` is required. A round is refused when spend so far plus its upper bound passes the cap. Actual cost from usage goes to `RUN/spend.json`.
- `step` refuses until `freeze` has written `protocol.json`. It also refuses if a prompt, the generator, config.json or cases.jsonl has changed since.
- `dryrun` runs the whole pipeline on a fake Batches endpoint with a mock model. The mock is the rule oracle plus noise calibrated on the v4 run. Input tokens come from at most 200 count_tokens calls. It prices the run at Batch rates and makes no Messages call.

## 6. Minimal flip search

For every case where a_V differs from a_H, find the smallest set of events whose insertion restores a_H.

1. Insert every event missing from V. If a_H does not return, label the case "not recoverable by insertion" and stop.
2. Run delta debugging (ddmin, Zeller and Hildebrandt 2002) over the missing events. Inserted events keep their dates and order.
3. Cap the search at 48 decider calls. Report cases that hit the cap.
4. Re-test the final set once with a reworded prompt. Keep it only if both runs agree.

Views. A memory system returns a `View`: `system`, `budget`, `text` (what the decider sees), `event_ids` (events quoted verbatim, empty for a summary) and `units` (event lines for an event view, sentences for a summary). The decider prompt shows the view text under "Records:". Inserted units follow in a block headed "Additional records:", in history order. The search candidates are the case events not in `view.event_ids`, so a summary view has every event as a candidate. A search may also take its own candidate list of units (the residue stage does). The group of an event is "type/subtype". A sentence unit is in the group "sentence".

The output is a Distinction Record v0:

```
case_id, system, budget, model
events: [ids and text of the minimal set]
action_from_view, action_restored
class, calls_used, verified
stage ("flip" or "residue"), units_inserted (texts of the minimal set)
```

On generated cases the planted D* gives exact scoring. Report precision and recall of the found set against D*.

## 7. Residue stage

This stage measures the Unsaid Residue and feeds the position paper.

1. Agent A gets V and the query and writes a handoff note N of at most 150 tokens, in plain text, for a colleague who will choose the action. The note-writer is an injected callable. The default is Ollama. The note is cut to 150 tokens with the shared token helper.
2. Agent B decides from N and the query, with the standard decider prompt and N as the view text.
3. Target = a_H on cases where a_H equals a*. A is the decider on V, so a_A = a_V. A residue flip is a case where a_A equals the target and a_B does not.
4. On a residue flip, run the flip search with the view text N. The candidates are the units of V that N does not contain verbatim: the event lines of an event view, the sentences of a summary. The result names what the note left out. The record has stage "residue".

Residue rate = P(a_B wrong | a_A right) on gated cases. Notes are written only for those cases. `lens flip run --residue` writes `residue.jsonl` (one row per note), `residue_records.jsonl` (the searches) and a `residue` block in summary.json.

### 7.2 Controls and hypotheses for the position paper

Added 2026-10-20 from the position-paper session's spec. Eligible cases are gated cases (a_H = a*) where a_A = a_V equals the target. Bounds are 95% paired bootstrap bounds over cases.

1. Rewording floor. Decide again from V with the v2 prompt. floor_rate = P(a_V' differs from a_A | eligible). It measures how often the decider disagrees with itself on the same view. B deciding from V is no control, because a_A = a_V by construction.
2. Truncation note. N_trunc is the last 150 tokens of V. trunc_rate = P(B wrong on N_trunc | eligible). It shows whether A's note beats plain cutting.
3. H2a: residue_rate minus floor_rate is above zero. It fails if the bounds of the paired difference include zero.
4. H2b: event-view systems only (recent, rag-bm25, rag-embed), on residue flips where every D* event is in V. Mean recall of D* in the found set is compared with recall in random subsets of the same size from the same candidates (1,000 draws per case). It fails if the bounds of the difference include zero.
5. H2c: the note-writer returns {"note": ..., "p_residue": ...}. p_residue is A's probability that something left out would change the colleague's action. Score its AUROC for residue flips over eligible cases. Baselines: 0.5 and the omission fraction (units of V not in N over units of V). It fails if the bounds of either difference include zero. Parse failures are counted, never imputed.
6. Power rule. H2b and H2c are tested only with at least 20 residue flips in the run. With fewer, they are reported as descriptive with bounds and no claim.
7. Prompt note. Asking for p_residue may change the notes. All residue rates are reported for the p_residue prompt. The plain-note prompt from 7 is kept for development runs only.
8. Freeze. Hypotheses, budgets, systems, models, seeds and prompts go into `docs/flip-protocol-v1.md` with sha256 hashes of the prompts and the generator before phase 1c.

Protocol v2 (Varma, 2026-10-20) replaces item 6. Each hypothesis is tested only when its own analysis set is large enough. H2a needs 20 eligible cases. H2b needs 20 of its own eligible residue flips. H2c needs 20 residue flips and 20 non-flips among the eligible cases. Below its minimum a hypothesis is descriptive: n and bounds, no label. The v2 hashes go into `docs/flip-protocol-v2.md`. v1 stays as frozen.

summary.json `residue` block: n_eligible, residue_rate, floor_rate, trunc_rate, h2a_diff and bounds, h2b_recall, h2b_random_recall, h2b_diff and bounds, h2c_auroc, h2c_auroc_omission, both differences and bounds, n_residue_flips, p_residue_parse_errors.

### 7.1 Summary and embedding systems

- summary: the rendered history is cut into chunks of 40 events. summary_k+1 = summarize(summary_k, chunk_k, B). The prompt asks for a compact factual summary within B tokens and never mentions the query, because the summary is written before the query is known. Each step is cut to B tokens. Units are the sentences of the final text. The summariser is an injected callable. The default is qwen3:30b-a3b with think off and num_predict of 1.5 B. The model writes its reasoning into free text even with think off, so the reply is asked for as a one-field JSON schema ({"text": ...}) and only the text is kept.
- rag-embed: nomic-embed-text through POST /api/embed. The reply holds "embeddings" (a list of vectors). The code also reads a single "embedding". Each rendered line and the query text are embedded. Vectors are cached by sha256 of the model and text. Lines are ranked by cosine similarity, ties go to the later event. Lines are added until the budget is full. No task prefixes are added.
- Every text and embedding call goes through the sqlite call cache.

### 7.3 A non-verbatim omission comparator for H2c

Status: exploratory, not part of frozen protocol v3. Proposal only. Nothing here has run beyond the free local timing below.

Any result from it is exploratory. It does not replace the frozen H2c test. The frozen verdict stays as computed.

Problem. The frozen omission baseline is the units of V not contained verbatim in N, over the units of V. Notes never quote units verbatim.

In the merged v3 report the baseline is 1.0 on 487 of the 490 eligible notes. The other three notes score 0.95, 0.95 and 0.96. The pooled AUROC is 0.49. The comparator is almost constant. "Beats the omission fraction" is then a weak bar.

Quantity. For each pair of a note N and a unit u of V, conveyed(u, N) is 1 or 0. The comparator is omission_nv = 1 minus the mean of conveyed over the units of V. It enters H2c as a second baseline, next to the frozen one.

Pairs. The 490 eligible notes give 11,240 (note, unit) pairs. That is 22.9 units per note.

| System | Eligible notes | Pairs |
| --- | --- | --- |
| recent | 98 | 2,771 |
| rag-bm25 | 163 | 4,397 |
| rag-embed | 144 | 2,494 |
| summary | 85 | 1,578 |

(a) Embedding coverage.

- Score: for each unit of V, the maximum cosine similarity between its embedding and the embeddings of the note's sentences.
- Model: local nomic-embed-text through POST /api/embed, as in the rag-embed system.
- Sentences: from the shared sentence splitter.
- Rule: a unit counts as conveyed when the maximum is at least a threshold τ.
- Size: 12,372 distinct texts (units and note sentences). Calls are cached.

Calibrating τ needs a hand-check set. The builder cannot label it. The plan is:

1. Sample: draw 200 (note, unit) pairs with a fixed seed. Take 50 per system.
2. Strata: within a system, split the pairs into 5 strata by the quantile of the maximum cosine. Take 10 per stratum. The borderline range is then well covered.
3. Labeler: Varma, or a judge he names, labels each pair blind. The labeler sees the note and the unit. The labeler does not see the cosine or the system.
4. Question: does the note convey this record's content? Answers are yes, partly or no. "Partly" counts as not conveyed. The report also shows the other reading.
5. Second labeler: a model from another family labels the same 200 pairs. The report gives the agreement and Cohen's kappa.
6. Split: halve the 200 labels into a tuning half and a check half.
7. Choice of τ: sweep τ over the observed maxima on the tuning half. Choose the τ that maximises sensitivity plus specificity minus 1. Weight each stratum by its share of all pairs.
8. Check: report accuracy, sensitivity and specificity at τ on the check half, with 95% bootstrap bounds.
9. Freeze: fix τ before any AUROC is computed on the 490 notes.

The risk is that unit lines are full of dates and amounts. Their embeddings may follow format more than content. The check half shows whether τ separates the labels at all.

(b) LLM judge. For each pair, ask: does the note convey this record's content? The reply is one field, {"conveyed": "YES" or "NO"}. The prompt scaffold is 166 characters. Two variants:

- b1: local qwen3:30b-a3b with think off and the reply format set to the schema. Free and slow.
- Without the schema, b1 writes its reasoning as free text. One probe spent 300 tokens in 2.8 s and gave no answer yet.
- b2: Haiku 5.5 through the Message Batches API with thinking disabled. 10 output tokens per judgement.

The same 200 hand labels check the judge. The report gives accuracy and kappa against them. There is no τ.

(c) Combination. Embedding first. A unit with a maximum at or above τ_hi is conveyed. A unit below τ_lo is not. A unit in between goes to the judge, b1 or b2.

τ_lo and τ_hi are chosen on the tuning half. The embedding label must be right on at least 95% of the pairs outside the band. The band's share of pairs is unknown until the hand-check exists. The table shows 10%, 25% and 50% as assumptions.

Cost and time from stored data. Tokens use the shared helper of 2.2 characters per token. Haiku Batch is $0.05 per million input tokens and $0.25 per million output tokens. Each judgement has 10 output tokens.

Local times were measured on 2026-10-21 on Ollama at localhost:11434. The embedding times come from 20 single-text calls after one warm-up call. The judge time comes from 5 calls on real pairs.

| Option | Pairs scored | Input tokens per pair | Haiku Batch cost | Local wall time |
| --- | --- | --- | --- | --- |
| (a) embedding | 11,240 | 49 (43 unit, 6 note share) | $0 | 142 s at 11.4 ms per call, or 113 s batched at 9.2 ms per text |
| (b1) local judge | 11,240 | 265 (76 scaffold, 43 unit, 147 note) | $0 | 2,255 s (37.6 min) at 0.201 s per call |
| (b2) Haiku judge | 11,240 | 265 | $0.177 | none |
| (c) 10% in band | 1,124 judged | 265 | $0.018 with b2 | 367 s with b1 |
| (c) 25% in band | 2,810 judged | 265 | $0.044 with b2 | 705 s with b1 |
| (c) 50% in band | 5,620 judged | 265 | $0.089 with b2 | 1,269 s with b1 |

Notes on the table:

- Option (b2): $0.149 for 2,984,177 input tokens and $0.028 for 112,400 output tokens.
- Option (c): it embeds first, so its local time includes the 142 s of embedding. Its input tokens per pair apply to the judged pairs only.
- Token check: the five judge calls reported 180 to 254 prompt tokens in Ollama's own tokenizer. The 2.2 estimate averages 265 per pair. The tokenizers differ. Treat the figure as an estimate.
- Judge quality is not known. The five timing answers are no evidence of quality. The hand-check supplies it.

Plan if Varma approves:

- Varma labels the 200 pairs or names the judge.
- The builder computes τ and runs (a) and (c) locally.
- The builder reports H2c with the new comparator as exploratory, beside the frozen result.
- Nothing is submitted or spent without Varma's go.

Implemented 2026-10-21 as `lens flip omission` (src/lens/flip/omission/, results in results/omission-explore-v1/, schema `lens.flip.omission/1`, exploratory, never touches summary_v3.json). Varma's decision replaced the plan above: he hand-labels 200 pairs (50 per system, strata likely verbatim, likely paraphrased and likely absent from local signals, seed 21, drawn from small clusters of notes; D* units are marked in the key only and capped at 8 per system because the frame holds 200 of them) in one vault file. `sample` writes that file and the key, `parse-labels` reads the filled file into labels.jsonl, and `run` embeds all 11,240 pairs and calibrates on the labels with "?" excluded. Mode `embedding` (default) picks the one tau that maximises Cohen's kappa for "cosine at or above tau means conveyed", reports agreement, precision, recall and kappa at that tau, and gives the two 90%-precision thresholds as context. It then computes each note's omission fraction and the H2c AUROC of p_residue and of the omission measure itself, pooled and per system, on the same 490 notes and bootstrap as the v3 summary (it reproduces the v3 p_residue AUROC as a check). Mode `embedding+judge` is built and tested with a fake judge and is not run: borderline pairs go to qwen3:30b-a3b with a {"conveyed": bool} schema only if it first reaches kappa 0.6 and accuracy 80% on the labelled borderline pairs. Otherwise the run stops and writes a priced Haiku Batch plan to judge_calibration.json. The plan makes no API call and submits nothing. The embedding-only result goes to summary_omission_v1.json once the labels exist.

## 8. Class labels

| Class | How it is assigned |
| --- | --- |
| Unobserved | By construction (U1). D* is absent from the history. |
| Unresolved | By construction (U2). Two mechanisms fit. |
| Unretained | D* is in H, absent from V and the decision flipped. |
| Unexpressed | D* is in V, absent from N and B's decision flipped. |

"Absent from V" means no D* event id appears in V and a yes-or-no judge finds none of its content. The judge is checked by hand on 100 cases.

## 9. Metrics

- False Closure Rate at budget B: committed decisions with a wrong action, over all committed decisions. Reported on rare cases and reweighted to 1%. summary.json carries two versions. `fcr_raw` counts every committed a_V. `fcr_gated` counts only cases where a_H equals a*, so a wrong full-history decision is not blamed on the memory view. Gated is the headline.
- Harmful flip rate: a_H is right and a_V is wrong. summary.json gives the count, the rate over all cases and the rate over gated cases.
  Protocol v2 splits it into three parts that sum to it: wrong commit (a_V is a committing action), abstention (a_V is REQUEST_INFO and a* is not) and parse error. `summary_v2.json` and `tables_v2.txt` carry the parts for every cell. The harmful rate itself is unchanged.
- Right restraint: REQUEST_INFO rate on U1 and U2.
- Over-caution: REQUEST_INFO rate on M0.
- Flip search quality: precision, recall and mean calls.
- Residue rate: B wrong given A right.
- Cost and latency per case.

All rates carry 95% bootstrap bounds over cases.

Gates, fixed before any run:

- Difficulty gate: the best baseline at B = 1000 must miss at least 20% of rare cases. If not, raise length or distance and re-run.
- Product gate: on public long-dialogue sets (phase 3), at least 2% of answers must flip between the memory view and full context. Below that, drop the product track. The 99/1 synthetic set cannot test this gate. Its rare share caps the rate near 1% by design.

## 10. Phases

| Phase | Scope | Spend |
| --- | --- | --- |
| 1a | Generator, checks, ddmin with a mock decider, three simple systems, local model, 200 cases | none |
| 1b | Summary and embedding systems, residue stage, 1,000 cases, local model | none |
| 1c | Headline run on Haiku and a Sonnet subset through Batch | capped, approved first |
| 2 | Mem0, Letta and LangMem adapters | small, approved first |
| 3 | Public long-dialogue sets after a licence check, product gate | capped, approved first |

Before 1c, a dry run counts tokens and prices the run. Varma approves a cap. The runner stops at the cap.

Protocol v3 (2026-10-21) reruns U2 only, in `results/headline-u2-v3` (30 cases, seed 12, ids `u3-NNNN`). A validity pre-check on the same 30 cases (`results/validity-u2-v3`, adaptive-high) runs first. The rerun starts only if U2 scores at least 24 of 30. `lens flip headline merge-report` then replaces the 30 old U2 cases of headline-300 with the new ones and reports the combined 300 in `summary_v3.json`. The protocol is `docs/flip-protocol-v3.md`.

## 11. Code layout

```
src/lens/flip/
  schema.py      Event, Case, View, Unit, DistinctionRecord
  generator.py   mechanisms, distractors, knobs, rule oracle
  render.py      event wording templates
  systems/       base, full, recent, summary, rag_bm25, rag_embed
  decider.py     prompt, JSON parsing, backends (ollama, anthropic-api)
  cache.py       sqlite call cache
  ddmin.py       delta debugging over event sets
  residue.py     handoff stage
  llm.py         cached Ollama text and embedding calls
  metrics.py     validity gate, raw and gated FCR (bootstrap bounds still to come)
  runner.py      run config to results/<run_id>/
  batch.py       Batch validity check (section 5.1)
  pddmin.py      round-parallel ddmin and group-first search
  headline/      Batch-round driver for the headline run (section 5.2)
  cli.py         lens flip gen | run | report
tests/flip/      generator checks, ddmin on a known set, metrics on a toy table
```

Standard library first. Dependencies are added with `uv add` only when needed (for example a BM25 package). Results go to `results/`, which git ignores.

## 12. Build order

| Step | Work | Tier |
| --- | --- | --- |
| 1 | schema, generator, rule oracle and the five generator checks | Sonnet |
| 2 | ddmin and its tests on a mock decider | Sonnet |
| 3 | full, recent and rag-bm25, plus the Ollama decider | Sonnet |
| 4 | Phase 1a run and report | Haiku |
| 5 | summary, rag-embed and the residue stage (built 2026-10-20) | Sonnet |
| 6 | Phase 1b run and report | Haiku |
| 7 | Review of design and first results | Opus (senior-reviewer) |
| 8 | Dry-run pricing, then phase 1c after approval | Haiku |

A step that fails twice moves up one tier.

## 13. Open questions for Varma

1. The spend cap for phase 1c.
2. Whether named third-party systems appear in the December release.
3. The local development model: qwen3:30b-a3b or qwen3:8b.

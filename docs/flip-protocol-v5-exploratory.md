# Decision Flip Test: protocol v5 (exploratory)

Status: FINAL since 2026-10-30 01:14 UTC. Varma confirmed the dry-run price on 2026-10-29 ("I approve $30" in chat; approval a42). The run started the same evening with a $30 cap; dry run likely $2.19, cap-limited upper $24.49.
Version: 5-exploratory
Written: 2026-10-29, branch `exploratory-v5` from `exploratory-v4` at db3a76c.
Approved: Varma, 2026-10-29, item ft-17: exploratory v5 arms for the journal version, on Haiku 5.5 through the Batch API, outside the frozen protocols v3 and v4. Price first. The run waits for his yes to the price.

## 0. Status of this protocol

This protocol is exploratory. It sits outside the frozen v3 protocol (`docs/flip-protocol-v3.md`) and the exploratory v4 protocol (`docs/flip-protocol-v4-exploratory.md`). Neither file is edited.

- No result here changes a v3 or v4 verdict or number.
- `results/headline-300`, `results/headline-u2-v3`, `results/headline-300-v3` and `results/exploratory-v4` are read only.
- Results are versioned apart in `results/exploratory-v5/`, schema `lens.flip.exploratory/1`.
- Every v5 claim carries the label "exploratory".
- The arms answer the open points of J. Raynal's review: the four missing controls, and the writer's weak self-estimate.
- The protocol check refuses to run while the status is DRAFT. `finalize` sets FINAL, and only on Varma's yes to the price.

## 1. Frame: the same 490 handoffs

- The 490 eligible handoffs of the v3 merge at B = 1,000, as in v4. No case is regenerated.
- Reused code: `explore4.frame` (`load_frame`, `load_cases`). Reused data: `results/exploratory-v4/frame.jsonl`, copied after its sha256 is checked.
- `frame.jsonl` sha256 `bed6b1d00288354439f4a0459dc7d512cfc83245b369a141e9e4f1a88f6eb69f`.
- `cases.jsonl` sha256 `432e55ca579b74766e9097428fe513bd5a8c96496892dec1b023365ba1a4ed3c`.
- By system: rag-bm25 163, rag-embed 144, recent 98, summary 85. v3 residue flips: 142, of which 41 are restored by insertion (the v3 recoverable count) and 101 are not.
- The v3 minimal restoring sets come from the `residue_records.jsonl` of the two source runs. They are written to `v3_min_sets.jsonl` (sha256 `434b26b8589ae65262445cfb3c7c79bc16f3179f7ada054b7fdc61ea2f713996`) and checked against the frame row of each flip.
- Bootstrap: the v3 bootstrap. 1,000 resamples of the 300 cases, seed 11. Every handoff of a resampled case comes along. Every comparison with a v3 quantity is paired by case and handoff.
- The decider, the note-writer and the flip search are exactly as in v3 and v4 unless an arm says otherwise.

## 2. The six arms

All requests go to claude-haiku-5-5 through the Batch API. Notes are cut at 150 tokens with the shared helper (ceil(chars / 2.2)).

### X1: reader-conditioned estimate

- A writes a new note at 150 tokens with the v3 note prompt plus one paragraph. The paragraph tells A who the reader is: a colleague who sees only the note and the request, and chooses among the five actions with the v1 action policy (quoted in the paragraph). It also asks for p_residue as that colleague would see it.
- Removing the paragraph gives the v3 note prompt back byte for byte (test).
- A returns {note, p_residue} in the v3 note schema. B decides from the note with the v1 prompt.
- No flip search.
- Metrics: residue rate; the difference to the v3 residue rate (paired); the AUROC of p_residue for X1's residue flips, against 0.5 and against the v3 p_residue AUROC (paired difference of the two AUROCs).

### X2: structured note at equal tokens

- A writes the note in five fields: leading explanation, live alternatives, missing evidence, next action, reversibility. The fields and their labels together must fit in the same 150 tokens.
- A returns the five fields and p_residue. The fields are rendered as five labelled lines, in that order, joined by newlines. An empty field reads "none". The rendered text is cut at 150 tokens by the shared helper, as the v3 free-text note is. The cut and the number of complete replies are reported.
- B decides from the rendered note.
- On X2 flips the v3 flip search runs: group-first, 48 calls, v2 verify, candidates = the units of V the note does not contain verbatim.
- Metrics: residue rate paired against v3 free text; AUROC of p_residue (against 0.5, and against v3 as in X1); recoverable rate (search status restored, cap hit included, as in v3), against the v3 recoverable rate (41 of 490) and the v3 rewording floor.
- The search is priced apart from the notes in section 7.

### C1: irrelevant insertion of equal length

- For each of the 142 v3 residue flips, K = 5 control sets. Each set holds units of V drawn at random without replacement from the units of V that are not in the v3 minimal restoring set. For a non-recoverable flip the pool is all of V's candidate units.
- The rendered length of a set is len of the units joined by newlines. It must fall within 10 percent of the minimal set's length. For a non-recoverable flip the target is the median minimal-set length of the recoverable flips.
- Seeds: `random.Random("x5-c1|<case>|<system>|<k>")`. The draw rule is in `explore5/controls.py` and its hash is below. In short: shuffle the pool, add units that keep the length at or below 1.1 times the target, stop at 0.9 times, repeat the shuffle up to 200 times, keep the first set inside the band. A copy of an earlier control of the same flip counts as a failed attempt.
- The controls are shown to B the way the v3 search shows inserted units: the v3 note as the view, the units under "Additional records". So targeted and irrelevant insertion differ in content only. B decides each note with the v1 prompt.
- A draw that cannot reach the band is kept and flagged. A set that is empty or a copy is not sent. On the frozen data: 710 draws, 626 sent, 85 draws miss the band, 84 are not sent. The draws are frozen in `c1_controls.jsonl` (sha256 `8a047e7ff8151436556073a64fc5f2955aa908ba85eefd86613bee79a09dcd1d`).
- Metric: the restore rate under irrelevant insertion (per flip, the share of its matched controls that return B to A's action) against targeted reinsertion (1 when the v3 search restored the flip), paired by flip. The primary comparison uses flips with at least one matched control. A sensitivity check uses every sent control.

### C2: content-preserving rewrite

- A rewriter (Haiku 5.5, thinking disabled, effort medium, JSON {"note": ...}) rewrites each of the 490 v3 notes. It keeps every fact and the advice and changes the wording only, within 150 tokens. It sees the note and nothing else.
- B decides on the rewrite with the v1 prompt.
- Metrics: the flip rate of the rewrite against A's action, and its paired difference to the v3 residue rate. This is the wording residue. Also reported: rewrites identical to the note, and rewrites that lose a number or date of the note.
- A rewrite that fails to parse is not decided and is counted.

### C3: repeated identical requests

- B decides on the identical v3 note R = 5 more times per handoff: 490 x 5 requests with distinct custom_ids.
- The call cache is bypassed for this arm only. Mechanism: the key of each repeat is the sha256 of the canonical request body plus the repeat tag, so the five identical bodies are five requests. The body sent to the API is unchanged. `config.json` says `cache_bypass: true`, `cache_bypass.json` records the scheme, and `summary_v5.json` carries the flag.
- Metrics: reader self-disagreement on the identical note (repeats against the v3 answer, and pairwise among the six answers); the repeat flip rate against the v3 residue rate; the share of the 142 v3 residue flips that a repeat reverses (at least one repeat returns to A's action, and the share of repeats that do).

### C4: semantic absence

- For every (note, unit of V) pair of the 490 v3 notes, a judge (Haiku 5.5, thinking disabled, effort medium) answers JSON {"conveyed": bool}: does the note convey the unit's content. The prompt is the one in `src/lens/flip/omission/judge.py`, unchanged. The reply schema adds additionalProperties false.
- Pairs: 11,240 (the sum of the view sizes).
- Metrics: the semantic omission fraction per note (units judged not conveyed over units judged); the H2c AUROC of the v3 p_residue and of the semantic omission fraction for the v3 residue flips, with their paired difference; the semantic recall of D* in the note on residue flips (event views with D* in view), and on non-flips for comparison.
- Unparsed replies are counted and left out of the fraction.

## 3. Decomposition of the residue (exploratory)

The residue r_q is the v3 residue rate. The decomposition reads the four controls together:

    r_q = omission residue + wording residue + unattributed

- Omission residue: from targeted against irrelevant insertion (C1). The sum over assessed v3 flips of (targeted restore minus irrelevant restore), divided by the 490 handoffs. Assessed means the flip has at least one matched control.
- Wording residue: from the rewrite (C2) net of reader noise (C3). The mean over handoffs with a rewrite of (rewrite flip minus the share of the five repeats that flip).
- Unattributed: r_q minus the two terms. It holds the reader-noise floor and anything neither control separates. It is reported with its bounds and never dropped.
- C4 is the check on the omission term: if the note conveys D* semantically, the flip is not an omission in the semantic sense.
- Bounds: the paired bootstrap above. Each term is a share of the 490 eligible handoffs. The identity holds in every resample.
- This is a descriptive account. It does not claim the terms are additive causes.

## 4. Models and request settings

| Role | Model | Thinking | Effort | max_tokens | Reply |
| --- | --- | --- | --- | --- | --- |
| A, X1 and X2 notes | claude-haiku-5-5 | adaptive (field omitted) | high | 6000 | note schema (X1), five fields (X2) |
| B, every arm | claude-haiku-5-5 | adaptive (field omitted) | high | 6000 | v3 action schema |
| C2 rewriter | claude-haiku-5-5 | disabled | medium | 512 | {"note"} |
| C4 judge | claude-haiku-5-5 | disabled | medium | 64 | {"conveyed"} |

Batch rates, USD per million tokens: Haiku 0.05 input and 0.25 output. Requests go through the v3 headline engine: one key per request body (C3 excepted), dedup, rounds, one retry of errored requests, the anthropic-api wrapper.

## 5. Runner, guards and order

- `uv run python -m lens.flip.explore5 <command>`: `frame`, `dryrun`, `freeze`, `finalize`, `init`, `run`, `report`.
- `run --arms X1,C2 --cap-usd N --price-confirmed` runs any subset, in the order C-arms first and X2 last (X1, C2, C3, C1, C4, X2). State is resumable: a rerun picks up each arm's saved state and call cache.
- `run` and `init` refuse without `--price-confirmed`. `run` refuses without `--cap-usd`. The engine refuses while the protocol is DRAFT or any hash differs.
- Spend guard, as in v4. Before each round, if the ledger total plus the round's likely cost passes the cap, the round is not submitted and the run stops. The round's upper bound (max_tokens on every request) must also fit inside the cap minus the other arms' spend. A refusal stops the run. It is never worked around.
- The ledger is `results/exploratory-v5/spend.json`, updated after every batch.
- Order of work when run: X1, C2, C3, C1, C4, X2.

## 6. Outputs

`results/exploratory-v5/summary_v5.json` (schema lens.flip.exploratory/1). It carries every metric with n, bounds, the prompt hashes and the spend. Code: `src/lens/flip/explore5/`. Tests: `tests/flip/test_explore5.py`.

## 7. Price (dry run, nothing spent)

Computed by `dryrun` from stored prompt sizes and the real usage of the v3 and v4 runs. No model call, no count_tokens call, no batch. Batch rates above.

| Arm | Requests | Likely USD | Upper USD | Rounds (likely / upper) |
| --- | --- | --- | --- | --- |
| X1 | 980 | 0.31 | 1.55 | 2 / 2 |
| X2 | 2,469 | 0.72 | 16.78 | 16 / 27 |
| of X2: notes and B | 980 | 0.31 | 1.55 | 2 / 2 |
| of X2: flip search | 1,489 | 0.41 | 15.23 | 14 / 25 |
| C1 | 626 | 0.17 | 0.98 | 1 / 1 |
| C2 | 980 | 0.13 | 0.84 | 2 / 2 |
| C3 | 2,450 | 0.48 | 3.81 | 1 / 1 |
| C4 | 11,240 | 0.37 | 0.54 | 1 / 1 |
| Total | 18,745 | 2.19 | 24.49 | |

- Likely total: $2.19. Cap-limited upper bound: $24.49. With every output 50 percent longer than assumed, the likely total is $2.90.
- The likely total is under $10.
- Upper bound: every request at max_tokens and every X2 flip search at 48 calls plus one verify, each call carrying the note and all candidate units. It counts 193 searches (39.4% of 490, the highest flip rate seen on any v4 arm). If all 490 handoffs flipped, the X2 search alone would reach $38.67. The per-round guard bounds that.
- Rounds: one per dependency step. X1 and C2 need two (note, then decision), C3, C1 and C4 need one. X2 needs two plus the search: the stored v3 and v4 searches took up to 13 rounds, plus one verify round. Add one retry round to any arm if a request errors.

Method.

- Input tokens per request: likely ceil(chars / 2.2) plus 331 (the output-format overhead seen on every v3 and v4 request). Upper bound ceil(chars / 1.902) plus 331. Against the real input tokens of the nine v4 stages with stored batches, the likely rule lands between -2 and +16 percent of the actual mean, and the upper bound covers every request except 6 of 5325 (`calibration_check` in `dryrun_v5.json`).
- Output tokens per request, likely: note writers 1,373 (v4 c_A300; v3 residue_A was 1,214), B on a note 594 (v4 x_B), B with inserted units 882 (v4 residue_flip), verify 729 (v4), rewriter 175 (thinking off, a JSON note of at most 150 tokens), judge 10 (the omission judge's own figure). Upper bound: max_tokens.
- Flip search: likely 32.0% of handoffs flip (mean of v3 142, v4 c300 162 and c600 167 of 490), 8.93 calls per search and 0.55 verifies per search (pooled over the 142 v3 flips and the 193 v4 extractive searches), at the observed v4 input and output tokens per call.
- The B requests of X1, X2 and C2 use the v3 note of the handoff as the stand-in for the new note's length. C1, C3 and C4 requests are priced from their exact prompts.

## 8. Hashes (sha256)

Rendered with sentinel inputs, as in v3 and v4. `x5.*` items are new. The v3 rows are reused unchanged and equal the v3 hashes.

| Item | sha256 |
| --- | --- |
| x5.c1.rule | `71879188097dc3d218375c3533fc35da007411dc4d58cd548c6a79ec707dafe7` |
| x5.c2.prompt | `bc44bd9316b1dfe0581d4f74262e99fbaae763c748a8c9eaf0b9b19742d91bfc` |
| x5.c2.schema | `30864c84deaaebb6f499ed43d9b222243ca1ebbc6a13bfbdf5c6dcc378ddfbe7` |
| x5.c3.rule | `062017a0205a2e97ac3dfe923ee0395684ece95dd08b998c111f0859f9ec5358` |
| x5.c4.prompt | `408d3c8182d83d619c8c103edc75e7a0163a3d5024abce0184eff5a4c246ca23` |
| x5.c4.prompt_source | `de77686c779f79a72720f64bbc0aeaa7f7fe1a34c8843c13f0bbd76ecd6a4838` |
| x5.c4.schema | `e04725a549cead4dbe5a210256463f461f969ebcebd05c1f760bbd0a63366ec5` |
| x5.x1.prompt | `0f922c5080f2b2e969d5e2decc2c0d2378c90c704ba43dd0e05facc4703cad16` |
| x5.x1.reader_paragraph | `f631e18161dac71f3a47291fa706403c4252af7aa024ade524271c3f18d3e5e7` |
| x5.x1.schema | `23b194ce2295f649c6a2a17cdaedf6bccbc5ae05eda412d1d6a65c1af2ae88d0` |
| x5.x2.prompt | `c540c232ed2c93791f44e9cab8795232c91feb3a8352f5ac2e3f748fe55bc113` |
| x5.x2.render | `e40f026dff8328cec8da32ab93f74899b87885fe7b144dce2029143a801eff33` |
| x5.x2.schema | `530c6bf414bc6259522fe144b3903a0873de5d577f2e711fcd2f84328876611a` |
| decider.action_schema | `b5c809762c595781e481572d7b6ead5fb9ef9a4afa1e2afcae093ca81a69efbb` |
| decider.v1 | `f0828f379ef244779cec7c9e8fd2260b23b53f9da81cfd35bf0a8c4dd3673f04` |
| decider.v1.inserted | `74be01664a51887455a9445a08edbf8eae8c1f75b9a34db9d924a1b9d9b5a8d0` |
| decider.v2 | `afa9c09a90ae59bfc1f92e858bc5d9d34125298c17c0bd74c1bfa70c9b9d718b` |
| event_templates | `38f5302ef4a810a19679a838828c909cc0e1c753fd2676f6df5776c5af3ac919` |
| file:render.py | `46567cebd52980297247f58f1c46ad315eac94ff3736427fe092bcb85dc6dfaf` |
| note.prompt_p_residue | `d06b8aafacbcc2e16280c4e6ebd5069a2e5ec8baeb8e23a2e0829d435a08b6cd` |
| note.schema | `23b194ce2295f649c6a2a17cdaedf6bccbc5ae05eda412d1d6a65c1af2ae88d0` |

Arm request settings, as `config.json` minus the assumed output tokens:

| Arm | sha256 |
| --- | --- |
| C1 | `56f4d1148346c2156ebbd392327c9e3d2f16c5b9a3a3680f3ef0065594789df7` |
| C2 | `57cb6cf4d8c79a0224271ad10f69fc6055d0b443b0138bd67e35e319e3058c2f` |
| C3 | `f8b4bdd70959e663478b22c4bfe2933737d4b17c17b597ee44d4f56d71843389` |
| C4 | `921629c56fbaeb854a40588e23f3e77eb7bd704e194351a6e3bf02fa69be5edb` |
| X1 | `5cb9340bbd11f3345ea0ce6ac3c11841f91142d2b9159248fed4f598693baef0` |
| X2 | `b3f297960352ebe64f3d081f7429823ea25da1b3e6c06da5d323ffc82ff22c68` |

Run files: frame `bed6b1d00288354439f4a0459dc7d512cfc83245b369a141e9e4f1a88f6eb69f`, cases `432e55ca579b74766e9097428fe513bd5a8c96496892dec1b023365ba1a4ed3c`, v3 minimal sets `434b26b8589ae65262445cfb3c7c79bc16f3179f7ada054b7fdc61ea2f713996`, C1 draws `8a047e7ff8151436556073a64fc5f2955aa908ba85eefd86613bee79a09dcd1d`, input-token rule `7dfa41aa22ab0772d3fc9e8b428f388ffa85f7e18686f9c958f14ec1f0727225`.

`results/exploratory-v5/protocol_v5.json` holds the same hashes with status DRAFT. The engine checks them before every step.

## 9. Known limits and readings

- One sample per request, as in v3. Haiku with adaptive thinking is not deterministic. C3 measures that noise for the v1 prompt on the v3 notes.
- C1 shows the units to B in the "Additional records" block, as the v3 search does, and does not paste them into the note text. This keeps targeted and irrelevant insertion on one footing.
- C1 cannot always reach the 10 percent band. A short minimal set (as short as 33 characters) or a view with few, long sentences leaves no usable draw. Those flips drop out of the primary C1 comparison and are counted. The summary view is where most of them sit.
- Targeted reinsertion in v3 is a search that stops at 48 calls. Its restore rate is the v3 recoverable share (41 of 142), so "restored" carries the search's own selection. The irrelevant rate is a single fixed draw per control.
- The decomposition terms are shares of the same denominator. They are not independent effects and the unattributed term can be negative.
- The rewriter and the judge are the same model family as the writer and the reader. The judge's prompt was written for a small local model and is reused as it is.
- The frozen v3 notes are model text and are used as they are. They sit in `results/`, which is git-ignored, and carry vocabulary from the private word list. The v5 code, prompts and this document do not (test).
- No correction for multiple comparisons. Per-system results are descriptive.
- Prices are estimates. The ledger holds the actual spend after any run.

# Decision Flip Test: protocol v7 (exploratory)

Status: FINAL since 2026-11-19 09:13 UTC. Frozen before any model call. `results/exploratory-v7/protocol_v7.json` sha256 `1290eb9d0b19167ede155b31408f4ccdb7adf76e0ad96d4a5790d433144e5706`.
Version: 7-exploratory
Written: 2026-11-19, branch `exploratory-v5`, on top of 5e58966.
Approved: Varma, 2026-11-19 00:00, question 68: "Run both (about 1.50 dollars, adds about one day)". Track cap $5.00 of actual spend, v6 included. v6 spent $0.432525, so the run cap is $4.56. Price gate $2.00 on the dry-run likely price.

## 0. Status of this protocol

This protocol is exploratory. It sits outside the frozen v3 protocol and the exploratory v4, v5 and v6 protocols. None of those files is edited.

- No result here changes an earlier number.
- The result folders of v3 to v6 are read only. Their call caches are opened read only.
- Results go to `results/exploratory-v7/`, schema `lens.flip.exploratory/1`.
- Every v7 claim carries the label "exploratory".
- The protocol check refuses to run while the status is DRAFT or any hash differs.
- No flip search in any arm.

Two studies.

- Study R: why the reader holds back. 105 of the 142 v3 flips are deferrals (B chose REQUEST_INFO). The paper reads this as the reader holding back on a thin note. That reading is untested.
- Study X: the reader-conditioned writer of v5 (X1) cut flips from 29.0 to 14.5 in 100 where A was right. Its paragraph does two things at once, and it was never scored where A was wrong.

## 1. Frame

All sets are at B = 1,000, from the v3 merge (`headline.merge.merge_data`), 300 cases, four systems.

| Set | Rule | Handoffs |
| --- | --- | --- |
| gated | the full-history action is right: a_H equals a_star | 1,160 |
| eligible | gated, and A's action from the view is right: a_V equals a_H | 490 |
| A-wrong | gated, and A's action from the view is wrong: a_V differs from a_H | 670 |

- The eligible set is the frame of v3 to v6. `frame.jsonl` sha256 `bed6b1d00288354439f4a0459dc7d512cfc83245b369a141e9e4f1a88f6eb69f`, `cases.jsonl` sha256 `432e55ca579b74766e9097428fe513bd5a8c96496892dec1b023365ba1a4ed3c`.
- A-wrong by system: rag-bm25 127, rag-embed 146, recent 192, summary 205. A's wrong action: REQUEST_INFO 505, OPEN_REVIEW 103, EXPLAIN 35, APPLY_CREDIT 14, CHECK_DEVICE 13. The right action: EXPLAIN 374, CHECK_DEVICE 87, APPLY_CREDIT 82, OPEN_REVIEW 72, REQUEST_INFO 55.
- The frozen run wrote no note for an A-wrong handoff. v7 writes them now. Each A-wrong row carries the stored view of its cell. `wrong_frame.jsonl` sha256 `0301a0f4a5535c1566a951f98952606ffb68b954f1b36718d56bb9f6999bdce9`.
- Check at staging: on the 490 eligible handoffs, the v3 note request built here is in the v3 call caches (490 of 490) and the X1 note request built here is in the v5 X1 call cache (490 of 490). So the two writer requests of XW are the frozen ones, byte for byte.
- Frozen references, action names only (`v5_refs.jsonl`): B's action on the X1 note of v5 (71 flips of 490) and B's five repeat actions of v5 arm C3.
- Bootstrap: the v3 one. 1,000 resamples of the 300 cases, seed 11. Every handoff of a resampled case comes along. Bounds are 95 percent.

## 2. Study R: why the reader holds back

Reader B is unchanged (section 5). Notes are the frozen v3 notes of the 490 eligible handoffs.

### R0: the cap and the flips (no model call, descriptive)

A note is "cut" when its reply was longer than the stored note. It "ends mid-sentence" when its last character is not a full stop, question mark or exclamation mark. 349 notes are cut. 331 end mid-sentence, all of them cut.

v3 rates in 100 handoffs, with bounds. Counts in round brackets.

| Group | n | Flip | Deferral flip | Commit flip |
| --- | --- | --- | --- | --- |
| all | 490 | 29.0 (142) | 21.4 (105) | 7.3 (36) |
| cut | 349 | 28.7 [23.8, 33.1] (100) | 21.5 [17.1, 26.2] (75) | 6.9 [4.4, 9.6] (24) |
| not cut | 141 | 29.8 [22.7, 37.6] (42) | 21.3 [14.8, 28.6] (30) | 8.5 [3.9, 13.8] (12) |
| cut minus not cut | | minus 1.1 [minus 9.7, 7.2] | 0.2 [minus 8.0, 7.8] | minus 1.6 [minus 7.6, 3.6] |
| ends mid-sentence | 331 | 27.8 [23.0, 32.5] (92) | 20.5 [16.0, 25.3] (68) | 6.9 [4.3, 9.8] (23) |
| ends a sentence | 159 | 31.4 [24.4, 38.7] (50) | 23.3 [16.6, 30.2] (37) | 8.2 [3.9, 13.2] (13) |
| mid-sentence minus ends a sentence | | minus 3.7 [minus 11.7, 4.5] | minus 2.7 [minus 10.7, 4.9] | minus 1.2 [minus 6.9, 3.7] |

One v3 flip is an unparsed reply. It is neither a deferral nor a commit. The differences are differences of two rates under the same case resamples. All bounds include zero. This is context, not a test.

### R1: note-framed reader

The v1 reader prompt, changed in one respect: B is told the text is a colleague's handoff note and not the records themselves. Two edits carry that one change.

| Part | v1 | R1 |
| --- | --- | --- |
| system line | "You are reviewing an account's records to choose one action." | "You are reviewing a colleague's handoff note about an account to choose one action. The note is not the records themselves." |
| block label | "Records:" | "Handoff note:" |
| context line, request, action policy, reply instruction, reply schema, request settings | | unchanged, byte for byte |

The R1 user message, with sentinels for the note and the request:

```
Context: each reading is the activity figure for its own period. A reset reading after a device change restarts the count. Statements list activity charges, other charges and credits.

Handoff note:
{NOTE}

{QUERY}

Action policy:
- EXPLAIN: the higher amount is explained by normal seasonal activity.
- OPEN_REVIEW: the statement rests on a figure that later records contradict.
- CHECK_DEVICE: a device change during the period may have distorted the figures.
- APPLY_CREDIT: an approved credit was never applied.
- REQUEST_INFO: the records do not settle the cause, or two or more different problems are present for the period, because each would call for a different action and a reviewer must decide first. Normal seasonal activity is never a problem or a cause for this rule.

Reply only with JSON: {"action": one of the five names, "reason": one sentence, "event_ids": ids you relied on}.
```

Replacing the label with "Records:" gives the v1 user message back (test). 490 calls.

### R2: hedges stripped

A fixed text rule, in the manner of the v4 advice strip. No model call. Sentences come from the shared splitter (`split_sentences`). A sentence is dropped when any rule fires. Kept sentences are joined by one space. An unchanged note stays byte-identical.

| Rule | Markers (whole words) |
| --- | --- |
| H1 | may (lower case only, so the month name is not a marker), might, could, possibly, possible, perhaps, probably, likely, unlikely, presumably, potential, potentially, apparently, seemingly |
| H2 | appear to, appears to, seem, seems, seemed, suggests |
| H3 | unclear, uncertain, uncertainty, unknown, unconfirmed, unverified, unresolved, inconclusive, ambiguous, not clear, not certain, not confirmed, not verified, not known, not sure, cannot, can not, can't, caveat, caveats |
| H4 | if, unless, whether, pending |

Left out on purpose: about, roughly, approximately, the tilde, estimate, only, but, confirm, verify, missing, no record. These mark figures, contrast, advice or stated absence, not doubt.

The marker list was chosen from the note text alone. No reader outcome was looked at.

On the frozen notes: 195 of 490 notes change, 0 become empty, 224 sentences are dropped. Notes by rule: H1 65, H2 8, H3 93, H4 60.

B reads the result with the v1 prompt. An unchanged note is not sent: its v3 decision stands. 195 calls.

### R3: trailing fragment removed

A fixed text rule. No model call.

- It applies only to a note that ends mid-sentence.
- A sentence-ending mark is a full stop, question mark or exclamation mark followed by white space. A decimal point is not one.
- The note keeps the text up to and including its last mark. The rest is dropped.
- A note with no mark would be empty. It is counted and not sent.
- Other notes are unchanged.

On the frozen notes: 331 notes change, 0 become empty. The dropped fragment names an action in 91 notes and names A's action in 58. 335 of the 490 notes name an action somewhere.

B reads the result with the v1 prompt. An unchanged note is not sent: its v3 decision stands. 40 notes give the same text under R2 and R3. Identical bodies share one request, so R3 sends 291 requests and reads 331 answers.

## 3. Study X: the reader-conditioned writer, split and two-sided

### The split

The frozen X1 paragraph of v5 has three sentences and the action policy.

| Piece | Text |
| --- | --- |
| S1 | "The colleague reads only your note and the request. They do not see the records." |
| S2 | "They choose one of EXPLAIN, OPEN_REVIEW, CHECK_DEVICE, APPLY_CREDIT or REQUEST_INFO with the policy below, and they apply it to your note as written." |
| S3 | "Judge p_residue for that colleague: the chance that, from your note alone, they choose a different action from the one you would choose from the records." |
| policy | the v1 action policy, five lines |

- X1p, policy only: S2 and the policy.
- X1r, reader only: S1 and S3.
- The pieces are cut from the frozen paragraph by slicing. No word is retyped. S1 + S2 + S3 + policy is the paragraph (test). Every word is in exactly one half (test).
- Each half sits where X1 puts the whole paragraph: between the reply instruction and the records. Removing the half gives the v3 note prompt back, byte for byte (test).
- What each arm asks for p_residue. X1p: as the v3 prompt asks ("that something you left out of the note would change the colleague's action"). X1r: as X1 asks, for that reader (S3). S3 cannot go to X1p, because it says the reader works from the note alone.
- The reply schema and the writer settings are those of v3 and X1.

The X1p prompt:

```
You are handing an account case to a colleague who will choose the action. Read the records and the request, then write a short handoff note for the colleague.
Write at most 150 tokens (about 330 characters) of plain text. Include what the colleague needs to choose well.
Also give p_residue: your probability, from 0 to 1, that something you left out of the note would change the colleague's action.
Reply only with JSON: {"note": the note, "p_residue": a number from 0 to 1}.

They choose one of EXPLAIN, OPEN_REVIEW, CHECK_DEVICE, APPLY_CREDIT or REQUEST_INFO with the policy below, and they apply it to your note as written.
Action policy:
- EXPLAIN: the higher amount is explained by normal seasonal activity.
- OPEN_REVIEW: the statement rests on a figure that later records contradict.
- CHECK_DEVICE: a device change during the period may have distorted the figures.
- APPLY_CREDIT: an approved credit was never applied.
- REQUEST_INFO: the records do not settle the cause, or two or more different problems are present for the period, because each would call for a different action and a reviewer must decide first. Normal seasonal activity is never a problem or a cause for this rule.

Records:
{RECORDS}

Request: {QUERY}
```

The X1r prompt:

```
You are handing an account case to a colleague who will choose the action. Read the records and the request, then write a short handoff note for the colleague.
Write at most 150 tokens (about 330 characters) of plain text. Include what the colleague needs to choose well.
Also give p_residue: your probability, from 0 to 1, that something you left out of the note would change the colleague's action.
Reply only with JSON: {"note": the note, "p_residue": a number from 0 to 1}.

The colleague reads only your note and the request. They do not see the records. Judge p_residue for that colleague: the chance that, from your note alone, they choose a different action from the one you would choose from the records.

Records:
{RECORDS}

Request: {QUERY}
```

### The arms

| Arm | Handoffs | Writer | Reader | Calls |
| --- | --- | --- | --- | --- |
| X1p | 490 eligible | v3 note prompt plus the policy half | v1 on the note | 490 notes, 490 decisions |
| X1r | 490 eligible | v3 note prompt plus the reader half | v1 on the note | 490 notes, 490 decisions |
| XW | 670 A-wrong | the frozen v3 note prompt, and the frozen X1 prompt | v1 on each note | 1,340 notes, 1,340 decisions |

Notes are cut at 150 tokens by the shared helper, as in v3. A note is answered before the decision that reads it.

## 4. Pre-registered analysis

Fixed here and in `explore7/analysis.py` (`ANALYSIS_SPEC`, hashed as x7.analysis) before any model call.

Outcomes on an eligible handoff.

| Term | Definition |
| --- | --- |
| flip | B's action differs from A's action. On eligible handoffs A's action is the right action. |
| deferral flip | A flip where B's action is REQUEST_INFO. |
| commit flip | A flip where B's action is one of the four other actions. |
| unparsed | A reply that is refused, cut, unparsed or failed twice. It is a flip, and neither a deferral nor a commit. Counted. |

Study R, per arm against v3 on the same handoffs, paired.

1. The flip rate difference, the deferral-flip rate difference and the commit-flip rate difference.
2. Each of the three again on the notes the arm changed. For R1 that is every note.
3. The arm's flip rate and deferral-flip rate split by whether the cap cut the v3 note (349 cut, 141 not), beside the v3 rates and the differences. The cut is the candidate mechanism.
4. Noise floor. The share of arm answers that differ from the v3 answer on changed notes, beside the same share of the five C3 repeats (9.3 in 100 over all handoffs). Each arm rate minus the mean rate of the five C3 repeats, paired. C3 is the identical note read again, so it shows what reader noise alone does.
5. R3 only, descriptive: the differences split by whether the dropped fragment names an action.

A note that a rule empties is not sent, is counted and is left out of that arm's rows.

Reading rule for R, per arm. The arm "reduces deferral" only if the upper bound of the deferral-flip difference on the handoffs with an arm decision is below zero. Otherwise it "does not". A move smaller than the C3 floor is not read as an effect.

Study X.

1. On the 490 eligible handoffs: the flip rates of v3, X1 (frozen v5 numbers), X1p and X1r. Paired differences: each against v3, X1p minus X1r, X1p minus X1, X1r minus X1.
   Reading rule. The gain "comes with the policy" if X1p minus v3 has its upper bound below zero and X1r minus v3 does not. It "comes with the reader description" in the mirrored case. "Both" if both upper bounds are below zero. "Neither separable" if neither is.
2. On the 670 A-wrong handoffs: the share where B ends on the right action ("rescue") and the share where B repeats A's wrong action ("repeat"), under v3 notes and under X1 notes, with paired differences.
3. On all 1,160 gated handoffs: B's accuracy under v3 notes and under X1 notes, with the paired difference. This is the two-sided score. On the eligible side the decisions are the frozen v3 and v5 ones.

Descriptive only: note lengths, reply status counts, B's action counts on A-wrong handoffs, the two-sided difference by system and by set.

No correction for multiple comparisons.

## 5. Models and request settings

Read from the v5 arm config (`explore5.engine.arm_config`). Nothing is new.

| Role | Model | Thinking | Effort | max_tokens | Reply |
| --- | --- | --- | --- | --- | --- |
| A, every writer arm | claude-haiku-5-5 | adaptive (field omitted) | high | 6000 | v3 note schema |
| B, every arm | claude-haiku-5-5 | adaptive (field omitted) | high | 6000 | v3 action schema |

Batch rates, USD per million tokens: 0.05 input and 0.25 output. Every request is keyed by its body. Identical bodies share one request. No cache bypass.

## 6. Runner, guards and rounds

- `uv run python -m lens.flip.explore7 <command>`: `frame`, `dryrun`, `freeze`, `finalize`, `run`, `report`.
- Two run folders: R (arms R1, R2, R3) and X (arms X1p, X1r, XW). A stage name carries its arm, so spend per arm is read by stage.
- `run --arms R,X --cap-usd 4.56 --price-confirmed`. Order R, then X. The CLI refuses a cap above 4.56. State is resumable. No request is paid twice.
- Spend guard, as in v4 to v6. Before each round, the ledger total plus the round's likely cost must stay at or under the cap. The round's upper bound (max_tokens on every request) must also fit inside the cap minus the other study's spend. A refusal stops the run. It is never worked around.
- Round size. A round holds at most 1,000 requests, the first ones by request key. The rest wait for the next round. This keeps each round's upper bound inside the cap. It changes no request and no key.
- Rounds planned: R in one round. X in five or more: notes first, then decisions as their notes arrive.
- `finalize` refuses when the dry-run likely price is above $2.00 or when the guard preview does not fit.

## 7. Outputs

- `results/exploratory-v7/summary_v7.json`: every number with n and bounds, the hashes and the spend.
- `results/exploratory-v7/handoff_metrics_v7.jsonl`: one flat row per gated handoff (490 eligible, then 670 A-wrong). Ids, system, mechanism group, action names, counts and 0/1 flags only. No note, view or prompt text. The summary is computed from this file alone.
- Code: `src/lens/flip/explore7/`. Tests: `tests/flip/test_explore7.py`.

## 8. Price (dry run, nothing spent)

Computed by `dryrun` from request bodies and stored usage. No model call, no count_tokens call, no batch.

| Arm | Requests | Likely USD | Upper USD |
| --- | --- | --- | --- |
| R1 | 490 | 0.092 | 0.762 |
| R2 | 195 | 0.036 | 0.302 |
| R3 | 291 | 0.054 | 0.452 |
| X1p | 980 | 0.289 | 1.549 |
| X1r | 980 | 0.283 | 1.543 |
| XW | 2,680 | 0.780 | 4.229 |
| Total | 5,616 | 1.53 | 8.84 |

- Likely total $1.53. It is under the $2.00 gate. With every output 50 percent longer, $2.11.
- Track total, likely: $0.43 spent by v6 plus $1.53 = $1.97 of the $5.00 cap.
- Upper bound $8.84 (max_tokens on every request). It is above the cap, which is why rounds are limited in size.
- Guard preview at the $4.56 cap. The largest possible round bound is $1.61. All requests paid at the likely price with outputs 50 percent longer cost $2.11. The sum, $3.73, is under the cap, so every round fits.
- Exact bodies: every R request and every writer request of X. Stand-ins: a B request of X reads a note that does not exist yet. X1p and X1r use the v3 note of the handoff. XW uses the longest v3 note.
- Input tokens per request: the v5 rule. Likely ceil(chars / 2.2) plus 331. Upper bound ceil(chars / 1.902) plus 331.
- Output tokens per request, likely: 548 for B on a note (stored: 546 in v5 C3, 535 in v3), 1,214 for the v3 writer (stored v3 mean), 1,231 for the X1-type writers (stored v5 X1 mean). Upper bound: max_tokens.

## 9. Hashes (sha256)

| Item | sha256 |
| --- | --- |
| x7.analysis | `aea43ecf5a65a167ed53f64dd71fb6818336c52f89fa2d0bbc378ee7a4260156` |
| x7.r1.diff | `9830da0357050e179f8834755532945f940a6047569346584adac30c38de0db7` |
| x7.r1.prompt | `fe975e9e8528a9e6e0709b473e6a2fbc8ac477aaabad625a88fa3f7f2fd478fb` |
| x7.r1.system | `9271ff9efd58a1ffb0cfea1cb48097b5cc405bfaac92c321d00d55521c239517` |
| x7.r2.rule | `9365caf91075aa1ca7a6e23485409d2ddc584c568b55932e6287ec23cb89a86d` |
| x7.r3.rule | `b8f0ac63cd97e64220c8c7d4c3b06c24b17da73ac9e48fd278db5085a513cb63` |
| x7.rounds.rule | `44fc14a2fc1886e1076cc3f0f8b4dcdd67e37817b19fcdeb9ce5d11fbc92a178` |
| x7.sets.rule | `ef7a42b513c1e178bbe54a5bcedade58222245b0e6e0c15b84854c300d97238b` |
| x7.x1.schema | `23b194ce2295f649c6a2a17cdaedf6bccbc5ae05eda412d1d6a65c1af2ae88d0` |
| x7.x1.split | `567830ee36c6a12f4774dbe45d98a588cfb12d8561eaa672295a2a0134ff4c11` |
| x7.x1p.prompt | `e992f7f2a3c0ef6b624061479ee2d87688c0d93b0fbd20fe6d7dc05368329c8b` |
| x7.x1r.prompt | `7691cfea86bad16e3511150c5cdb972de51d2b2a9ebfa9cdd79b926c78c2d76d` |
| decider.action_schema | `b5c809762c595781e481572d7b6ead5fb9ef9a4afa1e2afcae093ca81a69efbb` |
| decider.v1 | `f0828f379ef244779cec7c9e8fd2260b23b53f9da81cfd35bf0a8c4dd3673f04` |
| decider.v1.inserted | `74be01664a51887455a9445a08edbf8eae8c1f75b9a34db9d924a1b9d9b5a8d0` |
| decider.v2 | `afa9c09a90ae59bfc1f92e858bc5d9d34125298c17c0bd74c1bfa70c9b9d718b` |
| event_templates | `38f5302ef4a810a19679a838828c909cc0e1c753fd2676f6df5776c5af3ac919` |
| file:render.py | `46567cebd52980297247f58f1c46ad315eac94ff3736427fe092bcb85dc6dfaf` |
| note.prompt_p_residue | `d06b8aafacbcc2e16280c4e6ebd5069a2e5ec8baeb8e23a2e0829d435a08b6cd` |
| note.schema | `23b194ce2295f649c6a2a17cdaedf6bccbc5ae05eda412d1d6a65c1af2ae88d0` |
| x5.x1.prompt | `0f922c5080f2b2e969d5e2decc2c0d2378c90c704ba43dd0e05facc4703cad16` |
| x5.x1.reader_paragraph | `f631e18161dac71f3a47291fa706403c4252af7aa024ade524271c3f18d3e5e7` |
| x5.x1.schema | `23b194ce2295f649c6a2a17cdaedf6bccbc5ae05eda412d1d6a65c1af2ae88d0` |

The prompts are rendered with sentinel inputs, as in v3 to v5. The decider, note and x5 rows are reused unchanged and equal the earlier hashes.

Request settings per run folder, as `config.json` minus the assumed output tokens:

| Run folder | sha256 |
| --- | --- |
| R | `52d4041d5d5fcd4107bdaa4c5ba812674b6061366cbd8b6bfb56cc39c961137a` |
| X | `cd64183a94d85a8fb99ab593acfded92576ad8c464cd21550c0c42a99c5fef1c` |

Run files:

| File | sha256 |
| --- | --- |
| frame.jsonl | `bed6b1d00288354439f4a0459dc7d512cfc83245b369a141e9e4f1a88f6eb69f` |
| cases.jsonl | `432e55ca579b74766e9097428fe513bd5a8c96496892dec1b023365ba1a4ed3c` |
| wrong_frame.jsonl | `0301a0f4a5535c1566a951f98952606ffb68b954f1b36718d56bb9f6999bdce9` |
| r_notes.jsonl | `d864ab8dc1ac0150163706be181a46fa23599f5696c4641ba4c2f0ad850fd479` |
| v5_refs.jsonl | `bc10945c852bbadf04b9367bb63d67bde2fa62e49b07e36bf21594ddf70b2c64` |
| calibration_v7.json (the v5 input-token rule) | `7dfa41aa22ab0772d3fc9e8b428f388ffa85f7e18686f9c958f14ec1f0727225` |
| protocol_v7.json (FINAL) | `1290eb9d0b19167ede155b31408f4ccdb7adf76e0ad96d4a5790d433144e5706` |

`results/` is git-ignored, so the last row is the record of the frozen file. The engine checks every hash before every step.

## 10. Known limits and readings

- One sample per request. The reader is not deterministic. C3 measured that noise: 9.3 in 100 repeat answers differ from the v3 answer. Every R difference is read against it.
- R1 changes two strings to carry one change. The policy text still says "the records do not settle the cause". It is kept as it is, so a reader may still read the policy as being about records.
- R2 is a word list. It drops a whole sentence for one marker, so it also drops what else the sentence says, advice included. It misses hedges that use no listed word.
- R3 drops the cut-off tail. The tail names an action in 91 of 331 notes. So R3 can remove advice together with the fragment. The split by that flag is reported.
- R2 and R3 keep the v3 decision for unchanged notes. A changed note gets a new sample and an unchanged note does not. The changed-only rows and the C3 comparison are there for that reason.
- R0 shows no gap between cut and uncut notes in v3. R3 is still run, because the gap is between notes, and R3 changes one note at a time.
- X1p keeps sentence S2 whole. S2 says the policy is applied to the note as written. That is a small part of the reader description, left in X1p because no word may change.
- X1r drops the action list along with the policy, since S2 carries both.
- The X1 rate on the eligible handoffs is the frozen v5 result, one sample. X1p and X1r are new samples.
- On A-wrong handoffs the writer's own action is not asked. "Repeat" compares B's action with A's stored action from the view.
- 505 of the 670 A-wrong actions are REQUEST_INFO. "Repeat" is therefore mostly B deferring too.
- The two-sided score mixes frozen decisions (eligible side) with new ones (A-wrong side).
- No correction for multiple comparisons. Per-system results are descriptive.
- Prices are estimates. The ledger holds the actual spend after the run.
- The frozen notes are stored text and are used as they are. They sit in `results/`, which is git-ignored. The v7 code, rules and this document carry no word of the private list (test).

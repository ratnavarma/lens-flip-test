# Decision Flip Test: protocol v3 (U2 rerun)

Status: Frozen. Any change to a hashed item needs a new protocol version.
Version: 3
Written: 2026-10-21 19:04 UTC by `lens flip headline freeze`.
Design: `docs/flip-test-design.md`, sections 4 to 10 and 7.2.

## Changes from v2

v2 is `docs/flip-protocol-v2.md`. It stays as frozen. v3 changes the U2 cases only.

1. Diagnosis, declared before any v3 model call. In the headline-300 run U2 scored 21 of 30 on the full history. That is under the 80% gate. Every miss is an M1+M2 case: 4 of 13 right, against 11 of 11 for M1+M3 and 6 of 6 for M2+M3. In M1+M2 both causes act on the same reading of P. The planted later actual reading for P is normal. That points to the estimate alone. So two causes is not the only sensible reading of the full history. The ground truth is contested. This is a flaw in the case design.
2. U2 pair rule. U2 draws its pair only from M1+M3 and M2+M3, half each. A run plants the two pairs in turn by position. Any even count splits exactly half and half, crossed with the length. Each pair puts one cause on the activity charge (M1 or M2) and one on the other charges (M3). Both stay visible on their own. M1+M2 is never planted. The (type, subtype) counts still match across mechanisms. Each case carries a decoy for the slot it does not plant.
3. Nothing else in the generator moves. Prompts, the action policy and every other mechanism are byte-identical to v2. The plant stream is seeded per mechanism, so M0, M1, M2, M3 and U1 come out the same for the same seed. `generator_version` 2 regenerates the headline-300 cases byte for byte.
4. Rerun scope. Only U2 is rerun, in `results/headline-u2-v3`: 30 cases, seed 12, case ids `u3-NNNN`. Half the cases are at length 100 and half at length 400. Distances are drawn from the seeded stream. The ids cannot collide with headline-300 (`h11-NNNN`).
5. Pre-check gate. First run `results/validity-u2-v3`. It holds the same cases (same seed and ids). It uses the adaptive-high arm only, with one full-history decision per case. The full U2 rerun runs only if U2 scores at least 24 of 30 (80%). Otherwise stop and report. Varma approves the rerun's cap after the pre-check, as for any spend.
6. Merge rule. `lens flip headline merge-report --base results/headline-300 --replace-mechanism U2 --with results/headline-u2-v3 --protocol 3 --out results/headline-300-v3` takes every non-U2 case from the base run. It takes every case from the U2 run. It computes the full v2 metric set and the validity gate on the combined 300 cases. The bootstrap is over the combined cases. It writes `summary_v3.json` and `tables_v3.txt`. Base files stay untouched. The 30 old U2 cases and their 21 of 30 stay on record in the base run. They led to this change. They are not in the combined report.
7. Every other item is identical to v2: hypotheses, power rule, systems, budgets, stages, models, request settings, metrics, validity gate and known limits. The hashes in section 6 equal the v2 hashes in every row except `file:generator.py`. The rows are repeated here.

| Item | v2 sha256 | v3 sha256 | same |
| --- | --- | --- | --- |
| decider.action_schema | `b5c809762c595781e481572d7b6ead5fb9ef9a4afa1e2afcae093ca81a69efbb` | `b5c809762c595781e481572d7b6ead5fb9ef9a4afa1e2afcae093ca81a69efbb` | yes |
| decider.v1 | `f0828f379ef244779cec7c9e8fd2260b23b53f9da81cfd35bf0a8c4dd3673f04` | `f0828f379ef244779cec7c9e8fd2260b23b53f9da81cfd35bf0a8c4dd3673f04` | yes |
| decider.v1.inserted | `74be01664a51887455a9445a08edbf8eae8c1f75b9a34db9d924a1b9d9b5a8d0` | `74be01664a51887455a9445a08edbf8eae8c1f75b9a34db9d924a1b9d9b5a8d0` | yes |
| decider.v2 | `afa9c09a90ae59bfc1f92e858bc5d9d34125298c17c0bd74c1bfa70c9b9d718b` | `afa9c09a90ae59bfc1f92e858bc5d9d34125298c17c0bd74c1bfa70c9b9d718b` | yes |
| decider.v2.inserted | `5628b0bf90fc428bd722e17d4860046d4f0fa69f2692f025a17b6e85d717de2b` | `5628b0bf90fc428bd722e17d4860046d4f0fa69f2692f025a17b6e85d717de2b` | yes |
| event_templates | `38f5302ef4a810a19679a838828c909cc0e1c753fd2676f6df5776c5af3ac919` | `38f5302ef4a810a19679a838828c909cc0e1c753fd2676f6df5776c5af3ac919` | yes |
| file:generator.py | `df562bd8f38fb7827c061825c4c07fb2af3d5c6762f9afe1a96fbe5c716b5dbd` | `8878e1b4e1e854bfaa7bb82831dc5915a21592b2a00bbf2778c36a5a3076570a` | NO |
| file:render.py | `46567cebd52980297247f58f1c46ad315eac94ff3736427fe092bcb85dc6dfaf` | `46567cebd52980297247f58f1c46ad315eac94ff3736427fe092bcb85dc6dfaf` | yes |
| note.prompt_p_residue | `d06b8aafacbcc2e16280c4e6ebd5069a2e5ec8baeb8e23a2e0829d435a08b6cd` | `d06b8aafacbcc2e16280c4e6ebd5069a2e5ec8baeb8e23a2e0829d435a08b6cd` | yes |
| note.prompt_plain_dev | `0f5d2b2200c03ed60eca4fcaf3f9798adfbba5d1b13626819c2c6be06a2f9aa0` | `0f5d2b2200c03ed60eca4fcaf3f9798adfbba5d1b13626819c2c6be06a2f9aa0` | yes |
| note.schema | `23b194ce2295f649c6a2a17cdaedf6bccbc5ae05eda412d1d6a65c1af2ae88d0` | `23b194ce2295f649c6a2a17cdaedf6bccbc5ae05eda412d1d6a65c1af2ae88d0` | yes |
| summary.prompt | `c9e5b1c1517b4b78dfd8117282b94fd21a6fb746183b4c2b4f2051e8f155da77` | `c9e5b1c1517b4b78dfd8117282b94fd21a6fb746183b4c2b4f2051e8f155da77` | yes |
| summary.schema | `54669b5e6a1bfebb4d15788d41b5cd5fb8e51fc2d982eb2383a42262c748c90a` | `54669b5e6a1bfebb4d15788d41b5cd5fb8e51fc2d982eb2383a42262c748c90a` | yes |

Base run kept as is: `results/headline-300`, N 300, seed 11, cases.jsonl `74a1e3a421da9e658cf930f5017a92bfa92d775af6dfb0f14b2cdcb7c0731374`, config.json `cc00fc06f4d799ea0b5ba4cfb40c4253e7b4848674f53061f4f0dca399f155ab`.

## Changes from v1

v1 is `docs/flip-protocol-v1.md`. It stays as frozen. The pilot is readable under both protocols.

1. Power rule per hypothesis. Each hypothesis is tested only when its own analysis set is large enough. H2a needs at least 20 eligible cases. H2b needs at least 20 of its own eligible residue flips (event-view systems, restored search, every D* event in V). H2c needs at least 20 residue flips and at least 20 non-flips among the eligible cases. Below its minimum a hypothesis is descriptive: bounds and n, no claim, no label of fails or supported. v1 pooled one count of 20 residue flips over H2b and H2c. The pilot had 3 flips for H2b and 26 flips for H2c.
2. Harmful flips split. The harmful gated rate stays for continuity. It is reported with two parts that sum to it. Wrong commit: a_H equals a*, a_V differs from a* and a_V is a committing action (not REQUEST_INFO, not PARSE_ERROR). Abstention: a_H equals a*, a_V is REQUEST_INFO and a* is not. PARSE_ERROR flips are reported separately, so wrong commit plus abstention plus parse error equals the harmful rate.
3. Token helper, note only. The helper is unchanged from v1: ceil(characters / 2.2). The v1 pilot ran with it, and its usage gave 2.19 characters per input token on decision requests, net of the 331-token output-format overhead. Dry runs made before the 2026-10-20 calibration used 4 characters per token and are not comparable (their directories end in -v0-chars4). Run costs in v2 are priced from the pilot's real usage.

## 1. Hypotheses

Eligible cases: a_H equals a* and a_A = a_V equals a_H at the residue budget. All bounds are 95% paired bootstrap bounds over cases.

- H2a: residue_rate minus floor_rate is above zero. It fails if the bounds of the paired difference include zero.
- H2b: on event-view systems, on residue flips where every D* event is in V, mean recall of D* in the found set beats recall in random subsets of the same size from the same candidates (1,000 draws per case). It fails if the bounds of the difference include zero.
- H2c: the AUROC of p_residue for residue flips beats 0.5 and beats the omission fraction. It fails if the bounds of either difference include zero. Parse failures are counted, never imputed.
- Power rule, per hypothesis. H2a is tested with at least 20 eligible cases. H2b is tested with at least 20 eligible residue flips of its own (event-view systems, restored search, every D* event in V). H2c is tested with at least 20 residue flips and at least 20 non-flips among the eligible cases with a parsed p_residue.
- Below its minimum a hypothesis is descriptive. It carries n and bounds, no claim, and neither the label fails nor the label supported. The rule applies to the pooled set and to each system's own set. The report prints every hypothesis with its own n and its minimum.
- Prompt note: asking for p_residue may change the notes. Every residue rate is reported for the p_residue prompt.

## 2. Systems, budgets and stages

- Systems: recent, rag-bm25, rag-embed, summary. Budgets: 500, 1000, 2000 tokens.
- a_H: the decider on the full history, once per case.
- a_V: the decider on each view. Flip search where a_V differs from a_H and a_H parsed: target a_H, group-first, 48 calls per search, round-parallel ddmin, verified with the v2 prompt.
- Residue stage at B = 1000 for all systems on eligible cases: note A with p_residue, B on the note, the floor (v2 prompt on V), truncation (B on the last 150 tokens of V). A residue flip runs a flip search over the units of V missing from the note.
- Gated metrics count only cases where a_H equals a*. The other cases are excluded and counted.
- Validity gate: a_H right on at least 90% of cases and at least 80% of every mechanism. In v3 it is applied to the combined 300 cases of the merge report. The pre-check gate is separate: U2 at least 24 of 30.

## 3. Models and request settings

| Role | Model | Thinking | Effort | max_tokens | Reply format |
| --- | --- | --- | --- | --- | --- |
| decider | claude-haiku-5-5 | adaptive (field omitted) | high | 6000 | action schema |
| summarizer | claude-haiku-5-5 | {"type": "disabled"} | medium | 1.5 x B | one-field text schema |
| note-writer | claude-haiku-5-5 | adaptive (field omitted) | high | 6000 | note and p_residue schema |
| rag-embed | nomic-embed-text (local Ollama) | - | - | - | vectors, cached |

Summary chunks: 40 events per step. Each step is cut to B tokens. The note is cut to 150 tokens. Every request goes through the Message Batches API in rounds. Identical requests share one key and are sent once.

## 4. Sample and seeds

Mechanisms: U2 only. Lengths 100 and 400 alternate. The distance knob is drawn per case from the seeded stream. U2 pairs M1+M3 and M2+M3 alternate by position, half each. Generator version 3. Distractors: 2. Bootstrap: 1000 resamples.

| Run dir | N | Seed | sha256 of cases.jsonl | sha256 of config.json |
| --- | --- | --- | --- | --- |
| results/headline-u2-v3 | 30 | 12 | `388e5264c4622d432f1c4b1414c94dae70535c680b7caf6f6313cefe3e366d03` | `6b57796c90f5d1d9ae1fc8101593d7b0a5c9d74ea24b03729e64f3d569c36f25` |

## 5. Metrics

Per cell (system and budget): FCR raw and gated (all, rare only, reweighted to 1% rare), harmful flip rate (over all cases and over gated cases), right restraint on U1 and U2, over-caution on M0, flip-search precision and recall against D*, mean calls and cap hits. Residue block per system and pooled: n_eligible, residue_rate, floor_rate, trunc_rate, H2a, H2b, H2c, n_residue_flips and p_residue parse errors.
Protocol v2 adds, for every cell, the harmful flip split into wrong commit, abstention and parse error (counts, rate over all cases and rate over gated cases), and the analysis-set size and minimum of each hypothesis.
Files: `summary_v2.json` and `tables_v2.txt`.

## 6. Hashes (sha256)

Prompts are rendered with sentinel inputs, so a change of wording changes the hash.

| Item | sha256 |
| --- | --- |
| decider.action_schema | `b5c809762c595781e481572d7b6ead5fb9ef9a4afa1e2afcae093ca81a69efbb` |
| decider.v1 | `f0828f379ef244779cec7c9e8fd2260b23b53f9da81cfd35bf0a8c4dd3673f04` |
| decider.v1.inserted | `74be01664a51887455a9445a08edbf8eae8c1f75b9a34db9d924a1b9d9b5a8d0` |
| decider.v2 | `afa9c09a90ae59bfc1f92e858bc5d9d34125298c17c0bd74c1bfa70c9b9d718b` |
| decider.v2.inserted | `5628b0bf90fc428bd722e17d4860046d4f0fa69f2692f025a17b6e85d717de2b` |
| event_templates | `38f5302ef4a810a19679a838828c909cc0e1c753fd2676f6df5776c5af3ac919` |
| file:generator.py | `8878e1b4e1e854bfaa7bb82831dc5915a21592b2a00bbf2778c36a5a3076570a` |
| file:render.py | `46567cebd52980297247f58f1c46ad315eac94ff3736427fe092bcb85dc6dfaf` |
| note.prompt_p_residue | `d06b8aafacbcc2e16280c4e6ebd5069a2e5ec8baeb8e23a2e0829d435a08b6cd` |
| note.prompt_plain_dev | `0f5d2b2200c03ed60eca4fcaf3f9798adfbba5d1b13626819c2c6be06a2f9aa0` |
| note.schema | `23b194ce2295f649c6a2a17cdaedf6bccbc5ae05eda412d1d6a65c1af2ae88d0` |
| summary.prompt | `c9e5b1c1517b4b78dfd8117282b94fd21a6fb746183b4c2b4f2051e8f155da77` |
| summary.schema | `54669b5e6a1bfebb4d15788d41b5cd5fb8e51fc2d982eb2383a42262c748c90a` |

## 7. Known limits

- The shared token helper counts ceil(characters / 2.2), calibrated with count_tokens on Haiku 5.5 (2.16 to 2.24 characters per token by request kind). Budgets B are close to real model tokens.
- A summary step may stop at max_tokens. The partial text is kept and cut to B. The stop is counted.

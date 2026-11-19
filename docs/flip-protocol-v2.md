# Decision Flip Test: protocol v2 (headline run)

Status: Frozen. Any change to a hashed item needs a new protocol version.
Version: 2
Written: 2026-10-21 04:24 UTC by `lens flip headline freeze`.
Design: `docs/flip-test-design.md`, sections 4 to 10 and 7.2.

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
- Validity gate: a_H right on at least 90% of cases and at least 80% of every mechanism.

## 3. Models and request settings

| Role | Model | Thinking | Effort | max_tokens | Reply format |
| --- | --- | --- | --- | --- | --- |
| decider | claude-haiku-5-5 | adaptive (field omitted) | high | 6000 | action schema |
| summarizer | claude-haiku-5-5 | {"type": "disabled"} | medium | 1.5 x B | one-field text schema |
| note-writer | claude-haiku-5-5 | adaptive (field omitted) | high | 6000 | note and p_residue schema |
| rag-embed | nomic-embed-text (local Ollama) | - | - | - | vectors, cached |

Summary chunks: 40 events per step. Each step is cut to B tokens. The note is cut to 150 tokens. Every request goes through the Message Batches API in rounds. Identical requests share one key and are sent once.

## 4. Sample and seeds

Mechanism mix: the generator's default power mix (rare share 0.5: M0 0.5, each rare mechanism 0.1). Lengths 100 and 400 alternate within each mechanism. The distance knob is drawn per case from the seeded stream. Distractors: 2. Bootstrap: 1000 resamples.

| Run dir | N | Seed | sha256 of cases.jsonl | sha256 of config.json |
| --- | --- | --- | --- | --- |
| results/headline-300 | 300 | 11 | `74a1e3a421da9e658cf930f5017a92bfa92d775af6dfb0f14b2cdcb7c0731374` | `cc00fc06f4d799ea0b5ba4cfb40c4253e7b4848674f53061f4f0dca399f155ab` |

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
| file:generator.py | `df562bd8f38fb7827c061825c4c07fb2af3d5c6762f9afe1a96fbe5c716b5dbd` |
| file:render.py | `46567cebd52980297247f58f1c46ad315eac94ff3736427fe092bcb85dc6dfaf` |
| note.prompt_p_residue | `d06b8aafacbcc2e16280c4e6ebd5069a2e5ec8baeb8e23a2e0829d435a08b6cd` |
| note.prompt_plain_dev | `0f5d2b2200c03ed60eca4fcaf3f9798adfbba5d1b13626819c2c6be06a2f9aa0` |
| note.schema | `23b194ce2295f649c6a2a17cdaedf6bccbc5ae05eda412d1d6a65c1af2ae88d0` |
| summary.prompt | `c9e5b1c1517b4b78dfd8117282b94fd21a6fb746183b4c2b4f2051e8f155da77` |
| summary.schema | `54669b5e6a1bfebb4d15788d41b5cd5fb8e51fc2d982eb2383a42262c748c90a` |

## 7. Known limits

- The shared token helper counts ceil(characters / 2.2), calibrated with count_tokens on Haiku 5.5 (2.16 to 2.24 characters per token by request kind). Budgets B are close to real model tokens.
- A summary step may stop at max_tokens. The partial text is kept and cut to B. The stop is counted.

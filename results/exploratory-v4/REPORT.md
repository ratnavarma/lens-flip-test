# LENS flip test: exploratory v4 report

Exploratory. Outside the frozen v3 protocol. Protocol: `docs/flip-protocol-v4-exploratory.md`. All bounds are 95% paired bootstrap bounds over the 300 cases (1,000 resamples, seed 11). Denominator: 490 eligible handoffs at B = 1,000 unless a row says otherwise. Rates are in percent.

## v3 reference (recomputed from the frozen rows)

| Quantity | k/n | Rate [95% CI] |
| --- | --- | --- |
| Residue rate | 142/490 | 29.0 [24.8, 33.2] |
| Rewording floor | 51/490 | 10.4 [7.8, 13.3] |
| Insertion-recoverable flips | 41/490 | 8.4 [5.9, 11.1] |
| Recoverable minus floor | | -2.0 [-5.4, +1.8] |

## Study A1: extractive notes (A picks verbatim units, at most 150 tokens)

| Quantity | k/n | Rate or difference [95% CI] |
| --- | --- | --- |
| Residue rate, extractive | 193/490 | 39.4 [35.2, 43.7] |
| Insertion-recoverable flips, extractive | 144/490 | 29.4 [25.5, 33.6] |
| ... of which verified with the v2 prompt | 113/490 | 23.1 [19.4, 27.0] |
| Rewording floor (v3) | 51/490 | 10.4 [7.8, 13.3] |
| **Primary: recoverable minus floor** | | **+19.0 [+14.5, +24.0]** (above zero) |
| Verified recoverable minus floor | | +12.7 [+8.1, +17.7] |
| Recoverable, extractive minus v3 | | +21.0 [+16.5, +25.6] |
| Residue, extractive minus v3 | | +10.4 [+5.9, +15.2] |
| Residue, extractive minus floor | | +29.0 [+24.5, +33.7] |

Search outcomes on the 193 extractive flips: not_recoverable 49, restored_1_minimal 144. B chose REQUEST_INFO on 107 of them.

| System | n | Residue, extractive | Recoverable, extractive | Floor (v3) |
| --- | --- | --- | --- | --- |
| recent | 98 | 28.6 [19.8, 37.9] | 26.5 [17.9, 35.6] | 6.1 [2.0, 11.1] |
| rag-bm25 | 163 | 41.1 [33.7, 48.3] | 25.2 [19.2, 31.8] | 10.4 [6.1, 15.2] |
| rag-embed | 144 | 43.8 [35.3, 51.3] | 37.5 [29.6, 45.1] | 9.0 [4.7, 14.1] |
| summary | 85 | 41.2 [30.8, 51.9] | 27.1 [17.6, 37.6] | 17.6 [10.4, 26.2] |

H2c on extractive notes (exploratory): AUROC of A's p_residue 0.50 [0.44, 0.56]; AUROC of the omission fraction (now non-degenerate) 0.45 [0.40, 0.50]; p minus omission +0.04 [-0.04, +0.13].

Notes: mean 115.0 tokens, 2.51 units kept, 9 empty, 154 listed units skipped for the budget, 0 unknown labels, reply status {'ok': 490}. Rare cases on event views with every D* event kept: 95/163.

## Study A2: advice-stripped v3 notes

418 of 490 notes changed; 490 sentences dropped; 1 notes emptied. Rule hits: {'R1': 341, 'R3': 452, 'R4': 257, 'R2': 217}.

| Set | n | Residue, stripped | Residue, v3 | Stripped minus v3 |
| --- | --- | --- | --- | --- |
| All handoffs | 490 | 25.7 [21.6, 29.7] | 29.0 [24.8, 33.2] | -3.3 [-6.4, -0.2] |
| Changed notes only | 418 | 24.2 [19.9, 28.6] | 28.0 [23.8, 32.9] | -3.8 [-7.5, -0.2] |

Stripped minus floor: +15.3 [+10.3, +20.0]. Flip to no flip: 34; no flip to flip: 18.

| System | n | Residue, stripped | Residue, v3 |
| --- | --- | --- | --- |
| recent | 98 | 18.4 [11.1, 25.7] | 21.4 [13.5, 29.5] |
| rag-bm25 | 163 | 31.9 [25.2, 39.0] | 28.8 [22.2, 35.5] |
| rag-embed | 144 | 18.1 [12.0, 24.6] | 27.1 [19.7, 34.4] |
| summary | 85 | 35.3 [25.3, 45.1] | 41.2 [30.3, 52.9] |

## Study B: second readers on the frozen v3 notes

Local gate (qwen3:30b-a3b, full history, v1 prompt): 5/30, pass at 24: FAIL. Per mechanism {'M0': 5, 'M1': 0, 'M2': 0, 'M3': 0, 'U1': 0, 'U2': 0}. Mean 8.17 s per call.

| Reader | Residue rate | Minus Haiku | H2c AUROC of the writer's p_residue | Reader on V (v2) | Residue minus reader on V |
| --- | --- | --- | --- | --- | --- |
| haiku-5-5 (v3) | 142/490 29.0 [24.8, 33.2] | - | 0.64 [0.59, 0.69] | 10.4 [7.8, 13.3] | +18.6 [+14.2, +23.3] |
| sonnet-5-5 | 119/490 24.3 [20.5, 28.2] | -4.7 [-7.9, -1.7] | 0.62 [0.58, 0.67] | 6.9 [4.5, 9.3] | +17.3 [+12.9, +22.1] |

| Reader | recent residue | recent AUROC | rag-bm25 residue | rag-bm25 AUROC | rag-embed residue | rag-embed AUROC | summary residue | summary AUROC |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| haiku-5-5 (v3) | 21.4 [13.5, 29.5] | 0.58 [0.48, 0.69] | 28.8 [22.2, 35.5] | 0.71 [0.63, 0.77] | 27.1 [19.7, 34.4] | 0.71 [0.63, 0.80] | 41.2 [30.3, 52.9] | 0.37 [0.27, 0.49] |
| sonnet-5-5 | 19.4 [12.4, 27.3] | 0.61 [0.51, 0.71] | 25.8 [19.9, 32.3] | 0.64 [0.56, 0.72] | 20.8 [14.4, 27.4] | 0.73 [0.63, 0.82] | 32.9 [23.3, 43.7] | 0.36 [0.26, 0.49] |

## Study C: note-budget sweep

| Budget | Residue rate | Mean note tokens | Notes at cap | H2c AUROC |
| --- | --- | --- | --- | --- |
| 150 | 142/490 29.0 [24.8, 33.2] | 146.6 | 65.9 [61.6, 70.4] | 0.64 [0.59, 0.69] |
| 300 | 162/490 33.1 [27.9, 37.8] | 292.0 | 63.3 [59.4, 67.5] | 0.59 [0.54, 0.64] |
| 600 | 167/490 34.1 [29.1, 39.2] | 500.8 | 6.1 [4.2, 8.1] | 0.54 [0.49, 0.60] |

| Paired difference | Residue rate difference [95% CI] |
| --- | --- |
| 300 minus 150 | +4.1 [-0.4, +8.1] (includes zero) |
| 600 minus 150 | +5.1 [+0.8, +9.5] (above zero) |
| 600 minus 300 | +1.0 [-2.9, +5.1] (includes zero) |

| Budget | recent | rag-bm25 | rag-embed | summary |
| --- | --- | --- | --- | --- |
| 150 | 21.4 [13.5, 29.5] | 28.8 [22.2, 35.5] | 27.1 [19.7, 34.4] | 41.2 [30.3, 52.9] |
| 300 | 22.4 [14.4, 31.5] | 31.3 [23.5, 38.6] | 38.2 [30.2, 46.1] | 40.0 [29.7, 51.7] |
| 600 | 20.4 [12.8, 28.0] | 35.6 [28.5, 42.8] | 39.6 [32.2, 47.3] | 37.6 [27.6, 48.5] |

Monotone decrease across budgets: False.

## Spend

Total $4.7305 of the $30.00 cap (probes $0.0069).

| Arm | Actual USD |
| --- | --- |
| A-extractive | 0.7690 |
| A-stripped | 0.0689 |
| B-sonnet | 1.1876 |
| B-sonnet-view | 2.0758 |
| C-budget | 0.6223 |

## Skipped

- B-local: gate failed

## Prompt hashes (sha256)

| Item | sha256 |
| --- | --- |
| decider.action_schema | `b5c809762c595781e481572d7b6ead5fb9ef9a4afa1e2afcae093ca81a69efbb` |
| decider.v1 | `f0828f379ef244779cec7c9e8fd2260b23b53f9da81cfd35bf0a8c4dd3673f04` |
| decider.v1.inserted | `74be01664a51887455a9445a08edbf8eae8c1f75b9a34db9d924a1b9d9b5a8d0` |
| decider.v2 | `afa9c09a90ae59bfc1f92e858bc5d9d34125298c17c0bd74c1bfa70c9b9d718b` |
| event_templates | `38f5302ef4a810a19679a838828c909cc0e1c753fd2676f6df5776c5af3ac919` |
| file:render.py | `46567cebd52980297247f58f1c46ad315eac94ff3736427fe092bcb85dc6dfaf` |
| note.prompt_p_residue | `d06b8aafacbcc2e16280c4e6ebd5069a2e5ec8baeb8e23a2e0829d435a08b6cd` |
| note.schema | `23b194ce2295f649c6a2a17cdaedf6bccbc5ae05eda412d1d6a65c1af2ae88d0` |
| x4.extract.prompt | `c8b8697b944b9ebded64e427ea7f408869f83d6d845ae1d8a79bd21cf35f3711` |
| x4.extract.schema | `55976ef045f27a3d7d8f6ddeec88f80a0230c628286e62afef43d28371d10731` |
| x4.note_budget.prompt.150 | `d06b8aafacbcc2e16280c4e6ebd5069a2e5ec8baeb8e23a2e0829d435a08b6cd` |
| x4.note_budget.prompt.300 | `ed78331b70730c9f565ba6108b310b664f090fb557567c7c5dbbb47d0e80744f` |
| x4.note_budget.prompt.600 | `c98fcb27385d7ef144dd8d0675cd0f41c45c9ffd6a8fbddc0f68c403a0e4cc3e` |
| x4.note_budget.schema | `23b194ce2295f649c6a2a17cdaedf6bccbc5ae05eda412d1d6a65c1af2ae88d0` |
| x4.strip.rule | `11258fa7668fb82afb3ef0c4a26068532264300fe0da3b3f874e216598a0edc7` |

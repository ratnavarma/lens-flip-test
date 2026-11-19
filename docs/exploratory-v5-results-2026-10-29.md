---
tags: [lens, flip-test, exploratory, v5]
date: 2026-10-29
status: exploratory, outside frozen protocols v3 and v4
---

# Exploratory v5 results 2026-10-29

Six arms on the same 490 handoffs as protocol v3 (300 cases, four systems). Spend $2.13 of the $30 cap Varma approved. Protocol: `docs/flip-protocol-v5-exploratory.md` (FINAL, hashes in `results/exploratory-v5/protocol_v5.json`). Numbers: `results/exploratory-v5/summary_v5.json`. Bootstrap 1000 reps, paired by case. No correction for multiple comparisons. Reference: v3 residue rate 29.0 in 100 (142 of 490).

## 1. One table

| Arm | Question | Result | Against v3 | Reading |
| --- | --- | --- | --- | --- |
| X1 reader-conditioned | Does telling the writer who reads the note and what they decide cut flips? | residue 14.5 in 100 (71 of 490) | minus 14.5 points, CI below zero | Yes. Flips halve at equal length (147 tokens). AUROC of the residue score 0.71 against 0.64 (difference CI includes zero). |
| X2 structured note | Does a five-field form at equal length cut flips? | residue 61.0 in 100 (299 of 490) | plus 32 points, CI above zero | No, it doubles them. 477 of 490 forms were cut at 150 tokens, so the form ate the budget. AUROC 0.37, inverted. Recoverable 15.7 against 8.4 in 100. |
| C1 irrelevant insertion | Does putting back the missing fact restore more flips than putting back random lines of the same size? | targeted 26.4, irrelevant 15.3 in 100 (125 flips) | plus 11.1 points, CI above zero | Yes, but random lines restore 15 in 100 too. Among flips v3 called recoverable, random lines restore 45 in 100. |
| C2 content-preserving rewrite | Does rewording alone change the flip rate? | rewrite 27.3 in 100 (134 of 490) | minus 1.6 points, CI includes zero | No change in the rate. Flips churn: 33 flips vanished, 25 new ones appeared. 149 rewrites lost a number or date. |
| C3 repeated requests | How noisy is the reader on the identical note? | 9.3 in 100 answers differ from the v3 answer; 79 in 100 handoffs give six identical answers | repeat flip rate 28.4, CI includes zero | The noise floor is about 9 in 100 per call. One in four v3 flips vanishes on at least one of five repeats; 13.5 in 100 non-flips flip at least once. |
| C4 semantic absence | Can a judge tell which units the note failed to convey? | judge calls 93 in 100 units absent; AUROC 0.50 | v3 score AUROC 0.64 | The judge failed. It marks almost everything absent and cannot separate flips from non-flips. Not usable as built. |

## 2. Decomposition of the residue (exploratory)

r_q = omission + wording + unattributed, as shares of the 490 handoffs.

| Term | Value (points) | CI95 |
| --- | --- | --- |
| residue r_q | 29.0 | 24.8 to 33.2 |
| omission (C1: targeted minus irrelevant restore, summed over assessed flips) | 2.8 | 1.2 to 4.7 |
| wording (C2 net of C3) | minus 1.1 | minus 4.1 to 2.1 |
| unattributed | 27.3 | 21.7 to 32.9 |

Reading. With these tools, the specific missing fact accounts for a small share of the flips, wording for none, and most of the residue stays unattributed. The 48-call search of v3 restores 29 in 100 flips, random insertion restores 15 in 100, and reader noise alone moves 9 in 100 answers. The flip rate is real and stable (C2 and C3 reproduce it), but attributing a single flip to a single omitted fact is harder than the v3 write-up implied.

## 3. What this means

1. Keep. The reader-conditioned writer (X1) is the strongest result of the week: half the flips at equal length. It is a cheap intervention and belongs in the position paper as an exploratory finding and in Paper 1 as a baseline arm.
2. Keep. C2 and C3 answer J. Raynal's control question: wording alone does not move the rate, and the reader noise floor is about 9 in 100. The residue rate is about 20 points above that floor.
3. Soften. The per-flip attribution claim ("this flip is caused by this omitted fact") rests on the v3 search. C1 shows random insertion restores half as many flips as targeted insertion. Say "the residue is not explained by wording or noise" and stop short of "each flip has one cause" until the search improves.
4. Rerun. X2 is confounded by truncation (477 of 490 cut). A rerun with a form that fits the budget, or a budget that fits the form, costs about $1. Until then X2 says nothing about structure.
5. Rebuild. The C4 judge needs a new prompt and a calibration check against the hand-labelled 200 pairs before it is used again (see ft-16).

## 4. Spend

| Arm | Requests | Actual USD |
| --- | --- | --- |
| X1 | 980 | 0.27 |
| X2 | 2,486 | 0.88 |
| C1 | 626 | 0.15 |
| C2 | 980 | 0.12 |
| C3 | 2,450 | 0.44 |
| C4 | 11,240 | 0.27 |
| Total | 18,762 | 2.13 |

Dry run said $2.19 likely. Varma approved $30 on 2026-10-29 ("use as needed").

## 5. Run notes

- Started 2026-10-29 18:14 PDT from a private worktree (not released); three restarts (shell exit, a cap change to $10 by the prior-art session, back to $30 on Varma's word). Every restart resumed the open batch; no request was paid twice.
- Order X1, C2, C3, C1, C4, X2 by design; X2 last because it can cost the most.
- Per-system results are in `summary_v5.json` under `by_system` and are descriptive only.

## 6. C4 judge v2

Local judges only, no API call. Truth: 400 synthetic pairs from the frozen v3 notes, 200 conveyed and 200 absent (no hand labels were filled). Absent rate on real pairs: all 11,240 for the embedding and Haiku, a uniform sample of 300 for the local models.

| Judge | Balanced accuracy | Kappa | Absent rate (real) | AUROC on flips |
| --- | --- | --- | --- | --- |
| nomic-embed-text, cosine at or above 0.758 | 0.730 | 0.460 | 0.719 | not run |
| qwen3:30b-a3b, prompt v2 with a quote | 0.920 | 0.840 | 0.863 | not run |
| embedding band, qwen3:30b-a3b inside it | 0.897 | 0.795 | 0.867 | not run |
| qwen3:8b, prompt v2 with a quote | 0.883 | 0.765 | 0.797 | not run |
| embedding band, qwen3:8b inside it | 0.870 | 0.740 | 0.810 | not run |
| Haiku, v1 prompt (the v5 run) | 0.719 | 0.438 | 0.932 | 0.50 |

Pass bar: balanced accuracy at least 0.85 and a real absent rate under 0.75. Haiku is scored on the 160 unmodified pairs only.

The new 30B judge is accurate: 0.92 balanced accuracy and kappa 0.84. It still calls 86 in 100 real units absent, so no local judge passed. Two in three real units share no number or date with their note, so the bar looks out of reach.

- Status: no judge passed. The C4 rerun did not run. Owner: this session, done.
- Haiku Batch plan: $0.67 likely for calibration plus all pairs, $0.41 for the band only. Nothing submitted. Owner: Varma, yes or no.
- Next decision: run the 30B judge locally anyway (about 1.2 hours, $0) or revise the bar. Owner: Varma.
- Files: `results/exploratory-v5/judge2/` (`calibration_v2.json`, `calibration_set.jsonl`, `haiku_plan.json`). Code: `src/lens/flip/explore5/judge2.py`.

## 7. X2b rerun

Protocol: `docs/flip-protocol-v5b-x2-rerun.md` (FINAL, hashes in `results/exploratory-v5/protocol_v5b.json`). Numbers: `results/exploratory-v5/summary_v5b.json`. Paired bootstrap as above. Brackets are CI95.

| Measure | v3 free text | X2 five fields, cut | X2b three fields, fits | X2b minus v3 | X2b minus X2 |
| --- | --- | --- | --- | --- | --- |
| Residue, in 100 | 29.0 (142) | 61.0 (299) | 43.5 (213) | +14.5 [10.0, 19.2] | -17.6 [-22.3, -13.3] |
| Notes cut at 150 tokens | | 477 | 0 (gate at most 49: pass) | | |
| Re-asks with tighter caps | | | 0 of 490 | | |
| Mean note tokens | 146.6 | 148.4 | 108.3 (max 136) | | |
| Asks for an action | no | yes | no | | |
| AUROC of p_residue | 0.64 | 0.37 | 0.47 [0.42, 0.52] | -0.17 [-0.22, -0.11] | +0.10 [0.05, 0.15] |
| Recoverable, in 100 | 8.4 (41) | 15.7 (77) | 9.8 (48) | +1.4 [-2.0, 4.9] | -5.9 [-10.2, -2.0] |
| Spend, USD | | 0.88 | 0.60 (dry run 0.90) | | |

Reading. The form fit: none of the 490 notes was cut, so the gate passes. Cutting caused about half of the X2 excess, yet flips stay 14.5 points above v3. The writer's score lost its signal (AUROC 0.47) on notes 38 tokens shorter than v3.

## 8. After the reruns (2026-10-29 late evening)

Varma chose "Rerun X2 and rebuild the judge". Both are done. Spend $2.72 of $30; the judge rerun cost nothing (local model, 79 minutes).

| Rerun | Result | Reading |
| --- | --- | --- |
| X2b, three-field form that fits 150 tokens (0 of 490 cut) | residue 43.5 in 100 (213 of 490); plus 14.5 points against v3, minus 17.6 against X2; AUROC of the form's own residue score 0.47 | Structure at a tight length still loses to the free note. Two confounds: X2b notes ran 108 tokens against 147, and the form has no action field. |
| C4 with judge v2 (local qwen3:30b, quote required; balanced accuracy 0.92 on 400 synthetic pairs) | mean semantic omission 0.85; AUROC of the semantic-omission fraction for the v3 residue flips 0.46, CI 0.41 to 0.53; D* semantic recall 76 in 100 on flips, 81 in 100 on non-flips | A better judge does not rescue the signal. How many units a note conveys does not predict which notes flip. The specific distinction D* is conveyed in three of four flips, so most flips are not the D* fact being absent from the note. |

What changes in the readings of section 3:

1. The reader-conditioned writer (X1) stays the result of the week.
2. The structured-note idea is closed for now: cut or fitting, a form at 150 tokens loses to free text.
3. The omission story needs care. Three measurements now agree that the specific missing fact explains a small share of the flips: C1 (2.8 points of 29), the decomposition, and judge v2 (D* conveyed in 76 in 100 flips). The v3 residue score (AUROC 0.64) still predicts flips while the conveyed-unit count (0.46) does not. The paper should say the flips are real, repeatable and not wording or noise, and that what the writer knows about the reader matters more than which units the note carries.
4. The judge v2 calibration truth is synthetic. Varma's 200-record sheet (vault, "Omission calibration 2026-10-21.md") still has no labels; filling it gives the judge a human check.
5. Ledger note: `update_ledger` in `engine.py` knows the six v5 arms only; the X2b share lives in `X2b/spend.json` and is folded into `spend.json` by the x2b commands.


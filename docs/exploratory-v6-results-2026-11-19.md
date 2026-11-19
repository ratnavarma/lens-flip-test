---
tags: [lens, flip-test, exploratory, v6]
date: 2026-11-19
status: exploratory, outside frozen protocols v3, v4 and v5
---

# Exploratory v6 results 2026-11-19

Two arms on the same 490 handoffs as protocol v3 (300 cases, four systems). Reader B unchanged. Protocol: `docs/flip-protocol-v6-exploratory.md`, FINAL at commit f89cd87 before any model call, `protocol_v6.json` sha256 `1818614a6741b9624f6c241e9392832c967ca4effb03374761980d70dac95dbe`. Numbers: `results/exploratory-v6/summary_v6.json`. Per-handoff metric file, text free: `results/exploratory-v6/handoff_metrics_v6.jsonl`. Bootstrap 1,000 resamples by case, seed 11, paired. Brackets are CI95. Values are in 100 flips or in points of the 490 handoffs, as the row says. No correction for multiple comparisons.

## 1. One table

| # | Quantity | Result | Reading |
| --- | --- | --- | --- |
| Counts | Extractive flips; assessed; controls drawn, sent, matched | 193; 179; 965, 870, 858 | 14 flips have no matched control (all summary view). 7 have no sent control. |
| Counts | Fresh re-reads sent | E1 720 (144 sets), P1 205 (41 sets) | No reply failed or was unparsed. |
| Primary 1 | Extractive specificity: targeted, fresh minus irrelevant, 179 assessed flips | 52.4 minus 39.4 = 13.0 [6.7, 20.1] | Above zero. The minimal set restores more often than random units of equal length. Random units restore 39 in 100. |
| Primary 2 | Content-specific recoverable rate, points of 490 | 4.8 [2.4, 7.4] | Sum 23.3 flip-equivalents of 490. |
| Primary 2 | Minus the v3 rewording floor (10.4) | minus 5.7 [minus 9.5, minus 1.9] | Below zero. Verdict by the frozen rule: not supported. |
| Sensitivity | Every sent control, 186 flips | difference 14.7 [8.4, 21.4]; rate 5.6 [3.2, 8.2]; minus floor minus 4.8 [minus 8.8, minus 1.0] | Same verdict. |
| Sensitivity | Upper bracket: unassessed flips count in full | rate 6.4 [3.9, 9.1]; minus floor minus 4.0 [minus 7.9, minus 0.1] | Same verdict. |
| S1 | Targeted by the v5 definition (search outcome), 179 flips | 74.9 minus 39.4 = 35.5 [28.4, 42.6] | The definition behind Table 8. |
| S1 | Content-specific rate, v5 definition | 13.0 [10.0, 16.1]; minus floor 2.6 [minus 1.6, 6.6] | Includes zero. Not supported under this definition either. |
| S2 | Irrelevant restore rate by search outcome | restored 43.7 [37.8, 49.4] (134 flips); not restored 26.7 [19.0, 35.4] (45 flips) | Random units restore most where the search also restored. |
| S3 | At least one matched control restores | 68.7 [62.0, 75.9] (123 of 179) | |
| S4 | Fresh re-read rate, E1 | 70.8 [65.7, 75.9] (510 of 720) | 63 of 144 sets restore on all five reads. 13 never restore. |
| S4 | Fresh re-read rate, P1 | 57.6 [46.4, 69.4] (118 of 205) | 9 of 41 sets restore on all five reads. 6 never restore. |
| S5 | v5 omission term, v5 definition (recomputed) | 2.8 [1.2, 4.7] | Equals the value the paper reports. |
| S5 | v5 omission term, targeted, fresh | 0.1 [minus 1.0, 1.2] | Includes zero. On the 125 assessed v3 flips: targeted 15.5, irrelevant 15.3, difference 0.2 [minus 4.0, 4.7]. |
| Check | v4 claim recomputed from the metric rows | recoverable 29.4 [25.5, 33.6]; minus floor 19.0 [14.5, 24.0] | Equals the v4 numbers. |

Per system, descriptive only (S6).

| System | Flips / assessed | Targeted, fresh | Irrelevant | Difference | Content-specific rate, points of the system's handoffs | Fresh re-read rate |
| --- | --- | --- | --- | --- | --- | --- |
| rag-bm25 | 67 / 67 | 44.5 | 24.6 | 19.9 [7.9, 31.7] | 8.2 [3.2, 13.2] | 72.7 |
| rag-embed | 63 / 63 | 62.5 | 60.1 | 2.5 [minus 5.7, 10.2] | 1.1 [minus 2.5, 4.5] | 73.0 |
| recent | 28 / 28 | 47.9 | 31.3 | 16.5 [minus 0.5, 32.5] | 4.7 [minus 0.1, 9.8] | 51.5 |
| summary | 35 / 21 | 53.3 | 35.2 | 18.1 [minus 5.0, 41.9] | 4.5 [minus 1.2, 11.0] | 84.3 |

Also descriptive: sets the search verified with the v2 prompt restore again on 79.1 in 100 fresh reads (113 sets). Sets it did not verify restore on 40.6 in 100 (31 sets).

## 2. Reading

1. The extractive margin does not survive a specificity control. The v4 margin was 19.0 points above the floor. Once random units of equal length are netted out and the targeted set is read fresh, the content-specific rate is 4.8 points, which is 5.7 points below the floor. By the rule frozen before the run, the claim "survives its own falsification test" is not supported. Both sensitivity rows agree.
2. Two things remove the margin. Random units restore 39 in 100 extractive flips. A restoring set restores again on only 71 in 100 fresh reads. With the search's own outcome as the targeted side (S1), the margin is 2.6 points and its bounds include zero.
3. Content still matters. The minimal set restores 13.0 points more often than random units of equal length, and the bounds are above zero. The effect is real and small.
4. The v5 omission term was carried by selection. With a fresh targeted rate it falls from 2.8 to 0.1 points and its bounds include zero. On paraphrased notes, the targeted set and random units restore equally often (15.5 against 15.3).
5. The rule is strict in one direction. A control can hold relevant content, since the minimal set is not unique. That raises the irrelevant rate. The rag-embed row fits this: random units restore 60 in 100 there. The floor carries no such control. The verdict stands as frozen. A fairer test needs controls known to be irrelevant.
6. A served-model change between the v4 run and this run cannot be told apart from selection in the fresh re-read rate.

## 3. Spend

| Arm | Requests | Actual USD | Dry run likely USD |
| --- | --- | --- | --- |
| E1 controls | 870 | 0.2037 | 0.237 |
| E1 fresh re-reads | 720 | 0.1747 | 0.197 |
| P1 fresh re-reads | 205 | 0.0541 | 0.058 |
| Total | 1,795 | 0.4325 | 0.49 |

Cap $3.00. One round per arm. No request errored, so no retry round.

## 4. Run notes

- No deviation from the frozen protocol. The protocol check passes on the root and on both arm dirs after the run. The v5 and v5b checks still pass.
- The first start stopped before anything was sent. The key-holding wrapper refuses to run when the shell carries a key variable. The run was started again with that variable removed from the child process, as the wrapper asks. The engine resumed the unsent round. Nothing was paid twice.
- The spend guard accepted both rounds: E1 upper bound $2.4769 and P1 upper bound $0.3201 against the $3.00 cap.
- `results/exploratory-v6/` is a plain folder in this worktree. The older result folders here are links to the main checkout.
- Tests: `tests/flip/test_explore6.py`, 21 tests. The summary is rebuilt from the metric file alone (`analysis.rebuild`).

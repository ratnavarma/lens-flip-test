---
tags: [lens, flip-test, exploratory, v7]
date: 2026-11-19
status: exploratory, outside frozen protocols v3 to v6
---

# Exploratory v7 results 2026-11-19

Two studies on the frozen handoffs of the v3 merge (300 cases, four systems). Reader B unchanged. Protocol: `docs/flip-protocol-v7-exploratory.md`, FINAL at commit 19ddaf9 before any model call, `protocol_v7.json` sha256 `1290eb9d0b19167ede155b31408f4ccdb7adf76e0ad96d4a5790d433144e5706`. Numbers: `results/exploratory-v7/summary_v7.json`. Per-handoff metric file, text free: `results/exploratory-v7/handoff_metrics_v7.jsonl`. Bootstrap 1,000 resamples by case, seed 11, paired. Brackets are CI95. Values are in 100 handoffs. Differences are in points. No correction for multiple comparisons.

## 1. Study R: why the reader holds back

On the 490 eligible handoffs and their frozen v3 notes. v3 reference: flip 29.0 (142), deferral flip 21.4 (105), commit flip 7.3 (36).

### R0 (no call, descriptive)

| Group | n | Flip | Deferral flip |
| --- | --- | --- | --- |
| cut by the cap | 349 | 28.7 [23.8, 33.1] | 21.5 [17.1, 26.2] |
| not cut | 141 | 29.8 [22.7, 37.6] | 21.3 [14.8, 28.6] |
| cut minus not cut | | minus 1.1 [minus 9.7, 7.2] | 0.2 [minus 8.0, 7.8] |
| ends mid-sentence | 331 | 27.8 [23.0, 32.5] | 20.5 [16.0, 25.3] |
| ends a sentence | 159 | 31.4 [24.4, 38.7] | 23.3 [16.6, 30.2] |
| mid-sentence minus ends a sentence | | minus 3.7 [minus 11.7, 4.5] | minus 2.7 [minus 10.7, 4.9] |

### Arms against v3, all handoffs with an arm decision (n = 490 in every arm)

| Arm | New decisions | Flip rate | Flip minus v3 | Deferral flip minus v3 | Commit flip minus v3 | Verdict by the frozen rule |
| --- | --- | --- | --- | --- | --- | --- |
| R1 note-framed reader | 490 | 48.4 | 19.4 [14.7, 24.6] | 22.7 [18.1, 27.3] | minus 3.1 [minus 5.6, minus 0.6] | does not |
| R2 hedges stripped | 195 | 28.6 | minus 0.4 [minus 3.2, 2.4] | minus 1.8 [minus 4.3, 0.6] | 1.6 [minus 0.4, 3.6] | does not |
| R3 trailing fragment removed | 331 | 26.9 | minus 2.0 [minus 4.7, 0.6] | minus 3.5 [minus 5.8, minus 1.0] | 1.6 [minus 0.2, 3.4] | reduces deferral |

No note became empty under R2 or R3. No reply was unparsed. R2 kept the v3 decision for 295 unchanged notes and R3 for 159.

### Restricted to the notes the arm changed

| Arm | n | Flip minus v3 | Deferral flip minus v3 | Commit flip minus v3 |
| --- | --- | --- | --- | --- |
| R1 | 490 | 19.4 [14.7, 24.6] | 22.7 [18.1, 27.3] | minus 3.1 [minus 5.6, minus 0.6] |
| R2 | 195 | minus 1.0 [minus 8.0, 6.3] | minus 4.6 [minus 10.8, 1.8] | 4.1 [minus 1.1, 9.2] |
| R3 | 331 | minus 3.0 [minus 6.9, 0.9] | minus 5.1 [minus 8.5, minus 1.6] | 2.4 [minus 0.3, 4.9] |

### Split by whether the cap cut the v3 note

| Arm | Cut (349): flip, arm against v3 | Cut: flip minus v3 | Cut: deferral minus v3 | Not cut (141): flip, arm against v3 | Not cut: flip minus v3 | Not cut: deferral minus v3 |
| --- | --- | --- | --- | --- | --- | --- |
| R1 | 48.1 against 28.7 | 19.5 [14.1, 25.2] | 22.1 [16.7, 27.6] | 48.9 against 29.8 | 19.1 [11.5, 27.1] | 24.1 [16.3, 32.1] |
| R2 | 28.1 against 28.7 | minus 0.6 [minus 4.6, 3.2] | minus 2.3 [minus 5.3, 0.8] | 29.8 against 29.8 | 0.0 [minus 2.9, 3.2] | minus 0.7 [minus 4.6, 3.2] |
| R3 | 25.8 against 28.7 | minus 2.9 [minus 6.6, 0.9] | minus 4.9 [minus 8.0, minus 1.5] | 29.8 against 29.8 | 0.0, no note changed | 0.0, no note changed |

### Noise floor

C3 reference (the identical v3 note read five more times): 9.3 [7.4, 11.4] in 100 answers differ from the v3 answer. Repeat rates minus v3: flip minus 0.5 [minus 2.6, 1.6], deferral minus 0.9 [minus 2.7, 0.6], commit 0.6 [minus 0.6, 2.1].

| Arm | Answers that differ from v3, changed notes | C3 repeats that differ, same handoffs | Deferral flip minus C3 mean, all | Deferral flip minus C3 mean, changed |
| --- | --- | --- | --- | --- |
| R1 | 31.0 [26.6, 35.6] | 9.3 | 23.6 [19.2, 27.9] | 23.6 [19.2, 27.9] |
| R2 | 26.7 [20.3, 33.3] | 10.3 | minus 0.9 [minus 3.5, 2.0] | minus 4.0 [minus 9.3, 1.9] |
| R3 | 16.0 [11.9, 20.1] | 11.1 | minus 2.5 [minus 4.9, minus 0.1] | minus 4.3 [minus 7.5, minus 0.8] |

R3, descriptive: where the dropped fragment names an action (91 notes), deferral minus v3 is minus 4.4 [minus 11.1, 2.4]. Where it names none (240 notes), minus 5.4 [minus 9.7, minus 0.9].

### Reading

1. Telling the reader that the text is a note does not reduce deferral. It raises it. Deferral flips go from 21.4 to 44.1 in 100, and the flip rate from 29.0 to 48.4. Commit flips fall by 3.1 points. The move is far outside the C3 floor. The reader holds back more, not less, once it knows it reads a note and not the records.
2. Stripping hedging sentences does not reduce deferral. Every difference includes zero, also on the 195 changed notes. Answers do move on those notes (26.7 in 100 differ from v3, against 10.3 for the C3 repeats), but in both directions.
3. Removing the cut-off tail reduces deferral by the frozen rule: minus 3.5 points [minus 5.8, minus 1.0], and minus 5.1 on the 331 changed notes. The move is small. Net of the C3 repeats it is minus 2.5 [minus 4.9, minus 0.1]. The flip rate itself does not move (minus 2.0, bounds include zero), because commit flips rise by 1.6.
4. The cap does not explain the flips. R0 shows no gap between cut and uncut notes. R1 raises deferral equally on both. R3 removes at most about 5 of the 21 deferral points on cut notes.
5. Together: the deferral is not driven by hedging words, and only a little by the cut-off tail. It grows when the reader is told what it reads. One reading, not tested here: the reader treats a note as weaker evidence than records.

## 2. Study X: the reader-conditioned writer, split and two-sided

### (1) The split, 490 eligible handoffs

| Writer | Flip rate | Deferral flip rate | Minus v3 | Minus X1 |
| --- | --- | --- | --- | --- |
| v3 (frozen) | 29.0 [24.8, 33.2] (142) | 21.4 | | |
| X1, both halves (frozen v5) | 14.5 [11.6, 17.5] (71) | 9.2 | minus 14.5 [minus 19.1, minus 10.0] | |
| X1p, policy half | 20.6 [17.0, 24.4] (101) | 15.7 | minus 8.4 [minus 13.1, minus 4.0] | 6.1 [2.5, 9.8] |
| X1r, reader half | 24.3 [20.6, 28.2] (119) | 17.1 | minus 4.7 [minus 8.3, minus 0.8] | 9.8 [5.6, 14.1] |

X1p minus X1r: minus 3.7 [minus 7.7, 0.6]. Verdict by the frozen rule: both.

Notes: X1p mean 145.9 tokens, X1r 146.8. Every reply parsed.

### (2) The 670 A-wrong handoffs

| Measure | v3 notes | X1 notes | X1 minus v3 |
| --- | --- | --- | --- |
| Rescue: B ends on the right action | 15.2 [12.7, 17.6] (102) | 20.3 [17.3, 23.3] (136) | 5.1 [2.0, 8.5] |
| Repeat: B repeats A's wrong action | 76.1 [73.2, 79.0] (510) | 71.6 [68.5, 74.8] (480) | minus 4.5 [minus 7.8, minus 1.3] |
| Descriptive: B chooses REQUEST_INFO | 79.9 (535) | 68.1 (456) | minus 11.8 [minus 15.1, minus 8.0] |

### (3) The two-sided score, all 1,160 gated handoffs

| Set | n | B's accuracy, v3 notes | B's accuracy, X1 notes | X1 minus v3 |
| --- | --- | --- | --- | --- |
| all gated | 1,160 | 38.8 [35.7, 42.0] (450) | 47.8 [45.0, 50.6] (555) | 9.1 [6.4, 11.6] |
| eligible (frozen decisions) | 490 | 71.0 | 85.5 | 14.5 [10.0, 19.1] |
| A-wrong (new decisions) | 670 | 15.2 | 20.3 | 5.1 [2.0, 8.5] |

By system, descriptive, X1 minus v3: rag-bm25 6.9 [0.3, 13.4], rag-embed 16.9 [11.7, 22.2], recent 0.0 [minus 4.1, 4.1], summary 12.4 [8.2, 17.2].

### Reading

1. Both halves of the paragraph cut flips, and neither reaches the whole. The policy half removes 8.4 points and the reader half 4.7. Together in X1 they removed 14.5. The two halves are not separable from each other: their difference includes zero.
2. X1 is not a one-sided gain. Where A was wrong, an X1 note leads B to the right action more often (plus 5.1 points) and to A's wrong action less often (minus 4.5).
3. Over all gated handoffs B is right on 47.8 in 100 with X1 notes against 38.8 with v3 notes. The gain is 9.1 points, bounds above zero.
4. Most A-wrong handoffs stay wrong under either note: 80 in 100 under X1. The note carries what the view lacked only in a minority of cases.
5. Descriptive: the recent system shows no gain.

## 3. Spend

| Arm | Requests | Actual USD | Dry run likely USD |
| --- | --- | --- | --- |
| R1 | 490 | 0.1090 | 0.092 |
| R2 | 195 | 0.0320 | 0.036 |
| R3 | 291 | 0.0480 | 0.054 |
| X1p | 980 | 0.2818 | 0.289 |
| X1r | 980 | 0.2721 | 0.283 |
| XW | 2,680 | 0.7237 | 0.780 |
| Total v7 | 5,616 | 1.4667 | 1.53 |

Track total: v6 $0.4325 plus v7 $1.4667 = $1.8992 of the $5.00 cap.

## 4. Run notes

- No deviation from the frozen protocol. No guard refusal. The protocol check passes on the root and both run folders after the run. The v5 and v6 checks still pass.
- Rounds: R in one round of 976 requests. X in five rounds of 1,000, 1,000, 1,000, 1,000 and 640. Largest round bound $1.6026 against the $4.56 cap. No request errored.
- R3 sent 291 requests for 331 changed notes. 40 notes give the same text under R2 and R3 and share one request, booked under R2.
- The run was started with the key variable removed from the child process, as the key-holding wrapper asks.
- Limit on reading 3 of study R. The frozen text says a move smaller than the C3 floor is not read as an effect, and gives no number for it. The R3 verdict follows the bound rule. The comparison with C3 is printed beside it.
- `results/exploratory-v7/` is a plain folder in this worktree.
- Tests: `tests/flip/test_explore7.py`, 17 tests. The summary is rebuilt from the metric file alone (`analysis.rebuild`).

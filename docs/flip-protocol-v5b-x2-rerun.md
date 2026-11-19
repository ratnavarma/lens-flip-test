# Decision Flip Test: protocol v5b, arm X2b (exploratory addendum)

Status: addendum to `docs/flip-protocol-v5-exploratory.md` (FINAL). The v5 document, its hashes and its six arm directories are not changed.
Version: 5b-exploratory
Written: 2026-10-29, branch `exploratory-v5`.
Approved: Varma, 2026-10-29: rerun X2 with a form that fits the 150-token budget, at about $1, inside the standing $30 cap.

## 0. Why

Arm X2 asked for a five-field form and cut the rendered text at 150 tokens. 477 of 490 forms were cut. Its residue rate (61.0 in 100) measures the cut, not the structure. X2b asks for a form that fits.

## 1. Frame and fixed parts

- The same 490 handoffs, `frame.jsonl` and `cases.jsonl` of v5 (sha256 checked against `protocol_v5.json`).
- The same reader: B decides from the rendered note with the v1 prompt, claude-haiku-5-5, adaptive thinking, effort high, max_tokens 6000.
- The same writer settings as X2: claude-haiku-5-5, adaptive thinking, effort high, max_tokens 6000.
- The same budget: 150 tokens by the shared helper, ceil(chars / 2.2). That is 330 characters.
- The same flip search on X2b flips: group-first, 48 calls, v2 verify, candidates = units of V the note does not contain verbatim.
- The same bootstrap: 1,000 resamples of the 300 cases, seed 11, paired by case and handoff.

## 2. The form

Three fields, each with a hard word cap stated in the prompt.

| Field | Label | Asks for | Cap, first ask | Cap, re-ask |
| --- | --- | --- | --- | --- |
| facts | Facts | the facts the colleague needs for this decision | 20 words | 14 words |
| changed | Changed | what changed or what conflicts in the records | 10 words | 7 words |
| open | Open | what is uncertain or still open | 8 words | 5 words |

- The prompt states the budget three ways: 150 tokens, about 330 characters, about 47 words (330 / 7, the 6.9 characters per word seen on the X2 forms, rounded up). It states the sum of the caps as the word limit for the three fields (38 first, 26 on the re-ask).
- p_residue is kept, with the X2 wording, so its AUROC compares with the earlier arms.
- The form does not ask for the next action or for reversibility. X2 did. The writer may still state an action under Facts. This is a design difference from X2 and is reported with the result.
- Renderer: three lines `Facts: ...`, `Changed: ...`, `Open: ...`, in that order, joined by newlines. Whitespace inside a field is collapsed to single spaces. An empty field reads `none`. Labels are one word. The labels and separators take 24 characters.
- Reply schema: {facts, changed, open: string; p_residue: number}, additionalProperties false.

## 3. The re-ask and the cut

1. A writes the form with the first caps (stage `x2b_A`).
2. If the rendered form fits in 150 tokens, it is the note.
3. If not, A is asked once more with the re-ask caps (stage `x2b_A2`). Same records, same request, same template. Only the caps change. The second call is counted in the price and the ledger.
4. If the second form parses, it is used with its own p_residue. If it still passes 150 tokens, it is cut by the shared helper and flagged `cut`. If the second reply does not parse or the request fails twice, the first form is cut and flagged.
5. A reply that is not JSON is treated as in X2: its text is the note, cut if needed, complete False.

Gate. X2b is readable only if the cut share is at or under 10 in 100 (at most 49 of 490). The report prints the gate and its verdict. Above the gate the arm is reported as not readable, like X2.

## 4. Metrics

- Residue rate (B's action differs from A's), paired against the v3 free-text note and against X2.
- AUROC of p_residue for X2b flips against 0.5. Paired differences against the v3 AUROC and the X2 AUROC.
- Recoverable rate (search status restored, cap hit included), against the v3 recoverable rate (41 of 490), the v3 rewording floor and X2.
- Cut share and the gate, re-asks, re-asks that fit, mean and max note tokens, reply status.
- Per-system rates, descriptive only.
- Spend: X2b actual by stage, plus the shared ledger total.

## 5. Runner, guards, files

- `uv run python -m lens.flip.explore5.x2b dryrun | freeze [--price-confirmed] | run --price-confirmed --cap-usd N | report`.
- Code: `src/lens/flip/explore5/x2b_prompts.py`, `src/lens/flip/explore5/x2b.py`. They reuse the v5 engine, the headline Engine, the call cache, rounds, retries and the flip search by import. No v5 module is edited.
- `freeze` writes `results/exploratory-v5/protocol_v5b.json` with the hashes of section 8 as DRAFT. `freeze --price-confirmed` sets FINAL. It refuses if the dry-run likely price is above $1.50.
- The engine refuses unless three things hold: protocol_v5b is FINAL, its hashes match and the v5 protocol check passes (v5 prompts, v3 prompts, v5 arm configs, frame, cases, C1 draws, calibration). The sha256 of `protocol_v5.json` itself is also pinned.
- Spend guard as in v5. The ledger is the shared `results/exploratory-v5/spend.json`. X2b is added to it as its own arm. Before each round the ledger total plus the round's likely cost must stay under the cap. The round's upper bound must fit in the cap minus the other arms' spend. A refusal stops the run.
- Outputs: `results/exploratory-v5/X2b/` (run dir), `dryrun_v5b.json`, `protocol_v5b.json`, `summary_v5b.json`.
- Tests: `tests/flip/test_explore5_x2b.py`.

## 6. Price

The dry run prices X2b from stored prompt sizes and the real usage of arm X2. No model call.

- Writer and re-ask: exact prompt lengths. Output 1,851 tokens per request (X2's x2_A mean). Likely re-ask share 0.5 (no data yet), upper 1.0.
- B: the v3 note as the length stand-in. Output 434 tokens (X2's x2_B mean).
- Search: the v5 dry-run method (flip rate 32.0%, 8.93 calls and 0.55 verifies per search). A sensitivity line prices the search at X2's observed flip rate (61.0%).
- Upper bound: max_tokens on every request, a re-ask for every handoff, 39.4% of handoffs searched at 48 calls plus one verify.

Dry run, 2026-10-30 03:11 UTC: likely $0.90, upper $17.56 (2,714 requests likely). With the search at X2's flip rate the likely price is $1.27. Both sit under the $1.50 gate. Full table in `results/exploratory-v5/dryrun_v5b.json`. Actual spend: $0.60.

## 7. Known limits

- One sample per request. C3 measured reader noise at about 9 in 100 answers on the v3 notes.
- The form drops the action and reversibility fields of X2. X2b against X2 therefore changes the fields as well as the fit.
- The re-ask uses a fresh request, not a dialogue with the first draft. The second form may differ in content, not only in length.
- Exploratory. No result here changes a v3, v4 or v5 number.

## 8. Hashes (sha256)

Rendered with the v5 sentinel inputs. Frozen 2026-10-30 03:11 UTC, FINAL 2026-10-30 03:11 UTC.

| Item | sha256 |
| --- | --- |
| x5b.x2b.prompt_first | `438747015615b1123e293e8efd5cf28c93092116e2800ba1f2887d124286a187` |
| x5b.x2b.prompt_tight | `20093b63adccf3f068b1c3fbd89a1f346ba8dfb4f21e5983b0ec51ef33bc2e28` |
| x5b.x2b.render | `191a10a1761aa6b9a34a699b155ae332c9022d626842cc575c9507b788f429d5` |
| x5b.x2b.rule | `8c59b45dfffac3f7a8084bd3926fd06d1cb697c17b960edb763a26f911d3d4e4` |
| x5b.x2b.schema | `c7ccff819ff29e0648574be4829337cda3a715fa1f6bb142368da2e44abe2138` |
| arm config X2b | `4dd8c2ed838399db9bbcbd8523297644eb96930f4962fb3559701e7e2f9e7531` |
| calibration.json | `0e12af21a05647a9dae122af69b80704673d847bd63c94c510724d9e68da4f1f` |
| protocol_v5.json | `4356cf459a2920dbac89afdc1c0aad3cc0cc6cefffd6eec202255aebd81b6032` |

Frame and cases are the v5 files: frame `bed6b1d00288354439f4a0459dc7d512cfc83245b369a141e9e4f1a88f6eb69f`, cases `432e55ca579b74766e9097428fe513bd5a8c96496892dec1b023365ba1a4ed3c`.

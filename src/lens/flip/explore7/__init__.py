"""Exploratory protocol v7 (docs/flip-protocol-v7-exploratory.md). Outside the frozen v3 to v6 protocols.

Two studies on the frozen handoffs of the v3 merge, reader B unchanged.

    Study R  why the reader holds back on the 490 eligible handoffs and their frozen v3 notes
             R0  descriptive, no call: v3 flips by whether the cap cut the note
             R1  the v1 reader prompt with the note framed as a note
             R2  hedging sentences stripped by a fixed text rule
             R3  the trailing fragment of a note that ends mid-sentence removed by a fixed text rule
    Study X  the reader-conditioned writer of v5 (X1), split and scored on both sides
             X1p the v3 note prompt plus the policy half of the X1 paragraph
             X1r the v3 note prompt plus the reader half of the X1 paragraph
             XW  the 670 gated handoffs where A's action from the view was wrong: v3 notes and X1 notes, B on each

Nothing in the v3 to v6 modules or result dirs changes. v7 reuses the v5 engine and asker by import.
Results are written to results/exploratory-v7 with schema lens.flip.exploratory/1. No flip search in any arm.
"""

SCHEMA = "lens.flip.exploratory/1"
VERSION = "7-exploratory"

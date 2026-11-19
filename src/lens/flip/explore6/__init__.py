"""Exploratory protocol v6 (docs/flip-protocol-v6-exploratory.md). Outside the frozen v3, v4 and v5 protocols.

A specificity control for the extractive falsification margin of v4, and a symmetric targeted rate for the v5
omission term. Two arms on the same 490 handoffs, reader B unchanged:

    E1  on the 193 extractive flips of v4: K = 5 irrelevant control sets per flip (the v5 draw rule, seed tag
        x6-c1), and 5 fresh re-reads of the minimal restoring set of each of the 144 restored flips
    P1  on the 41 restored v3 flips: 5 fresh re-reads of the v3 minimal restoring set on the v3 note

Nothing in the v3, v4 or v5 modules or result dirs changes. v6 reuses the v5 engine, asker, salt rule and draw
rule by import. Results are written to results/exploratory-v6 with schema lens.flip.exploratory/1.
"""

SCHEMA = "lens.flip.exploratory/1"
VERSION = "6-exploratory"

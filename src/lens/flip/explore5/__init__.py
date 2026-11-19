"""Exploratory protocol v5 (docs/flip-protocol-v5-exploratory.md). Outside the frozen v3 and v4 protocols.

Six arms on the 490 eligible handoffs of results/headline-300-v3, for the journal version:
X1 reader-conditioned estimate, X2 structured note at equal tokens, and four controls on the residue:
C1 irrelevant insertion of equal length, C2 content-preserving rewrite, C3 repeated identical requests,
C4 semantic absence. Results are written to results/exploratory-v5 with schema lens.flip.exploratory/1.
"""

SCHEMA = "lens.flip.exploratory/1"
VERSION = "5-exploratory"

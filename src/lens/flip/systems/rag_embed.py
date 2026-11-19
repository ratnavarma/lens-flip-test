"""Retrieval by embeddings. The rendered event lines and the query text are embedded
(nomic-embed-text by default). Lines are ranked by cosine similarity to the query,
ties to the later event, and added best first until the next line does not fit.
The view is returned in history order.

The embedder is an injected callable: a list of texts in, a list of vectors out.
"""

from __future__ import annotations

import math
from typing import Callable, Sequence

from ..schema import Case, View
from .base import event_view, rendered, take_within_budget

Embedder = Callable[[Sequence[str]], "list[list[float]]"]


def cosine(a: Sequence[float], b: Sequence[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    return dot / (na * nb) if na and nb else 0.0


class RagEmbedSystem:
    name = "rag-embed"
    reference = False

    def __init__(self, embedder: Embedder):
        self.embedder = embedder

    def view(self, case: Case, budget_tokens: int) -> View:
        pairs = rendered(case)
        vecs = self.embedder([line for _, line in pairs] + [case.query_text])
        if len(vecs) != len(pairs) + 1:
            raise RuntimeError("embedder returned the wrong number of vectors")
        q = vecs[-1]
        scores = [cosine(v, q) for v in vecs[:-1]]
        order = sorted(range(len(pairs)), key=lambda i: (-scores[i], -i))
        ids = take_within_budget((pairs[i] for i in order), budget_tokens)
        return event_view(case, ids, self.name, budget_tokens)

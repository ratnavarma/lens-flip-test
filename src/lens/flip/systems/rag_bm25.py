"""BM25 retrieval over rendered event lines. Written by hand, no dependencies.

Score of a line d for query terms q: sum over unique terms t in q of
idf(t) * tf * (k1 + 1) / (tf + k1 * (1 - b + b * |d| / avgdl)),
with idf(t) = ln(1 + (N - df + 0.5) / (df + 0.5)), k1 = 1.5, b = 0.75.
Ties go to the later event. Events are added best first until the next line
does not fit; the view is returned in history order.
"""

from __future__ import annotations

import math
import re
from collections import Counter

from ..schema import Case, View
from .base import event_view, rendered, take_within_budget

K1 = 1.5
B = 0.75
_TOKEN = re.compile(r"[a-z0-9-]+")


def tokenise(text: str) -> list[str]:
    return _TOKEN.findall(text.lower())


def bm25_scores(docs: list[str], query: str) -> list[float]:
    toks = [tokenise(d) for d in docs]
    n = len(toks)
    if n == 0:
        return []
    avgdl = sum(len(t) for t in toks) / n or 1.0
    df: Counter[str] = Counter()
    for t in toks:
        df.update(set(t))
    tfs = [Counter(t) for t in toks]
    scores = [0.0] * n
    for term in sorted(set(tokenise(query))):
        if term not in df:
            continue
        idf = math.log(1.0 + (n - df[term] + 0.5) / (df[term] + 0.5))
        for i, tf in enumerate(tfs):
            f = tf.get(term, 0)
            if f:
                scores[i] += idf * f * (K1 + 1.0) / (f + K1 * (1.0 - B + B * len(toks[i]) / avgdl))
    return scores


class RagBm25System:
    name = "rag-bm25"
    reference = False

    def view(self, case: Case, budget_tokens: int) -> View:
        pairs = rendered(case)
        scores = bm25_scores([line for _, line in pairs], case.query_text)
        order = sorted(range(len(pairs)), key=lambda i: (-scores[i], -i))
        ids = take_within_budget((pairs[i] for i in order), budget_tokens)
        return event_view(case, ids, self.name, budget_tokens)

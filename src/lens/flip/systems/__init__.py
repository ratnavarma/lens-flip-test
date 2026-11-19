from __future__ import annotations

from typing import Any

from .base import MemorySystem, count_tokens
from .full import FullSystem
from .rag_bm25 import RagBm25System
from .rag_embed import RagEmbedSystem
from .recent import RecentSystem
from .summary import SummarySystem

SYSTEMS = {
    "full": FullSystem,
    "recent": RecentSystem,
    "summary": SummarySystem,
    "rag-bm25": RagBm25System,
    "rag-embed": RagEmbedSystem,
}


def get_system(name: str, *, summarizer: Any = None, embedder: Any = None) -> MemorySystem:
    """Build a system by name. summary needs a summarizer and rag-embed needs an embedder."""
    try:
        cls = SYSTEMS[name]
    except KeyError:
        raise ValueError(f"unknown system {name!r}; choose from {sorted(SYSTEMS)}") from None
    if cls is SummarySystem:
        if summarizer is None:
            raise ValueError("the summary system needs a summarizer")
        return SummarySystem(summarizer)
    if cls is RagEmbedSystem:
        if embedder is None:
            raise ValueError("the rag-embed system needs an embedder")
        return RagEmbedSystem(embedder)
    return cls()


__all__ = ["MemorySystem", "count_tokens", "get_system", "SYSTEMS", "FullSystem", "RecentSystem",
           "RagBm25System", "RagEmbedSystem", "SummarySystem"]

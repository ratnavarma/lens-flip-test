"""Live Ollama checks. Skipped unless LENS_OLLAMA=1."""

from __future__ import annotations

import os

import pytest

from lens.flip.decider import ollama_decider
from lens.flip.generator import Knobs, generate_cases
from lens.flip.systems.base import event_view

pytestmark = pytest.mark.skipif(os.environ.get("LENS_OLLAMA") != "1", reason="set LENS_OLLAMA=1 to call Ollama")


def test_live_decider_returns_an_action():
    case = generate_cases(2, 1, Knobs(length=100))[0]
    action = ollama_decider("qwen3:30b-a3b")(event_view(case, (e.id for e in case.events), "full"), [], case)
    assert action in case.actions


def test_live_embedding_shape_and_cosine():
    from lens.flip.llm import CachedEmbedder
    from lens.flip.systems.rag_embed import cosine

    a, b, c = CachedEmbedder()(["Payment received: 80.00.", "A payment of 80.00 was posted.", "Device replaced."])
    assert len(a) == len(b) == len(c) > 100
    assert cosine(a, b) > cosine(a, c)


def test_live_text_generation_returns_text_only():
    from lens.flip.llm import OllamaText

    out = OllamaText()("Write one short sentence about a payment of 80.00.", 60)
    assert out and not out.lstrip().startswith("{")

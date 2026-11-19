"""Rolling summary memory.

The rendered history is split into chunks of 40 events. A running summary is
updated chunk by chunk: summary_k+1 = summarize(summary_k, chunk_k, budget).
The summary is written before the query is known, so neither the prompt nor the
system ever reads the query. The final text is cut to the budget with the shared
token helper. The view has no verbatim events; its units are the summary sentences.

The summariser is an injected callable (summary, chunk lines, budget) -> text.
`ollama_summarizer` is the default backend: qwen3:30b-a3b, think off, plain text,
num_predict of about 1.5 times the budget.
"""

from __future__ import annotations

from typing import Callable, Sequence

from ..llm import TextGen
from ..schema import Case, View
from .base import CHARS_PER_TOKEN, rendered, split_sentences, truncate_tokens

CHUNK_EVENTS = 40
NUM_PREDICT_FACTOR = 1.5

Summarizer = Callable[[str, "list[str]", int], str]


def build_summary_prompt(summary: str, chunk: Sequence[str], budget_tokens: int) -> str:
    """Prompt for one summary update. It takes no case and no query by design."""
    return (
        "You keep a compact factual summary of an account's records. "
        "Update the current summary with the new records below.\n\n"
        "Keep what may matter for future decisions: amounts, periods, readings (actual or estimated), "
        "corrections and what they correct, device changes, plan changes, credits and whether they were applied, "
        "and contacts. Cite record ids. Write plain sentences, one fact per sentence, with no headings.\n"
        f"Stay within {budget_tokens} tokens (about {int(budget_tokens * CHARS_PER_TOKEN)} characters). "
        "Drop the least useful details first. Reply with the updated summary only.\n\n"
        f"Current summary:\n{summary.strip() or '(empty)'}\n\n"
        "New records:\n" + "\n".join(chunk)
    )


def ollama_summarizer(text_gen: TextGen) -> Summarizer:
    """Default summariser on a text generator (see llm.OllamaText)."""

    def summarize(summary: str, chunk: list[str], budget_tokens: int) -> str:
        return text_gen(build_summary_prompt(summary, chunk, budget_tokens),
                        int(NUM_PREDICT_FACTOR * budget_tokens))

    return summarize


class SummarySystem:
    name = "summary"
    reference = False

    def __init__(self, summarizer: Summarizer, chunk_events: int = CHUNK_EVENTS):
        if chunk_events < 1:
            raise ValueError("chunk_events must be at least 1")
        self.summarizer = summarizer
        self.chunk_events = chunk_events

    def view(self, case: Case, budget_tokens: int) -> View:
        lines = [line for _, line in rendered(case)]
        summary = ""
        for lo in range(0, len(lines), self.chunk_events):
            chunk = lines[lo : lo + self.chunk_events]
            summary = truncate_tokens(self.summarizer(summary, chunk, budget_tokens), budget_tokens)
        return View(system=self.name, budget=budget_tokens, text=summary, event_ids=[],
                    units=split_sentences(summary))

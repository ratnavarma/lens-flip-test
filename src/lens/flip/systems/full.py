"""Reference system: every event. The budget is ignored."""

from __future__ import annotations

from ..schema import Case, View
from .base import event_view


class FullSystem:
    name = "full"
    reference = True

    def view(self, case: Case, budget_tokens: int) -> View:
        return event_view(case, (e.id for e in case.events), self.name, budget_tokens)

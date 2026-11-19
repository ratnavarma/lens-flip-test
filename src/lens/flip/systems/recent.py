"""The latest events whose lines fit the budget, in history order."""

from __future__ import annotations

from ..schema import Case, View
from .base import event_view, rendered, take_within_budget


class RecentSystem:
    name = "recent"
    reference = False

    def view(self, case: Case, budget_tokens: int) -> View:
        newest_first = reversed(rendered(case))
        return event_view(case, take_within_budget(newest_first, budget_tokens), self.name, budget_tokens)

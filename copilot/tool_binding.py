"""Thin call-scoped tool binding for borrowed Copilot Sessions and Services.

The delegates own execution, history and observations. These facades retain
only the delegate and tool configuration; Memory acquisition and resource
lifecycle remain with the configured tools and host provider.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING

from llm_adapters.tool_binding import BoundToolCollection, bound_tools

if TYPE_CHECKING:
    from .failure_observability import CopilotFailureObservation
    from .runtime_observability import CopilotObservedResult
    from .service import CopilotService
    from .session import CopilotSession, CopilotTurn


__all__ = ("BoundCopilotSession", "BoundCopilotService")


@dataclass(frozen=True, slots=True, eq=False)
class BoundCopilotSession:
    """Bind one stable tool configuration around an existing Session's calls."""

    session: CopilotSession
    tools: BoundToolCollection

    def __post_init__(self) -> None:
        if not isinstance(self.tools, BoundToolCollection):
            raise TypeError("tools must be a BoundToolCollection")

    @property
    def history(self) -> tuple[CopilotTurn, ...]:
        return self.session.history

    @property
    def turn_count(self) -> int:
        return self.session.turn_count

    @property
    def model(self) -> str:
        return self.session.model

    @property
    def max_turns(self) -> int:
        return self.session.max_turns

    @property
    def experiment_context(self) -> dict[str, object] | None:
        return self.session.experiment_context

    def ask(self, question: str) -> str:
        with bound_tools(self.tools):
            return self.session.ask(question)

    def ask_with_result(self, question: str) -> CopilotTurn:
        with bound_tools(self.tools):
            return self.session.ask_with_result(question)

    def ask_with_observability(
        self,
        question: str,
        *,
        on_failure: Callable[[CopilotFailureObservation], None] | None = None,
    ) -> CopilotObservedResult:
        with bound_tools(self.tools):
            return self.session.ask_with_observability(question, on_failure=on_failure)

    def export_history(self) -> list[dict[str, object]]:
        return self.session.export_history()

    def reset(self) -> None:
        self.session.reset()


@dataclass(frozen=True, slots=True, eq=False)
class BoundCopilotService:
    """Activate tools during execution and wrap Sessions from the delegate."""

    service: CopilotService
    tools: BoundToolCollection

    def __post_init__(self) -> None:
        if not isinstance(self.tools, BoundToolCollection):
            raise TypeError("tools must be a BoundToolCollection")

    def run(self, question: str, **request_options: object) -> CopilotObservedResult:
        with bound_tools(self.tools):
            return self.service.run(question, **request_options)

    def _run_with_observability(
        self, question: str, **request_options: object,
    ) -> CopilotObservedResult:
        with bound_tools(self.tools):
            return self.service._run_with_observability(question, **request_options)

    def create_session(self, **request_options: object) -> BoundCopilotSession:
        session = self.service.create_session(**request_options)
        return BoundCopilotSession(session, self.tools)

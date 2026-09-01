"""Thin injected-client facade over existing Copilot primitives."""

from typing import Callable

from .failure_observability import (
    CopilotFailureObservation,
    run_copilot_turn_with_failure_observability,
)
from .runtime_observability import (
    CopilotObservedResult,
    run_copilot_turn_with_observability,
)
from .session import CopilotSession


__all__ = ("CopilotService",)


class CopilotService:
    """Bind a borrowed client and model to existing Copilot entry points."""

    def __init__(
        self,
        client: object,
        *,
        model: str,
    ) -> None:
        self._client = client
        self._model = model

    def run(
        self,
        question: str,
        *,
        experiment_context: dict[str, object] | None = None,
        turn_timeout_seconds: float | None = None,
        on_failure: (
            Callable[[CopilotFailureObservation], None] | None
        ) = None,
        **request_options: object,
    ) -> CopilotObservedResult:
        """Run one observed turn through the appropriate existing entry."""
        if on_failure is None:
            return run_copilot_turn_with_observability(
                self._client,
                model=self._model,
                question=question,
                experiment_context=experiment_context,
                turn_timeout_seconds=turn_timeout_seconds,
                **request_options,
            )
        return run_copilot_turn_with_failure_observability(
            self._client,
            model=self._model,
            question=question,
            experiment_context=experiment_context,
            turn_timeout_seconds=turn_timeout_seconds,
            on_failure=on_failure,
            **request_options,
        )

    def create_session(
        self,
        *,
        experiment_context: dict[str, object] | None = None,
        max_turns: int = 8,
        turn_timeout_seconds: float | None = None,
        **request_options: object,
    ) -> CopilotSession:
        """Create an independent session using the bound client and model."""
        return CopilotSession(
            self._client,
            model=self._model,
            experiment_context=experiment_context,
            max_turns=max_turns,
            turn_timeout_seconds=turn_timeout_seconds,
            **request_options,
        )

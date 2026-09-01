"""Failure observability for one bounded Copilot turn."""

from dataclasses import dataclass
from time import perf_counter as _perf_counter
from typing import Callable

from .runtime_observability import (
    CopilotObservedResult,
    CopilotRuntimeMetrics,
)
from .runtime_result import _run_copilot_turn_with_result


__all__ = (
    "CopilotFailureObservation",
    "run_copilot_turn_with_failure_observability",
)


@dataclass(frozen=True, slots=True)
class CopilotFailureObservation:
    """Minimal, payload-free facts about one failed Copilot turn."""

    stage: str
    provider_request_count: int
    tool_invocation_count: int
    elapsed_seconds: float


@dataclass(slots=True)
class _ProgressState:
    stage: str = "input_validation"
    provider_request_count: int = 0
    tool_invocation_count: int = 0

    def update(self, event: str) -> None:
        if event == "provider_request_started":
            self.provider_request_count += 1
            if self.provider_request_count == 1:
                self.stage = "first_provider_request"
            else:
                self.stage = "second_provider_request"
        elif event == "provider_response_received":
            if self.provider_request_count == 1:
                self.stage = "first_provider_response_validation"
            else:
                self.stage = "final_response_validation"
        elif event == "tool_call_validation":
            self.stage = "tool_call_validation"
        elif event == "tool_execution":
            self.tool_invocation_count += 1
            self.stage = "tool_execution"
        elif event == "tool_result_serialization":
            self.stage = "tool_result_serialization"


def run_copilot_turn_with_failure_observability(
    client: object,
    *,
    model: str,
    question: str,
    experiment_context: dict[str, object] | None = None,
    turn_timeout_seconds: float | None = None,
    on_failure: Callable[[CopilotFailureObservation], None],
    **request_options: object,
) -> CopilotObservedResult:
    """Run one bounded turn and report minimal facts if it fails."""
    if not callable(on_failure):
        raise TypeError("on_failure must be callable")

    progress = _ProgressState()
    start = _perf_counter()
    try:
        turn = _run_copilot_turn_with_result(
            client,
            progress.update,
            model=model,
            question=question,
            experiment_context=experiment_context,
            turn_timeout_seconds=turn_timeout_seconds,
            **request_options,
        )
    except BaseException:
        finish = _perf_counter()
        observation = CopilotFailureObservation(
            stage=progress.stage,
            provider_request_count=progress.provider_request_count,
            tool_invocation_count=progress.tool_invocation_count,
            elapsed_seconds=finish - start,
        )
        try:
            on_failure(observation)
        except BaseException:
            pass
        raise

    finish = _perf_counter()
    return CopilotObservedResult(
        turn=turn,
        metrics=CopilotRuntimeMetrics(
            provider_request_count=progress.provider_request_count,
            tool_invocation_count=progress.tool_invocation_count,
            elapsed_seconds=finish - start,
        ),
    )

"""Minimal observability for one successful Copilot turn."""

from dataclasses import dataclass
from time import perf_counter as _perf_counter

from .runtime_result import run_copilot_turn_with_result
from .session import CopilotTurn


__all__ = (
    "CopilotRuntimeMetrics",
    "CopilotObservedResult",
    "run_copilot_turn_with_observability",
)


@dataclass(frozen=True, slots=True)
class CopilotRuntimeMetrics:
    """Minimal metrics for one successful Copilot turn."""

    provider_request_count: int
    tool_invocation_count: int
    elapsed_seconds: float


@dataclass(frozen=True, slots=True)
class CopilotObservedResult:
    """A structured Copilot turn and its runtime metrics."""

    turn: CopilotTurn
    metrics: CopilotRuntimeMetrics


def run_copilot_turn_with_observability(
    client: object,
    *,
    model: str,
    question: str,
    experiment_context: dict[str, object] | None = None,
    **request_options: object,
) -> CopilotObservedResult:
    """Run one bounded Copilot turn and report minimal success metrics."""
    start = _perf_counter()
    turn = run_copilot_turn_with_result(
        client,
        model=model,
        question=question,
        experiment_context=experiment_context,
        **request_options,
    )
    finish = _perf_counter()
    tool_count = len(turn.tool_invocations)
    provider_count = 1 if tool_count == 0 else 2
    metrics = CopilotRuntimeMetrics(
        provider_request_count=provider_count,
        tool_invocation_count=tool_count,
        elapsed_seconds=finish - start,
    )
    return CopilotObservedResult(
        turn=turn,
        metrics=metrics,
    )

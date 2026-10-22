"""Minimal observability for one successful Copilot turn."""

from dataclasses import dataclass
from time import perf_counter as _perf_counter
from typing import Callable
from uuid import uuid4 as _uuid4

import copilot as _copilot

from .run_metadata import (
    CopilotProviderUsage,
    CopilotRunEvent,
    CopilotRunMetadata,
    _aggregate_run_usage,
)
from .runtime_result import (
    _capture_successful_turn_evidence,
    _run_copilot_turn_with_result,
    run_copilot_turn_with_result,
)
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
    run: CopilotRunMetadata | None = None


def _build_success_events(
    run_id: str,
    turn: CopilotTurn,
    *,
    provider_request_count: int,
    provider_response_count: int,
    provider_usages: tuple[CopilotProviderUsage, ...],
) -> tuple[CopilotRunEvent, ...]:
    """Describe a completed successful turn without retaining its payloads."""
    events: list[CopilotRunEvent] = []

    def emit(
        kind: str,
        *,
        usage: CopilotProviderUsage | None = None,
        tool_index: int | None = None,
    ) -> None:
        invocation = (
            None if tool_index is None else turn.tool_invocations[tool_index]
        )
        events.append(
            CopilotRunEvent(
                run_id=run_id,
                sequence=len(events),
                kind=kind,
                tool_invocation_index=tool_index,
                tool_call_id=(
                    None if invocation is None else invocation.tool_call_id
                ),
                tool_name=None if invocation is None else invocation.tool_name,
                usage=usage,
            )
        )

    emit("run.started")
    for request_index in range(provider_request_count):
        emit("provider.request.started")
        if request_index < provider_response_count:
            usage = (
                provider_usages[request_index]
                if request_index < len(provider_usages)
                else None
            )
            emit("provider.response.received", usage=usage)
            if request_index == 0 and turn.tool_invocations:
                emit("tool.calls.validation.started")
                for tool_index in range(len(turn.tool_invocations)):
                    emit("tool.execution.started", tool_index=tool_index)
                    emit("tool.result.accepted", tool_index=tool_index)
    emit("run.completed")
    return tuple(events)


def run_copilot_turn_with_observability(
    client: object,
    *,
    model: str,
    question: str,
    experiment_context: dict[str, object] | None = None,
    turn_timeout_seconds: float | None = None,
    **request_options: object,
) -> CopilotObservedResult:
    """Run one bounded Copilot turn and report minimal success metrics."""
    return _run_copilot_turn_with_observability(
        client,
        model=model,
        question=question,
        experiment_context=experiment_context,
        turn_timeout_seconds=turn_timeout_seconds,
        **request_options,
    )


def _run_copilot_turn_with_observability(
    client: object,
    message_builder: Callable[[str], list[dict]] | None = None,
    on_failure: "Callable[[_copilot.CopilotFailureObservation], None] | None" = None,
    /,
    *,
    model: str,
    question: str,
    experiment_context: dict[str, object] | None = None,
    turn_timeout_seconds: float | None = None,
    **request_options: object,
) -> CopilotObservedResult:
    """Observe a turn, optionally using Session-prepared messages."""
    progress = None
    if on_failure is not None:
        if not callable(on_failure):
            raise TypeError("on_failure must be callable")
        # Failure models depend on this module; import only during execution.
        from .failure_observability import _ProgressState, _notify_failure

        progress = _ProgressState()

    run_id = str(_uuid4())
    start = _perf_counter()
    with _capture_successful_turn_evidence() as evidence:
        try:
            if message_builder is None and progress is None:
                turn = run_copilot_turn_with_result(
                    client,
                    model=model,
                    question=question,
                    experiment_context=experiment_context,
                    turn_timeout_seconds=turn_timeout_seconds,
                    **request_options,
                )
            else:
                turn = _run_copilot_turn_with_result(
                    client,
                    None if progress is None else progress.update,
                    message_builder,
                    model=model,
                    question=question,
                    experiment_context=experiment_context,
                    turn_timeout_seconds=turn_timeout_seconds,
                    **request_options,
                )
        except BaseException:
            if on_failure is not None and progress is not None:
                _notify_failure(
                    on_failure,
                    progress=progress,
                    run_id=run_id,
                    elapsed_seconds=_perf_counter() - start,
                    evidence=evidence[0],
                )
            raise
    finish = _perf_counter()
    tool_count = len(turn.tool_invocations)
    provider_count = 1 if tool_count == 0 else 2
    metrics = CopilotRuntimeMetrics(
        provider_request_count=provider_count,
        tool_invocation_count=tool_count,
        elapsed_seconds=finish - start,
    )
    provider_request_count, provider_response_count, provider_usages = evidence[0]
    return CopilotObservedResult(
        turn=turn,
        metrics=metrics,
        run=CopilotRunMetadata(
            run_id=run_id,
            usage=_aggregate_run_usage(
                provider_usages,
                provider_request_count=provider_request_count,
                provider_response_count=provider_response_count,
                terminal_success=True,
            ),
            events=_build_success_events(
                run_id,
                turn,
                provider_request_count=provider_request_count,
                provider_response_count=provider_response_count,
                provider_usages=provider_usages,
            ),
        ),
    )

"""Failure observability for one bounded Copilot turn."""

from dataclasses import dataclass
from time import perf_counter as _perf_counter
from typing import Callable
from uuid import uuid4 as _uuid4

from .run_metadata import (
    CopilotProviderUsage,
    CopilotRunEvent,
    CopilotRunMetadata,
    _aggregate_run_usage,
)
from .runtime_observability import (
    CopilotObservedResult,
    CopilotRuntimeMetrics,
)
from .runtime_result import (
    _capture_successful_turn_evidence,
    _run_copilot_turn_with_result,
)


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
    run: CopilotRunMetadata | None = None


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


def _notify_failure(
    on_failure: Callable[[CopilotFailureObservation], None],
    *,
    progress: _ProgressState,
    run_id: str,
    elapsed_seconds: float,
    evidence: tuple[int, int, tuple[CopilotProviderUsage, ...]],
) -> None:
    """Publish Runtime-owned failure evidence without masking the failure."""
    provider_request_count, provider_response_count, provider_usages = evidence
    events = [
        CopilotRunEvent(
            run_id=run_id,
            sequence=0,
            kind="run.started",
        )
    ]
    for usage in provider_usages:
        events.append(
            CopilotRunEvent(
                run_id=run_id,
                sequence=len(events),
                kind="provider.response.received",
                usage=usage,
            )
        )
    events.append(
        CopilotRunEvent(
            run_id=run_id,
            sequence=len(events),
            kind="run.failed",
            failure_stage=progress.stage,
        )
    )
    run = CopilotRunMetadata(
        run_id=run_id,
        usage=_aggregate_run_usage(
            provider_usages,
            provider_request_count=provider_request_count,
            provider_response_count=provider_response_count,
            terminal_success=False,
        ),
        events=tuple(events),
    )
    observation = CopilotFailureObservation(
        stage=progress.stage,
        provider_request_count=progress.provider_request_count,
        tool_invocation_count=progress.tool_invocation_count,
        elapsed_seconds=elapsed_seconds,
        run=run,
    )
    try:
        on_failure(observation)
    except BaseException:
        pass


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

    run_id = str(_uuid4())
    progress = _ProgressState()
    start = _perf_counter()
    with _capture_successful_turn_evidence() as evidence:
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
            _notify_failure(
                on_failure,
                progress=progress,
                run_id=run_id,
                elapsed_seconds=finish - start,
                evidence=evidence[0],
            )
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

"""Structured result for one bounded Copilot turn."""

from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from typing import Callable

from llm_adapters.openai_tool_adapter import (
    _ProviderEvidence,
    _experiment_path_policy_scope,
    _run_tool_call_cycle_with_trace,
)
from llm_adapters.turn_deadline import (
    _check_turn_deadline,
    _turn_deadline_scope,
    _validate_turn_deadline_options,
)
from tool_layer.experiment_path_security import (
    _build_experiment_path_policy,
)

from .runtime import (
    _build_messages,
    _extract_final_content,
    _validate_context,
    _validate_question,
    _validate_request_options,
)
from .run_metadata import CopilotProviderUsage
from .session import CopilotTurn, _normalize_tool_trace


_SUCCESSFUL_TURN_EVIDENCE: ContextVar[
    list[tuple[int, int, tuple[CopilotProviderUsage, ...]]] | None
] = ContextVar("successful_turn_evidence", default=None)


@contextmanager
def _capture_successful_turn_evidence() -> Iterator[
    list[tuple[int, int, tuple[CopilotProviderUsage, ...]]]
]:
    """Collect evidence without changing the public CopilotTurn projection."""
    # One slot carries the completed (request count, response count, usages).
    evidence: list[tuple[int, int, tuple[CopilotProviderUsage, ...]]] = [
        (0, 0, ())
    ]
    token = _SUCCESSFUL_TURN_EVIDENCE.set(evidence)
    try:
        yield evidence
    finally:
        _SUCCESSFUL_TURN_EVIDENCE.reset(token)


def run_copilot_turn_with_result(
    client: object,
    *,
    model: str,
    question: str,
    experiment_context: dict[str, object] | None = None,
    turn_timeout_seconds: float | None = None,
    **request_options: object,
) -> CopilotTurn:
    """Run one bounded Copilot turn and return its structured result."""
    return _run_copilot_turn_with_result(
        client,
        None,
        model=model,
        question=question,
        experiment_context=experiment_context,
        turn_timeout_seconds=turn_timeout_seconds,
        **request_options,
    )


def _run_copilot_turn_with_result(
    client: object,
    progress_callback: Callable[[str], None] | None = None,
    message_builder: Callable[[str], list[dict]] | None = None,
    /,
    *,
    model: str,
    question: str,
    experiment_context: dict[str, object] | None = None,
    turn_timeout_seconds: float | None = None,
    **request_options: object,
) -> CopilotTurn:
    validated_question = _validate_question(question)
    validated_context = _validate_context(experiment_context)
    _validate_request_options(request_options)
    validated_timeout = _validate_turn_deadline_options(
        turn_timeout_seconds,
        request_options,
    )
    messages = (
        _build_messages(validated_question, validated_context)
        if message_builder is None
        else message_builder(validated_question)
    )
    path_policy = _build_experiment_path_policy(validated_context)
    with _turn_deadline_scope(validated_timeout):
        with _experiment_path_policy_scope(path_policy):
            captured_evidence = _SUCCESSFUL_TURN_EVIDENCE.get()
            provider_evidence = (
                _ProviderEvidence()
                if captured_evidence is not None
                else None
            )
            try:
                trace = _run_tool_call_cycle_with_trace(
                    client,
                    progress_callback,
                    provider_evidence,
                    model=model,
                    messages=messages,
                    **request_options,
                )
            finally:
                if (
                    captured_evidence is not None
                    and provider_evidence is not None
                ):
                    captured_evidence[0] = (
                        provider_evidence.provider_request_count,
                        provider_evidence.provider_response_count,
                        tuple(
                            CopilotProviderUsage(
                                input_tokens=usage.input_tokens,
                                output_tokens=usage.output_tokens,
                                total_tokens=usage.total_tokens,
                            )
                            for usage in provider_evidence.provider_usages
                        ),
                    )
        answer = _extract_final_content(trace.response)
        tool_call_content, tool_invocations = _normalize_tool_trace(
            trace.assistant_message,
            trace.tool_messages,
        )
        result = CopilotTurn(
            question=validated_question,
            answer=answer,
            tool_call_content=tool_call_content,
            tool_invocations=tool_invocations,
        )
        _check_turn_deadline()
        evidence = _SUCCESSFUL_TURN_EVIDENCE.get()
        if evidence is not None:
            evidence[0] = (
                trace.provider_request_count,
                trace.provider_response_count,
                tuple(
                    CopilotProviderUsage(
                        input_tokens=usage.input_tokens,
                        output_tokens=usage.output_tokens,
                        total_tokens=usage.total_tokens,
                    )
                    for usage in trace.provider_usages
                ),
            )
        return result

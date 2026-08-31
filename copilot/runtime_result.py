"""Structured result for one bounded Copilot turn."""

from typing import Callable

from llm_adapters.openai_tool_adapter import (
    _experiment_path_policy_scope,
    _run_tool_call_cycle_with_trace,
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
from .session import CopilotTurn, _normalize_tool_trace


def run_copilot_turn_with_result(
    client: object,
    *,
    model: str,
    question: str,
    experiment_context: dict[str, object] | None = None,
    **request_options: object,
) -> CopilotTurn:
    """Run one bounded Copilot turn and return its structured result."""
    return _run_copilot_turn_with_result(
        client,
        None,
        model=model,
        question=question,
        experiment_context=experiment_context,
        **request_options,
    )


def _run_copilot_turn_with_result(
    client: object,
    progress_callback: Callable[[str], None] | None = None,
    /,
    *,
    model: str,
    question: str,
    experiment_context: dict[str, object] | None = None,
    **request_options: object,
) -> CopilotTurn:
    validated_question = _validate_question(question)
    validated_context = _validate_context(experiment_context)
    _validate_request_options(request_options)
    messages = _build_messages(
        validated_question,
        validated_context,
    )
    path_policy = _build_experiment_path_policy(validated_context)
    with _experiment_path_policy_scope(path_policy):
        trace = _run_tool_call_cycle_with_trace(
            client,
            progress_callback,
            model=model,
            messages=messages,
            **request_options,
        )
    answer = _extract_final_content(trace.response)
    tool_call_content, tool_invocations = _normalize_tool_trace(
        trace.assistant_message,
        trace.tool_messages,
    )
    return CopilotTurn(
        question=validated_question,
        answer=answer,
        tool_call_content=tool_call_content,
        tool_invocations=tool_invocations,
    )

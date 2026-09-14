"""Component RED contracts for M9 failure run observability."""

from __future__ import annotations

from dataclasses import fields, is_dataclass
import json
from types import SimpleNamespace
from typing import get_type_hints
from uuid import UUID

import pytest

from copilot import (
    CopilotFailureObservation,
    CopilotRunMetadata,
    run_copilot_turn_with_failure_observability,
)
import llm_adapters.openai_tool_adapter as adapter


_MISSING = object()
FAILURE_STAGES = (
    "input_validation",
    "first_provider_request",
    "first_provider_response_validation",
    "tool_call_validation",
    "tool_execution",
    "tool_result_serialization",
    "second_provider_request",
    "final_response_validation",
)
EXPECTED_USAGE = {
    "input_validation": (0, 0, 0),
    "first_provider_request": (1, 0, 0),
    "first_provider_response_validation": (1, 1, 0),
    "tool_call_validation": (1, 1, 0),
    "tool_execution": (1, 1, 0),
    "tool_result_serialization": (1, 1, 0),
    "second_provider_request": (2, 1, 1),
    "final_response_validation": (2, 2, 0),
}
EVENT_KINDS = {
    "run.started",
    "provider.request.started",
    "provider.response.received",
    "tool.calls.validation.started",
    "tool.execution.started",
    "tool.result.accepted",
    "run.completed",
    "run.failed",
}


class _SequentialCompletions:
    def __init__(self, outcomes: list[object]) -> None:
        self.outcomes = list(outcomes)
        self.calls: list[dict[str, object]] = []

    def create(self, **kwargs: object) -> object:
        self.calls.append(kwargs)
        index = len(self.calls) - 1
        if index >= len(self.outcomes):
            raise AssertionError("unexpected extra provider request")
        outcome = self.outcomes[index]
        if isinstance(outcome, BaseException):
            raise outcome
        return outcome


class _FakeClient:
    def __init__(self, outcomes: list[object]) -> None:
        self.completions = _SequentialCompletions(outcomes)
        self.chat = SimpleNamespace(completions=self.completions)
        self.close_calls = 0

    def close(self) -> None:
        self.close_calls += 1


def _usage(
    input_tokens: int = 11,
    output_tokens: int = 2,
    total_tokens: int = 13,
) -> SimpleNamespace:
    return SimpleNamespace(
        prompt_tokens=input_tokens,
        completion_tokens=output_tokens,
        total_tokens=total_tokens,
    )


def _tool_call(*, arguments: str | None = None) -> SimpleNamespace:
    return SimpleNamespace(
        id="call_0",
        type="function",
        function=SimpleNamespace(
            name="tool_0",
            arguments=(
                arguments
                if arguments is not None
                else json.dumps({"value": "SECRET_TOOL_ARGUMENT"})
            ),
        ),
    )


def _response(
    tool_calls: object = None,
    *,
    content: object = "final answer",
    usage: object = _MISSING,
) -> SimpleNamespace:
    response = SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(
                    role="assistant",
                    content=content,
                    tool_calls=tool_calls,
                )
            )
        ],
        provider_payload="SECRET_RAW_PROVIDER_RESPONSE",
    )
    if usage is not _MISSING:
        response.usage = usage
    return response


def _install_tool(
    monkeypatch: pytest.MonkeyPatch,
    outcome: object = _MISSING,
) -> None:
    def invoke(name: str, arguments: dict) -> object:
        del name, arguments
        if isinstance(outcome, BaseException):
            raise outcome
        if outcome is not _MISSING:
            return outcome
        return {"value": "SECRET_TOOL_RESULT"}

    monkeypatch.setattr(adapter, "invoke_tool", invoke)


def _failure_case(
    stage: str,
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[_FakeClient, dict[str, object]]:
    options: dict[str, object] = {
        "question": "SECRET_PROMPT",
        "api_key": "SECRET_CREDENTIAL",
        "diagnostic_path": r"C:\SECRET\experiment",
    }
    if stage == "input_validation":
        options["question"] = ""
        client = _FakeClient([])
    elif stage == "first_provider_request":
        client = _FakeClient([RuntimeError("SECRET_PROVIDER_EXCEPTION")])
    elif stage == "first_provider_response_validation":
        client = _FakeClient([SimpleNamespace(choices=[])])
    elif stage == "tool_call_validation":
        client = _FakeClient(
            [_response([_tool_call(arguments="not-json")])]
        )
    elif stage == "tool_execution":
        _install_tool(
            monkeypatch,
            RuntimeError("SECRET_TOOL_EXECUTION_EXCEPTION"),
        )
        client = _FakeClient([_response([_tool_call()])])
    elif stage == "tool_result_serialization":
        _install_tool(
            monkeypatch,
            {"secret": "SECRET_TOOL_RESULT", "invalid": object()},
        )
        client = _FakeClient([_response([_tool_call()])])
    elif stage == "second_provider_request":
        _install_tool(monkeypatch)
        client = _FakeClient(
            [
                _response([_tool_call()], usage=_usage()),
                RuntimeError("SECRET_SECOND_PROVIDER_EXCEPTION"),
            ]
        )
    elif stage == "final_response_validation":
        _install_tool(monkeypatch)
        client = _FakeClient(
            [
                _response([_tool_call()]),
                _response(None, content=None),
            ]
        )
    else:
        raise AssertionError(f"unknown failure stage: {stage}")
    return client, options


def _run_failure(
    client: object,
    options: dict[str, object],
    callback: object,
) -> BaseException:
    arguments = dict(options)
    question = arguments.pop("question")
    with pytest.raises(Exception) as caught:
        run_copilot_turn_with_failure_observability(
            client,
            model="test-model",
            question=question,
            on_failure=callback,
            **arguments,
        )
    return caught.value


def _populated_run(observation: CopilotFailureObservation) -> CopilotRunMetadata:
    assert hasattr(observation, "run"), (
        "M9 failure observations require additive run metadata"
    )
    run = observation.run
    assert run is not None, "runtime-produced failure metadata must be populated"
    assert type(run) is CopilotRunMetadata
    return run


def _assert_uuid4(value: object) -> str:
    assert type(value) is str
    assert value == value.lower()
    parsed = UUID(value)
    assert parsed.version == 4
    assert str(parsed) == value
    return value


def _assert_minimal_failed_run(
    observation: CopilotFailureObservation,
) -> CopilotRunMetadata:
    run = _populated_run(observation)
    run_id = _assert_uuid4(run.run_id)
    assert [event.sequence for event in run.events] == list(
        range(len(run.events))
    )
    assert run.events[0].kind == "run.started"
    assert run.events[-1].kind == "run.failed"
    assert sum(
        event.kind in {"run.completed", "run.failed"}
        for event in run.events
    ) == 1
    assert {event.run_id for event in run.events} == {run_id}
    assert {event.kind for event in run.events} <= EVENT_KINDS
    assert run.events[-1].failure_stage == observation.stage
    assert all(not hasattr(event, "timestamp") for event in run.events)
    terminal = run.events[-1]
    assert terminal.provider_request_index is None
    assert terminal.tool_invocation_index is None
    assert terminal.tool_call_id is None
    assert terminal.tool_name is None
    assert terminal.usage is None
    assert "SECRET_" not in repr(run)
    return run


def test_failure_observation_run_field_is_additive_and_optional() -> None:
    assert is_dataclass(CopilotFailureObservation)
    assert CopilotFailureObservation.__dataclass_params__.frozen is True
    assert tuple(item.name for item in fields(CopilotFailureObservation)) == (
        "stage",
        "provider_request_count",
        "tool_invocation_count",
        "elapsed_seconds",
        "run",
    )
    assert tuple(CopilotFailureObservation.__slots__) == (
        "stage",
        "provider_request_count",
        "tool_invocation_count",
        "elapsed_seconds",
        "run",
    )
    assert get_type_hints(CopilotFailureObservation) == {
        "stage": str,
        "provider_request_count": int,
        "tool_invocation_count": int,
        "elapsed_seconds": float,
        "run": CopilotRunMetadata | None,
    }
    legacy = CopilotFailureObservation(
        "input_validation",
        0,
        0,
        0.0,
    )
    assert legacy.run is None


@pytest.mark.parametrize("stage", FAILURE_STAGES)
def test_failure_run_is_minimal_correlated_and_conservative(
    monkeypatch: pytest.MonkeyPatch,
    stage: str,
) -> None:
    client, options = _failure_case(stage, monkeypatch)
    observations: list[CopilotFailureObservation] = []
    _run_failure(client, options, observations.append)
    assert len(observations) == 1
    observation = observations[0]
    assert observation.stage == stage
    if stage == "tool_execution":
        assert observation.tool_invocation_count == 1

    run = _assert_minimal_failed_run(observation)
    expected_requests, expected_responses, expected_reports = EXPECTED_USAGE[
        stage
    ]
    assert run.usage.provider_request_count == expected_requests
    assert run.usage.provider_response_count == expected_responses
    assert run.usage.usage_report_count == expected_reports
    assert run.usage.input_tokens is None
    assert run.usage.output_tokens is None
    assert run.usage.total_tokens is None
    assert run.usage.complete is False
    assert client.close_calls == 0


def test_second_provider_request_failure_preserves_partial_usage_evidence(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_tool(monkeypatch)
    original = RuntimeError("SECRET_SECOND_PROVIDER_EXCEPTION")
    client = _FakeClient(
        [
            _response([_tool_call()], usage=_usage(11, 2, 13)),
            original,
        ]
    )
    observations: list[CopilotFailureObservation] = []
    caught = _run_failure(
        client,
        {
            "question": "SECRET_PROMPT",
            "api_key": "SECRET_CREDENTIAL",
        },
        observations.append,
    )
    assert caught is original
    assert len(observations) == 1
    run = _assert_minimal_failed_run(observations[0])
    assert run.usage.provider_request_count == 2
    assert run.usage.provider_response_count == 1
    assert run.usage.usage_report_count == 1
    assert run.usage.input_tokens is None
    assert run.usage.output_tokens is None
    assert run.usage.total_tokens is None
    assert run.usage.complete is False
    received = [
        event
        for event in run.events
        if event.kind == "provider.response.received"
    ]
    assert len(received) == 1
    assert received[0].usage is not None
    assert received[0].usage.input_tokens == 11
    assert received[0].usage.output_tokens == 2
    assert received[0].usage.total_tokens == 13
    assert client.close_calls == 0


def test_failure_callback_preserves_original_exception_and_run_metadata() -> None:
    original = RuntimeError("SECRET_ORIGINAL_EXCEPTION")
    client = _FakeClient([original])
    observations: list[CopilotFailureObservation] = []
    callback_runs: list[object] = []

    def callback(observation: CopilotFailureObservation) -> None:
        observations.append(observation)
        callback_runs.append(getattr(observation, "run", _MISSING))
        raise KeyboardInterrupt("SECRET_CALLBACK_EXCEPTION")

    caught = _run_failure(
        client,
        {
            "question": "SECRET_PROMPT",
            "api_key": "SECRET_CREDENTIAL",
        },
        callback,
    )
    assert caught is original
    assert len(observations) == 1
    run = _assert_minimal_failed_run(observations[0])
    assert callback_runs == [run]
    assert run.events[-1].failure_stage == "first_provider_request"
    assert client.close_calls == 0

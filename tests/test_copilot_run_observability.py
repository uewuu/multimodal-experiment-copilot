"""RED contracts for M9 run identity, usage, and lifecycle events."""

from __future__ import annotations

from dataclasses import fields, is_dataclass
import inspect
import json
from types import SimpleNamespace
from typing import get_type_hints
from uuid import UUID

import pytest

import copilot
from copilot import (
    CopilotFailureObservation,
    CopilotObservedResult,
    CopilotService,
    CopilotSession,
    CopilotTurn,
    run_copilot_turn,
    run_copilot_turn_with_failure_observability,
    run_copilot_turn_with_observability,
    run_copilot_turn_with_result,
)
import llm_adapters.openai_tool_adapter as adapter


MODEL_FIELDS = {
    "CopilotProviderUsage": (
        "input_tokens",
        "output_tokens",
        "total_tokens",
    ),
    "CopilotRunUsage": (
        "input_tokens",
        "output_tokens",
        "total_tokens",
        "provider_request_count",
        "provider_response_count",
        "usage_report_count",
        "complete",
    ),
    "CopilotRunEvent": (
        "run_id",
        "sequence",
        "kind",
        "provider_request_index",
        "tool_invocation_index",
        "tool_call_id",
        "tool_name",
        "usage",
        "failure_stage",
    ),
    "CopilotRunMetadata": (
        "run_id",
        "usage",
        "events",
    ),
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
_MISSING = object()


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
    *,
    input_tokens: object = 10,
    output_tokens: object = 5,
    total_tokens: object = 15,
) -> SimpleNamespace:
    return SimpleNamespace(
        prompt_tokens=input_tokens,
        completion_tokens=output_tokens,
        total_tokens=total_tokens,
    )


def _tool_call(
    index: int,
    *,
    name: object | None = None,
    arguments: object | None = None,
) -> SimpleNamespace:
    return SimpleNamespace(
        id=f"call_{index}",
        type="function",
        function=SimpleNamespace(
            name=name if name is not None else f"tool_{index}",
            arguments=(
                arguments
                if arguments is not None
                else json.dumps({"index": index})
            ),
        ),
    )


def _response(
    tool_calls: object = None,
    *,
    content: object = "final answer",
    role: object = "assistant",
    usage: object = _MISSING,
    provider_run_id: str | None = None,
) -> SimpleNamespace:
    response = SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(
                    role=role,
                    content=content,
                    tool_calls=tool_calls,
                )
            )
        ]
    )
    if usage is not _MISSING:
        response.usage = usage
    if provider_run_id is not None:
        response.run_id = provider_run_id
    return response


def _run_observed(client: object, **options: object) -> CopilotObservedResult:
    return run_copilot_turn_with_observability(
        client,
        model="test-model",
        question="analyze experiments",
        **options,
    )


def _run_metadata(value: object) -> object:
    assert hasattr(value, "run"), "M9 requires additive run metadata"
    run = getattr(value, "run")
    assert run is not None, "runtime-produced M9 metadata must be populated"
    return run


def _assert_uuid4(value: object) -> str:
    assert type(value) is str
    assert value == value.lower()
    parsed = UUID(value)
    assert parsed.version == 4
    assert str(parsed) == value
    return value


def _event_kinds(run: object) -> list[str]:
    return [event.kind for event in run.events]


def _assert_event_invariants(run: object, terminal: str) -> None:
    run_id = _assert_uuid4(run.run_id)
    assert [event.sequence for event in run.events] == list(
        range(len(run.events))
    )
    assert run.events[0].kind == "run.started"
    assert run.events[-1].kind == terminal
    assert sum(
        event.kind in {"run.completed", "run.failed"}
        for event in run.events
    ) == 1
    assert {event.run_id for event in run.events} == {run_id}
    assert {event.kind for event in run.events} <= EVENT_KINDS
    assert all(not hasattr(event, "timestamp") for event in run.events)


def _install_tool(
    monkeypatch: pytest.MonkeyPatch,
    outcome: object = _MISSING,
) -> list[tuple[str, dict]]:
    calls: list[tuple[str, dict]] = []

    def invoke(name: str, arguments: dict) -> object:
        calls.append((name, arguments))
        if isinstance(outcome, BaseException):
            raise outcome
        if outcome is not _MISSING:
            return outcome
        return {"accepted": arguments.get("index")}

    monkeypatch.setattr(adapter, "invoke_tool", invoke)
    return calls


def _failure_case(
    scenario: str,
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[_FakeClient, dict[str, object], str]:
    options: dict[str, object] = {}
    if scenario == "input_validation":
        client = _FakeClient([])
        options["question"] = ""
    elif scenario == "first_provider_request":
        client = _FakeClient([RuntimeError("provider secret")])
    elif scenario == "first_provider_response_validation":
        client = _FakeClient([SimpleNamespace(choices=[])])
    elif scenario == "tool_call_validation":
        client = _FakeClient(
            [_response([_tool_call(0, arguments="not-json")])]
        )
    elif scenario == "tool_execution":
        _install_tool(monkeypatch, RuntimeError("tool secret"))
        client = _FakeClient([_response([_tool_call(0)])])
    elif scenario == "tool_result_serialization":
        _install_tool(monkeypatch, {"invalid": object()})
        client = _FakeClient([_response([_tool_call(0)])])
    elif scenario == "second_provider_request":
        _install_tool(monkeypatch)
        client = _FakeClient(
            [
                _response(
                    [_tool_call(0)],
                    usage=_usage(),
                ),
                RuntimeError("second provider secret"),
            ]
        )
    elif scenario == "final_response_validation":
        _install_tool(monkeypatch)
        client = _FakeClient(
            [
                _response([_tool_call(0)]),
                _response(None, content=None),
            ]
        )
    else:
        raise AssertionError(f"unknown failure scenario: {scenario}")
    return client, options, scenario


def test_run_metadata_models_are_public_frozen_slotted_contracts() -> None:
    for name, expected_fields in MODEL_FIELDS.items():
        assert hasattr(copilot, name), f"M9 requires public {name}"
        model = getattr(copilot, name)
        assert name in copilot.__all__
        assert is_dataclass(model)
        assert model.__dataclass_params__.frozen is True
        assert tuple(item.name for item in fields(model)) == expected_fields
        assert tuple(model.__slots__) == expected_fields
        assert "__dict__" not in model.__slots__


def test_observed_and_failure_results_gain_only_additive_run_metadata() -> None:
    observed_fields = tuple(
        item.name for item in fields(CopilotObservedResult)
    )
    failure_fields = tuple(
        item.name for item in fields(CopilotFailureObservation)
    )
    assert observed_fields == ("turn", "metrics", "run")
    assert failure_fields == (
        "stage",
        "provider_request_count",
        "tool_invocation_count",
        "elapsed_seconds",
        "run",
    )
    observed_run = fields(CopilotObservedResult)[-1]
    failure_run = fields(CopilotFailureObservation)[-1]
    assert observed_run.default is None
    assert failure_run.default is None


def test_legacy_projections_and_copilot_turn_contract_remain_unchanged() -> None:
    assert tuple(item.name for item in fields(CopilotTurn)) == (
        "question",
        "answer",
        "tool_call_content",
        "tool_invocations",
    )
    assert not hasattr(CopilotTurn("q", "a", None, ()), "run")
    assert get_type_hints(run_copilot_turn)["return"] is str
    assert get_type_hints(run_copilot_turn_with_result)["return"] is (
        CopilotTurn
    )
    assert get_type_hints(CopilotSession.ask)["return"] is str
    assert get_type_hints(CopilotSession.ask_with_result)["return"] is (
        CopilotTurn
    )
    assert get_type_hints(CopilotService.run)["return"] is (
        CopilotObservedResult
    )
    assert "run_id" not in inspect.signature(run_copilot_turn).parameters
    assert "run_id" not in inspect.signature(CopilotService.run).parameters


def test_no_tool_runs_have_unique_canonical_runtime_owned_ids() -> None:
    first = _run_metadata(
        _run_observed(_FakeClient([_response(usage=_usage())]))
    )
    second = _run_metadata(
        _run_observed(_FakeClient([_response(usage=_usage())]))
    )
    assert _assert_uuid4(first.run_id) != _assert_uuid4(second.run_id)


def test_caller_provider_and_tool_cannot_choose_run_id(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    forbidden_ids = {
        "caller-controlled",
        "provider-controlled",
        "tool-controlled",
    }
    caller_client = _FakeClient([_response(usage=_usage())])
    try:
        caller_result = _run_observed(
            caller_client,
            run_id="caller-controlled",
        )
    except (TypeError, ValueError):
        assert caller_client.completions.calls == []
    else:
        caller_run = _run_metadata(caller_result)
        assert _assert_uuid4(caller_run.run_id) != "caller-controlled"

    _install_tool(monkeypatch, {"run_id": "tool-controlled"})
    provider_tool_client = _FakeClient(
        [
            _response(
                [_tool_call(0)],
                provider_run_id="provider-controlled",
            ),
            _response(None),
        ]
    )
    result = _run_observed(provider_tool_client)
    run = _run_metadata(result)
    assert _assert_uuid4(run.run_id) not in forbidden_ids
    assert {event.run_id for event in run.events}.isdisjoint(forbidden_ids)


def test_single_response_usage_and_no_tool_event_order() -> None:
    result = _run_observed(
        _FakeClient(
            [
                _response(
                    usage=_usage(
                        input_tokens=100,
                        output_tokens=20,
                        total_tokens=120,
                    )
                )
            ]
        )
    )
    run = _run_metadata(result)
    assert run.usage.input_tokens == 100
    assert run.usage.output_tokens == 20
    assert run.usage.total_tokens == 120
    assert run.usage.provider_request_count == 1
    assert run.usage.provider_response_count == 1
    assert run.usage.usage_report_count == 1
    assert run.usage.complete is True
    assert _event_kinds(run) == [
        "run.started",
        "provider.request.started",
        "provider.response.received",
        "run.completed",
    ]
    _assert_event_invariants(run, "run.completed")


@pytest.mark.parametrize(
    ("provider_usage", "expected", "report_count"),
    [
        (_MISSING, (None, None, None), 0),
        (None, (None, None, None), 0),
        (_usage(input_tokens="10", output_tokens="5", total_tokens="15"),
         (None, None, None), 0),
        (_usage(input_tokens=-1, output_tokens=-2, total_tokens=-3),
         (None, None, None), 0),
        (_usage(input_tokens=True, output_tokens=False, total_tokens=True),
         (None, None, None), 0),
        (_usage(input_tokens=10, output_tokens=None, total_tokens=None),
         (10, None, None), 1),
        (_usage(input_tokens=10, output_tokens=5, total_tokens="bad"),
         (10, 5, None), 1),
    ],
    ids=(
        "missing",
        "none",
        "malformed",
        "negative",
        "bool",
        "partial",
        "field-level-invalid",
    ),
)
def test_usage_report_count_is_response_based_and_usage_is_conservative(
    provider_usage: object,
    expected: tuple[int | None, int | None, int | None],
    report_count: int,
) -> None:
    """Count responses having at least one valid normalized usage field."""
    response = (
        _response()
        if provider_usage is _MISSING
        else _response(usage=provider_usage)
    )
    result = _run_observed(_FakeClient([response]))
    run = _run_metadata(result)
    assert result.turn.answer == "final answer"
    assert (
        run.usage.input_tokens,
        run.usage.output_tokens,
        run.usage.total_tokens,
    ) == expected
    assert run.usage.usage_report_count == report_count
    assert run.usage.complete is False


def test_two_response_usage_uses_field_level_complete_aggregation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_tool(monkeypatch)
    client = _FakeClient(
        [
            _response(
                [_tool_call(0)],
                usage=_usage(
                    input_tokens=100,
                    output_tokens=20,
                    total_tokens=120,
                ),
            ),
            _response(
                None,
                usage=_usage(
                    input_tokens=50,
                    output_tokens=None,
                    total_tokens=None,
                ),
            ),
        ]
    )
    run = _run_metadata(_run_observed(client))
    assert run.usage.input_tokens == 150
    assert run.usage.output_tokens is None
    assert run.usage.total_tokens is None
    assert run.usage.provider_request_count == 2
    assert run.usage.provider_response_count == 2
    assert run.usage.usage_report_count == 2
    assert run.usage.complete is False


def test_failed_second_request_keeps_only_conservative_response_evidence(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_tool(monkeypatch)
    failure = RuntimeError("provider failure must not enter metadata")
    client = _FakeClient(
        [
            _response(
                [_tool_call(0)],
                usage=_usage(
                    input_tokens=11,
                    output_tokens=2,
                    total_tokens=13,
                ),
            ),
            failure,
        ]
    )
    observations: list[CopilotFailureObservation] = []
    with pytest.raises(RuntimeError) as caught:
        run_copilot_turn_with_failure_observability(
            client,
            model="test-model",
            question="question",
            on_failure=observations.append,
        )
    assert caught.value is failure
    run = _run_metadata(observations[0])
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
    assert received[0].usage.input_tokens == 11
    assert "provider failure" not in repr(run)


@pytest.mark.parametrize("tool_count", [1, 3])
def test_tool_event_order_is_deterministic(
    monkeypatch: pytest.MonkeyPatch,
    tool_count: int,
) -> None:
    calls = _install_tool(monkeypatch)
    client = _FakeClient(
        [
            _response([_tool_call(index) for index in range(tool_count)]),
            _response(None),
        ]
    )
    run = _run_metadata(_run_observed(client))
    expected = [
        "run.started",
        "provider.request.started",
        "provider.response.received",
        "tool.calls.validation.started",
    ]
    for _ in range(tool_count):
        expected.extend(
            ["tool.execution.started", "tool.result.accepted"]
        )
    expected.extend(
        [
            "provider.request.started",
            "provider.response.received",
            "run.completed",
        ]
    )
    assert _event_kinds(run) == expected
    tool_events = [
        event
        for event in run.events
        if event.kind == "tool.execution.started"
    ]
    assert [event.tool_invocation_index for event in tool_events] == list(
        range(tool_count)
    )
    assert [event.tool_call_id for event in tool_events] == [
        f"call_{index}" for index in range(tool_count)
    ]
    assert [event.tool_name for event in tool_events] == [
        f"tool_{index}" for index in range(tool_count)
    ]
    assert len(calls) == tool_count
    _assert_event_invariants(run, "run.completed")


@pytest.mark.parametrize("scenario", FAILURE_STAGES)
def test_every_existing_failure_stage_ends_one_correlated_run(
    monkeypatch: pytest.MonkeyPatch,
    scenario: str,
) -> None:
    client, overrides, expected_stage = _failure_case(
        scenario,
        monkeypatch,
    )
    observations: list[CopilotFailureObservation] = []
    arguments: dict[str, object] = {
        "model": "test-model",
        "question": "question",
        "on_failure": observations.append,
    }
    arguments.update(overrides)
    with pytest.raises(Exception):
        run_copilot_turn_with_failure_observability(
            client,
            **arguments,
        )
    assert len(observations) == 1
    observation = observations[0]
    assert observation.stage == expected_stage
    run = _run_metadata(observation)
    _assert_event_invariants(run, "run.failed")
    assert run.events[-1].failure_stage == expected_stage
    if scenario == "tool_execution":
        assert observation.tool_invocation_count == 1


def test_failure_callback_cannot_mask_original_exception_or_metadata(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    del monkeypatch
    original = RuntimeError("SECRET_ORIGINAL_EXCEPTION")
    client = _FakeClient([original])
    observations: list[CopilotFailureObservation] = []

    def callback(observation: CopilotFailureObservation) -> None:
        observations.append(observation)
        raise KeyboardInterrupt("callback failure")

    with pytest.raises(RuntimeError) as caught:
        run_copilot_turn_with_failure_observability(
            client,
            model="test-model",
            question="question",
            on_failure=callback,
        )
    assert caught.value is original
    run = _run_metadata(observations[0])
    assert run.events[-1].failure_stage == "first_provider_request"
    assert "SECRET_ORIGINAL_EXCEPTION" not in repr(run)
    assert client.close_calls == 0


def test_events_have_closed_payload_free_fields() -> None:
    assert hasattr(copilot, "CopilotRunEvent")
    event_type = copilot.CopilotRunEvent
    assert tuple(item.name for item in fields(event_type)) == (
        "run_id",
        "sequence",
        "kind",
        "provider_request_index",
        "tool_invocation_index",
        "tool_call_id",
        "tool_name",
        "usage",
        "failure_stage",
    )
    forbidden = {
        "timestamp",
        "prompt",
        "messages",
        "arguments",
        "result",
        "api_key",
        "credentials",
        "response",
        "path",
        "exception",
        "message",
    }
    assert forbidden.isdisjoint(item.name for item in fields(event_type))


def test_session_adds_minimal_observed_method_without_changing_history() -> None:
    method = getattr(
        CopilotSession,
        "ask_with_observability",
        None,
    )
    assert method is not None, "M9 requires observed Session turn access"
    parameters = list(inspect.signature(method).parameters.values())
    assert [item.name for item in parameters] == [
        "self",
        "question",
        "on_failure",
    ]
    assert parameters[2].kind is inspect.Parameter.KEYWORD_ONLY
    assert parameters[2].default is None
    assert get_type_hints(method)["return"] is CopilotObservedResult
    client = _FakeClient([_response(), _response()])
    session = CopilotSession(client, model="test-model")
    first = method(session, "first")
    second = method(session, "second")
    first_run = _run_metadata(first)
    second_run = _run_metadata(second)
    assert first_run.run_id != second_run.run_id
    assert session.history == (first.turn, second.turn)
    assert all(not hasattr(turn, "run") for turn in session.history)
    assert session.export_history() == [
        {
            "question": "first",
            "answer": "final answer",
            "tool_call_content": None,
            "tool_invocations": [],
        },
        {
            "question": "second",
            "answer": "final answer",
            "tool_call_content": None,
            "tool_invocations": [],
        },
    ]


def test_failed_observed_session_turn_is_transactional() -> None:
    method = getattr(
        CopilotSession,
        "ask_with_observability",
        None,
    )
    assert method is not None, "M9 requires observed Session turn access"
    failure = RuntimeError("failed session turn")
    client = _FakeClient([_response(content="committed"), failure])
    session = CopilotSession(client, model="test-model", max_turns=1)
    committed = session.ask_with_result("committed question")
    before = session.history
    observations: list[CopilotFailureObservation] = []
    with pytest.raises(RuntimeError) as caught:
        method(session, "failed question", on_failure=observations.append)
    assert caught.value is failure
    assert session.history == before == (committed,)
    assert session.history[0] is committed
    assert _run_metadata(observations[0]).events[-1].kind == "run.failed"
    assert client.close_calls == 0


def test_service_run_propagates_populated_metadata_and_borrows_client() -> None:
    client = _FakeClient([_response(usage=_usage())])
    service = CopilotService(client, model="test-model")
    result = service.run("question")
    run = _run_metadata(result)
    _assert_event_invariants(run, "run.completed")
    assert result.turn.answer == "final answer"
    assert client.close_calls == 0

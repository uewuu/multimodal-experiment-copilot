"""RED component contracts for successful M9 Runtime run metadata."""

from __future__ import annotations

from dataclasses import asdict, fields
import json
from types import SimpleNamespace
from typing import get_type_hints
from uuid import UUID

import pytest

from copilot import (
    CopilotObservedResult,
    CopilotRunMetadata,
    CopilotRuntimeMetrics,
    CopilotTurn,
    run_copilot_turn_with_observability,
)
import copilot.runtime_result as runtime_result
import llm_adapters.openai_tool_adapter as adapter


_MISSING = object()
_EVENT_KINDS = {
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
    def __init__(self, responses: list[object]) -> None:
        self.responses = list(responses)
        self.calls: list[dict[str, object]] = []

    def create(self, **kwargs: object) -> object:
        self.calls.append(kwargs)
        index = len(self.calls) - 1
        if index >= len(self.responses):
            raise AssertionError("unexpected extra provider request")
        return self.responses[index]


class _FakeClient:
    def __init__(self, responses: list[object]) -> None:
        self.completions = _SequentialCompletions(responses)
        self.chat = SimpleNamespace(completions=self.completions)


def _usage(
    input_tokens: object = 10,
    output_tokens: object = 5,
    total_tokens: object = 15,
) -> SimpleNamespace:
    return SimpleNamespace(
        prompt_tokens=input_tokens,
        completion_tokens=output_tokens,
        total_tokens=total_tokens,
    )


def _response(
    tool_calls: object = None,
    *,
    content: str = "final answer",
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
        ]
    )
    if usage is not _MISSING:
        response.usage = usage
    return response


def _tool_call(
    index: int,
    arguments: dict[str, object] | None = None,
) -> SimpleNamespace:
    return SimpleNamespace(
        id=f"call_{index}",
        type="function",
        function=SimpleNamespace(
            name=f"tool_{index}",
            arguments=json.dumps(
                {"index": index} if arguments is None else arguments
            ),
        ),
    )


def _install_tool(
    monkeypatch: pytest.MonkeyPatch,
    outcome: object = _MISSING,
) -> list[tuple[str, dict]]:
    calls: list[tuple[str, dict]] = []

    def invoke(name: str, arguments: dict) -> object:
        calls.append((name, arguments))
        if outcome is not _MISSING:
            return outcome
        return {"accepted": arguments.get("index")}

    monkeypatch.setattr(adapter, "invoke_tool", invoke)
    return calls


def _observe(
    client: object,
    *,
    question: str = "analyze experiments",
    **options: object,
) -> CopilotObservedResult:
    return run_copilot_turn_with_observability(
        client,
        model="test-model",
        question=question,
        **options,
    )


def _run(result: CopilotObservedResult) -> CopilotRunMetadata:
    assert hasattr(result, "run"), (
        "Slice 1B requires successful Runtime result.run"
    )
    run = getattr(result, "run")
    assert run is not None, "successful Runtime must populate result.run"
    assert isinstance(run, CopilotRunMetadata)
    return run


def _assert_uuid4(value: object) -> str:
    assert type(value) is str
    assert value == value.lower()
    parsed = UUID(value)
    assert parsed.version == 4
    assert str(parsed) == value
    return value


def _assert_success_events(run: CopilotRunMetadata) -> None:
    run_id = _assert_uuid4(run.run_id)
    kinds = [event.kind for event in run.events]
    assert [event.sequence for event in run.events] == list(
        range(len(run.events))
    )
    assert kinds[0] == "run.started"
    assert kinds[-1] == "run.completed"
    assert sum(kind in {"run.completed", "run.failed"} for kind in kinds) == 1
    assert {event.run_id for event in run.events} == {run_id}
    assert set(kinds) <= _EVENT_KINDS
    assert all(not hasattr(event, "timestamp") for event in run.events)


def test_successful_runtime_result_adds_populated_run_metadata() -> None:
    result_fields = fields(CopilotObservedResult)
    assert tuple(item.name for item in result_fields) == (
        "turn", "metrics", "run"
    ), "Slice 1B requires the additive observed-result run field"
    assert result_fields[-1].default is None
    annotations = get_type_hints(CopilotObservedResult)
    assert annotations["turn"] is CopilotTurn
    assert annotations["metrics"] is CopilotRuntimeMetrics

    turn = CopilotTurn("question", "answer", None, ())
    metrics = CopilotRuntimeMetrics(1, 0, 0.0)
    legacy = CopilotObservedResult(turn, metrics)
    assert legacy.turn is turn
    assert legacy.metrics is metrics
    assert legacy.run is None
    assert tuple(item.name for item in fields(CopilotTurn)) == (
        "question", "answer", "tool_call_content", "tool_invocations"
    )
    assert not hasattr(turn, "run")

    result = _observe(_FakeClient([_response()]))
    assert isinstance(result.turn, CopilotTurn)
    assert isinstance(result.metrics, CopilotRuntimeMetrics)
    assert result.turn.answer == "final answer"
    assert result.metrics.provider_request_count == 1
    assert result.metrics.tool_invocation_count == 0
    _run(result)


def test_successful_runtime_creates_unique_canonical_uuid4_ids() -> None:
    first = _observe(_FakeClient([_response()]))
    second = _observe(_FakeClient([_response()]))
    assert _assert_uuid4(_run(first).run_id) != _assert_uuid4(
        _run(second).run_id
    )


def test_runtime_identity_is_not_chosen_by_caller_provider_or_tool(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    forbidden_ids = {
        "caller-controlled", "provider-controlled", "tool-controlled"
    }
    caller_client = _FakeClient([_response()])
    caller_result = None
    try:
        caller_result = _observe(caller_client, run_id="caller-controlled")
    except (TypeError, ValueError):
        assert caller_client.completions.calls == []

    _install_tool(monkeypatch, {"run_id": "tool-controlled"})
    provider_response = _response([_tool_call(0)])
    provider_response.run_id = "provider-controlled"
    result = _observe(_FakeClient([provider_response, _response()]))

    if caller_result is not None:
        assert _assert_uuid4(_run(caller_result).run_id) != "caller-controlled"
    run = _run(result)
    assert _assert_uuid4(run.run_id) not in forbidden_ids
    assert {event.run_id for event in run.events}.isdisjoint(forbidden_ids)


def test_no_tool_runtime_builds_success_lifecycle_and_complete_usage() -> None:
    result = _observe(_FakeClient([_response(usage=_usage(100, 20, 120))]))
    run = _run(result)
    assert (
        run.usage.input_tokens,
        run.usage.output_tokens,
        run.usage.total_tokens,
    ) == (100, 20, 120)
    assert run.usage.provider_request_count == 1
    assert run.usage.provider_response_count == 1
    assert run.usage.usage_report_count == 1
    assert run.usage.complete is True
    assert [event.kind for event in run.events] == [
        "run.started",
        "provider.request.started",
        "provider.response.received",
        "run.completed",
    ]
    _assert_success_events(run)


def test_runtime_consumes_adapter_private_usage_evidence(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Runtime must use the Adapter snapshot, not reread raw response usage."""
    cycle = runtime_result._run_tool_call_cycle_with_trace
    traces: list[object] = []

    def capture_trace(*args: object, **kwargs: object) -> object:
        trace = cycle(*args, **kwargs)
        traces.append(trace)
        trace.response.usage = _usage(900, 800, 1700)
        return trace

    monkeypatch.setattr(
        runtime_result, "_run_tool_call_cycle_with_trace", capture_trace
    )
    result = _observe(_FakeClient([_response(usage=_usage(31, 7, 38))]))
    assert len(traces) == 1
    evidence = traces[0]
    assert evidence.provider_usages[0].input_tokens == 31
    run = _run(result)
    assert (
        run.usage.input_tokens,
        run.usage.output_tokens,
        run.usage.total_tokens,
    ) == (31, 7, 38)
    assert run.usage.provider_request_count == evidence.provider_request_count
    assert run.usage.provider_response_count == evidence.provider_response_count
    assert run.usage.usage_report_count == 1
    assert run.usage.complete is True


@pytest.mark.parametrize(
    ("provider_usage", "expected", "report_count"),
    [
        (_MISSING, (None, None, None), 0),
        (None, (None, None, None), 0),
        (_usage("10", "5", "15"), (None, None, None), 0),
        (_usage(-1, -2, -3), (None, None, None), 0),
        (_usage(True, False, True), (None, None, None), 0),
        (_usage(10, None, None), (10, None, None), 1),
        (_usage(10, 5, "bad"), (10, 5, None), 1),
    ],
    ids=(
        "missing", "none", "malformed", "negative", "bool",
        "partial", "field-level-invalid",
    ),
)
def test_runtime_preserves_conservative_usage(
    provider_usage: object,
    expected: tuple[int | None, int | None, int | None],
    report_count: int,
) -> None:
    result = _observe(_FakeClient([_response(usage=provider_usage)]))
    assert result.turn.answer == "final answer"
    run = _run(result)
    assert (
        run.usage.input_tokens,
        run.usage.output_tokens,
        run.usage.total_tokens,
    ) == expected
    assert run.usage.usage_report_count == report_count
    assert run.usage.complete is False


def test_runtime_aggregates_usage_with_field_level_response_coverage(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_tool(monkeypatch)
    client = _FakeClient(
        [
            _response([_tool_call(0)], usage=_usage(100, 20, 120)),
            _response(usage=_usage(50, None, None)),
        ]
    )
    result = _observe(client)
    assert len(client.completions.calls) == 2
    run = _run(result)
    assert (
        run.usage.input_tokens,
        run.usage.output_tokens,
        run.usage.total_tokens,
    ) == (150, None, None)
    assert run.usage.provider_request_count == 2
    assert run.usage.provider_response_count == 2
    assert run.usage.usage_report_count == 2
    assert run.usage.complete is False


@pytest.mark.parametrize("tool_count", [1, 3])
def test_runtime_builds_deterministic_successful_tool_lifecycle(
    monkeypatch: pytest.MonkeyPatch,
    tool_count: int,
) -> None:
    calls = _install_tool(monkeypatch)
    client = _FakeClient(
        [
            _response([_tool_call(index) for index in range(tool_count)]),
            _response(),
        ]
    )
    result = _observe(client)
    assert len(calls) == tool_count
    run = _run(result)
    expected = [
        "run.started",
        "provider.request.started",
        "provider.response.received",
        "tool.calls.validation.started",
    ]
    for _ in range(tool_count):
        expected.extend(["tool.execution.started", "tool.result.accepted"])
    expected.extend(
        [
            "provider.request.started",
            "provider.response.received",
            "run.completed",
        ]
    )
    assert [event.kind for event in run.events] == expected
    executions = [
        event for event in run.events if event.kind == "tool.execution.started"
    ]
    assert [event.tool_invocation_index for event in executions] == list(
        range(tool_count)
    )
    assert [event.tool_call_id for event in executions] == [
        f"call_{index}" for index in range(tool_count)
    ]
    assert [event.tool_name for event in executions] == [
        f"tool_{index}" for index in range(tool_count)
    ]
    _assert_success_events(run)


def test_successful_runtime_metadata_excludes_sensitive_payloads(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    secrets = (
        "SECRET_PROMPT",
        "SECRET_ARGUMENT",
        "SECRET_RESULT",
        "SECRET_API_KEY",
        "SECRET_PROVIDER_PAYLOAD",
        "C:\\SECRET\\experiment",
    )
    _install_tool(monkeypatch, {"value": secrets[2]})
    first_response = _response(
        [_tool_call(0, {"payload": secrets[1], "path": secrets[5]})],
        content=secrets[0],
        usage=_usage(),
    )
    first_response.provider_payload = {
        "value": secrets[4], "credentials": secrets[3]
    }
    client = _FakeClient(
        [first_response, _response(content="safe answer", usage=_usage())]
    )
    client.api_key = secrets[3]
    result = _observe(client, question=secrets[0])
    invocation = result.turn.tool_invocations[0]
    assert secrets[1] in invocation.arguments_json
    assert secrets[2] in invocation.result_json
    serialized_run = json.dumps(asdict(_run(result)), ensure_ascii=False)
    for secret in secrets:
        assert json.dumps(secret, ensure_ascii=False)[1:-1] not in serialized_run

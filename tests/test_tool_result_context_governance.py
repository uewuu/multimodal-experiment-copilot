"""RED contracts for bounded tool-result provider context."""

from __future__ import annotations

from copy import deepcopy
import json
from types import SimpleNamespace

import pytest

from copilot import (
    CopilotSession,
    run_copilot_turn_with_failure_observability,
)
import llm_adapters.openai_tool_adapter as adapter


MAX_TOOL_RESULT_BYTES = 256 * 1024
MAX_TOOL_RESULTS_PER_CYCLE_BYTES = 512 * 1024


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
        self.chat = SimpleNamespace(
            completions=_SequentialCompletions(outcomes)
        )
        self.close_calls = 0

    def close(self) -> None:
        self.close_calls += 1


def _tool_call(
    index: int,
    *,
    name: str = "analyze_experiment",
    arguments: str | None = None,
) -> SimpleNamespace:
    arguments_text = (
        arguments
        if arguments is not None
        else json.dumps(
            {"experiment_dir": f"demo-{index}"},
            separators=(",", ":"),
        )
    )
    return SimpleNamespace(
        id=f"call-{index}",
        type="function",
        function=SimpleNamespace(
            name=name,
            arguments=arguments_text,
        ),
    )


def _response(
    tool_calls: object = None,
    *,
    content: object = "final answer",
) -> SimpleNamespace:
    return SimpleNamespace(
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


def _serialize(value: object) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def _payload_with_serialized_bytes(
    target_bytes: int,
    *,
    unicode_text: bool = False,
) -> dict[str, str]:
    overhead = len(_serialize({"data": ""}).encode("utf-8"))
    value_bytes = target_bytes - overhead
    assert value_bytes >= 0
    if unicode_text:
        unicode_count, ascii_count = divmod(value_bytes, 3)
        value = "界" * unicode_count + "x" * ascii_count
    else:
        value = "x" * value_bytes
    payload = {"data": value}
    assert len(_serialize(payload).encode("utf-8")) == target_bytes
    return payload


def _install_results(
    monkeypatch: pytest.MonkeyPatch,
    results: list[object],
) -> tuple[list[tuple[str, dict]], list[object]]:
    calls: list[tuple[str, dict]] = []
    returned: list[object] = []

    def invoke(name: str, arguments: dict) -> object:
        calls.append((name, deepcopy(arguments)))
        result = results[len(calls) - 1]
        returned.append(result)
        return result

    monkeypatch.setattr(adapter, "invoke_tool", invoke)
    return calls, returned


def _run_cycle(
    client: _FakeClient,
) -> object:
    return adapter.run_tool_call_cycle(
        client,
        model="test-model",
        messages=[{"role": "user", "content": "Analyze."}],
    )


def _tool_messages(request: dict[str, object]) -> list[dict]:
    messages = request["messages"]
    assert type(messages) is list
    return [
        message
        for message in messages
        if message.get("role") == "tool"
    ]


def test_no_tool_path_requests_once_and_never_invokes_governance_tool(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    response = _response(content="done")
    client = _FakeClient([response])

    def forbidden(*args: object, **kwargs: object) -> object:
        pytest.fail("no tool may be invoked")

    monkeypatch.setattr(adapter, "invoke_tool", forbidden)

    assert _run_cycle(client) is response
    assert len(client.chat.completions.calls) == 1
    assert client.close_calls == 0


def test_small_multiple_results_keep_content_order_and_ids(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    results = [
        {"position": 1, "text": "中文"},
        {"position": 2, "items": [None, True]},
    ]
    calls, returned = _install_results(monkeypatch, results)
    first = _response(
        [_tool_call(1), _tool_call(2)],
        content="working",
    )
    final = _response(content="done")
    client = _FakeClient([first, final])

    assert _run_cycle(client) is final

    messages = _tool_messages(client.chat.completions.calls[1])
    assert len(client.chat.completions.calls) == 2
    assert [message["tool_call_id"] for message in messages] == [
        "call-1",
        "call-2",
    ]
    assert [message["content"] for message in messages] == [
        _serialize(result) for result in results
    ]
    assert [arguments for _, arguments in calls] == [
        {"experiment_dir": "demo-1"},
        {"experiment_dir": "demo-2"},
    ]
    assert returned == results
    assert client.close_calls == 0


def test_result_exactly_at_per_result_limit_is_accepted(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    result = _payload_with_serialized_bytes(MAX_TOOL_RESULT_BYTES)
    _install_results(monkeypatch, [result])
    final = _response(content="done")
    client = _FakeClient([_response([_tool_call(1)]), final])

    assert _run_cycle(client) is final

    messages = _tool_messages(client.chat.completions.calls[1])
    assert len(client.chat.completions.calls) == 2
    assert messages == [
        {
            "role": "tool",
            "tool_call_id": "call-1",
            "content": _serialize(result),
        }
    ]
    assert len(messages[0]["content"].encode("utf-8")) == (
        MAX_TOOL_RESULT_BYTES
    )


def test_result_one_byte_over_per_result_limit_is_rejected(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    result = _payload_with_serialized_bytes(
        MAX_TOOL_RESULT_BYTES + 1
    )
    calls, _ = _install_results(monkeypatch, [result])
    client = _FakeClient(
        [_response([_tool_call(1)]), _response(content="forbidden")]
    )

    with pytest.raises(
        ValueError,
        match=r"tool result.*262144.*context limit",
    ):
        _run_cycle(client)

    assert len(calls) == 1
    assert len(client.chat.completions.calls) == 1
    assert client.close_calls == 0


def test_unicode_limit_uses_utf8_bytes_not_character_count(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    result = _payload_with_serialized_bytes(
        MAX_TOOL_RESULT_BYTES + 1,
        unicode_text=True,
    )
    serialized = _serialize(result)
    assert len(serialized) < MAX_TOOL_RESULT_BYTES
    assert len(serialized.encode("utf-8")) == (
        MAX_TOOL_RESULT_BYTES + 1
    )
    _install_results(monkeypatch, [result])
    client = _FakeClient(
        [_response([_tool_call(1)]), _response(content="forbidden")]
    )

    with pytest.raises(ValueError, match=r"tool result.*262144"):
        _run_cycle(client)

    assert len(client.chat.completions.calls) == 1


def test_results_exactly_at_cycle_limit_are_accepted(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    results = [
        _payload_with_serialized_bytes(MAX_TOOL_RESULT_BYTES),
        _payload_with_serialized_bytes(MAX_TOOL_RESULT_BYTES),
    ]
    _install_results(monkeypatch, results)
    final = _response(content="done")
    client = _FakeClient(
        [_response([_tool_call(1), _tool_call(2)]), final]
    )

    assert _run_cycle(client) is final

    messages = _tool_messages(client.chat.completions.calls[1])
    assert sum(
        len(message["content"].encode("utf-8"))
        for message in messages
    ) == MAX_TOOL_RESULTS_PER_CYCLE_BYTES
    assert [message["tool_call_id"] for message in messages] == [
        "call-1",
        "call-2",
    ]


def _aggregate_overflow_results() -> list[object]:
    return [
        _payload_with_serialized_bytes(MAX_TOOL_RESULT_BYTES),
        _payload_with_serialized_bytes(MAX_TOOL_RESULT_BYTES - 1),
        {},
        {"late": True},
    ]


def _aggregate_overflow_case(
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[_FakeClient, list[tuple[str, dict]], BaseException | None]:
    results = _aggregate_overflow_results()
    calls, _ = _install_results(monkeypatch, results)
    first = _response(
        [_tool_call(index) for index in range(1, 5)]
    )
    client = _FakeClient([first, _response(content="forbidden")])
    caught: BaseException | None = None
    try:
        _run_cycle(client)
    except BaseException as error:
        caught = error
    return client, calls, caught


def test_cycle_limit_plus_one_is_rejected_before_second_request(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client, _, caught = _aggregate_overflow_case(monkeypatch)

    assert type(caught) is ValueError
    assert "524288" in str(caught)
    assert "per-cycle context limit" in str(caught)
    assert len(client.chat.completions.calls) == 1


def test_cycle_overflow_tool_counts_as_started_and_later_tool_stops(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _, calls, _ = _aggregate_overflow_case(monkeypatch)

    assert [arguments["experiment_dir"] for _, arguments in calls] == [
        "demo-1",
        "demo-2",
        "demo-3",
    ]


def test_cycle_overflow_sends_no_partial_tool_messages(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client, _, caught = _aggregate_overflow_case(monkeypatch)

    assert isinstance(caught, ValueError)
    assert len(client.chat.completions.calls) == 1
    first_messages = client.chat.completions.calls[0]["messages"]
    assert all(message.get("role") != "tool" for message in first_messages)


def test_accepted_result_does_not_mutate_arguments_or_result(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    result = {"nested": {"values": [1, "中文"]}}
    result_before = deepcopy(result)
    arguments = '{ "experiment_dir" : "demo", "nested" : [1] }'
    tool_call = _tool_call(1, arguments=arguments)
    tool_call_before = deepcopy(tool_call)
    calls, _ = _install_results(monkeypatch, [result])
    client = _FakeClient(
        [_response([tool_call]), _response(content="done")]
    )

    _run_cycle(client)

    assert result == result_before
    assert vars(tool_call) == vars(tool_call_before)
    assert vars(tool_call.function) == vars(tool_call_before.function)
    assert tool_call.function.arguments == arguments
    assert calls == [
        (
            "analyze_experiment",
            {"experiment_dir": "demo", "nested": [1]},
        )
    ]
    assert client.close_calls == 0


def _deep_result(depth: int = 1500) -> dict[str, object]:
    root: dict[str, object] = {}
    cursor = root
    for _ in range(depth):
        child: dict[str, object] = {}
        cursor["child"] = child
        cursor = child
    return root


def _legacy_serialization_case(kind: str) -> tuple[object, type[BaseException]]:
    if kind == "nan":
        return float("nan"), ValueError
    if kind == "infinity":
        return float("inf"), ValueError
    if kind == "set":
        return {1}, TypeError
    if kind == "bytes":
        return b"value", TypeError
    if kind == "object":
        return object(), TypeError
    if kind == "circular":
        circular: dict[str, object] = {}
        circular["self"] = circular
        return circular, ValueError
    if kind == "deep":
        return _deep_result(), RecursionError
    raise AssertionError(f"unknown case: {kind}")


@pytest.mark.parametrize(
    "kind",
    ["nan", "infinity", "set", "bytes", "object", "circular", "deep"],
)
def test_existing_serialization_failures_keep_type_and_stop_follow_up(
    monkeypatch: pytest.MonkeyPatch,
    kind: str,
) -> None:
    result, error_type = _legacy_serialization_case(kind)
    _install_results(monkeypatch, [result])
    client = _FakeClient([_response([_tool_call(1)])])

    with pytest.raises(error_type):
        _run_cycle(client)

    assert len(client.chat.completions.calls) == 1


def test_oversized_result_is_observed_as_tool_result_serialization(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    result = _payload_with_serialized_bytes(
        MAX_TOOL_RESULT_BYTES + 1
    )
    _install_results(monkeypatch, [result])
    client = _FakeClient(
        [_response([_tool_call(1)], content=None), _response()]
    )
    observations: list[object] = []

    with pytest.raises(ValueError, match=r"tool result.*262144"):
        run_copilot_turn_with_failure_observability(
            client,
            model="test-model",
            question="Analyze.",
            on_failure=observations.append,
        )

    assert len(observations) == 1
    observation = observations[0]
    assert observation.stage == "tool_result_serialization"
    assert observation.provider_request_count == 1
    assert observation.tool_invocation_count == 1
    assert len(client.chat.completions.calls) == 1
    assert client.close_calls == 0


def test_aggregate_overflow_observes_all_started_tools(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls, _ = _install_results(
        monkeypatch,
        _aggregate_overflow_results(),
    )
    client = _FakeClient(
        [
            _response(
                [_tool_call(index) for index in range(1, 5)],
                content=None,
            ),
            _response(content="forbidden"),
        ]
    )
    observations: list[object] = []

    with pytest.raises(ValueError, match=r"tool results.*524288"):
        run_copilot_turn_with_failure_observability(
            client,
            model="test-model",
            question="Analyze.",
            on_failure=observations.append,
        )

    assert len(calls) == 3
    assert len(observations) == 1
    observation = observations[0]
    assert observation.stage == "tool_result_serialization"
    assert observation.provider_request_count == 1
    assert observation.tool_invocation_count == 3
    assert len(client.chat.completions.calls) == 1


@pytest.mark.parametrize(
    "callback_error",
    [
        RuntimeError("callback"),
        KeyboardInterrupt(),
        SystemExit(11),
        GeneratorExit(),
    ],
)
def test_governance_failure_survives_callback_baseexception(
    monkeypatch: pytest.MonkeyPatch,
    callback_error: BaseException,
) -> None:
    result = _payload_with_serialized_bytes(
        MAX_TOOL_RESULT_BYTES + 1
    )
    _install_results(monkeypatch, [result])
    client = _FakeClient(
        [_response([_tool_call(1)], content=None), _response()]
    )
    observations: list[object] = []

    def callback(observation: object) -> None:
        observations.append(observation)
        raise callback_error

    with pytest.raises(ValueError) as caught:
        run_copilot_turn_with_failure_observability(
            client,
            model="test-model",
            question="Analyze.",
            on_failure=callback,
        )

    assert "262144" in str(caught.value)
    assert len(observations) == 1
    observation = observations[0]
    assert observation.stage == "tool_result_serialization"
    assert observation.provider_request_count == 1
    assert observation.tool_invocation_count == 1
    assert len(client.chat.completions.calls) == 1


@pytest.mark.parametrize(
    "callback_error",
    [
        RuntimeError("callback"),
        KeyboardInterrupt(),
        SystemExit(7),
        GeneratorExit(),
    ],
)
def test_serialization_error_identity_survives_callback_baseexception(
    monkeypatch: pytest.MonkeyPatch,
    callback_error: BaseException,
) -> None:
    expected = ValueError("original serialization failure")
    first_response = _response([_tool_call(1)], content=None)

    def fail_serialization(*args: object, **kwargs: object) -> str:
        raise expected

    monkeypatch.setattr(adapter.json, "dumps", fail_serialization)
    _install_results(monkeypatch, [{"ok": True}])
    client = _FakeClient([first_response])
    observations: list[object] = []

    def callback(observation: object) -> None:
        observations.append(observation)
        raise callback_error

    with pytest.raises(ValueError) as caught:
        run_copilot_turn_with_failure_observability(
            client,
            model="test-model",
            question="Analyze.",
            on_failure=callback,
        )

    assert caught.value is expected
    assert len(observations) == 1
    observation = observations[0]
    assert observation.stage == "tool_result_serialization"
    assert observation.provider_request_count == 1
    assert observation.tool_invocation_count == 1


def test_session_governance_failure_does_not_commit_or_evict(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    result = _payload_with_serialized_bytes(
        MAX_TOOL_RESULT_BYTES + 1
    )
    _install_results(monkeypatch, [result])
    client = _FakeClient(
        [
            _response(content="answer one"),
            _response(content="answer two"),
            _response([_tool_call(1)], content=None),
            _response(content="forbidden"),
        ]
    )
    session = CopilotSession(client, model="test-model", max_turns=2)
    session.ask("question one")
    session.ask("question two")
    history_before = session.history

    with pytest.raises(ValueError, match=r"tool result.*262144"):
        session.ask("oversized result")

    assert session.history == history_before
    assert all(
        current is original
        for current, original in zip(
            session.history,
            history_before,
            strict=True,
        )
    )
    assert session.turn_count == 2
    assert len(client.chat.completions.calls) == 3
    assert client.close_calls == 0

import builtins
from copy import deepcopy
from dataclasses import FrozenInstanceError, fields, is_dataclass
import importlib
import inspect
import logging
from types import SimpleNamespace
from typing import Callable, get_type_hints

import pytest

import copilot
from copilot.runtime_observability import CopilotObservedResult
import llm_adapters.openai_tool_adapter as adapter


MODULE_NAME = "copilot.failure_observability"
PUBLIC_NAMES = (
    "CopilotFailureObservation",
    "run_copilot_turn_with_failure_observability",
)
STAGES = (
    "input_validation",
    "first_provider_request",
    "first_provider_response_validation",
    "tool_call_validation",
    "tool_execution",
    "tool_result_serialization",
    "second_provider_request",
    "final_response_validation",
)


class _SequentialCompletions:
    def __init__(self, outcomes: list[object]) -> None:
        self.outcomes = list(outcomes)
        self.calls: list[dict[str, object]] = []

    def create(self, **kwargs: object) -> object:
        self.calls.append(deepcopy(kwargs))
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


def _module() -> object:
    return importlib.import_module(MODULE_NAME)


def _tool_call(
    index: int,
    *,
    name: object = "analyze_experiment",
    arguments: object = '{"experiment_dir":"demo"}',
    call_id: object | None = None,
    call_type: object = "function",
    function: object | None = None,
) -> SimpleNamespace:
    function_value = function
    if function_value is None:
        function_value = SimpleNamespace(
            name=name,
            arguments=arguments,
        )
    return SimpleNamespace(
        id=f"call_{index}" if call_id is None else call_id,
        type=call_type,
        function=function_value,
    )


def _response(
    tool_calls: object = None,
    *,
    content: object = "final answer",
    role: object = "assistant",
) -> SimpleNamespace:
    return SimpleNamespace(
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


def _run(
    client: _FakeClient,
    callback: object,
    **overrides: object,
) -> object:
    arguments: dict[str, object] = {
        "model": "test-model",
        "question": "Analyze the experiment.",
        "on_failure": callback,
    }
    arguments.update(overrides)
    return _module().run_copilot_turn_with_failure_observability(
        client,
        **arguments,
    )


def _assert_observation(
    observation: object,
    *,
    stage: str,
    provider_count: int,
    tool_count: int,
) -> None:
    module = _module()
    assert isinstance(observation, module.CopilotFailureObservation)
    assert observation.stage == stage
    assert observation.provider_request_count == provider_count
    assert observation.tool_invocation_count == tool_count
    assert type(observation.elapsed_seconds) is float
    assert observation.elapsed_seconds >= 0.0


def _assert_failure(
    client: _FakeClient,
    error_type: type[BaseException],
    *,
    stage: str,
    provider_count: int,
    tool_count: int,
    **overrides: object,
) -> BaseException:
    observations: list[object] = []
    with pytest.raises(error_type) as caught:
        _run(client, observations.append, **overrides)
    assert len(observations) == 1
    _assert_observation(
        observations[0],
        stage=stage,
        provider_count=provider_count,
        tool_count=tool_count,
    )
    return caught.value


def test_module_and_package_export_exact_public_api() -> None:
    module = _module()
    assert tuple(module.__all__) == PUBLIC_NAMES
    for name in PUBLIC_NAMES:
        assert getattr(copilot, name) is getattr(module, name)
        assert name in copilot.__all__


def test_failure_runtime_has_exact_signature() -> None:
    module = _module()
    function = module.run_copilot_turn_with_failure_observability
    parameters = list(inspect.signature(function).parameters.values())
    assert [parameter.name for parameter in parameters] == [
        "client",
        "model",
        "question",
        "experiment_context",
        "on_failure",
        "request_options",
    ]
    assert [parameter.kind for parameter in parameters] == [
        inspect.Parameter.POSITIONAL_OR_KEYWORD,
        inspect.Parameter.KEYWORD_ONLY,
        inspect.Parameter.KEYWORD_ONLY,
        inspect.Parameter.KEYWORD_ONLY,
        inspect.Parameter.KEYWORD_ONLY,
        inspect.Parameter.VAR_KEYWORD,
    ]
    assert parameters[3].default is None
    assert parameters[4].default is inspect.Parameter.empty
    module = _module()
    assert get_type_hints(function) == {
        "client": object,
        "model": str,
        "question": str,
        "experiment_context": dict[str, object] | None,
        "on_failure": Callable[[module.CopilotFailureObservation], None],
        "request_options": object,
        "return": CopilotObservedResult,
    }


def test_failure_observation_is_exact_frozen_slots_dataclass() -> None:
    model = _module().CopilotFailureObservation
    assert is_dataclass(model)
    assert model.__dataclass_params__.frozen is True
    assert tuple(field.name for field in fields(model)) == (
        "stage",
        "provider_request_count",
        "tool_invocation_count",
        "elapsed_seconds",
    )
    assert get_type_hints(model) == {
        "stage": str,
        "provider_request_count": int,
        "tool_invocation_count": int,
        "elapsed_seconds": float,
    }
    assert STAGES == (
        "input_validation",
        "first_provider_request",
        "first_provider_response_validation",
        "tool_call_validation",
        "tool_execution",
        "tool_result_serialization",
        "second_provider_request",
        "final_response_validation",
    )
    observation = model("tool_execution", 1, 2, 0.5)
    with pytest.raises(FrozenInstanceError):
        observation.stage = "changed"
    assert tuple(model.__slots__) == (
        "stage",
        "provider_request_count",
        "tool_invocation_count",
        "elapsed_seconds",
    )
    assert not hasattr(observation, "__dict__")
def test_success_without_tools_returns_existing_result_and_no_callback() -> None:
    client = _FakeClient([_response(content="answer")])
    observations: list[object] = []
    result = _run(client, observations.append)
    assert isinstance(result, CopilotObservedResult)
    assert result.turn.answer == "answer"
    assert result.metrics.provider_request_count == 1
    assert result.metrics.tool_invocation_count == 0
    assert observations == []
    assert len(client.completions.calls) == 1
    assert client.close_calls == 0


def test_success_with_tools_preserves_order_and_no_callback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = [
        _tool_call(1, name="analyze_experiment"),
        _tool_call(2, name="compare_experiments"),
    ]
    client = _FakeClient([
        _response(calls, content=None),
        _response(content="done"),
    ])
    invoked: list[tuple[str, dict[str, object]]] = []

    def fake_invoke(name: str, arguments: dict[str, object]) -> dict:
        invoked.append((name, deepcopy(arguments)))
        return {"index": len(invoked)}

    monkeypatch.setattr(adapter, "invoke_tool", fake_invoke)
    observations: list[object] = []
    result = _run(client, observations.append)
    assert isinstance(result, CopilotObservedResult)
    assert result.turn.answer == "done"
    assert result.metrics.provider_request_count == 2
    assert result.metrics.tool_invocation_count == 2
    assert [name for name, _ in invoked] == [
        "analyze_experiment",
        "compare_experiments",
    ]
    assert observations == []
    assert len(client.completions.calls) == 2


@pytest.mark.parametrize(
    ("overrides", "error_type"),
    [
        ({"question": 1}, TypeError),
        ({"question": "   "}, ValueError),
        ({"experiment_context": []}, TypeError),
        ({"messages": []}, TypeError),
    ],
)
def test_input_validation_failures_are_observed_before_execution(
    overrides: dict[str, object],
    error_type: type[Exception],
) -> None:
    client = _FakeClient([])
    _assert_failure(
        client,
        error_type,
        stage="input_validation",
        provider_count=0,
        tool_count=0,
        **overrides,
    )
    assert client.completions.calls == []


def test_noncallable_callback_fails_before_clock_or_execution(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _module()
    clock_calls: list[None] = []

    def clock() -> float:
        clock_calls.append(None)
        return 1.0

    monkeypatch.setattr(module, "_perf_counter", clock)
    client = _FakeClient([])
    with pytest.raises(TypeError):
        _run(client, None)
    assert clock_calls == []
    assert client.completions.calls == []


@pytest.mark.parametrize(
    "error",
    [RuntimeError("provider"), KeyboardInterrupt(), SystemExit(7), GeneratorExit()],
)
def test_first_provider_failure_preserves_identity_and_counts_attempt(
    error: BaseException,
) -> None:
    client = _FakeClient([error])
    caught = _assert_failure(
        client,
        type(error),
        stage="first_provider_request",
        provider_count=1,
        tool_count=0,
    )
    assert caught is error
    assert len(client.completions.calls) == 1


@pytest.mark.parametrize(
    ("response", "error_type"),
    [
        (SimpleNamespace(), TypeError),
        (SimpleNamespace(choices=()), TypeError),
        (SimpleNamespace(choices=[]), ValueError),
        (SimpleNamespace(choices=[SimpleNamespace()]), TypeError),
        (_response(content=object()), TypeError),
        (_response(role="user"), ValueError),
    ],
)
def test_first_provider_response_structure_failures_are_observed(
    response: object,
    error_type: type[Exception],
) -> None:
    client = _FakeClient([response])
    _assert_failure(
        client,
        error_type,
        stage="first_provider_response_validation",
        provider_count=1,
        tool_count=0,
    )
    assert len(client.completions.calls) == 1


@pytest.mark.parametrize(
    "case",
    [
        "id",
        "type",
        "function",
        "name",
        "arguments_type",
        "malformed_json",
        "non_object_json",
    ],
)
def test_tool_call_validation_failures_have_no_tool_side_effects(
    monkeypatch: pytest.MonkeyPatch,
    case: str,
) -> None:
    values: dict[str, object] = {}
    if case == "id":
        values["call_id"] = ""
    elif case == "type":
        values["call_type"] = "computer"
    elif case == "function":
        values["function"] = {}
    elif case == "name":
        values["name"] = ""
    elif case == "arguments_type":
        values["arguments"] = None
    elif case == "malformed_json":
        values["arguments"] = "{"
    else:
        values["arguments"] = "[]"
    invoked: list[object] = []
    monkeypatch.setattr(adapter, "invoke_tool", invoked.append)
    client = _FakeClient([_response([_tool_call(1, **values)], content=None)])
    observations: list[object] = []
    with pytest.raises((TypeError, ValueError)):
        _run(client, observations.append)
    assert invoked == []
    assert len(observations) == 1
    _assert_observation(
        observations[0],
        stage="tool_call_validation",
        provider_count=1,
        tool_count=0,
    )


def test_all_tool_calls_are_prevalidated_before_any_invocation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    invoked: list[object] = []
    monkeypatch.setattr(adapter, "invoke_tool", invoked.append)
    client = _FakeClient([
        _response(
            [_tool_call(1), _tool_call(2, arguments="[")],
            content=None,
        )
    ])
    _assert_failure(
        client,
        ValueError,
        stage="tool_call_validation",
        provider_count=1,
        tool_count=0,
    )
    assert invoked == []


@pytest.mark.parametrize(
    "error",
    [RuntimeError("tool"), KeyboardInterrupt(), SystemExit(9), GeneratorExit()],
)
def test_tool_execution_failure_preserves_identity_and_counts_attempt(
    monkeypatch: pytest.MonkeyPatch,
    error: BaseException,
) -> None:
    def fail_tool(name: str, arguments: dict[str, object]) -> object:
        raise error

    monkeypatch.setattr(adapter, "invoke_tool", fail_tool)
    client = _FakeClient([_response([_tool_call(1)], content=None)])
    caught = _assert_failure(
        client,
        type(error),
        stage="tool_execution",
        provider_count=1,
        tool_count=1,
    )
    assert caught is error


def test_unknown_first_tool_is_an_observed_started_invocation() -> None:
    client = _FakeClient([
        _response(
            [_tool_call(1, name="not_registered")],
            content=None,
        )
    ])
    _assert_failure(
        client,
        KeyError,
        stage="tool_execution",
        provider_count=1,
        tool_count=1,
    )


def test_later_tool_failure_counts_only_started_invocations(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    invoked: list[int] = []
    error = RuntimeError("second tool failed")

    def invoke(name: str, arguments: dict[str, object]) -> dict:
        invoked.append(len(invoked) + 1)
        if len(invoked) == 2:
            raise error
        return {"ok": True}

    monkeypatch.setattr(adapter, "invoke_tool", invoke)
    client = _FakeClient([
        _response([_tool_call(1), _tool_call(2), _tool_call(3)], content=None)
    ])
    caught = _assert_failure(
        client,
        RuntimeError,
        stage="tool_execution",
        provider_count=1,
        tool_count=2,
    )
    assert caught is error
    assert invoked == [1, 2]


@pytest.mark.parametrize("result", [{1, 2}, float("nan"), float("inf")])
def test_tool_result_serialization_failure_is_observed(
    monkeypatch: pytest.MonkeyPatch,
    result: object,
) -> None:
    monkeypatch.setattr(adapter, "invoke_tool", lambda *args: result)
    client = _FakeClient([_response([_tool_call(1)], content=None)])
    observations: list[object] = []
    with pytest.raises((TypeError, ValueError)):
        _run(client, observations.append)
    assert len(observations) == 1
    _assert_observation(
        observations[0],
        stage="tool_result_serialization",
        provider_count=1,
        tool_count=1,
    )


@pytest.mark.parametrize(
    "error",
    [RuntimeError("second provider"), KeyboardInterrupt(), SystemExit(11)],
)
def test_second_provider_failure_preserves_identity_and_counts_attempt(
    monkeypatch: pytest.MonkeyPatch,
    error: BaseException,
) -> None:
    monkeypatch.setattr(adapter, "invoke_tool", lambda *args: {"ok": True})
    client = _FakeClient([
        _response([_tool_call(1)], content=None),
        error,
    ])
    caught = _assert_failure(
        client,
        type(error),
        stage="second_provider_request",
        provider_count=2,
        tool_count=1,
    )
    assert caught is error
    assert len(client.completions.calls) == 2


@pytest.mark.parametrize(
    "final_response",
    [
        SimpleNamespace(),
        SimpleNamespace(choices=[]),
        _response(content=None),
        _response(content="   "),
        _response([_tool_call(2)], content=None),
    ],
)
def test_final_response_validation_failures_are_observed(
    monkeypatch: pytest.MonkeyPatch,
    final_response: object,
) -> None:
    monkeypatch.setattr(adapter, "invoke_tool", lambda *args: {"ok": True})
    client = _FakeClient([
        _response([_tool_call(1)], content=None),
        final_response,
    ])
    observations: list[object] = []
    with pytest.raises((TypeError, ValueError)):
        _run(client, observations.append)
    assert len(observations) == 1
    _assert_observation(
        observations[0],
        stage="final_response_validation",
        provider_count=2,
        tool_count=1,
    )
    assert len(client.completions.calls) == 2


@pytest.mark.parametrize(
    "callback_error",
    [RuntimeError("callback"), KeyboardInterrupt(), SystemExit(13), GeneratorExit()],
)
def test_callback_failure_never_masks_original_failure(
    callback_error: BaseException,
) -> None:
    module = _module()
    original = RuntimeError("provider")
    client = _FakeClient([original])
    received: list[object] = []

    def callback(observation: object) -> None:
        received.append(observation)
        raise callback_error

    with pytest.raises(RuntimeError) as caught:
        module.run_copilot_turn_with_failure_observability(
            client,
            model="test-model",
            question="question",
            on_failure=callback,
        )
    assert caught.value is original
    assert len(received) == 1
    _assert_observation(
        received[0],
        stage="first_provider_request",
        provider_count=1,
        tool_count=0,
    )


def test_elapsed_time_excludes_callback_runtime(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _module()
    values = iter([10.0, 12.25, 99.0])
    clock_calls: list[float] = []

    def clock() -> float:
        value = next(values)
        clock_calls.append(value)
        return value

    monkeypatch.setattr(module, "_perf_counter", clock)
    error = RuntimeError("provider")
    client = _FakeClient([error])
    observations: list[object] = []

    def callback(observation: object) -> None:
        observations.append(observation)
        module._perf_counter()

    with pytest.raises(RuntimeError) as caught:
        _run(client, callback)
    assert caught.value is error
    assert observations[0].elapsed_seconds == 2.25
    assert clock_calls == [10.0, 12.25, 99.0]


def test_failure_path_preserves_inputs_and_has_no_runtime_side_effects(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    caplog: pytest.LogCaptureFixture,
) -> None:
    module = _module()
    context = {"experiment_dir": "demo"}
    options = {"temperature": 0.0, "metadata": {"trace": ["a"]}}
    context_before = deepcopy(context)
    options_before = deepcopy(options)
    error = RuntimeError("provider")
    client = _FakeClient([error])

    def fail_write(*args: object, **kwargs: object) -> object:
        raise AssertionError("file write is forbidden")

    monkeypatch.setattr(builtins, "open", fail_write)
    caplog.set_level(logging.NOTSET)
    with pytest.raises(RuntimeError) as caught:
        module.run_copilot_turn_with_failure_observability(
            client,
            model="test-model",
            question="question",
            experiment_context=context,
            on_failure=lambda observation: None,
            **options,
        )
    assert caught.value is error
    assert context == context_before
    assert options == options_before
    assert client.close_calls == 0
    assert len(client.completions.calls) == 1
    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err == ""
    assert caplog.records == []

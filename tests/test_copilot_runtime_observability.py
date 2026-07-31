import builtins
import copy
from dataclasses import FrozenInstanceError, fields, is_dataclass
import importlib
import inspect
import logging
import os
from pathlib import Path
import socket
import sys
from types import SimpleNamespace
from typing import get_type_hints

import pytest

import copilot
from copilot.session import CopilotToolInvocation, CopilotTurn
import llm_adapters.openai_tool_adapter as adapter


MODULE_NAME = "copilot.runtime_observability"
PUBLIC_NAMES = (
    "CopilotRuntimeMetrics",
    "CopilotObservedResult",
    "run_copilot_turn_with_observability",
)


class _SequentialCompletions:
    def __init__(self, outcomes: list[object]) -> None:
        self.outcomes = outcomes
        self.calls: list[dict[str, object]] = []

    @property
    def call_count(self) -> int:
        return len(self.calls)

    def create(self, **kwargs: object) -> object:
        self.calls.append(kwargs)
        index = self.call_count - 1
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


def _module():
    return importlib.import_module(MODULE_NAME)


def _tool_call(
    index: int,
    *,
    name: str | None = None,
    arguments: str | None = None,
) -> SimpleNamespace:
    return SimpleNamespace(
        id=f"call_{index}",
        type="function",
        function=SimpleNamespace(
            name=name or f"tool_{index}",
            arguments=arguments or f'{{ "index" : {index} }}',
        ),
    )


def _response(
    tool_calls: object,
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


def _plain_turn() -> CopilotTurn:
    return CopilotTurn(
        question="question",
        answer="answer",
        tool_call_content=None,
        tool_invocations=(),
    )


def _set_clock(
    monkeypatch: pytest.MonkeyPatch,
    module: object,
    values: list[float],
) -> list[None]:
    pending = iter(values)
    calls: list[None] = []

    def fake_clock() -> float:
        calls.append(None)
        return next(pending)

    monkeypatch.setattr(module, "_perf_counter", fake_clock)
    return calls


def _stub_structured_runtime(
    monkeypatch: pytest.MonkeyPatch,
    module: object,
    *,
    turn: CopilotTurn | None = None,
    error: BaseException | None = None,
) -> list[dict[str, object]]:
    calls: list[dict[str, object]] = []

    def fake_runtime(client: object, **kwargs: object) -> CopilotTurn:
        calls.append({"client": client, **kwargs})
        if error is not None:
            raise error
        assert turn is not None
        return turn

    monkeypatch.setattr(
        module,
        "run_copilot_turn_with_result",
        fake_runtime,
    )
    return calls


def _run_observed(
    module: object,
    client: object,
    **request_options: object,
) -> object:
    return module.run_copilot_turn_with_observability(
        client,
        model="test-model",
        question="question",
        **request_options,
    )


def test_module_exports_exact_public_api() -> None:
    module = _module()
    assert tuple(module.__all__) == PUBLIC_NAMES
    for name in PUBLIC_NAMES:
        assert hasattr(module, name)


def test_package_exports_are_identical_to_module_objects() -> None:
    module = _module()
    for name in PUBLIC_NAMES:
        assert getattr(copilot, name) is getattr(module, name)
        assert name in copilot.__all__


def test_observed_runtime_has_exact_signature_and_return_type() -> None:
    module = _module()
    function = module.run_copilot_turn_with_observability
    signature = inspect.signature(function)
    parameters = list(signature.parameters.values())
    assert [item.name for item in parameters] == [
        "client",
        "model",
        "question",
        "experiment_context",
        "request_options",
    ]
    assert parameters[0].kind is inspect.Parameter.POSITIONAL_OR_KEYWORD
    assert parameters[1].kind is inspect.Parameter.KEYWORD_ONLY
    assert parameters[2].kind is inspect.Parameter.KEYWORD_ONLY
    assert parameters[3].kind is inspect.Parameter.KEYWORD_ONLY
    assert parameters[3].default is None
    assert parameters[4].kind is inspect.Parameter.VAR_KEYWORD
    hints = get_type_hints(function)
    assert hints == {
        "client": object,
        "model": str,
        "question": str,
        "experiment_context": dict[str, object] | None,
        "request_options": object,
        "return": module.CopilotObservedResult,
    }


def test_observability_import_does_not_load_openai_sdk() -> None:
    was_loaded = "openai" in sys.modules
    _module()
    assert ("openai" in sys.modules) is was_loaded


def test_module_defines_no_other_public_observability_types() -> None:
    module = _module()
    public_types = {
        name
        for name, value in vars(module).items()
        if not name.startswith("_")
        and isinstance(value, type)
        and value.__module__ == MODULE_NAME
    }
    assert public_types == {
        "CopilotRuntimeMetrics",
        "CopilotObservedResult",
    }


def test_runtime_metrics_is_frozen_slots_dataclass() -> None:
    module = _module()
    metrics_type = module.CopilotRuntimeMetrics
    assert is_dataclass(metrics_type)
    assert metrics_type.__dataclass_params__.frozen is True
    assert tuple(metrics_type.__slots__) == (
        "provider_request_count",
        "tool_invocation_count",
        "elapsed_seconds",
    )


def test_runtime_metrics_has_exact_fields_and_annotations() -> None:
    module = _module()
    metrics_type = module.CopilotRuntimeMetrics
    assert [item.name for item in fields(metrics_type)] == [
        "provider_request_count",
        "tool_invocation_count",
        "elapsed_seconds",
    ]
    assert get_type_hints(metrics_type) == {
        "provider_request_count": int,
        "tool_invocation_count": int,
        "elapsed_seconds": float,
    }


@pytest.mark.parametrize(
    "forbidden",
    ["used_tools", "completed", "token_usage", "error_type"],
)
def test_runtime_metrics_excludes_deferred_fields(
    forbidden: str,
) -> None:
    module = _module()
    assert forbidden not in {
        item.name for item in fields(module.CopilotRuntimeMetrics)
    }


def test_runtime_metrics_instances_are_immutable() -> None:
    module = _module()
    metrics = module.CopilotRuntimeMetrics(
        provider_request_count=1,
        tool_invocation_count=0,
        elapsed_seconds=0.25,
    )
    with pytest.raises(FrozenInstanceError):
        metrics.elapsed_seconds = 1.0


def test_observed_result_is_frozen_slots_dataclass() -> None:
    module = _module()
    result_type = module.CopilotObservedResult
    assert is_dataclass(result_type)
    assert result_type.__dataclass_params__.frozen is True
    assert tuple(result_type.__slots__) == ("turn", "metrics")


def test_observed_result_has_exact_fields_and_annotations() -> None:
    module = _module()
    result_type = module.CopilotObservedResult
    assert [item.name for item in fields(result_type)] == [
        "turn",
        "metrics",
    ]
    assert get_type_hints(result_type) == {
        "turn": CopilotTurn,
        "metrics": module.CopilotRuntimeMetrics,
    }


@pytest.mark.parametrize(
    "forbidden",
    ["client", "response", "request_options", "experiment_context"],
)
def test_observed_result_excludes_provider_and_input_objects(
    forbidden: str,
) -> None:
    module = _module()
    assert forbidden not in {
        item.name for item in fields(module.CopilotObservedResult)
    }


def test_observed_result_instances_are_immutable() -> None:
    module = _module()
    turn = _plain_turn()
    metrics = module.CopilotRuntimeMetrics(1, 0, 0.0)
    result = module.CopilotObservedResult(turn, metrics)
    with pytest.raises(FrozenInstanceError):
        result.turn = _plain_turn()


def test_no_tool_success_preserves_turn_and_counts_requests(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _module()
    client = _FakeClient([_response(None)])
    clock_calls = _set_clock(monkeypatch, module, [10.0, 12.5])
    monkeypatch.setattr(
        adapter,
        "invoke_tool",
        lambda *args: pytest.fail("no tool may be invoked"),
    )
    result = _run_observed(module, client)
    assert isinstance(result, module.CopilotObservedResult)
    assert isinstance(result.turn, CopilotTurn)
    assert result.metrics.provider_request_count == 1
    assert result.metrics.tool_invocation_count == 0
    assert result.metrics.elapsed_seconds == 2.5
    assert client.completions.call_count == 1
    assert len(clock_calls) == 2


def test_no_tool_success_does_not_close_client(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _module()
    client = _FakeClient([_response(None)])
    _set_clock(monkeypatch, module, [1.0, 2.0])
    _run_observed(module, client)
    assert client.close_calls == 0


@pytest.mark.parametrize("tool_count", [1, 2, 3, 4])
def test_tool_success_reports_bounded_request_and_tool_counts(
    monkeypatch: pytest.MonkeyPatch,
    tool_count: int,
) -> None:
    module = _module()
    calls = [_tool_call(index) for index in range(tool_count)]
    client = _FakeClient(
        [
            _response(calls, content="using tools"),
            _response(None, content="complete"),
        ]
    )
    invocations: list[tuple[str, dict]] = []

    def fake_invoke(name: str, arguments: dict) -> dict:
        invocations.append((name, arguments))
        return {"index": arguments["index"]}

    monkeypatch.setattr(adapter, "invoke_tool", fake_invoke)
    _set_clock(monkeypatch, module, [3.0, 4.0])
    result = _run_observed(module, client)
    assert result.metrics.provider_request_count == 2
    assert result.metrics.tool_invocation_count == tool_count
    assert len(result.turn.tool_invocations) == tool_count
    assert client.completions.call_count == 2
    assert [name for name, _ in invocations] == [
        f"tool_{index}" for index in range(tool_count)
    ]
    assert len(invocations) == tool_count


def test_tool_trace_values_and_order_are_preserved() -> None:
    module = _module()
    invocations = (
        CopilotToolInvocation(
            tool_call_id="call_a",
            tool_name="first",
            arguments_json='{ "value" : "原始" }',
            result_json='{"result":"一"}',
        ),
        CopilotToolInvocation(
            tool_call_id="call_b",
            tool_name="second",
            arguments_json='{"value":2}',
            result_json='{"result":"二"}',
        ),
    )
    turn = CopilotTurn(
        question="question",
        answer="answer",
        tool_call_content="working",
        tool_invocations=invocations,
    )
    metrics = module.CopilotRuntimeMetrics(2, 2, 0.5)
    result = module.CopilotObservedResult(turn, metrics)
    assert result.turn is turn
    assert result.turn.tool_invocations is invocations
    assert result.turn.tool_invocations == invocations


@pytest.mark.parametrize(
    ("option_name", "option_value"),
    [
        ("temperature", 0),
        ("seed", 7),
        ("metadata", {"tags": ["original"]}),
        (
            "tool_choice",
            {"type": "function", "function": {"name": "tool"}},
        ),
    ],
)
def test_inputs_and_request_options_remain_unchanged(
    monkeypatch: pytest.MonkeyPatch,
    option_name: str,
    option_value: object,
) -> None:
    module = _module()
    turn = _plain_turn()
    calls = _stub_structured_runtime(
        monkeypatch,
        module,
        turn=turn,
    )
    _set_clock(monkeypatch, module, [1.0, 1.25])
    client = object()
    question = "  preserve question  "
    context = {"experiment_dir": "实验目录"}
    options = {option_name: option_value}
    context_before = copy.deepcopy(context)
    options_before = copy.deepcopy(options)
    result = module.run_copilot_turn_with_observability(
        client,
        model="test-model",
        question=question,
        experiment_context=context,
        **options,
    )
    assert result.turn is turn
    assert len(calls) == 1
    assert calls[0]["client"] is client
    assert calls[0]["question"] == question
    assert calls[0]["experiment_context"] is context
    assert calls[0][option_name] is option_value
    assert context == context_before
    assert options == options_before


def test_timing_supports_zero_duration(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _module()
    _stub_structured_runtime(monkeypatch, module, turn=_plain_turn())
    clock_calls = _set_clock(monkeypatch, module, [5.0, 5.0])
    result = _run_observed(module, object())
    assert result.metrics.elapsed_seconds == 0.0
    assert type(result.metrics.elapsed_seconds) is float
    assert len(clock_calls) == 2


def test_timing_supports_non_integer_duration(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _module()
    _stub_structured_runtime(monkeypatch, module, turn=_plain_turn())
    _set_clock(monkeypatch, module, [0.125, 1.75])
    result = _run_observed(module, object())
    assert result.metrics.elapsed_seconds == 1.625
    assert type(result.metrics.elapsed_seconds) is float


def test_clock_is_private_and_read_exactly_twice_on_success(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _module()
    _stub_structured_runtime(monkeypatch, module, turn=_plain_turn())
    calls = _set_clock(monkeypatch, module, [2.0, 3.0])
    _run_observed(module, object())
    assert calls == [None, None]
    assert "_perf_counter" not in module.__all__
    assert "clock" not in inspect.signature(
        module.run_copilot_turn_with_observability
    ).parameters


@pytest.mark.parametrize(
    "failure_kind",
    ["first-provider", "second-provider", "tool", "malformed"],
)
def test_ordinary_failures_propagate_without_retry_or_wrapping(
    monkeypatch: pytest.MonkeyPatch,
    failure_kind: str,
) -> None:
    module = _module()
    error = RuntimeError(f"{failure_kind} failed")
    calls = _stub_structured_runtime(
        monkeypatch,
        module,
        error=error,
    )
    _set_clock(monkeypatch, module, [1.0])
    with pytest.raises(RuntimeError) as error_info:
        _run_observed(module, object())
    assert error_info.value is error
    assert len(calls) == 1


def test_invalid_final_content_error_propagates_unchanged(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _module()
    error = ValueError("final content must not be empty")
    calls = _stub_structured_runtime(
        monkeypatch,
        module,
        error=error,
    )
    _set_clock(monkeypatch, module, [1.0])
    with pytest.raises(ValueError) as error_info:
        _run_observed(module, object())
    assert error_info.value is error
    assert len(calls) == 1


@pytest.mark.parametrize(
    "error",
    [KeyboardInterrupt(), SystemExit(0), SystemExit(7), GeneratorExit()],
)
def test_control_flow_exceptions_are_not_captured(
    monkeypatch: pytest.MonkeyPatch,
    error: BaseException,
) -> None:
    module = _module()
    calls = _stub_structured_runtime(
        monkeypatch,
        module,
        error=error,
    )
    _set_clock(monkeypatch, module, [1.0])
    with pytest.raises(type(error)) as error_info:
        _run_observed(module, object())
    assert error_info.value is error
    assert len(calls) == 1


def test_failure_does_not_return_partial_result_or_retry(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _module()
    error = RuntimeError("failed")
    calls = _stub_structured_runtime(
        monkeypatch,
        module,
        error=error,
    )
    _set_clock(monkeypatch, module, [1.0])
    observed_results: list[object] = []
    with pytest.raises(RuntimeError):
        observed_results.append(_run_observed(module, object()))
    assert observed_results == []
    assert len(calls) == 1


def test_execution_reads_no_credentials_or_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _module()
    _stub_structured_runtime(monkeypatch, module, turn=_plain_turn())
    _set_clock(monkeypatch, module, [1.0, 2.0])

    def fail_environment(*args: object, **kwargs: object) -> object:
        raise AssertionError("environment access is forbidden")

    with monkeypatch.context() as context:
        context.setattr(os, "getenv", fail_environment)
        context.setattr(type(os.environ), "get", fail_environment)
        context.setattr(
            type(os.environ),
            "__getitem__",
            fail_environment,
        )
        _run_observed(module, object())


def test_execution_does_not_access_real_network(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _module()
    _stub_structured_runtime(monkeypatch, module, turn=_plain_turn())
    _set_clock(monkeypatch, module, [1.0, 2.0])

    def fail_network(*args: object, **kwargs: object) -> object:
        raise AssertionError("network access is forbidden")

    with monkeypatch.context() as context:
        context.setattr(socket, "create_connection", fail_network)
        context.setattr(socket.socket, "connect", fail_network)
        _run_observed(module, object())


def test_execution_does_not_write_files_or_create_directories(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    module = _module()
    _stub_structured_runtime(monkeypatch, module, turn=_plain_turn())
    _set_clock(monkeypatch, module, [1.0, 2.0])
    before = list(tmp_path.iterdir())

    def fail_write(*args: object, **kwargs: object) -> object:
        raise AssertionError("filesystem write is forbidden")

    with monkeypatch.context() as context:
        context.setattr(builtins, "open", fail_write)
        context.setattr(Path, "write_text", fail_write)
        context.setattr(Path, "write_bytes", fail_write)
        context.setattr(Path, "mkdir", fail_write)
        _run_observed(module, object())
    assert list(tmp_path.iterdir()) == before


def test_execution_emits_no_output_or_logging(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    caplog: pytest.LogCaptureFixture,
) -> None:
    module = _module()
    _stub_structured_runtime(monkeypatch, module, turn=_plain_turn())
    _set_clock(monkeypatch, module, [1.0, 2.0])
    with caplog.at_level(logging.DEBUG):
        _run_observed(module, object())
    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err == ""
    assert caplog.records == []


def test_execution_does_not_touch_session_or_cli_state(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _module()
    _stub_structured_runtime(monkeypatch, module, turn=_plain_turn())
    _set_clock(monkeypatch, module, [1.0, 2.0])
    before = {
        name: sys.modules.get(name)
        for name in ("copilot.cli", "copilot.interactive")
    }
    _run_observed(module, object())
    after = {
        name: sys.modules.get(name)
        for name in ("copilot.cli", "copilot.interactive")
    }
    assert after == before

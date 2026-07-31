import builtins
from copy import deepcopy
from dataclasses import FrozenInstanceError, fields, is_dataclass
import importlib
import inspect
import os
from pathlib import Path
import socket
import sys
from types import SimpleNamespace
from typing import Callable

import pytest


class RecordingCompletions:
    def __init__(self, outcomes: list[object]) -> None:
        self._outcomes = list(outcomes)
        self.calls: list[dict[str, object]] = []

    def create(self, **kwargs: object) -> object:
        self.calls.append(deepcopy(kwargs))
        outcome = self._outcomes.pop(0)
        if isinstance(outcome, BaseException):
            raise outcome
        return outcome


class FakeClient:
    def __init__(self, outcomes: list[object]) -> None:
        self.completions = RecordingCompletions(outcomes)
        self.chat = SimpleNamespace(completions=self.completions)
        self.close_calls = 0

    def close(self) -> None:
        self.close_calls += 1


def _load_module() -> object:
    return importlib.import_module("copilot.runtime_result")


def _load_api() -> Callable[..., object]:
    module = _load_module()
    return getattr(module, "run_copilot_turn_with_result")


def _message(
    *,
    content: object,
    tool_calls: object,
    role: object = "assistant",
) -> object:
    return SimpleNamespace(
        role=role,
        content=content,
        tool_calls=tool_calls,
    )


def _response(message: object) -> object:
    return SimpleNamespace(
        choices=[SimpleNamespace(message=message)]
    )


def _final_response(content: object = "Final answer") -> object:
    return _response(
        _message(content=content, tool_calls=None)
    )


def _tool_call(
    call_id: str,
    name: str,
    arguments: str,
) -> object:
    return SimpleNamespace(
        id=call_id,
        type="function",
        function=SimpleNamespace(
            name=name,
            arguments=arguments,
        ),
    )


def _tool_response(
    *tool_calls: object,
    content: str | None = None,
) -> object:
    return _response(
        _message(
            content=content,
            tool_calls=list(tool_calls),
        )
    )


def _run(
    client: FakeClient,
    **overrides: object,
) -> object:
    arguments: dict[str, object] = {
        "model": "test-model",
        "question": "Analyze this experiment.",
    }
    arguments.update(overrides)
    return _load_api()(client, **arguments)


def _install_tool_recorder(
    monkeypatch: pytest.MonkeyPatch,
    results: dict[str, dict[str, object]] | None = None,
    error: BaseException | None = None,
) -> list[tuple[str, dict[str, object]]]:
    adapter = importlib.import_module(
        "llm_adapters.openai_tool_adapter"
    )
    calls: list[tuple[str, dict[str, object]]] = []

    def invoke_tool(
        tool_name: str,
        arguments: dict[str, object],
    ) -> dict[str, object]:
        calls.append((tool_name, deepcopy(arguments)))
        if error is not None:
            raise error
        if results is None:
            return {"tool": tool_name}
        return deepcopy(results[tool_name])

    monkeypatch.setattr(adapter, "invoke_tool", invoke_tool)
    return calls


def test_runtime_result_module_exports_public_function() -> None:
    module = _load_module()

    assert callable(
        getattr(module, "run_copilot_turn_with_result")
    )


def test_package_export_is_the_module_function() -> None:
    module = _load_module()
    package = importlib.import_module("copilot")

    assert (
        getattr(package, "run_copilot_turn_with_result")
        is getattr(module, "run_copilot_turn_with_result")
    )


def test_public_function_has_exact_signature_and_return_type() -> None:
    function = _load_api()
    session = importlib.import_module("copilot.session")
    signature = inspect.signature(function)
    parameters = list(signature.parameters.values())

    assert [parameter.name for parameter in parameters] == [
        "client",
        "model",
        "question",
        "experiment_context",
        "request_options",
    ]
    assert parameters[0].kind is inspect.Parameter.POSITIONAL_OR_KEYWORD
    assert all(
        parameter.kind is inspect.Parameter.KEYWORD_ONLY
        for parameter in parameters[1:4]
    )
    assert parameters[3].default is None
    assert parameters[4].kind is inspect.Parameter.VAR_KEYWORD
    assert signature.return_annotation is session.CopilotTurn


def test_import_defines_no_new_result_dataclass_or_sdk(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delitem(sys.modules, "openai", raising=False)
    module = _load_module()
    local_dataclasses = [
        value
        for value in vars(module).values()
        if isinstance(value, type)
        and is_dataclass(value)
        and value.__module__ == module.__name__
    ]

    assert local_dataclasses == []
    assert "openai" not in sys.modules


def test_no_tool_turn_returns_existing_complete_turn() -> None:
    session = importlib.import_module("copilot.session")
    client = FakeClient([_final_response("Observed answer")])

    result = _run(client)

    assert type(result) is session.CopilotTurn
    assert result.question == "Analyze this experiment."
    assert result.answer == "Observed answer"
    assert result.tool_call_content is None
    assert result.tool_invocations == ()
    assert len(client.completions.calls) == 1


def test_single_tool_turn_returns_existing_invocation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    session = importlib.import_module("copilot.session")
    calls = _install_tool_recorder(
        monkeypatch,
        {"analyze_experiment": {"best": 0.9}},
    )
    first = _tool_response(
        _tool_call(
            "call-1",
            "analyze_experiment",
            '{"experiment_dir":"demo"}',
        ),
        content="I will inspect it.",
    )
    client = FakeClient([first, _final_response("Done")])

    result = _run(client)

    assert len(result.tool_invocations) == 1
    assert type(result.tool_invocations[0]) is (
        session.CopilotToolInvocation
    )
    assert calls == [
        ("analyze_experiment", {"experiment_dir": "demo"})
    ]


def test_multiple_tools_preserve_original_order(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = _install_tool_recorder(
        monkeypatch,
        {
            "analyze_experiment": {"score": 1},
            "compare_experiments": {"score": 2},
        },
    )
    first = _tool_response(
        _tool_call(
            "call-a",
            "analyze_experiment",
            '{"experiment_dir":"a"}',
        ),
        _tool_call(
            "call-b",
            "compare_experiments",
            '{"experiment_root":"root"}',
        ),
    )
    client = FakeClient([first, _final_response()])

    result = _run(client)

    assert [
        invocation.tool_name
        for invocation in result.tool_invocations
    ] == ["analyze_experiment", "compare_experiments"]
    assert [name for name, _ in calls] == [
        "analyze_experiment",
        "compare_experiments",
    ]


def test_tool_invocation_preserves_identifiers_and_raw_json(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    arguments_json = '{ "experiment_dir": "实验 | demo" }'
    _install_tool_recorder(
        monkeypatch,
        {"analyze_experiment": {"值": "一"}},
    )
    client = FakeClient(
        [
            _tool_response(
                _tool_call(
                    "call-unicode",
                    "analyze_experiment",
                    arguments_json,
                )
            ),
            _final_response(),
        ]
    )

    result = _run(client)
    invocation = result.tool_invocations[0]

    assert invocation.tool_call_id == "call-unicode"
    assert invocation.tool_name == "analyze_experiment"
    assert invocation.arguments_json == arguments_json
    assert invocation.result_json == '{"值":"一"}'


def test_tool_turn_preserves_content_and_execution_bounds(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = _install_tool_recorder(
        monkeypatch,
        {
            "analyze_experiment": {"ok": True},
            "compare_experiments": {"ok": True},
        },
    )
    client = FakeClient(
        [
            _tool_response(
                _tool_call(
                    "call-1",
                    "analyze_experiment",
                    '{"experiment_dir":"one"}',
                ),
                _tool_call(
                    "call-2",
                    "compare_experiments",
                    '{"experiment_root":"many"}',
                ),
                content="Inspecting both.",
            ),
            _final_response("Comparison complete."),
        ]
    )

    result = _run(client)

    assert result.tool_call_content == "Inspecting both."
    assert result.answer == "Comparison complete."
    assert len(client.completions.calls) == 2
    assert calls == [
        ("analyze_experiment", {"experiment_dir": "one"}),
        ("compare_experiments", {"experiment_root": "many"}),
    ]


def test_caller_inputs_are_not_modified(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_tool_recorder(monkeypatch)
    context = {
        "experiment_dir": "examples/demo",
        "metrics_config": "metrics.yaml",
    }
    request_options = {
        "temperature": 0.0,
        "metadata": {"request": "original"},
    }
    context_before = deepcopy(context)
    options_before = deepcopy(request_options)
    question = "Keep this exact question."
    client = FakeClient([_final_response()])

    result = _run(
        client,
        question=question,
        experiment_context=context,
        **request_options,
    )

    assert result.question == question
    assert context == context_before
    assert request_options == options_before


def test_result_and_invocations_are_immutable_tuples(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_tool_recorder(monkeypatch)
    client = FakeClient(
        [
            _tool_response(
                _tool_call(
                    "call-1",
                    "analyze_experiment",
                    '{"experiment_dir":"demo"}',
                )
            ),
            _final_response(),
        ]
    )

    result = _run(client)

    assert type(result.tool_invocations) is tuple
    with pytest.raises(FrozenInstanceError):
        result.answer = "changed"
    with pytest.raises(FrozenInstanceError):
        result.tool_invocations[0].tool_name = "changed"


def test_result_excludes_client_and_provider_response() -> None:
    provider_response = _final_response("Safe answer")
    client = FakeClient([provider_response])

    result = _run(client)

    values = tuple(
        getattr(result, field.name)
        for field in fields(result)
    )

    assert client not in values
    assert provider_response not in values
    assert tuple(field.name for field in fields(result)) == (
        "question",
        "answer",
        "tool_call_content",
        "tool_invocations",
    )


@pytest.mark.parametrize("failure_stage", ["first", "second"])
def test_provider_failures_propagate_original_exception(
    monkeypatch: pytest.MonkeyPatch,
    failure_stage: str,
) -> None:
    error = RuntimeError(f"{failure_stage} provider failure")
    if failure_stage == "first":
        outcomes = [error]
        expected_requests = 1
        expected_tool_calls = 0
    else:
        outcomes = [
            _tool_response(
                _tool_call(
                    "call-1",
                    "analyze_experiment",
                    '{"experiment_dir":"demo"}',
                )
            ),
            error,
        ]
        expected_requests = 2
        expected_tool_calls = 1
    calls = _install_tool_recorder(monkeypatch)
    client = FakeClient(outcomes)

    with pytest.raises(RuntimeError) as caught:
        _run(client)

    assert caught.value is error
    assert len(client.completions.calls) == expected_requests
    assert len(calls) == expected_tool_calls


def test_tool_failure_propagates_original_exception(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    error = LookupError("tool failed")
    calls = _install_tool_recorder(monkeypatch, error=error)
    client = FakeClient(
        [
            _tool_response(
                _tool_call(
                    "call-1",
                    "analyze_experiment",
                    '{"experiment_dir":"demo"}',
                )
            )
        ]
    )

    with pytest.raises(LookupError) as caught:
        _run(client)

    assert caught.value is error
    assert len(calls) == 1
    assert len(client.completions.calls) == 1


@pytest.mark.parametrize(
    ("response", "expected_error"),
    [
        (SimpleNamespace(), TypeError),
        (SimpleNamespace(choices={}), TypeError),
        (SimpleNamespace(choices=[]), ValueError),
        (
            _response(
                _message(
                    content="answer",
                    tool_calls=None,
                    role="user",
                )
            ),
            ValueError,
        ),
        (
            _tool_response(
                _tool_call(
                    "call-1",
                    "analyze_experiment",
                    "not-json",
                )
            ),
            ValueError,
        ),
    ],
)
def test_malformed_provider_data_keeps_existing_error_semantics(
    response: object,
    expected_error: type[Exception],
) -> None:
    client = FakeClient([response])

    with pytest.raises(expected_error):
        _run(client)


@pytest.mark.parametrize("content", [None, "", "   "])
def test_invalid_final_content_is_rejected(content: object) -> None:
    client = FakeClient([_final_response(content)])

    with pytest.raises(ValueError):
        _run(client)


def test_failures_do_not_construct_partial_result(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load_module()
    error = RuntimeError("final request failed")
    constructions: list[dict[str, object]] = []
    original_turn = getattr(module, "CopilotTurn")

    def record_turn(**fields: object) -> object:
        constructions.append(fields)
        return original_turn(**fields)

    monkeypatch.setattr(module, "CopilotTurn", record_turn)
    _install_tool_recorder(monkeypatch)
    client = FakeClient(
        [
            _tool_response(
                _tool_call(
                    "call-1",
                    "analyze_experiment",
                    '{"experiment_dir":"demo"}',
                )
            ),
            error,
        ]
    )

    with pytest.raises(RuntimeError) as caught:
        _run(client)

    assert caught.value is error
    assert constructions == []


@pytest.mark.parametrize(
    "interrupt",
    [KeyboardInterrupt(), SystemExit(7)],
)
def test_control_flow_exceptions_are_not_caught(
    interrupt: BaseException,
) -> None:
    client = FakeClient([interrupt])

    with pytest.raises(type(interrupt)) as caught:
        _run(client)

    assert caught.value is interrupt


def test_runtime_result_does_not_read_credentials(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    forbidden = {
        "OPENAI_API_KEY",
        "GH_TOKEN",
        "GITHUB_TOKEN",
    }
    original_get = os._Environ.get
    original_getitem = os._Environ.__getitem__

    def guarded_get(
        environ: object,
        key: str,
        default: object = None,
    ) -> object:
        if key in forbidden:
            raise AssertionError(f"credential read: {key}")
        return original_get(environ, key, default)

    def guarded_getitem(environ: object, key: str) -> str:
        if key in forbidden:
            raise AssertionError(f"credential read: {key}")
        return original_getitem(environ, key)

    monkeypatch.setattr(os._Environ, "get", guarded_get)
    monkeypatch.setattr(
        os._Environ,
        "__getitem__",
        guarded_getitem,
    )
    client = FakeClient([_final_response()])

    assert _run(client).answer == "Final answer"


def test_runtime_result_does_not_access_network(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def blocked(*args: object, **kwargs: object) -> object:
        raise AssertionError("network access attempted")

    monkeypatch.setattr(socket, "create_connection", blocked)
    monkeypatch.setattr(socket.socket, "connect", blocked)
    client = FakeClient([_final_response()])

    assert _run(client).answer == "Final answer"


def test_runtime_result_does_not_write_files_or_create_directories(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def blocked(*args: object, **kwargs: object) -> object:
        raise AssertionError("filesystem write attempted")

    monkeypatch.setattr(builtins, "open", blocked)
    monkeypatch.setattr(Path, "write_text", blocked)
    monkeypatch.setattr(Path, "write_bytes", blocked)
    monkeypatch.setattr(Path, "mkdir", blocked)
    client = FakeClient([_final_response()])

    assert _run(client).answer == "Final answer"


def test_runtime_result_prints_nothing(
    capsys: pytest.CaptureFixture[str],
) -> None:
    client = FakeClient([_final_response()])

    _run(client)

    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err == ""


def test_runtime_result_does_not_close_borrowed_client() -> None:
    client = FakeClient([_final_response()])

    _run(client)

    assert client.close_calls == 0


def test_context_values_are_platform_independent() -> None:
    context = {
        "experiment_dir": "experiments/demo",
        "metrics_config": "configs/metrics.yaml",
    }
    client = FakeClient([_final_response()])

    result = _run(client, experiment_context=context)

    assert result.question == "Analyze this experiment."
    sent_messages = client.completions.calls[0]["messages"]
    assert "experiments/demo" in sent_messages[1]["content"]
    assert "configs/metrics.yaml" in sent_messages[1]["content"]

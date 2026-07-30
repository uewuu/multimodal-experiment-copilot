"""Behavioral contract tests for the bounded in-memory Copilot session."""

from __future__ import annotations

from dataclasses import FrozenInstanceError, fields, is_dataclass
import importlib
import inspect
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
from types import SimpleNamespace

import pytest


class _SequentialCompletions:
    def __init__(self, responses: list[object]) -> None:
        self.responses = list(responses)
        self.calls: list[dict[str, object]] = []

    def create(self, **kwargs: object) -> object:
        self.calls.append(kwargs)
        if not self.responses:
            raise AssertionError("unexpected provider request")
        response = self.responses.pop(0)
        if isinstance(response, BaseException):
            raise response
        return response


class _FakeClient:
    def __init__(self, responses: list[object]) -> None:
        self.chat = SimpleNamespace(
            completions=_SequentialCompletions(responses)
        )
        self.close_calls = 0

    def close(self) -> None:
        self.close_calls += 1


def _tool_call(
    call_id: str,
    name: str,
    arguments: str,
) -> SimpleNamespace:
    return SimpleNamespace(
        id=call_id,
        type="function",
        function=SimpleNamespace(
            name=name,
            arguments=arguments,
        ),
    )


def _response(
    content: object,
    *,
    tool_calls: object = None,
    role: object = "assistant",
) -> SimpleNamespace:
    return SimpleNamespace(
        id="provider-response-id",
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(
                    role=role,
                    content=content,
                    tool_calls=tool_calls,
                )
            )
        ],
    )


def _load_session_module():
    return importlib.import_module("copilot.session")


def _load_public_api():
    module = _load_session_module()
    package = importlib.import_module("copilot")
    return (
        module,
        package.CopilotSession,
        package.CopilotTurn,
        package.CopilotToolInvocation,
    )


def _new_session(
    responses: list[object],
    *,
    model: str = "test-model",
    experiment_context: dict[str, object] | None = None,
    max_turns: int = 8,
    **request_options: object,
):
    _, session_type, _, _ = _load_public_api()
    client = _FakeClient(responses)
    session = session_type(
        client,
        model=model,
        experiment_context=experiment_context,
        max_turns=max_turns,
        **request_options,
    )
    return session, client


def _patch_tool(
    monkeypatch: pytest.MonkeyPatch,
    result_or_error: object,
) -> list[tuple[str, dict]]:
    adapter = importlib.import_module(
        "llm_adapters.openai_tool_adapter"
    )
    calls: list[tuple[str, dict]] = []

    def fake_invoke(name: str, arguments: dict) -> dict:
        calls.append((name, arguments))
        if isinstance(result_or_error, BaseException):
            raise result_or_error
        return result_or_error  # type: ignore[return-value]

    monkeypatch.setattr(adapter, "invoke_tool", fake_invoke)
    return calls


def _single_tool_responses(
    *,
    tool_content: str | None = None,
    final_content: str = "final answer",
    arguments: str = '{"experiment_dir":"demo"}',
) -> list[object]:
    return [
        _response(
            tool_content,
            tool_calls=[
                _tool_call(
                    "call-1",
                    "analyze_experiment",
                    arguments,
                )
            ],
        ),
        _response(final_content),
    ]


def test_session_module_is_importable() -> None:
    module = _load_session_module()
    assert module.__name__ == "copilot.session"


@pytest.mark.parametrize(
    "name",
    ["CopilotSession", "CopilotTurn", "CopilotToolInvocation"],
)
def test_copilot_package_exports_session_api(name: str) -> None:
    module, _, _, _ = _load_public_api()
    package = importlib.import_module("copilot")
    assert getattr(package, name) is getattr(module, name)
    assert name in package.__all__


def test_session_import_is_sdk_free_and_side_effect_free() -> None:
    _load_public_api()
    before = set(Path.cwd().iterdir())
    script = """
import os
import socket
import sys

os.environ["OPENAI_API_KEY"] = "must-not-be-read"
socket.create_connection = lambda *a, **k: (_ for _ in ()).throw(
    AssertionError("network access")
)
import copilot.session
import copilot
assert "openai" not in sys.modules
assert all(
    hasattr(copilot, name)
    for name in ("CopilotSession", "CopilotTurn", "CopilotToolInvocation")
)
"""
    environment = os.environ.copy()
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    completed = subprocess.run(
        [sys.executable, "-c", script],
        cwd=Path.cwd(),
        env=environment,
        text=True,
        capture_output=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    assert completed.stdout == ""
    assert completed.stderr == ""
    assert set(Path.cwd().iterdir()) == before


@pytest.mark.parametrize(
    ("type_name", "expected_fields"),
    [
        (
            "CopilotToolInvocation",
            [
                "tool_call_id",
                "tool_name",
                "arguments_json",
                "result_json",
            ],
        ),
        (
            "CopilotTurn",
            [
                "question",
                "answer",
                "tool_call_content",
                "tool_invocations",
            ],
        ),
    ],
)
def test_value_objects_have_exact_fields(
    type_name: str,
    expected_fields: list[str],
) -> None:
    module, _, _, _ = _load_public_api()
    value_type = getattr(module, type_name)
    assert [field.name for field in fields(value_type)] == expected_fields


def test_value_objects_are_frozen_slotted_dataclasses() -> None:
    _, _, turn_type, invocation_type = _load_public_api()
    invocation = invocation_type("id", "tool", "{}", "{}")
    turn = turn_type("question", "answer", None, (invocation,))
    for value, field_name in (
        (invocation, "tool_name"),
        (turn, "question"),
    ):
        assert is_dataclass(value)
        assert type(value).__dataclass_params__.frozen is True
        assert "__dict__" not in dir(value)
        with pytest.raises(FrozenInstanceError):
            setattr(value, field_name, "changed")


def test_session_constructor_has_exact_signature() -> None:
    _, session_type, _, _ = _load_public_api()
    parameters = inspect.signature(session_type.__init__).parameters
    assert list(parameters) == [
        "self",
        "client",
        "model",
        "experiment_context",
        "max_turns",
        "request_options",
    ]
    assert parameters["client"].kind is inspect.Parameter.POSITIONAL_OR_KEYWORD
    for name in ("model", "experiment_context", "max_turns"):
        assert parameters[name].kind is inspect.Parameter.KEYWORD_ONLY
    assert parameters["experiment_context"].default is None
    assert parameters["max_turns"].default == 8
    assert parameters["request_options"].kind is inspect.Parameter.VAR_KEYWORD


def test_session_exposes_exact_state_and_action_properties() -> None:
    _, session_type, _, _ = _load_public_api()
    for name in (
        "ask",
        "history",
        "turn_count",
        "model",
        "max_turns",
        "experiment_context",
        "reset",
    ):
        assert hasattr(session_type, name)
    assert list(inspect.signature(session_type.ask).parameters) == [
        "self",
        "question",
    ]
    assert list(inspect.signature(session_type.reset).parameters) == ["self"]


def test_session_has_no_ownership_or_interactive_lifecycle_api() -> None:
    _, session_type, _, _ = _load_public_api()
    for name in ("close", "__enter__", "__exit__", "interactive"):
        assert not hasattr(session_type, name)


def test_constructor_defaults_are_observable_without_provider_calls() -> None:
    session, client = _new_session([])
    assert session.model == "test-model"
    assert session.max_turns == 8
    assert session.experiment_context is None
    assert session.history == ()
    assert session.turn_count == 0
    assert client.chat.completions.calls == []


@pytest.mark.parametrize("model", [None, 1, True, "", " \t\r\n"])
def test_constructor_rejects_invalid_model(model: object) -> None:
    _, session_type, _, _ = _load_public_api()
    client = _FakeClient([])
    expected = TypeError if not isinstance(model, str) else ValueError
    with pytest.raises(expected):
        session_type(client, model=model)
    assert client.chat.completions.calls == []


def test_constructor_accepts_supported_contexts_without_mutation() -> None:
    _, session_type, _, _ = _load_public_api()
    contexts = [
        None,
        {},
        {"experiment_dir": "demo"},
        {
            "experiment_root": "runs",
            "metrics_config": "metrics.yaml",
        },
    ]
    for context in contexts:
        original = None if context is None else dict(context)
        session = session_type(
            _FakeClient([]),
            model="model",
            experiment_context=context,
        )
        assert session.experiment_context == original
        assert context == original


@pytest.mark.parametrize(
    ("context", "error_type"),
    [
        ([], TypeError),
        ({"unknown": "value"}, ValueError),
        (
            {"experiment_dir": "one", "experiment_root": "many"},
            ValueError,
        ),
        ({"experiment_dir": 1}, TypeError),
        ({"experiment_dir": ""}, ValueError),
        ({"metrics_config": " \t"}, ValueError),
    ],
)
def test_constructor_rejects_invalid_context(
    context: object,
    error_type: type[Exception],
) -> None:
    _, session_type, _, _ = _load_public_api()
    client = _FakeClient([])
    with pytest.raises(error_type):
        session_type(
            client,
            model="model",
            experiment_context=context,
        )
    assert client.chat.completions.calls == []


@pytest.mark.parametrize(
    "max_turns",
    [None, True, False, 0, -1, 1.5, "2"],
)
def test_constructor_requires_positive_integer_max_turns(
    max_turns: object,
) -> None:
    _, session_type, _, _ = _load_public_api()
    expected = (
        ValueError
        if isinstance(max_turns, int) and not isinstance(max_turns, bool)
        else TypeError
    )
    with pytest.raises(expected):
        session_type(
            _FakeClient([]),
            model="model",
            max_turns=max_turns,
        )


def test_constructor_rejects_runtime_controlled_options() -> None:
    _, session_type, _, _ = _load_public_api()
    for option in (
        "messages",
        "system_prompt",
        "system_instruction",
        "tools",
        "tool_choice",
    ):
        with pytest.raises(TypeError):
            session_type(
                _FakeClient([]),
                model="model",
                **{option: object()},
            )


def test_constructor_copies_external_context_and_options() -> None:
    context = {"experiment_dir": "original"}
    metadata = {"nested": ["original"]}
    session, client = _new_session(
        [_response("answer")],
        experiment_context=context,
        metadata=metadata,
    )
    context["experiment_dir"] = "changed"
    metadata["nested"].append("changed")
    exposed_context = session.experiment_context
    assert exposed_context == {"experiment_dir": "original"}
    assert exposed_context is not context
    exposed_context["experiment_dir"] = "also changed"
    session.ask("question")
    call = client.chat.completions.calls[0]
    assert call["metadata"] == {"nested": ["original"]}
    assert "original" in call["messages"][1]["content"]
    assert "changed" not in call["messages"][1]["content"]


def test_constructor_never_uses_or_closes_borrowed_client() -> None:
    _, session_type, _, _ = _load_public_api()
    client = _FakeClient([])
    session_type(client, model="model")
    assert client.chat.completions.calls == []
    assert client.close_calls == 0


def test_new_session_has_empty_immutable_history() -> None:
    session, _ = _new_session([])
    assert type(session.history) is tuple
    assert session.history == ()
    assert session.turn_count == len(session.history) == 0


@pytest.mark.parametrize("question", [None, 1, True, "", " \t\r\n"])
def test_ask_rejects_invalid_question_before_provider_call(
    question: object,
) -> None:
    session, client = _new_session([])
    expected = TypeError if not isinstance(question, str) else ValueError
    with pytest.raises(expected):
        session.ask(question)
    assert client.chat.completions.calls == []
    assert session.history == ()


def test_first_no_tool_turn_commits_answer_and_history() -> None:
    session, client = _new_session([_response("answer")])
    assert session.ask("question") == "answer"
    assert session.turn_count == 1
    turn = session.history[0]
    assert turn.question == "question"
    assert turn.answer == "answer"
    assert turn.tool_call_content is None
    assert turn.tool_invocations == ()
    assert [item["role"] for item in client.chat.completions.calls[0]["messages"]] == [
        "system",
        "user",
    ]


@pytest.mark.parametrize(
    "question",
    ["plain question", "  preserve spacing  ", "中文问题🙂"],
)
def test_ask_preserves_valid_question_text(question: str) -> None:
    session, _ = _new_session([_response("answer")])
    session.ask(question)
    assert session.history[0].question == question


def test_multiple_no_tool_turns_rebuild_complete_ordered_messages() -> None:
    session, client = _new_session(
        [_response("answer one"), _response("answer two")]
    )
    session.ask("question one")
    session.ask("question two")
    assert [turn.question for turn in session.history] == [
        "question one",
        "question two",
    ]
    messages = client.chat.completions.calls[1]["messages"]
    assert [message["role"] for message in messages] == [
        "system",
        "user",
        "assistant",
        "user",
    ]
    assert messages[1]["content"] == "question one"
    assert messages[2]["content"] == "answer one"
    assert messages[3]["content"] == "question two"


def test_every_request_preserves_model_and_request_options() -> None:
    session, client = _new_session(
        [_response("one"), _response("two")],
        model="chosen-model",
        temperature=0.25,
        max_tokens=100,
    )
    session.ask("one")
    session.ask("two")
    assert len(client.chat.completions.calls) == 2
    for call in client.chat.completions.calls:
        assert call["model"] == "chosen-model"
        assert call["temperature"] == 0.25
        assert call["max_tokens"] == 100
        assert type(call["tools"]) is list


def test_experiment_context_is_injected_without_becoming_history() -> None:
    context = {"experiment_dir": "实验目录"}
    session, client = _new_session(
        [_response("one"), _response("two")],
        experiment_context=context,
    )
    session.ask("first")
    session.ask("second")
    for call in client.chat.completions.calls:
        assert "实验目录" in call["messages"][1]["content"]
    assert [turn.question for turn in session.history] == ["first", "second"]


def test_single_tool_turn_records_complete_json_trace(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = _patch_tool(monkeypatch, {"score": 0.75})
    arguments = '{ "experiment_dir" : "demo" }'
    session, client = _new_session(
        _single_tool_responses(arguments=arguments),
        temperature=0.25,
    )
    assert session.ask("analyze") == "final answer"
    invocation = session.history[0].tool_invocations[0]
    assert calls == [
        ("analyze_experiment", {"experiment_dir": "demo"})
    ]
    assert invocation.tool_call_id == "call-1"
    assert invocation.tool_name == "analyze_experiment"
    assert invocation.arguments_json == arguments
    assert invocation.result_json == '{"score":0.75}'
    assert len(client.chat.completions.calls) == 2
    for request in client.chat.completions.calls:
        assert request["model"] == "test-model"
        assert request["temperature"] == 0.25


def test_multiple_tool_invocations_preserve_provider_order(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    adapter = importlib.import_module("llm_adapters.openai_tool_adapter")
    invoked: list[tuple[str, dict]] = []

    def invoke(name: str, arguments: dict) -> dict:
        invoked.append((name, arguments))
        return {"name": name}

    monkeypatch.setattr(adapter, "invoke_tool", invoke)
    first = _response(
        "working",
        tool_calls=[
            _tool_call("a", "analyze_experiment", '{"x":1}'),
            _tool_call("b", "compare_experiments", '{"x":2}'),
        ],
    )
    session, _ = _new_session([first, _response("done")])
    session.ask("question")
    turn = session.history[0]
    assert invoked == [
        ("analyze_experiment", {"x": 1}),
        ("compare_experiments", {"x": 2}),
    ]
    assert [item.tool_call_id for item in turn.tool_invocations] == ["a", "b"]
    assert [item.tool_name for item in turn.tool_invocations] == [
        "analyze_experiment",
        "compare_experiments",
    ]


@pytest.mark.parametrize("tool_content", [None, "I will inspect the run."])
def test_tool_call_assistant_content_is_recorded_exactly(
    monkeypatch: pytest.MonkeyPatch,
    tool_content: str | None,
) -> None:
    _patch_tool(monkeypatch, {"ok": True})
    session, _ = _new_session(
        _single_tool_responses(tool_content=tool_content)
    )
    session.ask("question")
    assert session.history[0].tool_call_content == tool_content


def test_next_turn_replays_complete_prior_tool_protocol(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_tool(monkeypatch, {"ok": True})
    session, client = _new_session(
        [
            *_single_tool_responses(tool_content="checking"),
            _response("second answer"),
        ]
    )
    session.ask("first")
    session.ask("second")
    messages = client.chat.completions.calls[2]["messages"]
    assert [message["role"] for message in messages] == [
        "system",
        "user",
        "assistant",
        "tool",
        "assistant",
        "user",
    ]
    assert messages[2]["content"] == "checking"
    assert messages[2]["tool_calls"][0]["function"]["arguments"] == (
        '{"experiment_dir":"demo"}'
    )
    assert messages[3]["tool_call_id"] == "call-1"
    assert messages[3]["content"] == '{"ok":true}'
    assert messages[4]["content"] == "final answer"


def test_no_tool_turn_records_no_tool_trace() -> None:
    session, _ = _new_session([_response("answer")])
    session.ask("question")
    turn = session.history[0]
    assert turn.tool_call_content is None
    assert type(turn.tool_invocations) is tuple
    assert turn.tool_invocations == ()


def test_history_never_contains_provider_or_client_objects(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_tool(monkeypatch, {"ok": True})
    responses = _single_tool_responses()
    session, client = _new_session(responses)
    session.ask("question")
    serialized = repr(session.history)
    assert "provider-response-id" not in serialized
    assert repr(client) not in serialized
    assert all(
        not hasattr(turn, "response")
        for turn in session.history
    )


def test_max_turns_one_evicts_oldest_complete_turn() -> None:
    session, client = _new_session(
        [_response("one"), _response("two"), _response("three")],
        max_turns=1,
    )
    session.ask("first")
    session.ask("second")
    session.ask("third")
    assert session.turn_count == 1
    assert [(turn.question, turn.answer) for turn in session.history] == [
        ("third", "three")
    ]
    for call, current_question in zip(
        client.chat.completions.calls,
        ("first", "second", "third"),
        strict=True,
    ):
        messages = call["messages"]
        assert [message["role"] for message in messages] == [
            "system",
            "user",
        ]
        assert messages[-1]["content"] == current_question
    final_contents = [
        message.get("content")
        for message in client.chat.completions.calls[-1]["messages"]
    ]
    assert "first" not in final_contents
    assert "one" not in final_contents
    assert "second" not in final_contents
    assert "two" not in final_contents


def test_max_turns_two_preserves_exact_boundary_and_order() -> None:
    session, _ = _new_session(
        [_response("one"), _response("two"), _response("three")],
        max_turns=2,
    )
    for question in ("first", "second", "third"):
        session.ask(question)
    assert [(turn.question, turn.answer) for turn in session.history] == [
        ("second", "two"),
        ("third", "three"),
    ]


def test_default_max_turns_retains_exactly_eight_successes() -> None:
    session, _ = _new_session(
        [_response(str(index)) for index in range(9)]
    )
    for index in range(9):
        session.ask(f"question-{index}")
    assert session.turn_count == 8
    assert session.history[0].question == "question-1"
    assert session.history[-1].question == "question-8"


def test_tool_turn_is_evicted_only_as_a_complete_unit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_tool(monkeypatch, {"ok": True})
    session, client = _new_session(
        [
            _response("old answer"),
            *_single_tool_responses(tool_content="checking"),
            _response("current answer"),
        ],
        max_turns=2,
    )
    session.ask("old turn")
    session.ask("tool turn")
    session.ask("current turn")
    assert [turn.question for turn in session.history] == [
        "tool turn",
        "current turn",
    ]
    messages = client.chat.completions.calls[3]["messages"]
    roles = [message["role"] for message in messages]
    assert roles == [
        "system",
        "user",
        "assistant",
        "tool",
        "assistant",
        "user",
    ]
    contents = [message.get("content") for message in messages]
    assert "old turn" not in contents
    assert "old answer" not in contents
    assert messages[1]["content"] == "tool turn"
    assert messages[2]["content"] == "checking"
    assert messages[2]["tool_calls"][0]["function"]["arguments"] == (
        '{"experiment_dir":"demo"}'
    )
    assert messages[3]["tool_call_id"] == "call-1"
    assert messages[3]["content"] == '{"ok":true}'
    assert messages[4]["content"] == "final answer"
    assert messages[5]["content"] == "current turn"


def test_reset_clears_history_and_turn_count() -> None:
    session, _ = _new_session([_response("answer")])
    session.ask("question")
    session.reset()
    assert session.history == ()
    assert session.turn_count == 0
    session.reset()
    assert session.history == ()


def test_reset_preserves_configuration_and_future_usability() -> None:
    context = {"experiment_dir": "demo"}
    session, client = _new_session(
        [_response("one"), _response("two")],
        model="model",
        experiment_context=context,
        max_turns=2,
        temperature=0.1,
    )
    session.ask("first")
    session.reset()
    assert session.model == "model"
    assert session.max_turns == 2
    assert session.experiment_context == context
    assert session.ask("second") == "two"
    assert client.chat.completions.calls[-1]["temperature"] == 0.1
    assert [
        message["role"]
        for message in client.chat.completions.calls[-1]["messages"]
    ] == ["system", "user"]


@pytest.mark.parametrize("failure_stage", ["first_request", "follow_up"])
def test_provider_failures_leave_history_atomic(
    monkeypatch: pytest.MonkeyPatch,
    failure_stage: str,
) -> None:
    error = RuntimeError("provider failed")
    if failure_stage == "first_request":
        responses = [_response("committed"), error]
    else:
        _patch_tool(monkeypatch, {"ok": True})
        responses = [
            _response("committed"),
            _single_tool_responses()[0],
            error,
        ]
    session, _ = _new_session(responses)
    session.ask("committed")
    before = session.history
    with pytest.raises(RuntimeError) as caught:
        session.ask("failed")
    assert caught.value is error
    assert session.history == before
    assert session.turn_count == 1


@pytest.mark.parametrize(
    "invalid_response",
    [
        SimpleNamespace(),
        SimpleNamespace(choices=[]),
        _response(None),
        _response(" \t\r\n"),
    ],
)
def test_invalid_final_responses_do_not_commit(
    invalid_response: object,
) -> None:
    session, _ = _new_session([_response("kept"), invalid_response])
    session.ask("kept")
    before = session.history
    with pytest.raises((TypeError, ValueError)):
        session.ask("invalid")
    assert session.history == before


@pytest.mark.parametrize(
    "failure_kind",
    ["invalid_json", "unknown_tool", "tool_error", "serialization"],
)
def test_tool_cycle_failures_do_not_commit_partial_turns(
    monkeypatch: pytest.MonkeyPatch,
    failure_kind: str,
) -> None:
    if failure_kind == "invalid_json":
        arguments = "{"
        result: object = {"ok": True}
    elif failure_kind == "unknown_tool":
        arguments = "{}"
        result = KeyError("unknown tool")
    elif failure_kind == "tool_error":
        arguments = "{}"
        result = RuntimeError("tool failed")
    else:
        arguments = "{}"
        result = {"bad": object()}
    _patch_tool(monkeypatch, result)
    session, _ = _new_session(
        [
            _response("kept"),
            _single_tool_responses(arguments=arguments)[0],
        ]
    )
    session.ask("kept")
    before = session.history
    with pytest.raises((TypeError, ValueError, KeyError, RuntimeError)):
        session.ask("failed")
    assert session.history == before


@pytest.mark.parametrize(
    "error_type",
    [KeyboardInterrupt, SystemExit],
)
def test_base_exceptions_propagate_without_committing(
    error_type: type[BaseException],
) -> None:
    error = error_type()
    session, _ = _new_session([_response("kept"), error])
    session.ask("kept")
    before = session.history
    with pytest.raises(error_type) as caught:
        session.ask("failed")
    assert caught.value is error
    assert session.history == before


def test_failure_at_full_bound_does_not_evict_committed_history() -> None:
    error = RuntimeError("failure")
    session, _ = _new_session(
        [_response("one"), _response("two"), error],
        max_turns=2,
    )
    session.ask("one")
    session.ask("two")
    before = session.history
    with pytest.raises(RuntimeError):
        session.ask("three")
    assert session.history == before
    assert session.turn_count == 2


@pytest.mark.parametrize(
    "scenario",
    ["success", "failure", "reset"],
)
def test_session_never_closes_caller_owned_client(
    scenario: str,
) -> None:
    responses: list[object]
    if scenario == "failure":
        responses = [RuntimeError("failure")]
    else:
        responses = [_response("answer")]
    session, client = _new_session(responses)
    if scenario == "success":
        session.ask("question")
    elif scenario == "failure":
        with pytest.raises(RuntimeError):
            session.ask("question")
    else:
        session.reset()
    assert client.close_calls == 0


def test_ask_has_no_file_process_network_or_console_side_effects(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    session, _ = _new_session([_response("answer")])

    def forbidden(*args: object, **kwargs: object) -> None:
        raise AssertionError("forbidden side effect")

    monkeypatch.setattr("builtins.open", forbidden)
    monkeypatch.setattr(subprocess, "run", forbidden)
    monkeypatch.setattr(socket, "create_connection", forbidden)
    session.ask("question")
    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err == ""


def test_session_state_contains_no_sdk_key_or_raw_response(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_tool(monkeypatch, {"ok": True})
    secret = "secret-api-key"
    monkeypatch.setenv("OPENAI_API_KEY", secret)
    session, _ = _new_session(_single_tool_responses())
    session.ask("question")
    state = repr(session.history)
    assert secret not in state
    assert "provider-response-id" not in state
    assert "openai" not in sys.modules


def test_history_is_an_isolated_tuple_snapshot() -> None:
    session, _ = _new_session([_response("one"), _response("two")])
    session.ask("first")
    first_snapshot = session.history
    session.ask("second")
    assert type(first_snapshot) is tuple
    assert len(first_snapshot) == 1
    assert len(session.history) == 2
    assert first_snapshot[0] is session.history[0]


def test_tool_trace_preserves_raw_arguments_and_strict_result_json(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    arguments = '{ "text": "中文", "flag": true }'
    _patch_tool(monkeypatch, {"z": 1, "a": "中文"})
    session, _ = _new_session(
        _single_tool_responses(arguments=arguments)
    )
    session.ask("question")
    trace = session.history[0].tool_invocations[0]
    assert trace.arguments_json == arguments
    assert trace.result_json == '{"a":"中文","z":1}'
    assert json.loads(trace.result_json) == {"a": "中文", "z": 1}

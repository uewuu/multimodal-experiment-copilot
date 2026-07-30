"""Contract tests for JSON-safe Copilot session history export."""

from __future__ import annotations

import builtins
import inspect
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
from types import SimpleNamespace
from typing import Callable, get_type_hints

import pytest

import llm_adapters.openai_tool_adapter as adapter
from copilot.session import (
    CopilotSession,
    CopilotToolInvocation,
    CopilotTurn,
)


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


def _require_export_history() -> Callable[
    [CopilotSession],
    list[dict[str, object]],
]:
    return getattr(CopilotSession, "export_history")


def _response(
    content: object,
    *,
    tool_calls: object = None,
) -> SimpleNamespace:
    return SimpleNamespace(
        id="provider-response-id",
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(
                    role="assistant",
                    content=content,
                    tool_calls=tool_calls,
                )
            )
        ],
    )


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


def _new_session(
    responses: list[object],
    *,
    max_turns: int = 8,
) -> tuple[CopilotSession, _FakeClient]:
    client = _FakeClient(responses)
    session = CopilotSession(
        client,
        model="test-model",
        max_turns=max_turns,
    )
    return session, client


def _invocation(
    position: int,
    *,
    arguments_json: str | None = None,
    result_json: str | None = None,
) -> CopilotToolInvocation:
    return CopilotToolInvocation(
        tool_call_id=f"call-{position}",
        tool_name=f"tool-{position}",
        arguments_json=(
            arguments_json
            if arguments_json is not None
            else f'{{"position":{position}}}'
        ),
        result_json=(
            result_json
            if result_json is not None
            else f'{{"result":{position}}}'
        ),
    )


def _turn(
    position: int,
    *,
    tool_call_content: str | None = None,
    invocations: tuple[CopilotToolInvocation, ...] = (),
) -> CopilotTurn:
    return CopilotTurn(
        question=f"question-{position}",
        answer=f"answer-{position}",
        tool_call_content=tool_call_content,
        tool_invocations=invocations,
    )


def _set_history(
    session: CopilotSession,
    turns: tuple[CopilotTurn, ...],
) -> None:
    session._history = turns


def _assert_json_native(value: object) -> None:
    if value is None or isinstance(value, str):
        return
    if type(value) is list:
        for item in value:
            _assert_json_native(item)
        return
    if type(value) is dict:
        for key, item in value.items():
            assert type(key) is str
            _assert_json_native(item)
        return
    pytest.fail(f"non-JSON-native value: {type(value).__name__}")


def test_export_history_has_exact_public_signature() -> None:
    export_history = _require_export_history()
    descriptor = inspect.getattr_static(
        CopilotSession,
        "export_history",
    )
    signature = inspect.signature(export_history)

    assert inspect.isfunction(descriptor)
    assert not isinstance(descriptor, (property, classmethod, staticmethod))
    assert list(signature.parameters) == ["self"]
    assert get_type_hints(export_history)["return"] == list[
        dict[str, object]
    ]
    assert not inspect.iscoroutinefunction(export_history)
    assert not inspect.isgeneratorfunction(export_history)


def test_export_history_returns_empty_list_for_new_session() -> None:
    export_history = _require_export_history()
    session, _ = _new_session([])

    exported = export_history(session)

    assert type(exported) is list
    assert exported == []


def test_export_history_maps_plain_turn_exactly() -> None:
    export_history = _require_export_history()
    session, _ = _new_session([_response("final answer")])
    session.ask("What happened?")

    exported = export_history(session)

    assert exported == [
        {
            "question": "What happened?",
            "answer": "final answer",
            "tool_call_content": None,
            "tool_invocations": [],
        }
    ]
    assert list(exported[0]) == [
        "question",
        "answer",
        "tool_call_content",
        "tool_invocations",
    ]


@pytest.mark.parametrize(
    "tool_call_content",
    [None, "I will inspect the experiment."],
)
def test_export_history_preserves_tool_call_content(
    monkeypatch: pytest.MonkeyPatch,
    tool_call_content: str | None,
) -> None:
    export_history = _require_export_history()
    responses = [
        _response(
            tool_call_content,
            tool_calls=[
                _tool_call(
                    "call-1",
                    "analyze_experiment",
                    '{"experiment_dir":"demo"}',
                )
            ],
        ),
        _response("done"),
    ]
    session, _ = _new_session(responses)
    monkeypatch.setattr(
        adapter,
        "invoke_tool",
        lambda name, arguments: {"ok": True},
    )
    session.ask("Analyze it.")

    exported = export_history(session)

    assert exported[0]["tool_call_content"] is tool_call_content


@pytest.mark.parametrize("invocation_count", [1, 2, 3])
def test_export_history_preserves_tool_invocation_count_and_order(
    monkeypatch: pytest.MonkeyPatch,
    invocation_count: int,
) -> None:
    export_history = _require_export_history()
    calls = [
        _tool_call(
            f"call-{position}",
            f"tool-{position}",
            f'{{ "position" : {position} }}',
        )
        for position in range(1, invocation_count + 1)
    ]
    session, _ = _new_session(
        [_response("working", tool_calls=calls), _response("done")]
    )
    tool_positions = iter(range(1, invocation_count + 1))
    monkeypatch.setattr(
        adapter,
        "invoke_tool",
        lambda name, arguments: {"result": next(tool_positions)},
    )
    session.ask("Run the tools.")

    exported = export_history(session)
    invocations = exported[0]["tool_invocations"]

    assert type(invocations) is list
    assert invocations == [
        {
            "tool_call_id": f"call-{position}",
            "tool_name": f"tool-{position}",
            "arguments_json": f'{{ "position" : {position} }}',
            "result_json": f'{{"result":{position}}}',
        }
        for position in range(1, invocation_count + 1)
    ]
    assert all(
        list(invocation) == [
            "tool_call_id",
            "tool_name",
            "arguments_json",
            "result_json",
        ]
        for invocation in invocations
    )


def test_export_history_preserves_multiple_turn_order() -> None:
    export_history = _require_export_history()
    session, _ = _new_session(
        [_response("one"), _response("two"), _response("three")]
    )
    for question in ("first", "second", "third"):
        session.ask(question)

    exported = export_history(session)

    assert [turn["question"] for turn in exported] == [
        turn.question for turn in session.history
    ]
    assert [turn["answer"] for turn in exported] == [
        "one",
        "two",
        "three",
    ]


@pytest.mark.parametrize("max_turns", [1, 2, 3])
def test_export_history_reflects_bounded_history(
    max_turns: int,
) -> None:
    export_history = _require_export_history()
    session, _ = _new_session(
        [_response(f"answer-{position}") for position in range(4)],
        max_turns=max_turns,
    )
    for position in range(4):
        session.ask(f"question-{position}")

    exported = export_history(session)
    expected_questions = [
        f"question-{position}"
        for position in range(4 - max_turns, 4)
    ]

    assert [turn["question"] for turn in exported] == expected_questions
    assert [turn.question for turn in session.history] == expected_questions
    assert len(exported) == max_turns


def test_export_history_returns_empty_list_after_reset() -> None:
    export_history = _require_export_history()
    session, _ = _new_session([_response("answer")])
    session.ask("question")
    session.reset()

    assert export_history(session) == []
    assert session.history == ()


@pytest.mark.parametrize(
    "text",
    [
        "中文实验",
        "日本語の実験",
        "emoji 👩🏽‍💻 and e\u0301",
    ],
)
def test_export_history_preserves_unicode(
    monkeypatch: pytest.MonkeyPatch,
    text: str,
) -> None:
    export_history = _require_export_history()
    arguments_json = json.dumps(
        {"text": text},
        ensure_ascii=False,
        separators=(",", ":"),
    )
    session, _ = _new_session(
        [
            _response(
                text,
                tool_calls=[
                    _tool_call("call-unicode", "unicode_tool", arguments_json)
                ],
            ),
            _response(text),
        ]
    )
    monkeypatch.setattr(
        adapter,
        "invoke_tool",
        lambda name, arguments: {"text": text},
    )
    session.ask(text)

    exported = export_history(session)
    invocation = exported[0]["tool_invocations"][0]

    assert exported[0]["question"] == text
    assert exported[0]["answer"] == text
    assert exported[0]["tool_call_content"] == text
    assert invocation["arguments_json"] == arguments_json
    assert text in invocation["result_json"]


@pytest.mark.parametrize(
    ("arguments_json", "result_json"),
    [
        ('{  "z" : 1, "a" : [ true, null ] }', '{"b": 2, "a": 1}'),
        (
            '{"escaped":"\\u4e2d\\u6587","literal":"日本語"}',
            '{"escaped":"\\ud83d\\ude80","literal":"中文"}',
        ),
    ],
)
def test_export_history_preserves_raw_json_strings(
    arguments_json: str,
    result_json: str,
) -> None:
    export_history = _require_export_history()
    session, _ = _new_session([])
    invocation = _invocation(
        1,
        arguments_json=arguments_json,
        result_json=result_json,
    )
    _set_history(
        session,
        (_turn(1, tool_call_content="working", invocations=(invocation,)),),
    )

    exported_invocation = export_history(session)[0][
        "tool_invocations"
    ][0]

    assert exported_invocation["arguments_json"] == arguments_json
    assert exported_invocation["result_json"] == result_json


def test_export_history_is_serializable_by_standard_json() -> None:
    export_history = _require_export_history()
    session, _ = _new_session([])
    _set_history(
        session,
        (
            _turn(
                1,
                tool_call_content="分析中",
                invocations=(_invocation(1),),
            ),
        ),
    )

    serialized = json.dumps(
        export_history(session),
        ensure_ascii=False,
        allow_nan=False,
    )

    assert '"question-1"' in serialized
    assert '"分析中"' in serialized


def test_export_history_contains_only_recursive_json_native_types() -> None:
    export_history = _require_export_history()
    session, _ = _new_session([])
    _set_history(
        session,
        (
            _turn(1),
            _turn(
                2,
                tool_call_content="working",
                invocations=(_invocation(1), _invocation(2)),
            ),
        ),
    )

    exported = export_history(session)

    _assert_json_native(exported)


def test_export_history_returns_fresh_nested_containers() -> None:
    export_history = _require_export_history()
    session, _ = _new_session([])
    _set_history(
        session,
        (
            _turn(
                1,
                tool_call_content="working",
                invocations=(_invocation(1),),
            ),
            _turn(
                2,
                tool_call_content="working",
                invocations=(_invocation(2),),
            ),
        ),
    )

    first = export_history(session)
    second = export_history(session)

    assert first is not second
    for first_turn, second_turn in zip(first, second, strict=True):
        assert first_turn is not second_turn
        assert (
            first_turn["tool_invocations"]
            is not second_turn["tool_invocations"]
        )
        for first_invocation, second_invocation in zip(
            first_turn["tool_invocations"],
            second_turn["tool_invocations"],
            strict=True,
        ):
            assert first_invocation is not second_invocation


@pytest.mark.parametrize(
    "mutation",
    ["question", "invocation_list", "invocation_field"],
)
def test_export_history_mutation_is_isolated(
    mutation: str,
) -> None:
    export_history = _require_export_history()
    session, _ = _new_session([])
    _set_history(
        session,
        (
            _turn(
                1,
                tool_call_content="working",
                invocations=(_invocation(1),),
            ),
        ),
    )
    original_history = session.history
    untouched_export = export_history(session)
    mutated_export = export_history(session)

    if mutation == "question":
        mutated_export[0]["question"] = "changed"
    elif mutation == "invocation_list":
        mutated_export[0]["tool_invocations"].append(
            {"tool_call_id": "extra"}
        )
    else:
        mutated_export[0]["tool_invocations"][0][
            "tool_name"
        ] = "changed"

    assert session.history == original_history
    assert export_history(session) == untouched_export
    assert untouched_export == [
        {
            "question": "question-1",
            "answer": "answer-1",
            "tool_call_content": "working",
            "tool_invocations": [
                {
                    "tool_call_id": "call-1",
                    "tool_name": "tool-1",
                    "arguments_json": '{"position":1}',
                    "result_json": '{"result":1}',
                }
            ],
        }
    ]


def test_export_history_excludes_failed_ask_transactionally() -> None:
    export_history = _require_export_history()
    error = RuntimeError("provider failed")
    session, client = _new_session([_response("success"), error])
    session.ask("successful question")
    before_failure = export_history(session)

    with pytest.raises(RuntimeError) as error_info:
        session.ask("failed question")

    calls_before_export = len(client.chat.completions.calls)
    assert error_info.value is error
    assert export_history(session) == before_failure
    assert len(client.chat.completions.calls) == calls_before_export


def test_export_history_has_no_execution_side_effects(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    export_history = _require_export_history()
    session, client = _new_session(
        [
            _response(
                "working",
                tool_calls=[
                    _tool_call("call-1", "test_tool", '{"value":1}')
                ],
            ),
            _response("done"),
        ]
    )
    tool_calls: list[tuple[str, dict]] = []

    def fake_invoke(name: str, arguments: dict) -> dict:
        tool_calls.append((name, arguments))
        return {"ok": True}

    monkeypatch.setattr(adapter, "invoke_tool", fake_invoke)
    session.ask("question")
    history_before = session.history
    provider_count = len(client.chat.completions.calls)
    tool_count = len(tool_calls)
    close_count = client.close_calls

    export_history(session)

    assert len(client.chat.completions.calls) == provider_count
    assert len(tool_calls) == tool_count
    assert client.close_calls == close_count
    assert session.history == history_before


@pytest.mark.parametrize(
    "forbidden_fields",
    [
        (
            "client",
            "provider",
            "provider_response",
            "model",
            "experiment_context",
            "request_options",
            "api_key",
            "authorization",
            "environment",
        )
    ],
)
def test_export_history_excludes_security_fields(
    forbidden_fields: tuple[str, ...],
) -> None:
    export_history = _require_export_history()
    session, _ = _new_session([])
    arguments_json = (
        '{"api_key":"must-remain","environment":"must-remain"}'
    )
    result_json = (
        '{"authorization":"must-remain","client":"must-remain"}'
    )
    _set_history(
        session,
        (
            _turn(
                1,
                tool_call_content="working",
                invocations=(
                    _invocation(
                        1,
                        arguments_json=arguments_json,
                        result_json=result_json,
                    ),
                ),
            ),
        ),
    )

    exported = export_history(session)
    turn = exported[0]
    invocation = turn["tool_invocations"][0]

    assert set(turn) == {
        "question",
        "answer",
        "tool_call_content",
        "tool_invocations",
    }
    assert set(invocation) == {
        "tool_call_id",
        "tool_name",
        "arguments_json",
        "result_json",
    }
    assert not set(forbidden_fields).intersection(turn)
    assert not set(forbidden_fields).intersection(invocation)
    assert invocation["arguments_json"] == arguments_json
    assert invocation["result_json"] == result_json


def test_export_history_has_import_network_environment_and_file_safety(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    export_history = _require_export_history()
    environment = os.environ.copy()
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    environment.pop("OPENAI_API_KEY", None)
    script = """
import sys
import copilot.session
assert "openai" not in sys.modules
"""
    status_before = subprocess.run(
        ["git", "status", "--short", "--untracked-files=all"],
        cwd=Path.cwd(),
        text=True,
        capture_output=True,
        check=True,
    ).stdout
    import_result = subprocess.run(
        [sys.executable, "-c", script],
        cwd=Path.cwd(),
        env=environment,
        text=True,
        capture_output=True,
        check=False,
    )
    session, client = _new_session([])
    _set_history(session, (_turn(1),))
    history_before = session.history
    directory_before = tuple(tmp_path.iterdir())

    def fail_side_effect(*args: object, **kwargs: object) -> object:
        raise AssertionError("export must not perform external I/O")

    with monkeypatch.context() as context:
        context.setattr(socket, "create_connection", fail_side_effect)
        context.setattr(socket.socket, "connect", fail_side_effect)
        context.setattr(os, "getenv", fail_side_effect)
        context.setattr(type(os.environ), "get", fail_side_effect)
        context.setattr(
            type(os.environ),
            "__getitem__",
            fail_side_effect,
        )
        context.setattr(builtins, "open", fail_side_effect)
        context.setattr(Path, "open", fail_side_effect)
        context.setattr(Path, "write_text", fail_side_effect)
        context.setattr(Path, "write_bytes", fail_side_effect)
        context.setattr(Path, "touch", fail_side_effect)
        context.setattr(Path, "mkdir", fail_side_effect)
        exported = export_history(session)

    status_after = subprocess.run(
        ["git", "status", "--short", "--untracked-files=all"],
        cwd=Path.cwd(),
        text=True,
        capture_output=True,
        check=True,
    ).stdout

    assert import_result.returncode == 0, import_result.stderr
    assert import_result.stdout == ""
    assert import_result.stderr == ""
    assert exported[0]["question"] == "question-1"
    assert client.chat.completions.calls == []
    assert client.close_calls == 0
    assert session.history == history_before
    assert tuple(tmp_path.iterdir()) == directory_before
    assert status_after == status_before

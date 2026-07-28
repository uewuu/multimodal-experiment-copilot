import builtins
import copy
import inspect
import json
import os
from pathlib import Path
import socket
import subprocess
from types import SimpleNamespace
import urllib.request

import pytest

import copilot.runtime as runtime
from copilot import run_copilot_turn
import generate_report
import llm_adapters.openai_tool_adapter as adapter


_MISSING = object()


class _SequentialCompletions:
    def __init__(self, outcomes: list[object]) -> None:
        self.outcomes = outcomes
        self.calls: list[
            tuple[tuple[object, ...], dict[str, object]]
        ] = []

    @property
    def call_count(self) -> int:
        return len(self.calls)

    def create(
        self,
        *args: object,
        **kwargs: object,
    ) -> object:
        self.calls.append((args, kwargs))
        index = self.call_count - 1
        if index >= len(self.outcomes):
            raise AssertionError(
                "client.chat.completions.create was called more "
                f"than the {len(self.outcomes)} expected time(s)"
            )
        outcome = self.outcomes[index]
        if isinstance(outcome, BaseException):
            raise outcome
        return outcome


class _FakeClient:
    def __init__(self, outcomes: list[object]) -> None:
        self.completions = _SequentialCompletions(outcomes)
        self.chat = SimpleNamespace(completions=self.completions)


class _AttributeObject:
    def __init__(self, **attributes: object) -> None:
        for name, value in attributes.items():
            setattr(self, name, value)


def _tool_call(
    *,
    call_id: str = "call_001",
    name: str = "analyze_experiment",
    arguments: str = '{"experiment_dir":"demo"}',
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
    content: object = "final answer",
    *,
    tool_calls: object = None,
    role: object = "assistant",
) -> SimpleNamespace:
    message = SimpleNamespace(
        role=role,
        content=content,
        tool_calls=tool_calls,
    )
    return SimpleNamespace(
        choices=[SimpleNamespace(message=message)],
    )


def _without_attribute(
    value: SimpleNamespace,
    attribute: str,
) -> SimpleNamespace:
    copied = SimpleNamespace(**vars(value))
    delattr(copied, attribute)
    return copied


def _patch_cycle(
    monkeypatch: pytest.MonkeyPatch,
    outcome: object,
) -> list[dict[str, object]]:
    calls: list[dict[str, object]] = []

    def fake_cycle(
        client: object,
        *,
        model: str,
        messages: list[dict],
        **request_options: object,
    ) -> object:
        calls.append(
            {
                "client": client,
                "model": model,
                "messages": messages,
                "request_options": request_options,
            }
        )
        if isinstance(outcome, BaseException):
            raise outcome
        return outcome

    monkeypatch.setattr(runtime, "run_tool_call_cycle", fake_cycle)
    return calls


def _run_with_response(
    monkeypatch: pytest.MonkeyPatch,
    response: object,
) -> str:
    _patch_cycle(monkeypatch, response)
    return run_copilot_turn(
        object(),
        model="test-model",
        question="Analyze the experiment.",
    )


def _write_experiment(
    experiment_dir: Path,
    *,
    r2: float,
    racc: float,
) -> None:
    experiment_dir.mkdir(parents=True)
    (experiment_dir / "hparams.yaml").write_text(
        "config:\n  batch_size: 8\n",
        encoding="utf-8",
    )
    (experiment_dir / "history.json").write_text(
        json.dumps(
            {
                "valid": {
                    "app": {
                        "r2": [[0, r2 / 2], [1, r2]],
                        "racc": [[0, racc / 2], [1, racc]],
                    }
                }
            }
        ),
        encoding="utf-8",
    )


def test_package_publicly_exports_run_copilot_turn() -> None:
    import copilot

    assert copilot.run_copilot_turn is runtime.run_copilot_turn
    assert run_copilot_turn is runtime.run_copilot_turn


def test_run_copilot_turn_has_exact_public_signature() -> None:
    signature = inspect.signature(run_copilot_turn)

    assert str(signature) == (
        "(client: object, *, model: str, question: str, "
        "experiment_context: dict[str, object] | None = None, "
        "**request_options: object) -> str"
    )


def test_runtime_delegates_and_returns_only_assistant_text(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    provider_response = _response("completed")
    calls = _patch_cycle(monkeypatch, provider_response)
    client = object()

    result = run_copilot_turn(
        client,
        model="test-model",
        question="Analyze this experiment.",
    )

    assert result == "completed"
    assert type(result) is str
    assert result is not provider_response
    assert len(calls) == 1
    assert calls[0]["client"] is client


@pytest.mark.parametrize(
    "reserved_option",
    ["messages", "system_prompt", "system_instruction"],
)
def test_runtime_rejects_caller_controlled_message_or_prompt_options(
    reserved_option: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = _patch_cycle(monkeypatch, _response())

    with pytest.raises(TypeError):
        run_copilot_turn(
            object(),
            model="test-model",
            question="Analyze this.",
            **{reserved_option: []},
        )

    assert calls == []


@pytest.mark.parametrize(
    "question",
    ["Analyze this experiment.", "请分析实验结果。"],
)
def test_runtime_accepts_plain_and_unicode_questions(
    question: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = _patch_cycle(monkeypatch, _response())

    run_copilot_turn(
        object(),
        model="test-model",
        question=question,
    )

    messages = calls[0]["messages"]
    assert messages[1] == {"role": "user", "content": question}


def test_runtime_rejects_non_string_questions_before_delegation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = _patch_cycle(monkeypatch, _response())

    for invalid in (None, 1, [], {}, b"question"):
        with pytest.raises(TypeError):
            run_copilot_turn(
                object(),
                model="test-model",
                question=invalid,  # type: ignore[arg-type]
            )

    assert calls == []


@pytest.mark.parametrize("question", ["", " ", "\t\r\n"])
def test_runtime_rejects_empty_or_whitespace_questions(
    question: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = _patch_cycle(monkeypatch, _response())

    with pytest.raises(ValueError):
        run_copilot_turn(
            object(),
            model="test-model",
            question=question,
        )

    assert calls == []


def test_runtime_preserves_question_content_without_stripping(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    question = "  Analyze the original question exactly.  "
    before = copy.deepcopy(question)
    calls = _patch_cycle(monkeypatch, _response("  exact answer  "))

    result = run_copilot_turn(
        object(),
        model="test-model",
        question=question,
    )

    assert question == before
    assert calls[0]["messages"][1]["content"] == question
    assert result == "  exact answer  "


@pytest.mark.parametrize(
    "context",
    [
        None,
        {},
        {"experiment_dir": "experiments/demo"},
        {
            "experiment_dir": "experiments/demo",
            "metrics_config": "metrics.yaml",
        },
        {
            "experiment_root": "experiments",
            "metrics_config": "metrics.yaml",
        },
    ],
)
def test_runtime_accepts_supported_experiment_contexts(
    context: dict[str, object] | None,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = _patch_cycle(monkeypatch, _response())

    run_copilot_turn(
        object(),
        model="test-model",
        question="Analyze.",
        experiment_context=context,
    )

    user_content = calls[0]["messages"][1]["content"]
    if context:
        assert "experiment context" in user_content.lower()
        assert json.dumps(
            context,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
        ) in user_content
    else:
        assert user_content == "Analyze."


def test_runtime_rejects_conflicting_experiment_paths(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = _patch_cycle(monkeypatch, _response())

    with pytest.raises(ValueError):
        run_copilot_turn(
            object(),
            model="test-model",
            question="Analyze.",
            experiment_context={
                "experiment_dir": "experiment",
                "experiment_root": "experiments",
            },
        )

    assert calls == []


def test_runtime_rejects_unknown_context_fields(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = _patch_cycle(monkeypatch, _response())

    with pytest.raises((TypeError, ValueError), match="unknown"):
        run_copilot_turn(
            object(),
            model="test-model",
            question="Analyze.",
            experiment_context={"dataset": "private"},
        )

    assert calls == []


def test_runtime_requires_an_actual_context_dict(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = _patch_cycle(monkeypatch, _response())

    for invalid in ("experiment", [], (), SimpleNamespace()):
        with pytest.raises(TypeError):
            run_copilot_turn(
                object(),
                model="test-model",
                question="Analyze.",
                experiment_context=invalid,  # type: ignore[arg-type]
            )

    assert calls == []


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("experiment_dir", Path("experiment")),
        ("experiment_root", 1),
        ("metrics_config", False),
    ],
)
def test_runtime_rejects_non_string_context_values(
    field: str,
    value: object,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = _patch_cycle(monkeypatch, _response())

    with pytest.raises(TypeError):
        run_copilot_turn(
            object(),
            model="test-model",
            question="Analyze.",
            experiment_context={field: value},
        )

    assert calls == []


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("experiment_dir", ""),
        ("experiment_root", " "),
        ("metrics_config", "\t\r\n"),
    ],
)
def test_runtime_rejects_blank_context_values(
    field: str,
    value: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = _patch_cycle(monkeypatch, _response())

    with pytest.raises(ValueError):
        run_copilot_turn(
            object(),
            model="test-model",
            question="Analyze.",
            experiment_context={field: value},
        )

    assert calls == []


def test_runtime_does_not_modify_experiment_context(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    context = {
        "experiment_dir": r"C:\实验\demo",
        "metrics_config": r"C:\配置\指标.yaml",
    }
    before = copy.deepcopy(context)
    calls = _patch_cycle(monkeypatch, _response())

    run_copilot_turn(
        object(),
        model="test-model",
        question="Analyze.",
        experiment_context=context,
    )

    assert context == before
    assert calls[0]["messages"] is not context


def test_runtime_builds_two_ordered_messages_with_deterministic_context(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    context = {
        "metrics_config": r"C:\配置\指标.yaml",
        "experiment_dir": r"C:\实验\demo",
    }
    calls = _patch_cycle(monkeypatch, _response())

    run_copilot_turn(
        object(),
        model="test-model",
        question="比较指标。",
        experiment_context=context,
    )

    messages = calls[0]["messages"]
    encoded = json.dumps(
        context,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
    )
    assert len(messages) == 2
    assert [message["role"] for message in messages] == [
        "system",
        "user",
    ]
    assert messages[1]["content"].startswith("比较指标。")
    assert encoded in messages[1]["content"]
    assert r"C:\\实验\\demo" in messages[1]["content"]
    assert "配置" in messages[1]["content"]
    assert all(message["role"] != "context" for message in messages)


def test_runtime_omits_empty_context_block(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = _patch_cycle(monkeypatch, _response())

    run_copilot_turn(
        object(),
        model="test-model",
        question="Analyze.",
        experiment_context={},
    )

    assert calls[0]["messages"][1] == {
        "role": "user",
        "content": "Analyze.",
    }
    assert "{}" not in calls[0]["messages"][1]["content"]


def test_system_prompt_contains_required_safety_semantics(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = _patch_cycle(monkeypatch, _response())

    run_copilot_turn(
        object(),
        model="test-model",
        question="Analyze.",
    )

    prompt = calls[0]["messages"][0]["content"].lower()
    required_fragments = (
        "machine-learning experiment",
        "tool",
        "metrics",
        "epochs",
        "paths",
        "configurations",
        "training outcomes",
        "failed",
        "facts",
        "inferences",
        "recommendations",
        "fi",
        "example",
        "training",
    )
    for fragment in required_fragments:
        assert fragment in prompt


def test_no_tool_path_requests_once_and_uses_registry_tools() -> None:
    response = _response("analysis complete", tool_calls=None)
    client = _FakeClient([response])

    result = run_copilot_turn(
        client,
        model="test-model",
        question="Analyze.",
    )

    assert result == "analysis complete"
    assert client.completions.call_count == 1
    args, kwargs = client.completions.calls[0]
    assert args == ()
    assert kwargs["model"] == "test-model"
    assert [message["role"] for message in kwargs["messages"]] == [
        "system",
        "user",
    ]
    assert type(kwargs["tools"]) is list
    assert [tool["function"]["name"] for tool in kwargs["tools"]] == [
        "analyze_experiment",
        "compare_experiments",
    ]


def test_no_tool_path_preserves_final_content_exactly() -> None:
    content = "\n  保留首尾内容。  \n"
    client = _FakeClient([_response(content)])

    result = run_copilot_turn(
        client,
        model="test-model",
        question="Analyze.",
    )

    assert result == content


def test_runtime_forwards_model_and_request_options() -> None:
    metadata = {"tags": ["runtime"]}
    tool_choice = {
        "type": "function",
        "function": {"name": "analyze_experiment"},
    }
    client = _FakeClient([_response("done")])

    run_copilot_turn(
        client,
        model="gpt-test",
        question="Analyze.",
        metadata=metadata,
        tool_choice=tool_choice,
        temperature=0,
    )

    kwargs = client.completions.calls[0][1]
    assert kwargs["model"] == "gpt-test"
    assert kwargs["metadata"] is metadata
    assert kwargs["tool_choice"] is tool_choice
    assert kwargs["temperature"] == 0


def test_runtime_preserves_adapter_rejection_of_explicit_tools() -> None:
    client = _FakeClient([_response("unused")])

    with pytest.raises(
        TypeError,
        match="tools are provided by the tool registry",
    ):
        run_copilot_turn(
            client,
            model="test-model",
            question="Analyze.",
            tools=[],
        )

    assert client.completions.calls == []


def test_single_tool_path_builds_follow_up_and_returns_text(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    arguments = '{ "experiment_dir" : "demo" }'
    client = _FakeClient(
        [
            _response(
                None,
                tool_calls=[_tool_call(arguments=arguments)],
            ),
            _response("tool analysis complete"),
        ]
    )
    monkeypatch.setattr(
        adapter,
        "invoke_tool",
        lambda name, decoded: {"best_r2": 0.75},
    )

    result = run_copilot_turn(
        client,
        model="test-model",
        question="Analyze.",
    )

    assert result == "tool analysis complete"
    assert client.completions.call_count == 2
    follow_up = client.completions.calls[1][1]["messages"]
    assert [message["role"] for message in follow_up] == [
        "system",
        "user",
        "assistant",
        "tool",
    ]
    assert follow_up[2]["tool_calls"][0] == {
        "id": "call_001",
        "type": "function",
        "function": {
            "name": "analyze_experiment",
            "arguments": arguments,
        },
    }
    assert json.loads(follow_up[3]["content"]) == {"best_r2": 0.75}


def test_multiple_tools_preserve_invocation_and_message_order(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = [
        _tool_call(
            call_id="call_a",
            name="first_tool",
            arguments='{"position":1}',
        ),
        _tool_call(
            call_id="call_b",
            name="second_tool",
            arguments='{"position":2}',
        ),
    ]
    client = _FakeClient(
        [
            _response(None, tool_calls=calls),
            _response("complete"),
        ]
    )
    invocations: list[tuple[str, dict]] = []

    def fake_invoke(name: str, arguments: dict) -> dict:
        invocations.append((name, arguments))
        return {"name": name}

    monkeypatch.setattr(adapter, "invoke_tool", fake_invoke)

    run_copilot_turn(
        client,
        model="test-model",
        question="Compare.",
    )

    assert invocations == [
        ("first_tool", {"position": 1}),
        ("second_tool", {"position": 2}),
    ]
    tool_messages = client.completions.calls[1][1]["messages"][-2:]
    assert [message["tool_call_id"] for message in tool_messages] == [
        "call_a",
        "call_b",
    ]


def test_runtime_rejects_second_response_tool_calls_without_third_request(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    invoked: list[str] = []
    client = _FakeClient(
        [
            _response(
                None,
                tool_calls=[_tool_call(call_id="call_first")],
            ),
            _response(
                None,
                tool_calls=[_tool_call(call_id="call_second")],
            ),
        ]
    )

    def fake_invoke(name: str, arguments: dict) -> dict:
        invoked.append(name)
        return {"ok": True}

    monkeypatch.setattr(adapter, "invoke_tool", fake_invoke)

    with pytest.raises(ValueError, match="tool_calls"):
        run_copilot_turn(
            client,
            model="test-model",
            question="Analyze.",
        )

    assert invoked == ["analyze_experiment"]
    assert client.completions.call_count == 2


def test_runtime_rejects_invalid_response_container_shapes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    missing_tool_calls = _response("answer")
    missing_tool_calls.choices[0].message = _without_attribute(
        missing_tool_calls.choices[0].message,
        "tool_calls",
    )
    invalid_responses = [
        {},
        SimpleNamespace(),
        SimpleNamespace(choices=None),
        SimpleNamespace(choices=()),
        SimpleNamespace(choices=[]),
        SimpleNamespace(choices=[{}]),
        SimpleNamespace(choices=[SimpleNamespace()]),
        SimpleNamespace(
            choices=[SimpleNamespace(message={})]
        ),
        missing_tool_calls,
    ]

    for response in invalid_responses:
        _patch_cycle(monkeypatch, response)
        with pytest.raises((TypeError, ValueError)):
            run_copilot_turn(
                object(),
                model="test-model",
                question="Analyze.",
            )


@pytest.mark.parametrize("role", [None, "user", 1])
def test_runtime_rejects_missing_or_invalid_final_role(
    role: object,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    response = _response("answer", role=role)
    if role is None:
        response.choices[0].message = _without_attribute(
            response.choices[0].message,
            "role",
        )

    _patch_cycle(monkeypatch, response)
    with pytest.raises((TypeError, ValueError), match="role"):
        run_copilot_turn(
            object(),
            model="test-model",
            question="Analyze.",
        )


@pytest.mark.parametrize("tool_calls", [None, []])
def test_runtime_accepts_resolved_final_tool_calls(
    tool_calls: object,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    assert _run_with_response(
        monkeypatch,
        _response("answer", tool_calls=tool_calls),
    ) == "answer"


def test_runtime_rejects_invalid_final_tool_calls_type(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    for invalid in ((), {}, "calls"):
        _patch_cycle(
            monkeypatch,
            _response("answer", tool_calls=invalid),
        )
        with pytest.raises(TypeError, match="tool_calls"):
            run_copilot_turn(
                object(),
                model="test-model",
                question="Analyze.",
            )


def test_runtime_rejects_unresolved_final_tool_calls(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_cycle(
        monkeypatch,
        _response("pending", tool_calls=[_tool_call()]),
    )

    with pytest.raises(ValueError, match="tool_calls"):
        run_copilot_turn(
            object(),
            model="test-model",
            question="Analyze.",
        )


@pytest.mark.parametrize(
    ("content", "expected_exception"),
    [
        (_MISSING, ValueError),
        (None, ValueError),
        (1, TypeError),
        ([], TypeError),
        ("", ValueError),
        (" \t\r\n", ValueError),
    ],
)
def test_runtime_rejects_missing_invalid_or_blank_final_content(
    content: object,
    expected_exception: type[Exception],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    response = _response(content)
    if content is _MISSING:
        response.choices[0].message = _without_attribute(
            response.choices[0].message,
            "content",
        )
    _patch_cycle(monkeypatch, response)

    with pytest.raises(expected_exception, match="content"):
        run_copilot_turn(
            object(),
            model="test-model",
            question="Analyze.",
        )


def test_runtime_uses_only_first_choice(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    response = _response("first")
    response.choices.append(
        SimpleNamespace(
            message=SimpleNamespace(
                role="assistant",
                content=None,
                tool_calls=[_tool_call()],
            )
        )
    )

    assert _run_with_response(monkeypatch, response) == "first"


def test_runtime_accepts_non_namespace_attribute_response(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    response = _AttributeObject(
        choices=[
            _AttributeObject(
                message=_AttributeObject(
                    role="assistant",
                    content="custom response",
                    tool_calls=None,
                )
            )
        ]
    )

    assert _run_with_response(monkeypatch, response) == "custom response"


@pytest.mark.parametrize("failure_stage", ["first", "second"])
def test_runtime_propagates_client_exceptions_unchanged(
    failure_stage: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    error = RuntimeError(f"{failure_stage} request failed")
    if failure_stage == "first":
        client = _FakeClient([error])
    else:
        client = _FakeClient(
            [
                _response(None, tool_calls=[_tool_call()]),
                error,
            ]
        )
        monkeypatch.setattr(
            adapter,
            "invoke_tool",
            lambda name, arguments: {"ok": True},
        )

    with pytest.raises(RuntimeError) as error_info:
        run_copilot_turn(
            client,
            model="test-model",
            question="Analyze.",
        )

    assert error_info.value is error


@pytest.mark.parametrize(
    "error",
    [
        RuntimeError("tool failed"),
        FileNotFoundError("missing experiment"),
        NotADirectoryError("invalid root"),
        KeyError("unknown tool: missing"),
    ],
)
def test_runtime_propagates_tool_exceptions_unchanged(
    error: Exception,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = _FakeClient(
        [_response(None, tool_calls=[_tool_call()])]
    )

    def fail_tool(name: str, arguments: dict) -> dict:
        raise error

    monkeypatch.setattr(adapter, "invoke_tool", fail_tool)

    with pytest.raises(type(error)) as error_info:
        run_copilot_turn(
            client,
            model="test-model",
            question="Analyze.",
        )

    assert error_info.value is error


def test_runtime_preserves_malformed_tool_json_cause() -> None:
    client = _FakeClient(
        [
            _response(
                None,
                tool_calls=[_tool_call(arguments='{"broken":}')],
            )
        ]
    )

    with pytest.raises(ValueError) as error_info:
        run_copilot_turn(
            client,
            model="test-model",
            question="Analyze.",
        )

    assert isinstance(error_info.value.__cause__, json.JSONDecodeError)


@pytest.mark.parametrize(
    "arguments",
    ["[]", '"text"', "1", "true", "null"],
)
def test_runtime_rejects_non_object_tool_arguments(
    arguments: str,
) -> None:
    client = _FakeClient(
        [
            _response(
                None,
                tool_calls=[_tool_call(arguments=arguments)],
            )
        ]
    )

    with pytest.raises(TypeError, match="decode to an object"):
        run_copilot_turn(
            client,
            model="test-model",
            question="Analyze.",
        )


def test_runtime_propagates_tool_result_serialization_errors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = _FakeClient(
        [_response(None, tool_calls=[_tool_call()])]
    )
    monkeypatch.setattr(
        adapter,
        "invoke_tool",
        lambda name, arguments: object(),
    )

    with pytest.raises(TypeError):
        run_copilot_turn(
            client,
            model="test-model",
            question="Analyze.",
        )


def test_runtime_preserves_context_and_nested_request_options() -> None:
    context = {
        "experiment_dir": r"C:\实验\demo",
        "metrics_config": r"C:\实验\metrics.yaml",
    }
    metadata = {"tags": ["original"], "nested": {"value": 1}}
    tool_choice = {
        "type": "function",
        "function": {"name": "analyze_experiment"},
    }
    context_before = copy.deepcopy(context)
    metadata_before = copy.deepcopy(metadata)
    tool_choice_before = copy.deepcopy(tool_choice)
    client = _FakeClient([_response("done")])

    run_copilot_turn(
        client,
        model="test-model",
        question="Analyze.",
        experiment_context=context,
        metadata=metadata,
        tool_choice=tool_choice,
    )

    assert context == context_before
    assert metadata == metadata_before
    assert tool_choice == tool_choice_before
    request = client.completions.calls[0][1]
    assert request["metadata"] is metadata
    assert request["tool_choice"] is tool_choice


def test_system_prompt_is_stable_across_calls(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = _patch_cycle(monkeypatch, _response())

    run_copilot_turn(
        object(),
        model="test-model",
        question="First.",
    )
    first_prompt = calls[0]["messages"][0]["content"]
    run_copilot_turn(
        object(),
        model="test-model",
        question="Second.",
    )
    second_prompt = calls[1]["messages"][0]["content"]

    assert first_prompt == second_prompt
    assert type(first_prompt) is str
    assert calls[0]["messages"] is not calls[1]["messages"]
    assert calls[0]["messages"][0] is not calls[1]["messages"][0]


def test_runtime_has_no_network_key_sdk_file_or_subprocess_side_effects(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = Path(runtime.__file__).read_text(encoding="utf-8").lower()
    forbidden_fragments = (
        "import openai",
        "from openai",
        "openai_api_key",
        "gh_token",
        "github_token",
        "os.getenv",
        "os.environ",
    )
    for fragment in forbidden_fragments:
        assert fragment not in source

    calls = _patch_cycle(monkeypatch, _response("safe"))
    original_import = builtins.__import__

    def fail(name: str, *args: object, **kwargs: object) -> object:
        raise AssertionError(f"forbidden side effect: {name}")

    def guarded_import(
        name: str,
        globals: dict | None = None,
        locals: dict | None = None,
        fromlist: tuple | list = (),
        level: int = 0,
    ) -> object:
        if name.split(".", 1)[0] in {"openai", "requests"}:
            raise AssertionError(f"forbidden runtime import: {name}")
        return original_import(
            name,
            globals,
            locals,
            fromlist,
            level,
        )

    with monkeypatch.context() as context:
        context.setattr(socket, "create_connection", fail)
        context.setattr(socket.socket, "connect", fail)
        context.setattr(urllib.request, "urlopen", fail)
        context.setattr(os, "getenv", fail)
        context.setattr(type(os.environ), "get", fail)
        context.setattr(type(os.environ), "__getitem__", fail)
        context.setattr(subprocess, "run", fail)
        context.setattr(subprocess, "Popen", fail)
        context.setattr(builtins, "open", fail)
        context.setattr(Path, "write_text", fail)
        context.setattr(Path, "write_bytes", fail)
        context.setattr(Path, "mkdir", fail)
        context.setattr(generate_report, "write_summary_json", fail)
        context.setattr(generate_report, "write_markdown_report", fail)
        context.setattr(builtins, "__import__", guarded_import)

        result = run_copilot_turn(
            object(),
            model="test-model",
            question="Analyze.",
        )

    assert result == "safe"
    assert len(calls) == 1


def test_sequential_fake_client_rejects_unexpected_extra_request() -> None:
    client = _FakeClient([_response("only")])

    client.chat.completions.create(model="test", messages=[])
    with pytest.raises(
        AssertionError,
        match="called more than the 1 expected time",
    ):
        client.chat.completions.create(model="test", messages=[])

    assert client.completions.call_count == 2


def test_real_single_experiment_tool_flow(
    tmp_path: Path,
) -> None:
    experiment_dir = tmp_path / "单实验"
    _write_experiment(experiment_dir, r2=0.75, racc=0.9)
    arguments = json.dumps(
        {"experiment_dir": str(experiment_dir)},
        ensure_ascii=False,
    )
    client = _FakeClient(
        [
            _response(
                None,
                tool_calls=[
                    _tool_call(
                        name="analyze_experiment",
                        arguments=arguments,
                    )
                ],
            ),
            _response("The experiment was analyzed."),
        ]
    )

    result = run_copilot_turn(
        client,
        model="test-model",
        question="分析这个实验。",
        experiment_context={
            "experiment_dir": str(experiment_dir),
        },
    )

    assert result == "The experiment was analyzed."
    assert client.completions.call_count == 2
    tool_message = client.completions.calls[1][1]["messages"][-1]
    assert tool_message["role"] == "tool"
    payload = json.loads(tool_message["content"])
    assert payload["configuration"]["batch_size"] == 8
    assert payload["validation_metrics"]["r2"]["best_value"] == 0.75
    assert payload["validation_metrics"]["r2"]["best_epoch"] == 1
    assert not (tmp_path / "outputs").exists()


def test_real_multi_experiment_tool_flow(
    tmp_path: Path,
) -> None:
    experiment_root = tmp_path / "experiments"
    experiment_a = experiment_root / "experiment_a"
    experiment_b = experiment_root / "experiment_b"
    _write_experiment(experiment_a, r2=0.6, racc=0.8)
    _write_experiment(experiment_b, r2=0.9, racc=0.95)
    arguments = json.dumps(
        {"experiment_root": str(experiment_root)},
        ensure_ascii=False,
    )
    client = _FakeClient(
        [
            _response(
                None,
                tool_calls=[
                    _tool_call(
                        name="compare_experiments",
                        arguments=arguments,
                    )
                ],
            ),
            _response("The experiments were compared."),
        ]
    )

    result = run_copilot_turn(
        client,
        model="test-model",
        question="Compare the experiments.",
        experiment_context={
            "experiment_root": str(experiment_root),
        },
    )

    assert result == "The experiments were compared."
    assert client.completions.call_count == 2
    messages = client.completions.calls[1][1]["messages"]
    payload = json.loads(messages[-1]["content"])
    assert payload["experiment_counts"] == {
        "total": 2,
        "successful": 2,
        "failed": 0,
    }
    assert [
        record["experiment_name"]
        for record in payload["comparison_records"]
    ] == ["experiment_b", "experiment_a"]
    assert [
        record["best_r2"]
        for record in payload["comparison_records"]
    ] == [0.9, 0.6]
    assert not (tmp_path / "outputs").exists()

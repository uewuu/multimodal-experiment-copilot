import builtins
import copy
import inspect
import json
import os
import socket
from types import SimpleNamespace

import pytest

import llm_adapters
import llm_adapters.openai_tool_adapter as adapter


class _SequentialCompletions:
    def __init__(self, outcomes: list[object]) -> None:
        self.outcomes = outcomes
        self.calls: list[
            tuple[tuple[object, ...], dict[str, object]]
        ] = []
        self.args: list[tuple[object, ...]] = []
        self.kwargs: list[dict[str, object]] = []

    @property
    def call_count(self) -> int:
        return len(self.calls)

    def create(
        self,
        *args: object,
        **kwargs: object,
    ) -> object:
        self.calls.append((args, kwargs))
        self.args.append(args)
        self.kwargs.append(kwargs)
        outcome_index = self.call_count - 1
        if outcome_index >= len(self.outcomes):
            raise AssertionError(
                "client.chat.completions.create was called more "
                f"than the {len(self.outcomes)} expected time(s)"
            )

        outcome = self.outcomes[outcome_index]
        if isinstance(outcome, BaseException):
            raise outcome
        return outcome


class _FakeClient:
    def __init__(self, responses: list[object]) -> None:
        self.completions = _SequentialCompletions(responses)
        self.chat = SimpleNamespace(completions=self.completions)


class _AttributeObject:
    def __init__(self, **attributes: object) -> None:
        for name, value in attributes.items():
            setattr(self, name, value)


def _tool_call(
    *,
    call_id: object = "call_001",
    call_type: object = "function",
    name: object = "analyze_experiment",
    arguments: object = '{"experiment_dir":"demo"}',
) -> SimpleNamespace:
    return SimpleNamespace(
        id=call_id,
        type=call_type,
        function=SimpleNamespace(
            name=name,
            arguments=arguments,
        ),
    )


def _response(
    tool_calls: object,
    *,
    role: object = "assistant",
    content: object = None,
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


def test_package_publicly_exports_run_tool_call_cycle() -> None:
    assert llm_adapters.run_tool_call_cycle is adapter.run_tool_call_cycle


def test_run_tool_call_cycle_has_exact_signature() -> None:
    signature = inspect.signature(adapter.run_tool_call_cycle)
    parameters = list(signature.parameters.values())

    assert [parameter.name for parameter in parameters] == [
        "client",
        "model",
        "messages",
        "request_options",
    ]
    assert parameters[0].kind is inspect.Parameter.POSITIONAL_OR_KEYWORD
    assert parameters[0].annotation is object
    assert parameters[1].kind is inspect.Parameter.KEYWORD_ONLY
    assert parameters[1].annotation is str
    assert parameters[2].kind is inspect.Parameter.KEYWORD_ONLY
    assert parameters[2].annotation == list[dict]
    assert parameters[3].kind is inspect.Parameter.VAR_KEYWORD
    assert parameters[3].annotation is object
    assert signature.return_annotation is object


@pytest.mark.parametrize("tool_calls", [None, []])
def test_cycle_without_tool_calls_requests_once_and_returns_first_response(
    monkeypatch: pytest.MonkeyPatch,
    tool_calls: object,
) -> None:
    response = _response(tool_calls, content="final answer")
    client = _FakeClient([response])

    def fail_invoke(*args: object, **kwargs: object) -> object:
        pytest.fail("no tool may be invoked")

    monkeypatch.setattr(adapter, "invoke_tool", fail_invoke)

    result = adapter.run_tool_call_cycle(
        client,
        model="test-model",
        messages=[{"role": "user", "content": "Analyze this."}],
    )

    assert result is response
    assert client.completions.call_count == 1


def test_cycle_with_tool_calls_requests_twice_and_returns_second_response(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    first_response = _response([_tool_call()])
    second_response = _response(None, content="analysis complete")
    client = _FakeClient([first_response, second_response])
    monkeypatch.setattr(
        adapter,
        "invoke_tool",
        lambda tool_name, arguments: {"ok": True},
    )

    result = adapter.run_tool_call_cycle(
        client,
        model="test-model",
        messages=[{"role": "user", "content": "Analyze this."}],
    )

    assert result is second_response
    assert client.completions.call_count == 2


def test_cycle_builds_follow_up_messages_in_protocol_order(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    arguments = '{ "experiment_dir" : "demo" }'
    tool_call = _tool_call(arguments=arguments)
    first_response = _response(
        [tool_call],
        content="I will inspect the experiment.",
    )
    client = _FakeClient([first_response, _response(None)])
    original_messages = [{"role": "user", "content": "Analyze this."}]
    monkeypatch.setattr(
        adapter,
        "invoke_tool",
        lambda tool_name, decoded_arguments: {"metric": 0.75},
    )

    adapter.run_tool_call_cycle(
        client,
        model="test-model",
        messages=original_messages,
    )

    follow_up_messages = client.completions.kwargs[1]["messages"]
    assert follow_up_messages == [
        original_messages[0],
        {
            "role": "assistant",
            "content": "I will inspect the experiment.",
            "tool_calls": [
                {
                    "id": "call_001",
                    "type": "function",
                    "function": {
                        "name": "analyze_experiment",
                        "arguments": arguments,
                    },
                }
            ],
        },
        {
            "role": "tool",
            "tool_call_id": "call_001",
            "content": '{"metric":0.75}',
        },
    ]
    assert follow_up_messages[0] is original_messages[0]
    assert follow_up_messages[1]["tool_calls"] is not first_response.choices[
        0
    ].message.tool_calls


def test_cycle_executes_multiple_tools_in_original_order(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    first_response = _response(
        [
            _tool_call(
                call_id="call_001",
                name="first_tool",
                arguments='{"position":1}',
            ),
            _tool_call(
                call_id="call_002",
                name="second_tool",
                arguments='{"position":2}',
            ),
        ]
    )
    client = _FakeClient([first_response, _response(None)])
    invocations: list[tuple[str, dict]] = []

    def fake_invoke(tool_name: str, arguments: dict) -> dict:
        invocations.append((tool_name, arguments))
        return {"tool": tool_name}

    monkeypatch.setattr(adapter, "invoke_tool", fake_invoke)

    adapter.run_tool_call_cycle(
        client,
        model="test-model",
        messages=[],
    )

    assert invocations == [
        ("first_tool", {"position": 1}),
        ("second_tool", {"position": 2}),
    ]
    assert [
        message["tool_call_id"]
        for message in client.completions.kwargs[1]["messages"][-2:]
    ] == ["call_001", "call_002"]


def test_cycle_stops_after_second_response_even_with_more_tool_calls(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    first_response = _response([_tool_call(call_id="call_first")])
    second_response = _response([_tool_call(call_id="call_second")])
    client = _FakeClient([first_response, second_response])
    invoked_ids: list[str] = []

    def fake_invoke(tool_name: str, arguments: dict) -> dict:
        invoked_ids.append(arguments.get("id", "first"))
        return {"ok": True}

    first_response.choices[0].message.tool_calls[
        0
    ].function.arguments = '{"id":"first"}'
    second_response.choices[0].message.tool_calls[
        0
    ].function.arguments = '{"id":"second"}'
    monkeypatch.setattr(adapter, "invoke_tool", fake_invoke)

    result = adapter.run_tool_call_cycle(
        client,
        model="test-model",
        messages=[],
    )

    assert result is second_response
    assert invoked_ids == ["first"]
    assert client.completions.call_count == 2


@pytest.mark.parametrize("content", [None, "", "working"])
def test_cycle_accepts_supported_assistant_content(
    monkeypatch: pytest.MonkeyPatch,
    content: str | None,
) -> None:
    client = _FakeClient(
        [_response([_tool_call()], content=content), _response(None)]
    )
    monkeypatch.setattr(
        adapter,
        "invoke_tool",
        lambda tool_name, arguments: {},
    )

    adapter.run_tool_call_cycle(
        client,
        model="test-model",
        messages=[],
    )

    assistant_message = client.completions.kwargs[1]["messages"][0]
    assert assistant_message["content"] is content


@pytest.mark.parametrize("role", [None, 1, True, "user", ""])
def test_cycle_rejects_invalid_assistant_role_before_tool_execution(
    monkeypatch: pytest.MonkeyPatch,
    role: object,
) -> None:
    client = _FakeClient([_response([_tool_call()], role=role)])
    invocations: list[object] = []
    monkeypatch.setattr(
        adapter,
        "invoke_tool",
        lambda *args: invocations.append(args),
    )

    with pytest.raises((TypeError, ValueError), match="role"):
        adapter.run_tool_call_cycle(
            client,
            model="test-model",
            messages=[],
        )

    assert invocations == []
    assert client.completions.call_count == 1


@pytest.mark.parametrize("content", [1, True, [], {}])
def test_cycle_rejects_invalid_assistant_content_before_tool_execution(
    monkeypatch: pytest.MonkeyPatch,
    content: object,
) -> None:
    client = _FakeClient([_response([_tool_call()], content=content)])
    invocations: list[object] = []
    monkeypatch.setattr(
        adapter,
        "invoke_tool",
        lambda *args: invocations.append(args),
    )

    with pytest.raises(TypeError, match="content"):
        adapter.run_tool_call_cycle(
            client,
            model="test-model",
            messages=[],
        )

    assert invocations == []
    assert client.completions.call_count == 1


@pytest.mark.parametrize("attribute", ["role", "content"])
def test_cycle_rejects_missing_assistant_fields_before_tool_execution(
    monkeypatch: pytest.MonkeyPatch,
    attribute: str,
) -> None:
    response = _response([_tool_call()])
    response.choices[0].message = _without_attribute(
        response.choices[0].message,
        attribute,
    )
    client = _FakeClient([response])
    invocations: list[object] = []
    monkeypatch.setattr(
        adapter,
        "invoke_tool",
        lambda *args: invocations.append(args),
    )

    with pytest.raises(ValueError, match=attribute):
        adapter.run_tool_call_cycle(
            client,
            model="test-model",
            messages=[],
        )

    assert invocations == []


def test_cycle_prevalidates_all_tool_calls_before_any_execution(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    response = _response(
        [
            _tool_call(name="valid_tool"),
            _tool_call(
                call_id="call_002",
                name="invalid_tool",
                arguments="{",
            ),
        ]
    )
    client = _FakeClient([response])
    invocations: list[object] = []
    monkeypatch.setattr(
        adapter,
        "invoke_tool",
        lambda *args: invocations.append(args),
    )

    with pytest.raises(ValueError, match=r"tool_calls\[1\]"):
        adapter.run_tool_call_cycle(
            client,
            model="test-model",
            messages=[],
        )

    assert invocations == []
    assert client.completions.call_count == 1


def test_cycle_supports_attribute_objects_without_using_model_dump(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    response = _response([_tool_call()])

    def fail_model_dump() -> object:
        pytest.fail("model_dump must not be used")

    response.choices[0].message.model_dump = fail_model_dump
    response.choices[0].message.tool_calls[0].model_dump = fail_model_dump
    client = _FakeClient([response, _response(None)])
    monkeypatch.setattr(
        adapter,
        "invoke_tool",
        lambda tool_name, arguments: {},
    )

    adapter.run_tool_call_cycle(
        client,
        model="test-model",
        messages=[],
    )

    assert client.completions.call_count == 2


def test_cycle_rejects_dictionary_provider_response() -> None:
    response = {
        "choices": [
            {
                "message": {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [],
                }
            }
        ]
    }
    client = _FakeClient([response])

    with pytest.raises(TypeError):
        adapter.run_tool_call_cycle(
            client,
            model="test-model",
            messages=[],
        )


def test_cycle_preserves_request_options_for_both_requests(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = _FakeClient([_response([_tool_call()]), _response(None)])
    tool_choice = {
        "type": "function",
        "function": {"name": "analyze_experiment"},
    }
    metadata = {"request": ["unchanged"]}
    monkeypatch.setattr(
        adapter,
        "invoke_tool",
        lambda tool_name, arguments: {},
    )

    adapter.run_tool_call_cycle(
        client,
        model="test-model",
        messages=[],
        tool_choice=tool_choice,
        metadata=metadata,
        temperature=0,
    )

    for request in client.completions.kwargs:
        assert request["model"] == "test-model"
        assert request["tool_choice"] is tool_choice
        assert request["metadata"] is metadata
        assert request["temperature"] == 0
        assert "tools" in request


def test_cycle_does_not_modify_messages_or_request_options(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    messages = [
        {
            "role": "user",
            "content": {"parts": ["Analyze this."]},
        }
    ]
    request_options = {
        "tool_choice": {
            "type": "function",
            "function": {"name": "analyze_experiment"},
        },
        "metadata": {"tags": ["test"]},
    }
    messages_before = copy.deepcopy(messages)
    request_options_before = copy.deepcopy(request_options)
    client = _FakeClient([_response([_tool_call()]), _response(None)])
    monkeypatch.setattr(
        adapter,
        "invoke_tool",
        lambda tool_name, arguments: {},
    )

    adapter.run_tool_call_cycle(
        client,
        model="test-model",
        messages=messages,
        **request_options,
    )

    assert messages == messages_before
    assert request_options == request_options_before
    assert client.completions.kwargs[0]["messages"] is messages
    assert client.completions.kwargs[1]["messages"] is not messages


def test_cycle_rejects_explicit_tools_before_requesting() -> None:
    client = _FakeClient([_response(None)])

    with pytest.raises(
        TypeError,
        match="tools are provided by the tool registry",
    ):
        adapter.run_tool_call_cycle(
            client,
            model="test-model",
            messages=[],
            tools=[],
        )

    assert client.completions.calls == []


@pytest.mark.parametrize("tool_calls", [None, []])
def test_cycle_lists_tools_once_without_tool_calls(
    monkeypatch: pytest.MonkeyPatch,
    tool_calls: object,
) -> None:
    original_list_tools = adapter.list_tools
    list_tools_calls = 0

    def counted_list_tools() -> list[dict]:
        nonlocal list_tools_calls
        list_tools_calls += 1
        return original_list_tools()

    monkeypatch.setattr(adapter, "list_tools", counted_list_tools)
    client = _FakeClient([_response(tool_calls)])

    adapter.run_tool_call_cycle(
        client,
        model="test-model",
        messages=[],
    )

    assert list_tools_calls == 1
    assert client.completions.call_count == 1


def test_cycle_lists_tools_twice_with_tool_calls(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original_list_tools = adapter.list_tools
    list_tools_calls = 0

    def counted_list_tools() -> list[dict]:
        nonlocal list_tools_calls
        list_tools_calls += 1
        return original_list_tools()

    monkeypatch.setattr(adapter, "list_tools", counted_list_tools)
    monkeypatch.setattr(
        adapter,
        "invoke_tool",
        lambda tool_name, arguments: {},
    )
    client = _FakeClient([_response([_tool_call()]), _response(None)])

    adapter.run_tool_call_cycle(
        client,
        model="test-model",
        messages=[],
    )

    assert list_tools_calls == 2
    assert client.completions.call_count == 2


def test_cycle_does_not_list_tools_or_request_third_response(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original_list_tools = adapter.list_tools
    list_tools_calls = 0

    def counted_list_tools() -> list[dict]:
        nonlocal list_tools_calls
        list_tools_calls += 1
        return original_list_tools()

    monkeypatch.setattr(adapter, "list_tools", counted_list_tools)
    monkeypatch.setattr(
        adapter,
        "invoke_tool",
        lambda tool_name, arguments: {},
    )
    second_response = _response([_tool_call(call_id="call_second")])
    client = _FakeClient(
        [_response([_tool_call(call_id="call_first")]), second_response]
    )

    result = adapter.run_tool_call_cycle(
        client,
        model="test-model",
        messages=[],
    )

    assert result is second_response
    assert list_tools_calls == 2
    assert client.completions.call_count == 2


def test_cycle_propagates_first_request_exception(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    error = RuntimeError("first request failed")
    client = _FakeClient([error])
    invocations: list[object] = []
    monkeypatch.setattr(
        adapter,
        "invoke_tool",
        lambda *args: invocations.append(args),
    )

    with pytest.raises(RuntimeError) as error_info:
        adapter.run_tool_call_cycle(
            client,
            model="test-model",
            messages=[],
        )

    assert error_info.value is error
    assert client.completions.call_count == 1
    assert invocations == []


def test_cycle_propagates_second_request_exception(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    error = RuntimeError("second request failed")
    client = _FakeClient([_response([_tool_call()]), error])
    invocations: list[tuple[str, dict]] = []

    def fake_invoke(tool_name: str, arguments: dict) -> dict:
        invocations.append((tool_name, arguments))
        return {"ok": True}

    monkeypatch.setattr(adapter, "invoke_tool", fake_invoke)

    with pytest.raises(RuntimeError) as error_info:
        adapter.run_tool_call_cycle(
            client,
            model="test-model",
            messages=[],
        )

    assert error_info.value is error
    assert client.completions.call_count == 2
    assert invocations == [
        ("analyze_experiment", {"experiment_dir": "demo"})
    ]


def test_cycle_preserves_malformed_json_cause_without_side_effects(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = _FakeClient(
        [_response([_tool_call(arguments='{"invalid":}')])]
    )
    invocations: list[object] = []
    monkeypatch.setattr(
        adapter,
        "invoke_tool",
        lambda *args: invocations.append(args),
    )

    with pytest.raises(ValueError) as error_info:
        adapter.run_tool_call_cycle(
            client,
            model="test-model",
            messages=[],
        )

    assert isinstance(error_info.value.__cause__, json.JSONDecodeError)
    assert invocations == []
    assert client.completions.call_count == 1


@pytest.mark.parametrize(
    "arguments",
    ["[]", '"text"', "1", "true", "null"],
)
def test_cycle_rejects_non_object_json_without_side_effects(
    monkeypatch: pytest.MonkeyPatch,
    arguments: str,
) -> None:
    client = _FakeClient(
        [_response([_tool_call(arguments=arguments)])]
    )
    invocations: list[object] = []
    monkeypatch.setattr(
        adapter,
        "invoke_tool",
        lambda *args: invocations.append(args),
    )

    with pytest.raises(TypeError, match="must decode to an object"):
        adapter.run_tool_call_cycle(
            client,
            model="test-model",
            messages=[],
        )

    assert invocations == []
    assert client.completions.call_count == 1


def test_cycle_preserves_unknown_tool_error() -> None:
    client = _FakeClient(
        [
            _response(
                [
                    _tool_call(
                        name="unknown_tool",
                        arguments="{}",
                    )
                ]
            )
        ]
    )

    with pytest.raises(KeyError) as error_info:
        adapter.run_tool_call_cycle(
            client,
            model="test-model",
            messages=[],
        )

    assert error_info.value.args == ("unknown tool: unknown_tool",)
    assert client.completions.call_count == 1


def test_cycle_allows_prior_tool_side_effect_before_unknown_tool(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = _FakeClient(
        [
            _response(
                [
                    _tool_call(
                        call_id="call_first",
                        name="valid_tool",
                        arguments='{"position":1}',
                    ),
                    _tool_call(
                        call_id="call_second",
                        name="unknown_tool",
                        arguments='{"position":2}',
                    ),
                ]
            )
        ]
    )
    invocations: list[str] = []

    def fake_invoke(tool_name: str, arguments: dict) -> dict:
        invocations.append(tool_name)
        if tool_name == "unknown_tool":
            raise KeyError("unknown tool: unknown_tool")
        return {"ok": True}

    monkeypatch.setattr(adapter, "invoke_tool", fake_invoke)

    with pytest.raises(KeyError) as error_info:
        adapter.run_tool_call_cycle(
            client,
            model="test-model",
            messages=[],
        )

    assert error_info.value.args == ("unknown tool: unknown_tool",)
    assert invocations == ["valid_tool", "unknown_tool"]
    assert client.completions.call_count == 1


def test_cycle_propagates_tool_exception_without_wrapping(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    error = RuntimeError("tool failed")
    client = _FakeClient([_response([_tool_call()])])

    def fail_tool(tool_name: str, arguments: dict) -> dict:
        raise error

    monkeypatch.setattr(adapter, "invoke_tool", fail_tool)

    with pytest.raises(RuntimeError) as error_info:
        adapter.run_tool_call_cycle(
            client,
            model="test-model",
            messages=[],
        )

    assert error_info.value is error
    assert client.completions.call_count == 1


def test_cycle_propagates_tool_result_serialization_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = _FakeClient([_response([_tool_call()])])
    monkeypatch.setattr(
        adapter,
        "invoke_tool",
        lambda tool_name, arguments: object(),
    )

    with pytest.raises(TypeError):
        adapter.run_tool_call_cycle(
            client,
            model="test-model",
            messages=[],
        )

    assert client.completions.call_count == 1


def test_cycle_supports_non_namespace_attribute_objects(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    tool_call = _AttributeObject(
        id="call_custom",
        type="function",
        function=_AttributeObject(
            name="custom_tool",
            arguments='{"value":7}',
        ),
    )
    first_response = _AttributeObject(
        choices=[
            _AttributeObject(
                message=_AttributeObject(
                    role="assistant",
                    content=None,
                    tool_calls=[tool_call],
                )
            )
        ]
    )
    second_response = _AttributeObject(
        choices=[
            _AttributeObject(
                message=_AttributeObject(
                    role="assistant",
                    content="complete",
                    tool_calls=None,
                )
            )
        ]
    )
    client = _FakeClient([first_response, second_response])
    invocations: list[tuple[str, dict]] = []

    def fake_invoke(tool_name: str, arguments: dict) -> dict:
        invocations.append((tool_name, arguments))
        return {"ok": True}

    monkeypatch.setattr(adapter, "invoke_tool", fake_invoke)

    result = adapter.run_tool_call_cycle(
        client,
        model="test-model",
        messages=[],
    )

    assert result is second_response
    assert client.completions.call_count == 2
    assert invocations == [("custom_tool", {"value": 7})]


def test_cycle_has_no_runtime_network_environment_or_sdk_dependency(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = _FakeClient([_response([_tool_call()]), _response(None)])
    monkeypatch.setattr(
        adapter,
        "invoke_tool",
        lambda tool_name, arguments: {"ok": True},
    )
    original_import = builtins.__import__

    def fail_network(*args: object, **kwargs: object) -> object:
        raise AssertionError("cycle must not access the real network")

    def fail_environment(*args: object, **kwargs: object) -> object:
        raise AssertionError(
            "cycle must not read environment variables"
        )

    def guarded_import(
        name: str,
        globals: dict | None = None,
        locals: dict | None = None,
        fromlist: tuple | list = (),
        level: int = 0,
    ) -> object:
        if name.split(".", 1)[0] == "openai":
            raise AssertionError(
                "cycle must not dynamically import the OpenAI SDK"
            )
        return original_import(
            name,
            globals,
            locals,
            fromlist,
            level,
        )

    with monkeypatch.context() as context:
        context.setattr(socket, "create_connection", fail_network)
        context.setattr(socket.socket, "connect", fail_network)
        context.setattr(os, "getenv", fail_environment)
        context.setattr(type(os.environ), "get", fail_environment)
        context.setattr(
            type(os.environ),
            "__getitem__",
            fail_environment,
        )
        context.setattr(builtins, "__import__", guarded_import)

        result = adapter.run_tool_call_cycle(
            client,
            model="test-model",
            messages=[],
        )

    assert result is client.completions.outcomes[1]
    assert client.completions.call_count == 2

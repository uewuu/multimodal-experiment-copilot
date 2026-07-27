import copy
import importlib
import inspect
import json
import os
import socket
import sys
from collections import UserList
from pathlib import Path
from types import SimpleNamespace

import pytest


CORE_PUBLIC_NAMES = [
    "create_tool_call_response",
    "execute_tool_calls",
]

EXPECTED_PUBLIC_NAMES = [
    *CORE_PUBLIC_NAMES,
    "run_tool_call_cycle",
]


_MISSING = object()


def _load_package():
    try:
        return importlib.import_module("llm_adapters")
    except (ModuleNotFoundError, ImportError) as error:
        pytest.fail(
            f"llm_adapters package must implement Issue #16: {error}",
            pytrace=False,
        )


def _load_adapter_module():
    try:
        return importlib.import_module(
            "llm_adapters.openai_tool_adapter"
        )
    except (ModuleNotFoundError, ImportError) as error:
        pytest.fail(
            "llm_adapters.openai_tool_adapter must implement "
            f"Issue #16: {error}",
            pytrace=False,
        )


def _load_public_functions():
    package = _load_package()
    functions = []
    for name in CORE_PUBLIC_NAMES:
        if not hasattr(package, name):
            pytest.fail(
                f"llm_adapters must publicly export {name}",
                pytrace=False,
            )
        value = getattr(package, name)
        if not callable(value):
            pytest.fail(
                f"llm_adapters.{name} must be callable",
                pytrace=False,
            )
        functions.append(value)
    return tuple(functions)


class _FakeCompletions:
    def __init__(
        self,
        response: object,
        error: BaseException | None = None,
    ) -> None:
        self.response = response
        self.error = error
        self.calls: list[tuple[tuple[object, ...], dict[str, object]]] = []

    def create(self, *args: object, **kwargs: object) -> object:
        self.calls.append((args, kwargs))
        if self.error is not None:
            raise self.error
        return self.response


class _FakeClient:
    def __init__(
        self,
        response: object,
        error: BaseException | None = None,
    ) -> None:
        self.completions = _FakeCompletions(response, error)
        self.chat = SimpleNamespace(completions=self.completions)


def _function_call(
    *,
    call_id: object = "call_001",
    call_type: object = "function",
    name: object = "analyze_experiment",
    arguments: object = '{"experiment_dir":"demo"}',
):
    function = SimpleNamespace(name=name, arguments=arguments)
    return SimpleNamespace(
        id=call_id,
        type=call_type,
        function=function,
    )


def _response(tool_calls: object):
    message = SimpleNamespace(tool_calls=tool_calls)
    choice = SimpleNamespace(message=message)
    return SimpleNamespace(choices=[choice])


def _without_attribute(value: object, attribute: str):
    namespace = SimpleNamespace(**vars(value))
    delattr(namespace, attribute)
    return namespace


def _patch_invoke(
    monkeypatch: pytest.MonkeyPatch,
    implementation,
):
    module = _load_adapter_module()
    monkeypatch.setattr(module, "invoke_tool", implementation)
    return module


def test_llm_adapters_package_is_importable() -> None:
    package = _load_package()

    assert package.__name__ == "llm_adapters"


def test_openai_tool_adapter_module_is_importable() -> None:
    module = _load_adapter_module()

    assert module.__name__ == "llm_adapters.openai_tool_adapter"


def test_package_publicly_exports_adapter_functions() -> None:
    package = _load_package()

    for name in EXPECTED_PUBLIC_NAMES:
        assert hasattr(package, name)
        assert callable(getattr(package, name))


def test_package_all_has_exact_public_names() -> None:
    package = _load_package()

    assert package.__all__ == EXPECTED_PUBLIC_NAMES


def test_create_tool_call_response_has_exact_signature() -> None:
    create_tool_call_response, _ = _load_public_functions()
    signature = inspect.signature(create_tool_call_response)
    parameters = list(signature.parameters.values())

    assert [parameter.name for parameter in parameters] == [
        "client",
        "model",
        "messages",
        "request_options",
    ]
    assert parameters[0].kind is inspect.Parameter.POSITIONAL_OR_KEYWORD
    assert parameters[0].default is inspect.Parameter.empty
    assert parameters[0].annotation is object
    assert parameters[1].kind is inspect.Parameter.KEYWORD_ONLY
    assert parameters[1].default is inspect.Parameter.empty
    assert parameters[1].annotation is str
    assert parameters[2].kind is inspect.Parameter.KEYWORD_ONLY
    assert parameters[2].default is inspect.Parameter.empty
    assert parameters[2].annotation == list[dict]
    assert parameters[3].kind is inspect.Parameter.VAR_KEYWORD
    assert parameters[3].annotation is object
    assert signature.return_annotation is object


def test_execute_tool_calls_has_exact_signature() -> None:
    _, execute_tool_calls = _load_public_functions()
    signature = inspect.signature(execute_tool_calls)
    parameters = list(signature.parameters.values())

    assert [parameter.name for parameter in parameters] == ["response"]
    assert parameters[0].kind is inspect.Parameter.POSITIONAL_OR_KEYWORD
    assert parameters[0].default is inspect.Parameter.empty
    assert parameters[0].annotation is object
    assert signature.return_annotation == list[dict]


@pytest.mark.parametrize("invalid_model", [None, 1, True, [], {}])
def test_create_rejects_non_string_model(invalid_model: object) -> None:
    create_tool_call_response, _ = _load_public_functions()

    with pytest.raises(TypeError) as error_info:
        create_tool_call_response(
            _FakeClient(object()),
            model=invalid_model,
            messages=[],
        )

    assert "model" in str(error_info.value).lower()


@pytest.mark.parametrize("invalid_model", ["", " ", "\t", "\n"])
def test_create_rejects_blank_model(invalid_model: str) -> None:
    create_tool_call_response, _ = _load_public_functions()

    with pytest.raises(ValueError) as error_info:
        create_tool_call_response(
            _FakeClient(object()),
            model=invalid_model,
            messages=[],
        )

    assert "model" in str(error_info.value).lower()


@pytest.mark.parametrize(
    "model",
    ["gpt-test", " model-with-spaces-inside "],
)
def test_create_preserves_valid_model(model: str) -> None:
    create_tool_call_response, _ = _load_public_functions()
    client = _FakeClient(object())

    create_tool_call_response(client, model=model, messages=[])

    assert client.completions.calls[0][1]["model"] == model


def test_create_rejects_non_actual_list_messages() -> None:
    create_tool_call_response, _ = _load_public_functions()

    class CustomList(list):
        pass

    invalid_values = (
        None,
        (),
        {},
        "messages",
        UserList(),
        CustomList(),
    )
    for invalid_value in invalid_values:
        with pytest.raises(TypeError) as error_info:
            create_tool_call_response(
                _FakeClient(object()),
                model="gpt-test",
                messages=invalid_value,
            )
        assert "messages" in str(error_info.value).lower()


@pytest.mark.parametrize("messages", [[], [{"unexpected": object()}]])
def test_create_accepts_actual_list_messages(messages: list) -> None:
    create_tool_call_response, _ = _load_public_functions()
    client = _FakeClient(object())

    create_tool_call_response(
        client,
        model="gpt-test",
        messages=messages,
    )

    assert client.completions.calls[0][1]["messages"] is messages


def test_create_uses_injected_client_once() -> None:
    create_tool_call_response, _ = _load_public_functions()
    client = _FakeClient(object())

    create_tool_call_response(client, model="gpt-test", messages=[])

    assert len(client.completions.calls) == 1


def test_create_forwards_model_messages_and_returns_response() -> None:
    create_tool_call_response, _ = _load_public_functions()
    response = object()
    messages = [{"role": "user", "content": "analyze"}]
    client = _FakeClient(response)

    result = create_tool_call_response(
        client,
        model="gpt-test",
        messages=messages,
    )

    args, kwargs = client.completions.calls[0]
    assert args == ()
    assert kwargs["model"] == "gpt-test"
    assert kwargs["messages"] is messages
    assert result is response


def test_create_passes_registry_tools_by_identity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load_adapter_module()
    sentinel_tools = [{"sentinel": []}]
    call_count = 0

    def fake_list_tools() -> list[dict]:
        nonlocal call_count
        call_count += 1
        return sentinel_tools

    monkeypatch.setattr(module, "list_tools", fake_list_tools)
    client = _FakeClient(object())

    module.create_tool_call_response(
        client,
        model="gpt-test",
        messages=[],
    )

    assert call_count == 1
    assert client.completions.calls[0][1]["tools"] is sentinel_tools


def test_create_gets_fresh_tools_for_each_call(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load_adapter_module()
    produced: list[list[dict]] = []

    def fake_list_tools() -> list[dict]:
        tools = [{"call": len(produced)}]
        produced.append(tools)
        return tools

    monkeypatch.setattr(module, "list_tools", fake_list_tools)
    first_client = _FakeClient(object())
    second_client = _FakeClient(object())

    module.create_tool_call_response(
        first_client,
        model="gpt-test",
        messages=[],
    )
    module.create_tool_call_response(
        second_client,
        model="gpt-test",
        messages=[],
    )

    assert len(produced) == 2
    assert produced[0] is not produced[1]
    assert first_client.completions.calls[0][1]["tools"] is produced[0]
    assert second_client.completions.calls[0][1]["tools"] is produced[1]


def test_create_does_not_modify_registry_tools(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load_adapter_module()
    sentinel_tools = [{"function": {"properties": []}}]
    original = copy.deepcopy(sentinel_tools)
    monkeypatch.setattr(module, "list_tools", lambda: sentinel_tools)

    module.create_tool_call_response(
        _FakeClient(object()),
        model="gpt-test",
        messages=[],
    )

    assert sentinel_tools == original


def test_create_forwards_request_options_and_identity() -> None:
    create_tool_call_response, _ = _load_public_functions()
    metadata = {"nested": []}
    client = _FakeClient(object())

    create_tool_call_response(
        client,
        model="gpt-test",
        messages=[],
        temperature=0.25,
        metadata=metadata,
    )

    kwargs = client.completions.calls[0][1]
    assert kwargs["temperature"] == 0.25
    assert kwargs["metadata"] is metadata


def test_create_does_not_inject_tool_choice() -> None:
    create_tool_call_response, _ = _load_public_functions()
    client = _FakeClient(object())

    create_tool_call_response(client, model="gpt-test", messages=[])

    assert "tool_choice" not in client.completions.calls[0][1]


def test_create_forwards_explicit_tool_choice() -> None:
    create_tool_call_response, _ = _load_public_functions()
    tool_choice = {"type": "function", "function": {"name": "tool"}}
    client = _FakeClient(object())

    create_tool_call_response(
        client,
        model="gpt-test",
        messages=[],
        tool_choice=tool_choice,
    )

    assert client.completions.calls[0][1]["tool_choice"] is tool_choice


def test_create_does_not_modify_messages_or_request_values() -> None:
    create_tool_call_response, _ = _load_public_functions()
    messages = [{"role": "user", "content": ["original"]}]
    metadata = {"nested": ["original"]}
    messages_before = copy.deepcopy(messages)
    metadata_before = copy.deepcopy(metadata)

    create_tool_call_response(
        _FakeClient(object()),
        model="gpt-test",
        messages=messages,
        metadata=metadata,
    )

    assert messages == messages_before
    assert metadata == metadata_before


def test_create_rejects_tools_override() -> None:
    create_tool_call_response, _ = _load_public_functions()

    with pytest.raises(TypeError) as error_info:
        create_tool_call_response(
            _FakeClient(object()),
            model="gpt-test",
            messages=[],
            tools=[],
        )

    assert "tools" in str(error_info.value).lower()


def test_create_propagates_client_exception_identity() -> None:
    create_tool_call_response, _ = _load_public_functions()
    expected_error = RuntimeError("client failed")

    with pytest.raises(RuntimeError) as error_info:
        create_tool_call_response(
            _FakeClient(object(), expected_error),
            model="gpt-test",
            messages=[],
        )

    assert error_info.value is expected_error


def test_create_rejects_missing_client_path() -> None:
    create_tool_call_response, _ = _load_public_functions()
    clients = (
        SimpleNamespace(),
        SimpleNamespace(chat=SimpleNamespace()),
        SimpleNamespace(
            chat=SimpleNamespace(completions=SimpleNamespace())
        ),
    )

    for client in clients:
        with pytest.raises(AttributeError):
            create_tool_call_response(
                client,
                model="gpt-test",
                messages=[],
            )


def test_create_rejects_noncallable_create() -> None:
    create_tool_call_response, _ = _load_public_functions()
    client = SimpleNamespace(
        chat=SimpleNamespace(
            completions=SimpleNamespace(create=object())
        )
    )

    with pytest.raises(TypeError):
        create_tool_call_response(
            client,
            model="gpt-test",
            messages=[],
        )


@pytest.mark.parametrize(
    "response",
    [
        {"choices": []},
        SimpleNamespace(output=[]),
    ],
)
def test_execute_rejects_unsupported_response_shapes(
    response: object,
) -> None:
    _, execute_tool_calls = _load_public_functions()

    with pytest.raises(TypeError) as error_info:
        execute_tool_calls(response)

    assert "choices" in str(error_info.value).lower()


def test_execute_rejects_non_actual_list_choices() -> None:
    _, execute_tool_calls = _load_public_functions()

    class CustomList(list):
        pass

    for invalid_choices in ((), UserList(), CustomList()):
        with pytest.raises(TypeError):
            execute_tool_calls(
                SimpleNamespace(choices=invalid_choices)
            )


def test_execute_rejects_empty_choices() -> None:
    _, execute_tool_calls = _load_public_functions()

    with pytest.raises(ValueError) as error_info:
        execute_tool_calls(SimpleNamespace(choices=[]))

    assert "choices" in str(error_info.value).lower()


def test_execute_rejects_missing_message() -> None:
    _, execute_tool_calls = _load_public_functions()

    with pytest.raises(TypeError) as error_info:
        execute_tool_calls(
            SimpleNamespace(choices=[SimpleNamespace()])
        )

    assert "message" in str(error_info.value).lower()


def test_execute_rejects_missing_tool_calls() -> None:
    _, execute_tool_calls = _load_public_functions()
    response = SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace())]
    )

    with pytest.raises(TypeError) as error_info:
        execute_tool_calls(response)

    assert "tool_calls" in str(error_info.value).lower()


@pytest.mark.parametrize("tool_calls", [None, []])
def test_execute_returns_empty_list_without_tool_calls(
    tool_calls: object,
) -> None:
    _, execute_tool_calls = _load_public_functions()

    assert execute_tool_calls(_response(tool_calls)) == []


def test_execute_rejects_non_actual_list_tool_calls() -> None:
    _, execute_tool_calls = _load_public_functions()

    class CustomList(list):
        pass

    for invalid_tool_calls in ((), UserList(), CustomList()):
        with pytest.raises(TypeError):
            execute_tool_calls(_response(invalid_tool_calls))


def test_execute_handles_one_tool_call(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[str, dict]] = []
    result = {"b": 2, "a": "中文"}

    def fake_invoke(name: str, arguments: dict) -> dict:
        calls.append((name, arguments))
        return result

    module = _patch_invoke(monkeypatch, fake_invoke)

    messages = module.execute_tool_calls(
        _response([_function_call()])
    )

    assert calls == [
        ("analyze_experiment", {"experiment_dir": "demo"})
    ]
    assert messages == [
        {
            "role": "tool",
            "tool_call_id": "call_001",
            "content": '{"a":"中文","b":2}',
        }
    ]
    assert set(messages[0]) == {"role", "tool_call_id", "content"}
    assert type(messages[0]["content"]) is str


def test_execute_preserves_multiple_call_order(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[str, dict]] = []

    def fake_invoke(name: str, arguments: dict) -> dict:
        calls.append((name, arguments))
        return {"position": len(calls)}

    module = _patch_invoke(monkeypatch, fake_invoke)
    tool_calls = [
        _function_call(
            call_id="call_003",
            name="analyze_experiment",
            arguments='{"experiment_dir":"first"}',
        ),
        _function_call(
            call_id="call_001",
            name="compare_experiments",
            arguments='{"experiment_root":"second"}',
        ),
        _function_call(
            call_id="call_002",
            name="analyze_experiment",
            arguments='{"experiment_dir":"third"}',
        ),
    ]

    messages = module.execute_tool_calls(_response(tool_calls))

    assert [name for name, _ in calls] == [
        "analyze_experiment",
        "compare_experiments",
        "analyze_experiment",
    ]
    assert [message["tool_call_id"] for message in messages] == [
        "call_003",
        "call_001",
        "call_002",
    ]
    assert len({id(message) for message in messages}) == 3


def test_execute_validates_tool_call_id() -> None:
    _, execute_tool_calls = _load_public_functions()
    missing_id = _without_attribute(_function_call(), "id")
    cases = [
        (missing_id, ValueError),
        (_function_call(call_id=None), TypeError),
        (_function_call(call_id=1), TypeError),
        (_function_call(call_id=""), ValueError),
        (_function_call(call_id=" "), ValueError),
    ]

    for tool_call, error_type in cases:
        with pytest.raises(error_type) as error_info:
            execute_tool_calls(_response([tool_call]))
        message = str(error_info.value).lower()
        assert "id" in message
        assert "0" in message


def test_execute_preserves_original_tool_call_id(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _patch_invoke(
        monkeypatch,
        lambda name, arguments: {},
    )
    original_id = " call_with_spaces "

    messages = module.execute_tool_calls(
        _response([_function_call(call_id=original_id)])
    )

    assert messages[0]["tool_call_id"] == original_id


def test_execute_validates_tool_call_type() -> None:
    _, execute_tool_calls = _load_public_functions()
    missing_type = _without_attribute(_function_call(), "type")
    cases = [
        (missing_type, ValueError),
        (_function_call(call_type=None), TypeError),
        (_function_call(call_type=1), TypeError),
        (_function_call(call_type="FUNCTION"), ValueError),
        (_function_call(call_type="tool"), ValueError),
        (_function_call(call_type=""), ValueError),
        (_function_call(call_type=" "), ValueError),
    ]

    for tool_call, error_type in cases:
        with pytest.raises(error_type) as error_info:
            execute_tool_calls(_response([tool_call]))
        assert "type" in str(error_info.value).lower()


def test_execute_validates_function_object() -> None:
    _, execute_tool_calls = _load_public_functions()
    missing_function = _without_attribute(
        _function_call(),
        "function",
    )
    cases = [
        (missing_function, ValueError),
        (
            SimpleNamespace(
                id="call_001",
                type="function",
                function=None,
            ),
            TypeError,
        ),
        (
            SimpleNamespace(
                id="call_001",
                type="function",
                function={},
            ),
            TypeError,
        ),
    ]

    for tool_call, error_type in cases:
        with pytest.raises(error_type) as error_info:
            execute_tool_calls(_response([tool_call]))
        assert "function" in str(error_info.value).lower()


@pytest.mark.parametrize(
    "invalid_function",
    [1, "value", object()],
    ids=["integer", "string", "object"],
)
def test_execute_rejects_invalid_function_object_type(
    invalid_function: object,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    invoke_count = 0

    def fake_invoke(name: str, arguments: dict) -> dict:
        nonlocal invoke_count
        invoke_count += 1
        return {}

    module = _patch_invoke(monkeypatch, fake_invoke)
    tool_call = SimpleNamespace(
        id="call_001",
        type="function",
        function=invalid_function,
    )

    with pytest.raises(TypeError) as error_info:
        module.execute_tool_calls(_response([tool_call]))

    message = str(error_info.value).lower()
    assert "function" in message
    assert "0" in message
    assert ".name" not in message
    assert invoke_count == 0


def test_execute_validates_function_name() -> None:
    _, execute_tool_calls = _load_public_functions()
    missing_name_function = _without_attribute(
        _function_call().function,
        "name",
    )
    missing_name = SimpleNamespace(
        id="call_001",
        type="function",
        function=missing_name_function,
    )
    cases = [
        (missing_name, ValueError),
        (_function_call(name=None), TypeError),
        (_function_call(name=1), TypeError),
        (_function_call(name=""), ValueError),
        (_function_call(name=" "), ValueError),
    ]

    for tool_call, error_type in cases:
        with pytest.raises(error_type) as error_info:
            execute_tool_calls(_response([tool_call]))
        assert "name" in str(error_info.value).lower()


@pytest.mark.parametrize(
    "name",
    [" analyze_experiment", "Analyze_Experiment"],
)
def test_execute_does_not_normalize_function_name(
    name: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    received: list[str] = []

    def fake_invoke(tool_name: str, arguments: dict) -> dict:
        received.append(tool_name)
        return {}

    module = _patch_invoke(monkeypatch, fake_invoke)

    module.execute_tool_calls(
        _response([_function_call(name=name)])
    )

    assert received == [name]


def test_execute_validates_arguments_type() -> None:
    _, execute_tool_calls = _load_public_functions()
    missing_arguments_function = _without_attribute(
        _function_call().function,
        "arguments",
    )
    missing_arguments = SimpleNamespace(
        id="call_001",
        type="function",
        function=missing_arguments_function,
    )
    cases = [
        (missing_arguments, ValueError),
        (_function_call(arguments=None), TypeError),
        (_function_call(arguments={}), TypeError),
        (_function_call(arguments=b"{}"), TypeError),
        (_function_call(arguments=1), TypeError),
    ]

    for tool_call, error_type in cases:
        with pytest.raises(error_type) as error_info:
            execute_tool_calls(_response([tool_call]))
        assert "arguments" in str(error_info.value).lower()


@pytest.mark.parametrize(
    "arguments, expected",
    [
        ("{}", {}),
        ('{"experiment_dir":"demo"}', {"experiment_dir": "demo"}),
        (
            '{"nested":{"values":[1,true,null]},"text":"中文"}',
            {
                "nested": {"values": [1, True, None]},
                "text": "中文",
            },
        ),
        ('{"z":1,"a":2}', {"z": 1, "a": 2}),
    ],
)
def test_execute_accepts_json_objects(
    arguments: str,
    expected: dict,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    received: list[dict] = []

    def fake_invoke(name: str, decoded: dict) -> dict:
        received.append(decoded)
        return {}

    module = _patch_invoke(monkeypatch, fake_invoke)

    module.execute_tool_calls(
        _response([_function_call(arguments=arguments)])
    )

    assert received == [expected]
    assert type(received[0]) is dict


@pytest.mark.parametrize("arguments", ["", " ", "\t", "\n"])
def test_execute_rejects_blank_arguments(arguments: str) -> None:
    _, execute_tool_calls = _load_public_functions()

    with pytest.raises(ValueError) as error_info:
        execute_tool_calls(
            _response([_function_call(arguments=arguments)])
        )

    message = str(error_info.value)
    assert "arguments" in message.lower()
    assert arguments not in message or not arguments.strip()


@pytest.mark.parametrize("arguments", ["{", '{"x":}', "not-json"])
def test_execute_wraps_malformed_json(arguments: str) -> None:
    _, execute_tool_calls = _load_public_functions()

    with pytest.raises(ValueError) as error_info:
        execute_tool_calls(
            _response([_function_call(arguments=arguments)])
        )

    assert isinstance(error_info.value.__cause__, json.JSONDecodeError)
    message = str(error_info.value)
    assert "call_001" in message
    assert "analyze_experiment" in message
    assert arguments not in message


@pytest.mark.parametrize(
    "arguments",
    ["[]", '["a"]', '"text"', "1", "1.5", "true", "false", "null"],
)
def test_execute_rejects_non_object_json(arguments: str) -> None:
    _, execute_tool_calls = _load_public_functions()

    with pytest.raises(TypeError) as error_info:
        execute_tool_calls(
            _response([_function_call(arguments=arguments)])
        )

    assert "arguments" in str(error_info.value).lower()


def test_execute_prevalidates_all_calls_before_invocation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    invoke_count = 0

    def fake_invoke(name: str, arguments: dict) -> dict:
        nonlocal invoke_count
        invoke_count += 1
        return {}

    module = _patch_invoke(monkeypatch, fake_invoke)
    malformed_calls = [
        [
            _function_call(),
            _function_call(
                call_id="call_002",
                arguments="{",
            ),
        ],
        [
            _function_call(),
            _function_call(
                call_id="call_002",
                call_type="tool",
            ),
        ],
    ]

    for tool_calls in malformed_calls:
        with pytest.raises(ValueError):
            module.execute_tool_calls(_response(tool_calls))

    assert invoke_count == 0


def test_execute_propagates_unknown_tool_after_prior_success(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    expected_error = KeyError("unknown tool")
    calls: list[str] = []

    def fake_invoke(name: str, arguments: dict) -> dict:
        calls.append(name)
        if name == "unknown":
            raise expected_error
        return {"name": name}

    module = _patch_invoke(monkeypatch, fake_invoke)
    tool_calls = [
        _function_call(call_id="one", name="first"),
        _function_call(call_id="two", name="unknown"),
        _function_call(call_id="three", name="third"),
    ]

    with pytest.raises(KeyError) as error_info:
        module.execute_tool_calls(_response(tool_calls))

    assert error_info.value is expected_error
    assert calls == ["first", "unknown"]


def test_execute_propagates_tool_exception_and_stops(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    expected_error = RuntimeError("tool failed")
    calls: list[str] = []

    def fake_invoke(name: str, arguments: dict) -> dict:
        calls.append(name)
        if name == "second":
            raise expected_error
        return {}

    module = _patch_invoke(monkeypatch, fake_invoke)
    tool_calls = [
        _function_call(call_id="one", name="first"),
        _function_call(call_id="two", name="second"),
        _function_call(call_id="three", name="third"),
    ]

    with pytest.raises(RuntimeError) as error_info:
        module.execute_tool_calls(_response(tool_calls))

    assert error_info.value is expected_error
    assert calls == ["first", "second"]


def test_execute_serializes_strict_deterministic_json(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    results = [
        {"b": 2, "a": "中文"},
        {"nested": {"items": [None, True, 1.5]}},
        [],
        {},
    ]
    index = 0

    def fake_invoke(name: str, arguments: dict) -> object:
        nonlocal index
        result = results[index]
        index += 1
        return result

    module = _patch_invoke(monkeypatch, fake_invoke)
    tool_calls = [
        _function_call(call_id=f"call_{position}")
        for position in range(len(results))
    ]

    messages = module.execute_tool_calls(_response(tool_calls))

    assert [message["content"] for message in messages] == [
        '{"a":"中文","b":2}',
        '{"nested":{"items":[null,true,1.5]}}',
        "[]",
        "{}",
    ]


def test_execute_propagates_json_serialization_errors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cases = [
        (float("nan"), ValueError),
        (float("inf"), ValueError),
        (float("-inf"), ValueError),
        ({1}, TypeError),
        (object(), TypeError),
        (b"value", TypeError),
    ]

    for result, error_type in cases:
        module = _patch_invoke(
            monkeypatch,
            lambda name, arguments, result=result: result,
        )
        with pytest.raises(error_type):
            module.execute_tool_calls(
                _response([_function_call()])
            )


def test_execute_does_not_modify_inputs_or_tool_result(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    tool_call = _function_call(
        arguments='{"nested":{"values":[1]}}'
    )
    tool_call_before = copy.deepcopy(tool_call)
    result = {"nested": {"values": [1]}}
    result_before = copy.deepcopy(result)
    module = _patch_invoke(
        monkeypatch,
        lambda name, arguments: result,
    )

    first_messages = module.execute_tool_calls(
        _response([tool_call])
    )
    second_messages = module.execute_tool_calls(
        _response([tool_call])
    )

    assert vars(tool_call) == vars(tool_call_before)
    assert vars(tool_call.function) == vars(tool_call_before.function)
    assert tool_call.function.arguments == '{"nested":{"values":[1]}}'
    assert result == result_before
    assert first_messages[0] is not second_messages[0]


def test_adapter_has_no_sdk_key_or_network_dependency(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail_environment(*args: object, **kwargs: object) -> object:
        pytest.fail("adapter must not read environment variables")

    def fail_network(*args: object, **kwargs: object) -> object:
        pytest.fail("adapter must not access the real network")

    monkeypatch.setattr(os, "getenv", fail_environment)
    monkeypatch.setattr(socket, "create_connection", fail_network)
    sys.modules.pop("llm_adapters.openai_tool_adapter", None)
    sys.modules.pop("llm_adapters", None)
    module = _load_adapter_module()
    source = Path(module.__file__).read_text(encoding="utf-8").lower()
    forbidden_fragments = (
        "import openai",
        "from openai",
        "import requests",
        "from requests",
        "import httpx",
        "from httpx",
        "import socket",
        "from socket",
        "import dotenv",
        "from dotenv",
        "openai_api_key",
        "os.getenv",
        "os.environ",
    )

    for fragment in forbidden_fragments:
        assert fragment not in source

    client = _FakeClient(SimpleNamespace(choices=[]))
    with pytest.raises(ValueError):
        module.execute_tool_calls(client.completions.response)
    assert client.completions.calls == []


def test_existing_tool_registry_contract_remains_unchanged() -> None:
    package = importlib.import_module("tool_layer")
    list_tools = package.list_tools
    invoke_tool = package.invoke_tool

    assert [
        tool["function"]["name"]
        for tool in list_tools()
    ] == ["analyze_experiment", "compare_experiments"]
    assert str(inspect.signature(invoke_tool)) == (
        "(tool_name: str, arguments: dict) -> dict"
    )
    assert not hasattr(package, "openai_tool_adapter")

"""Chat Completions-compatible tool calling adapter."""

import json

from tool_layer import invoke_tool, list_tools


_MISSING = object()


def create_tool_call_response(
    client: object,
    *,
    model: str,
    messages: list[dict],
    **request_options: object,
) -> object:
    """Request one model response with the registered tools."""
    if not isinstance(model, str):
        raise TypeError("model must be a string")
    if not model.strip():
        raise ValueError("model must not be empty or whitespace")
    if type(messages) is not list:
        raise TypeError("messages must be a list")
    if "tools" in request_options:
        raise TypeError("tools are provided by the tool registry")

    tools = list_tools()
    return client.chat.completions.create(
        model=model,
        messages=messages,
        tools=tools,
        **request_options,
    )


def _call_context(
    index: int,
    tool_call_id: object = _MISSING,
    function_name: object = _MISSING,
) -> str:
    parts = [f"tool_calls[{index}]"]
    if isinstance(tool_call_id, str) and tool_call_id.strip():
        parts.append(f"id={tool_call_id!r}")
    if isinstance(function_name, str) and function_name.strip():
        parts.append(f"function={function_name!r}")
    return " ".join(parts)


def _is_attribute_container(value: object) -> bool:
    return (
        value is not None
        and not isinstance(value, dict)
        and (
            hasattr(value, "__dict__")
            or hasattr(type(value), "__slots__")
        )
    )


def _validate_tool_call(
    tool_call: object,
    index: int,
) -> tuple[str, str, dict]:
    tool_call_id = getattr(tool_call, "id", _MISSING)
    id_path = f"tool_calls[{index}].id"
    if tool_call_id is _MISSING:
        raise ValueError(f"{id_path} is required")
    if not isinstance(tool_call_id, str):
        raise TypeError(f"{id_path} must be a string")
    if not tool_call_id.strip():
        raise ValueError(f"{id_path} must not be empty or whitespace")

    call_type = getattr(tool_call, "type", _MISSING)
    type_path = f"tool_calls[{index}].type"
    context = _call_context(index, tool_call_id)
    if call_type is _MISSING:
        raise ValueError(f"{type_path} is required ({context})")
    if not isinstance(call_type, str):
        raise TypeError(f"{type_path} must be a string ({context})")
    if call_type != "function":
        raise ValueError(
            f"{type_path} must be 'function' ({context})"
        )

    function = getattr(tool_call, "function", _MISSING)
    function_path = f"tool_calls[{index}].function"
    if function is _MISSING:
        raise ValueError(f"{function_path} is required ({context})")
    if not _is_attribute_container(function):
        raise TypeError(
            f"{function_path} must be an attribute object ({context})"
        )

    function_name = getattr(function, "name", _MISSING)
    name_path = f"{function_path}.name"
    if function_name is _MISSING:
        raise ValueError(f"{name_path} is required ({context})")
    if not isinstance(function_name, str):
        raise TypeError(f"{name_path} must be a string ({context})")
    if not function_name.strip():
        raise ValueError(
            f"{name_path} must not be empty or whitespace ({context})"
        )

    context = _call_context(index, tool_call_id, function_name)
    arguments_text = getattr(function, "arguments", _MISSING)
    arguments_path = f"{function_path}.arguments"
    if arguments_text is _MISSING:
        raise ValueError(f"{arguments_path} is required ({context})")
    if not isinstance(arguments_text, str):
        raise TypeError(
            f"{arguments_path} must be a string ({context})"
        )
    if not arguments_text.strip():
        raise ValueError(
            f"{arguments_path} must not be empty or whitespace "
            f"({context})"
        )

    try:
        decoded_arguments = json.loads(arguments_text)
    except json.JSONDecodeError as error:
        raise ValueError(
            f"{arguments_path} contains invalid JSON ({context})"
        ) from error

    if type(decoded_arguments) is not dict:
        raise TypeError(
            f"{arguments_path} must decode to an object ({context})"
        )

    return tool_call_id, function_name, decoded_arguments


def execute_tool_calls(
    response: object,
) -> list[dict]:
    """Validate and execute all tool calls in one model response."""
    choices = getattr(response, "choices", _MISSING)
    if choices is _MISSING:
        raise TypeError("response.choices is required")
    if type(choices) is not list:
        raise TypeError("response.choices must be a list")
    if not choices:
        raise ValueError("response.choices must not be empty")

    message = getattr(choices[0], "message", _MISSING)
    if message is _MISSING:
        raise TypeError("response.choices[0].message is required")

    tool_calls = getattr(message, "tool_calls", _MISSING)
    if tool_calls is _MISSING:
        raise TypeError(
            "response.choices[0].message.tool_calls is required"
        )
    if tool_calls is None:
        return []
    if type(tool_calls) is not list:
        raise TypeError(
            "response.choices[0].message.tool_calls must be a list"
        )
    if not tool_calls:
        return []

    validated_calls = [
        _validate_tool_call(tool_call, index)
        for index, tool_call in enumerate(tool_calls)
    ]

    messages: list[dict] = []
    for tool_call_id, function_name, arguments in validated_calls:
        result = invoke_tool(function_name, arguments)
        serialized_result = json.dumps(
            result,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        messages.append(
            {
                "role": "tool",
                "tool_call_id": tool_call_id,
                "content": serialized_result,
            }
        )

    return messages


def _build_assistant_tool_call_message(
    response: object,
) -> dict | None:
    choices = getattr(response, "choices", _MISSING)
    if choices is _MISSING:
        raise TypeError("response.choices is required")
    if type(choices) is not list:
        raise TypeError("response.choices must be a list")
    if not choices:
        raise ValueError("response.choices must not be empty")

    message = getattr(choices[0], "message", _MISSING)
    if message is _MISSING:
        raise TypeError("response.choices[0].message is required")
    if not _is_attribute_container(message):
        raise TypeError(
            "response.choices[0].message must be an attribute object"
        )

    role = getattr(message, "role", _MISSING)
    role_path = "response.choices[0].message.role"
    if role is _MISSING:
        raise ValueError(f"{role_path} is required")
    if not isinstance(role, str):
        raise TypeError(f"{role_path} must be a string")
    if role != "assistant":
        raise ValueError(f"{role_path} must be 'assistant'")

    content = getattr(message, "content", _MISSING)
    content_path = "response.choices[0].message.content"
    if content is _MISSING:
        raise ValueError(f"{content_path} is required")
    if content is not None and not isinstance(content, str):
        raise TypeError(f"{content_path} must be a string or None")

    tool_calls = getattr(message, "tool_calls", _MISSING)
    tool_calls_path = "response.choices[0].message.tool_calls"
    if tool_calls is _MISSING:
        raise TypeError(f"{tool_calls_path} is required")
    if tool_calls is None:
        return None
    if type(tool_calls) is not list:
        raise TypeError(f"{tool_calls_path} must be a list")
    if not tool_calls:
        return None

    validated_calls = [
        _validate_tool_call(tool_call, index)
        for index, tool_call in enumerate(tool_calls)
    ]

    serialized_tool_calls: list[dict] = []
    for tool_call, (tool_call_id, function_name, _) in zip(
        tool_calls,
        validated_calls,
        strict=True,
    ):
        serialized_tool_calls.append(
            {
                "id": tool_call_id,
                "type": getattr(tool_call, "type"),
                "function": {
                    "name": function_name,
                    "arguments": getattr(
                        getattr(tool_call, "function"),
                        "arguments",
                    ),
                },
            }
        )

    return {
        "role": role,
        "content": content,
        "tool_calls": serialized_tool_calls,
    }


def run_tool_call_cycle(
    client: object,
    *,
    model: str,
    messages: list[dict],
    **request_options: object,
) -> object:
    """Run at most one tool execution step and one follow-up request."""
    first_response = create_tool_call_response(
        client,
        model=model,
        messages=messages,
        **request_options,
    )
    assistant_message = _build_assistant_tool_call_message(
        first_response
    )
    if assistant_message is None:
        return first_response

    tool_messages = execute_tool_calls(first_response)
    follow_up_messages = [
        *messages,
        assistant_message,
        *tool_messages,
    ]
    return create_tool_call_response(
        client,
        model=model,
        messages=follow_up_messages,
        **request_options,
    )

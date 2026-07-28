"""Minimal application runtime for one experiment Copilot turn."""

import json

from llm_adapters import run_tool_call_cycle


_MISSING = object()
_ALLOWED_CONTEXT_FIELDS = frozenset(
    {
        "experiment_dir",
        "experiment_root",
        "metrics_config",
    }
)
_RESERVED_REQUEST_OPTIONS = frozenset(
    {
        "messages",
        "system_prompt",
        "system_instruction",
    }
)
_SYSTEM_PROMPT = """
You are a machine-learning experiment analysis Copilot.

Use registered tools whenever answering questions about concrete experiment
data. Base metrics, epochs, paths, configurations, and training outcomes only
on successful tool results. Never invent those details, and never claim that
a failed tool call succeeded.

Clearly distinguish observed facts, inferences, and recommendations. Treat FI
personality recognition as an example use case rather than the product
boundary. Never claim that training occurred unless tool data demonstrates
that training.
""".strip()


def _is_attribute_container(value: object) -> bool:
    return (
        value is not None
        and not isinstance(value, dict)
        and (
            hasattr(value, "__dict__")
            or hasattr(type(value), "__slots__")
        )
    )


def _validate_question(question: object) -> str:
    if not isinstance(question, str):
        raise TypeError("question must be a string")
    if not question.strip():
        raise ValueError("question must not be empty or whitespace")
    return question


def _validate_context(
    experiment_context: object,
) -> dict[str, object] | None:
    if experiment_context is None:
        return None
    if type(experiment_context) is not dict:
        raise TypeError("experiment_context must be a dict or None")

    unknown_fields = (
        set(experiment_context) - _ALLOWED_CONTEXT_FIELDS
    )
    if unknown_fields:
        unknown = sorted(
            repr(field)
            for field in unknown_fields
        )
        raise ValueError(
            "experiment_context contains unknown field(s): "
            + ", ".join(unknown)
        )

    if (
        "experiment_dir" in experiment_context
        and "experiment_root" in experiment_context
    ):
        raise ValueError(
            "experiment_dir and experiment_root cannot both be provided"
        )

    for field, value in experiment_context.items():
        if not isinstance(value, str):
            raise TypeError(
                f"experiment_context[{field!r}] must be a string"
            )
        if not value.strip():
            raise ValueError(
                f"experiment_context[{field!r}] must not be "
                "empty or whitespace"
            )

    return experiment_context


def _validate_request_options(
    request_options: dict[str, object],
) -> None:
    reserved = _RESERVED_REQUEST_OPTIONS.intersection(
        request_options
    )
    if reserved:
        option = sorted(reserved)[0]
        raise TypeError(
            f"{option} is controlled by the Copilot runtime"
        )


def _build_messages(
    question: str,
    experiment_context: dict[str, object] | None,
) -> list[dict]:
    user_content = question
    if experiment_context:
        encoded_context = json.dumps(
            experiment_context,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
        )
        user_content = (
            f"{question}\n\n"
            f"Experiment context:\n{encoded_context}"
        )

    return [
        {
            "role": "system",
            "content": _SYSTEM_PROMPT,
        },
        {
            "role": "user",
            "content": user_content,
        },
    ]


def _extract_final_content(response: object) -> str:
    choices = getattr(response, "choices", _MISSING)
    if choices is _MISSING:
        raise TypeError("response.choices is required")
    if type(choices) is not list:
        raise TypeError("response.choices must be a list")
    if not choices:
        raise ValueError("response.choices must not be empty")

    first_choice = choices[0]
    if not _is_attribute_container(first_choice):
        raise TypeError(
            "response.choices[0] must be an attribute object"
        )

    message = getattr(first_choice, "message", _MISSING)
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

    tool_calls = getattr(message, "tool_calls", _MISSING)
    tool_calls_path = "response.choices[0].message.tool_calls"
    if tool_calls is _MISSING:
        raise TypeError(f"{tool_calls_path} is required")
    if tool_calls is not None:
        if type(tool_calls) is not list:
            raise TypeError(f"{tool_calls_path} must be a list")
        if tool_calls:
            raise ValueError(
                f"{tool_calls_path} must be empty in the final response"
            )

    content = getattr(message, "content", _MISSING)
    content_path = "response.choices[0].message.content"
    if content is _MISSING:
        raise ValueError(f"{content_path} is required")
    if content is None:
        raise ValueError(f"{content_path} must not be None")
    if not isinstance(content, str):
        raise TypeError(f"{content_path} must be a string")
    if not content.strip():
        raise ValueError(
            f"{content_path} must not be empty or whitespace"
        )
    return content


def run_copilot_turn(
    client: object,
    *,
    model: str,
    question: str,
    experiment_context: dict[str, object] | None = None,
    **request_options: object,
) -> str:
    """Run one bounded Copilot turn and return assistant text."""
    validated_question = _validate_question(question)
    validated_context = _validate_context(experiment_context)
    _validate_request_options(request_options)
    messages = _build_messages(
        validated_question,
        validated_context,
    )
    response = run_tool_call_cycle(
        client,
        model=model,
        messages=messages,
        **request_options,
    )
    return _extract_final_content(response)

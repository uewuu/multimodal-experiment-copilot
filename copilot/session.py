"""Bounded in-memory Copilot conversation session."""

from copy import deepcopy
from dataclasses import dataclass

from llm_adapters.openai_tool_adapter import (
    _run_tool_call_cycle_with_trace,
)

from .runtime import (
    _build_messages,
    _extract_final_content,
    _validate_context,
    _validate_question,
    _validate_request_options,
)


_SESSION_RESERVED_REQUEST_OPTIONS = frozenset(
    {
        "tool_choice",
        "tools",
    }
)


@dataclass(frozen=True, slots=True)
class CopilotToolInvocation:
    tool_call_id: str
    tool_name: str
    arguments_json: str
    result_json: str


@dataclass(frozen=True, slots=True)
class CopilotTurn:
    question: str
    answer: str
    tool_call_content: str | None
    tool_invocations: tuple[CopilotToolInvocation, ...]


def _validate_model(model: object) -> str:
    if not isinstance(model, str):
        raise TypeError("model must be a string")
    if not model.strip():
        raise ValueError("model must not be empty or whitespace")
    return model


def _validate_session_request_options(
    request_options: dict[str, object],
) -> None:
    _validate_request_options(request_options)
    reserved = _SESSION_RESERVED_REQUEST_OPTIONS.intersection(
        request_options
    )
    if reserved:
        option = sorted(reserved)[0]
        raise TypeError(
            f"{option} is controlled by the Copilot session"
        )


def _validate_max_turns(max_turns: object) -> int:
    if isinstance(max_turns, bool) or not isinstance(max_turns, int):
        raise TypeError("max_turns must be an integer")
    if max_turns < 1:
        raise ValueError("max_turns must be at least 1")
    return max_turns


def _normalize_tool_trace(
    assistant_message: dict[str, object] | None,
    tool_messages: tuple[dict[str, object], ...],
) -> tuple[str | None, tuple[CopilotToolInvocation, ...]]:
    if assistant_message is None:
        if tool_messages:
            raise ValueError(
                "tool messages require an assistant tool-call message"
            )
        return None, ()
    if type(assistant_message) is not dict:
        raise TypeError("assistant trace message must be a dict")
    if assistant_message.get("role") != "assistant":
        raise ValueError("assistant trace role must be 'assistant'")

    content = assistant_message.get("content")
    if content is not None and not isinstance(content, str):
        raise TypeError(
            "assistant trace content must be a string or None"
        )

    tool_calls = assistant_message.get("tool_calls")
    if type(tool_calls) is not list:
        raise TypeError("assistant trace tool_calls must be a list")
    if len(tool_calls) != len(tool_messages):
        raise ValueError(
            "assistant trace and tool message counts must match"
        )

    invocations: list[CopilotToolInvocation] = []
    for index, (tool_call, tool_message) in enumerate(
        zip(tool_calls, tool_messages, strict=True)
    ):
        if type(tool_call) is not dict:
            raise TypeError(
                f"assistant trace tool_calls[{index}] must be a dict"
            )
        if tool_call.get("type") != "function":
            raise ValueError(
                f"assistant trace tool_calls[{index}].type "
                "must be 'function'"
            )
        tool_call_id = tool_call.get("id")
        if not isinstance(tool_call_id, str):
            raise TypeError(
                f"assistant trace tool_calls[{index}].id "
                "must be a string"
            )
        function = tool_call.get("function")
        if type(function) is not dict:
            raise TypeError(
                f"assistant trace tool_calls[{index}].function "
                "must be a dict"
            )
        tool_name = function.get("name")
        if not isinstance(tool_name, str):
            raise TypeError(
                f"assistant trace tool_calls[{index}].function.name "
                "must be a string"
            )
        arguments_json = function.get("arguments")
        if not isinstance(arguments_json, str):
            raise TypeError(
                f"assistant trace tool_calls[{index}]."
                "function.arguments must be a string"
            )

        if type(tool_message) is not dict:
            raise TypeError(
                f"tool trace messages[{index}] must be a dict"
            )
        if tool_message.get("role") != "tool":
            raise ValueError(
                f"tool trace messages[{index}].role must be 'tool'"
            )
        if tool_message.get("tool_call_id") != tool_call_id:
            raise ValueError(
                f"tool trace messages[{index}].tool_call_id "
                "must match the assistant tool call"
            )
        result_json = tool_message.get("content")
        if not isinstance(result_json, str):
            raise TypeError(
                f"tool trace messages[{index}].content "
                "must be a string"
            )

        invocations.append(
            CopilotToolInvocation(
                tool_call_id=tool_call_id,
                tool_name=tool_name,
                arguments_json=arguments_json,
                result_json=result_json,
            )
        )

    return content, tuple(invocations)


class CopilotSession:
    """Run bounded, transactional Copilot turns with a borrowed client."""

    def __init__(
        self,
        client: object,
        *,
        model: str,
        experiment_context: dict[str, object] | None = None,
        max_turns: int = 8,
        **request_options: object,
    ) -> None:
        validated_model = _validate_model(model)
        validated_context = _validate_context(experiment_context)
        validated_max_turns = _validate_max_turns(max_turns)
        _validate_session_request_options(request_options)

        self._client = client
        self._model = validated_model
        self._experiment_context = deepcopy(validated_context)
        self._max_turns = validated_max_turns
        self._request_options = deepcopy(request_options)
        self._history: tuple[CopilotTurn, ...] = ()

    @property
    def history(self) -> tuple[CopilotTurn, ...]:
        return self._history

    @property
    def turn_count(self) -> int:
        return len(self._history)

    @property
    def model(self) -> str:
        return self._model

    @property
    def max_turns(self) -> int:
        return self._max_turns

    @property
    def experiment_context(
        self,
    ) -> dict[str, object] | None:
        return deepcopy(self._experiment_context)

    def _retained_history(self) -> tuple[CopilotTurn, ...]:
        if self._max_turns == 1:
            return ()
        return self._history[-(self._max_turns - 1):]

    def _build_turn_messages(
        self,
        question: str,
        retained: tuple[CopilotTurn, ...],
    ) -> list[dict]:
        current_messages = _build_messages(
            question,
            self._experiment_context,
        )
        messages = [current_messages[0]]

        for turn in retained:
            previous_user = _build_messages(
                turn.question,
                self._experiment_context,
            )[1]
            messages.append(previous_user)
            if turn.tool_invocations:
                messages.append(
                    {
                        "role": "assistant",
                        "content": turn.tool_call_content,
                        "tool_calls": [
                            {
                                "id": invocation.tool_call_id,
                                "type": "function",
                                "function": {
                                    "name": invocation.tool_name,
                                    "arguments": (
                                        invocation.arguments_json
                                    ),
                                },
                            }
                            for invocation in turn.tool_invocations
                        ],
                    }
                )
                messages.extend(
                    {
                        "role": "tool",
                        "tool_call_id": invocation.tool_call_id,
                        "content": invocation.result_json,
                    }
                    for invocation in turn.tool_invocations
                )
            messages.append(
                {
                    "role": "assistant",
                    "content": turn.answer,
                }
            )

        messages.append(current_messages[1])
        return messages

    def ask_with_result(self, question: str) -> CopilotTurn:
        validated_question = _validate_question(question)
        retained = self._retained_history()
        messages = self._build_turn_messages(
            validated_question,
            retained,
        )
        trace = _run_tool_call_cycle_with_trace(
            self._client,
            model=self._model,
            messages=messages,
            **deepcopy(self._request_options),
        )
        answer = _extract_final_content(trace.response)
        tool_call_content, tool_invocations = _normalize_tool_trace(
            trace.assistant_message,
            trace.tool_messages,
        )
        new_turn = CopilotTurn(
            question=validated_question,
            answer=answer,
            tool_call_content=tool_call_content,
            tool_invocations=tool_invocations,
        )
        self._history = (retained + (new_turn,))[
            -self._max_turns:
        ]
        return new_turn

    def ask(self, question: str) -> str:
        return self.ask_with_result(question).answer

    def export_history(self) -> list[dict[str, object]]:
        return [
            {
                "question": turn.question,
                "answer": turn.answer,
                "tool_call_content": turn.tool_call_content,
                "tool_invocations": [
                    {
                        "tool_call_id": invocation.tool_call_id,
                        "tool_name": invocation.tool_name,
                        "arguments_json": invocation.arguments_json,
                        "result_json": invocation.result_json,
                    }
                    for invocation in turn.tool_invocations
                ],
            }
            for turn in self._history
        ]

    def reset(self) -> None:
        self._history = ()

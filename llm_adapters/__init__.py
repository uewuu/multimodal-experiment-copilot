"""Public OpenAI-compatible tool calling adapters."""

from .openai_tool_adapter import (
    create_tool_call_response,
    execute_tool_calls,
)


__all__ = [
    "create_tool_call_response",
    "execute_tool_calls",
]

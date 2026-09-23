"""Optional host-owned tool collections, active only in a calling context.

Collections snapshot descriptions and dispatch ownership without acquiring any
tool resources. Tool sets retain responsibility for argument validation,
authorization and results; the default registry is never modified.
"""

from collections.abc import Callable, Iterator, Mapping
from contextlib import contextmanager
from contextvars import ContextVar
from copy import deepcopy
from dataclasses import dataclass
from types import MappingProxyType
from typing import Protocol

from tool_layer import tool_registry


__all__ = ("BoundToolCollection", "bound_tools")


class _ToolSet(Protocol):
    def list_tools(self) -> list[dict]: ...

    def invoke_tool(self, tool_name: str, arguments: dict) -> dict: ...


@dataclass(frozen=True, slots=True, init=False, eq=False, repr=False)
class BoundToolCollection:
    """Default tools followed by explicitly bound sets, with unique names.

    Host tool sets must keep their configured capabilities stable for the
    collection's lifetime. Only their descriptions and invocation callables are
    retained; construction never invokes a tool or a Memory provider.
    """

    _definitions: tuple[dict, ...]
    _dispatch: Mapping[str, Callable[[str, dict], dict]]

    def __init__(self, *tool_sets: _ToolSet):
        definitions = []
        dispatch = {}
        for tool_set in (tool_registry, *tool_sets):
            invoke = tool_set.invoke_tool
            for definition in tool_set.list_tools():
                snapshot = deepcopy(definition)
                name = snapshot["function"]["name"]
                if not isinstance(name, str) or not name.strip():
                    raise ValueError("tool name must be a non-empty string")
                if name in dispatch:
                    raise ValueError(f"duplicate tool name: {name}")
                definitions.append(snapshot)
                dispatch[name] = invoke
        object.__setattr__(self, "_definitions", tuple(definitions))
        object.__setattr__(self, "_dispatch", MappingProxyType(dispatch))

    def list_tools(self) -> list[dict]:
        """Return detached descriptions in the frozen collection order."""
        return deepcopy(list(self._definitions))

    def invoke_tool(self, tool_name: str, arguments: dict) -> dict:
        """Dispatch only an advertised name to its original owning tool set."""
        if not isinstance(tool_name, str):
            raise TypeError("tool_name must be a string")
        if not tool_name.strip():
            raise ValueError("tool_name must not be empty or whitespace")
        try:
            invoke = self._dispatch[tool_name]
        except KeyError:
            raise KeyError(f"unknown tool: {tool_name}") from None
        return invoke(tool_name, arguments)


_BOUND_TOOLS: ContextVar[BoundToolCollection | None] = ContextVar(
    "bound_tools", default=None,
)


def _get_bound_tools() -> BoundToolCollection | None:
    return _BOUND_TOOLS.get()


@contextmanager
def bound_tools(collection: BoundToolCollection) -> Iterator[None]:
    """Temporarily activate a collection; restore its predecessor on every exit."""
    if not isinstance(collection, BoundToolCollection):
        raise TypeError("collection must be a BoundToolCollection")
    token = _BOUND_TOOLS.set(collection)
    try:
        yield
    finally:
        _BOUND_TOOLS.reset(token)

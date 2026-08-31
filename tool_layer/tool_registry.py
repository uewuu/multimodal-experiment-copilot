"""Vendor-neutral tool descriptions, capabilities, and dispatch."""

from copy import deepcopy
from dataclasses import dataclass
from typing import Callable, Literal

from .experiment_tools import analyze_experiment, compare_experiments


@dataclass(frozen=True, slots=True)
class _WorkspacePathCapability:
    parameter_name: str
    path_kind: Literal["directory", "file"]


@dataclass(frozen=True, slots=True)
class _ToolCapabilities:
    workspace_paths: tuple[_WorkspacePathCapability, ...] = ()


@dataclass(frozen=True, slots=True)
class _ToolDefinition:
    name: str
    description: str
    parameters: dict[str, object]
    function: Callable[..., dict]
    capabilities: _ToolCapabilities


def _analyze_experiment(**arguments: object) -> dict:
    return analyze_experiment(**arguments)


def _compare_experiments(**arguments: object) -> dict:
    return compare_experiments(**arguments)


_TOOL_DEFINITIONS = (
    _ToolDefinition(
        name="analyze_experiment",
        description=(
            "Analyze one machine-learning experiment directory."
        ),
        parameters={
            "type": "object",
            "properties": {
                "experiment_dir": {
                    "type": "string",
                },
                "metrics_config": {
                    "type": ["string", "null"],
                    "default": None,
                },
                "include_diagnostics": {
                    "type": "boolean",
                    "default": False,
                },
            },
            "required": ["experiment_dir"],
            "additionalProperties": False,
        },
        function=_analyze_experiment,
        capabilities=_ToolCapabilities(
            workspace_paths=(
                _WorkspacePathCapability(
                    parameter_name="experiment_dir",
                    path_kind="directory",
                ),
                _WorkspacePathCapability(
                    parameter_name="metrics_config",
                    path_kind="file",
                ),
            )
        ),
    ),
    _ToolDefinition(
        name="compare_experiments",
        description=(
            "Compare multiple machine-learning experiment directories."
        ),
        parameters={
            "type": "object",
            "properties": {
                "experiment_root": {
                    "type": "string",
                },
                "sort_by": {
                    "type": ["string", "null"],
                    "default": None,
                },
                "descending": {
                    "type": "boolean",
                    "default": True,
                },
                "metrics_config": {
                    "type": ["string", "null"],
                    "default": None,
                },
                "include_diagnostics": {
                    "type": "boolean",
                    "default": False,
                },
            },
            "required": ["experiment_root"],
            "additionalProperties": False,
        },
        function=_compare_experiments,
        capabilities=_ToolCapabilities(
            workspace_paths=(
                _WorkspacePathCapability(
                    parameter_name="experiment_root",
                    path_kind="directory",
                ),
                _WorkspacePathCapability(
                    parameter_name="metrics_config",
                    path_kind="file",
                ),
            )
        ),
    ),
)


def _definition_for_tool(tool_name: str) -> _ToolDefinition | None:
    for definition in _TOOL_DEFINITIONS:
        if definition.name == tool_name:
            return definition
    return None


def _workspace_paths_for_tool(
    tool_name: str,
) -> tuple[_WorkspacePathCapability, ...]:
    definition = _definition_for_tool(tool_name)
    if definition is None:
        return ()
    return definition.capabilities.workspace_paths


def list_tools() -> list[dict]:
    """Return isolated JSON-safe descriptions for registered tools."""
    return [
        {
            "type": "function",
            "function": {
                "name": definition.name,
                "description": definition.description,
                "parameters": deepcopy(definition.parameters),
            },
        }
        for definition in _TOOL_DEFINITIONS
    ]


def invoke_tool(
    tool_name: str,
    arguments: dict,
) -> dict:
    """Invoke a registered tool by its exact name."""
    if not isinstance(tool_name, str):
        raise TypeError("tool_name must be a string")
    if not tool_name.strip():
        raise ValueError("tool_name must not be empty or whitespace")
    if type(arguments) is not dict:
        raise TypeError("arguments must be a dict")

    definition = _definition_for_tool(tool_name)
    if definition is None:
        raise KeyError(f"unknown tool: {tool_name}")

    return definition.function(**arguments)

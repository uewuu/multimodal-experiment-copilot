"""Vendor-neutral tool descriptions and dispatch."""

from copy import deepcopy

from .experiment_tools import analyze_experiment, compare_experiments


_TOOL_DESCRIPTIONS = [
    {
        "type": "function",
        "function": {
            "name": "analyze_experiment",
            "description": (
                "Analyze one machine-learning experiment directory."
            ),
            "parameters": {
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
        },
    },
    {
        "type": "function",
        "function": {
            "name": "compare_experiments",
            "description": (
                "Compare multiple machine-learning experiment directories."
            ),
            "parameters": {
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
        },
    },
]


def list_tools() -> list[dict]:
    """Return isolated JSON-safe descriptions for registered tools."""
    return deepcopy(_TOOL_DESCRIPTIONS)


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

    if tool_name == "analyze_experiment":
        tool_function = analyze_experiment
    elif tool_name == "compare_experiments":
        tool_function = compare_experiments
    else:
        raise KeyError(f"unknown tool: {tool_name}")

    return tool_function(**arguments)

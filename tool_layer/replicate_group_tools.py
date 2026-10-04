"""Explicitly bound comparison of host-supplied replicate group snapshots."""

from dataclasses import dataclass

import experiment_group_comparison as core


__all__ = ("ReplicateGroupComparisonTools",)

_TOOL_NAME = "compare_replicate_groups"


@dataclass(frozen=True, slots=True, init=False, eq=False, repr=False)
class ReplicateGroupComparisonTools:
    """Keep host selections detached and delegate every invocation to the core.

    The core's snapshot boundary owns approved types and input budgets. Reuse
    it for stored, validation and invocation data, without caching any result.
    """

    _group_a: dict
    _group_b: dict
    _targets: object

    def __init__(self, group_a, group_b, *, targets):
        snapshot = core._snapshot_inputs(group_a, group_b, targets)
        # Validate a separate copy so delegate mutation cannot change binding.
        a, b, mapping = core._snapshot_inputs(*snapshot)
        core.compare_replicate_groups(a, b, targets=mapping)
        object.__setattr__(self, "_group_a", snapshot[0])
        object.__setattr__(self, "_group_b", snapshot[1])
        object.__setattr__(self, "_targets", snapshot[2])

    def list_tools(self) -> list[dict]:
        """Return a fresh description containing no host data or model options."""
        return [{
            "type": "function",
            "function": {
                "name": _TOOL_NAME,
                "description": (
                    "Compare host-bound replicate group snapshots descriptively, "
                    "preserving coverage and provenance limitations. Differences "
                    "do not establish pairing, independence, significance or a winner."
                ),
                "parameters": {
                    "type": "object", "properties": {}, "additionalProperties": False,
                },
            },
        }]

    def invoke_tool(self, tool_name: str, arguments: dict) -> dict:
        """Reject model options before core execution and preserve its result."""
        if type(tool_name) is not str:
            raise TypeError("tool_name must be a string")
        if not tool_name.strip():
            raise ValueError("tool_name must not be empty or whitespace")
        if tool_name != _TOOL_NAME:
            raise KeyError("unknown replicate group comparison tool")
        if type(arguments) is not dict:
            raise TypeError("arguments must be a dict")
        if arguments:
            raise ValueError("replicate group comparison accepts no model arguments")
        a, b, mapping = core._snapshot_inputs(self._group_a, self._group_b, self._targets)
        return core.compare_replicate_groups(a, b, targets=mapping)

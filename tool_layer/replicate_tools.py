"""Explicitly bound replicate aggregation over host-supplied memory snapshots."""

from copy import deepcopy
from dataclasses import dataclass

import experiment_replicates


__all__ = ("ReplicateAggregationTools",)

_TOOL_NAME = "aggregate_replicate_evidence"


@dataclass(frozen=True, slots=True, init=False, eq=False, repr=False)
class ReplicateAggregationTools:
    """Keep a detached host snapshot and delegate aggregation to the core.

    Construction validates before copying, rejecting custom/lazy objects without
    invoking their copy hooks. No result is cached: every invocation uses the
    frozen core with the bound data. Acquisition remains the host's responsibility.
    """

    _executions: object
    _group_declaration: dict
    _targets: object

    def __init__(self, executions, *, group_declaration, targets):
        # The public core owns structural limits and declaration validation.
        # Its bounded traversal rejects dynamic objects before deepcopy can
        # dispatch any caller-defined copying behavior. Eligibility failures
        # are valid evidence results and do not prevent binding a snapshot.
        experiment_replicates.aggregate_replicate_evidence(
            executions, group_declaration=group_declaration, targets=targets,
        )
        snapshot = deepcopy((executions, group_declaration, targets))
        object.__setattr__(self, "_executions", snapshot[0])
        object.__setattr__(self, "_group_declaration", snapshot[1])
        object.__setattr__(self, "_targets", snapshot[2])

    def list_tools(self) -> list[dict]:
        """Return a fresh schema exposing no host data or model parameters."""
        return [{
            "type": "function",
            "function": {
                "name": _TOOL_NAME,
                "description": (
                    "Summarize host-bound replicate evidence snapshots with "
                    "descriptive statistics, coverage and eligibility limits. "
                    "Host declarations do not verify independence or checkpoint provenance."
                ),
                "parameters": {
                    "type": "object", "properties": {}, "additionalProperties": False,
                },
            },
        }]

    def invoke_tool(self, tool_name: str, arguments: dict) -> dict:
        """Reject model options before delegating; preserve the core result."""
        if type(tool_name) is not str:
            raise TypeError("tool_name must be a string")
        if not tool_name.strip():
            raise ValueError("tool_name must not be empty or whitespace")
        if tool_name != _TOOL_NAME:
            raise KeyError("unknown replicate aggregation tool")
        if type(arguments) is not dict:
            raise TypeError("arguments must be a dict")
        if arguments:
            raise ValueError("replicate aggregation accepts no model arguments")
        return experiment_replicates.aggregate_replicate_evidence(
            self._executions,
            group_declaration=self._group_declaration,
            targets=self._targets,
        )

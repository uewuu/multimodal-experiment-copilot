"""Explicitly bound replicate aggregation over host-supplied memory snapshots."""

from dataclasses import dataclass
import json

import experiment_replicates


__all__ = ("ReplicateAggregationTools",)

_TOOL_NAME = "aggregate_replicate_evidence"
_SNAPSHOT_TYPES = experiment_replicates._DATA_TYPES
_MAX_INPUT_BYTES = 2 * 1024 * 1024


def _data_snapshot(envelope):
    """Copy only the bounded data projection that the frozen core accepts.

    Exact built-ins and approved, ordinary-field dataclasses have no caller
    traversal protocols. Extra instance attributes are never inspected. Count
    the compact UTF-8 input projection while copying, before unbounded work.
    Scalar JSON encoding only measures size; it is not an object-copy mechanism.
    """
    used = 0
    ancestors = set()

    def charge(size):
        nonlocal used
        used += size
        if used > _MAX_INPUT_BYTES:
            raise ValueError("replicate input exceeds 2 MiB")

    def scalar(value):
        if type(value) is str and len(value) > _MAX_INPUT_BYTES:
            raise ValueError("replicate input exceeds 2 MiB")
        if type(value) is int and value.bit_length() > _MAX_INPUT_BYTES * 4:
            raise ValueError("replicate input integer exceeds budget")
        charge(len(json.dumps(value, ensure_ascii=False,
                              separators=(",", ":")).encode("utf-8")))
        return value

    def visit(value, depth):
        kind = type(value)
        if value is None or any(kind is allowed for allowed in (str, int, float, bool)):
            # Nonfinite floats are data, not executable objects. Preserve them
            # so the core can exclude invalid observations locally as contracted.
            return scalar(value)
        if not any(kind is allowed for allowed in (dict, list, tuple, *_SNAPSHOT_TYPES)):
            raise TypeError("snapshot requires exact data types and approved dataclasses")
        if depth > 16 or id(value) in ancestors:
            raise ValueError("input exceeds structure depth or contains a cycle")
        ancestors.add(id(value))
        try:
            charge(2)
            if kind is list or kind is tuple:
                copied = []
                for index, item in enumerate(value):
                    charge(int(index > 0))
                    copied.append(visit(item, depth + 1))
                return tuple(copied) if kind is tuple else copied
            entries = (value.items() if kind is dict else
                       experiment_replicates._data_fields(value))
            copied = {}
            for index, (key, item) in enumerate(entries):
                if type(key) is not str:
                    raise TypeError("snapshot mapping keys must be strings")
                charge(1 + int(index > 0))
                scalar(key)
                copied[key] = visit(item, depth + 1)
            # Only constructors of the four approved project types can run.
            return copied if kind is dict else kind(**copied)
        finally:
            ancestors.remove(id(value))

    return visit(envelope, 1)


@dataclass(frozen=True, slots=True, init=False, eq=False, repr=False)
class ReplicateAggregationTools:
    """Keep a detached host snapshot and delegate aggregation to the core.

    Construction projects data before core validation. Each core call receives
    an independent projection, keeping host, stored and delegated data isolated.
    No result is cached. Acquisition remains the host's responsibility.
    """

    _executions: object
    _group_declaration: dict
    _targets: object

    def __init__(self, executions, *, group_declaration, targets):
        snapshot = _data_snapshot({
            "executions": executions, "group_declaration": group_declaration, "targets": targets,
        })
        # Core validation receives its own safe copy too; it cannot expose or
        # mutate the snapshot subsequently retained by this binding.
        validation = _data_snapshot(snapshot)
        experiment_replicates.aggregate_replicate_evidence(
            validation["executions"], group_declaration=validation["group_declaration"],
            targets=validation["targets"],
        )
        object.__setattr__(self, "_executions", snapshot["executions"])
        object.__setattr__(self, "_group_declaration", snapshot["group_declaration"])
        object.__setattr__(self, "_targets", snapshot["targets"])

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
        invocation = _data_snapshot({
            "executions": self._executions,
            "group_declaration": self._group_declaration,
            "targets": self._targets,
        })
        return experiment_replicates.aggregate_replicate_evidence(
            invocation["executions"],
            group_declaration=invocation["group_declaration"],
            targets=invocation["targets"],
        )

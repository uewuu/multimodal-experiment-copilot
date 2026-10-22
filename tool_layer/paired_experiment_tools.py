"""Explicitly bound paired comparisons over host-supplied snapshots."""

from dataclasses import dataclass

import experiment_paired_comparison as _core


__all__ = ("PairedReplicateComparisonTools",)

_TOOL_NAME = "compare_paired_replicates"


def _snapshot_binding(group_a, group_b, pairing_declaration, targets):
    """Reuse the frozen Core's exact-type, callback-safe snapshot boundary.

    Each input keeps its own depth and byte budget. The real Core validates
    their combined envelope and all semantic contracts before binding succeeds.
    """
    return (
        _core._snapshot(group_a, 2 * 1024 * 1024, 16)[0],
        _core._snapshot(group_b, 2 * 1024 * 1024, 16)[0],
        _core._snapshot(pairing_declaration, 16 * 1024, 8)[0],
        _core._snapshot(targets, 16 * 1024, 8)[0],
    )


@dataclass(frozen=True, slots=True, init=False, eq=False, repr=False)
class PairedReplicateComparisonTools:
    """Keep host selections detached; delegate without caching any result."""

    _group_a: dict
    _group_b: dict
    _pairing_declaration: dict
    _targets: list

    def __init__(self, group_a, group_b, *, pairing_declaration, targets):
        snapshot = _snapshot_binding(group_a, group_b, pairing_declaration, targets)
        # Only validated plain data reaches _clone. Give constructor validation
        # a separate copy, so even delegate mutation cannot alter stored data.
        a, b, pairing, mapping = (_core._clone(value) for value in snapshot)
        _core.compare_paired_replicates(a, b, pairing_declaration=pairing, targets=mapping)
        object.__setattr__(self, "_group_a", snapshot[0])
        object.__setattr__(self, "_group_b", snapshot[1])
        object.__setattr__(self, "_pairing_declaration", snapshot[2])
        object.__setattr__(self, "_targets", snapshot[3])

    def list_tools(self) -> list[dict]:
        """Return an independent schema with no host data or model options."""
        return [{
            "type": "function",
            "function": {
                "name": _TOOL_NAME,
                "description": (
                    "Compare host-bound, explicitly paired replicate snapshots "
                    "using descriptive statistics and declared coverage. Pairing "
                    "and variant context are host declarations; results do not "
                    "verify experimental control, causality, significance or a winner."
                ),
                "parameters": {
                    "type": "object", "properties": {}, "additionalProperties": False,
                },
            },
        }]

    def invoke_tool(self, tool_name: str, arguments: dict) -> dict:
        """Reject model options before Core dispatch; propagate its exceptions."""
        if type(tool_name) is not str:
            raise TypeError("tool_name must be an exact string")
        if tool_name != _TOOL_NAME:
            raise KeyError("unknown paired replicate comparison tool")
        if type(arguments) is not dict:
            raise TypeError("arguments must be an exact dict")
        # Exact dict length does not inspect, hash or format any untrusted key.
        if arguments:
            raise ValueError("paired replicate comparison accepts no model arguments")
        a, b, pairing, mapping = _snapshot_binding(
            self._group_a, self._group_b, self._pairing_declaration, self._targets,
        )
        return _core.compare_paired_replicates(a, b, pairing_declaration=pairing, targets=mapping)

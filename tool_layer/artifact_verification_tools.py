"""Explicitly bound, in-memory artifact descriptor comparison tools."""

import experiment_artifact_verification as _core


__all__ = ("ArtifactVerificationTools",)


class ArtifactVerificationTools:
    """Keep host selections isolated and delegate every call to the Core.

    The Core's data-only snapshot boundary rejects caller protocols before
    validation. Delegates receive fresh copies, never the retained snapshot.
    """

    def __init__(self, declared_set, reported_observations):
        declared, reports = _core._snapshot_inputs(declared_set, reported_observations)
        # Validate a separate copy; retain neither delegate mutations nor results.
        _core.compare_artifact_descriptors(*_core._snapshot_inputs(declared, reports))
        self._declared_set = declared
        self._reported_observations = reports

    def list_tools(self) -> list[dict]:
        """Return a fresh schema containing no host data or model options."""
        return [{
            "type": "function",
            "function": {
                "name": "compare_artifact_descriptors",
                "description": (
                    "Compare host-bound artifact descriptors in memory. "
                    "Results describe host-reported fields only; no artifacts "
                    "are read or independently verified."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {},
                    "additionalProperties": False,
                },
            },
        }]

    def invoke_tool(self, tool_name: str, arguments: dict) -> dict:
        """Reject model options before passing detached selections to the Core."""
        if type(tool_name) is not str:
            raise TypeError("tool_name must be an exact string")
        if tool_name != "compare_artifact_descriptors":
            raise KeyError(f"unknown tool: {tool_name}")
        if type(arguments) is not dict:
            raise TypeError("arguments must be an exact dict")
        if arguments:
            raise ValueError("compare_artifact_descriptors accepts no model arguments")
        declared, reports = _core._snapshot_inputs(
            self._declared_set, self._reported_observations,
        )
        return _core.compare_artifact_descriptors(declared, reports)

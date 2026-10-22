"""Host-authorized, read-only tools over borrowed M12 ExperimentMemory.

These definitions are opt-in and do not register tools with the Agent. The host
owns storage preparation, connection threads and resource cleanup. M12 owns
queries and lifecycle; M11 owns comparability. Output limits bound model data,
not database scans, execution time or cross-call snapshots.
"""

from collections.abc import Callable, Collection, Mapping
from dataclasses import dataclass
import json

from experiment_identity import ExecutionIdentity, ExperimentIdentity, MetricResultProvenance
from experiment_memory import ExperimentMemory
from experiment_repository import (
    RepositoryCorruptionError,
    RepositoryError,
    UnsupportedRepositoryVersionError,
)


__all__ = ("MemoryAccessScope", "MemoryTools")

_GET = "get_historical_execution"
_LIST = "list_historical_executions"
_COMPARE = "find_historical_comparison_candidates"
_REQUIRED = {
    _GET: ("experiment_id", "execution_id"),
    _LIST: ("experiment_id",),
    _COMPARE: ("experiment_id", "reference_execution_id", "metric_name"),
}
_DESCRIPTIONS = {
    _GET: "Read-only lookup of one authorized historical execution with safe evidence fields.",
    _LIST: "Read-only history listing with exact provenance filters in execution-ID order, not score ranking.",
    _COMPARE: (
        "Read-only comparison candidates using a stored reference metric. Preserve comparable, "
        "incompatible and unknown states with reasons; these are not reuse recommendations. "
        "A self-reference is not independent replication."
    ),
}
_PROVENANCE_FIELDS = (
    "dataset_version", "split_id", "task", "target", "realized_seed", "source_revision",
)
_METRIC_FIELDS = (
    "metric_definition", "direction", "aggregation", "evaluation_protocol", "selection_protocol",
)
_ARTIFACT_FIELDS = ("checkpoint_ref", "result_artifact_ref", "history_ref")
_MAX_RESULT_BYTES = 32 * 1024


def _valid_identifier(value: object) -> bool:
    return type(value) is str and bool(value.strip()) and len(value) <= 256


@dataclass(frozen=True, slots=True, init=False)
class MemoryAccessScope:
    """Exact host declarations; an empty allowlist denies all, '*' is literal."""

    namespace: str
    allowed_experiment_ids: frozenset[str]
    allow_invalidated: bool

    def __init__(self, namespace, allowed_experiment_ids, *, allow_invalidated=False):
        if type(namespace) is not str or not namespace.strip():
            raise ValueError("namespace must be a nonblank string")
        if (
            not isinstance(allowed_experiment_ids, Collection)
            or isinstance(allowed_experiment_ids, (str, bytes, Mapping))
        ):
            raise ValueError("allowed_experiment_ids must be an explicit collection")
        identifiers = tuple(allowed_experiment_ids)
        if any(not _valid_identifier(value) for value in identifiers):
            raise ValueError("Invalid allowed experiment identity")
        if type(allow_invalidated) is not bool:
            raise ValueError("allow_invalidated must be a boolean")
        object.__setattr__(self, "namespace", namespace)
        object.__setattr__(self, "allowed_experiment_ids", frozenset(identifiers))
        object.__setattr__(self, "allow_invalidated", allow_invalidated)


def _parameters(tool_name: str) -> dict:
    properties = {
        name: {"type": "string", "minLength": 1, "maxLength": 256, "pattern": r"\S"}
        for name in _REQUIRED[tool_name]
    }
    if tool_name != _GET:
        properties.update({
            "filters": {
                "type": "object",
                "properties": {
                    name: {"type": "integer" if name == "realized_seed" else "string"}
                    for name in _PROVENANCE_FIELDS
                },
                "additionalProperties": False,
            },
            "limit": {"type": "integer", "minimum": 1, "maximum": 50, "default": 10},
        })
    properties["include_invalidated"] = {"type": "boolean", "default": False}
    return {"type": "object", "properties": properties,
            "required": list(_REQUIRED[tool_name]), "additionalProperties": False}


def _argument_error(tool_name: str, arguments: object) -> str | None:
    """Validate the tool boundary before borrowing any host resource."""
    if type(arguments) is not dict:
        return "invalid_arguments"
    allowed = {*_REQUIRED[tool_name], "include_invalidated"}
    if tool_name != _GET:
        allowed.update(("filters", "limit"))
    if arguments.keys() - allowed:
        return "unsupported_option"
    if any(not _valid_identifier(arguments.get(name)) for name in _REQUIRED[tool_name]):
        return "invalid_arguments"
    if type(arguments.get("include_invalidated", False)) is not bool:
        return "invalid_arguments"
    if tool_name != _GET:
        limit = arguments.get("limit", 10)
        if type(limit) is not int or not 1 <= limit <= 50:
            return "invalid_arguments"
        if "filters" in arguments:
            filters = arguments["filters"]
            if type(filters) is not dict:
                return "invalid_filter"
            for name, value in filters.items():
                if type(name) is not str or name not in _PROVENANCE_FIELDS:
                    return "invalid_filter"
                if type(value) is not (int if name == "realized_seed" else str):
                    return "invalid_filter"
    return None


def _error(code: str) -> dict:
    return {"format_version": 1, "ok": False, "error": {"code": code}}


def _success(data: dict) -> dict:
    result = {"format_version": 1, "ok": True, "data": data}
    encoded = json.dumps(result, ensure_ascii=False, allow_nan=False,
                         sort_keys=True, separators=(",", ":")).encode("utf-8")
    return _error("result_too_large") if len(encoded) > _MAX_RESULT_BYTES else result


def _execution(stored) -> dict:
    identity = stored.record.provenance.execution
    return {
        "execution": {"namespace": identity.experiment.namespace,
                      "experiment_id": identity.experiment.experiment_id,
                      "execution_id": identity.execution_id},
        "lifecycle": {"status": stored.status},
    }


def _entry(stored) -> dict:
    return {
        **_execution(stored),
        "provenance": {name: getattr(stored.record.provenance, name)
                       for name in _PROVENANCE_FIELDS},
        "comparability": None,
    }


def _metric(metric: MetricResultProvenance) -> dict:
    """Project declarations without resolving or revealing artifact references."""
    return {
        "metric_name": metric.metric_name, "value": metric.value, "best_epoch": metric.best_epoch,
        "metric_provenance": {
            **{name: getattr(metric.provenance, name)
               for name in ("task", "target", "dataset_version", "split_id")},
            **{name: getattr(metric, name) for name in _METRIC_FIELDS},
        },
        "artifacts": {name: "not_declared" if getattr(metric, name) is None else "withheld"
                      for name in _ARTIFACT_FIELDS},
    }


def _visible_execution(memory, execution, include_invalidated):
    # Only M12's exact missing-key boundary is translated. KeyErrors elsewhere
    # remain programmer exceptions rather than being disguised as missing data.
    try:
        stored = memory.get(execution)
    except KeyError:
        return None
    if stored.status == "invalidated" and not include_invalidated:
        return None
    return stored


def _query(memory, tool_name, experiment, arguments):
    include_invalidated = arguments.get("include_invalidated", False)
    if tool_name == _GET:
        stored = _visible_execution(
            memory, ExecutionIdentity(experiment, arguments["execution_id"]), include_invalidated,
        )
        if stored is None:
            return _error("execution_not_found")
        return _success({**_entry(stored), "metrics": {
            name: _metric(metric) for name, metric in stored.record.metric_results.items()
        }})

    limit = arguments.get("limit", 10)
    options = {"filters": arguments.get("filters"), "limit": limit + 1,
               "include_invalidated": include_invalidated}
    if tool_name == _LIST:
        stored_records = memory.list_executions(experiment, **options)
        return _success({"items": [_entry(stored) for stored in stored_records[:limit]],
                         "limit": limit, "has_more": len(stored_records) > limit})

    reference_identity = ExecutionIdentity(experiment, arguments["reference_execution_id"])
    reference = _visible_execution(memory, reference_identity, include_invalidated)
    if reference is None:
        return _error("execution_not_found")
    metric = reference.record.metric_results.get(arguments["metric_name"])
    if metric is None:
        return _error("metric_not_found")
    candidates = memory.find_comparison_candidates(
        experiment, metric, comparable_only=False, **options,
    )
    items = []
    for candidate in candidates[:limit]:
        assessment = candidate.comparability
        items.append({
            **_execution(candidate.stored), "metric": _metric(candidate.metric_result),
            "comparability": {"status": assessment.status,
                              "reason_codes": list(assessment.reason_codes),
                              "policy_version": assessment.policy_version},
            "is_reference": candidate.stored.record.provenance.execution == reference_identity,
        })
    return _success({"reference": {**_execution(reference), "metric": _metric(metric)},
                     "items": items, "limit": limit, "has_more": len(candidates) > limit})


class MemoryTools:
    """Three local read-only tools; the provider lends Memory on each calling thread.

    The provider must return an already configured Memory usable on that thread.
    This object stores only host configuration, never returned Memory or results.
    """

    __slots__ = ("_scope", "_memory_provider")

    def __init__(self, scope: MemoryAccessScope, memory_provider: Callable[[], ExperimentMemory]):
        if not isinstance(scope, MemoryAccessScope):
            raise TypeError("scope must be a MemoryAccessScope")
        if not callable(memory_provider):
            raise TypeError("memory_provider must be callable")
        self._scope = scope
        self._memory_provider = memory_provider

    def list_tools(self) -> list[dict]:
        """Build independent descriptions without acquiring Memory."""
        return [{"type": "function", "function": {
            "name": name, "description": _DESCRIPTIONS[name], "parameters": _parameters(name),
        }} for name in _REQUIRED]

    def invoke_tool(self, tool_name: str, arguments: dict) -> dict:
        """Validate, authorize, borrow once, then project a bounded query result."""
        if type(tool_name) is not str or tool_name not in _REQUIRED:
            raise KeyError("Unknown Memory tool")
        error = _argument_error(tool_name, arguments)
        if error is not None:
            return _error(error)
        # All accepted values are scalar except the flat filters mapping. Take
        # a detached request before executing the host provider.
        arguments = dict(arguments)
        if "filters" in arguments:
            arguments["filters"] = dict(arguments["filters"])
        scope = self._scope
        if (arguments["experiment_id"] not in scope.allowed_experiment_ids
                or (arguments.get("include_invalidated", False) and not scope.allow_invalidated)):
            return _error("unauthorized_scope")
        experiment = ExperimentIdentity(scope.namespace, arguments["experiment_id"])
        try:
            memory = self._memory_provider()
            if not isinstance(memory, ExperimentMemory):
                raise TypeError("memory_provider must return an ExperimentMemory")
            return _query(memory, tool_name, experiment, arguments)
        except RepositoryCorruptionError:
            return _error("repository_corrupt")
        except UnsupportedRepositoryVersionError:
            return _error("unsupported_repository_version")
        except RepositoryError:
            return _error("repository_unavailable")

"""Read-only historical queries over a borrowed experiment repository.

Repository owns persistence, lifecycle and independent evidence snapshots.
M11 owns comparability. Limits bound returned results, not repository reads.
"""

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from experiment_comparability import ComparabilityResult, compare_metric_results
from experiment_identity import ExecutionIdentity, ExperimentIdentity, MetricResultProvenance
from experiment_repository import ExperimentRepository


@dataclass(frozen=True)
class ComparisonCandidate:
    """Stored evidence and its M11 assessment, without a reuse recommendation."""

    # Preserve the repository's returned envelope without importing its private type.
    stored: Any
    metric_result: MetricResultProvenance
    comparability: ComparabilityResult


_FILTER_FIELDS = (
    "dataset_version", "split_id", "task", "target", "realized_seed", "source_revision",
)


def _validate_query(filters, limit):
    if type(limit) is not int or limit <= 0:
        raise ValueError("limit must be a positive integer")
    if filters is None:
        return ()
    if not isinstance(filters, Mapping):
        raise ValueError("filters must be a mapping or None")
    validated = []
    for name, value in filters.items():
        if type(name) is not str or name not in _FILTER_FIELDS:
            raise ValueError(f"Unsupported provenance filter: {name}")
        expected_type = int if name == "realized_seed" else str
        if type(value) is not expected_type:
            raise ValueError(f"{name} must be a concrete {expected_type.__name__} value")
        validated.append((name, value))
    return tuple(validated)


class ExperimentMemory:
    """Query one explicit experiment at a time, without owning its repository.

    Every call reads current repository state. No evidence cache, connection
    management, lifecycle writes or artifact access are performed here.
    """

    def __init__(self, repository: ExperimentRepository):
        self._repository = repository

    def get(self, execution: ExecutionIdentity):
        """Delegate exact lookup, preserving lifecycle and missing-key errors."""
        return self._repository.get(execution)

    def _matching_executions(self, experiment, filters, include_invalidated):
        stored_records = self._repository.list_executions(
            experiment, include_invalidated=include_invalidated,
        )
        return (
            stored for stored in stored_records
            if all(getattr(stored.record.provenance, name) == value for name, value in filters)
        )

    def list_executions(
        self,
        experiment: ExperimentIdentity,
        *,
        filters: Mapping[str, str | int] | None = None,
        include_invalidated: bool = False,
        limit: int = 100,
    ) -> tuple:
        """Return exact AND matches in case-sensitive execution-ID order.

        Unknown declarations remain unchanged and cannot match a concrete
        filter. Lifecycle visibility and provenance filtering precede limit.
        """
        validated = _validate_query(filters, limit)
        matches = self._matching_executions(experiment, validated, include_invalidated)
        ordered = sorted(matches, key=lambda stored: stored.record.provenance.execution.execution_id)
        return tuple(ordered[:limit])

    def find_comparison_candidates(
        self,
        experiment: ExperimentIdentity,
        reference_metric_result: MetricResultProvenance,
        *,
        filters: Mapping[str, str | int] | None = None,
        include_invalidated: bool = False,
        comparable_only: bool = False,
        limit: int = 100,
    ) -> tuple[ComparisonCandidate, ...]:
        """Assess same-metric evidence using M11, retaining all three states.

        References need not be stored and their identity is not excluded.
        Missing metrics are omitted. Explicit comparable-only selection occurs
        before sorting and limiting; invalidation never changes an assessment.
        """
        validated = _validate_query(filters, limit)
        candidates = []
        for stored in self._matching_executions(experiment, validated, include_invalidated):
            metric = stored.record.metric_results.get(reference_metric_result.metric_name)
            if metric is None:
                continue
            assessment = compare_metric_results(reference_metric_result, metric)
            if comparable_only and assessment.status != "comparable":
                continue
            candidates.append(ComparisonCandidate(stored, metric, assessment))
        candidates.sort(key=lambda candidate: candidate.stored.record.provenance.execution.execution_id)
        return tuple(candidates[:limit])

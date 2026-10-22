"""Pure comparison of declared metric results under metric-result-v1."""

from dataclasses import dataclass
from typing import Literal

from experiment_identity import MetricResultProvenance


@dataclass(frozen=True)
class ComparabilityResult:
    """Policy assessment with stable, ordered reasons."""

    status: Literal["comparable", "incompatible", "unknown"]
    reason_codes: tuple[str, ...]
    policy_version: Literal["metric-result-v1"] = "metric-result-v1"


_PROVENANCE_FIELDS = ("task", "target", "dataset_version", "split_id")
_REQUIRED_FIELDS = (
    *_PROVENANCE_FIELDS,
    "metric_definition",
    "direction",
    "aggregation",
    "evaluation_protocol",
    "selection_protocol",
)


def compare_metric_results(
    left: MetricResultProvenance, right: MetricResultProvenance,
) -> ComparabilityResult:
    """Assess declared comparison prerequisites, allowing model/config/seed variation.

    Comparable does not establish identity, truth, significance, causality, or
    safe substitution. Only the nine policy dimensions participate; metric
    values, epochs, and artifact references do not fill missing declarations.
    Known conflicts take precedence, retaining missing-field reasons as well.
    """
    reasons: list[str] = []
    has_conflict = False
    for name in _REQUIRED_FIELDS:
        left_source = left.provenance if name in _PROVENANCE_FIELDS else left
        right_source = right.provenance if name in _PROVENANCE_FIELDS else right
        left_value = getattr(left_source, name)
        right_value = getattr(right_source, name)
        if left_value is None or right_value is None:
            reasons.append(f"missing_{name}")
        elif left_value != right_value:
            reasons.append(f"{name}_mismatch")
            has_conflict = True

    if has_conflict:
        return ComparabilityResult("incompatible", tuple(reasons))
    if reasons:
        return ComparabilityResult("unknown", tuple(reasons))
    return ComparabilityResult("comparable", ())

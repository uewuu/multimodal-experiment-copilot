"""Explicit experiment identities and caller-declared provenance.

References are opaque values: this module neither resolves them nor infers
missing declarations. Value equality is not verification of provenance.
"""

from dataclasses import dataclass
from typing import Literal


@dataclass(frozen=True)
class ExperimentIdentity:
    """A caller-assigned experiment key within a namespace."""

    namespace: str
    experiment_id: str


@dataclass(frozen=True)
class ExecutionIdentity:
    """One explicitly identified execution of an experiment."""

    experiment: ExperimentIdentity
    execution_id: str


@dataclass(frozen=True)
class ExperimentProvenance:
    """Definition and execution declarations; None denotes unknown information.

    Realized seeds describe executions and never alter experiment identity.
    Locations and timestamps are record metadata, not identity sources.
    """

    execution: ExecutionIdentity
    configuration_ref: str | None = None
    task: str | None = None
    target: str | None = None
    model_ref: str | None = None
    dataset_version: str | None = None
    split_id: str | None = None
    source_revision: str | None = None
    initialization_checkpoint_ref: str | None = None
    realized_seed: int | None = None
    environment_ref: str | None = None
    location: str | None = None
    recorded_at: str | None = None


@dataclass(frozen=True)
class MetricResultProvenance:
    """One metric result with independently declared selection and artifacts.

    Equal epochs do not imply shared checkpoints. Artifact references remain
    absent unless the caller explicitly supplies them for this result.
    """

    metric_name: str
    value: float
    provenance: ExperimentProvenance
    metric_definition: str | None = None
    direction: str | None = None
    aggregation: str | None = None
    evaluation_protocol: str | None = None
    selection_protocol: str | None = None
    best_epoch: int | None = None
    checkpoint_ref: str | None = None
    result_artifact_ref: str | None = None
    history_ref: str | None = None


@dataclass(frozen=True)
class IdentityConsistency:
    """Agreement of declarations, without verification of their truth."""

    status: Literal["consistent", "conflict", "unknown"]
    reason_codes: tuple[str, ...]


_DEFINITION_FIELDS = (
    "configuration_ref",
    "task",
    "target",
    "model_ref",
    "dataset_version",
    "split_id",
    "source_revision",
    "initialization_checkpoint_ref",
)


def check_identity_consistency(
    left: ExperimentProvenance, right: ExperimentProvenance,
) -> IdentityConsistency:
    """Compare definition declarations under the same experiment identity.

    Different experiment identities are outside this check's scope and raise
    ValueError. Missing values on either side remain unknown, including when
    both are missing. Conflicts take precedence without discarding reasons.
    """
    if left.execution.experiment != right.execution.experiment:
        raise ValueError("Identity consistency requires the same experiment identity")

    reasons: list[str] = []
    has_conflict = False
    for name in _DEFINITION_FIELDS:
        left_value = getattr(left, name)
        right_value = getattr(right, name)
        if left_value is None or right_value is None:
            reasons.append(f"missing_{name}")
        elif left_value != right_value:
            reasons.append(f"{name}_mismatch")
            has_conflict = True

    if has_conflict:
        return IdentityConsistency("conflict", tuple(reasons))
    if reasons:
        return IdentityConsistency("unknown", tuple(reasons))
    return IdentityConsistency("consistent", ())

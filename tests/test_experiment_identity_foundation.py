"""M11 Slice 1 RED contracts for explicit identities and declared provenance.

Future experiment_identity exports frozen dataclass value models:
ExperimentIdentity(namespace, experiment_id), ExecutionIdentity(experiment,
execution_id), and ExperimentProvenance(execution, ...optional declarations).
References are caller-declared opaque strings, never resolved from files.

check_identity_consistency compares definition provenance under the same
declared experiment identity. Its status is consistent, conflict, or unknown;
consistent means declarations agree, not that their truth has been verified.
Known conflicts take precedence over missing information. Reasons are emitted
in DEFINITION_FIELDS order, as <field>_mismatch or missing_<field>.
Execution IDs, realized seeds, environments, locations, and timestamps are
execution/record provenance and do not participate in definition consistency.
"""

from copy import deepcopy
from dataclasses import FrozenInstanceError, fields, replace
import importlib
import importlib.util

import pytest


DEFINITION_FIELDS = (
    "configuration_ref", "task", "target", "model_ref", "dataset_version",
    "split_id", "source_revision", "initialization_checkpoint_ref",
)


def _api(*names):
    assert importlib.util.find_spec("experiment_identity") is not None, (
        "missing M11 experiment_identity capability: " + ", ".join(names)
    )
    module = importlib.import_module("experiment_identity")
    missing = [name for name in names if not callable(getattr(module, name, None))]
    assert not missing, f"missing M11 identity API: {missing}"
    return module


def _provenance(api, execution_id="execution-a", seed=11):
    return api.ExperimentProvenance(
        execution=api.ExecutionIdentity(
            experiment=api.ExperimentIdentity("lab-a", "experiment-x"),
            execution_id=execution_id,
        ),
        configuration_ref="config:v1",
        task="regression",
        target="target-a",
        model_ref="architecture:v1",
        dataset_version="dataset:v1",
        split_id="split:v1",
        source_revision="repository-a:revision-1",
        initialization_checkpoint_ref="initial-weights:v1",
        realized_seed=seed,
        environment_ref="environment:v1",
        location="records/original",
        recorded_at="2026-01-01T00:00:00Z",
    )


def test_experiment_identity_requires_explicit_keys_and_has_value_equality():
    api = _api("ExperimentIdentity")
    identity = api.ExperimentIdentity(namespace="lab-a", experiment_id="experiment-x")
    assert tuple(field.name for field in fields(identity)) == (
        "namespace", "experiment_id",
    )
    assert identity == api.ExperimentIdentity("lab-a", "experiment-x")
    assert identity != api.ExperimentIdentity("lab-a", "experiment-y")
    assert (identity.namespace, identity.experiment_id) == ("lab-a", "experiment-x")
    with pytest.raises(FrozenInstanceError):
        identity.experiment_id = "changed"
    for arguments in (
        {}, {"namespace": "lab-a"}, {"experiment_id": "experiment-x"},
        {"namespace": "lab-a", "location": "records/20260101"},
        {"namespace": "lab-a", "recorded_at": "2026-01-01T00:00:00Z"},
    ):
        with pytest.raises(TypeError):
            api.ExperimentIdentity(**arguments)


def test_execution_identity_supports_multiple_seeds_under_one_experiment():
    api = _api(
        "ExperimentIdentity", "ExecutionIdentity", "ExperimentProvenance",
        "check_identity_consistency",
    )
    first = _provenance(api, "execution-a", 11)
    second = _provenance(api, "execution-b", 29)
    assert first.execution.experiment == second.execution.experiment
    assert first.execution != second.execution
    assert first.execution == api.ExecutionIdentity(
        first.execution.experiment, "execution-a",
    )
    assert (first.realized_seed, second.realized_seed) == (11, 29)
    assert tuple(field.name for field in fields(first.execution)) == (
        "experiment", "execution_id",
    )
    with pytest.raises(TypeError):
        api.ExecutionIdentity(experiment=first.execution.experiment)
    with pytest.raises(FrozenInstanceError):
        first.execution.execution_id = "changed"
    moved = replace(
        second, location="copied/renamed", recorded_at="2026-09-16T12:00:00Z",
        environment_ref="environment:v2",
    )
    assert moved.execution == second.execution
    assessment = api.check_identity_consistency(first, moved)
    assert assessment.status == "consistent"
    assert assessment.reason_codes == ()
    # A caller may explicitly choose a separate definition for another seed.
    declared_variant = replace(second, execution=api.ExecutionIdentity(
        api.ExperimentIdentity("lab-a", "caller-declared-seed-variant"),
        "execution-c",
    ))
    assert declared_variant.execution.experiment != first.execution.experiment


def test_missing_provenance_is_unknown_even_when_both_sides_are_missing():
    api = _api(
        "ExperimentIdentity", "ExecutionIdentity", "ExperimentProvenance",
        "check_identity_consistency",
    )
    experiment = api.ExperimentIdentity("lab-a", "experiment-x")
    first = api.ExperimentProvenance(api.ExecutionIdentity(experiment, "a"))
    second = api.ExperimentProvenance(api.ExecutionIdentity(experiment, "b"))
    for name in (*DEFINITION_FIELDS, "realized_seed", "environment_ref"):
        assert getattr(first, name) is None
        assert getattr(second, name) is None
    before = deepcopy((first, second))
    result = api.check_identity_consistency(first, second)
    assert result.status == "unknown"
    assert result.reason_codes == tuple(f"missing_{name}" for name in DEFINITION_FIELDS)
    assert api.check_identity_consistency(first, second) == result
    assert (first, second) == before
    # Equal seeds and equal dataset locations cannot establish split identity.
    complete = _provenance(api)
    missing_dataset = replace(complete, dataset_version=None, location="datasets/v1")
    assert missing_dataset.dataset_version is None
    result = api.check_identity_consistency(missing_dataset, missing_dataset)
    assert result.status == "unknown"
    assert result.reason_codes == ("missing_dataset_version",)
    incomplete = replace(complete, split_id=None, location="data/shared")
    for left in (complete, incomplete):
        result = api.check_identity_consistency(left, incomplete)
        assert result.status == "unknown"
        assert result.reason_codes == ("missing_split_id",)


def test_same_declared_identity_preserves_known_provenance_conflicts():
    api = _api(
        "ExperimentIdentity", "ExecutionIdentity", "ExperimentProvenance",
        "check_identity_consistency",
    )
    first = _provenance(api)
    for name in DEFINITION_FIELDS:
        second = replace(_provenance(api, "execution-b"), **{name: "different"})
        assert first.execution.experiment == second.execution.experiment
        before = deepcopy((first, second))
        result = api.check_identity_consistency(first, second)
        assert result.status == "conflict"
        assert result.reason_codes == (f"{name}_mismatch",)
        assert api.check_identity_consistency(second, first) == result
        assert (first, second) == before
    mixed = replace(first, configuration_ref=None, dataset_version="dataset:v2")
    result = api.check_identity_consistency(first, mixed)
    assert result.status == "conflict"
    assert result.reason_codes == ("missing_configuration_ref", "dataset_version_mismatch")


def test_identity_namespaces_isolate_equal_experiment_and_execution_labels():
    api = _api("ExperimentIdentity", "ExecutionIdentity")
    first = api.ExperimentIdentity("lab-a", "experiment-x")
    second = api.ExperimentIdentity("lab-b", "experiment-x")
    assert first != second
    assert api.ExecutionIdentity(first, "execution-1") != api.ExecutionIdentity(
        second, "execution-1",
    )
    assert first == api.ExperimentIdentity("lab-a", "experiment-x")

"""M11 RED contract for declared metric-result comparability, policy v1.

experiment_identity.MetricResultProvenance is a frozen dataclass with metric_name,
value, provenance (ExperimentProvenance), and optional result declarations.
The experiment_comparability.compare_metric_results entry returns a value model
with status, reason_codes, and policy_version='metric-result-v1'.

REQUIRED_FIELDS are checked in order. Known conflicts take precedence over
unknown information; reasons retain both, using <field>_mismatch and
missing_<field>. Config/model/seed may vary between candidates. Matching required
declarations is not proof of truth, significance, causality, identity, or reuse.
Artifact references are explicit: neither metric names nor epochs allocate them.
"""

from copy import deepcopy
from dataclasses import FrozenInstanceError, fields, replace
import importlib
import importlib.util

import pytest

from copilot import CopilotRunMetadata
from copilot.evaluation import EvaluationCase


PROVENANCE_FIELDS = ("task", "target", "dataset_version", "split_id")
REQUIRED_FIELDS = (
    *PROVENANCE_FIELDS, "metric_definition", "direction", "aggregation",
    "evaluation_protocol", "selection_protocol",
)


def _module(name, *names):
    assert importlib.util.find_spec(name) is not None, (
        f"missing M11 {name} capability: {', '.join(names)}"
    )
    module = importlib.import_module(name)
    missing = [item for item in names if not callable(getattr(module, item, None))]
    assert not missing, f"missing M11 public API: {missing}"
    return module


def _apis():
    comparator = _module("experiment_comparability", "compare_metric_results")
    identity = _module(
        "experiment_identity", "ExperimentIdentity", "ExecutionIdentity",
        "ExperimentProvenance", "MetricResultProvenance",
    )
    return identity, comparator


def _record(api, experiment_id="experiment-a", execution_id="execution-a", seed=11):
    provenance = api.ExperimentProvenance(
        execution=api.ExecutionIdentity(
            api.ExperimentIdentity("lab-a", experiment_id), execution_id,
        ),
        configuration_ref="config:v1", task="regression", target="target-a",
        model_ref="architecture:v1", dataset_version="dataset:v1",
        split_id="split:v1", source_revision="repository-a:revision-1",
        initialization_checkpoint_ref="initial-weights:v1",
        realized_seed=seed, environment_ref="environment:v1",
    )
    return api.MetricResultProvenance(
        metric_name="r2", value=0.82, provenance=provenance,
        metric_definition="r2:definition-v1", direction="maximize",
        aggregation="macro:definition-v1", evaluation_protocol="evaluation:v1",
        selection_protocol="best-validation-per-metric:v1",
        best_epoch=3,
    )


def _change(record, name, value):
    if name in PROVENANCE_FIELDS:
        return replace(record, provenance=replace(record.provenance, **{name: value}))
    return replace(record, **{name: value})


def _assert_result(result, status, reasons):
    assert result.status == status
    assert result.reason_codes == reasons
    assert result.policy_version == "metric-result-v1"


def test_complete_declared_metric_results_are_comparable_deterministically():
    api, comparator = _apis()
    first = _record(api)
    second = replace(_record(api, "experiment-b", "execution-b"), value=0.91)
    before = deepcopy((first, second))
    result = comparator.compare_metric_results(first, second)
    _assert_result(result, "comparable", ())
    assert comparator.compare_metric_results(first, second) == result
    assert comparator.compare_metric_results(second, first) == result
    assert (first, second) == before
    # These records remain distinct; comparable is not experiment equality.
    assert first.provenance.execution.experiment != second.provenance.execution.experiment


def test_known_required_conflicts_are_incompatible_with_stable_reasons():
    api, comparator = _apis()
    first = _record(api)
    for name in REQUIRED_FIELDS:
        value = "minimize" if name == "direction" else "different"
        second = _change(_record(api, "experiment-b", "execution-b"), name, value)
        before = deepcopy((first, second))
        result = comparator.compare_metric_results(first, second)
        _assert_result(result, "incompatible", (f"{name}_mismatch",))
        assert comparator.compare_metric_results(second, first) == result
        assert (first, second) == before
    # A definite conflict cannot be downgraded to unknown by another absent field.
    mixed = _change(_change(first, "dataset_version", "dataset:v2"), "split_id", None)
    _assert_result(
        comparator.compare_metric_results(first, mixed), "incompatible",
        ("dataset_version_mismatch", "missing_split_id"),
    )


def test_missing_required_dimensions_are_unknown_on_one_or_both_sides():
    api, comparator = _apis()
    first = _record(api)
    for name in REQUIRED_FIELDS:
        missing = _change(first, name, None)
        for left in (first, missing):
            result = comparator.compare_metric_results(left, missing)
            _assert_result(result, "unknown", (f"missing_{name}",))
            assert comparator.compare_metric_results(missing, left) == result
    # Even the very same explicit experiment/execution and seed cannot fill a split.
    missing_split = _change(first, "split_id", None)
    assert missing_split.provenance.execution == first.provenance.execution
    assert missing_split.provenance.realized_seed == first.provenance.realized_seed
    _assert_result(
        comparator.compare_metric_results(missing_split, missing_split),
        "unknown", ("missing_split_id",),
    )


def test_policy_allows_multiple_seeds_and_explicit_model_config_variants():
    api, comparator = _apis()
    first = _record(api)
    repeated = _record(api, execution_id="execution-b", seed=29)
    assert first.provenance.execution.experiment == repeated.provenance.execution.experiment
    assert first.provenance.execution != repeated.provenance.execution
    _assert_result(comparator.compare_metric_results(first, repeated), "comparable", ())
    candidate = _record(api, "caller-declared-model-variant", "execution-c", seed=47)
    candidate = replace(candidate, provenance=replace(
        candidate.provenance, model_ref="architecture:v2", configuration_ref="config:v2",
    ))
    _assert_result(comparator.compare_metric_results(first, candidate), "comparable", ())
    assert (first.provenance.realized_seed, repeated.provenance.realized_seed) == (11, 29)


def test_metric_best_epochs_preserve_separate_explicit_artifact_provenance():
    api, comparator = _apis()
    r2 = _record(api)
    racc = replace(r2, metric_name="racc", metric_definition="racc:definition-v1",
                   value=0.91, best_epoch=7)
    assert r2.provenance.execution == racc.provenance.execution
    assert (r2.best_epoch, racc.best_epoch) == (3, 7)
    for result in (r2, racc):
        assert result.checkpoint_ref is None
        assert result.result_artifact_ref is None
        assert result.history_ref is None
    before = deepcopy((r2, racc))
    _assert_result(comparator.compare_metric_results(r2, racc), "incompatible",
                   ("metric_definition_mismatch",))
    assert (r2, racc) == before
    with pytest.raises(FrozenInstanceError):
        r2.checkpoint_ref = "inferred-from-epoch-3"
    declared_r2 = replace(r2, checkpoint_ref="checkpoint:3",
                          result_artifact_ref="result:r2", history_ref="history:v1")
    declared_racc = replace(racc, checkpoint_ref="checkpoint:7",
                            result_artifact_ref="result:racc", history_ref="history:v1")
    assert declared_r2.checkpoint_ref != declared_racc.checkpoint_ref
    # Equal epochs still imply no checkpoint identity without an explicit reference.
    equal_epoch = replace(racc, best_epoch=3)
    assert r2.checkpoint_ref is equal_epoch.checkpoint_ref is None
    shared_r2 = replace(r2, checkpoint_ref="caller-declared-shared-checkpoint")
    shared_racc = replace(equal_epoch, checkpoint_ref="caller-declared-shared-checkpoint")
    assert shared_r2.checkpoint_ref == shared_racc.checkpoint_ref


def test_ml_identity_does_not_accept_copilot_or_evaluation_identity_substitutes():
    api, _ = _apis()
    run = CopilotRunMetadata(run_id="copilot-run-a")
    case = EvaluationCase("evaluation-case-a", "1", "task", {}, {})
    identity = api.ExperimentIdentity("lab-a", "explicit-ml-experiment")
    assert tuple(field.name for field in fields(identity)) == ("namespace", "experiment_id")
    for substitute in ({"run_id": run.run_id}, {"case_id": case.case_id}):
        with pytest.raises(TypeError):
            api.ExperimentIdentity(namespace="lab-a", **substitute)
        with pytest.raises(TypeError):
            api.ExecutionIdentity(experiment=identity, **substitute)
    assert identity == api.ExperimentIdentity("lab-a", "explicit-ml-experiment")
    assert (run.run_id, case.case_id) == ("copilot-run-a", "evaluation-case-a")

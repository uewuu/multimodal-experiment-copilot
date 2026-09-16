"""M11 Slice 3A RED contracts for opt-in, in-memory experiment adaptation.

Future experiment_record exports ExperimentRecord and
build_experiment_record(parsed_experiment, *, provenance, metric_declarations=None).
The input is one existing analyze_experiment_dirs successful_experiments entry:
experiment_name, experiment_dir, summary. Provenance is an explicitly supplied
ExperimentProvenance, containing ExecutionIdentity and ExperimentIdentity.
No redundant identity arguments or implicit identity conversions are needed.

The record exposes experiment_name, experiment_dir, summary, provenance, and
metric_results (a name-to-MetricResultProvenance mapping). Each result takes its
value/best_epoch from that metric's existing best_value/best_epoch. Optional
metric_declarations maps metric names to explicit foundation declaration fields
(metric_definition, direction, aggregation, evaluation_protocol,
selection_protocol, checkpoint_ref, result_artifact_ref, history_ref).
Omitted declarations remain None. Existing summaries are not enriched in place.

Consumers call the frozen foundation comparators on these values directly;
the adapter introduces no comparison policy, parsing, ranking, or persistence.
"""

from copy import deepcopy
from dataclasses import replace
import importlib
import importlib.util
from pathlib import Path

import pytest

import compare_experiments
import metrics
import summarize_experiment
from experiment_comparability import compare_metric_results
from experiment_identity import (
    ExecutionIdentity,
    ExperimentIdentity,
    ExperimentProvenance,
    MetricResultProvenance,
    check_identity_consistency,
)


def _api():
    assert importlib.util.find_spec("experiment_record") is not None, (
        "missing M11 experiment-record integration capability: experiment_record"
    )
    module = importlib.import_module("experiment_record")
    for name in ("ExperimentRecord", "build_experiment_record"):
        assert callable(getattr(module, name, None)), (
            f"missing M11 experiment-record integration capability: {name}"
        )
    return module


@pytest.fixture
def parsed_experiment(monkeypatch):
    """Exercise real summary/extraction code; substitute only file readers."""
    config = {
        "batch_size": 8, "seed": 42, "sample_seed": 42, "n_epochs": 8,
        "feature_list": ["vision", "audio"],
    }
    history = {"valid": {"app": {
        "r2": [[1, 0.70], [3, 0.82], [7, 0.78]],
        "racc": [[1, 0.80], [3, 0.85], [7, 0.91]],
    }}}
    monkeypatch.setattr(summarize_experiment, "read_config", lambda path: config)
    monkeypatch.setattr(summarize_experiment, "read_history", lambda path: history)
    batch = compare_experiments.analyze_experiment_dirs([
        Path("outputs/dataset-v99/split-test/20260916/checkpoint-epoch-3.pt"),
    ])
    assert batch["failed_experiments"] == []
    assert len(batch["successful_experiments"]) == 1
    return batch["successful_experiments"][0]


def _provenance(execution_id="execution-a", seed=42):
    return ExperimentProvenance(
        execution=ExecutionIdentity(
            ExperimentIdentity("lab-a", "explicit-experiment"), execution_id,
        ),
        configuration_ref="config:v1", task="regression", target="target-a",
        model_ref="model:v1", dataset_version="dataset:v1", split_id="split:v1",
        source_revision="repository:revision-1",
        initialization_checkpoint_ref="initial-weights:v1", realized_seed=seed,
    )


def _declarations():
    return {name: {
        "metric_definition": f"{name}:definition-v1", "direction": "maximize",
        "aggregation": "macro:v1", "evaluation_protocol": "evaluation:v1",
        "selection_protocol": "best-validation-per-metric:v1",
    } for name in ("r2", "racc")}


def test_record_adapts_existing_summary_with_explicit_identity_without_mutation(
    parsed_experiment,
):
    api = _api()
    provenance, declarations = _provenance(), _declarations()
    before = deepcopy((parsed_experiment, provenance, declarations))
    record = api.build_experiment_record(
        parsed_experiment, provenance=provenance, metric_declarations=declarations,
    )
    assert isinstance(record, api.ExperimentRecord)
    assert record.experiment_name == parsed_experiment["experiment_name"]
    assert record.experiment_dir == parsed_experiment["experiment_dir"]
    assert record.summary == parsed_experiment["summary"]
    assert record.provenance == provenance
    assert record.provenance.execution == ExecutionIdentity(
        ExperimentIdentity("lab-a", "explicit-experiment"), "execution-a",
    )
    assert set(record.metric_results) == {"r2", "racc"}
    for name, result in record.metric_results.items():
        source = parsed_experiment["summary"]["validation_metrics"][name]
        assert isinstance(result, MetricResultProvenance)
        assert (result.metric_name, result.value, result.best_epoch) == (
            name, source["best_value"], source["best_epoch"],
        )
        assert result.provenance == provenance
    assert (parsed_experiment, provenance, declarations) == before
    with pytest.raises(TypeError):
        api.build_experiment_record(parsed_experiment)
    for substitute in (
        {"run_id": "copilot-run-a"}, {"case_id": "evaluation-case-a"},
        {"execution_id": parsed_experiment["experiment_dir"]},
    ):
        with pytest.raises(TypeError):
            api.build_experiment_record(parsed_experiment, **substitute)


def test_record_keeps_missing_provenance_unknown_without_path_discovery(
    parsed_experiment, monkeypatch,
):
    api = _api()
    provenance = ExperimentProvenance(_provenance().execution)
    before = deepcopy(parsed_experiment)

    def forbidden(*args, **kwargs):
        pytest.fail("record adaptation must not parse or discover filesystem provenance")

    with monkeypatch.context() as guard:
        guard.setattr("builtins.open", forbidden)
        for name in ("open", "read_text", "read_bytes", "iterdir", "glob", "rglob"):
            guard.setattr(Path, name, forbidden)
        for name in ("read_config", "read_history", "build_experiment_summary"):
            guard.setattr(summarize_experiment, name, forbidden)
        for name in ("build_experiment_summary", "analyze_experiment_dirs", "find_experiment_dirs"):
            guard.setattr(compare_experiments, name, forbidden)
        record = api.build_experiment_record(parsed_experiment, provenance=provenance)

    assert record.provenance == provenance
    for name in (
        "configuration_ref", "task", "target", "model_ref", "dataset_version",
        "split_id", "source_revision", "initialization_checkpoint_ref",
        "realized_seed", "environment_ref", "location", "recorded_at",
    ):
        assert getattr(record.provenance, name) is None
    for result in record.metric_results.values():
        for name in (
            "metric_definition", "direction", "aggregation", "evaluation_protocol",
            "selection_protocol", "checkpoint_ref", "result_artifact_ref", "history_ref",
        ):
            assert getattr(result, name) is None
    assert check_identity_consistency(record.provenance, record.provenance).status == "unknown"
    assert compare_metric_results(record.metric_results["r2"], record.metric_results["r2"]).status == "unknown"
    assert parsed_experiment == before


def test_record_preserves_metric_specific_epochs_and_explicit_artifacts(parsed_experiment):
    api = _api()
    provenance = _provenance()
    plain = api.build_experiment_record(parsed_experiment, provenance=provenance)
    assert (plain.metric_results["r2"].value, plain.metric_results["r2"].best_epoch) == (0.82, 3)
    assert (plain.metric_results["racc"].value, plain.metric_results["racc"].best_epoch) == (0.91, 7)
    for result in plain.metric_results.values():
        assert (result.checkpoint_ref, result.result_artifact_ref, result.history_ref) == (None, None, None)

    declarations = _declarations()
    for name, epoch in (("r2", 3), ("racc", 7)):
        declarations[name].update(
            checkpoint_ref=f"checkpoint:{epoch}",
            result_artifact_ref=f"result:{name}", history_ref=f"history:{name}",
        )
    before = deepcopy((parsed_experiment, declarations))
    declared = api.build_experiment_record(
        parsed_experiment, provenance=provenance, metric_declarations=declarations,
    )
    for name in ("r2", "racc"):
        result = declared.metric_results[name]
        for field, value in declarations[name].items():
            assert getattr(result, field) == value
        assert (result.value, result.best_epoch) == (
            plain.metric_results[name].value, plain.metric_results[name].best_epoch,
        )
    assert (parsed_experiment, declarations) == before

    equal_epoch = deepcopy(parsed_experiment)
    equal_epoch["summary"]["validation_metrics"]["racc"]["best_epoch"] = 3
    unknown = api.build_experiment_record(equal_epoch, provenance=provenance)
    assert unknown.metric_results["r2"].checkpoint_ref is None
    assert unknown.metric_results["racc"].checkpoint_ref is None
    shared = {name: {"checkpoint_ref": "explicit-shared-checkpoint"} for name in ("r2", "racc")}
    explicit = api.build_experiment_record(
        equal_epoch, provenance=provenance, metric_declarations=shared,
    )
    assert explicit.metric_results["r2"].checkpoint_ref == "explicit-shared-checkpoint"
    assert explicit.metric_results["racc"].checkpoint_ref == "explicit-shared-checkpoint"


def test_record_preserves_repeated_executions_and_seeds_under_one_experiment(parsed_experiment):
    api = _api()
    second_summary = deepcopy(parsed_experiment)
    second_summary["experiment_name"] = "renamed-experiment"
    second_summary["experiment_dir"] = "copied/20260917"
    second_summary["summary"]["configuration"]["seed"] = 108
    before = deepcopy((parsed_experiment, second_summary))
    first = api.build_experiment_record(parsed_experiment, provenance=_provenance("a", 42))
    second = api.build_experiment_record(second_summary, provenance=_provenance("b", 108))
    assert first.provenance.execution.experiment == second.provenance.execution.experiment
    assert first.provenance.execution != second.provenance.execution
    assert (first.provenance.execution.execution_id, second.provenance.execution.execution_id) == ("a", "b")
    assert (first.provenance.realized_seed, second.provenance.realized_seed) == (42, 108)
    assessment = check_identity_consistency(first.provenance, second.provenance)
    assert (assessment.status, assessment.reason_codes) == ("consistent", ())
    assert second.metric_results["r2"].provenance == second.provenance
    assert (parsed_experiment, second_summary) == before


def test_record_metric_results_use_frozen_three_state_comparability(parsed_experiment):
    api = _api()
    provenance, declarations = _provenance(), _declarations()
    first = api.build_experiment_record(
        parsed_experiment, provenance=provenance, metric_declarations=declarations,
    )
    cases = (
        (replace(provenance, realized_seed=108), "comparable", ()),
        (replace(provenance, dataset_version="dataset:v2"), "incompatible", ("dataset_version_mismatch",)),
        (replace(provenance, split_id=None), "unknown", ("missing_split_id",)),
        (replace(provenance, dataset_version="dataset:v2", split_id=None), "incompatible",
         ("dataset_version_mismatch", "missing_split_id")),
    )
    before = deepcopy((parsed_experiment, provenance, declarations))
    for supplied, status, reasons in cases:
        other = api.build_experiment_record(
            parsed_experiment, provenance=supplied, metric_declarations=declarations,
        )
        left, right = first.metric_results["r2"], other.metric_results["r2"]
        assessment = compare_metric_results(left, right)
        assert (assessment.status, assessment.reason_codes) == (status, reasons)
        assert assessment.policy_version == "metric-result-v1"
        assert compare_metric_results(right, left) == assessment
        assert compare_metric_results(left, right) == assessment
        if supplied.split_id is None and supplied.dataset_version == provenance.dataset_version:
            assert compare_metric_results(right, right).status == "unknown"
        if supplied.dataset_version != provenance.dataset_version:
            assert check_identity_consistency(first.provenance, other.provenance).status == "conflict"
    assert (parsed_experiment, provenance, declarations) == before


def test_record_adaptation_leaves_existing_summary_comparison_and_metrics_unchanged(
    parsed_experiment,
):
    second = deepcopy(parsed_experiment)
    second["experiment_name"] = "second"
    second["experiment_dir"] = "outputs/second"
    second["summary"]["validation_metrics"]["r2"]["best_value"] = 0.90
    second["summary"]["validation_metrics"]["racc"]["best_value"] = 0.88
    batch = {"successful_experiments": [parsed_experiment, second], "failed_experiments": []}
    records_before = compare_experiments.build_comparison_records(batch)
    payload_before = compare_experiments.build_comparison_payload(batch)
    racc_before = compare_experiments.build_comparison_payload(batch, sort_by="best_racc")
    history = [[1, 0.70], [3, 0.82], [7, 0.78]]
    metric_before = metrics.evaluate_metric_history(history, "maximize")
    assert set(parsed_experiment["summary"]) == {"configuration", "validation_metrics"}
    assert set(records_before[0]) == {
        "experiment_name", "experiment_dir", "best_r2", "best_r2_epoch", "best_racc", "best_racc_epoch",
    }
    assert (records_before[0]["best_r2_epoch"], records_before[0]["best_racc_epoch"]) == (3, 7)
    assert payload_before["comparison_records"][0]["experiment_name"] == "second"
    assert racc_before["comparison_records"][0]["experiment_name"] == parsed_experiment["experiment_name"]
    before = deepcopy((batch, history))

    api = _api()
    for index, source in enumerate(batch["successful_experiments"]):
        record = api.build_experiment_record(source, provenance=_provenance(f"execution-{index}"))
        assert record.summary == source["summary"]
    assert (batch, history) == before
    assert compare_experiments.build_comparison_records(batch) == records_before
    assert compare_experiments.build_comparison_payload(batch) == payload_before
    assert compare_experiments.build_comparison_payload(batch, sort_by="best_racc") == racc_before
    assert metrics.evaluate_metric_history(history, "maximize") == metric_before
    assert summarize_experiment.build_experiment_summary(
        config_path=Path("memory/hparams.yaml"), history_path=Path("memory/history.json"),
    ) == parsed_experiment["summary"]

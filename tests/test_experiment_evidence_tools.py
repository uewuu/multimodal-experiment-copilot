"""M16 RED composition API in tool_layer.experiment_tools.

analyze_experiment_evidence(experiment_dir, *, config_paths=(),
test_metrics_file=None, test_metric_specs=(), metrics_config=None,
include_diagnostics=False) -> dict. New sections: configuration_evidence,
validation_evidence (kind validation_history, source.file history.json),
test_results. Legacy fields remain equal to analyze_experiment's output.
Host file inputs are relative, contained in the selected experiment. Optional
CSV data errors preserve validation; authorization/configuration errors raise.
Test and MetricSpec selection must be specified together. No global defaults.
"""

import importlib
import inspect
import json
from pathlib import Path

import pytest
import yaml

from tool_layer.experiment_tools import analyze_experiment
from test_test_metrics_csv import spec, assert_provenance, OpenProbe


def api(name="analyze_experiment_evidence"):
    value = getattr(importlib.import_module("tool_layer.experiment_tools"), name, None)
    assert callable(value), f"missing M16 capability: tool_layer.experiment_tools.{name}"
    return value


def experiment(tmp_path, name="experiment", *, csv="score\n0.7\n"):
    root = tmp_path / name
    root.mkdir()
    (root / "hparams.yaml").write_text(yaml.safe_dump({"config": {
        "batch_size": 4, "n_epochs": 100, "optimizer": {"base_lr": 0.002},
        "component_enabled": False, "private": "UNSELECTED_SECRET"}}), encoding="utf-8")
    (root / "history.json").write_text(json.dumps({"valid": {"app": {
        "r2": [[0, 0.1], [2, 0.5], [3, 0.4]],
        "racc": [[0, 0.8], [2, 0.9], [3, 0.85]]}}}), encoding="utf-8")
    if csv is not None:
        (root / "heldout.csv").write_text(csv, encoding="utf-8")
    return root


def test_legacy_signature_and_output_remain_unchanged(tmp_path):
    assert str(inspect.signature(analyze_experiment)) == (
        "(experiment_dir: str, *, metrics_config: str | None = None, "
        "include_diagnostics: bool = False) -> dict")
    root = experiment(tmp_path)
    result = analyze_experiment(str(root), include_diagnostics=True)
    assert list(result) == ["configuration", "validation_metrics", "diagnostics"]
    assert "component_enabled" not in result["configuration"]
    assert "test_results" not in result


def test_new_sections_preserve_legacy_summary_and_separate_evidence(tmp_path):
    analyze = api()
    root = experiment(tmp_path)
    before = {p.name: p.read_bytes() for p in root.iterdir()}
    legacy = analyze_experiment(str(root), include_diagnostics=True)
    result = analyze(str(root), config_paths=[("component_enabled",)],
                     test_metrics_file="heldout.csv", test_metric_specs=[spec()],
                     include_diagnostics=True)
    assert {key: result[key] for key in legacy} == legacy
    assert result["configuration_evidence"]["fields"][0]["value"] is False
    assert result["validation_evidence"]["evidence_kind"] == "validation_history"
    assert result["validation_evidence"]["source"]["file"] == "history.json"
    assert result["test_results"]["metrics"]["score"]["value"] == 0.7
    assert result["validation_metrics"]["r2"]["best_value"] == 0.5
    assert_provenance(result["test_results"])
    assert "UNSELECTED_SECRET" not in json.dumps(result)
    assert {p.name: p.read_bytes() for p in root.iterdir()} == before


def test_default_never_reads_test_or_checkpoint(tmp_path, monkeypatch):
    analyze = api()
    root = experiment(tmp_path, csv="INVALID CSV")
    checkpoint = root / "checkpoint_valid_racc.ckpt"
    checkpoint.write_bytes(b"NOT_A_PICKLE")
    forbidden = {(root / "heldout.csv").resolve(), checkpoint.resolve()}
    def guarded(target):
        assert target not in forbidden, "unexpected artifact read"
    probe = OpenProbe(monkeypatch, before=guarded)
    result = analyze(str(root))
    assert not any(target in forbidden for target, _, _ in probe.attempts)
    assert result["test_results"]["status"] == "not_requested"
    assert result["test_results"]["metrics"] == {}
    assert result["configuration_evidence"]["fields"] == []


@pytest.mark.parametrize("csv,status", [(None, "missing"), ("score\nNaN\n", "invalid")])
def test_csv_data_failure_preserves_validation(tmp_path, csv, status):
    analyze = api()
    root = experiment(tmp_path, csv=csv)
    result = analyze(str(root), test_metrics_file="heldout.csv", test_metric_specs=[spec()])
    assert result["test_results"]["status"] == status
    assert result["test_results"]["metrics"] == {}
    assert result["validation_metrics"] == analyze_experiment(str(root))["validation_metrics"]
    assert_provenance(result["test_results"])


@pytest.mark.parametrize("options", [
    {"test_metrics_file": "heldout.csv"}, {"test_metric_specs": [spec()]},
    {"test_metrics_file": ""}, {"test_metrics_file": 12},
    {"include_diagnostics": "true"}, {"config_paths": [("*",)]},
    {"unexpected_option": True},
], ids=["file-without-specs", "specs-without-file", "blank-file", "wrong-file-type",
        "wrong-diagnostics-type", "wildcard", "unknown-option"])
def test_invalid_host_options_are_not_csv_missing(tmp_path, options):
    analyze = api()
    with pytest.raises((TypeError, ValueError)):
        analyze(str(experiment(tmp_path)), **options)


@pytest.mark.parametrize("target", ["../outside.csv", "../missing.csv", "C:\\outside.csv",
                                     "C:outside.csv", "\\\\server\\share\\data.csv"],
                         ids=["traversal-existing", "traversal-missing", "absolute",
                              "drive-relative", "unc"])
def test_csv_paths_cannot_escape_or_select_absolute_host_files(tmp_path, target):
    analyze = api()
    (tmp_path / "outside.csv").write_text("score\n99\n", encoding="utf-8")
    with pytest.raises(ValueError):
        analyze(str(experiment(tmp_path)), test_metrics_file=target, test_metric_specs=[spec()])


def test_nested_custom_csv_name_preserves_relative_source(tmp_path):
    analyze = api()
    root = experiment(tmp_path, csv=None)
    (root / "reports").mkdir()
    (root / "reports" / "evaluation.csv").write_text("score\n0.4\n", encoding="utf-8")
    result = analyze(str(root), test_metrics_file="reports/evaluation.csv", test_metric_specs=[spec()])
    assert result["test_results"]["source"]["file"] == "reports/evaluation.csv"


def test_dynamic_validation_and_test_metric_specs_are_independent(tmp_path):
    analyze = api()
    root = experiment(tmp_path)
    (root / "history.json").write_text(json.dumps({"validation": {"loss": [[0, 2], [1, 1]]}}))
    (root / "metrics.yaml").write_text(yaml.safe_dump({"metrics": [{"name": "loss",
        "path": ["validation", "loss"], "direction": "minimize", "display_name": "Loss"}]}))
    result = analyze(str(root), metrics_config="metrics.yaml", test_metrics_file="heldout.csv",
                     test_metric_specs=[spec("test_loss", "score", "minimize")])
    assert result["validation_metrics"]["loss"]["best_value"] == 1
    assert result["test_results"]["metrics"]["test_loss"]["value"] == 0.7
    assert "best_epoch" not in result["test_results"]["metrics"]["test_loss"]


def test_unexpected_programmer_exception_is_not_converted_to_success(tmp_path, monkeypatch):
    analyze = api()
    root = experiment(tmp_path)
    error = RuntimeError("programmer sentinel")
    def fail(*args, **kwargs):
        raise error
    monkeypatch.setattr("tool_layer.experiment_tools.analyze_experiment", fail)
    with pytest.raises(RuntimeError) as caught:
        analyze(str(root))
    assert caught.value is error

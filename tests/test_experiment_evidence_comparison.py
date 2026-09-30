"""M17 Contract RED: pure, bounded, explicit pairwise comparison (no GREEN)."""

import builtins
from copy import deepcopy
from dataclasses import replace
import importlib
import io
import json
from pathlib import Path

import pytest

import experiment_comparability as m11
from experiment_identity import MetricResultProvenance
from tool_layer.experiment_tools import analyze_experiment_evidence
from m17_contract_helpers import (
    UNKNOWN, assert_blocked, assert_no_winner, config_field, core_api,
    evidence, options, spec, target, write_experiment,
)


def test_fixture_matches_real_m16_without_extending_its_schema(tmp_path):
    actual = analyze_experiment_evidence(str(write_experiment(tmp_path, "a")), **options())
    synthetic = evidence()
    for key in ("test_results", "validation_evidence", "validation_metrics"):
        assert actual[key] == synthetic[key]
    assert actual["configuration_evidence"]["fields"] == [config_field(False)]
    assert "metric_specs" not in actual["validation_evidence"]


def test_target_fixture_is_a_valid_m11_pair_with_distinct_experiment_identities():
    selected = target()
    def declared(side, value):
        data = selected[side]
        return MetricResultProvenance(
            metric_name=data["metric_name"], value=value, provenance=data["provenance"],
            metric_definition=data["metric_definition"], direction=data["metric_spec"].direction,
            aggregation=data["aggregation"], evaluation_protocol=data["evaluation_protocol"],
            selection_protocol=data["selection_protocol"])
    left, right = declared("a", 0.5), declared("b", 0.75)
    assert left.provenance.execution.experiment != right.provenance.execution.experiment
    assert m11.compare_metric_results(left, right).status == "comparable"
    assert m11.compare_metric_results(left, replace(right, evaluation_protocol=None)).reason_codes == (
        "missing_evaluation_protocol",)
    assert m11.compare_metric_results(left, replace(right, direction="minimize")).reason_codes == (
        "direction_mismatch",)


def test_comparable_pair_calls_real_m11_and_uses_evidence_values(monkeypatch):
    compare = core_api()
    calls = []
    original = m11.compare_metric_results
    def traced(left, right):
        calls.append((left, right))
        return original(left, right)
    monkeypatch.setattr(m11, "compare_metric_results", traced)
    module = importlib.import_module("experiment_comparison")
    if getattr(module, "compare_metric_results", None) is original:
        monkeypatch.setattr(module, "compare_metric_results", traced)
    result = compare(evidence(0.5), evidence(0.75), targets=[target()])
    row = result["targets"]["quality"]
    assert result["schema_version"] == "experiment-comparison-v1"
    assert "pair" in result and "configuration_differences" in result
    assert row["comparability"] == {"status": "comparable", "reason_codes": [],
                                    "policy_version": "metric-result-v1"}
    assert row["alignment"]["status"] == "aligned"
    assert row["calculation"]["raw_delta"] == 0.25
    assert len(calls) == 1
    left, right = calls[0]
    assert isinstance(left, MetricResultProvenance) and isinstance(right, MetricResultProvenance)
    assert (left.value, right.value) == (0.5, 0.75)
    assert left.best_epoch is None and right.best_epoch is None
    assert left.checkpoint_ref is None and right.checkpoint_ref is None
    assert_no_winner(result)


@pytest.mark.parametrize("field", ["task", "target", "dataset_version", "split_id",
    "metric_definition", "direction", "aggregation", "evaluation_protocol", "selection_protocol"])
def test_known_m11_conflicts_preserve_existing_reasons_and_block(field):
    compare = core_api()
    selection = target()
    b = evidence(0.75)
    if field in ("task", "target", "dataset_version", "split_id"):
        selection["b"]["provenance"] = replace(selection["b"]["provenance"], **{field: "other"})
    elif field == "direction":
        selection["b"]["metric_spec"] = spec(direction="minimize")
        b["test_results"]["metrics"]["score"]["direction"] = "minimize"
    else:
        selection["b"][field] = "other"
    row = compare(evidence(), b, targets=[selection])["targets"]["quality"]
    assert row["comparability"] == {"status": "incompatible",
        "reason_codes": [f"{field}_mismatch"], "policy_version": "metric-result-v1"}
    assert_blocked(row)


@pytest.mark.parametrize("field", ["task", "target", "dataset_version", "split_id",
                                     "evaluation_protocol", "selection_protocol"])
def test_aligned_unknown_allows_only_limited_numerical_observation(field):
    compare = core_api()
    selection = target()
    for side in ("a", "b"):
        if field in ("task", "target", "dataset_version", "split_id"):
            selection[side]["provenance"] = replace(selection[side]["provenance"], **{field: None})
        else:
            selection[side][field] = None
    row = compare(evidence(), evidence(0.75), targets=[selection])["targets"]["quality"]
    assert row["comparability"] == {"status": "unknown", "reason_codes": [f"missing_{field}"],
                                    "policy_version": "metric-result-v1"}
    assert row["calculation"]["raw_delta"] == 0.25
    assert f"missing_{field}" in row["limitations"]


def test_conflict_precedence_does_not_erase_missing_reasons():
    compare = core_api()
    selected = target()
    selected["b"]["provenance"] = replace(selected["b"]["provenance"], task=None, split_id="other")
    row = compare(evidence(), evidence(), targets=[selected])["targets"]["quality"]
    assert row["comparability"]["status"] == "incompatible"
    assert row["comparability"]["reason_codes"] == ["missing_task", "split_id_mismatch"]
    assert_blocked(row)


@pytest.mark.parametrize("field", ["metric_spec", "metric_definition", "aggregation"])
def test_incomplete_metric_semantics_never_inferred_from_name(field):
    compare = core_api()
    selected = target()
    selected["b"][field] = None
    row = compare(evidence(), evidence(0.75), targets=[selected])["targets"]["quality"]
    assert row["value_b"] == 0.75
    assert row["alignment"]["status"] == "blocked"
    assert row["alignment"]["reason_codes"]
    assert_blocked(row)


def test_different_local_names_and_paths_need_explicit_alignment_not_display_labels():
    compare = core_api()
    selected = target()
    selected["b"]["metric_name"] = "quality_b"
    selected["b"]["metric_spec"] = replace(spec("quality_b"), path=("other_column",), display_name="Other")
    b = evidence(0.75, name="quality_b")
    b["test_results"]["metrics"]["quality_b"]["source_column"] = "other_column"
    b["test_results"]["metrics"]["quality_b"]["display_name"] = "Other"
    row = compare(evidence(), b, targets=[selected])["targets"]["quality"]
    assert row["comparability"]["status"] == "comparable"
    assert row["calculation"]["raw_delta"] == 0.25


@pytest.mark.parametrize("field,value", [("source_column", "wrong_column"), ("direction", "minimize")])
def test_test_metadata_disagrees_with_bound_spec_and_cannot_be_relabelled(field, value):
    compare = core_api()
    b = evidence(0.75)
    b["test_results"]["metrics"]["score"][field] = value
    row = compare(evidence(), b, targets=[target()])["targets"]["quality"]
    assert row["alignment"]["status"] == "blocked"
    assert row["alignment"]["reason_codes"]
    assert_blocked(row)


@pytest.mark.parametrize("section", ["validation", "test"])
def test_only_explicit_targets_are_returned(section):
    compare = core_api()
    a, b = evidence(), evidence(0.75)
    a["validation_metrics"]["unused"] = deepcopy(a["validation_metrics"]["score"])
    b["validation_metrics"]["unused"] = deepcopy(b["validation_metrics"]["score"])
    result = compare(a, b, targets=[target(section=section)])
    assert list(result["targets"]) == ["quality"]
    assert result["targets"]["quality"]["calculation"]["raw_delta"] == 0.25


def test_legacy_validation_values_survive_missing_spec_but_delta_does_not():
    compare = core_api()
    selected = target(section="validation")
    selected["a"]["metric_spec"] = selected["b"]["metric_spec"] = None
    row = compare(evidence(), evidence(0.75), targets=[selected])["targets"]["quality"]
    assert (row["value_a"], row["value_b"]) == (0.5, 0.75)
    assert row["alignment"]["status"] == "blocked"
    assert_blocked(row)


@pytest.mark.parametrize("mismatch", ["section", "evidence_kind"])
def test_source_isolation_even_when_names_values_and_declared_split_match(mismatch):
    compare = core_api()
    selected, b = target(), evidence(0.75)
    if mismatch == "section":
        selected["b"]["section"] = "validation"
        selected["b"]["metric_spec"] = spec(section="validation")
    else:
        b["test_results"]["evidence_kind"] = "validation_history"
    row = compare(evidence(), b, targets=[selected])["targets"]["quality"]
    assert (row["value_a"], row["value_b"]) == (0.5, 0.75)
    assert row["alignment"]["status"] == "blocked"
    assert_blocked(row)


@pytest.mark.parametrize("direction,a,b,interpretation", [
    ("maximize", 0.5, 0.75, "b_favorable"), ("maximize", 0.75, 0.5, "a_favorable"),
    ("minimize", 0.5, 0.75, "a_favorable"), ("minimize", 0.75, 0.5, "b_favorable"),
    ("maximize", 0.5, 0.5, "equal"), ("minimize", 0.5, 0.5, "equal"),
])
def test_raw_delta_has_fixed_sign_and_separate_direction(direction, a, b, interpretation):
    compare = core_api()
    result = compare(evidence(a, direction=direction), evidence(b, direction=direction),
                     targets=[target(direction=direction)])
    assert result["targets"]["quality"]["calculation"] == {
        "status": "calculated", "raw_delta": b - a, "direction": direction,
        "direction_interpretation": interpretation,
    }
    assert_no_winner(result)


def test_precision_is_display_only():
    compare = core_api()
    selected = target()
    a, b = evidence(0.5001), evidence(0.5002)
    for side, item in (("a", a), ("b", b)):
        selected[side]["metric_spec"] = replace(spec(), precision=1)
        item["test_results"]["metrics"]["score"]["precision"] = 1
    row = compare(a, b, targets=[selected])["targets"]["quality"]
    assert row["calculation"]["raw_delta"] == 0.5002 - 0.5001
    assert row["calculation"]["raw_delta"] != 0


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf"), True, "0.5"])
def test_invalid_numeric_input_is_rejected(value):
    compare = core_api()
    b = evidence()
    b["test_results"]["metrics"]["score"]["value"] = value
    with pytest.raises((TypeError, ValueError)):
        compare(evidence(), b, targets=[target()])


def test_subtraction_overflow_blocks_instead_of_serializing_infinity():
    compare = core_api()
    row = compare(evidence(-1e308), evidence(1e308), targets=[target()])["targets"]["quality"]
    assert_blocked(row)
    assert "delta_non_finite" in row["limitations"]
    json.dumps(row, allow_nan=False)


@pytest.mark.parametrize("status", ["missing", "invalid", "unavailable", "not_requested"])
@pytest.mark.parametrize("sides", [("a",), ("b",), ("a", "b")])
def test_unavailable_evidence_retains_target_and_does_not_fabricate_m11(status, sides, monkeypatch):
    compare = core_api()
    original = m11.compare_metric_results
    def deny(*args, **kwargs):
        raise AssertionError("missing numeric evidence must not be fabricated for M11")
    monkeypatch.setattr(m11, "compare_metric_results", deny)
    module = importlib.import_module("experiment_comparison")
    if getattr(module, "compare_metric_results", None) is original:
        monkeypatch.setattr(module, "compare_metric_results", deny)
    inputs = {"a": evidence(), "b": evidence(0.75)}
    for side in sides:
        inputs[side]["test_results"]["status"] = status
        inputs[side]["test_results"]["metrics"] = {}
        if status == "not_requested":
            inputs[side]["test_results"] = {"status": status, "metrics": {}}
    result = compare(inputs["a"], inputs["b"], targets=[target()])
    assert list(result["targets"]) == ["quality"]
    row = result["targets"]["quality"]
    for side in sides:
        assert row[f"availability_{side}"] == status
        assert row[f"value_{side}"] is None
    assert row["comparability"] is None and row["not_evaluated_reason"]
    assert_blocked(row)


def test_missing_metric_no_fallback_and_other_target_keeps_own_comparability():
    compare = core_api()
    missing = target("absent")
    missing["b"]["metric_name"] = "absent"
    missing["b"]["metric_spec"] = spec("absent")
    result = compare(evidence(), evidence(), targets=[target(), missing])
    assert result["targets"]["quality"]["comparability"]["status"] == "comparable"
    row = result["targets"]["absent"]
    assert row["availability_b"] == "missing" and row["value_b"] is None
    assert row["comparability"] is None
    assert_blocked(row)
    assert_no_winner(result)


def test_csv_provenance_preserved_without_binding_to_best_validation_epoch():
    compare = core_api()
    a, b = evidence(), evidence(0.75)
    a["validation_metrics"]["score"]["best_epoch"] = 71
    row = compare(a, b, targets=[target()])["targets"]["quality"]
    for side in ("a", "b"):
        source = row[f"source_{side}"]
        assert source["evidence_kind"] == "csv_reported"
        assert source["provenance"] == UNKNOWN
        assert source["metadata_source"] == "host_declared"
        assert source.get("best_epoch") is None
    assert row["calculation"]["raw_delta"] == 0.25
    assert "checkpoint_binding_not_verified" in row["limitations"]
    assert "statistical_significance_not_assessed" in row["limitations"]


@pytest.mark.parametrize("left,right,relation", [
    (config_field(False), config_field(0), "different"),
    (config_field(None), config_field(""), "different"),
    (config_field(None), config_field(status="missing"), "unknown"),
    (config_field(status="missing"), config_field(status="missing"), "unknown"),
    (config_field(status="unavailable"), config_field(False), "unknown"),
    (config_field(False), config_field(False), "equal"),
])
def test_configuration_context_preserves_types_and_never_decides_comparability(left, right, relation):
    compare = core_api()
    a, b = evidence(), evidence()
    a["configuration_evidence"]["fields"] = [left]
    b["configuration_evidence"]["fields"] = [right]
    result = compare(a, b, targets=[target()])
    assert result["configuration_differences"] == [
        {"path": ["enabled"], "a": left, "b": right, "relation": relation}]
    assert result["targets"]["quality"]["comparability"]["status"] == "comparable"


@pytest.mark.parametrize("case", ["duplicate", "empty", "too-many", "blank-id", "unknown-section",
    "value-override", "extra-target-key", "ambiguous-selector", "invalid-provenance"])
def test_invalid_targets_raise_without_partial_output(case):
    compare = core_api()
    selected = [target()]
    if case == "duplicate": selected *= 2
    elif case == "empty": selected = []
    elif case == "too-many": selected = [target(str(i)) for i in range(33)]
    elif case == "blank-id": selected[0]["target_id"] = ""
    elif case == "unknown-section": selected[0]["a"]["section"] = "training"
    elif case == "value-override": selected[0]["a"]["value"] = 99
    elif case == "extra-target-key": selected[0]["approval"] = True
    elif case == "ambiguous-selector": selected[0]["a"]["metric_name"] = ["score", "loss"]
    else: selected[0]["a"]["provenance"] = {"task": "invented"}
    with pytest.raises((TypeError, ValueError)):
        compare(evidence(), evidence(), targets=selected)


def test_maximum_target_count_is_inclusive_and_inputs_are_immutable():
    compare = core_api()
    a, b, selected = evidence(), evidence(0.75), [target(str(i)) for i in range(32)]
    before = deepcopy((a, b, selected))
    first = compare(a, b, targets=selected)
    assert len(first["targets"]) == 32
    assert first == compare(a, b, targets=selected)
    assert (a, b, selected) == before
    encoded = json.dumps(first, ensure_ascii=False, allow_nan=False, separators=(",", ":")).encode("utf-8")
    assert len(encoded) <= 64 * 1024
    first["targets"].clear()
    assert len(compare(a, b, targets=selected)["targets"]) == 32


def test_oversized_utf8_output_is_an_error_not_truncation():
    compare = core_api()
    # A retained target_id alone exceeds 64 KiB; no assumption about optional prose.
    selected = target("界" * (64 * 1024 // 3 + 1))
    with pytest.raises(ValueError):
        compare(evidence(), evidence(), targets=[selected])


def test_core_has_no_file_reads_or_writes_at_probed_python_entrypoints(monkeypatch):
    compare = core_api()
    def deny(*args, **kwargs):
        raise AssertionError("core attempted file I/O")
    with monkeypatch.context() as patch:
        patch.setattr(builtins, "open", deny)
        patch.setattr(io, "open", deny)
        patch.setattr(Path, "open", deny)
        result = compare(evidence(), evidence(0.75), targets=[target()])
    assert result["targets"]["quality"]["calculation"]["raw_delta"] == 0.25


def test_blocked_assertion_rejects_false_success_and_false_zero():
    for calculation in ({"status": "calculated", "raw_delta": None, "direction_interpretation": None},
                        {"status": "blocked", "raw_delta": 0, "direction_interpretation": None}):
        with pytest.raises(AssertionError):
            assert_blocked({"calculation": calculation})

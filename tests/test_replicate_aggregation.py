"""M18 core RED: independent arithmetic, membership, eligibility and coverage.

Missing production is asserted inside test bodies, so collection stays valid.
No implementation stub, skip, xfail, or aggregation oracle is supplied here.
"""

from copy import deepcopy
from dataclasses import asdict, replace
import importlib
import inspect
from itertools import combinations
import json
import math

import pytest

import experiment_comparability as m11
import experiment_identity as identities
from m17_contract_helpers import options, write_experiment
from m18_contract_helpers import (
    ROOT_LIMITATIONS, assert_empty_statistics, assert_partition, bundle, core_api,
    deny_external_access, identity, input_envelope, invoke, json_size, member_ids,
)


DEFINITION_FIELDS = (
    "configuration_ref", "task", "target", "model_ref", "dataset_version", "split_id",
    "source_revision", "initialization_checkpoint_ref",
)


def trace_primitive(monkeypatch, owner, name, calls):
    original = getattr(owner, name)
    def traced(left, right):
        calls.append((left, right))
        return original(left, right)
    monkeypatch.setattr(owner, name, traced)
    module = importlib.import_module("experiment_replicates")
    # Support both module imports and ordinary directly imported function aliases.
    for alias, value in list(vars(module).items()):
        if value is original:
            monkeypatch.setattr(module, alias, traced)


def test_public_api_signature_is_one_keyword_only_core():
    function = core_api()
    signature = inspect.signature(function)
    assert list(signature.parameters) == ["executions", "group_declaration", "targets"]
    assert signature.parameters["executions"].kind == inspect.Parameter.POSITIONAL_OR_KEYWORD
    for name in ("group_declaration", "targets"):
        assert signature.parameters[name].kind == inspect.Parameter.KEYWORD_ONLY
        assert signature.parameters[name].default is inspect.Parameter.empty
    assert not hasattr(importlib.import_module("experiment_replicates"), "assess_replicate_eligibility")


def test_complete_group_has_hand_calculated_statistics_and_explicit_members():
    result = invoke(core_api(), bundle())
    assert result["schema_version"] == "replicate-aggregation-v1"
    group, row = result["group"], result["targets"]["quality"]
    assert group["experiment_identity"] == asdict(identity(0).experiment)
    assert group["group_id"] == "group-v1" and group["membership_source"] == "host_declared"
    assert group["group_status"] == group["aggregation_eligibility"]["status"] == "eligible"
    assert group["identity_consistency"]["status"] == "consistent"
    assert row["target_status"] == "complete"
    assert row["metric_comparability"] == {
        "status": "comparable", "reason_codes": [], "policy_version": "metric-result-v1"}
    assert row["statistics"] == {"count": 3, "mean": 3.0, "median": 3.0,
                                 "sample_std": 2.0, "min": 1.0, "max": 5.0}
    assert [x["value"] for x in row["included_executions"]] == [1.0, 3.0, 5.0]
    assert_partition(row, [f"exec-{i:02d}" for i in range(3)])
    assert ROOT_LIMITATIONS <= set(result["limitations"])


@pytest.mark.parametrize("values,mean,median,std", [
    ((1.0, 2.0, 9.0), 4.0, 2.0, math.sqrt(19.0)),
    ((1.0, 3.0, 5.0, 7.0), 4.0, 4.0, math.sqrt(20.0 / 3.0)),
    ((4.0, 4.0), 4.0, 4.0, 0.0),
])
def test_equal_execution_weighting_odd_even_median_and_sample_std(values, mean, median, std):
    data = bundle(values)
    # Dataset sample sizes are not execution weights; keep them as opaque acquisition metadata.
    for i, run in enumerate(data[0]):
        run["snapshot_metadata"]["dataset_sample_count"] = 10 ** i
    row = invoke(core_api(), data)["targets"]["quality"]
    assert row["statistics"]["mean"] == pytest.approx(mean)
    assert row["statistics"]["median"] == median
    assert row["statistics"]["sample_std"] == pytest.approx(std)
    assert row["statistics"]["min"] == min(values) and row["statistics"]["max"] == max(values)


@pytest.mark.parametrize("count", [1, 16])
def test_execution_bounds_are_inclusive(count):
    data = bundle(tuple(float(i) for i in range(count)))
    row = invoke(core_api(), data)["targets"]["quality"]
    assert row["statistics"]["count"] == count and row["target_status"] == "complete"
    assert row["statistics"]["mean"] == (count - 1) / 2
    if count == 1:
        assert row["statistics"]["sample_std"] is None
        assert "insufficient_variability" in row["limitations"]


@pytest.mark.parametrize("count", [0, 17])
def test_expected_execution_bounds_reject(count):
    function = core_api()
    data = bundle(tuple(float(i) for i in range(count)))
    with pytest.raises(ValueError):
        invoke(function, data)


@pytest.mark.parametrize("where", ["roster", "snapshots", "selectors"])
def test_duplicate_execution_identity_rejected_without_deduplication(where):
    function = core_api()
    data = bundle()
    entries = {"roster": data[1]["expected_executions"], "snapshots": data[0],
               "selectors": data[2][0]["members"]}[where]
    entries.append(deepcopy(entries[0]))
    with pytest.raises(ValueError):
        invoke(function, data)


def test_unexpected_execution_not_silently_added_to_denominator():
    function = core_api()
    data = bundle((1.0, 3.0))
    data[0].append(bundle()[0][2])
    with pytest.raises(ValueError):
        invoke(function, data)


def test_different_experiment_identity_blocks_whole_group():
    data = bundle()
    other = replace(identity(2), experiment=replace(identity(2).experiment, experiment_id="other-method"))
    data[0][2]["execution"] = other
    data[0][2]["provenance"] = replace(data[0][2]["provenance"], execution=other)
    data[1]["expected_executions"][2]["execution"] = other
    data[2][0]["members"][2]["execution"] = other
    result = invoke(core_api(), data)
    assert result["group"]["group_status"] == "blocked"
    assert result["group"]["aggregation_eligibility"]["reason_codes"]
    assert_empty_statistics(result["targets"]["quality"])


def test_snapshot_provenance_cannot_substitute_execution_identity():
    function = core_api()
    data = bundle()
    data[0][0]["provenance"] = replace(data[0][0]["provenance"], execution=identity(1))
    with pytest.raises(ValueError):
        invoke(function, data)


def test_repeated_seed_preserved_and_not_called_independent_sample_count():
    result = invoke(core_api(), bundle(seeds=(42, 42, 2024)))
    row, seed = result["targets"]["quality"], result["group"]["seed_summary"]
    assert row["statistics"]["count"] == 3 and row["statistics"]["mean"] == 3.0
    assert seed["distinct_known_seed_count"] == 2 and seed["unknown_seed_count"] == 0
    assert seed["repeated_seeds"] == [{"seed": 42, "executions": [asdict(identity(0)), asdict(identity(1))]}]
    assert "repeated_seed" in result["limitations"]
    assert "independence_not_verified" in result["limitations"]


def test_unknown_realized_seed_never_filled_from_expected_or_configuration():
    data = bundle(seeds=(None, 43, 44))
    data[1]["expected_executions"][0]["expected_seed"] = 42
    data[0][0]["evidence"]["configuration"]["seed"] = 42
    data[0][0]["evidence"]["configuration"]["sample_seed"] = 99
    result = invoke(core_api(), data)
    assert result["group"]["seed_summary"]["unknown_seed_count"] == 1
    assert result["group"]["seed_summary"]["distinct_known_seed_count"] == 2
    assert "seed_unknown" in result["limitations"]
    assert result["targets"]["quality"]["statistics"]["count"] == 3


def test_known_expected_seed_mismatch_blocks_instead_of_relabeling():
    data = bundle()
    data[1]["expected_executions"][0]["expected_seed"] = 999
    result = invoke(core_api(), data)
    assert result["group"]["group_status"] == "blocked"
    assert "expected_seed_mismatch" in result["group"]["aggregation_eligibility"]["reason_codes"]
    assert_empty_statistics(result["targets"]["quality"])


@pytest.mark.parametrize("field", DEFINITION_FIELDS)
@pytest.mark.parametrize("value,status", [("different", "blocked"), (None, "insufficient")])
def test_identity_definition_conflicts_and_unknowns_do_not_form_a_subset(field, value, status):
    data = bundle()
    data[0][2]["provenance"] = replace(data[0][2]["provenance"], **{field: value})
    result = invoke(core_api(), data)
    assert result["group"]["group_status"] == status
    assert result["group"]["identity_consistency"]["status"] == ("conflict" if value else "unknown")
    row = result["targets"]["quality"]
    assert_empty_statistics(row)
    assert row["coverage"]["observed_count"] == 3
    assert_partition(row, [f"exec-{i:02d}" for i in range(3)])


@pytest.mark.parametrize("where", ["group", "member"])
@pytest.mark.parametrize("value,status", [("other-training", "blocked"), (None, "insufficient")])
def test_training_protocol_is_required_outside_m11(where, value, status):
    data = bundle()
    owner = data[1] if where == "group" else data[0][2]
    owner["training_protocol_ref"] = value
    result = invoke(core_api(), data)
    assert result["group"]["group_status"] == status
    assert_empty_statistics(result["targets"]["quality"])


@pytest.mark.parametrize("environment", [None, "other-environment"])
def test_environment_drift_is_limitation_not_seed_causality(environment):
    data = bundle()
    data[0][1]["provenance"] = replace(data[0][1]["provenance"], environment_ref=environment)
    result = invoke(core_api(), data)
    assert result["targets"]["quality"]["statistics"]["count"] == 3
    assert "environment_not_controlled" in result["limitations"]


def test_real_identity_primitive_is_reused_on_all_distinct_pairs(monkeypatch):
    function, calls = core_api(), []
    trace_primitive(monkeypatch, identities, "check_identity_consistency", calls)
    invoke(function, bundle())
    pairs = {frozenset((a.execution.execution_id, b.execution.execution_id)) for a, b in calls}
    assert pairs == {frozenset(pair) for pair in combinations(["exec-00", "exec-01", "exec-02"], 2)}


@pytest.mark.parametrize("field", ["metric_definition", "aggregation", "evaluation_protocol", "selection_protocol"])
@pytest.mark.parametrize("value,status", [("other", "incompatible"), (None, "unknown")])
def test_real_metric_policy_blocks_conflict_and_unknown_without_subset(field, value, status):
    data = bundle()
    data[2][0]["members"][2][field] = value
    row = invoke(core_api(), data)["targets"]["quality"]
    assert row["metric_comparability"]["status"] == status
    reason = f"{field}_mismatch" if value else f"missing_{field}"
    assert reason in row["metric_comparability"]["reason_codes"]
    assert row["target_status"] == ("blocked" if value else "insufficient")
    assert_empty_statistics(row)
    assert_partition(row, [f"exec-{i:02d}" for i in range(3)])


def test_direction_conflict_is_m11_incompatibility_not_sign_flip():
    data = bundle()
    member = data[2][0]["members"][2]
    member["metric_spec"] = replace(member["metric_spec"], direction="minimize")
    data[0][2]["evidence"]["test_results"]["metrics"]["score"]["direction"] = "minimize"
    row = invoke(core_api(), data)["targets"]["quality"]
    assert "direction_mismatch" in row["metric_comparability"]["reason_codes"]
    assert_empty_statistics(row)


def test_unknown_reference_must_not_hide_conflict_between_other_members(monkeypatch):
    function, calls = core_api(), []
    trace_primitive(monkeypatch, m11, "compare_metric_results", calls)
    data = bundle()
    for member, protocol in zip(data[2][0]["members"], (None, "protocol-b", "protocol-c")):
        member["evaluation_protocol"] = protocol
    row = invoke(function, data)["targets"]["quality"]
    assert len(calls) == 3
    assert row["metric_comparability"]["status"] == "incompatible"
    assert {"missing_evaluation_protocol", "evaluation_protocol_mismatch"} <= set(
        row["metric_comparability"]["reason_codes"])
    assert_empty_statistics(row)


def test_pair_bound_real_m11_values_and_types_are_exact(monkeypatch):
    function, calls = core_api(), []
    trace_primitive(monkeypatch, m11, "compare_metric_results", calls)
    result = invoke(function, bundle(tuple(float(i) for i in range(16))))
    assert len(calls) == 120
    actual = set()
    for a, b in calls:
        assert isinstance(a, identities.MetricResultProvenance)
        assert isinstance(b, identities.MetricResultProvenance)
        actual.add(frozenset((a.provenance.execution.execution_id, b.provenance.execution.execution_id)))
        for item in (a, b):
            assert item.value == float(item.provenance.execution.execution_id.split("-")[1])
            assert item.checkpoint_ref is None and item.best_epoch is None
    assert len(actual) == 120
    assert result["targets"]["quality"]["statistics"]["count"] == 16
    assert "pairwise_matrix" not in result


def test_comparable_models_can_still_be_ineligible_replicates():
    data = bundle()
    data[0][1]["provenance"] = replace(data[0][1]["provenance"], model_ref="different-model")
    result = invoke(core_api(), data)
    assert result["group"]["identity_consistency"]["status"] == "conflict"
    assert_empty_statistics(result["targets"]["quality"])


@pytest.mark.parametrize("section", ["test", "validation"])
def test_sections_select_their_own_values_and_preserve_source(section):
    data = bundle(section=section)
    for i, run in enumerate(data[0]):
        run["evidence"]["validation_metrics"]["score"]["best_value"] = 10.0 + i
    row = invoke(core_api(), data)["targets"]["quality"]
    assert row["statistics"]["mean"] == (3.0 if section == "test" else 11.0)
    for entry in row["included_executions"]:
        assert entry["source"]["section"] == section
        assert entry["source"]["metadata_source"] == "host_declared"
        if section == "validation":
            assert entry["source"]["best_epoch"] == 0
        else:
            assert entry["source"]["evidence_kind"] == "csv_reported"
            assert entry["source"]["provenance"]["checkpoint_binding"] == "not_verified"


@pytest.mark.parametrize("mutation", ["kind", "column", "direction", "value-key"])
def test_malformed_or_misaligned_observation_excluded_locally(mutation):
    data = bundle()
    test = data[0][2]["evidence"]["test_results"]
    if mutation == "kind": test["evidence_kind"] = "validation_history"
    elif mutation == "column": test["metrics"]["score"]["source_column"] = "wrong"
    elif mutation == "direction": test["metrics"]["score"]["direction"] = "minimize"
    else: del test["metrics"]["score"]["value"]
    row = invoke(core_api(), data)["targets"]["quality"]
    assert member_ids(row["excluded_executions"]) == ["exec-02"]
    assert row["statistics"]["count"] == 2 and row["statistics"]["mean"] == 2.0
    assert row["target_status"] == "partial"
    assert_partition(row, [f"exec-{i:02d}" for i in range(3)])


@pytest.mark.parametrize("kind", ["absent", "unavailable", "metric", "not_requested"])
def test_missing_replicate_or_observation_keeps_fixed_denominator(kind):
    data = bundle()
    if kind == "absent": data[0].pop()
    elif kind == "unavailable": data[0][2].update(status="unavailable", evidence=None)
    elif kind == "metric": data[0][2]["evidence"]["test_results"]["metrics"].clear()
    else: data[0][2]["evidence"]["test_results"] = {"status": "not_requested"}
    result = invoke(core_api(), data)
    row = result["targets"]["quality"]
    assert row["target_status"] == "partial"
    assert row["statistics"]["mean"] == 2.0
    assert row["statistics"]["sample_std"] == pytest.approx(math.sqrt(2.0))
    assert row["coverage"]["observed_count"] == 2
    assert member_ids(row["missing_executions"]) == ["exec-02"]
    assert_partition(row, [f"exec-{i:02d}" for i in range(3)])
    group = result["group"]["coverage"]
    assert group["expected_count"] == 3 and group["usable_count"] == 2
    assert group["coverage_ratio"] == 2 / 3
    assert group["available_count"] == (2 if kind in ("absent", "unavailable") else 3)
    assert member_ids(group["missing_replicates"]) == (["exec-02"] if kind in ("absent", "unavailable") else [])


def test_per_target_coverage_does_not_force_complete_case_subset():
    data = bundle()
    validation = deepcopy(data[2][0])
    validation.update(target_id="validation", section="validation")
    for member in validation["members"]:
        member["metric_spec"] = replace(member["metric_spec"], path=("validation", "measurement"))
    data[2].append(validation)
    data[0][0]["evidence"]["test_results"]["metrics"].clear()
    data[0][1]["evidence"]["validation_metrics"].clear()
    result = invoke(core_api(), data)
    assert member_ids(result["targets"]["quality"]["included_executions"]) == ["exec-01", "exec-02"]
    assert member_ids(result["targets"]["validation"]["included_executions"]) == ["exec-00", "exec-02"]
    assert result["targets"]["quality"]["statistics"]["mean"] == 4.0
    assert result["targets"]["validation"]["statistics"]["mean"] == 3.0
    assert result["group"]["coverage"]["usable_count"] == 1
    assert result["group"]["coverage"]["coverage_ratio"] == 1 / 3


def test_invalid_csv_is_excluded_and_validation_stays_usable():
    data = bundle()
    validation = bundle(section="validation")[2][0]
    validation["target_id"] = "validation"
    data[2].append(validation)
    data[0][2]["evidence"]["test_results"].update(status="invalid", metrics={})
    result = invoke(core_api(), data)
    assert result["targets"]["validation"]["statistics"]["count"] == 3
    row = result["targets"]["quality"]
    assert row["statistics"]["count"] == 2
    assert member_ids(row["excluded_executions"]) == ["exec-02"]
    assert "checkpoint_binding_not_verified" in row["limitations"]


def test_conflicting_member_cannot_be_hidden_by_missing_numeric_value():
    data = bundle()
    data[0][2]["evidence"]["test_results"]["metrics"].clear()
    data[2][0]["members"][2]["evaluation_protocol"] = "other"
    row = invoke(core_api(), data)["targets"]["quality"]
    assert row["target_status"] == "blocked"
    assert row["aggregation_eligibility"]["reason_codes"]
    assert_empty_statistics(row)
    assert_partition(row, [f"exec-{i:02d}" for i in range(3)])


def test_zero_supplied_executions_returns_coverage_not_fabricated_zero_metric():
    data = bundle()
    data[0].clear()
    result = invoke(core_api(), data)
    row = result["targets"]["quality"]
    assert result["group"]["group_status"] == "insufficient"
    assert row["target_status"] == "insufficient" and row["metric_comparability"] is None
    assert_empty_statistics(row)
    assert_partition(row, [f"exec-{i:02d}" for i in range(3)])


def test_one_observation_with_missing_roster_members_is_partial_with_null_std():
    data = bundle()
    del data[0][1:]
    row = invoke(core_api(), data)["targets"]["quality"]
    assert row["target_status"] == "partial"
    assert row["statistics"] == {"count": 1, "mean": 1.0, "median": 1.0,
                                 "sample_std": None, "min": 1.0, "max": 1.0}
    assert "insufficient_variability" in row["limitations"]


@pytest.mark.parametrize("value", [True, False, float("nan"), float("inf"), -float("inf"), "3", None, {}])
def test_invalid_numeric_observation_is_excluded_without_json_contamination(value):
    data = bundle()
    data[0][2]["evidence"]["test_results"]["metrics"]["score"]["value"] = value
    result = invoke(core_api(), data)
    row = result["targets"]["quality"]
    assert row["statistics"]["mean"] == 2.0 and row["statistics"]["count"] == 2
    assert row["coverage"]["observed_count"] == 2
    assert member_ids(row["excluded_executions"]) == ["exec-02"]
    json.dumps(result, allow_nan=False)


@pytest.mark.parametrize("values", [(1e308, 1e308), (-1.7e308, 1.7e308)])
def test_extreme_finite_arithmetic_never_emits_nonfinite_or_silent_zero(values):
    row = invoke(core_api(), bundle(values))["targets"]["quality"]
    # The contract permits a stable calculation OR an explicit numerical block.
    # It does not require an avoidable intermediate overflow from naive summing.
    if row["target_status"] == "blocked":
        assert "numerical_error" in row["aggregation_eligibility"]["reason_codes"]
        assert_empty_statistics(row)
        assert row["coverage"]["observed_count"] == 2
        assert_partition(row, ["exec-00", "exec-01"])
    else:
        assert values[0] == values[1]  # opposite extremes have unrepresentable sample std
        assert row["statistics"]["mean"] == 1e308
        assert row["statistics"]["sample_std"] == 0.0
    json.dumps(row, allow_nan=False)


def test_reordering_executions_roster_selectors_targets_is_deterministic():
    function = core_api()
    data = bundle()
    second = deepcopy(data[2][0])
    second["target_id"] = "another"
    data[2].append(second)
    before = invoke(function, data)
    data[0].reverse()
    data[1]["expected_executions"].reverse()
    data[2].reverse()
    for target in data[2]: target["members"].reverse()
    assert invoke(function, data) == before
    assert list(before["targets"]) == ["another", "quality"]


def test_input_immutability_and_deep_return_isolation():
    function, data = core_api(), bundle()
    before = deepcopy(data)
    result = invoke(function, data)
    assert data == before
    pristine = deepcopy(result)
    result["group"]["coverage"]["expected_count"] = 999
    result["targets"]["quality"]["included_executions"][0]["source"]["provenance"]["checkpoint_binding"] = "verified"
    assert invoke(function, data) == pristine and data == before


@pytest.mark.parametrize("count", [1, 32])
def test_target_bounds_are_inclusive(count):
    data = bundle()
    data[2][:] = [dict(deepcopy(data[2][0]), target_id=f"target-{i:02d}") for i in range(count)]
    result = invoke(core_api(), data)
    assert len(result["targets"]) == count
    assert json_size(result) <= 128 * 1024


@pytest.mark.parametrize("count", [0, 33])
def test_target_bounds_reject(count):
    function, data = core_api(), bundle()
    data[2][:] = [dict(deepcopy(data[2][0]), target_id=f"target-{i}") for i in range(count)]
    with pytest.raises(ValueError): invoke(function, data)


@pytest.mark.parametrize("mutation", ["duplicate-target", "missing-selector", "extra-selector", "policy", "section", "seed-bool"])
def test_invalid_declarations_reject_instead_of_guessing(mutation):
    function, data = core_api(), bundle()
    if mutation == "duplicate-target": data[2].append(deepcopy(data[2][0]))
    elif mutation == "missing-selector": data[2][0]["members"].pop()
    elif mutation == "extra-selector": data[2][0]["members"].append(bundle((1, 2, 3, 4))[2][0]["members"][3])
    elif mutation == "policy": data[1]["aggregation_policy"] = "best-run-v1"
    elif mutation == "section": data[2][0]["section"] = "mixed"
    else: data[1]["expected_executions"][0]["expected_seed"] = True
    with pytest.raises((TypeError, ValueError)): invoke(function, data)


def test_output_budget_is_utf8_bytes_and_never_silent_truncation():
    function, data = core_api(), bundle()
    data[1]["group_id"] = "界" * 50000  # mandatory echoed field >128 KiB; input <2 MiB
    assert json_size(input_envelope(data)) < 2 * 1024 * 1024
    with pytest.raises(ValueError, match="output"): invoke(function, data)


@pytest.mark.parametrize("extra,allowed", [(0, True), (1, False)])
def test_input_budget_includes_unused_snapshot_metadata_at_exact_boundary(extra, allowed):
    function, data = core_api(), bundle()
    metadata = data[0][0]["snapshot_metadata"]
    metadata["padding"] = ""
    metadata["padding"] = "x" * (2 * 1024 * 1024 - json_size(input_envelope(data)) + extra)
    assert json_size(input_envelope(data)) == 2 * 1024 * 1024 + extra
    if allowed:
        assert invoke(function, data)["targets"]["quality"]["statistics"]["count"] == 3
    else:
        with pytest.raises(ValueError, match="input"): invoke(function, data)


@pytest.mark.parametrize("kind", ["deep", "cycle"])
def test_bounded_structure_rejects_deep_and_cyclic_inputs(kind):
    function, data = core_api(), bundle()
    nested = {}
    if kind == "cycle": nested["again"] = nested
    else:
        for _ in range(17): nested = {"next": nested}
    data[0][0]["snapshot_metadata"]["extra"] = nested
    with pytest.raises(ValueError): invoke(function, data)


@pytest.mark.parametrize("kind", ["identity", "metric", "training"])
def test_singleton_still_requires_complete_declarations(kind):
    data = bundle((1.0,))
    if kind == "identity":
        data[0][0]["provenance"] = replace(data[0][0]["provenance"], initialization_checkpoint_ref=None)
    elif kind == "metric": data[2][0]["members"][0]["evaluation_protocol"] = None
    else: data[0][0]["training_protocol_ref"] = None
    row = invoke(core_api(), data)["targets"]["quality"]
    assert row["target_status"] == "insufficient"
    assert_empty_statistics(row)


def test_missing_metric_spec_is_insufficient_not_inferred_from_name():
    data = bundle()
    data[2][0]["members"][0]["metric_spec"] = None
    row = invoke(core_api(), data)["targets"]["quality"]
    assert row["target_status"] == "insufficient"
    assert row["aggregation_eligibility"]["reason_codes"]
    assert_empty_statistics(row)


def test_zero_available_metrics_retains_all_supplied_members_and_null_statistics():
    data = bundle()
    for run in data[0]: run["evidence"]["test_results"]["metrics"].clear()
    row = invoke(core_api(), data)["targets"]["quality"]
    assert row["target_status"] == "insufficient"
    assert row["metric_comparability"] is None
    assert row["coverage"]["observed_count"] == 0
    assert_empty_statistics(row)
    assert_partition(row, ["exec-00", "exec-01", "exec-02"])


def test_missing_member_stays_missing_when_other_members_conflict():
    data = bundle()
    data[0].pop()
    data[0][1]["provenance"] = replace(data[0][1]["provenance"], model_ref="other")
    result = invoke(core_api(), data)
    row = result["targets"]["quality"]
    assert result["group"]["group_status"] == "blocked"
    assert member_ids(row["missing_executions"]) == ["exec-02"]
    assert member_ids(row["excluded_executions"]) == ["exec-00", "exec-01"]
    assert_partition(row, ["exec-00", "exec-01", "exec-02"])


def test_real_m17_alignment_and_unknown_policy_remain_unchanged():
    from experiment_comparison import compare_experiment_evidence
    from m17_contract_helpers import evidence, target
    selected = target()
    selected["a"]["evaluation_protocol"] = selected["b"]["evaluation_protocol"] = None
    row = compare_experiment_evidence(evidence(1.0), evidence(3.0), targets=[selected])["targets"]["quality"]
    assert row["comparability"]["status"] == "unknown"
    assert row["calculation"]["raw_delta"] == 2.0
    assert "limited_numerical_observation" in row["limitations"]


def test_selected_evidence_values_agree_with_real_m17_projection():
    from experiment_comparison import compare_experiment_evidence
    data = bundle((1.0, 3.0))
    result = invoke(core_api(), data)
    sides = []
    for run, selector in zip(data[0], data[2][0]["members"]):
        side = {key: value for key, value in selector.items() if key != "execution"}
        side.update(section="test", provenance=run["provenance"])
        sides.append(side)
    pair = compare_experiment_evidence(data[0][0]["evidence"], data[0][1]["evidence"],
        targets=[{"target_id": "quality", "a": sides[0], "b": sides[1]}])["targets"]["quality"]
    row = result["targets"]["quality"]
    assert row["metric_comparability"] == pair["comparability"]
    assert [entry["value"] for entry in row["included_executions"]] == [pair["value_a"], pair["value_b"]]


def test_core_has_no_external_access_through_scoped_probe(monkeypatch):
    function, data = core_api(), bundle()
    with deny_external_access(monkeypatch) as calls:
        result = invoke(function, data)
    assert calls == [] and result["targets"]["quality"]["statistics"]["count"] == 3


def test_provenance_limitations_and_no_inferential_or_group_comparison_outputs():
    result = invoke(core_api(), bundle())
    assert ROOT_LIMITATIONS <= set(result["limitations"])
    row = result["targets"]["quality"]
    assert "checkpoint_binding_not_verified" in row["limitations"]
    forbidden = {"winner", "ranking", "best", "aggregate_delta", "raw_delta", "p_value",
                 "confidence_interval", "effective_independent_sample_size", "significantly_better",
                 "robustly_better", "generalizes_better", "causal_contribution", "state_of_the_art"}
    def inspect_tree(value):
        if isinstance(value, dict):
            assert not forbidden.intersection(value)
            for item in value.values(): inspect_tree(item)
        elif isinstance(value, list):
            for item in value: inspect_tree(item)
    inspect_tree(result)
    for entry in row["included_executions"]:
        provenance = entry["source"]["provenance"]
        assert provenance["checkpoint_identity"] is None
        assert provenance["checkpoint_binding"] == "not_verified"
        assert provenance["recomputed_from_predictions"] is False


# Independent checks run before GREEN too: future missing APIs cannot mask them.
def test_generic_fixture_matches_real_m16_selected_evidence(tmp_path):
    from tool_layer.experiment_tools import analyze_experiment_evidence
    root = write_experiment(tmp_path, "generic", value=1.0)
    actual = analyze_experiment_evidence(str(root), **options())
    fixture = bundle((1.0,))[0][0]["evidence"]
    for key in ("status", "evidence_kind", "metrics", "provenance"):
        assert fixture["test_results"][key] == actual["test_results"][key]
    for key in ("best_epoch", "best_value"):
        assert fixture["validation_metrics"]["score"][key] == actual["validation_metrics"]["score"][key]


def test_real_m11_unknown_anchor_counterexample_is_possible():
    a, b, c = [run["provenance"] for run in bundle()[0]]
    a = replace(a, split_id=None)
    c = replace(c, split_id="other")
    def result(p):
        return identities.MetricResultProvenance("score", 1.0, p, "quality-v1", "maximize", "macro", "eval", "select")
    assert m11.compare_metric_results(result(a), result(b)).status == "unknown"
    assert m11.compare_metric_results(result(a), result(c)).status == "unknown"
    assert m11.compare_metric_results(result(b), result(c)).status == "incompatible"


@pytest.mark.parametrize("corruption", ["duplicate", "missing", "count", "ratio", "reason"])
def test_partition_assertion_rejects_independent_bad_results(corruption):
    row = {"included_executions": [{"execution": asdict(identity(0))}],
           "missing_executions": [{"execution": asdict(identity(1)), "reason_codes": ["missing"]}],
           "excluded_executions": [], "coverage": {"expected_count": 2, "observed_count": 1,
           "included_count": 1, "eligible_count": 1, "missing_count": 1, "excluded_count": 0,
           "coverage_ratio": 0.5}, "statistics": {"count": 1}}
    assert_partition(row, ["exec-00", "exec-01"])
    if corruption == "duplicate": row["missing_executions"][0]["execution"] = asdict(identity(0))
    elif corruption == "missing": row["missing_executions"].clear()
    elif corruption == "count": row["statistics"]["count"] = 2
    elif corruption == "ratio": row["coverage"]["coverage_ratio"] = 1.0
    else: row["missing_executions"][0]["reason_codes"].clear()
    with pytest.raises(AssertionError): assert_partition(row, ["exec-00", "exec-01"])

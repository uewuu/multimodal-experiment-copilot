"""M19 deterministic group-comparison RED; public M18/M11 are real delegates."""

from copy import deepcopy
from dataclasses import asdict, replace
import importlib
import inspect
import json
import math

import pytest

import experiment_comparability as m11
import experiment_replicates as m18
from experiment_identity import MetricResultProvenance
from m18_contract_helpers import assert_partition, deny_external_access
from m19_contract_helpers import (
    GROUP_KEYS, ROOT_LIMITATIONS, SIDE_KEYS, aggregate, assert_no_claims, core_api,
    invoke, json_size, key, many_targets, pair, patch_public, set_direction,
)


def test_public_signature():
    signature = inspect.signature(core_api())
    assert list(signature.parameters) == ["group_a", "group_b", "targets"]
    assert signature.parameters["targets"].kind == inspect.Parameter.KEYWORD_ONLY
    assert signature.parameters["targets"].default is inspect.Parameter.empty


def test_output_shape_and_exact_real_m18_projections():
    function, data = core_api(), pair()
    expected = [aggregate(side) for side in data[:2]]
    result = invoke(function, data)
    assert set(result) == {"schema_version", "comparison_policy", "groups", "targets", "limitations"}
    assert result["schema_version"] == "replicate-group-comparison-v1"
    assert result["comparison_policy"] == "unpaired-descriptive-v1"
    assert set(result["groups"]) == {"a", "b"}
    row = result["targets"]["quality"]
    assert set(row) == {"a", "b", "alignment", "metric_comparability", "comparison", "limitations"}
    for label, source, old in zip(("a", "b"), data[:2], expected):
        projected = result["groups"][label]
        assert set(projected) == GROUP_KEYS
        for name in GROUP_KEYS - {"aggregation_policy", "training_protocol_ref", "expected_executions"}:
            assert projected[name] == old["group"][name]
        for name in ("aggregation_policy", "training_protocol_ref"):
            assert projected[name] == source["group_declaration"][name]
        assert projected["expected_executions"] == [
            {"execution": asdict(x["execution"]), "expected_seed": x["expected_seed"]}
            for x in source["group_declaration"]["expected_executions"]]
        side = row[label]
        assert set(side) == SIDE_KEYS
        source_id = source["targets"][0]["target_id"]
        assert (side["source_target_id"], side["section"]) == (source_id, "test")
        for name in SIDE_KEYS - {"source_target_id", "section"}:
            assert side[name] == old["targets"][source_id][name]
        assert_partition(side, [x["execution"].execution_id for x in source["executions"]])
    assert row["alignment"] == {"status": "aligned", "reason_codes": []}
    assert row["metric_comparability"] == {
        "status": "comparable", "policy_version": "metric-result-v1", "reason_codes": [],
        "assessed_pair_count": 9}
    assert row["comparison"] == {
        "eligibility": "eligible", "reason_codes": [], "calculation_status": "calculated",
        "raw_delta": 1., "direction": "maximize", "direction_interpretation": "b_favorable",
        "coverage_relation": {"a_complete": True, "b_complete": True,
                              "same_expected_count": True, "same_included_count": True}}
    assert ROOT_LIMITATIONS <= set(result["limitations"])
    assert_no_claims(result)
    json.dumps(result, allow_nan=False)


def test_exactly_two_public_m18_calls_with_original_declarations_and_real_results(monkeypatch):
    function, data, calls = core_api(), pair(), []
    original = m18.aggregate_replicate_evidence
    def traced(executions, *, group_declaration, targets):
        result = original(executions, group_declaration=group_declaration, targets=targets)
        calls.append((deepcopy((executions, group_declaration, targets)), deepcopy(result)))
        return result
    patch_public(monkeypatch, m18, "aggregate_replicate_evidence", traced,
                 importlib.import_module("experiment_group_comparison"))
    result = invoke(function, data)
    assert len(calls) == 2
    assert [c[0] for c in calls] == [
        (side["executions"], side["group_declaration"], side["targets"]) for side in data[:2]]
    for label, call in zip(("a", "b"), calls):
        assert result["targets"]["quality"][label]["statistics"] == next(iter(call[1]["targets"].values()))["statistics"]


@pytest.mark.parametrize("same_experiment", [False, True])
@pytest.mark.parametrize("size", [1, 3, 16])
def test_every_cross_pair_uses_real_execution_provenance_not_group_mean(monkeypatch, same_experiment, size):
    function = core_api()
    data = pair(tuple(range(1, size + 1)), tuple(range(101, 101 + size)), same_experiment=same_experiment)
    a_ids = {key(x["execution"]) for x in data[0]["executions"]}
    b_ids = {key(x["execution"]) for x in data[1]["executions"]}
    calls, original = [], m11.compare_metric_results
    def traced(left, right):
        result = original(left, right)
        if key(left.provenance.execution) in a_ids and key(right.provenance.execution) in b_ids:
            calls.append((left, right, result))
        elif key(right.provenance.execution) in a_ids and key(left.provenance.execution) in b_ids:
            calls.append((right, left, result))
        return result
    patch_public(monkeypatch, m11, "compare_metric_results", traced,
                 importlib.import_module("experiment_group_comparison"))
    row = invoke(function, data)["targets"]["quality"]
    assert len(calls) == size * size
    assert {(key(a.provenance.execution), key(b.provenance.execution)) for a, b, _ in calls} == {
        (a, b) for a in a_ids for b in b_ids}
    for a, b, assessment in calls:
        assert type(a) is type(b) is MetricResultProvenance
        assert assessment.status == "comparable"
        for actual, group in ((a, data[0]), (b, data[1])):
            snapshot = next(x for x in group["executions"] if x["execution"] == actual.provenance.execution)
            member = next(x for x in group["targets"][0]["members"] if x["execution"] == actual.provenance.execution)
            assert actual.provenance == snapshot["provenance"]
            assert actual.value == snapshot["evidence"]["test_results"]["metrics"]["score"]["value"]
            for field in ("metric_name", "metric_definition", "aggregation", "evaluation_protocol", "selection_protocol"):
                assert getattr(actual, field) == member[field]
            assert actual.direction == member["metric_spec"].direction
            assert actual.checkpoint_ref is None
    assert row["metric_comparability"]["assessed_pair_count"] == size * size
    assert row["comparison"]["raw_delta"] == 100


@pytest.mark.parametrize("field", ["task", "target", "dataset_version", "split_id",
                                    "metric_definition", "aggregation", "evaluation_protocol", "selection_protocol"])
@pytest.mark.parametrize("mode", ["conflict", "unknown"])
def test_cross_group_policy_is_not_inferred_from_within_group_comparability(field, mode):
    function, data = core_api(), pair()
    replacement = "different" if mode == "conflict" else None
    for snapshot in data[1]["executions"]:
        if field in ("task", "target", "dataset_version", "split_id"):
            snapshot["provenance"] = replace(snapshot["provenance"], **{field: replacement})
    for member in data[1]["targets"][0]["members"]:
        if field in member: member[field] = replacement
    row = invoke(function, data)["targets"]["quality"]
    assert row["metric_comparability"]["status"] == ("incompatible" if mode == "conflict" else "unknown")
    reason = f"{field}_mismatch" if mode == "conflict" else f"missing_{field}"
    assert reason in row["metric_comparability"]["reason_codes"]
    assert row["comparison"]["eligibility"] == ("blocked" if mode == "conflict" else "insufficient")
    assert row["comparison"]["raw_delta"] is None


@pytest.mark.parametrize("unavailable", ["snapshot", "metric", "invalid"])
def test_last_member_conflict_is_checked_even_without_numeric_value(monkeypatch, unavailable):
    function, data, seen = core_api(), pair(), []
    snapshot = data[1]["executions"][-1]
    data[1]["targets"][0]["members"][-1]["metric_definition"] = "conflicting-definition"
    if unavailable == "snapshot": snapshot.update(status="unavailable", evidence=None)
    elif unavailable == "metric": snapshot["evidence"]["test_results"]["metrics"].clear()
    else: snapshot["evidence"]["test_results"]["metrics"]["score"]["value"] = float("nan")
    original = m11.compare_metric_results
    def traced(left, right):
        if left.provenance.execution.experiment != right.provenance.execution.experiment:
            seen.append((left, right))
        return original(left, right)
    patch_public(monkeypatch, m11, "compare_metric_results", traced,
                 importlib.import_module("experiment_group_comparison"))
    row = invoke(function, data)["targets"]["quality"]
    assert len(seen) == 9
    unavailable_operands = [x for operands in seen for x in operands
                            if x.provenance.execution == snapshot["execution"]]
    assert len(unavailable_operands) == 3 and all(x.value is None for x in unavailable_operands)
    assert row["metric_comparability"]["status"] == "incompatible"
    assert "metric_definition_mismatch" in row["metric_comparability"]["reason_codes"]
    assert row["comparison"]["eligibility"] == "blocked" and row["comparison"]["raw_delta"] is None


def test_absent_snapshot_does_not_fabricate_provenance_or_hide_roster_slot():
    function, data = core_api(), pair()
    data[1]["executions"].pop()
    # An absent selector alone is not fabricated into a supplied provenance.
    data[1]["targets"][0]["members"][-1]["metric_definition"] = "absent-declaration"
    row = invoke(function, data)["targets"]["quality"]
    assert row["metric_comparability"]["assessed_pair_count"] == 6
    assert row["metric_comparability"]["status"] == "comparable"
    assert row["b"]["coverage"]["missing_count"] == 1
    assert row["comparison"]["eligibility"] == "limited"


def test_validation_pair_uses_validation_values_epochs_and_no_checkpoint_inference(monkeypatch):
    function, data, seen = core_api(), pair(section="validation"), []
    for side in data[:2]:
        for i, snapshot in enumerate(side["executions"]):
            snapshot["evidence"]["test_results"]["metrics"]["score"]["value"] = 999
            snapshot["evidence"]["validation_metrics"]["score"]["best_epoch"] = 10 + i
    original = m11.compare_metric_results
    def traced(left, right):
        if left.provenance.execution.experiment != right.provenance.execution.experiment:
            seen.append((left, right))
        return original(left, right)
    patch_public(monkeypatch, m11, "compare_metric_results", traced,
                 importlib.import_module("experiment_group_comparison"))
    result = invoke(function, data)
    assert len(seen) == 9
    snapshots = {s["execution"]: s for group in data[:2] for s in group["executions"]}
    for operands in seen:
        for operand in operands:
            metric = snapshots[operand.provenance.execution]["evidence"]["validation_metrics"]["score"]
            assert operand.value == metric["best_value"]
            assert operand.best_epoch == metric["best_epoch"]
            assert operand.checkpoint_ref is None
    row = result["targets"]["quality"]
    assert row["comparison"]["raw_delta"] == 1.
    assert row["a"]["included_executions"][0]["source"]["evidence_kind"] == "validation_history"


def test_missing_whole_group_has_no_cross_m11_calls(monkeypatch):
    function, data, seen = core_api(), pair(), []
    data[1]["executions"].clear()
    original = m11.compare_metric_results
    def traced(left, right):
        seen.append((left.provenance.execution, right.provenance.execution))
        return original(left, right)
    patch_public(monkeypatch, m11, "compare_metric_results", traced,
                 importlib.import_module("experiment_group_comparison"))
    row = invoke(function, data)["targets"]["quality"]
    assert seen and all(a.experiment == b.experiment == data[0]["group_declaration"]["experiment_identity"]
                        for a, b in seen)
    assert row["metric_comparability"]["assessed_pair_count"] == 0
    assert row["comparison"]["eligibility"] == "insufficient"
    assert row["comparison"]["raw_delta"] is None


@pytest.mark.parametrize("case", ["empty", "duplicate-id", "duplicate-a", "duplicate-b", "missing", "unknown-a", "unknown-b", "extra-field", "blank-id"])
def test_explicit_mapping_rejects_malformed_or_incomplete_selection(case):
    function, data = core_api(), many_targets(pair(), 2)
    mapping = data[2]
    if case == "empty": mapping.clear()
    elif case == "duplicate-id": mapping[1]["target_id"] = mapping[0]["target_id"]
    elif case in ("duplicate-a", "duplicate-b"):
        field = case[-1] + "_target_id"
        mapping[1][field] = mapping[0][field]
    elif case == "missing": mapping.pop()
    elif case.startswith("unknown-"): mapping[0][case[-1] + "_target_id"] = "unknown"
    elif case == "extra-field": mapping[0]["pair_mapping"] = []
    else: mapping[0]["target_id"] = " "
    with pytest.raises((TypeError, ValueError)): invoke(function, data)


def test_explicit_mapping_uses_ids_not_same_names_or_positions():
    function, data = core_api(), many_targets(pair(), 2)
    data[0]["targets"][1]["section"] = "validation"
    data[1]["targets"][0]["section"] = "validation"
    for side, index in ((data[0], 1), (data[1], 0)):
        for m in side["targets"][index]["members"]:
            m["metric_spec"] = replace(m["metric_spec"], path=("validation", "measurement"))
    data[2][0]["b_target_id"], data[2][1]["b_target_id"] = "t-01", "t-00"
    result = invoke(function, data)
    assert [r["alignment"]["status"] for r in result["targets"].values()] == ["aligned", "aligned"]
    assert result["targets"]["comparison-00"]["b"]["source_target_id"] == "t-01"


def test_section_mismatch_blocks_without_cross_section_fallback():
    function, data = core_api(), pair()
    data[1]["targets"][0]["section"] = "validation"
    row = invoke(function, data)["targets"]["quality"]
    assert row["alignment"]["status"] == "blocked"
    assert "section_mismatch" in row["alignment"]["reason_codes"]
    assert row["comparison"]["eligibility"] == "blocked"
    assert row["comparison"]["raw_delta"] is None


@pytest.mark.parametrize("supplied", [True, False])
def test_full_execution_identity_overlap_blocks_even_expected_only(supplied):
    function, data = core_api(), pair(same_experiment=True)
    a, b, _ = data
    shared = a["executions"][0]["execution"]
    b["group_declaration"]["expected_executions"][0]["execution"] = shared
    b["targets"][0]["members"][0]["execution"] = shared
    b["executions"][0]["execution"] = shared
    b["executions"][0]["provenance"] = replace(b["executions"][0]["provenance"], execution=shared)
    if not supplied:
        a["executions"].pop(0)
        b["executions"].pop(0)
    row = invoke(function, data)["targets"]["quality"]
    assert row["comparison"]["eligibility"] == "blocked"
    assert "overlapping_execution_identity" in row["comparison"]["reason_codes"]
    assert row["comparison"]["raw_delta"] is None
    assert row["a"]["coverage"]["expected_count"] == row["b"]["coverage"]["expected_count"] == 3


def test_same_execution_label_in_different_experiments_is_not_overlap():
    function, data = core_api(), pair()
    for i, snapshot in enumerate(data[1]["executions"]):
        execution = replace(snapshot["execution"], execution_id=data[0]["executions"][i]["execution"].execution_id)
        snapshot["execution"] = execution
        snapshot["provenance"] = replace(snapshot["provenance"], execution=execution)
        data[1]["group_declaration"]["expected_executions"][i]["execution"] = execution
        data[1]["targets"][0]["members"][i]["execution"] = execution
    assert invoke(function, data)["targets"]["quality"]["comparison"]["eligibility"] == "eligible"


@pytest.mark.parametrize("direction,values,delta,interpretation", [
    ("maximize", (2., 4., 6.), 1., "b_favorable"),
    ("maximize", (0., 2., 4.), -1., "a_favorable"),
    ("minimize", (2., 4., 6.), 1., "a_favorable"),
    ("minimize", (0., 2., 4.), -1., "b_favorable"),
    ("maximize", (1., 3., 5.), 0., "equal"),
    ("minimize", (1., 3., 5.), 0., "equal"),
])
def test_delta_b_minus_a_direction_and_swap(direction, values, delta, interpretation):
    function, data = core_api(), pair(b_values=values)
    for side in data[:2]: set_direction(side, direction)
    row = invoke(function, data)["targets"]["quality"]
    assert row["comparison"]["raw_delta"] == delta
    assert row["comparison"]["direction"] == direction
    assert row["comparison"]["direction_interpretation"] == interpretation
    swapped = function(data[1], data[0], targets=[{
        "target_id": "quality", "a_target_id": "quality-B", "b_target_id": "quality-A"}])
    assert swapped["targets"]["quality"]["comparison"]["raw_delta"] == -delta


@pytest.mark.parametrize("direction,status", [(None, "unknown"), ("minimize", "incompatible")])
def test_unknown_or_conflicting_direction_never_interpreted(direction, status):
    function, data = core_api(), pair()
    set_direction(data[1], direction)
    row = invoke(function, data)["targets"]["quality"]
    assert row["metric_comparability"]["status"] == status
    assert row["comparison"]["raw_delta"] is None
    assert row["comparison"]["direction_interpretation"] is None


@pytest.mark.parametrize("expected_a,observed_a,expected_b,observed_b,eligibility,delta", [
    (3, 3, 3, 3, "eligible", 10.), (3, 3, 3, 2, "limited", 9.5),
    (3, 2, 3, 2, "limited", 10.), (3, 3, 5, 5, "limited", 11.),
    (1, 1, 1, 1, "limited", 10.), (3, 0, 3, 3, "insufficient", None),
])
def test_target_coverage_controls_eligibility_and_own_means(expected_a, observed_a, expected_b, observed_b, eligibility, delta):
    function = core_api()
    data = pair(tuple(range(expected_a)), tuple(range(10, 10 + expected_b)))
    del data[0]["executions"][observed_a:]
    del data[1]["executions"][observed_b:]
    row = invoke(function, data)["targets"]["quality"]
    for label, side, expected, observed in (("a", data[0], expected_a, observed_a), ("b", data[1], expected_b, observed_b)):
        old = aggregate(side)["targets"][side["targets"][0]["target_id"]]
        assert row[label]["coverage"] == old["coverage"]
        assert row[label]["statistics"] == old["statistics"]
        assert row[label]["coverage"]["missing_count"] == expected - observed
        if observed < 2: assert row[label]["statistics"]["sample_std"] is None
    comparison = row["comparison"]
    assert comparison["eligibility"] == eligibility and comparison["raw_delta"] == delta
    assert comparison["coverage_relation"] == {
        "a_complete": observed_a == expected_a, "b_complete": observed_b == expected_b,
        "same_expected_count": expected_a == expected_b, "same_included_count": observed_a == observed_b}
    if expected_a != expected_b: assert "expected_roster_size_mismatch" in comparison["reason_codes"]
    if (expected_a, observed_a) != (expected_b, observed_b): assert "coverage_mismatch" in comparison["reason_codes"]
    if min(observed_a, observed_b) == 1: assert "insufficient_variability" in comparison["reason_codes"]
    if observed_a == 0:
        assert row["metric_comparability"]["assessed_pair_count"] == 0
        assert row["metric_comparability"]["status"] == "unknown"
        assert "no_supplied_cross_pairs" in row["metric_comparability"]["reason_codes"]


def test_target_specific_coverage_never_uses_group_intersection():
    function, data = core_api(), many_targets(pair(), 2)
    for side in data[:2]:
        side["targets"][1]["section"] = "validation"
        for member in side["targets"][1]["members"]:
            member["metric_spec"] = replace(member["metric_spec"], path=("validation", "measurement"))
    data[0]["executions"][0]["evidence"]["test_results"]["metrics"].clear()
    data[0]["executions"][2]["evidence"]["validation_metrics"].clear()
    result = invoke(function, data)
    assert result["groups"]["a"]["coverage"]["usable_count"] == 1
    assert result["targets"]["comparison-00"]["a"]["statistics"]["mean"] == 4.
    assert result["targets"]["comparison-01"]["a"]["statistics"]["mean"] == 2.
    assert result["targets"]["comparison-00"]["comparison"]["raw_delta"] == 0.
    assert result["targets"]["comparison-01"]["comparison"]["raw_delta"] == 2.


def test_m18_block_reasons_and_statistics_are_preserved():
    function, data = core_api(), pair()
    data[0]["executions"][1]["training_protocol_ref"] = "different"
    original = aggregate(data[0])["targets"]["quality-A"]
    row = invoke(function, data)["targets"]["quality"]
    assert row["a"]["target_status"] == "blocked"
    assert row["a"]["aggregation_eligibility"] == original["aggregation_eligibility"]
    assert set(original["aggregation_eligibility"]["reason_codes"]) <= set(row["comparison"]["reason_codes"])
    assert row["comparison"]["eligibility"] == "blocked" and row["comparison"]["raw_delta"] is None


@pytest.mark.parametrize("side", [0, 1])
def test_unsupported_group_policy_rejects(side):
    function, data = core_api(), pair()
    data[side]["group_declaration"]["aggregation_policy"] = "sample-weighted-v2"
    with pytest.raises(ValueError): invoke(function, data)


def test_supplied_snapshots_with_no_metrics_are_insufficient_not_zero():
    function, data = core_api(), pair()
    for snapshot in data[1]["executions"]: snapshot["evidence"]["test_results"]["metrics"].clear()
    row = invoke(function, data)["targets"]["quality"]
    assert row["b"]["statistics"]["count"] == 0 and row["b"]["statistics"]["mean"] is None
    assert row["comparison"]["eligibility"] == "insufficient" and row["comparison"]["raw_delta"] is None


def test_provenance_limits_same_seed_and_no_inference_or_pairing():
    function, data = core_api(), pair()
    # Both groups use the same three declared seeds; this is not a pairing map.
    result = invoke(function, data)
    for label in ("a", "b"):
        side = result["targets"]["quality"][label]
        assert ROOT_LIMITATIONS | {"checkpoint_binding_not_verified"} <= set(side["limitations"])
        for member in side["included_executions"]:
            source = member["source"]
            assert (source["evidence_kind"], source["metadata_source"]) == ("csv_reported", "host_declared")
            assert source["provenance"]["checkpoint_binding"] == "not_verified"
            assert source["provenance"]["checkpoint_identity"] is None
            assert source["provenance"]["recomputed_from_predictions"] is False
    assert_no_claims(result)
    assert result["targets"]["quality"]["a"]["statistics"]["sample_std"] == 2.
    with pytest.raises(TypeError): function(data[0], data[1], targets=data[2], pair_mapping=[])


@pytest.mark.parametrize("bad", [True, False, float("nan"), float("inf"), -float("inf")])
def test_invalid_measurement_is_local_exclusion_and_strict_json(bad):
    function, data = core_api(), pair()
    data[1]["executions"][-1]["evidence"]["test_results"]["metrics"]["score"]["value"] = bad
    result = invoke(function, data)
    row = result["targets"]["quality"]
    assert row["b"]["coverage"]["excluded_count"] == 1
    assert row["b"]["statistics"]["mean"] == 3.
    assert row["comparison"]["raw_delta"] == 0. and row["comparison"]["eligibility"] == "limited"
    json.dumps(result, allow_nan=False)


@pytest.mark.parametrize("a,b,overflow", [(-1e308, 1e308, True), (1e308, 1e308, False), (-1e300, 1e300, False)])
def test_extreme_finite_mean_subtraction(a, b, overflow):
    function, data = core_api(), pair((a,), (b,))
    result = invoke(function, data)
    comparison = result["targets"]["quality"]["comparison"]
    if overflow:
        assert comparison["eligibility"] == "blocked"
        assert comparison["raw_delta"] is None and "numeric_overflow" in comparison["reason_codes"]
    else:
        assert comparison["raw_delta"] == b - a and math.isfinite(comparison["raw_delta"])
    json.dumps(result, allow_nan=False)


def test_order_invariance_input_immutability_and_return_isolation():
    function, data = core_api(), many_targets(pair(), 2)
    pristine = deepcopy(data)
    first = invoke(function, data)
    assert data == pristine
    reordered = deepcopy(data)
    reordered[2].reverse()
    for side in reordered[:2]:
        side["executions"].reverse()
        side["group_declaration"]["expected_executions"].reverse()
        side["targets"].reverse()
        for target in side["targets"]: target["members"].reverse()
    assert invoke(function, reordered) == first
    first["targets"]["comparison-00"]["a"]["statistics"]["mean"] = -999
    assert invoke(function, data) == invoke(function, pristine)
    assert data == pristine


def test_reason_order_invariance_with_multiple_cross_conflicts():
    function, data = core_api(), pair()
    for member in data[1]["targets"][0]["members"]:
        member.update(metric_definition="different", evaluation_protocol=None, aggregation="different")
    first = invoke(function, data)
    for side in data[:2]: side["executions"].reverse()
    second = invoke(function, data)
    assert first == second
    for field in ("alignment", "metric_comparability", "comparison"):
        reasons = first["targets"]["quality"][field]["reason_codes"]
        assert reasons == sorted(set(reasons))


@pytest.mark.parametrize("kind", ["expected-zero", "expected-17", "supplied-17", "targets-33", "group-extra", "aggregate-only"])
def test_shape_and_cardinality_bounds(kind):
    function, data = core_api(), pair()
    if kind == "expected-zero": data[0]["group_declaration"]["expected_executions"].clear()
    elif kind == "expected-17": data = pair(tuple(range(17)))
    elif kind == "supplied-17": data[0]["executions"] *= 6
    elif kind == "targets-33": many_targets(data, 33)
    elif kind == "group-extra": data[0]["cached_summary"] = {}
    else: data = (aggregate(data[0]), aggregate(data[1]), data[2])
    with pytest.raises((TypeError, ValueError)): invoke(function, data)


def test_maximum_target_count_is_accepted_with_bounded_small_rosters():
    result = invoke(core_api(), many_targets(pair((1.,), (2.,)), 32))
    assert len(result["targets"]) == 32
    assert json_size(result) <= 128 * 1024


@pytest.mark.parametrize("extra", [0, 1])
@pytest.mark.parametrize("boundary", ["group", "mapping", "total"])
def test_inclusive_input_byte_budgets(boundary, extra):
    function, data = core_api(), pair((1.,), (2.,))
    if boundary == "mapping":
        data[2][0]["target_id"] = ""
        data[2][0]["target_id"] = "x" * (16 * 1024 - json_size(data[2]) + extra)
        assert json_size(data[2]) == 16 * 1024 + extra
    elif boundary == "group":
        meta = data[0]["executions"][0]["snapshot_metadata"]
        meta["padding"] = ""
        meta["padding"] = "x" * (2 * 1024 * 1024 - json_size(data[0]) + extra)
        assert json_size(data[0]) == 2 * 1024 * 1024 + extra
    else:
        envelope = {"group_a": data[0], "group_b": data[1], "targets": data[2]}
        for side in data[:2]:
            side["executions"][0]["snapshot_metadata"]["padding"] = ""
        meta_a = data[0]["executions"][0]["snapshot_metadata"]
        meta_b = data[1]["executions"][0]["snapshot_metadata"]
        meta_a["padding"] = "x" * (2 * 1024 * 1024 - json_size(data[0]))
        meta_b["padding"] = "x" * (4 * 1024 * 1024 - json_size(envelope) + extra)
        assert all(json_size(side) <= 2 * 1024 * 1024 for side in data[:2])
        assert json_size(envelope) == 4 * 1024 * 1024 + extra
    if extra:
        with pytest.raises(ValueError, match="input|mapping"): invoke(function, data)
    else:
        assert next(iter(invoke(function, data)["targets"].values()))["comparison"]["raw_delta"] == 1.


@pytest.mark.parametrize("kind", ["depth", "cycle"])
def test_unbounded_structure_rejects(kind):
    function, data, nested = core_api(), pair(), {}
    if kind == "cycle": nested["self"] = nested
    else:
        for _ in range(17): nested = {"child": nested}
    data[1]["executions"][0]["snapshot_metadata"]["payload"] = nested
    with pytest.raises(ValueError): invoke(function, data)


@pytest.mark.parametrize("extra", [0, 1])
def test_group_depth_count_starts_at_its_own_envelope(extra):
    function, data, nested = core_api(), pair(), {}
    # group=1, executions=2, snapshot=3, metadata=4, payload=5.
    for _ in range(11 + extra): nested = {"child": nested}
    data[0]["executions"][0]["snapshot_metadata"]["payload"] = nested
    if extra:
        with pytest.raises(ValueError): invoke(function, data)
    else:
        assert invoke(function, data)["targets"]["quality"]["comparison"]["raw_delta"] == 1.


def test_utf8_bytes_not_character_count_for_group_budget():
    function, data = core_api(), pair()
    data[0]["executions"][0]["snapshot_metadata"]["payload"] = "\u6d4b" * 800000
    assert json_size(data[0]) > 2 * 1024 * 1024
    with pytest.raises(ValueError, match="input"): invoke(function, data)


def output_overflow_fixture():
    data = pair((1.,), (2.,))
    for side in data[:2]:
        side["executions"][0]["evidence"]["test_results"]["source"]["file"] = "x" * 70000
    return data


def test_m19_output_budget_not_merely_each_m18_output_budget():
    function, data = core_api(), output_overflow_fixture()
    assert all(json_size(aggregate(side)) < 128 * 1024 for side in data[:2])
    with pytest.raises(ValueError, match="output"): invoke(function, data)


def test_core_no_io_network_memory_registry_or_reader(monkeypatch):
    function, data = core_api(), pair()
    with deny_external_access(monkeypatch) as calls:
        result = invoke(function, data)
    assert calls == [] and result["targets"]["quality"]["comparison"]["raw_delta"] == 1.


# These fixture/delegation counterexamples run even when M19 is absent.
def test_fixture_statistics_and_cross_conflict_are_independently_real():
    data = pair()
    a, b = [aggregate(side) for side in data[:2]]
    assert a["targets"]["quality-A"]["statistics"]["mean"] == 3.
    assert b["targets"]["quality-B"]["statistics"]["mean"] == 4.
    changed = deepcopy(data[1])
    for m in changed["targets"][0]["members"]: m["metric_definition"] = "other-definition"
    assert aggregate(changed) == b  # Aggregate summaries alone erase this distinction.
    operands = []
    for side in (data[0], changed):
        m, s = side["targets"][0]["members"][0], side["executions"][0]
        operands.append(MetricResultProvenance("score", s["evidence"]["test_results"]["metrics"]["score"]["value"],
            s["provenance"], metric_definition=m["metric_definition"], direction=m["metric_spec"].direction,
            aggregation=m["aggregation"], evaluation_protocol=m["evaluation_protocol"], selection_protocol=m["selection_protocol"]))
    assert m11.compare_metric_results(*operands).status == "incompatible"


def test_output_overflow_fixture_does_not_fail_m18_first():
    data = output_overflow_fixture()
    sizes = [json_size(aggregate(side)) for side in data[:2]]
    assert all(size < 128 * 1024 for size in sizes)
    assert sum(sizes) > 128 * 1024

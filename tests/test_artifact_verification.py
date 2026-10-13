"""M20 deterministic descriptor contracts; synthetic values, no live verification."""

from copy import deepcopy
import inspect
import json

import pytest

from m20_contract_helpers import (
    LIMITATIONS, SHA_A, SHA_B, assert_limits, core_api, data, deny_external, size,
)


def test_public_signature():
    signature = inspect.signature(core_api())
    assert list(signature.parameters) == ["declared_set", "reported_observations"]
    for parameter in signature.parameters.values():
        assert parameter.kind == inspect.Parameter.POSITIONAL_OR_KEYWORD
        assert parameter.default is inspect.Parameter.empty


def test_handwritten_complete_projection_and_fixed_limits():
    function, (declared, reported) = core_api(), data()
    result = function(declared, reported)
    assert_limits(result)
    assert result == {
        "schema_version": "artifact-verification-report-v1", "set_id": "bundle-1",
        "declared_by": "host-A", "execution_ref": {
            "namespace": "lab", "experiment_id": "generic", "execution_id": "run-1"},
        "data_snapshot_ref": "data-1", "evidence_basis": "host_reported",
        "artifacts": {"metrics": {
            "logical_role": "metrics",
            "declared": {"locator": "metrics.csv", "byte_size": 12, "sha256": SHA_A},
            "reported": {"observation_id": "report-1", "reported_by": "host-B",
                         "reported_at": "2026-01-01T00:00:00Z", "reported_status": "observed",
                         "locator": "metrics.csv", "byte_size": 12, "sha256": SHA_A, "mtime_ns": 123},
            "identity_status": "aligned_host_key", "reported_status": "observed",
            "integrity": {"sha256_comparison": "match", "size_comparison": "match"},
            "locator_comparison": "same", "currentness_status": "not_checked",
            "experiment_membership_status": "host_declared", "checkpoint_binding_status": "not_verified",
            "recomputation_status": "not_performed_by_copilot", "evidence_basis": "host_reported",
            "reason_codes": ["locator_match", "sha256_match", "size_match"], "limitations": LIMITATIONS}},
        "limitations": LIMITATIONS}


@pytest.mark.parametrize("field,left,right,expected", [
    ("sha256", SHA_A, SHA_A, "match"), ("sha256", SHA_A.upper(), SHA_A, "match"),
    ("sha256", SHA_A, SHA_B, "mismatch"), ("sha256", None, SHA_A, "unknown"),
    ("sha256", SHA_A, None, "unknown"), ("sha256", None, None, "unknown"),
    ("byte_size", 0, 0, "match"), ("byte_size", 12, 13, "mismatch"),
    ("byte_size", 2**63-1, 2**63-1, "match"), ("byte_size", None, 12, "unknown"),
    ("byte_size", 12, None, "unknown"), ("byte_size", None, None, "unknown"),
    ("locator", "", "", "same"), ("locator", "x", "x", "same"),
    ("locator", "a/../b", "b", "different"), ("locator", "A", "a", "different"),
    ("locator", None, "x", "unknown"), ("locator", "x", None, "unknown"),
    ("locator", None, None, "unknown"),
])
def test_independent_field_comparison(field, left, right, expected):
    function, (declared, reported) = core_api(), data()
    declared["artifacts"][0][field], reported[0][field] = left, right
    report = function(declared, reported)
    assert_limits(report)
    row = report["artifacts"]["metrics"]
    if field == "locator":
        assert row["locator_comparison"] == expected
        code = {"same": "locator_match", "different": "locator_mismatch", "unknown": "locator_unknown"}[expected]
    else:
        prefix = "sha256" if field == "sha256" else "size"
        assert row["integrity"][prefix + "_comparison"] == expected
        code = prefix + "_" + expected
    assert code in row["reason_codes"]
    if field == "sha256" and left is not None:
        assert row["declared"]["sha256"] == left.lower()


@pytest.mark.parametrize("status", ["missing", "unauthorized", "unreadable", "unobserved"])
def test_nonobserved_statuses_remain_distinct(status):
    function, (declared, reported) = core_api(), data()
    reported[0].update(reported_status=status, byte_size=None, sha256=None, mtime_ns=None)
    row = function(declared, reported)["artifacts"]["metrics"]
    assert row["reported_status"] == row["reported"]["reported_status"] == status
    assert row["identity_status"] == "aligned_host_key"
    assert row["integrity"] == {"sha256_comparison": "not_evaluated", "size_comparison": "not_evaluated"}
    assert row["locator_comparison"] == "not_evaluated"
    assert row["reported"]["locator"] == "metrics.csv"
    assert row["reason_codes"] == sorted([
        "reported_" + status, "locator_not_evaluated", "sha256_not_evaluated", "size_not_evaluated"])


def test_no_observation_does_not_mean_missing_file():
    function, (declared, _) = core_api(), data()
    row = function(declared, [])["artifacts"]["metrics"]
    assert row["identity_status"] == "missing_observation"
    assert row["reported"] is None and row["reported_status"] == "unobserved"
    assert row["reason_codes"] == ["locator_not_evaluated", "observation_missing",
                                   "sha256_not_evaluated", "size_not_evaluated"]


@pytest.mark.parametrize("status", ["missing", "unauthorized", "unreadable", "unobserved"])
@pytest.mark.parametrize("field,value", [("sha256", SHA_A), ("byte_size", 0), ("mtime_ns", 0)])
def test_nonobserved_integrity_fields_reject(status, field, value):
    function, (declared, reported) = core_api(), data()
    reported[0].update(reported_status=status, byte_size=None, sha256=None, mtime_ns=None)
    reported[0][field] = value
    with pytest.raises((TypeError, ValueError)):
        function(declared, reported)


def test_partial_matrix_and_conflicting_dimensions_do_not_collapse():
    function, (declared, reported) = core_api(), data()
    declared["artifacts"].append({"artifact_id": "checkpoint", "logical_role": "checkpoint_ref"})
    reported[0]["byte_size"] = 99
    result = function(declared, reported)
    assert_limits(result)
    assert result["artifacts"]["metrics"]["integrity"] == {
        "sha256_comparison": "match", "size_comparison": "mismatch"}
    assert result["artifacts"]["checkpoint"]["reported_status"] == "unobserved"


def test_explicit_id_alignment_and_order_independence():
    function, (declared, reported) = core_api(), data()
    # Same role/locator/hash must neither collapse nor join these distinct IDs.
    declared["artifacts"].append(dict(declared["artifacts"][0], artifact_id="config", byte_size=50))
    reported.append(dict(reported[0], artifact_id="config", byte_size=50))
    first = function(declared, list(reversed(reported)))
    declared["artifacts"].reverse()
    second = function(declared, reported)
    assert first == second and list(first["artifacts"]) == ["config", "metrics"]
    assert all(r["integrity"]["size_comparison"] == "match" for r in first["artifacts"].values())
    assert first["artifacts"]["config"]["reported"]["observation_id"] == "report-1"


@pytest.mark.parametrize("mode", ["duplicate-artifact", "duplicate-observation", "unknown-artifact", "case-mismatch"])
def test_invalid_alignment_is_not_repaired_by_equal_locator_role_hash_or_position(mode):
    function, (declared, reported) = core_api(), data()
    if mode == "duplicate-artifact": declared["artifacts"].append(deepcopy(declared["artifacts"][0]))
    elif mode == "duplicate-observation": reported.append(dict(reported[0], observation_id="other"))
    else: reported[0]["artifact_id"] = "METRICS" if mode == "case-mismatch" else "unknown"
    with pytest.raises((TypeError, ValueError)):
        function(declared, reported)


@pytest.mark.parametrize("placement", ["set", "artifact", "observation", "execution"])
@pytest.mark.parametrize("field", ["unknown", "verified", "trusted", "authenticated", "system_observed",
                                   "checkpoint_verified", "current", "producer_verified", "reader_verified"])
def test_strict_fields_block_host_trust_escalation(placement, field):
    function, (declared, reported) = core_api(), data()
    target = {"set": declared, "artifact": declared["artifacts"][0],
              "observation": reported[0], "execution": declared["execution_ref"]}[placement]
    target[field] = True
    with pytest.raises((TypeError, ValueError)):
        function(declared, reported)


@pytest.mark.parametrize("placement,field", [
    ("set", "schema_version"), ("set", "set_id"), ("set", "declared_by"), ("set", "artifacts"),
    ("artifact", "artifact_id"), ("artifact", "logical_role"),
    ("observation", "artifact_id"), ("observation", "observation_id"),
    ("observation", "reported_by"), ("observation", "reported_status"),
    ("execution", "namespace"), ("execution", "experiment_id"), ("execution", "execution_id"),
])
def test_required_fields(placement, field):
    function, (declared, reported) = core_api(), data()
    target = {"set": declared, "artifact": declared["artifacts"][0],
              "observation": reported[0], "execution": declared["execution_ref"]}[placement]
    del target[field]
    with pytest.raises((TypeError, ValueError)):
        function(declared, reported)


@pytest.mark.parametrize("value", ["", " ", 1, True, None, "declared-artifact-set-v2"])
def test_schema_version_reject(value):
    function, (declared, reported) = core_api(), data()
    declared["schema_version"] = value
    with pytest.raises((TypeError, ValueError)):
        function(declared, reported)


@pytest.mark.parametrize("field", ["byte_size", "mtime_ns"])
@pytest.mark.parametrize("value", [True, False, -1, 2**63, 1.0, float("nan"), float("inf"), "12",
                                  pytest.param(10**10000, id="huge-int")])
def test_exact_bounded_observation_integer(field, value):
    function, (declared, reported) = core_api(), data()
    reported[0][field] = value
    with pytest.raises((TypeError, ValueError)):
        function(declared, reported)


@pytest.mark.parametrize("value", [True, -1, 2**63, 1.0, "12"])
def test_declared_size_reject(value):
    function, (declared, reported) = core_api(), data()
    declared["artifacts"][0]["byte_size"] = value
    with pytest.raises((TypeError, ValueError)):
        function(declared, reported)


@pytest.mark.parametrize("side", ["declared", "reported"])
@pytest.mark.parametrize("value", ["", "a"*63, "a"*65, "g"*64, "Ａ"*64, " a"*32, 123, True, b"a"*64])
def test_sha_requires_exact_ascii_hex(side, value):
    function, (declared, reported) = core_api(), data()
    (declared["artifacts"][0] if side == "declared" else reported[0])["sha256"] = value
    with pytest.raises((TypeError, ValueError)):
        function(declared, reported)


@pytest.mark.parametrize("status", [None, True, "OBSERVED", "current", "verified", "available", ""])
def test_status_enum_is_closed(status):
    function, (declared, reported) = core_api(), data()
    reported[0]["reported_status"] = status
    with pytest.raises((TypeError, ValueError)):
        function(declared, reported)


@pytest.mark.parametrize("field", ["set_id", "declared_by", "artifact_id", "logical_role", "observation_id", "reported_by", "reported_at"])
@pytest.mark.parametrize("value", ["", " ", 1, True, "界"*43])
def test_identifier_and_source_label_constraints(field, value):
    function, (declared, reported) = core_api(), data()
    if field in ("set_id", "declared_by"): target = declared
    elif field in ("artifact_id", "logical_role"): target = declared["artifacts"][0]
    else: target = reported[0]
    target[field] = value
    with pytest.raises((TypeError, ValueError)):
        function(declared, reported)


def test_optional_omission_equals_null_and_does_not_claim_membership():
    function, (declared, reported) = core_api(), data()
    for name in ("execution_ref", "data_snapshot_ref"): declared.pop(name)
    for name in ("locator", "byte_size", "sha256"): declared["artifacts"][0].pop(name)
    for name in ("reported_at", "locator", "byte_size", "sha256", "mtime_ns"): reported[0].pop(name)
    omitted = function(declared, reported)
    declared.update(execution_ref=None, data_snapshot_ref=None)
    declared["artifacts"][0].update(locator=None, byte_size=None, sha256=None)
    reported[0].update(reported_at=None, locator=None, byte_size=None, sha256=None, mtime_ns=None)
    assert omitted == function(declared, reported)
    row = omitted["artifacts"]["metrics"]
    assert row["experiment_membership_status"] == "unknown"
    assert row["integrity"] == {"sha256_comparison": "unknown", "size_comparison": "unknown"}
    assert row["reason_codes"] == ["locator_unknown", "sha256_unknown", "size_unknown"]


@pytest.mark.parametrize("field", ["execution_ref", "data_snapshot_ref"])
@pytest.mark.parametrize("value", [True, 0, [], "", " " ])
def test_optional_root_references_are_strict(field, value):
    function, (declared, reported) = core_api(), data()
    declared[field] = value
    with pytest.raises((TypeError, ValueError)):
        function(declared, reported)


@pytest.mark.parametrize("field", ["namespace", "experiment_id", "execution_id"])
@pytest.mark.parametrize("value", [None, True, "", " ", "x"*129])
def test_execution_reference_fields_are_exact_identifiers(field, value):
    function, (declared, reported) = core_api(), data()
    declared["execution_ref"][field] = value
    with pytest.raises((TypeError, ValueError)):
        function(declared, reported)


@pytest.mark.parametrize("mtime", [0, 2**63-1])
def test_reported_time_snapshot_label_and_checkpoint_role_do_not_upgrade(mtime):
    function, (declared, reported) = core_api(), data()
    declared["artifacts"][0]["logical_role"] = "checkpoint_ref"
    declared["data_snapshot_ref"] = "latest-and-memory-active-host-label"
    declared["execution_ref"] = None
    reported[0].update(mtime_ns=mtime, reported_at="latest-system-verified-host-claim",
                       reported_by="trusted-producer-host-label")
    result = function(declared, reported)
    assert_limits(result)
    row = result["artifacts"]["metrics"]
    assert row["experiment_membership_status"] == "unknown"
    assert row["reported"]["reported_at"] == "latest-system-verified-host-claim"
    assert row["reported"]["reported_by"] == "trusted-producer-host-label"


def test_bounded_ids_preserve_whitespace_and_unicode_without_normalization():
    function, (declared, reported) = core_api(), data()
    declared["set_id"] = "s"*128
    declared["declared_by"] = "h"*128
    declared["artifacts"][0].update(artifact_id=" id ", logical_role="r"*128)
    reported[0].update(artifact_id=" id ", observation_id="o"*128, reported_by="p"*128,
                       reported_at="t"*128)
    result = function(declared, reported)
    assert list(result["artifacts"]) == [" id "]
    assert result["set_id"] == "s"*128
    declared["artifacts"][0]["locator"] = "é"
    reported[0]["locator"] = "e\u0301"
    assert function(declared, reported)["artifacts"][" id "]["locator_comparison"] == "different"


@pytest.mark.parametrize("value", ["../never", "C:/private/model.ckpt", "a\x00b", "https://invalid.example/x"])
def test_locators_are_opaque_not_authorized_paths(monkeypatch, value):
    function, (declared, reported) = core_api(), data()
    declared["artifacts"][0]["locator"] = reported[0]["locator"] = value
    with deny_external(monkeypatch) as calls:
        result = function(declared, reported)
    assert calls == []
    assert result["artifacts"]["metrics"]["locator_comparison"] == "same"
    assert_limits(result)


def test_input_result_and_cross_row_isolation_and_json():
    function, arguments = core_api(), data()
    pristine = deepcopy(arguments)
    first = function(*arguments)
    assert arguments == pristine
    first["artifacts"]["metrics"]["declared"]["sha256"] = SHA_B
    first["execution_ref"]["namespace"] = "changed"
    first["limitations"].clear()
    second = function(*arguments)
    assert second["artifacts"]["metrics"]["declared"]["sha256"] == SHA_A
    assert arguments == pristine
    assert_limits(second)
    assert json.loads(json.dumps(second, allow_nan=False)) == second


@pytest.mark.parametrize("count,valid", [(0, False), (1, True), (32, True), (33, False)])
def test_artifact_cardinality(count, valid):
    function, (declared, _) = core_api(), data()
    declared["artifacts"] = [{"artifact_id": str(i), "logical_role": "config"} for i in range(count)]
    if valid: assert len(function(declared, [])["artifacts"]) == count
    else:
        with pytest.raises(ValueError): function(declared, [])


@pytest.mark.parametrize("count", [0, 32, 33])
def test_observation_cardinality(count):
    function, (declared, reported) = core_api(), data()
    declared["artifacts"] = [{"artifact_id": str(i), "logical_role": "config"} for i in range(32)]
    reports = [dict(reported[0], artifact_id=str(i)) for i in range(count)]
    if count <= 32: assert len(function(declared, reports)["artifacts"]) == 32
    else:
        with pytest.raises(ValueError): function(declared, reports)


@pytest.mark.parametrize("field,limit", [("set_id", 128), ("locator", 512)])
@pytest.mark.parametrize("extra", [0, 1])
def test_utf8_byte_limits_are_inclusive(field, limit, extra):
    function, (declared, reported) = core_api(), data()
    value = "é" * (limit // 2) + "x" * extra
    if field == "set_id": declared[field] = value
    else: declared["artifacts"][0][field] = value
    if extra:
        with pytest.raises(ValueError): function(declared, reported)
    else: assert function(declared, reported)["set_id"] == declared["set_id"]


@pytest.mark.parametrize("placement", ["descriptor", "observation", "combined"])
def test_oversized_inputs_are_rejected_not_truncated(placement):
    function, (declared, reported) = core_api(), data()
    if placement == "descriptor": declared["artifacts"][0]["locator"] = "x" * 2049
    elif placement == "observation": reported[0]["reported_by"] = "x" * 2049
    else: declared["data_snapshot_ref"] = "x" * 131073
    with pytest.raises(ValueError): function(declared, reported)


def test_valid_large_projection_cannot_escape_output_budget():
    function, (declared, reported) = core_api(), data()
    declared["artifacts"] = [dict(declared["artifacts"][0], artifact_id=str(i),
                                  logical_role="r"*128, locator="x"*512) for i in range(32)]
    reported = [dict(reported[0], artifact_id=str(i), locator="y"*512,
                     observation_id="o"*128, reported_by="s"*128, reported_at="t"*128) for i in range(32)]
    assert all(size(x) <= 2048 for x in declared["artifacts"] + reported)
    assert size({"declared_set": declared, "reported_observations": reported}) <= 131072
    with pytest.raises(ValueError, match="output"):
        function(declared, reported)

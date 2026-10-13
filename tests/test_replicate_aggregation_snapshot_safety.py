"""Post-Slice-3 safety contracts: data snapshots must never invoke copy hooks.

All probes are in-memory counters. Extra attributes on approved dataclasses
are outside the input contract; declared fields still reject unsupported data.
Original M18 fixtures and real core arithmetic remain authoritative.
"""

from dataclasses import dataclass
import json

import pytest

import experiment_replicates as core
from experiment_identity import ExperimentProvenance
from m18_contract_helpers import TOOL, bundle, input_envelope, json_size, make_tools
from tool_layer.replicate_tools import ReplicateAggregationTools


def selected_object(data, kind):
    return {
        "provenance": data[0][0]["provenance"],
        "execution": data[0][0]["execution"],
        "experiment": data[1]["experiment_identity"],
        "spec": data[2][0]["members"][0]["metric_spec"],
    }[kind]


@pytest.mark.parametrize("kind", ["provenance", "execution", "experiment", "spec"])
def test_extra_attribute_copy_callback_never_runs(kind):
    data, calls = bundle(), []
    class CopyProbe:
        def __deepcopy__(self, memo):
            calls.append("copy callback")
            return self
    object.__setattr__(selected_object(data, kind), "_extra_metadata", CopyProbe())
    tools = make_tools(ReplicateAggregationTools, data)
    assert calls == [], f"extra attribute callback executed {len(calls)} time(s)"
    result = tools.invoke_tool(TOOL, {})
    assert calls == []
    assert result["targets"]["quality"]["statistics"]["mean"] == 3.0


@pytest.mark.parametrize("kind", ["provenance", "execution", "experiment", "spec"])
def test_extra_attributes_never_reach_core_or_result(monkeypatch, kind):
    data = bundle()
    original = selected_object(data, kind)
    object.__setattr__(original, "_extra_metadata", {"marker": "extra-attribute-sentinel"})
    function, seen = core.aggregate_replicate_evidence, []
    def traced(executions, *, group_declaration, targets):
        obj = selected_object((executions, group_declaration, targets), kind)
        seen.append((obj is original, hasattr(obj, "_extra_metadata")))
        return function(executions, group_declaration=group_declaration, targets=targets)
    monkeypatch.setattr(core, TOOL, traced)
    tools = make_tools(ReplicateAggregationTools, data)
    result = tools.invoke_tool(TOOL, {})
    assert seen and all(state == (False, False) for state in seen), seen
    assert "extra-attribute-sentinel" not in json.dumps(result)
    assert hasattr(original, "_extra_metadata")  # Never sanitize by mutating the host.


@pytest.mark.parametrize("protocol", ["__deepcopy__", "__copy__", "__reduce__", "__reduce_ex__", "__getstate__"])
@pytest.mark.parametrize("placement", ["metadata", "declared-field"])
def test_unsupported_declared_data_rejected_before_copy_protocol(protocol, placement):
    data, calls = bundle(), []
    def forbidden(*args, **kwargs):
        calls.append(protocol)
        raise AssertionError("unsupported copy protocol executed")
    probe = type("ProtocolProbe", (), {protocol: forbidden})()
    if placement == "metadata":
        data[0][0]["snapshot_metadata"]["payload"] = probe
    else:
        object.__setattr__(data[0][0]["provenance"], "configuration_ref", probe)
    with pytest.raises((TypeError, ValueError)):
        make_tools(ReplicateAggregationTools, data)
    assert calls == []


@pytest.mark.parametrize("kind", ["dict", "list", "tuple", "dataclass-subclass", "foreign-dataclass"])
def test_unapproved_types_rejected_without_user_protocols(kind):
    data, calls = bundle(), []
    def forbidden(*args, **kwargs):
        calls.append("protocol")
        raise AssertionError("unapproved object protocol executed")
    if kind in ("dict", "list", "tuple"):
        base = {"dict": dict, "list": list, "tuple": tuple}[kind]
        cls = type("ContainerProbe", (base,), {
            "__iter__": forbidden, "items": forbidden, "__deepcopy__": forbidden,
        })
        value = cls()
    elif kind == "dataclass-subclass":
        class Derived(ExperimentProvenance):
            __deepcopy__ = forbidden
        value = Derived(data[0][0]["execution"])
    else:
        @dataclass
        class Foreign:
            value: int = 1
            __deepcopy__ = forbidden
        value = Foreign()
    data[0][0]["snapshot_metadata"]["payload"] = value
    with pytest.raises((TypeError, ValueError)):
        make_tools(ReplicateAggregationTools, data)
    assert calls == []


def test_core_receives_detached_data_during_construction_and_each_invocation(monkeypatch):
    data, observed = bundle(), []
    function = core.aggregate_replicate_evidence
    def traced(executions, *, group_declaration, targets):
        observed.append((executions, group_declaration, targets))
        return function(executions, group_declaration=group_declaration, targets=targets)
    monkeypatch.setattr(core, TOOL, traced)
    tools = make_tools(ReplicateAggregationTools, data)
    tools.invoke_tool(TOOL, {})
    tools.invoke_tool(TOOL, {})
    # Constructor may validate through the public core. Invocation always does.
    assert len(observed) >= 2
    for call in observed:
        assert all(value is not host for value, host in zip(call, data))
        assert call[0][0]["evidence"] is not data[0][0]["evidence"]
    for a, b in zip(observed, observed[1:]):
        assert all(x is not y for x, y in zip(a, b))
        assert a[0][0]["evidence"] is not b[0][0]["evidence"]


def test_core_side_mutation_cannot_change_stored_snapshot(monkeypatch):
    tools = make_tools(ReplicateAggregationTools, bundle())
    function = core.aggregate_replicate_evidence
    calls = []
    def mutating(executions, *, group_declaration, targets):
        result = function(executions, group_declaration=group_declaration, targets=targets)
        calls.append(True)
        executions[0]["evidence"]["test_results"]["metrics"]["score"]["value"] = 999
        group_declaration["group_id"] = "changed-by-delegate"
        targets[0]["members"][0]["evaluation_protocol"] = None
        return result
    monkeypatch.setattr(core, TOOL, mutating)
    first = tools.invoke_tool(TOOL, {})
    second = tools.invoke_tool(TOOL, {})
    assert len(calls) == 2 and first == second
    assert second["targets"]["quality"]["statistics"]["mean"] == 3.0


def test_normal_behavior_external_mutation_and_return_isolation():
    data = bundle()
    tools = make_tools(ReplicateAggregationTools, data)
    data[0][0]["evidence"]["test_results"]["metrics"]["score"]["value"] = 999
    data[1]["expected_executions"].clear()
    data[2][0]["members"].clear()
    first = tools.invoke_tool(TOOL, {})
    assert first["targets"]["quality"]["statistics"]["mean"] == 3.0
    first["targets"]["quality"]["included_executions"][0]["source"]["provenance"]["status"] = "changed"
    first["group"]["group_id"] = "changed"
    second = tools.invoke_tool(TOOL, {})
    assert second["group"]["group_id"] == "group-v1"
    assert second["targets"]["quality"]["included_executions"][0]["source"]["provenance"]["status"] == "unknown"


@pytest.mark.parametrize("value", [True, float("nan"), float("inf"), -float("inf")])
def test_snapshot_preserves_core_local_invalid_observation_semantics(value):
    data = bundle()
    data[0][2]["evidence"]["test_results"]["metrics"]["score"]["value"] = value
    row = make_tools(ReplicateAggregationTools, data).invoke_tool(TOOL, {})["targets"]["quality"]
    assert row["target_status"] == "partial"
    assert row["statistics"]["count"] == 2 and row["statistics"]["mean"] == 2.0
    json.dumps(row, allow_nan=False)


@pytest.mark.parametrize("extra", [0, 1])
def test_snapshot_retains_exact_input_byte_boundary(extra):
    data = bundle()
    metadata = data[0][0]["snapshot_metadata"]
    metadata["padding"] = ""
    metadata["padding"] = "x" * (2 * 1024 * 1024 - json_size(input_envelope(data)) + extra)
    if extra:
        with pytest.raises(ValueError, match="input"):
            make_tools(ReplicateAggregationTools, data)
    else:
        assert make_tools(ReplicateAggregationTools, data).invoke_tool(TOOL, {})["targets"]["quality"]["statistics"]["count"] == 3


@pytest.mark.parametrize("kind", ["depth", "cycle"])
def test_snapshot_rejects_unbounded_structure_before_copy(kind):
    data, nested = bundle(), {}
    if kind == "cycle":
        nested["cycle"] = nested
    else:
        for _ in range(17):
            nested = {"child": nested}
    data[0][0]["snapshot_metadata"]["extra"] = nested
    with pytest.raises(ValueError):
        make_tools(ReplicateAggregationTools, data)

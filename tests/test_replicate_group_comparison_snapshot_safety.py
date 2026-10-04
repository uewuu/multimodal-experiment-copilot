"""M19 data-only snapshots, harmless callback probes and independent calibrations."""

from copy import copy, deepcopy
from dataclasses import dataclass
import importlib
import json

import pytest

import experiment_replicates as m18
from experiment_identity import ExperimentProvenance
from m19_contract_helpers import TOOL, core_api, invoke, pair, patch_public, tool_class


PROTOCOLS = ("__deepcopy__", "__copy__", "__reduce__", "__reduce_ex__", "__getstate__")


def probe(protocol, calls):
    def triggered(*args, **kwargs):
        calls.append(protocol)
        raise AssertionError("callback sentinel")
    return type("CallbackProbe", (), {protocol: triggered})()


def selected(group, kind):
    return {"provenance": group["executions"][0]["provenance"],
            "execution": group["executions"][0]["execution"],
            "experiment": group["group_declaration"]["experiment_identity"],
            "spec": group["targets"][0]["members"][0]["metric_spec"]}[kind]


def entry_api(entry):
    return core_api() if entry == "core" else tool_class()


def call(entry, function, data):
    result = invoke(function, data)
    return result if entry == "core" else result.invoke_tool(TOOL, {})


@pytest.mark.parametrize("protocol", PROTOCOLS)
def test_copy_probe_calibration_independent_of_m19(protocol):
    calls = []
    value = probe(protocol, calls)
    # Real copy libraries, not a direct call to the method under test.
    with pytest.raises(AssertionError, match="callback sentinel"):
        (copy if protocol == "__copy__" else deepcopy)(value)
    assert calls == [protocol]


@pytest.mark.parametrize("kind", ["provenance", "execution", "experiment", "spec"])
def test_approved_extra_attribute_calibration_really_reaches_nested_callback(kind):
    data, calls = pair(), []
    obj = selected(data[0], kind)
    object.__setattr__(obj, "_extra", probe("__deepcopy__", calls))
    with pytest.raises(AssertionError, match="callback sentinel"): deepcopy(obj)
    assert calls == ["__deepcopy__"]


@pytest.mark.parametrize("entry", ["core", "tool"])
@pytest.mark.parametrize("side", [0, 1], ids=["a", "b"])
@pytest.mark.parametrize("kind", ["provenance", "execution", "experiment", "spec"])
@pytest.mark.parametrize("protocol", PROTOCOLS)
def test_approved_extra_attributes_are_ignored_without_callbacks(entry, side, kind, protocol):
    function, data, calls = entry_api(entry), pair(), []
    obj = selected(data[side], kind)
    object.__setattr__(obj, "_extra", probe(protocol, calls))
    object.__setattr__(obj, "_extra_text", "must-not-leak-extra-attr")
    result = call(entry, function, data)
    assert calls == []
    assert result["targets"]["quality"]["comparison"]["raw_delta"] == 1.
    assert "must-not-leak-extra-attr" not in json.dumps(result)
    assert hasattr(obj, "_extra")  # Never sanitize the host by mutation.


@pytest.mark.parametrize("entry", ["core", "tool"])
@pytest.mark.parametrize("side", [0, 1], ids=["a", "b"])
@pytest.mark.parametrize("protocol", PROTOCOLS)
@pytest.mark.parametrize("placement", ["metadata", "declared-field"])
def test_unsafe_declared_data_rejects_before_callbacks_and_before_either_m18_call(monkeypatch, entry, side, protocol, placement):
    function, data, calls, delegates = entry_api(entry), pair(), [], []
    value = probe(protocol, calls)
    if placement == "metadata": data[side]["executions"][0]["snapshot_metadata"]["extra"] = value
    else: object.__setattr__(data[side]["executions"][0]["provenance"], "configuration_ref", value)
    core = importlib.import_module("experiment_group_comparison")
    def forbidden(*args, **kwargs):
        delegates.append(True)
        raise AssertionError("unsafe inputs reached M18")
    patch_public(monkeypatch, m18, "aggregate_replicate_evidence", forbidden, core)
    with pytest.raises((TypeError, ValueError)): call(entry, function, data)
    assert calls == delegates == []


@pytest.mark.parametrize("entry", ["core", "tool"])
@pytest.mark.parametrize("protocol", PROTOCOLS)
def test_mapping_protocol_object_rejected_before_callback(entry, protocol):
    function, data, calls = entry_api(entry), pair(), []
    data[2][0]["a_target_id"] = probe(protocol, calls)
    with pytest.raises((TypeError, ValueError)): call(entry, function, data)
    assert calls == []


def unapproved(kind, calls, data):
    def forbidden(*args, **kwargs):
        calls.append(kind)
        raise AssertionError("unsafe protocol sentinel")
    if kind in ("dict", "list", "tuple"):
        base = {"dict": dict, "list": list, "tuple": tuple}[kind]
        return type("ContainerProbe", (base,), {
            "__iter__": forbidden, "items": forbidden, "__deepcopy__": forbidden})()
    if kind == "subclass":
        class Derived(ExperimentProvenance):
            __deepcopy__ = forbidden
        return Derived(data[0]["executions"][0]["execution"])
    if kind == "foreign-dataclass":
        @dataclass
        class Foreign:
            value: int = 1
            __deepcopy__ = forbidden
        return Foreign()
    if kind == "callable": return forbidden
    if kind == "generator":
        def lazy():
            calls.append(kind)
            yield 1
        return lazy()
    class Provider:
        load = forbidden
        __iter__ = forbidden
        __deepcopy__ = forbidden
    return Provider()


@pytest.mark.parametrize("entry", ["core", "tool"])
@pytest.mark.parametrize("placement", ["a", "b", "mapping"])
@pytest.mark.parametrize("kind", ["dict", "list", "tuple", "subclass", "foreign-dataclass", "generator", "callable", "provider"])
def test_unapproved_types_reject_without_iteration_or_copy(entry, placement, kind):
    function, data, calls = entry_api(entry), pair(), []
    value = unapproved(kind, calls, data)
    if placement == "mapping": data[2][0]["b_target_id"] = value
    else: data[0 if placement == "a" else 1]["executions"][0]["snapshot_metadata"]["extra"] = value
    with pytest.raises((TypeError, ValueError)): call(entry, function, data)
    assert calls == []


@pytest.mark.parametrize("kind", ["dict", "list", "tuple", "subclass", "foreign-dataclass", "generator", "callable", "provider"])
def test_unapproved_type_probe_independent_calibration(kind):
    calls, data = [], pair()
    value = unapproved(kind, calls, data)
    if kind == "generator":
        assert next(value) == 1
    else:
        with pytest.raises(AssertionError, match="unsafe protocol sentinel"):
            if kind in ("dict", "list", "tuple"): iter(value)
            elif kind == "callable": value()
            elif kind == "provider": value.load()
            else: deepcopy(value)
    assert calls == [kind]


@pytest.mark.parametrize("side", [0, 1])
@pytest.mark.parametrize("kind", ["provenance", "execution", "experiment", "spec"])
def test_m18_receives_only_detached_declared_fields(monkeypatch, side, kind):
    function, data, seen = core_api(), pair(), []
    obj = selected(data[side], kind)
    object.__setattr__(obj, "_extra", "extra-sentinel")
    original = m18.aggregate_replicate_evidence
    def traced(executions, *, group_declaration, targets):
        if group_declaration["group_id"] == data[side]["group_declaration"]["group_id"]:
            copied = selected(dict(executions=executions, group_declaration=group_declaration, targets=targets), kind)
            seen.append((copied is obj, hasattr(copied, "_extra")))
        return original(executions, group_declaration=group_declaration, targets=targets)
    patch_public(monkeypatch, m18, "aggregate_replicate_evidence", traced,
                 importlib.import_module("experiment_group_comparison"))
    invoke(function, data)
    assert seen == [(False, False)]
    assert obj._extra == "extra-sentinel"


def test_core_m18_delegate_mutation_isolated_from_host_and_result(monkeypatch):
    function, data = core_api(), pair()
    pristine, calls = deepcopy(data), []
    original = m18.aggregate_replicate_evidence
    def mutating(executions, *, group_declaration, targets):
        result = original(executions, group_declaration=group_declaration, targets=targets)
        calls.append((executions, group_declaration, targets))
        executions[0]["evidence"]["test_results"]["metrics"]["score"]["value"] = 999
        targets[0]["members"][0]["metric_definition"] = "delegate-mutation"
        return result
    patch_public(monkeypatch, m18, "aggregate_replicate_evidence", mutating,
                 importlib.import_module("experiment_group_comparison"))
    first, second = invoke(function, data), invoke(function, data)
    assert data == pristine and first == second
    assert first["targets"]["quality"]["comparison"]["raw_delta"] == 1.
    assert len(calls) == 4
    assert calls[0][0] is not calls[2][0]


def test_tool_constructor_host_delegate_return_and_cross_call_isolation(monkeypatch):
    cls, function, data = tool_class(), core_api(), pair()
    calls = []
    def traced(a, b, *, targets):
        result = function(a, b, targets=targets)
        calls.append((a, b, targets))
        a["executions"][0]["evidence"]["test_results"]["metrics"]["score"]["value"] = 999
        b["group_declaration"]["group_id"] = "mutated-delegate"
        targets[0]["target_id"] = "mutated-mapping"
        return result
    patch_public(monkeypatch, importlib.import_module("experiment_group_comparison"), TOOL,
                 traced, importlib.import_module("tool_layer.replicate_group_tools"))
    tools = invoke(cls, data)
    assert data == pair()
    data[0]["executions"].clear()
    data[1]["group_declaration"]["expected_executions"].clear()
    data[2].clear()
    calls.clear()
    first = tools.invoke_tool(TOOL, {})
    pristine = deepcopy(first)
    first["targets"]["quality"]["b"]["included_executions"][0]["source"]["provenance"]["status"] = "polluted"
    assert tools.invoke_tool(TOOL, {}) == pristine
    assert len(calls) == 2
    assert all(a is not b for a, b in zip(calls[0], calls[1]))
    assert pristine["targets"]["quality"]["comparison"]["raw_delta"] == 1.
    assert pristine["groups"]["b"]["group_id"] == "group-B"

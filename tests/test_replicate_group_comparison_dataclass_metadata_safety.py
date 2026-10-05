"""Independent M19 regression for untrusted dataclass field metadata.

Probes only count in-memory callbacks. Class metadata replacement is scoped to
the M19 snapshot helper; downstream M18 class internals are not patched in the
end-to-end cases. Existing frozen contracts remain untouched.
"""

from copy import copy, deepcopy
from dataclasses import asdict, fields

import pytest

import experiment_comparability as m11
import experiment_group_comparison as core
import experiment_replicates as m18
from m19_contract_helpers import TOOL, invoke, pair
from test_replicate_group_comparison_snapshot_safety import selected
from tool_layer.replicate_group_tools import ReplicateGroupComparisonTools


KINDS = ("experiment", "execution", "provenance", "spec")
ENTRIES = ("core", "constructor")


class MetadataProbe:
    """Keep old fields(instance) functional, while recording its callback."""

    def __init__(self, declared_fields, calls):
        self.declared_fields = declared_fields
        self.calls = calls

    def values(self):
        self.calls.append("metadata.values")
        return self.declared_fields.values()


@pytest.mark.parametrize("kind", KINDS)
def test_metadata_probe_calibrates_against_real_dataclasses_fields(kind):
    obj, calls = selected(pair()[0], kind), []
    original = fields(type(obj))
    metadata = MetadataProbe(type(obj).__dataclass_fields__, calls)
    object.__setattr__(obj, "__dataclass_fields__", metadata)
    assert fields(obj) == original
    assert calls == ["metadata.values"]


@pytest.mark.parametrize("entry", ENTRIES)
@pytest.mark.parametrize("side", (0, 1), ids=("a", "b"))
@pytest.mark.parametrize("kind", KINDS)
def test_instance_metadata_never_runs_or_changes_real_delegation(monkeypatch, entry, side, kind):
    calls, traces = [], []
    aggregate, compare = m18.aggregate_replicate_evidence, m11.compare_metric_results

    def traced_aggregate(*args, **kwargs):
        traces.append(("m18", deepcopy((args, kwargs))))
        return aggregate(*args, **kwargs)

    def traced_compare(*args, **kwargs):
        traces.append(("m11", deepcopy((args, kwargs))))
        return compare(*args, **kwargs)

    monkeypatch.setattr(m18, "aggregate_replicate_evidence", traced_aggregate)
    monkeypatch.setattr(m11, "compare_metric_results", traced_compare)

    def run(data):
        if entry == "core":
            result = invoke(core.compare_replicate_groups, data)
            assert calls == []
            return result
        tools = invoke(ReplicateGroupComparisonTools, data)
        assert calls == []  # Constructor must be safe before any invocation.
        first = tools.invoke_tool(TOOL, {})
        assert calls == []
        assert tools.invoke_tool(TOOL, {}) == first
        assert calls == []
        return first

    expected = run(pair())
    expected_traces = traces[:]
    traces.clear()
    data = pair()
    obj = selected(data[side], kind)
    metadata = MetadataProbe(type(obj).__dataclass_fields__, calls)
    object.__setattr__(obj, "__dataclass_fields__", metadata)
    result = run(data)
    assert result == expected
    assert result["targets"]["quality"]["comparison"]["raw_delta"] == 1.
    assert traces == expected_traces
    assert obj.__dataclass_fields__ is metadata  # Host is not sanitized in place.


@pytest.mark.parametrize("kind", KINDS)
@pytest.mark.parametrize("override", ("empty", "reversed", "invented-field"))
def test_instance_metadata_cannot_change_snapshot_fields_order_or_budget(monkeypatch, kind, override):
    obj = selected(pair()[0], kind)
    encoded = []
    dumps = core.json.dumps

    def traced_dumps(value, **kwargs):
        encoded.append(value)
        return dumps(value, **kwargs)

    monkeypatch.setattr(core.json, "dumps", traced_dumps)
    expected, expected_size = core._data_snapshot(obj, limit=2 * 1024 * 1024)
    expected_encoding_order = encoded[:]
    encoded.clear()
    metadata = dict(reversed(tuple(type(obj).__dataclass_fields__.items())))
    if override == "empty":
        metadata.clear()
    elif override == "invented-field":
        field = copy(next(iter(metadata.values())))
        field.name = "_not_a_declared_field"
        metadata[field.name] = field
        object.__setattr__(obj, field.name, "must-not-enter-snapshot")
    object.__setattr__(obj, "__dataclass_fields__", metadata)
    actual, size = core._data_snapshot(obj, limit=2 * 1024 * 1024)
    assert actual == expected and actual is not obj
    assert asdict(actual) == asdict(expected)
    assert encoded == expected_encoding_order
    assert size == expected_size
    assert "__dataclass_fields__" not in vars(actual)
    assert not hasattr(actual, "_not_a_declared_field")


@pytest.mark.parametrize("kind", KINDS)
def test_snapshot_schema_does_not_read_mutable_class_metadata(monkeypatch, kind):
    obj, calls = selected(pair()[0], kind), []
    expected, expected_size = core._data_snapshot(obj, limit=2 * 1024 * 1024)
    metadata = MetadataProbe(type(obj).__dataclass_fields__, calls)
    with monkeypatch.context() as patch:
        patch.setattr(type(obj), "__dataclass_fields__", metadata)
        actual, size = core._data_snapshot(obj, limit=2 * 1024 * 1024)
    assert calls == []
    assert actual == expected and actual is not obj
    assert size == expected_size


@pytest.mark.parametrize("entry", ENTRIES)
@pytest.mark.parametrize("placement", ("a", "b", "mapping"))
def test_unapproved_metaclass_equality_and_hash_not_called(entry, placement):
    calls = []

    class Meta(type):
        def __eq__(cls, other):
            calls.append("equality")
            return False

        def __hash__(cls):
            calls.append("hash")
            return 0

    class Unapproved(metaclass=Meta):
        pass

    assert (Unapproved == dict) is False
    assert hash(Unapproved) == 0
    assert calls == ["equality", "hash"]
    calls.clear()
    data = pair()
    if placement == "mapping":
        data[2][0]["target_id"] = Unapproved()
    else:
        data[0 if placement == "a" else 1]["executions"][0]["snapshot_metadata"]["extra"] = Unapproved()
    function = core.compare_replicate_groups if entry == "core" else ReplicateGroupComparisonTools
    with pytest.raises(TypeError):
        invoke(function, data)
    assert calls == []

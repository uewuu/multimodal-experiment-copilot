"""R4b: calibrated callback probes at the two real M18 entry boundaries.

No frozen tests are weakened. Approved declared data remain supported; extra
dataclass metadata are not a traversal schema or executable host capability.
"""

from collections.abc import Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass, fields
import json
import os

import pytest

import experiment_replicates as core
from m18_contract_helpers import TOOL, bundle, input_envelope, invoke, json_size
from test_replicate_aggregation_snapshot_safety import selected_object
from tool_layer.replicate_tools import ReplicateAggregationTools


ENTRIES = ("core", "tool")
KINDS = ("experiment", "execution", "provenance", "spec")
PROTOCOLS = (
    "metaclass-equality", "metaclass-hash", "instancecheck", "class-attribute",
    "mapping", "sequence", "iter", "deepcopy", "attribute", "fspath",
    "foreign-dataclass", "dict-subclass", "list-subclass", "str-subclass",
    "descriptor",
)


def run(entry, data):
    if entry == "core":
        return invoke(core.aggregate_replicate_evidence, data)
    return invoke(ReplicateAggregationTools, data).invoke_tool(TOOL, {})


def probe(protocol, calls):
    def hit(*args, **kwargs):
        calls.append(protocol)
        raise RuntimeError("calibrated callback")

    if protocol in ("metaclass-equality", "metaclass-hash", "instancecheck", "class-attribute"):
        name = {"metaclass-equality": "__eq__", "metaclass-hash": "__hash__",
                "instancecheck": "__instancecheck__", "class-attribute": "__getattribute__"}[protocol]
        meta = type("ProbeMeta", (type,), {name: hit})
        cls = meta("Unsupported", (), {})
        value = cls()
        trigger = {"metaclass-equality": lambda: cls == dict,
                   "metaclass-hash": lambda: hash(cls),
                   "instancecheck": lambda: isinstance(object(), cls),
                   "class-attribute": lambda: cls.__name__}[protocol]
    elif protocol == "mapping":
        class Custom(Mapping):
            __iter__ = __len__ = __getitem__ = hit
        value = Custom()
        trigger = lambda: iter(value)
    elif protocol == "sequence":
        class Custom(Sequence):
            __len__ = __getitem__ = hit
        value = Custom()
        trigger = lambda: value[0]
    elif protocol == "foreign-dataclass":
        @dataclass
        class Custom:
            field: int = 1
            __deepcopy__ = hit
        value = Custom()
        trigger = lambda: deepcopy(value)
    elif protocol in ("dict-subclass", "list-subclass", "str-subclass"):
        base = {"dict-subclass": dict, "list-subclass": list, "str-subclass": str}[protocol]
        value = type("Custom", (base,), {"__iter__": hit, "__str__": hit})()
        trigger = (lambda: str(value)) if base is str else (lambda: iter(value))
    elif protocol == "descriptor":
        value = type("Custom", (), {"payload": property(hit)})()
        trigger = lambda: value.payload
    else:
        name = {"iter": "__iter__", "deepcopy": "__deepcopy__",
                "attribute": "__getattribute__", "fspath": "__fspath__"}[protocol]
        value = type("Custom", (), {name: hit})()
        trigger = {"iter": lambda: iter(value), "deepcopy": lambda: deepcopy(value),
                   "attribute": lambda: value.payload, "fspath": lambda: os.fspath(value)}[protocol]
    calls.clear()  # Construction is outside the tested boundary.
    return value, trigger


@pytest.mark.parametrize("protocol", PROTOCOLS)
def test_probe_calibration(protocol):
    calls = []
    _, trigger = probe(protocol, calls)
    with pytest.raises(RuntimeError, match="calibrated callback"):
        trigger()
    assert calls == [protocol]


@pytest.mark.parametrize("entry", ENTRIES)
@pytest.mark.parametrize("placement", ("metadata", "declared-field"))
@pytest.mark.parametrize("protocol", PROTOCOLS)
def test_unsupported_input_rejected_before_protocol(entry, placement, protocol):
    data, calls = bundle(), []
    value, _ = probe(protocol, calls)
    if placement == "metadata":
        data[0][0]["snapshot_metadata"]["payload"] = value
    else:
        object.__setattr__(data[0][0]["provenance"], "configuration_ref", value)
    with pytest.raises((TypeError, ValueError)):
        run(entry, data)
    assert calls == [], f"callback_count={len(calls)}: {calls}"


class MetadataProvider:
    def __init__(self, declared, calls):
        self.declared, self.calls = declared, calls

    def values(self):
        self.calls.append("metadata.values")
        return self.declared.values()


@pytest.mark.parametrize("kind", KINDS)
def test_metadata_provider_calibration(kind):
    value, calls = selected_object(bundle(), kind), []
    expected = fields(value)
    object.__setattr__(value, "__dataclass_fields__", MetadataProvider(type(value).__dataclass_fields__, calls))
    assert fields(value) == expected
    assert calls == ["metadata.values"]


@pytest.mark.parametrize("entry", ENTRIES)
@pytest.mark.parametrize("where", ("instance", "class"))
@pytest.mark.parametrize("kind", KINDS)
def test_metadata_provider_is_not_a_schema_or_callback(monkeypatch, entry, where, kind):
    expected = run(entry, bundle())
    data, calls = bundle(), []
    value = selected_object(data, kind)
    provider = MetadataProvider(type(value).__dataclass_fields__, calls)
    if where == "instance":
        object.__setattr__(value, "__dataclass_fields__", provider)
        actual = run(entry, data)
        assert value.__dataclass_fields__ is provider
    else:
        with monkeypatch.context() as patch:
            patch.setattr(type(value), "__dataclass_fields__", provider)
            actual = run(entry, data)
    assert calls == [], f"callback_count={len(calls)}: {calls}"
    assert actual == expected


@pytest.mark.parametrize("entry", ENTRIES)
def test_original_metadata_probe_rejected_without_callback(entry):
    data, calls = bundle(), []
    value = data[1]["experiment_identity"]
    object.__setattr__(value, "__dataclass_fields__", MetadataProvider({}, calls))
    with pytest.raises((TypeError, ValueError)):
        run(entry, ([value], {}, []))
    assert calls == [], f"callback_count={len(calls)}: {calls}"


@pytest.mark.parametrize("entry", ENTRIES)
def test_malicious_dict_key_rejects_without_rehash_or_equality(entry):
    calls = []
    class Key:
        def __hash__(self):
            calls.append("hash")
            return hash("snapshot_id")
        def __eq__(self, other):
            calls.append("equality")
            return False
    key = Key()
    assert hash(key) == hash("snapshot_id")
    assert (key == "snapshot_id") is False
    assert calls == ["hash", "equality"]
    data = bundle()
    data[0][0]["snapshot_metadata"][key] = "data"
    calls.clear()
    with pytest.raises(TypeError):
        run(entry, data)
    assert calls == []


@pytest.mark.parametrize("entry", ENTRIES)
@pytest.mark.parametrize("failure", ("cycle", "depth", "bytes"))
def test_frozen_resource_bounds_reject(entry, failure):
    data, value = bundle(), {}
    if failure == "cycle":
        value["cycle"] = value
    elif failure == "depth":
        for _ in range(17):
            value = {"child": value}
    else:
        value = "x" * (2 * 1024 * 1024 + 1)
    data[0][0]["snapshot_metadata"]["extra"] = value
    with pytest.raises(ValueError):
        run(entry, data)


@pytest.mark.parametrize("entry", ENTRIES)
@pytest.mark.parametrize("section", ("test", "validation"))
def test_exact_builtins_and_approved_types_remain_deterministic(entry, section):
    data = bundle(section=section)
    expected = run(entry, data)
    data[0][0]["snapshot_metadata"]["extra"] = {"plain": [None, True, 1, 2.5, "text", ()]}
    actual = run(entry, (tuple(data[0]), data[1], tuple(data[2])))
    assert actual == expected
    assert run(entry, data) == actual
    assert len(json.dumps(actual, ensure_ascii=False, allow_nan=False, separators=(",", ":")).encode()) <= 128 * 1024


@pytest.mark.parametrize("entry", ENTRIES)
@pytest.mark.parametrize("extra", (0, 1))
def test_exact_byte_boundary_with_ignored_metadata(entry, extra):
    data, calls = bundle(), []
    metadata = data[0][0]["snapshot_metadata"]
    metadata["padding"] = ""
    metadata["padding"] = "x" * (2 * 1024 * 1024 - json_size(input_envelope(data)) + extra)
    obj = data[1]["experiment_identity"]
    object.__setattr__(obj, "__dataclass_fields__", MetadataProvider({}, calls))
    if extra:
        with pytest.raises(ValueError, match="input"):
            run(entry, data)
    else:
        assert run(entry, data)["targets"]["quality"]["statistics"]["mean"] == 3.0
    assert calls == [], f"callback_count={len(calls)}: {calls}"

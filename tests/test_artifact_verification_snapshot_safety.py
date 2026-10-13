"""M20 data-only boundaries and independently live callback probes."""

from collections.abc import Mapping, Sequence
from copy import copy, deepcopy
from dataclasses import dataclass
import os

import pytest

from m20_contract_helpers import TOOL, core_api, data, patch_core, tool_class

COPY_PROTOCOLS = ("__copy__", "__deepcopy__", "__reduce__", "__reduce_ex__", "__getstate__")
KINDS = COPY_PROTOCOLS + ("dict-subclass", "list-subclass", "mapping", "sequence", "proxy",
                         "lazy", "callable", "generator", "pathlike", "descriptor",
                         "instance-metadata", "class-metadata", "str-subclass", "metaclass")


def probe(kind, calls):
    def hit(*args, **kwargs):
        calls.append(kind)
        raise AssertionError("callback sentinel")
    if kind in COPY_PROTOCOLS:
        return type("CopyProbe", (), {kind: hit})()
    if kind in ("dict-subclass", "list-subclass"):
        base = dict if kind == "dict-subclass" else list
        return type("ContainerProbe", (base,), {"__iter__": hit, "items": hit, "__deepcopy__": hit})()
    if kind == "mapping":
        class CustomMapping(Mapping):
            __getitem__ = __iter__ = __len__ = hit
        return CustomMapping()
    if kind == "sequence":
        class CustomSequence(Sequence):
            __getitem__ = __len__ = hit
        return CustomSequence()
    if kind == "proxy":
        return type("Proxy", (), {"__getattr__": hit})()
    if kind == "lazy":
        return type("Lazy", (), {"load": hit, "__iter__": hit})()
    if kind == "callable": return hit
    if kind == "generator":
        def generate():
            hit()
            yield None
        return generate()
    if kind == "pathlike":
        return type("PathProbe", (), {"__fspath__": hit})()
    if kind == "descriptor":
        return type("Descriptor", (), {"value": property(hit)})()
    if kind in ("instance-metadata", "class-metadata"):
        @dataclass
        class Foreign:
            value: int = 1
        value = Foreign()
        metadata = type("Metadata", (), {"values": hit})()
        if kind == "instance-metadata": value.__dataclass_fields__ = metadata
        else: Foreign.__dataclass_fields__ = metadata
        return value
    if kind == "str-subclass":
        return type("Text", (str,), {"__str__": hit, "__eq__": hit, "__hash__": str.__hash__})("metrics")
    if kind == "metaclass":
        class Meta(type):
            __eq__ = hit
            __hash__ = type.__hash__
        class Foreign(metaclass=Meta):
            pass
        return Foreign()
    raise AssertionError(kind)


@pytest.mark.parametrize("kind", KINDS)
def test_callback_probe_independent_calibration(kind):
    calls, value = [], None
    value = probe(kind, calls)
    with pytest.raises(AssertionError, match="callback sentinel"):
        if kind in COPY_PROTOCOLS: (copy if kind == "__copy__" else deepcopy)(value)
        elif kind in ("dict-subclass", "list-subclass", "mapping", "lazy"): iter(value)
        elif kind == "sequence": value[0]
        elif kind == "proxy": value.payload
        elif kind == "callable": value()
        elif kind == "generator": next(value)
        elif kind == "pathlike": os.fspath(value)
        elif kind == "descriptor": value.value
        elif kind in ("instance-metadata", "class-metadata"):
            from dataclasses import fields
            fields(value)
        elif kind == "str-subclass": str(value)
        elif kind == "metaclass": type(value) == dict
    assert calls == [kind]


@pytest.mark.parametrize("entrance", ["core", "tool"])
@pytest.mark.parametrize("placement", ["set", "artifacts", "artifact", "observations", "observation",
                                      "locator", "sha256", "source"])
@pytest.mark.parametrize("kind", KINDS)
def test_reject_arbitrary_objects_before_protocols(entrance, placement, kind):
    function = core_api() if entrance == "core" else tool_class()
    declared, reported = data()
    calls = []
    value = probe(kind, calls)
    if placement == "set": declared = value
    elif placement == "artifacts": declared["artifacts"] = value
    elif placement == "artifact": declared["artifacts"][0] = value
    elif placement == "observations": reported = value
    elif placement == "observation": reported[0] = value
    elif placement == "locator": declared["artifacts"][0]["locator"] = value
    elif placement == "sha256": reported[0]["sha256"] = value
    else: reported[0]["reported_by"] = value
    with pytest.raises((TypeError, ValueError)):
        function(declared, reported)
    assert calls == []


@pytest.mark.parametrize("kind", KINDS)
def test_constructor_rejects_unsafe_input_before_core(monkeypatch, kind):
    cls = tool_class()
    core_api()
    declared, reported = data()
    calls, delegates = [], []
    reported[0]["locator"] = probe(kind, calls)
    def forbidden(*args, **kwargs):
        delegates.append(True)
        raise AssertionError("unsafe graph reached core")
    patch_core(monkeypatch, forbidden)
    with pytest.raises((TypeError, ValueError)):
        cls(declared, reported)
    assert calls == delegates == []


@pytest.mark.parametrize("entrance", ["core", "tool"])
@pytest.mark.parametrize("shape", ["list-cycle", "dict-cycle", "mutual-cycle", "depth", "tuple", "bytes", "bool", "float"])
def test_cycles_depth_and_unapproved_primitive_types(entrance, shape):
    function = core_api() if entrance == "core" else tool_class()
    declared, reported = data()
    if shape == "list-cycle":
        value = []
        value.append(value)
    elif shape == "dict-cycle":
        value = {}
        value["self"] = value
    elif shape == "mutual-cycle":
        value, other = [], {}
        value.append(other)
        other["back"] = value
    elif shape == "depth":
        value = []
        for _ in range(10): value = [value]
    else: value = {"tuple": (), "bytes": b"x", "bool": True, "float": 1.0}[shape]
    declared["artifacts"] = value
    with pytest.raises((TypeError, ValueError)):
        function(declared, reported)


def test_constructor_and_delegate_mutation_cannot_change_bound_snapshot(monkeypatch):
    cls, function, arguments = tool_class(), core_api(), data()
    pristine = deepcopy(arguments)
    seen = []
    def mutating(declared_set, reported_observations):
        seen.append(deepcopy((declared_set, reported_observations)))
        result = function(declared_set, reported_observations)
        declared_set["artifacts"][0]["locator"] = "mutated-by-delegate"
        reported_observations.clear()
        return result
    patch_core(monkeypatch, mutating)
    tools = cls(*arguments)
    assert arguments == pristine
    arguments[0]["execution_ref"]["execution_id"] = "outside-mutation"
    arguments[0]["artifacts"].clear()
    arguments[1].clear()
    seen.clear()
    first = tools.invoke_tool(TOOL, {})
    first["artifacts"]["metrics"]["reported"]["sha256"] = "modified-return"
    second = tools.invoke_tool(TOOL, {})
    assert second == function(*pristine)
    assert seen == [pristine, pristine]


def test_core_rejects_nonstring_mapping_key_without_stringification():
    function, (declared, reported) = core_api(), data()
    calls = []
    value = probe("proxy", calls)
    declared[value] = "unapproved key"
    with pytest.raises((TypeError, ValueError)): function(declared, reported)
    assert calls == []

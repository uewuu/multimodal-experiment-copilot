"""Calibrated hostile protocols: rejection must not execute input callbacks."""
from collections.abc import Mapping, Sequence
from copy import copy, deepcopy
import os

import pytest

from m21_contract_helpers import api, binding, ROOT, TOOL


KINDS = ['copy', 'deepcopy', 'reduce', 'reduce_ex', 'getstate', 'fspath', 'str-subclass',
         'int-subclass', 'mapping', 'sequence', 'dict-subclass', 'list-subclass', 'proxy',
         'lazy', 'generator', 'callable', 'descriptor', 'key', 'repr', 'str', 'format',
         'meta-equality', 'meta-hash', 'meta-attribute']


def probe(kind):
    calls = []
    def trip(*args, **kwargs):
        calls.append(kind)
        raise RuntimeError('callback sentinel')
    class Hostile:
        __copy__ = __deepcopy__ = __reduce__ = __reduce_ex__ = __getstate__ = trip
        __fspath__ = __repr__ = __str__ = __format__ = trip
        __iter__ = __call__ = __get__ = __getattr__ = trip
        __hash__ = __eq__ = trip
    if kind == 'str-subclass':
        class Value(str):
            __str__ = __repr__ = __eq__ = __hash__ = trip
        value, trigger = Value('unsafe'), lambda x: str(x)
    elif kind == 'int-subclass':
        class Value(int):
            __int__ = __repr__ = trip
        value, trigger = Value(1048576), lambda x: int(x)
    elif kind in ('mapping', 'sequence'):
        if kind == 'mapping':
            class Value(Mapping):
                __iter__ = __getitem__ = __len__ = trip
        else:
            class Value(Sequence):
                __getitem__ = __len__ = __iter__ = trip
        value, trigger = Value(), lambda x: iter(x)
    elif kind in ('dict-subclass', 'list-subclass'):
        class Value(dict if kind == 'dict-subclass' else list):
            __iter__ = __repr__ = trip
            items = trip
        value, trigger = Value(), lambda x: iter(x)
    elif kind == 'generator':
        def generate():
            trip()
            yield None
        value, trigger = generate(), lambda x: next(x)
    elif kind.startswith('meta-'):
        class Meta(type):
            __eq__ = __hash__ = trip
            def __getattribute__(self, name):
                if name in ('__name__', '__dict__', '__dataclass_fields__', '__mro__'):
                    trip()
                return type.__getattribute__(self, name)
        class Value(metaclass=Meta):
            pass
        value = Value()
        trigger = {'meta-equality': lambda x: type(x) == dict,
                   'meta-hash': lambda x: hash(type(x)),
                   'meta-attribute': lambda x: type(x).__name__}[kind]
    elif kind == 'getstate':
        class Value:
            __getstate__ = trip
        value, trigger = Value(), lambda x: x.__getstate__()
    else:
        value = Hostile()
        trigger = {'copy': copy, 'deepcopy': deepcopy, 'reduce': lambda x: x.__reduce__(),
                   'reduce_ex': lambda x: x.__reduce_ex__(4), 'fspath': os.fspath,
                   'proxy': lambda x: x.schema, 'lazy': lambda x: iter(x),
                   'callable': lambda x: x(), 'descriptor': lambda x: x.__get__(None, object),
                   'key': hash, 'repr': repr, 'str': str, 'format': lambda x: format(x)}[kind]
    calls.clear()
    return value, trigger, calls


@pytest.mark.parametrize('kind', KINDS)
def test_callback_probe_calibration_independent_of_production(kind):
    value, trigger, calls = probe(kind)
    with pytest.raises(RuntimeError, match='callback sentinel'):
        trigger(value)
    assert calls == [kind]


@pytest.mark.parametrize('kind', KINDS)
@pytest.mark.parametrize('entrance', ['binding-root', 'binding-name', 'binding-ref',
                                    'tool-name', 'tool-args', 'tool-root', 'io-root', 'io-name'])
def test_rejection_without_copy_reflection_coercion_or_formatting(kind, entrance):
    layer = 'tool' if entrance.startswith('tool') else 'io' if entrance.startswith('io') else 'core'
    module = api(layer)
    value, _, calls = probe(kind)
    with pytest.raises((TypeError, ValueError)):
        if entrance.startswith('binding'):
            field = {'binding-root': 'authorized_root', 'binding-name': 'file_name', 'binding-ref': 'artifact_ref'}[entrance]
            module.validate_artifact_binding(binding(**{field: value}))
        elif entrance.startswith('io'):
            with module.open_authorized_artifact(value if entrance == 'io-root' else ROOT,
                                                  value if entrance == 'io-name' else 'artifact.bin'):
                pytest.fail('unsafe path accepted')
        else:
            tools = module.ArtifactObservationTools(binding())
            tools.invoke_tool(value if entrance == 'tool-name' else TOOL,
                              value if entrance == 'tool-args' else
                              {'experiment_dir': value if entrance == 'tool-root' else ROOT})
    assert calls == []


def keyed_probe():
    calls, armed = [], [False]
    class Key:
        def __hash__(self):
            if armed[0]:
                calls.append('hash')
                raise RuntimeError('key hash')
            return hash('authorized_root')
        def __eq__(self, other):
            calls.append('equality')
            raise RuntimeError('key equality')
        def __repr__(self):
            calls.append('repr')
            raise RuntimeError('key repr')
    # Insert alone while disarmed, avoiding test-setup collisions with exact keys.
    key = Key()
    value = {key: 'value'}
    armed[0] = True
    return key, value, calls


@pytest.mark.parametrize('operation', ['hash', 'equality', 'repr'])
def test_armed_dictionary_key_probe_calibration(operation):
    key, _, calls = keyed_probe()
    with pytest.raises(RuntimeError, match='key'):
        if operation == 'hash':
            hash(key)
        elif operation == 'equality':
            key == 'authorized_root'
        else:
            repr(key)
    assert calls == [operation]


@pytest.mark.parametrize('entrance', ['binding', 'tool-constructor', 'tool-arguments'])
def test_malicious_dictionary_key_not_hashed_compared_or_formatted(entrance):
    module = api('core' if entrance == 'binding' else 'tool')
    _, value, calls = keyed_probe()
    with pytest.raises((TypeError, ValueError)):
        if entrance == 'binding':
            module.validate_artifact_binding(value)
        elif entrance == 'tool-constructor':
            module.ArtifactObservationTools(value)
        else:
            module.ArtifactObservationTools(binding()).invoke_tool(TOOL, value)
    assert calls == []


@pytest.mark.parametrize('kind', ['dict-subclass', 'mapping', 'proxy', 'descriptor', 'meta-attribute'])
def test_constructor_rejects_nonplain_binding_without_callbacks(kind):
    cls = api('tool').ArtifactObservationTools
    value, _, calls = probe(kind)
    with pytest.raises((TypeError, ValueError)):
        cls(value)
    assert calls == []


def test_int_subclass_budget_is_not_coerced():
    core = api('core')
    value, _, calls = probe('int-subclass')
    with pytest.raises((TypeError, ValueError)):
        core.validate_artifact_binding(binding(max_bytes=value))
    assert calls == []


def test_cyclic_binding_rejects_without_copy_recursion():
    core = api('core')
    selected = binding()
    selected['artifact_ref'] = selected
    with pytest.raises((TypeError, ValueError)):
        core.validate_artifact_binding(selected)


def test_os_error_formatting_is_never_used_in_reader_report(monkeypatch):
    from m21_contract_helpers import Stream, install_stream, assert_report
    reader = api('reader')
    calls = []
    class HostileError(OSError):
        def __str__(self):
            calls.append('str')
            raise RuntimeError('error formatted')
        __repr__ = __str__
        def __format__(self, spec):
            return self.__str__()
    error = HostileError(5, 'private path')
    stream = Stream(error=error)
    observe = install_stream(monkeypatch, stream)
    assert_report(observe(binding()), 'io_error')
    assert calls == [] and reader is not None


@pytest.mark.parametrize('kind', KINDS)
def test_capture_snapshot_rejects_callback_values(kind):
    from m21_contract_helpers import capture
    core = api('core')
    value, _, calls = probe(kind)
    measured = capture()
    measured['sha256'] = value
    with pytest.raises((TypeError, ValueError)):
        core.project_artifact_observation(binding(), measured)
    assert calls == []


def test_error_formatting_probe_calibration():
    calls = []
    class Error(OSError):
        def __str__(self):
            calls.append(True)
            raise RuntimeError('formatting sentinel')
    with pytest.raises(RuntimeError, match='formatting sentinel'):
        str(Error(5, 'private'))
    assert calls == [True]

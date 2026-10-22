"""Calibrated callback rejection at the Core and bound Tool boundaries."""
from collections.abc import Mapping, Sequence
from copy import copy, deepcopy
from dataclasses import dataclass
from decimal import Decimal
import os
import pickle

import pytest
from m22_contract_helpers import core_api, fixture, invoke, metric, pairs, tool_class, TOOL


OPERATIONS = {
    '__str__': str, '__repr__': repr, '__format__': lambda x: format(x, ''),
    '__eq__': lambda x: x == 1, '__hash__': hash, '__iter__': iter,
    '__len__': len, '__getitem__': lambda x: x[0], '__contains__': lambda x: 0 in x,
    'keys': lambda x: x.keys(), 'items': lambda x: x.items(), 'values': lambda x: x.values(),
    '__copy__': copy, '__deepcopy__': deepcopy, '__reduce__': lambda x: x.__reduce__(),
    '__reduce_ex__': lambda x: pickle.dumps(x), '__getstate__': lambda x: x.__getstate__(),
    '__getattribute__': lambda x: x.payload, '__index__': lambda x: [0][x],
    '__int__': int, '__float__': float, '__fspath__': os.fspath,
}


def probe(protocol):
    calls = []
    def trap(*args, **kwargs):
        calls.append(protocol)
        raise RuntimeError('calibrated callback')
    cls = type('Probe', (), {protocol: trap})
    return cls(), calls


@pytest.mark.parametrize('protocol', list(OPERATIONS))
def test_callback_probe_independent_calibration(protocol):
    value, calls = probe(protocol)
    with pytest.raises(RuntimeError, match='calibrated callback'): OPERATIONS[protocol](value)
    assert calls == [protocol]


def place(data, where, value):
    if where == 'group': data['group_a'] = value
    elif where == 'snapshot': data['group_a']['executions'][0]['snapshot_metadata']['extra'] = value
    elif where == 'identity': pairs(data)[0]['a_execution']['execution_id'] = value
    elif where == 'pair_id': pairs(data)[0]['pair_id'] = value
    elif where == 'target_id': data['targets'][0]['target_id'] = value
    elif where == 'declaration': data['pairing_declaration'] = value
    elif where == 'measurement': metric(data)['value'] = value
    else: raise AssertionError('invalid test placement')


@pytest.mark.parametrize('protocol', list(OPERATIONS))
@pytest.mark.parametrize('entry', ['core', 'tool'])
def test_unsupported_protocols_never_run_before_rejection(protocol, entry):
    api = core_api() if entry == 'core' else tool_class()
    value, calls = probe(protocol); d = fixture()
    with pytest.raises(RuntimeError): OPERATIONS[protocol](value)
    calls.clear(); place(d, 'snapshot', value)
    with pytest.raises(TypeError): invoke(api, d)
    assert calls == []


@pytest.mark.parametrize('where', ['group', 'identity', 'pair_id', 'target_id', 'declaration', 'measurement'])
@pytest.mark.parametrize('entry', ['core', 'tool'])
def test_error_formatting_never_touches_rejected_objects(where, entry):
    calls = []
    def trap(*a, **kw): calls.append('format'); raise RuntimeError('format trap')
    cls = type('FormattingProbe', (), dict(__repr__=trap, __str__=trap, __format__=trap, __fspath__=trap))
    value = cls(); d = fixture(); place(d, where, value)
    with pytest.raises(RuntimeError): repr(value)
    assert calls == ['format']; calls.clear()
    api = core_api() if entry == 'core' else tool_class()
    with pytest.raises(TypeError) as caught: invoke(api, d)
    assert calls == [] and len(str(caught.value).encode('utf-8')) <= 1024


@pytest.mark.parametrize('where', ['group', 'declaration', 'mapping', 'args'])
def test_malicious_dictionary_keys_rejected_before_hash_eq_or_format(where):
    calls = []
    class Key:
        def __hash__(self): calls.append('hash'); return 3
        def __eq__(self, other): calls.append('eq'); return False
        def __repr__(self): calls.append('repr'); return 'key'
        def __str__(self): calls.append('str'); return 'key'
    value = Key(); bad = {value: 1}
    assert calls == ['hash']; calls.clear()
    api = tool_class() if where == 'args' else core_api()
    d = fixture()
    if where == 'args':
        tools = invoke(api, d)
        with pytest.raises((TypeError, ValueError)): tools.invoke_tool(TOOL, bad)
    else:
        if where == 'group': d['group_a']['executions'][0]['snapshot_metadata'] = bad
        elif where == 'declaration': d['pairing_declaration'] = bad
        else: d['targets'][0] = bad
        with pytest.raises(TypeError): invoke(api, d)
    assert calls == []


@pytest.mark.parametrize('form', ['instance', 'class'])
@pytest.mark.parametrize('where', ['core', 'tool_name', 'tool_args'])
def test_metaclass_identity_checks_do_not_call_equality_hash_or_attribute(form, where):
    calls = []
    class Meta(type):
        def __eq__(cls, other): calls.append('eq'); return False
        def __hash__(cls): calls.append('hash'); return 1
        def __getattribute__(cls, name):
            calls.append('attr'); return type.__getattribute__(cls, name)
    class Evil(metaclass=Meta): pass
    value = Evil() if form == 'instance' else Evil
    assert (Evil == dict) is False; hash(Evil); Evil.__name__
    assert set(calls) == {'eq', 'hash', 'attr'}; calls.clear()
    api = core_api() if where == 'core' else tool_class()
    d = fixture()
    if where == 'core':
        place(d, 'snapshot', value)
        with pytest.raises(TypeError): invoke(api, d)
    else:
        tool = invoke(api, d)
        with pytest.raises(TypeError): tool.invoke_tool(value if where == 'tool_name' else TOOL,
                                                       value if where == 'tool_args' else {})
    assert calls == []


@pytest.mark.parametrize('base', [str, int, float, dict, list], ids=lambda x: x.__name__)
@pytest.mark.parametrize('entry', ['core', 'tool'])
def test_builtin_subclasses_are_not_plain_data(base, entry):
    calls = []
    class Sub(base):
        def __repr__(self): calls.append('repr'); return 'sub'
        def __deepcopy__(self, memo): calls.append('deepcopy'); return self
    value = Sub(); assert repr(value) == 'sub'; deepcopy(value); assert calls == ['repr', 'deepcopy']
    calls.clear(); d = fixture(); place(d, 'snapshot', value)
    api = core_api() if entry == 'core' else tool_class()
    with pytest.raises(TypeError): invoke(api, d)
    assert calls == []


@pytest.mark.parametrize('kind', ['mapping', 'sequence', 'generator', 'callable', 'descriptor', 'dataclass', 'tuple', 'decimal', 'foreign-numeric'])
def test_nonplain_structures_rejected_without_protocols(kind):
    calls = []
    class Map(Mapping):
        def __iter__(self): calls.append('iter'); return iter(())
        def __len__(self): calls.append('len'); return 0
        def __getitem__(self, key): calls.append('getitem'); raise KeyError
    class Seq(Sequence):
        def __len__(self): calls.append('len'); return 0
        def __getitem__(self, key): calls.append('getitem'); raise IndexError
    class Descriptor:
        def __get__(self, instance, owner): calls.append('get'); return 1
    @dataclass
    class Data:
        value: int = 1
        def __getattribute__(self, name): calls.append('attr'); return object.__getattribute__(self, name)
    def lazy():
        calls.append('generator'); yield 1
    def callback(): calls.append('call'); return {}
    class ForeignNumeric:
        # No optional NumPy dependency in the frozen test environment. This
        # calibrated foreign scalar freezes rejection before numeric coercion;
        # it is not represented as a real numpy scalar acceptance test.
        def __float__(self): calls.append('float'); return 1.
        def __int__(self): calls.append('int'); return 1
    values = {'mapping': Map(), 'sequence': Seq(), 'generator': lazy(), 'callable': callback,
              'descriptor': Descriptor(), 'dataclass': Data(), 'tuple': (),
              'decimal': Decimal('1'), 'foreign-numeric': ForeignNumeric()}
    value = values[kind]; calls.clear(); d = fixture(); place(d, 'snapshot', value)
    if kind == 'mapping': list(value); value.keys(); assert calls
    elif kind == 'sequence': list(value); assert calls
    elif kind == 'generator': next(value); assert calls == ['generator']
    elif kind == 'callable': value(); assert calls == ['call']
    elif kind == 'descriptor': value.__get__(None, None); assert calls == ['get']
    elif kind == 'dataclass': value.value; assert calls == ['attr']
    elif kind == 'foreign-numeric': float(value); int(value); assert calls == ['float', 'int']
    calls.clear(); api = core_api()
    with pytest.raises(TypeError): invoke(api, d)
    assert calls == []


@pytest.mark.parametrize('entry', ['core', 'tool'])
@pytest.mark.parametrize('metadata', ['instance', 'class'])
def test_dataclass_metadata_is_never_inspected(entry, metadata):
    calls = []
    class Trap(dict):
        def values(self): calls.append('values'); raise RuntimeError('metadata')
    @dataclass
    class Data: value: int = 1
    value = Data(); trap = Trap(Data.__dataclass_fields__)
    if metadata == 'instance': value.__dataclass_fields__ = trap
    else: Data.__dataclass_fields__ = trap
    from dataclasses import fields
    with pytest.raises(RuntimeError): fields(value)
    assert calls == ['values']; calls.clear()
    api = core_api() if entry == 'core' else tool_class()
    d = fixture(); place(d, 'snapshot', value)
    with pytest.raises(TypeError): invoke(api, d)
    assert calls == []


@pytest.mark.parametrize('protocol', ['__repr__', '__str__', '__format__', '__fspath__', '__copy__', '__deepcopy__'])
@pytest.mark.parametrize('where', ['name', 'args'])
def test_tool_entry_objects_reject_without_callbacks(protocol, where):
    tools = invoke(tool_class(), fixture()); value, calls = probe(protocol)
    with pytest.raises(TypeError): tools.invoke_tool(value if where == 'name' else TOOL,
                                                    value if where == 'args' else {})
    assert calls == []


def test_constructor_snapshot_and_delegate_mutation_do_not_change_binding(monkeypatch):
    cls, function, d = tool_class(), core_api(), fixture()
    tools = invoke(cls, d); before = tools.invoke_tool(TOOL, {})
    d['group_a']['executions'].clear(); pairs(d).clear(); d['targets'].clear()
    assert tools.invoke_tool(TOOL, {}) == before
    import experiment_paired_comparison as core
    from m22_contract_helpers import patch_public
    def mutate(a, b, *, pairing_declaration, targets):
        result = function(a, b, pairing_declaration=pairing_declaration, targets=targets)
        a.clear(); b.clear(); pairing_declaration.clear(); targets.clear()
        return result
    patch_public(monkeypatch, core, TOOL, mutate)
    for _ in range(2): assert tools.invoke_tool(TOOL, {}) == before

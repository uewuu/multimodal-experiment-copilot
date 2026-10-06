"""Pure binding/projection contracts. No file bytes, clock or I/O in this layer."""
from copy import deepcopy
import hashlib
import builtins
from pathlib import Path

import pytest

from m21_contract_helpers import (api, binding, capture, assert_report, STATUSES,
                                  DATA_FIELDS, ROOT, LIMIT, CHUNK, SUCCESS_BASIS)


def test_binding_is_detached_and_preserves_host_context():
    core = api('core')
    supplied = binding()
    result = core.validate_artifact_binding(supplied)
    assert result == supplied and result is not supplied
    supplied['artifact_id'] = 'changed'
    assert result['artifact_id'] == 'metrics-1'


@pytest.mark.parametrize('field', list(binding()))
def test_each_binding_field_required(field):
    core = api('core')
    supplied = binding()
    del supplied[field]
    with pytest.raises((TypeError, ValueError)):
        core.validate_artifact_binding(supplied)


@pytest.mark.parametrize('field,value', [
    ('schema_version', 'future'), ('max_bytes', LIMIT + 1), ('max_bytes', True),
    ('max_bytes', LIMIT - 1), ('chunk_bytes', CHUNK + 1), ('chunk_bytes', False),
    ('artifact_id', ''), ('artifact_id', ' '), ('logical_role', None),
    ('artifact_ref', 'x' * 129), ('artifact_ref', '界' * 43),
    ('file_name', 'x' * 256), ('authorized_root', 'C:\\' + 'x' * 1022),
    ('authorized_root', ''), ('authorized_root', '.'), ('authorized_root', 'C:run'),
    ('authorized_root', r'\run'), ('authorized_root', r'\\server\share'),
    ('authorized_root', r'\\?\C:\run'), ('authorized_root', 'C:\\run\x00'),
    ('authorized_root', r'C:\run\..\other'), ('authorized_root', r'C:\run:stream'),
])
def test_invalid_binding_is_call_error(field, value):
    core = api('core')
    with pytest.raises((TypeError, ValueError)):
        core.validate_artifact_binding(binding(**{field: value}))


BAD_NAMES = ['', '.', '..', '../a', r'a\b', '/a', r'C:\a', 'C:a', r'\\host\file',
             r'\\.\NUL', 'a\x00b', 'a:stream', 'NUL', 'con.txt', 'PRN.json', 'AUX',
             'COM1', 'lpt9.txt', 'COM¹', 'LPT².csv', 'COM³.bin', 'a.', 'a ',
             '*.csv', 'a?', 'a<', 'a>', 'a"', 'a|', 'a\nb']


@pytest.mark.parametrize('name', BAD_NAMES)
def test_windows_single_component_name_rejections(name):
    core = api('core')
    with pytest.raises((TypeError, ValueError)):
        core.validate_artifact_binding(binding(file_name=name))


@pytest.mark.parametrize('name', ['metrics.json', 'config.yaml', 'history.json', 'COM10',
                                 'a' * 255, '界' * 85, 'experiment data.bin'])
def test_legal_component_inclusive_byte_bound(name):
    assert api('core').validate_artifact_binding(binding(file_name=name))['file_name'] == name


@pytest.mark.parametrize('field', ['sha256', 'byte_size', 'started_at', 'clock', 'trusted',
                                  'verified', 'current', 'checkpoint', 'approval', 'extra'])
def test_host_cannot_inject_observation_or_claim_fields(field):
    core = api('core')
    with pytest.raises((TypeError, ValueError)):
        core.validate_artifact_binding(binding(**{field: True}))


def test_projection_literal_oracle_is_deterministic_isolated_and_pure(monkeypatch):
    core = api('core')
    supplied, measured = binding(), capture()
    before = deepcopy((supplied, measured))
    calls = []
    def deny(*args, **kwargs):
        calls.append(True)
        raise AssertionError('pure projection performed external operation')
    with monkeypatch.context() as patch:
        for owner, name in [(builtins, 'open'), (Path, 'open'), (Path, 'resolve'),
                            (hashlib, 'sha256')]:
            patch.setattr(owner, name, deny)
        result = core.project_artifact_observation(supplied, measured)
        again = core.project_artifact_observation(supplied, measured)
    assert calls == [] and (supplied, measured) == before
    assert_report(result)
    assert result == dict(measured, schema_version='system-artifact-observation-v1',
                          artifact_id='metrics-1', logical_role='metrics', artifact_ref='host-artifact-1',
                          observation_basis=SUCCESS_BASIS, limitations=result['limitations'])
    result['limitations'].clear()
    assert again['limitations'] and measured == before[1]


@pytest.mark.parametrize('status', STATUSES[1:])
def test_failure_projection_no_partial_success_fields(status):
    core = api('core')
    measured = capture()
    measured.update({key: None for key in DATA_FIELDS})
    measured.update(status=status, consistency_status='changed' if status == 'changed_during_read' else 'not_evaluated')
    result = core.project_artifact_observation(binding(), measured)
    assert_report(result, status)
    assert ROOT not in str(result) and 'artifact.bin' not in str(result)


@pytest.mark.parametrize('field,value', [
    ('status', 'verified'), ('sha256', 'ab'), ('sha256', 'AB' * 32),
    ('byte_size', True), ('byte_size', 2), ('size_before', 4), ('size_after', 4),
    ('mtime_ns_before', -1), ('mtime_ns_after', 43), ('mtime_ns_after', 2**63),
    ('started_at', 'host-time'), ('completed_at', '2025-10-06T01:00:00.000000Z'),
    ('started_at', '2026-10-06T01:00:00'), ('consistency_status', 'atomic_snapshot_verified'),
    ('contents', 'abc'), ('file_id', [1, 2]), ('error', 'private path'),
])
def test_invalid_capture_cannot_be_projected_as_observation(field, value):
    core = api('core')
    measured = capture()
    measured[field] = value
    with pytest.raises((TypeError, ValueError)):
        core.project_artifact_observation(binding(), measured)


def test_nonobserved_capture_cannot_smuggle_digest():
    core = api('core')
    measured = capture()
    measured.update(status='oversized', consistency_status='not_evaluated')
    with pytest.raises((TypeError, ValueError)):
        core.project_artifact_observation(binding(), measured)


def test_max_legal_context_bounded_output():
    core = api('core')
    supplied = binding(artifact_id='界' * 42 + 'xx', logical_role='x' * 128,
                       artifact_ref='x' * 128, file_name='界' * 85,
                       authorized_root='C:\\' + 'a' * 1021)
    assert_report(core.project_artifact_observation(supplied, capture()))


def test_projection_source_has_no_acquisition_hash_or_current_clock():
    import ast
    import inspect
    core = api('core')
    tree = ast.parse(inspect.getsource(core))
    forbidden_calls = {'open', 'read_bytes', 'read_text', 'stat', 'fstat', 'resolve',
                       'sha256', 'now', 'utcnow', 'time', 'perf_counter', 'monotonic'}
    called = {node.func.id if isinstance(node.func, ast.Name) else node.func.attr
              for node in ast.walk(tree) if isinstance(node, ast.Call)
              and isinstance(node.func, (ast.Name, ast.Attribute))}
    assert not called & forbidden_calls

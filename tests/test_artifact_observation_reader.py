"""Bounded live Reader over a deterministic already-authorized handle seam."""
from contextlib import contextmanager
import hashlib
import json

import pytest

from m21_contract_helpers import (api, binding, Stream, install_stream, assert_report,
                                  patch_alias, LIMIT, CHUNK, ABC_SHA, EMPTY_SHA, ROOT)


@pytest.mark.parametrize('data', [b'', b'abc', b'x' * CHUNK, b'x' * (CHUNK + 7), b'x' * LIMIT],
                         ids=['empty', 'known', 'one-chunk', 'multi-chunk', 'exact-cap'])
def test_success_actual_count_stream_digest_timestamps_same_handle(monkeypatch, data):
    expected = hashlib.sha256(data).hexdigest()
    stream = Stream(data)
    observe = install_stream(monkeypatch, stream)
    result = observe(binding())
    assert_report(result)
    assert result['sha256'] == expected and result['byte_size'] == len(data)
    assert result['started_at'] == '2026-10-06T01:00:00.000000Z'
    assert result['completed_at'] == '2026-10-06T01:00:01.000000Z'
    assert stream.opens == stream.closes == 1
    assert stream.events[0] == stream.events[-1] == 'metadata'
    assert stream.events.count('metadata') == 2
    assert max(stream.requests) <= CHUNK and stream.position == len(data)
    assert result['size_before'] == result['size_after'] == len(data)


@pytest.mark.parametrize('mode', ['pre-size', 'overflow', 'growth'])
def test_oversized_no_pre_read_or_partial_digest(monkeypatch, mode):
    stream = Stream(b'x' * (LIMIT + 50), before={'size': LIMIT + 1 if mode == 'pre-size' else LIMIT})
    if mode == 'growth':
        stream.after['size'] = LIMIT + 50
    observe = install_stream(monkeypatch, stream)
    result = observe(binding(logical_role='checkpoint'))
    assert_report(result, 'oversized')
    assert stream.position == (0 if mode == 'pre-size' else LIMIT + 1)
    if mode == 'pre-size':
        assert stream.requests == []
    else:
        assert stream.requests[-1] == 1
    assert stream.opens == stream.closes == 1


@pytest.mark.parametrize('after,before', [({'file_id': (9, 9)}, {}), ({'size': 4}, {}),
    ({'mtime_ns': 43}, {}), ({'change_token': 8}, {}), ({}, {'size': 4})],
    ids=['identity', 'size', 'mtime', 'change-indicator', 'actual-count'])
def test_changed_observation_discards_digest(monkeypatch, after, before):
    stream = Stream(before=before, after=after)
    result = install_stream(monkeypatch, stream)(binding())
    assert_report(result, 'changed_during_read')
    assert stream.opens == stream.closes == 1


@pytest.mark.parametrize('status', ['missing', 'unauthorized', 'unreadable', 'unsupported_type',
                                   'unsupported_platform', 'io_error'])
def test_safe_open_rejection_before_read_and_hash_update(monkeypatch, status):
    reader, backend = api('reader'), api('io')
    accesses = []
    @contextmanager
    def rejected(*args):
        accesses.append('open')
        raise backend.ObservationIOError(status)
        yield  # pragma: no cover - contextmanager shape, never a success fallback
    patch_alias(monkeypatch, backend, 'open_authorized_artifact', reader, rejected)
    original = hashlib.sha256
    class Digest:
        def update(self, data):
            accesses.append('hash-bytes')
            raise AssertionError('hash before authorization')
        def hexdigest(self):
            return original().hexdigest()
    patch_alias(monkeypatch, hashlib, 'sha256', reader, lambda *a, **k: Digest())
    result = reader.observe_artifact(binding())
    assert_report(result, status)
    assert accesses == ['open']


@pytest.mark.parametrize('error,status', [(FileNotFoundError(2, 'C:\\Users\\PRIVATE'), 'missing'),
    (PermissionError(13, 'C:\\Users\\PRIVATE'), 'unreadable'), (OSError(5, 'PRIVATE'), 'io_error')])
def test_mid_read_os_errors_safe_report_no_retry(monkeypatch, error, status):
    stream = Stream(error=error)
    result = install_stream(monkeypatch, stream)(binding())
    assert_report(result, status)
    assert 'PRIVATE' not in json.dumps(result) and ROOT not in json.dumps(result)
    assert stream.opens == stream.closes == 1 and len(stream.requests) == 1


def test_reader_program_bug_not_mislabeled_io_error(monkeypatch):
    failure = RuntimeError('bug sentinel')
    stream = Stream(error=failure)
    observe = install_stream(monkeypatch, stream)
    with pytest.raises(RuntimeError) as caught:
        observe(binding())
    assert caught.value is failure and stream.closes == 1


def test_invalid_binding_before_authorized_open(monkeypatch):
    stream = Stream()
    observe = install_stream(monkeypatch, stream)
    with pytest.raises((TypeError, ValueError)):
        observe(binding(file_name='../outside'))
    assert stream.opens == stream.position == 0


def test_incremental_hash_updates_exactly_observed_stream(monkeypatch):
    stream = Stream(b'x' * (CHUNK + 5))
    observe = install_stream(monkeypatch, stream)
    original, updates = hashlib.sha256, []
    class Digest:
        def __init__(self, data=b''):
            assert data == b''
            self.delegate = original()
        def update(self, data):
            updates.append(data)
            self.delegate.update(data)
        def hexdigest(self):
            return self.delegate.hexdigest()
    patch_alias(monkeypatch, hashlib, 'sha256', api('reader'), Digest)
    result = observe(binding())
    assert b''.join(updates) == stream.data and max(map(len, updates)) <= CHUNK
    assert result['sha256'] == original(stream.data).hexdigest()


def test_reader_delegates_pure_projection_without_reinterpreting(monkeypatch):
    stream = Stream()
    observe = install_stream(monkeypatch, stream)
    core, reader = api('core'), api('reader')
    original, calls = core.project_artifact_observation, []
    def traced(selected, measured):
        result = original(selected, measured)
        calls.append(result)
        return result
    patch_alias(monkeypatch, core, 'project_artifact_observation', reader, traced)
    result = observe(binding())
    assert calls == [result] and result['sha256'] == ABC_SHA


def test_stream_and_digest_oracles_calibrated_without_production():
    assert hashlib.sha256(b'').hexdigest() == EMPTY_SHA
    assert hashlib.sha256(b'abc').hexdigest() == ABC_SHA
    stream = Stream()
    with stream.open(ROOT, 'artifact.bin') as handle:
        assert handle.metadata()['size'] == 3
        assert handle.read(2) == b'ab' and handle.read(2) == b'c' and handle.read(2) == b''
    assert stream.position == 3 and stream.opens == stream.closes == 1


def test_no_writes_discovery_parsing_memory_loading_network_or_parallelism(monkeypatch):
    from m21_contract_helpers import deny_side_effects
    stream = Stream(b'not JSON or YAML or checkpoint serialization')
    observe = install_stream(monkeypatch, stream)
    api('core')
    with deny_side_effects(monkeypatch) as calls:
        report = observe(binding(logical_role='checkpoint'))
    assert calls == []
    assert_report(report)
    assert report['byte_size'] == len(stream.data)


@pytest.mark.parametrize('entrance', ['write', 'append', 'truncate', 'rename', 'delete', 'chmod',
    'touch', 'copy', 'move', 'scan', 'glob', 'walk', 'memory', 'parse', 'm20', 'checkpoint', 'network', 'parallel'])
def test_no_side_effect_probe_calibration(monkeypatch, entrance):
    import os
    from pathlib import Path
    import shutil
    import socket
    import concurrent.futures
    from experiment_memory import ExperimentMemory
    import read_test_metrics
    import experiment_artifact_verification
    from m21_contract_helpers import deny_side_effects
    actions = {
        'write': lambda: Path('never-write').write_bytes(b'x'),
        'append': lambda: open('never-append', 'ab'),
        'truncate': lambda: os.truncate('never-truncate', 0),
        'rename': lambda: os.rename('never-rename', 'never-target'),
        'delete': lambda: os.unlink('never-delete'),
        'chmod': lambda: os.chmod('never-chmod', 0),
        'touch': lambda: Path('never-touch').touch(),
        'copy': lambda: shutil.copy('never-copy', 'never-target'),
        'move': lambda: shutil.move('never-move', 'never-target'),
        'scan': lambda: os.scandir('never-scan'),
        'glob': lambda: list(Path('never-glob').glob('*')),
        'walk': lambda: os.walk('never-walk'),
        'memory': lambda: ExperimentMemory.get(None, None),
        'parse': lambda: read_test_metrics.read_test_metrics(None, ()),
        'm20': lambda: experiment_artifact_verification.compare_artifact_descriptors({}, []),
        'checkpoint': lambda: __import__('torch').load('never-load'),
        'network': lambda: socket.create_connection(('invalid.example', 443)),
        'parallel': lambda: concurrent.futures.ThreadPoolExecutor(),
    }
    with deny_side_effects(monkeypatch) as calls:
        with pytest.raises(AssertionError, match='forbidden side effect sentinel'):
            actions[entrance]()
    assert calls == [True]


def test_digest_bug_propagates_and_handle_is_closed(monkeypatch):
    stream = Stream()
    observe = install_stream(monkeypatch, stream)
    failure = RuntimeError('digest implementation bug')
    class Digest:
        def update(self, data):
            raise failure
        def hexdigest(self):
            raise failure
    patch_alias(monkeypatch, hashlib, 'sha256', api('reader'), lambda *args: Digest())
    with pytest.raises(RuntimeError) as caught:
        observe(binding())
    assert caught.value is failure and stream.opens == stream.closes == 1

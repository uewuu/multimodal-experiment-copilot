"""Raw native seam tests: authorization before bytes, fixed access, fail closed."""
import builtins
from pathlib import Path

import pytest

from m21_contract_helpers import api, Native, install_native, ROOT


def test_native_open_flags_order_same_handle_and_cleanup(monkeypatch):
    native = Native()
    module = install_native(monkeypatch, native)
    with module.open_authorized_artifact(ROOT, 'artifact.bin') as handle:
        before = handle.metadata()
        assert handle.read(2) == b'ab'
        assert handle.read(2) == b'c'
        after = handle.metadata()
    assert before == after == dict(size=3, mtime_ns=42, file_id=(1, 2), change_token=7)
    opens = [e for e in native.events if e[0] == 'open']
    assert len(opens) == 2
    assert opens[0][1] == ROOT and opens[1][1] == ROOT + r'\artifact.bin'
    for index, entry in enumerate(opens):
        options = entry[2]
        assert options['access'] == (0x80 if index == 0 else 0x80000000)
        assert options['share'] == 1 and options['disposition'] == 3
        assert options['flags'] & 0x00200000
    assert opens[0][2]['flags'] & 0x02000000
    first_read = next(i for i, e in enumerate(native.events) if e[0] == 'read')
    assert sum(e[:2] == ('describe', native.root) for e in native.events[:first_read]) >= 2
    assert ('describe', native.file) in native.events[:first_read]
    assert [e[1] for e in native.events if e[0] == 'read'] == [native.file, native.file]
    assert [e[1] for e in native.events if e[0] == 'close'].count(native.file) == 1
    assert [e[1] for e in native.events if e[0] == 'close'].count(native.root) == 1


@pytest.mark.parametrize('where,update,status', [
    ('root', {'reparse': True}, 'unsupported_type'),
    ('root', {'kind': 'file'}, 'unsupported_type'),
    ('root', {'filesystem': 'FAT32'}, 'unsupported_platform'),
    ('root', {'local': False}, 'unsupported_platform'),
    ('root', {'final_path': r'C:\outside'}, 'unauthorized'),
    ('file', {'kind': 'directory'}, 'unsupported_type'),
    ('file', {'kind': 'fifo'}, 'unsupported_type'),
    ('file', {'kind': 'socket'}, 'unsupported_type'),
    ('file', {'kind': 'device'}, 'unsupported_type'),
    ('file', {'kind': 'unknown'}, 'unsupported_type'),
    ('file', {'reparse': True}, 'unsupported_type'),
    ('file', {'filesystem': 'ReFS'}, 'unsupported_platform'),
    ('file', {'local': False}, 'unsupported_platform'),
    ('file', {'final_path': r'C:\outside\artifact.bin'}, 'unauthorized'),
    ('file', {'final_path': ROOT + r'\nested\artifact.bin'}, 'unauthorized'),
    ('file', {'final_path': ROOT + r'\other.bin'}, 'unauthorized'),
    ('file', {'volume_id': 2}, 'unauthorized'),
    ('after', {'file_id': (2, 9)}, 'unauthorized'),
])
def test_rejection_before_content_and_all_acquired_handles_closed(monkeypatch, where, update, status):
    native = Native(**{{'root': 'root_update', 'file': 'file_update', 'after': 'root_after'}[where]: update})
    module = install_native(monkeypatch, native)
    with pytest.raises(module.ObservationIOError) as caught:
        with module.open_authorized_artifact(ROOT, 'artifact.bin') as handle:
            handle.read(1)
    assert caught.value.status == status
    assert not any(e[0] == 'read' for e in native.events)
    opened = len([e for e in native.events if e[0] == 'open'])
    assert len([e for e in native.events if e[0] == 'close']) == opened


@pytest.mark.parametrize('platform', ['linux', 'darwin', 'unknown'])
def test_unsupported_platform_before_native_open(monkeypatch, platform):
    native = Native()
    module = install_native(monkeypatch, native)
    monkeypatch.setattr(module, '_platform_name', lambda: platform)
    with pytest.raises(module.ObservationIOError) as caught:
        with module.open_authorized_artifact(ROOT, 'artifact.bin'):
            pytest.fail('unsupported platform yielded handle')
    assert caught.value.status == 'unsupported_platform' and native.events == []


@pytest.mark.parametrize('error,status', [(FileNotFoundError(2, 'PRIVATE'), 'missing'),
    (PermissionError(13, 'PRIVATE'), 'unreadable'), (OSError(5, 'PRIVATE'), 'io_error')])
def test_native_errors_no_retry_or_python_fallback(monkeypatch, error, status):
    native = Native(error=error)
    module = install_native(monkeypatch, native)
    calls = []
    def forbidden(*args, **kwargs):
        calls.append(True)
        raise AssertionError('unsafe generic fallback')
    with monkeypatch.context() as patch:
        patch.setattr(builtins, 'open', forbidden)
        patch.setattr(Path, 'open', forbidden)
        with pytest.raises(module.ObservationIOError) as caught:
            with module.open_authorized_artifact(ROOT, 'artifact.bin'):
                pytest.fail('failed native open yielded handle')
    assert caught.value.status == status and calls == []
    assert len([e for e in native.events if e[0] == 'open']) == 1


def test_backend_unavailable_is_fail_closed(monkeypatch):
    module = api('io')
    monkeypatch.setattr(module, '_platform_name', lambda: 'win32')
    monkeypatch.setattr(module, '_native', None)
    with pytest.raises(module.ObservationIOError) as caught:
        with module.open_authorized_artifact(ROOT, 'artifact.bin'):
            pytest.fail('backend unavailable')
    assert caught.value.status == 'unsupported_platform'


@pytest.mark.parametrize('phase', ['describe', 'read', 'body'])
def test_unexpected_bug_propagates_and_closes_handles(monkeypatch, phase):
    native = Native()
    module = install_native(monkeypatch, native)
    failure = RuntimeError('program bug')
    def fail(*args, **kwargs):
        raise failure
    if phase != 'body':
        monkeypatch.setattr(native, phase, fail)
    with pytest.raises(RuntimeError) as caught:
        with module.open_authorized_artifact(ROOT, 'artifact.bin') as handle:
            if phase == 'body':
                raise failure
            handle.read(1)
    assert caught.value is failure
    assert len([e for e in native.events if e[0] == 'close']) == len([e for e in native.events if e[0] == 'open'])


@pytest.mark.parametrize('root,name', [(ROOT, '../outside'), (ROOT, 'NUL'),
                                      ('.', 'artifact.bin'), (ROOT, 'a:stream')])
def test_invalid_open_arguments_do_not_touch_native(monkeypatch, root, name):
    native = Native()
    module = install_native(monkeypatch, native)
    with pytest.raises((TypeError, ValueError)):
        with module.open_authorized_artifact(root, name):
            pytest.fail('invalid arguments accepted')
    assert native.events == []


def test_native_probe_calibration_is_independent_of_missing_backend():
    native = Native()
    root = native.open(ROOT, access=0x80)
    child = native.open(ROOT + r'\artifact.bin', access=0x80000000)
    assert root is native.root and child is native.file
    assert native.describe(child)['size'] == 3 and native.read(child, 2) == b'ab'
    native.close(child)
    native.close(root)
    assert [e[0] for e in native.events] == ['open', 'open', 'describe', 'read', 'close', 'close']


def test_child_open_failure_closes_root_without_retry(monkeypatch):
    native = Native()
    module = install_native(monkeypatch, native)
    original = native.open
    def open_child_fails(path, **options):
        if path != ROOT:
            native.events.append(('open', path, options))
            raise FileNotFoundError(2, 'private child path')
        return original(path, **options)
    monkeypatch.setattr(native, 'open', open_child_fails)
    with pytest.raises(module.ObservationIOError) as caught:
        with module.open_authorized_artifact(ROOT, 'artifact.bin'):
            pytest.fail('missing child accepted')
    assert caught.value.status == 'missing'
    assert [e for e in native.events if e[0] == 'close'] == [('close', native.root)]
    assert len([e for e in native.events if e[0] == 'open']) == 2
    assert not any(e[0] == 'read' for e in native.events)


def test_final_handle_extended_path_and_case_normalization(monkeypatch):
    native = Native(root_update={'final_path': '\\\\?\\' + ROOT.upper()},
                    file_update={'final_path': '\\\\?\\' + ROOT.upper() + r'\ARTIFACT.BIN'})
    module = install_native(monkeypatch, native)
    with module.open_authorized_artifact(ROOT, 'artifact.bin') as handle:
        assert handle.read(3) == b'abc'

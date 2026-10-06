"""Real Windows/local-NTFS acceptance. No skip, no mock standing in for native I/O."""
from contextlib import contextmanager
import ctypes
from ctypes import wintypes
import os
from pathlib import Path
import subprocess
import sys

import pytest

from m21_contract_helpers import api, binding, assert_report, ABC_SHA


def kernel():
    assert sys.platform == 'win32', 'unsupported platform: Windows acceptance must run on Windows'
    dll = ctypes.WinDLL('kernel32', use_last_error=True)
    dll.CreateFileW.argtypes = [wintypes.LPCWSTR, wintypes.DWORD, wintypes.DWORD,
        ctypes.c_void_p, wintypes.DWORD, wintypes.DWORD, wintypes.HANDLE]
    dll.CreateFileW.restype = wintypes.HANDLE
    dll.CloseHandle.argtypes = [wintypes.HANDLE]
    dll.CloseHandle.restype = wintypes.BOOL
    dll.GetVolumeInformationW.argtypes = [wintypes.LPCWSTR, wintypes.LPWSTR, wintypes.DWORD,
        ctypes.POINTER(wintypes.DWORD), ctypes.POINTER(wintypes.DWORD), ctypes.POINTER(wintypes.DWORD),
        wintypes.LPWSTR, wintypes.DWORD]
    dll.GetVolumeInformationW.restype = wintypes.BOOL
    dll.GetDriveTypeW.argtypes = [wintypes.LPCWSTR]
    dll.GetDriveTypeW.restype = wintypes.UINT
    return dll


def require_ntfs(root):
    dll = kernel()
    volume = Path(root).anchor
    fs = ctypes.create_unicode_buffer(64)
    assert dll.GetVolumeInformationW(volume, None, 0, None, None, None, fs, len(fs)), ctypes.get_last_error()
    assert fs.value == 'NTFS' and dll.GetDriveTypeW(volume) == 3, 'local fixed NTFS acceptance required'
    return dll


@contextmanager
def writer_handle(path):
    """Test fixture opens an existing file for write ACCESS, never writes bytes."""
    dll = kernel()
    handle = dll.CreateFileW(str(path), 0x40000000, 7, None, 3, 0, None)
    assert handle != ctypes.c_void_p(-1).value, ctypes.get_last_error()
    try:
        yield handle
    finally:
        assert dll.CloseHandle(handle)


@contextmanager
def junction(path, target):
    assert sys.platform == 'win32'
    result = subprocess.run(['cmd', '/c', 'mklink', '/J', str(path), str(target)], capture_output=True)
    assert result.returncode == 0, result.stderr
    try:
        yield path
    finally:
        # Remove only the link itself; never recurse into the target.
        path.rmdir()


def test_windows_native_fixture_calibration_ntfs_share_and_junction(tmp_path):
    dll = require_ntfs(tmp_path)
    path = tmp_path / 'artifact.bin'
    path.write_bytes(b'abc')
    with writer_handle(path):
        blocked = dll.CreateFileW(str(path), 0x80000000, 1, None, 3, 0, None)
        code = ctypes.get_last_error()
        if blocked != ctypes.c_void_p(-1).value:
            dll.CloseHandle(blocked)
        assert blocked == ctypes.c_void_p(-1).value and code == 32
    target = tmp_path / 'target'
    target.mkdir()
    with junction(tmp_path / 'link', target) as link:
        assert os.lstat(link).st_file_attributes & 0x400
        assert link.resolve() == target.resolve()


def test_real_safe_open_same_handle_and_close(tmp_path):
    backend = api('io')
    require_ntfs(tmp_path)
    path = tmp_path / 'artifact.bin'
    path.write_bytes(b'abc')
    with backend.open_authorized_artifact(str(tmp_path), 'artifact.bin') as handle:
        before = handle.metadata()
        assert handle.read(2) == b'ab' and handle.read(2) == b'c'
        after = handle.metadata()
    assert before == after and before['size'] == 3
    # A leaked non-write-sharing read handle would prevent this writer open.
    with writer_handle(path):
        pass


def test_real_existing_writer_prevents_safe_read_without_retry(tmp_path):
    backend = api('io')
    require_ntfs(tmp_path)
    path = tmp_path / 'artifact.bin'
    path.write_bytes(b'abc')
    with writer_handle(path):
        with pytest.raises(backend.ObservationIOError) as caught:
            with backend.open_authorized_artifact(str(tmp_path), 'artifact.bin'):
                pytest.fail('safe open ignored existing writer')
    assert caught.value.status == 'unreadable'


@pytest.mark.parametrize('where', ['root', 'child'])
def test_real_junction_is_rejected_before_handle_is_exposed(tmp_path, where):
    backend = api('io')
    require_ntfs(tmp_path)
    outside = tmp_path / 'outside'
    outside.mkdir()
    (outside / 'artifact.bin').write_bytes(b'outside')
    allowed = tmp_path / 'allowed'
    allowed.mkdir()
    link = tmp_path / 'linked-root' if where == 'root' else allowed / 'artifact.bin'
    with junction(link, outside):
        root = link if where == 'root' else allowed
        with pytest.raises(backend.ObservationIOError) as caught:
            with backend.open_authorized_artifact(str(root), 'artifact.bin'):
                pytest.fail('reparse object yielded readable handle')
        assert caught.value.status == 'unsupported_type'


def test_real_directory_not_regular_file(tmp_path):
    backend = api('io')
    require_ntfs(tmp_path)
    (tmp_path / 'artifact.bin').mkdir()
    with pytest.raises(backend.ObservationIOError) as caught:
        with backend.open_authorized_artifact(str(tmp_path), 'artifact.bin'):
            pytest.fail('directory accepted')
    assert caught.value.status == 'unsupported_type'


def test_real_reader_known_digest_and_missing(tmp_path):
    reader = api('reader')
    require_ntfs(tmp_path)
    assert_report(reader.observe_artifact(binding(str(tmp_path))), 'missing')
    (tmp_path / 'artifact.bin').write_bytes(b'abc')
    result = reader.observe_artifact(binding(str(tmp_path)))
    assert_report(result)
    assert result['sha256'] == ABC_SHA and result['byte_size'] == 3


def test_real_reader_no_generic_open_or_explicit_side_effects(monkeypatch, tmp_path):
    from m21_contract_helpers import deny_side_effects
    reader = api('reader')
    api('io')
    api('core')
    require_ntfs(tmp_path)
    (tmp_path / 'artifact.bin').write_bytes(b'abc')
    with deny_side_effects(monkeypatch) as calls:
        result = reader.observe_artifact(binding(str(tmp_path)))
    assert calls == []
    assert_report(result)
    assert result['sha256'] == ABC_SHA

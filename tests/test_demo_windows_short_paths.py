"""Real Windows 8.3 aliases at Demo host boundaries; no simulated native PASS.

This file belongs to Windows Full, not the portable allowlist. An unavailable
alias or unsupported volume fails with a capability diagnostic, never a skip.
Only the Demo's temporary-directory factory is redirected; TEMP/TMP and all
production authorization, Reader and native handle checks remain unchanged.
"""

from contextlib import contextmanager
import ctypes
from ctypes import wintypes
import hashlib
import os
from pathlib import Path
import sys
from tempfile import TemporaryDirectory

import pytest

from m23_release_contract_helpers import ROOT, DEMO, assert_demo, load_future, offline


@pytest.fixture
def workspace_factory(tmp_path):
    assert sys.platform == 'win32', 'real Windows 8.3 acceptance requires Windows'
    parent = tmp_path / 'owned-long-directory-for-alias-regression'
    parent.mkdir()
    parent = parent.resolve(strict=True)
    dll = ctypes.WinDLL('kernel32', use_last_error=True)
    dll.GetShortPathNameW.argtypes = [wintypes.LPCWSTR, wintypes.LPWSTR, wintypes.DWORD]
    dll.GetShortPathNameW.restype = wintypes.DWORD
    dll.GetDriveTypeW.argtypes = [wintypes.LPCWSTR]
    dll.GetDriveTypeW.restype = wintypes.UINT
    dll.GetVolumeInformationW.argtypes = [wintypes.LPCWSTR, wintypes.LPWSTR, wintypes.DWORD,
        ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, wintypes.LPWSTR, wintypes.DWORD]
    dll.GetVolumeInformationW.restype = wintypes.BOOL
    filesystem = ctypes.create_unicode_buffer(64)
    assert dll.GetVolumeInformationW(parent.anchor, None, 0, None, None, None,
                                    filesystem, len(filesystem)), ctypes.get_last_error()
    assert filesystem.value == 'NTFS' and dll.GetDriveTypeW(parent.anchor) == 3
    buffer = ctypes.create_unicode_buffer(32768)
    length = dll.GetShortPathNameW(str(parent), buffer, len(buffer))
    assert 0 < length < len(buffer), 'GetShortPathNameW did not return an alias'
    short = buffer.value
    assert os.path.normcase(short) != os.path.normcase(str(parent)), 'real 8.3 alias unavailable'
    assert os.path.samefile(short, parent)
    assert Path(short).resolve(strict=True) == parent
    assert not os.lstat(parent).st_file_attributes & 0x400, 'reparse fixture forbidden'
    created = []

    def factory(spelling):
        @contextmanager
        def temporary(*, prefix):
            directory = short if spelling == 'short' else str(parent)
            with TemporaryDirectory(prefix=prefix, dir=directory) as raw:
                canonical = Path(raw).resolve(strict=True)
                assert os.path.samefile(raw, canonical)
                assert (Path(raw) != canonical) == (spelling == 'short')
                assert not os.lstat(raw).st_file_attributes & 0x400
                created.append(canonical)
                yield raw
        return temporary

    return factory, created


@pytest.mark.parametrize('spelling', ['short', 'canonical'])
def test_deterministic_demo_canonicalizes_before_fixture_and_context(
        monkeypatch, workspace_factory, spelling):
    module = load_future(ROOT / 'examples/deterministic_copilot_demo.py', monkeypatch)
    factory, created = workspace_factory
    monkeypatch.setattr(module, 'TemporaryDirectory', factory(spelling))
    offline(monkeypatch)
    original_write = module._write_experiment
    original_run = module.CopilotService.run
    writes, contexts = [], []

    def write(path, **kwargs):
        assert path.parent == path.parent.resolve(strict=True)
        writes.append(path)
        return original_write(path, **kwargs)

    def run(self, *args, **kwargs):
        root = Path(kwargs['experiment_context']['experiment_root'])
        assert root == root.resolve(strict=True)
        contexts.append(root)
        return original_run(self, *args, **kwargs)

    monkeypatch.setattr(module, '_write_experiment', write)
    monkeypatch.setattr(module.CopilotService, 'run', run)
    result = module.build_demo_result()
    assert result['runtime'] == {'provider_request_count': 2, 'tool_invocation_count': 1}
    assert len(writes) == 2 and contexts == created
    assert len(created) == 1 and not created[0].exists()


@pytest.mark.parametrize('spelling', ['short', 'canonical'])
@pytest.mark.parametrize('scenario', ['A', 'B', 'G'])
def test_v1_demo_canonical_host_root_reaches_real_stack(
        monkeypatch, workspace_factory, spelling, scenario):
    module = load_future(DEMO, monkeypatch)
    factory, created = workspace_factory
    monkeypatch.setattr(module, 'TemporaryDirectory', factory(spelling))
    offline(monkeypatch)
    original_host = module._host_binding
    original_fixture = module.fixtures.write_experiment
    import artifact_observation_reader as reader
    import artifact_observation_io as backend
    original_reader, original_open = reader.observe_artifact, backend._native.open
    contexts, reads, opens = [], [], []

    def fixture(workspace, *args, **kwargs):
        assert workspace == workspace.resolve(strict=True)
        return original_fixture(workspace, *args, **kwargs)

    def host(name, stack):
        tool, context = original_host(name, stack)
        root = Path(next(iter(context.values())))
        assert root == root.resolve(strict=True)
        contexts.append(root)
        if name == 'G':
            assert tool._binding['authorized_root'] == str(root)
        return tool, context

    def observe(binding):
        reads.append(binding['authorized_root'])
        return original_reader(binding)

    def native_open(path, **kwargs):
        opens.append(path)
        return original_open(path, **kwargs)

    monkeypatch.setattr(module.fixtures, 'write_experiment', fixture)
    monkeypatch.setattr(module, '_host_binding', host)
    monkeypatch.setattr(reader, 'observe_artifact', observe)
    monkeypatch.setattr(backend._native, 'open', native_open)
    report = module.run_scenario(scenario)
    assert_demo(report, scenario)
    assert len(contexts) == 1 and len(created) == 1 and not created[0].exists()
    if scenario == 'G':
        assert reads == [str(contexts[0])] and len(opens) == 2
        assert report['result']['byte_size'] == 3
        assert report['result']['sha256'] == hashlib.sha256(b'abc').hexdigest()
    else:
        assert reads == opens == []


def test_short_alias_g_unsupported_backend_keeps_exit_three(
        monkeypatch, workspace_factory, capsys):
    module = load_future(DEMO, monkeypatch)
    factory, created = workspace_factory
    monkeypatch.setattr(module, 'TemporaryDirectory', factory('short'))
    offline(monkeypatch)
    import artifact_observation_io as backend
    monkeypatch.setattr(backend, '_platform_name', lambda: 'linux')

    def deny(*args, **kwargs):
        pytest.fail('unsupported backend must not acquire native handles')

    monkeypatch.setattr(backend._native, 'open', deny)
    assert module.main(['--scenario', 'G']) == 3
    import json
    report = json.loads(capsys.readouterr().out)
    assert report['status'] == 'unsupported' and report['error_code'] is None
    assert report['facts'] == {'status': 'unsupported_platform'}
    assert len(created) == 1 and not created[0].exists()


def test_short_alias_does_not_authorize_a_different_session_root(
        monkeypatch, workspace_factory, tmp_path):
    module = load_future(DEMO, monkeypatch)
    factory, created = workspace_factory
    monkeypatch.setattr(module, 'TemporaryDirectory', factory('short'))
    offline(monkeypatch)
    other = tmp_path / 'different-root'
    other.mkdir()
    other = other.resolve(strict=True)
    original_host = module._host_binding
    original_ask = module.CopilotSession.ask_with_observability
    from llm_adapters.tool_binding import _get_bound_tools
    import artifact_observation_reader as reader
    sessions, reads = [], []

    def host(name, stack):
        tool, context = original_host(name, stack)
        assert not os.path.samefile(tool._binding['authorized_root'], other)
        return tool, {'experiment_dir': str(other)}

    def ask(self, *args, **kwargs):
        sessions.append(self)
        return original_ask(self, *args, **kwargs)

    def deny(*args, **kwargs):
        reads.append(True)
        pytest.fail('different actual root must be rejected before Reader')

    monkeypatch.setattr(module, '_host_binding', host)
    monkeypatch.setattr(module.CopilotSession, 'ask_with_observability', ask)
    monkeypatch.setattr(reader, 'observe_artifact', deny)
    previous = _get_bound_tools()
    report = module.run_scenario('G')
    assert report['status'] == 'error' and report['error_code'] == 'execution_failed'
    assert report['result'] is None and reads == []
    assert len(sessions) == 1 and sessions[0].history == ()
    assert _get_bound_tools() is previous
    assert len(created) == 1 and not created[0].exists()

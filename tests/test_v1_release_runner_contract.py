"""Real tiny subprocess acceptance plus deliberately corrupt child transports.

No test invokes the runner against this repository, avoiding recursive runs.
Fault doubles are not native/whole-repository acceptance evidence.
"""

from copy import deepcopy
import os
from pathlib import Path
import subprocess
import sys

import pytest

from m23_release_contract_helpers import (
    COUNTS, RUNNER, load_future, strict_json, tiny_repo, transport,
)


def assert_success(report, files):
    assert report['schema_version'] == 'v1-release-checks-v1'
    assert report['status'] == 'passed' and report['reason_codes'] == []
    assert report['lane'] == 'windows-full'
    assert report['selected_files'] == files and report['excluded_files'] == []
    ids = [file + '::test_one' for file in files]
    assert report['collection_node_ids'] == report['executed_node_ids'] == ids
    assert report['collection_total'] == report['executed_total'] == len(ids)
    assert report['coverage_complete'] is True
    assert report['counts'] == {key: len(ids) if key == 'passed' else 0 for key in COUNTS}
    assert [r['file'] for r in report['files']] == files
    assert all(row['exit_code'] == 0 for row in report['files'])
    assert report['python_version'] and report['platform'] and report['command_environment']


def test_runner_dynamic_collection_and_exact_node_identity_union(monkeypatch, tmp_path):
    module = load_future(RUNNER, monkeypatch)
    root, files = tiny_repo(tmp_path)
    collection, reports, calls = transport(monkeypatch, module, files)
    report = module.run_checks(root, lane='windows-full')
    assert_success(report, files)
    assert calls == [('collect', tuple(files)), *(('execute', f) for f in files)]
    # New file must change the collection; not frozen to 83 files or 4576 IDs.
    new = 'tests/test_added.py'
    (root / new).write_text('def test_one(): pass\n', encoding='utf8')
    files.append(new)
    files.sort()
    collection['node_ids'].append(new + '::test_one')
    collection['node_ids'].sort()
    reports[new] = dict(exit_code=0, node_ids=[new + '::test_one'],
                        outcomes=[dict(node_id=new + '::test_one', outcome='passed')], errors=[])
    assert_success(module.run_checks(root, lane='windows-full'), files)


@pytest.mark.parametrize('fault,reason', [
    ('file-omitted', 'coverage_mismatch'), ('node-omitted', 'coverage_mismatch'),
    ('same-count-wrong-id', 'coverage_mismatch'), ('extra-id', 'coverage_mismatch'),
    ('duplicate', 'duplicate_node_id'), ('collection-duplicate', 'duplicate_node_id'),
    ('file-collection-mismatch', 'coverage_mismatch'),
    ('file-collection-duplicate', 'duplicate_node_id'),
    ('collection-error', 'collection_failed'), ('collection-missing', 'missing_report'),
    ('empty-collection', 'empty_collection'), ('missing-report', 'missing_report'),
    ('malformed-report', 'malformed_report'), ('malformed-outcome', 'malformed_report'),
    ('empty-result', 'coverage_mismatch'), ('exit-failure', 'subprocess_failed'),
    ('start-failure', 'subprocess_start_failed'), ('timeout', 'timeout'),
    ('interrupt', 'interrupted'),
])
def test_runner_fail_closed_incomplete_or_corrupt_evidence(monkeypatch, tmp_path, fault, reason):
    module = load_future(RUNNER, monkeypatch)
    root, files = tiny_repo(tmp_path)
    collection, reports, calls = transport(monkeypatch, module, files)
    file = files[0]
    first = reports[file]
    if fault == 'file-omitted': collection['node_ids'] = collection['node_ids'][1:]
    elif fault == 'node-omitted': first['outcomes'] = []
    elif fault == 'same-count-wrong-id': first['outcomes'][0]['node_id'] += '_different'
    elif fault == 'extra-id': first['outcomes'].append(dict(node_id=file+'::test_extra', outcome='passed'))
    elif fault == 'duplicate': first['outcomes'] *= 2
    elif fault == 'collection-duplicate': collection['node_ids'] *= 2
    elif fault == 'file-collection-mismatch': first['node_ids'][0] += '_different'
    elif fault == 'file-collection-duplicate': first['node_ids'] *= 2
    elif fault == 'collection-error': collection.update(exit_code=2, errors=['collection_error'])
    elif fault == 'collection-missing': monkeypatch.setattr(module, '_collect', lambda *a: None)
    elif fault == 'empty-collection': collection['node_ids'] = []
    elif fault == 'missing-report': reports[file] = None
    elif fault == 'malformed-report': reports[file] = '{broken json'
    elif fault == 'malformed-outcome': first['outcomes'][0]['outcome'] = 'looks_good'
    elif fault == 'empty-result': first.update(node_ids=[], outcomes=[])
    elif fault == 'exit-failure': first['exit_code'] = 7
    elif fault == 'start-failure': reports[file] = OSError('SECRET PRIVATE PATH')
    elif fault == 'timeout': reports[file] = subprocess.TimeoutExpired('PRIVATE', 1)
    elif fault == 'interrupt': reports[file] = KeyboardInterrupt()
    report = module.run_checks(root, lane='windows-full')
    assert report['status'] == 'blocked' and reason in report['reason_codes']
    assert 'PRIVATE' not in str(report) and 'SECRET' not in str(report)
    monkeypatch.setattr(module, 'run_checks', lambda *a, **k: deepcopy(report))
    assert module.main(['--root', str(root), '--lane', 'windows-full']) == 1
    executed = [f for phase, f in calls if phase == 'execute']
    assert len(executed) == len(set(executed)), 'no retry'


@pytest.mark.parametrize('outcome,counter', [
    ('failed', 'failed'), ('error', 'errors'), ('skipped', 'skipped'),
    ('xfailed', 'xfailed'), ('xpassed', 'xpassed'),
])
def test_nonpass_outcomes_remain_visible_and_block(monkeypatch, tmp_path, outcome, counter):
    module = load_future(RUNNER, monkeypatch)
    root, files = tiny_repo(tmp_path)
    _, reports, _ = transport(monkeypatch, module, files)
    reports[files[0]]['outcomes'][0]['outcome'] = outcome
    if outcome in ('failed', 'error'): reports[files[0]]['exit_code'] = 1
    report = module.run_checks(root, lane='windows-full')
    assert report['status'] == 'blocked'
    assert report['counts'][counter] == 1
    assert report['counts']['passed'] == 1
    assert report['coverage_complete'] is True  # coverage is not correctness


def test_full_lane_cannot_be_relabelled_partial_selection(monkeypatch, tmp_path):
    module = load_future(RUNNER, monkeypatch)
    root, files = tiny_repo(tmp_path)
    with pytest.raises((TypeError, ValueError)):
        module.run_checks(root, lane='windows-full', test_files=files[:1])
    with pytest.raises((TypeError, ValueError)):
        module.run_checks(root, lane='portable', test_files=[])


def test_portable_lane_declares_selected_and_excluded_scope(monkeypatch, tmp_path):
    module = load_future(RUNNER, monkeypatch)
    root, files = tiny_repo(tmp_path)
    transport(monkeypatch, module, files[:1])
    report = module.run_checks(root, lane='portable', test_files=files[:1])
    assert report['status'] == 'passed' and report['lane'] == 'portable'
    assert report['selected_files'] == files[:1] and report['excluded_files'] == files[1:]
    assert report['collection_total'] == report['executed_total'] == 1
    assert report.get('native_acceptance') is not True


def test_reports_are_stable_machine_and_human_readable(monkeypatch, tmp_path, capsys):
    module = load_future(RUNNER, monkeypatch)
    root, files = tiny_repo(tmp_path)
    transport(monkeypatch, module, files)
    first = module.run_checks(root, lane='windows-full')
    assert first == module.run_checks(root, lane='windows-full')
    assert module.main(['--root', str(root), '--lane', 'windows-full']) == 0
    captured = capsys.readouterr()
    assert strict_json(captured.out) == first
    assert 'passed' in captured.err.lower()


def test_actual_pytest_children_use_current_interpreter_one_file_each(monkeypatch, tmp_path):
    module = load_future(RUNNER, monkeypatch)
    root, files = tiny_repo(tmp_path)
    original = subprocess.run
    calls = []
    def run(command, *args, **kwargs):
        assert not kwargs.get('shell', False)
        assert type(command) in (list, tuple)
        assert Path(command[0]).resolve() == Path(sys.executable).resolve()
        assert '-B' in command
        assert 'no:cacheprovider' in command
        assert kwargs.get('env', os.environ).get('PYTHONDONTWRITEBYTECODE') == '1'
        calls.append(list(map(str, command)))
        return original(command, *args, **kwargs)
    monkeypatch.setattr(subprocess, 'run', run)
    report = module.run_checks(root, lane='windows-full')
    assert_success(report, files)
    executions = [c for c in calls if '--collect-only' not in c]
    assert len(executions) == 2
    selected = []
    for command in executions:
        inputs = [arg.replace('\\', '/') for arg in command if arg.endswith('.py') and 'test_' in arg]
        assert len(inputs) == 1
        selected.append(inputs[0].split('/tests/')[-1].removeprefix('tests/'))
    assert sorted(selected) == ['test_a.py', 'test_b.py']
    assert not list(root.rglob('__pycache__')) and not list(root.rglob('.pytest_cache'))


def test_actual_pytest_setup_teardown_errors_are_not_double_counted(monkeypatch, tmp_path):
    module = load_future(RUNNER, monkeypatch)
    root, files = tiny_repo(tmp_path)
    (root / files[0]).write_text(
        'import pytest\n@pytest.fixture\ndef broken():\n    yield\n    raise RuntimeError("teardown")\n'
        'def test_one(broken):\n    assert True\n', encoding='utf8')
    report = module.run_checks(root, lane='windows-full')
    assert report['status'] == 'blocked' and report['coverage_complete'] is True
    assert report['executed_total'] == 2
    assert report['counts'] == dict(passed=1, failed=0, errors=1, skipped=0, xfailed=0, xpassed=0)

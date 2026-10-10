"""Isolated source-release test checks; no test-count or success shortcuts.

Only trusted local test projects are accepted. This is a host CLI, not a model
tool. Child pytest hooks supply machine evidence independently of console text.
"""

import argparse
from collections import Counter
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
from tempfile import TemporaryDirectory


_OUTCOMES = {'passed': 'passed', 'failed': 'failed', 'error': 'errors',
             'skipped': 'skipped', 'xfailed': 'xfailed', 'xpassed': 'xpassed'}
_TIMEOUT = 300


class _EvidencePlugin:
    """One terminal outcome after teardown; error overrides a passing call."""

    def __init__(self):
        self.node_ids = []
        self.outcomes = []
        self.errors = []
        self.pending = {}

    def pytest_collection_finish(self, session):
        self.node_ids = [item.nodeid for item in session.items]

    def pytest_collectreport(self, report):
        if report.failed:
            self.errors.append('collection_error')
        elif report.skipped:
            self.errors.append('collection_skipped')

    def pytest_internalerror(self, excrepr, excinfo):
        self.errors.append('pytest_internal_error')

    def pytest_runtest_logreport(self, report):
        outcome = None
        if report.failed:
            outcome = 'failed' if report.when == 'call' else 'error'
        elif report.skipped:
            outcome = 'xfailed' if hasattr(report, 'wasxfail') else 'skipped'
        elif report.when == 'call':
            outcome = 'xpassed' if hasattr(report, 'wasxfail') else 'passed'
        priority = {'passed': 0, 'skipped': 1, 'xfailed': 2, 'xpassed': 3, 'failed': 4, 'error': 5}
        previous = self.pending.get(report.nodeid)
        if outcome is not None and (previous is None or priority[outcome] > priority[previous]):
            self.pending[report.nodeid] = outcome
        if report.when == 'teardown':
            terminal = self.pending.pop(report.nodeid, None)
            if terminal is None:
                self.errors.append('missing_terminal_outcome')
            else:
                self.outcomes.append({'node_id': report.nodeid, 'outcome': terminal})


def _child_main():
    import pytest
    plugin = _EvidencePlugin()
    code = int(pytest.main(sys.argv[1:], plugins=[plugin]))
    payload = dict(exit_code=code, node_ids=plugin.node_ids,
                   outcomes=plugin.outcomes, errors=plugin.errors)
    Path(os.environ['M23_CHILD_REPORT']).write_text(
        json.dumps(payload, ensure_ascii=True, allow_nan=False), encoding='utf-8')
    raise SystemExit(code)


def _unique(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError('duplicate report key')
        result[key] = value
    return result


def _diagnostics(files, collect, stdout, stderr):
    """Keep pytest diagnostics on stderr, separate from stable JSON evidence."""
    print(('collect' if collect else 'execute') + ': ' + ', '.join(files), file=sys.stderr)
    for stream in (stdout, stderr):
        if stream:
            text = stream.decode('utf-8', errors='replace') if isinstance(stream, bytes) else stream
            print(text, file=sys.stderr, end='' if text.endswith('\n') else '\n')


def _child(root, files, *, collect):
    # subprocess.run kills/waits for its immediate child on timeout/exception.
    # No retry and no claim of preempting arbitrary descendant processes/I/O.
    with TemporaryDirectory(prefix='copilot-release-child-') as temporary:
        report_path = Path(temporary) / 'report.json'
        env = dict(os.environ, PYTHONDONTWRITEBYTECODE='1', PYTHONUTF8='1',
                   PYTEST_DISABLE_PLUGIN_AUTOLOAD='1', M23_CHILD_REPORT=str(report_path))
        env.pop('PYTEST_ADDOPTS', None)
        env.pop('PYTEST_PLUGINS', None)
        command = [sys.executable, '-B', '-c',
                   "import runpy,sys; runpy.run_path(sys.argv.pop(1))['_child_main']()",
                   str(Path(__file__).resolve()), '-p', 'no:cacheprovider', '-q',
                   '--rootdir', str(root)]
        if collect:
            command.append('--collect-only')
        command.extend(files)
        try:
            process = subprocess.run(command, cwd=root, env=env, shell=False,
                                     capture_output=True, timeout=_TIMEOUT)
        except subprocess.TimeoutExpired as exc:
            _diagnostics(files, collect, exc.stdout, exc.stderr)
            raise
        _diagnostics(files, collect, process.stdout, process.stderr)
        if not report_path.is_file():
            return None
        try:
            payload = json.loads(report_path.read_text(encoding='utf-8'), object_pairs_hook=_unique)
        except (ValueError, UnicodeError):
            return 'malformed_report'
        if type(payload) is dict:
            # The OS process exit is authoritative even if pytest reported 0.
            payload['exit_code'] = process.returncode
        return payload


def _collect(root, files):
    return _child(root, files, collect=True)


def _run_file(root, file):
    return _child(root, [file], collect=False)


def _payload(value, reasons):
    if value is None:
        reasons.add('missing_report')
        return None
    if (type(value) is not dict or set(value) != {'exit_code', 'node_ids', 'outcomes', 'errors'}
            or type(value['exit_code']) is not int
            or type(value['node_ids']) is not list or type(value['outcomes']) is not list
            or type(value['errors']) is not list
            or any(type(x) is not str or not x for x in value['node_ids'] + value['errors'])):
        reasons.add('malformed_report')
        return None
    for item in value['outcomes']:
        if (type(item) is not dict or set(item) != {'node_id', 'outcome'}
                or type(item['node_id']) is not str or not item['node_id']
                or type(item['outcome']) is not str or item['outcome'] not in _OUTCOMES):
            reasons.add('malformed_report')
            return None
    return value


def _transport(function, args, reasons):
    try:
        return _payload(function(*args), reasons)
    except subprocess.TimeoutExpired:
        reasons.add('timeout')
    except KeyboardInterrupt:
        reasons.add('interrupted')
    except OSError:
        reasons.add('subprocess_start_failed')
    return None


def _selection(root, lane, test_files):
    files = sorted(path.relative_to(root).as_posix() for path in (root / 'tests').rglob('test_*.py')
                   if path.is_file())
    for file in files:
        if not (root / file).resolve().is_relative_to(root):
            raise ValueError('test file escapes root')
    if lane == 'windows-full':
        if test_files is not None:
            raise ValueError('full lane cannot override selection')
        return files, []
    if lane != 'portable' or type(test_files) not in (list, tuple) or not test_files:
        raise ValueError('portable lane requires explicit test files')
    if any(type(file) is not str or file not in files for file in test_files):
        raise ValueError('allowlist must contain repository-relative test files')
    if len(set(test_files)) != len(test_files):
        raise ValueError('duplicate allowlist file')
    return sorted(test_files), sorted(set(files) - set(test_files))


def run_checks(root, *, lane, test_files=None):
    """Collect independently, execute each file once, reconcile exact IDs."""
    root = Path(root).resolve(strict=True)
    files, excluded = _selection(root, lane, test_files)
    reasons = set()
    if sys.version_info[:2] != (3, 11):
        reasons.add('unsupported_python')
    if lane == 'windows-full' and sys.platform != 'win32':
        reasons.add('unsupported_platform')
    if not files:
        reasons.add('empty_collection')
    collected, file_union, executed, rows = [], [], [], []
    counts = dict.fromkeys(_OUTCOMES.values(), 0)
    if not reasons:
        global_report = _transport(_collect, (root, files), reasons)
        if global_report is not None:
            collected = global_report['node_ids']
            if global_report['exit_code'] or global_report['errors']:
                reasons.add('collection_failed')
            if global_report['outcomes']:
                reasons.add('malformed_report')
            if not collected:
                reasons.add('empty_collection')
        # Even an incomplete collection cannot be relabelled a successful subset.
        for file in files:
            if 'interrupted' in reasons:
                break
            payload = _transport(_run_file, (root, file), reasons)
            row = dict(file=file, exit_code=None, node_ids=[], outcomes=[])
            if payload is not None:
                row.update(exit_code=payload['exit_code'], node_ids=sorted(payload['node_ids']),
                           outcomes=sorted(payload['outcomes'], key=lambda item: item['node_id']))
                if payload['exit_code']:
                    reasons.add('subprocess_failed')
                if payload['errors']:
                    reasons.add('child_report_error')
                node_ids = payload['node_ids']
                terminal_ids = [item['node_id'] for item in payload['outcomes']]
                file_union.extend(node_ids)
                executed.extend(terminal_ids)
                if not node_ids or not terminal_ids or Counter(node_ids) != Counter(terminal_ids):
                    reasons.add('coverage_mismatch')
                if any(not node.startswith(file + '::') for node in node_ids + terminal_ids):
                    reasons.add('coverage_mismatch')
                for item in payload['outcomes']:
                    counts[_OUTCOMES[item['outcome']]] += 1
                    if item['outcome'] != 'passed':
                        reasons.add('nonpass_outcome')
            rows.append(row)
    for sequence in (collected, file_union, executed):
        if len(sequence) != len(set(sequence)):
            reasons.add('duplicate_node_id')
    coverage = (bool(collected) and len(rows) == len(files)
                and len(collected) == len(set(collected))
                and Counter(collected) == Counter(file_union) == Counter(executed)
                and 'coverage_mismatch' not in reasons)
    if not coverage:
        reasons.add('coverage_mismatch')
    return dict(
        schema_version='v1-release-checks-v1', python_version=platform.python_version(),
        platform=sys.platform, lane=lane, selected_files=files, excluded_files=excluded,
        collection_node_ids=sorted(collected), per_file_collection_node_ids=sorted(file_union),
        executed_node_ids=sorted(executed), collection_total=len(collected), executed_total=len(executed),
        coverage_complete=coverage, counts=counts, files=rows,
        status='blocked' if reasons else 'passed', reason_codes=sorted(reasons),
        command_environment=dict(interpreter=sys.executable, bytecode=False, pytest_cache=False,
                                 plugin_autoload=False, shell=False, retry=False,
                                 child_timeout_seconds=_TIMEOUT, isolation='one_file_per_process'),
    )


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', default=str(Path(__file__).resolve().parents[1]))
    parser.add_argument('--lane', choices=['windows-full', 'portable'], required=True)
    parser.add_argument('--test-file', action='append')
    parser.add_argument('--test-list', help='UTF-8 JSON array of explicit portable test-file paths')
    args = parser.parse_args(argv)
    try:
        selected = args.test_file
        if args.test_list:
            if args.lane != 'portable' or selected is not None:
                parser.error('--test-list requires portable lane and cannot accompany --test-file')
            selected = json.loads(Path(args.test_list).read_text(encoding='utf-8'))
        report = run_checks(args.root, lane=args.lane, test_files=selected)
    except (ValueError, TypeError, OSError):
        parser.error('invalid root or lane selection')
    print(json.dumps(report, ensure_ascii=True, allow_nan=False, sort_keys=True))
    print(f"{report['status']}: {report['executed_total']}/{report['collection_total']} executed; "
          f"counts={report['counts']}; reasons={report['reason_codes']}", file=sys.stderr)
    return 0 if report['status'] == 'passed' else 1


if __name__ == '__main__':
    raise SystemExit(main())

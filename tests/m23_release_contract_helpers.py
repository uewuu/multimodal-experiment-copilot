"""M23 Slice 1: V1 Integration & Release -- scope/acceptance freeze, tests only.

SOURCE DELIVERY: Python 3.11, repository-root commands and documented deps.
No mandatory wheel/editable/PyPI. v0.2.0 is a candidate, NOT a frozen tag.
Frozen parent 0bb5fa9a8e722feb7e7056fdfcf8809463464e62. M11-M22 production
and tests stay unchanged. New/modified production modules: ZERO. Reuse real
CopilotSession, BoundCopilotSession, BoundToolCollection, Adapter, Tools/Core.
Default registry remains analyze_experiment + compare_experiments.

IN: seven independent generic synthetic workflows; offline fake Provider;
real stack; source setup; tested dependency constraints; examples/docs;
isolated full regression; Windows/local NTFS acceptance; Linux portable subset;
clean environment; CI correction; cumulative Git review and authorized release.
OUT: new framework/capability/default tools, Core redesign, M20/M21 automatic
composition, auto pairing, causal/significance/winner claims, training/HPO,
scheduling/discovery, public multi-user service/UI, large artifacts, POSIX
safe-open, automatic repair, unapproved merge/tag/PR/release.

DEMO (future examples/v1_bound_workflows_demo.py):
  python examples/v1_bound_workflows_demo.py --scenario A  (A through G)
  run_scenario(name) -> fresh strict-JSON dict; main(argv=None) -> exit code.
  Exactly one selected bound tool invocation through ask_with_observability;
  default tools remain advertised, but are not automatically invoked. Provider
  fake only emits messages, final {facts: ...} grounded in REAL tool output.
  _make_client(scenario) is an internal provider-message injection seam only.
  A/B host-select tiny files in TemporaryDirectory; C/D/E/F memory snapshots.
  G host-selects artifact.bin containing b'abc', through REAL M21 authorization.
  A: test metric 2; B: test comparison 3-2=1; C: values [1,3], mean 2.
  D: unpaired A=[1,3], B=[2,7], mean difference 2.5.
  E: same values, EXPLICIT crossed pairs: pair-0=(A0,B1), pair-1=(A1,B0),
     deltas [6,-1], mean 2.5; never infer pairing from order/seed/artifact.
  F: one host-reported metrics descriptor with matching digest/size.
  Envelope: schema_version=v1-bound-demo-v1, scenario, status (ok|unsupported|
  error), tool_name, result (full display-normalized tool report or None),
  facts, limitations (nonempty), runtime {provider_request_count,
  tool_invocation_count}, error_code (None on ok). No total verified/released.
  Only display projection may normalize absolute experiment_dir/root to their
  basename, run_id, started_at/completed_at, mtime_ns_before/after. Production
  data and statistical facts MUST remain intact. No timestamps in envelope.
  CLI stdout one strict JSON document; stderr optional safe diagnostic.
  Exit 0=ok; 2=invalid CLI selection; 3=unsupported G; 1=execution failure.
  Exceptions become error_code=execution_failed without raw exception/path;
  failed Session history is empty; original bound context restored.
  Non-Windows G: status unsupported, facts.status=unsupported_platform, no
  fabricated digest. G on Windows requires actual NTFS native success; no skip.
  Fixture preparation may write temporary files; tool operations cannot write.
  A-F must not call M21, F must not hash/read, and G must not call M20/M16.

RUNNER (future scripts/run_release_checks.py):
  python scripts/run_release_checks.py --lane windows-full [--root ROOT]
  run_checks(root, *, lane, test_files=None) -> strict-JSON report.
  windows-full dynamically enumerates tests/**/test_*.py; caller list override
  forbidden. portable requires an explicit nonempty relative file allowlist,
  reports exclusions, and can NEVER claim full platform/native acceptance.
  CLI portable takes repeated --test-file; invalid args exit 2. Overall report
  status passed => exit 0, otherwise blocked => exit 1. stdout JSON, stderr a
  human summary. Python 3.11/current sys.executable; shell=False, no retry;
  each selected test file exactly one pytest execution subprocess. Collection
  is separate. -B, -p no:cacheprovider, PYTHONDONTWRITEBYTECODE=1 for children.
  No hard timeout claim. If timeout/interruption occurs, fail closed.

  Internal transport seams for fault injection, NOT fake acceptance:
  _collect(root, files) and _run_file(root, file) return child payload dicts:
    exit_code: exact int; node_ids: list of collected IDs;
    outcomes: [{node_id, outcome}], errors: list of safe codes.
  Collection outcomes=[]; execution has ONE terminal outcome per executed ID.
  Outcomes passed/failed/error/skipped/xfailed/xpassed. Setup/teardown failures
  collapse to one error outcome per node, rather than duplicate test counts.
  Production transport must obtain machine reports from REAL pytest hooks;
  no parsing pass-count prose as proof of execution. Missing/malformed report
  cannot be replaced by exit_code=0. Transport errors may raise OSError,
  TimeoutExpired or KeyboardInterrupt; report blocks safely without raw text.

  Global selected collection IDs == union of per-file collection IDs ==
  executed terminal IDs, with no duplicates/extras/omissions. Compare exact
  identities, NOT totals. Empty collection/file or result blocks. Unexpected
  outcome, nonzero exit, collection errors, absent report, duplicate counts,
  unapproved skip/xfail/xpass all block. v1 allows NO success skips/xfails/xpasses.
  Report fields: schema_version=v1-release-checks-v1; python_version, platform,
  lane, selected_files, excluded_files, collection_node_ids, executed_node_ids,
  collection_total, executed_total, coverage_complete, counts, files,
  status, reason_codes, command_environment. Each files entry has file,
  exit_code (None on start failure), node_ids, outcomes. Counts use keys below.
  Deterministic ordering and safe codes; no wall time in stable report. No
  hard-coded final count: 83 files/4576 IDs are PRE-M23 collection evidence,
  NOT passed evidence. Runner contracts use tiny temporary test repos to avoid
  recursively executing the repository's runner tests.

ACCEPTANCE OUTSIDE PYTEST: matrix below is persisted here once, not duplicated
in planning docs/manifest. Each tuple: gate, method, expected, block, evidence,
owner. S2 examples/docs/dependency materials; S3 fresh install/CI/full regression;
S4 independent RC review; S5 separately authorized Git integration/release.
No docs title-count tests. No installation, fetch, CI edit or release in S1.
Implementation Frozen != Remote Synced != RC Ready != Released.
OS access-time changes are not program writes; Session rollback does not undo
host callback side effects. Providers may receive results; history/HTTP exports
are not automatically safe to publish. HTTP has no multi-user auth boundary.
M15 stays simulation-only; M21 is bounded, read-only and best-effort change
detection; M22 is host-declared descriptive. No uniform Reader byte/time bound.
"""

from copy import deepcopy
import importlib.util
import json
import os
from pathlib import Path
import socket
import sys
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
DEMO = ROOT / 'examples/v1_bound_workflows_demo.py'
RUNNER = ROOT / 'scripts/run_release_checks.py'
MISSING_DEMO = 'M23 expected RED: offline bound demo missing'
MISSING_RUNNER = 'M23 expected RED: isolated release runner missing'
COUNTS = ('passed', 'failed', 'errors', 'skipped', 'xfailed', 'xpassed')
TOOLS = dict(zip('ABCDEFG', (
    'analyze_experiment_evidence', 'compare_experiment_evidence',
    'aggregate_replicate_evidence', 'compare_replicate_groups',
    'compare_paired_replicates', 'compare_artifact_descriptors', 'observe_artifact')))
ABC_SHA = 'ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad'
GOLDEN = {
    'A': {'test_value': 2.0}, 'B': {'raw_delta': 1.0},
    'C': {'count': 2, 'mean': 2.0},
    'D': {'comparison_policy': 'unpaired-descriptive-v1', 'raw_delta': 2.5},
    'E': {'comparison_policy': 'explicit-paired-descriptive-v1',
          'pairing_source': 'host_declared', 'pair_deltas': [6.0, -1.0], 'mean': 2.5},
    'F': {'basis': 'host_reported', 'sha256_comparison': 'match',
          'currentness_status': 'not_checked', 'checkpoint_binding_status': 'not_verified'},
    'G': {'status': 'observed', 'sha256': ABC_SHA, 'byte_size': 3,
          'consistency_status': 'no_change_detected'},
}
ACCEPTANCE = (
    ('functional', 'run A-G independently', 'golden facts and real tool results',
     'missing or wrong scenario', 'commands + JSON', 'S2/S3'),
    ('compatibility', 'diff parent + old regression', 'frozen production/tests unchanged',
     'changed frozen file or regression', 'diff + per-file reports', 'S3/S4'),
    ('integration', 'delegate spies calling originals', 'real Session/Adapter/Tool/Core',
     'fabricated result or bypass', 'M23 contracts', 'S2'),
    ('offline', 'deny network/remove keys', 'demo succeeds without credentials',
     'external request or private fixture', 'offline smoke', 'S2/S3'),
    ('fresh_environment', 'new Python 3.11 venv + documented install', 'imports and demos succeed',
     'developer-only dependency/path', 'OS/Python/deps/command exits', 'S3'),
    ('windows_native', 'probe actual local NTFS + native M21 tests', 'real native tests pass',
     'wrong filesystem or seam-only evidence', 'environment probe + native report', 'S3'),
    ('linux_portable', 'explicit subset on Linux', 'listed scenarios/imports/fail-closed pass',
     'failure or native/full-platform claim', 'allowlist + exclusions + report', 'S3'),
    ('coverage', 'compare collected/executed node-ID multisets', 'exact union, no duplicates',
     'missing/extra/empty/malformed evidence', 'machine node-ID records', 'S3'),
    ('isolation', 'one file per current-Python child', 'Adapter separate, no shared process',
     'multi-file execution or hidden failure', 'commands + exit codes', 'S3'),
    ('security', 'deny/rollback/claim gates + secret review', 'existing authority and trust preserved',
     'privilege/claim upgrade or exposed secrets', 'contracts + review record', 'S2/S4'),
    ('documentation', 'execute documented commands and review API/source', 'accurate interfaces/limits',
     'stale pass counts or unsupported claims', 'command log + review', 'S2/S4'),
    ('dependencies', 'install tested constraints in clean env', 'reproducible supported combination',
     'missing dependency or incompatible resolution', 'resolved versions + install log', 'S3'),
    ('git_scope', 'fresh fetch + ancestry/cumulative diff + PR/protection review', 'all milestones retained',
     'unknown remote, divergence or unreviewed scope', 'SHAs/diff/review', 'S4/S5'),
    ('version', 'review tags/API support/RC identity', 'explicit version and tag target decision',
     'collision or assumed long-term API stability', 'approved version/target', 'S4/S5'),
    ('release_evidence', 'independent exact-SHA audit', 'all required gates satisfied',
     'historical evidence presented as current', 'RC evidence manifest', 'S4'),
    ('release_action', 'explicit authorization then post-action verification', 'approved target only',
     'missing authorization or remote verification', 'merge/tag/release receipts', 'S5'),
)


def load_future(path, monkeypatch):
    assert path.is_file(), MISSING_DEMO if path == DEMO else MISSING_RUNNER
    name = '_m23_contract_' + path.stem
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, name, module)
    spec.loader.exec_module(module)
    return module


def strict_json(text):
    def unique(pairs):
        result = {}
        for key, value in pairs:
            assert key not in result, 'duplicate JSON key'
            result[key] = value
        return result
    def invalid(value):
        raise AssertionError('non-finite JSON')
    return json.loads(text, object_pairs_hook=unique, parse_constant=invalid)


def offline(monkeypatch):
    calls = []
    def deny(*args, **kwargs):
        calls.append('network')
        raise AssertionError('real network forbidden')
    for key in tuple(os.environ):
        if any(token in key.upper() for token in ('API_KEY', 'OPENAI', 'ANTHROPIC', 'AZURE')):
            monkeypatch.delenv(key, raising=False)
    monkeypatch.setattr(socket, 'create_connection', deny)
    monkeypatch.setattr(socket.socket, 'connect', deny)
    monkeypatch.setattr(socket.socket, 'connect_ex', deny)
    monkeypatch.setattr(urllib.request, 'urlopen', deny)
    return calls


def display(value, key=None):
    """Independent allowed display projection, never statistical computation."""
    if key in ('started_at', 'completed_at', 'run_id') and value is not None:
        return '<normalized>'
    if key in ('mtime_ns_before', 'mtime_ns_after') and value is not None:
        return 0
    if key in ('experiment_dir', 'experiment_root') and type(value) is str:
        return Path(value).name
    if type(value) is dict:
        return {k: display(v, k) for k, v in value.items()}
    if type(value) is list:
        return [display(v) for v in value]
    return value


def assert_demo(report, scenario):
    assert set(report) == {'schema_version', 'scenario', 'status', 'tool_name',
                           'result', 'facts', 'limitations', 'runtime', 'error_code'}
    assert report['schema_version'] == 'v1-bound-demo-v1'
    assert report['scenario'] == scenario and report['tool_name'] == TOOLS[scenario]
    assert report['status'] == 'ok' and report['error_code'] is None
    assert report['facts'] == GOLDEN[scenario]
    assert type(report['result']) is dict and report['limitations']
    assert report['runtime'] == {'provider_request_count': 2, 'tool_invocation_count': 1}
    strict_json(json.dumps(report, allow_nan=False))


def tiny_repo(tmp_path):
    root = tmp_path / 'mini'
    tests = root / 'tests'
    tests.mkdir(parents=True)
    for name in ('a', 'b'):
        (tests / f'test_{name}.py').write_text('def test_one():\n    assert True\n', encoding='utf8')
    return root, ['tests/test_a.py', 'tests/test_b.py']


def transport(monkeypatch, module, files):
    """Calibrated child-report doubles only; no release runner implementation."""
    ids = [f + '::test_one' for f in files]
    collection = dict(exit_code=0, node_ids=list(ids), outcomes=[], errors=[])
    reports = {f: dict(exit_code=0, node_ids=[node],
                      outcomes=[dict(node_id=node, outcome='passed')], errors=[])
               for f, node in zip(files, ids)}
    calls = []
    def collect(root, selected):
        calls.append(('collect', tuple(selected)))
        return deepcopy(collection)
    def run(root, file):
        calls.append(('execute', file))
        item = reports[file]
        if isinstance(item, BaseException):
            raise item
        return deepcopy(item)
    monkeypatch.setattr(module, '_collect', collect)
    monkeypatch.setattr(module, '_run_file', run)
    return collection, reports, calls

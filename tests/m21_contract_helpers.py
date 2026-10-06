"""M21 Scope Freeze: Bounded Artifact Observation, Contract v1.

Only tests/helpers are added at RED. Future modules, in slice order:
2 artifact_observation_io: open_authorized_artifact(authorized_root, file_name)
  returns a context manager yielding metadata()/read(n); ObservationIOError has
  a fixed .status. Both path arguments are exact str; invalid inputs raise
  TypeError/ValueError BEFORE native access. Standalone calls are host-authorized
  by the explicit root argument; the backend is not a Session authorization API.
3 artifact_observation: validate_artifact_binding(binding),
  project_artifact_observation(binding, capture), both pure detached exact dicts;
  artifact_observation_reader.observe_artifact(binding), live on EVERY call.
4 tool_layer.artifact_observation_tools.ArtifactObservationTools(binding),
  list_tools()/invoke_tool(name, arguments). Existing protocols, explicit binding,
  minimal registry private capability observe_artifact -> experiment_dir/directory.
  Default registry, Path Policy, Adapter, Session, Runtime and M11-M20 frozen.

BINDING: exact dict with ALL and ONLY keys in binding() below. Version fixed;
max_bytes=1048576 and chunk_bytes=65536 exact int (not bool); not tunable in v1.
IDs/role/ref exact nonblank str <=128 UTF8 bytes, opaque host labels (never paths
to acquire). Root exact str <=1024 UTF8 bytes, absolute local Windows drive path;
no UNC/device/drive-relative/root-relative/NUL/dot components/ADS. A drive root
and a trailing separator are allowed; lexical case and slash equivalence allowed.
No I/O in normalization. Filename exact str, one component, 1..255 UTF8 bytes;
reject separators, . / .., colon, NUL/control chars, <>"|?*, trailing space/dot,
CON/PRN/AUX/NUL/COM1..9/LPT1..9 (also superscript 1/2/3), with any extension,
case-insensitive. Never coerce pathlike or string subclasses. Raw binding compact
UTF8 JSON <=4096; report <=8192. No truncation. Unknown fields reject, including
clock/digest/provenance claims. All callback-bearing input types reject before
reflection, iteration, copy, hash/equality or error formatting callbacks.

IO SEAM (private, not host/model API): _platform_name() -> 'win32'; _native
has open(path, *, access, share, disposition, flags), describe(handle),
read(handle, n), close(handle). describe returns fresh exact dicts from native
observations: kind ('directory'/'file'/other), reparse bool, final_path str,
filesystem str, local bool, volume_id int, file_id tuple[int,int], size int,
mtime_ns int, change_token int. Values are system metadata, NOT host claims.
These are raw test-double primitives, never authorization or hashing helpers.
Native adapter may expose additional internals, but this seam is stable.
Root opened READ_ATTRIBUTES, share READ only, OPEN_EXISTING, BACKUP_SEMANTICS |
OPEN_REPARSE_POINT; child GENERIC_READ, share READ only, OPEN_EXISTING,
OPEN_REPARSE_POINT. Additional safe flags allowed; no extra access/share bits.
Before child content: root and child non-reparse; root directory; child regular;
local NTFS; handle final paths match normalized selected root/direct child;
same volume; non-null identities. Root identity rechecked after child open.
Different final paths/volume/root identity -> unauthorized. Reparse/special ->
unsupported_type. Non-Windows/nonlocal/non-NTFS/backend unavailable ->
unsupported_platform. Native missing -> missing; access/share denied -> unreadable;
other OSError -> io_error. Unexpected programming exceptions propagate unchanged.
Close every acquired handle once on every exit. Never retry, broaden sharing,
or use generic open fallback. metadata() is obtained from the SAME file handle
as read(), with keys size/mtime_ns/file_id/change_token. No public native file ID.

READER: validate before safe open; private _utc_now() returns aware UTC datetime;
two timestamps formatted YYYY-MM-DDTHH:MM:SS.ffffffZ (start before open, end
after close). Bounded sequential reads <=65536/request and <=1048577 total;
SHA256 incremental from actual same-handle bytes. Pre size >limit -> no read.
Overflow -> oversized; discard hash. Identity/size/mtime/change-token mismatch,
or actual count inconsistent with metadata -> changed_during_read. No retries.
Safe-open typed error or classified OSError -> status; arbitrary bug propagates.
No file reopening for digest, no stat(path) source, no reported-size allocation.

CAPTURE: exact dict in capture() below, private Reader-to-projection data. UTC
timestamps canonical and ordered. All size/mtime fields exact nonnegative int
<=2**63-1 (successful size <=limit); sha256 canonical lowercase 64 hex chars.
Successful capture must have byte_size==size_before==size_after, equal mtimes,
consistency=no_change_detected. Failure capture has ALL six numeric/hash fields
None; consistency changed iff changed_during_read, otherwise not_evaluated.
Projection validates, never hashes/opens/resolves/reads clock; it does not confer
trust on arbitrary host captures. Only Reader-owned capture is system evidence.

REPORT: EXACT keys in REPORT_KEYS; schema_version=system-artifact-observation-v1;
host labels unchanged; observation_basis=SUCCESS_BASIS only for observed, else
['host_declared']; all other capture fields copied after validation. Limitations
always exactly LIMITATIONS (sorted). No contents, preview, base64, root, filename,
device path, raw errors, file ID or overall verified/trusted/current flag.
Host labels are intentional host-visible data, not system-discovered paths.
No-change means best-effort detection, NOT atomic snapshot/TOCTOU elimination.
No currentness/latest, checkpoint binding, producer authenticity, experiment
membership, semantics, recomputation or M16 parser same-bytes claim.

TOOL: model schema PARAMETERS. '.' is resolved by existing formal Path Policy
to the Session root before dispatch. Tool accepts only the secured absolute root
matching its host binding, using lexical Windows normalization, and rejects '.'
in direct invoke (no independent Session grant). Existing Adapter loses the
original spelling: an equivalent absolute root cannot be distinguished at Tool
entrance, but cannot expand access. No changes to Adapter to enforce enum.
Unknown args, missing root, wrong root and unsafe exact types reject before Reader.
Construct/list do no I/O. Binding/result/schema isolated. Each invoke calls real
Reader, no cached observation. Unknown exact-string tool name -> KeyError.

OUT: nested paths, discovery, arrays, parallelism, retry, writes/repair, parsing,
M20 composition, Memory, network, training/inference/checkpoint deserialization,
large checkpoints, hard read deadlines. Read-only does not promise no OS atime.
Windows local NTFS platform tests are required; no skip-based acceptance. Mocks
cannot establish native safety. POSIX fail-closed, no POSIX safety claim.

Helpers are fixtures, literal oracles and calibrated probes, not production
projection/validation/authorization implementations. Structured evals test the
existing explicit fact gate, not arbitrary natural-language claim detection.
"""

from contextlib import contextmanager
from copy import deepcopy
from datetime import datetime, timezone
import importlib
import json
from types import SimpleNamespace as NS

TOOL = 'observe_artifact'
LIMIT = 1048576
CHUNK = 65536
ROOT = r'C:\experiments\run-a'
PARAMETERS = {'type': 'object', 'properties': {
    'experiment_dir': {'type': 'string', 'enum': ['.']}},
    'required': ['experiment_dir'], 'additionalProperties': False}
STATUSES = ['observed', 'missing', 'unauthorized', 'unreadable', 'unsupported_type',
            'unsupported_platform', 'oversized', 'changed_during_read', 'io_error']
LIMITATIONS = sorted(['atomic_snapshot_not_guaranteed', 'artifact_currentness_not_checked',
    'checkpoint_binding_not_verified', 'producer_authenticity_not_verified',
    'experiment_membership_not_verified', 'semantic_correctness_not_verified',
    'metric_recomputation_not_performed', 'm16_parser_same_bytes_not_verified',
    'hard_read_timeout_not_guaranteed'])
SUCCESS_BASIS = ['host_declared', 'system_observed_metadata',
                 'system_observed_content', 'hash_computed_by_copilot']
EMPTY_SHA = 'e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855'
ABC_SHA = 'ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad'
DATA_FIELDS = ['byte_size', 'sha256', 'size_before', 'size_after',
               'mtime_ns_before', 'mtime_ns_after']
REPORT_KEYS = {'schema_version', 'artifact_id', 'logical_role', 'artifact_ref',
               'status', 'observation_basis', 'started_at', 'completed_at',
               'consistency_status', 'limitations', *DATA_FIELDS}


def api(layer):
    module, category = {
        'io': ('artifact_observation_io', 'Missing Platform Safety / Authorized Open'),
        'core': ('artifact_observation', 'Missing Reader / Projection'),
        'reader': ('artifact_observation_reader', 'Missing Reader / Projection'),
        'tool': ('tool_layer.artifact_observation_tools', 'Missing Tool / Capability / Integration'),
    }[layer]
    try:
        return importlib.import_module(module)
    except ModuleNotFoundError as error:
        if error.name != module:
            raise
        raise AssertionError(category) from error


def binding(root=ROOT, **updates):
    result = dict(schema_version='artifact-observation-binding-v1', artifact_id='metrics-1',
                  logical_role='metrics', artifact_ref='host-artifact-1', file_name='artifact.bin',
                  authorized_root=root, max_bytes=LIMIT, chunk_bytes=CHUNK)
    result.update(updates)
    return result


def capture():
    return dict(status='observed', started_at='2026-10-06T01:00:00.000000Z',
                completed_at='2026-10-06T01:00:01.000000Z', byte_size=3, sha256=ABC_SHA,
                size_before=3, size_after=3, mtime_ns_before=42, mtime_ns_after=42,
                consistency_status='no_change_detected')


def assert_report(report, status='observed'):
    assert type(report) is dict and set(report) == REPORT_KEYS
    assert report['schema_version'] == 'system-artifact-observation-v1'
    assert report['status'] == status
    assert report['limitations'] == LIMITATIONS
    assert report['observation_basis'] == (SUCCESS_BASIS if status == 'observed' else ['host_declared'])
    assert report['consistency_status'] == ('no_change_detected' if status == 'observed' else
                                          'changed' if status == 'changed_during_read' else 'not_evaluated')
    if status != 'observed':
        assert all(report[k] is None for k in DATA_FIELDS)
    assert len(json.dumps(report, ensure_ascii=False, allow_nan=False,
                          separators=(',', ':')).encode('utf8')) <= 8192


def patch_alias(monkeypatch, owner, name, consumer, replacement):
    original = getattr(owner, name)
    monkeypatch.setattr(owner, name, replacement)
    for alias, value in list(vars(consumer).items()):
        if value is original:
            monkeypatch.setattr(consumer, alias, replacement)


class Stream:
    """Already-authorized Reader seam: byte source, not a safe-open substitute."""
    def __init__(self, data=b'abc', before=None, after=None, error=None):
        self.data, self.position, self.requests = data, 0, []
        self.before = dict(size=len(data), mtime_ns=42, file_id=(1, 2), change_token=7)
        self.before.update(before or {})
        self.after = dict(self.before, **(after or {}))
        self.error, self.events, self.opens, self.closes = error, [], 0, 0

    def metadata(self):
        value = self.before if 'metadata' not in self.events else self.after
        self.events.append('metadata')
        return dict(value)

    def read(self, n):
        assert type(n) is int and 0 < n <= CHUNK
        self.events.append('read')
        self.requests.append(n)
        if self.error is not None:
            raise self.error
        result = self.data[self.position:self.position + n]
        self.position += len(result)
        return result

    @contextmanager
    def open(self, root, name):
        assert (root, name) == (ROOT, 'artifact.bin')
        self.opens += 1
        try:
            yield self
        finally:
            self.closes += 1


def install_stream(monkeypatch, stream):
    reader, backend = api('reader'), api('io')
    patch_alias(monkeypatch, backend, 'open_authorized_artifact', reader, stream.open)
    times = iter([datetime(2026, 10, 6, 1, 0, tzinfo=timezone.utc),
                  datetime(2026, 10, 6, 1, 0, 1, tzinfo=timezone.utc)])
    monkeypatch.setattr(reader, '_utc_now', lambda: next(times))
    return reader.observe_artifact


class Native:
    """Raw native syscall double; does NOT decide authorization or file safety."""
    def __init__(self, *, root_update=None, file_update=None, root_after=None, error=None):
        self.root, self.file = object(), object()
        self.root_info = dict(kind='directory', reparse=False, final_path=ROOT,
                              filesystem='NTFS', local=True, volume_id=1,
                              file_id=(1, 1), size=0, mtime_ns=42, change_token=7)
        self.file_info = dict(self.root_info, kind='file', final_path=ROOT + r'\artifact.bin',
                              file_id=(1, 2), size=3)
        self.root_info.update(root_update or {})
        self.file_info.update(file_update or {})
        self.root_after, self.error = root_after, error
        self.events, self.root_queries, self.position = [], 0, 0

    def open(self, path, **options):
        self.events.append(('open', path, options))
        if self.error is not None:
            raise self.error
        return self.root if len([e for e in self.events if e[0] == 'open']) == 1 else self.file

    def describe(self, handle):
        assert handle is self.root or handle is self.file
        self.events.append(('describe', handle))
        if handle is self.root:
            self.root_queries += 1
            return dict(self.root_info, **(self.root_after or {})) if self.root_queries > 1 else dict(self.root_info)
        return dict(self.file_info)

    def read(self, handle, n):
        assert handle is self.file
        self.events.append(('read', handle, n))
        data = b'abc'[self.position:self.position + n]
        self.position += len(data)
        return data

    def close(self, handle):
        self.events.append(('close', handle))


def install_native(monkeypatch, native):
    module = api('io')
    monkeypatch.setattr(module, '_platform_name', lambda: 'win32')
    monkeypatch.setattr(module, '_native', native)
    return module


class Client:
    def __init__(self, args=None, *, final='offline answer', error=None):
        self.args = {'experiment_dir': '.'} if args is None else args
        self.final, self.error, self.requests = final, error, []
        self.chat = NS(completions=self)

    def create(self, **kwargs):
        self.requests.append(deepcopy(kwargs))
        if len(self.requests) == 1:
            message = NS(role='assistant', content=None, tool_calls=[NS(id='observation-call',
                type='function', function=NS(name=TOOL, arguments=json.dumps(self.args)))])
        elif len(self.requests) == 2:
            if self.error is not None:
                raise self.error
            message = NS(role='assistant', content=self.final, tool_calls=None)
        else:
            raise AssertionError('unexpected provider retry')
        return NS(choices=[NS(message=message)], usage=NS(prompt_tokens=10, completion_tokens=3, total_tokens=13))


@contextmanager
def deny_side_effects(monkeypatch):
    """Named Python/project entrances; native access flags are tested separately."""
    import builtins
    import io
    import os
    from pathlib import Path
    import pickle
    import shutil
    import socket
    import sys
    import concurrent.futures
    import read_config
    import read_history
    import read_test_metrics
    import experiment_artifact_verification
    from experiment_memory import ExperimentMemory
    from experiment_repository import ExperimentRepository
    calls = []
    def deny(*args, **kwargs):
        calls.append(True)
        raise AssertionError('forbidden side effect sentinel')
    entrances = [(builtins, 'open'), (io, 'open'), (os, 'open'), (os, 'write'),
        (os, 'truncate'), (os, 'rename'), (os, 'replace'), (os, 'remove'), (os, 'unlink'),
        (os, 'chmod'), (os, 'utime'), (os, 'listdir'), (os, 'scandir'), (os, 'walk'),
        (Path, 'open'), (Path, 'read_bytes'), (Path, 'write_bytes'), (Path, 'write_text'),
        (Path, 'touch'), (Path, 'glob'), (Path, 'rglob'), (shutil, 'copy'), (shutil, 'move'),
        (pickle, 'load'), (pickle, 'loads'), (socket, 'create_connection'),
        (concurrent.futures, 'ThreadPoolExecutor'), (concurrent.futures, 'ProcessPoolExecutor'),
        (read_config, 'read_config_evidence'), (read_history, 'read_history'),
        (read_test_metrics, 'read_test_metrics'),
        (experiment_artifact_verification, 'compare_artifact_descriptors'),
        (ExperimentMemory, 'get'), (ExperimentMemory, 'list_executions'),
        (ExperimentRepository, '__init__')]
    with monkeypatch.context() as patch:
        for owner, name in entrances:
            original = getattr(owner, name)
            patch.setattr(owner, name, deny)
            for module_name in ('artifact_observation', 'artifact_observation_io',
                                'artifact_observation_reader', 'tool_layer.artifact_observation_tools'):
                module = sys.modules.get(module_name)
                if module is not None:
                    for alias, value in list(vars(module).items()):
                        if value is original:
                            patch.setattr(module, alias, deny)
        # No real torch import/checkpoint load is needed or permitted by fixtures.
        patch.setitem(sys.modules, 'torch', NS(load=deny, jit=NS(load=deny)))
        yield calls

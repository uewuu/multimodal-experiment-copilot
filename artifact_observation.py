"""Pure M21 binding validation and deterministic observation projection.

Projection validates structure, not the provenance of a supplied capture. Only
the Reader's own acquisition provides system observation evidence.
"""

from datetime import datetime
import json
import ntpath
import re

__all__ = ('validate_artifact_binding', 'project_artifact_observation')

MAX_BYTES = 1048576
CHUNK_BYTES = 65536
_BINDING_KEYS = ('schema_version', 'artifact_id', 'logical_role', 'artifact_ref',
                 'file_name', 'authorized_root', 'max_bytes', 'chunk_bytes')
_DATA_FIELDS = ('byte_size', 'sha256', 'size_before', 'size_after',
                'mtime_ns_before', 'mtime_ns_after')
_CAPTURE_KEYS = ('status', 'started_at', 'completed_at', *_DATA_FIELDS,
                 'consistency_status')
_STATUSES = ('observed', 'missing', 'unauthorized', 'unreadable', 'unsupported_type',
             'unsupported_platform', 'oversized', 'changed_during_read', 'io_error')
_LIMITATIONS = tuple(sorted((
    'atomic_snapshot_not_guaranteed', 'artifact_currentness_not_checked',
    'checkpoint_binding_not_verified', 'producer_authenticity_not_verified',
    'experiment_membership_not_verified', 'semantic_correctness_not_verified',
    'metric_recomputation_not_performed', 'm16_parser_same_bytes_not_verified',
    'hard_read_timeout_not_guaranteed')))


def _plain_fields(value, keys):
    if type(value) is not dict:
        raise TypeError('expected an exact dictionary')
    # Validate keys before any lookup/set construction can invoke foreign hash
    # or equality. Values are validated individually before copying/encoding.
    if len(value) != len(keys):
        raise ValueError('invalid fields')
    for key in value:
        if type(key) is not str:
            raise TypeError('field names must be exact strings')
        if key not in keys:
            raise ValueError('invalid fields')


def _text(value, limit):
    if type(value) is not str:
        raise TypeError('expected an exact string')
    if not value or len(value) > limit:
        raise ValueError('invalid string length')
    try:
        length = len(value.encode('utf-8'))
    except UnicodeError:
        raise ValueError('invalid string encoding') from None
    if length > limit or not value.strip():
        raise ValueError('invalid string length')


def _component(value):
    if (not value or value in ('.', '..') or value.endswith((' ', '.'))
            or any(ord(c) < 32 or c in '<>:"/\\|?*' for c in value)):
        raise ValueError('invalid path component')
    stem = value.split('.', 1)[0].upper()
    if (stem in ('CON', 'PRN', 'AUX', 'NUL')
            or (len(stem) == 4 and stem[:3] in ('COM', 'LPT')
                and stem[3] in '123456789\u00b9\u00b2\u00b3')):
        raise ValueError('reserved path component')


def _bounded_json(value, limit):
    if len(json.dumps(value, ensure_ascii=False, allow_nan=False,
                      separators=(',', ':')).encode('utf-8')) > limit:
        raise ValueError('structured data exceeds byte limit')


def validate_artifact_binding(binding):
    """Return detached exact scalar fields; lexical validation performs no I/O.

This is schema validation, not opened-object authorization. The frozen backend
independently authorizes the actual root and child handles before reading.
"""
    _plain_fields(binding, _BINDING_KEYS)
    _text(binding['schema_version'], 64)
    if binding['schema_version'] != 'artifact-observation-binding-v1':
        raise ValueError('invalid binding version')
    for key in ('artifact_id', 'logical_role', 'artifact_ref'):
        _text(binding[key], 128)
    for key, expected in (('max_bytes', MAX_BYTES), ('chunk_bytes', CHUNK_BYTES)):
        if type(binding[key]) is not int:
            raise TypeError('budget must be an exact integer')
        if binding[key] != expected:
            raise ValueError('budget is fixed in v1')
    _text(binding['file_name'], 255)
    _component(binding['file_name'])
    _text(binding['authorized_root'], 1024)
    root = binding['authorized_root'].replace('/', '\\')
    if (len(root) < 3 or root[0] not in
            'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ' or root[1:3] != ':\\'):
        raise ValueError('root must be a local absolute drive path')
    for part in root[3:].rstrip('\\').split('\\'):
        if part or len(root.rstrip('\\')) > 2:
            _component(part)
    result = {key: binding[key] for key in _BINDING_KEYS}
    _bounded_json(result, 4096)
    result['authorized_root'] = ntpath.normpath(root)
    return result


def _timestamp(value):
    _text(value, 27)
    if re.fullmatch(r'[0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9]{2}:[0-9]{2}:[0-9]{2}\.[0-9]{6}Z', value) is None:
        raise ValueError('timestamp must be canonical UTC')
    # Parse only an already bounded exact string; never consult the clock.
    datetime.fromisoformat(value[:-1] + '+00:00')


def project_artifact_observation(binding, capture):
    """Project a validated Reader capture without acquiring bytes or trust."""
    selected = validate_artifact_binding(binding)
    _plain_fields(capture, _CAPTURE_KEYS)
    _text(capture['status'], 32)
    status = capture['status']
    if status not in _STATUSES:
        raise ValueError('invalid observation status')
    for key in ('started_at', 'completed_at'):
        _timestamp(capture[key])
    if capture['completed_at'] < capture['started_at']:
        raise ValueError('observation timestamps are reversed')
    _text(capture['consistency_status'], 32)
    consistency = ('no_change_detected' if status == 'observed' else
                   'changed' if status == 'changed_during_read' else 'not_evaluated')
    if capture['consistency_status'] != consistency:
        raise ValueError('invalid consistency status')
    if status == 'observed':
        for key in _DATA_FIELDS:
            value = capture[key]
            if key == 'sha256':
                _text(value, 64)
                if re.fullmatch('[0-9a-f]{64}', value) is None:
                    raise ValueError('invalid SHA-256 encoding')
            else:
                if type(value) is not int:
                    raise TypeError('metadata must be an exact integer')
                if not 0 <= value <= (2**63 - 1):
                    raise ValueError('metadata out of range')
        if (not capture['byte_size'] == capture['size_before'] == capture['size_after']
                or capture['byte_size'] > MAX_BYTES
                or capture['mtime_ns_before'] != capture['mtime_ns_after']):
            raise ValueError('inconsistent successful capture')
    elif any(capture[key] is not None for key in _DATA_FIELDS):
        raise ValueError('failed capture contains success fields')
    report = dict(schema_version='system-artifact-observation-v1',
                  artifact_id=selected['artifact_id'], logical_role=selected['logical_role'],
                  artifact_ref=selected['artifact_ref'])
    report.update((key, capture[key]) for key in _CAPTURE_KEYS)
    report['observation_basis'] = (['host_declared', 'system_observed_metadata',
        'system_observed_content', 'hash_computed_by_copilot'] if status == 'observed'
        else ['host_declared'])
    report['limitations'] = list(_LIMITATIONS)
    _bounded_json(report, 8192)
    return report

"""One bounded byte observation through the frozen authorized-handle backend.

No parsing, path reopening, caching, retry, or provenance upgrade. Consistency
is best-effort detection, not an atomic snapshot or a currentness guarantee.
"""

from datetime import datetime, timezone
import hashlib

import artifact_observation as projection
import artifact_observation_io as backend

__all__ = ('observe_artifact',)


def _utc_now():
    """Internal clock seam, never a host/model argument."""
    return datetime.now(timezone.utc)


def _timestamp():
    return _utc_now().isoformat(timespec='microseconds').replace('+00:00', 'Z')


def _read_capture(handle):
    before = handle.metadata()
    if before['size'] > projection.MAX_BYTES:
        return {'status': 'oversized'}
    digest = hashlib.sha256()
    count = 0
    while True:
        request = min(projection.CHUNK_BYTES, projection.MAX_BYTES + 1 - count)
        chunk = handle.read(request)
        if not chunk:
            break
        count += len(chunk)
        if count > projection.MAX_BYTES:
            return {'status': 'oversized'}
        digest.update(chunk)
        # Do not retain the previous chunk during the next native read.
        del chunk
    after = handle.metadata()
    if (any(before[key] != after[key] for key in
            ('size', 'mtime_ns', 'file_id', 'change_token')) or count != before['size']):
        return {'status': 'changed_during_read'}
    return dict(status='observed', byte_size=count, sha256=digest.hexdigest(),
                size_before=before['size'], size_after=after['size'],
                mtime_ns_before=before['mtime_ns'], mtime_ns_after=after['mtime_ns'])


def observe_artifact(binding):
    """Observe one host-selected file afresh; invalid binding is a call error."""
    selected = projection.validate_artifact_binding(binding)
    started = _timestamp()
    try:
        with backend.open_authorized_artifact(selected['authorized_root'],
                                               selected['file_name']) as handle:
            capture = _read_capture(handle)
    except backend.ObservationIOError as error:
        capture = {'status': error.status}
    except OSError as error:
        capture = {'status': backend._os_status(error)}
    # The context has closed before the completion clock or projection runs.
    capture['started_at'] = started
    capture['completed_at'] = _timestamp()
    status = capture['status']
    capture['consistency_status'] = ('no_change_detected' if status == 'observed' else
        'changed' if status == 'changed_during_read' else 'not_evaluated')
    if status != 'observed':
        capture.update((key, None) for key in ('byte_size', 'sha256', 'size_before',
                       'size_after', 'mtime_ns_before', 'mtime_ns_after'))
    return projection.project_artifact_observation(selected, capture)

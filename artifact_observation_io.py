"""Authorized Windows/local-NTFS handle foundation for M21.

The caller grants one root and selects one direct child. Opening only queries
metadata; content is read solely on an explicit validated-handle read(n).
This is not a Reader, an atomic snapshot, or a provenance verification API.
"""

from contextlib import ExitStack, contextmanager
import ctypes
from ctypes import wintypes
import errno
import ntpath
import sys

__all__ = ('ObservationIOError', 'open_authorized_artifact')

_READ_ATTRIBUTES = 0x80
_GENERIC_READ = 0x80000000
_SHARE_READ = 1
_OPEN_EXISTING = 3
_BACKUP_SEMANTICS = 0x02000000
_OPEN_REPARSE_POINT = 0x00200000
_MAX_READ = 65536
_STATUSES = ('missing', 'unauthorized', 'unreadable', 'unsupported_type',
             'unsupported_platform', 'io_error')


class ObservationIOError(Exception):
    """A fixed safe code, without native messages or host paths."""

    def __init__(self, status):
        if type(status) is not str:
            raise TypeError('status must be an exact string')
        if status not in _STATUSES:
            raise ValueError('invalid observation I/O status')
        self.status = status
        super().__init__(status)


def _os_status(error):
    # Read the built-in slot without formatting or custom attribute lookup.
    code = OSError.errno.__get__(error)
    if type(code) is int:
        if code == errno.ENOENT:
            return 'missing'
        if code in (errno.EACCES, errno.EPERM):
            return 'unreadable'
    return 'io_error'


def _call(function, *args, **kwargs):
    try:
        return function(*args, **kwargs)
    except OSError as error:
        raise ObservationIOError(_os_status(error)) from None


def _exact_path(value, limit):
    if type(value) is not str:
        raise TypeError('path must be an exact string')
    try:
        size = len(value.encode('utf-8'))
    except UnicodeError:
        raise ValueError('invalid path encoding') from None
    if not size or size > limit:
        raise ValueError('invalid path length')


def _component(value):
    if (not value or value in ('.', '..') or value.endswith((' ', '.'))
            or any(ord(c) < 32 or c in '<>:"/\\|?*' for c in value)):
        raise ValueError('invalid path component')
    stem = value.split('.', 1)[0].upper()
    if (stem in ('CON', 'PRN', 'AUX', 'NUL')
            or (len(stem) == 4 and stem[:3] in ('COM', 'LPT')
                and stem[3] in '123456789\u00b9\u00b2\u00b3')):
        raise ValueError('reserved path component')


def _root(value):
    _exact_path(value, 1024)
    path = value.replace('/', '\\')
    if (len(path) < 3 or path[0] not in 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ'
            or path[1:3] != ':\\'):
        raise ValueError('root must be a local absolute drive path')
    for part in path[3:].rstrip('\\').split('\\'):
        if part or len(path.rstrip('\\')) > 2:
            _component(part)
    return ntpath.normpath(path)


def _final_path(value):
    # Only native DOS drive paths are eligible; do not accept other namespaces.
    if type(value) is not str:
        raise ObservationIOError('unauthorized')
    path = value
    if path.startswith('\\\\?\\'):
        path = path[4:]
    try:
        # A child path can be longer than the independently bounded root.
        _exact_path(path, 1280)
        normalized = path.replace('/', '\\')
        if (len(normalized) < 3 or normalized[0] not in
                'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ'
                or normalized[1:3] != ':\\'):
            raise ValueError('invalid native location')
        for part in normalized[3:].rstrip('\\').split('\\'):
            if part or len(normalized.rstrip('\\')) > 2:
                _component(part)
    except (TypeError, ValueError):
        raise ObservationIOError('unauthorized') from None
    return ntpath.normcase(ntpath.normpath(normalized))


def _verify(info, kind, expected_path):
    if info['reparse'] or info['kind'] != kind:
        raise ObservationIOError('unsupported_type')
    if not info['local'] or info['filesystem'] != 'NTFS':
        raise ObservationIOError('unsupported_platform')
    if (info['file_id'] is None or info['volume_id'] is None
            or _final_path(info['final_path']) != ntpath.normcase(expected_path)):
        raise ObservationIOError('unauthorized')


class _OwnedHandle:
    """One close attempt per acquired handle, also during exception unwinding."""

    def __init__(self, backend, value):
        self.backend = backend
        self.value = value

    def require_open(self):
        if self.value is None:
            raise ValueError('artifact handle is closed')
        return self.value

    def close(self):
        if self.value is not None:
            value, self.value = self.value, None
            _call(self.backend.close, value)


class _ValidatedHandle:
    """Created only after both opened objects pass the authorization checks."""

    def __init__(self, owned):
        self._owned = owned

    def metadata(self):
        owned = self._owned
        info = _call(owned.backend.describe, owned.require_open())
        return {key: info[key] for key in ('size', 'mtime_ns', 'file_id', 'change_token')}

    def read(self, n):
        if type(n) is not int:
            raise TypeError('read size must be an exact integer')
        if not 1 <= n <= _MAX_READ:
            raise ValueError('read size must be between 1 and 65536')
        owned = self._owned
        result = _call(owned.backend.read, owned.require_open(), n)
        if type(result) is not bytes or len(result) > n:
            raise RuntimeError('invalid native read result')
        return result

    def close(self):
        self._owned.close()


def _platform_name():
    return sys.platform


@contextmanager
def open_authorized_artifact(authorized_root, file_name):
    """Grant a managed readable handle to one host-selected direct child.

No content read occurs here. The root stays open until context exit. File close
is idempotent; read/metadata after close fail without reopening any path.
"""
    root_path = _root(authorized_root)
    _exact_path(file_name, 255)
    _component(file_name)
    if _platform_name() != 'win32' or _native is None:
        raise ObservationIOError('unsupported_platform')
    backend = _native
    child_path = ntpath.join(root_path, file_name)
    with ExitStack() as stack:
        root = _OwnedHandle(backend, _call(backend.open, root_path,
            access=_READ_ATTRIBUTES, share=_SHARE_READ, disposition=_OPEN_EXISTING,
            flags=_BACKUP_SEMANTICS | _OPEN_REPARSE_POINT))
        stack.callback(root.close)
        before = _call(backend.describe, root.require_open())
        _verify(before, 'directory', root_path)
        child = _OwnedHandle(backend, _call(backend.open, child_path,
            access=_GENERIC_READ, share=_SHARE_READ, disposition=_OPEN_EXISTING,
            flags=_BACKUP_SEMANTICS | _OPEN_REPARSE_POINT))
        stack.callback(child.close)
        info = _call(backend.describe, child.require_open())
        _verify(info, 'file', child_path)
        after = _call(backend.describe, root.require_open())
        _verify(after, 'directory', root_path)
        if (before['file_id'] != after['file_id']
                or before['volume_id'] != after['volume_id']
                or info['volume_id'] != before['volume_id']):
            raise ObservationIOError('unauthorized')
        yield _ValidatedHandle(child)


class _FileInformation(ctypes.Structure):
    _fields_ = [('attributes', wintypes.DWORD), ('creation', wintypes.FILETIME),
                ('access', wintypes.FILETIME), ('write', wintypes.FILETIME),
                ('volume', wintypes.DWORD), ('size_high', wintypes.DWORD),
                ('size_low', wintypes.DWORD), ('links', wintypes.DWORD),
                ('index_high', wintypes.DWORD), ('index_low', wintypes.DWORD)]


class _BasicInformation(ctypes.Structure):
    _fields_ = [('creation', ctypes.c_longlong), ('access', ctypes.c_longlong),
                ('write', ctypes.c_longlong), ('change', ctypes.c_longlong),
                ('attributes', wintypes.DWORD)]


class _WindowsNative:
    """Private fixed-signature Win32 primitives; no read loop or retry."""

    def __init__(self):
        dll = ctypes.WinDLL('kernel32', use_last_error=True)
        signatures = {
            'CreateFileW': ([wintypes.LPCWSTR, wintypes.DWORD, wintypes.DWORD,
                            ctypes.c_void_p, wintypes.DWORD, wintypes.DWORD, wintypes.HANDLE], wintypes.HANDLE),
            'CloseHandle': ([wintypes.HANDLE], wintypes.BOOL),
            'GetFileType': ([wintypes.HANDLE], wintypes.DWORD),
            'GetFileInformationByHandle': ([wintypes.HANDLE, ctypes.POINTER(_FileInformation)], wintypes.BOOL),
            'GetFileInformationByHandleEx': ([wintypes.HANDLE, ctypes.c_int, ctypes.c_void_p,
                                             wintypes.DWORD], wintypes.BOOL),
            'GetFinalPathNameByHandleW': ([wintypes.HANDLE, wintypes.LPWSTR,
                                          wintypes.DWORD, wintypes.DWORD], wintypes.DWORD),
            'GetVolumeInformationByHandleW': ([wintypes.HANDLE, wintypes.LPWSTR, wintypes.DWORD,
                ctypes.POINTER(wintypes.DWORD), ctypes.POINTER(wintypes.DWORD), ctypes.POINTER(wintypes.DWORD),
                wintypes.LPWSTR, wintypes.DWORD], wintypes.BOOL),
            'GetDriveTypeW': ([wintypes.LPCWSTR], wintypes.UINT),
            'ReadFile': ([wintypes.HANDLE, ctypes.c_void_p, wintypes.DWORD,
                          ctypes.POINTER(wintypes.DWORD), ctypes.c_void_p], wintypes.BOOL),
        }
        for name, (arguments, result) in signatures.items():
            function = getattr(dll, name)
            function.argtypes, function.restype = arguments, result
        self._dll = dll

    def open(self, path, *, access, share, disposition, flags):
        # Reject mapped network drives before attempting a file open.
        if self._dll.GetDriveTypeW(path[:3]) not in (2, 3):
            raise ObservationIOError('unsupported_platform')
        handle = self._dll.CreateFileW(path, access, share, None, disposition, flags, None)
        if handle == ctypes.c_void_p(-1).value:
            raise ctypes.WinError(ctypes.get_last_error())
        return handle

    def describe(self, handle):
        if self._dll.GetFileType(handle) != 1:  # FILE_TYPE_DISK only
            raise ObservationIOError('unsupported_type')
        info = _FileInformation()
        if not self._dll.GetFileInformationByHandle(handle, ctypes.byref(info)):
            raise ctypes.WinError(ctypes.get_last_error())
        if info.attributes & 0x400:  # FILE_ATTRIBUTE_REPARSE_POINT
            raise ObservationIOError('unsupported_type')
        basic = _BasicInformation()
        if not self._dll.GetFileInformationByHandleEx(handle, 0, ctypes.byref(basic), ctypes.sizeof(basic)):
            raise ctypes.WinError(ctypes.get_last_error())
        # Fixed bounded buffer; never retry a changing namespace or allocate from
        # an untrusted size. This exceeds the bounded root + child DOS path.
        path = ctypes.create_unicode_buffer(2048)
        length = self._dll.GetFinalPathNameByHandleW(handle, path, len(path), 0)
        if not length:
            raise ctypes.WinError(ctypes.get_last_error())
        if length >= len(path):
            raise ObservationIOError('unauthorized')
        final = _final_path(path.value)
        filesystem = ctypes.create_unicode_buffer(32)
        if not self._dll.GetVolumeInformationByHandleW(handle, None, 0, None, None, None,
                                                       filesystem, len(filesystem)):
            raise ctypes.WinError(ctypes.get_last_error())
        return dict(kind='directory' if info.attributes & 0x10 else 'file', reparse=False,
            final_path=path.value, filesystem=filesystem.value,
            local=self._dll.GetDriveTypeW(final[:3]) in (2, 3), volume_id=info.volume,
            file_id=(info.volume, (info.index_high << 32) | info.index_low),
            size=(info.size_high << 32) | info.size_low,
            mtime_ns=(basic.write - 116444736000000000) * 100, change_token=basic.change)

    def read(self, handle, n):
        buffer = ctypes.create_string_buffer(n)
        count = wintypes.DWORD()
        if not self._dll.ReadFile(handle, buffer, n, ctypes.byref(count), None):
            raise ctypes.WinError(ctypes.get_last_error())
        return buffer.raw[:count.value]

    def close(self, handle):
        if not self._dll.CloseHandle(handle):
            raise ctypes.WinError(ctypes.get_last_error())


def _load_native():
    if _platform_name() != 'win32':
        return None
    try:
        return _WindowsNative()
    except (OSError, AttributeError):  # Native library/API unavailable: no fallback.
        return None


_native = _load_native()

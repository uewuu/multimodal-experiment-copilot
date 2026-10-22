"""Private derived-path checks for the two host-authorized default tools.

This enforces static scan/read containment in a host-managed filesystem. It
does not make validation and open atomic or sandbox arbitrary host Python.
Direct host APIs outside this scope retain their original behavior.
"""

from contextlib import contextmanager
from contextvars import ContextVar
import os
from pathlib import Path
import stat


_ACCESS = ContextVar("default_tool_derived_path_access", default=None)


class _PathAuthorizationError(ValueError):
    def __init__(self):
        super().__init__("derived path is not an authorized experiment filesystem target")


@contextmanager
def _derived_access_scope(policy, tool_name):
    active = (policy, tool_name) if policy is not None and tool_name in (
        "analyze_experiment", "compare_experiments"
    ) else None
    token = _ACCESS.set(active)
    try:
        yield
    finally:
        _ACCESS.reset(token)


def _check_link_metadata(path):
    """Inspect lexical ancestors and link targets, without reading contents.

    resolve(strict=True) separately detects cycles. The visited set here avoids
    revisiting shared ancestors; it is not a replacement for OS resolution.
    """
    pending = [path]
    seen = set()
    while pending:
        current = pending.pop()
        for part in (*reversed(current.parents), current):
            if part in seen:
                continue
            seen.add(part)
            info = part.lstat()
            reparse = getattr(info, "st_file_attributes", 0) & getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0)
            if reparse and getattr(info, "st_reparse_tag", None) not in (
                stat.IO_REPARSE_TAG_SYMLINK, stat.IO_REPARSE_TAG_MOUNT_POINT
            ):
                raise _PathAuthorizationError()
            if reparse or stat.S_ISLNK(info.st_mode):
                target = Path(os.readlink(part))
                pending.append(target if target.is_absolute() else part.parent / target)


def _inspect(path, *, directory, discovery=False):
    policy, tool_name = _ACCESS.get()
    try:
        try:
            original = path.lstat()
        except FileNotFoundError:
            if not discovery:
                raise
            # Missing ordinary child names remain exclusions. A missing or
            # broken linked parent must still pass directory authorization.
            _authorize_derived_directory(path.parent)
            return None

        resolved = path.resolve(strict=True)
        field = ("experiment_root" if tool_name == "compare_experiments" else "experiment_dir") if directory and not discovery else "metrics_config"
        secured = policy(tool_name, {field: str(resolved)})
        target = Path(secured[field])
        _check_link_metadata(path)
        _check_link_metadata(target)
        mode = target.lstat().st_mode
        valid = stat.S_ISDIR(mode) if directory else stat.S_ISREG(mode)
        if not valid:
            # Preserve ordinary unrelated files and directory-as-required-file
            # exclusions; unsupported objects and link mismatches fail closed.
            ordinary = not stat.S_ISLNK(original.st_mode) and not (
                getattr(original, "st_file_attributes", 0) & getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0)
            )
            if discovery and ordinary and (stat.S_ISREG(mode) or stat.S_ISDIR(mode)):
                return None
            raise _PathAuthorizationError()
        return target
    except (OSError, ValueError):
        raise _PathAuthorizationError() from None
    except RuntimeError as error:
        # Python 3.11 pathlib reports POSIX symlink cycles as RuntimeError.
        # Do not convert unrelated programming errors into authorization errors.
        if str(error).startswith("Symlink loop from "):
            raise _PathAuthorizationError() from None
        raise


def _authorize_derived_directory(path):
    return path if _ACCESS.get() is None else _inspect(path, directory=True)


def _authorize_derived_file(path):
    return path if _ACCESS.get() is None else _inspect(path, directory=False)


def _is_experiment_directory(path):
    if _ACCESS.get() is None:
        return path.is_dir()
    # Membership precedes followed type checks even for link candidates.
    return _inspect(path, directory=True, discovery=True) is not None


def _is_experiment_file(path):
    if _ACCESS.get() is None:
        return path.is_file()
    return _inspect(path, directory=False, discovery=True) is not None


def _has_experiment_files(config_path, history_path):
    if _ACCESS.get() is None:
        return config_path.is_file() and history_path.is_file()
    # Do not silently skip an unsafe history link just because config is absent.
    config_exists = _is_experiment_file(config_path)
    history_exists = _is_experiment_file(history_path)
    return config_exists and history_exists

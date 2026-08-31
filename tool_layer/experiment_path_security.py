"""Request-scoped path policy for model-driven experiment tools."""

import os
from pathlib import Path, PureWindowsPath
from typing import Callable


_PathPolicy = Callable[[str, dict], dict]

_TOOL_PATH_FIELDS = {
    "analyze_experiment": (
        ("experiment_dir", True),
        ("metrics_config", False),
    ),
    "compare_experiments": (
        ("experiment_root", True),
        ("metrics_config", False),
    ),
}


def _boundary_error() -> ValueError:
    return ValueError(
        "path is outside the authorized experiment workspace"
    )


def _validate_model_path_form(
    value: str,
    trusted_root: Path | None,
) -> None:
    windows_path = PureWindowsPath(value)
    if value.startswith(("\\\\", "//")):
        raise _boundary_error()
    if value.startswith("\\"):
        raise _boundary_error()
    if (
        os.name == "nt"
        and windows_path.root
        and not windows_path.drive
    ):
        raise _boundary_error()
    if windows_path.drive and not windows_path.root:
        raise _boundary_error()

    if windows_path.drive and windows_path.root:
        trusted_drive = (
            None
            if trusted_root is None
            else PureWindowsPath(str(trusted_root)).drive
        )
        if (
            not trusted_drive
            or windows_path.drive.casefold()
            != trusted_drive.casefold()
        ):
            raise _boundary_error()


def _resolve_trusted_root(
    trusted_value: str,
) -> tuple[Path, Path]:
    trusted_input = Path(trusted_value)
    trusted_root = trusted_input.resolve(strict=True)
    if not trusted_root.is_dir():
        raise NotADirectoryError(trusted_value)
    return trusted_input, trusted_root


def _resolve_candidate(
    value: str,
    *,
    trusted_input: Path,
    trusted_root: Path,
    directory_required: bool,
) -> Path:
    _validate_model_path_form(value, trusted_root)
    candidate_input = Path(value)
    if candidate_input.is_absolute():
        unresolved = candidate_input
    elif candidate_input == trusted_input:
        unresolved = trusted_root
    else:
        unresolved = trusted_root / candidate_input

    candidate = unresolved.resolve(strict=False)
    try:
        candidate.relative_to(trusted_root)
    except ValueError:
        raise _boundary_error() from None

    if not candidate.exists():
        raise FileNotFoundError(value)
    if directory_required and not candidate.is_dir():
        raise NotADirectoryError(value)
    return candidate


def _secure_without_context(
    tool_name: str,
    arguments: dict,
) -> dict:
    fields = _TOOL_PATH_FIELDS.get(tool_name)
    if fields is None:
        return arguments

    for field, _ in fields:
        value = arguments.get(field)
        if not isinstance(value, str) or not value.strip():
            continue
        _validate_model_path_form(value, None)
        candidate = Path(value)
        if candidate.is_absolute():
            raise ValueError(
                "experiment filesystem access is not authorized"
            )
        try:
            candidate.resolve(strict=True)
        except OSError:
            continue
        raise ValueError(
            "experiment filesystem access is not authorized"
        )

    return arguments


def _build_experiment_path_policy(
    experiment_context: dict[str, object] | None,
) -> _PathPolicy:
    trusted_value: str | None = None
    if experiment_context is not None:
        for field in ("experiment_dir", "experiment_root"):
            value = experiment_context.get(field)
            if isinstance(value, str):
                trusted_value = value
                break

    def secure_arguments(
        tool_name: str,
        arguments: dict,
    ) -> dict:
        fields = _TOOL_PATH_FIELDS.get(tool_name)
        if fields is None:
            return arguments
        if trusted_value is None:
            return _secure_without_context(tool_name, arguments)

        trusted_input, trusted_root = _resolve_trusted_root(
            trusted_value
        )
        secured_arguments = dict(arguments)
        for field, directory_required in fields:
            value = arguments.get(field)
            if value is None and field == "metrics_config":
                continue
            if not isinstance(value, str) or not value.strip():
                continue
            secured_arguments[field] = str(
                _resolve_candidate(
                    value,
                    trusted_input=trusted_input,
                    trusted_root=trusted_root,
                    directory_required=directory_required,
                )
            )
        return secured_arguments

    return secure_arguments

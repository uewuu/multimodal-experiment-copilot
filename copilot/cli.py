"""Minimal command-line entry point for one experiment Copilot turn."""

import argparse
from collections.abc import Sequence
import math
from pathlib import Path
import sys

from copilot import run_copilot_turn
from generate_report import resolve_experiment_paths
from llm_clients import create_openai_client


_KNOWN_ERRORS = (
    FileNotFoundError,
    NotADirectoryError,
    ImportError,
    ValueError,
    TypeError,
)
_GENERIC_ERROR_MESSAGE = "The Copilot request could not be completed."
_INTERRUPTED_MESSAGE = "The Copilot request was interrupted."


def _non_blank_text(value: str) -> str:
    if not value.strip():
        raise argparse.ArgumentTypeError(
            "value must not be empty or whitespace"
        )
    return value


def _positive_finite_float(value: str) -> float:
    try:
        number = float(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            "value must be a number"
        ) from error

    if not math.isfinite(number) or number <= 0:
        raise argparse.ArgumentTypeError(
            "value must be finite and greater than zero"
        )
    return number


def build_parser() -> argparse.ArgumentParser:
    """Build the argument parser for the minimal Copilot CLI."""
    parser = argparse.ArgumentParser(
        description="Run one experiment Copilot turn.",
    )
    parser.add_argument(
        "--model",
        required=True,
        type=_non_blank_text,
        help="Model identifier used for the Copilot turn.",
    )
    parser.add_argument(
        "--question",
        required=True,
        type=_non_blank_text,
        help="Question to ask the experiment Copilot.",
    )
    parser.add_argument(
        "--experiment-dir",
        type=Path,
        default=None,
        help="Optional experiment directory to make available to tools.",
    )
    parser.add_argument(
        "--base-url",
        type=_non_blank_text,
        default=None,
        help="Optional provider-compatible API base URL.",
    )
    parser.add_argument(
        "--timeout",
        type=_positive_finite_float,
        default=None,
        help="Optional positive request timeout in seconds.",
    )
    return parser


def _build_experiment_context(
    experiment_dir: Path | None,
) -> dict[str, object] | None:
    if experiment_dir is None:
        return None
    if not experiment_dir.exists():
        raise FileNotFoundError(
            f"Experiment directory does not exist: {experiment_dir}"
        )
    if not experiment_dir.is_dir():
        raise NotADirectoryError(
            f"Experiment path is not a directory: {experiment_dir}"
        )

    resolve_experiment_paths(experiment_dir)
    return {"experiment_dir": str(experiment_dir)}


def _close_client(client: object) -> None:
    close = getattr(client, "close", None)
    if callable(close):
        close()


def _safe_error_message(error: Exception) -> str:
    if isinstance(error, _KNOWN_ERRORS):
        message = str(error)
        if message:
            return message
    return _GENERIC_ERROR_MESSAGE


def _write_error(message: str) -> None:
    sys.stderr.write(message.rstrip("\r\n") + "\n")


def main(
    argv: Sequence[str] | None = None,
) -> int:
    """Run one bounded Copilot request from command-line arguments."""
    args = build_parser().parse_args(argv)

    try:
        experiment_context = _build_experiment_context(
            args.experiment_dir
        )
        client = create_openai_client(
            base_url=args.base_url,
            timeout=args.timeout,
        )
    except KeyboardInterrupt:
        _write_error(_INTERRUPTED_MESSAGE)
        return 130
    except Exception as error:
        _write_error(_safe_error_message(error))
        return 1

    answer: str | None = None
    primary_error: Exception | None = None
    interrupted = False

    try:
        answer = run_copilot_turn(
            client,
            model=args.model,
            question=args.question,
            experiment_context=experiment_context,
        )
    except KeyboardInterrupt:
        interrupted = True
    except Exception as error:
        primary_error = error

    close_error: Exception | None = None
    try:
        _close_client(client)
    except KeyboardInterrupt:
        if primary_error is None:
            interrupted = True
    except Exception as error:
        close_error = error

    if interrupted:
        _write_error(_INTERRUPTED_MESSAGE)
        return 130
    if primary_error is not None:
        _write_error(_safe_error_message(primary_error))
        return 1
    if close_error is not None:
        _write_error(_safe_error_message(close_error))
        return 1
    if not isinstance(answer, str):
        _write_error(_GENERIC_ERROR_MESSAGE)
        return 1

    sys.stdout.write(answer.rstrip("\r\n") + "\n")
    return 0

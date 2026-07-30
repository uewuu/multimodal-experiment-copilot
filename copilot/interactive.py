"""Bounded interactive command-line entry point for the Copilot."""

import argparse
from collections.abc import Sequence
from pathlib import Path
import sys

from copilot.cli import (
    _INTERRUPTED_MESSAGE,
    _build_experiment_context,
    _close_client,
    _non_blank_text,
    _positive_finite_float,
    _safe_error_message,
    _write_error,
)
from copilot.session import CopilotSession
from llm_clients import create_openai_client


_PROMPT = "You> "
_ANSWER_PREFIX = "Copilot> "
_HELP_MESSAGE = "Commands: /help, /reset, /exit, /quit"
_RESET_MESSAGE = "Session history cleared."
_EXIT_COMMANDS = frozenset({"/exit", "/quit"})


def _positive_integer(value: str) -> int:
    try:
        number = int(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            "value must be an integer"
        ) from error

    if number < 1:
        raise argparse.ArgumentTypeError(
            "value must be greater than zero"
        )
    return number


def build_parser() -> argparse.ArgumentParser:
    """Build the argument parser for the interactive Copilot CLI."""
    parser = argparse.ArgumentParser(
        description="Run a bounded interactive Copilot session.",
    )
    parser.add_argument(
        "--model",
        required=True,
        type=_non_blank_text,
        help="Model identifier used for the Copilot session.",
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
    parser.add_argument(
        "--max-turns",
        type=_positive_integer,
        default=8,
        help="Maximum number of conversation turns retained in memory.",
    )
    return parser


def _write_output(message: str) -> None:
    sys.stdout.write(message.rstrip("\r\n") + "\n")


def _write_answer(answer: str) -> None:
    if not isinstance(answer, str):
        raise TypeError("answer must be a string")
    _write_output(_ANSWER_PREFIX + answer)


def _run_loop(session: CopilotSession) -> None:
    while True:
        try:
            raw_input = input(_PROMPT)
        except EOFError:
            return

        command = raw_input.strip().casefold()
        if not command:
            continue
        if command in _EXIT_COMMANDS:
            return
        if command == "/reset":
            session.reset()
            _write_output(_RESET_MESSAGE)
            continue
        if command == "/help":
            _write_output(_HELP_MESSAGE)
            continue

        _write_answer(session.ask(raw_input))


def main(
    argv: Sequence[str] | None = None,
) -> int:
    """Run one bounded interactive Copilot session."""
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

    primary_error: Exception | None = None
    interrupted = False

    try:
        session = CopilotSession(
            client,
            model=args.model,
            experiment_context=experiment_context,
            max_turns=args.max_turns,
        )
        _run_loop(session)
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
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

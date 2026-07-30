import argparse
import builtins
import importlib
import inspect
import io
import os
from pathlib import Path
import subprocess
import sys

import pytest


MODULE_NAME = "copilot.interactive"
SYNTHETIC_SECRET = "synthetic-secret-must-not-leak"
CLIENT_MARKER = "FAKE-INTERACTIVE-CLIENT-MUST-NOT-LEAK"
INTERRUPTED_MESSAGE = "The Copilot request was interrupted."


class _FakeClient:
    def __init__(self, close_error: BaseException | None = None) -> None:
        self.close_calls = 0
        self.close_error = close_error

    def close(self) -> None:
        self.close_calls += 1
        if self.close_error is not None:
            raise self.close_error

    def __repr__(self) -> str:
        return CLIENT_MARKER


class _ClientWithoutClose:
    pass


class _FailingWriter:
    def write(self, value: str) -> int:
        raise RuntimeError(SYNTHETIC_SECRET)

    def flush(self) -> None:
        return None


def _load_interactive_module():
    sys.modules.pop(MODULE_NAME, None)
    return importlib.import_module(MODULE_NAME)


def _minimal_args() -> list[str]:
    return ["--model", "test-model"]


def _make_experiment_dir(tmp_path: Path) -> Path:
    experiment_dir = tmp_path / "experiment"
    experiment_dir.mkdir()
    (experiment_dir / "hparams.yaml").write_text(
        "config:\n  model: demo\n",
        encoding="utf-8",
    )
    (experiment_dir / "history.json").write_text(
        '{"valid": {}}',
        encoding="utf-8",
    )
    return experiment_dir


def _call_main(interactive, arguments: list[str]) -> int:
    try:
        result = interactive.main(arguments)
    except SystemExit as error:
        return int(error.code)
    assert type(result) is int
    return result


def _patch_dependencies(
    monkeypatch: pytest.MonkeyPatch,
    interactive,
    *,
    events: list[object],
    answers: list[str] | None = None,
    client: object | None = None,
    factory_error: BaseException | None = None,
    session_error: BaseException | None = None,
    ask_error: BaseException | None = None,
    reset_error: BaseException | None = None,
) -> dict[str, object]:
    state: dict[str, object] = {
        "factory_calls": [],
        "session_calls": [],
        "ask_calls": [],
        "reset_calls": 0,
        "prompts": [],
    }
    selected_client = client if client is not None else _FakeClient()
    pending_events = list(events)
    pending_answers = list(answers or [])

    def fake_input(prompt: str) -> str:
        state["prompts"].append(prompt)
        if not pending_events:
            raise AssertionError("interactive CLI requested extra input")
        event = pending_events.pop(0)
        if isinstance(event, BaseException):
            raise event
        assert isinstance(event, str)
        return event

    def fake_factory(**kwargs: object) -> object:
        state["factory_calls"].append(kwargs)
        if factory_error is not None:
            raise factory_error
        return selected_client

    class FakeSession:
        def __init__(self, received_client: object, **kwargs: object) -> None:
            state["session_calls"].append(
                {"client": received_client, **kwargs}
            )
            if session_error is not None:
                raise session_error

        def ask(self, question: str) -> str:
            state["ask_calls"].append(question)
            if ask_error is not None:
                raise ask_error
            if pending_answers:
                return pending_answers.pop(0)
            return f"answer:{question}"

        def reset(self) -> None:
            state["reset_calls"] += 1
            if reset_error is not None:
                raise reset_error

    monkeypatch.setattr(builtins, "input", fake_input)
    monkeypatch.setattr(
        interactive,
        "create_openai_client",
        fake_factory,
    )
    monkeypatch.setattr(interactive, "CopilotSession", FakeSession)
    state["client"] = selected_client
    return state


def _module_subprocess(*arguments: str) -> subprocess.CompletedProcess[str]:
    environment = os.environ.copy()
    environment.pop("OPENAI_API_KEY", None)
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    return subprocess.run(
        [sys.executable, "-m", MODULE_NAME, *arguments],
        cwd=Path(__file__).resolve().parents[1],
        capture_output=True,
        text=True,
        encoding="utf-8",
        env=environment,
        check=False,
    )


def _assert_safe_failure(
    code: int,
    captured: pytest.CaptureFixture[str],
) -> None:
    assert code == 1
    assert captured.err
    assert "Traceback" not in captured.err
    assert SYNTHETIC_SECRET not in captured.err
    assert CLIENT_MARKER not in captured.err


def test_interactive_module_import_is_sdk_free_and_side_effect_free() -> None:
    _load_interactive_module()
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import sys; import copilot.interactive; "
                "print('openai' in sys.modules)"
            ),
        ],
        cwd=Path(__file__).resolve().parents[1],
        capture_output=True,
        text=True,
        encoding="utf-8",
        env={
            **os.environ,
            "PYTHONDONTWRITEBYTECODE": "1",
            "OPENAI_API_KEY": "",
        },
        check=False,
    )
    assert result.returncode == 0
    assert result.stdout == "False\n"
    assert result.stderr == ""


def test_public_functions_have_approved_signatures() -> None:
    interactive = _load_interactive_module()
    assert isinstance(interactive.build_parser(), argparse.ArgumentParser)
    signature = inspect.signature(interactive.main)
    assert list(signature.parameters) == ["argv"]
    assert signature.parameters["argv"].default is None


def test_parser_requires_model() -> None:
    interactive = _load_interactive_module()
    with pytest.raises(SystemExit) as error:
        interactive.build_parser().parse_args([])
    assert error.value.code == 2


@pytest.mark.parametrize("model", ["  模型-🚀  "])
def test_parser_preserves_valid_model_text(model: str) -> None:
    interactive = _load_interactive_module()
    args = interactive.build_parser().parse_args(["--model", model])
    assert args.model == model


def test_parser_defaults_optional_arguments() -> None:
    interactive = _load_interactive_module()
    args = interactive.build_parser().parse_args(_minimal_args())
    assert args.experiment_dir is None
    assert args.base_url is None
    assert args.timeout is None
    assert args.max_turns == 8


@pytest.mark.parametrize("blank", ["", " ", "\t"])
def test_parser_rejects_blank_model(blank: str) -> None:
    interactive = _load_interactive_module()
    with pytest.raises(SystemExit) as error:
        interactive.build_parser().parse_args(["--model", blank])
    assert error.value.code == 2


def test_parser_converts_experiment_directory_to_path() -> None:
    interactive = _load_interactive_module()
    args = interactive.build_parser().parse_args(
        [*_minimal_args(), "--experiment-dir", "relative/experiment"]
    )
    assert args.experiment_dir == Path("relative/experiment")


@pytest.mark.parametrize("blank", ["", " \t "])
def test_parser_rejects_blank_base_url(blank: str) -> None:
    interactive = _load_interactive_module()
    with pytest.raises(SystemExit) as error:
        interactive.build_parser().parse_args(
            [*_minimal_args(), "--base-url", blank]
        )
    assert error.value.code == 2


def test_parser_preserves_valid_base_url() -> None:
    interactive = _load_interactive_module()
    base_url = "  https://provider.invalid/v1  "
    args = interactive.build_parser().parse_args(
        [*_minimal_args(), "--base-url", base_url]
    )
    assert args.base_url == base_url


@pytest.mark.parametrize(
    "timeout",
    ["", "soon", "0", "-1", "nan", "inf"],
)
def test_parser_rejects_invalid_timeout(timeout: str) -> None:
    interactive = _load_interactive_module()
    with pytest.raises(SystemExit) as error:
        interactive.build_parser().parse_args(
            [*_minimal_args(), "--timeout", timeout]
        )
    assert error.value.code == 2


@pytest.mark.parametrize("timeout", ["0.25"])
def test_parser_accepts_positive_finite_timeout(timeout: str) -> None:
    interactive = _load_interactive_module()
    args = interactive.build_parser().parse_args(
        [*_minimal_args(), "--timeout", timeout]
    )
    assert args.timeout == float(timeout)


def test_parser_defaults_max_turns_to_session_default() -> None:
    interactive = _load_interactive_module()
    args = interactive.build_parser().parse_args(_minimal_args())
    assert args.max_turns == 8


@pytest.mark.parametrize(
    "max_turns",
    ["", "0", "-1", "1.5", "true", "false", "many"],
)
def test_parser_rejects_invalid_max_turns(max_turns: str) -> None:
    interactive = _load_interactive_module()
    with pytest.raises(SystemExit) as error:
        interactive.build_parser().parse_args(
            [*_minimal_args(), "--max-turns", max_turns]
        )
    assert error.value.code == 2


def test_parser_accepts_positive_integer_max_turns() -> None:
    interactive = _load_interactive_module()
    args = interactive.build_parser().parse_args(
        [*_minimal_args(), "--max-turns", "3"]
    )
    assert args.max_turns == 3


@pytest.mark.parametrize(
    "option",
    ["--question", "--api-key", "--unknown-option"],
)
def test_parser_rejects_forbidden_options(option: str) -> None:
    interactive = _load_interactive_module()
    with pytest.raises(SystemExit) as error:
        interactive.build_parser().parse_args(
            [*_minimal_args(), option, "forbidden"]
        )
    assert error.value.code == 2


def test_help_describes_only_interactive_arguments(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    interactive = _load_interactive_module()
    state = _patch_dependencies(
        monkeypatch,
        interactive,
        events=[],
    )
    with pytest.raises(SystemExit) as error:
        interactive.main(["--help"])
    assert error.value.code == 0
    output = capsys.readouterr().out
    for option in (
        "--model",
        "--experiment-dir",
        "--base-url",
        "--timeout",
        "--max-turns",
    ):
        assert option in output
    assert "--question" not in output
    assert "--api-key" not in output
    assert state["factory_calls"] == []
    assert state["session_calls"] == []


def test_session_receives_none_context_without_experiment_directory(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    interactive = _load_interactive_module()
    state = _patch_dependencies(
        monkeypatch,
        interactive,
        events=["/exit"],
    )
    assert _call_main(interactive, _minimal_args()) == 0
    assert state["session_calls"][0]["experiment_context"] is None


def test_valid_experiment_context_is_passed_to_session(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    interactive = _load_interactive_module()
    experiment_dir = _make_experiment_dir(tmp_path)
    state = _patch_dependencies(
        monkeypatch,
        interactive,
        events=["/exit"],
    )
    assert _call_main(
        interactive,
        [
            *_minimal_args(),
            "--experiment-dir",
            str(experiment_dir),
        ],
    ) == 0
    assert state["session_calls"][0]["experiment_context"] == {
        "experiment_dir": str(experiment_dir)
    }


@pytest.mark.parametrize(
    "invalid_kind",
    ["missing-directory", "file", "missing-config", "missing-history"],
)
def test_invalid_experiment_context_fails_before_factory(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    invalid_kind: str,
) -> None:
    interactive = _load_interactive_module()
    if invalid_kind == "missing-directory":
        experiment_dir = tmp_path / "missing"
    elif invalid_kind == "file":
        experiment_dir = tmp_path / "file"
        experiment_dir.write_text("not a directory", encoding="utf-8")
    else:
        experiment_dir = _make_experiment_dir(tmp_path)
        missing_name = (
            "hparams.yaml"
            if invalid_kind == "missing-config"
            else "history.json"
        )
        (experiment_dir / missing_name).unlink()
    state = _patch_dependencies(
        monkeypatch,
        interactive,
        events=[],
    )
    code = _call_main(
        interactive,
        [
            *_minimal_args(),
            "--experiment-dir",
            str(experiment_dir),
        ],
    )
    _assert_safe_failure(code, capsys.readouterr())
    assert state["factory_calls"] == []
    assert state["session_calls"] == []


def test_client_factory_is_called_once_with_connection_options(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    interactive = _load_interactive_module()
    state = _patch_dependencies(
        monkeypatch,
        interactive,
        events=["question", "/exit"],
    )
    assert _call_main(
        interactive,
        [
            *_minimal_args(),
            "--base-url",
            "https://provider.invalid/v1",
            "--timeout",
            "4.5",
        ],
    ) == 0
    assert state["factory_calls"] == [
        {
            "base_url": "https://provider.invalid/v1",
            "timeout": 4.5,
        }
    ]


def test_client_factory_never_receives_api_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    interactive = _load_interactive_module()
    state = _patch_dependencies(
        monkeypatch,
        interactive,
        events=["/exit"],
    )
    assert _call_main(interactive, _minimal_args()) == 0
    assert "api_key" not in state["factory_calls"][0]


def test_session_is_created_once_with_exact_configuration(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    interactive = _load_interactive_module()
    state = _patch_dependencies(
        monkeypatch,
        interactive,
        events=["first", "second", "/exit"],
    )
    assert _call_main(
        interactive,
        [*_minimal_args(), "--max-turns", "3"],
    ) == 0
    assert state["session_calls"] == [
        {
            "client": state["client"],
            "model": "test-model",
            "experiment_context": None,
            "max_turns": 3,
        }
    ]


def test_multiple_questions_reuse_the_same_session(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    interactive = _load_interactive_module()
    state = _patch_dependencies(
        monkeypatch,
        interactive,
        events=["first", "second", "/exit"],
    )
    assert _call_main(interactive, _minimal_args()) == 0
    assert len(state["session_calls"]) == 1
    assert state["ask_calls"] == ["first", "second"]


@pytest.mark.parametrize(
    "question",
    ["  preserve this whitespace  ", "请分析实验结果 🚀"],
)
def test_questions_are_passed_to_session_unchanged(
    monkeypatch: pytest.MonkeyPatch,
    question: str,
) -> None:
    interactive = _load_interactive_module()
    state = _patch_dependencies(
        monkeypatch,
        interactive,
        events=[question, "/exit"],
    )
    assert _call_main(interactive, _minimal_args()) == 0
    assert state["ask_calls"] == [question]


@pytest.mark.parametrize("blank", ["", " ", "\t"])
def test_blank_input_does_not_call_session(
    monkeypatch: pytest.MonkeyPatch,
    blank: str,
) -> None:
    interactive = _load_interactive_module()
    state = _patch_dependencies(
        monkeypatch,
        interactive,
        events=[blank, "/exit"],
    )
    assert _call_main(interactive, _minimal_args()) == 0
    assert state["ask_calls"] == []


@pytest.mark.parametrize(
    "command",
    ["/exit", "/QUIT", "  /Exit  "],
)
def test_exit_commands_are_case_insensitive_and_whitespace_tolerant(
    monkeypatch: pytest.MonkeyPatch,
    command: str,
) -> None:
    interactive = _load_interactive_module()
    state = _patch_dependencies(
        monkeypatch,
        interactive,
        events=[command],
    )
    assert _call_main(interactive, _minimal_args()) == 0
    assert state["ask_calls"] == []


@pytest.mark.parametrize("command", ["  /RESET  "])
def test_reset_uses_the_same_session_and_confirms(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    command: str,
) -> None:
    interactive = _load_interactive_module()
    state = _patch_dependencies(
        monkeypatch,
        interactive,
        events=["before", command, "after", "/exit"],
    )
    assert _call_main(interactive, _minimal_args()) == 0
    assert len(state["session_calls"]) == 1
    assert state["reset_calls"] == 1
    assert state["ask_calls"] == ["before", "after"]
    assert "Session history cleared.\n" in capsys.readouterr().out


def test_help_command_prints_commands_without_calling_session(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    interactive = _load_interactive_module()
    state = _patch_dependencies(
        monkeypatch,
        interactive,
        events=["/help", "/exit"],
    )
    assert _call_main(interactive, _minimal_args()) == 0
    assert state["ask_calls"] == []
    assert (
        "Commands: /help, /reset, /exit, /quit\n"
        in capsys.readouterr().out
    )


@pytest.mark.parametrize(
    "question",
    ["/unknown", "please explain /exit behavior"],
)
def test_unknown_or_embedded_commands_are_ordinary_questions(
    monkeypatch: pytest.MonkeyPatch,
    question: str,
) -> None:
    interactive = _load_interactive_module()
    state = _patch_dependencies(
        monkeypatch,
        interactive,
        events=[question, "/exit"],
    )
    assert _call_main(interactive, _minimal_args()) == 0
    assert state["ask_calls"] == [question]


@pytest.mark.parametrize(
    ("answer", "expected"),
    [
        ("answer", "Copilot> answer\n"),
        ("answer\n\n", "Copilot> answer\n"),
        ("first\nsecond", "Copilot> first\nsecond\n"),
    ],
)
def test_answers_have_prefix_and_one_trailing_newline(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    answer: str,
    expected: str,
) -> None:
    interactive = _load_interactive_module()
    state = _patch_dependencies(
        monkeypatch,
        interactive,
        events=["question", "/exit"],
        answers=[answer],
    )
    assert _call_main(interactive, _minimal_args()) == 0
    assert expected in capsys.readouterr().out
    assert state["prompts"] == ["You> ", "You> "]


def test_eof_exits_successfully_without_farewell(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    interactive = _load_interactive_module()
    state = _patch_dependencies(
        monkeypatch,
        interactive,
        events=[EOFError()],
    )
    assert _call_main(interactive, _minimal_args()) == 0
    captured = capsys.readouterr()
    assert "goodbye" not in captured.out.casefold()
    assert captured.err == ""
    assert state["ask_calls"] == []


@pytest.mark.parametrize(
    "location",
    ["input", "ask", "reset"],
)
def test_keyboard_interrupt_is_safe_and_returns_130(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    location: str,
) -> None:
    interactive = _load_interactive_module()
    events: list[object]
    if location == "input":
        events = [KeyboardInterrupt()]
    elif location == "ask":
        events = ["question"]
    else:
        events = ["/reset"]
    state = _patch_dependencies(
        monkeypatch,
        interactive,
        events=events,
        ask_error=KeyboardInterrupt() if location == "ask" else None,
        reset_error=(
            KeyboardInterrupt() if location == "reset" else None
        ),
    )
    assert _call_main(interactive, _minimal_args()) == 130
    captured = capsys.readouterr()
    assert INTERRUPTED_MESSAGE in captured.err
    assert "Traceback" not in captured.err
    assert state["client"].close_calls == 1


@pytest.mark.parametrize(
    "location",
    ["input", "ask", "reset", "output"],
)
def test_interaction_errors_fail_fast_without_secret_leaks(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    location: str,
) -> None:
    interactive = _load_interactive_module()
    error = RuntimeError(SYNTHETIC_SECRET)
    events: list[object]
    if location == "input":
        events = [error, "must-not-run"]
    elif location == "ask":
        events = ["question", "must-not-run"]
    elif location == "reset":
        events = ["/reset", "must-not-run"]
    else:
        events = ["question"]
    with monkeypatch.context() as patch:
        state = _patch_dependencies(
            patch,
            interactive,
            events=events,
            ask_error=error if location == "ask" else None,
            reset_error=error if location == "reset" else None,
        )
        if location == "output":
            patch.setattr(sys, "stdout", _FailingWriter())
        code = _call_main(interactive, _minimal_args())
    _assert_safe_failure(code, capsys.readouterr())
    assert len(state["prompts"]) == 1


def test_factory_and_session_constructor_failures_have_correct_ownership(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    interactive = _load_interactive_module()
    with monkeypatch.context() as patch:
        factory_state = _patch_dependencies(
            patch,
            interactive,
            events=[],
            factory_error=RuntimeError(SYNTHETIC_SECRET),
        )
        _assert_safe_failure(
            _call_main(interactive, _minimal_args()),
            capsys.readouterr(),
        )
        assert factory_state["client"].close_calls == 0
    with monkeypatch.context() as patch:
        session_state = _patch_dependencies(
            patch,
            interactive,
            events=[],
            session_error=ValueError("invalid session"),
        )
        _assert_safe_failure(
            _call_main(interactive, _minimal_args()),
            capsys.readouterr(),
        )
        assert session_state["client"].close_calls == 1


@pytest.mark.parametrize(
    "termination",
    ["success", "quit", "eof", "ask-error", "no-close"],
)
def test_client_is_closed_once_for_every_owned_lifecycle(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    termination: str,
) -> None:
    interactive = _load_interactive_module()
    if termination == "success":
        events: list[object] = ["question", "/exit"]
    elif termination == "quit":
        events = ["/quit"]
    elif termination == "eof":
        events = [EOFError()]
    elif termination == "no-close":
        events = ["question", "/exit"]
    else:
        events = ["question"]
    selected_client = (
        _ClientWithoutClose()
        if termination == "no-close"
        else None
    )
    state = _patch_dependencies(
        monkeypatch,
        interactive,
        events=events,
        client=selected_client,
        ask_error=(
            ValueError("ask failed")
            if termination == "ask-error"
            else None
        ),
    )
    code = _call_main(interactive, _minimal_args())
    capsys.readouterr()
    assert code == (1 if termination == "ask-error" else 0)
    if termination == "no-close":
        assert not hasattr(state["client"], "close")
    else:
        assert state["client"].close_calls == 1


@pytest.mark.parametrize(
    "primary",
    ["ordinary", "interrupt"],
)
def test_close_failure_never_masks_primary_failure(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    primary: str,
) -> None:
    interactive = _load_interactive_module()
    client = _FakeClient(close_error=RuntimeError("close failed"))
    primary_error: BaseException = (
        ValueError("primary failure")
        if primary == "ordinary"
        else KeyboardInterrupt()
    )
    state = _patch_dependencies(
        monkeypatch,
        interactive,
        events=["question"],
        client=client,
        ask_error=primary_error,
    )
    code = _call_main(interactive, _minimal_args())
    captured = capsys.readouterr()
    assert code == (1 if primary == "ordinary" else 130)
    if primary == "ordinary":
        assert "primary failure" in captured.err
    else:
        assert INTERRUPTED_MESSAGE in captured.err
    assert state["client"].close_calls == 1
    if primary == "ordinary":
        with monkeypatch.context() as patch:
            close_only_client = _FakeClient(
                close_error=RuntimeError("close-only failure")
            )
            close_only_state = _patch_dependencies(
                patch,
                interactive,
                events=["/exit"],
                client=close_only_client,
            )
            close_only_code = _call_main(
                interactive,
                _minimal_args(),
            )
        _assert_safe_failure(
            close_only_code,
            capsys.readouterr(),
        )
        assert close_only_state["client"].close_calls == 1


def test_cli_has_no_secret_persistence_or_debug_output(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    interactive = _load_interactive_module()
    before = sorted(path.relative_to(tmp_path) for path in tmp_path.rglob("*"))
    state = _patch_dependencies(
        monkeypatch,
        interactive,
        events=["question", "/exit"],
        answers=["safe answer"],
    )
    assert _call_main(interactive, _minimal_args()) == 0
    captured = capsys.readouterr()
    after = sorted(path.relative_to(tmp_path) for path in tmp_path.rglob("*"))
    assert after == before
    assert SYNTHETIC_SECRET not in captured.out + captured.err
    assert CLIENT_MARKER not in captured.out + captured.err
    assert state["session_calls"][0]["client"] is state["client"]


def test_module_help_uses_current_python_without_provider_access() -> None:
    _load_interactive_module()
    result = _module_subprocess("--help")
    assert result.returncode == 0
    assert "--model" in result.stdout
    assert "--max-turns" in result.stdout
    assert "--question" not in result.stdout
    assert "--api-key" not in result.stdout
    assert result.stderr == ""

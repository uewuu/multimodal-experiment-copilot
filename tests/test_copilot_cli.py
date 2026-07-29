import argparse
import ast
import builtins
import importlib
import inspect
import json
import math
import os
from pathlib import Path
import runpy
import subprocess
import sys

import pytest
import yaml


CLI_MODULE = "copilot.cli"
MAIN_MODULE = "copilot.__main__"
SYNTHETIC_API_KEY = "synthetic-test-api-key-never-log"
CLIENT_REPR_MARKER = "FAKE-CLIENT-REPR-MUST-NOT-LEAK"


class _FakeClient:
    def __init__(
        self,
        *,
        close_error: Exception | None = None,
    ) -> None:
        self.close_calls = 0
        self.close_error = close_error

    def close(self) -> None:
        self.close_calls += 1
        if self.close_error is not None:
            raise self.close_error

    def __repr__(self) -> str:
        return CLIENT_REPR_MARKER


class _ClientWithoutClose:
    pass


class _ClientWithNonCallableClose:
    close = "not-callable"


def _remove_cli_modules() -> None:
    sys.modules.pop(MAIN_MODULE, None)
    sys.modules.pop(CLI_MODULE, None)


def _import_cli():
    _remove_cli_modules()
    return importlib.import_module(CLI_MODULE)


def _make_experiment_dir(
    tmp_path: Path,
    *,
    config_content: str = "config:\n  model: demo\n",
    history_content: str = '{"valid": {}}',
) -> Path:
    experiment_dir = tmp_path / "experiment"
    experiment_dir.mkdir()
    (experiment_dir / "hparams.yaml").write_text(
        config_content,
        encoding="utf-8",
    )
    (experiment_dir / "history.json").write_text(
        history_content,
        encoding="utf-8",
    )
    return experiment_dir


def _call_main(cli, argv: list[str]) -> int:
    try:
        result = cli.main(argv)
    except SystemExit as error:
        return int(error.code)
    assert isinstance(result, int)
    return result


def _patch_dependencies(
    monkeypatch: pytest.MonkeyPatch,
    cli,
    *,
    client: object | None = None,
    answer: str = "analysis complete",
    factory_error: BaseException | None = None,
    runtime_error: BaseException | None = None,
) -> tuple[object, list[dict], list[dict]]:
    selected_client = client if client is not None else _FakeClient()
    factory_calls: list[dict] = []
    runtime_calls: list[dict] = []

    def fake_factory(**kwargs: object) -> object:
        factory_calls.append(kwargs)
        if factory_error is not None:
            raise factory_error
        return selected_client

    def fake_runtime(
        received_client: object,
        **kwargs: object,
    ) -> str:
        runtime_calls.append(
            {
                "client": received_client,
                **kwargs,
            }
        )
        if runtime_error is not None:
            raise runtime_error
        return answer

    monkeypatch.setattr(cli, "create_openai_client", fake_factory)
    monkeypatch.setattr(cli, "run_copilot_turn", fake_runtime)
    return selected_client, factory_calls, runtime_calls


def _minimal_argv() -> list[str]:
    return [
        "--model",
        "test-model",
        "--question",
        "Analyze the experiment.",
    ]


def _module_subprocess(*arguments: str) -> subprocess.CompletedProcess[str]:
    environment = os.environ.copy()
    environment.pop("OPENAI_API_KEY", None)
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    return subprocess.run(
        [sys.executable, "-m", "copilot", *arguments],
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
    assert captured.out == ""
    assert captured.err
    assert "Traceback" not in captured.err
    assert SYNTHETIC_API_KEY not in captured.err
    assert CLIENT_REPR_MARKER not in captured.err


# Import safety


def test_cli_module_is_importable() -> None:
    assert _import_cli().__name__ == CLI_MODULE


def test_cli_import_does_not_import_openai(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original_import = builtins.__import__

    def guarded_import(name: str, *args: object, **kwargs: object):
        if name == "openai" or name.startswith("openai."):
            raise AssertionError("CLI import attempted to load OpenAI SDK")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded_import)
    assert _import_cli().__name__ == CLI_MODULE


def test_cli_import_does_not_create_client(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import llm_clients

    monkeypatch.setattr(
        llm_clients,
        "create_openai_client",
        lambda **kwargs: pytest.fail("client created during import"),
    )
    _import_cli()


def test_cli_import_does_not_call_runtime(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import copilot

    monkeypatch.setattr(
        copilot,
        "run_copilot_turn",
        lambda *args, **kwargs: pytest.fail("runtime called during import"),
    )
    _import_cli()


def test_cli_import_does_not_read_api_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import llm_clients.openai_client_factory as factory_module

    class _UnreadableEnvironment(dict):
        def get(self, key: object, default: object = None) -> object:
            raise AssertionError("environment read during import")

    monkeypatch.setattr(
        factory_module.os,
        "environ",
        _UnreadableEnvironment(),
    )
    _import_cli()


def test_cli_import_does_not_access_experiment_files(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        Path,
        "open",
        lambda *args, **kwargs: pytest.fail("file opened during import"),
    )
    _import_cli()


def test_cli_import_prints_nothing(
    capsys: pytest.CaptureFixture[str],
) -> None:
    _import_cli()
    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err == ""


def test_cli_import_does_not_exit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        sys,
        "exit",
        lambda *args: pytest.fail("sys.exit called during import"),
    )
    _import_cli()


def test_cli_source_has_no_direct_openai_import() -> None:
    source = Path("copilot/cli.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    imported = {
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, ast.Import)
        for alias in node.names
    }
    assert "openai" not in imported


def test_cli_source_has_no_tool_layer_adapter_or_dotenv_import() -> None:
    source = Path("copilot/cli.py").read_text(encoding="utf-8")
    assert "tool_layer" not in source
    assert "llm_adapters" not in source
    assert "dotenv" not in source


# Parser contract


def test_build_parser_returns_argument_parser() -> None:
    assert isinstance(_import_cli().build_parser(), argparse.ArgumentParser)


def test_main_is_callable_and_has_approved_signature() -> None:
    main = _import_cli().main
    assert callable(main)
    signature = inspect.signature(main)
    assert list(signature.parameters) == ["argv"]
    assert signature.parameters["argv"].default is None


@pytest.mark.parametrize(
    "arguments",
    [
        ["--question", "question"],
        ["--model", "model"],
    ],
)
def test_parser_requires_model_and_question(
    arguments: list[str],
) -> None:
    with pytest.raises(SystemExit) as error:
        _import_cli().build_parser().parse_args(arguments)
    assert error.value.code == 2


def test_parser_accepts_minimal_arguments() -> None:
    args = _import_cli().build_parser().parse_args(_minimal_argv())
    assert args.model == "test-model"
    assert args.question == "Analyze the experiment."


def test_parser_preserves_unicode_question() -> None:
    question = "请分析实验结果 🚀"
    args = _import_cli().build_parser().parse_args(
        ["--model", "model", "--question", question]
    )
    assert args.question == question


def test_parser_preserves_model_and_question_text() -> None:
    model = "  provider-model  "
    question = "  keep meaningful whitespace  "
    args = _import_cli().build_parser().parse_args(
        ["--model", model, "--question", question]
    )
    assert args.model == model
    assert args.question == question


@pytest.mark.parametrize("blank", ["", " ", "\t", "\r\n"])
def test_parser_rejects_blank_model(blank: str) -> None:
    with pytest.raises(SystemExit) as error:
        _import_cli().build_parser().parse_args(
            ["--model", blank, "--question", "question"]
        )
    assert error.value.code == 2


@pytest.mark.parametrize("blank", ["", " ", "\t", "\r\n"])
def test_parser_rejects_blank_question(blank: str) -> None:
    with pytest.raises(SystemExit) as error:
        _import_cli().build_parser().parse_args(
            ["--model", "model", "--question", blank]
        )
    assert error.value.code == 2


def test_parser_defaults_optional_arguments_to_none() -> None:
    args = _import_cli().build_parser().parse_args(_minimal_argv())
    assert args.experiment_dir is None
    assert args.base_url is None
    assert args.timeout is None


def test_parser_rejects_blank_base_url() -> None:
    with pytest.raises(SystemExit) as error:
        _import_cli().build_parser().parse_args(
            [*_minimal_argv(), "--base-url", "  "]
        )
    assert error.value.code == 2


def test_parser_rejects_non_numeric_timeout() -> None:
    with pytest.raises(SystemExit) as error:
        _import_cli().build_parser().parse_args(
            [*_minimal_argv(), "--timeout", "soon"]
        )
    assert error.value.code == 2


@pytest.mark.parametrize("timeout", ["0", "-1", "-0.01"])
def test_parser_rejects_non_positive_timeout(timeout: str) -> None:
    with pytest.raises(SystemExit) as error:
        _import_cli().build_parser().parse_args(
            [*_minimal_argv(), "--timeout", timeout]
        )
    assert error.value.code == 2


@pytest.mark.parametrize("timeout", ["nan", "inf", "-inf"])
def test_parser_rejects_non_finite_timeout(timeout: str) -> None:
    with pytest.raises(SystemExit) as error:
        _import_cli().build_parser().parse_args(
            [*_minimal_argv(), "--timeout", timeout]
        )
    assert error.value.code == 2


def test_parser_accepts_positive_finite_timeout() -> None:
    args = _import_cli().build_parser().parse_args(
        [*_minimal_argv(), "--timeout", "12.5"]
    )
    assert math.isclose(args.timeout, 12.5)


def test_parser_rejects_api_key_option() -> None:
    with pytest.raises(SystemExit) as error:
        _import_cli().build_parser().parse_args(
            [*_minimal_argv(), "--api-key", SYNTHETIC_API_KEY]
        )
    assert error.value.code == 2


def test_main_help_returns_zero() -> None:
    assert _call_main(_import_cli(), ["--help"]) == 0


def test_main_argument_error_returns_two() -> None:
    assert _call_main(_import_cli(), []) == 2


# Experiment context


def test_context_is_none_without_experiment_directory(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cli = _import_cli()
    _, _, runtime_calls = _patch_dependencies(monkeypatch, cli)
    assert _call_main(cli, _minimal_argv()) == 0
    assert runtime_calls[0]["experiment_context"] is None


def test_context_contains_only_experiment_directory(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    experiment_dir = _make_experiment_dir(tmp_path)
    cli = _import_cli()
    _, _, runtime_calls = _patch_dependencies(monkeypatch, cli)
    assert _call_main(
        cli,
        [*_minimal_argv(), "--experiment-dir", str(experiment_dir)],
    ) == 0
    assert runtime_calls[0]["experiment_context"] == {
        "experiment_dir": str(experiment_dir)
    }


def test_missing_experiment_directory_fails_before_factory(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    cli = _import_cli()
    _, factory_calls, runtime_calls = _patch_dependencies(monkeypatch, cli)
    code = _call_main(
        cli,
        [*_minimal_argv(), "--experiment-dir", str(tmp_path / "missing")],
    )
    _assert_safe_failure(code, capsys.readouterr())
    assert factory_calls == []
    assert runtime_calls == []


def test_file_is_rejected_as_experiment_directory(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    path = tmp_path / "file"
    path.write_text("not a directory", encoding="utf-8")
    cli = _import_cli()
    _, factory_calls, runtime_calls = _patch_dependencies(monkeypatch, cli)
    code = _call_main(
        cli,
        [*_minimal_argv(), "--experiment-dir", str(path)],
    )
    _assert_safe_failure(code, capsys.readouterr())
    assert factory_calls == []
    assert runtime_calls == []


@pytest.mark.parametrize("missing_name", ["hparams.yaml", "history.json"])
def test_missing_required_file_fails_before_dependencies(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    missing_name: str,
) -> None:
    experiment_dir = _make_experiment_dir(tmp_path)
    (experiment_dir / missing_name).unlink()
    cli = _import_cli()
    _, factory_calls, runtime_calls = _patch_dependencies(monkeypatch, cli)
    code = _call_main(
        cli,
        [*_minimal_argv(), "--experiment-dir", str(experiment_dir)],
    )
    _assert_safe_failure(code, capsys.readouterr())
    assert factory_calls == []
    assert runtime_calls == []


def test_cli_does_not_create_reports_or_output_directory(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    experiment_dir = _make_experiment_dir(tmp_path)
    before = sorted(path.relative_to(tmp_path) for path in tmp_path.rglob("*"))
    cli = _import_cli()
    _patch_dependencies(monkeypatch, cli)
    assert _call_main(
        cli,
        [*_minimal_argv(), "--experiment-dir", str(experiment_dir)],
    ) == 0
    after = sorted(path.relative_to(tmp_path) for path in tmp_path.rglob("*"))
    assert after == before
    assert not (tmp_path / "outputs").exists()


def test_cli_does_not_default_to_demo_experiment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cli = _import_cli()
    _, _, runtime_calls = _patch_dependencies(monkeypatch, cli)
    assert _call_main(cli, _minimal_argv()) == 0
    assert runtime_calls[0]["experiment_context"] is None


def test_cli_reuses_resolve_experiment_paths(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    experiment_dir = _make_experiment_dir(tmp_path)
    cli = _import_cli()
    resolved: list[Path] = []

    def fake_resolver(path: Path) -> tuple[Path, Path]:
        resolved.append(path)
        return path / "hparams.yaml", path / "history.json"

    monkeypatch.setattr(cli, "resolve_experiment_paths", fake_resolver)
    _patch_dependencies(monkeypatch, cli)
    assert _call_main(
        cli,
        [*_minimal_argv(), "--experiment-dir", str(experiment_dir)],
    ) == 0
    assert resolved == [experiment_dir]


@pytest.mark.parametrize(
    ("config_content", "history_content"),
    [
        ("config: [", '{"valid": {}}'),
        ("config: {}", '{"valid":'),
    ],
)
def test_context_validation_does_not_parse_file_contents(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    config_content: str,
    history_content: str,
) -> None:
    experiment_dir = _make_experiment_dir(
        tmp_path,
        config_content=config_content,
        history_content=history_content,
    )
    cli = _import_cli()
    _patch_dependencies(monkeypatch, cli)
    assert _call_main(
        cli,
        [*_minimal_argv(), "--experiment-dir", str(experiment_dir)],
    ) == 0


@pytest.mark.parametrize(
    "runtime_error",
    [
        json.JSONDecodeError("bad json", "x", 0),
        yaml.YAMLError("bad yaml"),
    ],
)
def test_invalid_data_tool_error_is_reported_safely(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    runtime_error: Exception,
) -> None:
    cli = _import_cli()
    _patch_dependencies(
        monkeypatch,
        cli,
        runtime_error=runtime_error,
    )
    _assert_safe_failure(
        _call_main(cli, _minimal_argv()),
        capsys.readouterr(),
    )


# Factory and runtime integration


def test_factory_is_called_exactly_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cli = _import_cli()
    _, factory_calls, _ = _patch_dependencies(monkeypatch, cli)
    assert _call_main(cli, _minimal_argv()) == 0
    assert len(factory_calls) == 1


def test_factory_result_is_passed_to_runtime(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cli = _import_cli()
    client, _, runtime_calls = _patch_dependencies(monkeypatch, cli)
    assert _call_main(cli, _minimal_argv()) == 0
    assert runtime_calls[0]["client"] is client


def test_factory_never_receives_api_key_keyword(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cli = _import_cli()
    _, factory_calls, _ = _patch_dependencies(monkeypatch, cli)
    assert _call_main(cli, _minimal_argv()) == 0
    assert "api_key" not in factory_calls[0]


def test_factory_receives_base_url_and_timeout(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cli = _import_cli()
    _, factory_calls, _ = _patch_dependencies(monkeypatch, cli)
    assert _call_main(
        cli,
        [
            *_minimal_argv(),
            "--base-url",
            "https://provider.invalid/v1",
            "--timeout",
            "7.5",
        ],
    ) == 0
    assert factory_calls[0]["base_url"] == "https://provider.invalid/v1"
    assert factory_calls[0]["timeout"] == 7.5


def test_factory_handles_omitted_optional_settings(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cli = _import_cli()
    _, factory_calls, _ = _patch_dependencies(monkeypatch, cli)
    assert _call_main(cli, _minimal_argv()) == 0
    assert factory_calls[0].get("base_url") is None
    assert factory_calls[0].get("timeout") is None


@pytest.mark.parametrize(
    "factory_error",
    [
        ImportError("OpenAI SDK is unavailable"),
        ValueError("OPENAI_API_KEY is required"),
        TypeError("invalid client configuration"),
    ],
)
def test_known_factory_errors_are_reported_safely(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    factory_error: Exception,
) -> None:
    cli = _import_cli()
    _, _, runtime_calls = _patch_dependencies(
        monkeypatch,
        cli,
        factory_error=factory_error,
    )
    _assert_safe_failure(
        _call_main(cli, _minimal_argv()),
        capsys.readouterr(),
    )
    assert runtime_calls == []


def test_unknown_factory_error_is_sanitized(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    cli = _import_cli()
    _patch_dependencies(
        monkeypatch,
        cli,
        factory_error=RuntimeError(SYNTHETIC_API_KEY),
    )
    _assert_safe_failure(
        _call_main(cli, _minimal_argv()),
        capsys.readouterr(),
    )


def test_runtime_is_called_exactly_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cli = _import_cli()
    _, _, runtime_calls = _patch_dependencies(monkeypatch, cli)
    assert _call_main(cli, _minimal_argv()) == 0
    assert len(runtime_calls) == 1


def test_runtime_receives_exact_model_question_and_no_extra_options(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = "  exact-model  "
    question = "  exact question  "
    cli = _import_cli()
    _, _, runtime_calls = _patch_dependencies(monkeypatch, cli)
    assert _call_main(
        cli,
        ["--model", model, "--question", question],
    ) == 0
    assert runtime_calls[0] == {
        "client": runtime_calls[0]["client"],
        "model": model,
        "question": question,
        "experiment_context": None,
    }


@pytest.mark.parametrize(
    "runtime_error",
    [
        ValueError("invalid runtime input"),
        TypeError("invalid runtime type"),
        FileNotFoundError("experiment data missing"),
        json.JSONDecodeError("bad json", "x", 0),
        yaml.YAMLError("bad yaml"),
    ],
)
def test_known_runtime_errors_are_reported_safely(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    runtime_error: Exception,
) -> None:
    cli = _import_cli()
    _patch_dependencies(
        monkeypatch,
        cli,
        runtime_error=runtime_error,
    )
    _assert_safe_failure(
        _call_main(cli, _minimal_argv()),
        capsys.readouterr(),
    )


def test_unknown_runtime_error_does_not_leak_secret(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    cli = _import_cli()
    _patch_dependencies(
        monkeypatch,
        cli,
        runtime_error=RuntimeError(SYNTHETIC_API_KEY),
    )
    _assert_safe_failure(
        _call_main(cli, _minimal_argv()),
        capsys.readouterr(),
    )


# Output, exit codes, and lifecycle


@pytest.mark.parametrize(
    ("answer", "expected"),
    [
        ("answer", "answer\n"),
        ("answer\n", "answer\n"),
        ("answer\n\n\n", "answer\n"),
    ],
)
def test_success_has_exactly_one_trailing_newline(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    answer: str,
    expected: str,
) -> None:
    cli = _import_cli()
    _patch_dependencies(monkeypatch, cli, answer=answer)
    assert _call_main(cli, _minimal_argv()) == 0
    captured = capsys.readouterr()
    assert captured.out == expected
    assert captured.err == ""


def test_success_output_never_contains_client_repr_or_api_key(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    cli = _import_cli()
    _patch_dependencies(monkeypatch, cli, answer="safe answer")
    assert _call_main(cli, _minimal_argv()) == 0
    combined = capsys.readouterr().out
    assert CLIENT_REPR_MARKER not in combined
    assert SYNTHETIC_API_KEY not in combined


def test_success_output_never_contains_full_context(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    experiment_dir = _make_experiment_dir(tmp_path)
    cli = _import_cli()
    _patch_dependencies(monkeypatch, cli, answer="safe answer")
    assert _call_main(
        cli,
        [*_minimal_argv(), "--experiment-dir", str(experiment_dir)],
    ) == 0
    assert str(experiment_dir) not in capsys.readouterr().out


def test_failures_do_not_print_traceback(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    cli = _import_cli()
    _patch_dependencies(
        monkeypatch,
        cli,
        runtime_error=RuntimeError("provider failure"),
    )
    _assert_safe_failure(
        _call_main(cli, _minimal_argv()),
        capsys.readouterr(),
    )


def test_client_is_closed_once_after_success(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cli = _import_cli()
    client = _FakeClient()
    _patch_dependencies(monkeypatch, cli, client=client)
    assert _call_main(cli, _minimal_argv()) == 0
    assert client.close_calls == 1


def test_client_is_closed_once_after_runtime_failure(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    cli = _import_cli()
    client = _FakeClient()
    _patch_dependencies(
        monkeypatch,
        cli,
        client=client,
        runtime_error=ValueError("runtime failed"),
    )
    _assert_safe_failure(
        _call_main(cli, _minimal_argv()),
        capsys.readouterr(),
    )
    assert client.close_calls == 1


def test_client_is_closed_once_after_keyboard_interrupt(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    cli = _import_cli()
    client = _FakeClient()
    _patch_dependencies(
        monkeypatch,
        cli,
        client=client,
        runtime_error=KeyboardInterrupt(),
    )
    code = _call_main(cli, _minimal_argv())
    captured = capsys.readouterr()
    assert code == 130
    assert captured.out == ""
    assert captured.err
    assert "Traceback" not in captured.err
    assert client.close_calls == 1


@pytest.mark.parametrize(
    "client",
    [_ClientWithoutClose(), _ClientWithNonCallableClose()],
)
def test_client_without_callable_close_is_supported(
    monkeypatch: pytest.MonkeyPatch,
    client: object,
) -> None:
    cli = _import_cli()
    _patch_dependencies(monkeypatch, cli, client=client)
    assert _call_main(cli, _minimal_argv()) == 0


def test_close_failure_after_success_returns_one_and_suppresses_answer(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    cli = _import_cli()
    client = _FakeClient(close_error=RuntimeError("close failed"))
    _patch_dependencies(monkeypatch, cli, client=client)
    _assert_safe_failure(
        _call_main(cli, _minimal_argv()),
        capsys.readouterr(),
    )
    assert client.close_calls == 1


def test_close_failure_does_not_mask_runtime_error(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    cli = _import_cli()
    client = _FakeClient(close_error=RuntimeError("close failed"))
    _patch_dependencies(
        monkeypatch,
        cli,
        client=client,
        runtime_error=ValueError("primary runtime error"),
    )
    code = _call_main(cli, _minimal_argv())
    captured = capsys.readouterr()
    _assert_safe_failure(code, captured)
    assert "primary runtime error" in captured.err
    assert client.close_calls == 1


def test_close_failure_does_not_mask_keyboard_interrupt(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    cli = _import_cli()
    client = _FakeClient(close_error=RuntimeError("close failed"))
    _patch_dependencies(
        monkeypatch,
        cli,
        client=client,
        runtime_error=KeyboardInterrupt(),
    )
    code = _call_main(cli, _minimal_argv())
    captured = capsys.readouterr()
    assert code == 130
    assert captured.out == ""
    assert "Traceback" not in captured.err
    assert client.close_calls == 1


def test_factory_failure_does_not_attempt_close(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    cli = _import_cli()
    client = _FakeClient()
    _patch_dependencies(
        monkeypatch,
        cli,
        client=client,
        factory_error=ImportError("missing SDK"),
    )
    _assert_safe_failure(
        _call_main(cli, _minimal_argv()),
        capsys.readouterr(),
    )
    assert client.close_calls == 0


# Module entry


def test_module_help_works_without_openai_sdk() -> None:
    result = _module_subprocess("--help")
    assert result.returncode == 0
    assert "--model" in result.stdout
    assert "--question" in result.stdout
    assert result.stderr == ""


def test_module_missing_model_returns_two() -> None:
    result = _module_subprocess("--question", "question")
    assert result.returncode == 2
    assert result.stdout == ""
    assert result.stderr


def test_module_missing_question_returns_two() -> None:
    result = _module_subprocess("--model", "model")
    assert result.returncode == 2
    assert result.stdout == ""
    assert result.stderr


def test_module_entry_delegates_to_main_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cli = _import_cli()
    calls: list[None] = []

    def fake_main() -> int:
        calls.append(None)
        return 17

    monkeypatch.setattr(cli, "main", fake_main)
    sys.modules.pop(MAIN_MODULE, None)
    with pytest.raises(SystemExit) as error:
        runpy.run_module(MAIN_MODULE, run_name="__main__")
    assert error.value.code == 17
    assert calls == [None]


def test_importing_main_module_does_not_execute_main(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cli = _import_cli()
    monkeypatch.setattr(
        cli,
        "main",
        lambda: pytest.fail("main executed during ordinary import"),
    )
    sys.modules.pop(MAIN_MODULE, None)
    assert importlib.import_module(MAIN_MODULE).__name__ == MAIN_MODULE


def test_main_module_is_thin_and_has_no_openai_import() -> None:
    path = Path("copilot/__main__.py")
    source = path.read_text(encoding="utf-8")
    tree = ast.parse(source)
    assert "openai" not in source
    assert "create_openai_client" not in source
    assert "run_copilot_turn" not in source
    assert not any(
        isinstance(node, (ast.FunctionDef, ast.ClassDef))
        for node in tree.body
    )

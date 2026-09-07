"""Black-box contracts for the deterministic Copilot demonstration."""

from __future__ import annotations

import json
import os
from pathlib import Path
import runpy
import socket
import subprocess
import sys
from collections.abc import Iterator
import urllib.request

import pytest

from copilot import CopilotService
import llm_adapters.openai_tool_adapter as openai_adapter


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
DEMO_PATH = REPOSITORY_ROOT / "examples" / "deterministic_copilot_demo.py"


def _require_demo() -> None:
    if not DEMO_PATH.is_file():
        pytest.fail(
            "deterministic Copilot demo is not implemented at "
            "examples/deterministic_copilot_demo.py"
        )


def _forbid_provider_credentials(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    markers = ("OPENAI", "ANTHROPIC", "AZURE", "API_KEY", "TOKEN")
    for name in tuple(os.environ):
        if any(marker in name.upper() for marker in markers):
            monkeypatch.delenv(name, raising=False)


def _forbid_network(monkeypatch: pytest.MonkeyPatch) -> None:
    def fail_network(*args: object, **kwargs: object) -> object:
        raise AssertionError("the deterministic demo must not access network")

    monkeypatch.setattr(socket, "create_connection", fail_network)
    monkeypatch.setattr(socket.socket, "connect", fail_network)
    monkeypatch.setattr(socket.socket, "connect_ex", fail_network)
    monkeypatch.setattr(urllib.request, "urlopen", fail_network)


def _run_demo(
    run_directory: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> dict:
    _require_demo()
    run_directory.mkdir()
    _forbid_provider_credentials(monkeypatch)
    _forbid_network(monkeypatch)
    monkeypatch.setattr(sys, "dont_write_bytecode", True)
    monkeypatch.setattr(sys, "argv", [str(DEMO_PATH)])
    monkeypatch.chdir(run_directory)

    try:
        runpy.run_path(str(DEMO_PATH), run_name="__main__")
    except SystemExit as error:
        assert error.code in (None, 0)

    captured = capsys.readouterr()
    assert captured.out.strip(), "the demo must emit a JSON result"
    try:
        result = json.loads(captured.out)
    except json.JSONDecodeError as error:
        pytest.fail(f"demo stdout is not one JSON document: {error}")
    assert type(result) is dict
    return result


def _walk_json(value: object) -> Iterator[object]:
    yield value
    if type(value) is dict:
        for child in value.values():
            yield from _walk_json(child)
    elif type(value) is list:
        for child in value:
            yield from _walk_json(child)
    elif isinstance(value, str) and value.lstrip().startswith(("{", "[")):
        try:
            decoded = json.loads(value)
        except json.JSONDecodeError:
            return
        yield from _walk_json(decoded)


def _comparison_payload(result: dict) -> dict:
    for value in _walk_json(result):
        if (
            type(value) is dict
            and type(value.get("experiment_counts")) is dict
            and type(value.get("comparison_records")) is list
        ):
            return value
    pytest.fail("demo result does not contain a comparison tool payload")


def _git_status() -> str:
    completed = subprocess.run(
        ["git", "status", "--short", "--untracked-files=all"],
        cwd=REPOSITORY_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    return completed.stdout


def test_deterministic_copilot_demo_script_exists() -> None:
    _require_demo()


def test_demo_uses_real_copilot_stack_in_temporary_workspace(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    service_contexts: list[dict[str, object] | None] = []
    tool_calls: list[tuple[str, dict]] = []
    original_service_run = CopilotService.run
    original_invoke_tool = openai_adapter.invoke_tool

    def observe_service_run(
        service: CopilotService,
        question: str,
        **kwargs: object,
    ) -> object:
        context = kwargs.get("experiment_context")
        assert context is None or type(context) is dict
        service_contexts.append(context)
        assert type(context) is dict
        experiment_root = Path(context["experiment_root"])
        assert experiment_root.is_absolute()
        assert not experiment_root.is_relative_to(REPOSITORY_ROOT)
        experiments = [
            path
            for path in experiment_root.iterdir()
            if (path / "hparams.yaml").is_file()
            and (path / "history.json").is_file()
        ]
        assert len(experiments) == 2
        return original_service_run(service, question, **kwargs)

    def observe_tool_call(tool_name: str, arguments: dict) -> dict:
        tool_calls.append((tool_name, dict(arguments)))
        return original_invoke_tool(tool_name, arguments)

    monkeypatch.setattr(CopilotService, "run", observe_service_run)
    monkeypatch.setattr(openai_adapter, "invoke_tool", observe_tool_call)

    _run_demo(tmp_path / "run", monkeypatch, capsys)

    assert len(service_contexts) == 1
    context = service_contexts[0]
    assert type(context) is dict
    experiment_root = Path(context["experiment_root"])

    assert len(tool_calls) == 1
    tool_name, arguments = tool_calls[0]
    assert tool_name == "compare_experiments"
    assert Path(arguments["experiment_root"]) == experiment_root
    assert arguments["include_diagnostics"] is True


def test_demo_output_is_deterministic_and_meaningful(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    first = _run_demo(tmp_path / "first", monkeypatch, capsys)
    second = _run_demo(tmp_path / "second", monkeypatch, capsys)

    assert second == first
    json.dumps(first, ensure_ascii=False, allow_nan=False)

    comparison = _comparison_payload(first)
    assert comparison["experiment_counts"] == {
        "total": 2,
        "successful": 2,
        "failed": 0,
    }
    records = comparison["comparison_records"]
    assert len(records) == 2
    assert len({record["experiment_name"] for record in records}) == 2
    best_value_fields = {
        field
        for field in records[0]
        if field.startswith("best_")
        and not field.endswith("_epoch")
        and field in records[1]
    }
    assert best_value_fields
    assert any(
        records[0][field] != records[1][field]
        for field in best_value_fields
    )

    diagnostics = comparison.get("diagnostics")
    assert type(diagnostics) is dict
    assert diagnostics["diagnostics"] or diagnostics["recommendations"]


def test_demo_leaves_repository_unchanged(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    status_before = _git_status()

    _run_demo(tmp_path / "run", monkeypatch, capsys)

    assert _git_status() == status_before

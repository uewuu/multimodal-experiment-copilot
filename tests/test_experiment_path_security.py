import copy
import inspect
import json
import os
from pathlib import Path
import subprocess
from types import SimpleNamespace

import pytest

import copilot
from copilot.session import CopilotSession
import llm_adapters.openai_tool_adapter as adapter
import tool_layer.experiment_tools as experiment_tools
from tool_layer import (
    analyze_experiment,
    compare_experiments,
    invoke_tool,
    list_tools,
)


BASE_COMMIT = "d6d05fcbee03c2c0a296112b4f03c4df7fdeeb17"


class _SequentialCompletions:
    def __init__(self, outcomes: list[object]) -> None:
        self.outcomes = list(outcomes)
        self.calls: list[dict[str, object]] = []

    @property
    def call_count(self) -> int:
        return len(self.calls)

    def create(self, **kwargs: object) -> object:
        self.calls.append(kwargs)
        index = self.call_count - 1
        if index >= len(self.outcomes):
            raise AssertionError("unexpected extra provider request")
        outcome = self.outcomes[index]
        if isinstance(outcome, BaseException):
            raise outcome
        return outcome


class _FakeClient:
    def __init__(self, outcomes: list[object]) -> None:
        self.completions = _SequentialCompletions(outcomes)
        self.chat = SimpleNamespace(completions=self.completions)
        self.close_calls = 0

    def close(self) -> None:
        self.close_calls += 1


def _tool_call(
    name: str,
    arguments: dict[str, object],
    *,
    call_id: str = "call_path_security",
) -> SimpleNamespace:
    return SimpleNamespace(
        id=call_id,
        type="function",
        function=SimpleNamespace(
            name=name,
            arguments=json.dumps(arguments, ensure_ascii=False),
        ),
    )


def _response(
    content: str | None = "done",
    *,
    tool_calls: list[object] | None = None,
) -> SimpleNamespace:
    return SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(
                    role="assistant",
                    content=content,
                    tool_calls=tool_calls,
                )
            )
        ]
    )


def _tool_client(
    tool_name: str,
    arguments: dict[str, object],
) -> _FakeClient:
    return _FakeClient(
        [
            _response(
                None,
                tool_calls=[_tool_call(tool_name, arguments)],
            ),
            _response("done"),
        ]
    )


def _write_experiment(
    experiment_dir: Path,
    *,
    r2: float = 0.75,
    racc: float = 0.9,
) -> None:
    experiment_dir.mkdir(parents=True)
    (experiment_dir / "hparams.yaml").write_text(
        "config:\n  batch_size: 8\n",
        encoding="utf-8",
    )
    (experiment_dir / "history.json").write_text(
        json.dumps(
            {
                "valid": {
                    "app": {
                        "r2": [[0, r2 / 2], [1, r2]],
                        "racc": [[0, racc / 2], [1, racc]],
                    }
                }
            }
        ),
        encoding="utf-8",
    )


def _write_metrics_config(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "metrics:\n"
        "  - name: r2\n"
        "    path: [valid, app, r2]\n"
        "    direction: maximize\n"
        "    display_name: R2\n",
        encoding="utf-8",
    )


def _patch_tool_invocation_guard(
    monkeypatch: pytest.MonkeyPatch,
) -> list[tuple[str, dict[str, object]]]:
    calls: list[tuple[str, dict[str, object]]] = []

    def fake_invoke(
        tool_name: str,
        arguments: dict[str, object],
    ) -> dict:
        calls.append((tool_name, copy.deepcopy(arguments)))
        return {"unexpected": "tool executed"}

    monkeypatch.setattr(adapter, "invoke_tool", fake_invoke)
    return calls


def _run_runtime(
    client: _FakeClient,
    context: dict[str, object] | None,
) -> str:
    return copilot.run_copilot_turn(
        client,
        model="test-model",
        question="Analyze the authorized experiment workspace.",
        experiment_context=context,
    )


def _assert_runtime_boundary_rejection(
    monkeypatch: pytest.MonkeyPatch,
    *,
    context: dict[str, object] | None,
    tool_name: str,
    arguments: dict[str, object],
    trusted_root: Path | None = None,
) -> ValueError:
    guarded_calls = _patch_tool_invocation_guard(monkeypatch)
    client = _tool_client(tool_name, arguments)

    with pytest.raises(ValueError) as caught:
        _run_runtime(client, context)

    assert guarded_calls == []
    assert client.completions.call_count == 1
    assert client.close_calls == 0
    if trusted_root is not None:
        assert str(trusted_root) not in str(caught.value)
    return caught.value


def _make_directory_link(link: Path, target: Path) -> None:
    try:
        link.symlink_to(target, target_is_directory=True)
        return
    except OSError as symlink_error:
        if os.name != "nt":
            pytest.skip(f"directory symlink unavailable: {symlink_error}")

    completed = subprocess.run(
        ["cmd", "/c", "mklink", "/J", str(link), str(target)],
        capture_output=True,
        text=True,
        check=False,
    )
    if completed.returncode != 0:
        pytest.skip(
            "directory link unavailable: "
            + (completed.stderr or completed.stdout).strip()
        )


def test_public_tool_and_runtime_signatures_remain_unchanged() -> None:
    assert str(inspect.signature(analyze_experiment)) == (
        "(experiment_dir: str, *, metrics_config: str | None = None, "
        "include_diagnostics: bool = False) -> dict"
    )
    assert str(inspect.signature(compare_experiments)) == (
        "(experiment_root: str, sort_by: str | None = None, "
        "descending: bool = True, *, metrics_config: str | None = None, "
        "include_diagnostics: bool = False) -> dict"
    )
    assert str(inspect.signature(invoke_tool)) == (
        "(tool_name: str, arguments: dict) -> dict"
    )
    assert str(inspect.signature(copilot.run_copilot_turn)) == (
        "(client: object, *, model: str, question: str, "
        "experiment_context: dict[str, object] | None = None, "
        "**request_options: object) -> str"
    )
    assert str(inspect.signature(CopilotSession)) == (
        "(client: object, *, model: str, experiment_context: "
        "dict[str, object] | None = None, max_turns: int = 8, "
        "**request_options: object) -> None"
    )


def test_tool_json_schemas_remain_unchanged() -> None:
    tools = list_tools()

    assert [tool["function"]["name"] for tool in tools] == [
        "analyze_experiment",
        "compare_experiments",
    ]
    analyze_parameters = tools[0]["function"]["parameters"]
    compare_parameters = tools[1]["function"]["parameters"]
    assert list(analyze_parameters["properties"]) == [
        "experiment_dir",
        "metrics_config",
        "include_diagnostics",
    ]
    assert list(compare_parameters["properties"]) == [
        "experiment_root",
        "sort_by",
        "descending",
        "metrics_config",
        "include_diagnostics",
    ]
    assert analyze_parameters["additionalProperties"] is False
    assert compare_parameters["additionalProperties"] is False
    assert all(
        "experiment_context" not in parameters["properties"]
        for parameters in (analyze_parameters, compare_parameters)
    )


def test_trusted_direct_tool_call_remains_usable(tmp_path: Path) -> None:
    experiment_dir = tmp_path / "direct"
    _write_experiment(experiment_dir)

    result = invoke_tool(
        "analyze_experiment",
        {"experiment_dir": str(experiment_dir)},
    )

    assert result["validation_metrics"]["r2"]["best_value"] == 0.75


def test_candidate_equal_to_trusted_experiment_dir_is_allowed(
    tmp_path: Path,
) -> None:
    experiment_dir = tmp_path / "workspace" / "experiment"
    _write_experiment(experiment_dir)
    client = _tool_client(
        "analyze_experiment",
        {"experiment_dir": str(experiment_dir)},
    )

    assert _run_runtime(
        client,
        {"experiment_dir": str(experiment_dir)},
    ) == "done"
    assert client.completions.call_count == 2


@pytest.mark.parametrize(
    "candidate_factory",
    [
        lambda root, experiment: "experiment",
        lambda root, experiment: "nested/../experiment",
        lambda root, experiment: str(experiment),
    ],
    ids=["nested", "normalized-inside", "absolute-inside"],
)
def test_experiment_root_allows_resolved_paths_inside_workspace(
    tmp_path: Path,
    candidate_factory: object,
) -> None:
    root = tmp_path / "workspace"
    experiment_dir = root / "experiment"
    _write_experiment(experiment_dir)
    (root / "nested").mkdir()
    candidate = candidate_factory(root, experiment_dir)
    client = _tool_client(
        "analyze_experiment",
        {"experiment_dir": candidate},
    )

    assert _run_runtime(
        client,
        {"experiment_root": str(root)},
    ) == "done"
    assert client.completions.call_count == 2


def test_trusted_experiment_root_itself_remains_usable(
    tmp_path: Path,
) -> None:
    root = tmp_path / "workspace"
    _write_experiment(root / "a", r2=0.6)
    _write_experiment(root / "b", r2=0.9)
    client = _tool_client(
        "compare_experiments",
        {"experiment_root": str(root)},
    )

    assert _run_runtime(
        client,
        {"experiment_root": str(root)},
    ) == "done"
    assert client.completions.call_count == 2


@pytest.mark.parametrize("candidate", ["..", "../outside"])
def test_traversal_outside_workspace_is_rejected_before_tool_execution(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    candidate: str,
) -> None:
    root = tmp_path / "workspace"
    root.mkdir()
    (tmp_path / "outside").mkdir()

    _assert_runtime_boundary_rejection(
        monkeypatch,
        context={"experiment_root": str(root)},
        tool_name="analyze_experiment",
        arguments={"experiment_dir": candidate},
        trusted_root=root,
    )


def test_absolute_path_outside_workspace_is_rejected(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "workspace"
    outside = tmp_path / "outside"
    root.mkdir()
    outside.mkdir()

    _assert_runtime_boundary_rejection(
        monkeypatch,
        context={"experiment_root": str(root)},
        tool_name="analyze_experiment",
        arguments={"experiment_dir": str(outside)},
        trusted_root=root,
    )


@pytest.mark.parametrize(
    "candidate",
    [
        r"Z:\outside\experiment",
        r"\outside\experiment",
        r"Z:outside\experiment",
        r"\\invalid-host.invalid\share\experiment",
    ],
    ids=[
        "cross-drive",
        "rooted-without-drive",
        "drive-relative",
        "unc-network-form",
    ],
)
def test_windows_escape_forms_are_rejected_without_tool_execution(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    candidate: str,
) -> None:
    root = tmp_path / "workspace"
    root.mkdir()

    _assert_runtime_boundary_rejection(
        monkeypatch,
        context={"experiment_root": str(root)},
        tool_name="analyze_experiment",
        arguments={"experiment_dir": candidate},
        trusted_root=root,
    )


def test_resolved_directory_link_escape_is_rejected(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "workspace"
    outside = tmp_path / "outside"
    root.mkdir()
    outside.mkdir()
    link = root / "escape-link"
    _make_directory_link(link, outside)

    _assert_runtime_boundary_rejection(
        monkeypatch,
        context={"experiment_root": str(root)},
        tool_name="analyze_experiment",
        arguments={"experiment_dir": "escape-link"},
        trusted_root=root,
    )


def test_no_experiment_context_denies_model_driven_filesystem_tool(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    experiment_dir = tmp_path / "experiment"
    _write_experiment(experiment_dir)

    _assert_runtime_boundary_rejection(
        monkeypatch,
        context=None,
        tool_name="analyze_experiment",
        arguments={"experiment_dir": str(experiment_dir)},
    )


def test_experiment_dir_capability_cannot_expand_to_sibling(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    allowed = tmp_path / "workspace" / "allowed"
    sibling = tmp_path / "workspace" / "sibling"
    _write_experiment(allowed)
    _write_experiment(sibling)

    _assert_runtime_boundary_rejection(
        monkeypatch,
        context={"experiment_dir": str(allowed)},
        tool_name="analyze_experiment",
        arguments={"experiment_dir": str(sibling)},
        trusted_root=allowed,
    )


def test_external_metrics_config_is_rejected(
    tmp_path: Path,
) -> None:
    experiment_dir = tmp_path / "workspace" / "experiment"
    metrics_config = tmp_path / "outside" / "metrics.yaml"
    _write_experiment(experiment_dir)
    _write_metrics_config(metrics_config)
    client = _tool_client(
        "analyze_experiment",
        {
            "experiment_dir": str(experiment_dir),
            "metrics_config": str(metrics_config),
        },
    )

    with pytest.raises(ValueError) as caught:
        _run_runtime(
            client,
            {"experiment_dir": str(experiment_dir)},
        )

    assert str(experiment_dir) not in str(caught.value)
    assert client.completions.call_count == 1


def test_metrics_config_inside_workspace_is_allowed(
    tmp_path: Path,
) -> None:
    experiment_dir = tmp_path / "workspace" / "experiment"
    metrics_config = experiment_dir / "metrics.yaml"
    _write_experiment(experiment_dir)
    _write_metrics_config(metrics_config)
    client = _tool_client(
        "analyze_experiment",
        {
            "experiment_dir": str(experiment_dir),
            "metrics_config": "metrics.yaml",
        },
    )

    assert _run_runtime(
        client,
        {"experiment_dir": str(experiment_dir)},
    ) == "done"
    assert client.completions.call_count == 2


def test_nonexistent_candidate_preserves_file_not_found(
    tmp_path: Path,
) -> None:
    root = tmp_path / "workspace"
    root.mkdir()
    missing = root / "missing"
    client = _tool_client(
        "analyze_experiment",
        {"experiment_dir": str(missing)},
    )

    with pytest.raises(FileNotFoundError):
        _run_runtime(client, {"experiment_root": str(root)})


def test_nonexistent_trusted_root_preserves_file_not_found(
    tmp_path: Path,
) -> None:
    missing = tmp_path / "missing-workspace"
    client = _tool_client(
        "analyze_experiment",
        {"experiment_dir": str(missing)},
    )

    with pytest.raises(FileNotFoundError):
        _run_runtime(client, {"experiment_dir": str(missing)})


@pytest.mark.parametrize("context_field", ["experiment_dir", "experiment_root"])
def test_file_used_as_trusted_directory_preserves_not_a_directory(
    tmp_path: Path,
    context_field: str,
) -> None:
    file_path = tmp_path / "not-a-directory"
    file_path.write_text("data", encoding="utf-8")
    client = _tool_client(
        "analyze_experiment",
        {"experiment_dir": str(file_path)},
    )

    with pytest.raises(NotADirectoryError):
        _run_runtime(client, {context_field: str(file_path)})


def test_boundary_rejection_happens_before_target_file_reads(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "workspace"
    outside = tmp_path / "outside"
    root.mkdir()
    outside.mkdir()
    reads: list[Path] = []

    def forbidden_read(path: Path) -> tuple[Path, Path]:
        reads.append(path)
        raise AssertionError("target experiment files were accessed")

    monkeypatch.setattr(
        experiment_tools,
        "resolve_experiment_paths",
        forbidden_read,
    )
    client = _tool_client(
        "analyze_experiment",
        {"experiment_dir": str(outside)},
    )

    with pytest.raises(ValueError):
        _run_runtime(client, {"experiment_root": str(root)})

    assert reads == []
    assert client.completions.call_count == 1


def test_runtime_result_enforces_the_same_request_boundary(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "workspace"
    outside = tmp_path / "outside"
    root.mkdir()
    outside.mkdir()
    guarded_calls = _patch_tool_invocation_guard(monkeypatch)
    client = _tool_client(
        "analyze_experiment",
        {"experiment_dir": str(outside)},
    )

    with pytest.raises(ValueError):
        copilot.run_copilot_turn_with_result(
            client,
            model="test-model",
            question="Analyze.",
            experiment_context={"experiment_root": str(root)},
        )

    assert guarded_calls == []


def test_session_enforces_the_same_request_boundary(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "workspace"
    outside = tmp_path / "outside"
    root.mkdir()
    outside.mkdir()
    guarded_calls = _patch_tool_invocation_guard(monkeypatch)
    client = _tool_client(
        "analyze_experiment",
        {"experiment_dir": str(outside)},
    )
    session = CopilotSession(
        client,
        model="test-model",
        experiment_context={"experiment_root": str(root)},
    )

    with pytest.raises(ValueError):
        session.ask("Analyze.")

    assert guarded_calls == []
    assert session.history == ()


def test_failure_observability_keeps_security_rejection_at_tool_execution(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "workspace"
    outside = tmp_path / "outside"
    root.mkdir()
    outside.mkdir()
    guarded_calls = _patch_tool_invocation_guard(monkeypatch)
    client = _tool_client(
        "analyze_experiment",
        {"experiment_dir": str(outside)},
    )
    observations: list[object] = []

    with pytest.raises(ValueError):
        copilot.run_copilot_turn_with_failure_observability(
            client,
            model="test-model",
            question="Analyze.",
            experiment_context={"experiment_root": str(root)},
            on_failure=observations.append,
        )

    assert guarded_calls == []
    assert len(observations) == 1
    observation = observations[0]
    assert observation.stage == "tool_execution"
    assert observation.provider_request_count == 1
    assert observation.tool_invocation_count == 1


def test_boundary_rejection_does_not_mutate_caller_inputs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "workspace"
    outside = tmp_path / "outside"
    root.mkdir()
    outside.mkdir()
    context = {"experiment_root": str(root)}
    context_before = copy.deepcopy(context)
    arguments = {"experiment_dir": str(outside)}
    tool_call = _tool_call("analyze_experiment", arguments)
    arguments_json = tool_call.function.arguments
    client = _FakeClient(
        [
            _response(None, tool_calls=[tool_call]),
            _response("done"),
        ]
    )
    _patch_tool_invocation_guard(monkeypatch)

    with pytest.raises(ValueError):
        _run_runtime(client, context)

    assert context == context_before
    assert arguments == {"experiment_dir": str(outside)}
    assert tool_call.function.arguments == arguments_json
    assert client.close_calls == 0

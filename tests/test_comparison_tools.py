import importlib
import inspect
import json
from pathlib import Path

import pytest

import compare_experiments as comparison_core
import tool_layer
import tool_layer.experiment_tools as experiment_tools
from metrics import MetricSpec


def _comparison_tool():
    module = importlib.import_module("tool_layer")
    assert hasattr(module, "compare_experiments"), (
        "tool_layer must publicly export compare_experiments"
    )
    tool = getattr(module, "compare_experiments")
    assert callable(tool), "tool_layer.compare_experiments must be callable"
    return tool


def _empty_payload(
    *,
    sort_by: str = "best_r2",
    descending: bool = True,
) -> dict:
    return {
        "sort_by": sort_by,
        "descending": descending,
        "experiment_counts": {
            "total": 0,
            "successful": 0,
            "failed": 0,
        },
        "comparison_records": [],
        "failed_experiments": [],
    }


def _default_summary(
    r2: float = 0.75,
    racc: float = 0.9,
) -> dict:
    return {
        "configuration": {"batch_size": 8},
        "validation_metrics": {
            "r2": {
                "best_value": r2,
                "best_epoch": 1,
            },
            "racc": {
                "best_value": racc,
                "best_epoch": 2,
            },
        },
    }


def _metric_specs() -> tuple[MetricSpec, ...]:
    return (
        MetricSpec(
            name="accuracy",
            path=("validation", "metrics", "accuracy"),
            direction="maximize",
            display_name="Accuracy",
            precision=4,
        ),
        MetricSpec(
            name="validation_loss",
            path=("validation", "metrics", "loss"),
            direction="minimize",
            display_name="Validation Loss",
            precision=6,
        ),
    )


def _write_default_experiment(
    experiment_dir: Path,
    *,
    r2: float,
    racc: float,
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
                        "r2": [[0, r2]],
                        "racc": [[0, racc]],
                    }
                }
            }
        ),
        encoding="utf-8",
    )


def _patch_comparison_flow(
    monkeypatch: pytest.MonkeyPatch,
    *,
    experiment_dirs: list[Path] | None = None,
    batch_result: dict | None = None,
    payload: dict | None = None,
) -> dict[str, object]:
    if experiment_dirs is None:
        experiment_dirs = []
    if batch_result is None:
        batch_result = {
            "successful_experiments": [],
            "failed_experiments": [],
        }
    if payload is None:
        payload = _empty_payload()

    calls: dict[str, object] = {}

    def fake_find(root_path: Path) -> list[Path]:
        calls["root_path"] = root_path
        return experiment_dirs

    def fake_analyze(
        received_dirs: list[Path],
        metric_specs=None,
    ) -> dict:
        calls["experiment_dirs"] = received_dirs
        calls["analyze_metric_specs"] = metric_specs
        return batch_result

    def fake_build(
        received_batch: dict,
        sort_by: str = "best_r2",
        descending: bool = True,
        metric_specs=None,
        *,
        include_diagnostics: bool = False,
    ) -> dict:
        calls["batch_result"] = received_batch
        calls["sort_by"] = sort_by
        calls["descending"] = descending
        calls["payload_metric_specs"] = metric_specs
        calls["include_diagnostics"] = include_diagnostics
        return payload

    monkeypatch.setattr(
        experiment_tools,
        "find_experiment_dirs",
        fake_find,
        raising=False,
    )
    monkeypatch.setattr(
        experiment_tools,
        "analyze_experiment_dirs",
        fake_analyze,
        raising=False,
    )
    monkeypatch.setattr(
        experiment_tools,
        "build_comparison_payload",
        fake_build,
        raising=False,
    )
    return calls


def test_tool_layer_publicly_exports_compare_experiments() -> None:
    tool = _comparison_tool()

    assert tool_layer.compare_experiments is tool
    assert "compare_experiments" in tool_layer.__all__


def test_compare_experiments_has_expected_signature() -> None:
    tool = _comparison_tool()

    signature = inspect.signature(tool)
    parameters = list(signature.parameters.values())

    assert [parameter.name for parameter in parameters] == [
        "experiment_root",
        "sort_by",
        "descending",
        "metrics_config",
        "include_diagnostics",
    ]
    assert parameters[0].annotation is str
    assert parameters[0].default is inspect.Parameter.empty
    assert parameters[1].default is None
    assert parameters[2].default is True
    assert parameters[3].kind is inspect.Parameter.KEYWORD_ONLY
    assert parameters[3].default is None
    assert parameters[4].kind is inspect.Parameter.KEYWORD_ONLY
    assert parameters[4].default is False
    assert signature.return_annotation is dict


def test_compare_experiments_rejects_non_string_experiment_root() -> None:
    tool = _comparison_tool()

    for invalid_value in (None, Path("experiments"), 123):
        with pytest.raises(TypeError):
            tool(invalid_value)


def test_compare_experiments_rejects_blank_experiment_root() -> None:
    tool = _comparison_tool()

    for invalid_value in ("", " ", "\t\r\n"):
        with pytest.raises(ValueError):
            tool(invalid_value)


def test_compare_experiments_rejects_invalid_sort_by_type() -> None:
    tool = _comparison_tool()

    for invalid_value in (1, False, Path("best_r2")):
        with pytest.raises(TypeError):
            tool("experiments", sort_by=invalid_value)


def test_compare_experiments_rejects_blank_sort_by() -> None:
    tool = _comparison_tool()

    for invalid_value in ("", " ", "\t\r\n"):
        with pytest.raises(ValueError):
            tool("experiments", sort_by=invalid_value)


def test_compare_experiments_rejects_invalid_metrics_config_type() -> None:
    tool = _comparison_tool()

    for invalid_value in (0, False, Path("metrics.yaml")):
        with pytest.raises(TypeError):
            tool("experiments", metrics_config=invalid_value)


def test_compare_experiments_rejects_blank_metrics_config() -> None:
    tool = _comparison_tool()

    for invalid_value in ("", " ", "\t\r\n"):
        with pytest.raises(ValueError):
            tool("experiments", metrics_config=invalid_value)


def test_compare_experiments_requires_strict_boolean_descending() -> None:
    tool = _comparison_tool()

    for invalid_value in (0, 1, "true"):
        with pytest.raises(TypeError):
            tool("experiments", descending=invalid_value)


def test_compare_experiments_requires_strict_boolean_diagnostics() -> None:
    tool = _comparison_tool()

    for invalid_value in (0, 1, "false"):
        with pytest.raises(TypeError):
            tool("experiments", include_diagnostics=invalid_value)


def test_compare_experiments_preserves_valid_root_string(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    tool = _comparison_tool()
    calls = _patch_comparison_flow(monkeypatch)
    raw_root = " valid experiment root "

    tool(raw_root)

    assert calls["root_path"] == Path(raw_root)


def test_compare_experiments_preserves_valid_metrics_config_string(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    tool = _comparison_tool()
    calls = _patch_comparison_flow(monkeypatch)
    specs = _metric_specs()
    received_paths: list[Path] = []
    raw_config = " valid metrics config.yaml "

    def fake_read(config_path: Path) -> tuple[MetricSpec, ...]:
        received_paths.append(config_path)
        return specs

    monkeypatch.setattr(
        experiment_tools,
        "read_metric_specs_config",
        fake_read,
    )

    tool("experiments", metrics_config=raw_config)

    assert received_paths == [Path(raw_config)]
    assert calls["analyze_metric_specs"] is specs


def test_compare_experiments_returns_empty_default_payload(
    tmp_path: Path,
) -> None:
    tool = _comparison_tool()

    result = tool(str(tmp_path))

    assert result == _empty_payload()


def test_compare_experiments_runs_minimal_default_integration(
    tmp_path: Path,
) -> None:
    tool = _comparison_tool()
    root = tmp_path / "experiments"
    root.mkdir()
    experiment_dir = root / "experiment_a"
    _write_default_experiment(
        experiment_dir,
        r2=0.75,
        racc=0.9,
    )

    result = tool(str(root))

    assert isinstance(result, dict)
    assert result["experiment_counts"] == {
        "total": 1,
        "successful": 1,
        "failed": 0,
    }
    assert result["comparison_records"] == [
        {
            "experiment_name": "experiment_a",
            "experiment_dir": str(experiment_dir),
            "best_r2": 0.75,
            "best_r2_epoch": 0,
            "best_racc": 0.9,
            "best_racc_epoch": 0,
        }
    ]
    json.dumps(result, ensure_ascii=False, allow_nan=False)
    assert not (tmp_path / "outputs").exists()


def test_compare_experiments_discovers_only_complete_direct_children(
    tmp_path: Path,
) -> None:
    tool = _comparison_tool()
    root = tmp_path / "experiments"
    root.mkdir()
    experiment_a = root / "experiment_a"
    experiment_b = root / "experiment_b"
    _write_default_experiment(experiment_a, r2=0.6, racc=0.8)
    _write_default_experiment(experiment_b, r2=0.9, racc=0.95)
    _write_default_experiment(
        root / "group" / "nested",
        r2=1.0,
        racc=1.0,
    )
    incomplete = root / "incomplete"
    incomplete.mkdir()
    (incomplete / "hparams.yaml").write_text(
        "config: {}\n",
        encoding="utf-8",
    )

    result = tool(str(root))

    assert result["experiment_counts"]["total"] == 2
    assert [
        record["experiment_name"]
        for record in result["comparison_records"]
    ] == ["experiment_b", "experiment_a"]


def test_compare_experiments_passes_default_flow_arguments(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    tool = _comparison_tool()
    experiment_dirs = [Path("experiments/a"), Path("experiments/b")]
    batch_result = {
        "successful_experiments": [],
        "failed_experiments": [],
    }
    payload = _empty_payload()
    calls = _patch_comparison_flow(
        monkeypatch,
        experiment_dirs=experiment_dirs,
        batch_result=batch_result,
        payload=payload,
    )

    result = tool("experiments")

    assert calls == {
        "root_path": Path("experiments"),
        "experiment_dirs": experiment_dirs,
        "analyze_metric_specs": None,
        "batch_result": batch_result,
        "sort_by": "best_r2",
        "descending": True,
        "payload_metric_specs": None,
        "include_diagnostics": False,
    }
    assert result is payload


def test_compare_experiments_passes_ascending_sort_direction(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    tool = _comparison_tool()
    calls = _patch_comparison_flow(monkeypatch)

    tool("experiments", descending=False)

    assert calls["descending"] is False


def test_compare_experiments_passes_explicit_default_sort_field(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    tool = _comparison_tool()
    calls = _patch_comparison_flow(monkeypatch)

    tool("experiments", sort_by="best_racc")

    assert calls["sort_by"] == "best_racc"


def test_compare_experiments_preserves_invalid_default_sort_error(
    tmp_path: Path,
) -> None:
    tool = _comparison_tool()

    with pytest.raises(ValueError):
        tool(str(tmp_path), sort_by="accuracy")


def test_compare_experiments_loads_and_passes_dynamic_metric_specs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    tool = _comparison_tool()
    calls = _patch_comparison_flow(monkeypatch)
    specs = _metric_specs()
    config_path = Path("configs/metrics.yaml")
    read_paths: list[Path] = []

    def fake_read(path: Path) -> tuple[MetricSpec, ...]:
        read_paths.append(path)
        return specs

    monkeypatch.setattr(
        experiment_tools,
        "read_metric_specs_config",
        fake_read,
    )

    tool(
        "experiments",
        metrics_config=str(config_path),
    )

    assert read_paths == [config_path]
    assert calls["analyze_metric_specs"] is specs
    assert calls["payload_metric_specs"] is specs
    assert calls["sort_by"] == "accuracy"


def test_compare_experiments_passes_explicit_dynamic_sort_field(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    tool = _comparison_tool()
    calls = _patch_comparison_flow(monkeypatch)
    specs = _metric_specs()
    monkeypatch.setattr(
        experiment_tools,
        "read_metric_specs_config",
        lambda path: specs,
    )

    tool(
        "experiments",
        sort_by="validation_loss",
        metrics_config="metrics.yaml",
    )

    assert calls["sort_by"] == "validation_loss"


def test_compare_experiments_preserves_invalid_dynamic_sort_error(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    tool = _comparison_tool()
    specs = _metric_specs()
    monkeypatch.setattr(
        experiment_tools,
        "read_metric_specs_config",
        lambda path: specs,
    )

    with pytest.raises(ValueError):
        tool(
            str(tmp_path),
            sort_by="best_r2",
            metrics_config="metrics.yaml",
        )


def test_compare_experiments_propagates_metrics_config_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    tool = _comparison_tool()
    expected_error = ValueError("invalid metrics configuration")

    def fail(path: Path) -> tuple[MetricSpec, ...]:
        raise expected_error

    monkeypatch.setattr(
        experiment_tools,
        "read_metric_specs_config",
        fail,
    )

    with pytest.raises(ValueError) as error_info:
        tool("experiments", metrics_config="metrics.yaml")

    assert error_info.value is expected_error


def test_compare_experiments_preserves_partial_experiment_failures(
    tmp_path: Path,
) -> None:
    tool = _comparison_tool()
    root = tmp_path / "experiments"
    root.mkdir()
    _write_default_experiment(
        root / "valid",
        r2=0.8,
        racc=0.9,
    )
    broken = root / "broken"
    broken.mkdir()
    (broken / "hparams.yaml").write_text(
        "config: {}\n",
        encoding="utf-8",
    )
    (broken / "history.json").write_text(
        "{invalid json",
        encoding="utf-8",
    )

    result = tool(str(root))

    assert result["experiment_counts"] == {
        "total": 2,
        "successful": 1,
        "failed": 1,
    }
    assert result["comparison_records"][0]["experiment_name"] == "valid"
    assert result["failed_experiments"][0]["experiment_name"] == "broken"


def test_compare_experiments_preserves_all_failed_experiments(
    tmp_path: Path,
) -> None:
    tool = _comparison_tool()
    root = tmp_path / "experiments"
    root.mkdir()
    for name in ("broken_a", "broken_b"):
        experiment_dir = root / name
        experiment_dir.mkdir()
        (experiment_dir / "hparams.yaml").write_text(
            "config: {}\n",
            encoding="utf-8",
        )
        (experiment_dir / "history.json").write_text(
            "{invalid json",
            encoding="utf-8",
        )

    result = tool(str(root))

    assert result["experiment_counts"] == {
        "total": 2,
        "successful": 0,
        "failed": 2,
    }
    assert result["comparison_records"] == []
    assert [
        failure["experiment_name"]
        for failure in result["failed_experiments"]
    ] == ["broken_a", "broken_b"]


def test_compare_experiments_propagates_missing_root_error(
    tmp_path: Path,
) -> None:
    tool = _comparison_tool()

    with pytest.raises(FileNotFoundError):
        tool(str(tmp_path / "missing"))


def test_compare_experiments_propagates_non_directory_root_error(
    tmp_path: Path,
) -> None:
    tool = _comparison_tool()
    root_file = tmp_path / "root.txt"
    root_file.touch()

    with pytest.raises(NotADirectoryError):
        tool(str(root_file))


def test_compare_experiments_propagates_discovery_error_unchanged(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    tool = _comparison_tool()
    expected_error = OSError("discovery failed")

    def fail(path: Path) -> list[Path]:
        raise expected_error

    monkeypatch.setattr(
        experiment_tools,
        "find_experiment_dirs",
        fail,
        raising=False,
    )

    with pytest.raises(OSError) as error_info:
        tool("experiments")

    assert error_info.value is expected_error


def test_compare_experiments_propagates_payload_error_unchanged(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    tool = _comparison_tool()
    expected_error = KeyError("damaged comparison data")
    _patch_comparison_flow(monkeypatch)

    def fail(*args: object, **kwargs: object) -> dict:
        raise expected_error

    monkeypatch.setattr(
        experiment_tools,
        "build_comparison_payload",
        fail,
        raising=False,
    )

    with pytest.raises(KeyError) as error_info:
        tool("experiments")

    assert error_info.value is expected_error


@pytest.mark.parametrize("include_diagnostics", [False, True])
def test_compare_experiments_passes_diagnostics_option(
    include_diagnostics: bool,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    tool = _comparison_tool()
    calls = _patch_comparison_flow(monkeypatch)

    tool(
        "experiments",
        include_diagnostics=include_diagnostics,
    )

    assert calls["include_diagnostics"] is include_diagnostics


def test_compare_experiments_returns_existing_diagnostics_unchanged(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    tool = _comparison_tool()
    payload = _empty_payload()
    payload["diagnostics"] = {
        "facts": {"successful_experiments": 0},
        "diagnostics": [{"code": "no_successful_experiments"}],
        "recommendations": [{"code": "resolve_analysis_failures"}],
    }
    _patch_comparison_flow(monkeypatch, payload=payload)

    result = tool(
        "experiments",
        include_diagnostics=True,
    )

    assert result is payload
    assert result["diagnostics"] is payload["diagnostics"]


def test_compare_experiments_returns_original_json_safe_payload(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    tool = _comparison_tool()
    payload = _empty_payload()
    _patch_comparison_flow(monkeypatch, payload=payload)

    result = tool("experiments")

    assert result is payload
    assert isinstance(result, dict)
    assert "status" not in result
    assert "message" not in result
    assert "tool_name" not in result
    json.dumps(
        result,
        ensure_ascii=False,
        allow_nan=False,
    )


@pytest.mark.parametrize(
    "non_finite_value",
    [float("nan"), float("inf"), float("-inf")],
)
def test_compare_experiments_rejects_non_finite_payload_values(
    non_finite_value: float,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    tool = _comparison_tool()
    payload = _empty_payload()
    payload["non_finite"] = non_finite_value
    _patch_comparison_flow(monkeypatch, payload=payload)

    with pytest.raises(ValueError):
        tool("experiments")


def test_compare_experiments_rejects_non_json_payload_values(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    tool = _comparison_tool()
    payload = _empty_payload()
    payload["path"] = Path("not-json-native")
    _patch_comparison_flow(monkeypatch, payload=payload)

    with pytest.raises(TypeError):
        tool("experiments")


def test_compare_experiments_has_no_report_or_cli_side_effects(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    tool = _comparison_tool()
    _patch_comparison_flow(monkeypatch)

    def fail_if_called(*args: object, **kwargs: object) -> object:
        pytest.fail("comparison Tool must not call CLI or writer functions")

    forbidden_names = (
        "run_comparison_pipeline",
        "write_comparison_json",
        "write_comparison_markdown",
        "build_comparison_markdown",
        "parse_args",
        "main",
    )
    for name in forbidden_names:
        monkeypatch.setattr(
            experiment_tools,
            name,
            fail_if_called,
            raising=False,
        )
        monkeypatch.setattr(
            comparison_core,
            name,
            fail_if_called,
        )

    paths_before = sorted(tmp_path.rglob("*"))

    tool(str(tmp_path))

    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err == ""
    assert sorted(tmp_path.rglob("*")) == paths_before
    assert not (tmp_path / "outputs").exists()

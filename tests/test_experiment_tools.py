import inspect
import json
from pathlib import Path

import pytest

import tool_layer
import tool_layer.experiment_tools as experiment_tools
from metrics import MetricSpec
from tool_layer import analyze_experiment


def _create_experiment_dir(tmp_path: Path) -> Path:
    experiment_dir = tmp_path / "experiment"
    experiment_dir.mkdir()
    (experiment_dir / "hparams.yaml").touch()
    (experiment_dir / "history.json").touch()
    return experiment_dir


def _summary() -> dict:
    return {
        "configuration": {"batch_size": 8},
        "validation_metrics": {
            "r2": {"best_value": 0.75, "best_epoch": 3},
            "racc": {"best_value": 0.91, "best_epoch": 4},
        },
    }


def _patch_summary_builder(
    monkeypatch: pytest.MonkeyPatch,
    summary: dict,
) -> list[dict[str, object]]:
    calls: list[dict[str, object]] = []

    def fake_build_experiment_summary(**kwargs: object) -> dict:
        calls.append(kwargs)
        return summary

    monkeypatch.setattr(
        experiment_tools,
        "build_experiment_summary",
        fake_build_experiment_summary,
    )
    return calls


def test_analyze_experiment_has_expected_signature() -> None:
    signature = inspect.signature(analyze_experiment)

    assert str(signature) == (
        "(experiment_dir: str, *, metrics_config: str | None = None, "
        "include_diagnostics: bool = False) -> dict"
    )


def test_analyze_experiment_returns_dict_for_valid_directory(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    experiment_dir = _create_experiment_dir(tmp_path)
    expected = _summary()
    _patch_summary_builder(monkeypatch, expected)

    result = analyze_experiment(str(experiment_dir))

    assert isinstance(result, dict)
    assert result is expected


def test_analyze_experiment_result_is_strictly_json_serializable(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    experiment_dir = _create_experiment_dir(tmp_path)
    expected = _summary()
    _patch_summary_builder(monkeypatch, expected)

    result = analyze_experiment(str(experiment_dir))

    assert json.loads(
        json.dumps(
            result,
            ensure_ascii=False,
            allow_nan=False,
        )
    ) == expected


def test_analyze_experiment_runs_minimal_default_integration(
    tmp_path: Path,
) -> None:
    experiment_dir = tmp_path / "experiment"
    experiment_dir.mkdir()
    (experiment_dir / "hparams.yaml").write_text(
        "config:\n  batch_size: 8\n",
        encoding="utf-8",
    )
    (experiment_dir / "history.json").write_text(
        json.dumps(
            {
                "valid": {
                    "app": {
                        "r2": [[0, 0.25], [1, 0.5]],
                        "racc": [[0, 0.75], [1, 0.8]],
                    }
                }
            }
        ),
        encoding="utf-8",
    )

    result = analyze_experiment(str(experiment_dir))

    assert result["configuration"]["batch_size"] == 8
    assert result["validation_metrics"]["r2"]["best_value"] == 0.5
    assert result["validation_metrics"]["racc"]["best_value"] == 0.8


def test_analyze_experiment_defaults_diagnostics_to_false(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    experiment_dir = _create_experiment_dir(tmp_path)
    calls = _patch_summary_builder(monkeypatch, _summary())

    analyze_experiment(str(experiment_dir))

    assert calls[0]["include_diagnostics"] is False


def test_analyze_experiment_passes_enabled_diagnostics_to_summary(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    experiment_dir = _create_experiment_dir(tmp_path)
    calls = _patch_summary_builder(monkeypatch, _summary())

    analyze_experiment(
        str(experiment_dir),
        include_diagnostics=True,
    )

    assert calls[0]["include_diagnostics"] is True


def test_analyze_experiment_passes_resolved_paths_to_summary(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    experiment_dir = _create_experiment_dir(tmp_path)
    calls = _patch_summary_builder(monkeypatch, _summary())

    analyze_experiment(str(experiment_dir))

    assert calls[0]["config_path"] == experiment_dir / "hparams.yaml"
    assert calls[0]["history_path"] == experiment_dir / "history.json"


def test_analyze_experiment_does_not_read_metrics_config_by_default(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    experiment_dir = _create_experiment_dir(tmp_path)
    _patch_summary_builder(monkeypatch, _summary())

    def fail(path: Path) -> tuple[MetricSpec, ...]:
        raise AssertionError("metrics configuration must not be read")

    monkeypatch.setattr(
        experiment_tools,
        "read_metric_specs_config",
        fail,
    )

    analyze_experiment(str(experiment_dir))


def test_analyze_experiment_reads_provided_metrics_config(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    experiment_dir = _create_experiment_dir(tmp_path)
    metrics_path = tmp_path / "metrics.yaml"
    expected_specs: tuple[MetricSpec, ...] = ()
    received_paths: list[Path] = []
    _patch_summary_builder(monkeypatch, _summary())

    def fake_read(path: Path) -> tuple[MetricSpec, ...]:
        received_paths.append(path)
        return expected_specs

    monkeypatch.setattr(
        experiment_tools,
        "read_metric_specs_config",
        fake_read,
    )

    analyze_experiment(
        str(experiment_dir),
        metrics_config=str(metrics_path),
    )

    assert received_paths == [metrics_path]


def test_analyze_experiment_passes_loaded_metric_specs_to_summary(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    experiment_dir = _create_experiment_dir(tmp_path)
    spec = MetricSpec(
        name="accuracy",
        path=("valid", "accuracy"),
        direction="maximize",
        display_name="Accuracy",
    )
    expected_specs = (spec,)
    calls = _patch_summary_builder(monkeypatch, _summary())
    monkeypatch.setattr(
        experiment_tools,
        "read_metric_specs_config",
        lambda path: expected_specs,
    )

    analyze_experiment(
        str(experiment_dir),
        metrics_config=str(tmp_path / "metrics.yaml"),
    )

    assert calls[0]["metric_specs"] is expected_specs


@pytest.mark.parametrize("invalid_value", [None, 1, Path("experiment")])
def test_analyze_experiment_rejects_non_string_experiment_dir(
    invalid_value: object,
) -> None:
    with pytest.raises(
        TypeError,
        match="^experiment_dir must be a string$",
    ):
        analyze_experiment(invalid_value)  # type: ignore[arg-type]


@pytest.mark.parametrize("invalid_value", ["", " ", "\t\r\n"])
def test_analyze_experiment_rejects_empty_experiment_dir(
    invalid_value: str,
) -> None:
    with pytest.raises(
        ValueError,
        match="^experiment_dir must not be empty or whitespace$",
    ):
        analyze_experiment(invalid_value)


@pytest.mark.parametrize("invalid_value", [0, False, Path("metrics.yaml")])
def test_analyze_experiment_rejects_invalid_metrics_config_type(
    invalid_value: object,
) -> None:
    with pytest.raises(
        TypeError,
        match="^metrics_config must be a string or None$",
    ):
        analyze_experiment(
            "experiment",
            metrics_config=invalid_value,  # type: ignore[arg-type]
        )


@pytest.mark.parametrize("invalid_value", ["", " ", "\t\r\n"])
def test_analyze_experiment_rejects_empty_metrics_config(
    invalid_value: str,
) -> None:
    with pytest.raises(
        ValueError,
        match="^metrics_config must not be empty or whitespace$",
    ):
        analyze_experiment(
            "experiment",
            metrics_config=invalid_value,
        )


@pytest.mark.parametrize("invalid_value", [0, 1, "", "true", None])
def test_analyze_experiment_rejects_non_boolean_include_diagnostics(
    invalid_value: object,
) -> None:
    with pytest.raises(
        TypeError,
        match="^include_diagnostics must be a boolean$",
    ):
        analyze_experiment(
            "experiment",
            include_diagnostics=invalid_value,  # type: ignore[arg-type]
        )


def test_analyze_experiment_propagates_missing_directory_error(
    tmp_path: Path,
) -> None:
    with pytest.raises(FileNotFoundError):
        analyze_experiment(str(tmp_path / "missing"))


def test_analyze_experiment_propagates_missing_hparams_error(
    tmp_path: Path,
) -> None:
    experiment_dir = tmp_path / "experiment"
    experiment_dir.mkdir()
    (experiment_dir / "history.json").touch()

    with pytest.raises(FileNotFoundError):
        analyze_experiment(str(experiment_dir))


def test_analyze_experiment_propagates_missing_history_error(
    tmp_path: Path,
) -> None:
    experiment_dir = tmp_path / "experiment"
    experiment_dir.mkdir()
    (experiment_dir / "hparams.yaml").touch()

    with pytest.raises(FileNotFoundError):
        analyze_experiment(str(experiment_dir))


def test_analyze_experiment_propagates_summary_error_unchanged(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    experiment_dir = _create_experiment_dir(tmp_path)
    expected_error = ValueError("invalid metric history")

    def fail(**kwargs: object) -> dict:
        raise expected_error

    monkeypatch.setattr(
        experiment_tools,
        "build_experiment_summary",
        fail,
    )

    with pytest.raises(ValueError) as error_info:
        analyze_experiment(str(experiment_dir))

    assert error_info.value is expected_error


def test_analyze_experiment_does_not_create_output_files(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    experiment_dir = _create_experiment_dir(tmp_path)
    _patch_summary_builder(monkeypatch, _summary())
    paths_before = sorted(tmp_path.rglob("*"))

    analyze_experiment(str(experiment_dir))

    assert sorted(tmp_path.rglob("*")) == paths_before


def test_analyze_experiment_does_not_print(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    experiment_dir = _create_experiment_dir(tmp_path)
    _patch_summary_builder(monkeypatch, _summary())

    analyze_experiment(str(experiment_dir))

    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err == ""


def test_analyze_experiment_rejects_non_dict_summary(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    experiment_dir = _create_experiment_dir(tmp_path)
    monkeypatch.setattr(
        experiment_tools,
        "build_experiment_summary",
        lambda **kwargs: [],
    )

    with pytest.raises(
        TypeError,
        match="^build_experiment_summary must return a dict$",
    ):
        analyze_experiment(str(experiment_dir))


def test_analyze_experiment_propagates_json_type_error(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    experiment_dir = _create_experiment_dir(tmp_path)
    _patch_summary_builder(
        monkeypatch,
        {"non_json_value": Path("value")},
    )

    with pytest.raises(TypeError):
        analyze_experiment(str(experiment_dir))


def test_analyze_experiment_propagates_non_finite_json_value_error(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    experiment_dir = _create_experiment_dir(tmp_path)
    _patch_summary_builder(
        monkeypatch,
        {"non_finite_value": float("nan")},
    )

    with pytest.raises(ValueError):
        analyze_experiment(str(experiment_dir))


def test_tool_layer_exports_analyze_experiment() -> None:
    assert tool_layer.analyze_experiment is analyze_experiment

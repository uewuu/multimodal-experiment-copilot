from collections import UserList
from copy import deepcopy
from pathlib import Path

import pytest

import summarize_experiment
from metrics import MetricSpec
from summarize_experiment import build_experiment_summary


def _configuration() -> dict:
    return {
        "batch_size": 8,
        "num_workers": 4,
        "seed": 42,
        "sample_seed": 42,
        "n_epochs": 100,
        "feature_list": ["vision", "audio"],
        "attention_type": "linear",
        "use_modality_token_fusion": True,
        "use_behavior_state_token": True,
        "use_behavior_aware_cl": True,
        "use_regression_aware_cl": False,
        "use_trait_conditioned_modality_selection": True,
    }


def _default_history() -> dict:
    return {
        "valid": {
            "app": {
                "r2": [[0, 0.3], [1, 0.5], [2, 0.4]],
                "racc": [[0, 0.8], [1, 0.9], [2, 0.85]],
            },
        },
    }


def _dynamic_history() -> dict:
    return {
        "validation": {
            "metrics": {
                "accuracy": [[0, 0.70], [1, 0.80], [2, 0.75]],
                "loss": [[0, 0.50], [1, 0.30], [2, 0.40]],
            },
        },
    }


def _metric_spec(
    name: str = "accuracy",
    path: tuple[str, ...] = ("validation", "metrics", "accuracy"),
    direction: str = "maximize",
    display_name: str = "Accuracy",
) -> MetricSpec:
    return MetricSpec(
        name=name,
        path=path,
        direction=direction,
        display_name=display_name,
    )


def _patch_inputs(
    monkeypatch: pytest.MonkeyPatch,
    history: dict,
    configuration: dict | None = None,
) -> None:
    if configuration is None:
        configuration = _configuration()
    monkeypatch.setattr(
        summarize_experiment,
        "read_config",
        lambda path: configuration,
    )
    monkeypatch.setattr(
        summarize_experiment,
        "read_history",
        lambda path: history,
    )


def _build_summary(
    metric_specs=None,
) -> dict:
    return build_experiment_summary(
        config_path=Path("memory/hparams.yaml"),
        history_path=Path("memory/history.json"),
        metric_specs=metric_specs,
    )


def test_build_experiment_summary_preserves_default_metrics(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_inputs(monkeypatch, _default_history())

    summary = _build_summary()

    assert list(summary["validation_metrics"]) == ["r2", "racc"]
    assert summary["validation_metrics"] == {
        "r2": {
            "metric_name": "r2",
            "record_count": 3,
            "first_epoch": 0,
            "first_value": 0.3,
            "last_epoch": 2,
            "last_value": 0.4,
            "best_epoch": 1,
            "best_value": 0.5,
        },
        "racc": {
            "metric_name": "racc",
            "record_count": 3,
            "first_epoch": 0,
            "first_value": 0.8,
            "last_epoch": 2,
            "last_value": 0.85,
            "best_epoch": 1,
            "best_value": 0.9,
        },
    }


def test_build_experiment_summary_preserves_default_metric_field_order(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_inputs(monkeypatch, _default_history())
    expected_order = [
        "metric_name",
        "record_count",
        "first_epoch",
        "first_value",
        "last_epoch",
        "last_value",
        "best_epoch",
        "best_value",
    ]

    summary = _build_summary()

    assert list(summary["validation_metrics"]["r2"]) == expected_order
    assert list(summary["validation_metrics"]["racc"]) == expected_order


def test_build_experiment_summary_preserves_top_level_order(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_inputs(monkeypatch, _default_history())

    summary = _build_summary()

    assert list(summary) == ["configuration", "validation_metrics"]


def test_build_experiment_summary_preserves_configuration_structure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    configuration = _configuration()
    _patch_inputs(monkeypatch, _default_history(), configuration)

    summary = _build_summary()

    assert list(summary["configuration"]) == list(configuration)
    assert summary["configuration"] == configuration


def test_build_experiment_summary_default_mode_uses_legacy_validation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    history = {
        "valid": {
            "app": {
                "r2": [[True, False]],
                "racc": [[False, True]],
            },
        },
    }
    _patch_inputs(monkeypatch, history)

    summary = _build_summary()

    assert summary["validation_metrics"]["r2"]["first_epoch"] is True
    assert summary["validation_metrics"]["r2"]["best_value"] is False
    assert summary["validation_metrics"]["racc"]["first_epoch"] is False
    assert summary["validation_metrics"]["racc"]["best_value"] is True


def test_build_experiment_summary_explicit_none_uses_default_mode(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_inputs(monkeypatch, _default_history())

    summary = _build_summary(metric_specs=None)

    assert list(summary["validation_metrics"]) == ["r2", "racc"]


def test_build_experiment_summary_keeps_two_positional_paths_compatible(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    received: dict[str, Path] = {}

    def fake_read_config(path: Path) -> dict:
        received["config_path"] = path
        return _configuration()

    def fake_read_history(path: Path) -> dict:
        received["history_path"] = path
        return _default_history()

    monkeypatch.setattr(summarize_experiment, "read_config", fake_read_config)
    monkeypatch.setattr(summarize_experiment, "read_history", fake_read_history)
    config_path = Path("memory/config.yaml")
    history_path = Path("memory/history.json")

    build_experiment_summary(config_path, history_path)

    assert received == {
        "config_path": config_path,
        "history_path": history_path,
    }


def test_build_experiment_summary_supports_single_maximize_metric(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_inputs(monkeypatch, _dynamic_history())

    summary = _build_summary([_metric_spec()])

    assert summary["validation_metrics"]["accuracy"]["best_epoch"] == 1
    assert summary["validation_metrics"]["accuracy"]["best_value"] == 0.80


def test_build_experiment_summary_supports_single_minimize_metric(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_inputs(monkeypatch, _dynamic_history())
    spec = _metric_spec(
        name="loss",
        path=("validation", "metrics", "loss"),
        direction="minimize",
        display_name="Loss",
    )

    summary = _build_summary([spec])

    assert summary["validation_metrics"]["loss"]["best_epoch"] == 1
    assert summary["validation_metrics"]["loss"]["best_value"] == 0.30


def test_build_experiment_summary_supports_multiple_metrics(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_inputs(monkeypatch, _dynamic_history())
    loss_spec = _metric_spec(
        name="loss",
        path=("validation", "metrics", "loss"),
        direction="minimize",
        display_name="Loss",
    )

    summary = _build_summary([_metric_spec(), loss_spec])

    assert set(summary["validation_metrics"]) == {"accuracy", "loss"}


def test_build_experiment_summary_preserves_metric_spec_order(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_inputs(monkeypatch, _dynamic_history())
    loss_spec = _metric_spec(
        name="loss",
        path=("validation", "metrics", "loss"),
        direction="minimize",
        display_name="Loss",
    )

    summary = _build_summary([loss_spec, _metric_spec()])

    assert list(summary["validation_metrics"]) == ["loss", "accuracy"]


def test_build_experiment_summary_uses_custom_metric_name_as_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_inputs(monkeypatch, _dynamic_history())
    spec = _metric_spec(name="validation_accuracy")

    summary = _build_summary([spec])

    assert list(summary["validation_metrics"]) == ["validation_accuracy"]


def test_build_experiment_summary_sets_metric_name_from_spec(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_inputs(monkeypatch, _dynamic_history())
    spec = _metric_spec(name="validation_accuracy")

    summary = _build_summary([spec])

    assert (
        summary["validation_metrics"]["validation_accuracy"]["metric_name"]
        == "validation_accuracy"
    )


def test_build_experiment_summary_supports_nested_metric_path(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_inputs(monkeypatch, _dynamic_history())

    summary = _build_summary([_metric_spec()])

    assert summary["validation_metrics"]["accuracy"]["record_count"] == 3


def test_build_experiment_summary_returns_expected_dynamic_structure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_inputs(monkeypatch, _dynamic_history())

    metric = _build_summary([_metric_spec()])["validation_metrics"]["accuracy"]

    assert metric == {
        "metric_name": "accuracy",
        "record_count": 3,
        "first_epoch": 0,
        "first_value": 0.70,
        "last_epoch": 2,
        "last_value": 0.75,
        "best_epoch": 1,
        "best_value": 0.80,
    }
    assert list(metric) == [
        "metric_name",
        "record_count",
        "first_epoch",
        "first_value",
        "last_epoch",
        "last_value",
        "best_epoch",
        "best_value",
    ]


def test_build_experiment_summary_returns_empty_metrics_for_empty_list(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_inputs(monkeypatch, _dynamic_history())

    summary = _build_summary([])

    assert summary["validation_metrics"] == {}


def test_build_experiment_summary_returns_empty_metrics_for_empty_tuple(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_inputs(monkeypatch, _dynamic_history())

    summary = _build_summary(())

    assert summary["validation_metrics"] == {}


def test_build_experiment_summary_accepts_metric_spec_list(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_inputs(monkeypatch, _dynamic_history())

    summary = _build_summary([_metric_spec()])

    assert "accuracy" in summary["validation_metrics"]


def test_build_experiment_summary_accepts_metric_spec_tuple(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_inputs(monkeypatch, _dynamic_history())

    summary = _build_summary((_metric_spec(),))

    assert "accuracy" in summary["validation_metrics"]


def test_build_experiment_summary_accepts_other_sequence(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_inputs(monkeypatch, _dynamic_history())

    summary = _build_summary(UserList([_metric_spec()]))

    assert "accuracy" in summary["validation_metrics"]


def test_build_experiment_summary_preserves_first_maximum_on_tie(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    history = {"metrics": {"accuracy": [[0, 0.8], [1, 0.8]]}}
    _patch_inputs(monkeypatch, history)
    spec = _metric_spec(path=("metrics", "accuracy"))

    summary = _build_summary([spec])

    assert summary["validation_metrics"]["accuracy"]["best_epoch"] == 0


def test_build_experiment_summary_preserves_first_minimum_on_tie(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    history = {"metrics": {"loss": [[0, 0.3], [1, 0.3]]}}
    _patch_inputs(monkeypatch, history)
    spec = _metric_spec(
        name="loss",
        path=("metrics", "loss"),
        direction="minimize",
        display_name="Loss",
    )

    summary = _build_summary([spec])

    assert summary["validation_metrics"]["loss"]["best_epoch"] == 0


def test_build_experiment_summary_does_not_modify_metric_specs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_inputs(monkeypatch, _dynamic_history())
    metric_specs = [_metric_spec()]
    original_metric_specs = deepcopy(metric_specs)

    _build_summary(metric_specs)

    assert metric_specs == original_metric_specs


def test_build_experiment_summary_does_not_modify_history(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    history = _dynamic_history()
    original_history = deepcopy(history)
    _patch_inputs(monkeypatch, history)

    _build_summary([_metric_spec()])

    assert history == original_history


def test_build_experiment_summary_does_not_include_metric_metadata(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_inputs(monkeypatch, _dynamic_history())

    metric = _build_summary([_metric_spec()])["validation_metrics"]["accuracy"]

    assert not {"display_name", "direction", "precision", "path"} & set(metric)


@pytest.mark.parametrize(
    "metric_specs",
    [
        "metric_specs",
        b"metric_specs",
        bytearray(b"metric_specs"),
        123,
        {"accuracy": _metric_spec()},
        (spec for spec in [_metric_spec()]),
    ],
    ids=["string", "bytes", "bytearray", "integer", "mapping", "generator"],
)
def test_build_experiment_summary_rejects_invalid_metric_specs_container(
    monkeypatch: pytest.MonkeyPatch,
    metric_specs,
) -> None:
    _patch_inputs(monkeypatch, _dynamic_history())

    with pytest.raises(TypeError) as error_info:
        _build_summary(metric_specs)

    assert str(error_info.value) == "metric_specs must be a sequence"


@pytest.mark.parametrize(
    ("metric_specs", "index"),
    [
        ([{"name": "accuracy"}], 0),
        ([_metric_spec(), "accuracy"], 1),
        ([None], 0),
    ],
    ids=["dict-first", "string-second", "none-first"],
)
def test_build_experiment_summary_rejects_invalid_metric_spec_member(
    monkeypatch: pytest.MonkeyPatch,
    metric_specs,
    index: int,
) -> None:
    _patch_inputs(monkeypatch, _dynamic_history())

    with pytest.raises(TypeError) as error_info:
        _build_summary(metric_specs)

    assert str(error_info.value) == (
        f"metric_specs[{index}] must be a MetricSpec"
    )


def test_build_experiment_summary_rejects_duplicate_metric_name(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_inputs(monkeypatch, _dynamic_history())
    first = _metric_spec()
    second = _metric_spec(path=("validation", "metrics", "loss"))

    with pytest.raises(ValueError) as error_info:
        _build_summary([first, second])

    assert str(error_info.value) == (
        "duplicate metric name 'accuracy' at metric_specs[1]; "
        "first defined at metric_specs[0]"
    )


def test_build_experiment_summary_propagates_missing_top_level_path(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_inputs(monkeypatch, {})

    with pytest.raises(KeyError) as error_info:
        _build_summary([_metric_spec()])

    assert error_info.value.args[0] == "missing key at path 'validation'"


def test_build_experiment_summary_propagates_missing_nested_path(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_inputs(monkeypatch, {"validation": {"metrics": {}}})

    with pytest.raises(KeyError) as error_info:
        _build_summary([_metric_spec()])

    assert error_info.value.args[0] == (
        "missing key at path 'validation.metrics.accuracy'"
    )


def test_build_experiment_summary_propagates_non_mapping_intermediate_value(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_inputs(monkeypatch, {"validation": {"metrics": 0.8}})

    with pytest.raises(TypeError) as error_info:
        _build_summary([_metric_spec()])

    assert str(error_info.value) == (
        "value at path 'validation.metrics' must be a mapping"
    )


@pytest.mark.parametrize(
    ("records", "error_type", "message"),
    [
        ("records", TypeError, "records must be a sequence"),
        ([], ValueError, "records must not be empty"),
        ([[0]], ValueError, "records[0] must contain exactly two items"),
        (
            [["0", 0.8]],
            TypeError,
            "records[0][0] must be an integer",
        ),
        (
            [[0, "0.8"]],
            TypeError,
            "records[0][1] must be a number",
        ),
    ],
    ids=["string", "empty", "record-length", "epoch-type", "value-type"],
)
def test_build_experiment_summary_propagates_metric_history_errors(
    monkeypatch: pytest.MonkeyPatch,
    records,
    error_type,
    message: str,
) -> None:
    history = {"validation": {"metrics": {"accuracy": records}}}
    _patch_inputs(monkeypatch, history)

    with pytest.raises(error_type) as error_info:
        _build_summary([_metric_spec()])

    assert str(error_info.value) == message

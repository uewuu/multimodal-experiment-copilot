import inspect
import json
from collections import UserList
from copy import deepcopy
from pathlib import Path

import pytest

import summarize_experiment
from diagnostics import Diagnostic
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
    **kwargs,
) -> dict:
    return build_experiment_summary(
        config_path=Path("memory/hparams.yaml"),
        history_path=Path("memory/history.json"),
        metric_specs=metric_specs,
        **kwargs,
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


DIAGNOSTIC_FACT_KEYS = [
    "record_count",
    "first_epoch",
    "first_value",
    "last_epoch",
    "last_value",
    "best_epoch",
    "best_value",
    "best_record_index",
    "best_progress_ratio",
    "best_at_first_record",
    "best_at_last_record",
    "duplicate_epochs",
    "non_monotonic_epoch_transitions",
    "improvement_from_first",
    "regression_from_best",
    "recent_window_requested",
    "recent_window_size",
    "recent_transition_count",
    "recent_improving_steps",
    "recent_degrading_steps",
    "recent_flat_steps",
    "recent_net_change",
    "recent_trend",
]


def _loss_spec() -> MetricSpec:
    return _metric_spec(
        name="loss",
        path=("validation", "metrics", "loss"),
        direction="minimize",
        display_name="Loss",
    )


def _diagnostic_codes(summary: dict, metric_name: str) -> list[str]:
    return [
        item["code"]
        for item in summary["diagnostics"]["metrics"][metric_name][
            "diagnostics"
        ]
    ]


def test_build_experiment_summary_signature_adds_keyword_only_diagnostics():
    signature = inspect.signature(build_experiment_summary)
    include_parameter = signature.parameters["include_diagnostics"]
    window_parameter = signature.parameters["diagnostic_recent_window"]

    assert include_parameter.kind is inspect.Parameter.KEYWORD_ONLY
    assert window_parameter.kind is inspect.Parameter.KEYWORD_ONLY
    assert include_parameter.default is False
    assert window_parameter.default == 5


def test_build_experiment_summary_preserves_existing_parameter_order():
    parameters = list(inspect.signature(build_experiment_summary).parameters)

    assert parameters == [
        "config_path",
        "history_path",
        "metric_specs",
        "include_diagnostics",
        "diagnostic_recent_window",
    ]


def test_omitted_diagnostics_preserves_exact_default_output(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_inputs(monkeypatch, _default_history())

    implicit = _build_summary()
    explicit = _build_summary(include_diagnostics=False)

    assert implicit == explicit
    assert list(implicit) == ["configuration", "validation_metrics"]


def test_explicit_false_preserves_exact_dynamic_output(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_inputs(monkeypatch, _dynamic_history())
    specs = [_metric_spec(), _loss_spec()]

    implicit = _build_summary(specs)
    explicit = _build_summary(specs, include_diagnostics=False)

    assert implicit == explicit
    assert list(explicit) == ["configuration", "validation_metrics"]


def test_disabled_diagnostics_does_not_call_diagnostic_helpers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_inputs(monkeypatch, _default_history())

    def fail(*args, **kwargs):
        raise AssertionError("diagnostic helper must not be called")

    monkeypatch.setattr(summarize_experiment, "build_metric_facts", fail)
    monkeypatch.setattr(summarize_experiment, "build_metric_diagnostics", fail)
    monkeypatch.setattr(summarize_experiment, "build_recommendations", fail)
    monkeypatch.setattr(summarize_experiment, "diagnostic_to_dict", fail)
    monkeypatch.setattr(summarize_experiment, "recommendation_to_dict", fail)

    summary = _build_summary(include_diagnostics=False)

    assert "diagnostics" not in summary


@pytest.mark.parametrize("unused_window", [True, 1, 0, -1, "invalid", None])
def test_disabled_diagnostics_ignores_diagnostic_recent_window(
    monkeypatch: pytest.MonkeyPatch,
    unused_window,
) -> None:
    _patch_inputs(monkeypatch, _default_history())

    summary = _build_summary(
        include_diagnostics=False,
        diagnostic_recent_window=unused_window,
    )

    assert list(summary) == ["configuration", "validation_metrics"]


@pytest.mark.parametrize(
    "invalid_include_diagnostics",
    [0, 1, "true", None, [], {}],
    ids=["zero", "one", "string", "none", "list", "mapping"],
)
def test_build_experiment_summary_rejects_non_boolean_include_diagnostics(
    monkeypatch: pytest.MonkeyPatch,
    invalid_include_diagnostics,
) -> None:
    _patch_inputs(monkeypatch, _default_history())

    with pytest.raises(
        TypeError,
        match="^include_diagnostics must be a boolean$",
    ):
        _build_summary(include_diagnostics=invalid_include_diagnostics)


def test_include_diagnostics_validation_happens_before_file_reads(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = {"config": 0, "history": 0}

    def fake_read_config(path: Path) -> dict:
        calls["config"] += 1
        return _configuration()

    def fake_read_history(path: Path) -> dict:
        calls["history"] += 1
        return _default_history()

    monkeypatch.setattr(summarize_experiment, "read_config", fake_read_config)
    monkeypatch.setattr(summarize_experiment, "read_history", fake_read_history)

    with pytest.raises(TypeError):
        _build_summary(include_diagnostics=1)

    assert calls == {"config": 0, "history": 0}


def test_default_diagnostics_adds_exact_top_level_schema(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_inputs(monkeypatch, _default_history())

    summary = _build_summary(include_diagnostics=True)

    assert list(summary) == [
        "configuration",
        "validation_metrics",
        "diagnostics",
    ]
    assert list(summary["diagnostics"]) == ["metrics"]
    assert list(summary["diagnostics"]["metrics"]) == ["r2", "racc"]


def test_default_diagnostics_metric_entries_have_stable_schema(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_inputs(monkeypatch, _default_history())

    metrics = _build_summary(include_diagnostics=True)["diagnostics"]["metrics"]

    assert list(metrics["r2"]) == [
        "facts",
        "diagnostics",
        "recommendations",
    ]
    assert list(metrics["racc"]) == [
        "facts",
        "diagnostics",
        "recommendations",
    ]
    assert list(metrics["r2"]["facts"]) == DIAGNOSTIC_FACT_KEYS
    assert list(metrics["racc"]["facts"]) == DIAGNOSTIC_FACT_KEYS


def test_default_diagnostics_are_serialized_plain_objects(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_inputs(monkeypatch, _default_history())

    metrics = _build_summary(include_diagnostics=True)["diagnostics"]["metrics"]

    for metric in metrics.values():
        assert type(metric) is dict
        assert type(metric["facts"]) is dict
        assert type(metric["diagnostics"]) is list
        assert type(metric["recommendations"]) is list
        for diagnostic in metric["diagnostics"]:
            assert type(diagnostic) is dict
            assert list(diagnostic) == [
                "code",
                "severity",
                "message",
                "evidence",
            ]
            assert type(diagnostic["evidence"]) is dict
        for recommendation in metric["recommendations"]:
            assert type(recommendation) is dict
            assert list(recommendation) == [
                "code",
                "message",
                "diagnostic_codes",
            ]
            assert type(recommendation["diagnostic_codes"]) is list


def test_default_diagnostics_summary_is_json_serializable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_inputs(monkeypatch, _default_history())

    summary = _build_summary(include_diagnostics=True)

    assert json.loads(json.dumps(summary, allow_nan=False)) == summary


def test_diagnostics_does_not_change_default_validation_metrics(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_inputs(monkeypatch, _default_history())

    without_diagnostics = _build_summary()
    with_diagnostics = _build_summary(include_diagnostics=True)

    assert (
        with_diagnostics["validation_metrics"]
        == without_diagnostics["validation_metrics"]
    )
    assert not {
        "facts",
        "diagnostics",
    } & set(with_diagnostics["validation_metrics"]["r2"])


def test_enabled_summary_contains_recommendations_without_control_metadata(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_inputs(monkeypatch, _default_history())

    summary = _build_summary(include_diagnostics=True)
    serialized = json.dumps(summary)

    assert "recommendations" in serialized
    assert "include_diagnostics" not in summary
    assert "diagnostic_recent_window" not in summary


def test_default_diagnostics_use_raw_records_and_maximize_direction(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    history = _default_history()
    _patch_inputs(monkeypatch, history)
    received: list[tuple[object, str, int]] = []

    def fake_build_metric_facts(records, direction, *, recent_window):
        received.append((records, direction, recent_window))
        return {"record_count": len(records)}

    monkeypatch.setattr(
        summarize_experiment,
        "build_metric_facts",
        fake_build_metric_facts,
    )
    monkeypatch.setattr(
        summarize_experiment,
        "build_metric_diagnostics",
        lambda facts: (),
    )

    _build_summary(include_diagnostics=True)

    assert received == [
        (history["valid"]["app"]["r2"], "maximize", 5),
        (history["valid"]["app"]["racc"], "maximize", 5),
    ]
    assert received[0][0] is history["valid"]["app"]["r2"]
    assert received[1][0] is history["valid"]["app"]["racc"]


def test_default_diagnostics_read_each_input_file_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = {"config": 0, "history": 0}

    def fake_read_config(path: Path) -> dict:
        calls["config"] += 1
        return _configuration()

    def fake_read_history(path: Path) -> dict:
        calls["history"] += 1
        return _default_history()

    monkeypatch.setattr(summarize_experiment, "read_config", fake_read_config)
    monkeypatch.setattr(summarize_experiment, "read_history", fake_read_history)

    _build_summary(include_diagnostics=True)

    assert calls == {"config": 1, "history": 1}


def test_default_diagnostics_call_each_diagnostic_builder_twice(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_inputs(monkeypatch, _default_history())
    calls = {"facts": 0, "diagnostics": 0}

    def fake_build_metric_facts(records, direction, *, recent_window):
        calls["facts"] += 1
        return {"record_count": len(records)}

    def fake_build_metric_diagnostics(facts):
        calls["diagnostics"] += 1
        return ()

    monkeypatch.setattr(
        summarize_experiment,
        "build_metric_facts",
        fake_build_metric_facts,
    )
    monkeypatch.setattr(
        summarize_experiment,
        "build_metric_diagnostics",
        fake_build_metric_diagnostics,
    )

    _build_summary(include_diagnostics=True)

    assert calls == {"facts": 2, "diagnostics": 2}


def test_diagnostic_to_dict_called_for_every_generated_diagnostic(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_inputs(monkeypatch, _default_history())
    first = Diagnostic("synthetic_one", "info", "First.", {})
    second = Diagnostic("synthetic_two", "warning", "Second.", {})
    received: list[object] = []

    monkeypatch.setattr(
        summarize_experiment,
        "build_metric_facts",
        lambda records, direction, *, recent_window: {},
    )
    monkeypatch.setattr(
        summarize_experiment,
        "build_metric_diagnostics",
        lambda facts: (first, second),
    )

    def fake_diagnostic_to_dict(item):
        received.append(item)
        return {"serialized": True}

    monkeypatch.setattr(
        summarize_experiment,
        "diagnostic_to_dict",
        fake_diagnostic_to_dict,
    )

    summary = _build_summary(include_diagnostics=True)

    assert received == [first, second, first, second]
    assert summary["diagnostics"]["metrics"]["r2"]["diagnostics"] == [
        {"serialized": True},
        {"serialized": True},
    ]


def test_dynamic_diagnostics_preserves_metric_spec_order(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_inputs(monkeypatch, _dynamic_history())
    specs = (_loss_spec(), _metric_spec())

    summary = _build_summary(specs, include_diagnostics=True)

    assert list(summary["validation_metrics"]) == ["loss", "accuracy"]
    assert list(summary["diagnostics"]["metrics"]) == ["loss", "accuracy"]


def test_dynamic_diagnostics_preserves_validation_metrics(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_inputs(monkeypatch, _dynamic_history())
    specs = (_metric_spec(), _loss_spec())

    without_diagnostics = _build_summary(specs)
    with_diagnostics = _build_summary(specs, include_diagnostics=True)

    assert (
        with_diagnostics["validation_metrics"]
        == without_diagnostics["validation_metrics"]
    )


def test_dynamic_minimize_diagnostics_use_direction_semantics(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    history = {
        "validation": {
            "metrics": {
                "loss": [[0, 0.8], [1, 0.6], [2, 0.4]],
            },
        },
    }
    _patch_inputs(monkeypatch, history)

    summary = _build_summary([_loss_spec()], include_diagnostics=True)
    metric = summary["diagnostics"]["metrics"]["loss"]

    assert metric["facts"]["improvement_from_first"] == pytest.approx(0.4)
    assert metric["facts"]["regression_from_best"] == pytest.approx(0.0)
    assert metric["facts"]["recent_trend"] == "improving"
    assert "best_at_last_record" in _diagnostic_codes(summary, "loss")
    assert "recent_improvement" in _diagnostic_codes(summary, "loss")
    assert "post_best_regression" not in _diagnostic_codes(summary, "loss")


def test_dynamic_path_is_extracted_once_and_records_identity_is_reused(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    history = _dynamic_history()
    _patch_inputs(monkeypatch, history)
    records = history["validation"]["metrics"]["accuracy"]
    calls = {"path": 0, "evaluation": 0, "facts": 0, "diagnostics": 0}
    identities: dict[str, object] = {}

    def fake_get_value_at_path(data, path):
        calls["path"] += 1
        return records

    def fake_evaluate_metric_history(received_records, direction):
        calls["evaluation"] += 1
        identities["evaluation"] = received_records
        return {
            "record_count": 3,
            "first_epoch": 0,
            "first_value": 0.7,
            "last_epoch": 2,
            "last_value": 0.75,
            "best_epoch": 1,
            "best_value": 0.8,
        }

    def fake_build_metric_facts(
        received_records,
        direction,
        *,
        recent_window,
    ):
        calls["facts"] += 1
        identities["facts"] = received_records
        identities["direction"] = direction
        identities["window"] = recent_window
        return {"record_count": 3}

    def fake_build_metric_diagnostics(facts):
        calls["diagnostics"] += 1
        return ()

    monkeypatch.setattr(
        summarize_experiment,
        "get_value_at_path",
        fake_get_value_at_path,
    )
    monkeypatch.setattr(
        summarize_experiment,
        "evaluate_metric_history",
        fake_evaluate_metric_history,
    )
    monkeypatch.setattr(
        summarize_experiment,
        "build_metric_facts",
        fake_build_metric_facts,
    )
    monkeypatch.setattr(
        summarize_experiment,
        "build_metric_diagnostics",
        fake_build_metric_diagnostics,
    )

    _build_summary(
        [_metric_spec()],
        include_diagnostics=True,
        diagnostic_recent_window=2,
    )

    assert calls == {
        "path": 1,
        "evaluation": 1,
        "facts": 1,
        "diagnostics": 1,
    }
    assert identities["evaluation"] is records
    assert identities["facts"] is records
    assert identities["direction"] == "maximize"
    assert identities["window"] == 2


def test_dynamic_multiple_metrics_have_exact_call_counts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_inputs(monkeypatch, _dynamic_history())
    calls = {"path": 0, "evaluation": 0, "facts": 0, "diagnostics": 0}
    original_get = summarize_experiment.get_value_at_path
    original_evaluate = summarize_experiment.evaluate_metric_history
    original_facts = summarize_experiment.build_metric_facts
    original_diagnostics = summarize_experiment.build_metric_diagnostics

    def counted_get(*args, **kwargs):
        calls["path"] += 1
        return original_get(*args, **kwargs)

    def counted_evaluate(*args, **kwargs):
        calls["evaluation"] += 1
        return original_evaluate(*args, **kwargs)

    def counted_facts(*args, **kwargs):
        calls["facts"] += 1
        return original_facts(*args, **kwargs)

    def counted_diagnostics(*args, **kwargs):
        calls["diagnostics"] += 1
        return original_diagnostics(*args, **kwargs)

    monkeypatch.setattr(summarize_experiment, "get_value_at_path", counted_get)
    monkeypatch.setattr(
        summarize_experiment,
        "evaluate_metric_history",
        counted_evaluate,
    )
    monkeypatch.setattr(summarize_experiment, "build_metric_facts", counted_facts)
    monkeypatch.setattr(
        summarize_experiment,
        "build_metric_diagnostics",
        counted_diagnostics,
    )

    _build_summary(
        [_metric_spec(), _loss_spec()],
        include_diagnostics=True,
    )

    assert calls == {
        "path": 2,
        "evaluation": 2,
        "facts": 2,
        "diagnostics": 2,
    }


def test_dynamic_diagnostics_omit_metric_spec_metadata(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_inputs(monkeypatch, _dynamic_history())

    entry = _build_summary(
        [_metric_spec()],
        include_diagnostics=True,
    )["diagnostics"]["metrics"]["accuracy"]
    serialized = json.dumps(entry)

    for field in ("direction", "display_name", "path", "precision", "metric_name"):
        assert f'"{field}"' not in serialized


def test_dynamic_diagnostics_summary_is_json_serializable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_inputs(monkeypatch, _dynamic_history())

    summary = _build_summary(
        [_metric_spec(), _loss_spec()],
        include_diagnostics=True,
    )

    assert json.loads(json.dumps(summary, allow_nan=False)) == summary


def test_dynamic_diagnostics_does_not_modify_specs_or_history(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    history = _dynamic_history()
    original_history = deepcopy(history)
    specs = (_metric_spec(), _loss_spec())
    original_specs = deepcopy(specs)
    _patch_inputs(monkeypatch, history)

    _build_summary(specs, include_diagnostics=True)

    assert history == original_history
    assert specs == original_specs
    assert specs[0].path == original_specs[0].path


@pytest.mark.parametrize("recent_window", [5, 2, 20])
def test_diagnostic_recent_window_is_preserved_in_each_metric_facts(
    monkeypatch: pytest.MonkeyPatch,
    recent_window: int,
) -> None:
    _patch_inputs(monkeypatch, _default_history())

    metrics = _build_summary(
        include_diagnostics=True,
        diagnostic_recent_window=recent_window,
    )["diagnostics"]["metrics"]

    assert metrics["r2"]["facts"]["recent_window_requested"] == recent_window
    assert (
        metrics["racc"]["facts"]["recent_window_requested"]
        == recent_window
    )


@pytest.mark.parametrize(
    ("invalid_window", "error_type", "message"),
    [
        (True, TypeError, "recent_window must be an integer"),
        (1, ValueError, "recent_window must be at least 2"),
    ],
)
def test_enabled_diagnostics_propagates_recent_window_errors(
    monkeypatch: pytest.MonkeyPatch,
    invalid_window,
    error_type,
    message: str,
) -> None:
    _patch_inputs(monkeypatch, _default_history())

    with pytest.raises(error_type, match=f"^{message}$"):
        _build_summary(
            include_diagnostics=True,
            diagnostic_recent_window=invalid_window,
        )


def test_maximize_regression_real_integration(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    history = {
        "validation": {
            "metrics": {
                "accuracy": [[0, 0.5], [1, 0.8], [2, 0.7]],
            },
        },
    }
    _patch_inputs(monkeypatch, history)

    summary = _build_summary([_metric_spec()], include_diagnostics=True)
    facts = summary["diagnostics"]["metrics"]["accuracy"]["facts"]
    codes = _diagnostic_codes(summary, "accuracy")

    assert facts["regression_from_best"] == pytest.approx(0.1)
    assert facts["best_record_index"] == 1
    assert "post_best_regression" in codes
    assert "recent_mixed" in codes


def test_duplicate_and_non_monotonic_real_integration(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    history = {
        "validation": {
            "metrics": {
                "accuracy": [
                    [0, 0.4],
                    [2, 0.5],
                    [2, 0.6],
                    [1, 0.7],
                ],
            },
        },
    }
    _patch_inputs(monkeypatch, history)

    summary = _build_summary([_metric_spec()], include_diagnostics=True)
    metric = summary["diagnostics"]["metrics"]["accuracy"]
    diagnostics = {
        item["code"]: item
        for item in metric["diagnostics"]
    }

    assert metric["facts"]["duplicate_epochs"] == [2]
    assert metric["facts"]["non_monotonic_epoch_transitions"] == [
        {
            "previous_record_index": 2,
            "current_record_index": 3,
            "previous_epoch": 2,
            "current_epoch": 1,
        }
    ]
    assert diagnostics["duplicate_epochs"]["evidence"] == {
        "duplicate_epochs": [2]
    }
    assert diagnostics["non_monotonic_epochs"]["evidence"] == {
        "transitions": metric["facts"]["non_monotonic_epoch_transitions"]
    }


def test_single_record_real_integration(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    history = {
        "validation": {
            "metrics": {
                "accuracy": [[7, 0.5]],
            },
        },
    }
    _patch_inputs(monkeypatch, history)

    summary = _build_summary([_metric_spec()], include_diagnostics=True)
    facts = summary["diagnostics"]["metrics"]["accuracy"]["facts"]
    codes = _diagnostic_codes(summary, "accuracy")

    assert facts["best_progress_ratio"] is None
    assert codes == ["insufficient_history_for_trend"]
    assert "best_at_first_record" not in codes
    assert "best_at_last_record" not in codes
    assert "no_improvement" not in codes


@pytest.mark.parametrize(
    "value",
    [float("nan"), float("inf"), float("-inf")],
    ids=["nan", "positive-infinity", "negative-infinity"],
)
def test_enabled_dynamic_diagnostics_propagates_non_finite_value_error(
    monkeypatch: pytest.MonkeyPatch,
    value: float,
) -> None:
    history = {
        "validation": {
            "metrics": {
                "accuracy": [[0, 0.5], [1, value]],
            },
        },
    }
    _patch_inputs(monkeypatch, history)

    with pytest.raises(
        ValueError,
        match=r"^records\[1\]\[1\] must be finite$",
    ):
        _build_summary([_metric_spec()], include_diagnostics=True)


def test_enabled_dynamic_diagnostics_propagates_missing_path(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_inputs(monkeypatch, {"validation": {"metrics": {}}})

    with pytest.raises(KeyError) as error:
        _build_summary([_metric_spec()], include_diagnostics=True)

    assert error.value.args == (
        "missing key at path 'validation.metrics.accuracy'",
    )


def test_enabled_dynamic_diagnostics_propagates_non_mapping_path(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_inputs(monkeypatch, {"validation": {"metrics": 1}})

    with pytest.raises(
        TypeError,
        match="^value at path 'validation.metrics' must be a mapping$",
    ):
        _build_summary([_metric_spec()], include_diagnostics=True)


def test_enabled_dynamic_diagnostics_propagates_invalid_records(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    history = {"validation": {"metrics": {"accuracy": [[0]]}}}
    _patch_inputs(monkeypatch, history)

    with pytest.raises(
        ValueError,
        match=r"^records\[0\] must contain exactly two items$",
    ):
        _build_summary([_metric_spec()], include_diagnostics=True)


def test_build_metric_diagnostics_error_propagates_without_partial_summary(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_inputs(monkeypatch, _default_history())

    def raise_key_error(facts):
        raise KeyError("broken facts")

    monkeypatch.setattr(
        summarize_experiment,
        "build_metric_diagnostics",
        raise_key_error,
    )

    with pytest.raises(KeyError) as error:
        _build_summary(include_diagnostics=True)

    assert error.value.args == ("broken facts",)


def test_diagnostic_to_dict_error_propagates_without_partial_summary(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_inputs(monkeypatch, _default_history())
    diagnostic = Diagnostic("valid_code", "info", "Message", {})
    monkeypatch.setattr(
        summarize_experiment,
        "build_metric_diagnostics",
        lambda facts: (diagnostic,),
    )

    def raise_type_error(item):
        raise TypeError("serialization failed")

    monkeypatch.setattr(
        summarize_experiment,
        "diagnostic_to_dict",
        raise_type_error,
    )

    with pytest.raises(TypeError, match="^serialization failed$"):
        _build_summary(include_diagnostics=True)


def test_enabled_diagnostics_does_not_modify_config_or_default_history(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    configuration = _configuration()
    history = _default_history()
    original_configuration = deepcopy(configuration)
    original_history = deepcopy(history)
    _patch_inputs(monkeypatch, history, configuration)

    _build_summary(include_diagnostics=True)

    assert configuration == original_configuration
    assert history == original_history


def test_summary_keeps_exact_facts_object_returned_by_builder(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_inputs(monkeypatch, _default_history())
    facts_by_records: dict[int, dict[str, object]] = {}

    def fake_build_metric_facts(records, direction, *, recent_window):
        facts = {"record_count": len(records)}
        facts_by_records[id(records)] = facts
        return facts

    monkeypatch.setattr(
        summarize_experiment,
        "build_metric_facts",
        fake_build_metric_facts,
    )
    monkeypatch.setattr(
        summarize_experiment,
        "build_metric_diagnostics",
        lambda facts: (),
    )

    summary = _build_summary(include_diagnostics=True)
    history = _default_history()

    r2_facts = summary["diagnostics"]["metrics"]["r2"]["facts"]
    racc_facts = summary["diagnostics"]["metrics"]["racc"]["facts"]
    assert r2_facts in facts_by_records.values()
    assert racc_facts in facts_by_records.values()
    assert r2_facts is not racc_facts



def test_metric_recommendations_are_built_from_generated_diagnostics(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_inputs(monkeypatch, _default_history())
    diagnostics_seen: list[tuple[Diagnostic, ...]] = []

    def fake_build_recommendations(diagnostics):
        diagnostics_seen.append(diagnostics)
        return ()

    monkeypatch.setattr(
        summarize_experiment,
        "build_recommendations",
        fake_build_recommendations,
    )

    summary = _build_summary(include_diagnostics=True)

    assert len(diagnostics_seen) == 2
    assert all(isinstance(items, tuple) for items in diagnostics_seen)
    assert summary["diagnostics"]["metrics"]["r2"]["recommendations"] == []
    assert summary["diagnostics"]["metrics"]["racc"]["recommendations"] == []


def test_default_summary_serializes_deterministic_recommendations(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_inputs(monkeypatch, _default_history())

    metrics = _build_summary(include_diagnostics=True)["diagnostics"]["metrics"]

    assert [
        item["code"]
        for item in metrics["r2"]["recommendations"]
    ] == [
        "restore_best_checkpoint",
        "avoid_trend_conclusion",
    ]
    assert [
        item["code"]
        for item in metrics["racc"]["recommendations"]
    ] == [
        "restore_best_checkpoint",
        "avoid_trend_conclusion",
    ]


def test_recommendation_serialization_error_propagates(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_inputs(monkeypatch, _default_history())

    def raise_serialization_error(item):
        raise TypeError("recommendation serialization failed")

    monkeypatch.setattr(
        summarize_experiment,
        "recommendation_to_dict",
        raise_serialization_error,
    )

    with pytest.raises(
        TypeError,
        match="^recommendation serialization failed$",
    ):
        _build_summary(include_diagnostics=True)

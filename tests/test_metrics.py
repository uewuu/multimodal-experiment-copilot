from dataclasses import FrozenInstanceError

import pytest

from metrics import MetricSpec


def _valid_metric_kwargs() -> dict:
    return {
        "name": "accuracy",
        "path": ("valid", "metrics", "accuracy"),
        "direction": "maximize",
        "display_name": "Accuracy",
    }


def test_metric_spec_accepts_maximize_direction():
    metric = MetricSpec(**_valid_metric_kwargs())

    assert metric.direction == "maximize"


def test_metric_spec_accepts_minimize_direction():
    kwargs = _valid_metric_kwargs()
    kwargs["direction"] = "minimize"

    metric = MetricSpec(**kwargs)

    assert metric.direction == "minimize"


def test_metric_spec_accepts_nested_path():
    metric = MetricSpec(**_valid_metric_kwargs())

    assert metric.path == ("valid", "metrics", "accuracy")


def test_metric_spec_uses_default_precision():
    metric = MetricSpec(**_valid_metric_kwargs())

    assert metric.precision == 6


def test_metric_spec_accepts_zero_precision():
    metric = MetricSpec(**_valid_metric_kwargs(), precision=0)

    assert metric.precision == 0


def test_metric_spec_is_frozen():
    metric = MetricSpec(**_valid_metric_kwargs())

    with pytest.raises(FrozenInstanceError):
        metric.name = "loss"


def test_metric_spec_equal_values_compare_equal():
    first = MetricSpec(**_valid_metric_kwargs())
    second = MetricSpec(**_valid_metric_kwargs())

    assert first == second


def test_metric_spec_different_values_compare_not_equal():
    first = MetricSpec(**_valid_metric_kwargs())
    kwargs = _valid_metric_kwargs()
    kwargs["name"] = "macro_f1"
    second = MetricSpec(**kwargs)

    assert first != second


def test_metric_spec_preserves_original_values():
    name = "macro_f1"
    path = ("validation", "scores", "macro_f1")
    direction = "maximize"
    display_name = "Macro F1"
    precision = 4

    metric = MetricSpec(name, path, direction, display_name, precision)

    assert metric.name == name
    assert metric.path == path
    assert metric.direction == direction
    assert metric.display_name == display_name
    assert metric.precision == precision


@pytest.mark.parametrize(
    ("value", "error_type", "message"),
    [
        (123, TypeError, "name must be a string"),
        ("", ValueError, "name must not be empty or whitespace"),
        ("   ", ValueError, "name must not be empty or whitespace"),
        (
            " accuracy",
            ValueError,
            "name must not contain leading or trailing whitespace",
        ),
        (
            "accuracy ",
            ValueError,
            "name must not contain leading or trailing whitespace",
        ),
    ],
    ids=["non-string", "empty", "whitespace", "leading-space", "trailing-space"],
)
def test_metric_spec_rejects_invalid_name(value, error_type, message):
    kwargs = _valid_metric_kwargs()
    kwargs["name"] = value

    with pytest.raises(error_type) as error_info:
        MetricSpec(**kwargs)

    assert str(error_info.value) == message


@pytest.mark.parametrize(
    ("value", "error_type", "message"),
    [
        (["valid", "accuracy"], TypeError, "path must be a tuple"),
        ("valid.accuracy", TypeError, "path must be a tuple"),
        ((), ValueError, "path must not be empty"),
        (("valid", 1), TypeError, "path[1] must be a string"),
        (("valid", ""), ValueError, "path[1] must not be empty or whitespace"),
        (("valid", "  "), ValueError, "path[1] must not be empty or whitespace"),
        (
            ("valid", " accuracy"),
            ValueError,
            "path[1] must not contain leading or trailing whitespace",
        ),
        (
            ("valid", "accuracy "),
            ValueError,
            "path[1] must not contain leading or trailing whitespace",
        ),
    ],
    ids=[
        "list",
        "string",
        "empty",
        "non-string-member",
        "empty-member",
        "whitespace-member",
        "leading-space-member",
        "trailing-space-member",
    ],
)
def test_metric_spec_rejects_invalid_path(value, error_type, message):
    kwargs = _valid_metric_kwargs()
    kwargs["path"] = value

    with pytest.raises(error_type) as error_info:
        MetricSpec(**kwargs)

    assert str(error_info.value) == message


@pytest.mark.parametrize(
    ("value", "error_type", "message"),
    [
        (1, TypeError, "direction must be a string"),
        ("max", ValueError, "direction must be 'maximize' or 'minimize'"),
        ("min", ValueError, "direction must be 'maximize' or 'minimize'"),
        ("MAXIMIZE", ValueError, "direction must be 'maximize' or 'minimize'"),
        ("", ValueError, "direction must be 'maximize' or 'minimize'"),
    ],
    ids=["non-string", "max-alias", "min-alias", "uppercase", "empty"],
)
def test_metric_spec_rejects_invalid_direction(value, error_type, message):
    kwargs = _valid_metric_kwargs()
    kwargs["direction"] = value

    with pytest.raises(error_type) as error_info:
        MetricSpec(**kwargs)

    assert str(error_info.value) == message


@pytest.mark.parametrize(
    ("value", "error_type", "message"),
    [
        (123, TypeError, "display_name must be a string"),
        ("", ValueError, "display_name must not be empty or whitespace"),
        ("   ", ValueError, "display_name must not be empty or whitespace"),
        (
            " Accuracy",
            ValueError,
            "display_name must not contain leading or trailing whitespace",
        ),
        (
            "Accuracy ",
            ValueError,
            "display_name must not contain leading or trailing whitespace",
        ),
    ],
    ids=["non-string", "empty", "whitespace", "leading-space", "trailing-space"],
)
def test_metric_spec_rejects_invalid_display_name(value, error_type, message):
    kwargs = _valid_metric_kwargs()
    kwargs["display_name"] = value

    with pytest.raises(error_type) as error_info:
        MetricSpec(**kwargs)

    assert str(error_info.value) == message


@pytest.mark.parametrize(
    ("value", "error_type", "message"),
    [
        (1.5, TypeError, "precision must be an integer"),
        ("6", TypeError, "precision must be an integer"),
        (None, TypeError, "precision must be an integer"),
        (True, TypeError, "precision must be an integer"),
        (False, TypeError, "precision must be an integer"),
        (-1, ValueError, "precision must be greater than or equal to 0"),
    ],
    ids=["float", "string", "none", "true", "false", "negative"],
)
def test_metric_spec_rejects_invalid_precision(value, error_type, message):
    kwargs = _valid_metric_kwargs()
    kwargs["precision"] = value

    with pytest.raises(error_type) as error_info:
        MetricSpec(**kwargs)

    assert str(error_info.value) == message

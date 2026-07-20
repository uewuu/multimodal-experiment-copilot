from dataclasses import FrozenInstanceError
from types import MappingProxyType

import pytest

from metrics import (
    MetricSpec,
    evaluate_metric_history,
    get_value_at_path,
    parse_metric_specs,
)


def _valid_metric_kwargs() -> dict:
    return {
        "name": "accuracy",
        "path": ("valid", "metrics", "accuracy"),
        "direction": "maximize",
        "display_name": "Accuracy",
    }


def _valid_metric_definition() -> dict:
    return {
        "name": "accuracy",
        "path": ["valid", "metrics", "accuracy"],
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


def test_get_value_at_path_reads_top_level_value():
    assert get_value_at_path({"accuracy": 0.92}, ("accuracy",)) == 0.92


def test_get_value_at_path_reads_nested_value():
    data = {"valid": {"metrics": {"accuracy": 0.92}}}

    assert get_value_at_path(data, ("valid", "metrics", "accuracy")) == 0.92


def test_get_value_at_path_returns_none_value():
    assert get_value_at_path({"result": None}, ("result",)) is None


def test_get_value_at_path_returns_list_without_copying():
    values = [0.8, 0.9]

    result = get_value_at_path({"values": values}, ("values",))

    assert result is values


def test_get_value_at_path_returns_mapping_without_copying():
    metrics = {"accuracy": 0.92}

    result = get_value_at_path({"metrics": metrics}, ("metrics",))

    assert result is metrics


def test_get_value_at_path_supports_mapping_proxy():
    metrics = MappingProxyType({"accuracy": 0.92})
    data = MappingProxyType({"metrics": metrics})

    assert get_value_at_path(data, ("metrics", "accuracy")) == 0.92


def test_get_value_at_path_treats_dot_as_literal_key_content():
    data = {"metrics.accuracy": 0.92}

    assert get_value_at_path(data, ("metrics.accuracy",)) == 0.92


def test_get_value_at_path_does_not_modify_input_mapping():
    data = {"valid": {"metrics": {"accuracy": 0.92}}}
    expected = {"valid": {"metrics": {"accuracy": 0.92}}}

    get_value_at_path(data, ("valid", "metrics", "accuracy"))

    assert data == expected


def test_get_value_at_path_does_not_modify_path():
    path = ("valid", "metrics", "accuracy")

    get_value_at_path({"valid": {"metrics": {"accuracy": 0.92}}}, path)

    assert path == ("valid", "metrics", "accuracy")


def test_get_value_at_path_accepts_boolean_final_value():
    assert get_value_at_path({"converged": True}, ("converged",)) is True


@pytest.mark.parametrize(
    "data",
    [[], "not-a-mapping", None],
    ids=["list", "string", "none"],
)
def test_get_value_at_path_rejects_non_mapping_data(data):
    with pytest.raises(TypeError) as error_info:
        get_value_at_path(data, ("value",))

    assert str(error_info.value) == "data must be a mapping"


@pytest.mark.parametrize(
    ("path", "error_type", "message"),
    [
        (["value"], TypeError, "path must be a tuple"),
        ("value", TypeError, "path must be a tuple"),
        ((), ValueError, "path must not be empty"),
        (("valid", 1), TypeError, "path[1] must be a string"),
        (("valid", ""), ValueError, "path[1] must not be empty or whitespace"),
        (("valid", "  "), ValueError, "path[1] must not be empty or whitespace"),
        (
            ("valid", " metrics"),
            ValueError,
            "path[1] must not contain leading or trailing whitespace",
        ),
        (
            ("valid", "metrics "),
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
def test_get_value_at_path_rejects_invalid_path(path, error_type, message):
    with pytest.raises(error_type) as error_info:
        get_value_at_path({"valid": {}}, path)

    assert str(error_info.value) == message


def test_get_value_at_path_reports_missing_top_level_key():
    with pytest.raises(KeyError) as error_info:
        get_value_at_path({}, ("valid",))

    assert error_info.value.args[0] == "missing key at path 'valid'"


def test_get_value_at_path_reports_missing_nested_key():
    data = {"valid": {"metrics": {}}}

    with pytest.raises(KeyError) as error_info:
        get_value_at_path(data, ("valid", "metrics", "accuracy"))

    assert (
        error_info.value.args[0]
        == "missing key at path 'valid.metrics.accuracy'"
    )


def test_get_value_at_path_rejects_non_mapping_after_first_level():
    with pytest.raises(TypeError) as error_info:
        get_value_at_path({"valid": 0.92}, ("valid", "metrics"))

    assert str(error_info.value) == "value at path 'valid' must be a mapping"


def test_get_value_at_path_rejects_non_mapping_at_nested_level():
    data = {"valid": {"metrics": 0.92}}

    with pytest.raises(TypeError) as error_info:
        get_value_at_path(data, ("valid", "metrics", "accuracy"))

    assert (
        str(error_info.value)
        == "value at path 'valid.metrics' must be a mapping"
    )


def test_evaluate_metric_history_maximizes_values():
    result = evaluate_metric_history(
        [[0, 0.60], [1, 0.72], [2, 0.68]],
        "maximize",
    )

    assert result["best_epoch"] == 1
    assert result["best_value"] == 0.72


def test_evaluate_metric_history_minimizes_values():
    result = evaluate_metric_history(
        [[0, 0.60], [1, 0.42], [2, 0.48]],
        "minimize",
    )

    assert result["best_epoch"] == 1
    assert result["best_value"] == 0.42


def test_evaluate_metric_history_returns_expected_structure():
    result = evaluate_metric_history([[0, 0.60]], "maximize")

    assert set(result) == {
        "record_count",
        "first_epoch",
        "first_value",
        "last_epoch",
        "last_value",
        "best_epoch",
        "best_value",
    }


def test_evaluate_metric_history_reports_first_record():
    result = evaluate_metric_history([[5, 0.60], [6, 0.72]], "maximize")

    assert result["first_epoch"] == 5
    assert result["first_value"] == 0.60


def test_evaluate_metric_history_reports_last_record():
    result = evaluate_metric_history([[5, 0.60], [6, 0.72]], "maximize")

    assert result["last_epoch"] == 6
    assert result["last_value"] == 0.72


def test_evaluate_metric_history_reports_record_count():
    result = evaluate_metric_history([[0, 0.60], [1, 0.72], [2, 0.68]], "maximize")

    assert result["record_count"] == 3


def test_evaluate_metric_history_preserves_first_maximum_on_tie():
    result = evaluate_metric_history([[0, 0.72], [1, 0.60], [2, 0.72]], "maximize")

    assert result["best_epoch"] == 0


def test_evaluate_metric_history_preserves_first_minimum_on_tie():
    result = evaluate_metric_history([[0, 0.42], [1, 0.60], [2, 0.42]], "minimize")

    assert result["best_epoch"] == 0


def test_evaluate_metric_history_accepts_tuple_records():
    result = evaluate_metric_history([(0, 0.60), (1, 0.72)], "maximize")

    assert result["best_epoch"] == 1


def test_evaluate_metric_history_accepts_tuple_container():
    result = evaluate_metric_history(([0, 0.60], [1, 0.72]), "maximize")

    assert result["record_count"] == 2


def test_evaluate_metric_history_accepts_integer_values():
    result = evaluate_metric_history([[0, 2], [1, 3]], "maximize")

    assert result["best_value"] == 3
    assert isinstance(result["best_value"], int)


def test_evaluate_metric_history_accepts_negative_values():
    result = evaluate_metric_history([[0, -2.0], [1, -3.0]], "minimize")

    assert result["best_value"] == -3.0


def test_evaluate_metric_history_accepts_negative_epoch():
    result = evaluate_metric_history([[-1, 0.60], [0, 0.72]], "maximize")

    assert result["first_epoch"] == -1


def test_evaluate_metric_history_does_not_sort_by_epoch():
    result = evaluate_metric_history([[10, 0.60], [1, 0.72], [5, 0.68]], "maximize")

    assert result["first_epoch"] == 10
    assert result["last_epoch"] == 5
    assert result["best_epoch"] == 1


def test_evaluate_metric_history_does_not_modify_records():
    records = [[0, 0.60], [1, 0.72]]
    expected = [[0, 0.60], [1, 0.72]]

    evaluate_metric_history(records, "maximize")

    assert records == expected


def test_evaluate_metric_history_handles_single_record():
    result = evaluate_metric_history([[4, 0.75]], "minimize")

    assert result == {
        "record_count": 1,
        "first_epoch": 4,
        "first_value": 0.75,
        "last_epoch": 4,
        "last_value": 0.75,
        "best_epoch": 4,
        "best_value": 0.75,
    }


@pytest.mark.parametrize(
    "records",
    [
        "records",
        b"records",
        bytearray(b"records"),
        {"epoch": 0, "value": 0.60},
        (record for record in [[0, 0.60]]),
        123,
    ],
    ids=["string", "bytes", "bytearray", "mapping", "generator", "integer"],
)
def test_evaluate_metric_history_rejects_non_sequence_records(records):
    with pytest.raises(TypeError) as error_info:
        evaluate_metric_history(records, "maximize")

    assert str(error_info.value) == "records must be a sequence"


@pytest.mark.parametrize("records", [[], ()], ids=["list", "tuple"])
def test_evaluate_metric_history_rejects_empty_records(records):
    with pytest.raises(ValueError) as error_info:
        evaluate_metric_history(records, "maximize")

    assert str(error_info.value) == "records must not be empty"


@pytest.mark.parametrize(
    ("direction", "error_type", "message"),
    [
        (1, TypeError, "direction must be a string"),
        ("max", ValueError, "direction must be 'maximize' or 'minimize'"),
        ("min", ValueError, "direction must be 'maximize' or 'minimize'"),
        ("MAXIMIZE", ValueError, "direction must be 'maximize' or 'minimize'"),
        ("", ValueError, "direction must be 'maximize' or 'minimize'"),
    ],
    ids=["non-string", "max-alias", "min-alias", "uppercase", "empty"],
)
def test_evaluate_metric_history_rejects_invalid_direction(
    direction,
    error_type,
    message,
):
    with pytest.raises(error_type) as error_info:
        evaluate_metric_history([[0, 0.60]], direction)

    assert str(error_info.value) == message


@pytest.mark.parametrize(
    ("record", "error_type", "message"),
    [
        (123, TypeError, "records[0] must be a sequence"),
        ("record", TypeError, "records[0] must be a sequence"),
        (b"record", TypeError, "records[0] must be a sequence"),
        (bytearray(b"record"), TypeError, "records[0] must be a sequence"),
        ({"epoch": 0}, TypeError, "records[0] must be a sequence"),
        ([0], ValueError, "records[0] must contain exactly two items"),
        ([0, 0.60, 1], ValueError, "records[0] must contain exactly two items"),
        ([], ValueError, "records[0] must contain exactly two items"),
    ],
    ids=[
        "integer",
        "string",
        "bytes",
        "bytearray",
        "mapping",
        "one-item",
        "three-items",
        "empty",
    ],
)
def test_evaluate_metric_history_rejects_invalid_record(
    record,
    error_type,
    message,
):
    with pytest.raises(error_type) as error_info:
        evaluate_metric_history([record], "maximize")

    assert str(error_info.value) == message


@pytest.mark.parametrize(
    "epoch",
    [1.5, "0", None, True, False],
    ids=["float", "string", "none", "true", "false"],
)
def test_evaluate_metric_history_rejects_invalid_epoch(epoch):
    with pytest.raises(TypeError) as error_info:
        evaluate_metric_history([[epoch, 0.60]], "maximize")

    assert str(error_info.value) == "records[0][0] must be an integer"


@pytest.mark.parametrize(
    "value",
    ["0.60", None, [], True, False],
    ids=["string", "none", "list", "true", "false"],
)
def test_evaluate_metric_history_rejects_invalid_value(value):
    with pytest.raises(TypeError) as error_info:
        evaluate_metric_history([[0, value]], "maximize")

    assert str(error_info.value) == "records[0][1] must be a number"


def test_parse_metric_specs_parses_single_metric():
    specs = parse_metric_specs([_valid_metric_definition()])

    assert specs == (
        MetricSpec(
            name="accuracy",
            path=("valid", "metrics", "accuracy"),
            direction="maximize",
            display_name="Accuracy",
        ),
    )


def test_parse_metric_specs_parses_multiple_metrics():
    second = {
        "name": "validation_loss",
        "path": ["valid", "all", "avg_loss"],
        "direction": "minimize",
        "display_name": "Validation Loss",
    }

    specs = parse_metric_specs([_valid_metric_definition(), second])

    assert len(specs) == 2
    assert specs[1].name == "validation_loss"


def test_parse_metric_specs_preserves_metric_order():
    second = _valid_metric_definition()
    second["name"] = "macro_f1"
    second["display_name"] = "Macro F1"

    specs = parse_metric_specs([_valid_metric_definition(), second])

    assert tuple(spec.name for spec in specs) == ("accuracy", "macro_f1")


def test_parse_metric_specs_returns_tuple():
    specs = parse_metric_specs([_valid_metric_definition()])

    assert isinstance(specs, tuple)


def test_parse_metric_specs_converts_list_path_to_tuple():
    spec = parse_metric_specs([_valid_metric_definition()])[0]

    assert spec.path == ("valid", "metrics", "accuracy")
    assert isinstance(spec.path, tuple)


def test_parse_metric_specs_accepts_tuple_path():
    definition = _valid_metric_definition()
    definition["path"] = ("valid", "metrics", "accuracy")

    spec = parse_metric_specs([definition])[0]

    assert spec.path == ("valid", "metrics", "accuracy")


def test_parse_metric_specs_uses_default_precision():
    spec = parse_metric_specs([_valid_metric_definition()])[0]

    assert spec.precision == 6


def test_parse_metric_specs_preserves_explicit_precision():
    definition = _valid_metric_definition()
    definition["precision"] = 4

    spec = parse_metric_specs([definition])[0]

    assert spec.precision == 4


def test_parse_metric_specs_accepts_maximize_direction():
    spec = parse_metric_specs([_valid_metric_definition()])[0]

    assert spec.direction == "maximize"


def test_parse_metric_specs_accepts_minimize_direction():
    definition = _valid_metric_definition()
    definition["direction"] = "minimize"

    spec = parse_metric_specs([definition])[0]

    assert spec.direction == "minimize"


def test_parse_metric_specs_supports_mapping_proxy():
    definition = MappingProxyType(_valid_metric_definition())

    spec = parse_metric_specs([definition])[0]

    assert spec.name == "accuracy"


def test_parse_metric_specs_supports_tuple_container():
    specs = parse_metric_specs((_valid_metric_definition(),))

    assert len(specs) == 1


def test_parse_metric_specs_returns_empty_tuple_for_empty_definitions():
    assert parse_metric_specs([]) == ()


def test_parse_metric_specs_does_not_modify_definitions():
    definitions = [_valid_metric_definition()]
    expected = [_valid_metric_definition()]

    parse_metric_specs(definitions)

    assert definitions == expected


def test_parse_metric_specs_does_not_modify_original_path_list():
    path = ["valid", "metrics", "accuracy"]
    definition = _valid_metric_definition()
    definition["path"] = path

    parse_metric_specs([definition])

    assert path == ["valid", "metrics", "accuracy"]
    assert isinstance(path, list)


def test_parse_metric_specs_allows_duplicate_paths():
    second = _valid_metric_definition()
    second["name"] = "macro_f1"
    second["display_name"] = "Macro F1"

    specs = parse_metric_specs([_valid_metric_definition(), second])

    assert specs[0].path == specs[1].path


def test_parse_metric_specs_allows_duplicate_display_names():
    second = _valid_metric_definition()
    second["name"] = "macro_f1"

    specs = parse_metric_specs([_valid_metric_definition(), second])

    assert specs[0].display_name == specs[1].display_name


@pytest.mark.parametrize(
    "definitions",
    [
        "definitions",
        b"definitions",
        None,
        {"metrics": []},
        (definition for definition in [_valid_metric_definition()]),
    ],
    ids=["string", "bytes", "none", "mapping", "generator"],
)
def test_parse_metric_specs_rejects_non_sequence_definitions(definitions):
    with pytest.raises(TypeError) as error_info:
        parse_metric_specs(definitions)

    assert str(error_info.value) == "definitions must be a sequence"


@pytest.mark.parametrize(
    "definition",
    [123, "definition", [], None],
    ids=["integer", "string", "list", "none"],
)
def test_parse_metric_specs_rejects_non_mapping_definition(definition):
    with pytest.raises(TypeError) as error_info:
        parse_metric_specs([definition])

    assert str(error_info.value) == "definitions[0] must be a mapping"


def test_parse_metric_specs_rejects_non_string_definition_key():
    definition = _valid_metric_definition()
    definition[1] = "unexpected"

    with pytest.raises(TypeError) as error_info:
        parse_metric_specs([definition])

    assert str(error_info.value) == "definitions[0] keys must be strings"


@pytest.mark.parametrize(
    "field",
    ["name", "path", "direction", "display_name"],
)
def test_parse_metric_specs_reports_missing_required_field(field):
    definition = _valid_metric_definition()
    del definition[field]

    with pytest.raises(ValueError) as error_info:
        parse_metric_specs([definition])

    assert (
        str(error_info.value)
        == f"definitions[0] is missing required field '{field}'"
    )


@pytest.mark.parametrize("field", ["typo", "extra"])
def test_parse_metric_specs_rejects_unknown_field(field):
    definition = _valid_metric_definition()
    definition[field] = "unexpected"

    with pytest.raises(ValueError) as error_info:
        parse_metric_specs([definition])

    assert (
        str(error_info.value)
        == f"definitions[0] contains unknown field '{field}'"
    )


@pytest.mark.parametrize(
    "path",
    [
        "valid.metrics.accuracy",
        {"valid": "accuracy"},
        {"valid", "accuracy"},
        (part for part in ["valid", "accuracy"]),
        None,
    ],
    ids=["string", "mapping", "set", "generator", "none"],
)
def test_parse_metric_specs_rejects_invalid_path_container(path):
    definition = _valid_metric_definition()
    definition["path"] = path

    with pytest.raises(TypeError) as error_info:
        parse_metric_specs([definition])

    assert str(error_info.value) == (
        "definitions[0].path must be a list or tuple"
    )


@pytest.mark.parametrize(
    ("field", "value", "error_type", "message"),
    [
        ("name", 1, TypeError, "name must be a string"),
        ("name", "   ", ValueError, "name must not be empty or whitespace"),
        ("path", ["valid", 1], TypeError, "path[1] must be a string"),
        (
            "direction",
            "max",
            ValueError,
            "direction must be 'maximize' or 'minimize'",
        ),
        (
            "display_name",
            "",
            ValueError,
            "display_name must not be empty or whitespace",
        ),
        ("precision", True, TypeError, "precision must be an integer"),
        (
            "precision",
            -1,
            ValueError,
            "precision must be greater than or equal to 0",
        ),
    ],
    ids=[
        "name-type",
        "blank-name",
        "path-member-type",
        "direction",
        "empty-display-name",
        "bool-precision",
        "negative-precision",
    ],
)
def test_parse_metric_specs_prefixes_metric_spec_errors(
    field,
    value,
    error_type,
    message,
):
    definition = _valid_metric_definition()
    definition[field] = value

    with pytest.raises(error_type) as error_info:
        parse_metric_specs([definition])

    assert str(error_info.value) == f"definitions[0]: {message}"


def test_parse_metric_specs_rejects_duplicate_metric_name():
    first = _valid_metric_definition()
    second = _valid_metric_definition()

    with pytest.raises(ValueError) as error_info:
        parse_metric_specs([first, second])

    assert str(error_info.value) == (
        "duplicate metric name 'accuracy' at definitions[1]; "
        "first defined at definitions[0]"
    )

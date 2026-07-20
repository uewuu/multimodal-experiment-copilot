from copy import deepcopy
from pathlib import Path

import pytest
import yaml

import read_metrics_config
from metrics import MetricSpec
from read_metrics_config import read_metric_specs_config


def write_yaml_text(
    tmp_path: Path,
    content: str,
) -> Path:
    config_path = tmp_path / "metrics.yaml"
    config_path.write_text(content, encoding="utf-8")
    return config_path


def valid_metric_definition() -> dict:
    return {
        "name": "accuracy",
        "path": ["validation", "metrics", "accuracy"],
        "direction": "maximize",
        "display_name": "Accuracy",
    }


def patch_yaml_data(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    data: object,
) -> Path:
    config_path = write_yaml_text(tmp_path, "placeholder: true\n")
    monkeypatch.setattr(
        read_metrics_config.yaml,
        "safe_load",
        lambda file: data,
    )
    return config_path


def test_read_metric_specs_config_reads_single_metric(tmp_path: Path) -> None:
    config_path = write_yaml_text(
        tmp_path,
        """metrics:
  - name: accuracy
    path: [validation, metrics, accuracy]
    direction: maximize
    display_name: Accuracy
""",
    )

    specs = read_metric_specs_config(config_path)

    assert specs == (
        MetricSpec(
            name="accuracy",
            path=("validation", "metrics", "accuracy"),
            direction="maximize",
            display_name="Accuracy",
        ),
    )


def test_read_metric_specs_config_reads_multiple_metrics(tmp_path: Path) -> None:
    config_path = write_yaml_text(
        tmp_path,
        """metrics:
  - name: accuracy
    path: [validation, metrics, accuracy]
    direction: maximize
    display_name: Accuracy
  - name: validation_loss
    path: [validation, metrics, loss]
    direction: minimize
    display_name: Validation Loss
""",
    )

    specs = read_metric_specs_config(config_path)

    assert len(specs) == 2
    assert specs[1].name == "validation_loss"


def test_read_metric_specs_config_returns_tuple(tmp_path: Path) -> None:
    config_path = write_yaml_text(
        tmp_path,
        """metrics:
  - name: accuracy
    path: [validation, accuracy]
    direction: maximize
    display_name: Accuracy
""",
    )

    assert isinstance(read_metric_specs_config(config_path), tuple)


def test_read_metric_specs_config_returns_metric_spec_instances(
    tmp_path: Path,
) -> None:
    config_path = write_yaml_text(
        tmp_path,
        """metrics:
  - name: accuracy
    path: [validation, accuracy]
    direction: maximize
    display_name: Accuracy
""",
    )

    specs = read_metric_specs_config(config_path)

    assert all(isinstance(spec, MetricSpec) for spec in specs)


def test_read_metric_specs_config_preserves_metric_order(tmp_path: Path) -> None:
    config_path = write_yaml_text(
        tmp_path,
        """metrics:
  - name: validation_loss
    path: [validation, loss]
    direction: minimize
    display_name: Validation Loss
  - name: accuracy
    path: [validation, accuracy]
    direction: maximize
    display_name: Accuracy
""",
    )

    specs = read_metric_specs_config(config_path)

    assert tuple(spec.name for spec in specs) == (
        "validation_loss",
        "accuracy",
    )


def test_read_metric_specs_config_converts_path_to_tuple(tmp_path: Path) -> None:
    config_path = write_yaml_text(
        tmp_path,
        """metrics:
  - name: accuracy
    path: [validation, metrics, accuracy]
    direction: maximize
    display_name: Accuracy
""",
    )

    spec = read_metric_specs_config(config_path)[0]

    assert spec.path == ("validation", "metrics", "accuracy")
    assert isinstance(spec.path, tuple)


def test_read_metric_specs_config_uses_default_precision(tmp_path: Path) -> None:
    config_path = write_yaml_text(
        tmp_path,
        """metrics:
  - name: accuracy
    path: [validation, accuracy]
    direction: maximize
    display_name: Accuracy
""",
    )

    assert read_metric_specs_config(config_path)[0].precision == 6


def test_read_metric_specs_config_preserves_explicit_precision(
    tmp_path: Path,
) -> None:
    config_path = write_yaml_text(
        tmp_path,
        """metrics:
  - name: accuracy
    path: [validation, accuracy]
    direction: maximize
    display_name: Accuracy
    precision: 4
""",
    )

    assert read_metric_specs_config(config_path)[0].precision == 4


def test_read_metric_specs_config_accepts_maximize(tmp_path: Path) -> None:
    config_path = write_yaml_text(
        tmp_path,
        """metrics:
  - name: accuracy
    path: [validation, accuracy]
    direction: maximize
    display_name: Accuracy
""",
    )

    assert read_metric_specs_config(config_path)[0].direction == "maximize"


def test_read_metric_specs_config_accepts_minimize(tmp_path: Path) -> None:
    config_path = write_yaml_text(
        tmp_path,
        """metrics:
  - name: loss
    path: [validation, loss]
    direction: minimize
    display_name: Loss
""",
    )

    assert read_metric_specs_config(config_path)[0].direction == "minimize"


def test_read_metric_specs_config_reads_utf8_display_name(tmp_path: Path) -> None:
    config_path = write_yaml_text(
        tmp_path,
        """metrics:
  - name: accuracy
    path: [validation, accuracy]
    direction: maximize
    display_name: 验证集准确率
""",
    )

    assert read_metric_specs_config(config_path)[0].display_name == "验证集准确率"


def test_read_metric_specs_config_accepts_flow_style_path(tmp_path: Path) -> None:
    config_path = write_yaml_text(
        tmp_path,
        """metrics:
  - name: accuracy
    path: [validation, metrics, accuracy]
    direction: maximize
    display_name: Accuracy
""",
    )

    assert read_metric_specs_config(config_path)[0].path == (
        "validation",
        "metrics",
        "accuracy",
    )


def test_read_metric_specs_config_accepts_block_style_path(tmp_path: Path) -> None:
    config_path = write_yaml_text(
        tmp_path,
        """metrics:
  - name: accuracy
    path:
      - validation
      - metrics
      - accuracy
    direction: maximize
    display_name: Accuracy
""",
    )

    assert read_metric_specs_config(config_path)[0].path == (
        "validation",
        "metrics",
        "accuracy",
    )


def test_read_metric_specs_config_does_not_use_config_wrapper(
    tmp_path: Path,
) -> None:
    config_path = write_yaml_text(
        tmp_path,
        """config:
  metrics:
    - name: accuracy
      path: [validation, accuracy]
      direction: maximize
      display_name: Accuracy
""",
    )

    with pytest.raises(ValueError) as error_info:
        read_metric_specs_config(config_path)

    assert str(error_info.value) == (
        "metrics configuration is missing required field 'metrics'"
    )


def test_read_metric_specs_config_propagates_file_not_found_error(
    tmp_path: Path,
) -> None:
    with pytest.raises(FileNotFoundError):
        read_metric_specs_config(tmp_path / "missing.yaml")


def test_read_metric_specs_config_propagates_is_a_directory_error(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def raise_is_a_directory_error(*args: object, **kwargs: object):
        raise IsADirectoryError("is a directory")

    monkeypatch.setattr(Path, "open", raise_is_a_directory_error)

    with pytest.raises(IsADirectoryError, match="is a directory"):
        read_metric_specs_config(tmp_path)


def test_read_metric_specs_config_propagates_permission_error(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def raise_permission_error(*args: object, **kwargs: object):
        raise PermissionError("permission denied")

    monkeypatch.setattr(Path, "open", raise_permission_error)

    with pytest.raises(PermissionError, match="permission denied"):
        read_metric_specs_config(tmp_path / "metrics.yaml")


def test_read_metric_specs_config_propagates_yaml_error(tmp_path: Path) -> None:
    config_path = write_yaml_text(tmp_path, "metrics: [\n")

    with pytest.raises(yaml.YAMLError):
        read_metric_specs_config(config_path)


@pytest.mark.parametrize(
    "content",
    ["", "null\n", "- metric\n", "metrics\n", "42\n"],
    ids=["empty", "null", "list", "string", "number"],
)
def test_read_metric_specs_config_rejects_non_mapping_top_level(
    tmp_path: Path,
    content: str,
) -> None:
    config_path = write_yaml_text(tmp_path, content)

    with pytest.raises(ValueError) as error_info:
        read_metric_specs_config(config_path)

    assert str(error_info.value) == (
        "metrics configuration must be a mapping"
    )


def test_read_metric_specs_config_rejects_non_string_top_level_key(
    tmp_path: Path,
) -> None:
    config_path = write_yaml_text(
        tmp_path,
        """1: invalid
metrics: []
""",
    )

    with pytest.raises(TypeError) as error_info:
        read_metric_specs_config(config_path)

    assert str(error_info.value) == (
        "metrics configuration keys must be strings"
    )


def test_read_metric_specs_config_rejects_missing_metrics(tmp_path: Path) -> None:
    config_path = write_yaml_text(tmp_path, "version: 1\n")

    with pytest.raises(ValueError) as error_info:
        read_metric_specs_config(config_path)

    assert str(error_info.value) == (
        "metrics configuration is missing required field 'metrics'"
    )


def test_read_metric_specs_config_rejects_unknown_top_level_field(
    tmp_path: Path,
) -> None:
    config_path = write_yaml_text(
        tmp_path,
        """metrics:
  - name: accuracy
    path: [validation, accuracy]
    direction: maximize
    display_name: Accuracy
version: 1
""",
    )

    with pytest.raises(ValueError) as error_info:
        read_metric_specs_config(config_path)

    assert str(error_info.value) == (
        "metrics configuration contains unknown field 'version'"
    )


def test_read_metric_specs_config_reports_first_unknown_field(
    tmp_path: Path,
) -> None:
    config_path = write_yaml_text(
        tmp_path,
        """second: 2
metrics:
  - name: accuracy
    path: [validation, accuracy]
    direction: maximize
    display_name: Accuracy
first: 1
""",
    )

    with pytest.raises(ValueError) as error_info:
        read_metric_specs_config(config_path)

    assert str(error_info.value) == (
        "metrics configuration contains unknown field 'second'"
    )


@pytest.mark.parametrize(
    "metrics_value",
    [
        "metrics",
        b"metrics",
        bytearray(b"metrics"),
        {"name": "accuracy"},
        1,
        True,
        None,
        {"accuracy"},
        (value for value in [valid_metric_definition()]),
    ],
    ids=[
        "string",
        "bytes",
        "bytearray",
        "mapping",
        "integer",
        "boolean",
        "null",
        "set",
        "generator",
    ],
)
def test_read_metric_specs_config_rejects_invalid_metrics_container(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    metrics_value: object,
) -> None:
    config_path = patch_yaml_data(
        tmp_path,
        monkeypatch,
        {"metrics": metrics_value},
    )

    with pytest.raises(TypeError) as error_info:
        read_metric_specs_config(config_path)

    assert str(error_info.value) == (
        "metrics configuration field 'metrics' must be a sequence"
    )


@pytest.mark.parametrize(
    "metrics_value",
    [[], ()],
    ids=["list", "tuple"],
)
def test_read_metric_specs_config_rejects_empty_metrics_sequence(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    metrics_value: object,
) -> None:
    config_path = patch_yaml_data(
        tmp_path,
        monkeypatch,
        {"metrics": metrics_value},
    )

    with pytest.raises(ValueError) as error_info:
        read_metric_specs_config(config_path)

    assert str(error_info.value) == (
        "metrics configuration field 'metrics' must not be empty"
    )


@pytest.mark.parametrize(
    ("case", "error_type", "message"),
    [
        (
            "non_mapping",
            TypeError,
            "definitions[0] must be a mapping",
        ),
        (
            "missing_name",
            ValueError,
            "definitions[0] is missing required field 'name'",
        ),
        (
            "missing_path",
            ValueError,
            "definitions[0] is missing required field 'path'",
        ),
        (
            "unknown_field",
            ValueError,
            "definitions[0] contains unknown field 'extra'",
        ),
        (
            "path_type",
            TypeError,
            "definitions[0].path must be a list or tuple",
        ),
        (
            "path_member",
            TypeError,
            "definitions[0]: path[1] must be a string",
        ),
        (
            "direction",
            ValueError,
            "definitions[0]: direction must be 'maximize' or 'minimize'",
        ),
        (
            "display_name",
            ValueError,
            "definitions[0]: display_name must not be empty or whitespace",
        ),
        (
            "precision_bool",
            TypeError,
            "definitions[0]: precision must be an integer",
        ),
        (
            "precision_negative",
            ValueError,
            "definitions[0]: precision must be greater than or equal to 0",
        ),
        (
            "duplicate_name",
            ValueError,
            "duplicate metric name 'accuracy' at definitions[1]; "
            "first defined at definitions[0]",
        ),
    ],
)
def test_read_metric_specs_config_preserves_parse_metric_specs_errors(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    case: str,
    error_type: type[Exception],
    message: str,
) -> None:
    definition: object = valid_metric_definition()
    if case == "non_mapping":
        definition = 1
    elif case == "missing_name":
        del definition["name"]
    elif case == "missing_path":
        del definition["path"]
    elif case == "unknown_field":
        definition["extra"] = True
    elif case == "path_type":
        definition["path"] = "validation.accuracy"
    elif case == "path_member":
        definition["path"] = ["validation", 1]
    elif case == "direction":
        definition["direction"] = "maximum"
    elif case == "display_name":
        definition["display_name"] = ""
    elif case == "precision_bool":
        definition["precision"] = True
    elif case == "precision_negative":
        definition["precision"] = -1

    definitions = [definition]
    if case == "duplicate_name":
        definitions.append(valid_metric_definition())
    config_path = patch_yaml_data(
        tmp_path,
        monkeypatch,
        {"metrics": definitions},
    )

    with pytest.raises(error_type) as error_info:
        read_metric_specs_config(config_path)

    assert str(error_info.value) == message
    assert str(config_path) not in str(error_info.value)


def test_read_metric_specs_config_does_not_modify_parsed_data(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = ["validation", "metrics", "accuracy"]
    definitions = [
        {
            "name": "accuracy",
            "path": path,
            "direction": "maximize",
            "display_name": "Accuracy",
        }
    ]
    data = {"metrics": definitions}
    original_data = deepcopy(data)
    config_path = patch_yaml_data(tmp_path, monkeypatch, data)

    specs = read_metric_specs_config(config_path)

    assert data == original_data
    assert data["metrics"] is definitions
    assert definitions[0]["path"] is path
    assert isinstance(path, list)
    assert specs[0].path == tuple(path)

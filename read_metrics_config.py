"""独立指标 YAML 配置读取。"""

from collections.abc import Mapping, Sequence
from pathlib import Path

import yaml

from metrics import MetricSpec, parse_metric_specs


def read_metric_specs_config(
    config_path: Path,
) -> tuple[MetricSpec, ...]:
    """读取并解析独立指标 YAML 配置。"""
    with config_path.open(
        "r",
        encoding="utf-8",
    ) as file:
        data = yaml.safe_load(file)

    if not isinstance(data, Mapping):
        raise ValueError("metrics configuration must be a mapping")
    if any(not isinstance(key, str) for key in data):
        raise TypeError("metrics configuration keys must be strings")
    if "metrics" not in data:
        raise ValueError(
            "metrics configuration is missing required field 'metrics'"
        )
    for field in data:
        if field != "metrics":
            raise ValueError(
                "metrics configuration contains unknown field "
                f"'{field}'"
            )

    metrics_value = data["metrics"]
    if (
        isinstance(
            metrics_value,
            (str, bytes, bytearray, Mapping),
        )
        or not isinstance(metrics_value, Sequence)
    ):
        raise TypeError(
            "metrics configuration field 'metrics' "
            "must be a sequence"
        )
    if not metrics_value:
        raise ValueError(
            "metrics configuration field 'metrics' "
            "must not be empty"
        )

    return parse_metric_specs(metrics_value)

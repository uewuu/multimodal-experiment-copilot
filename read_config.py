from collections.abc import Mapping, Sequence
import json
import math
from pathlib import Path

import yaml

from experiment_path_access import _authorize_derived_file

CONFIG_PATH = Path("examples/demo_experiment/hparams.yaml")


def read_config(config_path: Path) -> dict:
    """读取 YAML 配置文件，并返回其中的配置字典。"""
    with _authorize_derived_file(config_path).open("r", encoding="utf-8") as file:
        data = yaml.safe_load(file)

    if not isinstance(data, dict):
        raise ValueError("YAML 文件的最外层不是字典结构。")

    # 当前 hparams.yaml 的配置位于最外层 config 节点中。
    config = data.get("config", data)

    if not isinstance(config, dict):
        raise ValueError("config 节点不是有效的字典结构。")

    return config


def _evidence_paths(field_paths: Sequence[Sequence[str]]) -> tuple[tuple[str, ...], ...]:
    """Validate and snapshot ordered host selections before reading a file."""
    if isinstance(field_paths, (str, bytes)) or not isinstance(field_paths, Sequence):
        raise TypeError("field_paths must be a sequence of segment sequences")
    if len(field_paths) > 32:
        raise ValueError("at most 32 configuration fields may be selected")

    paths = []
    seen = set()
    for path in field_paths:
        if isinstance(path, (str, bytes)) or not isinstance(path, Sequence):
            raise TypeError("each field path must be a sequence of string segments")
        if not 1 <= len(path) <= 8:
            raise ValueError("field path depth must be between 1 and 8")
        for segment in path:
            if not isinstance(segment, str):
                raise TypeError("field path segments must be strings")
            if (not segment or segment != segment.strip() or "*" in segment
                    or len(segment.encode("utf-8")) > 64):
                raise ValueError("invalid configuration field path segment")
        selected = tuple(path)
        if selected in seen:
            raise ValueError("duplicate configuration field path")
        seen.add(selected)
        paths.append(selected)
    return tuple(paths)


def _configuration_field(config: Mapping, path: tuple[str, ...]) -> dict:
    """Project one literal path; never stringify or expand unsupported values."""
    field = {"path": list(path), "evidence_kind": "configuration_declared"}
    value = config
    for segment in path:
        if not isinstance(value, Mapping):
            return {**field, "status": "unavailable", "reason": "non_mapping_parent"}
        if segment not in value:
            return {**field, "status": "missing"}
        value = value[segment]

    reason = None
    if value is not None and type(value) not in (bool, int, float, str):
        reason = "unsupported_value_type"
    elif type(value) is float and not math.isfinite(value):
        reason = "non_finite_value"
    elif isinstance(value, str) and len(value.encode("utf-8")) > 512:
        reason = "value_too_large"
    if reason is not None:
        return {**field, "status": "unavailable", "reason": reason}
    return {**field, "status": "present", "value": value}


def read_config_evidence(config_path: Path, field_paths: Sequence[Sequence[str]]) -> dict:
    """Read bounded evidence for explicitly selected YAML configuration fields.

    Paths are literal segments relative to the same root used by read_config:
    the ``config`` mapping when present, otherwise the document mapping. Dots
    inside a segment remain literal. Values describe file declarations, not
    proof of training behavior. Missing fields omit ``value``; explicit null,
    false, zero and empty strings retain their types and values.

    Invalid selections and output over 16 KiB raise TypeError/ValueError;
    filesystem and safe YAML parsing errors propagate. No partial projection
    is returned on output overflow. This reader adds no path authorization.
    """
    paths = _evidence_paths(field_paths)
    with config_path.open("r", encoding="utf-8") as file:
        data = yaml.safe_load(file)
    if not isinstance(data, dict):
        raise ValueError("YAML 文件的最外层不是字典结构。")
    config = data.get("config", data)
    if not isinstance(config, dict):
        raise ValueError("config 节点不是有效的字典结构。")

    result = {
        "schema_version": "configuration-evidence-v1",
        "source_file": config_path.name,
        "root_path": ["config"] if "config" in data else [],
        "fields": [_configuration_field(config, path) for path in paths],
    }
    serialized = json.dumps(result, ensure_ascii=False, allow_nan=False,
                            separators=(",", ":")).encode("utf-8")
    if len(serialized) > 16 * 1024:
        raise ValueError("configuration evidence exceeds 16 KiB")
    return result


def show_config(config: dict) -> None:
    """显示当前阶段关注的核心配置字段。"""
    fields = [
        "batch_size",
        "num_workers",
        "seed",
        "sample_seed",
        "n_epochs",
        "feature_list",
        "attention_type",
        "use_modality_token_fusion",
        "use_behavior_state_token",
        "use_behavior_aware_cl",
        "use_regression_aware_cl",
        "use_trait_conditioned_modality_selection",
    ]

    print("实验核心配置")
    print("-" * 50)

    for field in fields:
        value = config.get(field, "未找到")
        print(f"{field}: {value}")


def main() -> None:
    try:
        config = read_config(CONFIG_PATH)
        show_config(config)

    except FileNotFoundError:
        print(f"错误：没有找到配置文件：{CONFIG_PATH}")

    except yaml.YAMLError as error:
        print(f"错误：YAML 文件格式不正确：{error}")

    except UnicodeDecodeError:
        print("错误：无法使用 UTF-8 编码读取配置文件。")

    except ValueError as error:
        print(f"错误：{error}")


if __name__ == "__main__":
    main()

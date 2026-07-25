"""单实验分析 Tool Layer。"""

import json
from pathlib import Path

from compare_experiments import (
    analyze_experiment_dirs,
    build_comparison_payload,
    find_experiment_dirs,
)
from generate_report import resolve_experiment_paths
from read_metrics_config import read_metric_specs_config
from summarize_experiment import build_experiment_summary


def analyze_experiment(
    experiment_dir: str,
    *,
    metrics_config: str | None = None,
    include_diagnostics: bool = False,
) -> dict:
    """分析单个实验目录并返回严格 JSON 友好的结构化摘要。

    Args:
        experiment_dir: 包含 hparams.yaml 和 history.json 的目录路径。
        metrics_config: 可选的独立指标 YAML 配置文件路径。
        include_diagnostics: 是否在摘要中包含规则诊断和建议。

    Returns:
        ``build_experiment_summary`` 返回的原始摘要字典。

    Raises:
        TypeError: 参数类型、摘要类型或 JSON 值类型无效。
        ValueError: 字符串参数为空，或摘要包含非有限数值。
        OSError: 实验文件或指标配置文件无法访问。
        Exception: 其他底层解析、指标和诊断异常原样传播。

    此函数不打印业务结果，不创建输出目录，也不写入报告文件。
    """
    if not isinstance(experiment_dir, str):
        raise TypeError("experiment_dir must be a string")
    if not experiment_dir.strip():
        raise ValueError(
            "experiment_dir must not be empty or whitespace"
        )

    if metrics_config is not None:
        if not isinstance(metrics_config, str):
            raise TypeError(
                "metrics_config must be a string or None"
            )
        if not metrics_config.strip():
            raise ValueError(
                "metrics_config must not be empty or whitespace"
            )

    if not isinstance(include_diagnostics, bool):
        raise TypeError("include_diagnostics must be a boolean")

    config_path, history_path = resolve_experiment_paths(
        Path(experiment_dir)
    )

    if metrics_config is None:
        result = build_experiment_summary(
            config_path=config_path,
            history_path=history_path,
            include_diagnostics=include_diagnostics,
        )
    else:
        metric_specs = read_metric_specs_config(
            Path(metrics_config)
        )
        result = build_experiment_summary(
            config_path=config_path,
            history_path=history_path,
            metric_specs=metric_specs,
            include_diagnostics=include_diagnostics,
        )

    if not isinstance(result, dict):
        raise TypeError(
            "build_experiment_summary must return a dict"
        )

    json.dumps(
        result,
        ensure_ascii=False,
        allow_nan=False,
    )
    return result


def compare_experiments(
    experiment_root: str,
    sort_by: str | None = None,
    descending: bool = True,
    *,
    metrics_config: str | None = None,
    include_diagnostics: bool = False,
) -> dict:
    """Compare experiment directories and return a JSON-safe payload."""
    if not isinstance(experiment_root, str):
        raise TypeError("experiment_root must be a string")
    if not experiment_root.strip():
        raise ValueError(
            "experiment_root must not be empty or whitespace"
        )

    if sort_by is not None:
        if not isinstance(sort_by, str):
            raise TypeError("sort_by must be a string or None")
        if not sort_by.strip():
            raise ValueError(
                "sort_by must not be empty or whitespace"
            )

    if metrics_config is not None:
        if not isinstance(metrics_config, str):
            raise TypeError(
                "metrics_config must be a string or None"
            )
        if not metrics_config.strip():
            raise ValueError(
                "metrics_config must not be empty or whitespace"
            )

    if not isinstance(descending, bool):
        raise TypeError("descending must be a boolean")
    if not isinstance(include_diagnostics, bool):
        raise TypeError("include_diagnostics must be a boolean")

    root_path = Path(experiment_root)
    metric_specs = (
        None
        if metrics_config is None
        else read_metric_specs_config(Path(metrics_config))
    )
    experiment_dirs = find_experiment_dirs(root_path)
    batch_result = analyze_experiment_dirs(
        experiment_dirs,
        metric_specs=metric_specs,
    )

    if sort_by is not None:
        resolved_sort_by = sort_by
    elif metric_specs is None:
        resolved_sort_by = "best_r2"
    else:
        resolved_sort_by = metric_specs[0].name

    payload = build_comparison_payload(
        batch_result,
        sort_by=resolved_sort_by,
        descending=descending,
        metric_specs=metric_specs,
        include_diagnostics=include_diagnostics,
    )
    json.dumps(
        payload,
        ensure_ascii=False,
        allow_nan=False,
    )
    return payload

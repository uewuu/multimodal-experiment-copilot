import argparse
import json
from collections.abc import Sequence
from pathlib import Path

from diagnostics import (
    build_comparison_diagnostics,
    build_comparison_facts,
    build_recommendations,
    diagnostic_to_dict,
    recommendation_to_dict,
)
from metrics import MetricSpec
from read_config import CONFIG_PATH
from read_history import HISTORY_PATH
from read_metrics_config import read_metric_specs_config
from summarize_experiment import build_experiment_summary


DEFAULT_EXPERIMENT_ROOT = Path("examples")
DEFAULT_COMPARISON_OUTPUT_PATH = Path(
    "outputs/comparison.json"
)
DEFAULT_COMPARISON_MARKDOWN_OUTPUT_PATH = Path(
    "outputs/comparison.md"
)

CONFIG_FILENAME = CONFIG_PATH.name
HISTORY_FILENAME = HISTORY_PATH.name

SORTABLE_COMPARISON_FIELDS = {
    "best_r2",
    "best_racc",
}


def _validate_metric_specs(
    metric_specs: Sequence[MetricSpec],
) -> tuple[MetricSpec, ...]:
    if (
        isinstance(metric_specs, (str, bytes, bytearray))
        or not isinstance(metric_specs, Sequence)
    ):
        raise TypeError("metric_specs must be a sequence")
    if not metric_specs:
        raise ValueError("metric_specs must not be empty")

    validated_specs: list[MetricSpec] = []
    name_indexes: dict[str, int] = {}

    for index, spec in enumerate(metric_specs):
        if not isinstance(spec, MetricSpec):
            raise TypeError(
                f"metric_specs[{index}] must be a MetricSpec"
            )

        if spec.name in name_indexes:
            first_index = name_indexes[spec.name]
            raise ValueError(
                "duplicate metric name "
                f"'{spec.name}' at metric_specs[{index}]; "
                f"first defined at metric_specs[{first_index}]"
            )

        name_indexes[spec.name] = index
        validated_specs.append(spec)

    if isinstance(metric_specs, tuple):
        return metric_specs
    return tuple(validated_specs)


def _resolve_default_sort_by(
    requested_sort_by: str | None,
) -> str:
    resolved_sort_by = (
        requested_sort_by
        if requested_sort_by is not None
        else "best_r2"
    )
    allowed_fields = ("best_r2", "best_racc")
    if resolved_sort_by not in SORTABLE_COMPARISON_FIELDS:
        raise ValueError(
            "sort_by must be one of: " + ", ".join(allowed_fields)
        )
    return resolved_sort_by


def _resolve_dynamic_sort_by(
    requested_sort_by: str | None,
    metric_specs: Sequence[MetricSpec],
) -> str:
    metric_names = tuple(spec.name for spec in metric_specs)
    resolved_sort_by = (
        requested_sort_by
        if requested_sort_by is not None
        else metric_names[0]
    )
    if resolved_sort_by not in metric_names:
        raise ValueError(
            "sort_by must be one of configured metric names: "
            + ", ".join(metric_names)
        )
    return resolved_sort_by


def find_experiment_dirs(root_dir: Path) -> list[Path]:
    """查找根目录下所有有效的实验子目录。"""
    if not root_dir.exists():
        raise FileNotFoundError(
            f"实验根目录不存在：{root_dir}"
        )

    if not root_dir.is_dir():
        raise NotADirectoryError(
            f"指定路径不是目录：{root_dir}"
        )

    experiment_dirs: list[Path] = []

    for candidate_dir in root_dir.iterdir():
        if not candidate_dir.is_dir():
            continue

        config_path = candidate_dir / CONFIG_FILENAME
        history_path = candidate_dir / HISTORY_FILENAME

        if config_path.is_file() and history_path.is_file():
            experiment_dirs.append(candidate_dir)

    return sorted(experiment_dirs)


def analyze_experiment_dirs(
    experiment_dirs: list[Path],
    metric_specs: Sequence[MetricSpec] | None = None,
) -> dict:
    """按传入顺序分析实验目录并收集成功与失败结果。"""
    validated_metric_specs = (
        None
        if metric_specs is None
        else _validate_metric_specs(metric_specs)
    )
    successful_experiments: list[dict] = []
    failed_experiments: list[dict] = []

    for experiment_dir in experiment_dirs:
        config_path = experiment_dir / CONFIG_FILENAME
        history_path = experiment_dir / HISTORY_FILENAME

        try:
            if validated_metric_specs is None:
                summary = build_experiment_summary(
                    config_path=config_path,
                    history_path=history_path,
                )
            else:
                summary = build_experiment_summary(
                    config_path=config_path,
                    history_path=history_path,
                    metric_specs=validated_metric_specs,
                )
        except Exception as error:
            failed_experiments.append(
                {
                    "experiment_name": experiment_dir.name,
                    "experiment_dir": str(experiment_dir),
                    "error_type": type(error).__name__,
                    "error_message": str(error),
                }
            )
            continue

        successful_experiments.append(
            {
                "experiment_name": experiment_dir.name,
                "experiment_dir": str(experiment_dir),
                "summary": summary,
            }
        )

    return {
        "successful_experiments": successful_experiments,
        "failed_experiments": failed_experiments,
    }


def build_comparison_records(
    batch_result: dict,
    metric_specs: Sequence[MetricSpec] | None = None,
) -> list[dict]:
    """从成功实验摘要中构建统一的实验对比记录。"""
    validated_metric_specs = (
        None
        if metric_specs is None
        else _validate_metric_specs(metric_specs)
    )
    comparison_records: list[dict] = []

    for experiment in batch_result["successful_experiments"]:
        metrics = experiment["summary"]["validation_metrics"]
        if validated_metric_specs is None:
            comparison_records.append(
                {
                    "experiment_name": experiment["experiment_name"],
                    "experiment_dir": experiment["experiment_dir"],
                    "best_r2": metrics["r2"]["best_value"],
                    "best_r2_epoch": metrics["r2"]["best_epoch"],
                    "best_racc": metrics["racc"]["best_value"],
                    "best_racc_epoch": metrics["racc"]["best_epoch"],
                }
            )
            continue

        dynamic_metrics: dict[str, dict] = {}
        for spec in validated_metric_specs:
            metric = metrics[spec.name]
            dynamic_metrics[spec.name] = {
                "record_count": metric["record_count"],
                "first_epoch": metric["first_epoch"],
                "first_value": metric["first_value"],
                "last_epoch": metric["last_epoch"],
                "last_value": metric["last_value"],
                "best_epoch": metric["best_epoch"],
                "best_value": metric["best_value"],
            }

        comparison_records.append(
            {
                "experiment_name": experiment["experiment_name"],
                "experiment_dir": experiment["experiment_dir"],
                "metrics": dynamic_metrics,
            }
        )

    return comparison_records


def rank_comparison_records(
    comparison_records: list[dict],
    sort_by: str = "best_r2",
    descending: bool = True,
    metric_specs: Sequence[MetricSpec] | None = None,
) -> list[dict]:
    """按指定指标对实验对比记录进行稳定排序。"""
    if metric_specs is None:
        if sort_by not in SORTABLE_COMPARISON_FIELDS:
            available_fields = ", ".join(sorted(SORTABLE_COMPARISON_FIELDS))
            raise ValueError(
                f"不支持的排序字段：{sort_by}；"
                f"可用字段：{available_fields}"
            )
        sort_key = lambda record: record[sort_by]
    else:
        validated_metric_specs = _validate_metric_specs(metric_specs)
        metric_names = tuple(
            spec.name for spec in validated_metric_specs
        )
        if sort_by not in metric_names:
            raise ValueError(
                "sort_by must be one of configured metric names: "
                + ", ".join(metric_names)
            )
        sort_key = lambda record: record["metrics"][sort_by]["best_value"]

    return sorted(
        comparison_records,
        key=sort_key,
        reverse=descending,
    )


def build_comparison_payload(
    batch_result: dict,
    sort_by: str = "best_r2",
    descending: bool = True,
    metric_specs: Sequence[MetricSpec] | None = None,
    *,
    include_diagnostics: bool = False,
) -> dict:
    """构建包含排序记录、实验数量和失败信息的对比载荷。"""
    if not isinstance(include_diagnostics, bool):
        raise TypeError("include_diagnostics must be a boolean")

    if metric_specs is None:
        comparison_records = build_comparison_records(batch_result)
        ranked_records = rank_comparison_records(
            comparison_records,
            sort_by=sort_by,
            descending=descending,
        )
        serialized_metric_specs = None
    else:
        validated_metric_specs = _validate_metric_specs(metric_specs)
        comparison_records = build_comparison_records(
            batch_result,
            metric_specs=validated_metric_specs,
        )
        ranked_records = rank_comparison_records(
            comparison_records,
            sort_by=sort_by,
            descending=descending,
            metric_specs=validated_metric_specs,
        )
        serialized_metric_specs = [
            {
                "name": spec.name,
                "path": list(spec.path),
                "direction": spec.direction,
                "display_name": spec.display_name,
                "precision": spec.precision,
            }
            for spec in validated_metric_specs
        ]

    successful_count = len(
        batch_result["successful_experiments"]
    )
    failed_count = len(
        batch_result["failed_experiments"]
    )
    total_count = successful_count + failed_count

    payload = {
        "sort_by": sort_by,
        "descending": descending,
        "experiment_counts": {
            "total": total_count,
            "successful": successful_count,
            "failed": failed_count,
        },
        "comparison_records": ranked_records,
        "failed_experiments": batch_result["failed_experiments"],
    }

    if serialized_metric_specs is not None:
        payload = {
            "sort_by": payload["sort_by"],
            "descending": payload["descending"],
            "metric_specs": serialized_metric_specs,
            "experiment_counts": payload["experiment_counts"],
            "comparison_records": payload["comparison_records"],
            "failed_experiments": payload["failed_experiments"],
        }

    if not include_diagnostics:
        return payload

    facts = build_comparison_facts(payload)
    diagnostics = build_comparison_diagnostics(facts)
    recommendations = build_recommendations(diagnostics)
    payload["diagnostics"] = {
        "facts": facts,
        "diagnostics": [
            diagnostic_to_dict(diagnostic)
            for diagnostic in diagnostics
        ],
        "recommendations": [
            recommendation_to_dict(recommendation)
            for recommendation in recommendations
        ],
    }
    return payload


def _escape_markdown_cell(value: object) -> str:
    """Escape a value for safe use in a Markdown table cell."""
    text = str(value)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    return text.replace("\n", "<br>").replace("|", "\\|")


def _append_comparison_diagnostics_section(
    report_lines: list[str],
    payload: dict,
) -> None:
    """Append comparison diagnostics when the payload includes them."""
    if "diagnostics" not in payload:
        return

    diagnostic_items = payload["diagnostics"]["diagnostics"]
    report_lines.extend(
        [
            "",
            "## Diagnostics",
            "",
            "| Severity | Code | Message | Evidence |",
            "| --- | --- | --- | --- |",
        ]
    )

    if not diagnostic_items:
        report_lines.append(
            "| — | — | No diagnostic findings were generated. | — |"
        )
        return

    for diagnostic in diagnostic_items:
        evidence_text = json.dumps(
            diagnostic["evidence"],
            ensure_ascii=False,
            sort_keys=True,
        )
        report_lines.append(
            "| "
            f"{_escape_markdown_cell(diagnostic['severity'])} | "
            f"`{_escape_markdown_cell(diagnostic['code'])}` | "
            f"{_escape_markdown_cell(diagnostic['message'])} | "
            f"`{_escape_markdown_cell(evidence_text)}` |"
        )


def _append_comparison_recommendations_section(
    report_lines: list[str],
    payload: dict,
) -> None:
    """Append deterministic recommendations when available."""
    diagnostic_payload = payload.get("diagnostics")
    if not isinstance(diagnostic_payload, dict):
        return
    if "recommendations" not in diagnostic_payload:
        return

    recommendation_items = diagnostic_payload["recommendations"]
    report_lines.extend(
        [
            "",
            "## Recommendations",
            "",
            "| Code | Recommendation | Triggered Diagnostics |",
            "| --- | --- | --- |",
        ]
    )

    if not recommendation_items:
        report_lines.append(
            "| — | No recommendations were generated. | — |"
        )
        return

    for recommendation in recommendation_items:
        trigger_text = ", ".join(
            recommendation["diagnostic_codes"]
        )
        report_lines.append(
            "| "
            f"`{_escape_markdown_cell(recommendation['code'])}` | "
            f"{_escape_markdown_cell(recommendation['message'])} | "
            f"`{_escape_markdown_cell(trigger_text)}` |"
        )


def build_comparison_markdown(payload: dict) -> str:
    """Build a Markdown report from a multi-experiment payload."""
    counts = payload["experiment_counts"]
    comparison_records = payload["comparison_records"]
    failed_experiments = payload["failed_experiments"]
    sort_direction = (
        "Descending" if payload["descending"] else "Ascending"
    )

    report_lines = [
        "# Multi-experiment Comparison Report",
        "",
        "## Overview",
        "",
        f'- Sort field: `{payload["sort_by"]}`',
        f"- Sort direction: {sort_direction}",
        f'- Total experiments: {counts["total"]}',
        f'- Successful experiments: {counts["successful"]}',
        f'- Failed experiments: {counts["failed"]}',
        "",
        "## Ranked Experiments",
        "",
    ]

    if comparison_records:
        if "metric_specs" in payload:
            metric_specs = payload["metric_specs"]
            header_cells = [
                "Rank",
                "Experiment",
                "Directory",
                *[
                    _escape_markdown_cell(spec_metadata["display_name"])
                    for spec_metadata in metric_specs
                ],
            ]
            separator_cells = [
                "---:",
                "---",
                "---",
                *["---:" for _ in metric_specs],
            ]
            report_lines.extend(
                [
                    "| " + " | ".join(header_cells) + " |",
                    "| " + " | ".join(separator_cells) + " |",
                ]
            )
            for rank, record in enumerate(comparison_records, start=1):
                metric_cells = []
                for spec_metadata in metric_specs:
                    metric = record["metrics"][spec_metadata["name"]]
                    formatted_value = (
                        f'{metric["best_value"]:.{spec_metadata["precision"]}f}'
                    )
                    metric_cells.append(
                        f'{formatted_value} (epoch {metric["best_epoch"]})'
                    )
                row_cells = [
                    str(rank),
                    _escape_markdown_cell(record["experiment_name"]),
                    _escape_markdown_cell(record["experiment_dir"]),
                    *metric_cells,
                ]
                report_lines.append(
                    "| " + " | ".join(row_cells) + " |"
                )
        else:
            report_lines.extend(
                [
                    "| Rank | Experiment | Directory | Best R² | R² Epoch | Best RACC | RACC Epoch |",
                    "| ---: | --- | --- | ---: | ---: | ---: | ---: |",
                ]
            )
            for rank, record in enumerate(comparison_records, start=1):
                report_lines.append(
                    "| "
                    f'{rank} | {_escape_markdown_cell(record["experiment_name"])} | '
                    f'{_escape_markdown_cell(record["experiment_dir"])} | '
                    f'{record["best_r2"]:.6f} | {record["best_r2_epoch"]} | '
                    f'{record["best_racc"]:.6f} | {record["best_racc_epoch"]} |'
                )
    else:
        report_lines.append("No successful experiments were analyzed.")

    if failed_experiments:
        report_lines.extend(
            [
                "",
                "## Failed Experiments",
                "",
                "| Experiment | Directory | Error Type | Error Message |",
                "| --- | --- | --- | --- |",
            ]
        )
        for experiment in failed_experiments:
            report_lines.append(
                "| "
                f'{_escape_markdown_cell(experiment["experiment_name"])} | '
                f'{_escape_markdown_cell(experiment["experiment_dir"])} | '
                f'{_escape_markdown_cell(experiment["error_type"])} | '
                f'{_escape_markdown_cell(experiment["error_message"])} |'
            )

    _append_comparison_diagnostics_section(report_lines, payload)
    _append_comparison_recommendations_section(report_lines, payload)
    return "\n".join(report_lines) + "\n"


def write_comparison_markdown(
    markdown_text: str,
    output_path: Path,
) -> Path:
    """将 Markdown 对比报告以 UTF-8 写入指定路径。"""
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    normalized_text = markdown_text.rstrip("\r\n") + "\n"

    with output_path.open(
        "w",
        encoding="utf-8",
    ) as file:
        file.write(normalized_text)

    return output_path


def write_comparison_json(
    payload: dict,
    output_path: Path,
) -> Path:
    """将实验对比载荷以 UTF-8 编码写入 JSON 文件。"""
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with output_path.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            payload,
            file,
            ensure_ascii=False,
            indent=2,
        )
        file.write("\n")

    return output_path


def run_comparison_pipeline(
    experiment_root: Path,
    output_path: Path,
    sort_by: str = "best_r2",
    descending: bool = True,
    markdown_output_path: Path | None = None,
    metric_specs: Sequence[MetricSpec] | None = None,
    *,
    include_diagnostics: bool = False,
) -> dict:
    """总是写入 JSON，可选写入 Markdown，并返回实验对比载荷。"""
    if not isinstance(include_diagnostics, bool):
        raise TypeError("include_diagnostics must be a boolean")

    validated_metric_specs = (
        None
        if metric_specs is None
        else _validate_metric_specs(metric_specs)
    )
    experiment_dirs = find_experiment_dirs(
        experiment_root
    )
    if validated_metric_specs is None:
        batch_result = analyze_experiment_dirs(
            experiment_dirs
        )
        if include_diagnostics:
            payload = build_comparison_payload(
                batch_result,
                sort_by=sort_by,
                descending=descending,
                include_diagnostics=True,
            )
        else:
            payload = build_comparison_payload(
                batch_result,
                sort_by=sort_by,
                descending=descending,
            )
    else:
        batch_result = analyze_experiment_dirs(
            experiment_dirs,
            metric_specs=validated_metric_specs,
        )
        if include_diagnostics:
            payload = build_comparison_payload(
                batch_result,
                sort_by=sort_by,
                descending=descending,
                metric_specs=validated_metric_specs,
                include_diagnostics=True,
            )
        else:
            payload = build_comparison_payload(
                batch_result,
                sort_by=sort_by,
                descending=descending,
                metric_specs=validated_metric_specs,
            )
    write_comparison_json(
        payload,
        output_path,
    )

    if markdown_output_path is not None:
        markdown_text = build_comparison_markdown(payload)
        write_comparison_markdown(
            markdown_text,
            markdown_output_path,
        )

    return payload


def parse_args(
    argv: list[str] | None = None,
) -> argparse.Namespace:
    """解析批量比较机器学习实验所需的命令行参数。"""
    parser = argparse.ArgumentParser(
        description="批量比较机器学习实验并生成 JSON 结果。"
    )
    parser.add_argument(
        "--experiment-root",
        type=Path,
        default=DEFAULT_EXPERIMENT_ROOT,
        help="包含多个实验目录的根目录",
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=DEFAULT_COMPARISON_OUTPUT_PATH,
        help="实验对比结果的 JSON 输出路径",
    )
    parser.add_argument(
        "--markdown-output-path",
        type=Path,
        default=DEFAULT_COMPARISON_MARKDOWN_OUTPUT_PATH,
        help="Markdown 对比报告输出路径",
    )
    parser.add_argument(
        "--metrics-config",
        type=Path,
        default=None,
        help="独立指标 YAML 配置文件路径",
    )
    parser.add_argument(
        "--sort-by",
        type=str,
        default=None,
        help=(
            "排序字段。默认模式未指定时使用 best_r2；"
            "动态指标模式未指定时使用配置中的第一个指标"
        ),
    )
    parser.add_argument(
        "--ascending",
        action="store_true",
        help="按指定指标升序排列",
    )
    parser.add_argument(
        "--include-diagnostics",
        action="store_true",
        help="在 JSON 和 Markdown 对比报告中加入规则诊断与建议",
    )

    return parser.parse_args(argv)


def main(
    argv: list[str] | None = None,
) -> None:
    args = parse_args(argv)

    metric_specs = None
    if args.metrics_config is None:
        resolved_sort_by = _resolve_default_sort_by(args.sort_by)
    else:
        metric_specs = read_metric_specs_config(args.metrics_config)
        resolved_sort_by = _resolve_dynamic_sort_by(
            args.sort_by,
            metric_specs,
        )

    try:
        if metric_specs is None:
            if args.include_diagnostics:
                payload = run_comparison_pipeline(
                    experiment_root=args.experiment_root,
                    output_path=args.output_path,
                    sort_by=resolved_sort_by,
                    descending=not args.ascending,
                    markdown_output_path=args.markdown_output_path,
                    include_diagnostics=True,
                )
            else:
                payload = run_comparison_pipeline(
                    experiment_root=args.experiment_root,
                    output_path=args.output_path,
                    sort_by=resolved_sort_by,
                    descending=not args.ascending,
                    markdown_output_path=args.markdown_output_path,
                )
        else:
            if args.include_diagnostics:
                payload = run_comparison_pipeline(
                    experiment_root=args.experiment_root,
                    output_path=args.output_path,
                    sort_by=resolved_sort_by,
                    descending=not args.ascending,
                    markdown_output_path=args.markdown_output_path,
                    metric_specs=metric_specs,
                    include_diagnostics=True,
                )
            else:
                payload = run_comparison_pipeline(
                    experiment_root=args.experiment_root,
                    output_path=args.output_path,
                    sort_by=resolved_sort_by,
                    descending=not args.ascending,
                    markdown_output_path=args.markdown_output_path,
                    metric_specs=metric_specs,
                )
    except (FileNotFoundError, NotADirectoryError) as error:
        raise SystemExit(
            f"实验比较失败：{error}"
        ) from error

    counts = payload["experiment_counts"]

    if counts["total"] == 0:
        print(f"没有找到有效实验目录：{args.experiment_root}")
        print(f"- JSON 输出：{args.output_path}")
        print(f"- Markdown 输出：{args.markdown_output_path}")
        if args.include_diagnostics:
            print("- 规则诊断：已启用")
        return

    sort_direction = "降序" if payload["descending"] else "升序"
    print("实验比较完成：")
    print(f"- 总实验数：{counts['total']}")
    print(f"- 成功实验：{counts['successful']}")
    print(f"- 失败实验：{counts['failed']}")
    print(f"- 排序方式：{payload['sort_by']}（{sort_direction}）")
    print(f"- JSON 输出：{args.output_path}")
    print(f"- Markdown 输出：{args.markdown_output_path}")
    if args.include_diagnostics:
        print("- 规则诊断：已启用")


if __name__ == "__main__":
    main()

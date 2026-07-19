import argparse
import json
from pathlib import Path

from read_config import CONFIG_PATH
from read_history import HISTORY_PATH
from summarize_experiment import build_experiment_summary


DEFAULT_EXPERIMENT_ROOT = Path("examples")
DEFAULT_COMPARISON_OUTPUT_PATH = Path(
    "outputs/comparison.json"
)

CONFIG_FILENAME = CONFIG_PATH.name
HISTORY_FILENAME = HISTORY_PATH.name

SORTABLE_COMPARISON_FIELDS = {
    "best_r2",
    "best_racc",
}


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
) -> dict:
    """按传入顺序分析实验目录并收集成功与失败结果。"""
    successful_experiments: list[dict] = []
    failed_experiments: list[dict] = []

    for experiment_dir in experiment_dirs:
        config_path = experiment_dir / CONFIG_FILENAME
        history_path = experiment_dir / HISTORY_FILENAME

        try:
            summary = build_experiment_summary(
                config_path=config_path,
                history_path=history_path,
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
) -> list[dict]:
    """从成功实验摘要中构建统一的实验对比记录。"""
    comparison_records: list[dict] = []

    for experiment in batch_result["successful_experiments"]:
        metrics = experiment["summary"]["validation_metrics"]
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

    return comparison_records


def rank_comparison_records(
    comparison_records: list[dict],
    sort_by: str = "best_r2",
    descending: bool = True,
) -> list[dict]:
    """按指定指标对实验对比记录进行稳定排序。"""
    if sort_by not in SORTABLE_COMPARISON_FIELDS:
        available_fields = ", ".join(sorted(SORTABLE_COMPARISON_FIELDS))
        raise ValueError(
            f"不支持的排序字段：{sort_by}；"
            f"可用字段：{available_fields}"
        )

    return sorted(
        comparison_records,
        key=lambda record: record[sort_by],
        reverse=descending,
    )


def build_comparison_payload(
    batch_result: dict,
    sort_by: str = "best_r2",
    descending: bool = True,
) -> dict:
    """构建包含排序记录、实验数量和失败信息的对比载荷。"""
    comparison_records = build_comparison_records(batch_result)
    ranked_records = rank_comparison_records(
        comparison_records,
        sort_by=sort_by,
        descending=descending,
    )

    successful_count = len(
        batch_result["successful_experiments"]
    )
    failed_count = len(
        batch_result["failed_experiments"]
    )
    total_count = successful_count + failed_count

    return {
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


def _escape_markdown_cell(value: object) -> str:
    """Escape a value for safe use in a Markdown table cell."""
    text = str(value)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    return text.replace("\n", "<br>").replace("|", "\\|")


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

    return "\n".join(report_lines) + "\n"


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
) -> dict:
    """执行实验发现、分析、载荷构建和 JSON 写入流程。"""
    experiment_dirs = find_experiment_dirs(
        experiment_root
    )
    batch_result = analyze_experiment_dirs(
        experiment_dirs
    )
    payload = build_comparison_payload(
        batch_result,
        sort_by=sort_by,
        descending=descending,
    )
    write_comparison_json(
        payload,
        output_path,
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
        "--sort-by",
        choices=sorted(SORTABLE_COMPARISON_FIELDS),
        default="best_r2",
        help="实验对比记录的排序指标",
    )
    parser.add_argument(
        "--ascending",
        action="store_true",
        help="按指定指标升序排列",
    )

    return parser.parse_args(argv)


def main(
    argv: list[str] | None = None,
) -> None:
    args = parse_args(argv)

    try:
        payload = run_comparison_pipeline(
            experiment_root=args.experiment_root,
            output_path=args.output_path,
            sort_by=args.sort_by,
            descending=not args.ascending,
        )
    except (FileNotFoundError, NotADirectoryError) as error:
        raise SystemExit(
            f"实验比较失败：{error}"
        ) from error

    counts = payload["experiment_counts"]

    if counts["total"] == 0:
        print(f"没有找到有效实验目录：{args.experiment_root}")
        print(f"- JSON 输出：{args.output_path}")
        return

    sort_direction = "降序" if payload["descending"] else "升序"
    print("实验比较完成：")
    print(f"- 总实验数：{counts['total']}")
    print(f"- 成功实验：{counts['successful']}")
    print(f"- 失败实验：{counts['failed']}")
    print(f"- 排序方式：{payload['sort_by']}（{sort_direction}）")
    print(f"- JSON 输出：{args.output_path}")


if __name__ == "__main__":
    main()

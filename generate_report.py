from pathlib import Path
import json

from read_config import CONFIG_PATH
from read_history import HISTORY_PATH
from summarize_experiment import build_experiment_summary


OUTPUT_DIR = Path("outputs")
SUMMARY_JSON_FILENAME = "experiment_summary.json"
REPORT_MD_FILENAME = "experiment_report.md"


def write_summary_json(summary: dict, output_path: Path) -> None:
    """将结构化实验摘要写入 JSON 文件。"""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as file:
        json.dump(
            summary,
            file,
            ensure_ascii=False,
            indent=2,
        )


def build_markdown_report(summary: dict) -> str:
    """根据结构化摘要生成 Markdown 实验报告。"""
    configuration = summary["configuration"]
    validation_metrics = summary["validation_metrics"]

    r2_summary = validation_metrics["r2"]
    racc_summary = validation_metrics["racc"]

    feature_list = configuration.get("feature_list") or []
    feature_text = "、".join(map(str, feature_list))

    # 先取得较长的配置字段，避免在 f-string 中跨行调用函数。
    use_mtf = configuration.get("use_modality_token_fusion")
    use_behavior_token = configuration.get("use_behavior_state_token")
    use_behavior_cl = configuration.get("use_behavior_aware_cl")
    use_regression_cl = configuration.get("use_regression_aware_cl")
    use_tcms = configuration.get(
        "use_trait_conditioned_modality_selection"
    )

    report_lines = [
        "# 多模态实验摘要",
        "",
        "## 1. 实验配置",
        "",
        f"- Batch Size：{configuration.get('batch_size')}",
        f"- Num Workers：{configuration.get('num_workers')}",
        f"- Seed：{configuration.get('seed')}",
        f"- Sample Seed：{configuration.get('sample_seed')}",
        f"- 配置训练轮数：{configuration.get('n_epochs')}",
        f"- 输入模态：{feature_text}",
        f"- Attention Type：{configuration.get('attention_type')}",
        "",
        "## 2. 模块开关",
        "",
        f"- Modality Token Fusion：{use_mtf}",
        f"- Behavior State Token：{use_behavior_token}",
        f"- Behavior-aware CL：{use_behavior_cl}",
        f"- Regression-aware CL：{use_regression_cl}",
        f"- Trait-conditioned Modality Selection：{use_tcms}",
        "",
        "## 3. 验证集指标",
        "",
        "| 指标 | 记录数 | 最佳值 | 最佳 Epoch | 最后一轮值 | 最后一轮 Epoch |",
        "|---|---:|---:|---:|---:|---:|",
        (
            f"| R² | {r2_summary['record_count']} "
            f"| {r2_summary['best_value']:.6f} "
            f"| {r2_summary['best_epoch']} "
            f"| {r2_summary['last_value']:.6f} "
            f"| {r2_summary['last_epoch']} |"
        ),
        (
            f"| RACC | {racc_summary['record_count']} "
            f"| {racc_summary['best_value']:.6f} "
            f"| {racc_summary['best_epoch']} "
            f"| {racc_summary['last_value']:.6f} "
            f"| {racc_summary['last_epoch']} |"
        ),
        "",
        "## 4. 自动分析",
        "",
        (
            f"- 验证日志共包含 "
            f"{r2_summary['record_count']} 条 Epoch 记录。"
        ),
        (
            f"- 日志记录范围为 Epoch "
            f"{r2_summary['first_epoch']} 至 "
            f"{r2_summary['last_epoch']}。"
        ),
        (
            f"- 最佳验证 R² 为 "
            f"{r2_summary['best_value']:.6f}，"
            f"出现在 Epoch {r2_summary['best_epoch']}。"
        ),
        (
            f"- 最佳验证 RACC 为 "
            f"{racc_summary['best_value']:.6f}，"
            f"出现在 Epoch {racc_summary['best_epoch']}。"
        ),
    ]

    if r2_summary["best_epoch"] == racc_summary["best_epoch"]:
        report_lines.append(
            f"- R² 与 RACC 均在 Epoch "
            f"{r2_summary['best_epoch']} 达到最佳值。"
        )
    else:
        report_lines.append(
            "- R² 与 RACC 的最佳 Epoch 不一致，"
            "选择检查点时需要明确主要监控指标。"
        )

    configured_epochs = configuration.get("n_epochs")
    actual_last_epoch = r2_summary["last_epoch"]

    if (
        isinstance(configured_epochs, int)
        and actual_last_epoch + 1 < configured_epochs
    ):
        report_lines.append(
            f"- 配置计划训练 {configured_epochs} 个 Epoch，"
            f"但日志仅记录到 Epoch {actual_last_epoch}；"
            "可能由早停、手动中断或其他训练终止条件造成。"
        )

    return "\n".join(report_lines) + "\n"


def write_markdown_report(
    report_content: str,
    output_path: Path,
) -> None:
    """将 Markdown 报告写入文件。"""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as file:
        file.write(report_content)


def generate_experiment_report(
    config_path: Path = CONFIG_PATH,
    history_path: Path = HISTORY_PATH,
    output_dir: Path = OUTPUT_DIR,
) -> tuple[Path, Path]:
    """读取实验文件并生成 JSON 摘要和 Markdown 报告。"""
    summary = build_experiment_summary(
        config_path=config_path,
        history_path=history_path,
    )
    report_content = build_markdown_report(summary)

    summary_json_path = output_dir / SUMMARY_JSON_FILENAME
    report_md_path = output_dir / REPORT_MD_FILENAME

    write_summary_json(summary, summary_json_path)
    write_markdown_report(report_content, report_md_path)

    return summary_json_path, report_md_path


def main() -> None:
    summary_json_path, report_md_path = generate_experiment_report()

    print("实验报告生成完成")
    print("-" * 50)
    print(f"JSON 摘要：{summary_json_path}")
    print(f"Markdown 报告：{report_md_path}")


if __name__ == "__main__":
    main()
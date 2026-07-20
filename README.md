# Multimodal Experiment Copilot

A lightweight experiment analysis tool for reading multimodal machine learning
configurations and training histories, extracting key validation metrics, and
generating structured JSON summaries and Markdown reports.

中文名称：多模态实验分析智能体。

项目定位：面向机器学习研发团队的通用实验分析与决策 Copilot。

## Overview

Multimodal machine learning experiments often produce configuration files,
training histories, validation metrics, and multiple output artifacts.

This project provides a reproducible workflow for:

1. Reading experiment configurations from YAML.
2. Reading training histories from JSON.
3. Extracting validation R² and RACC metrics.
4. Identifying the best value and corresponding epoch.
5. Building a structured experiment summary.
6. Exporting JSON and Markdown reports.
7. Running the workflow through a command-line interface.

The project is being developed incrementally as a portfolio project for
AI Agent, LLM application, RAG, and AI backend engineering roles.

## Current Features

- Read `hparams.yaml` experiment configurations.
- Read `history.json` training histories.
- Validate metric records in `[epoch, value]` format.
- Analyze validation R² and RACC.
- Calculate:
  - record count;
  - first epoch and value;
  - last epoch and value;
  - best epoch and value.
- Build a structured experiment summary.
- Generate:
  - `experiment_summary.json`;
  - `experiment_report.md`.
- Support custom experiment and output directories.
- Provide friendly errors when required files are missing.
- Provide a command-line help interface with `argparse`.
- Discover valid experiment directories under a shared root directory.
- Analyze multiple experiments in one run.
- Isolate per-experiment failures without interrupting the full batch.
- Build normalized comparison records.
- Sort comparison results by best R² or best RACC.
- Export UTF-8 JSON comparison results.
- Generate a human-readable multi-experiment Markdown report.
- Provide a configurable multi-experiment comparison CLI.
- Define general experiment metrics in an independent YAML file.
- Select per-experiment best values with `maximize` or `minimize` semantics.
- Generate dynamic JSON and Markdown comparisons for configured metrics.
- Cover the comparison workflow with automated pytest tests.
- Validate the full pytest suite automatically with GitHub Actions.
- Run CI with Python 3.11 on `ubuntu-latest`.
- Run automatic checks for pull requests targeting `main` and pushes to
  `main`.

## Project Structure

```text
multimodal-experiment-copilot/
├── .github/
│   └── workflows/
│       └── tests.yml
├── configs/
│   └── metrics.example.yaml
├── examples/
│   └── demo_experiment/
│       ├── history.json
│       └── hparams.yaml
├── notes/
│   └── day01.md
├── outputs/
│   ├── comparison.json
│   └── comparison.md
├── tests/
│   └── test_compare_experiments.py
├── .gitignore
├── README.md
├── compare_experiments.py
├── generate_report.py
├── metrics.py
├── read_config.py
├── read_history.py
├── read_metrics_config.py
├── requirements.txt
└── summarize_experiment.py
```

The `outputs/` directory is ignored by Git because it contains automatically
generated files.

## Requirements

- Python 3.11 or later
- PyYAML
- pytest for development and testing

Runtime dependency:

```bash
pip install pyyaml
```

Test dependency:

```bash
pip install pytest
```

## Continuous Integration

The `Tests` GitHub Actions workflow automatically validates the project when:

- a pull request targets `main`;
- a commit is pushed to `main`.

The workflow runs on `ubuntu-latest` with Python 3.11, installs dependencies
from `requirements.txt`, and executes the complete test suite with:

```bash
python -m pytest ./tests -q
```

The GitHub-hosted workflow has been verified successfully for the current
version, including a passing pull request check.

## Usage

### 1. Display CLI help

```bash
python generate_report.py --help
```

### 2. Run with default paths

```bash
python generate_report.py
```

Default experiment directory:

```text
examples/demo_experiment
```

Default output directory:

```text
outputs
```

### 3. Specify experiment and output directories

```bash
python generate_report.py --experiment-dir examples/demo_experiment --output-dir outputs/demo_experiment
```

The experiment directory must contain:

```text
hparams.yaml
history.json
```

## Multi-experiment Comparison / 多实验比较

### 实验目录要求

实验根目录的直接子目录只有同时包含以下文件时才会被识别：

- `hparams.yaml`
- `history.json`

例如：

```text
examples/
├── experiment_a/
│   ├── hparams.yaml
│   └── history.json
└── experiment_b/
    ├── hparams.yaml
    └── history.json
```

当前只扫描实验根目录的直接子目录，不递归扫描更深层目录。

### 默认运行命令

```bash
python compare_experiments.py
```

默认值：

- 实验根目录：`examples`
- JSON 输出：`outputs/comparison.json`
- Markdown 输出：`outputs/comparison.md`
- 排序字段：`best_r2`
- 排序方向：降序

### 自定义运行示例

PowerShell：

```powershell
python compare_experiments.py `
  --experiment-root examples `
  --output-path outputs/custom_comparison.json `
  --markdown-output-path outputs/custom_comparison.md `
  --sort-by best_racc `
  --ascending
```

### 参数说明

- `--experiment-root`：包含多个实验目录的根目录。
- `--output-path`：JSON 输出文件路径。
- `--markdown-output-path`：Markdown 对比报告输出路径。
- `--metrics-config`：独立指标 YAML 配置文件路径；提供时启用动态指标模式。
- `--sort-by`：默认模式使用 `best_r2` 或 `best_racc`；动态模式使用配置中的指标 `name`。
- `--ascending`：出现时使用升序；未出现时使用降序。

### JSON 输出结构

```json
{
  "sort_by": "best_r2",
  "descending": true,
  "experiment_counts": {
    "total": 2,
    "successful": 1,
    "failed": 1
  },
  "comparison_records": [
    {
      "experiment_name": "experiment_a",
      "experiment_dir": "examples/experiment_a",
      "best_r2": 0.72,
      "best_r2_epoch": 18,
      "best_racc": 0.94,
      "best_racc_epoch": 20
    }
  ],
  "failed_experiments": [
    {
      "experiment_name": "experiment_b",
      "experiment_dir": "examples/experiment_b",
      "error_type": "ValueError",
      "error_message": "history.json 数据结构无效"
    }
  ]
}
```

### Markdown 报告内容

多实验 Markdown 报告是 JSON 对比结果的人类可读版本，至少包含：

- `Overview`；
- 排序字段和排序方向；
- 实验总数、成功数和失败数；
- `Ranked Experiments` 排名表格；
- 每个成功实验的 Best R²、对应 epoch、Best RACC 和对应 epoch；
- 存在失败实验时的 `Failed Experiments` 章节。

### 错误隔离行为

- 单个实验分析失败时不会中断其他实验。
- 失败实验会记录在 `failed_experiments` 中。
- 实验根目录不存在或根路径不是目录时，程序会退出并显示错误。

### 测试

```powershell
python -m pytest .\tests -v
```

当前版本已使用上述命令完成本地测试验证。

## Configurable Metrics / 可配置指标

多实验比较支持显式加载通用指标定义。提供 `--metrics-config` 时，程序会从
独立 YAML 文件读取指标，并为每个实验生成动态 JSON 记录和 Markdown 列；未提供
该参数时，程序继续使用原有 R²/RACC 兼容模式。

示例配置位于 [`configs/metrics.example.yaml`](configs/metrics.example.yaml)。

### 默认兼容模式

未提供 `--metrics-config` 时，默认比较字段仍为 `best_r2` 和 `best_racc`。
默认排序为 `sort_by=best_r2`、`descending=True`，动态模式不会替代此行为。

PowerShell：

```powershell
& "C:\Users\admin\.conda\envs\agent311\python.exe" `
  .\compare_experiments.py `
  --experiment-root .\examples `
  --output-path .\outputs\comparison.json `
  --markdown-output-path .\outputs\comparison.md `
  --sort-by best_r2
```

也可以省略 `--sort-by`，此时仍使用 `best_r2`。默认 comparison record 结构为：

```json
{
  "experiment_name": "experiment_a",
  "experiment_dir": "examples/experiment_a",
  "best_r2": 0.48,
  "best_r2_epoch": 32,
  "best_racc": 0.91,
  "best_racc_epoch": 30
}
```

默认 Markdown 排名表使用固定表头：

```text
| Rank | Experiment | Directory | Best R² | R² Epoch | Best RACC | RACC Epoch |
```

### 指标 YAML schema

独立配置文件的顶层只能包含非空的 `metrics` 序列。例如：

```yaml
metrics:
  - name: accuracy
    path:
      - validation
      - metrics
      - accuracy
    direction: maximize
    display_name: Accuracy
    precision: 4

  - name: validation_loss
    path:
      - validation
      - metrics
      - loss
    direction: minimize
    display_name: Validation Loss
    precision: 6
```

每个指标包含以下字段：

- `name`：稳定且唯一的指标标识符，用作 JSON key 和 `--sort-by` 值；它不是
  Markdown 显示标题。
- `path`：`history.json` 中指标历史序列的嵌套字符串键路径，YAML 中必须写成
  字符串列表。每个被比较实验都必须在该路径提供指标历史。
- `direction`：只允许 `maximize` 或 `minimize`，用于选择单个实验内部的
  `best_value`。
- `display_name`：Markdown 表头使用的可读文本，可以是中文，也可以包含需要
  Markdown 转义的字符；不能用作 `--sort-by`。
- `precision`：Markdown 中 `best_value` 的小数位数，默认为 `6`，必须是非负
  整数；它不改变 JSON 原始数值，也不表示百分比。

例如：

```yaml
path:
  - validation
  - metrics
  - accuracy
```

对应：

```python
history["validation"]["metrics"]["accuracy"]
```

`path` 不支持点号字符串、JSONPath、通配符、数组索引、自动搜索或指标发现。

`maximize` 表示在单个实验的历史记录中选择最大值；`minimize` 表示选择最小值。
`direction` 只控制实验内部的最佳值选择，不会自动决定跨实验排序方向。

### 动态 CLI

PowerShell：

```powershell
& "C:\Users\admin\.conda\envs\agent311\python.exe" `
  .\compare_experiments.py `
  --experiment-root .\examples `
  --metrics-config .\configs\metrics.example.yaml `
  --sort-by validation_loss `
  --ascending `
  --output-path .\outputs\comparison.json `
  --markdown-output-path .\outputs\comparison.md
```

- `--metrics-config` 启用动态指标模式。
- `--sort-by` 必须使用配置中的 `name`，不能使用 `display_name`。
- `--ascending` 表示跨实验按所选指标的 `best_value` 从小到大排序；未提供时
  从大到小排序。
- `direction=minimize` 不会自动启用升序。比较 loss 时通常需要显式添加
  `--ascending`。
- 动态模式省略 `--sort-by` 时，使用 YAML 中第一个指标的 `name`。

### 动态 JSON 输出

以下是省略 `experiment_counts` 和 `failed_experiments` 的精简 payload 示例：

```json
{
  "sort_by": "validation_loss",
  "descending": false,
  "metric_specs": [
    {
      "name": "validation_loss",
      "path": [
        "validation",
        "metrics",
        "loss"
      ],
      "direction": "minimize",
      "display_name": "Validation Loss",
      "precision": 6
    }
  ],
  "comparison_records": [
    {
      "experiment_name": "experiment_a",
      "experiment_dir": "examples/experiment_a",
      "metrics": {
        "validation_loss": {
          "record_count": 20,
          "first_epoch": 0,
          "first_value": 0.8,
          "last_epoch": 19,
          "last_value": 0.25,
          "best_epoch": 18,
          "best_value": 0.24
        }
      }
    }
  ]
}
```

JSON 保留指标历史评估结果中的记录数量、首尾值和最佳值，不受 `precision`
格式化影响。

### 动态 Markdown 输出

动态 Markdown 为 YAML 中每个指标生成一列，表头使用 `display_name`，列顺序与
`metrics` 顺序一致。单元格格式为 `<best_value> (epoch <best_epoch>)`，小数
位数由 `precision` 控制。例如：

```text
| Rank | Experiment | Directory | Validation Loss |
| ---: | --- | --- | ---: |
| 1 | experiment_a | examples/experiment_a | 0.240000 (epoch 18) |
```

`comparison_records` 的已有排序顺序直接用于 Rank。Markdown 不展示
`first_value`、`last_value` 或 `record_count`。

### 配置错误与当前限制

- YAML 顶层只能包含 `metrics`，且 `metrics` 必须是非空序列。
- 同一配置中的指标 `name` 必须唯一。
- `path` 必须与实际 `history.json` 一致；所有被比较实验都应提供配置中的指标
  路径。单个实验缺少路径时会作为失败实验隔离记录。
- 动态 `--sort-by` 必须使用配置中的 `name`；默认模式只允许 `best_r2` 或
  `best_racc`。
- 当前不支持 JSONPath、自动发现指标、由 `direction` 自动决定跨实验排序、
  图表生成或 Web UI。

## Generated Outputs

### Single-experiment report

After a successful `generate_report.py` run, the output directory contains:

```text
experiment_summary.json
experiment_report.md
```

The Markdown report currently includes:

1. Experiment configuration.
2. Module switches.
3. Validation metric table.
4. Automatic experiment analysis.

### Multi-experiment comparison

By default, `compare_experiments.py` generates:

```text
outputs/comparison.json
outputs/comparison.md
```

The comparison JSON contains:

1. The selected sort field and sort direction.
2. Total, successful, and failed experiment counts.
3. Sorted experiment metric records.
4. Failed experiments and their error details.

The human-readable Markdown report contains:

1. An overview of sorting and experiment counts.
2. A ranked experiment table with best R² and RACC values and epochs in
   default mode, or configured metric columns in dynamic mode.
3. A failed-experiment table when failures are present.

## Example Analysis

The demo experiment report identifies:

- the best validation R²;
- the best validation RACC;
- their corresponding epochs;
- the last recorded epoch;
- whether the configured epoch count differs from the actual log length.

When the log ends earlier than the configured training length, the report uses
an objective statement:

```text
The run may have ended because of early stopping, manual interruption,
or another termination condition.
```

It does not assume that early stopping was definitely responsible.

## Development History

The project has been developed through small, verifiable Git commits:

1. Add YAML configuration reader.
2. Analyze validation R² history.
3. Generalize validation metric analysis.
4. Build structured experiment summary.
5. Generate JSON and Markdown reports.
6. Parameterize experiment input paths.
7. Parameterize report output paths.
8. Add command-line report interface.
9. Discover valid experiment directories.
10. Add batch experiment analysis with failure isolation.
11. Build normalized experiment comparison records.
12. Add metric-based comparison sorting.
13. Generate structured comparison JSON output.
14. Add an end-to-end comparison pipeline.
15. Add the multi-experiment comparison CLI.
16. Add automated tests for the comparison workflow.
17. Document multi-experiment comparison usage.
18. Build and write the multi-experiment Markdown comparison report.
19. Integrate Markdown output into the comparison pipeline and CLI.

The complete evolution is available in the repository commit history.

## Roadmap

- [x] YAML configuration reader
- [x] JSON training history reader
- [x] Validation R² analysis
- [x] Generalized metric analysis
- [x] Structured experiment summary
- [x] JSON report generation
- [x] Markdown report generation
- [x] Configurable input and output paths
- [x] Command-line interface
- [x] Multi-experiment batch analysis
- [x] Experiment comparison tables
- [x] Multi-experiment Markdown comparison report
- [x] Configurable experiment metrics from YAML
- [ ] Trait-wise metric summaries
- [ ] Configuration and schema validation
- [x] Automated tests with `pytest`
- [x] Automated pytest validation with GitHub Actions
- [ ] LLM Tool Calling
- [ ] LangGraph workflow
- [ ] RAG support
- [ ] MCP integration
- [ ] FastAPI service
- [ ] Human-in-the-loop review
- [ ] Docker deployment
- [ ] Web interface

## Planned Architecture

```text
Experiment Files
      ↓
Configuration and History Readers
      ↓
Validation and Metric Analysis
      ↓
Structured Experiment Summary
      ↓
JSON / Markdown Reports
      ↓
Multi-experiment Comparison
      ↓
LLM and Agent Tool Interface
```

## Notes

The files under `examples/` are demonstration inputs. Local dataset paths,
credentials, API keys, and private experiment data should not be committed.

## License

A license has not yet been selected.

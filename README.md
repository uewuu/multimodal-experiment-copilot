# Multimodal Experiment Copilot

A lightweight experiment analysis tool for reading multimodal machine learning
configurations and training histories, extracting key validation metrics, and
generating structured JSON summaries and Markdown reports.

中文名称：多模态实验分析智能体。

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
- Cover the comparison workflow with automated pytest tests.

## Project Structure

```text
multimodal-experiment-copilot/
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
├── read_config.py
├── read_history.py
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
- `--sort-by`：排序字段，可选 `best_r2` 或 `best_racc`。
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
2. A ranked experiment table with best R² and RACC values and epochs.
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
- [ ] Trait-wise metric summaries
- [ ] Configuration and schema validation
- [x] Automated tests with `pytest`
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

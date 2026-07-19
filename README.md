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
├── .gitignore
├── README.md
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

Install the current dependency with:

```bash
pip install pyyaml
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

## 多实验比较

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
- 排序字段：`best_r2`
- 排序方向：降序

### 自定义运行示例

PowerShell：

```powershell
python compare_experiments.py `
  --experiment-root examples `
  --output-path outputs/comparison.json `
  --sort-by best_racc `
  --ascending
```

### 参数说明

- `--experiment-root`：包含多个实验目录的根目录。
- `--output-path`：JSON 输出文件路径。
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

### 错误隔离行为

- 单个实验分析失败时不会中断其他实验。
- 失败实验会记录在 `failed_experiments` 中。
- 实验根目录不存在或根路径不是目录时，程序会退出并显示错误。

### 测试

```powershell
python -m pytest .\tests -v
```

当前版本的本地验证记录为 64 个测试通过。

## Generated Outputs

After a successful run, the output directory contains:

```text
experiment_summary.json
experiment_report.md
```

The Markdown report currently includes:

1. Experiment configuration.
2. Module switches.
3. Validation metric table.
4. Automatic experiment analysis.

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
- [ ] Multi-experiment batch analysis
- [ ] Experiment comparison tables
- [ ] Trait-wise metric summaries
- [ ] Configuration and schema validation
- [ ] Automated tests with `pytest`
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

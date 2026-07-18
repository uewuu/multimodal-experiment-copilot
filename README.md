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
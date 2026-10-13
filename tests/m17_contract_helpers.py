"""Test-only M17 vocabulary and generic fixtures; no comparison implementation.

Core: experiment_comparison.compare_experiment_evidence(a, b, *, targets).
Targets are a sequence of mappings with exactly target_id/a/b. Each side has
section, metric_name, metric_spec (MetricSpec or None), provenance
(ExperimentProvenance), metric_definition, aggregation, evaluation_protocol,
selection_protocol. No caller-supplied metric value is accepted.

Tool: ExperimentComparisonTools(experiment_a, experiment_b, *, options_a=None,
options_b=None, targets). Each options mapping accepts only M16 config_paths,
test_metrics_file, test_metric_specs and metrics_config (experiment-relative).
Dynamic validation specs supplement the detached targets at invocation time;
the source metrics config remains subject to M16 containment checks.

Output: experiment-comparison-v1, pair, targets keyed by target_id, and
configuration_differences. Target alignment has status aligned/blocked and
reason_codes. Calculation has status calculated/blocked, raw_delta, direction,
direction_interpretation. Comparability is the unchanged M11 projection or
None with not_evaluated_reason. Sources retain section/evidence_kind/file,
metadata_source=host_declared and original Test provenance. Limitations are
stable code strings, including M11 missing reasons and
checkpoint_binding_not_verified/statistical_significance_not_assessed.
Configuration rows contain path, a, b (original projected field or None), and
relation equal/different/unknown. Unknown fields never establish equality.
JSON size uses UTF-8, ensure_ascii=False, allow_nan=False, compact separators.
"""

from copy import deepcopy
import importlib
import importlib.util
import json
from types import SimpleNamespace as NS

import yaml

from experiment_identity import ExperimentIdentity, ExecutionIdentity, ExperimentProvenance
from metrics import MetricSpec


TOOL = "compare_experiment_evidence"
UNKNOWN = {
    "status": "unknown", "checkpoint_identity": None,
    "checkpoint_binding": "not_verified", "dataset_split_identity": None,
    "recomputed_from_predictions": False,
}


def core_api():
    assert importlib.util.find_spec("experiment_comparison") is not None, (
        "missing M17 capability: experiment_comparison.compare_experiment_evidence")
    function = getattr(importlib.import_module("experiment_comparison"), TOOL, None)
    assert callable(function), "missing M17 capability: compare_experiment_evidence"
    return function


def tool_class():
    cls = getattr(importlib.import_module("tool_layer.experiment_tools"),
                  "ExperimentComparisonTools", None)
    assert callable(cls), "missing M17 capability: ExperimentComparisonTools"
    return cls


def spec(name="score", direction="maximize", section="test", precision=6):
    path = ("measurement",) if section == "test" else ("validation", "measurement")
    return MetricSpec(name, path, direction, name.title(), precision)


def provenance(side, section="test"):
    return ExperimentProvenance(
        ExecutionIdentity(ExperimentIdentity("synthetic-lab", side), "run-1"),
        task="regression", target="quality", dataset_version="dataset-v1",
        split_id="heldout-v1" if section == "test" else "validation-v1",
    )


def target(target_id="quality", *, section="test", direction="maximize"):
    def side(label):
        return {
            "section": section, "metric_name": "score",
            "metric_spec": spec(direction=direction, section=section),
            "provenance": provenance(label, section),
            "metric_definition": "quality-score-v1", "aggregation": "macro",
            "evaluation_protocol": "evaluation-v1", "selection_protocol": "selection-v1",
        }
    return {"target_id": target_id, "a": side("A"), "b": side("B")}


def evidence(value=0.5, *, direction="maximize", name="score"):
    """M16-shaped evidence; fixture fidelity is independently tested against M16."""
    return {
        "configuration": {},
        "configuration_evidence": {
            "schema_version": "configuration-evidence-v1", "source_file": "hparams.yaml",
            "root_path": ["config"], "fields": [],
        },
        "validation_evidence": {
            "evidence_kind": "validation_history", "source": {"file": "history.json"},
        },
        "validation_metrics": {name: {
            "metric_name": name, "record_count": 2, "first_epoch": 0,
            "first_value": value, "last_epoch": 1, "last_value": value,
            "best_epoch": 0, "best_value": value,
        }},
        "test_results": {
            "schema_version": "test-results-v1", "status": "available",
            "evidence_kind": "csv_reported",
            "source": {"file": "heldout.csv", "data_row": 1, "selection": "exactly_one_data_row"},
            "metrics": {name: {"value": value, "source_column": "measurement",
                               "direction": direction, "display_name": name.title(), "precision": 6}},
            "issues": [], "provenance": deepcopy(UNKNOWN),
        },
    }


def config_field(value=None, *, status="present", path=("enabled",)):
    result = {"path": list(path),
              "evidence_kind": "configuration_declared", "status": status}
    if status == "present":
        result["value"] = value
    elif status == "unavailable":
        result["reason"] = "unsupported_value_type"
    return result


def write_experiment(parent, name, *, value=0.5, csv=True):
    root = parent / name
    root.mkdir()
    (root / "hparams.yaml").write_text(yaml.safe_dump({"config": {"enabled": False}}), encoding="utf-8")
    (root / "history.json").write_text(json.dumps({"validation": {
        "measurement": [[0, value], [1, value]]}}), encoding="utf-8")
    s = spec(section="validation")
    (root / "metrics.yaml").write_text(yaml.safe_dump({"metrics": [{
        "name": s.name, "path": list(s.path), "direction": s.direction,
        "display_name": s.display_name, "precision": s.precision}]}), encoding="utf-8")
    if csv:
        (root / "heldout.csv").write_text(f"measurement\n{value}\n", encoding="utf-8")
    return root


def options():
    return {"config_paths": [["enabled"]], "metrics_config": "metrics.yaml",
            "test_metrics_file": "heldout.csv", "test_metric_specs": [spec()]}


def arguments(a, b):
    return {"experiment_a": str(a), "experiment_b": str(b)}


class Client:
    """Provider double only. It never replaces a tool result or reads files."""
    def __init__(self, args, *, final="offline answer", error=None):
        self.args, self.final, self.error = args, final, error
        self.requests = []
        self.chat = NS(completions=self)

    def create(self, **kwargs):
        self.requests.append(deepcopy(kwargs))
        if len(self.requests) == 1:
            message = NS(role="assistant", content=None, tool_calls=[NS(
                id="comparison-call", type="function", function=NS(
                    name=TOOL, arguments=json.dumps(self.args)))])
        elif len(self.requests) == 2:
            if self.error is not None:
                raise self.error
            message = NS(role="assistant", content=self.final, tool_calls=None)
        else:
            raise AssertionError("unexpected provider retry")
        return NS(choices=[NS(message=message)])


def assert_blocked(row):
    assert row["calculation"]["status"] == "blocked"
    assert row["calculation"]["raw_delta"] is None
    assert row["calculation"]["direction_interpretation"] is None


def assert_no_winner(result):
    if isinstance(result, dict):
        assert not {"winner", "best_experiment", "recommended_experiment",
                    "global_comparable", "statistically_significant"}.intersection(result)
        for value in result.values():
            assert_no_winner(value)
    elif isinstance(result, list):
        for value in result:
            assert_no_winner(value)

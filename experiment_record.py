"""Opt-in adaptation of existing experiment summaries and explicit provenance.

Parsing and best-metric selection remain owned by existing experiment modules.
This adapter only maps their already-computed results into foundation values.
"""

from collections.abc import Mapping
from copy import deepcopy
from dataclasses import dataclass

from experiment_identity import ExperimentProvenance, MetricResultProvenance


@dataclass(frozen=True)
class ExperimentRecord:
    """An existing summary accompanied by explicit experiment provenance.

    The builder supplies an independent summary copy and a new metric mapping.
    These containers remain ordinary dictionaries; nested summary values are
    preserved rather than converted into a replacement summary schema.
    """

    experiment_name: str
    experiment_dir: str
    summary: dict
    provenance: ExperimentProvenance
    metric_results: dict[str, MetricResultProvenance]


def build_experiment_record(
    parsed_experiment: Mapping[str, object],
    *,
    provenance: ExperimentProvenance,
    metric_declarations: Mapping[str, Mapping[str, str | None]] | None = None,
) -> ExperimentRecord:
    """Adapt one successful_experiments entry without modifying caller data.

    Identity and execution come exclusively from the supplied provenance.
    Each metric keeps its own best_value and best_epoch from the summary.
    Optional declarations supply metric definition, direction, aggregation,
    evaluation/selection protocols and checkpoint/result/history references.
    Missing declarations retain the foundation model's None defaults.

    Consumers pass metric_results values directly to compare_metric_results;
    this module neither evaluates comparability nor infers shared artifacts.
    """
    if not isinstance(provenance, ExperimentProvenance):
        raise TypeError("provenance must be an ExperimentProvenance")

    summary = deepcopy(parsed_experiment["summary"])
    declarations = {} if metric_declarations is None else metric_declarations
    metric_results = {}
    for name, metric in summary["validation_metrics"].items():
        metric_results[name] = MetricResultProvenance(
            metric_name=name,
            value=metric["best_value"],
            best_epoch=metric["best_epoch"],
            provenance=provenance,
            **declarations.get(name, {}),
        )

    return ExperimentRecord(
        experiment_name=parsed_experiment["experiment_name"],
        experiment_dir=parsed_experiment["experiment_dir"],
        summary=summary,
        provenance=provenance,
        metric_results=metric_results,
    )

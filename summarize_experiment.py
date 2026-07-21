from collections.abc import Sequence
from pathlib import Path
from pprint import pprint

from diagnostics import (
    build_metric_diagnostics,
    build_metric_facts,
    diagnostic_to_dict,
)
from metrics import (
    MetricDirection,
    MetricSpec,
    evaluate_metric_history,
    get_value_at_path,
)
from read_config import CONFIG_PATH, read_config
from read_history import (
    HISTORY_PATH,
    analyze_validation_metric,
    read_history,
)


def _validate_metric_specs(
    metric_specs: Sequence[MetricSpec],
) -> tuple[MetricSpec, ...]:
    if (
        isinstance(metric_specs, (str, bytes, bytearray))
        or not isinstance(metric_specs, Sequence)
    ):
        raise TypeError("metric_specs must be a sequence")

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

    return tuple(validated_specs)


def _build_metric_diagnostic_payload(
    records: Sequence[Sequence[object]],
    direction: MetricDirection,
    recent_window: int,
) -> dict[str, object]:
    facts = build_metric_facts(
        records,
        direction,
        recent_window=recent_window,
    )
    diagnostics = build_metric_diagnostics(facts)

    return {
        "facts": facts,
        "diagnostics": [
            diagnostic_to_dict(item)
            for item in diagnostics
        ],
    }


def build_experiment_summary(
    config_path: Path = CONFIG_PATH,
    history_path: Path = HISTORY_PATH,
    metric_specs: Sequence[MetricSpec] | None = None,
    *,
    include_diagnostics: bool = False,
    diagnostic_recent_window: int = 5,
) -> dict:
    """读取指定配置和训练历史文件，生成结构化实验摘要。"""
    if not isinstance(include_diagnostics, bool):
        raise TypeError("include_diagnostics must be a boolean")

    config = read_config(config_path)
    history = read_history(history_path)
    metric_diagnostic_payloads: dict[str, dict[str, object]] = {}

    if metric_specs is None:
        r2_summary = analyze_validation_metric(history, "r2")
        racc_summary = analyze_validation_metric(history, "racc")
        validation_metrics = {
            "r2": r2_summary,
            "racc": racc_summary,
        }
        if include_diagnostics:
            default_records = history["valid"]["app"]
            metric_diagnostic_payloads["r2"] = (
                _build_metric_diagnostic_payload(
                    default_records["r2"],
                    "maximize",
                    diagnostic_recent_window,
                )
            )
            metric_diagnostic_payloads["racc"] = (
                _build_metric_diagnostic_payload(
                    default_records["racc"],
                    "maximize",
                    diagnostic_recent_window,
                )
            )
    else:
        validated_metric_specs = _validate_metric_specs(metric_specs)
        validation_metrics: dict[str, dict] = {}

        for spec in validated_metric_specs:
            records = get_value_at_path(history, spec.path)
            evaluation = evaluate_metric_history(
                records,
                spec.direction,
            )
            validation_metrics[spec.name] = {
                "metric_name": spec.name,
                **evaluation,
            }
            if include_diagnostics:
                metric_diagnostic_payloads[spec.name] = (
                    _build_metric_diagnostic_payload(
                        records,
                        spec.direction,
                        diagnostic_recent_window,
                    )
                )

    summary = {
        "configuration": {
            "batch_size": config.get("batch_size"),
            "num_workers": config.get("num_workers"),
            "seed": config.get("seed"),
            "sample_seed": config.get("sample_seed"),
            "n_epochs": config.get("n_epochs"),
            "feature_list": config.get("feature_list"),
            "attention_type": config.get("attention_type"),
            "use_modality_token_fusion": config.get(
                "use_modality_token_fusion"
            ),
            "use_behavior_state_token": config.get(
                "use_behavior_state_token"
            ),
            "use_behavior_aware_cl": config.get(
                "use_behavior_aware_cl"
            ),
            "use_regression_aware_cl": config.get(
                "use_regression_aware_cl"
            ),
            "use_trait_conditioned_modality_selection": config.get(
                "use_trait_conditioned_modality_selection"
            ),
        },
        "validation_metrics": validation_metrics,
    }

    if include_diagnostics:
        summary["diagnostics"] = {
            "metrics": metric_diagnostic_payloads,
        }

    return summary


def main() -> None:
    summary = build_experiment_summary()

    print("结构化实验摘要")
    print("-" * 50)
    pprint(summary, sort_dicts=False)


if __name__ == "__main__":
    main()

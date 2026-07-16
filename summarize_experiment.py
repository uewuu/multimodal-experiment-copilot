from pprint import pprint

from read_config import CONFIG_PATH, read_config
from read_history import (
    HISTORY_PATH,
    analyze_validation_metric,
    read_history,
)


def build_experiment_summary() -> dict:
    """读取配置和训练历史，生成结构化实验摘要。"""
    config = read_config(CONFIG_PATH)
    history = read_history(HISTORY_PATH)

    r2_summary = analyze_validation_metric(history, "r2")
    racc_summary = analyze_validation_metric(history, "racc")

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
        "validation_metrics": {
            "r2": r2_summary,
            "racc": racc_summary,
        },
    }

    return summary


def main() -> None:
    summary = build_experiment_summary()

    print("结构化实验摘要")
    print("-" * 50)
    pprint(summary, sort_dicts=False)


if __name__ == "__main__":
    main()
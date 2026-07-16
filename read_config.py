from pathlib import Path

import yaml


CONFIG_PATH = Path("examples/demo_experiment/hparams.yaml")


def read_config(config_path: Path) -> dict:
    """读取 YAML 配置文件，并返回其中的配置字典。"""
    with config_path.open("r", encoding="utf-8") as file:
        data = yaml.safe_load(file)

    if not isinstance(data, dict):
        raise ValueError("YAML 文件的最外层不是字典结构。")

    # 当前 hparams.yaml 的配置位于最外层 config 节点中。
    config = data.get("config", data)

    if not isinstance(config, dict):
        raise ValueError("config 节点不是有效的字典结构。")

    return config


def show_config(config: dict) -> None:
    """显示当前阶段关注的核心配置字段。"""
    fields = [
        "batch_size",
        "num_workers",
        "seed",
        "sample_seed",
        "n_epochs",
        "feature_list",
        "attention_type",
        "use_modality_token_fusion",
        "use_behavior_state_token",
        "use_behavior_aware_cl",
        "use_regression_aware_cl",
        "use_trait_conditioned_modality_selection",
    ]

    print("实验核心配置")
    print("-" * 50)

    for field in fields:
        value = config.get(field, "未找到")
        print(f"{field}: {value}")


def main() -> None:
    try:
        config = read_config(CONFIG_PATH)
        show_config(config)

    except FileNotFoundError:
        print(f"错误：没有找到配置文件：{CONFIG_PATH}")

    except yaml.YAMLError as error:
        print(f"错误：YAML 文件格式不正确：{error}")

    except UnicodeDecodeError:
        print("错误：无法使用 UTF-8 编码读取配置文件。")

    except ValueError as error:
        print(f"错误：{error}")


if __name__ == "__main__":
    main()
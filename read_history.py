from pathlib import Path
import json


HISTORY_PATH = Path("examples/demo_experiment/history.json")


def read_history(history_path: Path) -> dict:
    """读取训练历史 JSON 文件，并返回字典。"""
    with history_path.open("r", encoding="utf-8") as file:
        data = json.load(file)

    if not isinstance(data, dict):
        raise ValueError("history.json 的最外层不是字典结构。")

    return data


def show_top_level_keys(history: dict) -> None:
    """显示 history.json 的最外层字段。"""
    print("history.json 最外层字段")
    print("-" * 50)

    for key in history:
        print(key)

def show_validation_keys(history: dict) -> None:
    """显示验证集记录中的字段。"""
    valid_history = history.get("valid")

    if not isinstance(valid_history, dict):
        raise ValueError("history.json 中缺少有效的 valid 字典。")

    print()
    print("valid 节点中的字段")
    print("-" * 50)

    for key in valid_history:
        print(key)

def show_validation_app_keys(history: dict) -> None:
    """显示 valid.app 节点中的字段。"""
    valid_history = history.get("valid")

    if not isinstance(valid_history, dict):
        raise ValueError("history.json 中缺少有效的 valid 字典。")

    app_history = valid_history.get("app")

    if not isinstance(app_history, dict):
        raise ValueError("valid 节点中缺少有效的 app 字典。")

    print()
    print("valid.app 节点中的字段")
    print("-" * 50)

    for key in app_history:
        print(key)

def analyze_validation_metric(
    history: dict,
    metric_name: str,
) -> dict:
    """分析指定验证集指标，并返回结构化摘要。"""
    valid_history = history.get("valid")

    if not isinstance(valid_history, dict):
        raise ValueError("history.json 中缺少有效的 valid 字典。")

    app_history = valid_history.get("app")

    if not isinstance(app_history, dict):
        raise ValueError("valid 节点中缺少有效的 app 字典。")

    metric_history = app_history.get(metric_name)

    if not isinstance(metric_history, list):
        raise ValueError(
            f"valid.app 节点中缺少有效的 {metric_name} 列表。"
        )

    if not metric_history:
        raise ValueError(f"验证集 {metric_name} 记录为空。")

    best_epoch = None
    best_value = None

    for record in metric_history:
        if not isinstance(record, list) or len(record) != 2:
            raise ValueError(
                f"{metric_name} 记录格式不正确，"
                "预期格式为 [epoch, metric_value]。"
            )

        epoch, metric_value = record

        if not isinstance(epoch, int):
            raise ValueError(
                f"{metric_name} 记录中的 Epoch 不是整数。"
            )

        if not isinstance(metric_value, (int, float)):
            raise ValueError(
                f"{metric_name} 记录中的指标值不是数字。"
            )

        if best_value is None or metric_value > best_value:
            best_epoch = epoch
            best_value = metric_value

    first_record = metric_history[0]
    last_record = metric_history[-1]

    summary = {
        "metric_name": metric_name,
        "record_count": len(metric_history),
        "first_epoch": first_record[0],
        "first_value": first_record[1],
        "last_epoch": last_record[0],
        "last_value": last_record[1],
        "best_epoch": best_epoch,
        "best_value": best_value,
    }

    return summary


def show_metric_summary(summary: dict) -> None:
    """显示单个验证集指标的结构化摘要。"""
    metric_name = summary["metric_name"].upper()

    print()
    print(f"验证集 {metric_name} 记录摘要")
    print("-" * 50)
    print(f"记录数量：{summary['record_count']}")
    print(
        f"第一条记录："
        f"[{summary['first_epoch']}, {summary['first_value']}]"
    )
    print(
        f"最后一条记录："
        f"[{summary['last_epoch']}, {summary['last_value']}]"
    )
    print(f"最佳验证 {metric_name}：{summary['best_value']:.6f}")
    print(f"最佳 {metric_name} 对应 Epoch：{summary['best_epoch']}")

def main() -> None:
    try:
        history = read_history(HISTORY_PATH)
        show_top_level_keys(history)
        show_validation_keys(history)
        show_validation_app_keys(history)
        r2_summary = analyze_validation_metric(history, "r2")
        racc_summary = analyze_validation_metric(history, "racc")

        show_metric_summary(r2_summary)
        show_metric_summary(racc_summary)

        print()
        print("结构化指标摘要")
        print("-" * 50)
        print(r2_summary)
        print(racc_summary)

    except FileNotFoundError:
        print(f"错误：没有找到历史文件：{HISTORY_PATH}")

    except json.JSONDecodeError as error:
        print(f"错误：JSON 文件格式不正确：{error}")

    except UnicodeDecodeError:
        print("错误：无法使用 UTF-8 编码读取历史文件。")

    except ValueError as error:
        print(f"错误：{error}")


if __name__ == "__main__":
    main()
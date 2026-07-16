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

def show_validation_r2_summary(history: dict) -> dict:
    """显示验证集 R² 记录摘要，并查找最佳验证 R²。"""
    valid_history = history.get("valid")

    if not isinstance(valid_history, dict):
        raise ValueError("history.json 中缺少有效的 valid 字典。")

    app_history = valid_history.get("app")

    if not isinstance(app_history, dict):
        raise ValueError("valid 节点中缺少有效的 app 字典。")

    r2_history = app_history.get("r2")

    if not isinstance(r2_history, list):
        raise ValueError("valid.app 节点中缺少有效的 r2 列表。")

    print()
    print("验证集 R² 记录摘要")
    print("-" * 50)
    print(f"记录数量：{len(r2_history)}")

    if not r2_history:
        raise ValueError("验证集 R² 记录为空。")

    first_record = r2_history[0]
    last_record = r2_history[-1]

    print(f"第一条记录：{first_record}")
    print(f"最后一条记录：{last_record}")

    best_epoch = None
    best_r2 = None

    for record in r2_history:
        if not isinstance(record, list) or len(record) != 2:
            raise ValueError(
                "R² 记录格式不正确，预期格式为 [epoch, r2_value]。"
            )

        epoch, r2_value = record

        if not isinstance(epoch, int):
            raise ValueError("R² 记录中的 Epoch 不是整数。")

        if not isinstance(r2_value, (int, float)):
            raise ValueError("R² 记录中的指标值不是数字。")

        if best_r2 is None or r2_value > best_r2:
            best_epoch = epoch
            best_r2 = r2_value

    print(f"最佳验证 R²：{best_r2:.6f}")
    print(f"最佳 R² 对应 Epoch：{best_epoch}")
    
    summary = {
    "record_count": len(r2_history),
    "first_epoch": first_record[0],
    "first_r2": first_record[1],
    "last_epoch": last_record[0],
    "last_r2": last_record[1],
    "best_epoch": best_epoch,
    "best_r2": best_r2,
    }
    return summary

def main() -> None:
    try:
        history = read_history(HISTORY_PATH)
        show_top_level_keys(history)
        show_validation_keys(history)
        show_validation_app_keys(history)
        r2_summary = show_validation_r2_summary(history)

        print()
        print("结构化 R² 摘要")
        print("-" * 50)
        print(r2_summary)

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
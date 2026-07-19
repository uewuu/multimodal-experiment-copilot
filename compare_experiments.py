from pathlib import Path

from read_config import CONFIG_PATH
from read_history import HISTORY_PATH
from summarize_experiment import build_experiment_summary


DEFAULT_EXPERIMENT_ROOT = Path("examples")

CONFIG_FILENAME = CONFIG_PATH.name
HISTORY_FILENAME = HISTORY_PATH.name


def find_experiment_dirs(root_dir: Path) -> list[Path]:
    """查找根目录下所有有效的实验子目录。"""
    if not root_dir.exists():
        raise FileNotFoundError(
            f"实验根目录不存在：{root_dir}"
        )

    if not root_dir.is_dir():
        raise NotADirectoryError(
            f"指定路径不是目录：{root_dir}"
        )

    experiment_dirs: list[Path] = []

    for candidate_dir in root_dir.iterdir():
        if not candidate_dir.is_dir():
            continue

        config_path = candidate_dir / CONFIG_FILENAME
        history_path = candidate_dir / HISTORY_FILENAME

        if config_path.is_file() and history_path.is_file():
            experiment_dirs.append(candidate_dir)

    return sorted(experiment_dirs)


def analyze_experiment_dirs(
    experiment_dirs: list[Path],
) -> dict:
    """按传入顺序分析实验目录并收集成功与失败结果。"""
    successful_experiments: list[dict] = []
    failed_experiments: list[dict] = []

    for experiment_dir in experiment_dirs:
        config_path = experiment_dir / CONFIG_FILENAME
        history_path = experiment_dir / HISTORY_FILENAME

        try:
            summary = build_experiment_summary(
                config_path=config_path,
                history_path=history_path,
            )
        except Exception as error:
            failed_experiments.append(
                {
                    "experiment_name": experiment_dir.name,
                    "experiment_dir": str(experiment_dir),
                    "error_type": type(error).__name__,
                    "error_message": str(error),
                }
            )
            continue

        successful_experiments.append(
            {
                "experiment_name": experiment_dir.name,
                "experiment_dir": str(experiment_dir),
                "summary": summary,
            }
        )

    return {
        "successful_experiments": successful_experiments,
        "failed_experiments": failed_experiments,
    }


def main() -> None:
    try:
        experiment_dirs = find_experiment_dirs(
            DEFAULT_EXPERIMENT_ROOT
        )
    except (FileNotFoundError, NotADirectoryError) as error:
        raise SystemExit(
            f"实验目录扫描失败：{error}"
        ) from error

    if not experiment_dirs:
        print(
            f"没有找到有效实验目录："
            f"{DEFAULT_EXPERIMENT_ROOT}"
        )
        return

    print(f"共找到 {len(experiment_dirs)} 个有效实验目录：")
    print("-" * 50)

    for experiment_dir in experiment_dirs:
        print(f"- {experiment_dir}")


if __name__ == "__main__":
    main()

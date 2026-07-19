from copy import deepcopy
from pathlib import Path

import pytest

import compare_experiments
from compare_experiments import (
    analyze_experiment_dirs,
    build_comparison_records,
    find_experiment_dirs,
    rank_comparison_records,
)


def create_required_files(experiment_dir: Path) -> None:
    experiment_dir.mkdir(parents=True)
    (experiment_dir / "hparams.yaml").touch()
    (experiment_dir / "history.json").touch()


def test_empty_root_directory_returns_empty_list(tmp_path: Path) -> None:
    assert find_experiment_dirs(tmp_path) == []


def test_directory_with_both_required_files_is_found(tmp_path: Path) -> None:
    experiment_dir = tmp_path / "experiment_a"
    create_required_files(experiment_dir)

    assert find_experiment_dirs(tmp_path) == [experiment_dir]


def test_multiple_experiment_directories_are_sorted_by_path_name(
    tmp_path: Path,
) -> None:
    experiment_b = tmp_path / "experiment_b"
    experiment_a = tmp_path / "experiment_a"
    create_required_files(experiment_b)
    create_required_files(experiment_a)

    assert find_experiment_dirs(tmp_path) == [experiment_a, experiment_b]


def test_regular_files_in_root_directory_are_ignored(tmp_path: Path) -> None:
    (tmp_path / "unrelated.txt").touch()

    assert find_experiment_dirs(tmp_path) == []


def test_directory_missing_hparams_file_is_ignored(tmp_path: Path) -> None:
    experiment_dir = tmp_path / "missing_hparams"
    experiment_dir.mkdir()
    (experiment_dir / "history.json").touch()

    assert find_experiment_dirs(tmp_path) == []


def test_directory_missing_history_file_is_ignored(tmp_path: Path) -> None:
    experiment_dir = tmp_path / "missing_history"
    experiment_dir.mkdir()
    (experiment_dir / "hparams.yaml").touch()

    assert find_experiment_dirs(tmp_path) == []


@pytest.mark.parametrize("directory_name", ["hparams.yaml", "history.json"])
def test_required_filename_that_is_directory_is_ignored(
    tmp_path: Path,
    directory_name: str,
) -> None:
    experiment_dir = tmp_path / "invalid_experiment"
    create_required_files(experiment_dir)
    (experiment_dir / directory_name).unlink()
    (experiment_dir / directory_name).mkdir()

    assert find_experiment_dirs(tmp_path) == []


def test_experiment_nested_at_second_level_is_not_found(tmp_path: Path) -> None:
    nested_experiment = tmp_path / "group" / "nested_experiment"
    create_required_files(nested_experiment)

    assert find_experiment_dirs(tmp_path) == []


def test_root_directory_itself_is_not_returned_as_experiment(
    tmp_path: Path,
) -> None:
    (tmp_path / "hparams.yaml").touch()
    (tmp_path / "history.json").touch()

    assert find_experiment_dirs(tmp_path) == []


def test_nonexistent_root_path_raises_file_not_found_error(
    tmp_path: Path,
) -> None:
    nonexistent_root = tmp_path / "does_not_exist"

    with pytest.raises(FileNotFoundError):
        find_experiment_dirs(nonexistent_root)


def test_root_path_that_is_regular_file_raises_not_a_directory_error(
    tmp_path: Path,
) -> None:
    root_file = tmp_path / "root_file.txt"
    root_file.touch()

    with pytest.raises(NotADirectoryError):
        find_experiment_dirs(root_file)


def test_analyze_empty_experiment_list_returns_empty_results() -> None:
    result = analyze_experiment_dirs([])

    assert result == {
        "successful_experiments": [],
        "failed_experiments": [],
    }


def test_analyze_one_experiment_records_success(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    experiment_dir = tmp_path / "experiment_a"
    experiment_dir.mkdir()
    expected_summary = {"validation_metrics": {"r2": {"best_value": 0.8}}}
    monkeypatch.setattr(
        compare_experiments,
        "build_experiment_summary",
        lambda **kwargs: expected_summary,
    )

    result = analyze_experiment_dirs([experiment_dir])

    assert result == {
        "successful_experiments": [
            {
                "experiment_name": "experiment_a",
                "experiment_dir": str(experiment_dir),
                "summary": expected_summary,
            }
        ],
        "failed_experiments": [],
    }


def test_analyze_passes_correct_config_and_history_paths(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    experiment_dir = tmp_path / "experiment_a"
    experiment_dir.mkdir()
    received_paths: dict[str, Path] = {}

    def fake_build_experiment_summary(
        *,
        config_path: Path,
        history_path: Path,
    ) -> dict:
        received_paths["config_path"] = config_path
        received_paths["history_path"] = history_path
        return {}

    monkeypatch.setattr(
        compare_experiments,
        "build_experiment_summary",
        fake_build_experiment_summary,
    )

    analyze_experiment_dirs([experiment_dir])

    assert received_paths == {
        "config_path": experiment_dir / "hparams.yaml",
        "history_path": experiment_dir / "history.json",
    }


def test_analyze_one_experiment_records_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    experiment_dir = tmp_path / "broken_experiment"
    experiment_dir.mkdir()

    def raise_invalid_history(**kwargs: Path) -> dict:
        raise ValueError("invalid history")

    monkeypatch.setattr(
        compare_experiments,
        "build_experiment_summary",
        raise_invalid_history,
    )

    result = analyze_experiment_dirs([experiment_dir])

    assert result == {
        "successful_experiments": [],
        "failed_experiments": [
            {
                "experiment_name": "broken_experiment",
                "experiment_dir": str(experiment_dir),
                "error_type": "ValueError",
                "error_message": "invalid history",
            }
        ],
    }


def test_analyze_failure_does_not_interrupt_other_experiments(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    failed_dir = tmp_path / "failed_experiment"
    successful_dir = tmp_path / "successful_experiment"
    failed_dir.mkdir()
    successful_dir.mkdir()
    expected_summary = {"status": "success"}

    def build_or_fail(*, config_path: Path, history_path: Path) -> dict:
        if config_path.parent == failed_dir:
            raise RuntimeError("analysis failed")
        return expected_summary

    monkeypatch.setattr(
        compare_experiments,
        "build_experiment_summary",
        build_or_fail,
    )

    result = analyze_experiment_dirs([failed_dir, successful_dir])

    assert result["successful_experiments"] == [
        {
            "experiment_name": "successful_experiment",
            "experiment_dir": str(successful_dir),
            "summary": expected_summary,
        }
    ]
    assert result["failed_experiments"] == [
        {
            "experiment_name": "failed_experiment",
            "experiment_dir": str(failed_dir),
            "error_type": "RuntimeError",
            "error_message": "analysis failed",
        }
    ]


def test_analyze_preserves_input_directory_order(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    experiment_b = tmp_path / "experiment_b"
    experiment_a = tmp_path / "experiment_a"
    experiment_b.mkdir()
    experiment_a.mkdir()
    monkeypatch.setattr(
        compare_experiments,
        "build_experiment_summary",
        lambda **kwargs: {},
    )

    result = analyze_experiment_dirs([experiment_b, experiment_a])

    assert [
        experiment["experiment_name"]
        for experiment in result["successful_experiments"]
    ] == ["experiment_b", "experiment_a"]


def test_main_reports_when_no_experiment_directories_are_found(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr(
        compare_experiments,
        "find_experiment_dirs",
        lambda root_dir: [],
    )

    compare_experiments.main()

    captured = capsys.readouterr()
    assert "没有找到有效实验目录：" in captured.out
    assert "共找到 0 个有效实验目录" not in captured.out


def test_build_comparison_records_returns_empty_list_for_no_successes() -> None:
    batch_result = {
        "successful_experiments": [],
        "failed_experiments": [],
    }

    assert build_comparison_records(batch_result) == []


def test_build_comparison_records_builds_record_for_one_success() -> None:
    batch_result = {
        "successful_experiments": [
            {
                "experiment_name": "experiment_a",
                "experiment_dir": "experiments/experiment_a",
                "summary": {
                    "validation_metrics": {
                        "r2": {"best_value": 0.5, "best_epoch": 3},
                        "racc": {"best_value": 0.9, "best_epoch": 4},
                    }
                },
            }
        ],
        "failed_experiments": [],
    }

    assert build_comparison_records(batch_result) == [
        {
            "experiment_name": "experiment_a",
            "experiment_dir": "experiments/experiment_a",
            "best_r2": 0.5,
            "best_r2_epoch": 3,
            "best_racc": 0.9,
            "best_racc_epoch": 4,
        }
    ]


def test_build_comparison_records_extracts_metrics_from_best_fields() -> None:
    batch_result = {
        "successful_experiments": [
            {
                "experiment_name": "experiment_a",
                "experiment_dir": "experiments/experiment_a",
                "summary": {
                    "validation_metrics": {
                        "r2": {
                            "best_value": 0.41,
                            "best_epoch": 7,
                            "last_value": -1.0,
                        },
                        "racc": {
                            "best_value": 0.92,
                            "best_epoch": 11,
                            "last_value": -2.0,
                        },
                    }
                },
            }
        ],
        "failed_experiments": [],
    }

    record = build_comparison_records(batch_result)[0]

    assert record["best_r2"] == 0.41
    assert record["best_r2_epoch"] == 7
    assert record["best_racc"] == 0.92
    assert record["best_racc_epoch"] == 11


def test_build_comparison_records_preserves_successful_experiment_order() -> None:
    def successful_experiment(name: str, value: float) -> dict:
        return {
            "experiment_name": name,
            "experiment_dir": f"experiments/{name}",
            "summary": {
                "validation_metrics": {
                    "r2": {"best_value": value, "best_epoch": 1},
                    "racc": {"best_value": value, "best_epoch": 2},
                }
            },
        }

    batch_result = {
        "successful_experiments": [
            successful_experiment("experiment_b", 0.8),
            successful_experiment("experiment_a", 0.9),
        ],
        "failed_experiments": [],
    }

    records = build_comparison_records(batch_result)

    assert [record["experiment_name"] for record in records] == [
        "experiment_b",
        "experiment_a",
    ]


def test_build_comparison_records_excludes_failed_experiments() -> None:
    batch_result = {
        "successful_experiments": [
            {
                "experiment_name": "successful_experiment",
                "experiment_dir": "experiments/successful_experiment",
                "summary": {
                    "validation_metrics": {
                        "r2": {"best_value": 0.6, "best_epoch": 5},
                        "racc": {"best_value": 0.93, "best_epoch": 6},
                    }
                },
            }
        ],
        "failed_experiments": [
            {
                "experiment_name": "failed_experiment",
                "experiment_dir": "experiments/failed_experiment",
                "error_type": "ValueError",
                "error_message": "invalid history",
            }
        ],
    }

    records = build_comparison_records(batch_result)

    assert len(records) == 1
    assert records[0]["experiment_name"] == "successful_experiment"


def test_build_comparison_records_does_not_modify_batch_result() -> None:
    batch_result = {
        "successful_experiments": [
            {
                "experiment_name": "experiment_a",
                "experiment_dir": "experiments/experiment_a",
                "summary": {
                    "configuration": {"seed": 42},
                    "validation_metrics": {
                        "r2": {"best_value": 0.7, "best_epoch": 8},
                        "racc": {"best_value": 0.94, "best_epoch": 9},
                    },
                },
            }
        ],
        "failed_experiments": [],
    }
    original_batch_result = deepcopy(batch_result)

    build_comparison_records(batch_result)

    assert batch_result == original_batch_result


def test_rank_comparison_records_returns_empty_list_for_empty_input() -> None:
    assert rank_comparison_records([]) == []


def test_rank_comparison_records_defaults_to_best_r2_descending() -> None:
    records = [
        {"experiment_name": "lower", "best_r2": 0.4},
        {"experiment_name": "higher", "best_r2": 0.8},
    ]

    ranked = rank_comparison_records(records)

    assert [record["experiment_name"] for record in ranked] == [
        "higher",
        "lower",
    ]


def test_rank_comparison_records_sorts_by_best_racc_descending() -> None:
    records = [
        {"experiment_name": "lower", "best_racc": 0.88},
        {"experiment_name": "higher", "best_racc": 0.95},
    ]

    ranked = rank_comparison_records(records, sort_by="best_racc")

    assert [record["experiment_name"] for record in ranked] == [
        "higher",
        "lower",
    ]


def test_rank_comparison_records_sorts_ascending_when_requested() -> None:
    records = [
        {"experiment_name": "higher", "best_r2": 0.8},
        {"experiment_name": "lower", "best_r2": 0.4},
    ]

    ranked = rank_comparison_records(records, descending=False)

    assert [record["experiment_name"] for record in ranked] == [
        "lower",
        "higher",
    ]


def test_rank_comparison_records_preserves_order_for_equal_values() -> None:
    records = [
        {"experiment_name": "experiment_b", "best_r2": 0.7},
        {"experiment_name": "experiment_a", "best_r2": 0.7},
    ]

    ranked = rank_comparison_records(records)

    assert [record["experiment_name"] for record in ranked] == [
        "experiment_b",
        "experiment_a",
    ]


def test_rank_comparison_records_does_not_modify_original_list() -> None:
    records = [
        {"experiment_name": "lower", "best_r2": 0.4},
        {"experiment_name": "higher", "best_r2": 0.8},
    ]
    original_records = deepcopy(records)

    ranked = rank_comparison_records(records)

    assert records == original_records
    assert ranked is not records


def test_rank_comparison_records_rejects_unsupported_sort_field() -> None:
    invalid_field = "best_loss"

    with pytest.raises(ValueError) as error_info:
        rank_comparison_records([], sort_by=invalid_field)

    error_message = str(error_info.value)
    assert invalid_field in error_message
    assert "best_r2" in error_message
    assert "best_racc" in error_message


def test_rank_comparison_records_raises_key_error_for_missing_field() -> None:
    records = [{"experiment_name": "experiment_a", "best_r2": 0.7}]

    with pytest.raises(KeyError):
        rank_comparison_records(records, sort_by="best_racc")

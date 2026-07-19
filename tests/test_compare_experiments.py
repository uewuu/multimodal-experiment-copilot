from copy import deepcopy
from pathlib import Path

import pytest

import compare_experiments
from compare_experiments import (
    DEFAULT_COMPARISON_OUTPUT_PATH,
    analyze_experiment_dirs,
    build_comparison_markdown,
    build_comparison_payload,
    build_comparison_records,
    find_experiment_dirs,
    parse_args,
    rank_comparison_records,
    run_comparison_pipeline,
    write_comparison_json,
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
        "run_comparison_pipeline",
        lambda **kwargs: {
            "sort_by": "best_r2",
            "descending": True,
            "experiment_counts": {
                "total": 0,
                "successful": 0,
                "failed": 0,
            },
            "comparison_records": [],
            "failed_experiments": [],
        },
    )

    compare_experiments.main([])

    captured = capsys.readouterr()
    assert "没有找到有效实验目录：" in captured.out
    assert "JSON 输出：" in captured.out


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


def make_successful_experiment(
    name: str,
    best_r2: float,
    best_racc: float,
) -> dict:
    return {
        "experiment_name": name,
        "experiment_dir": f"experiments/{name}",
        "summary": {
            "validation_metrics": {
                "r2": {"best_value": best_r2, "best_epoch": 3},
                "racc": {"best_value": best_racc, "best_epoch": 4},
            }
        },
    }


def test_build_comparison_payload_builds_empty_payload() -> None:
    batch_result = {
        "successful_experiments": [],
        "failed_experiments": [],
    }

    payload = build_comparison_payload(batch_result)

    assert payload["experiment_counts"] == {
        "total": 0,
        "successful": 0,
        "failed": 0,
    }
    assert payload["comparison_records"] == []
    assert payload["failed_experiments"] == []


def test_build_comparison_payload_calculates_experiment_counts() -> None:
    batch_result = {
        "successful_experiments": [
            make_successful_experiment("experiment_a", 0.7, 0.91),
            make_successful_experiment("experiment_b", 0.8, 0.92),
        ],
        "failed_experiments": [
            {
                "experiment_name": "failed_experiment",
                "error_type": "ValueError",
            }
        ],
    }

    payload = build_comparison_payload(batch_result)

    assert payload["experiment_counts"] == {
        "total": 3,
        "successful": 2,
        "failed": 1,
    }


def test_build_comparison_payload_defaults_to_best_r2_descending() -> None:
    batch_result = {
        "successful_experiments": [
            make_successful_experiment("lower", 0.5, 0.95),
            make_successful_experiment("higher", 0.8, 0.90),
        ],
        "failed_experiments": [],
    }

    payload = build_comparison_payload(batch_result)

    assert payload["sort_by"] == "best_r2"
    assert payload["descending"] is True
    assert [
        record["experiment_name"]
        for record in payload["comparison_records"]
    ] == ["higher", "lower"]


def test_build_comparison_payload_sorts_best_racc_ascending() -> None:
    batch_result = {
        "successful_experiments": [
            make_successful_experiment("higher", 0.5, 0.95),
            make_successful_experiment("lower", 0.8, 0.90),
        ],
        "failed_experiments": [],
    }

    payload = build_comparison_payload(
        batch_result,
        sort_by="best_racc",
        descending=False,
    )

    assert payload["sort_by"] == "best_racc"
    assert payload["descending"] is False
    assert [
        record["experiment_name"]
        for record in payload["comparison_records"]
    ] == ["lower", "higher"]


def test_build_comparison_payload_preserves_only_failed_experiments() -> None:
    failed_experiments = [
        {
            "experiment_name": "failed_experiment",
            "experiment_dir": "experiments/failed_experiment",
            "error_type": "ValueError",
            "error_message": "invalid history",
        }
    ]
    batch_result = {
        "successful_experiments": [
            make_successful_experiment("successful", 0.7, 0.93)
        ],
        "failed_experiments": failed_experiments,
    }

    payload = build_comparison_payload(batch_result)

    assert payload["failed_experiments"] is failed_experiments
    assert [
        record["experiment_name"]
        for record in payload["comparison_records"]
    ] == ["successful"]


def test_build_comparison_payload_rejects_invalid_sort_field() -> None:
    batch_result = {
        "successful_experiments": [],
        "failed_experiments": [],
    }

    with pytest.raises(ValueError):
        build_comparison_payload(batch_result, sort_by="best_loss")


def test_build_comparison_payload_does_not_modify_batch_result() -> None:
    batch_result = {
        "successful_experiments": [
            make_successful_experiment("experiment_a", 0.7, 0.93)
        ],
        "failed_experiments": [
            {
                "experiment_name": "failed_experiment",
                "error_type": "ValueError",
            }
        ],
    }
    original_batch_result = deepcopy(batch_result)

    build_comparison_payload(batch_result)

    assert batch_result == original_batch_result


def test_write_comparison_json_writes_simple_payload(tmp_path: Path) -> None:
    payload = {"sort_by": "best_r2", "comparison_records": []}
    output_path = tmp_path / "comparison.json"

    write_comparison_json(payload, output_path)

    assert output_path.is_file()
    assert output_path.read_text(encoding="utf-8")


def test_write_comparison_json_returns_output_path(tmp_path: Path) -> None:
    output_path = tmp_path / "comparison.json"

    result = write_comparison_json({}, output_path)

    assert result == output_path


def test_write_comparison_json_preserves_chinese_utf8(tmp_path: Path) -> None:
    payload = {"message": "实验对比结果"}
    output_path = tmp_path / "comparison.json"

    write_comparison_json(payload, output_path)

    text = output_path.read_text(encoding="utf-8")
    assert "实验对比结果" in text


def test_write_comparison_json_uses_two_space_indentation(
    tmp_path: Path,
) -> None:
    output_path = tmp_path / "comparison.json"

    write_comparison_json({"sort_by": "best_r2"}, output_path)

    text = output_path.read_text(encoding="utf-8")
    assert '  "sort_by": "best_r2"' in text


def test_write_comparison_json_ends_with_newline(tmp_path: Path) -> None:
    output_path = tmp_path / "comparison.json"

    write_comparison_json({}, output_path)

    text = output_path.read_text(encoding="utf-8")
    assert text.endswith("\n")


def test_write_comparison_json_creates_nested_parent_directories(
    tmp_path: Path,
) -> None:
    output_path = tmp_path / "nested" / "reports" / "comparison.json"

    write_comparison_json({}, output_path)

    assert output_path.is_file()


def test_write_comparison_json_does_not_modify_payload(tmp_path: Path) -> None:
    payload = {
        "sort_by": "best_r2",
        "comparison_records": [{"experiment_name": "experiment_a"}],
    }
    original_payload = deepcopy(payload)

    write_comparison_json(payload, tmp_path / "comparison.json")

    assert payload == original_payload


def test_write_comparison_json_raises_type_error_for_invalid_value(
    tmp_path: Path,
) -> None:
    payload = {"invalid": {1, 2, 3}}

    with pytest.raises(TypeError):
        write_comparison_json(payload, tmp_path / "comparison.json")


def test_run_comparison_pipeline_calls_dependencies_in_order(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[str] = []

    def fake_find(experiment_root: Path) -> list[Path]:
        calls.append("find")
        return []

    def fake_analyze(experiment_dirs: list[Path]) -> dict:
        calls.append("analyze")
        return {"successful_experiments": [], "failed_experiments": []}

    def fake_build(batch_result: dict, **kwargs: object) -> dict:
        calls.append("build")
        return {"comparison_records": []}

    def fake_write(payload: dict, output_path: Path) -> Path:
        calls.append("write")
        return output_path

    monkeypatch.setattr(compare_experiments, "find_experiment_dirs", fake_find)
    monkeypatch.setattr(compare_experiments, "analyze_experiment_dirs", fake_analyze)
    monkeypatch.setattr(compare_experiments, "build_comparison_payload", fake_build)
    monkeypatch.setattr(compare_experiments, "write_comparison_json", fake_write)

    run_comparison_pipeline(tmp_path / "experiments", tmp_path / "result.json")

    assert calls == ["find", "analyze", "build", "write"]


def test_run_comparison_pipeline_passes_experiment_root_to_find(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    experiment_root = tmp_path / "experiments"
    received: dict[str, Path] = {}

    def fake_find(root: Path) -> list[Path]:
        received["experiment_root"] = root
        return []

    monkeypatch.setattr(compare_experiments, "find_experiment_dirs", fake_find)
    monkeypatch.setattr(compare_experiments, "analyze_experiment_dirs", lambda dirs: {})
    monkeypatch.setattr(compare_experiments, "build_comparison_payload", lambda result, **kwargs: {})
    monkeypatch.setattr(compare_experiments, "write_comparison_json", lambda payload, path: path)

    run_comparison_pipeline(experiment_root, tmp_path / "result.json")

    assert received["experiment_root"] == experiment_root


def test_run_comparison_pipeline_passes_found_dirs_to_analyze(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    experiment_dirs = [tmp_path / "experiment_a", tmp_path / "experiment_b"]
    received: dict[str, list[Path]] = {}

    def fake_analyze(dirs: list[Path]) -> dict:
        received["experiment_dirs"] = dirs
        return {}

    monkeypatch.setattr(compare_experiments, "find_experiment_dirs", lambda root: experiment_dirs)
    monkeypatch.setattr(compare_experiments, "analyze_experiment_dirs", fake_analyze)
    monkeypatch.setattr(compare_experiments, "build_comparison_payload", lambda result, **kwargs: {})
    monkeypatch.setattr(compare_experiments, "write_comparison_json", lambda payload, path: path)

    run_comparison_pipeline(tmp_path, tmp_path / "result.json")

    assert received["experiment_dirs"] is experiment_dirs


def test_run_comparison_pipeline_passes_batch_and_sort_options_to_build(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    batch_result = {"successful_experiments": [], "failed_experiments": []}
    received: dict[str, object] = {}

    def fake_build(
        result: dict,
        *,
        sort_by: str,
        descending: bool,
    ) -> dict:
        received["batch_result"] = result
        received["sort_by"] = sort_by
        received["descending"] = descending
        return {}

    monkeypatch.setattr(compare_experiments, "find_experiment_dirs", lambda root: [])
    monkeypatch.setattr(compare_experiments, "analyze_experiment_dirs", lambda dirs: batch_result)
    monkeypatch.setattr(compare_experiments, "build_comparison_payload", fake_build)
    monkeypatch.setattr(compare_experiments, "write_comparison_json", lambda payload, path: path)

    run_comparison_pipeline(
        tmp_path,
        tmp_path / "result.json",
        sort_by="best_racc",
        descending=False,
    )

    assert received == {
        "batch_result": batch_result,
        "sort_by": "best_racc",
        "descending": False,
    }


def test_run_comparison_pipeline_passes_payload_and_path_to_writer(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    payload = {"comparison_records": []}
    output_path = tmp_path / "result.json"
    received: dict[str, object] = {}

    def fake_write(result: dict, path: Path) -> Path:
        received["payload"] = result
        received["output_path"] = path
        return path

    monkeypatch.setattr(compare_experiments, "find_experiment_dirs", lambda root: [])
    monkeypatch.setattr(compare_experiments, "analyze_experiment_dirs", lambda dirs: {})
    monkeypatch.setattr(compare_experiments, "build_comparison_payload", lambda result, **kwargs: payload)
    monkeypatch.setattr(compare_experiments, "write_comparison_json", fake_write)

    run_comparison_pipeline(tmp_path, output_path)

    assert received == {"payload": payload, "output_path": output_path}


def test_run_comparison_pipeline_returns_built_payload(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    payload = {"comparison_records": [{"experiment_name": "experiment_a"}]}
    monkeypatch.setattr(compare_experiments, "find_experiment_dirs", lambda root: [])
    monkeypatch.setattr(compare_experiments, "analyze_experiment_dirs", lambda dirs: {})
    monkeypatch.setattr(compare_experiments, "build_comparison_payload", lambda result, **kwargs: payload)
    monkeypatch.setattr(compare_experiments, "write_comparison_json", lambda result, path: path)

    result = run_comparison_pipeline(tmp_path, tmp_path / "result.json")

    assert result is payload


def test_run_comparison_pipeline_propagates_file_not_found_error(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def raise_missing_root(root: Path) -> list[Path]:
        raise FileNotFoundError("missing root")

    monkeypatch.setattr(
        compare_experiments,
        "find_experiment_dirs",
        raise_missing_root,
    )

    with pytest.raises(FileNotFoundError, match="missing root"):
        run_comparison_pipeline(tmp_path, tmp_path / "result.json")


def test_run_comparison_pipeline_propagates_value_error_without_writing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    write_called = False

    def raise_invalid_sort(result: dict, **kwargs: object) -> dict:
        raise ValueError("invalid sort field")

    def fake_write(payload: dict, path: Path) -> Path:
        nonlocal write_called
        write_called = True
        return path

    monkeypatch.setattr(compare_experiments, "find_experiment_dirs", lambda root: [])
    monkeypatch.setattr(compare_experiments, "analyze_experiment_dirs", lambda dirs: {})
    monkeypatch.setattr(compare_experiments, "build_comparison_payload", raise_invalid_sort)
    monkeypatch.setattr(compare_experiments, "write_comparison_json", fake_write)

    with pytest.raises(ValueError, match="invalid sort field"):
        run_comparison_pipeline(tmp_path, tmp_path / "result.json")

    assert write_called is False


def test_parse_args_uses_default_values() -> None:
    args = parse_args([])

    assert args.experiment_root == compare_experiments.DEFAULT_EXPERIMENT_ROOT
    assert args.output_path == DEFAULT_COMPARISON_OUTPUT_PATH
    assert args.sort_by == "best_r2"
    assert args.ascending is False


def test_parse_args_accepts_custom_paths() -> None:
    args = parse_args(
        [
            "--experiment-root",
            "custom_experiments",
            "--output-path",
            "custom_outputs/result.json",
        ]
    )

    assert args.experiment_root == Path("custom_experiments")
    assert isinstance(args.experiment_root, Path)
    assert args.output_path == Path("custom_outputs/result.json")
    assert isinstance(args.output_path, Path)


def test_parse_args_accepts_best_racc_sort_field() -> None:
    args = parse_args(["--sort-by", "best_racc"])

    assert args.sort_by == "best_racc"


def test_parse_args_enables_ascending_order() -> None:
    args = parse_args(["--ascending"])

    assert args.ascending is True


def test_parse_args_rejects_invalid_sort_field() -> None:
    with pytest.raises(SystemExit):
        parse_args(["--sort-by", "invalid_metric"])


def test_main_passes_parsed_arguments_to_pipeline(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    received: dict[str, object] = {}

    def fake_pipeline(**kwargs: object) -> dict:
        received.update(kwargs)
        return {
            "sort_by": "best_racc",
            "descending": False,
            "experiment_counts": {
                "total": 2,
                "successful": 2,
                "failed": 0,
            },
            "comparison_records": [],
            "failed_experiments": [],
        }

    monkeypatch.setattr(
        compare_experiments,
        "run_comparison_pipeline",
        fake_pipeline,
    )

    compare_experiments.main(
        [
            "--experiment-root",
            "custom_experiments",
            "--output-path",
            "custom_outputs/comparison.json",
            "--sort-by",
            "best_racc",
            "--ascending",
        ]
    )

    assert received == {
        "experiment_root": Path("custom_experiments"),
        "output_path": Path("custom_outputs/comparison.json"),
        "sort_by": "best_racc",
        "descending": False,
    }


def test_main_prints_comparison_summary(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr(
        compare_experiments,
        "run_comparison_pipeline",
        lambda **kwargs: {
            "sort_by": "best_r2",
            "descending": True,
            "experiment_counts": {
                "total": 3,
                "successful": 2,
                "failed": 1,
            },
            "comparison_records": [],
            "failed_experiments": [],
        },
    )

    compare_experiments.main([])

    captured = capsys.readouterr()
    assert "实验比较完成：" in captured.out
    assert "总实验数：3" in captured.out
    assert "成功实验：2" in captured.out
    assert "失败实验：1" in captured.out
    assert "best_r2" in captured.out
    assert "降序" in captured.out
    assert "JSON 输出" in captured.out


def markdown_payload() -> dict:
    return {
        "sort_by": "best_r2",
        "descending": True,
        "experiment_counts": {"total": 0, "successful": 0, "failed": 0},
        "comparison_records": [],
        "failed_experiments": [],
    }


def comparison_record(name: str, value: float = 0.5) -> dict:
    return {
        "experiment_name": name,
        "experiment_dir": f"examples/{name}",
        "best_r2": value,
        "best_r2_epoch": 2,
        "best_racc": value,
        "best_racc_epoch": 3,
    }


def test_build_comparison_markdown_handles_no_successful_experiments() -> None:
    markdown = build_comparison_markdown(markdown_payload())

    assert "No successful experiments were analyzed." in markdown


def test_build_comparison_markdown_builds_ranked_table_for_one_experiment() -> None:
    payload = markdown_payload()
    payload["comparison_records"] = [comparison_record("experiment_a", 0.9)]

    markdown = build_comparison_markdown(payload)

    assert "| Rank | Experiment | Directory | Best R² |" in markdown
    assert "| 1 | experiment_a | examples/experiment_a |" in markdown


def test_build_comparison_markdown_preserves_record_order() -> None:
    payload = markdown_payload()
    payload["comparison_records"] = [
        comparison_record("second", 0.2),
        comparison_record("first", 0.9),
    ]

    markdown = build_comparison_markdown(payload)

    assert markdown.index("second") < markdown.index("first")


def test_build_comparison_markdown_numbers_ranks_from_one() -> None:
    payload = markdown_payload()
    payload["comparison_records"] = [
        comparison_record("a"),
        comparison_record("b"),
    ]

    markdown = build_comparison_markdown(payload)

    assert "| 1 | a |" in markdown
    assert "| 2 | b |" in markdown


def test_build_comparison_markdown_formats_metrics_to_six_decimals() -> None:
    payload = markdown_payload()
    record = comparison_record("a")
    record["best_r2"] = 0.123456789
    record["best_racc"] = 0.9
    payload["comparison_records"] = [record]

    markdown = build_comparison_markdown(payload)

    assert "0.123457" in markdown
    assert "0.900000" in markdown


def test_build_comparison_markdown_displays_descending_direction() -> None:
    markdown = build_comparison_markdown(markdown_payload())

    assert "- Sort direction: Descending" in markdown


def test_build_comparison_markdown_displays_ascending_direction() -> None:
    payload = markdown_payload()
    payload["descending"] = False

    markdown = build_comparison_markdown(payload)

    assert "- Sort direction: Ascending" in markdown


def test_build_comparison_markdown_omits_failed_section_when_no_failures() -> None:
    markdown = build_comparison_markdown(markdown_payload())

    assert "## Failed Experiments" not in markdown


def test_build_comparison_markdown_includes_failed_experiment_table() -> None:
    payload = markdown_payload()
    payload["failed_experiments"] = [
        {
            "experiment_name": "broken",
            "experiment_dir": "examples/broken",
            "error_type": "ValueError",
            "error_message": "invalid metric",
        }
    ]

    markdown = build_comparison_markdown(payload)

    assert "## Failed Experiments" in markdown
    assert "| broken | examples/broken | ValueError | invalid metric |" in markdown


def test_build_comparison_markdown_escapes_table_pipes() -> None:
    payload = markdown_payload()
    payload["failed_experiments"] = [
        {
            "experiment_name": "a|b",
            "experiment_dir": "dir|name",
            "error_type": "Type|Error",
            "error_message": "left|right",
        }
    ]

    markdown = build_comparison_markdown(payload)

    for escaped_value in ["a\\|b", "dir\\|name", "Type\\|Error", "left\\|right"]:
        assert escaped_value in markdown


def test_build_comparison_markdown_replaces_multiline_cells_with_html_breaks() -> None:
    payload = markdown_payload()
    payload["failed_experiments"] = [
        {
            "experiment_name": "broken",
            "experiment_dir": "line1\r\nline2",
            "error_type": "ValueError",
            "error_message": "first\rsecond\nthird",
        }
    ]

    markdown = build_comparison_markdown(payload)

    assert "line1<br>line2" in markdown
    assert "first<br>second<br>third" in markdown


def test_build_comparison_markdown_does_not_modify_payload() -> None:
    payload = markdown_payload()
    payload["comparison_records"] = [comparison_record("a")]
    original_payload = deepcopy(payload)

    build_comparison_markdown(payload)

    assert payload == original_payload


def test_build_comparison_markdown_raises_key_error_for_missing_required_key() -> None:
    payload = markdown_payload()
    del payload["sort_by"]

    with pytest.raises(KeyError, match="sort_by"):
        build_comparison_markdown(payload)


def test_parse_args_help_describes_experiment_root(
    capsys: pytest.CaptureFixture[str],
) -> None:
    with pytest.raises(SystemExit) as error_info:
        parse_args(["--help"])

    captured = capsys.readouterr()
    assert error_info.value.code == 0
    assert "--experiment-root" in captured.out
    assert "包含多个实验目录的根目录" in captured.out
    assert "--output-path" in captured.out
    assert "实验对比结果的 JSON 输出路径" in captured.out

    help_lines = captured.out.splitlines()
    experiment_option_index = next(
        index
        for index, line in enumerate(help_lines)
        if line.strip().startswith("--experiment-root")
    )
    output_option_index = next(
        index
        for index, line in enumerate(help_lines)
        if line.strip().startswith("--output-path")
    )

    assert help_lines[experiment_option_index + 1].strip() == (
        "包含多个实验目录的根目录"
    )
    assert help_lines[output_option_index + 1].strip() == (
        "实验对比结果的 JSON 输出路径"
    )

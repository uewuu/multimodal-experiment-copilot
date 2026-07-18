from pathlib import Path

import pytest

from compare_experiments import find_experiment_dirs


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
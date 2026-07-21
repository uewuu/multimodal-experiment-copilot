import json
from copy import deepcopy
from pathlib import Path

import pytest

import generate_report
from generate_report import (
    REPORT_MD_FILENAME,
    SUMMARY_JSON_FILENAME,
    build_markdown_report,
    generate_experiment_report,
    main,
    parse_arguments,
    write_summary_json,
)


def make_summary(*, include_diagnostics: bool = False) -> dict:
    summary = {
        "configuration": {
            "batch_size": 8,
            "num_workers": 4,
            "seed": 42,
            "sample_seed": 42,
            "n_epochs": 10,
            "feature_list": ["vision", "audio"],
            "attention_type": "linear",
            "use_modality_token_fusion": True,
            "use_behavior_state_token": True,
            "use_behavior_aware_cl": True,
            "use_regression_aware_cl": False,
            "use_trait_conditioned_modality_selection": True,
        },
        "validation_metrics": {
            "r2": {
                "record_count": 3,
                "first_epoch": 0,
                "first_value": 0.1,
                "last_epoch": 2,
                "last_value": 0.3,
                "best_epoch": 2,
                "best_value": 0.3,
            },
            "racc": {
                "record_count": 3,
                "first_epoch": 0,
                "first_value": 0.8,
                "last_epoch": 2,
                "last_value": 0.9,
                "best_epoch": 2,
                "best_value": 0.9,
            },
        },
    }
    if include_diagnostics:
        summary["diagnostics"] = {
            "metrics": {
                "r2": {
                    "facts": {"recent_trend": "improving"},
                    "diagnostics": [
                        {
                            "code": "best_at_last_record",
                            "severity": "info",
                            "message": (
                                "The best value occurs at the last record."
                            ),
                            "evidence": {
                                "best_record_index": 2,
                                "best_epoch": 2,
                            },
                        },
                        {
                            "code": "recent_improvement",
                            "severity": "info",
                            "message": (
                                "The recent metric history is improving."
                            ),
                            "evidence": {
                                "recent_window_size": 3,
                                "recent_net_change": 0.2,
                            },
                        },
                    ],
                    "recommendations": [
                        {
                            "code": "consider_extended_training",
                            "message": (
                                "Consider extending training or patience."
                            ),
                            "diagnostic_codes": [
                                "best_at_last_record",
                                "recent_improvement",
                            ],
                        }
                    ],
                },
                "racc": {
                    "facts": {"recent_trend": "mixed"},
                    "diagnostics": [
                        {
                            "code": "recent_mixed",
                            "severity": "warning",
                            "message": (
                                "The recent metric history has mixed direction."
                            ),
                            "evidence": {
                                "label": "left|right",
                                "note": "line1\nline2",
                            },
                        }
                    ],
                    "recommendations": [
                        {
                            "code": "avoid_trend_conclusion",
                            "message": (
                                "Avoid strong trend conclusions until "
                                "additional records clarify the mixed direction."
                            ),
                            "diagnostic_codes": ["recent_mixed"],
                        }
                    ],
                },
            }
        }
    return summary


def test_parse_arguments_defaults_diagnostics_to_false() -> None:
    args = parse_arguments([])

    assert args.include_diagnostics is False


def test_parse_arguments_accepts_include_diagnostics_flag() -> None:
    args = parse_arguments(["--include-diagnostics"])

    assert args.include_diagnostics is True


def test_parse_arguments_preserves_existing_path_defaults() -> None:
    args = parse_arguments([])

    assert args.experiment_dir == generate_report.DEFAULT_EXPERIMENT_DIR
    assert args.output_dir == generate_report.OUTPUT_DIR


def test_parse_arguments_accepts_existing_paths_with_diagnostics() -> None:
    args = parse_arguments(
        [
            "--experiment-dir",
            "examples/demo",
            "--output-dir",
            "outputs/demo",
            "--include-diagnostics",
        ]
    )

    assert args.experiment_dir == Path("examples/demo")
    assert args.output_dir == Path("outputs/demo")
    assert args.include_diagnostics is True


def test_help_describes_include_diagnostics(
    capsys: pytest.CaptureFixture[str],
) -> None:
    with pytest.raises(SystemExit) as error_info:
        parse_arguments(["--help"])

    assert error_info.value.code == 0
    assert "--include-diagnostics" in capsys.readouterr().out


def test_default_markdown_omits_diagnostics_section() -> None:
    markdown = build_markdown_report(make_summary())

    assert "## 5. 规则诊断" not in markdown
    assert "best_at_last_record" not in markdown


def test_diagnostics_markdown_appends_section_after_automatic_analysis() -> None:
    markdown = build_markdown_report(
        make_summary(include_diagnostics=True)
    )

    assert markdown.index("## 4. 自动分析") < markdown.index(
        "## 5. 规则诊断"
    )
    assert markdown.endswith("\n")


def test_diagnostics_markdown_contains_stable_table_header() -> None:
    markdown = build_markdown_report(
        make_summary(include_diagnostics=True)
    )

    assert "| 指标 | 级别 | 代码 | 说明 | 证据 |" in markdown
    assert "|---|---|---|---|---|" in markdown


def test_diagnostics_markdown_preserves_metric_and_rule_order() -> None:
    markdown = build_markdown_report(
        make_summary(include_diagnostics=True)
    )

    assert markdown.index("best_at_last_record") < markdown.index(
        "recent_improvement"
    )
    assert markdown.index("recent_improvement") < markdown.index(
        "recent_mixed"
    )


def test_diagnostics_markdown_uses_default_metric_display_names() -> None:
    markdown = build_markdown_report(
        make_summary(include_diagnostics=True)
    )

    assert "| R² | info | `best_at_last_record` |" in markdown
    assert "| RACC | warning | `recent_mixed` |" in markdown


def test_diagnostics_markdown_serializes_evidence_deterministically() -> None:
    markdown = build_markdown_report(
        make_summary(include_diagnostics=True)
    )

    assert (
        '`{"best_epoch": 2, "best_record_index": 2}`'
        in markdown
    )


def test_diagnostics_markdown_escapes_evidence_table_content() -> None:
    markdown = build_markdown_report(
        make_summary(include_diagnostics=True)
    )

    assert "left\\|right" in markdown
    assert "line1\\nline2" in markdown


def test_diagnostics_markdown_handles_no_generated_diagnostics() -> None:
    summary = make_summary()
    summary["diagnostics"] = {
        "metrics": {
            "r2": {
                "facts": {},
                "diagnostics": [],
                "recommendations": [],
            },
            "racc": {
                "facts": {},
                "diagnostics": [],
                "recommendations": [],
            },
        }
    }

    markdown = build_markdown_report(summary)

    assert "| — | — | — | 未生成诊断信息 | — |" in markdown


def test_build_markdown_report_does_not_modify_summary() -> None:
    summary = make_summary(include_diagnostics=True)
    original = deepcopy(summary)

    build_markdown_report(summary)

    assert summary == original


@pytest.mark.parametrize("invalid_value", [None, 0, 1, "true", [], {}])
def test_generate_experiment_report_rejects_non_boolean_diagnostics(
    invalid_value: object,
    tmp_path: Path,
) -> None:
    with pytest.raises(
        TypeError,
        match="^include_diagnostics must be a boolean$",
    ):
        generate_experiment_report(
            output_dir=tmp_path,
            include_diagnostics=invalid_value,  # type: ignore[arg-type]
        )


def test_generate_report_default_keeps_legacy_summary_call_shape(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    expected_summary = make_summary()
    received: dict[str, Path] = {}

    def strict_builder(*, config_path: Path, history_path: Path) -> dict:
        received["config_path"] = config_path
        received["history_path"] = history_path
        return expected_summary

    monkeypatch.setattr(
        generate_report,
        "build_experiment_summary",
        strict_builder,
    )

    config_path = tmp_path / "hparams.yaml"
    history_path = tmp_path / "history.json"
    generate_experiment_report(
        config_path=config_path,
        history_path=history_path,
        output_dir=tmp_path / "outputs",
    )

    assert received == {
        "config_path": config_path,
        "history_path": history_path,
    }


def test_generate_report_enabled_passes_diagnostics_true(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    received: dict[str, object] = {}

    def diagnostic_builder(**kwargs: object) -> dict:
        received.update(kwargs)
        return make_summary(include_diagnostics=True)

    monkeypatch.setattr(
        generate_report,
        "build_experiment_summary",
        diagnostic_builder,
    )

    generate_experiment_report(
        config_path=tmp_path / "hparams.yaml",
        history_path=tmp_path / "history.json",
        output_dir=tmp_path / "outputs",
        include_diagnostics=True,
    )

    assert received["include_diagnostics"] is True


def test_generate_report_enabled_writes_diagnostics_to_json_and_markdown(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    expected_summary = make_summary(include_diagnostics=True)
    monkeypatch.setattr(
        generate_report,
        "build_experiment_summary",
        lambda **kwargs: expected_summary,
    )
    output_dir = tmp_path / "outputs"

    json_path, markdown_path = generate_experiment_report(
        output_dir=output_dir,
        include_diagnostics=True,
    )

    assert json_path == output_dir / SUMMARY_JSON_FILENAME
    assert markdown_path == output_dir / REPORT_MD_FILENAME
    assert json.loads(json_path.read_text(encoding="utf-8")) == (
        expected_summary
    )
    assert "## 5. 规则诊断" in markdown_path.read_text(
        encoding="utf-8"
    )


def test_generate_report_default_writes_no_diagnostics(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    expected_summary = make_summary()
    monkeypatch.setattr(
        generate_report,
        "build_experiment_summary",
        lambda **kwargs: expected_summary,
    )
    output_dir = tmp_path / "outputs"

    json_path, markdown_path = generate_experiment_report(
        output_dir=output_dir,
    )

    assert "diagnostics" not in json.loads(
        json_path.read_text(encoding="utf-8")
    )
    assert "## 5. 规则诊断" not in markdown_path.read_text(
        encoding="utf-8"
    )


def test_write_summary_json_round_trips_diagnostics(
    tmp_path: Path,
) -> None:
    summary = make_summary(include_diagnostics=True)
    output_path = tmp_path / "summary.json"

    write_summary_json(summary, output_path)

    assert json.loads(output_path.read_text(encoding="utf-8")) == summary


def test_main_default_preserves_legacy_generate_call_shape(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    experiment_dir = tmp_path / "experiment"
    output_dir = tmp_path / "outputs"
    config_path = experiment_dir / "hparams.yaml"
    history_path = experiment_dir / "history.json"
    received: dict[str, object] = {}

    monkeypatch.setattr(
        generate_report,
        "resolve_experiment_paths",
        lambda path: (config_path, history_path),
    )

    def strict_generate(
        *,
        config_path: Path,
        history_path: Path,
        output_dir: Path,
    ) -> tuple[Path, Path]:
        received.update(
            config_path=config_path,
            history_path=history_path,
            output_dir=output_dir,
        )
        return (
            output_dir / SUMMARY_JSON_FILENAME,
            output_dir / REPORT_MD_FILENAME,
        )

    monkeypatch.setattr(
        generate_report,
        "generate_experiment_report",
        strict_generate,
    )

    main(
        [
            "--experiment-dir",
            str(experiment_dir),
            "--output-dir",
            str(output_dir),
        ]
    )

    assert received == {
        "config_path": config_path,
        "history_path": history_path,
        "output_dir": output_dir,
    }
    assert "规则诊断：已启用" not in capsys.readouterr().out


def test_main_enabled_passes_flag_and_prints_status(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    experiment_dir = tmp_path / "experiment"
    output_dir = tmp_path / "outputs"
    config_path = experiment_dir / "hparams.yaml"
    history_path = experiment_dir / "history.json"
    received: dict[str, object] = {}

    monkeypatch.setattr(
        generate_report,
        "resolve_experiment_paths",
        lambda path: (config_path, history_path),
    )

    def diagnostic_generate(**kwargs: object) -> tuple[Path, Path]:
        received.update(kwargs)
        return (
            output_dir / SUMMARY_JSON_FILENAME,
            output_dir / REPORT_MD_FILENAME,
        )

    monkeypatch.setattr(
        generate_report,
        "generate_experiment_report",
        diagnostic_generate,
    )

    main(
        [
            "--experiment-dir",
            str(experiment_dir),
            "--output-dir",
            str(output_dir),
            "--include-diagnostics",
        ]
    )

    assert received["include_diagnostics"] is True
    assert "规则诊断：已启用" in capsys.readouterr().out


def test_main_missing_files_keeps_existing_error_message(
    tmp_path: Path,
) -> None:
    with pytest.raises(SystemExit, match="^报告生成失败："):
        main(["--experiment-dir", str(tmp_path)])



def test_default_markdown_omits_recommendations_section() -> None:
    markdown = build_markdown_report(make_summary())

    assert "## 6. 规则建议" not in markdown


def test_recommendations_markdown_follows_diagnostics_section() -> None:
    markdown = build_markdown_report(
        make_summary(include_diagnostics=True)
    )

    assert markdown.index("## 5. 规则诊断") < markdown.index(
        "## 6. 规则建议"
    )


def test_recommendations_markdown_contains_stable_table() -> None:
    markdown = build_markdown_report(
        make_summary(include_diagnostics=True)
    )

    assert "| 指标 | 建议代码 | 建议 | 触发诊断 |" in markdown
    assert "|---|---|---|---|" in markdown
    assert "| R² | `consider_extended_training` |" in markdown
    assert "| RACC | `avoid_trend_conclusion` |" in markdown


def test_recommendations_markdown_serializes_trigger_codes_in_order() -> None:
    markdown = build_markdown_report(
        make_summary(include_diagnostics=True)
    )

    assert (
        "`best_at_last_record、recent_improvement`"
        in markdown
    )


def test_recommendations_markdown_handles_empty_lists() -> None:
    summary = make_summary(include_diagnostics=True)
    for metric in summary["diagnostics"]["metrics"].values():
        metric["recommendations"] = []

    markdown = build_markdown_report(summary)

    assert "| — | — | 未生成规则建议 | — |" in markdown


def test_generated_report_writes_recommendations_to_json_and_markdown(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    expected_summary = make_summary(include_diagnostics=True)
    monkeypatch.setattr(
        generate_report,
        "build_experiment_summary",
        lambda **kwargs: expected_summary,
    )

    json_path, markdown_path = generate_experiment_report(
        output_dir=tmp_path,
        include_diagnostics=True,
    )

    payload = json.loads(json_path.read_text(encoding="utf-8"))
    markdown = markdown_path.read_text(encoding="utf-8")
    assert payload["diagnostics"]["metrics"]["r2"]["recommendations"]
    assert "## 6. 规则建议" in markdown

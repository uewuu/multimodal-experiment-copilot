import json
from copy import deepcopy
from dataclasses import FrozenInstanceError, fields
from types import MappingProxyType

import pytest

from diagnostics import (
    Diagnostic,
    Recommendation,
    build_comparison_diagnostics,
    build_comparison_facts,
    build_metric_diagnostics,
    build_metric_facts,
    build_recommendations,
    diagnostic_to_dict,
    recommendation_to_dict,
)


FACT_KEYS = [
    "record_count",
    "first_epoch",
    "first_value",
    "last_epoch",
    "last_value",
    "best_epoch",
    "best_value",
    "best_record_index",
    "best_progress_ratio",
    "best_at_first_record",
    "best_at_last_record",
    "duplicate_epochs",
    "non_monotonic_epoch_transitions",
    "improvement_from_first",
    "regression_from_best",
    "recent_window_requested",
    "recent_window_size",
    "recent_transition_count",
    "recent_improving_steps",
    "recent_degrading_steps",
    "recent_flat_steps",
    "recent_net_change",
    "recent_trend",
]


def _diagnostic() -> Diagnostic:
    return Diagnostic(
        code="recent_degradation",
        severity="warning",
        message="The recent metric history is degrading.",
        evidence={"count": 2, "details": {"epochs": [1, 2]}},
    )


def _facts_for(
    values: list[int | float],
    *,
    direction: str = "maximize",
    epochs: list[int] | None = None,
    recent_window: int = 5,
) -> dict[str, object]:
    resolved_epochs = epochs if epochs is not None else list(range(len(values)))
    records = [
        [epoch, value]
        for epoch, value in zip(resolved_epochs, values, strict=True)
    ]
    return build_metric_facts(
        records,
        direction,
        recent_window=recent_window,
    )


def _diagnostic_by_code(
    facts: dict[str, object],
    code: str,
) -> Diagnostic:
    return next(
        diagnostic
        for diagnostic in build_metric_diagnostics(facts)
        if diagnostic.code == code
    )


def test_diagnostic_accepts_valid_values():
    diagnostic = _diagnostic()

    assert diagnostic.code == "recent_degradation"
    assert diagnostic.severity == "warning"
    assert diagnostic.message == "The recent metric history is degrading."
    assert dict(diagnostic.evidence) == {
        "count": 2,
        "details": {"epochs": [1, 2]},
    }


def test_diagnostic_fields_have_stable_order():
    assert [field.name for field in fields(Diagnostic)] == [
        "code",
        "severity",
        "message",
        "evidence",
    ]


def test_diagnostic_is_frozen():
    diagnostic = _diagnostic()

    with pytest.raises(FrozenInstanceError):
        diagnostic.code = "changed"


def test_diagnostic_copies_original_evidence_mapping():
    evidence = {"count": 1}
    diagnostic = Diagnostic("valid_code", "info", "Message", evidence)

    evidence["count"] = 2
    evidence["extra"] = 3

    assert dict(diagnostic.evidence) == {"count": 1}


def test_diagnostic_evidence_is_top_level_read_only():
    diagnostic = _diagnostic()

    with pytest.raises(TypeError):
        diagnostic.evidence["count"] = 3


def test_diagnostic_evidence_is_mapping_proxy():
    assert isinstance(_diagnostic().evidence, MappingProxyType)


def test_diagnostic_evidence_preserves_insertion_order():
    diagnostic = Diagnostic(
        "valid_code",
        "info",
        "Message",
        {"second": 2, "first": 1, "third": 3},
    )

    assert list(diagnostic.evidence) == ["second", "first", "third"]


@pytest.mark.parametrize("invalid_code", [None, 1, True, b"valid_code"])
def test_diagnostic_rejects_non_string_code(invalid_code):
    with pytest.raises(TypeError, match="^code must be a string$"):
        Diagnostic(invalid_code, "info", "Message", {})


@pytest.mark.parametrize(
    "invalid_code",
    [
        "",
        "DuplicateEpochs",
        "duplicate-epochs",
        "duplicate epochs",
        "_duplicate_epochs",
        "duplicate_epochs_",
        "duplicate__epochs",
    ],
)
def test_diagnostic_rejects_invalid_snake_case_code(invalid_code):
    with pytest.raises(
        ValueError,
        match="^code must be a non-empty snake_case string$",
    ):
        Diagnostic(invalid_code, "info", "Message", {})


def test_diagnostic_accepts_numeric_snake_case_component():
    diagnostic = Diagnostic(
        "metric_2_warning",
        "warning",
        "Message",
        {},
    )

    assert diagnostic.code == "metric_2_warning"


@pytest.mark.parametrize("invalid_severity", [None, 1, True, b"info"])
def test_diagnostic_rejects_non_string_severity(invalid_severity):
    with pytest.raises(TypeError, match="^severity must be a string$"):
        Diagnostic("valid_code", invalid_severity, "Message", {})


@pytest.mark.parametrize("invalid_severity", ["", "critical", "INFO"])
def test_diagnostic_rejects_unknown_severity(invalid_severity):
    with pytest.raises(
        ValueError,
        match="^severity must be one of: info, warning$",
    ):
        Diagnostic("valid_code", invalid_severity, "Message", {})


@pytest.mark.parametrize("invalid_message", [None, 1, True, b"Message"])
def test_diagnostic_rejects_non_string_message(invalid_message):
    with pytest.raises(TypeError, match="^message must be a string$"):
        Diagnostic("valid_code", "info", invalid_message, {})


@pytest.mark.parametrize("invalid_message", ["", " ", "\t\r\n"])
def test_diagnostic_rejects_empty_message(invalid_message):
    with pytest.raises(ValueError, match="^message must not be empty$"):
        Diagnostic("valid_code", "info", invalid_message, {})


def test_diagnostic_does_not_strip_message():
    diagnostic = Diagnostic("valid_code", "info", " Message ", {})

    assert diagnostic.message == " Message "


@pytest.mark.parametrize("invalid_evidence", [None, [], (), "evidence"])
def test_diagnostic_rejects_non_mapping_evidence(invalid_evidence):
    with pytest.raises(TypeError, match="^evidence must be a mapping$"):
        Diagnostic("valid_code", "info", "Message", invalid_evidence)


@pytest.mark.parametrize("invalid_key", [1, None, True, ("key",)])
def test_diagnostic_rejects_non_string_evidence_key(invalid_key):
    with pytest.raises(
        TypeError,
        match="^evidence keys must be strings$",
    ):
        Diagnostic("valid_code", "info", "Message", {invalid_key: 1})


def test_diagnostic_to_dict_has_stable_field_order():
    result = diagnostic_to_dict(_diagnostic())

    assert list(result) == ["code", "severity", "message", "evidence"]


def test_diagnostic_to_dict_returns_plain_evidence_dict():
    result = diagnostic_to_dict(_diagnostic())

    assert type(result) is dict
    assert type(result["evidence"]) is dict


def test_diagnostic_to_dict_result_is_json_serializable():
    result = diagnostic_to_dict(_diagnostic())

    assert json.loads(json.dumps(result, allow_nan=False)) == result


def test_diagnostic_to_dict_evidence_is_independent():
    diagnostic = _diagnostic()
    result = diagnostic_to_dict(diagnostic)

    result["evidence"]["count"] = 99

    assert diagnostic.evidence["count"] == 2


def test_diagnostic_to_dict_deep_copies_nested_evidence():
    diagnostic = _diagnostic()
    result = diagnostic_to_dict(diagnostic)

    result["evidence"]["details"]["epochs"].append(3)

    assert diagnostic.evidence["details"] == {"epochs": [1, 2]}


@pytest.mark.parametrize("invalid_diagnostic", [None, {}, "diagnostic", 1])
def test_diagnostic_to_dict_rejects_non_diagnostic(invalid_diagnostic):
    with pytest.raises(
        TypeError,
        match="^diagnostic must be a Diagnostic$",
    ):
        diagnostic_to_dict(invalid_diagnostic)


def test_build_metric_facts_supports_maximize():
    facts = _facts_for([0.2, 0.5, 0.4])

    assert facts["best_value"] == pytest.approx(0.5)
    assert facts["best_epoch"] == 1
    assert facts["improvement_from_first"] == pytest.approx(0.3)
    assert facts["regression_from_best"] == pytest.approx(0.1)


def test_build_metric_facts_supports_minimize():
    facts = _facts_for([0.8, 0.5, 0.6], direction="minimize")

    assert facts["best_value"] == pytest.approx(0.5)
    assert facts["best_epoch"] == 1
    assert facts["improvement_from_first"] == pytest.approx(0.3)
    assert facts["regression_from_best"] == pytest.approx(0.1)


@pytest.mark.parametrize(
    ("values", "expected_index", "expected_ratio"),
    [
        ([3.0, 2.0, 1.0], 0, 0.0),
        ([1.0, 3.0, 2.0], 1, 0.5),
        ([1.0, 2.0, 3.0], 2, 1.0),
    ],
)
def test_build_metric_facts_reports_best_position(
    values,
    expected_index,
    expected_ratio,
):
    facts = _facts_for(values)

    assert facts["best_record_index"] == expected_index
    assert facts["best_progress_ratio"] == pytest.approx(expected_ratio)


def test_build_metric_facts_retains_first_tied_best_record():
    facts = _facts_for([1.0, 3.0, 2.0, 3.0])

    assert facts["best_epoch"] == 1
    assert facts["best_record_index"] == 1
    assert facts["best_progress_ratio"] == pytest.approx(1 / 3)


def test_build_metric_facts_handles_single_record():
    facts = _facts_for([0.5])

    assert facts["best_record_index"] == 0
    assert facts["best_progress_ratio"] is None
    assert facts["best_at_first_record"] is True
    assert facts["best_at_last_record"] is True
    assert facts["recent_trend"] == "insufficient_data"
    assert facts["recent_net_change"] == 0


def test_build_metric_facts_accepts_non_consecutive_increasing_epochs():
    facts = _facts_for([0.1, 0.2, 0.3], epochs=[0, 3, 10])

    assert facts["duplicate_epochs"] == []
    assert facts["non_monotonic_epoch_transitions"] == []


def test_build_metric_facts_supports_negative_metric_values():
    facts = _facts_for([-3.0, -1.0, -2.0])

    assert facts["improvement_from_first"] == pytest.approx(2.0)
    assert facts["regression_from_best"] == pytest.approx(1.0)


def test_build_metric_facts_supports_zero_metric_values():
    facts = _facts_for([0.0, 1.0, 0.0])

    assert facts["first_value"] == 0.0
    assert facts["last_value"] == 0.0
    assert facts["improvement_from_first"] == pytest.approx(1.0)


def test_build_metric_facts_preserves_integer_result_types():
    facts = _facts_for([1, 4, 2])

    assert facts["best_value"] == 4
    assert type(facts["best_value"]) is int
    assert type(facts["improvement_from_first"]) is int
    assert type(facts["regression_from_best"]) is int


def test_build_metric_facts_supports_float_values():
    facts = _facts_for([1.25, 1.75])

    assert facts["best_value"] == pytest.approx(1.75)
    assert facts["recent_net_change"] == pytest.approx(0.5)


def test_build_metric_facts_does_not_modify_records():
    records = [[0, 0.2], [1, 0.4], [2, 0.3]]
    original_records = deepcopy(records)

    build_metric_facts(records, "maximize")

    assert records == original_records


def test_build_metric_facts_has_exact_field_order():
    facts = _facts_for([0.2, 0.4])

    assert list(facts) == FACT_KEYS


def test_build_metric_facts_returns_json_friendly_data():
    facts = _facts_for([0.2, 0.4, 0.3], epochs=[0, 2, 1])

    assert json.loads(json.dumps(facts, allow_nan=False)) == facts


def test_build_metric_facts_reports_no_duplicate_epochs():
    assert _facts_for([1, 2, 3])["duplicate_epochs"] == []


def test_build_metric_facts_reports_adjacent_duplicate_epoch():
    facts = _facts_for([1, 2, 3], epochs=[0, 1, 1])

    assert facts["duplicate_epochs"] == [1]
    assert facts["non_monotonic_epoch_transitions"] == []


def test_build_metric_facts_reports_non_adjacent_duplicate_epoch():
    facts = _facts_for([1, 2, 3], epochs=[0, 1, 0])

    assert facts["duplicate_epochs"] == [0]


def test_build_metric_facts_lists_repeated_epoch_once():
    facts = _facts_for([1, 2, 3, 4], epochs=[0, 0, 0, 0])

    assert facts["duplicate_epochs"] == [0]


def test_build_metric_facts_preserves_duplicate_discovery_order():
    facts = _facts_for(
        [1, 2, 3, 4, 5, 6],
        epochs=[0, 1, 1, 2, 0, 1],
    )

    assert facts["duplicate_epochs"] == [1, 0]


def test_duplicate_epoch_is_not_itself_a_decreasing_transition():
    facts = _facts_for([1, 2], epochs=[3, 3])

    assert facts["duplicate_epochs"] == [3]
    assert facts["non_monotonic_epoch_transitions"] == []


def test_build_metric_facts_reports_single_epoch_decrease():
    facts = _facts_for([1, 2, 3, 4], epochs=[0, 3, 2, 5])

    assert facts["non_monotonic_epoch_transitions"] == [
        {
            "previous_record_index": 1,
            "current_record_index": 2,
            "previous_epoch": 3,
            "current_epoch": 2,
        }
    ]


def test_build_metric_facts_reports_multiple_epoch_decreases():
    facts = _facts_for([1, 2, 3, 4], epochs=[4, 2, 3, 1])

    assert facts["non_monotonic_epoch_transitions"] == [
        {
            "previous_record_index": 0,
            "current_record_index": 1,
            "previous_epoch": 4,
            "current_epoch": 2,
        },
        {
            "previous_record_index": 2,
            "current_record_index": 3,
            "previous_epoch": 3,
            "current_epoch": 1,
        },
    ]


def test_epoch_transition_fields_have_stable_order():
    facts = _facts_for([1, 2], epochs=[5, 1])
    transition = facts["non_monotonic_epoch_transitions"][0]

    assert list(transition) == [
        "previous_record_index",
        "current_record_index",
        "previous_epoch",
        "current_epoch",
    ]


def test_maximize_and_minimize_have_symmetric_change_facts():
    maximize = _facts_for([0.2, 0.4, 0.3])
    minimize = _facts_for([0.8, 0.6, 0.7], direction="minimize")

    for field in (
        "improvement_from_first",
        "regression_from_best",
        "recent_improving_steps",
        "recent_degrading_steps",
        "recent_flat_steps",
        "recent_net_change",
        "recent_trend",
    ):
        if isinstance(maximize[field], float):
            assert minimize[field] == pytest.approx(maximize[field])
        else:
            assert minimize[field] == maximize[field]


@pytest.mark.parametrize(
    ("values", "expected_trend", "improving", "degrading", "flat"),
    [
        ([1, 2, 3, 4], "improving", 3, 0, 0),
        ([4, 3, 2, 1], "degrading", 0, 3, 0),
        ([2, 2, 2, 2], "flat", 0, 0, 3),
        ([1, 2, 1, 2, 1], "mixed", 2, 2, 0),
        ([1, 2, 3, 4, 3], "improving", 3, 1, 0),
        ([4, 3, 2, 1, 2], "degrading", 1, 3, 0),
        ([1, 2, 2, 2, 2], "mixed", 1, 0, 3),
        ([2, 1, 1, 1, 1], "mixed", 0, 1, 3),
    ],
)
def test_build_metric_facts_classifies_recent_trend(
    values,
    expected_trend,
    improving,
    degrading,
    flat,
):
    facts = _facts_for(values)

    assert facts["recent_trend"] == expected_trend
    assert facts["recent_improving_steps"] == improving
    assert facts["recent_degrading_steps"] == degrading
    assert facts["recent_flat_steps"] == flat


def test_recent_window_uses_only_last_requested_records():
    facts = _facts_for([1, 4, 3, 2], recent_window=3)

    assert facts["recent_window_requested"] == 3
    assert facts["recent_window_size"] == 3
    assert facts["recent_transition_count"] == 2
    assert facts["recent_degrading_steps"] == 2
    assert facts["recent_trend"] == "degrading"
    assert facts["recent_net_change"] == -2


def test_recent_window_larger_than_history_uses_all_records():
    facts = _facts_for([1, 2, 3], recent_window=20)

    assert facts["recent_window_requested"] == 20
    assert facts["recent_window_size"] == 3
    assert facts["recent_transition_count"] == 2


def test_recent_window_two_uses_one_transition():
    facts = _facts_for([3, 1, 2], recent_window=2)

    assert facts["recent_window_size"] == 2
    assert facts["recent_transition_count"] == 1
    assert facts["recent_improving_steps"] == 1
    assert facts["recent_trend"] == "improving"


@pytest.mark.parametrize("invalid_window", [True, False, 2.0, "2", None])
def test_build_metric_facts_rejects_non_integer_recent_window(
    invalid_window,
):
    with pytest.raises(
        TypeError,
        match="^recent_window must be an integer$",
    ):
        build_metric_facts([[0, 1.0]], "maximize", recent_window=invalid_window)


@pytest.mark.parametrize("invalid_window", [1, 0, -1, -10])
def test_build_metric_facts_rejects_too_small_recent_window(
    invalid_window,
):
    with pytest.raises(
        ValueError,
        match="^recent_window must be at least 2$",
    ):
        build_metric_facts([[0, 1.0]], "maximize", recent_window=invalid_window)


@pytest.mark.parametrize(
    ("records", "invalid_index"),
    [
        ([[0, float("nan")], [1, 1.0]], 0),
        ([[0, 1.0], [1, float("inf")], [2, 2.0]], 1),
        ([[0, 1.0], [1, 2.0], [2, float("-inf")]], 2),
    ],
)
def test_build_metric_facts_rejects_non_finite_float(
    records,
    invalid_index,
):
    with pytest.raises(
        ValueError,
        match=rf"^records\[{invalid_index}\]\[1\] must be finite$",
    ):
        build_metric_facts(records, "maximize")


@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
def test_non_finite_error_identifies_last_record(value):
    with pytest.raises(
        ValueError,
        match=r"^records\[2\]\[1\] must be finite$",
    ):
        build_metric_facts([[0, 1.0], [1, 2.0], [2, value]], "maximize")


def test_finite_integer_values_are_accepted():
    assert _facts_for([1, 2, 3])["best_value"] == 3


def test_build_metric_facts_propagates_empty_records_error():
    with pytest.raises(ValueError, match="^records must not be empty$"):
        build_metric_facts([], "maximize")


def test_build_metric_facts_propagates_records_type_error():
    with pytest.raises(TypeError, match="^records must be a sequence$"):
        build_metric_facts({"epoch": 1}, "maximize")


def test_build_metric_facts_propagates_record_type_error():
    with pytest.raises(
        TypeError,
        match=r"^records\[0\] must be a sequence$",
    ):
        build_metric_facts([1], "maximize")


def test_build_metric_facts_propagates_record_length_error():
    with pytest.raises(
        ValueError,
        match=r"^records\[0\] must contain exactly two items$",
    ):
        build_metric_facts([[0, 1.0, 2.0]], "maximize")


def test_build_metric_facts_propagates_boolean_epoch_error():
    with pytest.raises(
        TypeError,
        match=r"^records\[0\]\[0\] must be an integer$",
    ):
        build_metric_facts([[True, 1.0]], "maximize")


def test_build_metric_facts_propagates_boolean_value_error():
    with pytest.raises(
        TypeError,
        match=r"^records\[0\]\[1\] must be a number$",
    ):
        build_metric_facts([[0, True]], "maximize")


def test_build_metric_facts_propagates_invalid_direction_error():
    with pytest.raises(
        ValueError,
        match="^direction must be 'maximize' or 'minimize'$",
    ):
        build_metric_facts([[0, 1.0]], "sideways")


def test_duplicate_epochs_diagnostic_has_fixed_content():
    diagnostic = _diagnostic_by_code(
        _facts_for([1, 2, 3], epochs=[0, 1, 1]),
        "duplicate_epochs",
    )

    assert diagnostic.severity == "warning"
    assert diagnostic.message == (
        "Duplicate epoch values were found in the metric history."
    )
    assert dict(diagnostic.evidence) == {"duplicate_epochs": [1]}


def test_non_monotonic_epochs_diagnostic_has_fixed_content():
    facts = _facts_for([1, 2, 3], epochs=[0, 2, 1])
    diagnostic = _diagnostic_by_code(facts, "non_monotonic_epochs")

    assert diagnostic.severity == "warning"
    assert diagnostic.message == (
        "Epoch values decrease in one or more adjacent records."
    )
    assert dict(diagnostic.evidence) == {
        "transitions": facts["non_monotonic_epoch_transitions"]
    }


def test_best_at_first_record_diagnostic_has_fixed_content():
    diagnostic = _diagnostic_by_code(
        _facts_for([3.0, 2.0, 1.0]),
        "best_at_first_record",
    )

    assert diagnostic.severity == "info"
    assert diagnostic.message == "The best value occurs at the first record."
    assert dict(diagnostic.evidence) == {
        "best_record_index": 0,
        "best_epoch": 0,
    }


def test_best_at_last_record_diagnostic_has_fixed_content():
    diagnostic = _diagnostic_by_code(
        _facts_for([1.0, 2.0, 3.0]),
        "best_at_last_record",
    )

    assert diagnostic.severity == "info"
    assert diagnostic.message == "The best value occurs at the last record."
    assert dict(diagnostic.evidence) == {
        "best_record_index": 2,
        "best_epoch": 2,
    }


def test_single_record_skips_both_best_position_diagnostics():
    codes = [
        diagnostic.code
        for diagnostic in build_metric_diagnostics(_facts_for([1.0]))
    ]

    assert "best_at_first_record" not in codes
    assert "best_at_last_record" not in codes


def test_no_improvement_diagnostic_has_fixed_content():
    diagnostic = _diagnostic_by_code(
        _facts_for([3.0, 2.0, 1.0]),
        "no_improvement",
    )

    assert diagnostic.severity == "warning"
    assert diagnostic.message == (
        "The metric did not improve beyond its first recorded value."
    )
    assert dict(diagnostic.evidence) == {
        "first_value": 3.0,
        "best_value": 3.0,
        "improvement_from_first": 0.0,
    }


def test_improvement_skips_no_improvement_diagnostic():
    codes = [
        diagnostic.code
        for diagnostic in build_metric_diagnostics(
            _facts_for([1.0, 2.0, 1.5])
        )
    ]

    assert "no_improvement" not in codes


def test_post_best_regression_diagnostic_has_fixed_content():
    diagnostic = _diagnostic_by_code(
        _facts_for([1.0, 3.0, 2.0]),
        "post_best_regression",
    )

    assert diagnostic.severity == "warning"
    assert diagnostic.message == (
        "The final value is worse than the best recorded value."
    )
    assert dict(diagnostic.evidence) == {
        "best_epoch": 1,
        "best_value": 3.0,
        "last_epoch": 2,
        "last_value": 2.0,
        "regression_from_best": 1.0,
    }


def test_last_equal_to_best_skips_post_best_regression():
    codes = [
        diagnostic.code
        for diagnostic in build_metric_diagnostics(
            _facts_for([1.0, 2.0, 2.0])
        )
    ]

    assert "post_best_regression" not in codes


@pytest.mark.parametrize(
    ("values", "code", "severity", "message"),
    [
        (
            [1, 2, 3],
            "recent_improvement",
            "info",
            "The recent metric history is improving.",
        ),
        (
            [3, 2, 1],
            "recent_degradation",
            "warning",
            "The recent metric history is degrading.",
        ),
        (
            [1, 1, 1],
            "recent_flat",
            "info",
            "The recent metric history is flat.",
        ),
        (
            [1, 2, 1],
            "recent_mixed",
            "info",
            "The recent metric history has mixed direction.",
        ),
    ],
)
def test_recent_trend_diagnostic_has_fixed_content(
    values,
    code,
    severity,
    message,
):
    facts = _facts_for(values)
    diagnostic = _diagnostic_by_code(facts, code)

    assert diagnostic.severity == severity
    assert diagnostic.message == message
    assert list(diagnostic.evidence) == [
        "recent_window_size",
        "recent_transition_count",
        "recent_improving_steps",
        "recent_degrading_steps",
        "recent_flat_steps",
        "recent_net_change",
    ]
    assert dict(diagnostic.evidence) == {
        key: facts[key]
        for key in diagnostic.evidence
    }


def test_insufficient_history_diagnostic_has_fixed_content():
    diagnostic = _diagnostic_by_code(
        _facts_for([1.0]),
        "insufficient_history_for_trend",
    )

    assert diagnostic.severity == "info"
    assert diagnostic.message == (
        "At least two records are required to determine a recent trend."
    )
    assert dict(diagnostic.evidence) == {
        "record_count": 1,
        "recent_window_size": 1,
    }


def test_build_metric_diagnostics_has_stable_rule_order():
    facts = _facts_for(
        [5.0, 4.0, 3.0],
        epochs=[2, 2, 1],
    )

    assert [
        diagnostic.code
        for diagnostic in build_metric_diagnostics(facts)
    ] == [
        "duplicate_epochs",
        "non_monotonic_epochs",
        "best_at_first_record",
        "no_improvement",
        "post_best_regression",
        "recent_degradation",
    ]


def test_build_metric_diagnostics_returns_tuple():
    assert isinstance(
        build_metric_diagnostics(_facts_for([1, 2, 3])),
        tuple,
    )


def test_build_metric_diagnostics_does_not_modify_facts():
    facts = _facts_for([3, 2, 1], epochs=[1, 1, 0])
    original_facts = deepcopy(facts)

    build_metric_diagnostics(facts)

    assert facts == original_facts


def test_build_metric_diagnostics_propagates_missing_field_key_error():
    facts = _facts_for([1, 2, 3])
    del facts["duplicate_epochs"]

    with pytest.raises(KeyError) as error:
        build_metric_diagnostics(facts)

    assert error.value.args == ("duplicate_epochs",)


@pytest.mark.parametrize("invalid_facts", [None, [], (), "facts", 1])
def test_build_metric_diagnostics_rejects_non_mapping(invalid_facts):
    with pytest.raises(TypeError, match="^facts must be a mapping$"):
        build_metric_diagnostics(invalid_facts)


def test_all_generated_diagnostics_are_json_serializable():
    facts = _facts_for([5.0, 4.0, 3.0], epochs=[2, 2, 1])
    serialized = [
        diagnostic_to_dict(diagnostic)
        for diagnostic in build_metric_diagnostics(facts)
    ]

    assert json.loads(json.dumps(serialized, allow_nan=False)) == serialized


def test_diagnostic_evidence_does_not_alias_nested_fact_lists():
    facts = _facts_for([1, 2, 3], epochs=[0, 2, 1])
    diagnostic = _diagnostic_by_code(facts, "non_monotonic_epochs")

    facts["non_monotonic_epoch_transitions"][0]["current_epoch"] = 99

    assert diagnostic.evidence["transitions"][0]["current_epoch"] == 1

COMPARISON_FACT_KEYS = [
    "total_experiments",
    "successful_experiments",
    "failed_experiments",
    "success_rate",
    "sort_by",
    "descending",
    "ranked_experiment_count",
    "top_experiment_name",
    "top_value",
    "second_experiment_name",
    "second_value",
    "worst_experiment_name",
    "worst_value",
    "top_vs_second_gap",
    "best_vs_worst_gap",
    "tied_best_experiments",
    "has_failures",
    "has_successful_experiments",
    "single_successful_experiment",
]


def _default_comparison_payload(
    values: list[int | float],
    *,
    sort_by: str = "best_r2",
    descending: bool = True,
    failed: int = 0,
) -> dict[str, object]:
    records = []
    for index, value in enumerate(values):
        records.append(
            {
                "experiment_name": f"exp_{index + 1}",
                "experiment_dir": f"/experiments/exp_{index + 1}",
                "best_r2": value,
                "best_r2_epoch": index,
                "best_racc": value,
                "best_racc_epoch": index,
            }
        )

    return {
        "sort_by": sort_by,
        "descending": descending,
        "experiment_counts": {
            "total": len(values) + failed,
            "successful": len(values),
            "failed": failed,
        },
        "comparison_records": records,
        "failed_experiments": [
            {
                "experiment_name": f"failed_{index + 1}",
                "experiment_dir": f"/experiments/failed_{index + 1}",
                "error_type": "ValueError",
                "error_message": "Invalid experiment",
            }
            for index in range(failed)
        ],
    }


def _dynamic_comparison_payload(
    values: list[int | float],
    *,
    metric_name: str = "mae",
    descending: bool = False,
    failed: int = 0,
) -> dict[str, object]:
    records = []
    for index, value in enumerate(values):
        records.append(
            {
                "experiment_name": f"exp_{index + 1}",
                "experiment_dir": f"/experiments/exp_{index + 1}",
                "metrics": {
                    metric_name: {
                        "record_count": 3,
                        "first_epoch": 0,
                        "first_value": value,
                        "last_epoch": 2,
                        "last_value": value,
                        "best_epoch": 1,
                        "best_value": value,
                    },
                },
            }
        )

    return {
        "sort_by": metric_name,
        "descending": descending,
        "metric_specs": [
            {
                "name": metric_name,
                "path": ["valid", "app", metric_name],
                "direction": (
                    "maximize" if descending else "minimize"
                ),
                "display_name": metric_name.upper(),
                "precision": 4,
            }
        ],
        "experiment_counts": {
            "total": len(values) + failed,
            "successful": len(values),
            "failed": failed,
        },
        "comparison_records": records,
        "failed_experiments": [],
    }


def _comparison_diagnostic_by_code(
    facts: dict[str, object],
    code: str,
) -> Diagnostic:
    return next(
        diagnostic
        for diagnostic in build_comparison_diagnostics(facts)
        if diagnostic.code == code
    )


def test_build_comparison_facts_has_exact_field_order():
    facts = build_comparison_facts(
        _default_comparison_payload([0.9, 0.8])
    )

    assert list(facts) == COMPARISON_FACT_KEYS


def test_build_comparison_facts_handles_empty_batch():
    facts = build_comparison_facts(
        _default_comparison_payload([])
    )

    assert facts == {
        "total_experiments": 0,
        "successful_experiments": 0,
        "failed_experiments": 0,
        "success_rate": None,
        "sort_by": "best_r2",
        "descending": True,
        "ranked_experiment_count": 0,
        "top_experiment_name": None,
        "top_value": None,
        "second_experiment_name": None,
        "second_value": None,
        "worst_experiment_name": None,
        "worst_value": None,
        "top_vs_second_gap": None,
        "best_vs_worst_gap": None,
        "tied_best_experiments": [],
        "has_failures": False,
        "has_successful_experiments": False,
        "single_successful_experiment": False,
    }


def test_build_comparison_facts_handles_all_failed_experiments():
    facts = build_comparison_facts(
        _default_comparison_payload([], failed=3)
    )

    assert facts["total_experiments"] == 3
    assert facts["successful_experiments"] == 0
    assert facts["failed_experiments"] == 3
    assert facts["success_rate"] == 0.0
    assert facts["has_failures"] is True
    assert facts["has_successful_experiments"] is False


def test_build_comparison_facts_handles_single_success():
    facts = build_comparison_facts(
        _default_comparison_payload([0.8])
    )

    assert facts["top_experiment_name"] == "exp_1"
    assert facts["top_value"] == pytest.approx(0.8)
    assert facts["second_experiment_name"] is None
    assert facts["second_value"] is None
    assert facts["worst_experiment_name"] == "exp_1"
    assert facts["worst_value"] == pytest.approx(0.8)
    assert facts["top_vs_second_gap"] is None
    assert facts["best_vs_worst_gap"] == pytest.approx(0.0)
    assert facts["tied_best_experiments"] == ["exp_1"]
    assert facts["single_successful_experiment"] is True


def test_build_comparison_facts_calculates_descending_gaps():
    facts = build_comparison_facts(
        _default_comparison_payload([0.9, 0.7, 0.4])
    )

    assert facts["top_experiment_name"] == "exp_1"
    assert facts["second_experiment_name"] == "exp_2"
    assert facts["worst_experiment_name"] == "exp_3"
    assert facts["top_vs_second_gap"] == pytest.approx(0.2)
    assert facts["best_vs_worst_gap"] == pytest.approx(0.5)


def test_build_comparison_facts_calculates_ascending_gaps():
    facts = build_comparison_facts(
        _default_comparison_payload(
            [0.1, 0.3, 0.8],
            descending=False,
        )
    )

    assert facts["top_vs_second_gap"] == pytest.approx(0.2)
    assert facts["best_vs_worst_gap"] == pytest.approx(0.7)


def test_build_comparison_facts_supports_negative_ranked_values():
    facts = build_comparison_facts(
        _default_comparison_payload([-1.0, -2.0, -4.0])
    )

    assert facts["top_value"] == pytest.approx(-1.0)
    assert facts["top_vs_second_gap"] == pytest.approx(1.0)
    assert facts["best_vs_worst_gap"] == pytest.approx(3.0)


def test_build_comparison_facts_supports_zero_ranked_values():
    facts = build_comparison_facts(
        _default_comparison_payload([0, -1, -2])
    )

    assert facts["top_value"] == 0
    assert facts["top_vs_second_gap"] == 1
    assert facts["best_vs_worst_gap"] == 2


def test_build_comparison_facts_extracts_default_best_racc():
    payload = _default_comparison_payload(
        [0.92, 0.90],
        sort_by="best_racc",
    )
    payload["comparison_records"][0]["best_r2"] = 0.1
    payload["comparison_records"][1]["best_r2"] = 0.99

    facts = build_comparison_facts(payload)

    assert facts["sort_by"] == "best_racc"
    assert facts["top_value"] == pytest.approx(0.92)


def test_build_comparison_facts_extracts_dynamic_best_value():
    facts = build_comparison_facts(
        _dynamic_comparison_payload([0.08, 0.10, 0.14])
    )

    assert facts["sort_by"] == "mae"
    assert facts["descending"] is False
    assert facts["top_value"] == pytest.approx(0.08)
    assert facts["second_value"] == pytest.approx(0.10)
    assert facts["worst_value"] == pytest.approx(0.14)


def test_dynamic_minimize_comparison_has_positive_gaps():
    facts = build_comparison_facts(
        _dynamic_comparison_payload([0.07, 0.08, 0.11])
    )

    assert facts["top_vs_second_gap"] == pytest.approx(0.01)
    assert facts["best_vs_worst_gap"] == pytest.approx(0.04)


def test_build_comparison_facts_reports_tied_best_experiments():
    facts = build_comparison_facts(
        _default_comparison_payload([0.9, 0.9, 0.8])
    )

    assert facts["top_vs_second_gap"] == pytest.approx(0.0)
    assert facts["tied_best_experiments"] == ["exp_1", "exp_2"]


def test_build_comparison_facts_reports_all_ranked_experiments_tied():
    facts = build_comparison_facts(
        _dynamic_comparison_payload([0.2, 0.2, 0.2])
    )

    assert facts["tied_best_experiments"] == [
        "exp_1",
        "exp_2",
        "exp_3",
    ]
    assert facts["top_vs_second_gap"] == pytest.approx(0.0)
    assert facts["best_vs_worst_gap"] == pytest.approx(0.0)


def test_build_comparison_facts_calculates_success_rate():
    facts = build_comparison_facts(
        _default_comparison_payload([0.9, 0.8], failed=2)
    )

    assert facts["success_rate"] == pytest.approx(0.5)


def test_build_comparison_facts_does_not_modify_payload():
    payload = _dynamic_comparison_payload(
        [0.08, 0.09],
        failed=1,
    )
    original_payload = deepcopy(payload)

    build_comparison_facts(payload)

    assert payload == original_payload


def test_build_comparison_facts_returns_json_friendly_data():
    facts = build_comparison_facts(
        _default_comparison_payload([0.9, 0.8], failed=1)
    )

    assert json.loads(json.dumps(facts, allow_nan=False)) == facts


@pytest.mark.parametrize("invalid_payload", [None, [], (), "payload", 1])
def test_build_comparison_facts_rejects_non_mapping(invalid_payload):
    with pytest.raises(
        TypeError,
        match="^comparison_payload must be a mapping$",
    ):
        build_comparison_facts(invalid_payload)


def test_build_comparison_facts_rejects_non_mapping_counts():
    payload = _default_comparison_payload([0.9])
    payload["experiment_counts"] = []

    with pytest.raises(
        TypeError,
        match=(
            r"^comparison_payload\['experiment_counts'\] "
            r"must be a mapping$"
        ),
    ):
        build_comparison_facts(payload)


@pytest.mark.parametrize("key", ["total", "successful", "failed"])
@pytest.mark.parametrize("invalid_value", [True, 1.0, "1", None])
def test_build_comparison_facts_rejects_non_integer_counts(
    key,
    invalid_value,
):
    payload = _default_comparison_payload([0.9])
    payload["experiment_counts"][key] = invalid_value

    with pytest.raises(
        TypeError,
        match=(
            rf"^experiment_counts\['{key}'\] "
            r"must be an integer$"
        ),
    ):
        build_comparison_facts(payload)


@pytest.mark.parametrize("key", ["total", "successful", "failed"])
def test_build_comparison_facts_rejects_negative_counts(key):
    payload = _default_comparison_payload([0.9])
    payload["experiment_counts"][key] = -1

    with pytest.raises(
        ValueError,
        match=(
            rf"^experiment_counts\['{key}'\] "
            r"must be non-negative$"
        ),
    ):
        build_comparison_facts(payload)


def test_build_comparison_facts_rejects_inconsistent_counts():
    payload = _default_comparison_payload([0.9])
    payload["experiment_counts"]["total"] = 5

    with pytest.raises(
        ValueError,
        match=(
            "^experiment counts must satisfy "
            "total == successful \\+ failed$"
        ),
    ):
        build_comparison_facts(payload)


def test_build_comparison_facts_rejects_non_sequence_records():
    payload = _default_comparison_payload([0.9])
    payload["comparison_records"] = {}

    with pytest.raises(
        TypeError,
        match=(
            r"^comparison_payload\['comparison_records'\] "
            r"must be a sequence$"
        ),
    ):
        build_comparison_facts(payload)


def test_build_comparison_facts_rejects_record_count_mismatch():
    payload = _default_comparison_payload([0.9])
    payload["comparison_records"].append(
        deepcopy(payload["comparison_records"][0])
    )

    with pytest.raises(
        ValueError,
        match=(
            "^comparison_records length must equal "
            "successful experiment count$"
        ),
    ):
        build_comparison_facts(payload)


@pytest.mark.parametrize("invalid_sort_by", [None, 1, True, []])
def test_build_comparison_facts_rejects_non_string_sort_by(
    invalid_sort_by,
):
    payload = _default_comparison_payload([0.9])
    payload["sort_by"] = invalid_sort_by

    with pytest.raises(
        TypeError,
        match=r"^comparison_payload\['sort_by'\] must be a string$",
    ):
        build_comparison_facts(payload)


def test_build_comparison_facts_rejects_empty_sort_by():
    payload = _default_comparison_payload([0.9])
    payload["sort_by"] = ""

    with pytest.raises(
        ValueError,
        match=r"^comparison_payload\['sort_by'\] must not be empty$",
    ):
        build_comparison_facts(payload)


@pytest.mark.parametrize("invalid_descending", [0, 1, None, "true"])
def test_build_comparison_facts_rejects_non_boolean_descending(
    invalid_descending,
):
    payload = _default_comparison_payload([0.9])
    payload["descending"] = invalid_descending

    with pytest.raises(
        TypeError,
        match=(
            r"^comparison_payload\['descending'\] "
            r"must be a boolean$"
        ),
    ):
        build_comparison_facts(payload)


def test_build_comparison_facts_rejects_non_mapping_record():
    payload = _default_comparison_payload([0.9])
    payload["comparison_records"][0] = []

    with pytest.raises(
        TypeError,
        match=r"^comparison_records\[0\] must be a mapping$",
    ):
        build_comparison_facts(payload)


@pytest.mark.parametrize("invalid_name", [None, 1, True, []])
def test_build_comparison_facts_rejects_non_string_name(invalid_name):
    payload = _default_comparison_payload([0.9])
    payload["comparison_records"][0]["experiment_name"] = invalid_name

    with pytest.raises(
        TypeError,
        match=(
            r"^comparison_records\[0\]\['experiment_name'\] "
            r"must be a string$"
        ),
    ):
        build_comparison_facts(payload)


def test_build_comparison_facts_rejects_empty_experiment_name():
    payload = _default_comparison_payload([0.9])
    payload["comparison_records"][0]["experiment_name"] = ""

    with pytest.raises(
        ValueError,
        match=(
            r"^comparison_records\[0\]\['experiment_name'\] "
            r"must not be empty$"
        ),
    ):
        build_comparison_facts(payload)


def test_build_comparison_facts_propagates_missing_default_metric_key():
    payload = _default_comparison_payload([0.9])
    del payload["comparison_records"][0]["best_r2"]

    with pytest.raises(KeyError) as error:
        build_comparison_facts(payload)

    assert error.value.args == ("best_r2",)


def test_build_comparison_facts_propagates_missing_dynamic_metrics_key():
    payload = _dynamic_comparison_payload([0.08])
    del payload["comparison_records"][0]["metrics"]

    with pytest.raises(KeyError) as error:
        build_comparison_facts(payload)

    assert error.value.args == ("metrics",)


@pytest.mark.parametrize("invalid_value", [None, "0.9", True, []])
def test_build_comparison_facts_rejects_non_numeric_ranked_value(
    invalid_value,
):
    payload = _default_comparison_payload([0.9])
    payload["comparison_records"][0]["best_r2"] = invalid_value

    with pytest.raises(
        TypeError,
        match=(
            "^comparison_records\\[0\\] ranked metric "
            "value must be a number$"
        ),
    ):
        build_comparison_facts(payload)


@pytest.mark.parametrize(
    "invalid_value",
    [float("nan"), float("inf"), float("-inf")],
)
def test_build_comparison_facts_rejects_non_finite_ranked_value(
    invalid_value,
):
    payload = _default_comparison_payload([0.9])
    payload["comparison_records"][0]["best_r2"] = invalid_value

    with pytest.raises(
        ValueError,
        match=(
            "^comparison_records\\[0\\] ranked metric "
            "value must be finite$"
        ),
    ):
        build_comparison_facts(payload)


def test_no_successful_experiments_diagnostic_has_fixed_content():
    facts = build_comparison_facts(
        _default_comparison_payload([], failed=2)
    )
    diagnostic = _comparison_diagnostic_by_code(
        facts,
        "no_successful_experiments",
    )

    assert diagnostic.severity == "warning"
    assert diagnostic.message == (
        "No experiments were analyzed successfully."
    )
    assert list(diagnostic.evidence) == [
        "total_experiments",
        "failed_experiments",
    ]
    assert dict(diagnostic.evidence) == {
        "total_experiments": 2,
        "failed_experiments": 2,
    }


def test_failed_experiments_present_diagnostic_has_fixed_content():
    facts = build_comparison_facts(
        _default_comparison_payload([0.9], failed=2)
    )
    diagnostic = _comparison_diagnostic_by_code(
        facts,
        "failed_experiments_present",
    )

    assert diagnostic.severity == "warning"
    assert diagnostic.message == (
        "One or more experiments failed during analysis."
    )
    assert list(diagnostic.evidence) == [
        "failed_experiments",
        "total_experiments",
    ]
    assert dict(diagnostic.evidence) == {
        "failed_experiments": 2,
        "total_experiments": 3,
    }


def test_single_successful_experiment_diagnostic_has_fixed_content():
    facts = build_comparison_facts(
        _default_comparison_payload([0.9])
    )
    diagnostic = _comparison_diagnostic_by_code(
        facts,
        "single_successful_experiment",
    )

    assert diagnostic.severity == "info"
    assert diagnostic.message == (
        "Only one experiment was analyzed successfully."
    )
    assert list(diagnostic.evidence) == [
        "successful_experiments",
        "top_experiment_name",
    ]
    assert dict(diagnostic.evidence) == {
        "successful_experiments": 1,
        "top_experiment_name": "exp_1",
    }


def test_tied_best_experiments_diagnostic_has_fixed_content():
    facts = build_comparison_facts(
        _default_comparison_payload([0.9, 0.9, 0.8])
    )
    diagnostic = _comparison_diagnostic_by_code(
        facts,
        "tied_best_experiments",
    )

    assert diagnostic.severity == "info"
    assert diagnostic.message == (
        "Multiple experiments share the best ranked value."
    )
    assert list(diagnostic.evidence) == [
        "sort_by",
        "top_value",
        "experiment_names",
    ]
    assert dict(diagnostic.evidence) == {
        "sort_by": "best_r2",
        "top_value": 0.9,
        "experiment_names": ["exp_1", "exp_2"],
    }


def test_build_comparison_diagnostics_skips_inactive_rules():
    facts = build_comparison_facts(
        _default_comparison_payload([0.9, 0.8])
    )

    assert build_comparison_diagnostics(facts) == ()


def test_build_comparison_diagnostics_has_stable_rule_order():
    facts = {
        "has_successful_experiments": False,
        "has_failures": True,
        "single_successful_experiment": True,
        "tied_best_experiments": ["exp_1", "exp_2"],
        "total_experiments": 2,
        "failed_experiments": 2,
        "successful_experiments": 1,
        "top_experiment_name": "exp_1",
        "sort_by": "best_r2",
        "top_value": 0.9,
    }

    assert [
        diagnostic.code
        for diagnostic in build_comparison_diagnostics(facts)
    ] == [
        "no_successful_experiments",
        "failed_experiments_present",
        "single_successful_experiment",
        "tied_best_experiments",
    ]


def test_build_comparison_diagnostics_returns_tuple():
    facts = build_comparison_facts(
        _default_comparison_payload([0.9], failed=1)
    )

    assert isinstance(build_comparison_diagnostics(facts), tuple)


def test_build_comparison_diagnostics_does_not_modify_facts():
    facts = build_comparison_facts(
        _default_comparison_payload([0.9, 0.9], failed=1)
    )
    original_facts = deepcopy(facts)

    build_comparison_diagnostics(facts)

    assert facts == original_facts


def test_comparison_diagnostic_copies_tied_name_evidence():
    facts = build_comparison_facts(
        _default_comparison_payload([0.9, 0.9])
    )
    diagnostic = _comparison_diagnostic_by_code(
        facts,
        "tied_best_experiments",
    )

    facts["tied_best_experiments"].append("exp_3")

    assert diagnostic.evidence["experiment_names"] == [
        "exp_1",
        "exp_2",
    ]


def test_build_comparison_diagnostics_propagates_missing_key():
    facts = build_comparison_facts(
        _default_comparison_payload([0.9])
    )
    del facts["has_successful_experiments"]

    with pytest.raises(KeyError) as error:
        build_comparison_diagnostics(facts)

    assert error.value.args == ("has_successful_experiments",)


@pytest.mark.parametrize("invalid_facts", [None, [], (), "facts", 1])
def test_build_comparison_diagnostics_rejects_non_mapping(
    invalid_facts,
):
    with pytest.raises(TypeError, match="^facts must be a mapping$"):
        build_comparison_diagnostics(invalid_facts)


def test_all_comparison_diagnostics_are_json_serializable():
    facts = build_comparison_facts(
        _default_comparison_payload([0.9, 0.9], failed=1)
    )
    serialized = [
        diagnostic_to_dict(diagnostic)
        for diagnostic in build_comparison_diagnostics(facts)
    ]

    assert json.loads(json.dumps(serialized, allow_nan=False)) == serialized

# ---------------------------------------------------------------------------
# Stage 9E-1: deterministic diagnostic recommendations
# ---------------------------------------------------------------------------


def _recommendation() -> Recommendation:
    return Recommendation(
        code="restore_best_checkpoint",
        message=(
            "Prefer the best checkpoint over the final checkpoint and "
            "review the cause of recent regression."
        ),
        diagnostic_codes=(
            "post_best_regression",
            "recent_degradation",
        ),
    )


def _diagnostic_with_code(code: str) -> Diagnostic:
    return Diagnostic(
        code=code,
        severity="warning",
        message=f"Diagnostic for {code}.",
        evidence={},
    )


def test_recommendation_accepts_valid_values():
    recommendation = _recommendation()

    assert recommendation.code == "restore_best_checkpoint"
    assert recommendation.message.startswith("Prefer the best checkpoint")
    assert recommendation.diagnostic_codes == (
        "post_best_regression",
        "recent_degradation",
    )


def test_recommendation_fields_have_stable_order():
    assert [field.name for field in fields(Recommendation)] == [
        "code",
        "message",
        "diagnostic_codes",
    ]


def test_recommendation_is_frozen():
    recommendation = _recommendation()

    with pytest.raises(FrozenInstanceError):
        recommendation.code = "changed"


def test_recommendation_normalizes_diagnostic_codes_to_tuple():
    recommendation = Recommendation(
        "valid_recommendation",
        "Message",
        ["first_code", "second_code"],
    )

    assert recommendation.diagnostic_codes == (
        "first_code",
        "second_code",
    )
    assert type(recommendation.diagnostic_codes) is tuple


@pytest.mark.parametrize("invalid_code", [None, 1, True, b"valid_code"])
def test_recommendation_rejects_non_string_code(invalid_code):
    with pytest.raises(TypeError, match="^code must be a string$"):
        Recommendation(invalid_code, "Message", ("diagnostic_code",))


@pytest.mark.parametrize(
    "invalid_code",
    [
        "",
        "RestoreBest",
        "restore-best",
        "restore best",
        "_restore_best",
        "restore_best_",
        "restore__best",
    ],
)
def test_recommendation_rejects_invalid_code(invalid_code):
    with pytest.raises(
        ValueError,
        match="^code must be a non-empty snake_case string$",
    ):
        Recommendation(invalid_code, "Message", ("diagnostic_code",))


@pytest.mark.parametrize("invalid_message", [None, 1, True, b"Message"])
def test_recommendation_rejects_non_string_message(invalid_message):
    with pytest.raises(TypeError, match="^message must be a string$"):
        Recommendation(
            "valid_recommendation",
            invalid_message,
            ("diagnostic_code",),
        )


@pytest.mark.parametrize("invalid_message", ["", " ", "\t\r\n"])
def test_recommendation_rejects_empty_message(invalid_message):
    with pytest.raises(ValueError, match="^message must not be empty$"):
        Recommendation(
            "valid_recommendation",
            invalid_message,
            ("diagnostic_code",),
        )


@pytest.mark.parametrize(
    "invalid_codes",
    [None, 1, {"diagnostic_code": True}, "diagnostic_code", b"code"],
)
def test_recommendation_rejects_invalid_diagnostic_codes_container(
    invalid_codes,
):
    with pytest.raises(
        TypeError,
        match="^diagnostic_codes must be a sequence$",
    ):
        Recommendation(
            "valid_recommendation",
            "Message",
            invalid_codes,
        )


def test_recommendation_rejects_empty_diagnostic_codes():
    with pytest.raises(
        ValueError,
        match="^diagnostic_codes must not be empty$",
    ):
        Recommendation("valid_recommendation", "Message", ())


@pytest.mark.parametrize("invalid_code", [None, 1, True, b"code"])
def test_recommendation_rejects_non_string_diagnostic_code(
    invalid_code,
):
    with pytest.raises(
        TypeError,
        match=r"^diagnostic_codes\[0\] must be a string$",
    ):
        Recommendation(
            "valid_recommendation",
            "Message",
            (invalid_code,),
        )


@pytest.mark.parametrize(
    "invalid_code",
    ["", "DiagnosticCode", "diagnostic-code", "diagnostic code"],
)
def test_recommendation_rejects_invalid_diagnostic_code(
    invalid_code,
):
    with pytest.raises(
        ValueError,
        match=(
            r"^diagnostic_codes\[0\] must be a "
            r"non-empty snake_case string$"
        ),
    ):
        Recommendation(
            "valid_recommendation",
            "Message",
            (invalid_code,),
        )


def test_recommendation_rejects_duplicate_diagnostic_codes():
    with pytest.raises(
        ValueError,
        match="^diagnostic_codes must not contain duplicates$",
    ):
        Recommendation(
            "valid_recommendation",
            "Message",
            ("same_code", "same_code"),
        )


def test_recommendation_to_dict_has_stable_field_order():
    result = recommendation_to_dict(_recommendation())

    assert list(result) == [
        "code",
        "message",
        "diagnostic_codes",
    ]


def test_recommendation_to_dict_returns_json_friendly_data():
    result = recommendation_to_dict(_recommendation())

    assert type(result) is dict
    assert type(result["diagnostic_codes"]) is list
    assert json.loads(json.dumps(result, allow_nan=False)) == result


def test_recommendation_to_dict_result_is_independent():
    recommendation = _recommendation()
    result = recommendation_to_dict(recommendation)

    result["diagnostic_codes"].append("another_code")

    assert recommendation.diagnostic_codes == (
        "post_best_regression",
        "recent_degradation",
    )


@pytest.mark.parametrize(
    "invalid_recommendation",
    [None, {}, "recommendation", 1],
)
def test_recommendation_to_dict_rejects_invalid_value(
    invalid_recommendation,
):
    with pytest.raises(
        TypeError,
        match="^recommendation must be a Recommendation$",
    ):
        recommendation_to_dict(invalid_recommendation)


def test_build_recommendations_returns_empty_tuple_for_no_diagnostics():
    assert build_recommendations(()) == ()


def test_build_recommendations_returns_tuple():
    result = build_recommendations(
        (_diagnostic_with_code("recent_degradation"),)
    )

    assert isinstance(result, tuple)


@pytest.mark.parametrize(
    "invalid_diagnostics",
    [None, 1, {}, "diagnostic", b"diagnostic"],
)
def test_build_recommendations_rejects_invalid_container(
    invalid_diagnostics,
):
    with pytest.raises(
        TypeError,
        match="^diagnostics must be a sequence$",
    ):
        build_recommendations(invalid_diagnostics)


@pytest.mark.parametrize("invalid_member", [None, {}, "diagnostic", 1])
def test_build_recommendations_rejects_invalid_member(invalid_member):
    with pytest.raises(
        TypeError,
        match=r"^diagnostics\[0\] must be a Diagnostic$",
    ):
        build_recommendations([invalid_member])


def test_build_recommendations_ignores_unknown_diagnostic_codes():
    diagnostics = (
        _diagnostic_with_code("future_unknown_diagnostic"),
    )

    assert build_recommendations(diagnostics) == ()


def test_build_recommendations_consolidates_metric_history_repairs():
    diagnostics = (
        _diagnostic_with_code("non_monotonic_epochs"),
        _diagnostic_with_code("duplicate_epochs"),
    )

    recommendation = build_recommendations(diagnostics)[0]

    assert recommendation.code == "repair_metric_history"
    assert recommendation.diagnostic_codes == (
        "duplicate_epochs",
        "non_monotonic_epochs",
    )


def test_build_recommendations_consolidates_regression_signals():
    diagnostics = (
        _diagnostic_with_code("recent_degradation"),
        _diagnostic_with_code("post_best_regression"),
    )

    recommendation = build_recommendations(diagnostics)[0]

    assert recommendation.code == "restore_best_checkpoint"
    assert recommendation.diagnostic_codes == (
        "post_best_regression",
        "recent_degradation",
    )


def test_build_recommendations_deduplicates_repeated_diagnostics():
    diagnostic = _diagnostic_with_code("recent_degradation")

    recommendations = build_recommendations(
        (diagnostic, diagnostic, diagnostic)
    )

    assert [item.code for item in recommendations] == [
        "restore_best_checkpoint"
    ]


def test_build_recommendations_has_canonical_order():
    diagnostics = (
        _diagnostic_with_code("tied_best_experiments"),
        _diagnostic_with_code("recent_degradation"),
        _diagnostic_with_code("duplicate_epochs"),
        _diagnostic_with_code("single_successful_experiment"),
    )

    assert [
        recommendation.code
        for recommendation in build_recommendations(diagnostics)
    ] == [
        "repair_metric_history",
        "restore_best_checkpoint",
        "add_comparison_experiments",
        "apply_secondary_tie_breaker",
    ]


def test_build_recommendations_is_independent_of_input_order():
    diagnostics = (
        _diagnostic_with_code("duplicate_epochs"),
        _diagnostic_with_code("recent_degradation"),
        _diagnostic_with_code("single_successful_experiment"),
    )

    forward = build_recommendations(diagnostics)
    reversed_result = build_recommendations(tuple(reversed(diagnostics)))

    assert forward == reversed_result


def test_build_recommendations_does_not_modify_input_sequence():
    diagnostics = [
        _diagnostic_with_code("recent_degradation"),
        _diagnostic_with_code("post_best_regression"),
    ]
    original = list(diagnostics)

    build_recommendations(diagnostics)

    assert diagnostics == original


@pytest.mark.parametrize(
    ("diagnostic_code", "recommendation_code"),
    [
        ("duplicate_epochs", "repair_metric_history"),
        ("non_monotonic_epochs", "repair_metric_history"),
        ("best_at_first_record", "verify_training_progress"),
        ("no_improvement", "verify_training_progress"),
        ("best_at_last_record", "consider_extended_training"),
        ("recent_improvement", "consider_extended_training"),
        ("post_best_regression", "restore_best_checkpoint"),
        ("recent_degradation", "restore_best_checkpoint"),
        (
            "insufficient_history_for_trend",
            "collect_more_history",
        ),
        ("recent_flat", "review_optimization_plateau"),
        ("recent_mixed", "avoid_trend_conclusion"),
        ("no_successful_experiments", "resolve_analysis_failures"),
        ("failed_experiments_present", "resolve_analysis_failures"),
        (
            "single_successful_experiment",
            "add_comparison_experiments",
        ),
        ("tied_best_experiments", "apply_secondary_tie_breaker"),
    ],
)
def test_every_current_diagnostic_code_has_a_recommendation(
    diagnostic_code,
    recommendation_code,
):
    recommendations = build_recommendations(
        (_diagnostic_with_code(diagnostic_code),)
    )

    assert [item.code for item in recommendations] == [
        recommendation_code
    ]


def test_all_recommendations_are_json_serializable():
    diagnostics = (
        _diagnostic_with_code("duplicate_epochs"),
        _diagnostic_with_code("recent_degradation"),
        _diagnostic_with_code("failed_experiments_present"),
    )
    serialized = [
        recommendation_to_dict(recommendation)
        for recommendation in build_recommendations(diagnostics)
    ]

    assert json.loads(json.dumps(serialized, allow_nan=False)) == serialized

import json
from copy import deepcopy
from dataclasses import FrozenInstanceError, fields
from types import MappingProxyType

import pytest

from diagnostics import (
    Diagnostic,
    build_metric_diagnostics,
    build_metric_facts,
    diagnostic_to_dict,
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

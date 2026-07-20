"""通用实验指标的确定性规则诊断。"""

from collections.abc import Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass
from math import isfinite
from re import fullmatch
from types import MappingProxyType
from typing import Literal

from metrics import MetricDirection, evaluate_metric_history


DiagnosticSeverity = Literal["info", "warning"]
RecentTrend = Literal[
    "improving",
    "degrading",
    "flat",
    "mixed",
    "insufficient_data",
]


@dataclass(frozen=True)
class Diagnostic:
    """描述一个固定规则产生的结构化诊断。"""

    code: str
    severity: DiagnosticSeverity
    message: str
    evidence: Mapping[str, object]

    def __post_init__(self) -> None:
        if not isinstance(self.code, str):
            raise TypeError("code must be a string")
        if fullmatch(r"[a-z][a-z0-9]*(?:_[a-z0-9]+)*", self.code) is None:
            raise ValueError(
                "code must be a non-empty snake_case string"
            )

        if not isinstance(self.severity, str):
            raise TypeError("severity must be a string")
        if self.severity not in ("info", "warning"):
            raise ValueError(
                "severity must be one of: info, warning"
            )

        if not isinstance(self.message, str):
            raise TypeError("message must be a string")
        if not self.message.strip():
            raise ValueError("message must not be empty")

        if not isinstance(self.evidence, Mapping):
            raise TypeError("evidence must be a mapping")
        if any(not isinstance(key, str) for key in self.evidence):
            raise TypeError("evidence keys must be strings")

        object.__setattr__(
            self,
            "evidence",
            MappingProxyType(dict(self.evidence)),
        )


def diagnostic_to_dict(
    diagnostic: Diagnostic,
) -> dict[str, object]:
    """将诊断转换为可安全序列化的普通字典。"""
    if not isinstance(diagnostic, Diagnostic):
        raise TypeError("diagnostic must be a Diagnostic")

    return {
        "code": diagnostic.code,
        "severity": diagnostic.severity,
        "message": diagnostic.message,
        "evidence": deepcopy(dict(diagnostic.evidence)),
    }


def _normalized_change(
    earlier_value: int | float,
    later_value: int | float,
    direction: MetricDirection,
) -> int | float:
    if direction == "maximize":
        return later_value - earlier_value
    return earlier_value - later_value


def build_metric_facts(
    records: Sequence[Sequence[object]],
    direction: MetricDirection,
    *,
    recent_window: int = 5,
) -> dict[str, object]:
    """根据原始指标历史计算确定性的诊断事实。"""
    if isinstance(recent_window, bool) or not isinstance(recent_window, int):
        raise TypeError("recent_window must be an integer")
    if recent_window < 2:
        raise ValueError("recent_window must be at least 2")

    metric_summary = evaluate_metric_history(
        records,
        direction,
    )

    for index, record in enumerate(records):
        value = record[1]
        if isinstance(value, float) and not isfinite(value):
            raise ValueError(
                f"records[{index}][1] must be finite"
            )

    record_count = metric_summary["record_count"]
    best_epoch = metric_summary["best_epoch"]
    best_value = metric_summary["best_value"]
    best_record_index = next(
        index
        for index, record in enumerate(records)
        if record[0] == best_epoch and record[1] == best_value
    )
    best_progress_ratio = (
        best_record_index / (record_count - 1)
        if record_count > 1
        else None
    )

    duplicate_epochs: list[int] = []
    seen_epochs: set[int] = set()
    recorded_duplicate_epochs: set[int] = set()
    non_monotonic_epoch_transitions: list[dict[str, int]] = []

    for index, record in enumerate(records):
        epoch = record[0]
        if epoch in seen_epochs and epoch not in recorded_duplicate_epochs:
            duplicate_epochs.append(epoch)
            recorded_duplicate_epochs.add(epoch)
        seen_epochs.add(epoch)

        if index > 0:
            previous_epoch = records[index - 1][0]
            if epoch < previous_epoch:
                non_monotonic_epoch_transitions.append(
                    {
                        "previous_record_index": index - 1,
                        "current_record_index": index,
                        "previous_epoch": previous_epoch,
                        "current_epoch": epoch,
                    }
                )

    first_value = metric_summary["first_value"]
    last_value = metric_summary["last_value"]
    improvement_from_first = _normalized_change(
        first_value,
        best_value,
        direction,
    )
    regression_from_best = -_normalized_change(
        best_value,
        last_value,
        direction,
    )

    recent_window_size = min(recent_window, record_count)
    recent_start_index = record_count - recent_window_size
    recent_values = [
        records[index][1]
        for index in range(recent_start_index, record_count)
    ]
    recent_transition_count = max(recent_window_size - 1, 0)
    recent_improving_steps = 0
    recent_degrading_steps = 0
    recent_flat_steps = 0

    for index in range(1, recent_window_size):
        change = _normalized_change(
            recent_values[index - 1],
            recent_values[index],
            direction,
        )
        if change > 0:
            recent_improving_steps += 1
        elif change < 0:
            recent_degrading_steps += 1
        else:
            recent_flat_steps += 1

    recent_net_change = (
        _normalized_change(
            recent_values[0],
            recent_values[-1],
            direction,
        )
        if recent_transition_count > 0
        else 0
    )

    recent_trend: RecentTrend
    if recent_transition_count == 0:
        recent_trend = "insufficient_data"
    elif recent_flat_steps == recent_transition_count:
        recent_trend = "flat"
    elif recent_improving_steps > recent_transition_count / 2:
        recent_trend = "improving"
    elif recent_degrading_steps > recent_transition_count / 2:
        recent_trend = "degrading"
    else:
        recent_trend = "mixed"

    return {
        "record_count": record_count,
        "first_epoch": metric_summary["first_epoch"],
        "first_value": first_value,
        "last_epoch": metric_summary["last_epoch"],
        "last_value": last_value,
        "best_epoch": best_epoch,
        "best_value": best_value,
        "best_record_index": best_record_index,
        "best_progress_ratio": best_progress_ratio,
        "best_at_first_record": best_record_index == 0,
        "best_at_last_record": best_record_index == record_count - 1,
        "duplicate_epochs": duplicate_epochs,
        "non_monotonic_epoch_transitions": (
            non_monotonic_epoch_transitions
        ),
        "improvement_from_first": improvement_from_first,
        "regression_from_best": regression_from_best,
        "recent_window_requested": recent_window,
        "recent_window_size": recent_window_size,
        "recent_transition_count": recent_transition_count,
        "recent_improving_steps": recent_improving_steps,
        "recent_degrading_steps": recent_degrading_steps,
        "recent_flat_steps": recent_flat_steps,
        "recent_net_change": recent_net_change,
        "recent_trend": recent_trend,
    }


def build_metric_diagnostics(
    facts: Mapping[str, object],
) -> tuple[Diagnostic, ...]:
    """根据指标事实生成固定规则诊断。"""
    if not isinstance(facts, Mapping):
        raise TypeError("facts must be a mapping")

    diagnostics: list[Diagnostic] = []

    if facts["duplicate_epochs"]:
        diagnostics.append(
            Diagnostic(
                code="duplicate_epochs",
                severity="warning",
                message=(
                    "Duplicate epoch values were found "
                    "in the metric history."
                ),
                evidence={
                    "duplicate_epochs": deepcopy(
                        facts["duplicate_epochs"]
                    ),
                },
            )
        )

    if facts["non_monotonic_epoch_transitions"]:
        diagnostics.append(
            Diagnostic(
                code="non_monotonic_epochs",
                severity="warning",
                message=(
                    "Epoch values decrease in one or more "
                    "adjacent records."
                ),
                evidence={
                    "transitions": deepcopy(
                        facts["non_monotonic_epoch_transitions"]
                    ),
                },
            )
        )

    if facts["record_count"] > 1:
        if facts["best_at_first_record"] is True:
            diagnostics.append(
                Diagnostic(
                    code="best_at_first_record",
                    severity="info",
                    message=(
                        "The best value occurs at the first record."
                    ),
                    evidence={
                        "best_record_index": facts["best_record_index"],
                        "best_epoch": facts["best_epoch"],
                    },
                )
            )

        if facts["best_at_last_record"] is True:
            diagnostics.append(
                Diagnostic(
                    code="best_at_last_record",
                    severity="info",
                    message=(
                        "The best value occurs at the last record."
                    ),
                    evidence={
                        "best_record_index": facts["best_record_index"],
                        "best_epoch": facts["best_epoch"],
                    },
                )
            )

        if facts["improvement_from_first"] == 0:
            diagnostics.append(
                Diagnostic(
                    code="no_improvement",
                    severity="warning",
                    message=(
                        "The metric did not improve beyond "
                        "its first recorded value."
                    ),
                    evidence={
                        "first_value": facts["first_value"],
                        "best_value": facts["best_value"],
                        "improvement_from_first": facts[
                            "improvement_from_first"
                        ],
                    },
                )
            )

    if facts["regression_from_best"] > 0:
        diagnostics.append(
            Diagnostic(
                code="post_best_regression",
                severity="warning",
                message=(
                    "The final value is worse than "
                    "the best recorded value."
                ),
                evidence={
                    "best_epoch": facts["best_epoch"],
                    "best_value": facts["best_value"],
                    "last_epoch": facts["last_epoch"],
                    "last_value": facts["last_value"],
                    "regression_from_best": facts[
                        "regression_from_best"
                    ],
                },
            )
        )

    recent_trend = facts["recent_trend"]
    if recent_trend == "insufficient_data":
        diagnostics.append(
            Diagnostic(
                code="insufficient_history_for_trend",
                severity="info",
                message=(
                    "At least two records are required "
                    "to determine a recent trend."
                ),
                evidence={
                    "record_count": facts["record_count"],
                    "recent_window_size": facts["recent_window_size"],
                },
            )
        )
    else:
        recent_trend_evidence = {
            "recent_window_size": facts["recent_window_size"],
            "recent_transition_count": facts[
                "recent_transition_count"
            ],
            "recent_improving_steps": facts[
                "recent_improving_steps"
            ],
            "recent_degrading_steps": facts[
                "recent_degrading_steps"
            ],
            "recent_flat_steps": facts["recent_flat_steps"],
            "recent_net_change": facts["recent_net_change"],
        }
        trend_diagnostics = {
            "improving": (
                "recent_improvement",
                "info",
                "The recent metric history is improving.",
            ),
            "degrading": (
                "recent_degradation",
                "warning",
                "The recent metric history is degrading.",
            ),
            "flat": (
                "recent_flat",
                "info",
                "The recent metric history is flat.",
            ),
            "mixed": (
                "recent_mixed",
                "info",
                "The recent metric history has mixed direction.",
            ),
        }
        code, severity, message = trend_diagnostics[recent_trend]
        diagnostics.append(
            Diagnostic(
                code=code,
                severity=severity,
                message=message,
                evidence=recent_trend_evidence,
            )
        )

    return tuple(diagnostics)

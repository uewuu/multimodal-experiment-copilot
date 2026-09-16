"""Deterministic evaluation of recorded, structured Copilot answers.

This module consumes observations without executing tools or managing runs.
Only the answer's JSON ``facts`` mapping is scored; free text and causal claims
outside that mapping are not evaluated. Case mappings are copied on construction
and remain caller-editable; the dataclass fields themselves are frozen.
"""

from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
from dataclasses import dataclass
import json
from math import isfinite
from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    from .runtime_observability import CopilotObservedResult


__all__ = (
    "EvaluationCase",
    "EvaluationResult",
    "EvaluationComparison",
    "evaluate",
    "compare_results",
)

_MISSING = object()


@dataclass(frozen=True, slots=True)
class EvaluationCase:
    """Versioned task and independent facts for the fixed three-check profile.

    scoring_spec contains required_tools (ordered names) and fact_paths (fact
    name to a sequence of object keys within a required tool's JSON result).
    Change scenario_version when the task, expected facts, or rules change.
    """

    case_id: str
    scenario_version: str
    prompt: str
    expected_facts: Mapping[str, object]
    scoring_spec: Mapping[str, object]

    def __post_init__(self) -> None:
        for name in ("case_id", "scenario_version", "prompt"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{name} must be a non-empty string")
        object.__setattr__(self, "expected_facts", deepcopy(dict(self.expected_facts)))
        object.__setattr__(self, "scoring_spec", deepcopy(dict(self.scoring_spec)))


@dataclass(frozen=True, slots=True)
class EvaluationResult:
    """Evaluation-owned score with an optional borrowed runtime identity."""

    case_id: str
    scenario_version: str
    passed: bool
    score: float
    failure_reasons: tuple[str, ...]
    run_id: str | None = None

    def __post_init__(self) -> None:
        if (
            type(self.score) not in (int, float)
            or not isfinite(self.score)
            or not 0 <= self.score <= 1
        ):
            raise ValueError("score must be a finite number between 0 and 1")
        object.__setattr__(self, "failure_reasons", tuple(self.failure_reasons))


@dataclass(frozen=True, slots=True)
class EvaluationComparison:
    """Direction and signed score change for a compatible pair of results."""

    change: Literal["regression", "improvement", "unchanged"]
    score_delta: float


def _json_object(payload: str) -> dict[str, object] | None:
    """Reject malformed, non-finite, or ambiguous evidence without raising."""
    def object_pairs(pairs: list[tuple[str, object]]) -> dict[str, object]:
        result: dict[str, object] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError("duplicate JSON key")
            result[key] = value
        return result

    def finite_float(value: str) -> float:
        number = float(value)
        if not isfinite(number):
            raise ValueError("non-finite JSON number")
        return number

    def invalid_constant(value: str) -> None:
        raise ValueError("non-finite JSON constant")

    try:
        value = json.loads(
            payload,
            object_pairs_hook=object_pairs,
            parse_float=finite_float,
            parse_constant=invalid_constant,
        )
    except (TypeError, ValueError, RecursionError):
        return None
    return value if isinstance(value, dict) else None


def _same_fact(actual: object, expected: object) -> bool:
    # JSON booleans must not compare equal to numeric facts (True == 1).
    if isinstance(actual, bool) or isinstance(expected, bool):
        return type(actual) is type(expected) and actual == expected
    if isinstance(actual, dict) and isinstance(expected, Mapping):
        return actual.keys() == expected.keys() and all(
            _same_fact(actual[key], expected[key]) for key in actual
        )
    if isinstance(actual, list) and isinstance(expected, list):
        return len(actual) == len(expected) and all(
            _same_fact(left, right) for left, right in zip(actual, expected)
        )
    return actual == expected


def _fact_at_path(evidence: object, path: tuple[str, ...]) -> object:
    for key in path:
        if not isinstance(evidence, dict) or key not in evidence:
            return _MISSING
        evidence = evidence[key]
    return evidence


def evaluate(
    case: EvaluationCase,
    observed_output: CopilotObservedResult,
) -> EvaluationResult:
    """Score tool selection, exact answer facts, and their recorded evidence.

    Each check contributes one third. Failure reasons use a fixed order and
    contain check names only, so they do not copy sensitive runtime payloads.
    """
    required_tools = tuple(case.scoring_spec["required_tools"])
    fact_paths = case.scoring_spec["fact_paths"]
    invocations = observed_output.turn.tool_invocations
    answer = _json_object(observed_output.turn.answer)
    facts = None if answer is None else answer.get("facts")
    evidence = [
        _json_object(invocation.result_json)
        for invocation in invocations
        if invocation.tool_name in required_tools
    ]
    checks = (
        ("tool_selection", tuple(item.tool_name for item in invocations)
         == required_tools),
        ("answer_facts", isinstance(facts, dict)
         and _same_fact(facts, case.expected_facts)),
        ("evidence_grounding", isinstance(facts, dict) and all(
            name in fact_paths and any(
                _same_fact(_fact_at_path(item, fact_paths[name]), value)
                for item in evidence
            )
            for name, value in facts.items()
        )),
    )
    failures = tuple(name for name, passed in checks if not passed)
    return EvaluationResult(
        case_id=case.case_id,
        scenario_version=case.scenario_version,
        passed=not failures,
        score=(len(checks) - len(failures)) / len(checks),
        failure_reasons=failures,
        run_id=None if observed_output.run is None else observed_output.run.run_id,
    )


def compare_results(
    previous_result: EvaluationResult,
    current_result: EvaluationResult,
) -> EvaluationComparison:
    """Compare scores only within the same case and scenario version."""
    if (
        previous_result.case_id != current_result.case_id
        or previous_result.scenario_version != current_result.scenario_version
    ):
        raise ValueError("comparison requires the same case_id and scenario_version")
    delta = current_result.score - previous_result.score
    change = "improvement" if delta > 0 else "regression" if delta < 0 else "unchanged"
    return EvaluationComparison(change=change, score_delta=delta)

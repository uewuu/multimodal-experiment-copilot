"""Sequential evaluation batches and explicit, in-memory baseline comparison.

M10 owns execution, scoring and result comparison. Suite definition policy is
owned by evaluation_suite. This module borrows the host's Session factory and
does not close resources, retry execution, persist artifacts or promote baselines.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from math import fsum
from typing import TYPE_CHECKING

from . import evaluation, evaluation_runner, evaluation_suite

if TYPE_CHECKING:
    from .session import CopilotSession


__all__ = (
    "BatchScenarioOutcome", "BatchEvaluationReport", "BatchEvaluationRunner",
    "BaselineValidationError", "BatchScenarioComparison", "ReleaseGateResult",
    "compare_batch_to_baseline",
)


@dataclass(frozen=True, slots=True)
class BatchScenarioOutcome:
    """One suite slot containing the original M10 result or a safe error code."""

    position: int
    case_id: str
    scenario_version: str
    content_fingerprint: str
    result: evaluation.EvaluationResult | None
    error_code: str | None

    def __post_init__(self) -> None:
        if self.result is None:
            if self.error_code != "runner_error":
                raise ValueError("An absent result requires runner_error")
        elif (
            not isinstance(self.result, evaluation.EvaluationResult)
            or self.error_code is not None
            or (self.result.case_id, self.result.scenario_version)
            != (self.case_id, self.scenario_version)
        ):
            raise ValueError("Outcome requires a matching M10 result without an error code")


@dataclass(frozen=True, slots=True)
class BatchEvaluationReport:
    """Complete canonical suite outcomes; counts derive from preserved results."""

    suite: evaluation_suite.EvaluationSuite
    outcomes: tuple[BatchScenarioOutcome, ...]

    def __post_init__(self) -> None:
        if type(self.suite) is not evaluation_suite.EvaluationSuite:
            raise ValueError("Expected an EvaluationSuite")
        outcomes = tuple(self.outcomes)
        if len(outcomes) != len(self.suite.scenarios):
            raise ValueError("Report must contain every suite scenario")
        for position, (outcome, scenario) in enumerate(zip(outcomes, self.suite.scenarios)):
            if (
                type(outcome) is not BatchScenarioOutcome
                or type(outcome.position) is not int
                or (outcome.position, outcome.case_id, outcome.scenario_version,
                    outcome.content_fingerprint) != (position, *scenario.key, scenario.fingerprint)
            ):
                raise ValueError("Report outcomes must match canonical suite positions and definitions")
        object.__setattr__(self, "outcomes", outcomes)

    @property
    def total(self) -> int:
        return len(self.outcomes)

    @property
    def completed(self) -> int:
        return sum(item.result is not None for item in self.outcomes)

    @property
    def passed(self) -> int:
        return sum(bool(item.result.passed) for item in self.outcomes if item.result is not None)

    @property
    def failed(self) -> int:
        return self.completed - self.passed

    @property
    def execution_failures(self) -> int:
        return sum(
            not item.result.passed and "execution_failure" in item.result.failure_reasons
            for item in self.outcomes if item.result is not None
        )

    @property
    def runner_errors(self) -> int:
        return self.total - self.completed

    @property
    def aggregate_score(self) -> float | None:
        if self.runner_errors:
            return None
        # Suite construction guarantees a nonempty definition, and the report
        # validates complete coverage. No error slot can shrink this denominator.
        return fsum(item.result.score for item in self.outcomes) / self.total


class BatchEvaluationRunner:
    """Attempt each scenario once; the host factory must supply fresh Sessions."""

    def __init__(self, session_factory: Callable[[], CopilotSession]):
        if not callable(session_factory):
            raise ValueError("session_factory must be callable")
        self._session_factory = session_factory

    def run(self, suite: evaluation_suite.EvaluationSuite) -> BatchEvaluationReport:
        if type(suite) is not evaluation_suite.EvaluationSuite:
            raise ValueError("Expected an EvaluationSuite")
        outcomes = []
        for position, scenario in enumerate(suite.scenarios):
            identity = (position, *scenario.key, scenario.fingerprint)
            try:
                # M10 alone obtains the Session and sequences context/final turns.
                # Module lookup preserves instrumentation of the owning API.
                result = evaluation_runner.run_evaluation_case(
                    scenario.case, self._session_factory,
                    context_prompts=scenario.context_prompts,
                )
                outcome = BatchScenarioOutcome(*identity, result=result, error_code=None)
            except Exception:
                # Do not retain exception text, traceback or fabricated run data.
                # Process-control BaseExceptions propagate and stop the batch.
                outcome = BatchScenarioOutcome(*identity, result=None, error_code="runner_error")
            outcomes.append(outcome)
        return BatchEvaluationReport(suite=suite, outcomes=tuple(outcomes))


class BaselineValidationError(ValueError):
    """Stable definition or result-set mismatch reasons, without baseline mutation."""

    def __init__(self, reason_codes: Sequence[str]):
        self.reason_codes = tuple(reason_codes)
        super().__init__("Invalid evaluation baseline: " + ", ".join(self.reason_codes))


@dataclass(frozen=True, slots=True)
class BatchScenarioComparison:
    case_id: str
    scenario_version: str
    comparison: evaluation.EvaluationComparison


@dataclass(frozen=True, slots=True)
class ReleaseGateResult:
    """Per-case quality decision; no model ranking or statistical inference."""

    passed: bool
    reason_codes: tuple[str, ...]
    comparisons: tuple[BatchScenarioComparison, ...]


def _baseline_results_by_identity(manifest, results):
    if not isinstance(results, Sequence) or isinstance(results, (str, bytes)):
        raise BaselineValidationError(("invalid_baseline_result",))
    by_identity = {}
    duplicate = False
    for result in results:
        if not isinstance(result, evaluation.EvaluationResult) or any(
            type(value) is not str for value in (result.case_id, result.scenario_version)
        ):
            raise BaselineValidationError(("invalid_baseline_result",))
        key = (result.case_id, result.scenario_version)
        if key in by_identity:
            duplicate = True
        else:
            by_identity[key] = result
    expected = {(item.case_id, item.scenario_version) for item in manifest.scenarios}
    actual = set(by_identity)
    reasons = []
    if duplicate:
        reasons.append("duplicate_baseline_result")
    if expected - actual:
        reasons.append("missing_baseline_result")
    if actual - expected:
        reasons.append("unexpected_baseline_result")
    if reasons:
        raise BaselineValidationError(reasons)
    return by_identity


def compare_batch_to_baseline(
    report: BatchEvaluationReport,
    manifest: evaluation_suite.BaselineManifest,
    baseline_results: Sequence[evaluation.EvaluationResult],
) -> ReleaseGateResult:
    """Validate the full baseline before comparing any original M10 result pair."""
    if type(report) is not BatchEvaluationReport:
        raise ValueError("Expected a BatchEvaluationReport")
    compatibility = evaluation_suite.check_baseline_compatibility(report.suite, manifest)
    if not compatibility.compatible:
        raise BaselineValidationError(compatibility.reason_codes)
    previous = _baseline_results_by_identity(manifest, baseline_results)

    comparisons = []
    runner_errors = []
    failed_evaluations = []
    regressions = []
    # Report validation guarantees every scenario is present in suite order.
    for outcome in report.outcomes:
        if outcome.result is None:
            runner_errors.append(f"runner_error:{outcome.case_id}")
            continue
        key = (outcome.case_id, outcome.scenario_version)
        comparison = evaluation.compare_results(previous[key], outcome.result)
        comparisons.append(BatchScenarioComparison(*key, comparison))
        if not outcome.result.passed:
            failed_evaluations.append(f"failed_evaluation:{outcome.case_id}")
        if comparison.change == "regression":
            regressions.append(f"regression:{outcome.case_id}")
    reasons = tuple(runner_errors + failed_evaluations + regressions)
    return ReleaseGateResult(passed=not reasons, reason_codes=reasons,
                             comparisons=tuple(comparisons))

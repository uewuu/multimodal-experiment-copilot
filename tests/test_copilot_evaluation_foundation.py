"""M10 Slice 1 RED contracts for the future copilot.evaluation module.

Evaluation is a leaf consumer of existing observations, with no package-root
re-export requirement. These contracts require no provider, files, or tools to
execute. Fixture facts are hand-written, not computed by the code under test.

The initial scoring profile has three equally weighted binary checks:
tool_selection (exact required tool names, in order), answer_facts (the JSON
answer's facts exactly match expected_facts), and evidence_grounding (every
answer fact matches its fact_paths location in a required tool's JSON result).
Missing evidence and extra claims fail the relevant check. A result passes only
when all checks pass; failure_reasons contains the failed check names. This
profile evaluates structured facts, not free-text explanations or causality.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import fields, replace
import importlib
import importlib.util
import json
from types import ModuleType

import pytest

from copilot import (
    CopilotObservedResult,
    CopilotRunEvent,
    CopilotRunMetadata,
    CopilotRunUsage,
    CopilotRuntimeMetrics,
    CopilotToolInvocation,
    CopilotTurn,
)


_MODULE = "copilot.evaluation"
_CASE_ID = "single-experiment-best-r2"
_VERSION = "1"
_PROMPT = 'Report best R2 as JSON: {"facts": {"best_r2": number}}.'


def _evaluation_api(*names: str) -> ModuleType:
    # Discover inside each test so missing capability is an assertion failure,
    # not a collection error. Do not mask imports broken inside a future module.
    assert importlib.util.find_spec(_MODULE) is not None, (
        f"missing M10 evaluation capability: {_MODULE} ({', '.join(names)})"
    )
    module = importlib.import_module(_MODULE)
    missing = [name for name in names if not callable(getattr(module, name, None))]
    assert not missing, f"missing M10 public evaluation API: {missing}"
    return module


def _case(api: ModuleType) -> object:
    return api.EvaluationCase(
        case_id=_CASE_ID,
        scenario_version=_VERSION,
        prompt=_PROMPT,
        expected_facts={"best_r2": 0.82},
        scoring_spec={
            "required_tools": ("analyze_experiment",),
            "fact_paths": {
                "best_r2": ("validation_metrics", "r2", "best_value"),
            },
        },
    )


def _observed(
    *,
    answer: str = '{"facts": {"best_r2": 0.82}}',
    tool_name: str = "analyze_experiment",
    evidence_value: float = 0.82,
) -> CopilotObservedResult:
    # Matches the existing analyze_experiment summary shape. Paths are inert
    # fixture labels: evaluation reads recorded evidence, never experiment files.
    invocation = CopilotToolInvocation(
        tool_call_id="call-fixture-1",
        tool_name=tool_name,
        arguments_json='{"experiment_dir": "fixture/experiment-a"}',
        result_json=json.dumps({
            "configuration": {"seed": 7},
            "validation_metrics": {
                "r2": {"best_value": evidence_value, "best_epoch": 3},
            },
        }),
    )
    run_id = "runtime-owned-fixture-run"
    return CopilotObservedResult(
        turn=CopilotTurn(_PROMPT, answer, None, (invocation,)),
        metrics=CopilotRuntimeMetrics(2, 1, 0.25),
        run=CopilotRunMetadata(
            run_id=run_id,
            usage=CopilotRunUsage(10, 2, 12, 2, 2, 2, True),
            events=(
                CopilotRunEvent(run_id, 0, "run.started"),
                CopilotRunEvent(run_id, 1, "run.completed"),
            ),
        ),
    )


def _result_values(result: object) -> tuple:
    return (
        result.case_id,
        result.scenario_version,
        result.passed,
        result.score,
        tuple(result.failure_reasons),
        result.run_id,
    )


def test_evaluation_case_represents_versioned_task_facts_and_scoring() -> None:
    api = _evaluation_api("EvaluationCase")
    case = _case(api)
    assert isinstance(case, api.EvaluationCase)
    assert case.case_id == _CASE_ID
    assert case.scenario_version == _VERSION
    assert case.prompt == _PROMPT
    assert case.expected_facts == {"best_r2": 0.82}
    assert case.scoring_spec == {
        "required_tools": ("analyze_experiment",),
        "fact_paths": {
            "best_r2": ("validation_metrics", "r2", "best_value"),
        },
    }


def test_evaluate_scores_observed_output_deterministically() -> None:
    api = _evaluation_api("EvaluationCase", "EvaluationResult", "evaluate")
    case = _case(api)
    observed = _observed()
    result = api.evaluate(case, observed)
    assert isinstance(result, api.EvaluationResult)
    assert _result_values(result) == (
        _CASE_ID, _VERSION, True, 1.0, (), observed.run.run_id,
    )
    assert _result_values(api.evaluate(case, observed)) == _result_values(result)

    # Independent negative controls prevent unconditional success, answer-only
    # matching, and ignoring the selected tool or unsupported structured claims.
    controls = (
        (_observed(answer='{"facts": {"best_r2": 0.99}}'),
         {"answer_facts", "evidence_grounding"}, 1 / 3),
        (_observed(tool_name="compare_experiments"),
         {"tool_selection", "evidence_grounding"}, 1 / 3),
        (_observed(evidence_value=0.40), {"evidence_grounding"}, 2 / 3),
        (_observed(answer='{"facts": {"best_r2": 0.82, "causal_gain": 1}}'),
         {"answer_facts", "evidence_grounding"}, 1 / 3),
        (replace(observed, turn=replace(observed.turn, tool_invocations=())),
         {"tool_selection", "evidence_grounding"}, 1 / 3),
    )
    for bad_observation, reasons, score in controls:
        failed = api.evaluate(case, bad_observation)
        assert isinstance(failed, api.EvaluationResult)
        assert failed.passed is False
        assert failed.score == pytest.approx(score)
        assert set(failed.failure_reasons) == reasons
        assert _result_values(api.evaluate(case, bad_observation)) == (
            _result_values(failed)
        )


def test_evaluation_result_exposes_identity_scores_and_failure_reasons() -> None:
    api = _evaluation_api("EvaluationResult")
    result = api.EvaluationResult(
        case_id=_CASE_ID,
        scenario_version=_VERSION,
        passed=False,
        score=2 / 3,
        failure_reasons=("evidence_grounding",),
        run_id="runtime-owned-fixture-run",
    )
    assert _result_values(result) == (
        _CASE_ID, _VERSION, False, 2 / 3,
        ("evidence_grounding",), "runtime-owned-fixture-run",
    )


def test_compare_results_detects_regression_improvement_and_unchanged() -> None:
    api = _evaluation_api("EvaluationResult", "compare_results")

    def result(score: float, run_id: str) -> object:
        reasons = {
            1.0: (),
            2 / 3: ("evidence_grounding",),
            1 / 3: ("answer_facts", "evidence_grounding"),
        }
        return api.EvaluationResult(
            case_id=_CASE_ID,
            scenario_version=_VERSION,
            passed=score == 1.0,
            score=score,
            failure_reasons=reasons[score],
            run_id=run_id,
        )

    # Higher scores are better; changes within failing results also matter.
    for old_score, new_score, change in (
        (1.0, 2 / 3, "regression"),
        (2 / 3, 1.0, "improvement"),
        (1.0, 1.0, "unchanged"),
        (2 / 3, 1 / 3, "regression"),
        (1 / 3, 2 / 3, "improvement"),
        (2 / 3, 2 / 3, "unchanged"),
    ):
        previous = result(old_score, "previous-runtime-run")
        current = result(new_score, "current-runtime-run")
        before = (_result_values(previous), _result_values(current))
        comparison = api.compare_results(previous, current)
        assert comparison.change == change
        assert comparison.score_delta == pytest.approx(new_score - old_score)
        assert (_result_values(previous), _result_values(current)) == before

    # Comparing different cases or scenario versions is not a valid baseline.
    for case_id, version in (("different-case", _VERSION), (_CASE_ID, "2")):
        incompatible = api.EvaluationResult(
            case_id=case_id, scenario_version=version, passed=True,
            score=1.0, failure_reasons=(), run_id="other-runtime-run",
        )
        with pytest.raises(ValueError):
            api.compare_results(result(1.0, "baseline-run"), incompatible)


def test_evaluation_preserves_turn_run_metadata_and_runtime_ownership() -> None:
    api = _evaluation_api("EvaluationCase", "evaluate")
    case = _case(api)
    observed = _observed()
    before = deepcopy(observed)
    original_turn, original_run = observed.turn, observed.run
    original_case = (deepcopy(case.expected_facts), deepcopy(case.scoring_spec))

    first = api.evaluate(case, observed)
    # Legacy observations without metadata remain valid. Evaluation must not
    # invent a runtime identity or reuse a previous observation's identity.
    without_run = replace(observed, run=None)
    legacy = api.evaluate(case, without_run)
    again = api.evaluate(case, observed)

    assert observed == before
    assert observed.turn is original_turn
    assert observed.run is original_run
    assert without_run.run is None
    assert first.run_id == original_run.run_id
    assert legacy.run_id is None
    assert legacy.case_id == _CASE_ID
    assert legacy.passed is True
    assert legacy.score == first.score
    assert _result_values(first) == _result_values(again)
    assert (case.expected_facts, case.scoring_spec) == original_case
    assert tuple(field.name for field in fields(CopilotTurn)) == (
        "question", "answer", "tool_call_content", "tool_invocations",
    )
    assert tuple(field.name for field in fields(CopilotRunMetadata)) == (
        "run_id", "usage", "events",
    )

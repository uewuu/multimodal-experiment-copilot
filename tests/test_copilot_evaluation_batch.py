"""M13 Slice 3A RED integration contracts; no batch implementation here.

Future copilot.evaluation_batch API:
  BatchEvaluationRunner(session_factory).run(suite) -> BatchEvaluationReport.
  compare_batch_to_baseline(report, manifest, baseline_results) -> gate result.
  BaselineValidationError(ValueError), exposing tuple reason_codes.

run is synchronous, sequential and never retries. It delegates each canonical
scenario exactly once to evaluation_runner.run_evaluation_case(case, factory,
context_prompts=...). M10 invokes the borrowed zero-argument factory once per
scenario. Owning-module public functions are looked up at call time so hosts
can instrument M10 execution/comparison and Slice 2 compatibility transparently.

Report.suite preserves the frozen definition. outcomes is an immutable tuple
of frozen dataclass values with exactly: position (zero-based), case_id,
scenario_version, content_fingerprint, result, error_code. A result is the
actual M10 EvaluationResult object, including execution_failure; error_code is
None. Ordinary delegated exceptions produce result=None, error_code="runner_error"
and retain their slot. No exception payload, invented result or run ID belongs
in that outcome. KeyboardInterrupt/SystemExit propagate and abort the batch.

Report counts: total, completed, passed, failed, execution_failures, runner_errors.
completed counts EvaluationResults (including execution_failure); failed counts
all completed results with passed=False; execution_failures is a subset of
failed. total=completed+runner_errors; completed=passed+failed. aggregate_score
is the arithmetic mean over ALL suite cases only when runner_errors==0,
otherwise None. Client shutdown and Session reset remain host-owned.

Comparison first reuses evaluation_suite.check_baseline_compatibility. An
incompatible definition raises BaselineValidationError with its exact reasons.
Baseline scores are a separate explicit sequence of frozen M10 EvaluationResult
values, never fields added to BaselineManifest. Before any M10 comparison,
validate exact (case_id, scenario_version) coverage and reject duplicates.
Result-set reason codes, in fixed order, are duplicate_baseline_result,
missing_baseline_result, unexpected_baseline_result (one per present category).
Wrong versions count as missing plus unexpected. Never compare an intersection
or silently overwrite duplicate results. No manifest/result/file auto-update.

A valid comparison calls evaluation.compare_results(previous, current) once
per available current result, in suite order, matched by exact identity rather
than input position. Its original EvaluationComparison is retained in each
gate.comparisons entry (case_id, scenario_version, comparison). M10 comparison
errors propagate. Runner-error slots have no fabricated comparison.

Gate exposes passed, reason_codes, comparisons (tuple). Passing requires valid
baseline definitions/results, no runner errors, every current result passed,
and no per-case regression. Reasons are category-ordered, then case-ID ordered:
runner_error:<id>, failed_evaluation:<id>, regression:<id>. An improved average
cannot hide a regression. No ranking, statistics, persistence or release wiring.
"""

from copy import deepcopy
from dataclasses import asdict, fields, replace
import importlib
import importlib.util
import json
from pathlib import Path
from threading import get_ident
from types import SimpleNamespace

import pytest

from copilot import CopilotSession
from copilot import evaluation, evaluation_runner, evaluation_suite
from copilot.evaluation import EvaluationCase, EvaluationResult
from copilot.evaluation_suite import BaselineManifest, EvaluationScenario, EvaluationSuite
import llm_adapters.openai_tool_adapter as adapter


def _api(*names):
    # Called after real frozen suite construction in every test. Missing M13
    # capability is a test failure, never an import-time or fixture error.
    module_name = "copilot.evaluation_batch"
    assert importlib.util.find_spec(module_name) is not None, (
        "missing M13 batch capability: copilot.evaluation_batch / BatchEvaluationRunner"
    )
    module = importlib.import_module(module_name)
    missing = [name for name in ("BatchEvaluationRunner", *names)
               if not callable(getattr(module, name, None))]
    assert not missing, f"missing M13 batch API: {missing}"
    return module


def _scenario(case_id, *, context=(), version="1", prompt=None, fixture="fixture-v1"):
    return EvaluationScenario(EvaluationCase(
        case_id=case_id, scenario_version=version,
        prompt=f"{case_id}: report best R2 as JSON facts." if prompt is None else prompt,
        expected_facts={"best_r2": 0.82},
        scoring_spec={"required_tools": ("analyze_experiment",),
                      "fact_paths": {"best_r2": ("validation_metrics", "r2", "best_value")}},
    ), context_prompts=context, fixture_version=fixture, metadata={"kind": "integration"})


def _suite(scenarios=None, **changes):
    values = dict(suite_id="copilot-batch", suite_version="1",
                  scenarios=[_scenario("case-a"), _scenario("case-b")]
                  if scenarios is None else scenarios,
                  evaluation_profile="m10-structured-three-checks-v1",
                  driver_version="fake-provider-driver-v1")
    return EvaluationSuite(**{**values, **changes})


def _response(content, tool_calls=None):
    return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(
        role="assistant", content=content, tool_calls=tool_calls,
    ))])


def _answers(value=0.82):
    call = SimpleNamespace(id="fixture-call", type="function", function=SimpleNamespace(
        name="analyze_experiment", arguments='{"experiment_dir":"fixture/experiment-a"}',
    ))
    return [_response(None, [call]), _response(json.dumps({"facts": {"best_r2": value}}))]


class _Client:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []
        self.close_calls = 0
        self.chat = SimpleNamespace(completions=self)

    def create(self, **kwargs):
        self.calls.append(deepcopy(kwargs))
        assert self.responses, "unexpected retry or extra provider call"
        response = self.responses.pop(0)
        if isinstance(response, BaseException):
            raise response
        return response

    def close(self):
        self.close_calls += 1


class _Session(CopilotSession):
    def __init__(self, client):
        super().__init__(client, model="fixture-model", max_turns=8)
        self.questions = []
        self.observations = []
        self.failures = []
        self.failure_histories = []
        self.reset_calls = 0

    def ask_with_observability(self, question, *, on_failure=None):
        self.questions.append(question)

        def record_failure(observation):
            self.failures.append(observation)
            self.failure_histories.append(self.history)
            if on_failure is not None:
                on_failure(observation)

        observed = super().ask_with_observability(question, on_failure=record_failure)
        self.observations.append(observed)
        return observed

    def reset(self):
        self.reset_calls += 1
        super().reset()


class _Factory:
    def __init__(self, plans):
        self.plans = list(plans)
        self.calls = 0
        self.sessions = []
        self.owner_thread = get_ident()

    def __call__(self):
        assert get_ident() == self.owner_thread, "batch must execute on the calling thread"
        self.calls += 1
        assert self.plans, "factory called more than once per scenario"
        plan = self.plans.pop(0)
        if isinstance(plan, BaseException):
            raise plan
        session = _Session(plan)
        self.sessions.append(session)
        return session


@pytest.fixture
def delegated(monkeypatch):
    """Observe real M10/Session execution; only provider/tool data are faked."""
    state = SimpleNamespace(calls=[], events=[], tool_calls=[])
    real_run = evaluation_runner.run_evaluation_case
    owner_thread = get_ident()

    def run(case, factory, *, context_prompts=()):
        assert get_ident() == owner_thread
        call = SimpleNamespace(case=case, factory=factory,
                               context=tuple(context_prompts), result=None)
        state.calls.append(call)
        state.events.append(("start", case.case_id))
        result = real_run(case, factory, context_prompts=context_prompts)
        call.result = result
        state.events.append(("end", case.case_id))
        return result

    def tool(name, arguments):
        assert name == "analyze_experiment"
        assert arguments == {"experiment_dir": "fixture/experiment-a"}
        state.tool_calls.append((name, deepcopy(arguments)))
        return {"validation_metrics": {"r2": {"best_value": 0.82}}}

    monkeypatch.setattr(evaluation_runner, "run_evaluation_case", run)
    monkeypatch.setattr(adapter, "invoke_tool", tool)
    return state


def _counts(report, expected):
    assert tuple(getattr(report, field) for field in (
        "total", "completed", "passed", "failed", "execution_failures", "runner_errors",
    )) == expected
    assert report.total == report.completed + report.runner_errors == len(report.outcomes)
    assert report.completed == report.passed + report.failed
    assert report.execution_failures <= report.failed


def _slots(report, suite):
    assert report.suite.to_json() == suite.to_json()
    assert isinstance(report.outcomes, tuple)
    assert len(report.outcomes) == len(suite.scenarios)
    for position, (outcome, scenario) in enumerate(zip(report.outcomes, suite.scenarios)):
        assert tuple(field.name for field in fields(outcome)) == (
            "position", "case_id", "scenario_version", "content_fingerprint", "result", "error_code",
        )
        assert (outcome.position, outcome.case_id, outcome.scenario_version,
                outcome.content_fingerprint) == (position, *scenario.key, scenario.fingerprint)
        with pytest.raises((AttributeError, TypeError)):
            outcome.position = 99


def _baseline_result(scenario, score=1.0):
    # Explicit historical values using the frozen M10 model; no score storage
    # is added to the definition-only BaselineManifest.
    reasons = {1.0: (), 2 / 3: ("evidence_grounding",),
               1 / 3: ("answer_facts", "evidence_grounding"),
               0.0: ("tool_selection", "answer_facts", "evidence_grounding")}
    return EvaluationResult(*scenario.key, passed=score == 1.0, score=score,
                            failure_reasons=reasons[score], run_id="historical-runtime-id")


def _comparison_spy(monkeypatch):
    calls = []
    original = evaluation.compare_results

    def compare(previous, current):
        result = original(previous, current)
        calls.append((previous, current, result))
        return result

    monkeypatch.setattr(evaluation, "compare_results", compare)
    return calls


def test_batch_uses_canonical_sequential_order_and_fresh_session_history(delegated):
    suite = _suite([_scenario("z-case"), _scenario("A-case", context=(
        "PRIVATE_A_FIRST", "PRIVATE_A_SECOND",
    ))])
    before = suite.to_json()
    api = _api()
    client = _Client([_response("A reply one"), _response("A reply two"),
                      *_answers(), *_answers()])
    factory = _Factory([client, client])  # Shared host client, separate Sessions.
    report = api.BatchEvaluationRunner(factory).run(suite)
    _slots(report, suite)
    assert delegated.events == [("start", "A-case"), ("end", "A-case"),
                                ("start", "z-case"), ("end", "z-case")]
    assert factory.calls == len(factory.sessions) == 2
    first, second = factory.sessions
    assert first is not second
    assert first.questions == ["PRIVATE_A_FIRST", "PRIVATE_A_SECOND", suite.scenarios[0].case.prompt]
    assert second.questions == [suite.scenarios[1].case.prompt]
    assert len(first.history) == 3 and len(second.history) == 1
    assert "PRIVATE_A" not in json.dumps(client.calls[4:])
    for call, scenario in zip(delegated.calls, suite.scenarios):
        assert asdict(call.case) == asdict(scenario.case)
        assert call.context == scenario.context_prompts
        assert call.factory is factory
    assert len(client.calls) == 6 and len(delegated.tool_calls) == 2
    assert first.reset_calls == second.reset_calls == client.close_calls == 0
    assert suite.to_json() == before
    _counts(report, (2, 2, 2, 0, 0, 0))
    assert report.aggregate_score == 1.0


def test_batch_preserves_actual_m10_results_and_counts_normal_scoring_failures(delegated):
    suite = _suite()
    api = _api()
    client = _Client([*_answers(), *_answers(0.99)])
    factory = _Factory([client, client])
    report = api.BatchEvaluationRunner(factory).run(suite)
    _slots(report, suite)
    assert len(delegated.calls) == 2
    for outcome, call, session in zip(report.outcomes, delegated.calls, factory.sessions):
        assert outcome.result is call.result
        assert isinstance(outcome.result, EvaluationResult)
        assert outcome.result == evaluation.evaluate(call.case, session.observations[-1])
        assert outcome.result.run_id == session.observations[-1].run.run_id
        assert outcome.error_code is None
    assert report.outcomes[1].result.failure_reasons == ("answer_facts", "evidence_grounding")
    _counts(report, (2, 2, 1, 1, 0, 0))
    assert report.aggregate_score == pytest.approx(2 / 3)
    assert client.close_calls == 0


def test_batch_preserves_execution_failure_and_session_transaction_semantics(delegated):
    suite = _suite([_scenario("case-a", context=("Committed context",)), _scenario("case-b")])
    api = _api()
    client = _Client([_response("Committed reply"), RuntimeError("SECRET_PROVIDER_PAYLOAD"),
                      *_answers()])
    factory = _Factory([client, client])
    report = api.BatchEvaluationRunner(factory).run(suite)
    _slots(report, suite)
    failed = report.outcomes[0]
    assert failed.result is delegated.calls[0].result
    assert failed.error_code is None
    assert failed.result.failure_reasons == ("execution_failure",)
    assert failed.result.passed is False and failed.result.score == 0.0
    first = factory.sessions[0]
    assert failed.result.run_id == first.failures[0].run.run_id
    assert first.history == first.failure_histories[0] == (first.observations[0].turn,)
    assert report.outcomes[1].result is delegated.calls[1].result
    assert report.outcomes[1].result.passed is True
    assert "SECRET_PROVIDER_PAYLOAD" not in json.dumps(asdict(failed))
    assert factory.calls == 2 and len(client.calls) == 4
    assert client.close_calls == 0
    _counts(report, (2, 2, 1, 1, 1, 0))
    assert report.aggregate_score == 0.5


def test_batch_runner_errors_keep_slots_sanitize_and_continue_without_fabricated_results(
    delegated, monkeypatch,
):
    suite = _suite([_scenario(name) for name in ("a-factory", "b-scoring", "c-success")])
    api = _api("compare_batch_to_baseline")
    real_evaluate = evaluation.evaluate

    def score(case, observed):
        if case.case_id == "b-scoring":
            raise ValueError("SECRET_SCORER_TOKEN request-body=private")
        return real_evaluate(case, observed)

    monkeypatch.setattr(evaluation, "evaluate", score)
    client = _Client([*_answers(), *_answers()])
    factory = _Factory([RuntimeError("SECRET_FACTORY_TOKEN"), client, client])
    report = api.BatchEvaluationRunner(factory).run(suite)
    _slots(report, suite)
    for outcome in report.outcomes[:2]:
        assert outcome.result is None and outcome.error_code == "runner_error"
        assert asdict(outcome) == {
            "position": outcome.position, "case_id": outcome.case_id,
            "scenario_version": "1", "content_fingerprint": outcome.content_fingerprint,
            "result": None, "error_code": "runner_error",
        }
    assert report.outcomes[2].result is delegated.calls[2].result
    assert report.outcomes[2].error_code is None
    assert factory.calls == 3 and len(factory.sessions) == 2
    assert len(delegated.calls) == 3 and len(client.calls) == 4
    assert client.close_calls == 0 and all(s.reset_calls == 0 for s in factory.sessions)
    _counts(report, (3, 1, 1, 0, 0, 2))
    assert report.aggregate_score is None
    gate = api.compare_batch_to_baseline(report, BaselineManifest.from_suite(suite),
                                         [_baseline_result(s) for s in suite.scenarios])
    assert gate.passed is False
    assert gate.reason_codes == ("runner_error:a-factory", "runner_error:b-scoring")
    assert tuple(item.case_id for item in gate.comparisons) == ("c-success",)


def test_batch_propagates_control_exceptions_and_stops_before_next_scenario(delegated, monkeypatch):
    suite = _suite()
    api = _api()
    for source in ("factory", "scoring"):
        for exception in (KeyboardInterrupt("control"), SystemExit(7)):
            client = _Client(_answers())
            later = _Client(_answers())
            factory = _Factory([exception if source == "factory" else client, later])

            def interrupt(*args):
                raise exception

            before = len(delegated.calls)
            with monkeypatch.context() as patch:
                if source == "scoring":
                    patch.setattr(evaluation, "evaluate", interrupt)
                with pytest.raises(type(exception)) as caught:
                    api.BatchEvaluationRunner(factory).run(suite)
            assert caught.value is exception
            assert factory.calls == 1 and len(factory.plans) == 1
            assert len(delegated.calls) == before + 1
            assert later.calls == []
            assert client.close_calls == later.close_calls == 0


def test_batch_runner_error_never_shrinks_aggregate_denominator(delegated):
    suite = _suite([_scenario(f"case-{i:02d}") for i in reversed(range(10))])
    api = _api()
    client = _Client(_answers() * 9)
    factory = _Factory([client] * 4 + [RuntimeError("unavailable")] + [client] * 5)
    report = api.BatchEvaluationRunner(factory).run(suite)
    _slots(report, suite)
    _counts(report, (10, 9, 9, 0, 0, 1))
    assert report.outcomes[4].case_id == "case-04"
    assert report.outcomes[4].result is None
    assert report.aggregate_score is None
    assert factory.calls == 10 and len(client.calls) == 18
    complete = _suite([s for s in suite.scenarios if s.key[0] != "case-04"])
    other_client = _Client(_answers() * 9)
    other = api.BatchEvaluationRunner(_Factory([other_client] * 9)).run(complete)
    _counts(other, (9, 9, 9, 0, 0, 0))
    assert other.aggregate_score == 1.0
    assert client.close_calls == other_client.close_calls == 0


def test_batch_comparison_reuses_m10_by_identity_and_never_updates_baseline(delegated, monkeypatch):
    suite = _suite()
    manifest = BaselineManifest.from_suite(suite, metadata={"release": "frozen"})
    baseline = [_baseline_result(suite.scenarios[1]), _baseline_result(suite.scenarios[0], 1 / 3)]
    before = (manifest.to_json(), deepcopy(baseline))
    api = _api("compare_batch_to_baseline")
    calls = _comparison_spy(monkeypatch)

    def forbid_write(*args, **kwargs):
        pytest.fail("batch execution/comparison must not read or promote artifact files")

    with monkeypatch.context() as patch:
        patch.setattr("builtins.open", forbid_write)
        patch.setattr(Path, "write_text", forbid_write)
        patch.setattr(Path, "write_bytes", forbid_write)
        client = _Client(_answers() * 2)
        report = api.BatchEvaluationRunner(_Factory([client, client])).run(suite)
        gate = api.compare_batch_to_baseline(report, manifest, baseline)
    assert gate.passed is True and gate.reason_codes == ()
    assert isinstance(gate.comparisons, tuple)
    assert len(calls) == len(gate.comparisons) == 2
    for i, entry in enumerate(gate.comparisons):
        assert (entry.case_id, entry.scenario_version) == suite.scenarios[i].key
        assert entry.comparison is calls[i][2]
        assert calls[i][0] is baseline[1 - i]
        assert calls[i][1] is report.outcomes[i].result
    assert [item.comparison.change for item in gate.comparisons] == ["improvement", "unchanged"]
    assert (manifest.to_json(), baseline) == before

    # M13 must not replace M10 comparison error semantics with a gate verdict.
    sentinel = ValueError("comparison rejected")

    def reject_comparison(*args):
        raise sentinel

    monkeypatch.setattr(evaluation, "compare_results", reject_comparison)
    with pytest.raises(ValueError) as caught:
        api.compare_batch_to_baseline(report, manifest, baseline)
    assert caught.value is sentinel


def test_batch_rejects_incompatible_definitions_before_any_score_comparison(delegated, monkeypatch):
    suite = _suite()
    api = _api("compare_batch_to_baseline", "BaselineValidationError")
    client = _Client(_answers() * 2)
    report = api.BatchEvaluationRunner(_Factory([client, client])).run(suite)
    baseline = [_baseline_result(s) for s in suite.scenarios]
    compare_calls = _comparison_spy(monkeypatch)
    real_check = evaluation_suite.check_baseline_compatibility
    checks = []

    def check(current, manifest):
        checks.append((current, manifest))
        return real_check(current, manifest)

    monkeypatch.setattr(evaluation_suite, "check_baseline_compatibility", check)
    variants = [_suite(**{field: "different"}) for field in (
        "suite_id", "suite_version", "evaluation_profile", "driver_version",
    )]
    variants += [
        _suite([_scenario("case-a")]),
        _suite([*suite.scenarios, _scenario("case-extra")]),
        _suite([_scenario("case-a", version="2"), _scenario("case-b")]),
        _suite([_scenario("case-a", prompt="Changed task."), _scenario("case-b")]),
        _suite([_scenario("case-a", fixture="fixture-v2"), _scenario("case-b")]),
    ]
    assert issubclass(api.BaselineValidationError, ValueError)
    for definition in variants:
        manifest = BaselineManifest.from_suite(definition)
        before = manifest.to_json()
        expected = real_check(suite, manifest)
        assert expected.compatible is False
        with pytest.raises(api.BaselineValidationError) as caught:
            api.compare_batch_to_baseline(report, manifest, baseline)
        assert caught.value.reason_codes == expected.reason_codes
        assert manifest.to_json() == before
    assert len(checks) == len(variants)
    assert all(current.to_json() == suite.to_json() for current, _ in checks)
    assert compare_calls == []


def test_batch_requires_complete_unique_exact_baseline_result_identities(delegated, monkeypatch):
    suite = _suite()
    manifest = BaselineManifest.from_suite(suite)
    api = _api("compare_batch_to_baseline", "BaselineValidationError")
    client = _Client(_answers() * 2)
    report = api.BatchEvaluationRunner(_Factory([client, client])).run(suite)
    a, b = [_baseline_result(s) for s in suite.scenarios]
    extra = replace(a, case_id="unexpected")
    variants = (
        ([a], ("missing_baseline_result",)),
        ([a, b, extra], ("unexpected_baseline_result",)),
        ([a, b, replace(a, score=0.0, passed=False)], ("duplicate_baseline_result",)),
        ([a, replace(b, scenario_version="2")],
         ("missing_baseline_result", "unexpected_baseline_result")),
        ([a, replace(b, case_id="CASE-B")],
         ("missing_baseline_result", "unexpected_baseline_result")),
        ([a, a, extra], ("duplicate_baseline_result", "missing_baseline_result",
                         "unexpected_baseline_result")),
    )
    calls = _comparison_spy(monkeypatch)
    for results, expected in variants:
        for ordered in (results, list(reversed(results))):
            before = deepcopy(ordered)
            with pytest.raises(api.BaselineValidationError) as caught:
                api.compare_batch_to_baseline(report, manifest, ordered)
            assert caught.value.reason_codes == expected
            assert ordered == before
    assert calls == []
    assert manifest.to_json() == BaselineManifest.from_suite(suite).to_json()


def test_batch_release_gate_rejects_regression_despite_average_improvement(delegated, monkeypatch):
    suite = _suite()
    manifest = BaselineManifest.from_suite(suite)
    api = _api("compare_batch_to_baseline")
    client = _Client([*_answers(), *_answers(0.99)])
    report = api.BatchEvaluationRunner(_Factory([client, client])).run(suite)
    baseline = [_baseline_result(suite.scenarios[0], 0.0), _baseline_result(suite.scenarios[1])]
    calls = _comparison_spy(monkeypatch)
    gate = api.compare_batch_to_baseline(report, manifest, baseline)
    assert report.aggregate_score > sum(item.score for item in baseline) / 2
    assert [item.comparison.change for item in gate.comparisons] == ["improvement", "regression"]
    assert all(entry.comparison is call[2] for entry, call in zip(gate.comparisons, calls))
    assert gate.passed is False
    assert gate.reason_codes == ("failed_evaluation:case-b", "regression:case-b")
    # No regression alone is insufficient: an unchanged failing case still
    # blocks release, so a gate cannot simply check score deltas or averages.
    unchanged = api.compare_batch_to_baseline(report, manifest,
                                              [item.result for item in report.outcomes])
    assert unchanged.passed is False
    assert unchanged.reason_codes == ("failed_evaluation:case-b",)
    assert [item.comparison.change for item in unchanged.comparisons] == ["unchanged", "unchanged"]

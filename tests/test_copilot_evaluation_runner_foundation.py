"""M10 Slice 3 RED contracts for copilot.evaluation_runner.

Public API: run_evaluation_case(case, session_factory, *, context_prompts=()).
The zero-argument factory supplies a fresh Session per invocation and retains
client ownership. Context prompts run in order before case.prompt; only the
last successful observation is scored through copilot.evaluation.evaluate.
Execution failure stops the scenario and returns score 0 with the stable reason
execution_failure and the failing Runtime observation's ID, or None if absent.

Session and Runtime execute normally. Only provider responses and tool results
are faked; no experiment files, network, or real LLM are needed.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, replace
import importlib
import importlib.util
import json
from types import SimpleNamespace

import pytest

from copilot import CopilotObservedResult, CopilotSession
from copilot import evaluation
from copilot.evaluation import EvaluationCase, EvaluationResult
import llm_adapters.openai_tool_adapter as adapter


_MODULE = "copilot.evaluation_runner"
_ANSWER = '{"facts": {"best_r2": 0.82}}'


def _runner():
    # Missing capability is an execution assertion, not a collection error.
    assert importlib.util.find_spec(_MODULE) is not None, (
        "missing M10 observed evaluation runner capability: "
        "copilot.evaluation_runner.run_evaluation_case"
    )
    module = importlib.import_module(_MODULE)
    entry = getattr(module, "run_evaluation_case", None)
    assert callable(entry), "missing M10 public run_evaluation_case entry point"
    return entry


def _case(case_id="case-a", prompt="Report best R2 as JSON facts."):
    return EvaluationCase(
        case_id=case_id,
        scenario_version="1",
        prompt=prompt,
        expected_facts={"best_r2": 0.82},
        scoring_spec={
            "required_tools": ("analyze_experiment",),
            "fact_paths": {
                "best_r2": ("validation_metrics", "r2", "best_value"),
            },
        },
    )


def _response(content, tool_calls=None):
    return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(
        role="assistant", content=content, tool_calls=tool_calls,
    ))])


def _success_outcomes(answer=_ANSWER):
    tool_call = SimpleNamespace(
        id="fixture-tool-call",
        type="function",
        function=SimpleNamespace(
            name="analyze_experiment",
            arguments='{"experiment_dir": "fixture/experiment-a"}',
        ),
    )
    return [_response(None, [tool_call]), _response(answer)]


class _FakeClient:
    def __init__(self, outcomes):
        self.outcomes = list(outcomes)
        self.calls = []
        self.close_calls = 0
        self.chat = SimpleNamespace(completions=self)

    def create(self, **kwargs):
        self.calls.append(deepcopy(kwargs))
        assert self.outcomes, "unexpected extra provider request"
        outcome = self.outcomes.pop(0)
        if isinstance(outcome, BaseException):
            raise outcome
        return outcome

    def close(self):
        self.close_calls += 1


class _RecordingSession(CopilotSession):
    """Spy on the public method while preserving actual Session execution."""

    def __init__(self, client):
        super().__init__(client, model="fixture-model", max_turns=8)
        self.questions = []
        self.observations = []
        self.observation_snapshots = []
        self.failures = []
        self.failure_histories = []
        self.callbacks = []
        self.reset_calls = 0

    def ask_with_observability(self, question, *, on_failure=None):
        self.questions.append(question)
        self.callbacks.append(on_failure)

        def record_failure(observation):
            self.failures.append(observation)
            self.failure_histories.append(self.history)
            if on_failure is not None:
                on_failure(observation)

        result = super().ask_with_observability(
            question, on_failure=record_failure,
        )
        self.observations.append(result)
        self.observation_snapshots.append(deepcopy(result))
        return result

    def reset(self):
        self.reset_calls += 1
        super().reset()


class _SessionFactory:
    def __init__(self, client):
        self.client = client
        self.sessions = []

    def __call__(self):
        session = _RecordingSession(self.client)
        self.sessions.append(session)
        return session


@pytest.fixture
def fixed_tool(monkeypatch):
    calls = []

    def invoke(name, arguments):
        assert name == "analyze_experiment"
        assert arguments == {"experiment_dir": "fixture/experiment-a"}
        calls.append((name, deepcopy(arguments)))
        return {"validation_metrics": {"r2": {"best_value": 0.82}}}

    monkeypatch.setattr(adapter, "invoke_tool", invoke)
    return calls


def test_runner_public_entry_executes_observed_case_and_returns_evaluation(fixed_tool):
    run = _runner()
    case = _case()
    before = asdict(case)
    client = _FakeClient(_success_outcomes())
    factory = _SessionFactory(client)

    result = run(case, factory)

    assert isinstance(result, EvaluationResult)
    assert len(factory.sessions) == 1
    session = factory.sessions[0]
    assert session.questions == [case.prompt]
    assert len(session.observations) == 1
    assert result == evaluation.evaluate(case, session.observations[0])
    assert result.passed is True
    assert result.score == 1.0
    assert result.failure_reasons == ()
    assert (result.case_id, result.scenario_version) == (case.case_id, "1")
    assert asdict(case) == before
    assert len(client.calls) == 2
    assert len(fixed_tool) == 1
    assert client.close_calls == 0
    assert session.reset_calls == 0


def test_runner_references_runtime_identity_without_replacing_observations(fixed_tool):
    run = _runner()
    case = _case()
    # Deliberately wrong final facts must preserve the evaluator's failing score.
    client = _FakeClient(_success_outcomes('{"facts": {"best_r2": 0.99}}'))
    factory = _SessionFactory(client)

    result = run(case, factory)

    session = factory.sessions[0]
    observed = session.observations[0]
    assert isinstance(observed, CopilotObservedResult)
    assert observed.run is not None
    assert result.run_id == observed.run.run_id
    assert result == evaluation.evaluate(case, observed)
    assert result.passed is False
    assert result.score == pytest.approx(1 / 3)
    assert observed == session.observation_snapshots[0]
    assert session.history == (observed.turn,)
    assert session.history[0] is observed.turn
    assert client.close_calls == 0


def test_runner_creates_separate_session_history_for_each_case(fixed_tool):
    run = _runner()
    case_a = _case("case-a", "CASE_A_PRIVATE_CONTEXT: report best R2.")
    case_b = _case("case-b", "CASE_B_CONTEXT: report best R2.")
    client = _FakeClient(_success_outcomes() + _success_outcomes())
    factory = _SessionFactory(client)

    result_a = run(case_a, factory)
    first_history = factory.sessions[0].history
    result_b = run(case_b, factory)

    assert result_a.passed is result_b.passed is True
    assert len(factory.sessions) == 2
    first, second = factory.sessions
    assert first is not second
    assert first.history == first_history
    assert first.history[0] is first_history[0]
    assert [turn.question for turn in first.history] == [case_a.prompt]
    assert [turn.question for turn in second.history] == [case_b.prompt]
    assert len(client.calls) == 4
    for request in client.calls[2:]:
        assert "CASE_A_PRIVATE_CONTEXT" not in json.dumps(request["messages"])
    assert [message["content"] for message in client.calls[2]["messages"]
            if message["role"] == "user"] == [case_b.prompt]
    assert first.reset_calls == second.reset_calls == 0
    assert client.close_calls == 0


def test_runner_executes_context_turns_in_order_and_scores_only_final_turn(
    fixed_tool, monkeypatch,
):
    case = _case()
    context_prompts = ("Remember CONTEXT_ONE.", "Now consider CONTEXT_TWO.")
    client = _FakeClient([
        _response("FIRST_CONTEXT_REPLY"),
        _response("SECOND_CONTEXT_REPLY"),
        *_success_outcomes(),
    ])
    factory = _SessionFactory(client)
    original_evaluate = evaluation.evaluate
    scored = []

    def record_evaluate(received_case, observed):
        assert len(factory.sessions) == 1
        session = factory.sessions[0]
        assert session.questions == [*context_prompts, case.prompt]
        assert len(session.history) == 3
        assert observed is session.observations[-1]
        assert received_case is case
        scored.append(observed)
        return original_evaluate(received_case, observed)

    monkeypatch.setattr(evaluation, "evaluate", record_evaluate)
    run = _runner()
    result = run(case, factory, context_prompts=context_prompts)

    session = factory.sessions[0]
    assert len(scored) == 1
    assert result == original_evaluate(case, session.observations[-1])
    assert result.passed is True
    assert result.run_id == session.observations[-1].run.run_id
    assert len(client.calls) == 4
    assert [message["content"] for message in client.calls[2]["messages"][1:]] == [
        context_prompts[0], "FIRST_CONTEXT_REPLY",
        context_prompts[1], "SECOND_CONTEXT_REPLY", case.prompt,
    ]
    assert all(turn is observed.turn for turn, observed in zip(
        session.history, session.observations, strict=True,
    ))
    assert session.reset_calls == client.close_calls == 0


def test_runner_reports_execution_failure_with_runtime_id_and_preserves_history(
    monkeypatch,
):
    run = _runner()
    case = _case()

    def forbid_scoring(*args, **kwargs):
        pytest.fail("an incomplete scenario must not be scored as a success")

    monkeypatch.setattr(evaluation, "evaluate", forbid_scoring)
    # Both failures occur in real Runtime execution. The second variant removes
    # only callback metadata at the consumer boundary to test legacy absence.
    original_ask = _RecordingSession.ask_with_observability
    for omit_run in (False, True):
        def ask(self, question, *, on_failure=None):
            assert callable(on_failure), "runner must connect failure observation"

            def forward(observation):
                on_failure(replace(observation, run=None) if omit_run else observation)

            return original_ask(self, question, on_failure=forward)

        monkeypatch.setattr(_RecordingSession, "ask_with_observability", ask)
        client = _FakeClient([
            _response("COMMITTED_CONTEXT_REPLY"),
            RuntimeError("SECRET_PROVIDER_FAILURE_PAYLOAD"),
        ])
        factory = _SessionFactory(client)
        result = run(
            case, factory,
            context_prompts=("Committed context.", "Failing context."),
        )

        assert isinstance(result, EvaluationResult)
        assert len(factory.sessions) == 1
        session = factory.sessions[0]
        assert session.questions == ["Committed context.", "Failing context."]
        assert len(session.failures) == 1
        failure = session.failures[0]
        assert failure.run is not None
        assert result.run_id == (None if omit_run else failure.run.run_id)
        assert (result.case_id, result.scenario_version) == (case.case_id, "1")
        assert result.passed is False
        assert result.score == 0.0
        assert result.failure_reasons == ("execution_failure",)
        assert "SECRET_PROVIDER_FAILURE_PAYLOAD" not in json.dumps(asdict(result))
        assert session.history == session.failure_histories[0]
        assert session.history == (session.observations[0].turn,)
        assert session.history[0] is session.observations[0].turn
        assert session.observations[0] == session.observation_snapshots[0]
        assert len(client.calls) == 2
        assert session.reset_calls == client.close_calls == 0

"""Component RED contracts for M9 Session observability."""

from __future__ import annotations

from copy import deepcopy
import inspect
from types import SimpleNamespace
from typing import get_type_hints

import pytest

from copilot import (
    CopilotFailureObservation,
    CopilotObservedResult,
    CopilotRunMetadata,
    CopilotSession,
    CopilotTurn,
)


class _SequentialCompletions:
    def __init__(self, outcomes: list[object]) -> None:
        self.outcomes = list(outcomes)
        self.calls: list[dict[str, object]] = []

    def create(self, **kwargs: object) -> object:
        self.calls.append(deepcopy(kwargs))
        if not self.outcomes:
            raise AssertionError("unexpected extra provider request")
        outcome = self.outcomes.pop(0)
        if isinstance(outcome, BaseException):
            raise outcome
        return outcome


class _FakeClient:
    def __init__(self, outcomes: list[object]) -> None:
        self.completions = _SequentialCompletions(outcomes)
        self.chat = SimpleNamespace(completions=self.completions)
        self.close_calls = 0

    def close(self) -> None:
        self.close_calls += 1


def _response(content: str) -> SimpleNamespace:
    return SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(
                    role="assistant",
                    content=content,
                    tool_calls=None,
                )
            )
        ]
    )


def _run_metadata(
    value: CopilotObservedResult | CopilotFailureObservation,
) -> CopilotRunMetadata:
    # Runtime suites own UUID, usage, and event internals. Session consumers
    # only require populated metadata with an opaque, non-empty run ID.
    run = value.run
    assert isinstance(run, CopilotRunMetadata)
    assert isinstance(run.run_id, str)
    assert run.run_id
    return run


def test_session_observability_method_has_minimal_public_contract() -> None:
    method = getattr(CopilotSession, "ask_with_observability", None)
    assert method is not None, "Slice 1D requires Session observability"
    parameters = list(inspect.signature(method).parameters.values())
    assert [parameter.name for parameter in parameters] == [
        "self",
        "question",
        "on_failure",
    ]
    assert parameters[2].kind is inspect.Parameter.KEYWORD_ONLY
    assert parameters[2].default is None
    assert get_type_hints(method)["return"] is CopilotObservedResult


def test_observed_session_success_preserves_runtime_result_and_history() -> None:
    client = _FakeClient(
        [
            _response("seed answer"),
            _response("first answer"),
            _response("second answer"),
        ]
    )
    session = CopilotSession(client, model="test-model", max_turns=2)
    seed = session.ask_with_result("seed question")

    first = session.ask_with_observability("first question")
    assert isinstance(first, CopilotObservedResult)
    first_run = _run_metadata(first)
    assert first.turn.question == "first question"
    assert first.turn.answer == "first answer"
    assert session.history == (seed, first.turn)
    assert session.history[0] is seed
    assert session.history[1] is first.turn
    assert session.turn_count == 2
    assert all(type(turn) is CopilotTurn for turn in session.history)
    assert client.close_calls == 0

    second = session.ask_with_observability("second question")
    assert isinstance(second, CopilotObservedResult)
    second_run = _run_metadata(second)
    assert first.run is first_run
    assert second_run is not first_run
    assert first_run.run_id != second_run.run_id
    assert second.turn.question == "second question"
    assert second.turn.answer == "second answer"

    # Both observed turns are committed once; the seed is evicted at the bound.
    assert session.history == (first.turn, second.turn)
    assert session.history[0] is first.turn
    assert session.history[1] is second.turn
    assert session.turn_count == session.max_turns == 2
    assert all(type(turn) is CopilotTurn for turn in session.history)
    assert all(not hasattr(turn, "run") for turn in session.history)

    assert len(client.completions.calls) == 3
    messages = client.completions.calls[2]["messages"]
    assert [message["role"] for message in messages] == [
        "system",
        "user",
        "assistant",
        "user",
    ]
    assert [message["content"] for message in messages[1:]] == [
        "first question",
        "first answer",
        "second question",
    ]
    assert session.export_history() == [
        {
            "question": "first question",
            "answer": "first answer",
            "tool_call_content": None,
            "tool_invocations": [],
        },
        {
            "question": "second question",
            "answer": "second answer",
            "tool_call_content": None,
            "tool_invocations": [],
        },
    ]
    assert client.close_calls == 0


def test_observed_session_failure_is_transactional_and_preserves_ownership() -> None:
    failure = RuntimeError("failed observed Session provider request")
    client = _FakeClient([_response("committed answer"), failure])
    session = CopilotSession(client, model="test-model", max_turns=1)
    committed = session.ask_with_result("committed question")
    before = session.history
    exported_before = session.export_history()
    assert before == (committed,)
    assert session.turn_count == 1
    assert client.close_calls == 0

    observations: list[CopilotFailureObservation] = []
    callback_history: list[tuple[CopilotTurn, ...]] = []

    def on_failure(observation: CopilotFailureObservation) -> None:
        observations.append(observation)
        callback_history.append(session.history)

    with pytest.raises(RuntimeError) as caught:
        session.ask_with_observability(
            "failed question",
            on_failure=on_failure,
        )

    assert caught.value is failure
    assert len(observations) == 1
    assert isinstance(observations[0], CopilotFailureObservation)
    _run_metadata(observations[0])
    assert callback_history == [before]
    assert callback_history[0][0] is committed
    assert session.history == before == (committed,)
    assert session.history[0] is committed
    assert type(session.history[0]) is CopilotTurn
    assert session.turn_count == session.max_turns == 1
    assert session.export_history() == exported_before
    assert len(client.completions.calls) == 2
    assert client.close_calls == 0

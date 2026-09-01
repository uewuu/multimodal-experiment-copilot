"""RED contract tests for the bounded Copilot turn deadline."""

from __future__ import annotations

from copy import deepcopy
import inspect
import math
from types import SimpleNamespace
import sys
import time
from typing import Callable

import pytest

import copilot
import copilot.failure_observability as failure_observability
from copilot.session import CopilotSession
import llm_adapters.openai_tool_adapter as adapter


_PUBLIC_TURN_APIS = (
    "run_copilot_turn",
    "run_copilot_turn_with_result",
    "run_copilot_turn_with_observability",
    "run_copilot_turn_with_failure_observability",
)


class _FakeClock:
    def __init__(self, now: float = 0.0) -> None:
        self.now = now

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


class _SequenceClock:
    def __init__(self, values: list[float]) -> None:
        self._values = iter(values)
        self._last = values[-1]

    def __call__(self) -> float:
        try:
            self._last = next(self._values)
        except StopIteration:
            pass
        return self._last


class _FakeCompletions:
    def __init__(
        self,
        steps: list[object],
        on_request: Callable[[int], None] | None = None,
    ) -> None:
        self._steps = iter(steps)
        self._on_request = on_request
        self.calls: list[dict[str, object]] = []

    def create(self, **kwargs: object) -> object:
        self.calls.append(deepcopy(kwargs))
        index = len(self.calls) - 1
        if self._on_request is not None:
            self._on_request(index)
        step = next(self._steps)
        if isinstance(step, BaseException):
            raise step
        return step


class _FakeClient:
    def __init__(
        self,
        steps: list[object],
        on_request: Callable[[int], None] | None = None,
    ) -> None:
        self.chat = SimpleNamespace(
            completions=_FakeCompletions(steps, on_request)
        )
        self.close_count = 0

    def close(self) -> None:
        self.close_count += 1


class _ProviderTimeout(RuntimeError):
    pass


def _response(
    content: str | None = "completed",
    tool_calls: list[object] | None = None,
) -> object:
    return SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(
                    role="assistant",
                    content=content,
                    tool_calls=tool_calls,
                )
            )
        ]
    )


def _tool_call(
    call_id: str,
    name: str = "analyze_experiment",
    arguments: str = '{"experiment_dir":"demo"}',
) -> object:
    return SimpleNamespace(
        id=call_id,
        type="function",
        function=SimpleNamespace(name=name, arguments=arguments),
    )


def _install_clock(
    monkeypatch: pytest.MonkeyPatch,
    clock: Callable[[], float],
) -> None:
    """Patch monotonic seams without requiring one private helper name."""
    monkeypatch.setattr(time, "perf_counter", clock)
    monkeypatch.setattr(time, "monotonic", clock)
    for module_name, module in tuple(sys.modules.items()):
        if module is None or not module_name.startswith(
            ("copilot", "llm_adapters")
        ):
            continue
        for attribute in (
            "_perf_counter",
            "perf_counter",
            "_monotonic",
            "monotonic",
            "_clock",
        ):
            if hasattr(module, attribute):
                monkeypatch.setattr(module, attribute, clock)


def _run_failure_observed(
    client: object,
    observations: list[object],
    **options: object,
) -> object:
    return copilot.run_copilot_turn_with_failure_observability(
        client,
        model="test-model",
        question="Analyze the experiment.",
        on_failure=observations.append,
        **options,
    )


@pytest.mark.parametrize("name", _PUBLIC_TURN_APIS)
def test_public_turn_api_declares_explicit_deadline_parameter(
    name: str,
) -> None:
    parameters = inspect.signature(getattr(copilot, name)).parameters
    timeout = parameters["turn_timeout_seconds"]
    assert timeout.kind is inspect.Parameter.KEYWORD_ONLY
    assert timeout.default is None
    assert timeout.annotation == float | None


def test_session_declares_explicit_deadline_parameter() -> None:
    parameters = inspect.signature(CopilotSession).parameters
    timeout = parameters["turn_timeout_seconds"]
    assert timeout.kind is inspect.Parameter.KEYWORD_ONLY
    assert timeout.default is None
    assert timeout.annotation == float | None


@pytest.mark.parametrize("invalid", [True, "1", object()])
def test_invalid_deadline_types_fail_before_provider(
    invalid: object,
) -> None:
    client = _FakeClient([_response()])
    with pytest.raises(TypeError):
        copilot.run_copilot_turn(
            client,
            model="test-model",
            question="Analyze.",
            turn_timeout_seconds=invalid,
        )
    assert client.chat.completions.calls == []


@pytest.mark.parametrize(
    "invalid",
    [0, -1, -0.25, math.nan, math.inf, -math.inf],
)
def test_invalid_deadline_values_fail_before_provider(
    invalid: int | float,
) -> None:
    client = _FakeClient([_response()])
    with pytest.raises(ValueError):
        copilot.run_copilot_turn(
            client,
            model="test-model",
            question="Analyze.",
            turn_timeout_seconds=invalid,
        )
    assert client.chat.completions.calls == []


@pytest.mark.parametrize("invalid", [False, "2", object()])
def test_session_validates_deadline_at_construction(
    invalid: object,
) -> None:
    client = _FakeClient([_response()])
    with pytest.raises(TypeError):
        CopilotSession(
            client,
            model="test-model",
            turn_timeout_seconds=invalid,
        )
    assert client.chat.completions.calls == []


def test_failure_observability_reports_deadline_validation() -> None:
    client = _FakeClient([_response()])
    observations: list[object] = []
    with pytest.raises(ValueError):
        _run_failure_observed(
            client,
            observations,
            turn_timeout_seconds=0,
        )
    assert len(observations) == 1
    observation = observations[0]
    assert observation.stage == "input_validation"
    assert observation.provider_request_count == 0
    assert observation.tool_invocation_count == 0
    assert client.chat.completions.calls == []


def test_default_none_does_not_change_no_tool_request() -> None:
    client = _FakeClient([_response("done")])
    options = {"temperature": 0, "tool_choice": "auto"}
    original = deepcopy(options)

    result = copilot.run_copilot_turn(
        client,
        model="test-model",
        question="Analyze.",
        turn_timeout_seconds=None,
        **options,
    )

    assert result == "done"
    assert options == original
    assert len(client.chat.completions.calls) == 1
    call = client.chat.completions.calls[0]
    assert call["temperature"] == 0
    assert call["tool_choice"] == "auto"
    assert "turn_timeout_seconds" not in call
    assert "timeout" not in call
    assert client.close_count == 0


def test_default_none_preserves_explicit_provider_timeout(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = _FakeClient(
        [
            _response(None, [_tool_call("call-1")]),
            _response("done"),
        ]
    )
    monkeypatch.setattr(
        adapter,
        "invoke_tool",
        lambda *args: {"ok": True},
    )

    assert copilot.run_copilot_turn(
        client,
        model="test-model",
        question="Analyze.",
        turn_timeout_seconds=None,
        timeout={"read": 3},
    ) == "done"

    assert len(client.chat.completions.calls) == 2
    for call in client.chat.completions.calls:
        assert call["timeout"] == {"read": 3}
        assert "turn_timeout_seconds" not in call


def test_turn_deadline_conflicts_with_provider_request_timeout() -> None:
    client = _FakeClient([_response()])
    with pytest.raises(TypeError, match="timeout"):
        copilot.run_copilot_turn(
            client,
            model="test-model",
            question="Analyze.",
            turn_timeout_seconds=5,
            timeout=2,
        )
    assert client.chat.completions.calls == []


@pytest.mark.parametrize("timeout", [2, 2.5])
def test_first_provider_receives_positive_float_remaining_budget(
    monkeypatch: pytest.MonkeyPatch,
    timeout: int | float,
) -> None:
    clock = _FakeClock()
    _install_clock(monkeypatch, clock)
    client = _FakeClient([_response()])

    copilot.run_copilot_turn(
        client,
        model="test-model",
        question="Analyze.",
        turn_timeout_seconds=timeout,
    )

    passed = client.chat.completions.calls[0]["timeout"]
    assert type(passed) is float
    assert 0 < passed <= float(timeout)
    assert "turn_timeout_seconds" not in client.chat.completions.calls[0]


def test_second_provider_receives_recomputed_remaining_budget(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    clock = _FakeClock()
    _install_clock(monkeypatch, clock)
    client = _FakeClient(
        [
            _response(None, [_tool_call("call-1")]),
            _response("done"),
        ],
        on_request=lambda index: clock.advance(2) if index == 0 else None,
    )

    def invoke(*args: object) -> dict[str, bool]:
        clock.advance(2)
        return {"ok": True}

    monkeypatch.setattr(adapter, "invoke_tool", invoke)

    assert copilot.run_copilot_turn(
        client,
        model="test-model",
        question="Analyze.",
        turn_timeout_seconds=10,
    ) == "done"

    assert [call["timeout"] for call in client.chat.completions.calls] == [
        10.0,
        6.0,
    ]


def test_expired_deadline_prevents_first_provider(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_clock(monkeypatch, _SequenceClock([0.0, 2.0]))
    client = _FakeClient([_response()])
    with pytest.raises(TimeoutError):
        copilot.run_copilot_turn(
            client,
            model="test-model",
            question="Analyze.",
            turn_timeout_seconds=1,
        )
    assert client.chat.completions.calls == []


def test_deadline_expired_after_provider_stops_progression(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    clock = _FakeClock()
    _install_clock(monkeypatch, clock)
    started: list[str] = []
    client = _FakeClient(
        [
            _response(None, [_tool_call("call-1")]),
            _response("forbidden follow-up"),
        ],
        on_request=lambda index: clock.advance(2),
    )
    monkeypatch.setattr(
        adapter,
        "invoke_tool",
        lambda name, arguments: started.append(name),
    )

    with pytest.raises(TimeoutError):
        copilot.run_copilot_turn(
            client,
            model="test-model",
            question="Analyze.",
            turn_timeout_seconds=1,
        )

    assert len(client.chat.completions.calls) == 1
    assert started == []


def test_tool_that_crosses_deadline_returns_before_timeout_is_raised(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    clock = _FakeClock()
    _install_clock(monkeypatch, clock)
    events: list[str] = []
    client = _FakeClient(
        [
            _response(None, [_tool_call("call-1")]),
            _response("forbidden follow-up"),
        ]
    )

    def invoke(*args: object) -> dict[str, bool]:
        events.append("tool-started")
        clock.advance(2)
        events.append("tool-returned")
        return {"ok": True}

    monkeypatch.setattr(adapter, "invoke_tool", invoke)

    with pytest.raises(TimeoutError):
        copilot.run_copilot_turn(
            client,
            model="test-model",
            question="Analyze.",
            turn_timeout_seconds=1,
        )

    assert events == ["tool-started", "tool-returned"]
    assert len(client.chat.completions.calls) == 1


def test_deadline_after_one_tool_prevents_later_tools(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    clock = _FakeClock()
    _install_clock(monkeypatch, clock)
    started: list[str] = []
    client = _FakeClient(
        [
            _response(
                None,
                [_tool_call("call-1"), _tool_call("call-2")],
            ),
            _response("forbidden follow-up"),
        ]
    )

    def invoke(name: str, arguments: dict) -> dict[str, bool]:
        started.append(name)
        clock.advance(2)
        return {"ok": True}

    monkeypatch.setattr(adapter, "invoke_tool", invoke)

    with pytest.raises(TimeoutError):
        copilot.run_copilot_turn(
            client,
            model="test-model",
            question="Analyze.",
            turn_timeout_seconds=1,
        )

    assert started == ["analyze_experiment"]
    assert len(client.chat.completions.calls) == 1


def test_first_provider_timeout_identity_and_observation() -> None:
    error = _ProviderTimeout("first request timed out")
    client = _FakeClient([error])
    observations: list[object] = []

    with pytest.raises(_ProviderTimeout) as caught:
        _run_failure_observed(
            client,
            observations,
            turn_timeout_seconds=5,
        )

    assert caught.value is error
    assert len(client.chat.completions.calls) == 1
    assert len(observations) == 1
    assert observations[0].stage == "first_provider_request"
    assert observations[0].provider_request_count == 1
    assert observations[0].tool_invocation_count == 0
    assert client.close_count == 0


def test_second_provider_timeout_identity_and_observation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    error = _ProviderTimeout("second request timed out")
    client = _FakeClient(
        [_response(None, [_tool_call("call-1")]), error]
    )
    monkeypatch.setattr(
        adapter,
        "invoke_tool",
        lambda *args: {"ok": True},
    )
    observations: list[object] = []

    with pytest.raises(_ProviderTimeout) as caught:
        _run_failure_observed(
            client,
            observations,
            turn_timeout_seconds=5,
        )

    assert caught.value is error
    assert len(client.chat.completions.calls) == 2
    assert len(observations) == 1
    assert observations[0].stage == "second_provider_request"
    assert observations[0].provider_request_count == 2
    assert observations[0].tool_invocation_count == 1


@pytest.mark.parametrize(
    "callback_error",
    [RuntimeError("callback"), KeyboardInterrupt(), SystemExit(4), GeneratorExit()],
)
def test_failure_callback_cannot_replace_provider_timeout(
    callback_error: BaseException,
) -> None:
    original = _ProviderTimeout("provider")
    client = _FakeClient([original])

    def callback(observation: object) -> None:
        raise callback_error

    with pytest.raises(_ProviderTimeout) as caught:
        copilot.run_copilot_turn_with_failure_observability(
            client,
            model="test-model",
            question="Analyze.",
            turn_timeout_seconds=5,
            on_failure=callback,
        )
    assert caught.value is original


@pytest.mark.parametrize(
    "control_flow",
    [KeyboardInterrupt(), SystemExit(7), GeneratorExit()],
)
def test_control_flow_exceptions_are_not_converted_to_timeout(
    control_flow: BaseException,
) -> None:
    client = _FakeClient([control_flow])
    with pytest.raises(type(control_flow)) as caught:
        copilot.run_copilot_turn(
            client,
            model="test-model",
            question="Analyze.",
            turn_timeout_seconds=5,
        )
    assert caught.value is control_flow
    assert len(client.chat.completions.calls) == 1


def test_timeout_turn_is_transactional_and_next_turn_works() -> None:
    timeout = _ProviderTimeout("provider")
    client = _FakeClient([_response("first"), timeout, _response("third")])
    session = CopilotSession(
        client,
        model="test-model",
        max_turns=1,
        turn_timeout_seconds=5,
    )

    first = session.ask_with_result("first question")
    with pytest.raises(_ProviderTimeout) as caught:
        session.ask("timed out question")
    assert caught.value is timeout
    assert session.history == (first,)
    assert session.ask("third question") == "third"
    assert session.turn_count == 1
    assert session.history[0].question == "third question"
    assert client.close_count == 0


def test_path_policy_is_restored_after_project_timeout(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: object,
) -> None:
    clock = _FakeClock()
    _install_clock(monkeypatch, clock)
    client = _FakeClient(
        [_response("late")],
        on_request=lambda index: clock.advance(2),
    )

    with pytest.raises(TimeoutError):
        copilot.run_copilot_turn(
            client,
            model="test-model",
            question="Analyze.",
            experiment_context={"experiment_dir": str(tmp_path)},
            turn_timeout_seconds=1,
        )

    assert adapter._TOOL_PATH_POLICY.get() is None
    follow_up = _FakeClient([_response("fresh")])
    assert copilot.run_copilot_turn(
        follow_up,
        model="test-model",
        question="Continue.",
    ) == "fresh"


def test_failure_stage_vocabulary_remains_exactly_eight() -> None:
    progress = failure_observability._ProgressState()
    stages = {progress.stage}
    for event in (
        "provider_request_started",
        "provider_response_received",
        "tool_call_validation",
        "tool_execution",
        "tool_result_serialization",
        "provider_request_started",
        "provider_response_received",
    ):
        progress.update(event)
        stages.add(progress.stage)

    assert stages == {
        "input_validation",
        "first_provider_request",
        "first_provider_response_validation",
        "tool_call_validation",
        "tool_execution",
        "tool_result_serialization",
        "second_provider_request",
        "final_response_validation",
    }


def test_deadline_does_not_add_retry_or_close_borrowed_client() -> None:
    error = _ProviderTimeout("provider")
    client = _FakeClient([error, _response("forbidden retry")])
    with pytest.raises(_ProviderTimeout) as caught:
        copilot.run_copilot_turn(
            client,
            model="test-model",
            question="Analyze.",
            turn_timeout_seconds=5,
        )
    assert caught.value is error
    assert len(client.chat.completions.calls) == 1
    assert client.close_count == 0

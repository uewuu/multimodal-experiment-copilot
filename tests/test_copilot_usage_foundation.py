"""Component contracts for the M9 metadata and usage foundation."""

from __future__ import annotations

from dataclasses import FrozenInstanceError, fields, is_dataclass
import importlib
from types import SimpleNamespace

import pytest

import copilot
import llm_adapters.openai_tool_adapter as adapter
from copilot import CopilotTurn


EVENT_KINDS = (
    "run.started",
    "provider.request.started",
    "provider.response.received",
    "tool.calls.validation.started",
    "tool.execution.started",
    "tool.result.accepted",
    "run.completed",
    "run.failed",
)
MODEL_FIELDS = {
    "CopilotProviderUsage": (
        "input_tokens",
        "output_tokens",
        "total_tokens",
    ),
    "CopilotRunUsage": (
        "input_tokens",
        "output_tokens",
        "total_tokens",
        "provider_request_count",
        "provider_response_count",
        "usage_report_count",
        "complete",
    ),
    "CopilotRunEvent": (
        "run_id",
        "sequence",
        "kind",
        "provider_request_index",
        "tool_invocation_index",
        "tool_call_id",
        "tool_name",
        "usage",
        "failure_stage",
    ),
    "CopilotRunMetadata": (
        "run_id",
        "usage",
        "events",
    ),
}
_MISSING = object()


class _SequentialCompletions:
    def __init__(self, outcomes: list[object]) -> None:
        self.outcomes = list(outcomes)
        self.calls: list[dict[str, object]] = []

    def create(self, **kwargs: object) -> object:
        self.calls.append(kwargs)
        outcome = self.outcomes[len(self.calls) - 1]
        if isinstance(outcome, BaseException):
            raise outcome
        return outcome


class _FakeClient:
    def __init__(self, outcomes: list[object]) -> None:
        self.completions = _SequentialCompletions(outcomes)
        self.chat = SimpleNamespace(completions=self.completions)


class _ExplodingUsage:
    @property
    def prompt_tokens(self) -> object:
        raise RuntimeError("malformed usage must be ignored")

    @property
    def completion_tokens(self) -> object:
        raise RuntimeError("malformed usage must be ignored")

    @property
    def total_tokens(self) -> object:
        raise RuntimeError("malformed usage must be ignored")


def _response(usage: object = _MISSING, tool_calls: object = None) -> object:
    response = SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(
                    role="assistant",
                    content="answer",
                    tool_calls=tool_calls,
                )
            )
        ]
    )
    if usage is not _MISSING:
        response.usage = usage
    return response


def _tool_call() -> object:
    return SimpleNamespace(
        id="call_0",
        type="function",
        function=SimpleNamespace(name="tool_0", arguments="{}"),
    )


def _trace(*responses: object) -> object:
    return adapter._run_tool_call_cycle_with_trace(
        _FakeClient(list(responses)),
        model="test-model",
        messages=[{"role": "user", "content": "question"}],
    )


def _metadata_module() -> object:
    return importlib.import_module("copilot.run_metadata")


def _models() -> tuple[type, type, type, type]:
    names = tuple(MODEL_FIELDS)
    missing = [name for name in names if not hasattr(copilot, name)]
    assert not missing, f"missing M9 public metadata models: {missing}"
    return tuple(getattr(copilot, name) for name in names)  # type: ignore[return-value]


def test_public_metadata_models_are_frozen_slotted_and_exported() -> None:
    for name, expected_fields in MODEL_FIELDS.items():
        assert hasattr(copilot, name), f"M9 requires public {name}"
        model = getattr(copilot, name)
        assert name in copilot.__all__
        assert is_dataclass(model)
        assert model.__dataclass_params__.frozen is True
        assert tuple(item.name for item in fields(model)) == expected_fields
        assert tuple(model.__slots__) == expected_fields


def test_metadata_defaults_are_conservative_and_events_are_stored_as_tuple() -> None:
    ProviderUsage, RunUsage, RunEvent, RunMetadata = _models()
    provider_usage = ProviderUsage()
    assert (
        provider_usage.input_tokens,
        provider_usage.output_tokens,
        provider_usage.total_tokens,
    ) == (None, None, None)
    run_usage = RunUsage()
    assert run_usage == RunUsage(None, None, None, 0, 0, 0, False)
    event = RunEvent("run-id", 0, "run.started")
    assert event.provider_request_index is None
    assert event.tool_invocation_index is None
    assert event.tool_call_id is None
    assert event.tool_name is None
    assert event.usage is None
    assert event.failure_stage is None
    metadata = RunMetadata("run-id", run_usage, [event])
    assert metadata.events == (event,)
    assert type(metadata.events) is tuple
    with pytest.raises(FrozenInstanceError):
        metadata.run_id = "changed"


def test_event_kind_vocabulary_is_closed_without_runtime_emission() -> None:
    _, _, RunEvent, _ = _models()
    for sequence, kind in enumerate(EVENT_KINDS):
        assert RunEvent("run-id", sequence, kind).kind == kind
    with pytest.raises(ValueError, match="event kind"):
        RunEvent("run-id", 0, "provider.retry.started")


def test_copilot_turn_remains_free_of_execution_metadata() -> None:
    assert tuple(item.name for item in fields(CopilotTurn)) == (
        "question",
        "answer",
        "tool_call_content",
        "tool_invocations",
    )


@pytest.mark.parametrize(
    ("usage", "expected"),
    [
        (SimpleNamespace(prompt_tokens=0, completion_tokens=1, total_tokens=1),
         (0, 1, 1)),
        (None, (None, None, None)),
        (SimpleNamespace(), (None, None, None)),
        (SimpleNamespace(prompt_tokens="1", completion_tokens=1.0,
                         total_tokens=-1), (None, None, None)),
        (SimpleNamespace(prompt_tokens=True, completion_tokens=False,
                         total_tokens=True), (None, None, None)),
        (_ExplodingUsage(), (None, None, None)),
    ],
    ids=("valid", "none", "missing", "invalid", "bool", "malformed"),
)
def test_adapter_normalizes_usage_nonfatally(
    usage: object,
    expected: tuple[int | None, int | None, int | None],
) -> None:
    trace = _trace(_response(usage))
    assert trace.provider_request_count == 1
    assert trace.provider_response_count == 1
    assert len(trace.provider_usages) == 1
    normalized = trace.provider_usages[0]
    assert (
        normalized.input_tokens,
        normalized.output_tokens,
        normalized.total_tokens,
    ) == expected


def test_adapter_records_each_started_request_and_received_response(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(adapter, "invoke_tool", lambda name, arguments: {})
    trace = _trace(
        _response(
            SimpleNamespace(
                prompt_tokens=10,
                completion_tokens=2,
                total_tokens=12,
            ),
            [_tool_call()],
        ),
        _response(
            SimpleNamespace(
                prompt_tokens=5,
                completion_tokens=None,
                total_tokens=None,
            )
        ),
    )
    assert trace.provider_request_count == 2
    assert trace.provider_response_count == 2
    assert [usage.input_tokens for usage in trace.provider_usages] == [10, 5]


@pytest.mark.parametrize(
    ("usages", "expected_report_count"),
    [
        (((None, None, None),), 0),
        (((10, None, None),), 1),
        (((10, 2, 12),), 1),
        (((10, 2, 12), (5, None, None)), 2),
    ],
    ids=("all-invalid", "one-valid-field", "all-valid", "per-response"),
)
def test_usage_report_count_counts_responses_not_fields(
    usages: tuple[tuple[int | None, int | None, int | None], ...],
    expected_report_count: int,
) -> None:
    ProviderUsage, _, _, _ = _models()
    normalized = tuple(ProviderUsage(*usage) for usage in usages)
    result = _metadata_module()._aggregate_run_usage(
        normalized,
        provider_request_count=len(normalized),
        provider_response_count=len(normalized),
        terminal_success=True,
    )
    assert result.usage_report_count == expected_report_count


def test_usage_aggregation_requires_field_level_complete_coverage() -> None:
    ProviderUsage, _, _, _ = _models()
    result = _metadata_module()._aggregate_run_usage(
        (
            ProviderUsage(100, 20, 120),
            ProviderUsage(50, None, None),
        ),
        provider_request_count=2,
        provider_response_count=2,
        terminal_success=True,
    )
    assert result.input_tokens == 150
    assert result.output_tokens is None
    assert result.total_tokens is None
    assert result.usage_report_count == 2
    assert result.complete is False


@pytest.mark.parametrize(
    ("request_count", "response_count", "terminal_success", "expected"),
    [
        (2, 2, True, True),
        (2, 2, False, False),
        (2, 1, True, False),
    ],
    ids=("complete", "failed-run", "missing-response"),
)
def test_complete_requires_success_and_full_usage_coverage(
    request_count: int,
    response_count: int,
    terminal_success: bool,
    expected: bool,
) -> None:
    ProviderUsage, _, _, _ = _models()
    usages = (ProviderUsage(10, 2, 12),) * response_count
    result = _metadata_module()._aggregate_run_usage(
        usages,
        provider_request_count=request_count,
        provider_response_count=response_count,
        terminal_success=terminal_success,
    )
    assert result.complete is expected
    if response_count != request_count:
        assert (
            result.input_tokens,
            result.output_tokens,
            result.total_tokens,
        ) == (None, None, None)

"""Immutable metadata values for one Copilot run."""

from __future__ import annotations

from dataclasses import dataclass, field


__all__ = (
    "CopilotProviderUsage",
    "CopilotRunEvent",
    "CopilotRunMetadata",
    "CopilotRunUsage",
)


_EVENT_KINDS = frozenset(
    {
        "run.started",
        "provider.request.started",
        "provider.response.received",
        "tool.calls.validation.started",
        "tool.execution.started",
        "tool.result.accepted",
        "run.completed",
        "run.failed",
    }
)


@dataclass(frozen=True, slots=True)
class CopilotProviderUsage:
    """Normalized token usage reported by one provider response."""

    input_tokens: int | None = None
    output_tokens: int | None = None
    total_tokens: int | None = None


@dataclass(frozen=True, slots=True)
class CopilotRunUsage:
    """Conservative provider-usage totals for one logical run."""

    input_tokens: int | None = None
    output_tokens: int | None = None
    total_tokens: int | None = None
    provider_request_count: int = 0
    provider_response_count: int = 0
    usage_report_count: int = 0
    complete: bool = False


@dataclass(frozen=True, slots=True)
class CopilotRunEvent:
    """One payload-free event in a Copilot run lifecycle."""

    run_id: str
    sequence: int
    kind: str
    provider_request_index: int | None = None
    tool_invocation_index: int | None = None
    tool_call_id: str | None = None
    tool_name: str | None = None
    usage: CopilotProviderUsage | None = None
    failure_stage: str | None = None

    def __post_init__(self) -> None:
        if self.kind not in _EVENT_KINDS:
            raise ValueError(f"unsupported Copilot run event kind: {self.kind!r}")


@dataclass(frozen=True, slots=True)
class CopilotRunMetadata:
    """Identity, usage, and immutable lifecycle events for one run."""

    run_id: str
    usage: CopilotRunUsage = field(default_factory=CopilotRunUsage)
    events: tuple[CopilotRunEvent, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "events", tuple(self.events))


def _valid_token_count(value: object) -> bool:
    return type(value) is int and value >= 0


def _aggregate_run_usage(
    provider_usages: tuple[CopilotProviderUsage, ...],
    *,
    provider_request_count: int,
    provider_response_count: int,
    terminal_success: bool,
) -> CopilotRunUsage:
    """Build conservative totals from normalized response-level evidence."""
    complete_response_coverage = (
        provider_request_count > 0
        and provider_response_count == provider_request_count
        and len(provider_usages) == provider_response_count
    )

    def aggregate_field(name: str) -> int | None:
        values = tuple(getattr(usage, name) for usage in provider_usages)
        if not complete_response_coverage or not all(
            _valid_token_count(value) for value in values
        ):
            return None
        return sum(values)

    input_tokens = aggregate_field("input_tokens")
    output_tokens = aggregate_field("output_tokens")
    total_tokens = aggregate_field("total_tokens")
    usage_report_count = sum(
        any(
            _valid_token_count(value)
            for value in (
                usage.input_tokens,
                usage.output_tokens,
                usage.total_tokens,
            )
        )
        for usage in provider_usages
    )
    complete = (
        terminal_success is True
        and input_tokens is not None
        and output_tokens is not None
        and total_tokens is not None
    )
    return CopilotRunUsage(
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        total_tokens=total_tokens,
        provider_request_count=provider_request_count,
        provider_response_count=provider_response_count,
        usage_report_count=usage_report_count,
        complete=complete,
    )

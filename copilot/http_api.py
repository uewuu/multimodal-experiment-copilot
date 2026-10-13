"""Injected, serialized FastAPI transport for Copilot operations."""

import threading

from fastapi import Body, FastAPI, HTTPException, Response, status
from pydantic import BaseModel, ConfigDict, field_validator

from .failure_observability import CopilotFailureObservation
from .run_metadata import (
    CopilotProviderUsage,
    CopilotRunEvent,
    CopilotRunMetadata,
    CopilotRunUsage,
)
from .runtime_observability import (
    CopilotObservedResult,
    CopilotRuntimeMetrics,
)
from .service import CopilotService
from .session import CopilotToolInvocation, CopilotTurn
from .session_repository import CopilotSessionRepository


__all__ = ("create_app",)

_RUN_HEADER = "X-Copilot-Run-Id"


def _map_provider_usage(
    usage: CopilotProviderUsage,
) -> dict[str, int | None]:
    return {
        "input_tokens": usage.input_tokens,
        "output_tokens": usage.output_tokens,
        "total_tokens": usage.total_tokens,
    }


def _map_run_usage(usage: CopilotRunUsage) -> dict[str, object]:
    return {
        "input_tokens": usage.input_tokens,
        "output_tokens": usage.output_tokens,
        "total_tokens": usage.total_tokens,
        "provider_request_count": usage.provider_request_count,
        "provider_response_count": usage.provider_response_count,
        "usage_report_count": usage.usage_report_count,
        "complete": usage.complete,
    }


def _map_run_event(event: CopilotRunEvent) -> dict[str, object]:
    return {
        "run_id": event.run_id,
        "sequence": event.sequence,
        "kind": event.kind,
        "provider_request_index": event.provider_request_index,
        "tool_invocation_index": event.tool_invocation_index,
        "tool_call_id": event.tool_call_id,
        "tool_name": event.tool_name,
        "usage": (
            None if event.usage is None else _map_provider_usage(event.usage)
        ),
        "failure_stage": event.failure_stage,
    }


def _map_run(run: CopilotRunMetadata) -> dict[str, object]:
    return {
        "run_id": run.run_id,
        "usage": _map_run_usage(run.usage),
        "events": [_map_run_event(event) for event in run.events],
    }


def _run_headers(run: CopilotRunMetadata | None) -> dict[str, str]:
    return {} if run is None else {_RUN_HEADER: run.run_id}


def _map_tool_invocation(
    invocation: CopilotToolInvocation,
) -> dict[str, str]:
    return {
        "tool_call_id": invocation.tool_call_id,
        "tool_name": invocation.tool_name,
        "arguments_json": invocation.arguments_json,
        "result_json": invocation.result_json,
    }


def _map_turn(turn: CopilotTurn) -> dict[str, object]:
    return {
        "question": turn.question,
        "answer": turn.answer,
        "tool_call_content": turn.tool_call_content,
        "tool_invocations": [
            _map_tool_invocation(invocation)
            for invocation in turn.tool_invocations
        ],
    }


def _map_metrics(
    metrics: CopilotRuntimeMetrics,
) -> dict[str, int | float]:
    return {
        "provider_request_count": metrics.provider_request_count,
        "tool_invocation_count": metrics.tool_invocation_count,
        "elapsed_seconds": metrics.elapsed_seconds,
    }


def _map_observed_result(
    result: CopilotObservedResult,
) -> dict[str, object]:
    mapped = {
        "turn": _map_turn(result.turn),
        "metrics": _map_metrics(result.metrics),
    }
    if result.run is not None:
        mapped["run"] = _map_run(result.run)
    return mapped


def create_app(
    service: CopilotService,
    session_repository: CopilotSessionRepository,
    *,
    experiment_context: dict[str, object] | None = None,
    max_turns: int = 8,
    turn_timeout_seconds: float | None = None,
) -> FastAPI:
    """Create an HTTP adapter around borrowed Copilot dependencies."""
    class _QuestionRequest(BaseModel):
        model_config = ConfigDict(extra="forbid", strict=True)

        question: str

        @field_validator("question")
        @classmethod
        def _require_nonblank_question(cls, value: str) -> str:
            if not value.strip():
                raise ValueError(
                    "question must not be empty or whitespace"
                )
            return value

    class _EmptyRequest(BaseModel):
        model_config = ConfigDict(extra="forbid", strict=True)

    class _ToolInvocationResponse(BaseModel):
        tool_call_id: str
        tool_name: str
        arguments_json: str
        result_json: str

    class _TurnResponse(BaseModel):
        question: str
        answer: str
        tool_call_content: str | None
        tool_invocations: list[_ToolInvocationResponse]

    class _RuntimeMetricsResponse(BaseModel):
        provider_request_count: int
        tool_invocation_count: int
        elapsed_seconds: float

    class _ProviderUsageResponse(BaseModel):
        input_tokens: int | None
        output_tokens: int | None
        total_tokens: int | None

    class _RunUsageResponse(_ProviderUsageResponse):
        provider_request_count: int
        provider_response_count: int
        usage_report_count: int
        complete: bool

    class _RunEventResponse(BaseModel):
        run_id: str
        sequence: int
        kind: str
        provider_request_index: int | None
        tool_invocation_index: int | None
        tool_call_id: str | None
        tool_name: str | None
        usage: _ProviderUsageResponse | None
        failure_stage: str | None

    class _RunResponse(BaseModel):
        run_id: str
        usage: _RunUsageResponse
        events: list[_RunEventResponse]

    class _ObservedResultResponse(BaseModel):
        turn: _TurnResponse
        metrics: _RuntimeMetricsResponse
        run: _RunResponse | None = None

    class _ObservedTurnResponse(_TurnResponse):
        run: _RunResponse | None = None

    class _SessionCreatedResponse(BaseModel):
        session_id: str

    class _HealthResponse(BaseModel):
        status: str

    application = FastAPI()
    business_lock = threading.Lock()

    @application.post(
        "/v1/copilot/turns",
        response_model=_ObservedResultResponse,
        response_model_exclude_unset=True,
    )
    def run_one_shot(
        request: _QuestionRequest,
        *,
        response: Response = None,
    ) -> dict[str, object]:
        # Keep only transport headers for this request, never observation state.
        failure_headers: dict[str, str] = {}

        def on_failure(observation: CopilotFailureObservation) -> None:
            failure_headers.update(_run_headers(observation.run))

        with business_lock:
            try:
                observed_run = getattr(service, "_run_with_observability", None)
                if callable(observed_run):
                    result = observed_run(
                        request.question,
                        experiment_context=experiment_context,
                        turn_timeout_seconds=turn_timeout_seconds,
                        on_failure=on_failure,
                    )
                else:
                    # Preserve the original contract for injected services.
                    result = service.run(
                        request.question,
                        experiment_context=experiment_context,
                        turn_timeout_seconds=turn_timeout_seconds,
                    )
            except TimeoutError:
                raise HTTPException(
                    status_code=status.HTTP_504_GATEWAY_TIMEOUT,
                    detail="Copilot turn timed out",
                    headers=failure_headers or None,
                ) from None
            except Exception:
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail="Internal server error",
                    headers=failure_headers or None,
                ) from None
        if response is not None:
            response.headers.update(_run_headers(result.run))
        return _map_observed_result(result)

    @application.post(
        "/v1/sessions",
        status_code=status.HTTP_201_CREATED,
        response_model=_SessionCreatedResponse,
    )
    def create_session(
        request: _EmptyRequest | None = Body(default=None),
    ) -> dict[str, str]:
        del request
        with business_lock:
            try:
                session_id = session_repository.create(
                    experiment_context=experiment_context,
                    max_turns=max_turns,
                    turn_timeout_seconds=turn_timeout_seconds,
                )
            except RuntimeError:
                raise HTTPException(
                    status_code=status.HTTP_409_CONFLICT,
                    detail="Session repository conflict",
                ) from None
            except Exception:
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail="Internal server error",
                ) from None
        return {"session_id": session_id}

    @application.post(
        "/v1/sessions/{session_id}/turns",
        response_model=_ObservedTurnResponse,
        response_model_exclude_unset=True,
    )
    def run_session_turn(
        session_id: str,
        request: _QuestionRequest,
        *,
        response: Response = None,
    ) -> dict[str, object]:
        failure_headers: dict[str, str] = {}

        def on_failure(observation: CopilotFailureObservation) -> None:
            failure_headers.update(_run_headers(observation.run))

        result = None
        with business_lock:
            try:
                session = session_repository.get(session_id)
            except KeyError:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail="Session not found",
                ) from None
            except Exception:
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail="Internal server error",
                ) from None

            try:
                observed_ask = getattr(session, "ask_with_observability", None)
                if callable(observed_ask):
                    result = observed_ask(
                        request.question,
                        on_failure=on_failure,
                    )
                    turn = result.turn
                else:
                    turn = session.ask_with_result(request.question)
            except TimeoutError:
                raise HTTPException(
                    status_code=status.HTTP_504_GATEWAY_TIMEOUT,
                    detail="Copilot turn timed out",
                    headers=failure_headers or None,
                ) from None
            except Exception:
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail="Internal server error",
                    headers=failure_headers or None,
                ) from None
        mapped = _map_turn(turn)
        if result is not None and result.run is not None:
            mapped["run"] = _map_run(result.run)
            if response is not None:
                response.headers.update(_run_headers(result.run))
        return mapped

    @application.delete(
        "/v1/sessions/{session_id}",
        status_code=status.HTTP_204_NO_CONTENT,
        response_class=Response,
    )
    def delete_session(session_id: str) -> Response:
        with business_lock:
            try:
                session_repository.delete(session_id)
            except KeyError:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail="Session not found",
                ) from None
            except Exception:
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail="Internal server error",
                ) from None
        return Response(status_code=status.HTTP_204_NO_CONTENT)

    @application.get("/health", response_model=_HealthResponse)
    def health() -> dict[str, str]:
        return {"status": "ok"}

    return application

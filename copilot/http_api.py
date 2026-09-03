"""Injected, serialized FastAPI transport for Copilot operations."""

import threading

from fastapi import Body, FastAPI, HTTPException, Response, status
from pydantic import BaseModel, ConfigDict, field_validator

from .runtime_observability import (
    CopilotObservedResult,
    CopilotRuntimeMetrics,
)
from .service import CopilotService
from .session import CopilotToolInvocation, CopilotTurn
from .session_repository import CopilotSessionRepository


__all__ = ("create_app",)


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
    return {
        "turn": _map_turn(result.turn),
        "metrics": _map_metrics(result.metrics),
    }


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

    class _ObservedResultResponse(BaseModel):
        turn: _TurnResponse
        metrics: _RuntimeMetricsResponse

    class _SessionCreatedResponse(BaseModel):
        session_id: str

    class _HealthResponse(BaseModel):
        status: str

    application = FastAPI()
    business_lock = threading.Lock()

    @application.post(
        "/v1/copilot/turns",
        response_model=_ObservedResultResponse,
    )
    def run_one_shot(request: _QuestionRequest) -> dict[str, object]:
        with business_lock:
            try:
                result = service.run(
                    request.question,
                    experiment_context=experiment_context,
                    turn_timeout_seconds=turn_timeout_seconds,
                )
            except TimeoutError:
                raise HTTPException(
                    status_code=status.HTTP_504_GATEWAY_TIMEOUT,
                    detail="Copilot turn timed out",
                ) from None
            except Exception:
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail="Internal server error",
                ) from None
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
        response_model=_TurnResponse,
    )
    def run_session_turn(
        session_id: str,
        request: _QuestionRequest,
    ) -> dict[str, object]:
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
                turn = session.ask_with_result(request.question)
            except TimeoutError:
                raise HTTPException(
                    status_code=status.HTTP_504_GATEWAY_TIMEOUT,
                    detail="Copilot turn timed out",
                ) from None
            except Exception:
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail="Internal server error",
                ) from None
        return _map_turn(turn)

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

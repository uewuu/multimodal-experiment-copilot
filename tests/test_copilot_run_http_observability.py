"""RED HTTP contracts for M9 run observability transport mapping."""

from __future__ import annotations

import json
from types import SimpleNamespace
from uuid import UUID

import pytest
from fastapi.testclient import TestClient

from copilot import CopilotService, CopilotSessionRepository
import llm_adapters.openai_tool_adapter as adapter


RUN_HEADER = "X-Copilot-Run-Id"
_MISSING = object()


class _SequentialCompletions:
    def __init__(self, outcomes: list[object]) -> None:
        self.outcomes = list(outcomes)
        self.calls: list[dict[str, object]] = []

    def create(self, **kwargs: object) -> object:
        self.calls.append(kwargs)
        index = len(self.calls) - 1
        if index >= len(self.outcomes):
            raise AssertionError("unexpected extra provider request")
        outcome = self.outcomes[index]
        if isinstance(outcome, BaseException):
            raise outcome
        return outcome


class _FakeClient:
    def __init__(self, outcomes: list[object]) -> None:
        self.completions = _SequentialCompletions(outcomes)
        self.chat = SimpleNamespace(completions=self.completions)


def _usage() -> SimpleNamespace:
    return SimpleNamespace(
        prompt_tokens=10,
        completion_tokens=5,
        total_tokens=15,
    )


def _tool_call() -> SimpleNamespace:
    return SimpleNamespace(
        id="call_0",
        type="function",
        function=SimpleNamespace(
            name="analyze_experiment",
            arguments=json.dumps({"experiment_dir": "SECRET_ARGUMENT"}),
        ),
    )


def _response(
    tool_calls: object = None,
    *,
    content: object = "final answer",
    usage: object = _MISSING,
) -> SimpleNamespace:
    response = SimpleNamespace(
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
    if usage is not _MISSING:
        response.usage = usage
    return response


def _assert_uuid4(value: object) -> str:
    assert type(value) is str
    assert value == value.lower()
    parsed = UUID(value)
    assert parsed.version == 4
    assert str(parsed) == value
    return value


def _assert_json_native(value: object) -> None:
    if value is None or type(value) in {str, int, float, bool}:
        return
    if type(value) is list:
        for item in value:
            _assert_json_native(item)
        return
    if type(value) is dict:
        for key, item in value.items():
            assert type(key) is str
            _assert_json_native(item)
        return
    pytest.fail(f"non-JSON-native transport value: {type(value).__name__}")


def _install_tool(monkeypatch: pytest.MonkeyPatch) -> None:
    def invoke(name: str, arguments: dict) -> object:
        del name, arguments
        return {"value": "SECRET_RESULT"}

    monkeypatch.setattr(adapter, "invoke_tool", invoke)


def test_http_one_shot_adds_explicit_run_mapping_and_correlation_header() -> None:
    from copilot.http_api import create_app

    client = _FakeClient([_response(usage=_usage())])
    service = CopilotService(client, model="test-model")
    repository = CopilotSessionRepository(service, max_sessions=2)
    app = create_app(service, repository)
    with TestClient(app) as http:
        response = http.post(
            "/v1/copilot/turns",
            json={"question": "question"},
        )
    assert response.status_code == 200
    payload = response.json()
    assert {"turn", "metrics", "run"} <= payload.keys()
    assert response.headers[RUN_HEADER] == payload["run"]["run_id"]
    _assert_uuid4(payload["run"]["run_id"])
    _assert_json_native(payload)


def test_http_session_turn_adds_metadata_without_replacing_turn_fields() -> None:
    from copilot.http_api import create_app

    client = _FakeClient([_response(usage=_usage())])
    service = CopilotService(client, model="test-model")
    repository = CopilotSessionRepository(
        service,
        max_sessions=2,
        id_factory=lambda: "session-id",
    )
    app = create_app(service, repository)
    with TestClient(app) as http:
        created = http.post("/v1/sessions")
        response = http.post(
            "/v1/sessions/session-id/turns",
            json={"question": "question"},
        )
    assert created.status_code == 201
    assert response.status_code == 200
    payload = response.json()
    assert {
        "question",
        "answer",
        "tool_call_content",
        "tool_invocations",
        "run",
    } <= payload.keys()
    assert "metrics" not in payload
    assert response.headers[RUN_HEADER] == payload["run"]["run_id"]
    _assert_json_native(payload)


@pytest.mark.parametrize(
    ("error", "status_code", "detail"),
    [
        (RuntimeError("SECRET_RUNTIME_FAILURE"), 500, "Internal server error"),
        (TimeoutError("SECRET_TIMEOUT_FAILURE"), 504, "Copilot turn timed out"),
    ],
    ids=("unexpected", "timeout"),
)
def test_http_runtime_failure_has_run_header_and_sanitized_body(
    error: Exception,
    status_code: int,
    detail: str,
) -> None:
    from copilot.http_api import create_app

    client = _FakeClient([error])
    service = CopilotService(client, model="test-model")
    repository = CopilotSessionRepository(service, max_sessions=1)
    app = create_app(service, repository)
    with TestClient(app, raise_server_exceptions=False) as http:
        response = http.post(
            "/v1/copilot/turns",
            json={"question": "question"},
        )
    assert response.status_code == status_code
    assert response.json() == {"detail": detail}
    _assert_uuid4(response.headers[RUN_HEADER])
    assert "SECRET_" not in response.text


def test_http_pre_runtime_failures_do_not_fabricate_run_ids() -> None:
    from copilot.http_api import create_app

    client = _FakeClient([])
    service = CopilotService(client, model="test-model")
    repository = CopilotSessionRepository(service, max_sessions=1)
    app = create_app(service, repository)
    with TestClient(app, raise_server_exceptions=False) as http:
        missing = http.post(
            "/v1/sessions/missing/turns",
            json={"question": "question"},
        )
        assert http.post("/v1/sessions").status_code == 201
        full = http.post("/v1/sessions")
    assert missing.status_code == 404
    assert missing.json() == {"detail": "Session not found"}
    assert RUN_HEADER not in missing.headers
    assert full.status_code == 409
    assert full.json() == {"detail": "Session repository conflict"}
    assert RUN_HEADER not in full.headers
    assert client.completions.calls == []


def test_http_run_mapping_never_leaks_sensitive_runtime_payloads(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from copilot.http_api import create_app

    secret_values = (
        "SECRET_PROMPT",
        "SECRET_ARGUMENT",
        "SECRET_RESULT",
        "SECRET_API_KEY",
        "C:\\SECRET\\experiment",
    )
    _install_tool(monkeypatch)
    client = _FakeClient(
        [
            _response(
                [_tool_call()],
                content="SECRET_PROMPT",
                usage=_usage(),
            ),
            _response(None, content="safe answer", usage=_usage()),
        ]
    )
    service = CopilotService(client, model="test-model")
    repository = CopilotSessionRepository(service, max_sessions=1)
    app = create_app(service, repository)
    with TestClient(app) as http:
        response = http.post(
            "/v1/copilot/turns",
            json={"question": "SECRET_API_KEY"},
        )
    assert response.status_code == 200
    run_payload = response.json()["run"]
    serialized_run = json.dumps(run_payload, sort_keys=True)
    for secret in secret_values:
        assert secret not in serialized_run
    _assert_json_native(run_payload)

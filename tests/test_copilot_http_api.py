"""RED contracts for the injected, serialized FastAPI interface."""

from __future__ import annotations

from collections.abc import Callable
import builtins
import importlib
from importlib.util import find_spec
import inspect
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import threading
from types import SimpleNamespace

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

import copilot
from copilot import (
    CopilotObservedResult,
    CopilotRuntimeMetrics,
    CopilotSessionRepository,
    CopilotToolInvocation,
    CopilotTurn,
)


MODULE_NAME = "copilot.http_api"
HTTP_API_AVAILABLE = find_spec(MODULE_NAME) is not None
requires_http_api = pytest.mark.skipif(
    not HTTP_API_AVAILABLE,
    reason="copilot.http_api is the missing M8 production capability",
)
REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
REQUIREMENTS_PATH = REPOSITORY_ROOT / "requirements.txt"


class _FakeService:
    def __init__(self, outcome: object | None = None) -> None:
        self.outcome = _observed_result() if outcome is None else outcome
        self.calls: list[tuple[str, dict[str, object]]] = []
        self.close_count = 0

    def run(self, question: str, **kwargs: object) -> object:
        self.calls.append((question, kwargs))
        if isinstance(self.outcome, BaseException):
            raise self.outcome
        return self.outcome

    def close(self) -> None:
        self.close_count += 1


class _FakeSession:
    def __init__(self, outcome: object | None = None) -> None:
        self.outcome = _turn() if outcome is None else outcome
        self.questions: list[str] = []
        self.close_count = 0
        self.reset_count = 0

    def ask_with_result(self, question: str) -> object:
        self.questions.append(question)
        if isinstance(self.outcome, BaseException):
            raise self.outcome
        return self.outcome

    def close(self) -> None:
        self.close_count += 1

    def reset(self) -> None:
        self.reset_count += 1


class _FakeRepository:
    def __init__(
        self,
        *,
        create_outcome: object = "session-id",
        session: object | None = None,
        get_error: BaseException | None = None,
        delete_error: BaseException | None = None,
    ) -> None:
        self.create_outcome = create_outcome
        self.session = _FakeSession() if session is None else session
        self.get_error = get_error
        self.delete_error = delete_error
        self.create_calls: list[dict[str, object]] = []
        self.get_calls: list[str] = []
        self.delete_calls: list[str] = []
        self.close_count = 0

    def create(self, **kwargs: object) -> str:
        self.create_calls.append(kwargs)
        if isinstance(self.create_outcome, BaseException):
            raise self.create_outcome
        return self.create_outcome  # type: ignore[return-value]

    def get(self, session_id: str) -> object:
        self.get_calls.append(session_id)
        if self.get_error is not None:
            raise self.get_error
        return self.session

    def delete(self, session_id: str) -> None:
        self.delete_calls.append(session_id)
        if self.delete_error is not None:
            raise self.delete_error

    def close(self) -> None:
        self.close_count += 1


def _turn(
    *,
    question: str = "question",
    answer: str = "answer",
    tool_call_content: str | None = None,
    invocations: tuple[CopilotToolInvocation, ...] = (),
) -> CopilotTurn:
    return CopilotTurn(
        question=question,
        answer=answer,
        tool_call_content=tool_call_content,
        tool_invocations=invocations,
    )


def _observed_result() -> CopilotObservedResult:
    return CopilotObservedResult(
        turn=_turn(),
        metrics=CopilotRuntimeMetrics(
            provider_request_count=1,
            tool_invocation_count=0,
            elapsed_seconds=0.25,
        ),
    )


def _load_http_module() -> object:
    return importlib.import_module(MODULE_NAME)


@pytest.fixture
def http_module() -> object:
    if not HTTP_API_AVAILABLE:
        pytest.skip("copilot.http_api is not implemented")
    return _load_http_module()


def _create_app(
    module: object,
    service: object | None = None,
    repository: object | None = None,
    **kwargs: object,
) -> FastAPI:
    selected_service = _FakeService() if service is None else service
    selected_repository = (
        _FakeRepository() if repository is None else repository
    )
    return module.create_app(  # type: ignore[attr-defined,no-any-return]
        selected_service,
        selected_repository,
        **kwargs,
    )


def _assert_json_native(value: object) -> None:
    if value is None or type(value) in (str, int, float, bool):
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
    pytest.fail(f"non-JSON-native value: {type(value).__name__}")


def _thread_call(action: Callable[[], object]) -> tuple[threading.Thread, dict]:
    outcome: dict[str, object] = {}

    def target() -> None:
        try:
            outcome["result"] = action()
        except BaseException as error:
            outcome["error"] = error

    thread = threading.Thread(target=target)
    thread.start()
    return thread, outcome


def _join(thread: threading.Thread) -> None:
    thread.join(timeout=3)
    assert not thread.is_alive(), "concurrent HTTP request did not finish"


# Dependency, TestClient, and healthy RED loading contracts.


def test_http_dependency_manifest_has_only_approved_direct_requirements() -> None:
    lines = REQUIREMENTS_PATH.read_text(encoding="utf-8").splitlines()
    assert lines == [
        "PyYAML>=6.0,<7",
        "pytest>=8.0,<9",
        "fastapi>=0.116,<1",
        "pydantic>=2.11,<3",
        "httpx>=0.28,<1",
    ]
    lowered = "\n".join(lines).lower()
    for forbidden in ("uvicorn", "openai", "starlette", "anyio", "lxml"):
        assert forbidden not in lowered


def test_resolved_http_stack_imports_and_testclient_are_compatible() -> None:
    app = FastAPI()
    app.add_api_route(
        "/smoke",
        lambda: {"status": "ok"},
        methods=["GET"],
    )
    with TestClient(app) as client:
        response = client.get("/smoke")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_core_package_import_remains_http_dependency_independent() -> None:
    script = r'''
import builtins
import sys

sys.dont_write_bytecode = True
original_import = builtins.__import__
blocked = ("fastapi", "pydantic", "starlette", "httpx", "openai")

def guarded_import(name, *args, **kwargs):
    if any(name == item or name.startswith(item + ".") for item in blocked):
        raise AssertionError(f"blocked dependency imported: {name}")
    return original_import(name, *args, **kwargs)

builtins.__import__ = guarded_import
import copilot
assert "create_app" not in copilot.__all__
assert not hasattr(copilot, "create_app")
assert not hasattr(copilot, "app")
'''
    environment = os.environ.copy()
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    completed = subprocess.run(
        [sys.executable, "-c", script],
        cwd=REPOSITORY_ROOT,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    assert completed.stdout == ""
    assert completed.stderr == ""


def test_package_root_does_not_export_http_api() -> None:
    assert "create_app" not in copilot.__all__
    assert not hasattr(copilot, "create_app")
    assert not hasattr(copilot, "app")


def test_http_module_is_implemented() -> None:
    assert HTTP_API_AVAILABLE, "copilot.http_api is not implemented"


# Module, construction, and route surface contracts.


@requires_http_api
def test_http_module_import_is_side_effect_free() -> None:
    script = r'''
import builtins
import concurrent.futures
import os
from pathlib import Path
import socket
import sys
import threading

sys.dont_write_bytecode = True
import fastapi
import pydantic
import starlette
import httpx
concurrent.futures.ThreadPoolExecutor
concurrent.futures.ProcessPoolExecutor
original_import = builtins.__import__
original_open = builtins.open

def guarded_import(name, *args, **kwargs):
    if name == "openai" or name.startswith("openai."):
        raise AssertionError("OpenAI SDK import")
    if name == "llm_clients" or name.startswith("llm_clients."):
        raise AssertionError("provider client factory import")
    return original_import(name, *args, **kwargs)

def guarded_open(file, mode="r", *args, **kwargs):
    if any(flag in mode for flag in "wax+"):
        raise AssertionError("filesystem write")
    return original_open(file, mode, *args, **kwargs)

def forbidden(*args, **kwargs):
    raise AssertionError("HTTP module import side effect")

builtins.__import__ = guarded_import
builtins.open = guarded_open
os.getenv = forbidden
os._Environ.get = forbidden
os._Environ.__getitem__ = forbidden
Path.mkdir = forbidden
Path.touch = forbidden
Path.write_bytes = forbidden
Path.write_text = forbidden
socket.create_connection = forbidden
socket.socket.connect = forbidden
threading.Lock = forbidden
threading.Thread.__init__ = forbidden
concurrent.futures.ThreadPoolExecutor.__init__ = forbidden
concurrent.futures.ProcessPoolExecutor.__init__ = forbidden

import copilot.http_api
'''
    environment = os.environ.copy()
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    completed = subprocess.run(
        [sys.executable, "-c", script],
        cwd=REPOSITORY_ROOT,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    assert completed.stdout == ""
    assert completed.stderr == ""


@requires_http_api
def test_module_exposes_factory_without_global_application() -> None:
    module = _load_http_module()
    assert callable(module.create_app)
    for name in ("app", "run_server", "main", "serve"):
        assert not hasattr(module, name)


def test_create_app_has_exact_signature(http_module: object) -> None:
    parameters = list(
        inspect.signature(http_module.create_app).parameters.values()
    )
    assert [parameter.name for parameter in parameters] == [
        "service",
        "session_repository",
        "experiment_context",
        "max_turns",
        "turn_timeout_seconds",
    ]
    assert parameters[0].kind is inspect.Parameter.POSITIONAL_OR_KEYWORD
    assert parameters[1].kind is inspect.Parameter.POSITIONAL_OR_KEYWORD
    for parameter in parameters[2:]:
        assert parameter.kind is inspect.Parameter.KEYWORD_ONLY
    assert parameters[2].default is None
    assert parameters[3].default == 8
    assert parameters[4].default is None


def test_create_app_returns_fastapi_without_business_calls(
    http_module: object,
) -> None:
    service = _FakeService()
    repository = _FakeRepository()
    app = _create_app(http_module, service, repository)
    assert isinstance(app, FastAPI)
    assert service.calls == []
    assert repository.create_calls == []
    assert repository.get_calls == []
    assert repository.delete_calls == []
    assert service.close_count == 0
    assert repository.close_count == 0


def test_approved_route_surface_is_exact(http_module: object) -> None:
    app = _create_app(http_module)
    routes = {
        (method, route.path)
        for route in app.routes
        for method in getattr(route, "methods", set())
        if route.path == "/health" or route.path.startswith("/v1/")
    }
    assert routes == {
        ("POST", "/v1/copilot/turns"),
        ("POST", "/v1/sessions"),
        ("POST", "/v1/sessions/{session_id}/turns"),
        ("DELETE", "/v1/sessions/{session_id}"),
        ("GET", "/health"),
    }


def test_health_is_exact_and_has_no_business_side_effect(
    http_module: object,
) -> None:
    service = _FakeService()
    repository = _FakeRepository()
    app = _create_app(http_module, service, repository)
    with TestClient(app) as client:
        response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}
    assert service.calls == []
    assert repository.create_calls == []
    assert repository.get_calls == []
    assert repository.delete_calls == []


# Strict request and trusted host-policy contracts.


@pytest.mark.parametrize("path", ["/v1/copilot/turns", "/v1/sessions/id/turns"])
@pytest.mark.parametrize(
    "body",
    [
        {},
        {"question": None},
        {"question": 1},
        {"question": True},
        {"question": ""},
        {"question": " "},
        {"question": "\t\n"},
        {"question": "valid", "model": "forbidden"},
        {"question": "valid", "request_options": {}},
        {"question": "valid", "experiment_context": {}},
        {"question": "valid", "experiment_dir": "forbidden"},
        {"question": "valid", "experiment_root": "forbidden"},
        {"question": "valid", "metrics_config": "forbidden"},
        {"question": "valid", "turn_timeout_seconds": 5},
        {"question": "valid", "timeout": 5},
        {"question": "valid", "max_turns": 3},
        {"question": "valid", "tools": []},
        {"question": "valid", "tool_choice": "auto"},
        {"question": "valid", "on_failure": "forbidden"},
        {"question": "valid", "temperature": 2},
    ],
)
def test_question_body_is_strict(
    http_module: object,
    path: str,
    body: dict,
) -> None:
    app = _create_app(http_module)
    with TestClient(app) as client:
        response = client.post(path, json=body)
    assert response.status_code == 422


def test_one_shot_forwards_exact_host_policy(http_module: object) -> None:
    service = _FakeService()
    repository = _FakeRepository()
    context = {"experiment_dir": "trusted"}
    app = _create_app(
        http_module,
        service,
        repository,
        experiment_context=context,
        turn_timeout_seconds=7.5,
    )
    question = "  preserve question spacing  "
    with TestClient(app) as client:
        response = client.post(
            "/v1/copilot/turns",
            json={"question": question},
        )
    assert response.status_code == 200
    assert service.calls == [
        (
            question,
            {
                "experiment_context": context,
                "turn_timeout_seconds": 7.5,
            },
        )
    ]
    assert service.calls[0][1]["experiment_context"] is context


def test_session_create_forwards_exact_host_policy(
    http_module: object,
) -> None:
    service = _FakeService()
    repository = _FakeRepository(create_outcome="opaque-id")
    context = {"experiment_root": "trusted"}
    app = _create_app(
        http_module,
        service,
        repository,
        experiment_context=context,
        max_turns=5,
        turn_timeout_seconds=8.0,
    )
    with TestClient(app) as client:
        response = client.post("/v1/sessions")
    assert response.status_code == 201
    assert response.json() == {"session_id": "opaque-id"}
    assert repository.create_calls == [
        {
            "experiment_context": context,
            "max_turns": 5,
            "turn_timeout_seconds": 8.0,
        }
    ]
    assert repository.create_calls[0]["experiment_context"] is context


@pytest.mark.parametrize(
    "body",
    [
        {"experiment_context": {"experiment_root": "outside"}},
        {"max_turns": 1000},
        {"turn_timeout_seconds": None},
        {"request_options": {"temperature": 2}},
        {"max_sessions": 1000},
    ],
)
def test_session_create_rejects_client_policy(
    http_module: object,
    body: dict,
) -> None:
    repository = _FakeRepository()
    app = _create_app(http_module, repository=repository)
    with TestClient(app) as client:
        response = client.post("/v1/sessions", json=body)
    assert response.status_code == 422
    assert repository.create_calls == []


# Explicit transport serialization and endpoint delegation.


def test_one_shot_explicitly_maps_complete_observed_result(
    http_module: object,
) -> None:
    invocation = CopilotToolInvocation(
        tool_call_id="call-1",
        tool_name="analyze_experiment",
        arguments_json='{ "experiment_dir" : "demo" }',
        result_json='{"score":0.75}',
    )
    result = CopilotObservedResult(
        turn=_turn(
            question="analyze",
            answer="done",
            tool_call_content="working",
            invocations=(invocation,),
        ),
        metrics=CopilotRuntimeMetrics(2, 1, 0.25),
    )
    app = _create_app(http_module, service=_FakeService(result))
    with TestClient(app) as client:
        response = client.post(
            "/v1/copilot/turns",
            json={"question": "analyze"},
        )
    assert response.status_code == 200
    assert response.json() == {
        "turn": {
            "question": "analyze",
            "answer": "done",
            "tool_call_content": "working",
            "tool_invocations": [
                {
                    "tool_call_id": "call-1",
                    "tool_name": "analyze_experiment",
                    "arguments_json": '{ "experiment_dir" : "demo" }',
                    "result_json": '{"score":0.75}',
                }
            ],
        },
        "metrics": {
            "provider_request_count": 2,
            "tool_invocation_count": 1,
            "elapsed_seconds": 0.25,
        },
    }
    _assert_json_native(response.json())


def test_session_turn_uses_exact_session_and_maps_turn(
    http_module: object,
) -> None:
    turn = _turn(question="follow up", answer="session answer")
    session = _FakeSession(turn)
    repository = _FakeRepository(session=session)
    service = _FakeService()
    app = _create_app(http_module, service, repository)
    with TestClient(app) as client:
        response = client.post(
            "/v1/sessions/session-1/turns",
            json={"question": "follow up"},
        )
    assert response.status_code == 200
    assert response.json() == {
        "question": "follow up",
        "answer": "session answer",
        "tool_call_content": None,
        "tool_invocations": [],
    }
    assert repository.get_calls == ["session-1"]
    assert session.questions == ["follow up"]
    assert service.calls == []
    _assert_json_native(response.json())


def test_delete_calls_repository_once_and_returns_empty_204(
    http_module: object,
) -> None:
    repository = _FakeRepository()
    app = _create_app(http_module, repository=repository)
    with TestClient(app) as client:
        response = client.delete("/v1/sessions/session-1")
    assert response.status_code == 204
    assert response.content == b""
    assert repository.delete_calls == ["session-1"]
    assert repository.session.reset_count == 0
    assert repository.session.close_count == 0


def test_http_module_does_not_use_generic_dataclass_serialization(
    http_module: object,
) -> None:
    source = inspect.getsource(http_module)
    assert "asdict" not in source


# Narrow, sanitized error translation contracts.


@pytest.mark.parametrize("operation", ["get", "delete"])
def test_missing_session_maps_to_404_only_at_repository_boundary(
    http_module: object,
    operation: str,
) -> None:
    missing = KeyError("SECRET_MISSING_ID")
    repository = _FakeRepository(
        get_error=missing if operation == "get" else None,
        delete_error=missing if operation == "delete" else None,
    )
    app = _create_app(http_module, repository=repository)
    with TestClient(app, raise_server_exceptions=False) as client:
        if operation == "get":
            response = client.post(
                "/v1/sessions/missing/turns",
                json={"question": "question"},
            )
        else:
            response = client.delete("/v1/sessions/missing")
    assert response.status_code == 404
    assert response.json() == {"detail": "Session not found"}
    assert "SECRET_MISSING_ID" not in response.text


def test_session_internal_key_error_is_sanitized_500(
    http_module: object,
) -> None:
    session = _FakeSession(KeyError("SECRET_TOOL_KEY"))
    app = _create_app(
        http_module,
        repository=_FakeRepository(session=session),
    )
    with TestClient(app, raise_server_exceptions=False) as client:
        response = client.post(
            "/v1/sessions/id/turns",
            json={"question": "question"},
        )
    assert response.status_code == 500
    assert response.json() == {"detail": "Internal server error"}
    assert "SECRET_TOOL_KEY" not in response.text


def test_repository_create_conflict_maps_to_sanitized_409(
    http_module: object,
) -> None:
    repository = _FakeRepository(
        create_outcome=RuntimeError("SECRET_REPOSITORY_STATE")
    )
    app = _create_app(http_module, repository=repository)
    with TestClient(app, raise_server_exceptions=False) as client:
        response = client.post("/v1/sessions")
    assert response.status_code == 409
    assert response.json() == {"detail": "Session repository conflict"}
    assert "SECRET_REPOSITORY_STATE" not in response.text


@pytest.mark.parametrize("target", ["one-shot", "session-turn"])
def test_turn_timeout_maps_to_sanitized_504(
    http_module: object,
    target: str,
) -> None:
    error = TimeoutError("SECRET_TIMEOUT_DETAIL")
    service = _FakeService(error if target == "one-shot" else None)
    session = _FakeSession(error if target == "session-turn" else None)
    repository = _FakeRepository(session=session)
    app = _create_app(http_module, service, repository)
    with TestClient(app, raise_server_exceptions=False) as client:
        if target == "one-shot":
            response = client.post(
                "/v1/copilot/turns",
                json={"question": "question"},
            )
        else:
            response = client.post(
                "/v1/sessions/id/turns",
                json={"question": "question"},
            )
    assert response.status_code == 504
    assert response.json() == {"detail": "Copilot turn timed out"}
    assert "SECRET_TIMEOUT_DETAIL" not in response.text


@pytest.mark.parametrize("target", ["one-shot", "session-turn", "delete"])
def test_unexpected_exception_maps_to_sanitized_500(
    http_module: object,
    target: str,
) -> None:
    error = RuntimeError("SECRET_INTERNAL_PATH_OR_PROVIDER_PAYLOAD")
    service = _FakeService(error if target == "one-shot" else None)
    session = _FakeSession(error if target == "session-turn" else None)
    repository = _FakeRepository(
        session=session,
        delete_error=error if target == "delete" else None,
    )
    app = _create_app(http_module, service, repository)
    with TestClient(app, raise_server_exceptions=False) as client:
        if target == "one-shot":
            response = client.post(
                "/v1/copilot/turns",
                json={"question": "question"},
            )
        elif target == "session-turn":
            response = client.post(
                "/v1/sessions/id/turns",
                json={"question": "question"},
            )
        else:
            response = client.delete("/v1/sessions/id")
    assert response.status_code == 500
    assert response.json() == {"detail": "Internal server error"}
    assert "SECRET_INTERNAL_PATH_OR_PROVIDER_PAYLOAD" not in response.text


@pytest.mark.parametrize(
    "error",
    [KeyboardInterrupt(), SystemExit(7), GeneratorExit()],
    ids=["keyboard-interrupt", "system-exit", "generator-exit"],
)
def test_base_exceptions_propagate_by_identity(
    http_module: object,
    error: BaseException,
) -> None:
    app = _create_app(http_module, service=_FakeService(error))
    route = next(
        route
        for route in app.routes
        if route.path == "/v1/copilot/turns"
        and "POST" in getattr(route, "methods", set())
    )
    endpoint = route.endpoint
    request_type = inspect.signature(endpoint).parameters[
        "request"
    ].annotation
    request = request_type(question="question")
    with pytest.raises(type(error)) as caught:
        endpoint(request)
    assert caught.value is error


def test_adapter_never_closes_borrowed_dependencies(
    http_module: object,
) -> None:
    service = _FakeService()
    session = _FakeSession()
    repository = _FakeRepository(session=session)
    app = _create_app(http_module, service, repository)
    with TestClient(app) as client:
        assert client.get("/health").status_code == 200
        assert client.post(
            "/v1/copilot/turns",
            json={"question": "one"},
        ).status_code == 200
        assert client.post("/v1/sessions").status_code == 201
        assert client.post(
            "/v1/sessions/id/turns",
            json={"question": "two"},
        ).status_code == 200
        assert client.delete("/v1/sessions/id").status_code == 204
    assert service.close_count == 0
    assert repository.close_count == 0
    assert session.close_count == 0
    assert session.reset_count == 0


# Deterministic app-scoped global business serialization contracts.


class _BlockingSession(_FakeSession):
    def __init__(self) -> None:
        super().__init__()
        self.entered = threading.Event()
        self.release = threading.Event()
        self.overlap = threading.Event()
        self._call_count = 0
        self._count_lock = threading.Lock()

    def ask_with_result(self, question: str) -> CopilotTurn:
        with self._count_lock:
            self._call_count += 1
            position = self._call_count
        if position == 1:
            self.entered.set()
            assert self.release.wait(3), "first turn was not released"
        elif not self.release.is_set():
            self.overlap.set()
        self.questions.append(question)
        return _turn(question=question, answer=f"answer-{position}")


class _BlockingService(_FakeService):
    def __init__(self) -> None:
        super().__init__()
        self.entered = threading.Event()
        self.release = threading.Event()

    def run(self, question: str, **kwargs: object) -> CopilotObservedResult:
        self.calls.append((question, kwargs))
        self.entered.set()
        assert self.release.wait(3), "one-shot call was not released"
        return _observed_result()


def test_same_session_turns_are_serialized(http_module: object) -> None:
    session = _BlockingSession()
    repository = _FakeRepository(session=session)
    app = _create_app(http_module, repository=repository)
    with TestClient(app) as client:
        first, first_outcome = _thread_call(
            lambda: client.post(
                "/v1/sessions/id/turns",
                json={"question": "first"},
            )
        )
        assert session.entered.wait(1)
        second, second_outcome = _thread_call(
            lambda: client.post(
                "/v1/sessions/id/turns",
                json={"question": "second"},
            )
        )
        try:
            assert not session.overlap.wait(0.25)
        finally:
            session.release.set()
        _join(first)
        _join(second)
    assert "error" not in first_outcome
    assert "error" not in second_outcome
    assert sorted(
        result.status_code
        for result in (
            first_outcome["result"],
            second_outcome["result"],
        )
    ) == [200, 200]
    assert len(session.questions) == 2
    assert not session.overlap.is_set()


def test_concurrent_create_cannot_exceed_real_repository_capacity(
    http_module: object,
) -> None:
    class BlockingCreationService:
        def __init__(self) -> None:
            self.calls = 0
            self.entered = threading.Event()
            self.release = threading.Event()

        def create_session(self, **kwargs: object) -> object:
            self.calls += 1
            self.entered.set()
            assert self.release.wait(3), "Session creation was not released"
            return object()

    service = BlockingCreationService()
    identifiers = iter(("first-id", "second-id"))
    repository = CopilotSessionRepository(
        service,  # type: ignore[arg-type]
        max_sessions=1,
        id_factory=lambda: next(identifiers),
    )
    app = _create_app(http_module, _FakeService(), repository)
    with TestClient(app) as client:
        first, first_outcome = _thread_call(
            lambda: client.post("/v1/sessions")
        )
        assert service.entered.wait(1)
        second, second_outcome = _thread_call(
            lambda: client.post("/v1/sessions")
        )
        try:
            assert service.calls == 1
        finally:
            service.release.set()
        _join(first)
        _join(second)
    statuses = sorted(
        outcome["result"].status_code
        for outcome in (first_outcome, second_outcome)
    )
    assert statuses == [201, 409]
    assert service.calls == 1
    assert repository.get("first-id") is not None


def test_delete_cannot_interleave_with_active_turn(
    http_module: object,
) -> None:
    session = _BlockingSession()

    class DeleteProbeRepository(_FakeRepository):
        def __init__(self) -> None:
            super().__init__(session=session)
            self.delete_entered = threading.Event()

        def delete(self, session_id: str) -> None:
            self.delete_entered.set()
            super().delete(session_id)

    repository = DeleteProbeRepository()
    app = _create_app(http_module, repository=repository)
    with TestClient(app) as client:
        turn, turn_outcome = _thread_call(
            lambda: client.post(
                "/v1/sessions/id/turns",
                json={"question": "question"},
            )
        )
        assert session.entered.wait(1)
        delete, delete_outcome = _thread_call(
            lambda: client.delete("/v1/sessions/id")
        )
        try:
            assert not repository.delete_entered.wait(0.25)
        finally:
            session.release.set()
        _join(turn)
        _join(delete)
    assert turn_outcome["result"].status_code == 200
    assert delete_outcome["result"].status_code == 204
    assert repository.delete_entered.is_set()


def test_one_shot_and_session_create_share_one_business_lock(
    http_module: object,
) -> None:
    service = _BlockingService()

    class CreateProbeRepository(_FakeRepository):
        def __init__(self) -> None:
            super().__init__()
            self.create_entered = threading.Event()

        def create(self, **kwargs: object) -> str:
            self.create_entered.set()
            return super().create(**kwargs)

    repository = CreateProbeRepository()
    app = _create_app(http_module, service, repository)
    with TestClient(app) as client:
        one_shot, one_shot_outcome = _thread_call(
            lambda: client.post(
                "/v1/copilot/turns",
                json={"question": "question"},
            )
        )
        assert service.entered.wait(1)
        create, create_outcome = _thread_call(
            lambda: client.post("/v1/sessions")
        )
        try:
            assert not repository.create_entered.wait(0.25)
        finally:
            service.release.set()
        _join(one_shot)
        _join(create)
    assert one_shot_outcome["result"].status_code == 200
    assert create_outcome["result"].status_code == 201
    assert repository.create_entered.is_set()


def test_business_locks_are_independent_per_app(http_module: object) -> None:
    first_service = _BlockingService()
    second_repository = _FakeRepository()
    first_app = _create_app(http_module, first_service, _FakeRepository())
    second_app = _create_app(
        http_module,
        _FakeService(),
        second_repository,
    )
    with TestClient(first_app) as first_client, TestClient(
        second_app
    ) as second_client:
        first, first_outcome = _thread_call(
            lambda: first_client.post(
                "/v1/copilot/turns",
                json={"question": "question"},
            )
        )
        assert first_service.entered.wait(1)
        try:
            response = second_client.post("/v1/sessions")
            assert response.status_code == 201
            assert len(second_repository.create_calls) == 1
        finally:
            first_service.release.set()
        _join(first)
    assert first_outcome["result"].status_code == 200


def test_health_bypasses_busy_business_lock(http_module: object) -> None:
    service = _BlockingService()
    app = _create_app(http_module, service, _FakeRepository())
    with TestClient(app) as client:
        business, business_outcome = _thread_call(
            lambda: client.post(
                "/v1/copilot/turns",
                json={"question": "question"},
            )
        )
        assert service.entered.wait(1)
        health, health_outcome = _thread_call(lambda: client.get("/health"))
        try:
            _join(health)
            assert health_outcome["result"].status_code == 200
            assert health_outcome["result"].json() == {"status": "ok"}
        finally:
            service.release.set()
        _join(business)
    assert business_outcome["result"].status_code == 200


def test_http_scope_has_no_server_auth_cors_or_provider_factory(
    http_module: object,
) -> None:
    source = inspect.getsource(http_module).lower()
    for forbidden in (
        "uvicorn",
        "corsmiddleware",
        "allow_origins",
        "oauth",
        "jwt",
        "create_openai_client",
        "llm_clients",
    ):
        assert forbidden not in source


def test_existing_application_primitives_remain_unchanged() -> None:
    assert list(inspect.signature(copilot.CopilotService).parameters) == [
        "client",
        "model",
    ]
    assert list(
        inspect.signature(copilot.CopilotSessionRepository).parameters
    ) == ["service", "max_sessions", "id_factory"]
    assert list(inspect.signature(copilot.CopilotSession).parameters) == [
        "client",
        "model",
        "experiment_context",
        "max_turns",
        "turn_timeout_seconds",
        "request_options",
    ]

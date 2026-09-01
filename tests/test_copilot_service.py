"""Contracts for the thin injected-client Copilot service facade."""

from collections.abc import Callable
import importlib
from importlib.util import find_spec
import inspect
import os
from pathlib import Path
import subprocess
import sys
from typing import get_args, get_origin, get_type_hints

import pytest

import copilot
from copilot import (
    CopilotFailureObservation,
    CopilotObservedResult,
    CopilotSession,
)


MODULE_NAME = "copilot.service"
SERVICE_AVAILABLE = find_spec(MODULE_NAME) is not None
requires_service = pytest.mark.skipif(
    not SERVICE_AVAILABLE,
    reason="copilot.service is the missing M6 production capability",
)


class _BorrowedClient:
    def __init__(self) -> None:
        self.close_count = 0

    def close(self) -> None:
        self.close_count += 1


def _load_service_module(contract: str) -> object:
    try:
        module = importlib.import_module(MODULE_NAME)
    except ModuleNotFoundError as error:
        if error.name == MODULE_NAME:
            pytest.fail(
                f"{contract}: copilot.service is not implemented"
            )
        raise
    assert hasattr(module, "CopilotService"), (
        f"{contract}: CopilotService is not implemented"
    )
    return module


def _service_type(contract: str) -> type:
    module = _load_service_module(contract)
    service_type = getattr(module, "CopilotService")
    assert isinstance(service_type, type), (
        f"{contract}: CopilotService must be a class"
    )
    return service_type


def _forbidden(*args: object, **kwargs: object) -> object:
    raise AssertionError("an unselected dependency was called")


def test_existing_runtime_primitives_remain_public() -> None:
    expected = {
        "run_copilot_turn",
        "run_copilot_turn_with_result",
        "run_copilot_turn_with_observability",
        "run_copilot_turn_with_failure_observability",
        "CopilotSession",
    }
    assert expected <= set(copilot.__all__)
    assert all(hasattr(copilot, name) for name in expected)


def test_existing_result_types_remain_public() -> None:
    expected = {
        "CopilotObservedResult",
        "CopilotFailureObservation",
        "CopilotTurn",
    }
    assert expected <= set(copilot.__all__)
    assert all(hasattr(copilot, name) for name in expected)


def test_existing_runtime_signatures_remain_compatible() -> None:
    for function_name in (
        "run_copilot_turn",
        "run_copilot_turn_with_result",
        "run_copilot_turn_with_observability",
        "run_copilot_turn_with_failure_observability",
    ):
        parameters = inspect.signature(
            getattr(copilot, function_name)
        ).parameters
        assert list(parameters)[:6] == [
            "client",
            "model",
            "question",
            "experiment_context",
            "turn_timeout_seconds",
            (
                "on_failure"
                if function_name.endswith("failure_observability")
                else "request_options"
            ),
        ]


def test_existing_session_signature_remains_compatible() -> None:
    parameters = inspect.signature(CopilotSession).parameters
    assert list(parameters) == [
        "client",
        "model",
        "experiment_context",
        "max_turns",
        "turn_timeout_seconds",
        "request_options",
    ]
    assert parameters["max_turns"].default == 8
    assert parameters["turn_timeout_seconds"].default is None


def test_service_module_is_available() -> None:
    assert SERVICE_AVAILABLE, "copilot.service is not implemented"


@requires_service
def test_service_has_exact_public_signatures() -> None:
    service_type = _service_type("public signature contract")
    constructor = list(
        inspect.signature(service_type.__init__).parameters.values()
    )
    assert [item.name for item in constructor] == [
        "self",
        "client",
        "model",
    ]
    assert constructor[1].kind is inspect.Parameter.POSITIONAL_OR_KEYWORD
    assert constructor[2].kind is inspect.Parameter.KEYWORD_ONLY
    assert constructor[2].default is inspect.Parameter.empty

    run_parameters = list(
        inspect.signature(service_type.run).parameters.values()
    )
    assert [item.name for item in run_parameters] == [
        "self",
        "question",
        "experiment_context",
        "turn_timeout_seconds",
        "on_failure",
        "request_options",
    ]
    assert run_parameters[1].kind is inspect.Parameter.POSITIONAL_OR_KEYWORD
    for item in run_parameters[2:5]:
        assert item.kind is inspect.Parameter.KEYWORD_ONLY
    assert run_parameters[2].default is None
    assert run_parameters[3].default is None
    assert run_parameters[4].default is None
    assert run_parameters[5].kind is inspect.Parameter.VAR_KEYWORD

    session_parameters = list(
        inspect.signature(service_type.create_session).parameters.values()
    )
    assert [item.name for item in session_parameters] == [
        "self",
        "experiment_context",
        "max_turns",
        "turn_timeout_seconds",
        "request_options",
    ]
    for item in session_parameters[1:4]:
        assert item.kind is inspect.Parameter.KEYWORD_ONLY
    assert session_parameters[1].default is None
    assert session_parameters[2].default == 8
    assert session_parameters[3].default is None
    assert session_parameters[4].kind is inspect.Parameter.VAR_KEYWORD

    run_hints = get_type_hints(service_type.run)
    assert run_hints["question"] is str
    assert run_hints["experiment_context"] == (
        dict[str, object] | None
    )
    assert run_hints["turn_timeout_seconds"] == float | None
    callback_union = get_args(run_hints["on_failure"])
    assert type(None) in callback_union
    callback_hint = next(
        hint for hint in callback_union if hint is not type(None)
    )
    assert get_origin(callback_hint) is Callable
    assert get_args(callback_hint) == (
        [CopilotFailureObservation],
        type(None),
    )
    assert run_hints["return"] is CopilotObservedResult
    assert get_type_hints(service_type.create_session)["return"] is (
        CopilotSession
    )


@requires_service
def test_service_adds_no_factory_or_result_types() -> None:
    module = _load_service_module("minimal public concept contract")
    for name in (
        "create_copilot_service",
        "ServiceResult",
        "ServiceTrace",
        "ServiceObservation",
        "ServiceError",
        "CopilotServiceError",
        "ServiceConfig",
        "DependencyContainer",
    ):
        assert not hasattr(module, name)


@requires_service
def test_construction_has_no_execution_or_ownership_side_effects(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load_service_module("construction contract")
    service_type = _service_type("construction contract")
    monkeypatch.setattr(
        module,
        "run_copilot_turn_with_observability",
        _forbidden,
    )
    monkeypatch.setattr(
        module,
        "run_copilot_turn_with_failure_observability",
        _forbidden,
    )
    monkeypatch.setattr(module, "CopilotSession", _forbidden)
    client = _BorrowedClient()

    service_type(client, model="fixed-model")

    assert client.close_count == 0


@requires_service
def test_service_has_no_client_lifecycle_api() -> None:
    service_type = _service_type("borrowed client ownership contract")
    for name in (
        "close",
        "__enter__",
        "__exit__",
        "__aenter__",
        "__aexit__",
    ):
        assert not hasattr(service_type, name)


@requires_service
def test_run_without_callback_delegates_exactly_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load_service_module("ordinary one-shot delegation")
    service_type = _service_type("ordinary one-shot delegation")
    client = _BorrowedClient()
    returned = object()
    calls: list[tuple[object, dict[str, object]]] = []

    def observed(
        received_client: object,
        **kwargs: object,
    ) -> object:
        calls.append((received_client, kwargs))
        return returned

    monkeypatch.setattr(
        module,
        "run_copilot_turn_with_observability",
        observed,
    )
    monkeypatch.setattr(
        module,
        "run_copilot_turn_with_failure_observability",
        _forbidden,
    )
    context = {"experiment_dir": "workspace"}
    metadata = {"nested": ["unchanged"]}
    provider_timeout = object()
    service = service_type(client, model="fixed-model")

    result = service.run(
        "  preserve question  ",
        experiment_context=context,
        turn_timeout_seconds=7.5,
        temperature=0.25,
        timeout=provider_timeout,
        metadata=metadata,
    )

    assert result is returned
    assert calls == [
        (
            client,
            {
                "model": "fixed-model",
                "question": "  preserve question  ",
                "experiment_context": context,
                "turn_timeout_seconds": 7.5,
                "temperature": 0.25,
                "timeout": provider_timeout,
                "metadata": metadata,
            },
        )
    ]
    received = calls[0][1]
    assert received["experiment_context"] is context
    assert received["timeout"] is provider_timeout
    assert received["metadata"] is metadata
    assert "on_failure" not in received
    assert metadata == {"nested": ["unchanged"]}
    assert client.close_count == 0


@requires_service
def test_run_with_callback_selects_failure_observability(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load_service_module("failure callback delegation")
    service_type = _service_type("failure callback delegation")
    client = _BorrowedClient()
    returned = object()
    calls: list[tuple[object, dict[str, object]]] = []

    def callback(observation: CopilotFailureObservation) -> None:
        raise AssertionError(
            f"Service invoked the callback directly: {observation!r}"
        )

    def failure_observed(
        received_client: object,
        **kwargs: object,
    ) -> object:
        calls.append((received_client, kwargs))
        return returned

    monkeypatch.setattr(
        module,
        "run_copilot_turn_with_observability",
        _forbidden,
    )
    monkeypatch.setattr(
        module,
        "run_copilot_turn_with_failure_observability",
        failure_observed,
    )
    service = service_type(client, model="fixed-model")

    result = service.run(
        "question",
        on_failure=callback,
        seed=3,
    )

    assert result is returned
    assert calls == [
        (
            client,
            {
                "model": "fixed-model",
                "question": "question",
                "experiment_context": None,
                "turn_timeout_seconds": None,
                "on_failure": callback,
                "seed": 3,
            },
        )
    ]
    assert calls[0][1]["on_failure"] is callback
    assert client.close_count == 0


@pytest.mark.parametrize(
    "error",
    [
        TypeError("type failure"),
        ValueError("value failure"),
        TimeoutError("deadline failure"),
        RuntimeError("provider failure"),
        KeyboardInterrupt(),
        SystemExit(7),
        GeneratorExit(),
    ],
    ids=[
        "type-error",
        "value-error",
        "timeout-error",
        "provider-error",
        "keyboard-interrupt",
        "system-exit",
        "generator-exit",
    ],
)
@requires_service
def test_run_preserves_exception_identity(
    monkeypatch: pytest.MonkeyPatch,
    error: BaseException,
) -> None:
    module = _load_service_module("exception identity contract")
    service_type = _service_type("exception identity contract")
    client = _BorrowedClient()

    def raise_original(*args: object, **kwargs: object) -> object:
        raise error

    monkeypatch.setattr(
        module,
        "run_copilot_turn_with_observability",
        raise_original,
    )
    monkeypatch.setattr(
        module,
        "run_copilot_turn_with_failure_observability",
        _forbidden,
    )
    service = service_type(client, model="fixed-model")

    with pytest.raises(type(error)) as caught:
        service.run("question")

    assert caught.value is error
    assert client.close_count == 0


@requires_service
def test_experiment_context_is_scoped_to_each_run(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load_service_module("request-scoped context contract")
    service_type = _service_type("request-scoped context contract")
    contexts: list[object] = []

    def observed(client: object, **kwargs: object) -> object:
        contexts.append(kwargs["experiment_context"])
        return object()

    monkeypatch.setattr(
        module,
        "run_copilot_turn_with_observability",
        observed,
    )
    monkeypatch.setattr(
        module,
        "run_copilot_turn_with_failure_observability",
        _forbidden,
    )
    service = service_type(_BorrowedClient(), model="fixed-model")
    first = {"experiment_dir": "first"}
    second = {"experiment_root": "second"}

    service.run("one", experiment_context=first)
    service.run("two", experiment_context=second)

    assert contexts == [first, second]
    assert contexts[0] is first
    assert contexts[1] is second


@requires_service
def test_run_does_not_duplicate_runtime_validation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load_service_module("no duplicate Runtime policy")
    service_type = _service_type("no duplicate Runtime policy")
    returned = object()
    received: list[dict[str, object]] = []

    def observed(client: object, **kwargs: object) -> object:
        received.append(kwargs)
        return returned

    monkeypatch.setattr(
        module,
        "run_copilot_turn_with_observability",
        observed,
    )
    monkeypatch.setattr(
        module,
        "run_copilot_turn_with_failure_observability",
        _forbidden,
    )
    service = service_type(_BorrowedClient(), model="fixed-model")
    invalid_context = object()
    invalid_timeout = object()

    result = service.run(
        "",
        experiment_context=invalid_context,
        turn_timeout_seconds=invalid_timeout,
        tools=object(),
    )

    assert result is returned
    assert received[0]["question"] == ""
    assert received[0]["experiment_context"] is invalid_context
    assert received[0]["turn_timeout_seconds"] is invalid_timeout
    assert "tools" in received[0]


@requires_service
def test_create_session_forwards_exact_configuration(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load_service_module("Session construction contract")
    service_type = _service_type("Session construction contract")
    client = _BorrowedClient()
    fake_session = object()
    calls: list[tuple[object, dict[str, object]]] = []

    def session_constructor(
        received_client: object,
        **kwargs: object,
    ) -> object:
        calls.append((received_client, kwargs))
        return fake_session

    monkeypatch.setattr(module, "CopilotSession", session_constructor)
    service = service_type(client, model="fixed-model")
    context = {"experiment_root": "workspace"}
    metadata = {"nested": []}
    timeout = object()

    result = service.create_session(
        experiment_context=context,
        max_turns=5,
        turn_timeout_seconds=timeout,
        temperature=0.2,
        metadata=metadata,
    )

    assert result is fake_session
    assert calls == [
        (
            client,
            {
                "model": "fixed-model",
                "experiment_context": context,
                "max_turns": 5,
                "turn_timeout_seconds": timeout,
                "temperature": 0.2,
                "metadata": metadata,
            },
        )
    ]
    assert calls[0][1]["experiment_context"] is context
    assert calls[0][1]["turn_timeout_seconds"] is timeout
    assert calls[0][1]["metadata"] is metadata
    assert client.close_count == 0


@requires_service
def test_create_session_does_not_duplicate_session_validation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load_service_module("Session validation ownership")
    service_type = _service_type("Session validation ownership")
    received: list[dict[str, object]] = []
    returned = object()

    def session_constructor(
        client: object,
        **kwargs: object,
    ) -> object:
        received.append(kwargs)
        return returned

    monkeypatch.setattr(module, "CopilotSession", session_constructor)
    service = service_type(_BorrowedClient(), model="fixed-model")
    invalid_context = object()
    invalid_max_turns = object()
    invalid_timeout = object()

    result = service.create_session(
        experiment_context=invalid_context,
        max_turns=invalid_max_turns,
        turn_timeout_seconds=invalid_timeout,
        tools=object(),
    )

    assert result is returned
    assert received[0]["experiment_context"] is invalid_context
    assert received[0]["max_turns"] is invalid_max_turns
    assert received[0]["turn_timeout_seconds"] is invalid_timeout
    assert "tools" in received[0]


@requires_service
def test_each_create_session_call_returns_new_constructor_result(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load_service_module("independent Session contract")
    service_type = _service_type("independent Session contract")
    created: list[object] = []

    def session_constructor(
        client: object,
        **kwargs: object,
    ) -> object:
        session = object()
        created.append(session)
        return session

    monkeypatch.setattr(module, "CopilotSession", session_constructor)
    service = service_type(_BorrowedClient(), model="fixed-model")

    first = service.create_session()
    second = service.create_session()

    assert len(created) == 2
    assert first is created[0]
    assert second is created[1]
    assert first is not second


@requires_service
def test_create_session_does_not_execute_session_actions(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load_service_module("Session state ownership contract")
    service_type = _service_type("Session state ownership contract")

    class FakeSession:
        def ask(self, question: str) -> str:
            raise AssertionError("Service called Session.ask")

        def ask_with_result(self, question: str) -> object:
            raise AssertionError("Service called Session.ask_with_result")

        def reset(self) -> None:
            raise AssertionError("Service called Session.reset")

        def export_history(self) -> list[dict[str, object]]:
            raise AssertionError("Service called Session.export_history")

    fake_session = FakeSession()
    monkeypatch.setattr(
        module,
        "CopilotSession",
        lambda *args, **kwargs: fake_session,
    )
    service = service_type(_BorrowedClient(), model="fixed-model")

    result = service.create_session()

    assert result is fake_session


@requires_service
def test_fixed_model_is_used_for_run_and_session(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load_service_module("fixed model binding contract")
    service_type = _service_type("fixed model binding contract")
    models: list[object] = []

    def observed(client: object, **kwargs: object) -> object:
        models.append(kwargs["model"])
        return object()

    def session_constructor(
        client: object,
        **kwargs: object,
    ) -> object:
        models.append(kwargs["model"])
        return object()

    monkeypatch.setattr(
        module,
        "run_copilot_turn_with_observability",
        observed,
    )
    monkeypatch.setattr(
        module,
        "run_copilot_turn_with_failure_observability",
        _forbidden,
    )
    monkeypatch.setattr(module, "CopilotSession", session_constructor)
    service = service_type(_BorrowedClient(), model="fixed-model")

    service.run("question")
    service.create_session()

    assert models == ["fixed-model", "fixed-model"]


def test_package_root_exports_service_without_replacing_primitives() -> None:
    service_type = getattr(copilot, "CopilotService", None)
    assert service_type is not None, (
        "CopilotService is not exported from the copilot package"
    )
    module = importlib.import_module(MODULE_NAME)
    assert getattr(copilot, "CopilotService", None) is service_type
    assert "CopilotService" in copilot.__all__
    for name in (
        "run_copilot_turn",
        "run_copilot_turn_with_result",
        "run_copilot_turn_with_observability",
        "run_copilot_turn_with_failure_observability",
        "CopilotSession",
    ):
        assert name in copilot.__all__
        assert hasattr(copilot, name)


def test_service_import_is_sdk_free_and_side_effect_free() -> None:
    repository_root = Path(__file__).resolve().parents[1]
    script = r'''
import builtins
import concurrent.futures
import os
from pathlib import Path
import socket
import sys
import threading

sys.dont_write_bytecode = True
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
    raise AssertionError("import side effect")

builtins.__import__ = guarded_import
builtins.open = guarded_open
os.getenv = forbidden
os._Environ.get = forbidden
os._Environ.__getitem__ = forbidden
os.mkdir = forbidden
os.makedirs = forbidden
Path.mkdir = forbidden
Path.touch = forbidden
Path.write_bytes = forbidden
Path.write_text = forbidden
socket.create_connection = forbidden
socket.socket.connect = forbidden
threading.Thread.__init__ = forbidden
concurrent.futures.ThreadPoolExecutor.__init__ = forbidden
concurrent.futures.ProcessPoolExecutor.__init__ = forbidden

import copilot.service
'''
    environment = os.environ.copy()
    environment["PYTHONDONTWRITEBYTECODE"] = "1"

    completed = subprocess.run(
        [sys.executable, "-c", script],
        cwd=repository_root,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    assert completed.stdout == ""
    assert completed.stderr == ""

"""RED contracts for the bounded in-memory Copilot Session repository."""

from __future__ import annotations

from importlib import import_module
from importlib.util import find_spec
import inspect
import os
from pathlib import Path
import subprocess
import sys
from typing import get_type_hints
from uuid import UUID

import pytest

import copilot
from copilot import CopilotService, CopilotSession


MODULE_NAME = "copilot.session_repository"
REPOSITORY_AVAILABLE = find_spec(MODULE_NAME) is not None
requires_repository = pytest.mark.skipif(
    not REPOSITORY_AVAILABLE,
    reason="copilot.session_repository is the missing M7 capability",
)


class _SequenceIdFactory:
    def __init__(self, values: list[object]) -> None:
        self._values = iter(values)
        self.call_count = 0

    def __call__(self) -> str:
        self.call_count += 1
        value = next(self._values)
        if isinstance(value, BaseException):
            raise value
        return value  # type: ignore[return-value]


class _FakeService:
    def __init__(self, outcomes: list[object] | None = None) -> None:
        self._outcomes = iter([] if outcomes is None else outcomes)
        self.calls: list[dict[str, object]] = []
        self.close_count = 0

    def create_session(self, **kwargs: object) -> object:
        self.calls.append(kwargs)
        outcome = next(self._outcomes)
        if isinstance(outcome, BaseException):
            raise outcome
        return outcome

    def close(self) -> None:
        self.close_count += 1


class _GuardedSession:
    def __init__(self) -> None:
        self.reset_count = 0
        self.close_count = 0

    def reset(self) -> None:
        self.reset_count += 1
        raise AssertionError("Repository reset the stored Session")

    def close(self) -> None:
        self.close_count += 1
        raise AssertionError("Repository closed the stored Session")


def _repository_type(contract: str) -> type:
    try:
        module = import_module(MODULE_NAME)
    except ModuleNotFoundError as error:
        if error.name == MODULE_NAME:
            pytest.fail(
                f"{contract}: copilot.session_repository is missing"
            )
        raise
    repository_type = getattr(module, "CopilotSessionRepository", None)
    assert isinstance(repository_type, type), (
        f"{contract}: CopilotSessionRepository is missing"
    )
    return repository_type


def _new_repository(
    service: object,
    *,
    max_sessions: object,
    id_factory: object = None,
) -> object:
    repository_type = _repository_type("repository construction")
    return repository_type(
        service,
        max_sessions=max_sessions,
        id_factory=id_factory,
    )


def test_existing_session_application_api_remains_public() -> None:
    expected = {
        "CopilotService",
        "CopilotSession",
        "CopilotObservedResult",
        "CopilotFailureObservation",
        "run_copilot_turn",
        "run_copilot_turn_with_result",
        "run_copilot_turn_with_observability",
        "run_copilot_turn_with_failure_observability",
    }
    assert expected <= set(copilot.__all__)
    assert all(hasattr(copilot, name) for name in expected)


def test_existing_service_and_session_signatures_remain_compatible() -> None:
    assert list(inspect.signature(CopilotService).parameters) == [
        "client",
        "model",
    ]
    assert list(inspect.signature(CopilotService.create_session).parameters) == [
        "self",
        "experiment_context",
        "max_turns",
        "turn_timeout_seconds",
        "request_options",
    ]
    assert list(inspect.signature(CopilotSession).parameters) == [
        "client",
        "model",
        "experiment_context",
        "max_turns",
        "turn_timeout_seconds",
        "request_options",
    ]


def test_repository_module_is_available() -> None:
    assert REPOSITORY_AVAILABLE, (
        "copilot.session_repository is not implemented"
    )


def test_package_root_exports_repository() -> None:
    assert hasattr(copilot, "CopilotSessionRepository"), (
        "CopilotSessionRepository is not exported from copilot"
    )
    assert "CopilotSessionRepository" in copilot.__all__


@requires_repository
def test_repository_has_exact_public_signatures() -> None:
    repository_type = _repository_type("public signature contract")

    constructor = list(
        inspect.signature(repository_type.__init__).parameters.values()
    )
    assert [item.name for item in constructor] == [
        "self",
        "service",
        "max_sessions",
        "id_factory",
    ]
    assert constructor[1].kind is inspect.Parameter.POSITIONAL_OR_KEYWORD
    assert constructor[2].kind is inspect.Parameter.KEYWORD_ONLY
    assert constructor[2].default is inspect.Parameter.empty
    assert constructor[3].kind is inspect.Parameter.KEYWORD_ONLY
    assert constructor[3].default is None

    create_parameters = list(
        inspect.signature(repository_type.create).parameters.values()
    )
    assert [item.name for item in create_parameters] == [
        "self",
        "experiment_context",
        "max_turns",
        "turn_timeout_seconds",
        "request_options",
    ]
    for item in create_parameters[1:4]:
        assert item.kind is inspect.Parameter.KEYWORD_ONLY
    assert create_parameters[1].default is None
    assert create_parameters[2].default == 8
    assert create_parameters[3].default is None
    assert create_parameters[4].kind is inspect.Parameter.VAR_KEYWORD

    for method_name in ("get", "delete"):
        parameters = list(
            inspect.signature(
                getattr(repository_type, method_name)
            ).parameters.values()
        )
        assert [item.name for item in parameters] == [
            "self",
            "session_id",
        ]
        assert parameters[1].kind is (
            inspect.Parameter.POSITIONAL_OR_KEYWORD
        )

    constructor_hints = get_type_hints(repository_type.__init__)
    create_hints = get_type_hints(repository_type.create)
    get_hints = get_type_hints(repository_type.get)
    delete_hints = get_type_hints(repository_type.delete)
    assert constructor_hints["service"] is CopilotService
    assert constructor_hints["max_sessions"] is int
    assert create_hints["return"] is str
    assert get_hints["session_id"] is str
    assert get_hints["return"] is CopilotSession
    assert delete_hints["session_id"] is str
    assert delete_hints["return"] is type(None)


@requires_repository
def test_repository_has_no_scope_creep_or_ownership_api() -> None:
    repository_type = _repository_type("minimal public surface")
    for name in (
        "list_ids",
        "items",
        "values",
        "clear",
        "ask",
        "ask_with_result",
        "reset",
        "export_history",
        "close",
        "__enter__",
        "__exit__",
        "__aenter__",
        "__aexit__",
    ):
        assert not hasattr(repository_type, name)


@requires_repository
def test_positive_max_sessions_is_accepted_without_side_effects() -> None:
    for capacity in (1, 2, 100):
        service = _FakeService()
        factory = _SequenceIdFactory([])

        _new_repository(
            service,
            max_sessions=capacity,
            id_factory=factory,
        )

        assert service.calls == []
        assert service.close_count == 0
        assert factory.call_count == 0


@pytest.mark.parametrize(
    "invalid",
    [True, False, 1.5, "2", None, object()],
)
@requires_repository
def test_max_sessions_rejects_non_integer_values(
    invalid: object,
) -> None:
    service = _FakeService()
    factory = _SequenceIdFactory([])

    with pytest.raises(TypeError):
        _new_repository(
            service,
            max_sessions=invalid,
            id_factory=factory,
        )

    assert service.calls == []
    assert service.close_count == 0
    assert factory.call_count == 0


@pytest.mark.parametrize("invalid", [0, -1, -100])
@requires_repository
def test_max_sessions_requires_a_positive_value(invalid: int) -> None:
    service = _FakeService()
    factory = _SequenceIdFactory([])

    with pytest.raises(ValueError):
        _new_repository(
            service,
            max_sessions=invalid,
            id_factory=factory,
        )

    assert service.calls == []
    assert service.close_count == 0
    assert factory.call_count == 0


@pytest.mark.parametrize("invalid", [object(), "factory", 42])
@requires_repository
def test_id_factory_must_be_callable_or_none(invalid: object) -> None:
    service = _FakeService()

    with pytest.raises(TypeError):
        _new_repository(
            service,
            max_sessions=1,
            id_factory=invalid,
        )

    assert service.calls == []
    assert service.close_count == 0


@requires_repository
def test_construction_does_not_invoke_factory_or_service() -> None:
    service = _FakeService()
    factory = _SequenceIdFactory(
        [AssertionError("ID generated during construction")]
    )

    _new_repository(
        service,
        max_sessions=1,
        id_factory=factory,
    )

    assert factory.call_count == 0
    assert service.calls == []
    assert service.close_count == 0


@requires_repository
def test_create_forwards_configuration_exactly_once_without_copying() -> None:
    session = object()
    service = _FakeService([session])
    factory = _SequenceIdFactory(["session-1"])
    repository = _new_repository(
        service,
        max_sessions=1,
        id_factory=factory,
    )
    context = object()
    max_turns = object()
    timeout = object()
    nested_option = {"nested": ["preserve identity"]}

    session_id = repository.create(
        experiment_context=context,
        max_turns=max_turns,
        turn_timeout_seconds=timeout,
        temperature=0.25,
        nested_option=nested_option,
    )

    assert session_id == "session-1"
    assert len(service.calls) == 1
    received = service.calls[0]
    assert received == {
        "experiment_context": context,
        "max_turns": max_turns,
        "turn_timeout_seconds": timeout,
        "temperature": 0.25,
        "nested_option": nested_option,
    }
    assert received["experiment_context"] is context
    assert received["max_turns"] is max_turns
    assert received["turn_timeout_seconds"] is timeout
    assert received["nested_option"] is nested_option
    assert repository.get(session_id) is session
    assert factory.call_count == 1


@requires_repository
def test_create_stores_exact_service_result_and_returns_only_id() -> None:
    session = object()
    service = _FakeService([session])
    repository = _new_repository(
        service,
        max_sessions=1,
        id_factory=_SequenceIdFactory(["session-123"]),
    )

    returned = repository.create()

    assert returned == "session-123"
    assert type(returned) is str
    assert repository.get(returned) is session


@requires_repository
def test_default_factory_returns_a_uuid4_based_string() -> None:
    session = object()
    repository = _new_repository(
        _FakeService([session]),
        max_sessions=1,
    )

    session_id = repository.create()

    assert isinstance(session_id, str)
    assert session_id.strip()
    assert UUID(session_id).version == 4
    assert repository.get(session_id) is session


@requires_repository
def test_memberships_are_independent_and_delete_isolated() -> None:
    first_session = object()
    second_session = object()
    repository = _new_repository(
        _FakeService([first_session, second_session]),
        max_sessions=2,
        id_factory=_SequenceIdFactory(["session-1", "session-2"]),
    )

    first_id = repository.create()
    second_id = repository.create()

    assert first_id != second_id
    assert repository.get(first_id) is first_session
    assert repository.get(second_id) is second_session
    assert repository.delete(first_id) is None
    with pytest.raises(KeyError):
        repository.get(first_id)
    assert repository.get(second_id) is second_session


@requires_repository
def test_full_capacity_fails_before_id_generation_and_service_call() -> None:
    existing = object()
    factory = _SequenceIdFactory(["existing", "forbidden-id"])
    service = _FakeService([existing, object()])
    repository = _new_repository(
        service,
        max_sessions=1,
        id_factory=factory,
    )
    existing_id = repository.create()

    with pytest.raises(RuntimeError):
        repository.create()

    assert factory.call_count == 1
    assert len(service.calls) == 1
    assert repository.get(existing_id) is existing


@requires_repository
def test_capacity_accepts_exact_boundary_then_rejects_without_eviction() -> None:
    first = object()
    second = object()
    service = _FakeService([first, second, object()])
    factory = _SequenceIdFactory(["one", "two", "three"])
    repository = _new_repository(
        service,
        max_sessions=2,
        id_factory=factory,
    )

    first_id = repository.create()
    second_id = repository.create()
    with pytest.raises(RuntimeError):
        repository.create()

    assert factory.call_count == 2
    assert len(service.calls) == 2
    assert repository.get(first_id) is first
    assert repository.get(second_id) is second


@requires_repository
def test_delete_releases_capacity() -> None:
    first = object()
    second = object()
    service = _FakeService([first, second])
    factory = _SequenceIdFactory(["one", "two"])
    repository = _new_repository(
        service,
        max_sessions=1,
        id_factory=factory,
    )

    first_id = repository.create()
    repository.delete(first_id)
    second_id = repository.create()

    assert second_id == "two"
    assert repository.get(second_id) is second
    with pytest.raises(KeyError):
        repository.get(first_id)


@requires_repository
def test_valid_opaque_id_is_not_normalized() -> None:
    session = object()
    repository = _new_repository(
        _FakeService([session]),
        max_sessions=1,
        id_factory=_SequenceIdFactory([" session "]),
    )

    session_id = repository.create()

    assert session_id == " session "
    assert repository.get(" session ") is session


@pytest.mark.parametrize("invalid", [123, None, object()])
@requires_repository
def test_generated_id_must_be_a_string(invalid: object) -> None:
    service = _FakeService([object()])
    repository = _new_repository(
        service,
        max_sessions=1,
        id_factory=_SequenceIdFactory([invalid]),
    )

    with pytest.raises(TypeError):
        repository.create()

    assert service.calls == []


@pytest.mark.parametrize("invalid", ["", "   ", "\t\n"])
@requires_repository
def test_generated_id_must_not_be_empty_or_whitespace(
    invalid: str,
) -> None:
    service = _FakeService([object()])
    repository = _new_repository(
        service,
        max_sessions=1,
        id_factory=_SequenceIdFactory([invalid]),
    )

    with pytest.raises(ValueError):
        repository.create()

    assert service.calls == []


@requires_repository
def test_collision_is_deterministic_without_retry_or_overwrite() -> None:
    existing = object()
    replacement = object()
    factory = _SequenceIdFactory(["same-id", "same-id", "unused"])
    service = _FakeService([existing, replacement])
    repository = _new_repository(
        service,
        max_sessions=2,
        id_factory=factory,
    )
    repository.create()

    with pytest.raises(RuntimeError):
        repository.create()

    assert factory.call_count == 2
    assert len(service.calls) == 1
    assert repository.get("same-id") is existing


@pytest.mark.parametrize(
    "error",
    [
        RuntimeError("factory"),
        KeyboardInterrupt(),
        SystemExit(7),
        GeneratorExit(),
    ],
)
@requires_repository
def test_id_factory_failure_preserves_identity_and_membership(
    error: BaseException,
) -> None:
    existing = object()
    factory = _SequenceIdFactory(["existing", error])
    service = _FakeService([existing, object()])
    repository = _new_repository(
        service,
        max_sessions=2,
        id_factory=factory,
    )
    repository.create()

    with pytest.raises(type(error)) as caught:
        repository.create()

    assert caught.value is error
    assert len(service.calls) == 1
    assert repository.get("existing") is existing


@pytest.mark.parametrize(
    "error",
    [
        TypeError("service"),
        ValueError("service"),
        RuntimeError("provider-like"),
        KeyboardInterrupt(),
        SystemExit(9),
        GeneratorExit(),
    ],
)
@requires_repository
def test_service_failure_preserves_identity_and_membership(
    error: BaseException,
) -> None:
    existing = object()
    service = _FakeService([existing, error])
    repository = _new_repository(
        service,
        max_sessions=2,
        id_factory=_SequenceIdFactory(["existing", "failed-id"]),
    )
    repository.create()

    with pytest.raises(type(error)) as caught:
        repository.create()

    assert caught.value is error
    assert repository.get("existing") is existing
    with pytest.raises(KeyError):
        repository.get("failed-id")


@requires_repository
def test_invalid_generated_id_does_not_mutate_existing_membership() -> None:
    existing = object()
    service = _FakeService([existing, object()])
    repository = _new_repository(
        service,
        max_sessions=2,
        id_factory=_SequenceIdFactory(["existing", ""]),
    )
    repository.create()

    with pytest.raises(ValueError):
        repository.create()

    assert len(service.calls) == 1
    assert repository.get("existing") is existing


@requires_repository
def test_get_missing_id_uses_native_key_error() -> None:
    repository = _new_repository(
        _FakeService(),
        max_sessions=1,
        id_factory=_SequenceIdFactory([]),
    )

    with pytest.raises(KeyError):
        repository.get("missing")


@requires_repository
def test_delete_only_removes_membership() -> None:
    session = _GuardedSession()
    service = _FakeService([session])
    repository = _new_repository(
        service,
        max_sessions=1,
        id_factory=_SequenceIdFactory(["session"]),
    )
    session_id = repository.create()

    assert repository.delete(session_id) is None

    with pytest.raises(KeyError):
        repository.get(session_id)
    with pytest.raises(KeyError):
        repository.delete(session_id)
    assert session.reset_count == 0
    assert session.close_count == 0
    assert service.close_count == 0


@requires_repository
def test_borrowed_service_is_never_closed_across_repository_actions() -> None:
    successful = object()
    failure = RuntimeError("failed create")
    service = _FakeService([successful, failure])
    repository = _new_repository(
        service,
        max_sessions=2,
        id_factory=_SequenceIdFactory(["successful", "failed"]),
    )

    successful_id = repository.create()
    assert repository.get(successful_id) is successful
    with pytest.raises(RuntimeError) as caught:
        repository.create()
    assert caught.value is failure
    repository.delete(successful_id)

    assert service.close_count == 0


@requires_repository
def test_repository_import_is_sdk_free_and_side_effect_free() -> None:
    repository_root = Path(__file__).resolve().parents[1]
    script = r'''
import builtins
import concurrent.futures
import os
from pathlib import Path
import socket
import sys
import threading
import uuid

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
uuid.uuid4 = forbidden

import copilot.session_repository
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

import ast
import builtins
from decimal import Decimal
from fractions import Fraction
import importlib
import inspect
import logging
import os
from pathlib import Path
import socket
import subprocess
import sys
import threading
import time
from types import ModuleType
import urllib.request

import pytest

import copilot.runtime as runtime
from copilot import run_copilot_turn
import llm_clients
from llm_clients import create_openai_client


class _AttributeObject:
    def __init__(self, **attributes: object) -> None:
        for name, value in attributes.items():
            setattr(self, name, value)


class _NoToolCompletions:
    def __init__(self, content: str = "factory client answer") -> None:
        self.content = content
        self.calls: list[
            tuple[tuple[object, ...], dict[str, object]]
        ] = []

    def create(
        self,
        *args: object,
        **kwargs: object,
    ) -> _AttributeObject:
        self.calls.append((args, kwargs))
        message = _AttributeObject(
            role="assistant",
            content=self.content,
            tool_calls=None,
        )
        return _AttributeObject(
            choices=[_AttributeObject(message=message)]
        )


class _AttributeClient:
    def __init__(
        self,
        identifier: int,
        *,
        content: str = "factory client answer",
    ) -> None:
        self.identifier = identifier
        self.completions = _NoToolCompletions(content)
        self.chat = _AttributeObject(completions=self.completions)
        self.responses = _ForbiddenClientAttribute("responses")
        self.models = _ForbiddenClientAttribute("models")
        self.close_calls = 0

    def close(self) -> None:
        self.close_calls += 1
        raise AssertionError("factory must not close the created client")


class _ForbiddenClientAttribute:
    def __init__(self, name: str) -> None:
        self.name = name

    def __getattr__(self, attribute: str) -> object:
        raise AssertionError(
            f"factory must not access client.{self.name}.{attribute}"
        )


class _ConstructorError(Exception):
    pass


class _FakeOpenAIConstructor:
    def __init__(self) -> None:
        self.calls: list[
            tuple[tuple[object, ...], dict[str, object]]
        ] = []
        self.outcomes: list[object] = []

    @property
    def call_count(self) -> int:
        return len(self.calls)

    def __call__(
        self,
        *args: object,
        **kwargs: object,
    ) -> object:
        self.calls.append((args, kwargs))
        if self.outcomes:
            outcome = self.outcomes.pop(0)
            if isinstance(outcome, BaseException):
                raise outcome
            return outcome
        return _AttributeClient(self.call_count)


@pytest.fixture
def fake_openai(
    monkeypatch: pytest.MonkeyPatch,
) -> _FakeOpenAIConstructor:
    constructor = _FakeOpenAIConstructor()
    module = ModuleType("openai")
    module.OpenAI = constructor  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "openai", module)
    return constructor


def _raise_assertion(message: str):
    def fail(*args: object, **kwargs: object) -> object:
        raise AssertionError(message)

    return fail


def _imported_roots(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    roots: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            roots.update(alias.name.split(".", 1)[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            roots.add(node.module.split(".", 1)[0])
    return roots


def _call_kwargs(
    constructor: _FakeOpenAIConstructor,
) -> dict[str, object]:
    assert constructor.call_count == 1
    args, kwargs = constructor.calls[0]
    assert args == ()
    return kwargs


def test_package_publicly_exports_only_create_openai_client() -> None:
    assert llm_clients.__all__ == ["create_openai_client"]
    assert llm_clients.create_openai_client is create_openai_client
    public_names = {
        name for name in vars(llm_clients) if not name.startswith("_")
    }
    assert public_names == {"create_openai_client"}


def test_create_openai_client_has_exact_keyword_only_signature() -> None:
    signature = inspect.signature(create_openai_client)

    assert str(signature) == (
        "(*, api_key: str | None = None, "
        "base_url: str | None = None, "
        "timeout: float | None = None) -> object"
    )
    assert all(
        parameter.kind is inspect.Parameter.KEYWORD_ONLY
        for parameter in signature.parameters.values()
    )
    assert all(
        parameter.default is None
        for parameter in signature.parameters.values()
    )
    assert "max_retries" not in signature.parameters


@pytest.mark.parametrize(
    ("args", "kwargs"),
    [
        (("explicit-key",), {}),
        ((), {"max_retries": 1}),
        ((), {"organization": "org"}),
        ((), {"project": "project"}),
        ((), {"client_options": {}}),
    ],
)
def test_factory_rejects_positional_or_unapproved_options(
    args: tuple[object, ...],
    kwargs: dict[str, object],
) -> None:
    with pytest.raises(TypeError):
        create_openai_client(*args, **kwargs)


def test_factory_returns_the_exact_constructor_result(
    fake_openai: _FakeOpenAIConstructor,
) -> None:
    expected = _AttributeClient(99)
    fake_openai.outcomes.append(expected)

    result = create_openai_client(api_key="explicit-key")

    assert result is expected
    assert _call_kwargs(fake_openai) == {
        "api_key": "explicit-key",
        "max_retries": 0,
    }


def test_package_exposes_no_public_helper_class_or_configuration() -> None:
    public_values = {
        name: value
        for name, value in vars(llm_clients).items()
        if not name.startswith("_")
    }

    assert public_values == {
        "create_openai_client": create_openai_client
    }
    assert not any(
        inspect.isclass(value)
        for value in public_values.values()
    )


def test_package_reload_is_idempotent_and_keeps_public_namespace_clean(
) -> None:
    for _ in range(2):
        reloaded_package = importlib.reload(llm_clients)
        public_names = {
            name
            for name in vars(reloaded_package)
            if not name.startswith("_")
        }

        assert reloaded_package.__all__ == ["create_openai_client"]
        assert public_names == {"create_openai_client"}
        assert callable(reloaded_package.create_openai_client)
        assert not hasattr(
            reloaded_package,
            "openai_client_factory",
        )


def test_llm_clients_import_is_sdk_free_and_has_no_side_effects(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original_import = builtins.__import__

    def guarded_import(
        name: str,
        globals: dict | None = None,
        locals: dict | None = None,
        fromlist: tuple | list = (),
        level: int = 0,
    ) -> object:
        if name.split(".", 1)[0] in {
            "openai",
            "dotenv",
            "requests",
            "httpx",
        }:
            raise AssertionError(
                f"import must not load optional package {name!r}"
            )
        return original_import(name, globals, locals, fromlist, level)

    fail_environment = _raise_assertion(
        "import must not read environment variables"
    )
    fail_network = _raise_assertion("import must not access the network")
    fail_process = _raise_assertion("import must not run subprocesses")
    fail_write = _raise_assertion("import must not write files")

    with monkeypatch.context() as context:
        context.delitem(sys.modules, "llm_clients", raising=False)
        context.delitem(
            sys.modules,
            "llm_clients.openai_client_factory",
            raising=False,
        )
        context.setattr(builtins, "__import__", guarded_import)
        context.setattr(os, "getenv", fail_environment)
        context.setattr(type(os.environ), "get", fail_environment)
        context.setattr(
            type(os.environ),
            "__getitem__",
            fail_environment,
        )
        context.setattr(socket, "create_connection", fail_network)
        context.setattr(socket.socket, "connect", fail_network)
        context.setattr(urllib.request, "urlopen", fail_network)
        context.setattr(subprocess, "run", fail_process)
        context.setattr(subprocess, "Popen", fail_process)
        context.setattr(Path, "write_text", fail_write)
        context.setattr(Path, "write_bytes", fail_write)

        imported = importlib.import_module("llm_clients")

    assert callable(imported.create_openai_client)
    assert imported.__all__ == ["create_openai_client"]


def test_imported_factory_module_has_no_client_singleton_or_cache() -> None:
    module = importlib.import_module(
        "llm_clients.openai_client_factory"
    )
    forbidden_names = {
        "client",
        "_client",
        "cached_client",
        "_cached_client",
        "client_cache",
        "_client_cache",
    }

    assert forbidden_names.isdisjoint(vars(module))


def test_missing_openai_package_raises_safe_import_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original_import = builtins.__import__
    missing = ModuleNotFoundError(
        "No module named 'openai'",
        name="openai",
    )

    def missing_openai(
        name: str,
        globals: dict | None = None,
        locals: dict | None = None,
        fromlist: tuple | list = (),
        level: int = 0,
    ) -> object:
        if name.split(".", 1)[0] == "openai":
            raise missing
        return original_import(name, globals, locals, fromlist, level)

    with monkeypatch.context() as context:
        context.delitem(sys.modules, "openai", raising=False)
        context.setattr(builtins, "__import__", missing_openai)
        with pytest.raises(ImportError) as error_info:
            create_openai_client(api_key="never-report-this-key")

    message = str(error_info.value)
    assert "OpenAI SDK is required" in message
    assert "requirements-openai.txt" in message
    assert "never-report-this-key" not in message
    assert error_info.value.__cause__ is missing
    assert error_info.value.__cause__.name == "openai"


def test_sdk_internal_module_not_found_error_is_not_translated(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original_import = builtins.__import__
    error = ModuleNotFoundError(
        "No module named 'broken_sdk_dependency'",
        name="broken_sdk_dependency",
    )

    def broken_sdk(
        name: str,
        globals: dict | None = None,
        locals: dict | None = None,
        fromlist: tuple | list = (),
        level: int = 0,
    ) -> object:
        if name.split(".", 1)[0] == "openai":
            raise error
        return original_import(name, globals, locals, fromlist, level)

    with monkeypatch.context() as context:
        context.delitem(sys.modules, "openai", raising=False)
        context.setattr(builtins, "__import__", broken_sdk)
        with pytest.raises(ModuleNotFoundError) as error_info:
            create_openai_client(api_key="explicit-key")

    assert error_info.value is error


def test_sdk_internal_import_error_is_not_translated(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original_import = builtins.__import__
    error = ImportError("broken OpenAI SDK installation")

    def broken_sdk(
        name: str,
        globals: dict | None = None,
        locals: dict | None = None,
        fromlist: tuple | list = (),
        level: int = 0,
    ) -> object:
        if name.split(".", 1)[0] == "openai":
            raise error
        return original_import(name, globals, locals, fromlist, level)

    with monkeypatch.context() as context:
        context.delitem(sys.modules, "openai", raising=False)
        context.setattr(builtins, "__import__", broken_sdk)
        with pytest.raises(ImportError) as error_info:
            create_openai_client(api_key="explicit-key")

    assert error_info.value is error


@pytest.mark.parametrize(
    "api_key",
    [
        "explicit-key",
        "provider-compatible-token",
        "  preserve-surrounding-space  ",
        "兼容提供商密钥\nvalue",
    ],
)
def test_explicit_api_key_is_passed_unchanged(
    api_key: str,
    fake_openai: _FakeOpenAIConstructor,
) -> None:
    create_openai_client(api_key=api_key)

    assert _call_kwargs(fake_openai)["api_key"] is api_key


@pytest.mark.parametrize("api_key", ["", " ", "\t\r\n"])
def test_blank_explicit_api_key_is_rejected(
    api_key: str,
    fake_openai: _FakeOpenAIConstructor,
) -> None:
    with pytest.raises(
        ValueError,
        match=(
            "api_key must be provided explicitly or through "
            "OPENAI_API_KEY"
        ),
    ):
        create_openai_client(api_key=api_key)

    assert fake_openai.calls == []


def test_non_string_explicit_api_key_is_rejected(
    fake_openai: _FakeOpenAIConstructor,
) -> None:
    invalid_values = [
        1,
        1.5,
        True,
        b"key",
        bytearray(b"key"),
        ["key"],
        ("key",),
        {"key": "value"},
        object(),
    ]

    for api_key in invalid_values:
        with pytest.raises(
            TypeError,
            match="api_key must be a string or None",
        ):
            create_openai_client(
                api_key=api_key,  # type: ignore[arg-type]
            )

    assert fake_openai.calls == []


@pytest.mark.parametrize("api_key", ["", "   "])
def test_blank_explicit_key_never_falls_back_to_environment(
    api_key: str,
    monkeypatch: pytest.MonkeyPatch,
    fake_openai: _FakeOpenAIConstructor,
) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "environment-key")

    with pytest.raises(ValueError):
        create_openai_client(api_key=api_key)

    assert fake_openai.calls == []


def test_explicit_key_does_not_read_any_environment_variable(
    monkeypatch: pytest.MonkeyPatch,
    fake_openai: _FakeOpenAIConstructor,
) -> None:
    fail_environment = _raise_assertion(
        "explicit key must prevent environment access"
    )

    with monkeypatch.context() as context:
        context.setattr(os, "getenv", fail_environment)
        context.setattr(type(os.environ), "get", fail_environment)
        context.setattr(
            type(os.environ),
            "__getitem__",
            fail_environment,
        )
        create_openai_client(api_key="explicit-key")

    assert _call_kwargs(fake_openai)["api_key"] == "explicit-key"


@pytest.mark.parametrize(
    "environment_key",
    [
        "environment-key",
        "provider-key-without-prefix",
        "  环境密钥保持原样  ",
    ],
)
def test_environment_api_key_is_used_unchanged(
    environment_key: str,
    monkeypatch: pytest.MonkeyPatch,
    fake_openai: _FakeOpenAIConstructor,
) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", environment_key)

    create_openai_client()

    assert _call_kwargs(fake_openai)["api_key"] is environment_key


@pytest.mark.parametrize(
    "environment_key",
    [None, "", " ", "\t\r\n"],
)
def test_missing_or_blank_environment_api_key_is_rejected(
    environment_key: str | None,
    monkeypatch: pytest.MonkeyPatch,
    fake_openai: _FakeOpenAIConstructor,
) -> None:
    if environment_key is None:
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    else:
        monkeypatch.setenv("OPENAI_API_KEY", environment_key)

    with pytest.raises(
        ValueError,
        match=(
            "api_key must be provided explicitly or through "
            "OPENAI_API_KEY"
        ),
    ):
        create_openai_client()

    assert fake_openai.calls == []


def test_factory_reads_only_openai_api_key_and_preserves_environment(
    monkeypatch: pytest.MonkeyPatch,
    fake_openai: _FakeOpenAIConstructor,
) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "environment-key")
    monkeypatch.setenv("OPENAI_BASE_URL", "never-read")
    monkeypatch.setenv("OPENAI_ORG_ID", "never-read")
    monkeypatch.setenv("OPENAI_PROJECT", "never-read")
    monkeypatch.setenv("AZURE_OPENAI_API_KEY", "never-read")
    before = dict(os.environ)
    accessed: list[str] = []
    environ_type = type(os.environ)
    original_get = environ_type.get
    original_getitem = environ_type.__getitem__

    def tracked_get(
        environ: object,
        key: str,
        default: object = None,
    ) -> object:
        accessed.append(key)
        return original_get(environ, key, default)

    def tracked_getitem(environ: object, key: str) -> str:
        accessed.append(key)
        return original_getitem(environ, key)

    with monkeypatch.context() as context:
        context.setattr(environ_type, "get", tracked_get)
        context.setattr(environ_type, "__getitem__", tracked_getitem)
        create_openai_client()

    assert accessed
    assert set(accessed) == {"OPENAI_API_KEY"}
    assert dict(os.environ) == before
    assert _call_kwargs(fake_openai)["api_key"] == "environment-key"


@pytest.mark.parametrize(
    "base_url",
    [
        "https://api.example.test/v1",
        "http://localhost:8000/custom",
        "https://例子.测试/模型",
        "provider://compatible/path",
        "https://example.test/v1/",
        "https://example.test/v1///",
        "  HTTPS://Example.Test/Path?A=B  ",
    ],
)
def test_valid_base_url_is_passed_unchanged(
    base_url: str,
    fake_openai: _FakeOpenAIConstructor,
) -> None:
    create_openai_client(
        api_key="explicit-key",
        base_url=base_url,
    )

    assert _call_kwargs(fake_openai)["base_url"] is base_url


def test_none_base_url_is_omitted(
    fake_openai: _FakeOpenAIConstructor,
) -> None:
    create_openai_client(
        api_key="explicit-key",
        base_url=None,
    )

    assert "base_url" not in _call_kwargs(fake_openai)


@pytest.mark.parametrize("base_url", ["", " ", "\t\r\n"])
def test_blank_base_url_is_rejected(
    base_url: str,
    fake_openai: _FakeOpenAIConstructor,
) -> None:
    with pytest.raises(ValueError):
        create_openai_client(
            api_key="explicit-key",
            base_url=base_url,
        )

    assert fake_openai.calls == []


def test_non_string_base_url_is_rejected(
    fake_openai: _FakeOpenAIConstructor,
) -> None:
    invalid_values = [
        1,
        1.5,
        True,
        b"url",
        ["url"],
        {"url": "value"},
        Path("url"),
    ]

    for base_url in invalid_values:
        with pytest.raises(TypeError):
            create_openai_client(
                api_key="explicit-key",
                base_url=base_url,  # type: ignore[arg-type]
            )

    assert fake_openai.calls == []


def test_factory_does_not_read_openai_base_url(
    monkeypatch: pytest.MonkeyPatch,
    fake_openai: _FakeOpenAIConstructor,
) -> None:
    monkeypatch.setenv(
        "OPENAI_BASE_URL",
        "https://must-not-be-read.test",
    )

    create_openai_client(api_key="explicit-key")

    assert "base_url" not in _call_kwargs(fake_openai)


@pytest.mark.parametrize("timeout", [1, 30, 0.25, 5e-324])
def test_positive_finite_timeout_is_passed_as_float(
    timeout: int | float,
    fake_openai: _FakeOpenAIConstructor,
) -> None:
    create_openai_client(
        api_key="explicit-key",
        timeout=timeout,
    )

    passed = _call_kwargs(fake_openai)["timeout"]
    assert type(passed) is float
    assert passed == float(timeout)


def test_none_timeout_is_omitted(
    fake_openai: _FakeOpenAIConstructor,
) -> None:
    create_openai_client(
        api_key="explicit-key",
        timeout=None,
    )

    assert "timeout" not in _call_kwargs(fake_openai)


def test_invalid_timeout_type_is_rejected(
    fake_openai: _FakeOpenAIConstructor,
) -> None:
    invalid_values = [
        True,
        False,
        "1",
        Decimal("1"),
        Fraction(1, 2),
        b"1",
        [1],
        {"seconds": 1},
        1 + 0j,
    ]

    for timeout in invalid_values:
        with pytest.raises(TypeError):
            create_openai_client(
                api_key="explicit-key",
                timeout=timeout,  # type: ignore[arg-type]
            )

    assert fake_openai.calls == []


@pytest.mark.parametrize(
    "timeout",
    [0, 0.0, -1, -0.5, float("nan"), float("inf"), float("-inf")],
)
def test_non_positive_or_non_finite_timeout_is_rejected(
    timeout: int | float,
    fake_openai: _FakeOpenAIConstructor,
) -> None:
    with pytest.raises(ValueError):
        create_openai_client(
            api_key="explicit-key",
            timeout=timeout,
        )

    assert fake_openai.calls == []


@pytest.mark.parametrize(
    ("base_url", "timeout", "expected"),
    [
        (
            None,
            None,
            {"api_key": "key", "max_retries": 0},
        ),
        (
            "https://example.test/",
            None,
            {
                "api_key": "key",
                "base_url": "https://example.test/",
                "max_retries": 0,
            },
        ),
        (
            None,
            3,
            {
                "api_key": "key",
                "timeout": 3.0,
                "max_retries": 0,
            },
        ),
        (
            "local://endpoint",
            0.5,
            {
                "api_key": "key",
                "base_url": "local://endpoint",
                "timeout": 0.5,
                "max_retries": 0,
            },
        ),
    ],
)
def test_constructor_receives_only_approved_kwargs(
    base_url: str | None,
    timeout: int | float | None,
    expected: dict[str, object],
    fake_openai: _FakeOpenAIConstructor,
) -> None:
    create_openai_client(
        api_key="key",
        base_url=base_url,
        timeout=timeout,
    )

    assert _call_kwargs(fake_openai) == expected
    assert type(expected["max_retries"]) is int


def test_factory_constructs_once_without_using_or_closing_client(
    fake_openai: _FakeOpenAIConstructor,
) -> None:
    client = _AttributeClient(1)
    fake_openai.outcomes.append(client)

    result = create_openai_client(api_key="key")

    assert result is client
    assert fake_openai.call_count == 1
    assert client.completions.calls == []
    assert client.close_calls == 0


def test_each_factory_call_creates_a_distinct_uncached_client(
    fake_openai: _FakeOpenAIConstructor,
) -> None:
    first = create_openai_client(api_key="key")
    second = create_openai_client(api_key="key")

    assert first is not second
    assert fake_openai.call_count == 2


@pytest.mark.parametrize(
    "error",
    [
        TypeError("constructor type error"),
        ValueError("constructor value error"),
        RuntimeError("constructor runtime error"),
        _ConstructorError("custom constructor error"),
    ],
)
def test_constructor_exceptions_propagate_by_identity_without_retry(
    error: Exception,
    fake_openai: _FakeOpenAIConstructor,
) -> None:
    fake_openai.outcomes.append(error)

    with pytest.raises(type(error)) as error_info:
        create_openai_client(api_key="secret-not-in-context")

    assert error_info.value is error
    assert error_info.value.args == error.args
    assert fake_openai.call_count == 1
    assert "secret-not-in-context" not in str(error_info.value)


def test_constructor_exception_preserves_existing_cause(
    fake_openai: _FakeOpenAIConstructor,
) -> None:
    cause = RuntimeError("original cause")
    error = _ConstructorError("constructor failed")
    error.__cause__ = cause
    fake_openai.outcomes.append(error)

    with pytest.raises(_ConstructorError) as error_info:
        create_openai_client(api_key="key")

    assert error_info.value is error
    assert error_info.value.__cause__ is cause
    assert fake_openai.call_count == 1


def test_factory_has_no_network_process_file_thread_or_retry_side_effect(
    monkeypatch: pytest.MonkeyPatch,
    fake_openai: _FakeOpenAIConstructor,
) -> None:
    fail_network = _raise_assertion("factory must not access network")
    fail_process = _raise_assertion("factory must not run a process")
    fail_write = _raise_assertion("factory must not write files")
    fail_thread = _raise_assertion("factory must not start threads")
    fail_sleep = _raise_assertion("factory must not sleep or retry")
    original_import = builtins.__import__
    original_open = builtins.open

    def guarded_import(
        name: str,
        globals: dict | None = None,
        locals: dict | None = None,
        fromlist: tuple | list = (),
        level: int = 0,
    ) -> object:
        if name.split(".", 1)[0] in {
            "dotenv",
            "requests",
            "httpx",
        }:
            raise AssertionError(
                f"factory must not import {name!r}"
            )
        return original_import(name, globals, locals, fromlist, level)

    def guarded_open(
        file: object,
        mode: str = "r",
        *args: object,
        **kwargs: object,
    ) -> object:
        if any(flag in mode for flag in "wax+"):
            raise AssertionError("factory must not write files")
        if str(file).lower().endswith(".env"):
            raise AssertionError("factory must not read .env")
        return original_open(file, mode, *args, **kwargs)

    with monkeypatch.context() as context:
        context.setattr(builtins, "__import__", guarded_import)
        context.setattr(builtins, "open", guarded_open)
        context.setattr(socket, "create_connection", fail_network)
        context.setattr(socket.socket, "connect", fail_network)
        context.setattr(urllib.request, "urlopen", fail_network)
        context.setattr(subprocess, "run", fail_process)
        context.setattr(subprocess, "Popen", fail_process)
        context.setattr(os, "system", fail_process)
        context.setattr(Path, "write_text", fail_write)
        context.setattr(Path, "write_bytes", fail_write)
        context.setattr(threading.Thread, "start", fail_thread)
        context.setattr(time, "sleep", fail_sleep)

        result = create_openai_client(api_key="explicit-key")

    assert isinstance(result, _AttributeClient)
    assert fake_openai.call_count == 1


def test_factory_never_outputs_logs_or_persists_api_key(
    caplog: pytest.LogCaptureFixture,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
    fake_openai: _FakeOpenAIConstructor,
) -> None:
    secret = "factory-secret-value"
    fail_write = _raise_assertion("factory must not persist secrets")

    with monkeypatch.context() as context:
        context.setattr(Path, "write_text", fail_write)
        context.setattr(Path, "write_bytes", fail_write)
        with caplog.at_level(logging.DEBUG):
            create_openai_client(api_key=secret)

    captured = capsys.readouterr()
    assert secret not in captured.out
    assert secret not in captured.err
    assert all(secret not in record.getMessage() for record in caplog.records)


def test_factory_client_is_compatible_with_injected_runtime(
    fake_openai: _FakeOpenAIConstructor,
) -> None:
    client = _AttributeClient(
        1,
        content="The experiment is stable.",
    )
    fake_openai.outcomes.append(client)

    created_client = create_openai_client(api_key="test-key")
    result = run_copilot_turn(
        created_client,
        model="test-model",
        question="Analyze this experiment.",
    )

    assert result == "The experiment is stable."
    assert client.completions.calls
    assert fake_openai.call_count == 1
    assert inspect.signature(runtime.run_copilot_turn) == (
        inspect.signature(run_copilot_turn)
    )


@pytest.mark.parametrize(
    ("path", "forbidden"),
    [
        (Path("copilot/runtime.py"), {"llm_clients", "openai"}),
        (
            Path("llm_adapters/openai_tool_adapter.py"),
            {"llm_clients", "openai"},
        ),
        (
            Path("tool_layer/tool_registry.py"),
            {"llm_clients", "openai"},
        ),
        (
            Path("llm_clients/openai_client_factory.py"),
            {"copilot", "llm_adapters", "tool_layer"},
        ),
    ],
)
def test_dependency_direction_remains_isolated(
    path: Path,
    forbidden: set[str],
) -> None:
    assert forbidden.isdisjoint(_imported_roots(path))


def test_core_requirements_and_default_ci_remain_sdk_free() -> None:
    requirements = Path("requirements.txt").read_text(
        encoding="utf-8"
    )
    workflow = Path(".github/workflows/tests.yml").read_text(
        encoding="utf-8"
    )

    requirement_names = {
        line.split("=", 1)[0].split("<", 1)[0].strip().lower()
        for line in requirements.splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    }
    assert "openai" not in requirement_names
    assert "requirements-openai.txt" not in workflow
    assert "openai" not in workflow.lower()

"""M14 Slice 3A RED contracts: optional, call-scoped Agent Memory exposure.

Future API (no implementation or frozen signature changes in this slice):
  llm_adapters.tool_binding.BoundToolCollection(*tool_sets)
    snapshots descriptions; defaults first, then bound sets in supplied order;
    list_tools()/invoke_tool() share ownership and reject duplicate names.
  llm_adapters.tool_binding.bound_tools(collection)
    context manager, nested/context-local, restores even on BaseException.
  copilot.tool_binding.BoundCopilotSession(session, tools)
  copilot.tool_binding.BoundCopilotService(service, tools)
    immutable public session/service and tools references; delegate existing
    execution APIs inside scope, including HTTP's _run_with_observability hook.
    Service.create_session returns a bound facade around its delegate's Session.

The host owns resources. Probes below call the real frozen implementations;
they only record delegation and close their own SQLite resources in the owning
thread. They do NOT implement tool binding, dispatch or authorization. All model
responses are scripted, and no filesystem or external provider is used.
"""

from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager, nullcontext
from contextvars import Context
from copy import deepcopy
import importlib
import importlib.util
import json
from threading import get_ident, local
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from copilot.http_api import create_app
from copilot.service import CopilotService
from copilot.session import CopilotSession
from copilot.session_repository import CopilotSessionRepository
from experiment_identity import ExecutionIdentity, ExperimentIdentity, ExperimentProvenance
from experiment_memory import ExperimentMemory
from experiment_record import build_experiment_record
from experiment_repository import ExperimentRepository
import llm_adapters.openai_tool_adapter as adapter
from tool_layer import tool_registry as registry
from tool_layer.memory_tools import MemoryAccessScope, MemoryTools


DEFAULTS = ("analyze_experiment", "compare_experiments")
MEMORY = ("get_historical_execution", "list_historical_executions",
          "find_historical_comparison_candidates")
GET = {"experiment_id": "exp", "execution_id": "A-reference"}
LIST = {"experiment_id": "exp", "filters": {"realized_seed": 7}, "limit": 1}
COMPARE = {"experiment_id": "exp", "reference_execution_id": "A-reference",
           "metric_name": "r2", "limit": 10}
HOST_KEYS = ("namespace", "database_path", "allowed_experiment_ids",
             "memory_provider", "allow_invalidated", "memory_config")


def _api(*, facades=False):
    name = "llm_adapters.tool_binding"
    assert importlib.util.find_spec(name) is not None, f"missing {name}"
    binding = importlib.import_module(name)
    for symbol in ("BoundToolCollection", "bound_tools"):
        assert callable(getattr(binding, symbol, None)), f"missing {name}.{symbol}"
    if not facades:
        return binding
    name = "copilot.tool_binding"
    assert importlib.util.find_spec(name) is not None, f"missing {name}"
    facade = importlib.import_module(name)
    for symbol in ("BoundCopilotSession", "BoundCopilotService"):
        assert callable(getattr(facade, symbol, None)), f"missing {name}.{symbol}"
    return binding, facade


def _record(execution_id, *, seed=7, protocol="eval-v1", unknown=False):
    identity = ExecutionIdentity(ExperimentIdentity("lab", "exp"), execution_id)
    provenance = ExperimentProvenance(
        identity, task="regression", target="target", dataset_version="dataset-v1",
        split_id="validation", source_revision="revision-v1", realized_seed=seed,
        configuration_ref="SECRET/config", location="SECRET/location",
    ) if not unknown else ExperimentProvenance(identity, realized_seed=seed)
    return build_experiment_record({
        "experiment_name": "fixture", "experiment_dir": "SECRET/experiment",
        "summary": {"private": "SECRET/summary", "validation_metrics": {
            "r2": {"best_value": 0.8, "best_epoch": 3},
        }},
    }, provenance=provenance, metric_declarations={"r2": {
        "metric_definition": "r2-v1", "direction": "maximize", "aggregation": "macro",
        "evaluation_protocol": protocol, "selection_protocol": "best-validation",
        "checkpoint_ref": "SECRET/checkpoint", "result_artifact_ref": None,
        "history_ref": "SECRET/history",
    }})


class _ThreadMemory(ExperimentMemory):
    def __init__(self, repository, host):
        super().__init__(repository)
        self.host = host
        self.owner = get_ident()

    def _record_thread(self):
        assert get_ident() == self.owner
        self.host.query_threads.append(get_ident())

    def get(self, identity):
        self._record_thread()
        return super().get(identity)

    def list_executions(self, identity, **options):
        self._record_thread()
        return super().list_executions(identity, **options)

    def find_comparison_candidates(self, identity, reference, **options):
        self._record_thread()
        return super().find_comparison_candidates(identity, reference, **options)


class _Host:
    """Test host resource owner; acquisition is deliberately deferred per call."""

    def __init__(self):
        self.records = (
            _record("A-reference"), _record("B-comparable"),
            _record("C-incompatible", seed=8, protocol="eval-v2"),
            _record("D-unknown", seed=9, unknown=True),
        )
        self.state = local()
        self.provider_threads = []
        self.query_threads = []
        self.memories = []
        self.closed_threads = []
        self.failure = None
        # Validate real M11/M12 fixtures before the missing-capability assertion.
        repository = self._repository()
        try:
            assert len(ExperimentMemory(repository).list_executions(
                ExperimentIdentity("lab", "exp"))) == 4
        finally:
            repository.close()

    def _repository(self):
        repository = ExperimentRepository(":memory:")
        try:
            for record in self.records:
                repository.add(record)
        except BaseException:
            repository.close()
            raise
        return repository

    @contextmanager
    def execution(self):
        previous = getattr(self.state, "resources", None)
        resources = []
        self.state.resources = resources
        try:
            yield
        finally:
            for repository in resources:
                try:
                    # Integration borrows resources; only this host may close.
                    repository.get(self.records[0].provenance.execution)
                finally:
                    repository.close()
                    self.closed_threads.append(get_ident())
            self.state.resources = previous

    def provider(self):
        self.provider_threads.append(get_ident())
        assert self.state.resources is not None, "provider acquired outside host execution"
        if self.failure is not None:
            raise self.failure
        repository = self._repository()
        self.state.resources.append(repository)
        memory = _ThreadMemory(repository, self)
        self.memories.append(memory)  # Keep identities alive to detect reuse.
        return memory

    def tools(self, allowed=("exp",)):
        return MemoryTools(MemoryAccessScope("lab", allowed), self.provider)


@pytest.fixture
def host():
    return _Host()


def _response(*, content="done", calls=None):
    return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(
        role="assistant", content=content, tool_calls=calls,
    ))])


def _tool_response(name=MEMORY[0], arguments=None):
    return _response(content=None, calls=[SimpleNamespace(
        id="memory-call", type="function", function=SimpleNamespace(
            name=name, arguments=json.dumps(GET if arguments is None else arguments),
        ),
    )])


def _outcomes(name=MEMORY[0], arguments=None, facts=None):
    return [_tool_response(name, arguments),
            _response(content=json.dumps({"facts": {} if facts is None else facts}))]


class _Client:
    def __init__(self, outcomes=()):
        self.outcomes = list(outcomes)
        self.requests = []
        self.threads = []
        self.chat = SimpleNamespace(completions=self)

    def create(self, **kwargs):
        self.requests.append(deepcopy(kwargs))
        self.threads.append(get_ident())
        assert self.outcomes, "unexpected provider request (no retries allowed)"
        result = self.outcomes.pop(0)
        if isinstance(result, BaseException):
            raise result
        return result


class _SessionProbe(CopilotSession):
    def __init__(self, client, host, **kwargs):
        super().__init__(client, **kwargs)
        self.host = host
        self.delegated = []
        self.failures = []

    def ask(self, question):
        result = super().ask(question)
        self.delegated.append(("ask", result))
        return result

    def ask_with_result(self, question):
        with self.host.execution():
            result = super().ask_with_result(question)
        self.delegated.append(("ask_with_result", result))
        return result

    def ask_with_observability(self, question, *, on_failure=None):
        def capture(observation):
            self.failures.append(observation)
            if on_failure is not None:
                on_failure(observation)
        with self.host.execution():
            result = super().ask_with_observability(question, on_failure=capture)
        self.delegated.append(("ask_with_observability", result))
        return result


class _ServiceProbe(CopilotService):
    def __init__(self, client, host):
        super().__init__(client, model="fixture-model")
        self.client = client
        self.host = host
        self.delegated = []
        self.created = []
        self.failures = []

    def _capture_failure(self, callback):
        def capture(observation):
            self.failures.append(observation)
            if callback is not None:
                callback(observation)
        return capture

    def run(self, question, **options):
        with self.host.execution():
            result = super().run(question, **options)
        self.delegated.append(("run", result))
        return result

    def _run_with_observability(self, question, **options):
        options["on_failure"] = self._capture_failure(options.get("on_failure"))
        with self.host.execution():
            result = super()._run_with_observability(question, **options)
        self.delegated.append(("_run_with_observability", result))
        return result

    def create_session(self, **options):
        # Preserve the real factory call and exact resulting object. Install
        # only host lifecycle/recording on this instance, not binding behavior.
        session = super().create_session(**options)
        original = session.ask_with_observability
        session.failures = []
        session.delegated = []

        def observed(question, *, on_failure=None):
            def capture(observation):
                session.failures.append(observation)
                if on_failure is not None:
                    on_failure(observation)
            with self.host.execution():
                result = original(question, on_failure=capture)
            session.delegated.append(("ask_with_observability", result))
            return result

        session.ask_with_observability = observed
        self.created.append((deepcopy(options), session))
        return session


def _names(definitions):
    return tuple(item["function"]["name"] for item in definitions)


def _dispatch(name=MEMORY[0], arguments=None):
    messages = adapter.execute_tool_calls(_tool_response(name, arguments))
    assert len(messages) == 1
    return json.loads(messages[0]["content"])


def _assert_unbound():
    client = _Client([_response()])
    adapter.create_tool_call_response(client, model="fixture-model", messages=[])
    assert _names(client.requests[0]["tools"]) == DEFAULTS
    assert _names(registry.list_tools()) == DEFAULTS
    for name in MEMORY:
        with pytest.raises(KeyError):
            _dispatch(name)


def test_bound_collection_preserves_defaults_and_rejects_name_collisions(host):
    api = _api()
    defaults = registry.list_tools()
    memory = host.tools()
    collection = api.BoundToolCollection(memory)
    assert collection.list_tools() == defaults + memory.list_tools()
    descriptions = collection.list_tools()
    descriptions[-1]["function"]["parameters"].clear()
    assert collection.list_tools() == defaults + memory.list_tools()
    with pytest.raises(TypeError):
        collection.invoke_tool(DEFAULTS[0], {})  # Real default dispatch: missing required argument.
    with pytest.raises(KeyError):
        collection.invoke_tool("hidden_tool", {})
    for name in (*DEFAULTS, MEMORY[0]):
        duplicate = SimpleNamespace(
            list_tools=lambda name=name: [{"type": "function", "function": {
                "name": name, "description": "collision", "parameters": {"type": "object"},
            }}],
            invoke_tool=lambda *_: pytest.fail("colliding tool must not dispatch"),
        )
        with pytest.raises(ValueError):
            api.BoundToolCollection(memory, duplicate)
    with pytest.raises(ValueError):
        api.BoundToolCollection(memory, host.tools())
    assert registry.list_tools() == defaults
    assert host.provider_threads == []


def test_no_binding_preserves_adapter_injection_and_unknown_memory_semantics(host, monkeypatch):
    api = _api()
    _assert_unbound()
    defaults = registry.list_tools()
    calls = []

    def descriptions():
        calls.append("list")
        return deepcopy(defaults)

    def dispatch(name, arguments):
        calls.append((name, arguments))
        return {"injected": True}

    with monkeypatch.context() as patch:
        patch.setattr(adapter, "list_tools", descriptions)
        patch.setattr(adapter, "invoke_tool", dispatch)
        client = _Client([_tool_response(DEFAULTS[0], {"experiment_dir": "fixture"})])
        response = adapter.create_tool_call_response(
            client, model="fixture-model", messages=[], temperature=0,
        )
        assert json.loads(adapter.execute_tool_calls(response)[0]["content"]) == {"injected": True}
        assert calls == ["list", (DEFAULTS[0], {"experiment_dir": "fixture"})]
        assert set(client.requests[0]) == {"model", "messages", "tools", "temperature"}
    for bound in (False, True):
        collection = api.BoundToolCollection(host.tools())
        with api.bound_tools(collection) if bound else nullcontext():
            with pytest.raises(TypeError, match="tools are provided by the tool registry"):
                adapter.create_tool_call_response(_Client(), model="fixture-model", messages=[], tools=[])
    _assert_unbound()


def test_adapter_advertises_and_dispatches_the_same_real_memory_collection(host):
    api = _api()
    collection = api.BoundToolCollection(host.tools())
    for _ in range(2):  # Sequential bindings cannot accumulate definitions.
        with host.execution(), api.bound_tools(collection):
            for name, arguments in zip(MEMORY, (GET, LIST, COMPARE)):
                client = _Client([_tool_response(name, arguments)])
                response = adapter.create_tool_call_response(
                    client, model="fixture-model", messages=[], temperature=0,
                )
                assert client.requests[0]["tools"] == collection.list_tools()
                assert _names(client.requests[0]["tools"]) == DEFAULTS + MEMORY
                assert set(client.requests[0]) == {"model", "messages", "tools", "temperature"}
                result = json.loads(adapter.execute_tool_calls(response)[0]["content"])
                assert result["ok"] is True
                assert "SECRET" not in json.dumps(result)
                assert registry.list_tools() == collection.list_tools()[:2]
        _assert_unbound()
    assert len(host.provider_threads) == len(host.memories) == 6
    assert all(thread == get_ident() for thread in host.provider_threads + host.query_threads)


def test_call_scope_restores_nested_independent_contexts_and_all_exit_paths(host):
    api = _api()
    allowed = api.BoundToolCollection(host.tools())
    denied = api.BoundToolCollection(host.tools(()))
    with host.execution():
        for error_type in (None, RuntimeError, TimeoutError, KeyboardInterrupt, SystemExit):
            with api.bound_tools(allowed):
                assert _dispatch()["ok"] is True
                try:
                    with api.bound_tools(denied):
                        assert _dispatch() == {"format_version": 1, "ok": False,
                                               "error": {"code": "unauthorized_scope"}}
                        if error_type:
                            raise error_type("fixture")
                except BaseException as error:
                    assert error_type is not None and type(error) is error_type
                assert _dispatch()["ok"] is True
            _assert_unbound()
        first, second = Context(), Context()
        first_scope, second_scope = api.bound_tools(allowed), api.bound_tools(denied)
        first.run(first_scope.__enter__)
        try:
            second.run(second_scope.__enter__)
            try:
                assert first.run(_dispatch)["ok"] is True
                assert second.run(_dispatch)["error"]["code"] == "unauthorized_scope"
                _assert_unbound()
            finally:
                second.run(second_scope.__exit__, None, None, None)
            assert first.run(_dispatch)["ok"] is True
            second.run(_assert_unbound)
        finally:
            first.run(first_scope.__exit__, None, None, None)
        first.run(_assert_unbound)
    _assert_unbound()


def test_authorization_host_configuration_and_no_write_capability(host, monkeypatch):
    api = _api()
    authorized_at_tool_layer = []
    original_invoke = MemoryTools.invoke_tool

    def record_tool_boundary(self, name, arguments):
        authorized_at_tool_layer.append((name, deepcopy(arguments)))
        return original_invoke(self, name, arguments)

    monkeypatch.setattr(MemoryTools, "invoke_tool", record_tool_boundary)
    collection = api.BoundToolCollection(host.tools())
    with api.bound_tools(collection):
        for name, arguments in zip(MEMORY, (GET, LIST, COMPARE)):
            result = _dispatch(name, {**arguments, "experiment_id": "not-authorized"})
            assert result == {"format_version": 1, "ok": False,
                              "error": {"code": "unauthorized_scope"}}
            assert _dispatch(name, {**arguments, "include_invalidated": True})["error"]["code"] == "unauthorized_scope"
            for key in HOST_KEYS:
                assert _dispatch(name, {**arguments, key: "model-injection"})["error"]["code"] == "unsupported_option"
        # Every rejected request reaches the frozen MemoryTools boundary.
        # Adapter/facade must not substitute a second authorization policy.
        assert len(authorized_at_tool_layer) == len(MEMORY) * (2 + len(HOST_KEYS))
        client = _Client([_response()])
        adapter.create_tool_call_response(client, model="fixture-model", messages=[])
        definitions = client.requests[0]["tools"]
        assert _names(definitions) == DEFAULTS + MEMORY
        for definition in definitions[2:]:
            schema = definition["function"]["parameters"]
            assert schema["additionalProperties"] is False
            assert set(HOST_KEYS).isdisjoint(schema["properties"])
        for name in ("add", "invalidate", "delete", "overwrite", "import",
                     "add_historical_execution", "invalidate_historical_execution",
                     "edit_config", "launch_training", "retry", "workflow"):
            with pytest.raises(KeyError):
                _dispatch(name, {})
        assert set(client.requests[0]) == {"model", "messages", "tools"}
    assert host.provider_threads == host.memories == []
    _assert_unbound()


def test_bound_session_delegates_history_observation_and_immutable_scope(host):
    api, facades = _api(facades=True)
    client = _Client(_outcomes() * 3)
    original = _SessionProbe(client, host, model="fixture-model", max_turns=3, temperature=0)
    tools = api.BoundToolCollection(host.tools())
    session = facades.BoundCopilotSession(original, tools)
    assert session.session is original and session.tools is tools
    assert host.provider_threads == []
    answer = session.ask("first")
    assert answer is original.delegated[-1][1]
    turn = session.ask_with_result("second")
    assert turn is original.delegated[-1][1]
    observed = session.ask_with_observability("third")
    assert observed is original.delegated[-1][1]
    assert session.history == original.history
    assert session.history[-1] is observed.turn
    assert session.turn_count == original.turn_count == 3
    assert session.model == original.model and session.max_turns == original.max_turns
    assert session.experiment_context == original.experiment_context
    assert session.export_history() == original.export_history()
    assert observed.run.run_id
    assert {event.run_id for event in observed.run.events} == {observed.run.run_id}
    assert {event.tool_name for event in observed.run.events if event.tool_name} == {MEMORY[0]}
    assert observed.turn.tool_invocations[0].tool_name == MEMORY[0]
    assert observed.metrics.tool_invocation_count == 1
    assert [message["content"] for message in client.requests[4]["messages"]
            if message["role"] == "user"] == ["first", "second", "third"]
    for attribute, replacement in (("tools", api.BoundToolCollection(host.tools(()))),
                                   ("session", CopilotSession(_Client(), model="fixture-model"))):
        with pytest.raises(AttributeError):
            setattr(session, attribute, replacement)
    assert session.tools is tools and session.history == original.history
    assert len(host.memories) == len({id(item) for item in host.memories}) == 3
    assert len(host.closed_threads) == 3
    session.reset()
    assert original.history == session.history == ()
    _assert_unbound()


def test_bound_session_failure_preserves_transaction_and_restores_binding(host):
    api, facades = _api(facades=True)
    for error_type in (RuntimeError, TimeoutError, KeyboardInterrupt, SystemExit):
        client = _Client(_outcomes() + [_tool_response()] + _outcomes())
        original = _SessionProbe(client, host, model="fixture-model")
        session = facades.BoundCopilotSession(original, api.BoundToolCollection(host.tools()))
        first = session.ask_with_observability("committed")
        before = original.history
        failure = error_type("SECRET failure")
        host.failure = failure
        observed_failures = []
        try:
            with pytest.raises(error_type) as caught:
                session.ask_with_observability("roll back", on_failure=observed_failures.append)
            assert caught.value is failure
        finally:
            host.failure = None
        assert original.history == session.history == before == (first.turn,)
        assert observed_failures == original.failures
        assert len(observed_failures) == 1
        assert observed_failures[0].run.run_id != first.run.run_id
        _assert_unbound()
        recovered = session.ask_with_observability("recover")
        assert session.history == (first.turn, recovered.turn)
        assert [message["content"] for message in client.requests[-2]["messages"]
                if message["role"] == "user"] == ["committed", "recover"]
        _assert_unbound()


def test_bound_service_delegates_one_shot_observed_and_session_factory(host):
    api, facades = _api(facades=True)
    client = _Client(_outcomes() * 3)
    original = _ServiceProbe(client, host)
    tools = api.BoundToolCollection(host.tools())
    service = facades.BoundCopilotService(original, tools)
    session = service.create_session(max_turns=2, temperature=0)
    assert original.created[0][0] == {"max_turns": 2, "temperature": 0}
    assert session.session is original.created[0][1] and session.tools is tools
    assert service.service is original and service.tools is tools
    assert host.provider_threads == []
    for method in ("run", "_run_with_observability"):
        result = getattr(service, method)("one shot", temperature=0)
        assert original.delegated[-1] == (method, result)
        assert original.delegated[-1][1] is result
        assert result.turn.tool_invocations[0].tool_name == MEMORY[0]
        assert session.history == ()
        _assert_unbound()
    result = session.ask_with_observability("session")
    assert result is original.created[0][1].delegated[-1][1]
    assert session.history == (result.turn,)
    for attribute, replacement in (("tools", api.BoundToolCollection(host.tools(()))),
                                   ("service", CopilotService(_Client(), model="other"))):
        with pytest.raises(AttributeError):
            setattr(service, attribute, replacement)
    assert len(host.memories) == len(host.closed_threads) == 3
    _assert_unbound()


def test_provider_acquires_fresh_sqlite_memory_on_actual_execution_thread(host):
    api, facades = _api(facades=True)
    main_thread = get_ident()
    client = _Client(_outcomes() * 2)
    original = _SessionProbe(client, host, model="fixture-model")
    session = facades.BoundCopilotSession(original, api.BoundToolCollection(host.tools()))
    assert host.provider_threads == []

    def execute(question):
        result = session.ask_with_observability(question)
        _assert_unbound()  # Inspect the worker context, not only the caller.
        return result

    with ThreadPoolExecutor(max_workers=1) as executor:
        first = executor.submit(execute, "first request").result()
        second = executor.submit(execute, "second request").result()
    assert len(host.memories) == 2 and host.memories[0] is not host.memories[1]
    assert len(host.provider_threads) == 2
    assert set(host.provider_threads) == set(host.query_threads) == set(client.threads) == set(host.closed_threads)
    assert main_thread not in host.provider_threads
    assert len(host.closed_threads) == 2
    assert session.history == (first.turn, second.turn)
    assert first.run.run_id != second.run.run_id
    _assert_unbound()


def test_http_host_binding_preserves_schema_worker_execution_headers_and_errors(host):
    api, facades = _api(facades=True)
    client = _Client()
    original = _ServiceProbe(client, host)
    plain_repository = CopilotSessionRepository(original, max_sessions=2)
    plain_schema = create_app(original, plain_repository).openapi()
    service = facades.BoundCopilotService(original, api.BoundToolCollection(host.tools()))
    repository = CopilotSessionRepository(service, max_sessions=2, id_factory=lambda: "memory-session")
    app = create_app(service, repository)
    assert app.openapi() == plain_schema
    assert host.provider_threads == []
    with TestClient(app) as http:
        assert http.post("/v1/sessions", json={}).json() == {"session_id": "memory-session"}
        session = repository.get("memory-session")
        assert session.session is original.created[0][1]
        assert host.provider_threads == []
        routes = ("/v1/copilot/turns", "/v1/sessions/memory-session/turns")
        for route in routes:
            for key in HOST_KEYS:
                assert http.post(route, json={"question": "read", key: "untrusted"}).status_code == 422
            assert client.requests == []
        for route in routes:
            client.outcomes = _outcomes()
            response = http.post(route, json={"question": "read"})
            assert response.status_code == 200
            observed = (original.delegated[-1][1] if route == routes[0]
                        else session.session.delegated[-1][1])
            payload = response.json()
            assert response.headers["X-Copilot-Run-Id"] == payload["run"]["run_id"] == observed.run.run_id
            turn_payload = payload["turn"] if route == routes[0] else payload
            assert turn_payload["question"] == observed.turn.question
            assert turn_payload["answer"] == observed.turn.answer
            assert turn_payload["tool_invocations"][0]["tool_name"] == MEMORY[0]
            assert "SECRET" not in response.text
            assert (host.provider_threads[-1] == host.query_threads[-1]
                    == host.closed_threads[-1] == client.threads[-2] == client.threads[-1])
            for error, status, detail in ((RuntimeError("SECRET"), 500, "Internal server error"),
                                          (TimeoutError("SECRET"), 504, "Copilot turn timed out")):
                before = session.history
                host.failure = error
                client.outcomes = [_tool_response()]
                try:
                    failure = http.post(route, json={"question": "fail"})
                finally:
                    host.failure = None
                assert failure.status_code == status
                assert failure.json() == {"detail": detail}
                observation = (original.failures[-1] if route == routes[0]
                               else session.session.failures[-1])
                assert failure.headers["X-Copilot-Run-Id"] == observation.run.run_id
                assert session.history == before
                assert host.provider_threads[-1] == client.threads[-1]
        assert http.delete("/v1/sessions/memory-session").status_code == 204
    assert len(host.memories) == len(host.closed_threads) == 2
    assert len(host.provider_threads) == 6
    assert get_ident() not in host.provider_threads
    assert set(host.provider_threads) == set(client.threads)
    assert set(host.query_threads) == set(host.closed_threads)
    assert all(_names(request["tools"]) == DEFAULTS + MEMORY
               for request in client.requests if "tools" in request)
    _assert_unbound()

"""M20 explicit binding, real offline integration and unchanged infrastructure."""

from copy import deepcopy
import inspect
import json

import pytest

from copilot.session import CopilotSession
from llm_adapters.tool_binding import BoundToolCollection, bound_tools, _get_bound_tools
from tool_layer import tool_registry as registry
from test_tool_capability_metadata import EXPECTED_PROVIDER_TOOLS
from m20_contract_helpers import (
    TOOL, PARAMETERS, Client, assert_limits, core_api, data, deny_external, patch_core, tool_class,
)


def test_tool_constructor_signature():
    signature = inspect.signature(tool_class())
    assert list(signature.parameters) == ["declared_set", "reported_observations"]
    assert all(p.kind == inspect.Parameter.POSITIONAL_OR_KEYWORD and p.default is inspect.Parameter.empty
               for p in signature.parameters.values())


def test_empty_schema_independent_and_no_host_payload():
    tools = tool_class()(*data())
    listed = tools.list_tools()
    assert len(listed) == 1 and listed[0]["type"] == "function"
    description = listed[0]["function"]
    assert description["name"] == TOOL and description["parameters"] == PARAMETERS
    assert type(description["description"]) is str and description["description"].strip()
    assert not any(x in json.dumps(listed) for x in ("bundle-1", "metrics.csv", "host-A", "host-B"))
    description["parameters"]["properties"]["trusted"] = {"type": "boolean"}
    assert tools.list_tools()[0]["function"]["parameters"] == PARAMETERS


def test_every_invocation_delegates_real_core_without_cached_report(monkeypatch):
    cls, function, arguments = tool_class(), core_api(), data()
    calls = []
    def traced(declared_set, reported_observations):
        result = function(declared_set, reported_observations)
        calls.append((deepcopy((declared_set, reported_observations)), deepcopy(result)))
        return result
    patch_core(monkeypatch, traced)
    tools = cls(*arguments)
    calls.clear()
    first, second = tools.invoke_tool(TOOL, {}), tools.invoke_tool(TOOL, {})
    assert len(calls) == 2 and calls[0][0] == calls[1][0] == arguments
    assert first == second == calls[0][1] == calls[1][1]


@pytest.mark.parametrize("field", ["path", "artifact_id", "sha256", "observation", "trusted", "verified",
                                  "current", "approval", "checkpoint", "declared_set", "reported_observations"])
def test_any_model_option_rejects_before_core(monkeypatch, field):
    tools = tool_class()(*data())
    core_api()
    calls = []
    def forbidden(*args, **kwargs):
        calls.append(True)
        raise AssertionError("model option reached core")
    patch_core(monkeypatch, forbidden)
    with pytest.raises((TypeError, ValueError)): tools.invoke_tool(TOOL, {field: True})
    assert calls == []


@pytest.mark.parametrize("arguments", [None, [], (), "{}", 0, True])
def test_model_arguments_require_exact_dict(arguments):
    tools = tool_class()(*data())
    with pytest.raises((TypeError, ValueError)): tools.invoke_tool(TOOL, arguments)


def test_unknown_tool_has_no_fallback():
    tools = tool_class()(*data())
    with pytest.raises(KeyError): tools.invoke_tool("analyze_experiment", {})


def test_explicit_collection_advertises_and_dispatches_same_tool_and_rejects_collision():
    tools = tool_class()(*data())
    collection = BoundToolCollection(tools)
    assert collection.list_tools() == EXPECTED_PROVIDER_TOOLS + tools.list_tools()
    assert_limits(collection.invoke_tool(TOOL, {}))
    with pytest.raises(ValueError): BoundToolCollection(tools, tools)
    assert registry.list_tools() == EXPECTED_PROVIDER_TOOLS
    with pytest.raises(KeyError): registry.invoke_tool(TOOL, {})


def test_default_registry_and_no_path_capability_independent_of_m20():
    assert registry.list_tools() == EXPECTED_PROVIDER_TOOLS
    assert [s["function"]["name"] for s in registry.list_tools()] == ["analyze_experiment", "compare_experiments"]
    assert registry._workspace_paths_for_tool(TOOL) == ()
    with pytest.raises(KeyError): registry.invoke_tool(TOOL, {})


def test_constructor_listing_and_invocation_do_not_read_resolve_hash_or_acquire(monkeypatch):
    cls, arguments = tool_class(), data()
    core_api()
    with deny_external(monkeypatch) as calls:
        tools = cls(*arguments)
        listed, result = tools.list_tools(), tools.invoke_tool(TOOL, {})
    assert calls == [] and listed[0]["function"]["parameters"] == PARAMETERS
    assert_limits(result)


@pytest.mark.parametrize("entrance", ["open", "stat", "resolve", "hash", "memory", "reader", "policy", "capability", "network"])
def test_external_probe_calibration_independent_of_missing_modules(monkeypatch, entrance):
    import hashlib
    from pathlib import Path
    import socket
    import read_test_metrics
    from experiment_memory import ExperimentMemory
    # Lookup patched functions at use time, including the policy module.
    from tool_layer import experiment_path_security
    actions = {
        "open": lambda: open("never-read", "rb"),
        "stat": lambda: Path("never-stat").stat(),
        "resolve": lambda: Path("never-resolve").resolve(),
        "hash": lambda: hashlib.sha256(b"not hashed"),
        "memory": lambda: ExperimentMemory.get(None, None),
        "reader": lambda: read_test_metrics.read_test_metrics(Path("never-read"), ()),
        "policy": lambda: experiment_path_security._build_experiment_path_policy(None),
        "capability": lambda: registry._workspace_paths_for_tool(TOOL),
        "network": lambda: socket.create_connection(("invalid.example", 443)),
    }
    with deny_external(monkeypatch) as calls:
        with pytest.raises(AssertionError, match="external access sentinel"):
            actions[entrance]()
    assert calls == [True]


def test_real_session_adapter_runtime_usage_history_and_correlated_events():
    tools, client = tool_class()(*data()), Client()
    collection = BoundToolCollection(tools)
    session = CopilotSession(client, model="offline", experiment_context=None)
    with bound_tools(collection): observed = session.ask_with_observability("Compare the bound descriptors")
    assert len(client.requests) == observed.metrics.provider_request_count == 2
    assert observed.metrics.tool_invocation_count == 1
    assert client.requests[0]["tools"] == collection.list_tools()
    invocation = observed.turn.tool_invocations[0]
    assert (invocation.tool_name, invocation.tool_call_id) == (TOOL, "artifact-call")
    payload = json.loads(invocation.result_json)
    assert_limits(payload)
    delivered = [m for m in client.requests[1]["messages"] if m["role"] == "tool"]
    assert len(delivered) == 1 and json.loads(delivered[0]["content"]) == payload
    events = observed.run.events
    assert [e.sequence for e in events] == list(range(len(events)))
    assert [e.kind for e in events] == [
        "run.started", "provider.request.started", "provider.response.received",
        "tool.calls.validation.started", "tool.execution.started", "tool.result.accepted",
        "provider.request.started", "provider.response.received", "run.completed"]
    for event in [e for e in events if e.tool_name is not None]:
        assert (event.run_id, event.tool_name, event.tool_call_id, event.tool_invocation_index) == (
            observed.run.run_id, TOOL, "artifact-call", 0)
    assert observed.run.usage.complete is True
    assert (observed.run.usage.input_tokens, observed.run.usage.output_tokens,
            observed.run.usage.total_tokens) == (20, 6, 26)
    assert session.history == (observed.turn,) and session.turn_count == 1


@pytest.mark.parametrize("mode", ["invalid-args", "tool-failure", "provider-failure"])
def test_failure_history_rollback_binding_restore_and_original_exception(monkeypatch, mode):
    tools = tool_class()(*data())
    core_api()
    failure = RuntimeError("offline failure sentinel")
    if mode == "tool-failure":
        def fail(*args, **kwargs): raise failure
        patch_core(monkeypatch, fail)
    client = Client({"approval": True} if mode == "invalid-args" else {},
                    error=failure if mode == "provider-failure" else None)
    session, failures, outer = CopilotSession(client, model="offline"), [], BoundToolCollection()
    with bound_tools(outer):
        with pytest.raises((TypeError, ValueError) if mode == "invalid-args" else RuntimeError) as caught:
            with bound_tools(BoundToolCollection(tools)):
                session.ask_with_observability("compare", on_failure=failures.append)
        if mode != "invalid-args": assert caught.value is failure
        assert _get_bound_tools() is outer
        assert session.history == () and session.turn_count == 0
        assert len(client.requests) == (2 if mode == "provider-failure" else 1)
        assert len(failures) == 1 and failures[0].run.events[-1].kind == "run.failed"
        if mode != "provider-failure": assert failures[0].stage == "tool_execution"
        else: assert failures[0].tool_invocation_count == 1
        fresh = Client()
        with pytest.raises(KeyError): CopilotSession(fresh, model="offline").ask("unbound")
        assert fresh.requests[0]["tools"] == EXPECTED_PROVIDER_TOOLS


@pytest.mark.parametrize("mode", ["mismatch", "missing", "unobserved"])
def test_completed_run_preserves_descriptor_disagreement_and_limits(mode):
    cls, (declared, reported) = tool_class(), data()
    if mode == "mismatch": reported[0]["byte_size"] = 99
    elif mode == "missing": reported[0].update(reported_status="missing", byte_size=None, sha256=None, mtime_ns=None)
    else: reported.clear()
    session = CopilotSession(Client(), model="offline")
    with bound_tools(BoundToolCollection(cls(declared, reported))):
        observed = session.ask_with_observability("report limits")
    payload = json.loads(observed.turn.tool_invocations[0].result_json)
    assert_limits(payload)
    row = payload["artifacts"]["metrics"]
    if mode == "mismatch": assert row["integrity"]["size_comparison"] == "mismatch"
    else: assert row["reported_status"] == mode
    assert observed.run.events[-1].kind == "run.completed"

"""M19 explicit binding over snapshots through the real Adapter/Session/Runtime."""

from copy import deepcopy
import importlib
import inspect
import json
from pathlib import Path
import socket

import pytest

from copilot.session import CopilotSession
from llm_adapters.tool_binding import BoundToolCollection, bound_tools, _get_bound_tools
from tool_layer import tool_registry as registry
from test_tool_capability_metadata import EXPECTED_PROVIDER_TOOLS
from m18_contract_helpers import deny_external_access
from m19_contract_helpers import (
    TOOL, PARAMETERS, Client, core_api, invoke, pair, patch_public, tool_class,
)


def test_constructor_signature():
    signature = inspect.signature(tool_class())
    assert list(signature.parameters) == ["group_a", "group_b", "targets"]
    assert signature.parameters["targets"].kind == inspect.Parameter.KEYWORD_ONLY
    assert signature.parameters["targets"].default is inspect.Parameter.empty


def test_empty_schema_no_host_data_and_independent_results():
    tools = invoke(tool_class(), pair())
    listed = tools.list_tools()
    assert len(listed) == 1 and listed[0]["type"] == "function"
    schema = listed[0]["function"]
    assert schema["name"] == TOOL and schema["parameters"] == PARAMETERS
    assert type(schema["description"]) is str and schema["description"].strip()
    assert not any(x in json.dumps(listed) for x in ("snapshot-0", "recipe-v1", "quality-A"))
    schema["parameters"]["properties"]["approval"] = {"type": "boolean"}
    assert tools.list_tools()[0]["function"]["parameters"] == PARAMETERS


def test_invocations_really_delegate_to_core_each_time_without_cached_result(monkeypatch):
    cls, function, data = tool_class(), core_api(), pair()
    calls = []
    def traced(a, b, *, targets):
        result = function(a, b, targets=targets)
        calls.append((deepcopy((a, b, targets)), deepcopy(result)))
        return result
    patch_public(monkeypatch, importlib.import_module("experiment_group_comparison"), TOOL,
                 traced, importlib.import_module("tool_layer.replicate_group_tools"))
    tools = invoke(cls, data)
    calls.clear()
    first, second = tools.invoke_tool(TOOL, {}), tools.invoke_tool(TOOL, {})
    assert len(calls) == 2
    assert calls[0][0] == calls[1][0] == data
    assert first == calls[0][1] == second == calls[1][1]


@pytest.mark.parametrize("arguments", [
    {"group_a": {}}, {"group_b": {}}, {"targets": []}, {"executions": []},
    {"approval": True}, {"path": "../outside"}, {"directory": "C:/outside"},
    {"directories": ["A", "B"]}, {"pair_mapping": []}, {"provider": "memory"},
    {"callback": "loader"}, {"unknown": 1},
])
def test_nonempty_model_arguments_rejected_before_core(monkeypatch, arguments):
    cls, function = tool_class(), core_api()
    tools, calls = invoke(cls, pair()), []
    def forbidden(*args, **kwargs):
        calls.append(True)
        raise AssertionError("invalid arguments reached core")
    patch_public(monkeypatch, importlib.import_module("experiment_group_comparison"), TOOL,
                 forbidden, importlib.import_module("tool_layer.replicate_group_tools"))
    with pytest.raises((TypeError, ValueError)): tools.invoke_tool(TOOL, arguments)
    assert calls == []


@pytest.mark.parametrize("arguments", [None, [], (), "{}", 1, True])
def test_runtime_arguments_are_exact_empty_dictionary(arguments):
    tools = invoke(tool_class(), pair())
    with pytest.raises((TypeError, ValueError)): tools.invoke_tool(TOOL, arguments)


def test_unknown_tool_rejected_without_fallback():
    tools = invoke(tool_class(), pair())
    with pytest.raises(KeyError): tools.invoke_tool("analyze_experiment", {})


def test_explicit_binding_announcement_dispatch_and_collision():
    tools = invoke(tool_class(), pair())
    collection = BoundToolCollection(tools)
    assert collection.list_tools() == EXPECTED_PROVIDER_TOOLS + tools.list_tools()
    assert collection.invoke_tool(TOOL, {})["targets"]["quality"]["comparison"]["raw_delta"] == 1.
    with pytest.raises(ValueError): BoundToolCollection(tools, tools)
    assert registry.list_tools() == EXPECTED_PROVIDER_TOOLS
    with pytest.raises(KeyError): registry.invoke_tool(TOOL, {})


def test_no_external_access_in_constructor_listing_or_invocation(monkeypatch):
    cls, data = tool_class(), pair()
    core_api()
    with deny_external_access(monkeypatch) as calls:
        tools = invoke(cls, data)
        description, result = tools.list_tools(), tools.invoke_tool(TOOL, {})
    assert calls == [] and description[0]["function"]["parameters"] == PARAMETERS
    assert result["targets"]["quality"]["comparison"]["raw_delta"] == 1.


def test_real_session_adapter_runtime_without_experiment_path_authorization():
    tools = invoke(tool_class(), pair())
    collection, client = BoundToolCollection(tools), Client()
    session = CopilotSession(client, model="offline", experiment_context=None)
    with bound_tools(collection):
        observed = session.ask_with_observability("Compare the bound groups descriptively")
    assert len(client.requests) == observed.metrics.provider_request_count == 2
    assert observed.metrics.tool_invocation_count == 1
    assert client.requests[0]["tools"] == collection.list_tools()
    invocation = observed.turn.tool_invocations[0]
    assert (invocation.tool_name, invocation.tool_call_id) == (TOOL, "group-comparison-call")
    payload = json.loads(invocation.result_json)
    assert payload["targets"]["quality"]["comparison"]["raw_delta"] == 1.
    delivered = [m for m in client.requests[1]["messages"] if m["role"] == "tool"]
    assert len(delivered) == 1 and json.loads(delivered[0]["content"]) == payload
    events = observed.run.events
    assert [e.sequence for e in events] == list(range(len(events)))
    started = [e for e in events if e.kind == "tool.execution.started"]
    accepted = [e for e in events if e.kind == "tool.result.accepted"]
    assert len(started) == len(accepted) == 1
    for event in started + accepted:
        assert (event.run_id, event.tool_name, event.tool_call_id, event.tool_invocation_index) == (
            observed.run.run_id, TOOL, "group-comparison-call", 0)
    assert started[0].sequence < accepted[0].sequence < events[-1].sequence
    assert events[-1].kind == "run.completed"
    assert observed.run.usage.complete is False and observed.run.usage.total_tokens is None
    assert session.history == (observed.turn,) and session.turn_count == 1


@pytest.mark.parametrize("mode", ["invalid-args", "tool-failure", "provider-failure"])
def test_failure_exception_history_observation_and_binding_restore(monkeypatch, mode):
    cls, function = tool_class(), core_api()
    tools = invoke(cls, pair())
    failure = RuntimeError("offline failure sentinel")
    if mode == "tool-failure":
        def fail(*args, **kwargs): raise failure
        patch_public(monkeypatch, importlib.import_module("experiment_group_comparison"), TOOL,
                     fail, importlib.import_module("tool_layer.replicate_group_tools"))
    client = Client({"approval": True} if mode == "invalid-args" else {},
                    error=failure if mode == "provider-failure" else None)
    session, observations, outer = CopilotSession(client, model="offline"), [], BoundToolCollection()
    with bound_tools(outer):
        with pytest.raises((TypeError, ValueError) if mode == "invalid-args" else RuntimeError) as caught:
            with bound_tools(BoundToolCollection(tools)):
                session.ask_with_observability("compare", on_failure=observations.append)
        if mode != "invalid-args": assert caught.value is failure
        assert _get_bound_tools() is outer and session.history == ()
        assert len(client.requests) == (2 if mode == "provider-failure" else 1)
        assert len(observations) == 1 and observations[0].run.events[-1].kind == "run.failed"
        if mode != "provider-failure": assert observations[0].stage == "tool_execution"
        else: assert observations[0].tool_invocation_count == 1
        fresh = Client()
        with pytest.raises(KeyError): CopilotSession(fresh, model="offline").ask("unbound")
        assert fresh.requests[0]["tools"] == EXPECTED_PROVIDER_TOOLS


@pytest.mark.parametrize("mode,eligibility", [("partial", "limited"), ("conflict", "blocked"), ("unknown", "insufficient")])
def test_runtime_completion_does_not_upgrade_evidence(mode, eligibility):
    cls, data = tool_class(), pair()
    if mode == "partial": data[1]["executions"][-1]["evidence"]["test_results"].update(status="invalid", metrics={})
    else:
        for m in data[1]["targets"][0]["members"]:
            m["metric_definition"] = "other" if mode == "conflict" else None
    session = CopilotSession(Client(), model="offline")
    with bound_tools(BoundToolCollection(invoke(cls, data))):
        observed = session.ask_with_observability("compare available evidence")
    row = json.loads(observed.turn.tool_invocations[0].result_json)["targets"]["quality"]
    assert row["comparison"]["eligibility"] == eligibility
    if mode == "partial":
        assert row["b"]["coverage"]["excluded_count"] == 1
        assert row["b"]["coverage"]["included_count"] == 2
    else: assert row["comparison"]["raw_delta"] is None
    assert observed.run.events[-1].kind == "run.completed"


# Independent PASS contracts: the missing tool must not mask probe validation.
def test_default_registry_and_private_capability_are_unchanged():
    assert registry.list_tools() == EXPECTED_PROVIDER_TOOLS
    assert registry._workspace_paths_for_tool(TOOL) == ()
    with pytest.raises(KeyError): registry.invoke_tool(TOOL, {})


@pytest.mark.parametrize("entrance", ["open", "resolve", "network", "memory", "registry", "reader", "capability"])
def test_external_access_probe_independent_calibration(monkeypatch, entrance):
    from experiment_memory import ExperimentMemory
    from tool_layer import experiment_tools
    actions = {
        "open": lambda: open("never-read.ckpt", "rb"),
        "resolve": lambda: Path("never-resolved").resolve(),
        "network": lambda: socket.create_connection(("invalid.example", 443)),
        "memory": lambda: ExperimentMemory.list_executions(None, None),
        "registry": lambda: registry.invoke_tool("analyze_experiment", {}),
        "reader": lambda: experiment_tools.analyze_experiment_evidence("never-read"),
        "capability": lambda: registry._workspace_paths_for_tool(TOOL),
    }
    with deny_external_access(monkeypatch) as calls:
        with pytest.raises(AssertionError, match="unexpected external access"): actions[entrance]()
    assert len(calls) == 1

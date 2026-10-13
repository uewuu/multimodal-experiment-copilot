"""M18 RED: host snapshots only; real binding, Session, Runtime, and core.

No new path capability: the tool cannot acquire data. Provider messages alone
are doubled in end-to-end tests. Boundary spies have independent calibration.
"""

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
from m18_contract_helpers import (
    PARAMETERS, TOOL, Client, bundle, core_api, deny_external_access, make_tools, tool_class,
)


def test_constructor_contract():
    cls = tool_class()
    signature = inspect.signature(cls)
    assert list(signature.parameters) == ["executions", "group_declaration", "targets"]
    for name in ("group_declaration", "targets"):
        assert signature.parameters[name].kind == inspect.Parameter.KEYWORD_ONLY
        assert signature.parameters[name].default is inspect.Parameter.empty


def test_empty_schema_isolated_and_no_host_data_leak():
    tools = make_tools(tool_class(), bundle())
    first = tools.list_tools()
    assert len(first) == 1 and first[0]["type"] == "function"
    definition = first[0]["function"]
    assert definition["name"] == TOOL and definition["parameters"] == PARAMETERS
    assert type(definition["description"]) is str and definition["description"].strip()
    assert "snapshot-0" not in json.dumps(first) and "recipe-v1" not in json.dumps(first)
    first[0]["function"]["parameters"]["properties"]["injected"] = {}
    assert tools.list_tools()[0]["function"]["parameters"] == PARAMETERS


def test_host_snapshots_are_deep_and_outputs_do_not_pollute_future_calls():
    cls, data = tool_class(), bundle()
    tools = make_tools(cls, data)
    data[0][0]["evidence"]["test_results"]["metrics"]["score"]["value"] = 999
    data[0][0]["snapshot_metadata"]["binding"] = "file_verified"
    data[1]["expected_executions"].pop()
    data[1]["group_id"] = "mutated"
    data[2][0]["members"][0]["evaluation_protocol"] = None
    result = tools.invoke_tool(TOOL, {})
    assert result["group"]["group_id"] == "group-v1"
    assert result["group"]["coverage"]["expected_count"] == 3
    assert result["targets"]["quality"]["statistics"]["mean"] == 3.0
    pristine = deepcopy(result)
    result["targets"]["quality"]["statistics"]["mean"] = 123
    assert tools.invoke_tool(TOOL, {}) == pristine


def test_real_core_delegation_not_a_second_aggregation_implementation(monkeypatch):
    cls, function = tool_class(), core_api()
    data = bundle()
    core, tool_module = importlib.import_module("experiment_replicates"), importlib.import_module("tool_layer.replicate_tools")
    calls = []
    def traced(*args, **kwargs):
        result = function(*args, **kwargs)
        calls.append((deepcopy(args), deepcopy(kwargs), deepcopy(result)))
        return result
    monkeypatch.setattr(core, TOOL, traced)
    for alias, value in list(vars(tool_module).items()):
        if value is function: monkeypatch.setattr(tool_module, alias, traced)
    tools = make_tools(cls, data)
    calls.clear()  # constructor validation may assess the snapshots
    result = tools.invoke_tool(TOOL, {})
    assert len(calls) == 1 and result == calls[0][2]
    assert calls[0][0] == (data[0],)
    assert calls[0][1] == {"group_declaration": data[1], "targets": data[2]}


@pytest.mark.parametrize("arguments", [
    {"path": "../outside"}, {"directory": "C:/outside"}, {"execution": "other"},
    {"executions": []}, {"target": "other"}, {"targets": []}, {"seed": 42},
    {"group_declaration": {}}, {"approval": True}, {"callback": "loader"},
    {"provider": "memory"}, {"unknown": 1},
])
def test_nonempty_model_arguments_rejected_before_core(monkeypatch, arguments):
    cls, function = tool_class(), core_api()
    tools = make_tools(cls, bundle())
    core, tool_module = importlib.import_module("experiment_replicates"), importlib.import_module("tool_layer.replicate_tools")
    calls = []
    def forbidden(*args, **kwargs):
        calls.append((args, kwargs))
        raise AssertionError("invalid arguments reached core")
    monkeypatch.setattr(core, TOOL, forbidden)
    for alias, value in list(vars(tool_module).items()):
        if value is function: monkeypatch.setattr(tool_module, alias, forbidden)
    with pytest.raises((TypeError, ValueError)):
        tools.invoke_tool(TOOL, arguments)
    assert calls == []


@pytest.mark.parametrize("arguments", [None, [], "{}", 1, True])
def test_runtime_arguments_must_be_a_mapping(arguments):
    tools = make_tools(tool_class(), bundle())
    with pytest.raises((TypeError, ValueError)): tools.invoke_tool(TOOL, arguments)


def test_unknown_tool_does_not_fallback_to_registry():
    tools = make_tools(tool_class(), bundle())
    with pytest.raises(KeyError): tools.invoke_tool("analyze_experiment", {})


@pytest.mark.parametrize("placement", ["executions", "metadata", "group", "target"])
@pytest.mark.parametrize("kind", ["callable", "provider", "generator"])
def test_lazy_or_dynamic_data_rejected_without_invocation(placement, kind):
    cls, data, calls = tool_class(), bundle(), []
    def loader():
        calls.append("called")
        return []
    class Provider:
        def load(self): return loader()
        def __iter__(self):
            calls.append("iterated")
            return iter(())
        def __deepcopy__(self, memo):
            calls.append("deepcopied")
            raise AssertionError("unsafe custom copy executed")
    def lazy():
        calls.append("advanced")
        yield 1
    value = {"callable": loader, "provider": Provider(), "generator": lazy()}[kind]
    if placement == "executions": data = (value, data[1], data[2])
    elif placement == "metadata": data[0][0]["snapshot_metadata"]["loader"] = value
    elif placement == "group": data[1]["membership_selection"] = value
    else: data[2][0]["members"][0]["evaluation_protocol"] = value
    with pytest.raises((TypeError, ValueError)): make_tools(cls, data)
    assert calls == []


def test_real_bound_collection_announces_exact_dispatch_owner_and_rejects_collision():
    tools = make_tools(tool_class(), bundle())
    collection = BoundToolCollection(tools)
    assert collection.list_tools() == EXPECTED_PROVIDER_TOOLS + tools.list_tools()
    assert collection.invoke_tool(TOOL, {})["targets"]["quality"]["statistics"]["mean"] == 3.0
    with pytest.raises(ValueError): BoundToolCollection(tools, tools)
    assert registry.list_tools() == EXPECTED_PROVIDER_TOOLS
    with pytest.raises(KeyError): registry.invoke_tool(TOOL, {})


def test_constructor_listing_and_invocation_have_no_external_access(monkeypatch):
    cls, data = tool_class(), bundle()
    # Load the delegated module before import machinery is denied.
    core_api()
    with deny_external_access(monkeypatch) as calls:
        tools = make_tools(cls, data)
        description = tools.list_tools()
        result = tools.invoke_tool(TOOL, {})
    assert calls == [] and description[0]["function"]["parameters"] == PARAMETERS
    assert result["targets"]["quality"]["statistics"]["mean"] == 3.0


def test_real_session_without_filesystem_authorization_publishes_snapshot_tool_and_observes_it():
    tools = make_tools(tool_class(), bundle())
    collection, client = BoundToolCollection(tools), Client()
    session = CopilotSession(client, model="offline", experiment_context=None)
    with bound_tools(collection):
        observed = session.ask_with_observability("Summarize the bound execution snapshots")
    assert len(client.requests) == observed.metrics.provider_request_count == 2
    assert observed.metrics.tool_invocation_count == 1
    assert client.requests[0]["tools"] == collection.list_tools()
    invocation = observed.turn.tool_invocations[0]
    assert (invocation.tool_name, invocation.tool_call_id) == (TOOL, "replicate-call")
    payload = json.loads(invocation.result_json)
    assert payload["targets"]["quality"]["statistics"]["mean"] == 3.0
    delivered = [message for message in client.requests[1]["messages"] if message["role"] == "tool"]
    assert len(delivered) == 1 and json.loads(delivered[0]["content"]) == payload
    events = observed.run.events
    assert [event.sequence for event in events] == list(range(len(events)))
    started = [e for e in events if e.kind == "tool.execution.started"]
    accepted = [e for e in events if e.kind == "tool.result.accepted"]
    assert len(started) == len(accepted) == 1
    for event in started + accepted:
        assert (event.run_id, event.tool_name, event.tool_call_id, event.tool_invocation_index) == (
            observed.run.run_id, TOOL, "replicate-call", 0)
    assert started[0].sequence < accepted[0].sequence < events[-1].sequence
    assert events[-1].kind == "run.completed"
    assert observed.run.usage.complete is False and observed.run.usage.total_tokens is None
    assert session.history == (observed.turn,) and session.turn_count == 1


def test_formal_adapter_rejects_injection_and_does_not_commit_history():
    tools = make_tools(tool_class(), bundle())
    client = Client({"directory": "../../outside"})
    session, failures = CopilotSession(client, model="offline"), []
    with bound_tools(BoundToolCollection(tools)):
        with pytest.raises((TypeError, ValueError)):
            session.ask_with_observability("summarize", on_failure=failures.append)
    assert session.history == () and len(client.requests) == 1
    assert len(failures) == 1 and failures[0].stage == "tool_execution"
    assert failures[0].run.events[-1].kind == "run.failed"


def test_final_provider_failure_preserves_exception_history_and_binding_restore():
    tools = make_tools(tool_class(), bundle())
    failure = RuntimeError("offline final response sentinel")
    client = Client(error=failure)
    session, observations = CopilotSession(client, model="offline"), []
    outer = BoundToolCollection()
    with bound_tools(outer):
        with pytest.raises(RuntimeError) as caught:
            with bound_tools(BoundToolCollection(tools)):
                session.ask_with_observability("summarize", on_failure=observations.append)
        assert _get_bound_tools() is outer
        assert caught.value is failure and session.history == ()
        assert len(client.requests) == 2
        assert len(observations) == 1 and observations[0].tool_invocation_count == 1
        assert observations[0].run.events[-1].kind == "run.failed"
        fresh = Client()
        with pytest.raises(KeyError): CopilotSession(fresh, model="offline").ask("unbound")
        assert fresh.requests[0]["tools"] == EXPECTED_PROVIDER_TOOLS


def test_runtime_completion_keeps_partial_coverage_and_invalid_exclusion():
    data = bundle()
    data[0][2]["evidence"]["test_results"].update(status="invalid", metrics={})
    tools = make_tools(tool_class(), data)
    session = CopilotSession(Client(), model="offline")
    with bound_tools(BoundToolCollection(tools)):
        observed = session.ask_with_observability("summarize available evidence")
    row = json.loads(observed.turn.tool_invocations[0].result_json)["targets"]["quality"]
    assert row["target_status"] == "partial"
    assert row["coverage"]["expected_count"] == 3 and row["coverage"]["included_count"] == 2
    assert row["excluded_executions"][0]["execution"]["execution_id"] == "exec-02"
    assert observed.run.events[-1].kind == "run.completed"


# These checks must pass before implementation; no missing-class masking.
def test_default_registry_and_private_capabilities_remain_frozen():
    assert registry.list_tools() == EXPECTED_PROVIDER_TOOLS
    assert registry._workspace_paths_for_tool(TOOL) == ()
    with pytest.raises(KeyError): registry.invoke_tool(TOOL, {})


@pytest.mark.parametrize("entrance", ["open", "resolve", "network", "memory", "registry", "reader"])
def test_external_access_probe_catches_independent_counterexamples(monkeypatch, entrance):
    from experiment_memory import ExperimentMemory
    from tool_layer.experiment_tools import analyze_experiment_evidence
    actions = {
        "open": lambda: open("never-read.ckpt", "rb"),
        "resolve": lambda: Path("never-resolved").resolve(),
        "network": lambda: socket.create_connection(("invalid.example", 443)),
        "memory": lambda: ExperimentMemory.list_executions(None, None),
        "registry": lambda: registry.invoke_tool("analyze_experiment", {}),
        "reader": lambda: analyze_experiment_evidence("never-read"),
    }
    # Resolve reader at call time so the calibration exercises the installed probe.
    if entrance == "reader":
        from tool_layer import experiment_tools
        actions[entrance] = lambda: experiment_tools.analyze_experiment_evidence("never-read")
    with deny_external_access(monkeypatch) as calls:
        with pytest.raises(AssertionError, match="unexpected external access"):
            actions[entrance]()
    assert len(calls) == 1

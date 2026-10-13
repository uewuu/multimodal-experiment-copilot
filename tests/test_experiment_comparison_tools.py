"""M17 RED: optional ordered pair binding through the existing Agent path.

Only the Provider is simulated on success paths. Composition/metadata readers,
core, Registry capabilities, Path Policy, binding, Session and Runtime are real.
Authorization probes advertise the new name without requiring the missing class
so a missing private capability cannot be hidden behind a missing class failure.
"""

from copy import deepcopy
from dataclasses import replace
import inspect
import json
import os
from pathlib import Path
import subprocess

import pytest

from copilot.session import CopilotSession
from llm_adapters.tool_binding import BoundToolCollection, bound_tools
from tool_layer import experiment_tools as existing
from tool_layer import tool_registry as registry
from test_tool_capability_metadata import EXPECTED_PROVIDER_TOOLS
from m17_contract_helpers import (
    TOOL, Client, arguments, assert_blocked, options, target, tool_class, write_experiment,
)


PARAMETERS = {"type": "object", "properties": {
    "experiment_a": {"type": "string"}, "experiment_b": {"type": "string"}},
    "required": ["experiment_a", "experiment_b"], "additionalProperties": False}


def pair(tmp_path):
    return write_experiment(tmp_path, "a"), write_experiment(tmp_path, "b", value=0.75)


def make_tools(a, b, **kwargs):
    return tool_class()(str(a), str(b), options_a=options(), options_b=options(),
                        targets=kwargs.pop("targets", [target()]), **kwargs)


def directory_link(link, destination):
    try:
        link.symlink_to(destination, target_is_directory=True)
    except OSError:
        if os.name != "nt":
            raise
        result = subprocess.run(["cmd", "/c", "mklink", "/J", str(link), str(destination)],
                                capture_output=True, check=False)
        assert result.returncode == 0, "environment failure: directory link unavailable"


def test_default_registry_schema_order_dispatch_and_m16_private_capability_unchanged():
    assert registry.list_tools() == EXPECTED_PROVIDER_TOOLS
    with pytest.raises(KeyError):
        registry.invoke_tool(TOOL, {"experiment_a": "a", "experiment_b": "b"})
    assert [(x.parameter_name, x.path_kind) for x in
            registry._workspace_paths_for_tool("analyze_experiment_evidence")] == [
                ("experiment_dir", "directory")]


def test_comparison_has_two_private_directory_capabilities_without_default_registration():
    assert [(x.parameter_name, x.path_kind) for x in registry._workspace_paths_for_tool(TOOL)] == [
        ("experiment_a", "directory"), ("experiment_b", "directory")
    ], "missing M17 capability: two comparison directory paths"
    assert registry.list_tools() == EXPECTED_PROVIDER_TOOLS


@pytest.mark.parametrize("case", ["a-outside", "b-outside", "wrong-root", "none", "a-link", "b-link"])
def test_both_paths_authorized_before_dispatch_independently_of_missing_tool_class(tmp_path, case):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    a, b = pair(workspace)
    outside = write_experiment(tmp_path, "outside")
    selected = arguments(a, b)
    context = {"experiment_root": str(workspace)}
    if case in ("a-outside", "b-outside"):
        selected["experiment_" + case[0]] = str(outside)
    elif case == "wrong-root": context = {"experiment_root": str(outside)}
    elif case == "none": context = None
    else:
        directory_link(workspace / "escape", outside)
        selected["experiment_" + case[0]] = str(workspace / "escape")
    dispatches = []
    class Probe:
        def list_tools(self):
            return [{"type": "function", "function": {
                "name": TOOL, "description": "Authorization probe", "parameters": deepcopy(PARAMETERS)}}]
        def invoke_tool(self, name, args):
            dispatches.append((name, args))
            raise AssertionError("M17 comparison dispatched before authorization")
    client = Client(selected)
    session = CopilotSession(client, model="offline", experiment_context=context)
    failures = []
    with bound_tools(BoundToolCollection(Probe())):
        with pytest.raises(ValueError):
            session.ask_with_observability("compare", on_failure=failures.append)
    assert dispatches == []
    assert len(client.requests) == 1 and session.history == ()
    assert len(failures) == 1 and failures[0].stage == "tool_execution"
    assert failures[0].run.events[-1].kind == "run.failed"


def test_schema_copies_binding_collision_and_announced_dispatch_set(tmp_path):
    tool_class()
    a, b = pair(tmp_path)
    tools = make_tools(a, b)
    schema = tools.list_tools()
    assert len(schema) == 1 and schema[0]["function"]["name"] == TOOL
    assert schema[0]["function"]["parameters"] == PARAMETERS
    assert isinstance(schema[0]["function"]["description"], str) and schema[0]["function"]["description"].strip()
    collection = BoundToolCollection(tools)
    assert collection.list_tools() == EXPECTED_PROVIDER_TOOLS + schema
    with pytest.raises(ValueError):
        BoundToolCollection(tools, tools)
    schema[0]["function"]["parameters"]["properties"].clear()
    assert tools.list_tools()[0]["function"]["parameters"] == PARAMETERS
    assert collection.invoke_tool(TOOL, arguments(a, b))["targets"]["quality"]["calculation"]["raw_delta"] == 0.25
    with pytest.raises(KeyError):
        tools.invoke_tool("analyze_experiment", arguments(a, b))


def test_nested_host_choices_are_detached_and_schema_does_not_leak_them(tmp_path):
    cls = tool_class()
    a, b = pair(tmp_path)
    oa, ob, selections = options(), options(), [target()]
    tools = cls(str(a), str(b), options_a=oa, options_b=ob, targets=selections)
    oa["config_paths"][0][0] = "not_selected"
    oa["test_metric_specs"].clear()
    ob["test_metrics_file"] = "missing.csv"
    ob["metrics_config"] = "missing.yaml"
    selections[0]["b"]["metric_definition"] = "mutated-definition"
    selections[0]["b"]["provenance"] = replace(selections[0]["b"]["provenance"], split_id="wrong")
    result = tools.invoke_tool(TOOL, arguments(a, b))
    row = result["targets"]["quality"]
    assert row["comparability"]["status"] == "comparable"
    assert row["calculation"]["raw_delta"] == 0.25
    assert result["configuration_differences"][0]["path"] == ["enabled"]
    assert tools.list_tools()[0]["function"]["parameters"] == PARAMETERS


@pytest.mark.parametrize("case", ["swap", "a-sibling", "b-sibling"])
def test_ordered_pair_rejects_substitution_under_broad_root_before_either_read(tmp_path, monkeypatch, case):
    tool_class()
    a, b = pair(tmp_path)
    c = write_experiment(tmp_path, "c")
    tools = make_tools(a, b)
    args = arguments(b, a) if case == "swap" else arguments(a, b)
    if case != "swap": args["experiment_" + case[0]] = str(c)
    reads = []
    def deny(*args, **kwargs):
        reads.append(args)
        raise AssertionError("composition entered before pair validation")
    monkeypatch.setattr(existing, "analyze_experiment_evidence", deny)
    client = Client(args)
    session = CopilotSession(client, model="offline", experiment_context={"experiment_root": str(tmp_path)})
    with bound_tools(BoundToolCollection(tools)):
        with pytest.raises(ValueError):
            session.ask_with_observability("compare")
    assert reads == [] and session.history == () and len(client.requests) == 1


@pytest.mark.parametrize("extra", [
    {"targets": []}, {"config_paths": []}, {"test_metrics_file": "other.csv"},
    {"test_metric_specs": []}, {"metrics_config": "other.yaml"},
    {"provenance": {}}, {"evaluation_protocol": "forged"}, {"approval": True},
    {"options_a": {}}, {"include_diagnostics": True}, {"unknown": 1},
])
def test_model_cannot_inject_host_choices_or_undeclared_arguments(tmp_path, monkeypatch, extra):
    tool_class()
    a, b = pair(tmp_path)
    tools = make_tools(a, b)
    reads = []
    def deny(*args, **kwargs):
        reads.append(args)
        raise AssertionError("invalid model arguments reached composition")
    monkeypatch.setattr(existing, "analyze_experiment_evidence", deny)
    with pytest.raises((TypeError, ValueError)):
        tools.invoke_tool(TOOL, {**arguments(a, b), **extra})
    assert reads == []


@pytest.mark.parametrize("bad", [None, 1, [], {}, ""])
def test_model_directory_types_are_validated_at_runtime(tmp_path, bad):
    tool_class()
    a, b = pair(tmp_path)
    tools = make_tools(a, b)
    with pytest.raises((TypeError, ValueError)):
        tools.invoke_tool(TOOL, {"experiment_a": str(a), "experiment_b": bad})


def test_relative_equivalent_pair_resolves_through_existing_policy(tmp_path):
    tool_class()
    a, b = pair(tmp_path)
    client = Client({"experiment_a": "./a", "experiment_b": "b/."})
    session = CopilotSession(client, model="offline", experiment_context={"experiment_root": str(tmp_path)})
    with bound_tools(BoundToolCollection(make_tools(a, b))):
        observed = session.ask_with_observability("compare")
    payload = json.loads(observed.turn.tool_invocations[0].result_json)
    assert payload["targets"]["quality"]["calculation"]["raw_delta"] == 0.25


def test_real_session_composes_twice_and_preserves_events_usage_and_history(tmp_path, monkeypatch):
    tool_class()
    a, b = pair(tmp_path)
    before = {str(p): p.read_bytes() for root in (a, b) for p in root.iterdir()}
    original, calls = existing.analyze_experiment_evidence, []
    def traced(root, **kwargs):
        result = original(root, **kwargs)
        calls.append((Path(root).resolve(), deepcopy(kwargs), deepcopy(result)))
        return result
    monkeypatch.setattr(existing, "analyze_experiment_evidence", traced)
    collection = BoundToolCollection(make_tools(a, b))
    client = Client(arguments(a, b))
    session = CopilotSession(client, model="offline", experiment_context={"experiment_root": str(tmp_path)})
    with bound_tools(collection):
        observed = session.ask_with_observability("compare")
    assert [c[0] for c in calls] == [a.resolve(), b.resolve()]
    assert all(c[1].get("include_diagnostics", False) is False for c in calls)
    assert all(c[2]["test_results"]["status"] == "available" for c in calls)
    assert len(client.requests) == observed.metrics.provider_request_count == 2
    assert observed.metrics.tool_invocation_count == 1
    assert client.requests[0]["tools"] == collection.list_tools()
    invocation = observed.turn.tool_invocations[0]
    assert (invocation.tool_name, invocation.tool_call_id) == (TOOL, "comparison-call")
    payload = json.loads(invocation.result_json)
    assert payload["targets"]["quality"]["calculation"]["raw_delta"] == 0.25
    delivered = [m for m in client.requests[1]["messages"] if m["role"] == "tool"]
    assert len(delivered) == 1 and json.loads(delivered[0]["content"]) == payload
    events = observed.run.events
    assert [e.sequence for e in events] == list(range(len(events)))
    starts = [e for e in events if e.kind == "tool.execution.started"]
    accepted = [e for e in events if e.kind == "tool.result.accepted"]
    assert len(starts) == len(accepted) == 1
    for event in starts + accepted:
        assert (event.run_id, event.tool_name, event.tool_call_id, event.tool_invocation_index) == (
            observed.run.run_id, TOOL, "comparison-call", 0)
    assert starts[0].sequence < accepted[0].sequence < events[-1].sequence
    assert events[-1].kind == "run.completed"
    assert observed.run.usage.complete is False and observed.run.usage.total_tokens is None
    assert session.history == (observed.turn,)
    assert before == {str(p): p.read_bytes() for root in (a, b) for p in root.iterdir()}


def test_dynamic_validation_metadata_supplement_does_not_modify_m16_result(tmp_path, monkeypatch):
    tool_class()
    a, b = pair(tmp_path)
    selected = target(section="validation")
    selected["a"]["metric_spec"] = selected["b"]["metric_spec"] = None
    original, captured = existing.analyze_experiment_evidence, []
    def traced(*args, **kwargs):
        result = original(*args, **kwargs)
        captured.append((result, deepcopy(result)))
        return result
    monkeypatch.setattr(existing, "analyze_experiment_evidence", traced)
    result = make_tools(a, b, targets=[selected]).invoke_tool(TOOL, arguments(a, b))
    row = result["targets"]["quality"]
    assert row["calculation"]["raw_delta"] == 0.25
    assert row["source_a"]["metadata_source"] == "host_declared"
    assert len(captured) == 2
    assert all(actual == snapshot for actual, snapshot in captured)
    assert all("metric_specs" not in actual["validation_evidence"] for actual, _ in captured)
    assert selected["a"]["metric_spec"] is None


@pytest.mark.parametrize("file_option", ["metrics_config", "test_metrics_file"])
def test_host_subfile_options_remain_experiment_relative(tmp_path, file_option):
    cls = tool_class()
    a, b = pair(tmp_path)
    ob = options()
    ob[file_option] = "../outside.csv"
    with pytest.raises(ValueError):
        tools = cls(str(a), str(b), options_a=options(), options_b=ob, targets=[target()])
        tools.invoke_tool(TOOL, arguments(a, b))


def test_provider_failure_keeps_original_error_history_and_binding_semantics(tmp_path, monkeypatch):
    tool_class()
    a, b = pair(tmp_path)
    reads = []
    original = existing.analyze_experiment_evidence
    def traced(*args, **kwargs):
        result = original(*args, **kwargs)
        reads.append(args[0])
        return result
    monkeypatch.setattr(existing, "analyze_experiment_evidence", traced)
    error = RuntimeError("offline final provider sentinel")
    client = Client(arguments(a, b), error=error)
    session = CopilotSession(client, model="offline", experiment_context={"experiment_root": str(tmp_path)})
    failures = []
    with pytest.raises(RuntimeError) as caught:
        with bound_tools(BoundToolCollection(make_tools(a, b))):
            session.ask_with_observability("compare", on_failure=failures.append)
    assert caught.value is error and len(reads) == 2
    assert session.history == () and len(client.requests) == 2
    assert len(failures) == 1 and failures[0].tool_invocation_count == 1
    assert failures[0].run.events[-1].kind == "run.failed"
    fresh = Client(arguments(a, b))
    unbound = CopilotSession(fresh, model="offline", experiment_context={"experiment_root": str(tmp_path)})
    with pytest.raises(KeyError):
        unbound.ask_with_observability("not bound")
    assert fresh.requests[0]["tools"] == EXPECTED_PROVIDER_TOOLS


def test_invalid_csv_completes_with_validation_preserved_and_no_available_claim(tmp_path):
    tool_class()
    a, b = pair(tmp_path)
    (b / "heldout.csv").write_text("measurement\nNaN\n", encoding="utf-8")
    selected = [target(), target("validation", section="validation")]
    client = Client(arguments(a, b))
    session = CopilotSession(client, model="offline", experiment_context={"experiment_root": str(tmp_path)})
    with bound_tools(BoundToolCollection(make_tools(a, b, targets=selected))):
        observed = session.ask_with_observability("compare")
    payload = json.loads(observed.turn.tool_invocations[0].result_json)
    test_row = payload["targets"]["quality"]
    assert test_row["availability_b"] == "invalid"
    assert_blocked(test_row)
    assert payload["targets"]["validation"]["calculation"]["raw_delta"] == 0.25
    assert observed.run.events[-1].kind == "run.completed" and session.turn_count == 1


def test_legacy_compare_signature_sort_diagnostics_partial_failure_unchanged(tmp_path):
    assert str(inspect.signature(existing.compare_experiments)) == (
        "(experiment_root: str, sort_by: str | None = None, descending: bool = True, *, "
        "metrics_config: str | None = None, include_diagnostics: bool = False) -> dict")
    a, b = pair(tmp_path)
    broken = write_experiment(tmp_path, "broken")
    (broken / "history.json").write_text("not json", encoding="utf-8")
    result = existing.compare_experiments(str(tmp_path), metrics_config=str(a / "metrics.yaml"),
                                          include_diagnostics=True)
    assert result["experiment_counts"] == {"total": 3, "successful": 2, "failed": 1}
    assert [r["experiment_name"] for r in result["comparison_records"]] == ["b", "a"]
    assert "diagnostics" in result and "targets" not in result
    ascending = existing.compare_experiments(str(tmp_path), descending=False,
                                             metrics_config=str(a / "metrics.yaml"))
    assert [r["experiment_name"] for r in ascending["comparison_records"]] == ["a", "b"]

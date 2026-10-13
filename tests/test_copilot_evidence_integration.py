"""M16 RED: ExperimentEvidenceTools(experiment_dir, *, config_paths=(),
test_metrics_file=None, test_metric_specs=(), metrics_config=None).

The host set lives in tool_layer.experiment_tools. It advertises only
analyze_experiment_evidence(experiment_dir, include_diagnostics=False).
Host options are immutable snapshots and absent from model schema. A real
Session/Adapter/Binding and real readers execute; only Provider is a double.
Capability-only metadata must protect this bound-only name before dispatch.
"""

from copy import deepcopy
import json
import os
from pathlib import Path
import subprocess
from types import SimpleNamespace as NS

import pytest

from copilot.session import CopilotSession
from llm_adapters.tool_binding import BoundToolCollection, bound_tools
from tool_layer import tool_registry as registry
from tool_layer.experiment_path_security import _build_experiment_path_policy
from test_tool_capability_metadata import EXPECTED_PROVIDER_TOOLS
from test_experiment_evidence_tools import api, experiment
from test_test_metrics_csv import spec, OpenProbe


TOOL = "analyze_experiment_evidence"


class Client:
    def __init__(self, arguments, *, tool=TOOL, final_error=None):
        self.arguments, self.tool, self.final_error = arguments, tool, final_error
        self.requests = []
        self.chat = NS(completions=self)

    def create(self, **kwargs):
        self.requests.append(deepcopy(kwargs))
        if len(self.requests) == 1:
            message = NS(role="assistant", content=None, tool_calls=[NS(
                id="evidence-call", type="function", function=NS(
                    name=self.tool, arguments=json.dumps(self.arguments)))])
        elif len(self.requests) == 2:
            if self.final_error is not None:
                raise self.final_error
            message = NS(role="assistant", content="offline answer", tool_calls=None)
        else:
            raise AssertionError("unexpected provider retry")
        return NS(choices=[NS(message=message)])


def tools_for(root, **options):
    return api("ExperimentEvidenceTools")(str(root), **options)


def test_default_registry_and_unbound_dispatch_remain_frozen():
    assert registry.list_tools() == EXPECTED_PROVIDER_TOOLS
    with pytest.raises(KeyError):
        registry.invoke_tool(TOOL, {"experiment_dir": "not-read"})


def test_optional_tool_has_explicit_private_path_capability():
    descriptors = registry._workspace_paths_for_tool(TOOL)
    assert [(d.parameter_name, d.path_kind) for d in descriptors] == [
        ("experiment_dir", "directory")], "missing M16 capability: evidence tool workspace metadata"
    assert registry.list_tools() == EXPECTED_PROVIDER_TOOLS


def test_path_policy_for_new_name_rejects_before_dispatch_without_binding(tmp_path):
    root = tmp_path / "root"
    outside = tmp_path / "outside"
    root.mkdir()
    outside.mkdir()
    policy = _build_experiment_path_policy({"experiment_dir": str(root)})
    with pytest.raises(ValueError):
        policy(TOOL, {"experiment_dir": str(outside)})


def test_schema_binding_snapshot_and_collisions(tmp_path):
    cls = api("ExperimentEvidenceTools")
    root = experiment(tmp_path)
    selected = [["component_enabled"]]
    specs = [spec()]
    tools = cls(str(root), config_paths=selected, test_metrics_file="heldout.csv", test_metric_specs=specs)
    selected[0][0] = "private"
    specs.clear()
    schema = tools.list_tools()
    assert len(schema) == 1 and schema[0]["function"]["name"] == TOOL
    parameters = schema[0]["function"]["parameters"]
    assert parameters == {"type": "object", "properties": {
        "experiment_dir": {"type": "string"},
        "include_diagnostics": {"type": "boolean", "default": False}},
        "required": ["experiment_dir"], "additionalProperties": False}
    collection = BoundToolCollection(tools)
    assert collection.list_tools() == EXPECTED_PROVIDER_TOOLS + schema
    with pytest.raises(ValueError):
        BoundToolCollection(tools, tools)
    schema[0]["function"]["parameters"]["properties"].clear()
    assert tools.list_tools()[0]["function"]["parameters"]["properties"]
    result = collection.invoke_tool(TOOL, {"experiment_dir": str(root)})
    assert result["configuration_evidence"]["fields"][0]["path"] == ["component_enabled"]
    assert result["test_results"]["status"] == "available"
    assert registry.list_tools() == EXPECTED_PROVIDER_TOOLS


def test_real_session_adapter_binding_metrics_and_correlated_events(tmp_path):
    api("ExperimentEvidenceTools")
    root = experiment(tmp_path)
    tools = tools_for(root, config_paths=[("optimizer", "base_lr")],
                      test_metrics_file="heldout.csv", test_metric_specs=[spec()])
    collection = BoundToolCollection(tools)
    client = Client({"experiment_dir": str(root), "include_diagnostics": True})
    session = CopilotSession(client, model="offline", experiment_context={"experiment_dir": str(root)})
    with bound_tools(collection):
        result = session.ask_with_observability("read evidence")
    assert client.requests[0]["tools"] == collection.list_tools()
    assert len(client.requests) == result.metrics.provider_request_count == 2
    assert result.metrics.tool_invocation_count == len(result.turn.tool_invocations) == 1
    invocation = result.turn.tool_invocations[0]
    assert invocation.tool_name == TOOL and invocation.tool_call_id == "evidence-call"
    payload = json.loads(invocation.result_json)
    assert payload["test_results"]["metrics"]["score"]["value"] == 0.7
    assert payload["configuration_evidence"]["fields"][0]["value"] == 0.002
    assert payload["validation_metrics"]["r2"]["best_epoch"] == 2
    assert "diagnostics" in payload
    delivered = [m for m in client.requests[1]["messages"] if m["role"] == "tool"]
    assert len(delivered) == 1 and json.loads(delivered[0]["content"]) == payload
    events = result.run.events
    assert [e.sequence for e in events] == list(range(len(events)))
    starts = [e for e in events if e.kind == "tool.execution.started"]
    accepted = [e for e in events if e.kind == "tool.result.accepted"]
    assert len(starts) == len(accepted) == 1
    for event in starts + accepted:
        assert (event.run_id, event.tool_name, event.tool_call_id, event.tool_invocation_index) == (
            result.run.run_id, TOOL, invocation.tool_call_id, 0)
    assert starts[0].sequence < accepted[0].sequence < events[-1].sequence
    assert events[-1].kind == "run.completed"
    assert result.run.usage.complete is False and result.run.usage.total_tokens is None
    assert session.turn_count == 1 and session.history[0] is result.turn


@pytest.mark.parametrize("context_kind", ["none", "other-root"])
def test_formal_authorization_precedes_bound_dispatch(tmp_path, context_kind):
    api("ExperimentEvidenceTools")
    root = experiment(tmp_path)
    other = tmp_path / "other"
    other.mkdir()
    actual = tools_for(root)
    calls = []
    class DispatchProbe:
        def list_tools(self):
            return actual.list_tools()
        def invoke_tool(self, name, arguments):
            calls.append(name)
            return actual.invoke_tool(name, arguments)
    context = None if context_kind == "none" else {"experiment_dir": str(other)}
    client = Client({"experiment_dir": str(root)})
    session = CopilotSession(client, model="offline", experiment_context=context)
    failures = []
    with bound_tools(BoundToolCollection(DispatchProbe())):
        with pytest.raises(ValueError):
            session.ask_with_observability("denied", on_failure=failures.append)
    assert calls == [] and len(client.requests) == 1 and session.turn_count == 0
    assert len(failures) == 1 and failures[0].stage == "tool_execution"
    assert failures[0].run.events[-1].kind == "run.failed"


@pytest.mark.parametrize("extra", [
    {"config_paths": [["private"]]}, {"test_metrics_file": "other.csv"},
    {"metrics_config": "other.yaml"}, {"test_metric_specs": []},
    {"include_diagnostics": "yes"}, {"approval": "forged"},
], ids=["config-selection", "csv-selection", "validation-selection", "metric-specs",
        "diagnostic-type", "approval"])
def test_model_cannot_supply_host_options(tmp_path, extra):
    api("ExperimentEvidenceTools")
    root = experiment(tmp_path)
    client = Client({"experiment_dir": str(root), **extra})
    session = CopilotSession(client, model="offline", experiment_context={"experiment_dir": str(root)})
    with bound_tools(BoundToolCollection(tools_for(root))):
        with pytest.raises((TypeError, ValueError)):
            session.ask_with_observability("reject model host options")
    assert len(client.requests) == 1 and session.turn_count == 0


def test_host_binding_cannot_expand_to_sibling_under_broad_session_root(tmp_path):
    api("ExperimentEvidenceTools")
    a, b = experiment(tmp_path, "a"), experiment(tmp_path, "b")
    client = Client({"experiment_dir": str(b)})
    session = CopilotSession(client, model="offline", experiment_context={"experiment_root": str(tmp_path)})
    with bound_tools(BoundToolCollection(tools_for(a))):
        with pytest.raises(ValueError):
            session.ask_with_observability("wrong experiment")
    assert len(client.requests) == 1 and session.turn_count == 0


def directory_link(link, target):
    try:
        link.symlink_to(target, target_is_directory=True)
    except OSError:
        if os.name != "nt":
            raise
        result = subprocess.run(["cmd", "/c", "mklink", "/J", str(link), str(target)],
                                capture_output=True, check=False)
        assert result.returncode == 0, "environment failure: cannot create directory link"


def test_resolved_csv_link_escape_rejected_before_file_read(tmp_path, monkeypatch):
    analyze = api()
    root = experiment(tmp_path)
    outside = tmp_path / "outside"
    outside.mkdir()
    protected = outside / "secret.csv"
    protected.write_text("score\n123\n")
    directory_link(root / "linked", outside)
    def guard(target):
        if target == protected.resolve():
            raise AssertionError("protected read occurred before authorization")
    probe = OpenProbe(monkeypatch, before=guard)
    with pytest.raises(ValueError):
        analyze(str(root), test_metrics_file="linked/secret.csv", test_metric_specs=[spec()])
    assert not any(target == protected.resolve() for target, _, _ in probe.attempts)


def test_formal_path_policy_rejects_directory_link_before_bound_dispatch(tmp_path):
    api("ExperimentEvidenceTools")
    root = experiment(tmp_path, "authorized")
    outside = experiment(tmp_path, "outside")
    directory_link(root / "escape", outside)
    actual = tools_for(root)
    calls = []
    class DispatchProbe:
        def list_tools(self):
            return actual.list_tools()
        def invoke_tool(self, name, arguments):
            calls.append(name)
            return actual.invoke_tool(name, arguments)
    client = Client({"experiment_dir": str(root / "escape")})
    session = CopilotSession(client, model="offline", experiment_context={"experiment_dir": str(root)})
    with bound_tools(BoundToolCollection(DispatchProbe())):
        with pytest.raises(ValueError):
            session.ask_with_observability("reject linked outside directory")
    assert calls == [] and len(client.requests) == 1 and session.turn_count == 0


def test_final_provider_failure_rolls_back_history_and_restores_binding(tmp_path):
    api("ExperimentEvidenceTools")
    root = experiment(tmp_path)
    error = RuntimeError("offline final answer failure")
    client = Client({"experiment_dir": str(root)}, final_error=error)
    session = CopilotSession(client, model="offline", experiment_context={"experiment_dir": str(root)})
    failures = []
    with pytest.raises(RuntimeError) as caught:
        with bound_tools(BoundToolCollection(tools_for(root))):
            session.ask_with_observability("fail", on_failure=failures.append)
    assert caught.value is error and session.history == ()
    assert len(client.requests) == 2 and failures[0].tool_invocation_count == 1
    fresh = Client({"experiment_dir": str(root)})
    unbound = CopilotSession(fresh, model="offline", experiment_context={"experiment_dir": str(root)})
    with pytest.raises(KeyError):
        unbound.ask_with_observability("not bound")
    assert fresh.requests[0]["tools"] == EXPECTED_PROVIDER_TOOLS


def test_runtime_completion_does_not_claim_invalid_csv_is_available(tmp_path):
    api("ExperimentEvidenceTools")
    root = experiment(tmp_path, csv="score\nNaN\n")
    client = Client({"experiment_dir": str(root)})
    session = CopilotSession(client, model="offline", experiment_context={"experiment_dir": str(root)})
    with bound_tools(BoundToolCollection(tools_for(root, test_metrics_file="heldout.csv",
                                                 test_metric_specs=[spec()]))):
        result = session.ask_with_observability("report invalid test evidence")
    payload = json.loads(result.turn.tool_invocations[0].result_json)
    assert payload["test_results"]["status"] == "invalid"
    assert payload["test_results"]["metrics"] == {}
    assert payload["validation_metrics"]["r2"]["best_value"] == 0.5
    assert result.run.events[-1].kind == "run.completed"
    assert session.turn_count == 1

"""M15 Slice 3A RED: optional tools borrowing the frozen host controller.

Proposed API: WorkflowTools(scope, controller, *, identity_provider,
approval_provider), list_tools(), invoke_tool(name, arguments).
scope is a trusted WorkflowScope; controller remains the sole authority even
if a host accidentally binds a different scope. identity_provider() supplies
(action_id, idempotency_key) for the host's current intent; approval_provider
(action_id) supplies the controller's original grant or None. Neither callback
is model-selected. The host owns their lifetime, storage and replay policy.

Model operations (in descriptor order):
  propose_simulated_action(target, arguments, preconditions)
  get_simulated_action(action_id)
  simulate_approved_action(action_id)
Proposal's action_type/contract_version are fixed simulate/v1; scope fields
come from the host. Unknown options -> unsupported_option; missing fields,
non-object requests and invalid action IDs -> invalid_arguments. ActionSpec
and Controller errors retain their original codes (including invalid_action).

Envelope version is integer 1. Successful reads/proposals/executions have
ok=True and exactly the six safe data fields asserted by _projected. The
requires_approval flag describes policy, not whether a grant is available.
Failed/stale/rejected/uncertain records have the same data and ok=False with
error.code == result_code; pre-record rejections contain only error.code.
No spec/value, host identity, key, grant, transitions or callback data is
projected. Each serialized envelope is <=2048 UTF-8 bytes. Unexpected errors
propagate; only designated WorkflowError and SimulationFailure semantics may
be converted. These are in-memory simulation contracts, not durable execution.
"""

import builtins
from copy import deepcopy
from dataclasses import replace
import importlib
import importlib.util
import io
import json
import os
from pathlib import Path
import socket
import sqlite3
import subprocess
import uuid

import pytest

from copilot.controlled_workflow import (
    ActionSpec, WorkflowController, WorkflowScope, WorkflowError, SimulationFailure,
)


PROPOSE = "propose_simulated_action"
GET = "get_simulated_action"
SIMULATE = "simulate_approved_action"
WORKFLOW = (PROPOSE, GET, SIMULATE)
CALLER = {"principal": "host-user", "context_id": "context-a"}
ACTION = {"action_id": "action-a"}
SECRET = "SECRET-host-only-C:\\private\\credential"


def _api():
    module = "tool_layer.workflow_tools"
    assert importlib.util.find_spec(module) is not None, (
        "missing M15 workflow tool capability: tool_layer.workflow_tools"
    )
    api = importlib.import_module(module)
    assert callable(getattr(api, "WorkflowTools", None)), "missing WorkflowTools API"
    return api


def _arguments(**changes):
    return {"target": "target-a", "arguments": {"value": {"labels": ["literal"]}},
            "preconditions": {"target_version": "v1", "expected": {"enabled": True}},
            **changes}


def _scope(**changes):
    return WorkflowScope(**{
        "scope_id": "scope-a", "policy_version": "policy-v1", **CALLER,
        "allowed_action_types": ["simulate"], "allowed_targets": ["target-a"],
        **changes,
    })


def _spec(arguments=None):
    return ActionSpec(contract_version="v1", action_type="simulate",
                      scope_id="scope-a", policy_version="policy-v1",
                      **(_arguments() if arguments is None else arguments))


class _Host:
    """Host data/identity doubles; every lifecycle decision uses real Controller."""

    def __init__(self, *, scope=None, failure=None):
        self.scope = _scope() if scope is None else scope
        self.now = 10.0
        self.target = deepcopy(_arguments()["preconditions"])
        self.reads, self.calls, self.issues, self.approval_reads = [], [], [], []
        self.identity = ("action-a", "key-a")
        self.grants = {}
        self.failure = failure
        self.controller = WorkflowController(
            self.scope, target_provider=self.read, clock=lambda: self.now,
            simulator=self.simulate,
        )

    def read(self, target):
        self.reads.append(target)
        return deepcopy(self.target)

    def simulate(self, spec):
        self.calls.append(spec)
        if self.failure is not None:
            raise self.failure
        return spec.arguments

    def issue(self):
        self.issues.append(self.identity)
        return self.identity

    def approval(self, action_id):
        self.approval_reads.append(action_id)
        return self.grants.get(action_id)

    def tools(self, api, *, scope=None):
        return api.WorkflowTools(
            self.scope if scope is None else scope, self.controller,
            identity_provider=self.issue, approval_provider=self.approval,
        )

    def propose(self, arguments=None):
        return self.controller.propose(_spec(arguments), action_id=self.identity[0],
                                       idempotency_key=self.identity[1], **CALLER)

    def approve(self, action_id="action-a"):
        grant = self.controller.approve(action_id, approval_id="approval-" + action_id,
                                         expires_at=20.0, **CALLER)
        self.grants[action_id] = grant
        return grant

    def record(self, action_id="action-a"):
        return self.controller.get(action_id, **CALLER)


def _error(result, code):
    assert result == {"format_version": 1, "ok": False, "error": {"code": code}}


def _projected(result, *, state="proposed", code=None, fingerprint=None, action_id="action-a"):
    expected = {"format_version": 1, "ok": state not in {"failed", "uncertain", "stale", "rejected"},
                "data": {"action_id": action_id, "fingerprint": _spec().fingerprint if fingerprint is None else fingerprint,
                         "state": state, "execution_mode": "simulation", "requires_approval": True,
                         "result_code": code}}
    if not expected["ok"]:
        expected["error"] = {"code": code}
    assert result == expected
    encoded = json.dumps(result, ensure_ascii=False, allow_nan=False, sort_keys=True,
                         separators=(",", ":")).encode("utf-8")
    assert len(encoded) <= 2048 and SECRET.encode() not in encoded


def test_descriptions_are_strict_deterministic_detached_and_host_free():
    api = _api()
    host = _Host()
    tools = host.tools(api)
    definitions = tools.list_tools()
    assert tuple(item["function"]["name"] for item in definitions) == WORKFLOW
    assert definitions == tools.list_tools() == host.tools(api).list_tools()
    for definition, fields in zip(definitions, (("target", "arguments", "preconditions"),
                                                ("action_id",), ("action_id",))):
        assert definition["type"] == "function"
        function = definition["function"]
        assert "simulat" in function["description"].lower()
        schema = function["parameters"]
        assert schema["type"] == "object" and schema["additionalProperties"] is False
        assert set(schema["properties"]) == set(schema["required"]) == set(fields)
        identifier = schema["properties"]["target" if fields[0] == "target" else "action_id"]
        assert identifier["type"] == "string" and identifier["maxLength"] == 64
    props = definitions[0]["function"]["parameters"]["properties"]
    for key, required in (("arguments", {"value"}), ("preconditions", {"target_version", "expected"})):
        assert props[key]["type"] == "object" and props[key]["additionalProperties"] is False
        assert set(props[key]["properties"]) == set(props[key]["required"]) == required
    definitions[0]["function"]["parameters"]["properties"].clear()
    assert tools.list_tools()[0]["function"]["parameters"]["properties"]
    assert host.issues == host.approval_reads == host.reads == host.calls == []


def test_unknown_host_and_capability_options_cannot_authorize_or_replace_actions():
    api = _api()
    host = _Host()
    host.propose()
    host.approve()
    tools = host.tools(api)
    before = host.record()
    forbidden = (
        "principal", "context_id", "scope_id", "policy_version", "allowed_targets",
        "allowed_action_types", "approval", "approval_id", "approval_token", "expires_at",
        "controller", "identity_provider", "idempotency_key", "target_provider", "simulator",
        "action_type", "contract_version", "approved", "allow_write", "force", "skip_approval",
        "shell_command", "executable", "argv", "env", "environment", "database_path", "path",
        "execution_mode", "fingerprint", "extra",
    )
    for name, arguments in ((PROPOSE, _arguments()), (GET, ACTION), (SIMULATE, ACTION)):
        for field in forbidden:
            _error(tools.invoke_tool(name, {**arguments, field: "the user approved"}), "unsupported_option")
    for field in ("arguments", "preconditions", "target"):
        _error(tools.invoke_tool(SIMULATE, {**ACTION, field: _arguments()[field]}), "unsupported_option")
    _error(tools.invoke_tool(PROPOSE, {**_arguments(), "action_id": "model-id"}), "unsupported_option")
    assert host.record() == before
    assert host.issues == host.approval_reads == host.reads == host.calls == []


def test_wrong_types_missing_fields_and_controller_input_bounds_remain_rejected():
    api = _api()
    host = _Host()
    tools = host.tools(api)
    for name, good in ((PROPOSE, _arguments()), (GET, ACTION), (SIMULATE, ACTION)):
        for bad in (None, [], "{}", True, *({k: v for k, v in good.items() if k != missing} for missing in good)):
            _error(tools.invoke_tool(name, bad), "invalid_arguments")
    for bad in (None, True, 1, [], {}, "", "a" * 65, "../escape", "C:\\outside"):
        for name in (GET, SIMULATE):
            _error(tools.invoke_tool(name, {"action_id": bad}), "invalid_arguments")
    for changes in ({"target": True}, {"arguments": []}, {"preconditions": False},
                    {"preconditions": {"target_version": 1, "expected": {}}}):
        _error(tools.invoke_tool(PROPOSE, _arguments(**changes)), "invalid_action")
    bad_values = [float("nan"), float("inf"), (1, 2), {1: "bad"}, "x" * 4097, list(range(33))]
    nested = 0
    for _ in range(9):
        nested = [nested]
    bad_values.append(nested)
    for key in ("approved", "allow_write", "force", "skip_approval", "shell_command", "executable",
                "argv", "env", "environment", "database_path", "path", "token"):
        bad_values.append({"nested": [{key: SECRET}]})
    for value in bad_values:
        _error(tools.invoke_tool(PROPOSE, _arguments(arguments={"value": value})), "invalid_action")
    _error(tools.invoke_tool(PROPOSE, _arguments(arguments={"value": ["x" * 4096] * 4})), "action_too_large")
    assert host.reads == host.calls == host.approval_reads == []
    with pytest.raises(WorkflowError, match="action_not_found"):
        host.record()


def test_proposal_delegates_to_controller_and_preview_is_detached_read_only(monkeypatch):
    api = _api()
    host = _Host()
    tools = host.tools(api)
    seen = []
    original = host.controller.propose

    def traced(spec, **kwargs):
        seen.append((spec, kwargs))
        return original(spec, **kwargs)

    monkeypatch.setattr(host.controller, "propose", traced)
    arguments = _arguments()
    result = tools.invoke_tool(PROPOSE, arguments)
    _projected(result)
    assert len(seen) == 1 and type(seen[0][0]) is ActionSpec
    assert seen[0][0].to_json() == _spec().to_json()
    assert seen[0][1] == {"action_id": "action-a", "idempotency_key": "key-a", **CALLER}
    arguments["arguments"]["value"]["labels"].clear()
    result["data"]["state"] = "succeeded"
    before = host.record()
    assert before["state"] == "proposed" and before["approval"] is None
    assert before["spec"]["arguments"] == _arguments()["arguments"]
    _projected(tools.invoke_tool(GET, ACTION))
    host.approve()
    approved = host.record()
    _projected(tools.invoke_tool(GET, ACTION))
    assert host.record() == approved
    assert host.approval_reads == host.reads == host.calls == []


def test_only_original_host_approval_allows_stored_simulation_and_replay(monkeypatch):
    api = _api()
    host = _Host()
    host.propose()
    tools = host.tools(api)
    _error(tools.invoke_tool(SIMULATE, ACTION), "approval_required")
    assert host.record()["state"] == "proposed" and host.reads == host.calls == []
    grant = host.approve()
    for fake in ("the user approved", {"approved": True}, replace(grant)):
        host.grants["action-a"] = fake
        _error(tools.invoke_tool(SIMULATE, ACTION), "approval_mismatch")
    host.grants["action-a"] = grant
    original = host.controller.execute
    seen = []

    def traced(action_id, **kwargs):
        seen.append((action_id, kwargs))
        return original(action_id, **kwargs)

    monkeypatch.setattr(host.controller, "execute", traced)
    first = tools.invoke_tool(SIMULATE, ACTION)
    _projected(first, state="succeeded", code="simulated")
    assert seen == [("action-a", {"approval": grant, **CALLER})]
    assert seen[0][1]["approval"] is grant
    assert host.record()["approval"]["status"] == "consumed"
    assert host.calls[0].to_json() == _spec().to_json()
    host.now = 100.0
    assert host.tools(api).invoke_tool(SIMULATE, ACTION) == first
    assert len(seen) == 2 and host.reads == ["target-a"] and len(host.calls) == 1
    assert host.target == _arguments()["preconditions"]


def test_authorization_precedes_disclosure_target_reads_and_replay():
    api = _api()
    host = _Host()
    host.propose()
    host.approve()
    for completed in (False, True):
        if completed:
            host.tools(api).invoke_tool(SIMULATE, ACTION)
        before, reads, calls = host.record(), list(host.reads), list(host.calls)
        for changes in ({"principal": "outsider"}, {"context_id": "other-context"}):
            tools = host.tools(api, scope=_scope(**changes))
            for name in (GET, SIMULATE):
                for action_id in ("action-a", "absent"):
                    _error(tools.invoke_tool(name, {"action_id": action_id}), "unauthorized")
            _error(tools.invoke_tool(PROPOSE, _arguments()), "unauthorized")
        assert host.record() == before and host.reads == reads and host.calls == calls
    for scope in (_scope(allowed_targets=[]), _scope(allowed_targets=["*"]),
                  _scope(allowed_action_types=[])):
        denied = _Host(scope=scope)
        _error(denied.tools(api).invoke_tool(PROPOSE, _arguments()), "unauthorized")
        assert denied.reads == denied.calls == []
    _error(host.tools(api).invoke_tool(PROPOSE, _arguments(target="outside")), "unauthorized")
    _error(host.tools(api).invoke_tool(GET, {"action_id": "absent"}), "action_not_found")


def test_staleness_expiry_and_host_rejection_preserve_controller_lifecycle():
    api = _api()
    for mode in ("stale", "expiry", "rejected"):
        host = _Host()
        host.propose()
        host.approve()
        tools = host.tools(api)
        if mode == "expiry":
            before = host.record()
            host.now = 20.0
            _error(tools.invoke_tool(SIMULATE, ACTION), "approval_expired")
            assert host.record() == before and host.reads == []
        else:
            if mode == "stale":
                host.target["target_version"] = "v2"
                code = "stale_precondition"
            else:
                host.controller.reject("action-a", **CALLER)
                code = "action_rejected"
            result = tools.invoke_tool(SIMULATE, ACTION)
            _projected(result, state=mode, code=code)
            assert tools.invoke_tool(SIMULATE, ACTION) == result
            assert tools.invoke_tool(GET, ACTION) == result
            assert host.reads == (["target-a"] if mode == "stale" else [])
        assert host.calls == [] and host.record()["approval"]["status"] == "available"


def test_host_idempotency_conflicts_capacity_and_controller_reuse():
    api = _api()
    host = _Host(scope=_scope(max_actions=1))
    tools = host.tools(api)
    first = tools.invoke_tool(PROPOSE, _arguments())
    host.identity = ("candidate", "key-a")
    assert host.tools(api).invoke_tool(PROPOSE, _arguments()) == first
    _error(tools.invoke_tool(PROPOSE, _arguments(arguments={"value": "changed"})), "idempotency_conflict")
    host.identity = ("action-a", "different-key")
    _error(tools.invoke_tool(PROPOSE, _arguments()), "action_id_conflict")
    host.identity = ("action-b", "key-b")
    _error(host.tools(api).invoke_tool(PROPOSE, _arguments()), "capacity_exceeded")
    host.approve()
    tools.invoke_tool(SIMULATE, ACTION)
    _error(host.tools(api).invoke_tool(PROPOSE, _arguments()), "capacity_exceeded")
    host.identity = ("candidate", "key-a")
    _projected(host.tools(api).invoke_tool(PROPOSE, _arguments()), state="succeeded", code="simulated")
    isolated = _Host()
    _error(isolated.tools(api).invoke_tool(GET, ACTION), "action_not_found")
    assert len(host.calls) == 1 and isolated.calls == []


def test_known_simulation_failure_is_safe_failed_evidence_without_retry():
    api = _api()
    host = _Host(failure=SimulationFailure(SECRET))
    host.propose()
    host.approve()
    result = host.tools(api).invoke_tool(SIMULATE, ACTION)
    _projected(result, state="failed", code="simulation_failed")
    assert host.tools(api).invoke_tool(SIMULATE, ACTION) == result
    assert host.tools(api).invoke_tool(GET, ACTION) == result
    assert host.record()["result"] == {"code": "simulation_failed", "data": {}}
    assert len(host.calls) == 1


@pytest.mark.parametrize("exception_type", [ValueError, KeyboardInterrupt, SystemExit],
                         ids=["programmer", "interrupt", "exit"])
def test_unexpected_errors_propagate_and_uncertain_replay_is_sanitized(exception_type):
    api = _api()
    error = exception_type(SECRET)
    host = _Host(failure=error)
    host.propose()
    host.approve()
    with pytest.raises(exception_type) as captured:
        host.tools(api).invoke_tool(SIMULATE, ACTION)
    assert captured.value is error
    result = host.tools(api).invoke_tool(GET, ACTION)
    _projected(result, state="uncertain", code="simulation_uncertain")
    assert host.tools(api).invoke_tool(SIMULATE, ACTION) == result and len(host.calls) == 1


def test_projection_withholds_large_literal_payloads_grants_and_internal_records():
    api = _api()
    host = _Host()
    arguments = _arguments(arguments={"value": [SECRET, "x" * 4096, "y" * 4096]})
    fingerprint = _spec(arguments).fingerprint
    tools = host.tools(api)
    _projected(tools.invoke_tool(PROPOSE, arguments), fingerprint=fingerprint)
    host.approve()
    result = tools.invoke_tool(SIMULATE, ACTION)
    _projected(result, state="succeeded", code="simulated", fingerprint=fingerprint)
    assert host.record()["result"]["data"] == arguments["arguments"]
    result["data"].clear()
    _projected(tools.invoke_tool(GET, ACTION), state="succeeded", code="simulated", fingerprint=fingerprint)


def test_tools_remain_simulation_only_without_external_resources_or_id_generation(monkeypatch):
    api = _api()
    from experiment_repository import ExperimentRepository

    host = _Host()
    attempts = []
    environment = dict(os.environ)

    def forbidden(*args, **kwargs):
        attempts.append("external capability")
        raise AssertionError("workflow tools must borrow host resources only")

    with monkeypatch.context() as guard:
        for owner, names in (
            (builtins, ("open",)), (io, ("open",)),
            (Path, ("open", "write_text", "write_bytes", "mkdir", "touch", "unlink", "rename", "replace")),
            (os, ("open", "write", "mkdir", "remove", "unlink", "rename", "replace", "system", "popen", "putenv")),
            (subprocess, ("Popen", "run", "call", "check_call", "check_output")),
            (socket, ("socket", "create_connection")), (sqlite3, ("connect",)),
            (ExperimentRepository, ("__init__", "add", "invalidate", "delete")),
            (uuid, ("uuid1", "uuid4")), (WorkflowController, ("__init__",)),
        ):
            for name in names:
                guard.setattr(owner, name, forbidden)
        tools = host.tools(api)
        tools.invoke_tool(PROPOSE, _arguments())
        host.approve()
        result = tools.invoke_tool(SIMULATE, ACTION)
        _projected(result, state="succeeded", code="simulated")
        for name in ("approve", "approve_action", "write_file", "launch_training", "execute_shell"):
            with pytest.raises(KeyError):
                tools.invoke_tool(name, {})
    assert attempts == [] and dict(os.environ) == environment
    assert host.target == _arguments()["preconditions"] and len(host.calls) == 1

"""M15 Slice 1 RED: one-instance, simulation-only action/approval contracts.

Public API frozen here (no production substitute lives in these tests):
  ActionSpec(**the seven _payload fields), to_json(), from_json(text),
    fingerprint; arguments/preconditions are detached JSON projections.
  WorkflowScope(scope_id, policy_version, principal, context_id, *,
    allowed_action_types=(), allowed_targets=(), execution_mode="simulation",
    max_actions=4, max_action_bytes=16384). Read-only host configuration.
  WorkflowController(scope, *, target_provider, clock, simulator=None).
    propose(spec, *, action_id, idempotency_key, principal, context_id)
    approve(action_id, *, approval_id, expires_at, principal, context_id)
    execute(action_id, *, approval=None, principal, context_id)
    get(action_id, *, principal, context_id)
    reject(action_id, *, principal, context_id)
  WorkflowError(code): safe business rejection, str(error) == error.code.
  SimulationFailure: designated simulator failure; message is never exposed.

All controller methods except approve return detached JSON record snapshots.
approve is a TRUSTED HOST API, absent from model proposal schemas. It returns
a frozen dataclass ApprovalEvidence with approval_id, action_id, fingerprint,
scope_id, policy_version, principal, context_id, expires_at. Only the issuing
controller's registered evidence object is accepted; serialized lookalikes,
copies, changed dataclasses and evidence from another instance are not grants.
This is a host-library boundary, not authentication of arbitrary Python code.
One approval per action; reissuance is rejected, so approval storage is bounded.

Spec v1 has exactly _payload's fields. arguments has exactly 'value';
preconditions has exactly 'target_version' and 'expected'. value/expected are
finite JSON data, not executable instructions. Reserved capability/credential
keys are rejected recursively. All identifiers use [A-Za-z0-9_-]{1,64}; scope
allowlist entries additionally allow literal '*', never a wildcard. Strings
inside JSON data preserve literal content, at most 4096 characters; containers
at most 32 members and at most 8 nested containers along any path below value
or expected. One target and one 'simulate' operation only, no paths or metadata.
Canonical JSON: sorted keys, compact separators, ensure_ascii=False,
allow_nan=False, UTF-8; lowercase SHA-256. Action ID/key/clock are NOT content.
Canonical spec limit is inclusive 16384 bytes; a scope may lower that limit.
max_actions is an integer in 1..1024; max_action_bytes in 1..16384 (not bool).

Records have exactly _RECORD_FIELDS. approval is None or a safe projection
{approval_id, status, expires_at}; status is available/consumed. result is None
or {code, data}; successes have code=simulated and data={value: ...}.
Failures/uncertainty have empty data. transitions contains {state, at} entries
from the injected finite host clock. No Runtime/Experiment identity is issued.

Authorization precedes record lookup, approval consumption, target reads and
replay results. Same key+fingerprint returns the first identity/result even if
the host supplies another candidate ID. Same key+different content conflicts;
same ID+another key conflicts. get is read-only; no separate preview API.
Expired means now >= expires_at. A terminal successful execute with its issued
approval returns the existing result, even after expiry, without another entry
or consumption. A consumed approval used for another action is rejected.

The target provider is borrowed at execute time only, returns the same shape
as spec.preconditions, and never changes a target. Reservation/approval
consumption/state change are synchronized within the controller; simulator
receives the stored ActionSpec, once. An in-flight duplicate returns executing
without waiting for the simulator; a later replay returns the terminal result.
No lock/queue outside the instance is required.
Normal lifecycle: proposed -> executing -> succeeded/failed/uncertain;
host rejection/staleness: proposed -> rejected/stale. Invalid outsider requests
do not change state. Known simulation failure becomes failed; unexpected
programmer exceptions propagate AND leave sanitized uncertain state, with no
retry. Terminal records are retained until the whole instance is discarded.
No restart recovery, durable ledger, global state or exactly-once claim.

This slice uses only in-memory host doubles. It has no Session, HTTP, provider,
filesystem or real process execution. Later integration/evaluation is deferred.
"""

import builtins
from copy import deepcopy
from dataclasses import FrozenInstanceError, replace
import importlib
import importlib.util
import io
import json
import os
from pathlib import Path
import socket
import sqlite3
import subprocess
import threading
import uuid

import pytest


_MODULE = "copilot.controlled_workflow"
_CALLER = {"principal": "host-user", "context_id": "context-a"}
_DIGEST = "ec4cf74cb44e35b4fe13f9ce4f3dc3479aeb8e60cde832c3a308d88cbd65101d"
_RECORD_FIELDS = {
    "action_id", "fingerprint", "idempotency_key", "spec", "principal",
    "context_id", "state", "execution_mode", "approval", "result", "transitions",
}
_FORBIDDEN = (
    "approved", "allow_write", "force", "skip_approval", "shell_command",
    "executable", "argv", "env", "environment", "database_path", "path",
    "cwd", "workspace", "api_key", "password", "token",
)


def _api():
    # Delayed lookup: every RED is a test-body failure, never a collection error.
    assert importlib.util.find_spec(_MODULE) is not None, (
        "missing M15 controlled workflow capability: copilot.controlled_workflow"
    )
    api = importlib.import_module(_MODULE)
    for name in ("ActionSpec", "WorkflowScope", "WorkflowController",
                 "ApprovalEvidence", "WorkflowError", "SimulationFailure"):
        assert callable(getattr(api, name, None)), f"missing M15 API: {name}"
    return api


def _payload(**changes):
    return {
        "contract_version": "v1", "action_type": "simulate", "target": "target-a",
        "arguments": {"value": {"labels": ["\u8bd5\u9a8c", " unchanged "],
                                  "enabled": True, "count": 1}},
        "preconditions": {"target_version": "v1", "expected": {"flags": [True, False]}},
        "scope_id": "scope-a", "policy_version": "policy-v1", **changes,
    }


def _canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False, allow_nan=False)


def _scope(api, **changes):
    values = dict(scope_id="scope-a", policy_version="policy-v1", **_CALLER,
                  allowed_action_types=["simulate"], allowed_targets=["target-a"],
                  execution_mode="simulation", max_actions=4, max_action_bytes=16384)
    return api.WorkflowScope(**{**values, **changes})


class _Host:
    """Only host data sources and a simulator spy, never workflow decisions."""

    def __init__(self):
        self.now = 10.0
        self.target = deepcopy(_payload()["preconditions"])
        self.reads = []
        self.calls = []

    def clock(self):
        return self.now

    def read(self, target):
        self.reads.append(target)
        return deepcopy(self.target)

    def simulate(self, spec):
        self.calls.append(spec)
        return {"value": deepcopy(spec.arguments["value"])}


def _controller(api, host, *, scope=None, simulator=None):
    return api.WorkflowController(
        _scope(api) if scope is None else scope,
        target_provider=host.read, clock=host.clock,
        simulator=host.simulate if simulator is None else simulator,
    )


def _propose(api, controller, *, action_id="action-a", key="key-a", payload=None, **caller):
    return controller.propose(
        api.ActionSpec(**(_payload() if payload is None else payload)),
        action_id=action_id, idempotency_key=key, **{**_CALLER, **caller},
    )


def _approve(controller, action_id="action-a", *, approval_id="approval-a", expires_at=20.0):
    return controller.approve(action_id, approval_id=approval_id,
                              expires_at=expires_at, **_CALLER)


def _error(api, code, call):
    with pytest.raises(api.WorkflowError) as captured:
        call()
    assert captured.value.code == code
    assert str(captured.value) == code
    return captured.value


def _record(record, state):
    assert type(record) is dict and set(record) == _RECORD_FIELDS
    assert record["state"] == state
    assert record["execution_mode"] == "simulation"
    json.dumps(record, allow_nan=False)
    assert "run_id" not in record and "execution_id" not in record


def test_action_spec_has_exact_canonical_utf8_fingerprint_and_literal_semantics():
    api = _api()
    original = _payload()
    spec = api.ActionSpec(**original)
    assert spec.to_json() == _canonical(original)
    assert spec.fingerprint == _DIGEST
    reordered = dict(reversed(list(original.items())))
    reordered["arguments"] = {"value": dict(reversed(list(original["arguments"]["value"].items())))}
    assert api.ActionSpec(**reordered).fingerprint == _DIGEST
    assert api.ActionSpec.from_json(spec.to_json()).to_json() == spec.to_json()
    for value in (
        {"labels": [" unchanged ", "\u8bd5\u9a8c"], "enabled": True, "count": 1},
        {"labels": ["\u8bd5\u9a8c", "unchanged"], "enabled": True, "count": 1},
        {"labels": ["\u8bd5\u9a8c", " unchanged "], "enabled": 1, "count": 1},
    ):
        assert api.ActionSpec(**_payload(arguments={"value": value})).fingerprint != _DIGEST
    for changes in ({"target": "target-b"}, {"scope_id": "scope-b"},
                    {"policy_version": "policy-v2"},
                    {"preconditions": {"target_version": "v2", "expected": {}}}):
        assert api.ActionSpec(**_payload(**changes)).fingerprint != _DIGEST


def test_action_parser_rejects_ambiguous_malformed_and_capability_fields():
    api = _api()
    for value in (float("nan"), float("inf"), -float("inf"), {1: "bad"},
                  (1, 2), {1, 2}, object()):
        _error(api, "invalid_action", lambda: api.ActionSpec(**_payload(arguments={"value": value})))
    cycle = []
    cycle.append(cycle)
    _error(api, "invalid_action", lambda: api.ActionSpec(**_payload(arguments={"value": cycle})))
    valid = _canonical(_payload())
    for text in ("{", "[]", valid[:-1] + ',"target":"target-b"}',
                 valid.replace('"count":1', '"count":1,"count":2'),
                 valid.replace('"count":1', '"count":NaN'),
                 _canonical(_payload(extra="not allowed")),
                 _canonical({k: v for k, v in _payload().items() if k != "target"})):
        _error(api, "invalid_action", lambda: api.ActionSpec.from_json(text))
    for field in _FORBIDDEN:
        for payload in (_payload(**{field: "forbidden"}),
                        _payload(arguments={"value": {}, field: True}),
                        _payload(arguments={"value": {field: "forbidden"}})):
            _error(api, "invalid_action", lambda: api.ActionSpec.from_json(_canonical(payload)))
    for changes in ({"contract_version": "v2"}, {"action_type": "launch"},
                    {"target": ["target-a", "target-b"]}, {"target": "../escape"},
                    {"target": "C:\\outside"}, {"target": "//host/share"},
                    {"target": "C:relative"}, {"target": "a;whoami"},
                    {"preconditions": {"target_version": "v1", "expected": {}, "extra": 1}}):
        _error(api, "invalid_action", lambda: api.ActionSpec(**_payload(**changes)))


def test_snapshots_isolate_nested_inputs_public_reads_and_approved_content():
    api = _api()
    host = _Host()
    controller = _controller(api, host)
    payload = _payload()
    spec = api.ActionSpec(**payload)
    controller.propose(spec, action_id="action-a", idempotency_key="key-a", **_CALLER)
    approval = _approve(controller)
    payload["arguments"]["value"]["labels"].append("caller mutation")
    payload["preconditions"]["expected"]["flags"].clear()
    spec.arguments["value"]["labels"].clear()
    spec.preconditions["expected"]["flags"].clear()
    with pytest.raises((FrozenInstanceError, AttributeError)):
        spec.target = "target-b"
    public = controller.get("action-a", **_CALLER)
    public["spec"]["arguments"]["value"]["labels"].clear()
    public["spec"]["preconditions"]["expected"]["flags"].clear()
    public["transitions"].clear()
    public["approval"]["status"] = "consumed"
    assert controller.get("action-a", **_CALLER)["spec"] == _payload()
    assert spec.fingerprint == approval.fingerprint == _DIGEST
    result = controller.execute("action-a", approval=approval, **_CALLER)
    assert host.calls[0].to_json() == _canonical(_payload())
    result["result"]["data"]["value"]["labels"].clear()
    assert controller.get("action-a", **_CALLER)["result"]["data"] == _payload()["arguments"]


def test_host_scope_is_independent_immutable_default_deny_and_literal_allowlist():
    api = _api()
    from tool_layer.memory_tools import MemoryAccessScope

    targets, types = ["target-a"], ["simulate"]
    scope = _scope(api, allowed_targets=targets, allowed_action_types=types)
    targets.append("target-b")
    types.append("launch")
    assert scope.allowed_targets == frozenset({"target-a"})
    assert scope.allowed_action_types == frozenset({"simulate"})
    assert not isinstance(scope, MemoryAccessScope)
    for name, value in (("principal", "attacker"), ("policy_version", "policy-v2"),
                        ("allowed_targets", frozenset({"target-b"})), ("execution_mode", "real")):
        with pytest.raises((FrozenInstanceError, AttributeError)):
            setattr(scope, name, value)
    denied_scopes = (
        api.WorkflowScope("scope-a", "policy-v1", **_CALLER),
        _scope(api, allowed_targets=[]), _scope(api, allowed_action_types=[]),
        _scope(api, allowed_targets=["*"]), _scope(api, allowed_action_types=["*"]),
    )
    for denied in denied_scopes:
        host = _Host()
        controller = _controller(api, host, scope=denied)
        _error(api, "unauthorized", lambda: _propose(api, controller))
        assert host.reads == host.calls == []
    _error(api, "invalid_scope", lambda: _scope(api, execution_mode="real"))
    host = _Host()
    controller = _controller(api, host, scope=scope)
    for changes in ({"target": "target-b"}, {"scope_id": "scope-b"}, {"policy_version": "policy-v2"}):
        _error(api, "unauthorized", lambda: _propose(api, controller, payload=_payload(**changes)))
    assert host.reads == host.calls == []


def test_host_action_ids_and_replay_keys_are_distinct_from_content_and_run_identity():
    api = _api()
    host = _Host()
    controller = _controller(api, host)
    first = _propose(api, controller)
    host.now = 11.0
    second = _propose(api, controller, action_id="action-b", key="key-b")
    assert first["action_id"] == "action-a" and second["action_id"] == "action-b"
    assert first["idempotency_key"] == "key-a" and second["idempotency_key"] == "key-b"
    assert first["fingerprint"] == second["fingerprint"] == _DIGEST
    assert first["transitions"] == [{"state": "proposed", "at": 10.0}]
    for record in (first, second):
        _record(record, "proposed")
        assert record["spec"] == _payload() and record["approval"] is None
        assert record["result"] is None
        assert record["principal"] == "host-user" and record["context_id"] == "context-a"
    for action_id in ("", "a" * 65):
        _error(api, "invalid_action", lambda: _propose(api, controller, action_id=action_id))
    _error(api, "invalid_action", lambda: _propose(api, controller, key=""))
    assert host.reads == host.calls == []


def test_authorization_precedes_lookup_approval_consumption_and_replay_disclosure():
    api = _api()
    host = _Host()
    controller = _controller(api, host)
    _propose(api, controller)
    approval = _approve(controller)
    before = controller.get("action-a", **_CALLER)
    for caller in ({**_CALLER, "principal": "outsider"}, {**_CALLER, "context_id": "other"}):
        for action_id in ("action-a", "absent"):
            _error(api, "unauthorized", lambda: controller.execute(action_id, approval=approval, **caller))
            _error(api, "unauthorized", lambda: controller.get(action_id, **caller))
            _error(api, "unauthorized", lambda: controller.reject(action_id, **caller))
        _error(api, "unauthorized", lambda: _propose(api, controller, **caller))
        _error(api, "unauthorized", lambda: controller.approve(
            "action-a", approval_id="forged", expires_at=20, **caller))
    assert host.reads == host.calls == []
    assert controller.get("action-a", **_CALLER) == before
    controller.execute("action-a", approval=approval, **_CALLER)
    _error(api, "unauthorized", lambda: controller.get("action-a", principal="outsider", context_id="context-a"))
    _error(api, "unauthorized", lambda: _propose(api, controller, principal="outsider"))
    _error(api, "unauthorized", lambda: controller.execute(
        "action-a", approval=approval, principal="outsider", context_id="context-a"))
    assert len(host.calls) == 1


def test_trusted_approval_required_and_execute_cannot_replace_frozen_arguments():
    api = _api()
    host = _Host()
    controller = _controller(api, host)
    _propose(api, controller)
    _error(api, "approval_required", lambda: controller.execute("action-a", **_CALLER))
    for fake in ({"approved": True}, "user approved", {"approval_id": "approval-a"}):
        _error(api, "approval_mismatch", lambda: controller.execute("action-a", approval=fake, **_CALLER))
    approval = _approve(controller)
    assert isinstance(approval, api.ApprovalEvidence)
    assert (approval.action_id, approval.fingerprint, approval.scope_id,
            approval.policy_version, approval.principal, approval.context_id, approval.expires_at) == (
                "action-a", _DIGEST, "scope-a", "policy-v1", "host-user", "context-a", 20.0)
    with pytest.raises((FrozenInstanceError, AttributeError)):
        approval.fingerprint = "0" * 64
    with pytest.raises(TypeError):
        controller.execute("action-a", approval=approval, arguments={"value": "replacement"}, **_CALLER)
    _error(api, "approval_exists", lambda: _approve(controller, approval_id="another"))
    before = controller.get("action-a", **_CALLER)
    assert controller.get("action-a", **_CALLER) == before  # read does not consume/reserve
    assert before["approval"] == {"approval_id": "approval-a", "status": "available", "expires_at": 20.0}
    assert host.reads == host.calls == []
    result = controller.execute("action-a", approval=approval, **_CALLER)
    _record(result, "succeeded")


def test_approval_binding_rejects_tampering_cross_instance_and_expiry_at_boundary():
    api = _api()
    host = _Host()
    controller = _controller(api, host)
    _propose(api, controller)
    approval = _approve(controller)
    before = controller.get("action-a", **_CALLER)
    for changes in ({"action_id": "action-b"}, {"fingerprint": "0" * 64},
                    {"scope_id": "scope-b"}, {"policy_version": "policy-v2"},
                    {"principal": "outsider"}, {"context_id": "other"},
                    {"expires_at": 999.0}, {"approval_id": "other"}, {}):
        forged = replace(approval, **changes)
        _error(api, "approval_mismatch", lambda: controller.execute("action-a", approval=forged, **_CALLER))
    fresh_host = _Host()
    fresh = _controller(api, fresh_host)
    _propose(api, fresh)
    own_approval = _approve(fresh)  # Same public IDs do not restore a prior grant.
    _error(api, "approval_mismatch", lambda: fresh.execute("action-a", approval=approval, **_CALLER))
    assert fresh_host.reads == fresh_host.calls == []
    fresh.execute("action-a", approval=own_approval, **_CALLER)
    assert controller.get("action-a", **_CALLER) == before
    host.now = 20.0
    _error(api, "approval_expired", lambda: controller.execute("action-a", approval=approval, **_CALLER))
    assert controller.get("action-a", **_CALLER) == before
    assert host.reads == host.calls == []
    for expiry in (10.0, float("nan"), float("inf"), True):
        other = _controller(api, _Host())
        _propose(api, other)
        _error(api, "invalid_approval", lambda: _approve(other, expires_at=expiry))


def test_single_use_approval_and_successful_replay_never_reenter_simulator():
    api = _api()
    host = _Host()
    controller = _controller(api, host)
    _propose(api, controller)
    _propose(api, controller, action_id="action-b", key="key-b")
    approval = _approve(controller)
    first = controller.execute("action-a", approval=approval, **_CALLER)
    assert first["approval"]["status"] == "consumed"
    assert first["result"] == {"code": "simulated", "data": _payload()["arguments"]}
    host.now = 100.0
    assert controller.execute("action-a", approval=approval, **_CALLER) == first
    _error(api, "approval_consumed", lambda: controller.execute("action-b", approval=approval, **_CALLER))
    assert controller.get("action-b", **_CALLER)["state"] == "proposed"
    assert host.reads == ["target-a"] and len(host.calls) == 1


def test_preconditions_are_read_at_execution_boundary_and_stale_is_terminal():
    api = _api()
    for change in ("version", "nested"):
        host = _Host()
        controller = _controller(api, host)
        _propose(api, controller)
        approval = _approve(controller)
        assert host.reads == []
        if change == "version":
            host.target["target_version"] = "v2"
        else:
            host.target["expected"]["flags"][0] = False
        host.now = 12.0
        stale = controller.execute("action-a", approval=approval, **_CALLER)
        _record(stale, "stale")
        assert stale["result"] == {"code": "stale_precondition", "data": {}}
        assert stale["approval"]["status"] == "available"
        assert stale["transitions"] == [{"state": "proposed", "at": 10.0}, {"state": "stale", "at": 12.0}]
        host.target = deepcopy(_payload()["preconditions"])
        assert controller.execute("action-a", approval=approval, **_CALLER) == stale
        assert host.reads == ["target-a"] and host.calls == []


def test_idempotency_reuses_original_intent_and_rejects_content_or_id_conflicts():
    api = _api()
    host = _Host()
    controller = _controller(api, host)
    original = _propose(api, controller)
    assert _propose(api, controller, action_id="candidate-id") == original
    changed = _payload(arguments={"value": "different"})
    _error(api, "idempotency_conflict", lambda: _propose(api, controller, action_id="action-b", payload=changed))
    _error(api, "action_id_conflict", lambda: _propose(api, controller, key="key-b"))
    assert host.reads == host.calls == []
    result = controller.execute("action-a", approval=_approve(controller), **_CALLER)
    assert _propose(api, controller, action_id="another-candidate") == result
    assert controller.get("action-a", **_CALLER) == result
    assert len(host.calls) == 1


def test_concurrent_duplicate_reservation_consumes_once_and_simulates_at_most_once():
    api = _api()
    host = _Host()
    entered, release = threading.Event(), threading.Event()
    duplicate_done = threading.Event()
    calls, results, errors = [], [], []
    result_lock = threading.Lock()

    def simulate(spec):
        with result_lock:
            calls.append(spec.fingerprint)
        entered.set()
        assert release.wait(5), "test host did not release simulation"
        return {"value": deepcopy(spec.arguments["value"])}

    controller = _controller(api, host, simulator=simulate)
    _propose(api, controller)
    approval = _approve(controller)

    def invoke(*, duplicate=False):
        try:
            result = controller.execute("action-a", approval=approval, **_CALLER)
            with result_lock:
                results.append(result)
        except BaseException as error:
            with result_lock:
                errors.append(error)
        finally:
            if duplicate:
                duplicate_done.set()

    first = threading.Thread(target=invoke, daemon=True)
    duplicate = threading.Thread(target=lambda: invoke(duplicate=True), daemon=True)
    threads = [first]
    try:
        first.start()
        assert entered.wait(5), "simulator was never entered"
        duplicate.start()
        threads.append(duplicate)
        assert duplicate_done.wait(5), "duplicate blocked instead of observing the reservation"
        assert errors == [] and len(results) == 1
        _record(results[0], "executing")
        assert results[0]["approval"]["status"] == "consumed"
        assert results[0]["result"] is None
    finally:
        release.set()
        for thread in threads:
            thread.join(timeout=5)
    assert not any(thread.is_alive() for thread in threads)
    assert errors == [] and len(results) == 2
    assert calls == [_DIGEST] and host.reads == ["target-a"]
    assert all(result["state"] in {"executing", "succeeded"} for result in results)
    final = controller.get("action-a", **_CALLER)
    assert final["approval"]["status"] == "consumed"
    assert [step["state"] for step in final["transitions"]] == ["proposed", "executing", "succeeded"]


def test_terminal_lifecycle_safe_failures_programmer_errors_and_no_automatic_retry():
    api = _api()
    for mode in ("success", "rejected", "failure", "programmer"):
        host = _Host()
        calls = []
        secret = "SECRET-canary-C:\\private\\token"
        bug = ValueError(secret)

        def simulate(spec):
            calls.append(spec.fingerprint)
            if mode == "failure":
                raise api.SimulationFailure(secret)
            if mode == "programmer":
                raise bug
            return {"value": deepcopy(spec.arguments["value"])}

        controller = _controller(api, host, simulator=simulate)
        _propose(api, controller)
        approval = _approve(controller)
        host.now = 12.0
        if mode == "rejected":
            controller.reject("action-a", **_CALLER)
            state, code = "rejected", "action_rejected"
        elif mode == "programmer":
            with pytest.raises(ValueError) as captured:
                controller.execute("action-a", approval=approval, **_CALLER)
            assert captured.value is bug
            state, code = "uncertain", "simulation_uncertain"
        else:
            controller.execute("action-a", approval=approval, **_CALLER)
            state, code = (("failed", "simulation_failed") if mode == "failure"
                           else ("succeeded", "simulated"))
        record = controller.get("action-a", **_CALLER)
        _record(record, state)
        assert record["result"]["code"] == code
        if mode != "success":
            assert record["result"]["data"] == {}
        steps = ["proposed", state] if mode == "rejected" else ["proposed", "executing", state]
        assert record["transitions"] == [{"state": s, "at": 10.0 if i == 0 else 12.0}
                                          for i, s in enumerate(steps)]
        assert secret not in json.dumps(record) and "traceback" not in json.dumps(record)
        assert controller.execute("action-a", approval=approval, **_CALLER) == record
        _error(api, "action_terminal", lambda: _approve(controller, approval_id="new-approval"))
        assert len(calls) == (0 if mode == "rejected" else 1)


def test_action_input_bounds_are_inclusive_utf8_bytes_not_output_limits():
    api = _api()
    # Four individually bounded strings reach the total byte boundary exactly.
    payload = _payload(arguments={"value": ["\u00e9" * 2200, "x" * 4000, "x" * 4000, ""]})
    padding = 16384 - len(_canonical(payload).encode("utf-8"))
    assert 0 < padding <= 4096  # Fixture arithmetic is independent of production.
    payload["arguments"]["value"][-1] = "x" * padding
    assert len(_canonical(payload).encode("utf-8")) == 16384
    assert api.ActionSpec(**payload).to_json() == _canonical(payload)
    payload["arguments"]["value"][-1] += "x"
    _error(api, "action_too_large", lambda: api.ActionSpec(**payload))
    for value in ("x" * 4097, list(range(33)), {f"k{i}": i for i in range(33)}):
        _error(api, "invalid_action", lambda: api.ActionSpec(**_payload(arguments={"value": value})))
    for value in ("x" * 4096, list(range(32)), {f"k{i}": i for i in range(32)}):
        api.ActionSpec(**_payload(arguments={"value": value}))
    nested = 0
    for _ in range(8):
        nested = [nested]
    api.ActionSpec(**_payload(arguments={"value": nested}))
    _error(api, "invalid_action", lambda: api.ActionSpec(**_payload(arguments={"value": [nested]})))
    _error(api, "invalid_action", lambda: api.ActionSpec(**_payload(target="a" * 65)))
    api.ActionSpec(**_payload(target="a" * 64))
    size = len(_canonical(_payload()).encode("utf-8"))
    host = _Host()
    small = _controller(api, host, scope=_scope(api, max_action_bytes=size - 1))
    _error(api, "action_too_large", lambda: _propose(api, small))
    exact = _controller(api, host, scope=_scope(api, max_action_bytes=size))
    _propose(api, exact)
    assert host.calls == host.reads == []


def test_controller_capacity_retains_terminal_replay_records_and_bounds_host_config():
    api = _api()
    for changes in ({"max_actions": 0}, {"max_actions": True}, {"max_actions": 1025},
                    {"max_action_bytes": 0}, {"max_action_bytes": True}, {"max_action_bytes": 16385}):
        _error(api, "invalid_scope", lambda: _scope(api, **changes))
    host = _Host()
    controller = _controller(api, host, scope=_scope(api, max_actions=1))
    _propose(api, controller)
    approval = _approve(controller)
    result = controller.execute("action-a", approval=approval, **_CALLER)
    _error(api, "capacity_exceeded", lambda: _propose(api, controller, action_id="action-b", key="key-b"))
    assert _propose(api, controller, action_id="ignored-new-id") == result
    assert controller.execute("action-a", approval=approval, **_CALLER) == result
    assert len(host.calls) == 1
    for state in ("rejected", "proposed"):
        other = _controller(api, _Host(), scope=_scope(api, max_actions=1))
        _propose(api, other)
        if state == "rejected":
            other.reject("action-a", **_CALLER)
        _error(api, "capacity_exceeded", lambda: _propose(api, other, action_id="action-b", key="key-b"))


def test_simulation_only_default_executor_has_no_external_effect_or_identity_generation(monkeypatch):
    api = _api()
    from experiment_repository import ExperimentRepository

    host = _Host()
    attempts = []
    environment = dict(os.environ)

    def forbidden(*args, **kwargs):
        attempts.append("external capability entered")
        raise AssertionError("foundation must be in-memory simulation only")

    # Imports precede guards; no real file, process, DB or network operation is
    # used to test a guard. context restores everything before pytest reporting.
    with monkeypatch.context() as guard:
        for owner, names in (
            (builtins, ("open",)), (io, ("open",)),
            (Path, ("open", "write_text", "write_bytes", "mkdir", "touch", "unlink", "rename", "replace", "rmdir")),
            (os, ("open", "write", "mkdir", "remove", "unlink", "rename", "replace", "system", "popen", "putenv", "unsetenv")),
            (subprocess, ("Popen", "run", "call", "check_call", "check_output")),
            (socket, ("socket", "create_connection")), (sqlite3, ("connect",)),
            (ExperimentRepository, ("__init__", "add", "invalidate", "delete")),
            (uuid, ("uuid1", "uuid4")),
        ):
            for name in names:
                guard.setattr(owner, name, forbidden)
        controller = api.WorkflowController(_scope(api), target_provider=host.read, clock=host.clock)
        _propose(api, controller)
        approval = _approve(controller)
        result = controller.execute("action-a", approval=approval, **_CALLER)
        assert controller.execute("action-a", approval=approval, **_CALLER) == result
        _record(result, "succeeded")
        assert result["result"] == {"code": "simulated", "data": _payload()["arguments"]}
        assert result["spec"] == _payload()
    assert attempts == [] and dict(os.environ) == environment
    assert host.target == _payload()["preconditions"]
    assert not hasattr(controller, "session") and not hasattr(controller, "repository")

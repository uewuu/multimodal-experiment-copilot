"""Bounded, process-local control of one deterministic simulation operation.

The host supplies identities, scope, clock and approval decisions. These Python
APIs are not remote authentication or model tools. Borrowed target providers
and optional simulator doubles must be trusted, side-effect-free host code;
this module does not sandbox arbitrary Python callbacks.

No state survives controller disposal. Neither approval nor idempotency is
durable, and simulated success is not evidence of a real resource change.
"""

from collections.abc import Collection, Mapping
from copy import deepcopy
from dataclasses import dataclass
from hashlib import sha256
import json
from math import isfinite
import re
from threading import Lock


__all__ = (
    "ActionSpec", "WorkflowScope", "ApprovalEvidence", "WorkflowController",
    "WorkflowError", "SimulationFailure",
)

_MAX_BYTES = 16384
_IDENTIFIER = re.compile(r"[A-Za-z0-9_-]{1,64}\Z")
_SPEC_FIELDS = frozenset({
    "contract_version", "action_type", "target", "arguments", "preconditions",
    "scope_id", "policy_version",
})
_FORBIDDEN = frozenset({
    "approved", "allow_write", "force", "skip_approval", "shell_command",
    "executable", "argv", "env", "environment", "database_path", "path",
    "cwd", "workspace", "api_key", "password", "token",
})
_TRANSITIONS = {
    "proposed": frozenset({"executing", "rejected", "stale"}),
    "executing": frozenset({"succeeded", "failed", "uncertain"}),
}


class WorkflowError(ValueError):
    """A stable business code, without caller data or exception diagnostics."""

    def __init__(self, code: str):
        self.code = code
        super().__init__(code)


class SimulationFailure(Exception):
    """Host simulator's known failure; its message never enters action evidence."""


def _identifier(value, code="invalid_action"):
    if type(value) is not str or _IDENTIFIER.fullmatch(value) is None:
        raise WorkflowError(code)
    return value


def _fields(value, fields):
    if type(value) is not dict or value.keys() != fields:
        raise WorkflowError("invalid_action")


def _canonical(value):
    try:
        text = json.dumps(value, sort_keys=True, separators=(",", ":"),
                          ensure_ascii=False, allow_nan=False)
        text.encode("utf-8")
        return text
    except (ValueError, UnicodeError, RecursionError):
        raise WorkflowError("invalid_action") from None


def _copy_data(value):
    """Copy finite literal JSON with depth/member/string and traversal bounds."""
    remaining = _MAX_BYTES

    def spend(count):
        nonlocal remaining
        remaining -= count
        if remaining < 0:
            raise WorkflowError("action_too_large")

    def visit(item, depth):
        kind = type(item)
        if kind in (list, dict):
            if depth >= 8 or len(item) > 32:
                raise WorkflowError("invalid_action")
            spend(2 + max(0, len(item) - 1))
            if kind is list:
                return [visit(child, depth + 1) for child in item]
            result = {}
            for key, child in item.items():
                if type(key) is not str or len(key) > 4096 or key in _FORBIDDEN:
                    raise WorkflowError("invalid_action")
                spend(len(_canonical(key).encode("utf-8")) + 1)
                result[key] = visit(child, depth + 1)
            return result
        if kind is str:
            if len(item) > 4096:
                raise WorkflowError("invalid_action")
        elif item is None or kind in (bool, int):
            pass
        elif kind is float and isfinite(item):
            pass
        else:
            raise WorkflowError("invalid_action")
        spend(len(_canonical(item).encode("utf-8")))
        return item

    return visit(value, 0)


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise WorkflowError("invalid_action")
        result[key] = value
    return result


def _invalid_constant(value):
    raise WorkflowError("invalid_action")


@dataclass(frozen=True, slots=True, init=False)
class ActionSpec:
    """An immutable canonical snapshot; nested public reads are fresh copies."""

    contract_version: str
    action_type: str
    target: str
    scope_id: str
    policy_version: str
    fingerprint: str
    _json: str

    def __init__(self, *, contract_version, action_type, target, arguments,
                 preconditions, scope_id, policy_version):
        if type(contract_version) is not str or contract_version != "v1":
            raise WorkflowError("invalid_action")
        if type(action_type) is not str or action_type != "simulate":
            raise WorkflowError("invalid_action")
        _fields(arguments, {"value"})
        _fields(preconditions, {"target_version", "expected"})
        payload = {
            "contract_version": contract_version, "action_type": action_type,
            "target": _identifier(target), "scope_id": _identifier(scope_id),
            "policy_version": _identifier(policy_version),
            "arguments": {"value": _copy_data(arguments["value"])},
            "preconditions": {
                "target_version": _identifier(preconditions["target_version"]),
                "expected": _copy_data(preconditions["expected"]),
            },
        }
        text = _canonical(payload)
        encoded = text.encode("utf-8")
        if len(encoded) > _MAX_BYTES:
            raise WorkflowError("action_too_large")
        for name in ("contract_version", "action_type", "target", "scope_id", "policy_version"):
            object.__setattr__(self, name, payload[name])
        object.__setattr__(self, "_json", text)
        object.__setattr__(self, "fingerprint", sha256(encoded).hexdigest())

    @property
    def arguments(self):
        return json.loads(self._json)["arguments"]

    @property
    def preconditions(self):
        return json.loads(self._json)["preconditions"]

    def to_json(self):
        return self._json

    @classmethod
    def from_json(cls, text):
        if type(text) is not str:
            raise WorkflowError("invalid_action")
        try:
            payload = json.loads(text, object_pairs_hook=_unique_object,
                                 parse_constant=_invalid_constant)
        except (ValueError, RecursionError):
            raise WorkflowError("invalid_action") from None
        _fields(payload, _SPEC_FIELDS)
        return cls(**payload)


def _allowlist(values):
    if not isinstance(values, Collection) or isinstance(values, (str, bytes, Mapping)):
        raise WorkflowError("invalid_scope")
    result = []
    for value in values:
        if type(value) is not str or value != "*":
            _identifier(value, "invalid_scope")
        result.append(value)
    return frozenset(result)


@dataclass(frozen=True, slots=True, init=False)
class WorkflowScope:
    """Independent host authority. Empty allowlists deny; '*' is only literal."""

    scope_id: str
    policy_version: str
    principal: str
    context_id: str
    allowed_action_types: frozenset[str]
    allowed_targets: frozenset[str]
    execution_mode: str
    max_actions: int
    max_action_bytes: int

    def __init__(self, scope_id, policy_version, principal, context_id, *,
                 allowed_action_types=(), allowed_targets=(), execution_mode="simulation",
                 max_actions=4, max_action_bytes=_MAX_BYTES):
        if type(execution_mode) is not str or execution_mode != "simulation":
            raise WorkflowError("invalid_scope")
        for value, upper in ((max_actions, 1024), (max_action_bytes, _MAX_BYTES)):
            if type(value) is not int or not 1 <= value <= upper:
                raise WorkflowError("invalid_scope")
        values = {
            "scope_id": _identifier(scope_id, "invalid_scope"),
            "policy_version": _identifier(policy_version, "invalid_scope"),
            "principal": _identifier(principal, "invalid_scope"),
            "context_id": _identifier(context_id, "invalid_scope"),
            "allowed_action_types": _allowlist(allowed_action_types),
            "allowed_targets": _allowlist(allowed_targets),
            "execution_mode": execution_mode, "max_actions": max_actions,
            "max_action_bytes": max_action_bytes,
        }
        for name, value in values.items():
            object.__setattr__(self, name, value)


@dataclass(frozen=True, slots=True)
class ApprovalEvidence:
    """A host-issued value; constructing an equal value does not issue a grant."""

    approval_id: str
    action_id: str
    fingerprint: str
    scope_id: str
    policy_version: str
    principal: str
    context_id: str
    expires_at: float


@dataclass
class _Action:
    spec: ActionSpec
    record: dict
    approval: ApprovalEvidence | None = None


def _simulate(spec):
    return spec.arguments


class WorkflowController:
    """One bounded in-memory ledger, with no adapters or persistent resources.

    Host callbacks are borrowed. Target reads run under the reservation lock
    and must be short and non-reentrant; the simulator runs outside the lock.
    """

    def __init__(self, scope, *, target_provider, clock, simulator=None):
        if type(scope) is not WorkflowScope:
            raise TypeError("scope must be a WorkflowScope")
        if not callable(target_provider) or not callable(clock):
            raise TypeError("target_provider and clock must be callable")
        if simulator is not None and not callable(simulator):
            raise TypeError("simulator must be callable or None")
        self._scope = scope
        self._target_provider = target_provider
        self._clock = clock
        self._simulator = _simulate if simulator is None else simulator
        self._actions: dict[str, _Action] = {}
        self._keys: dict[str, str] = {}
        self._lock = Lock()

    def _now(self):
        value = self._clock()
        if type(value) not in (int, float) or not isfinite(value):
            raise ValueError("host clock must return finite numeric time")
        return value

    def _authorize(self, principal, context_id):
        if (type(principal) is not str or type(context_id) is not str
                or principal != self._scope.principal or context_id != self._scope.context_id):
            raise WorkflowError("unauthorized")

    def _authorize_spec(self, spec):
        scope = self._scope
        if (spec.scope_id != scope.scope_id or spec.policy_version != scope.policy_version
                or spec.action_type not in scope.allowed_action_types
                or spec.target not in scope.allowed_targets):
            raise WorkflowError("unauthorized")

    def _entry(self, action_id):
        _identifier(action_id)
        try:
            entry = self._actions[action_id]
        except KeyError:
            raise WorkflowError("action_not_found") from None
        self._authorize_spec(entry.spec)
        return entry

    @staticmethod
    def _transition(entry, state, at, result=None):
        record = entry.record
        if state not in _TRANSITIONS.get(record["state"], ()):
            raise RuntimeError("invalid internal workflow transition")
        record["state"] = state
        record["result"] = result
        record["transitions"].append({"state": state, "at": at})

    def propose(self, spec, *, action_id, idempotency_key, principal, context_id):
        with self._lock:
            self._authorize(principal, context_id)
            if type(spec) is not ActionSpec:
                raise WorkflowError("invalid_action")
            self._authorize_spec(spec)
            _identifier(action_id)
            _identifier(idempotency_key)
            if len(spec.to_json().encode("utf-8")) > self._scope.max_action_bytes:
                raise WorkflowError("action_too_large")
            if action_id in self._actions and self._actions[action_id].record["idempotency_key"] != idempotency_key:
                raise WorkflowError("action_id_conflict")
            if idempotency_key in self._keys:
                entry = self._entry(self._keys[idempotency_key])
                if entry.spec.fingerprint != spec.fingerprint:
                    raise WorkflowError("idempotency_conflict")
                return deepcopy(entry.record)
            if len(self._actions) >= self._scope.max_actions:
                raise WorkflowError("capacity_exceeded")
            snapshot = ActionSpec.from_json(spec.to_json())
            record = {
                "action_id": action_id, "fingerprint": snapshot.fingerprint,
                "idempotency_key": idempotency_key, "spec": json.loads(snapshot.to_json()),
                "principal": principal, "context_id": context_id, "state": "proposed",
                "execution_mode": "simulation", "approval": None, "result": None,
                "transitions": [{"state": "proposed", "at": self._now()}],
            }
            self._actions[action_id] = _Action(snapshot, record)
            self._keys[idempotency_key] = action_id
            return deepcopy(record)

    def get(self, action_id, *, principal, context_id):
        with self._lock:
            self._authorize(principal, context_id)
            return deepcopy(self._entry(action_id).record)

    def approve(self, action_id, *, approval_id, expires_at, principal, context_id):
        """Trusted host decision; never exposed as a model-callable operation."""
        with self._lock:
            self._authorize(principal, context_id)
            entry = self._entry(action_id)
            if entry.record["state"] != "proposed":
                raise WorkflowError("action_terminal")
            if entry.approval is not None:
                raise WorkflowError("approval_exists")
            _identifier(approval_id, "invalid_approval")
            if (type(expires_at) not in (int, float) or not isfinite(expires_at)
                    or expires_at <= self._now()):
                raise WorkflowError("invalid_approval")
            if any(item.approval is not None and item.approval.approval_id == approval_id
                   for item in self._actions.values()):
                raise WorkflowError("approval_exists")
            entry.approval = ApprovalEvidence(
                approval_id, action_id, entry.spec.fingerprint,
                self._scope.scope_id, self._scope.policy_version,
                principal, context_id, expires_at,
            )
            entry.record["approval"] = {
                "approval_id": approval_id, "status": "available", "expires_at": expires_at,
            }
            return entry.approval

    def _approval(self, entry, approval):
        if approval is None:
            raise WorkflowError("approval_required")
        if type(approval) is not ApprovalEvidence or type(approval.action_id) is not str:
            raise WorkflowError("approval_mismatch")
        owner = self._actions.get(approval.action_id)
        if owner is None or owner.approval is not approval:
            raise WorkflowError("approval_mismatch")
        if owner is not entry:
            code = ("approval_consumed" if owner.record["approval"]["status"] == "consumed"
                    else "approval_mismatch")
            raise WorkflowError(code)
        public = entry.record["approval"]
        expected = ApprovalEvidence(
            public["approval_id"], entry.record["action_id"], entry.spec.fingerprint,
            self._scope.scope_id, self._scope.policy_version,
            self._scope.principal, self._scope.context_id, public["expires_at"],
        )
        if approval != expected:
            raise WorkflowError("approval_mismatch")

    def reject(self, action_id, *, principal, context_id):
        with self._lock:
            self._authorize(principal, context_id)
            entry = self._entry(action_id)
            if entry.record["state"] != "proposed":
                raise WorkflowError("action_terminal")
            self._transition(entry, "rejected", self._now(), {"code": "action_rejected", "data": {}})
            return deepcopy(entry.record)

    def execute(self, action_id, *, approval=None, principal, context_id):
        with self._lock:
            self._authorize(principal, context_id)
            entry = self._entry(action_id)
            self._approval(entry, approval)
            if entry.record["state"] != "proposed":
                return deepcopy(entry.record)
            if self._now() >= approval.expires_at:
                raise WorkflowError("approval_expired")
            current = self._target_provider(entry.spec.target)
            # Canonical comparison preserves bool/number and literal distinctions.
            _fields(current, {"target_version", "expected"})
            actual = {"target_version": _identifier(current["target_version"]),
                      "expected": _copy_data(current["expected"])}
            now = self._now()
            if now >= approval.expires_at:
                raise WorkflowError("approval_expired")
            if _canonical(actual) != _canonical(entry.spec.preconditions):
                self._transition(entry, "stale", now, {"code": "stale_precondition", "data": {}})
                return deepcopy(entry.record)
            # All duplicate callers see this reservation before any simulator entry.
            entry.record["approval"]["status"] = "consumed"
            self._transition(entry, "executing", now)

        try:
            try:
                data = self._simulator(entry.spec)
            except SimulationFailure:
                return self._complete(entry, "failed", "simulation_failed", {})
            else:
                # The sole simulation is a literal echo, not a generic result backend.
                if type(data) is not dict or data.keys() != {"value"}:
                    raise ValueError("invalid simulation result")
                if _canonical({"value": _copy_data(data["value"])}) != _canonical(entry.spec.arguments):
                    raise ValueError("invalid simulation result")
                return self._complete(entry, "succeeded", "simulated", entry.spec.arguments)
        except BaseException:
            # Keep diagnostics out of records and preserve the original exception.
            with self._lock:
                if entry.record["state"] == "executing":
                    try:
                        at = self._now()
                    except BaseException:
                        at = entry.record["transitions"][-1]["at"]
                    self._transition(entry, "uncertain", at,
                                     {"code": "simulation_uncertain", "data": {}})
            raise

    def _complete(self, entry, state, code, data):
        with self._lock:
            self._transition(entry, state, self._now(), {"code": code, "data": data})
            return deepcopy(entry.record)

"""Optional model tools over a borrowed, simulation-only workflow controller.

The host owns the controller, identity allocation and approval decisions. This
adapter neither issues approvals nor acquires resources. Host callbacks retain
the Foundation's trusted, side-effect-free contract; they are not sandboxed.
Only bounded record projections cross the model boundary.
"""

from dataclasses import dataclass
import json
import re

from copilot.controlled_workflow import (
    ActionSpec, WorkflowController, WorkflowError, WorkflowScope,
)


__all__ = ("WorkflowTools",)

_PROPOSE = "propose_simulated_action"
_GET = "get_simulated_action"
_SIMULATE = "simulate_approved_action"
_NAMES = (_PROPOSE, _GET, _SIMULATE)
_IDENTIFIER = re.compile(r"[A-Za-z0-9_-]{1,64}\Z")


def _object_schema(properties):
    return {"type": "object", "properties": properties,
            "required": list(properties), "additionalProperties": False}


def _identifier_schema():
    return {"type": "string", "minLength": 1, "maxLength": 64,
            "pattern": r"^[A-Za-z0-9_-]+$"}


def _error(code):
    return {"format_version": 1, "ok": False, "error": {"code": code}}


def _project(record):
    """Allowlist scalar evidence, excluding specs, grants and callback data."""
    code = None if record["result"] is None else record["result"]["code"]
    ok = record["state"] not in {"failed", "stale", "rejected", "uncertain"}
    result = {
        "format_version": 1, "ok": ok,
        "data": {
            "action_id": record["action_id"],
            "fingerprint": record["fingerprint"],
            "state": record["state"],
            "execution_mode": record["execution_mode"],
            "requires_approval": True,
            "result_code": code,
        },
    }
    if not ok:
        result["error"] = {"code": code}
    # Foundation bounds these scalar fields. Fail closed if that contract drifts.
    encoded = json.dumps(result, ensure_ascii=False, allow_nan=False,
                         sort_keys=True, separators=(",", ":")).encode("utf-8")
    if len(encoded) > 2048:
        raise ValueError("workflow result exceeds projection limit")
    return result


@dataclass(frozen=True, slots=True, init=False, eq=False, repr=False)
class WorkflowTools:
    """Stable tool configuration; all action state belongs to the controller."""

    _scope: WorkflowScope
    _controller: WorkflowController
    _identity_provider: object
    _approval_provider: object

    def __init__(self, scope, controller, *, identity_provider, approval_provider):
        if type(scope) is not WorkflowScope:
            raise TypeError("scope must be a WorkflowScope")
        if not isinstance(controller, WorkflowController):
            raise TypeError("controller must be a WorkflowController")
        if not callable(identity_provider) or not callable(approval_provider):
            raise TypeError("identity_provider and approval_provider must be callable")
        object.__setattr__(self, "_scope", scope)
        object.__setattr__(self, "_controller", controller)
        object.__setattr__(self, "_identity_provider", identity_provider)
        object.__setattr__(self, "_approval_provider", approval_provider)

    def list_tools(self) -> list[dict]:
        """Return fresh strict schemas with no host authority arguments."""
        descriptions = (
            "Propose a simulation only; a separate host decision must approve it.",
            "Read safe evidence for a stored simulation without executing it.",
            "Simulate a stored action using host approval; no real resource changes.",
        )
        schemas = (
            _object_schema({
                "target": _identifier_schema(),
                "arguments": _object_schema({"value": {}}),
                "preconditions": _object_schema({
                    "target_version": _identifier_schema(), "expected": {},
                }),
            }),
            _object_schema({"action_id": _identifier_schema()}),
            _object_schema({"action_id": _identifier_schema()}),
        )
        return [
            {"type": "function", "function": {
                "name": name, "description": description, "parameters": schema,
            }}
            for name, description, schema in zip(_NAMES, descriptions, schemas)
        ]

    def invoke_tool(self, tool_name: str, arguments: dict) -> dict:
        """Validate transport inputs, then delegate lifecycle decisions intact."""
        if not isinstance(tool_name, str):
            raise TypeError("tool_name must be a string")
        if not tool_name.strip():
            raise ValueError("tool_name must not be empty or whitespace")
        if tool_name not in _NAMES:
            raise KeyError("unknown workflow tool")
        if type(arguments) is not dict:
            return _error("invalid_arguments")
        fields = ({"target", "arguments", "preconditions"}
                  if tool_name == _PROPOSE else {"action_id"})
        if arguments.keys() - fields:
            return _error("unsupported_option")
        if arguments.keys() != fields:
            return _error("invalid_arguments")
        if tool_name != _PROPOSE:
            action_id = arguments["action_id"]
            if type(action_id) is not str or _IDENTIFIER.fullmatch(action_id) is None:
                return _error("invalid_arguments")

        caller = {"principal": self._scope.principal,
                  "context_id": self._scope.context_id}
        try:
            if tool_name == _PROPOSE:
                spec = ActionSpec(
                    contract_version="v1", action_type="simulate",
                    scope_id=self._scope.scope_id,
                    policy_version=self._scope.policy_version, **arguments,
                )
                action_id, idempotency_key = self._identity_provider()
                record = self._controller.propose(
                    spec, action_id=action_id, idempotency_key=idempotency_key,
                    **caller,
                )
            else:
                # Authorize before consulting even the host approval provider.
                # execute() rechecks authority and validates the original grant.
                record = self._controller.get(action_id, **caller)
                if tool_name == _SIMULATE:
                    approval = self._approval_provider(action_id)
                    record = self._controller.execute(
                        action_id, approval=approval, **caller,
                    )
        except WorkflowError as error:
            return _error(error.code)
        return _project(record)

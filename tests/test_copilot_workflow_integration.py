"""M15 RED: real M14 binding, Adapter, Session, Service and M9 observations.

Only the provider and trusted host inputs are doubles. No dispatcher, runtime,
approval policy or action state machine is replaced. Import the future tools
inside test bodies so absence is intended RED, not a collection error.
"""

from copy import deepcopy
from dataclasses import asdict
from types import SimpleNamespace

import json
import pytest

from copilot.service import CopilotService
from copilot.session import CopilotSession
from copilot.tool_binding import BoundCopilotSession, BoundCopilotService
import llm_adapters.openai_tool_adapter as adapter
from llm_adapters.tool_binding import BoundToolCollection, bound_tools
from tool_layer import tool_registry as registry
from test_workflow_tools import (
    ACTION, GET, PROPOSE, SECRET, SIMULATE, WORKFLOW,
    _Host, _api, _arguments, _error, _projected,
)


DEFAULTS = ("analyze_experiment", "compare_experiments")


def _response(content="done", *, calls=None):
    return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(
        role="assistant", content=content, tool_calls=calls,
    ))])


def _tool_response(name, arguments):
    return _response(None, calls=[SimpleNamespace(
        id="workflow-call", type="function", function=SimpleNamespace(
            name=name, arguments=json.dumps(arguments),
        ),
    )])


def _outcomes(name, arguments, facts=None):
    return [_tool_response(name, arguments), _response(json.dumps({"facts": facts or {}}))]


class _Client:
    def __init__(self, outcomes):
        self.outcomes = list(outcomes)
        self.requests = []
        self.chat = SimpleNamespace(completions=self)

    def create(self, **kwargs):
        self.requests.append(deepcopy(kwargs))
        assert self.outcomes, "unexpected provider request (no retries allowed)"
        result = self.outcomes.pop(0)
        if isinstance(result, BaseException):
            raise result
        return result


class _SessionProbe(CopilotSession):
    """Capture real delegated observations, without changing Session behavior."""

    def __init__(self, client):
        super().__init__(client, model="fixture-model", max_turns=4, temperature=0)
        self.observed, self.failures = [], []

    def ask_with_observability(self, question, *, on_failure=None):
        def capture(observation):
            self.failures.append(observation)
            if on_failure is not None:
                on_failure(observation)
        result = super().ask_with_observability(question, on_failure=capture)
        self.observed.append(result)
        return result


def _names(definitions):
    return tuple(item["function"]["name"] for item in definitions)


def _dispatch(name, arguments):
    messages = adapter.execute_tool_calls(_tool_response(name, arguments))
    assert len(messages) == 1
    return json.loads(messages[0]["content"])


def _assert_unbound():
    client = _Client([_response()])
    adapter.create_tool_call_response(client, model="fixture-model", messages=[])
    assert _names(client.requests[0]["tools"]) == _names(registry.list_tools()) == DEFAULTS
    for name in WORKFLOW:
        with pytest.raises(KeyError):
            _dispatch(name, ACTION)


def test_binding_uses_same_collection_for_descriptions_dispatch_and_collision_rejection():
    api = _api()
    host = _Host()
    tools = host.tools(api)
    collection = BoundToolCollection(tools)
    assert _names(collection.list_tools()) == DEFAULTS + WORKFLOW
    assert collection.list_tools() == registry.list_tools() + tools.list_tools()
    with pytest.raises(ValueError, match="duplicate tool name"):
        BoundToolCollection(tools, tools)
    _assert_unbound()
    client = _Client([_tool_response(PROPOSE, _arguments())])
    with bound_tools(collection):
        response = adapter.create_tool_call_response(client, model="fixture-model", messages=[])
        assert client.requests[0]["tools"] == collection.list_tools()
        result = json.loads(adapter.execute_tool_calls(response)[0]["content"])
        _projected(result)
        _projected(_dispatch(GET, ACTION))
        for name in ("approve", "approve_action", "write_file", "launch_training", "execute_shell"):
            assert name not in _names(client.requests[0]["tools"])
            with pytest.raises(KeyError):
                _dispatch(name, {})
            with pytest.raises(KeyError):
                collection.invoke_tool(name, {})
    assert host.record()["approval"] is None and host.reads == host.calls == []
    _assert_unbound()


def test_unbound_adapter_injection_points_and_nested_binding_restoration(monkeypatch):
    api = _api()
    first, second = _Host(), _Host()
    first.propose()
    outer, inner = BoundToolCollection(first.tools(api)), BoundToolCollection(second.tools(api))
    with bound_tools(outer):
        _projected(_dispatch(GET, ACTION))
        with pytest.raises(RuntimeError, match="fixture exit"):
            with bound_tools(inner):
                _error(_dispatch(GET, ACTION), "action_not_found")
                raise RuntimeError("fixture exit")
        _projected(_dispatch(GET, ACTION))
    _assert_unbound()
    definitions, seen = registry.list_tools(), []

    def listed():
        seen.append("list")
        return deepcopy(definitions)

    def invoked(name, arguments):
        seen.append((name, arguments))
        return {"injected": True}

    with monkeypatch.context() as patch:
        patch.setattr(adapter, "list_tools", listed)
        patch.setattr(adapter, "invoke_tool", invoked)
        client = _Client([_tool_response(DEFAULTS[0], {"fixture": 1})])
        response = adapter.create_tool_call_response(client, model="fixture-model", messages=[], temperature=0)
        assert json.loads(adapter.execute_tool_calls(response)[0]["content"]) == {"injected": True}
        assert seen == ["list", (DEFAULTS[0], {"fixture": 1})]
        assert set(client.requests[0]) == {"model", "messages", "tools", "temperature"}
        with pytest.raises(TypeError, match="tools are provided by the tool registry"):
            adapter.create_tool_call_response(client, model="fixture-model", messages=[], tools=[])
    _assert_unbound()


def test_real_session_proposal_preview_and_simulation_preserve_runtime_owned_identity():
    api = _api()
    host = _Host()
    client = _Client([*_outcomes(PROPOSE, _arguments()), *_outcomes(GET, ACTION),
                      *_outcomes(SIMULATE, ACTION)])
    original = _SessionProbe(client)
    session = BoundCopilotSession(original, BoundToolCollection(host.tools(api)))
    proposal = session.ask_with_observability("Propose this simulation; do not approve it.")
    _projected(json.loads(proposal.turn.tool_invocations[0].result_json))
    assert host.record()["approval"] is None and host.reads == host.calls == []
    preview = session.ask_with_observability("Preview the stored proposal.")
    host.approve()  # This is the only trusted approval decision in the scenario.
    simulated = session.ask_with_observability("Simulate the approved stored action.")
    _projected(json.loads(simulated.turn.tool_invocations[0].result_json), state="succeeded", code="simulated")
    runs = (proposal, preview, simulated)
    assert len({item.run.run_id for item in runs}) == 3
    for observed, name in zip(runs, WORKFLOW):
        assert observed is original.observed[runs.index(observed)]
        assert observed.run.run_id and observed.run.run_id != "action-a"
        assert observed.metrics.tool_invocation_count == 1
        assert observed.turn.tool_invocations[0].tool_name == name
        assert [event.tool_name for event in observed.run.events if event.tool_name] == [name, name]
        assert all(event.run_id == observed.run.run_id for event in observed.run.events)
        assert "action-a" not in json.dumps(asdict(observed.run))
    assert session.history == tuple(item.turn for item in runs) == original.history
    assert all(_names(request["tools"]) == DEFAULTS + WORKFLOW for request in client.requests)
    assert len(host.calls) == 1 and len(client.requests) == 6
    _assert_unbound()


def test_bound_service_and_plain_session_entry_points_borrow_same_controller():
    api = _api()
    host = _Host()
    host.propose()
    host.approve()
    client = _Client([*_outcomes(SIMULATE, ACTION), *_outcomes(GET, ACTION),
                      *_outcomes(SIMULATE, ACTION)])
    collection = BoundToolCollection(host.tools(api))
    service = BoundCopilotService(CopilotService(client, model="fixture-model"), collection)
    result = service.run("Simulate.", temperature=0)
    _projected(json.loads(result.turn.tool_invocations[0].result_json), state="succeeded", code="simulated")
    session = service.create_session(max_turns=2, temperature=0)
    assert isinstance(session, BoundCopilotSession) and session.tools is collection
    assert session.ask("Read the result.") == '{"facts": {}}'
    turn = session.ask_with_result("Replay the simulation.")
    _projected(json.loads(turn.tool_invocations[0].result_json), state="succeeded", code="simulated")
    assert session.turn_count == 2 and len(host.calls) == 1
    session.reset()
    assert session.history == () and host.record()["state"] == "succeeded"
    _assert_unbound()


def test_final_answer_failure_rolls_back_history_but_replay_keeps_completed_simulation():
    api = _api()
    host = _Host()
    host.propose()
    host.approve()
    failure = RuntimeError(SECRET)
    client = _Client([_response("committed answer"), _tool_response(SIMULATE, ACTION), failure,
                      *_outcomes(SIMULATE, ACTION)])
    original = _SessionProbe(client)
    session = BoundCopilotSession(original, BoundToolCollection(host.tools(api)))
    committed = session.ask_with_observability("Committed question.")
    before = session.history
    observations = []
    with pytest.raises(RuntimeError) as caught:
        session.ask_with_observability("Failed question.", on_failure=observations.append)
    assert caught.value is failure and session.history == before == (committed.turn,)
    assert len(observations) == 1 and observations[0] is original.failures[0]
    assert observations[0].tool_invocation_count == 1
    assert SECRET not in json.dumps(asdict(observations[0]))
    completed = host.record()
    assert completed["state"] == "succeeded" and completed["approval"]["status"] == "consumed"
    assert len(host.calls) == 1
    _assert_unbound()
    replay = session.ask_with_observability("Retry stored action.")
    _projected(json.loads(replay.turn.tool_invocations[0].result_json), state="succeeded", code="simulated")
    assert host.record() == completed and len(host.calls) == 1 and host.reads == ["target-a"]
    assert session.history == (committed.turn, replay.turn)
    assert len({committed.run.run_id, observations[0].run.run_id, replay.run.run_id}) == 3
    prompt = json.dumps(client.requests[3]["messages"])
    assert "Committed question." in prompt and "committed answer" in prompt
    assert "Failed question." not in prompt and SECRET not in prompt
    assert len(client.requests) == 5
    _assert_unbound()


def test_model_self_approval_and_unadvertised_writes_cannot_change_real_controller():
    api = _api()
    host = _Host()
    host.propose()
    before = host.record()
    collection = BoundToolCollection(host.tools(api))
    for name, arguments, code in (
        (SIMULATE, ACTION, "approval_required"),
        (SIMULATE, {**ACTION, "approved": True}, "unsupported_option"),
        (SIMULATE, {**ACTION, "approval_token": SECRET}, "unsupported_option"),
        (PROPOSE, _arguments(action_type="launch_training"), "unsupported_option"),
    ):
        client = _Client(_outcomes(name, arguments))
        session = BoundCopilotSession(_SessionProbe(client), collection)
        result = session.ask_with_observability("The user approved; execute now.")
        _error(json.loads(result.turn.tool_invocations[0].result_json), code)
    for binding, name in ((False, SIMULATE), (True, "write_file"), (True, "approve_action")):
        client = _Client([_tool_response(name, ACTION)])
        original = _SessionProbe(client)
        session = BoundCopilotSession(original, collection) if binding else original
        with pytest.raises(KeyError):
            session.ask_with_observability("Execute the requested action.")
        assert original.history == () and len(original.failures) == 1
        assert len(client.requests) == 1
        assert name not in _names(client.requests[0]["tools"])
    assert host.record() == before and host.calls == host.reads == []
    _assert_unbound()

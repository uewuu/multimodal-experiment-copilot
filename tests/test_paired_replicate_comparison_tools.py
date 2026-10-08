"""Explicit snapshot binding through the unchanged Adapter/Session/Runtime."""
from copy import deepcopy
import inspect
import json

import pytest
from copilot.session import CopilotSession
from llm_adapters.tool_binding import BoundToolCollection, bound_tools, _get_bound_tools
from tool_layer import tool_registry as registry
from m22_contract_helpers import (
    Client, PARAMETERS, TOOL, core_api, fixture, invoke, no_access, pairs,
    patch_public, row, tool_class,
)


def test_frozen_default_registry_and_capability_independent_of_missing_m22():
    assert [d['function']['name'] for d in registry.list_tools()] == ['analyze_experiment', 'compare_experiments']
    assert registry._workspace_paths_for_tool(TOOL) == ()
    with pytest.raises(KeyError): registry.invoke_tool(TOOL, {})


def test_constructor_signature_and_schema_snapshot():
    cls = tool_class(); p = inspect.signature(cls).parameters
    assert list(p) == ['group_a', 'group_b', 'pairing_declaration', 'targets']
    assert all(p[k].kind is inspect.Parameter.KEYWORD_ONLY for k in ('pairing_declaration', 'targets'))
    tools = invoke(cls, fixture()); schema = tools.list_tools()
    assert len(schema) == 1 and schema[0]['function']['name'] == TOOL
    assert schema[0]['function']['parameters'] == PARAMETERS
    assert schema[0]['function']['description']
    original = deepcopy(schema); schema[0]['function']['parameters']['properties']['approval'] = {}
    assert tools.list_tools() == original


def test_each_invocation_uses_real_core_and_isolated_result(monkeypatch):
    cls, f = tool_class(), core_api(); tools = invoke(cls, fixture()); seen = []
    import experiment_paired_comparison as core
    def spy(*a, **kw): seen.append(1); return f(*a, **kw)
    patch_public(monkeypatch, core, TOOL, spy)
    first = tools.invoke_tool(TOOL, {}); expected = deepcopy(first)
    row(first)['statistics']['mean'] = 999
    row(first)['pair_results'].clear(); row(first)['coverage'].clear(); first['limitations'].clear()
    assert tools.invoke_tool(TOOL, {}) == expected and seen == [1, 1]


@pytest.mark.parametrize('args', [{'group_a': {}}, {'pairs': []}, {'targets': []}, {'approval': True},
                                {'path': '.'}, {'seed': 42}, {'current': True}])
def test_nonempty_model_args_reject_before_core(monkeypatch, args):
    tools = invoke(tool_class(), fixture()); import experiment_paired_comparison as core
    seen = []
    def fail(*a, **kw): seen.append(1); raise AssertionError('core reached')
    patch_public(monkeypatch, core, TOOL, fail)
    with pytest.raises((TypeError, ValueError)): tools.invoke_tool(TOOL, args)
    assert seen == []


@pytest.mark.parametrize('args', [None, [], (), '', 0, True])
def test_runtime_requires_exact_empty_dictionary(args):
    tools = invoke(tool_class(), fixture())
    with pytest.raises(TypeError): tools.invoke_tool(TOOL, args)


def test_unknown_name_no_fallback_collision_and_nested_restore():
    tools = invoke(tool_class(), fixture()); defaults = registry.list_tools()
    with pytest.raises(KeyError): tools.invoke_tool('analyze_experiment', {})
    bound = BoundToolCollection(tools)
    assert bound.list_tools() == defaults + tools.list_tools()
    assert row(bound.invoke_tool(TOOL, {}))['statistics']['mean'] == 2.5
    with pytest.raises(ValueError): BoundToolCollection(tools, tools)
    class Collision:
        def list_tools(self): return defaults
        def invoke_tool(self, *a): raise AssertionError('shadow dispatch')
    with pytest.raises(ValueError): BoundToolCollection(Collision())
    outer = BoundToolCollection()
    with bound_tools(outer):
        with bound_tools(bound): assert _get_bound_tools() is bound
        assert _get_bound_tools() is outer
    assert registry.list_tools() == defaults


def test_real_session_two_providers_history_usage_and_correlated_runtime():
    tools = invoke(tool_class(), fixture()); client = Client()
    session = CopilotSession(client, model='offline', experiment_context=None)
    collection = BoundToolCollection(tools)
    with bound_tools(collection): observed = session.ask_with_observability('Describe the declared pairs')
    assert len(client.requests) == observed.metrics.provider_request_count == 2
    assert observed.metrics.tool_invocation_count == 1
    assert client.requests[0]['tools'] == collection.list_tools()
    call = observed.turn.tool_invocations[0]
    assert (call.tool_name, call.tool_call_id) == (TOOL, 'paired-call')
    payload = json.loads(call.result_json)
    assert row(payload)['statistics']['mean'] == 2.5
    delivered = [m for m in client.requests[1]['messages'] if m['role'] == 'tool']
    assert len(delivered) == 1 and json.loads(delivered[0]['content']) == payload
    events = observed.run.events
    assert [e.sequence for e in events] == list(range(len(events)))
    started = [e for e in events if e.kind == 'tool.execution.started']
    accepted = [e for e in events if e.kind == 'tool.result.accepted']
    assert len(started) == len(accepted) == 1
    for e in started + accepted:
        assert (e.run_id, e.tool_name, e.tool_call_id, e.tool_invocation_index) == (observed.run.run_id, TOOL, 'paired-call', 0)
    assert started[0].sequence < accepted[0].sequence < events[-1].sequence
    assert events[-1].kind == 'run.completed'
    assert observed.run.usage.complete is True and observed.run.usage.total_tokens == 28
    assert session.history == (observed.turn,) and session.turn_count == 1


@pytest.mark.parametrize('mode', ['invalid-args', 'core-validation', 'pair-declaration', 'computation', 'provider'])
def test_failures_preserve_exception_history_binding_and_observation(monkeypatch, mode):
    cls, f = tool_class(), core_api(); tools = invoke(cls, fixture())
    import experiment_paired_comparison as core
    failure = RuntimeError('computation sentinel')
    if mode in ('core-validation', 'pair-declaration', 'computation'):
        def failing(*a, **kw):
            if mode == 'computation': raise failure
            invalid = fixture()
            if mode == 'pair-declaration': pairs(invalid)[1]['pair_id'] = pairs(invalid)[0]['pair_id']
            else: invalid['group_a']['group_declaration']['aggregation_policy'] = 'invalid'
            return invoke(f, invalid)  # real core validation, not fake result
        patch_public(monkeypatch, core, TOOL, failing)
    client = Client({'approval': True} if mode == 'invalid-args' else {},
                    error=failure if mode == 'provider' else None)
    session, failures, outer = CopilotSession(client, model='offline'), [], BoundToolCollection()
    expected = RuntimeError if mode in ('computation', 'provider') else (ValueError, TypeError)
    with bound_tools(outer):
        with pytest.raises(expected) as caught:
            with bound_tools(BoundToolCollection(tools)):
                session.ask_with_observability('compare', on_failure=failures.append)
        if mode in ('computation', 'provider'): assert caught.value is failure
        assert _get_bound_tools() is outer and session.history == ()
        assert len(client.requests) == (2 if mode == 'provider' else 1)
        assert len(failures) == 1 and failures[0].run.events[-1].kind == 'run.failed'
        assert failures[0].run.usage.complete is False
        assert failures[0].run.usage.total_tokens == (None if mode == 'provider' else 14)
        if mode != 'provider': assert failures[0].stage == 'tool_execution'
        fresh = Client()
        with pytest.raises(KeyError): CopilotSession(fresh, model='offline').ask('unbound')
        assert fresh.requests[0]['tools'] == registry.list_tools()


@pytest.mark.parametrize('mode,status', [('partial', 'limited'), ('conflict', 'blocked'), ('unknown', 'insufficient')])
def test_runtime_completed_does_not_upgrade_pair_evidence(mode, status):
    cls, d = tool_class(), fixture()
    if mode == 'partial': d['group_b']['executions'].pop()
    else:
        for m in d['group_b']['targets'][0]['members']:
            m['metric_definition'] = 'other' if mode == 'conflict' else None
    session = CopilotSession(Client(), model='offline')
    with bound_tools(BoundToolCollection(invoke(cls, d))):
        observed = session.ask_with_observability('describe available evidence')
    t = row(json.loads(observed.turn.tool_invocations[0].result_json))
    assert t['summary_eligibility']['status'] == status
    assert observed.run.events[-1].kind == 'run.completed'
    if mode == 'partial': assert t['coverage']['coverage_ratio'] == .5
    else: assert t['statistics']['mean'] is None


def test_constructor_list_invoke_are_memory_only(monkeypatch):
    cls, d = tool_class(), fixture()
    with no_access(monkeypatch) as calls:
        tools = invoke(cls, d); schema = tools.list_tools(); result = tools.invoke_tool(TOOL, {})
    assert calls == [] and schema[0]['function']['parameters'] == PARAMETERS
    assert row(result)['statistics']['mean'] == 2.5


@pytest.mark.parametrize('mode', ['success', 'tool-error', 'provider-error'])
def test_provider_usage_fixture_calibration_against_frozen_m19(monkeypatch, mode):
    import m22_contract_helpers as helpers
    from tool_layer.replicate_group_tools import ReplicateGroupComparisonTools
    from m22_contract_helpers import trusted_m18
    d = fixture(); monkeypatch.setattr(helpers, 'TOOL', 'compare_replicate_groups')
    tool = ReplicateGroupComparisonTools(trusted_m18(d['group_a']), trusted_m18(d['group_b']), targets=d['targets'])
    client = Client({'approval': True} if mode == 'tool-error' else {},
                    error=RuntimeError('provider') if mode == 'provider-error' else None)
    session, failures = CopilotSession(client, model='offline'), []
    with bound_tools(BoundToolCollection(tool)):
        if mode == 'success':
            result = session.ask_with_observability('calibrate')
            assert result.run.usage.complete and result.run.usage.total_tokens == 28
        else:
            with pytest.raises((ValueError, RuntimeError)):
                session.ask_with_observability('calibrate', on_failure=failures.append)
            assert failures[0].run.usage.complete is False
            assert failures[0].run.usage.total_tokens == (None if mode == 'provider-error' else 14)


@pytest.mark.parametrize('entrance', ['observation', 'verification', 'unpaired', 'subprocess'])
def test_no_access_probe_independent_calibration(monkeypatch, entrance):
    import artifact_observation_reader as obs
    import experiment_artifact_verification as verify
    import experiment_group_comparison as unpaired
    import subprocess
    fn = {'observation': obs.observe_artifact, 'verification': verify.compare_artifact_descriptors,
          'unpaired': unpaired.compare_replicate_groups, 'subprocess': subprocess.Popen}
    # Resolve AFTER entering patch scope, so the probe calls the patched API.
    with no_access(monkeypatch) as calls:
        with pytest.raises(AssertionError, match='M22 attempted'):
            {'observation': obs.observe_artifact, 'verification': verify.compare_artifact_descriptors,
             'unpaired': unpaired.compare_replicate_groups, 'subprocess': subprocess.Popen}[entrance]()
    assert calls == ['forbidden'] and callable(fn[entrance])

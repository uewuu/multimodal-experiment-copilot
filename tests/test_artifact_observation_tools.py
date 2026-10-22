"""Three authorization layers and real offline Session/Adapter/Runtime dispatch."""
from copy import deepcopy
import json

import pytest

from copilot.session import CopilotSession
from llm_adapters.tool_binding import BoundToolCollection, bound_tools, _get_bound_tools
from tool_layer import tool_registry as registry
from test_tool_capability_metadata import EXPECTED_PROVIDER_TOOLS
from m21_contract_helpers import (api, binding, assert_report, Client, TOOL, PARAMETERS,
                                  patch_alias, ABC_SHA)


def test_private_capability_exact_and_default_collection_unchanged():
    found = registry._workspace_paths_for_tool(TOOL)
    assert [(d.parameter_name, d.path_kind) for d in found] == [('experiment_dir', 'directory')], \
        'Missing Tool / Capability / Integration'
    assert registry.list_tools() == EXPECTED_PROVIDER_TOOLS
    with pytest.raises(KeyError):
        registry.invoke_tool(TOOL, {'experiment_dir': '.'})


def test_default_registry_remains_unbound_independent_of_m21():
    assert registry.list_tools() == EXPECTED_PROVIDER_TOOLS
    with pytest.raises(KeyError):
        registry.invoke_tool(TOOL, {'experiment_dir': '.'})


def test_schema_binding_isolation_collision_and_no_constructor_read(monkeypatch, tmp_path):
    cls, reader = api('tool').ArtifactObservationTools, api('reader')
    calls = []
    def forbidden(*args, **kwargs):
        calls.append(True)
        raise AssertionError('construction/listing read file')
    patch_alias(monkeypatch, reader, 'observe_artifact', api('tool'), forbidden)
    supplied = binding(str(tmp_path))
    tools = cls(supplied)
    supplied['file_name'] = 'different.bin'
    listed = tools.list_tools()
    assert len(listed) == 1 and listed[0]['type'] == 'function'
    function = listed[0]['function']
    assert function['name'] == TOOL and function['parameters'] == PARAMETERS
    assert type(function['description']) is str and function['description'].strip()
    assert 'artifact.bin' not in json.dumps(listed) and str(tmp_path) not in json.dumps(listed)
    collection = BoundToolCollection(tools)
    assert collection.list_tools() == EXPECTED_PROVIDER_TOOLS + listed
    with pytest.raises(ValueError):
        BoundToolCollection(tools, tools)
    function['parameters']['properties'].clear()
    assert tools.list_tools()[0]['function']['parameters'] == PARAMETERS and calls == []


@pytest.mark.parametrize('field', ['file_name', 'filename', 'artifact_id', 'path', 'locator',
    'sha256', 'max_bytes', 'trusted', 'verified', 'current', 'checkpoint', 'approval', 'started_at'])
def test_model_injection_rejects_before_reader(monkeypatch, tmp_path, field):
    module, reader = api('tool'), api('reader')
    tools, calls = module.ArtifactObservationTools(binding(str(tmp_path))), []
    def forbidden(*args):
        calls.append(True)
        raise AssertionError('injection reached Reader')
    patch_alias(monkeypatch, reader, 'observe_artifact', module, forbidden)
    with pytest.raises((TypeError, ValueError)):
        tools.invoke_tool(TOOL, {'experiment_dir': str(tmp_path), field: True})
    assert calls == []


@pytest.mark.parametrize('arguments', [None, [], True, {}, {'experiment_dir': None},
    {'experiment_dir': 1}, {'experiment_dir': ''}, {'experiment_dir': '.'},
    {'experiment_dir': 'elsewhere'}, {'experiment_dir': r'C:\other-root'}])
def test_invalid_direct_arguments_before_reader(monkeypatch, tmp_path, arguments):
    module, reader = api('tool'), api('reader')
    tools, calls = module.ArtifactObservationTools(binding(str(tmp_path))), []
    def forbidden(*args):
        calls.append(True)
        raise AssertionError('invalid root reached Reader')
    patch_alias(monkeypatch, reader, 'observe_artifact', module, forbidden)
    with pytest.raises((TypeError, ValueError)):
        tools.invoke_tool(TOOL, arguments)
    assert calls == []


def test_unknown_name_no_fallback(tmp_path):
    tools = api('tool').ArtifactObservationTools(binding(str(tmp_path)))
    with pytest.raises(KeyError):
        tools.invoke_tool('analyze_experiment', {})


def test_each_call_delegates_real_reader_binding_and_results_isolated(monkeypatch, tmp_path):
    module, reader = api('tool'), api('reader')
    (tmp_path / 'artifact.bin').write_bytes(b'abc')
    supplied, calls = binding(str(tmp_path)), []
    original = reader.observe_artifact
    def traced(selected):
        result = original(selected)
        calls.append((deepcopy(selected), deepcopy(result)))
        return result
    patch_alias(monkeypatch, reader, 'observe_artifact', module, traced)
    tools = module.ArtifactObservationTools(supplied)
    supplied.update(file_name='wrong.bin', artifact_id='mutated')
    first = tools.invoke_tool(TOOL, {'experiment_dir': str(tmp_path)})
    assert first == calls[0][1]
    first['limitations'].clear()
    first['sha256'] = 'corrupted'
    (tmp_path / 'artifact.bin').write_bytes(b'')
    second = tools.invoke_tool(TOOL, {'experiment_dir': str(tmp_path)})
    assert len(calls) == 2 and calls[0][0] == calls[1][0] == binding(str(tmp_path))
    assert second == calls[1][1] and second['byte_size'] == 0
    assert calls[0][1]['sha256'] == ABC_SHA and second['limitations']


@pytest.mark.parametrize('mode', ['none', 'outside'])
def test_formal_authorization_before_bound_dispatch(tmp_path, mode):
    # Intentionally independent of missing Tool so capability deficiency is visible.
    calls = []
    class Probe:
        def list_tools(self):
            return [{'type': 'function', 'function': {'name': TOOL, 'description': 'probe', 'parameters': PARAMETERS}}]
        def invoke_tool(self, *args):
            calls.append(args)
            raise AssertionError('Missing Tool / Capability / Integration: authorization reached dispatch')
    outside = tmp_path / 'outside'
    outside.mkdir()
    client = Client({'experiment_dir': '.'} if mode == 'none' else {'experiment_dir': str(outside)})
    root = tmp_path / 'allowed'
    root.mkdir()
    session = CopilotSession(client, model='offline', experiment_context=None if mode == 'none' else {'experiment_dir': str(root)})
    with bound_tools(BoundToolCollection(Probe())):
        with pytest.raises(ValueError):
            session.ask('observe')
    assert calls == [] and session.history == () and len(client.requests) == 1


def test_session_root_different_from_host_rejects_before_reader(monkeypatch, tmp_path):
    module, reader = api('tool'), api('reader')
    host, session_root = tmp_path / 'host', tmp_path / 'session'
    host.mkdir()
    session_root.mkdir()
    calls = []
    def forbidden(*args):
        calls.append(True)
        raise AssertionError('wrong Session root reached Reader')
    patch_alias(monkeypatch, reader, 'observe_artifact', module, forbidden)
    session = CopilotSession(Client(), model='offline', experiment_context={'experiment_dir': str(session_root)})
    with bound_tools(BoundToolCollection(module.ArtifactObservationTools(binding(str(host))))):
        with pytest.raises(ValueError):
            session.ask('observe')
    assert calls == [] and session.history == ()


def test_real_session_authorized_dot_selected_file_usage_and_events(tmp_path):
    cls = api('tool').ArtifactObservationTools
    (tmp_path / 'artifact.bin').write_bytes(b'abc')
    (tmp_path / 'unselected.bin').write_bytes(b'not selected')
    client = Client()
    session = CopilotSession(client, model='offline', experiment_context={'experiment_dir': str(tmp_path)})
    collection = BoundToolCollection(cls(binding(str(tmp_path))))
    with bound_tools(collection):
        observed = session.ask_with_observability('Observe the selected artifact')
    assert len(client.requests) == observed.metrics.provider_request_count == 2
    assert observed.metrics.tool_invocation_count == 1
    assert client.requests[0]['tools'] == collection.list_tools()
    invocation = observed.turn.tool_invocations[0]
    assert (invocation.tool_name, invocation.tool_call_id) == (TOOL, 'observation-call')
    result = json.loads(invocation.result_json)
    assert_report(result)
    assert result['sha256'] == ABC_SHA and result['byte_size'] == 3
    delivered = [m for m in client.requests[1]['messages'] if m['role'] == 'tool']
    assert len(delivered) == 1 and json.loads(delivered[0]['content']) == result
    events = observed.run.events
    assert [e.sequence for e in events] == list(range(len(events)))
    assert [e.kind for e in events] == ['run.started', 'provider.request.started',
        'provider.response.received', 'tool.calls.validation.started', 'tool.execution.started',
        'tool.result.accepted', 'provider.request.started', 'provider.response.received', 'run.completed']
    for event in [e for e in events if e.tool_name is not None]:
        assert (event.run_id, event.tool_name, event.tool_call_id, event.tool_invocation_index) == (
            observed.run.run_id, TOOL, 'observation-call', 0)
    assert observed.run.usage.complete
    assert (observed.run.usage.input_tokens, observed.run.usage.output_tokens, observed.run.usage.total_tokens) == (20, 6, 26)
    assert session.history == (observed.turn,)


@pytest.mark.parametrize('mode', ['invalid-args', 'reader-bug', 'projection-bug', 'final-provider'])
def test_failure_original_exception_history_rollback_binding_restore(monkeypatch, tmp_path, mode):
    module, reader = api('tool'), api('reader')
    (tmp_path / 'artifact.bin').write_bytes(b'abc')
    failure = RuntimeError('failure sentinel')
    def fail(*args, **kwargs):
        raise failure
    if mode == 'reader-bug':
        patch_alias(monkeypatch, reader, 'observe_artifact', module, fail)
    elif mode == 'projection-bug':
        patch_alias(monkeypatch, api('core'), 'project_artifact_observation', reader, fail)
    client = Client({'experiment_dir': '.', 'approval': True} if mode == 'invalid-args' else None,
                    error=failure if mode == 'final-provider' else None)
    session = CopilotSession(client, model='offline', experiment_context={'experiment_dir': str(tmp_path)})
    outer, failures = BoundToolCollection(), []
    with bound_tools(outer):
        with pytest.raises((TypeError, ValueError) if mode == 'invalid-args' else RuntimeError) as caught:
            with bound_tools(BoundToolCollection(module.ArtifactObservationTools(binding(str(tmp_path))))):
                session.ask_with_observability('observe', on_failure=failures.append)
        if mode != 'invalid-args':
            assert caught.value is failure
        assert _get_bound_tools() is outer and session.history == () and session.turn_count == 0
        assert len(client.requests) == (2 if mode == 'final-provider' else 1)
        assert len(failures) == 1 and failures[0].run.events[-1].kind == 'run.failed'
        # Existing Runtime counts attempted executions, including failed dispatch.
        assert failures[0].tool_invocation_count == 1
        assert failures[0].stage == ('second_provider_request' if mode == 'final-provider' else 'tool_execution')
        with pytest.raises(KeyError):
            CopilotSession(Client(), model='offline', experiment_context={'experiment_dir': str(tmp_path)}).ask('unbound')


def test_missing_artifact_can_complete_without_false_observed_claim(tmp_path):
    cls = api('tool').ArtifactObservationTools
    session = CopilotSession(Client(), model='offline', experiment_context={'experiment_dir': str(tmp_path)})
    with bound_tools(BoundToolCollection(cls(binding(str(tmp_path))))):
        observed = session.ask_with_observability('observe missing artifact')
    assert_report(json.loads(observed.turn.tool_invocations[0].result_json), 'missing')
    assert observed.run.events[-1].kind == 'run.completed'


def test_formal_policy_rejects_link_escape_before_bound_dispatch(tmp_path):
    from test_artifact_observation_windows import junction, require_ntfs
    require_ntfs(tmp_path)
    allowed, outside = tmp_path / 'allowed', tmp_path / 'outside'
    allowed.mkdir()
    outside.mkdir()
    calls = []
    class Probe:
        def list_tools(self):
            return [{'type': 'function', 'function': {'name': TOOL, 'description': 'probe', 'parameters': PARAMETERS}}]
        def invoke_tool(self, *args):
            calls.append(args)
            raise AssertionError('Missing Tool / Capability / Integration: link escape reached dispatch')
    with junction(allowed / 'link', outside) as link:
        session = CopilotSession(Client({'experiment_dir': str(link)}), model='offline',
                                 experiment_context={'experiment_dir': str(allowed)})
        with bound_tools(BoundToolCollection(Probe())):
            with pytest.raises(ValueError):
                session.ask('outside link')
    assert calls == [] and session.history == ()

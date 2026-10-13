"""Missing delivery behavior fails inside test bodies, never at collection."""

from copy import deepcopy
import importlib
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from m23_release_contract_helpers import (
    DEMO, GOLDEN, ROOT, TOOLS, assert_demo, display, load_future, offline, strict_json,
)


@pytest.mark.parametrize('scenario', list('ABCDEFG'))
def test_independent_scenario_real_session_binding_core_and_golden(monkeypatch, scenario):
    # Spies call originals. Counting constructor-only Core calls is insufficient:
    # require at least one real Core/Reader call DURING the Session invocation.
    module = load_future(DEMO, monkeypatch)
    from copilot.session import CopilotSession
    from llm_adapters.tool_binding import BoundToolCollection, _get_bound_tools
    owners = {
        'A': ('tool_layer.experiment_tools', 'analyze_experiment_evidence'),
        'B': ('experiment_comparison', 'compare_experiment_evidence'),
        'C': ('experiment_replicates', 'aggregate_replicate_evidence'),
        'D': ('experiment_group_comparison', 'compare_replicate_groups'),
        'E': ('experiment_paired_comparison', 'compare_paired_replicates'),
        'F': ('experiment_artifact_verification', 'compare_artifact_descriptors'),
        'G': ('artifact_observation_reader', 'observe_artifact'),
    }
    owner_name, name = owners[scenario]
    owner = importlib.import_module(owner_name)
    original_core = getattr(owner, name)
    original_session = CopilotSession.ask_with_observability
    original_dispatch = BoundToolCollection.invoke_tool
    sessions, observations, dispatches, computations = [], [], [], []
    active = []

    def core(*args, **kwargs):
        result = original_core(*args, **kwargs)
        if active:
            computations.append(deepcopy(result))
        return result

    # Include already imported aliases without replacing any computation.
    for consumer in tuple(sys.modules.values()):
        if consumer is None or not getattr(consumer, '__name__', '').startswith(
                ('tool_layer', 'experiment_', '_m23_contract_')):
            continue
        for alias, value in list(vars(consumer).items()):
            if value is original_core:
                monkeypatch.setattr(consumer, alias, core)
    monkeypatch.setattr(owner, name, core)

    def session(self, *args, **kwargs):
        sessions.append(self)
        assert _get_bound_tools() is not None
        active.append(True)
        try:
            result = original_session(self, *args, **kwargs)
            observations.append(result)
            return result
        finally:
            active.pop()

    def dispatch(self, tool_name, arguments):
        assert active, 'tool must execute through Session'
        result = original_dispatch(self, tool_name, arguments)
        dispatches.append((tool_name, deepcopy(result)))
        return result

    monkeypatch.setattr(CopilotSession, 'ask_with_observability', session)
    monkeypatch.setattr(BoundToolCollection, 'invoke_tool', dispatch)
    calls = offline(monkeypatch)
    before = _get_bound_tools()
    report = module.run_scenario(scenario)
    assert calls == [] and _get_bound_tools() is before
    if scenario == 'G' and sys.platform != 'win32':
        assert report['status'] == 'unsupported'
        assert report['facts']['status'] == 'unsupported_platform'
        assert 'sha256' not in report['facts'] or report['facts']['sha256'] is None
        return  # Explicit unsupported contract, NOT native PASS or pytest.skip.
    assert_demo(report, scenario)
    assert len(sessions) == len(observations) == len(dispatches) == 1
    assert sessions[0].turn_count == 1 and computations
    assert dispatches[0][0] == TOOLS[scenario]
    assert report['result'] == display(dispatches[0][1])
    observed = observations[0]
    assert len(observed.turn.tool_invocations) == 1
    assert strict_json(observed.turn.tool_invocations[0].result_json) == dispatches[0][1]
    assert strict_json(observed.turn.answer)['facts'] == GOLDEN[scenario]


@pytest.mark.parametrize('scenario', list('ABCDEF'))
def test_repeated_result_is_deterministic_detached_and_no_observation_composition(monkeypatch, scenario):
    module = load_future(DEMO, monkeypatch)
    import artifact_observation_reader as m21
    def forbidden(*args, **kwargs):
        raise AssertionError('A-F must not automatically acquire M21 observations')
    monkeypatch.setattr(m21, 'observe_artifact', forbidden)
    offline(monkeypatch)
    first = module.run_scenario(scenario)
    second = module.run_scenario(scenario)
    assert_demo(first, scenario)
    assert first == second and first is not second
    first['facts'].clear()
    assert second['facts'] == GOLDEN[scenario]


@pytest.mark.parametrize('scenario', ['A', 'E', 'F'])
def test_cli_emits_single_json_without_credentials_or_user_paths(tmp_path, monkeypatch, scenario):
    load_future(DEMO, monkeypatch)
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE='1')
    for key in tuple(env):
        if any(token in key.upper() for token in ('API_KEY', 'OPENAI', 'ANTHROPIC', 'AZURE')):
            env.pop(key)
    # CLI invocation complements the in-process network-denial tests above.
    result = subprocess.run([sys.executable, '-B', str(DEMO), '--scenario', scenario],
                            cwd=tmp_path, env=env, capture_output=True, text=True, timeout=60)
    assert result.returncode == 0, result.stderr
    report = strict_json(result.stdout)
    assert_demo(report, scenario)
    assert str(ROOT) not in result.stdout and str(tmp_path) not in result.stdout


def test_cli_invalid_selection_never_runs_scenario(monkeypatch, capsys):
    module = load_future(DEMO, monkeypatch)
    def deny(*args, **kwargs):
        raise AssertionError('invalid selection must not dispatch')
    monkeypatch.setattr(module, 'run_scenario', deny)
    try:
        code = module.main(['--scenario', 'ALL'])
    except SystemExit as error:
        code = error.code
    assert code == 2
    assert not capsys.readouterr().out


@pytest.mark.parametrize('mode', ['invalid-tool-args', 'final-provider-error'])
def test_failure_restores_binding_does_not_commit_history_or_leak_exception(monkeypatch, mode):
    module = load_future(DEMO, monkeypatch)
    from copilot.session import CopilotSession
    from llm_adapters.tool_binding import BoundToolCollection, bound_tools, _get_bound_tools
    original_factory = module._make_client
    original_session = CopilotSession.ask_with_observability
    sessions, failures = [], []

    def factory(scenario):
        client = original_factory(scenario)
        original_create = client.chat.completions.create
        count = 0
        def create(**request):
            nonlocal count
            count += 1
            if count == 2 and mode == 'final-provider-error':
                raise RuntimeError('SECRET C:\\Users\\PRIVATE\\key.txt')
            response = original_create(**request)
            if count == 1 and mode == 'invalid-tool-args':
                response.choices[0].message.tool_calls[0].function.arguments = '{"approval":true}'
            return response
        monkeypatch.setattr(client.chat.completions, 'create', create)
        return client

    def session(self, *args, **kwargs):
        sessions.append(self)
        kwargs['on_failure'] = failures.append
        return original_session(self, *args, **kwargs)

    monkeypatch.setattr(module, '_make_client', factory)
    monkeypatch.setattr(CopilotSession, 'ask_with_observability', session)
    offline(monkeypatch)
    previous = BoundToolCollection()
    with bound_tools(previous):
        report = module.run_scenario('E')
        assert _get_bound_tools() is previous
    assert len(sessions) == 1 and sessions[0].history == () and failures
    assert report['status'] == 'error' and report['error_code'] == 'execution_failed'
    assert report['result'] is None
    assert not any(token in json.dumps(report) for token in ('SECRET', 'PRIVATE', 'key.txt'))
    monkeypatch.setattr(module, 'run_scenario', lambda name: deepcopy(report))
    assert module.main(['--scenario', 'E']) == 1


def test_g_unsupported_platform_cannot_claim_native_success(monkeypatch):
    module = load_future(DEMO, monkeypatch)
    import artifact_observation_io as backend
    monkeypatch.setattr(backend, '_platform_name', lambda: 'linux')
    def deny(*args, **kwargs):
        raise AssertionError('unsupported backend must not open a native handle')
    if backend._native is not None:
        monkeypatch.setattr(backend._native, 'open', deny)
    offline(monkeypatch)
    report = module.run_scenario('G')
    assert report['status'] == 'unsupported'
    assert report['facts']['status'] == 'unsupported_platform'
    assert report['facts'].get('sha256') is None
    monkeypatch.setattr(module, 'run_scenario', lambda name: deepcopy(report))
    assert module.main(['--scenario', 'G']) == 3


def test_paired_unpaired_reports_expose_different_statistical_objects(monkeypatch):
    module = load_future(DEMO, monkeypatch)
    offline(monkeypatch)
    unpaired, paired = module.run_scenario('D'), module.run_scenario('E')
    assert_demo(unpaired, 'D')
    assert_demo(paired, 'E')
    p = next(iter(paired['result']['targets'].values()))
    rows = [p['pair_results'][key] for key in sorted(p['pair_results'])]
    assert [r['calculation']['raw_delta'] for r in rows] == [6., -1.]
    assert [(r['a']['value'], r['b']['value']) for r in rows] == [(1., 7.), (3., 2.)]
    assert all(not value for value in paired['result']['claim_boundaries'].values())
    assert 'pair_results' not in next(iter(unpaired['result']['targets'].values()))

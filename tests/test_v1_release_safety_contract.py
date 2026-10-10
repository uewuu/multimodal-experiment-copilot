"""Calibrate guards against actual callbacks; preserve frozen trust boundaries.

Named Python entrance guards are regression probes, not an OS sandbox claim.
No monkeypatch substitutes a successful Core or native observation result.
"""

from copy import deepcopy
import builtins
import io
import json
from pathlib import Path
import runpy
import socket
import subprocess
import sys

import pytest

from m23_release_contract_helpers import (
    DEMO, GOLDEN, ROOT, TOOLS, assert_demo, load_future, offline, strict_json,
)


@pytest.mark.parametrize('payload', ['{"a":1,"a":2}', '{"a":NaN}', '{"a":Infinity}'])
def test_machine_json_probe_rejects_ambiguous_or_nonfinite_results(payload):
    with pytest.raises(AssertionError):
        strict_json(payload)
    assert strict_json('{"a":false,"b":0}') == {'a': False, 'b': 0}


def test_network_guard_calibration_is_effective(monkeypatch):
    calls = offline(monkeypatch)
    with pytest.raises(AssertionError, match='network forbidden'):
        socket.create_connection(('invalid.example', 443))
    assert calls == ['network']


def test_existing_offline_demo_really_runs_with_network_guard(monkeypatch, capsys):
    calls = offline(monkeypatch)
    existing = runpy.run_path(str(ROOT / 'examples/deterministic_copilot_demo.py'))
    result = existing['build_demo_result']()
    assert result['runtime'] == {'provider_request_count': 2, 'tool_invocation_count': 1}
    assert result['tool_call']['name'] == 'compare_experiments'
    assert calls == []


def test_crossed_pair_golden_calibrated_against_frozen_real_core():
    from m22_contract_helpers import fixture, invoke
    from experiment_paired_comparison import compare_paired_replicates
    data = fixture(a=(1., 3.), b=(2., 7.))
    pairs = data['pairing_declaration']['pairs']
    pairs[0]['b_execution'], pairs[1]['b_execution'] = pairs[1]['b_execution'], pairs[0]['b_execution']
    result = invoke(compare_paired_replicates, data)
    row = next(iter(result['targets'].values()))
    assert [p['calculation']['raw_delta'] for p in row['pair_results'].values()] == GOLDEN['E']['pair_deltas']
    assert row['statistics']['mean'] == GOLDEN['E']['mean']
    assert result['pairing_source'] == 'host_declared'
    assert not any(result['claim_boundaries'].values())


@pytest.mark.parametrize('case', ['missing-pair-member', 'unpaired-conflict'])
def test_frozen_pair_coverage_and_full_group_gate_remain_authoritative(case):
    from m22_contract_helpers import fixture, invoke
    from experiment_paired_comparison import compare_paired_replicates
    data = fixture(a=(1., 3.), b=(2., 7.))
    if case == 'missing-pair-member':
        data['group_b']['executions'].pop()
    else:
        data['pairing_declaration']['pairs'].pop()
        data['group_a']['executions'][1]['provenance']['configuration_ref'] = 'conflict'
    result = invoke(compare_paired_replicates, data)
    row = next(iter(result['targets'].values()))
    if case == 'missing-pair-member':
        assert row['coverage']['missing_count'] == 1
        assert row['coverage']['coverage_ratio'] == 0.5
        assert row['summary_eligibility']['status'] == 'limited'
    else:
        assert result['pairing_coverage']['unpaired_a']
        assert row['summary_eligibility']['status'] == 'blocked'
        assert row['statistics']['count'] == 0
        assert all(p['calculation']['raw_delta'] is None for p in row['pair_results'].values())
    assert not any(result['claim_boundaries'].values())


def test_unpaired_and_aggregation_golden_calibrated_against_frozen_core():
    from m19_contract_helpers import pair, invoke, aggregate
    from experiment_group_comparison import compare_replicate_groups
    data = pair(a_values=(1., 3.), b_values=(2., 7.))
    result = invoke(compare_replicate_groups, data)
    row = next(iter(result['targets'].values()))
    assert result['comparison_policy'] == GOLDEN['D']['comparison_policy']
    assert row['comparison']['raw_delta'] == GOLDEN['D']['raw_delta']
    aggregated = aggregate(data[0])
    statistics = next(iter(aggregated['targets'].values()))['statistics']
    assert statistics['count'] == GOLDEN['C']['count']
    assert statistics['mean'] == GOLDEN['C']['mean']


def test_existing_registry_exposure_remains_exact_and_bound_names_not_dispatchable():
    from tool_layer import tool_registry
    assert [t['function']['name'] for t in tool_registry.list_tools()] == [
        'analyze_experiment', 'compare_experiments']
    for name in TOOLS.values():
        with pytest.raises(KeyError):
            tool_registry.invoke_tool(name, {})


@pytest.mark.parametrize('scenario', list('CDEF'))
def test_snapshot_scenarios_do_not_acquire_files_memory_or_start_processes(monkeypatch, scenario):
    module = load_future(DEMO, monkeypatch)
    import sqlite3
    calls = offline(monkeypatch)
    def deny(*args, **kwargs):
        calls.append('external_access')
        raise AssertionError('snapshot demo attempted external acquisition')
    with monkeypatch.context() as guard:
        for owner, name in ((builtins, 'open'), (io, 'open'), (Path, 'open'),
                            (Path, 'read_bytes'), (Path, 'write_bytes'), (Path, 'write_text'),
                            (sqlite3, 'connect'), (subprocess, 'run'), (subprocess, 'Popen')):
            guard.setattr(owner, name, deny)
        report = module.run_scenario(scenario)
    assert_demo(report, scenario)
    assert calls == []


def test_io_guard_calibration_rejects_before_opening_file(monkeypatch, tmp_path):
    calls = []
    def deny(*args, **kwargs):
        calls.append('open')
        raise AssertionError('guard')
    monkeypatch.setattr(builtins, 'open', deny)
    with pytest.raises(AssertionError, match='guard'):
        open(tmp_path / 'must-not-exist', 'wb')
    assert calls == ['open'] and not (tmp_path / 'must-not-exist').exists()


@pytest.mark.parametrize('scenario', list('ABG'))
def test_fixture_preparation_does_not_authorize_tool_writes(monkeypatch, scenario):
    module = load_future(DEMO, monkeypatch)
    from llm_adapters.tool_binding import BoundToolCollection
    original = BoundToolCollection.invoke_tool
    writes = []
    original_open = builtins.open
    original_io_open = io.open
    def check_open(delegate):
        def guarded(file, mode='r', *args, **kwargs):
            if any(flag in mode for flag in 'wax+'):
                writes.append('write')
                raise AssertionError('tool cannot write')
            return delegate(file, mode, *args, **kwargs)
        return guarded
    def deny(*args, **kwargs):
        writes.append('mutation')
        raise AssertionError('tool cannot mutate files')
    def invoke(self, *args, **kwargs):
        with monkeypatch.context() as guard:
            guard.setattr(builtins, 'open', check_open(original_open))
            guard.setattr(io, 'open', check_open(original_io_open))
            for name in ('write_text', 'write_bytes', 'unlink', 'rename', 'replace', 'touch', 'chmod'):
                guard.setattr(Path, name, deny)
            return original(self, *args, **kwargs)
    monkeypatch.setattr(BoundToolCollection, 'invoke_tool', invoke)
    offline(monkeypatch)
    report = module.run_scenario(scenario)
    if scenario == 'G' and sys.platform != 'win32':
        assert report['status'] == 'unsupported'
    else:
        assert_demo(report, scenario)
    assert writes == []


def test_native_g_does_not_auto_call_descriptor_comparison_or_old_parsers(monkeypatch):
    module = load_future(DEMO, monkeypatch)
    import experiment_artifact_verification as m20
    from tool_layer import experiment_tools
    calls = []
    def deny(*args, **kwargs):
        calls.append('composition')
        raise AssertionError('observation is independent from descriptor comparison/parsing')
    monkeypatch.setattr(m20, 'compare_artifact_descriptors', deny)
    monkeypatch.setattr(experiment_tools, 'analyze_experiment_evidence', deny)
    offline(monkeypatch)
    report = module.run_scenario('G')
    if sys.platform == 'win32':
        assert_demo(report, 'G')
        assert report['result']['observation_basis'] == [
            'host_declared', 'system_observed_metadata', 'system_observed_content',
            'hash_computed_by_copilot']
        assert 'checkpoint_binding_not_verified' in report['result']['limitations']
    else:
        assert report['status'] == 'unsupported'
    assert calls == []


def test_demo_invalid_path_request_rejects_before_reader(monkeypatch, tmp_path):
    module = load_future(DEMO, monkeypatch)
    import artifact_observation_reader as reader
    original_factory = module._make_client
    calls = []
    def deny(*args, **kwargs):
        calls.append('reader')
        raise AssertionError('unauthorized reader entry')
    monkeypatch.setattr(reader, 'observe_artifact', deny)
    outside = tmp_path / 'outside'
    outside.mkdir()
    def factory(scenario):
        client = original_factory(scenario)
        original = client.chat.completions.create
        def create(**request):
            response = original(**request)
            message = response.choices[0].message
            if message.tool_calls:
                message.tool_calls[0].function.arguments = json.dumps({'experiment_dir': str(outside)})
            return response
        monkeypatch.setattr(client.chat.completions, 'create', create)
        return client
    monkeypatch.setattr(module, '_make_client', factory)
    report = module.run_scenario('G')
    assert report['status'] in ('error', 'unsupported')
    if sys.platform == 'win32':
        assert report['status'] == 'error'
    assert calls == []

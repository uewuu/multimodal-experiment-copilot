"""Pair declaration, real M18/M11 reuse and hand-calculated M22 contracts."""
from copy import deepcopy
from dataclasses import asdict
import inspect
import json

import pytest
import experiment_comparability as m11
import experiment_replicates as m18
from m22_contract_helpers import (
    CLAIMS, LIMITS, aggregate, assert_json, assert_partition, core_api, empty_stats,
    fixture, invoke, key, many_targets, metric, no_access, pairs, patch_public,
    row, set_direction, trusted_m18,
)


def test_fixture_bridge_and_manual_trap_are_real_m18():
    data = fixture((1., 100.), (3.,))
    a, b = [aggregate(data[s]) for s in ('group_a', 'group_b')]
    assert a['targets']['quality-A']['statistics']['mean'] == 50.5
    assert b['targets']['quality-B']['statistics']['mean'] == 3.
    assert 3. - 50.5 == -47.5 and 3. - 1. == 2.
    assert_json(data)


def test_fixture_unpaired_conflict_really_blocks_m18_but_trimmed_group_passes():
    data = fixture((1., 100.), (3.,)); g = data['group_a']
    g['executions'][1]['provenance']['configuration_ref'] = 'conflicting-config'
    assert aggregate(g)['group']['group_status'] == 'blocked'
    g['executions'].pop(); g['group_declaration']['expected_executions'].pop()
    g['targets'][0]['members'].pop()
    assert aggregate(g)['group']['group_status'] == 'eligible'


def test_public_signature():
    p = inspect.signature(core_api()).parameters
    assert list(p) == ['group_a', 'group_b', 'pairing_declaration', 'targets']
    assert all(p[x].kind is inspect.Parameter.KEYWORD_ONLY for x in ('pairing_declaration', 'targets'))


def test_complete_shape_traceability_and_hand_calculated_statistics():
    result = invoke(core_api(), fixture()); target = row(result)
    assert set(result) == {'schema_version', 'comparison_policy', 'pairing_source', 'groups',
                           'pairing_coverage', 'targets', 'variant_context', 'claim_boundaries', 'limitations'}
    assert result['schema_version'] == 'paired-replicate-comparison-v1'
    assert result['comparison_policy'] == 'explicit-paired-descriptive-v1'
    assert result['pairing_source'] == 'host_declared'
    assert result['variant_context'] is None and result['claim_boundaries'] == CLAIMS
    assert LIMITS <= set(result['limitations'])
    assert target['summary_eligibility']['status'] == 'eligible'
    assert target['statistics'] == dict(count=2, mean=2.5, median=2.5,
        sample_std=pytest.approx(0.7071067811865476), min=2., max=3.,
        positive_count=2, zero_count=0, negative_count=0)
    for i, (name, item) in enumerate(target['pair_results'].items()):
        assert name == f'pair-{i:02d}'
        assert item['a_execution'] == pairs(fixture())[i]['a_execution']
        assert item['b_execution'] == pairs(fixture())[i]['b_execution']
        assert item['a']['value'] == [1., 2.][i]
        assert item['b']['value'] == [3., 5.][i]
        assert item['a']['source']['evidence_kind'] == 'csv_reported'
        assert item['calculation']['raw_delta'] == [2., 3.][i]
    assert_partition(target, 2); assert_json(result)
    for label, name in (('a', 'quality-A'), ('b', 'quality-B')):
        real = aggregate(fixture()['group_' + label])
        assert result['groups'][label] == real['group']
        for index, p in enumerate(target['pair_results'].values()):
            assert p[label]['source'] == real['targets'][name]['included_executions'][index]['source']
        assert set(real['targets'][name]['limitations']) <= set(target['limitations'])
    def ordered_codes(value):
        if type(value) is dict:
            for k, v in value.items():
                if k in ('reason_codes', 'limitations'): assert v == sorted(set(v))
                ordered_codes(v)
        elif type(value) is list:
            for v in value: ordered_codes(v)
    ordered_codes(result)


@pytest.mark.parametrize('a,b,mean,std,counts', [
    ((5.,), (1.,), -4., None, (0, 0, 1)),
    ((2., 2., 2.), (0., 2., 6.), 2 / 3, 3.0550504633038935, (1, 1, 1)),
])
def test_single_and_mixed_sign_statistics(a, b, mean, std, counts):
    t = row(invoke(core_api(), fixture(a, b))); s = t['statistics']
    assert s['mean'] == pytest.approx(mean)
    assert s['sample_std'] == (None if std is None else pytest.approx(std))
    assert tuple(s[k] for k in ('positive_count', 'zero_count', 'negative_count')) == counts
    if len(a) == 1:
        assert s['median'] == s['min'] == s['max'] == -4.
        assert t['summary_eligibility']['status'] == 'limited'


def test_paired_observations_not_unpaired_group_means():
    r = invoke(core_api(), fixture((1., 100.), (3.,))); t = row(r)
    assert t['statistics']['mean'] == 2. and t['statistics']['mean'] != -47.5
    assert t['summary_eligibility']['status'] == 'limited'
    c = r['pairing_coverage']
    assert c['a_expected_count'] == 2 and c['a_referenced_count'] == 1
    assert c['a_pairing_coverage'] == .5 and c['b_pairing_coverage'] == 1.
    assert len(c['unpaired_a']) == 1 and c['unpaired_b'] == []


@pytest.mark.parametrize('direction,delta,interpretation', [
    ('maximize', 2., 'b_favorable'), ('minimize', 2., 'a_favorable'),
    ('maximize', -2., 'a_favorable'), ('minimize', -2., 'b_favorable'),
    ('maximize', 0., 'equal'), ('minimize', 0., 'equal'),
    ('maximize', 1e-10, 'b_favorable'),
])
def test_raw_b_minus_a_orientation_swap_and_no_rounding(direction, delta, interpretation):
    f = core_api(); d = fixture((0.,), (delta,)); set_direction(d, direction)
    p = row(invoke(f, d))['pair_results']['pair-00']['calculation']
    assert p == dict(status='calculated', raw_delta=delta, direction=direction,
                     direction_interpretation=interpretation)
    d['group_a'], d['group_b'] = d['group_b'], d['group_a']
    for p in pairs(d): p['a_execution'], p['b_execution'] = p['b_execution'], p['a_execution']
    for m in d['targets']: m['a_target_id'], m['b_target_id'] = m['b_target_id'], m['a_target_id']
    assert row(invoke(f, d))['statistics']['mean'] == -delta


@pytest.mark.parametrize('case', ['duplicate-id', 'duplicate-a', 'duplicate-b', 'self',
                                  'unknown', 'wrong-side', 'empty', 'source', 'extra-field'])
def test_malformed_pairing_rejected_not_repaired(case):
    f, d = core_api(), fixture(); p = pairs(d)
    if case == 'duplicate-id': p[1]['pair_id'] = p[0]['pair_id']
    elif case in ('duplicate-a', 'duplicate-b'):
        field = case[-1] + '_execution'; p[1][field] = deepcopy(p[0][field])
    elif case in ('self', 'wrong-side'): p[0]['b_execution'] = deepcopy(p[0]['a_execution'])
    elif case == 'unknown': p[0]['a_execution']['execution_id'] = 'absent'
    elif case == 'empty': p.clear()
    elif case == 'source': d['pairing_declaration']['pairing_source'] = 'system_verified'
    else: p[0]['auto_pair_by_seed'] = True
    with pytest.raises(ValueError): invoke(f, d)


def test_full_identity_overlap_in_unsupplied_unpaired_roster_rejects():
    f, d = core_api(), fixture(); pairs(d).pop()
    old = d['group_b']['group_declaration']['expected_executions'][1]['execution']
    old.clear(); old.update(deepcopy(pairs(d)[0]['a_execution']))
    d['group_b']['executions'].pop()
    d['group_b']['targets'][0]['members'][1]['execution'] = deepcopy(old)
    with pytest.raises(ValueError): invoke(f, d)


def test_same_execution_label_across_distinct_experiments_is_not_overlap():
    f, d = core_api(), fixture()
    for g in (d['group_a'], d['group_b']):
        for i, (s, r, m) in enumerate(zip(g['executions'], g['group_declaration']['expected_executions'], g['targets'][0]['members'])):
            for identity in (s['execution'], s['provenance']['execution'], r['execution'], m['execution']):
                identity['execution_id'] = f'same-{i}'
    for i, p in enumerate(pairs(d)):
        p['a_execution']['execution_id'] = p['b_execution']['execution_id'] = f'same-{i}'
    assert row(invoke(f, d))['summary_eligibility']['status'] == 'eligible'


@pytest.mark.parametrize('side', ['a', 'b', 'both'])
@pytest.mark.parametrize('kind', ['snapshot', 'metric', 'unavailable'])
def test_missing_keeps_declared_denominator_and_partition(side, kind):
    f, d = core_api(), fixture()
    for s in ('a', 'b') if side == 'both' else (side,):
        g = d['group_' + s]
        if kind == 'snapshot': g['executions'].pop()
        elif kind == 'unavailable': g['executions'][1]['status'] = 'unavailable'
        else: g['executions'][1]['evidence']['test_results']['metrics'].clear()
    t = row(invoke(f, d)); c = t['coverage']; p = t['pair_results']['pair-01']
    assert_partition(t, 2)
    assert (c['included_count'], c['missing_count'], c['coverage_ratio']) == (1, 1, .5)
    assert p['calculation']['raw_delta'] is None
    assert p['missing_a'] is (side in ('a', 'both'))
    assert p['missing_b'] is (side in ('b', 'both'))
    assert t['statistics']['mean'] == 2. and t['summary_eligibility']['status'] == 'limited'


def test_no_snapshots_no_fabricated_provenance_or_zero():
    f, d = core_api(), fixture(); d['group_a']['executions'].clear(); d['group_b']['executions'].clear()
    t = row(invoke(f, d)); assert_partition(t, 2)
    assert t['summary_eligibility']['status'] == 'insufficient'
    assert t['statistics'] == empty_stats()
    assert all(p['metric_comparability'] is None for p in t['pair_results'].values())


@pytest.mark.parametrize('field,value,status', [
    ('configuration_ref', 'conflict', 'blocked'), ('configuration_ref', None, 'insufficient'),
    ('metric_definition', 'conflict', 'blocked'), ('metric_definition', None, 'insufficient'),
    ('training_protocol_ref', 'conflict', 'blocked'), ('training_protocol_ref', None, 'insufficient'),
])
def test_unpaired_member_cannot_be_dropped_to_bypass_full_m18_gate(field, value, status):
    f, d = core_api(), fixture((1., 100.), (3.,)); g = d['group_a']
    if field == 'metric_definition': g['targets'][0]['members'][1][field] = value
    elif field == 'training_protocol_ref': g['executions'][1][field] = value
    else: g['executions'][1]['provenance'][field] = value
    t = row(invoke(f, d)); assert t['summary_eligibility']['status'] == status
    assert t['statistics'] == empty_stats()
    assert t['pair_results']['pair-00']['calculation']['raw_delta'] is None
    assert t['coverage']['excluded_count'] == 1


def test_two_real_public_m18_calls_full_groups_and_delegate_isolation(monkeypatch):
    f, d, seen = core_api(), fixture((1., 100.), (3.,)), []
    original = m18.aggregate_replicate_evidence; before = deepcopy(d)
    def spy(executions, *, group_declaration, targets):
        seen.append(deepcopy(dict(executions=executions, group_declaration=group_declaration, targets=targets)))
        result = original(executions, group_declaration=group_declaration, targets=targets)
        executions.clear()  # a delegate cannot corrupt the retained cross-pair declarations
        return result
    patch_public(monkeypatch, m18, 'aggregate_replicate_evidence', spy)
    r = invoke(f, d)
    assert seen == [trusted_m18(d['group_a']), trusted_m18(d['group_b'])]
    assert row(r)['statistics']['mean'] == 2. and d == before
    assert r['groups']['a'] == aggregate(before['group_a'])['group']


@pytest.mark.parametrize('field', ['task', 'target', 'dataset_version', 'split_id',
                                 'metric_definition', 'direction', 'aggregation',
                                 'evaluation_protocol', 'selection_protocol'])
@pytest.mark.parametrize('mode,status', [('conflict', 'blocked'), ('unknown', 'insufficient')])
def test_real_cross_m11_policy_fields(field, mode, status):
    f, d = core_api(), fixture(); g = d['group_b']; value = 'different' if mode == 'conflict' else None
    if field in ('task', 'target', 'dataset_version', 'split_id'):
        for s in g['executions']: s['provenance'][field] = value
    elif field == 'direction':
        for m in g['targets'][0]['members']:
            if mode == 'unknown': m['metric_spec'] = None
            else: m['metric_spec']['direction'] = 'minimize'
        for s in g['executions']: s['evidence']['test_results']['metrics']['metric_score']['direction'] = 'minimize'
    else:
        for m in g['targets'][0]['members']: m[field] = value
    t = row(invoke(f, d))
    assert t['summary_eligibility']['status'] == status and t['statistics'] == empty_stats()


@pytest.mark.parametrize('absent', [False, True])
def test_public_m11_receives_only_declared_cross_pairs_original_operands(monkeypatch, absent):
    f, d, seen = core_api(), fixture(), []
    if absent: d['group_b']['executions'].pop()
    else: d['group_b']['executions'][1]['evidence']['test_results']['metrics'].clear()
    original = m11.compare_metric_results
    def spy(a, b):
        if a.provenance.execution.experiment != b.provenance.execution.experiment:
            seen.append((a, b))
        return original(a, b)
    patch_public(monkeypatch, m11, 'compare_metric_results', spy)
    invoke(f, d)
    assert len(seen) == (1 if absent else 2)
    for i, (a, b) in enumerate(seen):
        assert asdict(a.provenance) == d['group_a']['executions'][i]['provenance']
        assert asdict(b.provenance) == d['group_b']['executions'][i]['provenance']
        assert a.value == [1., 2.][i] and b.value == [3., None][i]


def test_known_conflict_wins_over_unknown_and_cannot_salvage_other_pair():
    f, d = core_api(), fixture()
    d['group_b']['executions'][1]['provenance']['split_id'] = 'other'
    d['group_a']['targets'][0]['members'][0]['evaluation_protocol'] = None
    t = row(invoke(f, d)); assert t['summary_eligibility']['status'] == 'blocked'
    assert t['statistics'] == empty_stats()
    assert all(p['calculation']['raw_delta'] is None for p in t['pair_results'].values())


@pytest.mark.parametrize('case', ['empty', 'unknown', 'duplicate', 'unmapped', 'extra'])
def test_explicit_target_mapping_errors(case):
    f, d = core_api(), fixture()
    if case == 'empty': d['targets'].clear()
    elif case == 'unknown': d['targets'][0]['a_target_id'] = 'absent'
    elif case == 'duplicate': d['targets'].append(deepcopy(d['targets'][0]))
    elif case == 'unmapped': many_targets(d, 2); d['targets'].pop()
    else: d['targets'][0]['metric_name'] = 'metric_score'
    with pytest.raises(ValueError): invoke(f, d)


def test_equal_names_seeds_and_order_do_not_supply_missing_declarations():
    f, d = core_api(), fixture()
    for g in (d['group_a'], d['group_b']): g['targets'][0]['target_id'] = 'same-target'
    d['targets'].clear()
    with pytest.raises(ValueError): invoke(f, d)
    d['targets'] = [dict(target_id='quality', a_target_id='same-target', b_target_id='same-target')]
    pairs(d).clear()
    with pytest.raises(ValueError): invoke(f, d)


def test_matching_artifact_descriptors_never_override_incompatible_pairs():
    f, d = core_api(), fixture()
    for g in (d['group_a'], d['group_b']):
        for s in g['executions']:
            s['snapshot_metadata']['artifact'] = {'sha256': 'a' * 64, 'byte_size': 10, 'status': 'observed'}
    for s in d['group_b']['executions']: s['provenance']['split_id'] = 'different'
    t = row(invoke(f, d))
    assert t['summary_eligibility']['status'] == 'blocked' and t['statistics'] == empty_stats()


def test_section_mismatch_blocks_and_validation_preserves_epoch():
    f, d = core_api(), fixture(section='validation')
    t = row(invoke(f, d)); assert t['statistics']['mean'] == 2.5
    assert t['pair_results']['pair-00']['a']['source']['best_epoch'] == 0
    d['group_b'] = fixture()['group_b']
    t = row(invoke(f, d)); assert t['summary_eligibility']['status'] == 'blocked'
    assert 'section_mismatch' in t['summary_eligibility']['reason_codes']


def test_target_specific_coverage_no_cross_target_complete_case_intersection():
    f, d = core_api(), fixture(); many_targets(d, 2)
    for g in (d['group_a'], d['group_b']):
        for m in g['targets'][1]['members']:
            m['metric_name'] = m['metric_spec']['name'] = 'metric_aux'
        for s in g['executions']:
            metrics = s['evidence']['test_results']['metrics']
            metrics['metric_aux'] = deepcopy(metrics['metric_score'])
    del d['group_b']['executions'][1]['evidence']['test_results']['metrics']['metric_aux']
    r = invoke(f, d)
    assert r['targets']['c-00']['coverage']['included_count'] == 2
    assert r['targets']['c-01']['coverage']['included_count'] == 1
    assert set(r['targets']) == {'c-00', 'c-01'}


def test_seed_protocol_context_never_extends_m11_or_pairs_automatically():
    f, d = core_api(), fixture()
    for i, s in enumerate(d['group_b']['executions']):
        s['training_protocol_ref'] = 'different-training'
        s['provenance']['realized_seed'] += 100
        d['group_b']['group_declaration']['expected_executions'][i]['expected_seed'] += 100
    d['group_b']['group_declaration']['training_protocol_ref'] = 'different-training'
    r = invoke(f, d)
    assert row(r)['statistics']['mean'] == 2.5 and r['claim_boundaries'] == CLAIMS
    pairs(d).clear()
    with pytest.raises(ValueError): invoke(f, d)


def test_optional_variant_metadata_remains_host_declared_and_rejects_control_engine():
    f, d = core_api(), fixture()
    context = dict(baseline_label='baseline', candidate_label='variant', host_declared_changed_factor='optimizer')
    d['pairing_declaration']['variant_context'] = context
    r = invoke(f, d)
    assert r['variant_context'] == dict(context, source='host_declared')
    assert r['claim_boundaries'] == CLAIMS
    context['verified_controls'] = True
    with pytest.raises(ValueError): invoke(f, d)


def test_permutation_determinism_and_input_result_isolation():
    f, d = core_api(), fixture(); many_targets(d, 2)
    before = deepcopy(d); expected = invoke(f, d); assert d == before
    for g in (d['group_a'], d['group_b']):
        g['executions'].reverse(); g['group_declaration']['expected_executions'].reverse()
        g['targets'].reverse()
        for t in g['targets']: t['members'].reverse()
    pairs(d).reverse(); d['targets'].reverse()
    def reverse(x):
        if type(x) is dict: return {k: reverse(v) for k, v in reversed(list(x.items()))}
        if type(x) is list: return [reverse(v) for v in x]
        return x
    assert invoke(f, reverse(d)) == expected
    detached = deepcopy(expected); d['group_a']['executions'].clear()
    assert expected == detached
    expected['targets'].clear(); assert invoke(f, before) == detached


def test_core_has_no_io_or_artifact_or_memory_dependency(monkeypatch):
    f, d = core_api(), fixture()
    with no_access(monkeypatch) as calls:
        r = invoke(f, d)
    assert calls == [] and row(r)['statistics']['mean'] == 2.5


def test_static_core_boundary_no_m18_private_helpers_or_acquisition():
    import ast
    f = core_api()
    source = inspect.getsource(inspect.getmodule(f))
    tree = ast.parse(source)
    forbidden = {'os', 'io', 'pathlib', 'socket', 'requests', 'urllib', 'subprocess',
                 'hashlib', 'experiment_memory', 'experiment_repository',
                 'experiment_group_comparison', 'experiment_artifact_verification',
                 'artifact_observation_reader', 'artifact_observation_io', 'copilot',
                 'tool_layer', 'read_config', 'read_test_metrics', 'read_history'}
    aliases = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for name in node.names:
                assert name.name.split('.')[0] not in forbidden
                if name.name == 'experiment_replicates': aliases.add(name.asname or name.name)
        elif isinstance(node, ast.ImportFrom):
            assert (node.module or '').split('.')[0] not in forbidden
            if node.module == 'experiment_replicates':
                assert all(n.name == 'aggregate_replicate_evidence' for n in node.names)
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            leaf = node.func.id if isinstance(node.func, ast.Name) else getattr(node.func, 'attr', '')
            assert leaf not in {'open', 'read_bytes', 'write_bytes', 'CreateFileW', 'glob', 'rglob', 'listdir', 'scandir', 'walk'}
        if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name) and node.value.id in aliases:
            assert node.attr == 'aggregate_replicate_evidence'

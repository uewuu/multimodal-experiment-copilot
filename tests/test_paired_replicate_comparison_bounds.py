"""Inclusive budgets, strict numeric observations and finite paired arithmetic."""
from copy import deepcopy
from decimal import Decimal
import json

import pytest
from m22_contract_helpers import (
    aggregate, assert_json, assert_partition, core_api, empty_stats, fixture,
    invoke, many_targets, metric, pairs, row, size,
)


@pytest.mark.parametrize('count', [1, 16])
def test_pair_and_execution_cardinality_inclusive(count):
    f = core_api(); d = fixture(tuple(range(count)), tuple(range(1, count + 1)))
    t = row(invoke(f, d)); assert_partition(t, count)
    assert t['statistics']['mean'] == 1.


@pytest.mark.parametrize('count', [1, 32])
def test_target_count_inclusive(count):
    f, d = core_api(), fixture((1.,), (2.,)); many_targets(d, count)
    assert len(invoke(f, d)['targets']) == count


@pytest.mark.parametrize('case', ['expected-0', 'expected-17', 'supplied-17', 'pairs-0', 'pairs-17', 'targets-0', 'targets-33'])
def test_cardinality_rejections(case):
    f, d = core_api(), fixture(); name, count = case.split('-'); count = int(count)
    if name == 'expected':
        g = d['group_a']['group_declaration']; g['expected_executions'] = [deepcopy(g['expected_executions'][0]) for _ in range(count)]
    elif name == 'supplied': d['group_a']['executions'] *= 9
    elif name == 'pairs': d['pairing_declaration']['pairs'] = [deepcopy(pairs(d)[0]) for _ in range(count)]
    else: many_targets(d, count)
    with pytest.raises(ValueError): invoke(f, d)


@pytest.mark.parametrize('field,limit', [('pair_id', 128), ('basis', 256), ('label', 256), ('factor', 256), ('target_id', 128)])
@pytest.mark.parametrize('extra', [0, 1])
def test_utf8_text_limits(field, limit, extra):
    f, d = core_api(), fixture(); value = '\u6d4b' * (limit // 3) + 'x' * (limit % 3 + extra)
    assert len(value.encode('utf-8')) == limit + extra
    if field == 'pair_id': pairs(d)[0]['pair_id'] = value
    elif field == 'basis': pairs(d)[0]['declared_pair_basis'] = value
    elif field in ('label', 'factor'):
        d['pairing_declaration']['variant_context'] = {
            'baseline_label' if field == 'label' else 'host_declared_changed_factor': value}
    else: d['targets'][0]['target_id'] = value
    if extra:
        with pytest.raises(ValueError): invoke(f, d)
    else: assert invoke(f, d)['targets']


@pytest.mark.parametrize('component', ['namespace', 'experiment_id', 'execution_id'])
@pytest.mark.parametrize('extra', [0, 1])
def test_full_identity_component_limits(component, extra):
    f, d = core_api(), fixture((1.,), (2.,))
    value = 'x' * (128 + extra)
    def replace(x):
        if type(x) is dict:
            for k, v in x.items():
                if k == component: x[k] = value
                else: replace(v)
        elif type(x) is list:
            for v in x: replace(v)
    # Only A; same namespace/experiment labels in B must not create an overlap.
    replace(d['group_a']); replace(pairs(d)[0]['a_execution'])
    if extra:
        with pytest.raises(ValueError): invoke(f, d)
    else: assert row(invoke(f, d))['statistics']['mean'] == 1.


def padded(boundary, extra):
    d = fixture((1.,), (2.,))
    for s in ('group_a', 'group_b'): d[s]['executions'][0]['snapshot_metadata']['padding'] = ''
    a = d['group_a']['executions'][0]['snapshot_metadata']
    a['padding'] = 'x' * (2 * 1024 * 1024 - size(d['group_a']) + (extra if boundary == 'group' else 0))
    if boundary == 'total':
        b = d['group_b']['executions'][0]['snapshot_metadata']
        b['padding'] = 'x' * (4 * 1024 * 1024 - size(d) + extra)
    return d


@pytest.mark.parametrize('boundary', ['group', 'total'])
@pytest.mark.parametrize('extra', [0, 1])
def test_budget_fixture_calibration(boundary, extra):
    d = padded(boundary, extra)
    assert size(d['group_a'] if boundary == 'group' else d) == (2 if boundary == 'group' else 4) * 1024 * 1024 + extra
    if boundary == 'total': assert all(size(d[s]) <= 2 * 1024 * 1024 for s in ('group_a', 'group_b'))


@pytest.mark.parametrize('boundary', ['group', 'total'])
@pytest.mark.parametrize('extra', [0, 1])
def test_inclusive_group_and_combined_budget(boundary, extra):
    f, d = core_api(), padded(boundary, extra)
    if extra:
        with pytest.raises(ValueError, match='input|budget'): invoke(f, d)
    else: assert row(invoke(f, d))['statistics']['mean'] == 1.


def declaration_boundary(which, extra):
    if which == 'mapping':
        d = fixture((1.,), (2.,)); many_targets(d, 32)
        # Escaped control characters remain opaque identifier bytes. UTF8 label
        # budget and serialized JSON budget are distinct, both must apply.
        for entry in d['targets']:
            remaining = 16 * 1024 + extra - size(d['targets'])
            if remaining <= 0: break
            escaped = min(120, remaining // 6)
            entry['target_id'] += '\x01' * escaped
            remaining = 16 * 1024 + extra - size(d['targets'])
            if 0 < remaining < 6: entry['target_id'] += 'x' * remaining
        return d
    d = fixture(tuple(range(16)), tuple(range(16)))
    def expand(x):
        if type(x) is dict:
            for k, v in x.items():
                if k in ('namespace', 'experiment_id', 'execution_id'):
                    x[k] = v + 'x' * (96 - len(v))
                else: expand(v)
        elif type(x) is list:
            for v in x: expand(v)
    expand(d)
    for p in pairs(d):
        p['pair_id'] += 'x' * (100 - len(p['pair_id']))
        p['declared_pair_basis'] = ''
    # Optional nonempty strings: supply one character before filling budget.
    for p in pairs(d): p['declared_pair_basis'] = 'x'
    for p in pairs(d):
        remaining = 16 * 1024 + extra - size(d['pairing_declaration'])
        p['declared_pair_basis'] += 'x' * min(255, max(0, remaining))
    return d


@pytest.mark.parametrize('which', ['pairing', 'mapping'])
@pytest.mark.parametrize('extra', [0, 1])
def test_declaration_budget_fixture_calibration(which, extra):
    d = declaration_boundary(which, extra)
    assert size(d['targets'] if which == 'mapping' else d['pairing_declaration']) == 16 * 1024 + extra
    assert all(len(m['target_id'].encode('utf-8')) <= 128 for m in d['targets'])
    assert all(len(p['declared_pair_basis'].encode('utf-8')) <= 256 for p in pairs(d))
    assert all(size(aggregate(d[s])) < 128 * 1024 for s in ('group_a', 'group_b'))


@pytest.mark.parametrize('which', ['pairing', 'mapping'])
@pytest.mark.parametrize('extra', [0, 1])
def test_pairing_and_mapping_inclusive_serialized_budgets(which, extra):
    f, d = core_api(), declaration_boundary(which, extra)
    if extra:
        with pytest.raises(ValueError): invoke(f, d)
    else: assert invoke(f, d)['targets']


@pytest.mark.parametrize('where', ['mapping', 'pairing'])
def test_oversized_declaration_or_mapping_rejects(where):
    f, d = core_api(), fixture()
    if where == 'mapping': d['targets'][0]['target_id'] = 'x' * (16 * 1024)
    else: pairs(d)[0]['declared_pair_basis'] = 'x' * (16 * 1024)
    assert size(d['targets'] if where == 'mapping' else d['pairing_declaration']) > 16 * 1024
    with pytest.raises(ValueError): invoke(f, d)


@pytest.mark.parametrize('extra', [0, 1])
def test_group_depth_inclusive_and_cycles(extra):
    f, d, nested = core_api(), fixture(), {}
    for _ in range(11 + extra): nested = {'child': nested}
    d['group_a']['executions'][0]['snapshot_metadata']['payload'] = nested
    if extra:
        with pytest.raises(ValueError): invoke(f, d)
    else: assert row(invoke(f, d))['statistics']['mean'] == 2.5
    d['group_a']['executions'][0]['snapshot_metadata']['payload'] = d
    with pytest.raises(ValueError): invoke(f, d)


@pytest.mark.parametrize('place', ['pairing_declaration', 'targets'])
def test_cyclic_or_deep_pairing_and_mapping_reject(place):
    f, d = core_api(), fixture(); nested = {}
    for _ in range(9): nested = {'child': nested}
    if place == 'targets': d[place][0]['target_id'] = nested
    else: d[place]['pairs'][0]['declared_pair_basis'] = nested
    with pytest.raises((ValueError, TypeError)): invoke(f, d)


@pytest.mark.parametrize('bad', [True, False, float('nan'), float('inf'), -float('inf')])
def test_invalid_measurement_inherits_m18_local_exclusion(bad):
    f, d = core_api(), fixture(); metric(d, 'b', 1)['value'] = bad
    t = row(invoke(f, d)); assert_partition(t, 2)
    assert (t['coverage']['included_count'], t['coverage']['excluded_count']) == (1, 1)
    assert t['statistics']['mean'] == 2. and t['summary_eligibility']['status'] == 'limited'
    assert_json(t)


@pytest.mark.parametrize('bad', [Decimal('1'), complex(1, 0)])
def test_noncontract_numbers_rejected(bad):
    f, d = core_api(), fixture(); metric(d)['value'] = bad
    with pytest.raises(TypeError): invoke(f, d)


@pytest.mark.parametrize('a,b', [((-1e308,), (1e308,)), ((0., 0.), (-1.7e308, 1.7e308))])
def test_nonfinite_subtraction_or_sample_std_blocks(a, b):
    f, d = core_api(), fixture(a, b); t = row(invoke(f, d))
    assert t['summary_eligibility']['status'] == 'blocked'
    assert t['statistics'] == empty_stats(); assert_json(t)


def test_large_finite_mean_must_not_overflow_naive_sum():
    t = row(invoke(core_api(), fixture((0., 0.), (1e308, 1e308))))
    assert t['statistics']['mean'] == 1e308 and t['statistics']['sample_std'] == 0.
    assert_json(t)


def overflow_fixture():
    d = fixture((1.,), (2.,))
    for g in (d['group_a'], d['group_b']):
        # Escaping, not character length, pushes the merged source projection
        # over budget while each independent M18 report remains small enough.
        g['executions'][0]['evidence']['test_results']['source']['file'] = '\n' * 34000
    return d


def test_output_overflow_fixture_independent_m18_calibration():
    d = overflow_fixture()
    reports = [aggregate(d[s]) for s in ('group_a', 'group_b')]
    assert all(size(r) < 128 * 1024 for r in reports)
    sources = [r['targets'][name]['included_executions'][0]['source'] for r, name in zip(reports, ('quality-A', 'quality-B'))]
    assert size(sources) > 128 * 1024


def test_output_overflow_rejects_not_truncates_sources_or_pairs():
    f = core_api()
    with pytest.raises(ValueError, match='output'): invoke(f, overflow_fixture())


def test_nonfinite_metadata_not_measurement_is_rejected():
    f, d = core_api(), fixture()
    d['group_a']['executions'][0]['snapshot_metadata']['bad'] = float('nan')
    with pytest.raises(ValueError): invoke(f, d)


def test_utf8_and_json_escaping_count_not_character_length():
    f, d = core_api(), fixture()
    d['group_a']['executions'][0]['snapshot_metadata']['padding'] = '\u6d4b' * 800000
    assert size(d['group_a']) > 2 * 1024 * 1024
    with pytest.raises(ValueError): invoke(f, d)

"""M22 Scope Freeze: Explicit Paired Replicate Comparison, tests only.

PUBLIC: experiment_paired_comparison.compare_paired_replicates(group_a, group_b,
*, pairing_declaration, targets). PairedReplicateComparisonTools in
tool_layer.paired_experiment_tools has the same constructor; list_tools and
invoke_tool(name, arguments). Tool name compare_paired_replicates, empty schema.

INPUT: exact dict/list/scalars only, no external dataclasses/tuples/protocols.
Groups are M18 envelopes with fixed-field JSON projections of its four project
models. Only after safe bounded validation may production construct internal
ExperimentIdentity/ExecutionIdentity/ExperimentProvenance/MetricSpec instances.
MetricSpec.path is a JSON list externally. Missing declarations stay None.
Measurement bool/nonfinite floats inherit M18 local exclusion (the sole finite
input exception); nonfinite metadata rejects. Unsupported objects reject before
callbacks, including unused fields. All keys must be exact str before hashing,
set comparison or formatting; type checks must not call metaclass protocols.

PAIR DECLARATION: exact schema_version='explicit-pairing-v1',
pairing_source='host_declared', pairs, optional variant_context. Each pair has
pair_id, a_execution, b_execution, optional declared_pair_basis. Identity is
{experiment:{namespace,experiment_id},execution_id}. IDs are opaque, nonblank,
without surrounding whitespace. Unique pair IDs and unique endpoints per side;
any full identity overlap between group rosters rejects, including unpaired or
unsupplied members. Unknown/wrong-side endpoint rejects, expected-but-unsupplied
is missing. Empty pairs/mapping reject, never auto-discover. Context accepts only
optional baseline_label/candidate_label/host_declared_changed_factor strings.
No controlled-field engine. Context output adds source='host_declared'.
Targets are M19-shaped {target_id,a_target_id,b_target_id}, bijectively cover
the submitted source targets; no name/order matching or cross-section fallback.

DELEGATION: exactly two REAL public M18 aggregate_replicate_evidence calls per
valid invocation, with FULL groups, including unpaired members. Independent
safe copies protect original/stored/cross-assessment data. M18 owns observation
selection and within-group gates. Cross-side M11 compare_metric_results is
called for each declared pair with both snapshots supplied, per target; absent
snapshot has no fabricated operand, unavailable measurement is None. Operand
uses original provenance/declarations and finite observation, never group mean.
M11 calls nested inside M18 remain unchanged. Never call M18 private helpers,
M19 comparison, M20/M21, readers, Memory or a fake delta experiment.

OUTPUT root exactly schema_version='paired-replicate-comparison-v1',
comparison_policy='explicit-paired-descriptive-v1', pairing_source, groups,
pairing_coverage, targets, variant_context, claim_boundaries, limitations.
groups={a,b}: each is the real M18 group projection, detached (no full raw input).
pairing_coverage={declared_pair_count,a_expected_count,b_expected_count,
a_referenced_count,b_referenced_count,a_pairing_coverage,b_pairing_coverage,
unpaired_a,unpaired_b}. Unpaired identities sorted by full key.
Each target has pair_results keyed in sorted pair_id order, coverage,
summary_eligibility {status,reason_codes}, statistics, limitations.
Pair row: a_execution,b_execution,a,b,metric_comparability,partition,
missing_a,missing_b,calculation,reason_codes. Side={availability,value,source}:
availability included/missing/excluded from full M18 target; source is original
M18 included source or None. metric_comparability is M11 JSON projection or None
if either snapshot absent. calculation={status:calculated|blocked,raw_delta,
direction,direction_interpretation}; interpretation a_favorable/b_favorable/equal
or None, no improvement_delta. Delta always B-A, exact zero, no epsilon.

Per-target coverage={declared_pair_count,included_count,missing_count,
excluded_count,missing_a_count,missing_b_count,coverage_ratio}. Partition is
included/missing/excluded, mutually exclusive; any missing side => missing,
otherwise ineligible => excluded. Missing-side counts may overlap. Denominator
always declared count. Statistics count equals included count. Missing values
never become zero. Pair source limits and M18 root/target limits are retained.

SUMMARY precedence blocked > insufficient > limited > eligible. M18 blocked,
section mismatch, cross M11 incompatible, numeric failure => blocked. M18
insufficient or cross M11 unknown => insufficient. Never salvage a compatible
subset: on either global gate all otherwise included pairs become excluded,
all deltas null, stats empty; missing pairs remain missing. Independent side
values may remain for traceability. No included pairs => insufficient unless
blocked already. Otherwise unpaired roster/partial coverage/n=1 => limited.
At least two included, full declared and roster coverage => eligible.
Stats exactly count,mean,median,sample_std,min,max,positive_count,zero_count,
negative_count. Equal pair weights, sample_std denominator n-1. n=0 numeric
stats null/counts zero; n=1 std null. Nonfinite arithmetic => blocked, no partial
statistics. Known reasons include section_mismatch,numeric_error,
incomplete_pair_coverage,unpaired_executions,insufficient_variability; preserve
M18/M11 reasons. Baseline limitations alone do not downgrade eligibility.

DETERMINISM: all mapping keys/members/reasons/limits canonical; semantic input
permutations equal. Fresh isolated snapshots/results/schema on each call; no
cache. Bounds inclusive: 2 groups; each expected 1..16/supplied 0..16; pairs
1..16; targets 1..32 (<=512 rows). Identity components/group/target/pair IDs
<=128 UTF8 bytes; optional labels/factor/basis <=256. Per-group <=2MiB, pairing
<=16KiB, target mapping <=16KiB, whole {group_a,group_b,pairing_declaration,
targets} <=4MiB compact ensure_ascii=False JSON. Budget counts unused metadata
and escapes; invalid measurement tokens may be counted as in M18. Group depth
<=16 from own root=1, pairing/mapping <=8, cycles reject. Bounded primitive
checks precede encoding/copy. Output strict JSON <=128KiB, overflow ValueError
containing 'output', never truncate. Sub-budgets and total both apply.

SAFETY/OUT: no acquisition, I/O/network/Memory/config scan, artifact verification
or observation, hashing, training, writes, workflow, automatic pairing, N-way,
causality, inferential statistics, winner/recommendation, release work. No new
Registry/default dispatch/capability. Session/Adapter/Runtime unchanged. Exact
empty args reject before core, errors fixed/bounded without object formatting.
Evaluation gates structured claims only, not arbitrary natural-language safety.
Only tests/helpers in Slice 1; future Slice 2 Core, Slice 3 Tool, Slice 4 Freeze.
"""

from contextlib import contextmanager
from copy import deepcopy
from dataclasses import asdict
import importlib
import importlib.util
import json
from types import SimpleNamespace as NS

from experiment_identity import ExperimentIdentity, ExecutionIdentity, ExperimentProvenance
from metrics import MetricSpec
from m19_contract_helpers import group as old_group
from m18_contract_helpers import deny_external_access

TOOL = 'compare_paired_replicates'
PARAMETERS = {'type': 'object', 'properties': {}, 'additionalProperties': False}
CORE_MISSING = 'missing M22 core: experiment_paired_comparison.compare_paired_replicates'
TOOL_MISSING = 'missing M22 tool: tool_layer.paired_experiment_tools.PairedReplicateComparisonTools'
CLAIMS = dict.fromkeys((
    'same_seed_means_controlled', 'paired_delta_is_causal',
    'positive_mean_is_significant', 'all_positive_is_universally_robust',
    'host_factor_is_verified_only_change', 'partial_coverage_is_complete',
    'missing_metric_means_zero', 'unpaired_run_is_invalid',
    'explicit_pair_verifies_rationale', 'full_coverage_is_unbiased',
    'artifact_equality_establishes_pairing',
), False)
LIMITS = {'statistical_inference_not_performed', 'independence_not_verified',
          'roster_host_declared_not_preregistered', 'snapshot_currentness_not_verified',
          'pairing_host_declared_not_verified', 'causal_attribution_not_performed',
          'controlled_factors_not_verified'}


def core_api():
    assert importlib.util.find_spec('experiment_paired_comparison') is not None, CORE_MISSING
    value = getattr(importlib.import_module('experiment_paired_comparison'), TOOL, None)
    assert callable(value), CORE_MISSING
    return value


def tool_class():
    assert importlib.util.find_spec('tool_layer.paired_experiment_tools') is not None, TOOL_MISSING
    value = getattr(importlib.import_module('tool_layer.paired_experiment_tools'),
                    'PairedReplicateComparisonTools', None)
    assert callable(value), TOOL_MISSING
    return value


def fixture(a=(1., 2.), b=(3., 5.), *, section='test'):
    # Convert only trusted test builders, never arbitrary caller data.
    groups = [json.loads(json.dumps(old_group(label, values, section=section), default=asdict))
              for label, values in (('A', a), ('B', b))]
    for g in groups:
        for s in g['executions']:
            for field in ('validation_metrics',):
                metric = s['evidence'][field].pop('score')
                metric['metric_name'] = 'metric_score'
                s['evidence'][field]['metric_score'] = metric
            metric = s['evidence']['test_results']['metrics'].pop('score')
            s['evidence']['test_results']['metrics']['metric_score'] = metric
        for m in g['targets'][0]['members']:
            m['metric_name'] = m['metric_spec']['name'] = 'metric_score'
    pairs = [{'pair_id': f'pair-{i:02d}',
              'a_execution': deepcopy(groups[0]['group_declaration']['expected_executions'][i]['execution']),
              'b_execution': deepcopy(groups[1]['group_declaration']['expected_executions'][i]['execution']),
              'declared_pair_basis': 'host selected corresponding executions'}
             for i in range(min(len(a), len(b)))]
    return dict(group_a=groups[0], group_b=groups[1],
                pairing_declaration={'schema_version': 'explicit-pairing-v1',
                                     'pairing_source': 'host_declared', 'pairs': pairs},
                targets=[{'target_id': 'quality', 'a_target_id': 'quality-A', 'b_target_id': 'quality-B'}])


def invoke(api, data):
    return api(data['group_a'], data['group_b'],
               pairing_declaration=data['pairing_declaration'], targets=data['targets'])


def key(identity):
    return (identity['experiment']['namespace'], identity['experiment']['experiment_id'], identity['execution_id'])


def trusted_m18(group):
    """Fixture adapter to REAL M18, not a reference M22 implementation."""
    g = deepcopy(group)
    def identity(x):
        return ExecutionIdentity(ExperimentIdentity(**x['experiment']), x['execution_id'])
    d = g['group_declaration']
    d['experiment_identity'] = ExperimentIdentity(**d['experiment_identity'])
    for r in d['expected_executions']:
        r['execution'] = identity(r['execution'])
    for s in g['executions']:
        s['execution'] = identity(s['execution'])
        p = s['provenance']; p['execution'] = identity(p['execution'])
        s['provenance'] = ExperimentProvenance(**p)
    for t in g['targets']:
        for m in t['members']:
            m['execution'] = identity(m['execution'])
            if m['metric_spec'] is not None:
                m['metric_spec']['path'] = tuple(m['metric_spec']['path'])
                m['metric_spec'] = MetricSpec(**m['metric_spec'])
    return g


def aggregate(group):
    from experiment_replicates import aggregate_replicate_evidence
    g = trusted_m18(group)
    return aggregate_replicate_evidence(g['executions'], group_declaration=g['group_declaration'], targets=g['targets'])


def row(result):
    return result['targets']['quality']


def pairs(data):
    return data['pairing_declaration']['pairs']


def metric(data, side='a', index=0, section='test'):
    e = data['group_' + side]['executions'][index]['evidence']
    return (e['test_results']['metrics'] if section == 'test' else e['validation_metrics'])['metric_score']


def set_direction(data, direction):
    for side in ('a', 'b'):
        for m in data['group_' + side]['targets'][0]['members']:
            m['metric_spec']['direction'] = direction
        for s in data['group_' + side]['executions']:
            s['evidence']['test_results']['metrics']['metric_score']['direction'] = direction


def many_targets(data, count):
    for side in ('a', 'b'):
        g = data['group_' + side]
        g['targets'] = [dict(deepcopy(g['targets'][0]), target_id=f't-{i:02d}') for i in range(count)]
    data['targets'] = [dict(target_id=f'c-{i:02d}', a_target_id=f't-{i:02d}', b_target_id=f't-{i:02d}')
                       for i in range(count)]


def size(value):
    return len(json.dumps(value, ensure_ascii=False, separators=(',', ':')).encode('utf-8'))


def assert_json(value):
    kind = type(value)
    assert value is None or kind in (dict, list, str, int, float, bool)
    if kind is dict:
        assert all(type(k) is str for k in value)
        for item in value.values(): assert_json(item)
    if kind is list:
        for item in value: assert_json(item)
    json.dumps(value, allow_nan=False)


def assert_partition(target, count):
    rows, c = target['pair_results'], target['coverage']
    assert len(rows) == count and list(rows) == sorted(rows)
    assert c['declared_pair_count'] == count
    for partition in ('included', 'missing', 'excluded'):
        assert c[partition + '_count'] == sum(x['partition'] == partition for x in rows.values())
    assert sum(c[k + '_count'] for k in ('included', 'missing', 'excluded')) == count
    for side in ('a', 'b'):
        assert c['missing_' + side + '_count'] == sum(x['missing_' + side] for x in rows.values())
    assert c['coverage_ratio'] == c['included_count'] / count
    assert target['statistics']['count'] == c['included_count']


def empty_stats():
    return dict(count=0, mean=None, median=None, sample_std=None, min=None, max=None,
                positive_count=0, zero_count=0, negative_count=0)


def patch_public(monkeypatch, owner, name, wrapper):
    original = getattr(owner, name)
    monkeypatch.setattr(owner, name, wrapper)
    for module_name in ('experiment_paired_comparison', 'tool_layer.paired_experiment_tools'):
        import sys
        module = sys.modules.get(module_name)
        if module:
            for alias, value in list(vars(module).items()):
                if value is original: monkeypatch.setattr(module, alias, wrapper)


@contextmanager
def no_access(monkeypatch):
    # Scoped, calibrated public entrance sentinels; not an OS sandbox claim.
    import artifact_observation_reader as observation
    import experiment_artifact_verification as verification
    import experiment_group_comparison as unpaired
    import artifact_observation_io as artifact_io
    import os
    from pathlib import Path
    import subprocess
    calls = []
    def fail(*a, **kw):
        calls.append('forbidden')
        raise AssertionError('M22 attempted external or unrelated capability')
    with monkeypatch.context() as patch:
        for module, name in ((observation, 'observe_artifact'),
                             (verification, 'compare_artifact_descriptors'),
                             (unpaired, 'compare_replicate_groups'), (subprocess, 'Popen'),
                             (artifact_io, 'open_authorized_artifact'),
                             (os, 'listdir'), (os, 'scandir'), (os, 'walk'),
                             (Path, 'glob'), (Path, 'rglob')):
            patch.setattr(module, name, fail)
        with deny_external_access(patch) as inherited:
            yield calls
            assert inherited == []


class Client:
    """Only provider messages are simulated; no tool result doubles."""
    def __init__(self, args=None, *, final='offline answer', error=None):
        self.args = {} if args is None else args
        self.final, self.error, self.requests = final, error, []
        self.chat = NS(completions=self)

    def create(self, **kwargs):
        self.requests.append(deepcopy(kwargs))
        if len(self.requests) == 1:
            message = NS(role='assistant', content=None, tool_calls=[NS(
                id='paired-call', type='function', function=NS(name=TOOL, arguments=json.dumps(self.args)))])
        elif len(self.requests) == 2:
            if self.error is not None: raise self.error
            message = NS(role='assistant', content=self.final, tool_calls=None)
        else:
            raise AssertionError('unexpected provider retry')
        return NS(choices=[NS(message=message)], usage=NS(prompt_tokens=10, completion_tokens=4, total_tokens=14))

"""M13 per-case gates on structured paired evidence, not a prose classifier."""
from copy import deepcopy
import json

import pytest
from copilot.evaluation import EvaluationCase, EvaluationResult
from copilot.evaluation_runner import run_evaluation_case
from copilot.evaluation_batch import BatchEvaluationRunner, compare_batch_to_baseline
from copilot.evaluation_suite import BaselineManifest, EvaluationScenario, EvaluationSuite
from copilot.session import CopilotSession
from copilot.tool_binding import BoundCopilotSession
from llm_adapters.tool_binding import BoundToolCollection
from m22_contract_helpers import CLAIMS, Client, TOOL, fixture, invoke, tool_class


def data_for(name):
    d = fixture((1., 100.), (3.,)) if name == 'unpaired' else fixture()
    if name == 'partial': d['group_b']['executions'].pop()
    d['pairing_declaration']['variant_context'] = {'host_declared_changed_factor': 'optimizer'}
    return d


def case_for(name):
    mean, count, coverage, status = {
        'complete': (2.5, 2, 1., 'eligible'), 'partial': (2., 1, .5, 'limited'),
        'unpaired': (2., 1, 1., 'limited'),
    }[name]
    facts = dict(mean=mean, count=count, coverage=coverage, eligibility=status,
                 pairing_source='host_declared', context_source='host_declared', **CLAIMS)
    prefix = ['targets', 'quality']
    paths = dict(mean=prefix + ['statistics', 'mean'], count=prefix + ['statistics', 'count'],
                 coverage=prefix + ['coverage', 'coverage_ratio'],
                 eligibility=prefix + ['summary_eligibility', 'status'],
                 pairing_source=['pairing_source'], context_source=['variant_context', 'source'])
    paths.update({name: ['claim_boundaries', name] for name in CLAIMS})
    return EvaluationCase(name, '1', 'Return JSON facts, preserving declared pairing and limitations.',
                          facts, {'required_tools': [TOOL], 'fact_paths': paths})


def suite_definition():
    return EvaluationSuite('m22-paired-comparison', '1', [
        EvaluationScenario(case_for(name), fixture_version='generic-pairs-v1')
        for name in ('complete', 'partial', 'unpaired')],
        evaluation_profile='m10-structured-facts-v1', driver_version='m22-snapshot-agent-v1')


def run_case(name, facts):
    cls = tool_class(); client = Client(final=json.dumps({'facts': facts}))
    session = CopilotSession(client, model='offline')
    result = run_evaluation_case(case_for(name), lambda: BoundCopilotSession(
        session, BoundToolCollection(invoke(cls, data_for(name)))))
    assert len(client.requests) == 2 and session.turn_count == 1 and result.run_id
    evidence = json.loads(session.history[0].tool_invocations[0].result_json)
    assert evidence['comparison_policy'] == 'explicit-paired-descriptive-v1'
    return result


def test_evaluation_definition_roundtrip_and_literal_expectations():
    suite = suite_definition()
    assert EvaluationSuite.from_json(suite.to_json()).to_json() == suite.to_json()
    assert len(suite.scenarios) == 3
    assert case_for('partial').expected_facts['coverage'] == .5
    assert case_for('unpaired').expected_facts['mean'] == 2.
    assert all(value is False for value in CLAIMS.values())


@pytest.mark.parametrize('name', ['complete', 'partial', 'unpaired'])
def test_real_bound_tool_facts_pass_evaluation(name):
    result = run_case(name, case_for(name).expected_facts)
    assert result.passed and result.score == 1. and result.failure_reasons == ()


@pytest.mark.parametrize('claim', list(CLAIMS))
def test_unsupported_claims_fail_answer_and_grounding(claim):
    name = 'partial' if claim in ('partial_coverage_is_complete', 'missing_metric_means_zero') else 'complete'
    facts = deepcopy(case_for(name).expected_facts); facts[claim] = True
    result = run_case(name, facts)
    assert not result.passed and result.score == pytest.approx(1 / 3)
    assert result.failure_reasons == ('answer_facts', 'evidence_grounding')


@pytest.mark.parametrize('fact,value', [('coverage', 1.), ('mean', 0.), ('eligibility', 'eligible')])
def test_hiding_partial_coverage_or_zero_filling_fails(fact, value):
    facts = deepcopy(case_for('partial').expected_facts); facts[fact] = value
    result = run_case('partial', facts)
    assert not result.passed and 'evidence_grounding' in result.failure_reasons


def test_m13_batch_gate_catches_one_paired_coverage_regression():
    cls = tool_class(); suite = suite_definition(); manifest = BaselineManifest.from_suite(suite)
    baseline = tuple(EvaluationResult(s.key[0], '1', True, 1., ()) for s in suite.scenarios)
    original = manifest.to_json()
    def batch(corrupt):
        pending, sessions = iter(suite.scenarios), []
        def factory():
            scenario = next(pending); name = scenario.key[0]
            facts = deepcopy(scenario.case.expected_facts)
            if corrupt and name == 'partial': facts['coverage'] = 1.
            session = CopilotSession(Client(final=json.dumps({'facts': facts})), model='offline'); sessions.append(session)
            return BoundCopilotSession(session, BoundToolCollection(invoke(cls, data_for(name))))
        report = BatchEvaluationRunner(factory).run(suite)
        assert len({id(s) for s in sessions}) == 3 and all(s.turn_count == 1 for s in sessions)
        assert report.runner_errors == report.execution_failures == 0
        return report
    good, bad = batch(False), batch(True)
    assert good.passed == 3 and compare_batch_to_baseline(good, manifest, baseline).passed
    gate = compare_batch_to_baseline(bad, manifest, baseline)
    assert (bad.passed, bad.failed) == (2, 1) and not gate.passed
    assert 'regression:partial' in gate.reason_codes and manifest.to_json() == original

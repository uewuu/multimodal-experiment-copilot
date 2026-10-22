"""Structured negative-claim gate with real tiny-file Reader/Tool evidence."""
from copy import deepcopy
import json

import pytest

from copilot.evaluation import EvaluationCase
from copilot.evaluation_runner import run_evaluation_case
from copilot.session import CopilotSession
from copilot.tool_binding import BoundCopilotSession
from llm_adapters.tool_binding import BoundToolCollection
from m21_contract_helpers import (api, binding, Client, TOOL, ABC_SHA, LIMITATIONS,
                                  SUCCESS_BASIS, assert_report)


def case_for():
    facts = {'status': 'observed', 'sha256': ABC_SHA, 'byte_size': 3,
             'observation_basis': SUCCESS_BASIS, 'consistency_status': 'no_change_detected',
             'limitations': LIMITATIONS}
    return EvaluationCase('m21-observation', '1', 'Return JSON facts from the bound observation.',
                          facts, {'required_tools': [TOOL], 'fact_paths': {key: [key] for key in facts}})


def run_case(tmp_path, facts):
    cls = api('tool').ArtifactObservationTools
    (tmp_path / 'artifact.bin').write_bytes(b'abc')
    session = CopilotSession(Client(final=json.dumps({'facts': facts})), model='offline',
                             experiment_context={'experiment_dir': str(tmp_path)})
    result = run_evaluation_case(case_for(), lambda: BoundCopilotSession(
        session, BoundToolCollection(cls(binding(str(tmp_path))))))
    assert session.turn_count == 1 and result.run_id
    assert_report(json.loads(session.history[0].tool_invocations[0].result_json))
    return result


def test_allowed_observation_facts_grounded_in_real_read(tmp_path):
    result = run_case(tmp_path, case_for().expected_facts)
    assert result.passed and result.score == 1. and result.failure_reasons == ()


@pytest.mark.parametrize('claim', ['producer', 'checkpoint', 'current', 'locator', 'exists',
                                  'mtime', 'atomic', 'm16-parser'])
def test_observation_never_authorizes_stronger_structured_claim(tmp_path, claim):
    facts = deepcopy(case_for().expected_facts)
    facts['conclusion'] = {
        'producer': 'system observation proves trusted producer',
        'checkpoint': 'computed SHA proves checkpoint verified',
        'current': 'fresh read proves latest current artifact',
        'locator': 'same locator proves same bytes',
        'exists': 'file exists proves valid artifact',
        'mtime': 'same mtime proves historically unchanged',
        'atomic': 'no change detected proves atomic snapshot',
        'm16-parser': 'M21 observation proves M16 parser used these bytes',
    }[claim]
    result = run_case(tmp_path, facts)
    assert not result.passed and result.score == pytest.approx(1 / 3)
    assert result.failure_reasons == ('answer_facts', 'evidence_grounding')


def test_evaluation_definition_independent_of_future_modules():
    case = case_for()
    assert case.expected_facts['sha256'] == ABC_SHA
    assert case.expected_facts['consistency_status'] == 'no_change_detected'
    assert case.expected_facts['limitations'] == LIMITATIONS

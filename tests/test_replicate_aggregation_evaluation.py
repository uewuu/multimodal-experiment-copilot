"""M18 RED: M10 grounding and M13 case gates over real snapshot aggregation.

Scored structured facts only; this is not a universal free-text claim filter.
Expected answers below are literals independent of the future implementation.
"""

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
from m18_contract_helpers import TOOL, Client, bundle, make_tools, tool_class


def case_for(name):
    partial = name == "partial"
    facts = {
        "expected": 3, "observed": 2 if partial else 3,
        "included": 2 if partial else 3, "missing": 1 if partial else 0,
        "coverage": 2 / 3 if partial else 1.0,
        "mean": 2.0 if partial else 3.0,
        "membership_source": "host_declared",
        "status": "partial" if partial else "complete",
    }
    # M10 fact paths address mappings, not array indices. Source limitations
    # are asserted against real output separately, without exact-list coupling.
    paths = {
        "expected": ["targets", "quality", "coverage", "expected_count"],
        "observed": ["targets", "quality", "coverage", "observed_count"],
        "included": ["targets", "quality", "coverage", "included_count"],
        "missing": ["targets", "quality", "coverage", "missing_count"],
        "coverage": ["targets", "quality", "coverage", "coverage_ratio"],
        "mean": ["targets", "quality", "statistics", "mean"],
        "membership_source": ["group", "membership_source"],
        "status": ["targets", "quality", "target_status"],
    }
    return EvaluationCase(name, "1", "Return JSON facts about the bound replicate snapshots and their limits.",
                          facts, {"required_tools": [TOOL], "fact_paths": paths})


def data_for(name):
    data = bundle(seeds=(42, 42, 2024))
    if name == "partial": data[0].pop()
    return data


def run_case(name, facts):
    tools = make_tools(tool_class(), data_for(name))
    client = Client(final=json.dumps({"facts": facts}))
    sessions = []
    def factory():
        session = CopilotSession(client, model="offline")
        sessions.append(session)
        return BoundCopilotSession(session, BoundToolCollection(tools))
    result = run_evaluation_case(case_for(name), factory)
    assert len(sessions) == 1 and sessions[0].turn_count == 1
    assert len(client.requests) == 2 and result.run_id
    return result


@pytest.mark.parametrize("name", ["complete", "partial"])
def test_correct_coverage_missing_and_statistics_are_grounded(name):
    result = run_case(name, case_for(name).expected_facts)
    assert result.passed and result.score == 1.0 and result.failure_reasons == ()


@pytest.mark.parametrize("corruption", [
    "hide-missing", "observed-as-expected", "wrong-mean", "independent-seeds",
    "verified-checkpoint", "significant", "robust", "generalizes", "verified-stable",
    "winner", "causal", "state-of-the-art", "file-verified",
])
def test_negative_claims_fail_structured_facts_and_grounding(corruption):
    wrong = deepcopy(case_for("partial").expected_facts)
    if corruption == "hide-missing": wrong["missing"] = 0
    elif corruption == "observed-as-expected": wrong["observed"] = 3
    elif corruption == "wrong-mean": wrong["mean"] = 5.0
    elif corruption == "file-verified": wrong["membership_source"] = "file_verified"
    else:
        wrong["conclusion"] = {
            "independent-seeds": "verified independent replicates",
            "verified-checkpoint": "verified checkpoint metrics",
            "significant": "significantly better", "robust": "robustly better",
            "generalizes": "generalizes better", "verified-stable": "verified stable",
            "winner": "winner", "causal": "causal improvement", "state-of-the-art": "state-of-the-art",
        }[corruption]
    result = run_case("partial", wrong)
    assert not result.passed and result.score == pytest.approx(1 / 3)
    assert result.failure_reasons == ("answer_facts", "evidence_grounding")


def test_real_output_retains_limits_under_evaluation_driver():
    tools = make_tools(tool_class(), data_for("partial"))
    client = Client(final=json.dumps({"facts": case_for("partial").expected_facts}))
    session = CopilotSession(client, model="offline")
    result = run_evaluation_case(case_for("partial"),
        lambda: BoundCopilotSession(session, BoundToolCollection(tools)))
    assert result.passed
    payload = json.loads(session.history[0].tool_invocations[0].result_json)
    assert "checkpoint_binding_not_verified" in payload["targets"]["quality"]["limitations"]
    assert {"independence_not_verified", "repeated_seed", "snapshot_currentness_not_verified"} <= set(payload["limitations"])
    for entry in payload["targets"]["quality"]["included_executions"]:
        assert entry["source"]["provenance"]["checkpoint_binding"] == "not_verified"


def suite_definition():
    return EvaluationSuite("m18-replicates", "1", [
        EvaluationScenario(case_for(name), fixture_version="generic-executions-v1")
        for name in ("complete", "partial")], evaluation_profile="m10-structured-facts-v1",
        driver_version="m18-snapshot-agent-v1")


def test_m13_per_case_gate_catches_coverage_regression():
    tool_class()
    suite = suite_definition()
    manifest = BaselineManifest.from_suite(suite)
    before = manifest.to_json()
    baseline = tuple(EvaluationResult(s.key[0], "1", True, 1.0, ()) for s in suite.scenarios)
    def batch(corrupt):
        pending, sessions = iter(suite.scenarios), []
        def factory():
            scenario = next(pending)
            name = scenario.key[0]
            facts = deepcopy(scenario.case.expected_facts)
            if corrupt and name == "partial": facts["coverage"] = 1.0
            session = CopilotSession(Client(final=json.dumps({"facts": facts})), model="offline")
            sessions.append(session)
            return BoundCopilotSession(session, BoundToolCollection(make_tools(tool_class(), data_for(name))))
        report = BatchEvaluationRunner(factory).run(suite)
        assert len({id(session) for session in sessions}) == 2
        assert all(session.turn_count == 1 for session in sessions)
        assert report.runner_errors == report.execution_failures == 0
        return report
    good = batch(False)
    assert good.passed == 2 and compare_batch_to_baseline(good, manifest, baseline).passed
    bad = batch(True)
    assert (bad.passed, bad.failed) == (1, 1)
    gate = compare_batch_to_baseline(bad, manifest, baseline)
    assert not gate.passed and "regression:partial" in gate.reason_codes
    assert manifest.to_json() == before


def test_evaluation_definitions_roundtrip_without_missing_implementation():
    suite = suite_definition()
    assert len(suite.scenarios) == 2
    assert EvaluationSuite.from_json(suite.to_json()).to_json() == suite.to_json()
    assert case_for("partial").expected_facts["expected"] == 3
    assert case_for("partial").expected_facts["missing"] == 1
    assert data_for("partial")[0][0]["provenance"].realized_seed == data_for("partial")[0][1]["provenance"].realized_seed

"""M20 structured facts grounded in real bound reports, not a prose classifier."""

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
from m20_contract_helpers import TOOL, LIMITATIONS, Client, assert_limits, data, tool_class


def case_for(name):
    sha, count, status, locator = {
        "agreement": ("match", "match", "observed", "same"),
        "size-mismatch": ("match", "mismatch", "observed", "same"),
        "missing": ("not_evaluated", "not_evaluated", "missing", "not_evaluated"),
        "unobserved": ("not_evaluated", "not_evaluated", "unobserved", "not_evaluated"),
    }[name]
    facts = {"sha": sha, "size": count, "status": status, "locator": locator,
             "basis": "host_reported", "currentness": "not_checked", "checkpoint": "not_verified",
             "recomputation": "not_performed_by_copilot", "membership": "host_declared",
             "limitations": list(LIMITATIONS)}
    row = ["artifacts", "metrics"]
    paths = {"sha": row + ["integrity", "sha256_comparison"],
             "size": row + ["integrity", "size_comparison"],
             "status": row + ["reported_status"], "locator": row + ["locator_comparison"],
             "basis": ["evidence_basis"], "currentness": row + ["currentness_status"],
             "checkpoint": row + ["checkpoint_binding_status"],
             "recomputation": row + ["recomputation_status"],
             "membership": row + ["experiment_membership_status"], "limitations": ["limitations"]}
    return EvaluationCase(name, "1", "Return JSON descriptor facts and preserve the host-reported limitations.",
                          facts, {"required_tools": [TOOL], "fact_paths": paths})


def data_for(name):
    declared, reported = data()
    if name == "size-mismatch": reported[0]["byte_size"] = 99
    elif name == "missing": reported[0].update(reported_status="missing", byte_size=None, sha256=None, mtime_ns=None)
    elif name == "unobserved": reported.clear()
    return declared, reported


def run_case(name, facts):
    cls = tool_class()
    session = CopilotSession(Client(final=json.dumps({"facts": facts})), model="offline")
    result = run_evaluation_case(case_for(name), lambda: BoundCopilotSession(
        session, BoundToolCollection(cls(*data_for(name)))))
    assert session.turn_count == 1 and result.run_id
    assert_limits(json.loads(session.history[0].tool_invocations[0].result_json))
    return result


@pytest.mark.parametrize("name", ["agreement", "size-mismatch", "missing", "unobserved"])
def test_allowed_structured_claims_are_grounded_in_real_report(name):
    result = run_case(name, case_for(name).expected_facts)
    assert result.passed and result.score == 1. and result.failure_reasons == ()


@pytest.mark.parametrize("claim", [
    "hash-checkpoint", "hash-current", "locator-content", "observed-system", "host-trusted",
    "filename-checkpoint", "missing-unauthorized", "unobserved-missing", "memory-current",
    "descriptor-authentic", "mtime-current", "snapshot-version", "hash-semantic",
    "hash-experiment", "exists-current", "omit-checkpoint", "omit-filesystem", "omit-currentness",
])
def test_unsupported_structured_upgrade_fails_answer_and_grounding(claim):
    name = "missing" if claim == "missing-unauthorized" else (
        "unobserved" if claim == "unobserved-missing" else "agreement")
    wrong = deepcopy(case_for(name).expected_facts)
    if claim == "missing-unauthorized": wrong["status"] = "unauthorized"
    elif claim == "unobserved-missing": wrong["status"] = "missing"
    elif claim == "hash-checkpoint": wrong["checkpoint"] = "verified"
    elif claim in ("hash-current", "memory-current", "mtime-current", "exists-current"):
        wrong["currentness"] = "current"
    elif claim == "observed-system": wrong["basis"] = "system_observed"
    elif claim == "host-trusted": wrong["basis"] = "trusted_producer"
    elif claim.startswith("omit-"):
        wrong["limitations"].remove({"omit-checkpoint": "checkpoint_binding_not_verified",
                                     "omit-filesystem": "filesystem_not_observed_by_copilot",
                                     "omit-currentness": "artifact_currentness_not_checked"}[claim])
    else:
        wrong["conclusion"] = {
            "locator-content": "matching locator proves identical bytes",
            "filename-checkpoint": "matching filename proves checkpoint identity",
            "descriptor-authentic": "descriptor agreement proves authenticity",
            "snapshot-version": "matching snapshot label proves the same file version",
            "hash-semantic": "matching digest proves semantic correctness",
            "hash-experiment": "matching digest verifies the experiment",
        }[claim]
    result = run_case(name, wrong)
    assert not result.passed and result.score == pytest.approx(1 / 3)
    assert result.failure_reasons == ("answer_facts", "evidence_grounding")


def suite_definition():
    return EvaluationSuite("m20-artifact-descriptors", "1", [
        EvaluationScenario(case_for(name), fixture_version="generic-descriptors-v1")
        for name in ("agreement", "size-mismatch", "missing", "unobserved")],
        evaluation_profile="m10-structured-facts-v1", driver_version="m20-host-report-v1")


def test_m13_per_case_gate_detects_currentness_upgrade():
    cls, suite = tool_class(), suite_definition()
    manifest = BaselineManifest.from_suite(suite)
    before = manifest.to_json()
    baseline = tuple(EvaluationResult(s.key[0], "1", True, 1., ()) for s in suite.scenarios)
    def batch(corrupt):
        pending, sessions = iter(suite.scenarios), []
        def factory():
            scenario = next(pending)
            name = scenario.key[0]
            facts = deepcopy(scenario.case.expected_facts)
            if corrupt and name == "agreement": facts["currentness"] = "current"
            session = CopilotSession(Client(final=json.dumps({"facts": facts})), model="offline")
            sessions.append(session)
            return BoundCopilotSession(session, BoundToolCollection(cls(*data_for(name))))
        report = BatchEvaluationRunner(factory).run(suite)
        assert len({id(s) for s in sessions}) == 4 and all(s.turn_count == 1 for s in sessions)
        assert report.runner_errors == report.execution_failures == 0
        return report
    good, bad = batch(False), batch(True)
    assert good.passed == 4 and compare_batch_to_baseline(good, manifest, baseline).passed
    assert (bad.passed, bad.failed) == (3, 1)
    gate = compare_batch_to_baseline(bad, manifest, baseline)
    assert not gate.passed and "regression:agreement" in gate.reason_codes
    assert manifest.to_json() == before


def test_evaluation_definitions_and_literal_oracles_independent_of_m20():
    suite = suite_definition()
    assert len(suite.scenarios) == 4
    assert EvaluationSuite.from_json(suite.to_json()).to_json() == suite.to_json()
    assert case_for("size-mismatch").expected_facts["sha"] == "match"
    assert case_for("size-mismatch").expected_facts["size"] == "mismatch"
    assert case_for("unobserved").expected_facts["status"] == "unobserved"
    assert data_for("unobserved")[1] == []
    assert data_for("size-mismatch")[0]["artifacts"][0]["byte_size"] == 12
    assert data_for("size-mismatch")[1][0]["byte_size"] == 99
    assert case_for("agreement").expected_facts["limitations"] == LIMITATIONS

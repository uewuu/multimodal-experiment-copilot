"""M19 structured-fact grounding and per-case gates over real bound comparison.

Provider doubles supply messages, never fake tool results. These contracts make
no claim to detect arbitrary unsupported prose outside the scored facts.
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
from m19_contract_helpers import TOOL, ROOT_LIMITATIONS, Client, invoke, pair, tool_class


def case_for(name):
    expected, included, mean, delta = {"complete": (3, 3, 4., 1.),
                                     "partial": (3, 2, 3., 0.),
                                     "unequal-roster": (5, 5, 6., 3.)}[name]
    facts = {"a_mean": 3., "b_mean": mean, "raw_delta": delta,
             "b_expected": expected, "b_included": included, "b_missing": expected - included,
             "b_coverage": included / expected,
             "eligibility": "eligible" if name == "complete" else "limited",
             "membership_source": "host_declared", "policy": "unpaired-descriptive-v1",
             "a_limits": sorted(ROOT_LIMITATIONS | {"checkpoint_binding_not_verified"})}
    prefix = ["targets", "quality"]
    paths = {"a_mean": prefix + ["a", "statistics", "mean"],
             "b_mean": prefix + ["b", "statistics", "mean"],
             "raw_delta": prefix + ["comparison", "raw_delta"],
             "eligibility": prefix + ["comparison", "eligibility"],
             "membership_source": ["groups", "a", "membership_source"],
             "policy": ["comparison_policy"], "a_limits": prefix + ["a", "limitations"]}
    for fact, field in (("b_expected", "expected_count"), ("b_included", "included_count"),
                        ("b_missing", "missing_count"), ("b_coverage", "coverage_ratio")):
        paths[fact] = prefix + ["b", "coverage", field]
    return EvaluationCase(name, "1", "Return JSON facts including coverage and provenance limitations.",
                          facts, {"required_tools": [TOOL], "fact_paths": paths})


def data_for(name):
    values = {"complete": (3., 4., 5.), "partial": (2., 4., 6.),
              "unequal-roster": (2., 4., 6., 8., 10.)}[name]
    data = pair(b_values=values)
    if name == "partial": data[1]["executions"].pop()
    return data


def run_case(name, facts):
    cls = tool_class()
    client = Client(final=json.dumps({"facts": facts}))
    session = CopilotSession(client, model="offline")
    result = run_evaluation_case(case_for(name), lambda: BoundCopilotSession(
        session, BoundToolCollection(invoke(cls, data_for(name)))))
    assert session.turn_count == 1 and len(client.requests) == 2 and result.run_id
    payload = json.loads(session.history[0].tool_invocations[0].result_json)
    assert payload["comparison_policy"] == "unpaired-descriptive-v1"
    return result


@pytest.mark.parametrize("name", ["complete", "partial", "unequal-roster"])
def test_correct_structured_facts_are_grounded_in_real_comparison(name):
    result = run_case(name, case_for(name).expected_facts)
    assert result.passed and result.score == 1. and result.failure_reasons == ()


@pytest.mark.parametrize("corruption", [
    "significant-improvement", "verified-stability", "same-seed-paired", "verified-independent",
    "full-coverage-unbiased", "verified-checkpoint", "winner", "causal-effect",
    "hide-partial-coverage", "hide-roster-mismatch", "omit-checkpoint-limit", "omit-currentness-limit",
])
def test_unsupported_structured_claims_fail_answer_and_grounding(corruption):
    name = "unequal-roster" if corruption == "hide-roster-mismatch" else (
        "partial" if corruption == "hide-partial-coverage" else "complete")
    wrong = deepcopy(case_for(name).expected_facts)
    if corruption == "hide-partial-coverage": wrong["b_coverage"] = 1.
    elif corruption == "hide-roster-mismatch": wrong["b_expected"] = 3
    elif corruption == "omit-checkpoint-limit": wrong["a_limits"].remove("checkpoint_binding_not_verified")
    elif corruption == "omit-currentness-limit": wrong["a_limits"].remove("snapshot_currentness_not_verified")
    else:
        wrong["conclusion"] = {
            "significant-improvement": "mean difference proves significant improvement",
            "verified-stability": "lower sample std proves method stability",
            "same-seed-paired": "matching seeds establish paired executions",
            "verified-independent": "multiple executions are verified independent samples",
            "full-coverage-unbiased": "full roster coverage proves unbiased study",
            "verified-checkpoint": "CSV results have verified checkpoint provenance",
            "winner": "directionally favorable group is the winner",
            "causal-effect": "group mean difference is a causal effect",
        }[corruption]
    result = run_case(name, wrong)
    assert not result.passed and result.score == pytest.approx(1 / 3)
    assert result.failure_reasons == ("answer_facts", "evidence_grounding")


def suite_definition():
    return EvaluationSuite("m19-group-comparison", "1", [
        EvaluationScenario(case_for(name), fixture_version="generic-groups-v1")
        for name in ("complete", "partial", "unequal-roster")],
        evaluation_profile="m10-structured-facts-v1", driver_version="m19-snapshot-agent-v1")


def test_m13_per_case_gate_detects_coverage_regression():
    cls = tool_class()
    suite = suite_definition()
    manifest = BaselineManifest.from_suite(suite)
    before = manifest.to_json()
    baseline = tuple(EvaluationResult(s.key[0], "1", True, 1., ()) for s in suite.scenarios)
    def batch(corrupt):
        pending, sessions = iter(suite.scenarios), []
        def factory():
            scenario = next(pending)
            name = scenario.key[0]
            facts = deepcopy(scenario.case.expected_facts)
            if corrupt and name == "partial": facts["b_coverage"] = 1.
            session = CopilotSession(Client(final=json.dumps({"facts": facts})), model="offline")
            sessions.append(session)
            return BoundCopilotSession(session, BoundToolCollection(invoke(cls, data_for(name))))
        report = BatchEvaluationRunner(factory).run(suite)
        assert len({id(s) for s in sessions}) == 3 and all(s.turn_count == 1 for s in sessions)
        assert report.runner_errors == report.execution_failures == 0
        return report
    good = batch(False)
    assert good.passed == 3 and compare_batch_to_baseline(good, manifest, baseline).passed
    bad = batch(True)
    assert (bad.passed, bad.failed) == (2, 1)
    gate = compare_batch_to_baseline(bad, manifest, baseline)
    assert not gate.passed and "regression:partial" in gate.reason_codes
    assert manifest.to_json() == before


def test_evaluation_definitions_roundtrip_and_literals_without_m19():
    suite = suite_definition()
    assert len(suite.scenarios) == 3
    assert EvaluationSuite.from_json(suite.to_json()).to_json() == suite.to_json()
    assert case_for("partial").expected_facts["raw_delta"] == 0.
    assert case_for("unequal-roster").expected_facts["b_expected"] == 5
    assert "snapshot_currentness_not_verified" in case_for("complete").expected_facts["a_limits"]


def test_evaluation_fixtures_really_have_lower_std_and_matching_seeds():
    from m19_contract_helpers import aggregate
    data = data_for("complete")
    a, b = [aggregate(side) for side in data[:2]]
    assert a["targets"]["quality-A"]["statistics"]["sample_std"] == 2.
    assert b["targets"]["quality-B"]["statistics"]["sample_std"] == 1.
    assert [x["provenance"].realized_seed for x in data[0]["executions"]] == [
        x["provenance"].realized_seed for x in data[1]["executions"]]

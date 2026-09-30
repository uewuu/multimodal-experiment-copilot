"""M17 RED: M10 grounding and M13 per-case gates over real bound comparisons.

Expected facts are literals, not extracted from the production result. Only the
Provider answer is doubled. Negative claims live in scored JSON facts: this
does not claim to verify arbitrary natural-language significance or causality.
Authorization and Provider exceptions are tested in the integration file,
not mislabeled as successful evaluation cases.
"""

from copy import deepcopy
from dataclasses import replace
import json

import pytest

from copilot.evaluation import EvaluationCase, EvaluationResult
from copilot.evaluation_runner import run_evaluation_case
from copilot.evaluation_batch import BatchEvaluationRunner, compare_batch_to_baseline
from copilot.evaluation_suite import BaselineManifest, EvaluationScenario, EvaluationSuite
from copilot.session import CopilotSession
from copilot.tool_binding import BoundCopilotSession
from llm_adapters.tool_binding import BoundToolCollection
from m17_contract_helpers import TOOL, Client, arguments, options, spec, target, tool_class, write_experiment


SCENARIOS = ("comparable", "unknown", "split-mismatch", "definition-mismatch",
             "direction-mismatch", "missing", "invalid")


def case_for(name):
    status, reasons, delta, interpretation, availability = "comparable", [], 0.25, "b_favorable", "available"
    if name == "unknown": status, reasons = "unknown", ["missing_evaluation_protocol"]
    elif name.endswith("mismatch"):
        status = "incompatible"
        reasons = [{"split-mismatch": "split_id_mismatch", "definition-mismatch": "metric_definition_mismatch",
                    "direction-mismatch": "direction_mismatch"}[name]]
        delta, interpretation = None, None
    elif name in ("missing", "invalid"):
        availability, delta, interpretation = name, None, None
    comparability = None if name in ("missing", "invalid") else {
        "status": status, "reason_codes": reasons, "policy_version": "metric-result-v1"}
    facts = {"comparability": comparability, "availability": availability,
             "delta": delta, "interpretation": interpretation, "checkpoint_binding": "not_verified"}
    paths = {"comparability": ["targets", "quality", "comparability"],
             "availability": ["targets", "quality", "availability_b"],
             "delta": ["targets", "quality", "calculation", "raw_delta"],
             "interpretation": ["targets", "quality", "calculation", "direction_interpretation"],
             "checkpoint_binding": ["targets", "quality", "source_b", "provenance", "checkpoint_binding"]}
    return EvaluationCase(name, "1", "Return JSON facts with comparison limits.", facts,
                          {"required_tools": [TOOL], "fact_paths": paths})


def host(tmp_path, name):
    root = tmp_path / name
    root.mkdir()
    a = write_experiment(root, "a")
    b = write_experiment(root, "b", value=0.75, csv=name != "missing")
    selected, ob = target(), options()
    if name == "unknown": selected["b"]["evaluation_protocol"] = None
    elif name == "split-mismatch":
        selected["b"]["provenance"] = replace(selected["b"]["provenance"], split_id="other")
    elif name == "definition-mismatch": selected["b"]["metric_definition"] = "other-definition"
    elif name == "direction-mismatch":
        selected["b"]["metric_spec"] = spec(direction="minimize")
        ob["test_metric_specs"] = [spec(direction="minimize")]
    elif name == "invalid": (b / "heldout.csv").write_text("measurement\nNaN\n", encoding="utf-8")
    tools = tool_class()(str(a), str(b), options_a=options(), options_b=ob, targets=[selected])
    return root, a, b, BoundToolCollection(tools)


@pytest.mark.parametrize("name", SCENARIOS)
def test_real_comparison_facts_are_grounded_via_existing_m10(tmp_path, name):
    tool_class()
    case = case_for(name)
    root, a, b, collection = host(tmp_path, name)
    client = Client(arguments(a, b), final=json.dumps({"facts": case.expected_facts}))
    sessions = []
    def factory():
        session = CopilotSession(client, model="offline", experiment_context={"experiment_root": str(root)})
        sessions.append(session)
        return BoundCopilotSession(session, collection)
    result = run_evaluation_case(case, factory)
    assert result.passed and result.score == 1.0 and result.failure_reasons == ()
    assert result.run_id and len(sessions) == 1 and sessions[0].turn_count == 1
    assert len(client.requests) == 2


@pytest.mark.parametrize("corruption", ["verified-checkpoint", "false-comparable", "false-improvement"])
def test_negative_stronger_claims_fail_facts_and_grounding(tmp_path, corruption):
    tool_class()
    case = case_for("unknown")
    root, a, b, collection = host(tmp_path, "unknown")
    wrong = deepcopy(case.expected_facts)
    if corruption == "verified-checkpoint": wrong["checkpoint_binding"] = "verified"
    elif corruption == "false-comparable": wrong["comparability"]["status"] = "comparable"
    else: wrong["interpretation"] = "statistically_significant_improvement"
    client = Client(arguments(a, b), final=json.dumps({"facts": wrong}))
    def factory():
        return BoundCopilotSession(CopilotSession(client, model="offline",
            experiment_context={"experiment_root": str(root)}), collection)
    result = run_evaluation_case(case, factory)
    assert not result.passed and result.score == pytest.approx(1 / 3)
    assert result.failure_reasons == ("answer_facts", "evidence_grounding")
    assert len(client.requests) == 2


def test_m13_suite_and_per_case_gate_detect_false_verified_claim(tmp_path):
    tool_class()
    cases = [case_for(name) for name in ("unknown", "comparable")]
    suite = EvaluationSuite("m17-comparison", "1", [
        EvaluationScenario(case, fixture_version="generic-pair-v1") for case in cases],
        evaluation_profile="m10-structured-facts-v1", driver_version="m17-agent-v1")
    manifest = BaselineManifest.from_suite(suite)
    baseline = tuple(EvaluationResult(s.key[0], "1", True, 1.0, ()) for s in suite.scenarios)
    before = manifest.to_json()
    def batch(corrupt, folder):
        folder.mkdir()
        pending, sessions = iter(suite.scenarios), []
        def factory():
            scenario = next(pending)
            root, a, b, collection = host(folder, scenario.key[0])
            facts = deepcopy(scenario.case.expected_facts)
            if corrupt and scenario.key[0] == "unknown": facts["checkpoint_binding"] = "verified"
            client = Client(arguments(a, b), final=json.dumps({"facts": facts}))
            session = CopilotSession(client, model="offline", experiment_context={"experiment_root": str(root)})
            sessions.append(session)
            return BoundCopilotSession(session, collection)
        report = BatchEvaluationRunner(factory).run(suite)
        assert len(sessions) == len({id(s) for s in sessions}) == 2
        assert all(s.turn_count == 1 for s in sessions)
        assert report.runner_errors == report.execution_failures == 0
        return report
    good = batch(False, tmp_path / "good")
    assert good.passed == 2 and compare_batch_to_baseline(good, manifest, baseline).passed
    bad = batch(True, tmp_path / "bad")
    assert (bad.passed, bad.failed) == (1, 1)
    gate = compare_batch_to_baseline(bad, manifest, baseline)
    assert not gate.passed and "regression:unknown" in gate.reason_codes
    assert manifest.to_json() == before


def test_evaluation_definitions_are_valid_without_m17_implementation():
    suite = EvaluationSuite("m17-definitions", "1", [
        EvaluationScenario(case_for(name), fixture_version="generic-pair-v1") for name in SCENARIOS],
        evaluation_profile="m10-structured-facts-v1", driver_version="m17-agent-v1")
    assert len(suite.scenarios) == 7
    assert EvaluationSuite.from_json(suite.to_json()).to_json() == suite.to_json()
    assert case_for("missing").expected_facts["comparability"] is None
    assert case_for("unknown").expected_facts["checkpoint_binding"] == "not_verified"

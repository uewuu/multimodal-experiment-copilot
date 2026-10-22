"""M14 Slice 3A RED: real Memory/binding through frozen M10/M13 machinery.

The companion integration file owns shared host/model fixtures, not production
binding substitutes. Expected facts below are independent literal projections;
they are never derived from an Agent answer or used to replace tool dispatch.
M10 scores JSON facts only. No claim is made about arbitrary free-text truth.
Array-valued evidence is scored as a whole: frozen M10 paths traverse object
keys, not array indices. Baseline results remain separate from M13 definitions.
"""

from copy import deepcopy
from dataclasses import replace
from hashlib import sha256
import json

import pytest

from copilot.evaluation import EvaluationCase, EvaluationResult
from copilot.evaluation_batch import (
    BaselineValidationError, BatchEvaluationRunner, compare_batch_to_baseline,
)
from copilot.evaluation_runner import run_evaluation_case
from copilot.evaluation_suite import (
    BaselineManifest, EvaluationScenario, EvaluationSuite, check_baseline_compatibility,
)
from test_copilot_memory_integration import (
    COMPARE, DEFAULTS, GET, MEMORY, _Client, _Host, _SessionProbe, _api,
    _assert_unbound, _names, _outcomes, host,
)


def _identity(execution):
    return {"namespace": "lab", "experiment_id": "exp", "execution_id": execution}


def _metric(*, protocol="eval-v1", unknown=False):
    return {
        "metric_name": "r2", "value": 0.8, "best_epoch": 3,
        "metric_provenance": {
            "task": None if unknown else "regression",
            "target": None if unknown else "target",
            "dataset_version": None if unknown else "dataset-v1",
            "split_id": None if unknown else "validation",
            "metric_definition": "r2-v1", "direction": "maximize", "aggregation": "macro",
            "evaluation_protocol": protocol, "selection_protocol": "best-validation",
        },
        "artifacts": {"checkpoint_ref": "withheld", "result_artifact_ref": "not_declared",
                      "history_ref": "withheld"},
    }


def _entry(execution):
    return {
        "execution": _identity(execution), "lifecycle": {"status": "active"},
        "provenance": {"dataset_version": "dataset-v1", "split_id": "validation",
                       "task": "regression", "target": "target", "realized_seed": 7,
                       "source_revision": "revision-v1"},
        "comparability": None,
    }


def _candidate(execution, state, reasons=(), *, unknown=False, protocol="eval-v1"):
    return {
        "execution": _identity(execution), "lifecycle": {"status": "active"},
        "metric": _metric(unknown=unknown, protocol=protocol),
        "comparability": {"status": state, "reason_codes": list(reasons),
                          "policy_version": "metric-result-v1"},
        "is_reference": execution == "A-reference",
    }


def _definition(key, tool, arguments, facts, paths):
    case = EvaluationCase(
        case_id=key, scenario_version="1", prompt=f"Return structured evidence for {key}.",
        expected_facts=facts, scoring_spec={"required_tools": [tool], "fact_paths": paths},
    )
    return case, tool, arguments


def _catalog():
    lookup = {"identity": _identity("A-reference"), "lifecycle": {"status": "active"},
              "provenance": _entry("A-reference")["provenance"], "metric": _metric()}
    definitions = [
        _definition("lookup", MEMORY[0], GET, lookup, {
            "identity": ["data", "execution"], "lifecycle": ["data", "lifecycle"],
            "provenance": ["data", "provenance"], "metric": ["data", "metrics", "r2"],
        }),
        _definition("filtered-list", MEMORY[1], {
            "experiment_id": "exp", "filters": {"realized_seed": 7}, "limit": 10,
        }, {"items": [_entry("A-reference"), _entry("B-comparable")], "has_more": False},
            {"items": ["data", "items"], "has_more": ["data", "has_more"]}),
        _definition("bounded-list", MEMORY[1], {
            "experiment_id": "exp", "filters": {"realized_seed": 7}, "limit": 1,
        }, {"items": [_entry("A-reference")], "has_more": True},
            {"items": ["data", "items"], "has_more": ["data", "has_more"]}),
    ]
    for key, seed, candidates in (
        ("comparable", 7, [_candidate("A-reference", "comparable"),
                           _candidate("B-comparable", "comparable")]),
        ("incompatible", 8, [_candidate("C-incompatible", "incompatible",
                                        ("evaluation_protocol_mismatch",), protocol="eval-v2")]),
        ("unknown", 9, [_candidate("D-unknown", "unknown",
                                   ("missing_task", "missing_target", "missing_dataset_version",
                                    "missing_split_id"), unknown=True)]),
    ):
        definitions.append(_definition(key, MEMORY[2], {**COMPARE, "filters": {"realized_seed": seed}},
                                       {"candidates": candidates, "has_more": False},
                                       {"candidates": ["data", "items"], "has_more": ["data", "has_more"]}))
    for key, tool, arguments, error in (
        ("missing-execution", MEMORY[0], {**GET, "execution_id": "absent"}, "execution_not_found"),
        ("missing-metric", MEMORY[2], {**COMPARE, "metric_name": "absent"}, "metric_not_found"),
        ("unauthorized", MEMORY[0], {**GET, "experiment_id": "outside-scope"}, "unauthorized_scope"),
    ):
        definitions.append(_definition(key, tool, arguments, {"error": error}, {"error": ["error", "code"]}))
    return {case.case_id: (case, tool, arguments) for case, tool, arguments in definitions}


def _suite(catalog):
    # Deliberately supply reverse order: M13, not a fixture runner, canonicalizes.
    scenarios = [EvaluationScenario(
        catalog[key][0], fixture_version="memory-fixture-v1",
        metadata={"milestone": "M14", "capability": "read-only-memory"},
    ) for key in sorted(catalog, reverse=True)]
    return EvaluationSuite("m14-memory-agent", "1", scenarios,
                           evaluation_profile="m10-structured-facts-v1", driver_version="m14-agent-v1")


@pytest.fixture
def catalog():
    definitions = _catalog()
    # Preflight the frozen primitives before any missing integration assertion:
    # malformed records, wrong projection paths or invalid M13 definitions are
    # fixture errors, never acceptable RED evidence.
    _suite(definitions)
    probe = _Host()
    with probe.execution():
        tools = probe.tools()
        for case, name, arguments in definitions.values():
            before = len(probe.provider_threads)
            evidence = tools.invoke_tool(name, arguments)
            for fact, path in case.scoring_spec["fact_paths"].items():
                value = evidence
                for key in path:
                    value = value[key]
                assert value == case.expected_facts[fact]
            if case.case_id == "unauthorized":
                assert len(probe.provider_threads) == before
    return definitions


def _run(definition, host, api, facades, *, facts=None, binding=True):
    case, name, arguments = definition
    client = _Client(_outcomes(name, arguments, case.expected_facts if facts is None else facts))
    sessions = []

    def factory():
        original = _SessionProbe(client, host, model="fixture-model")
        sessions.append(original)
        return (facades.BoundCopilotSession(original, api.BoundToolCollection(host.tools()))
                if binding else original)

    result = run_evaluation_case(case, factory)
    assert len(sessions) == 1
    session = sessions[0]
    if result.failure_reasons == ("execution_failure",):
        assert session.history == () and len(session.failures) == 1
        assert result.run_id == session.failures[0].run.run_id
    else:
        observed = session.delegated[-1][1]
        assert result.run_id == observed.run.run_id
        assert observed.turn.tool_invocations[0].tool_name == name
        assert _names(client.requests[0]["tools"]) == DEFAULTS + MEMORY
    assert result.run_id
    _assert_unbound()
    return result, client


def test_m10_lookup_filtered_list_and_bounded_evidence_detect_missing_facts(host, catalog):
    api, facades = _api(facades=True)
    for key in ("lookup", "filtered-list", "bounded-list"):
        definition = catalog[key]
        result, _ = _run(definition, host, api, facades)
        assert result.passed and result.score == 1.0 and result.failure_reasons == ()
        for omitted in definition[0].expected_facts:
            facts = deepcopy(definition[0].expected_facts)
            del facts[omitted]
            negative, _ = _run(definition, host, api, facades, facts=facts)
            assert not negative.passed and negative.failure_reasons == ("answer_facts",)
    wrong_order = deepcopy(catalog["filtered-list"][0].expected_facts)
    wrong_order["items"].reverse()
    negative, _ = _run(catalog["filtered-list"], host, api, facades, facts=wrong_order)
    assert negative.failure_reasons == ("answer_facts", "evidence_grounding")


def test_m10_preserves_three_states_and_rejects_false_comparable_claims(host, catalog):
    api, facades = _api(facades=True)
    for state in ("comparable", "incompatible", "unknown"):
        result, _ = _run(catalog[state], host, api, facades)
        assert result.passed and result.score == 1.0
    for state in ("incompatible", "unknown"):
        wrong = deepcopy(catalog[state][0].expected_facts)
        wrong["candidates"][0]["comparability"]["status"] = "comparable"
        result, _ = _run(catalog[state], host, api, facades, facts=wrong)
        assert not result.passed
        assert result.failure_reasons == ("answer_facts", "evidence_grounding")
        assert result.score == pytest.approx(1 / 3)


def test_m10_missing_and_unauthorized_errors_preserve_safe_facts(host, catalog):
    api, facades = _api(facades=True)
    for key in ("missing-execution", "missing-metric", "unauthorized"):
        before = len(host.provider_threads)
        result, client = _run(catalog[key], host, api, facades)
        assert result.passed and result.score == 1.0
        tool_messages = [message for message in client.requests[-1]["messages"] if message["role"] == "tool"]
        assert len(tool_messages) == 1
        assert json.loads(tool_messages[0]["content"]) == {
            "format_version": 1, "ok": False,
            "error": {"code": catalog[key][0].expected_facts["error"]},
        }
        assert len(host.provider_threads) - before == (0 if key == "unauthorized" else 1)
        guessed, _ = _run(catalog[key], host, api, facades, facts={"error": "comparable"})
        assert guessed.failure_reasons == ("answer_facts", "evidence_grounding")


def test_m10_no_binding_and_write_attempts_cannot_execute_memory(host, catalog):
    api, facades = _api(facades=True)
    for name in MEMORY:
        definition = _definition("unbound", name, GET, {"forbidden": True}, {"forbidden": ["ok"]})
        result, client = _run(definition, host, api, facades, binding=False)
        assert not result.passed and result.score == 0.0
        assert result.failure_reasons == ("execution_failure",)
        assert _names(client.requests[0]["tools"]) == DEFAULTS
        assert len(client.requests) == 1
    for name in ("add", "invalidate", "delete", "overwrite", "import"):
        definition = _definition("write-attempt", name, {}, {"forbidden": True}, {"forbidden": ["ok"]})
        result, client = _run(definition, host, api, facades)
        assert result.failure_reasons == ("execution_failure",) and result.score == 0.0
        assert _names(client.requests[0]["tools"]) == DEFAULTS + MEMORY
        assert name not in _names(client.requests[0]["tools"])
        assert len(client.requests) == 1
    assert host.provider_threads == []


def test_m13_versioned_batch_fingerprints_and_baseline_gate_protect_memory_facts(host, catalog):
    suite = _suite(catalog)
    manifest = BaselineManifest.from_suite(suite, metadata={"selection": "explicit-host-baseline"})
    api, facades = _api(facades=True)
    expected_order = ("bounded-list", "comparable", "filtered-list", "incompatible",
                      "lookup", "missing-execution", "missing-metric", "unauthorized", "unknown")
    assert tuple(scenario.key[0] for scenario in suite.scenarios) == expected_order
    assert (suite.suite_id, suite.suite_version) == ("m14-memory-agent", "1")
    for scenario in suite.scenarios:
        case = scenario.case
        content = {"format_version": "v1", "prompt": case.prompt, "context_prompts": [],
                   "expected_facts": case.expected_facts, "scoring_spec": case.scoring_spec,
                   "fixture_version": "memory-fixture-v1",
                   "metadata": {"milestone": "M14", "capability": "read-only-memory"}}
        expected = sha256(json.dumps(content, ensure_ascii=False, allow_nan=False,
                                    sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()
        assert scenario.fingerprint == expected
    serialized_suite, serialized_manifest = suite.to_json(), manifest.to_json()
    assert EvaluationSuite.from_json(serialized_suite).to_json() == serialized_suite
    assert BaselineManifest.from_json(serialized_manifest).to_json() == serialized_manifest
    assert check_baseline_compatibility(suite, manifest).compatible
    # Historical quality values are explicitly separate, never inserted into the manifest.
    baseline_results = tuple(EvaluationResult(key, "1", True, 1.0, ()) for key in expected_order)
    original_results = deepcopy(baseline_results)

    def batch(*, corrupt=False):
        sessions = []
        pending = iter(suite.scenarios)

        def factory():
            scenario = next(pending)
            case, name, arguments = catalog[scenario.key[0]]
            facts = deepcopy(case.expected_facts)
            if corrupt and case.case_id in ("incompatible", "unknown"):
                facts["candidates"][0]["comparability"]["status"] = "comparable"
            client = _Client(_outcomes(name, arguments, facts))
            original = _SessionProbe(client, host, model="fixture-model")
            sessions.append(original)
            return facades.BoundCopilotSession(original, api.BoundToolCollection(host.tools()))

        report = BatchEvaluationRunner(factory).run(suite)
        assert len(sessions) == len({id(session) for session in sessions}) == 9
        assert tuple(item.case_id for item in report.outcomes) == expected_order
        assert tuple(item.content_fingerprint for item in report.outcomes) == tuple(
            scenario.fingerprint for scenario in suite.scenarios)
        for position, (outcome, session) in enumerate(zip(report.outcomes, sessions)):
            assert outcome.position == position and outcome.error_code is None
            assert len(session.history) == 1
            assert outcome.result.run_id == session.delegated[-1][1].run.run_id
        _assert_unbound()
        return report

    good = batch()
    assert (good.total, good.completed, good.passed, good.failed, good.runner_errors,
            good.execution_failures, good.aggregate_score) == (9, 9, 9, 0, 0, 0, 1.0)
    gate = compare_batch_to_baseline(good, manifest, baseline_results)
    assert gate.passed and gate.reason_codes == () and len(gate.comparisons) == 9
    negative = batch(corrupt=True)
    assert (negative.passed, negative.failed, negative.runner_errors) == (7, 2, 0)
    gate = compare_batch_to_baseline(negative, manifest, baseline_results)
    assert not gate.passed
    assert gate.reason_codes == ("failed_evaluation:incompatible", "failed_evaluation:unknown",
                                 "regression:incompatible", "regression:unknown")
    assert {item.case_id: item.comparison.change for item in gate.comparisons
            if item.comparison.change != "unchanged"} == {"incompatible": "regression", "unknown": "regression"}
    changed = dict(catalog)
    case, name, arguments = changed["lookup"]
    changed["lookup"] = (replace(case, prompt="Changed evidence definition"), name, arguments)
    drifted_manifest = BaselineManifest.from_suite(_suite(changed))
    compatibility = check_baseline_compatibility(suite, drifted_manifest)
    assert compatibility.reason_codes == ("content_fingerprint_mismatch:lookup",)
    with pytest.raises(BaselineValidationError) as caught:
        compare_batch_to_baseline(good, drifted_manifest, baseline_results)
    assert caught.value.reason_codes == compatibility.reason_codes
    assert suite.to_json() == serialized_suite and manifest.to_json() == serialized_manifest
    assert baseline_results == original_results

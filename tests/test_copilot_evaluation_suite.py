"""M13 Slice 1 RED contracts: definitions and compatibility, no execution.

Future copilot.evaluation_suite API:
  EvaluationScenario(case, *, context_prompts=(), fixture_version, metadata=None)
    wraps the actual M10 EvaluationCase; key is (case_id, scenario_version).
    Scalar fields are read-only. metadata is declared scenario content, unlike
    BaselineManifest.metadata, which is descriptive comparison metadata only.
  ScenarioRegistry().register(scenario); get(case_id, scenario_version).
    Duplicate keys raise ValueError even for identical content; missing keys
    raise KeyError. Registration and reads isolate nested mutable containers.
  EvaluationSuite(suite_id, suite_version, scenarios, *, evaluation_profile,
                  driver_version)
    snapshots 1..100 scenarios, one version per case_id, sorted by exact
    (case_id, scenario_version). scenarios returns independent snapshots.
  EvaluationSuite.to_json()/from_json(text).
  BaselineManifest.from_suite(suite, *, metadata=None); to_json()/from_json(text).
  check_baseline_compatibility(suite, baseline) -> compatible, reason_codes.

The v1 content fingerprint is lowercase SHA-256 over UTF-8 JSON of exactly:
format_version="v1", prompt, context_prompts, expected_facts, scoring_spec,
fixture_version, metadata. JSON uses sort_keys=True, separators=(",", ":"),
ensure_ascii=False, allow_nan=False. Only string object keys and finite JSON
values are accepted. M10 scoring tuples and context sequences encode as arrays;
object order is irrelevant, array order and prompt whitespace are significant.
Explicit case/version identity is checked separately from content. No paths,
clock, runtime IDs or Python object identities are consulted to derive keys.

Suite JSON contains format_version, suite_id, suite_version, evaluation_profile,
driver_version, scenarios. Each scenario contains case (the five M10 fields),
context_prompts, fixture_version, metadata, content_fingerprint. Imports verify
fingerprints; they never silently repair stale content or drop duplicate cases.
Baseline JSON has the same suite fields plus descriptive metadata; scenarios
contains only case_id, scenario_version, content_fingerprint, fixture_version.
Both formats require v1 and all declared fields; invalid input raises ValueError.
Host owns files. These APIs consume/produce JSON text entirely in memory.

Compatibility collects ALL differences, in category order:
suite_id_mismatch, suite_version_mismatch, evaluation_profile_mismatch,
driver_version_mismatch, scenario_added:<id>, scenario_removed:<id>,
scenario_version_mismatch:<id>, content_fingerprint_mismatch:<id>,
fixture_version_mismatch:<id>. Within a category, IDs sort case-sensitively.
Added/removed are relative to the baseline. Shared case IDs are checked even
when suite fields differ; fixture drift also changes the content fingerprint.
No intersection-only comparison, scoring, run execution or release gate here.
"""

from collections.abc import Mapping
from copy import deepcopy
from dataclasses import replace
import importlib
import importlib.util
import json

import pytest

from copilot.evaluation import EvaluationCase


_MODULE = "copilot.evaluation_suite"
_PROFILE = "m10-structured-three-checks-v1"
_DRIVER = "fake-provider-driver-v1"
_CONTEXT = ("Use declared evidence.", "Keep the supplied task context.")
_FINGERPRINT = "9e3af8542ff90f29800fbdc34f6e9cd26c2a1d78b5af6f489e651d86b2c21fda"


def _api(*names):
    # Missing capability fails during test execution, never during collection.
    # An import error inside a future module is deliberately not swallowed.
    assert importlib.util.find_spec(_MODULE) is not None, (
        f"missing M13 evaluation suite capability: {_MODULE}"
    )
    module = importlib.import_module(_MODULE)
    missing = [name for name in names if not callable(getattr(module, name, None))]
    assert not missing, f"missing M13 evaluation suite API: {missing}"
    return module


def _case(case_id="case-a", scenario_version="1", **changes):
    values = {
        "case_id": case_id,
        "scenario_version": scenario_version,
        "prompt": "Report best R2 and fixture details as JSON facts.",
        "expected_facts": {
            "best_r2": 0.82, "details": {"epochs": [3, 7], "label": "validation"},
        },
        "scoring_spec": {
            "required_tools": ("analyze_experiment",),
            "fact_paths": {
                "best_r2": ("validation_metrics", "r2", "best_value"),
                "details": ("fixture_details",),
            },
        },
    }
    return EvaluationCase(**{**values, **changes})


def _scenario(api, case=None, **changes):
    values = {
        "case": _case() if case is None else case,
        "context_prompts": _CONTEXT,
        "fixture_version": "fixture-v1",
        "metadata": {"tags": ["structured"], "owner": {"team": "quality"}},
    }
    return api.EvaluationScenario(**{**values, **changes})


def _suite(api, scenarios=None, **changes):
    values = {
        "suite_id": "copilot-core", "suite_version": "1",
        "scenarios": [_scenario(api)] if scenarios is None else scenarios,
        "evaluation_profile": _PROFILE, "driver_version": _DRIVER,
    }
    return api.EvaluationSuite(**{**values, **changes})


def _plain(value):
    # Normalize only containers for assertions, never scoring or fingerprints.
    if isinstance(value, Mapping):
        return {key: _plain(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_plain(item) for item in value]
    return value


def _case_values(case):
    assert isinstance(case, EvaluationCase), "M13 must reuse M10's case model"
    return {name: _plain(getattr(case, name)) for name in (
        "case_id", "scenario_version", "prompt", "expected_facts", "scoring_spec",
    )}


def _scenario_values(scenario):
    return {
        "case": _case_values(scenario.case),
        "context_prompts": list(scenario.context_prompts),
        "fixture_version": scenario.fixture_version,
        "metadata": _plain(scenario.metadata),
        "content_fingerprint": scenario.fingerprint,
    }


def _try_mutation(container, key, value):
    # Read-only returned containers are permitted; mutable ones must be detached.
    try:
        container[key] = value
    except (TypeError, AttributeError):
        pass


def _mutate_scenario(scenario):
    _try_mutation(scenario.case.expected_facts["details"]["epochs"], 0, 999)
    _try_mutation(scenario.case.scoring_spec["fact_paths"], "best_r2", ["wrong"])
    _try_mutation(scenario.metadata["owner"], "team", "mutated")
    _try_mutation(scenario.context_prompts, 0, "Mutated context.")


def _reversed_objects(value):
    if isinstance(value, dict):
        return {key: _reversed_objects(item) for key, item in reversed(value.items())}
    if isinstance(value, list):
        return [_reversed_objects(item) for item in value]
    return value


def _assert_compatibility(api, suite, baseline, reasons=()):
    result = api.check_baseline_compatibility(suite, baseline)
    assert result.compatible is (not reasons)
    assert result.reason_codes == reasons


def test_scenario_preserves_explicit_m10_case_and_ordered_context():
    api = _api("EvaluationScenario")
    case = _case()
    before = _case_values(case)
    scenario = _scenario(api, case)
    assert scenario.key == ("case-a", "1")
    assert _case_values(scenario.case) == before
    assert tuple(scenario.context_prompts) == _CONTEXT
    assert scenario.fixture_version == "fixture-v1"
    assert _plain(scenario.metadata) == {
        "tags": ["structured"], "owner": {"team": "quality"},
    }
    assert _case_values(case) == before
    for field, value in (("case", _case("other")), ("fixture_version", "changed")):
        with pytest.raises((AttributeError, TypeError)):
            setattr(scenario, field, value)
    for changes in (
        {"fixture_version": ""}, {"fixture_version": None},
        {"context_prompts": "not a prompt sequence"},
        {"context_prompts": ["valid", 42]}, {"context_prompts": [" "]},
        {"metadata": {"unsupported": object()}},
        {"metadata": {1: "ambiguous object key"}},
        {"case": replace(case, expected_facts={"best_r2": float("nan")})},
        {"case": replace(case, expected_facts={"best_r2": float("inf")})},
    ):
        with pytest.raises(ValueError):
            _scenario(api, **changes)
    with pytest.raises(TypeError):
        api.EvaluationScenario(fixture_version="fixture-v1")


def test_registry_rejects_duplicate_keys_and_preserves_exact_lookup():
    api = _api("EvaluationScenario", "ScenarioRegistry")
    registry = api.ScenarioRegistry()
    original = _scenario(api)
    registry.register(original)
    before = _scenario_values(registry.get("case-a", "1"))
    for duplicate in (
        _scenario(api), _scenario(api, _case(prompt="Changed task, same key.")),
    ):
        with pytest.raises(ValueError):
            registry.register(duplicate)
        assert _scenario_values(registry.get("case-a", "1")) == before

    other_version = _scenario(api, _case(scenario_version="2"))
    other_case = _scenario(api, _case("Case-a"))
    registry.register(other_version)
    registry.register(other_case)
    second = api.ScenarioRegistry()
    for scenario in (other_case, other_version, original):
        second.register(scenario)
    for key in (("case-a", "1"), ("case-a", "2"), ("Case-a", "1")):
        assert registry.get(*key).key == key
        assert _scenario_values(registry.get(*key)) == _scenario_values(second.get(*key))
    for key in (("missing", "1"), ("case-a", "3"), ("CASE-A", "1")):
        with pytest.raises(KeyError):
            registry.get(*key)
    assert _scenario_values(registry.get("case-a", "1")) == before


def test_registry_isolates_caller_and_retrieved_nested_snapshots():
    api = _api("EvaluationScenario", "ScenarioRegistry")
    case = _case()
    context = list(_CONTEXT)
    metadata = {"tags": ["structured"], "owner": {"team": "quality"}}
    scenario = _scenario(api, case, context_prompts=context, metadata=metadata)
    registry = api.ScenarioRegistry()
    registry.register(scenario)
    before = _scenario_values(registry.get("case-a", "1"))

    # Mutate the actual M10 case maps, not only the already-copied constructor
    # inputs: M10 itself does not promise immutability of these nested values.
    case.expected_facts["details"]["epochs"][0] = 888
    case.scoring_spec["fact_paths"]["best_r2"] = ["changed"]
    context.reverse()
    metadata["owner"]["team"] = "caller-changed"
    metadata["tags"].append("extra")
    _mutate_scenario(scenario)
    assert _scenario_values(registry.get("case-a", "1")) == before

    retrieved = registry.get("case-a", "1")
    _mutate_scenario(retrieved)
    assert _scenario_values(registry.get("case-a", "1")) == before
    assert registry.get("case-a", "1").fingerprint == _FINGERPRINT


def test_fingerprint_has_stable_canonical_encoding_and_preserves_sequence_meaning():
    api = _api("EvaluationScenario", "EvaluationSuite")
    original = _scenario(api)
    assert original.fingerprint == _FINGERPRINT  # Fixed, independent SHA-256 vector.
    normalized = _case_values(_case())
    reordered = _reversed_objects(normalized)
    equivalent = _scenario(api, EvaluationCase(**reordered))
    assert equivalent.fingerprint == original.fingerprint
    assert _scenario(api).fingerprint == original.fingerprint
    # Identity is explicit and separate from content; it is not a content hash.
    renamed = _scenario(api, _case("different-id", "2"))
    assert renamed.key == ("different-id", "2")
    assert renamed.fingerprint == original.fingerprint

    suite = _suite(api)
    reformatted = json.dumps(_reversed_objects(json.loads(suite.to_json())), indent=4)
    restored = api.EvaluationSuite.from_json(reformatted)
    assert restored.scenarios[0].fingerprint == original.fingerprint
    assert restored.to_json() == suite.to_json()

    variants = (
        _scenario(api, context_prompts=tuple(reversed(_CONTEXT))),
        _scenario(api, _case(prompt=_case().prompt + " ")),
        _scenario(api, _case(expected_facts={
            "best_r2": 0.82, "details": {"epochs": [7, 3], "label": "validation"},
        })),
    )
    assert all(item.fingerprint != original.fingerprint for item in variants)
    bool_fact = _scenario(api, _case(expected_facts={"flag": True}))
    numeric_fact = _scenario(api, _case(expected_facts={"flag": 1}))
    assert bool_fact.fingerprint != numeric_fact.fingerprint
    tools = {"required_tools": ["analyze_experiment", "compare_experiments"],
             "fact_paths": _case().scoring_spec["fact_paths"]}
    left = _scenario(api, _case(scoring_spec=tools))
    right = _scenario(api, _case(scoring_spec={
        **tools, "required_tools": list(reversed(tools["required_tools"])),
    }))
    assert left.fingerprint != right.fingerprint


def test_same_identity_content_drift_is_detected_without_auto_versioning():
    api = _api("EvaluationScenario", "EvaluationSuite", "BaselineManifest",
               "check_baseline_compatibility")
    baseline = api.BaselineManifest.from_suite(_suite(api))
    changed_spec = deepcopy(_case().scoring_spec)
    changed_spec["fact_paths"]["best_r2"] = ("validation_metrics", "r2", "mean")
    variants = (
        ("prompt", {"case": _case(prompt="A different task.")}),
        ("context", {"context_prompts": ("Changed context.", _CONTEXT[1])}),
        ("facts", {"case": _case(expected_facts={"best_r2": 0.99})}),
        ("scoring", {"case": _case(scoring_spec=changed_spec)}),
        ("fixture", {"fixture_version": "fixture-v2"}),
        ("metadata", {"metadata": {"tags": ["changed"]}}),
    )
    for field, changes in variants:
        scenario = _scenario(api, **changes)
        assert scenario.key == ("case-a", "1"), field
        assert scenario.fingerprint != _FINGERPRINT, field
        reasons = ("content_fingerprint_mismatch:case-a",)
        if field == "fixture":
            reasons += ("fixture_version_mismatch:case-a",)
        _assert_compatibility(api, _suite(api, [scenario]), baseline, reasons)
    assert baseline.scenarios[0].scenario_version == "1"
    assert baseline.scenarios[0].content_fingerprint == _FINGERPRINT


def test_suite_requires_explicit_bounded_unique_cases_in_canonical_order():
    api = _api("EvaluationScenario", "EvaluationSuite")
    scenarios = [_scenario(api, _case(name)) for name in ("z-case", "a-case", "A-case")]
    suite = _suite(api, scenarios)
    assert (suite.suite_id, suite.suite_version) == ("copilot-core", "1")
    assert (suite.evaluation_profile, suite.driver_version) == (_PROFILE, _DRIVER)
    assert tuple(item.key for item in suite.scenarios) == (
        ("A-case", "1"), ("a-case", "1"), ("z-case", "1"),
    )
    assert _suite(api, list(reversed(scenarios))).to_json() == suite.to_json()
    with pytest.raises((AttributeError, TypeError)):
        suite.suite_version = "changed"
    for field in ("suite_id", "suite_version", "evaluation_profile", "driver_version"):
        for invalid in ("", " ", None, 1):
            with pytest.raises(ValueError):
                _suite(api, **{field: invalid})
    with pytest.raises(TypeError):
        api.EvaluationSuite(scenarios=scenarios, evaluation_profile=_PROFILE,
                            driver_version=_DRIVER)
    for invalid in (
        [], [_scenario(api), _scenario(api)],
        [_scenario(api), _scenario(api, _case(scenario_version="2"))],
    ):
        with pytest.raises(ValueError):
            _suite(api, invalid)
    boundary = [_scenario(api, _case(f"case-{index:03d}")) for index in range(100)]
    assert len(_suite(api, boundary).scenarios) == 100
    with pytest.raises(ValueError):
        _suite(api, boundary + [_scenario(api, _case("case-100"))])


def test_suite_is_an_independent_snapshot_not_a_live_registry_view():
    api = _api("EvaluationScenario", "ScenarioRegistry", "EvaluationSuite",
               "BaselineManifest", "check_baseline_compatibility")
    registry = api.ScenarioRegistry()
    original = _scenario(api)
    registry.register(original)
    selected = registry.get("case-a", "1")
    supplied = [selected]
    suite = _suite(api, supplied)
    before = suite.to_json()
    baseline = api.BaselineManifest.from_suite(suite)
    _mutate_scenario(original)
    _mutate_scenario(selected)
    supplied.append(_scenario(api, _case("caller-added")))
    registry.register(_scenario(api, _case("registry-added")))
    registry.register(_scenario(api, _case(scenario_version="2")))
    exposed = suite.scenarios
    _mutate_scenario(exposed[0])
    _try_mutation(exposed, 0, _scenario(api, _case("replacement")))
    assert suite.to_json() == before
    assert tuple(item.key for item in suite.scenarios) == (("case-a", "1"),)
    assert suite.scenarios[0].fingerprint == _FINGERPRINT
    _assert_compatibility(api, suite, baseline)


def test_suite_json_round_trip_rejects_invalid_versions_duplicates_and_stale_content(monkeypatch):
    api = _api("EvaluationScenario", "EvaluationSuite")
    suite = _suite(api, [_scenario(api, _case("z-case")), _scenario(api)])

    def forbid_file_access(*args, **kwargs):
        pytest.fail("suite import/export must operate on in-memory JSON text")

    with monkeypatch.context() as guard:
        guard.setattr("builtins.open", forbid_file_access)
        encoded = suite.to_json()
        restored = api.EvaluationSuite.from_json(encoded)
    assert restored.to_json() == encoded == suite.to_json()
    assert [_scenario_values(item) for item in restored.scenarios] == [
        _scenario_values(item) for item in suite.scenarios
    ]
    payload = json.loads(encoded)
    assert payload == {
        "format_version": "v1", "suite_id": "copilot-core", "suite_version": "1",
        "evaluation_profile": _PROFILE, "driver_version": _DRIVER,
        "scenarios": [_scenario_values(item) for item in suite.scenarios],
    }
    reversed_manifest = deepcopy(payload)
    reversed_manifest["scenarios"].reverse()
    assert api.EvaluationSuite.from_json(json.dumps(reversed_manifest)).to_json() == encoded
    _mutate_scenario(restored.scenarios[0])
    assert suite.to_json() == encoded

    invalid_manifests = []
    for field in payload:
        missing = deepcopy(payload)
        del missing[field]
        invalid_manifests.append(missing)
    invalid_manifests += [
        {**payload, "format_version": "v99"}, {**payload, "format_version": 1},
        {**payload, "scenarios": []}, {**payload, "suite_id": ""},
        {**payload, "scenarios": "not an array"},
    ]
    duplicate = deepcopy(payload)
    duplicate["scenarios"].append(deepcopy(duplicate["scenarios"][0]))
    invalid_manifests.append(duplicate)
    two_versions = deepcopy(duplicate)
    two_versions["scenarios"][-1]["case"]["scenario_version"] = "2"
    invalid_manifests.append(two_versions)
    for field in payload["scenarios"][0]:
        missing = deepcopy(payload)
        del missing["scenarios"][0][field]
        invalid_manifests.append(missing)
    for field in payload["scenarios"][0]["case"]:
        missing = deepcopy(payload)
        del missing["scenarios"][0]["case"][field]
        invalid_manifests.append(missing)
    for field, value in (
        ("prompt", "Drift with a stale fingerprint."), ("scenario_version", None),
        ("expected_facts", []), ("scoring_spec", None), ("case_id", ""),
    ):
        invalid = deepcopy(payload)
        invalid["scenarios"][0]["case"][field] = value
        invalid_manifests.append(invalid)
    for field, value in (
        ("content_fingerprint", "0" * 64), ("context_prompts", "not an array"),
        ("fixture_version", None), ("metadata", []),
    ):
        invalid = deepcopy(payload)
        invalid["scenarios"][0][field] = value
        invalid_manifests.append(invalid)
    for index, invalid in enumerate(invalid_manifests):
        with pytest.raises(ValueError):
            api.EvaluationSuite.from_json(json.dumps(invalid))
        assert suite.to_json() == encoded, index
    for invalid_text in (
        "{", "null", "[]",
        '{"format_version":"v99",' + encoded.lstrip()[1:],
        encoded.replace('0.82', 'NaN', 1),
    ):
        with pytest.raises(ValueError):
            api.EvaluationSuite.from_json(invalid_text)


def test_baseline_manifest_preserves_exact_definition_and_excludes_run_metadata_from_identity():
    api = _api("EvaluationScenario", "EvaluationSuite", "BaselineManifest",
               "check_baseline_compatibility")
    suite = _suite(api, [_scenario(api, _case("z-case")), _scenario(api)])
    metadata = {"release": "candidate-a", "model": "fixture-model-a",
                "run_id": "borrowed-run-a", "recorded_at": "fixed-label-a",
                "notes": {"labels": ["baseline"]}}
    baseline = api.BaselineManifest.from_suite(suite, metadata=metadata)
    encoded = baseline.to_json()
    expected_entries = [{
        "case_id": name, "scenario_version": "1", "content_fingerprint": _FINGERPRINT,
        "fixture_version": "fixture-v1",
    } for name in ("case-a", "z-case")]
    assert json.loads(encoded) == {
        "format_version": "v1", "suite_id": "copilot-core", "suite_version": "1",
        "evaluation_profile": _PROFILE, "driver_version": _DRIVER,
        "scenarios": expected_entries, "metadata": metadata,
    }
    assert (baseline.suite_id, baseline.suite_version) == ("copilot-core", "1")
    assert (baseline.evaluation_profile, baseline.driver_version) == (_PROFILE, _DRIVER)
    assert [(entry.case_id, entry.scenario_version, entry.content_fingerprint,
             entry.fixture_version) for entry in baseline.scenarios] == [
        (name, "1", _FINGERPRINT, "fixture-v1") for name in ("case-a", "z-case")
    ]
    metadata["notes"]["labels"].append("caller-change")
    _try_mutation(baseline.metadata["notes"]["labels"], 0, "retrieved-change")
    assert baseline.to_json() == encoded
    restored = api.BaselineManifest.from_json(json.dumps(
        _reversed_objects(json.loads(encoded)), indent=2,
    ))
    assert restored.to_json() == encoded
    _assert_compatibility(api, suite, restored)
    alternate = api.BaselineManifest.from_suite(suite, metadata={
        "release": "candidate-b", "model": "fixture-model-b",
        "run_id": "borrowed-run-b", "recorded_at": "fixed-label-b",
    })
    _assert_compatibility(api, suite, alternate)
    assert json.loads(alternate.to_json())["scenarios"] == expected_entries

    payload = json.loads(encoded)
    invalid_manifests = [{**payload, "format_version": "v99"}]
    for field in payload:
        missing = deepcopy(payload)
        del missing[field]
        invalid_manifests.append(missing)
    for field in expected_entries[0]:
        missing = deepcopy(payload)
        del missing["scenarios"][0][field]
        invalid_manifests.append(missing)
    for field, value in (("content_fingerprint", "not-sha256"),
                         ("scenario_version", ""), ("fixture_version", None)):
        invalid = deepcopy(payload)
        invalid["scenarios"][0][field] = value
        invalid_manifests.append(invalid)
    for version in ("1", "2"):
        duplicate = deepcopy(payload)
        duplicate["scenarios"].append({**expected_entries[0], "scenario_version": version})
        invalid_manifests.append(duplicate)
    invalid_manifests.append({**payload, "scenarios": []})
    for invalid in invalid_manifests:
        with pytest.raises(ValueError):
            api.BaselineManifest.from_json(json.dumps(invalid))


def test_baseline_compatibility_reports_all_definition_differences_in_stable_order():
    api = _api("EvaluationScenario", "EvaluationSuite", "BaselineManifest",
               "check_baseline_compatibility")
    suite = _suite(api)
    baseline = api.BaselineManifest.from_suite(suite)
    before = (suite.to_json(), baseline.to_json())
    _assert_compatibility(api, suite, baseline)
    for field in ("suite_id", "suite_version", "evaluation_profile", "driver_version"):
        _assert_compatibility(api, _suite(api, **{field: "different"}), baseline,
                              (f"{field}_mismatch",))
    _assert_compatibility(api, _suite(api, [_scenario(api, _case(scenario_version="2"))]),
                          baseline, ("scenario_version_mismatch:case-a",))
    _assert_compatibility(api, _suite(api, [_scenario(api), _scenario(api, _case("new"))]),
                          baseline, ("scenario_added:new",))
    larger = api.BaselineManifest.from_suite(_suite(api, [
        _scenario(api), _scenario(api, _case("removed")),
    ]))
    _assert_compatibility(api, suite, larger, ("scenario_removed:removed",))

    old = _suite(api, [_scenario(api, _case(name)) for name in
                       ("z-removed", "shared", "a-removed")])
    shared = _scenario(api, _case("shared", "2", prompt="Changed shared task."),
                       fixture_version="fixture-v2")
    current_cases = [_scenario(api, _case("z-added")), shared,
                     _scenario(api, _case("a-added"))]
    changes = {"suite_id": "other-suite", "suite_version": "2",
               "evaluation_profile": "other-profile", "driver_version": "driver-v2"}
    old_manifest = api.BaselineManifest.from_suite(old)
    expected = (
        "suite_id_mismatch", "suite_version_mismatch", "evaluation_profile_mismatch",
        "driver_version_mismatch", "scenario_added:a-added", "scenario_added:z-added",
        "scenario_removed:a-removed", "scenario_removed:z-removed",
        "scenario_version_mismatch:shared", "content_fingerprint_mismatch:shared",
        "fixture_version_mismatch:shared",
    )
    for ordered in (current_cases, list(reversed(current_cases))):
        current = _suite(api, ordered, **changes)
        _assert_compatibility(api, current, old_manifest, expected)
        _assert_compatibility(api, current, old_manifest, expected)
    assert (suite.to_json(), baseline.to_json()) == before

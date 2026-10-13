"""Explicit evaluation definitions, immutable snapshots and v1 JSON artifacts.

M10 owns scoring and execution. This module only registers definitions and
checks whether a suite matches a baseline definition. All serialization is in
memory; the host owns file placement and baseline selection.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from hashlib import sha256
import json
from math import isfinite

from .evaluation import EvaluationCase


__all__ = (
    "EvaluationScenario", "ScenarioRegistry", "EvaluationSuite",
    "BaselineManifest", "BaselineCompatibility", "check_baseline_compatibility",
)

_FORMAT_VERSION = "v1"
_MAX_SCENARIOS = 100
_DEFINITION_FIELDS = (
    "suite_id", "suite_version", "evaluation_profile", "driver_version",
)
_CASE_FIELDS = (
    "case_id", "scenario_version", "prompt", "expected_facts", "scoring_spec",
)
_SCENARIO_FIELDS = (
    "case", "context_prompts", "fixture_version", "metadata", "content_fingerprint",
)
_ENTRY_FIELDS = (
    "case_id", "scenario_version", "content_fingerprint", "fixture_version",
)


def _text(value: object, name: str) -> str:
    if type(value) is not str or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")
    return value


def _json_copy(value: object, *, allow_tuples: bool = False) -> object:
    """Copy explicit JSON types without coercing keys, booleans or numbers."""
    def visit(item):
        if item is None or type(item) in (str, bool, int):
            return item
        if type(item) is float and isfinite(item):
            return item
        if type(item) is list or (allow_tuples and type(item) is tuple):
            return [visit(child) for child in item]
        if isinstance(item, Mapping) and all(type(key) is str for key in item):
            return {key: visit(child) for key, child in item.items()}
        raise ValueError("Expected finite JSON values with string object keys")

    try:
        return visit(value)
    except RecursionError as exc:
        raise ValueError("JSON content is cyclic or too deeply nested") from exc


def _mapping(value: object, *, allow_tuples: bool = False) -> dict:
    if not isinstance(value, Mapping):
        raise ValueError("Expected an object mapping")
    return _json_copy(value, allow_tuples=allow_tuples)


def _canonical(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False, allow_nan=False)


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("Duplicate JSON object key")
        result[key] = value
    return result


def _finite_float(text):
    value = float(text)
    if not isfinite(value):
        raise ValueError("Non-finite JSON number")
    return value


def _invalid_constant(text):
    raise ValueError("Non-finite JSON constant")


def _load_json(text: str) -> object:
    if type(text) is not str:
        raise ValueError("Expected JSON text")
    try:
        return json.loads(text, object_pairs_hook=_unique_object,
                          parse_float=_finite_float, parse_constant=_invalid_constant)
    except RecursionError as exc:
        raise ValueError("JSON content is too deeply nested") from exc


def _fields(value, names):
    if type(value) is not dict or set(value) != set(names):
        raise ValueError("Invalid or missing definition fields")


def _texts(value, name):
    if type(value) not in (list, tuple):
        raise ValueError(f"{name} must be an ordered sequence")
    return tuple(_text(item, name) for item in value)


def _case_values(values):
    _fields(values, _CASE_FIELDS)
    result = {name: _text(values[name], name)
              for name in ("case_id", "scenario_version", "prompt")}
    result["expected_facts"] = _mapping(values["expected_facts"])
    scoring = _mapping(values["scoring_spec"], allow_tuples=True)
    # Validate definition shape only. Fact coverage and score behavior belong
    # to M10; extra JSON scoring declarations are preserved in the fingerprint.
    if "required_tools" not in scoring or "fact_paths" not in scoring:
        raise ValueError("scoring_spec requires required_tools and fact_paths")
    _texts(scoring["required_tools"], "required_tools")
    paths = _mapping(scoring["fact_paths"])
    for path in paths.values():
        # M10 traverses object keys literally, including an empty-string key;
        # an empty path selects the whole evidence object.
        if type(path) is not list or any(type(key) is not str for key in path):
            raise ValueError("fact_paths values must be arrays of string keys")
    result["scoring_spec"] = scoring
    return result


def _fingerprint_text(value):
    if (type(value) is not str or len(value) != 64
            or any(char not in "0123456789abcdef" for char in value)):
        raise ValueError("Expected a lowercase SHA-256 fingerprint")
    return value


@dataclass(frozen=True, slots=True, init=False)
class EvaluationScenario:
    """Immutable definition; case and metadata reads return detached values.

    Scoring tuples encode as JSON arrays, which M10 consumes with the same
    ordering semantics. Case/version identity is separate from content hashing.
    """

    _case_json: str
    _metadata_json: str
    context_prompts: tuple[str, ...]
    fixture_version: str
    key: tuple[str, str]
    fingerprint: str

    def __init__(self, case: EvaluationCase, *, context_prompts=(),
                 fixture_version: str, metadata=None):
        if not isinstance(case, EvaluationCase):
            raise ValueError("Expected an M10 EvaluationCase")
        values = _case_values({name: getattr(case, name) for name in _CASE_FIELDS})
        context = _texts(context_prompts, "context_prompts")
        fixture = _text(fixture_version, "fixture_version")
        declared_metadata = _mapping({} if metadata is None else metadata)
        content = {
            "format_version": _FORMAT_VERSION,
            "prompt": values["prompt"], "context_prompts": list(context),
            "expected_facts": values["expected_facts"],
            "scoring_spec": values["scoring_spec"],
            "fixture_version": fixture, "metadata": declared_metadata,
        }
        digest = sha256(_canonical(content).encode("utf-8")).hexdigest()
        object.__setattr__(self, "_case_json", _canonical(values))
        object.__setattr__(self, "_metadata_json", _canonical(declared_metadata))
        object.__setattr__(self, "context_prompts", context)
        object.__setattr__(self, "fixture_version", fixture)
        object.__setattr__(self, "key", (values["case_id"], values["scenario_version"]))
        object.__setattr__(self, "fingerprint", digest)

    @property
    def case(self) -> EvaluationCase:
        return EvaluationCase(**json.loads(self._case_json))

    @property
    def metadata(self) -> dict:
        return json.loads(self._metadata_json)

    def _payload(self):
        return {
            "case": json.loads(self._case_json),
            "context_prompts": list(self.context_prompts),
            "fixture_version": self.fixture_version, "metadata": self.metadata,
            "content_fingerprint": self.fingerprint,
        }


def _snapshot(scenario):
    if type(scenario) is not EvaluationScenario:
        raise ValueError("Expected an EvaluationScenario")
    return EvaluationScenario(scenario.case, context_prompts=scenario.context_prompts,
                              fixture_version=scenario.fixture_version,
                              metadata=scenario.metadata)


class ScenarioRegistry:
    """Explicit instance-owned registry; duplicate registration never overwrites."""

    def __init__(self):
        self._scenarios: dict[tuple[str, str], EvaluationScenario] = {}

    def register(self, scenario: EvaluationScenario) -> None:
        snapshot = _snapshot(scenario)
        if snapshot.key in self._scenarios:
            raise ValueError("Scenario key is already registered")
        self._scenarios[snapshot.key] = snapshot

    def get(self, case_id: str, scenario_version: str) -> EvaluationScenario:
        key = (_text(case_id, "case_id"), _text(scenario_version, "scenario_version"))
        # The value contains no exposed mutable storage; case/metadata properties
        # construct independent containers on every read.
        return self._scenarios[key]


def _set_definition(instance, suite_id, suite_version, evaluation_profile, driver_version):
    for name, value in zip(_DEFINITION_FIELDS,
                           (suite_id, suite_version, evaluation_profile, driver_version)):
        object.__setattr__(instance, name, _text(value, name))


def _definition_payload(definition):
    return {"format_version": _FORMAT_VERSION,
            **{name: getattr(definition, name) for name in _DEFINITION_FIELDS}}


def _bounded_sequence(values):
    if (not isinstance(values, Sequence) or isinstance(values, (str, bytes))
            or not 1 <= len(values) <= _MAX_SCENARIOS):
        raise ValueError("A suite requires 1..100 scenarios")
    return tuple(values)


def _unique_cases(keys):
    case_ids = [case_id for case_id, version in keys]
    if len(set(case_ids)) != len(case_ids):
        raise ValueError("A suite permits only one scenario per case_id")


def _manifest(text, *, baseline=False):
    payload = _load_json(text)
    required = ("format_version", *_DEFINITION_FIELDS, "scenarios")
    _fields(payload, (*required, "metadata") if baseline else required)
    if payload["format_version"] != _FORMAT_VERSION:
        raise ValueError("Unsupported evaluation definition format version")
    for name in _DEFINITION_FIELDS:
        _text(payload[name], name)
    if type(payload["scenarios"]) is not list:
        raise ValueError("scenarios must be a JSON array")
    _bounded_sequence(payload["scenarios"])
    return payload


def _scenario_from_payload(payload):
    _fields(payload, _SCENARIO_FIELDS)
    values = _case_values(payload["case"])
    scenario = EvaluationScenario(
        EvaluationCase(**values), context_prompts=payload["context_prompts"],
        fixture_version=payload["fixture_version"], metadata=_mapping(payload["metadata"]),
    )
    if scenario.fingerprint != _fingerprint_text(payload["content_fingerprint"]):
        raise ValueError("Scenario content does not match its fingerprint")
    return scenario


@dataclass(frozen=True, slots=True, init=False)
class EvaluationSuite:
    """Bounded, canonically ordered snapshot of selected scenario definitions."""

    suite_id: str
    suite_version: str
    evaluation_profile: str
    driver_version: str
    scenarios: tuple[EvaluationScenario, ...]

    def __init__(self, suite_id: str, suite_version: str, scenarios, *,
                 evaluation_profile: str, driver_version: str):
        _set_definition(self, suite_id, suite_version, evaluation_profile, driver_version)
        snapshots = tuple(_snapshot(item) for item in _bounded_sequence(scenarios))
        _unique_cases(item.key for item in snapshots)
        object.__setattr__(self, "scenarios", tuple(sorted(snapshots, key=lambda item: item.key)))

    def to_json(self) -> str:
        return _canonical({**_definition_payload(self),
                           "scenarios": [item._payload() for item in self.scenarios]})

    @classmethod
    def from_json(cls, text: str) -> EvaluationSuite:
        payload = _manifest(text)
        return cls(**{name: payload[name] for name in _DEFINITION_FIELDS},
                   scenarios=[_scenario_from_payload(item) for item in payload["scenarios"]])


@dataclass(frozen=True, slots=True)
class _BaselineScenario:
    case_id: str
    scenario_version: str
    content_fingerprint: str
    fixture_version: str

    def __post_init__(self):
        for name in ("case_id", "scenario_version", "fixture_version"):
            _text(getattr(self, name), name)
        _fingerprint_text(self.content_fingerprint)

    def _payload(self):
        return {name: getattr(self, name) for name in _ENTRY_FIELDS}


@dataclass(frozen=True, slots=True, init=False)
class BaselineManifest:
    """Definition identity plus isolated, non-identifying comparison metadata."""

    suite_id: str
    suite_version: str
    evaluation_profile: str
    driver_version: str
    scenarios: tuple[_BaselineScenario, ...]
    _metadata_json: str

    def __init__(self, suite_id: str, suite_version: str, scenarios, *,
                 evaluation_profile: str, driver_version: str, metadata=None):
        _set_definition(self, suite_id, suite_version, evaluation_profile, driver_version)
        entries = _bounded_sequence(scenarios)
        if any(type(item) is not _BaselineScenario for item in entries):
            raise ValueError("Expected baseline scenario definitions")
        _unique_cases((item.case_id, item.scenario_version) for item in entries)
        object.__setattr__(self, "scenarios", tuple(sorted(
            entries, key=lambda item: (item.case_id, item.scenario_version),
        )))
        object.__setattr__(self, "_metadata_json",
                           _canonical(_mapping({} if metadata is None else metadata)))

    @property
    def metadata(self) -> dict:
        return json.loads(self._metadata_json)

    @classmethod
    def from_suite(cls, suite: EvaluationSuite, *, metadata=None) -> BaselineManifest:
        if type(suite) is not EvaluationSuite:
            raise ValueError("Expected an EvaluationSuite")
        entries = [_BaselineScenario(*item.key, item.fingerprint, item.fixture_version)
                   for item in suite.scenarios]
        return cls(**{name: getattr(suite, name) for name in _DEFINITION_FIELDS},
                   scenarios=entries, metadata=metadata)

    def to_json(self) -> str:
        return _canonical({**_definition_payload(self), "metadata": self.metadata,
                           "scenarios": [item._payload() for item in self.scenarios]})

    @classmethod
    def from_json(cls, text: str) -> BaselineManifest:
        payload = _manifest(text, baseline=True)
        entries = []
        for item in payload["scenarios"]:
            _fields(item, _ENTRY_FIELDS)
            entries.append(_BaselineScenario(**item))
        return cls(**{name: payload[name] for name in _DEFINITION_FIELDS},
                   scenarios=entries, metadata=_mapping(payload["metadata"]))


@dataclass(frozen=True, slots=True)
class BaselineCompatibility:
    """Definition compatibility only; no score or release decision."""

    compatible: bool
    reason_codes: tuple[str, ...]


def check_baseline_compatibility(
    suite: EvaluationSuite, baseline: BaselineManifest,
) -> BaselineCompatibility:
    """Report all definition differences in category order, then case-ID order."""
    if type(suite) is not EvaluationSuite or type(baseline) is not BaselineManifest:
        raise ValueError("Expected an EvaluationSuite and BaselineManifest")
    reasons = [f"{name}_mismatch" for name in _DEFINITION_FIELDS
               if getattr(suite, name) != getattr(baseline, name)]
    current = {item.key[0]: item for item in suite.scenarios}
    previous = {item.case_id: item for item in baseline.scenarios}
    reasons.extend(f"scenario_added:{key}" for key in sorted(current.keys() - previous.keys()))
    reasons.extend(f"scenario_removed:{key}" for key in sorted(previous.keys() - current.keys()))
    shared = sorted(current.keys() & previous.keys())
    reasons.extend(f"scenario_version_mismatch:{key}" for key in shared
                   if current[key].key[1] != previous[key].scenario_version)
    reasons.extend(f"content_fingerprint_mismatch:{key}" for key in shared
                   if current[key].fingerprint != previous[key].content_fingerprint)
    reasons.extend(f"fixture_version_mismatch:{key}" for key in shared
                   if current[key].fixture_version != previous[key].fixture_version)
    return BaselineCompatibility(compatible=not reasons, reason_codes=tuple(reasons))

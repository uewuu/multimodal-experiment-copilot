"""M14 Slice 1 RED: host-authorized read-only Memory tools, no Agent binding.

Future tool_layer.memory_tools API:
  MemoryAccessScope(namespace, allowed_experiment_ids, *, allow_invalidated=False)
    is immutable, snapshots its explicit allowlist; empty means no access.
  MemoryTools(scope, memory_provider)
    borrows a zero-argument callable returning a Memory for THIS call/thread.
    It acquires once per authorized invocation, never caches or closes Memory.
  MemoryTools.list_tools() -> isolated OpenAI-compatible descriptions.
  MemoryTools.invoke_tool(name, arguments) -> strict JSON-compatible dictionary.
    Unknown tool names raise KeyError; model argument failures use safe envelopes.

Exactly three tools are declared, in NAMES order below. Namespace is host-only.
All input objects reject unknown keys. IDs/metric names are nonblank exact
strings, at most 256 characters, without normalization. Filters retain M12's
six concrete-value AND semantics (including empty strings); null is invalid.
Limit is a non-bool int 1..50, default 10; include_invalidated is a strict bool.
Unknown top-level options -> unsupported_option; bad filters -> invalid_filter;
other malformed arguments -> invalid_arguments. Validation/authorization happen
before provider acquisition. Disallowed experiment or invalidation access ->
unauthorized_scope, with no scope/record disclosures.

Envelopes are exactly {format_version: 1, ok: True, data: ...} or
{format_version: 1, ok: False, error: {code: ...}}. Projection shapes are frozen
by the explicit expected values below. Unknown declarations remain null.
Record references, locations, full summaries and invalidation text are omitted.
Metric artifact references become 'not_declared' or 'withheld', never paths.
Lookup/list comparability is null. Candidate assessments retain M11 values and
reason order; the stored reference itself is retained with is_reference=True.

Lists probe Memory with limit+1 and return the first limit, plus has_more. This
is NOT pagination, a database-scan/I/O bound, a duration bound, or a cross-call
snapshot guarantee. Candidate references are retrieved in the same authorized
experiment, using the SAME acquired Memory as the candidate query. No arbitrary
reference values or comparable-only/ranking selector is exposed.

Every successful envelope is bounded to 32768 bytes under json.dumps with
ensure_ascii=False, allow_nan=False, sort_keys=True, separators=(',', ':').
Exactly the boundary is accepted; overflow replaces the WHOLE result with
result_too_large. No silent evidence clipping or skipping. Adapter bounds remain
separate and unchanged. RepositoryError maps safely; unexpected programmer
exceptions, TimeoutError and control exceptions propagate, with no retry.

Fixtures use real frozen M11/M12 components and in-memory SQLite, prepared by
the host before any missing-capability assertion. No filesystem, network, LLM,
Adapter, Session, Runtime, HTTP or M10/M13 evaluation integration belongs here.
"""

from contextlib import contextmanager
from copy import deepcopy
from dataclasses import asdict, replace
import importlib
import importlib.util
import json
import sqlite3
from threading import Thread, get_ident

import pytest

from experiment_identity import ExecutionIdentity, ExperimentIdentity, ExperimentProvenance
from experiment_memory import ExperimentMemory
from experiment_record import build_experiment_record
from experiment_repository import (
    ExperimentRepository, RepositoryCorruptionError, RepositoryError,
    UnsupportedRepositoryVersionError,
)


NAMES = (
    "get_historical_execution", "list_historical_executions",
    "find_historical_comparison_candidates",
)
FILTERS = ("dataset_version", "split_id", "task", "target", "realized_seed", "source_revision")
METRIC_FIELDS = ("metric_definition", "direction", "aggregation", "evaluation_protocol", "selection_protocol")
ARTIFACTS = ("checkpoint_ref", "result_artifact_ref", "history_ref")
FORBIDDEN = ("add", "invalidate", "delete", "overwrite", "import", "repair", "migrate",
             "promote_baseline", "launch_training", "edit_config", "mutate_artifact", "reuse_result")


def _api():
    name = "tool_layer.memory_tools"
    assert importlib.util.find_spec(name) is not None, f"missing {name}"
    module = importlib.import_module(name)
    for symbol in ("MemoryAccessScope", "MemoryTools"):
        assert callable(getattr(module, symbol, None)), f"missing {name}.{symbol}"
    return module


def _key(record):
    return record.provenance.execution


def _record(execution_id="A-reference", *, experiment_id="exp", namespace="lab", seed=7,
            value=0.8, task="regression", protocol="eval-v1", unknown=False):
    key = ExecutionIdentity(ExperimentIdentity(namespace, experiment_id), execution_id)
    provenance = ExperimentProvenance(
        key, task=task, target="target", dataset_version="dataset-v1", split_id="validation",
        realized_seed=seed, source_revision="revision-v1", configuration_ref="SECRET/config.yaml",
        model_ref="SECRET/model", location="SECRET/location", environment_ref="SECRET/env",
        initialization_checkpoint_ref="SECRET/initial.pt", recorded_at="fixed-host-label",
    )
    if unknown:
        provenance = ExperimentProvenance(key, realized_seed=seed)
    return build_experiment_record({
        "experiment_name": "untrusted display name", "experiment_dir": "SECRET/experiment",
        "summary": {"private": "SECRET/full-summary", "validation_metrics": {
            "r2": {"metric_name": "r2", "best_value": value, "best_epoch": 3},
        }},
    }, provenance=provenance, metric_declarations={"r2": {
        "metric_definition": "r2-v1", "direction": "maximize", "aggregation": "macro",
        "evaluation_protocol": protocol, "selection_protocol": "best-validation",
        "checkpoint_ref": "SECRET/checkpoint.pt", "result_artifact_ref": None,
        "history_ref": "SECRET/history.json",
    }})


class _MemorySpy(ExperimentMemory):
    def __init__(self, repository):
        super().__init__(repository)
        self.calls = []
        self.retrieved = []

    def get(self, execution):
        self.calls.append(("get", execution))
        result = super().get(execution)
        self.retrieved.append(result)
        return result

    def list_executions(self, experiment, **options):
        self.calls.append(("list", experiment, deepcopy(options)))
        return super().list_executions(experiment, **options)

    def find_comparison_candidates(self, experiment, reference_metric_result, **options):
        self.calls.append(("candidates", experiment, reference_metric_result, deepcopy(options)))
        return super().find_comparison_candidates(experiment, reference_metric_result, **options)


class _Provider:
    def __init__(self, repository):
        self.repository = repository
        self.memories = []
        self.threads = []

    def __call__(self):
        self.threads.append(get_ident())
        memory = _MemorySpy(self.repository)
        self.memories.append(memory)
        return memory


@contextmanager
def _host_repository(records, invalidated=()):
    repository = ExperimentRepository(":memory:")
    try:
        for record in records:
            assert repository.add(record).record == record
        for record in invalidated:
            repository.invalidate(_key(record), reason="SECRET/invalidation-reason")
        yield repository
    finally:
        repository.close()


@pytest.fixture
def history():
    records = [
        _record("z-comparable", seed=9, value=0.1),
        _record("b-unknown", unknown=True),
        _record("a-incompatible", protocol="eval-v2", value=0.99),
        _record("d-invalidated"), _record(),
        _record(namespace="other-lab", value=0.2),
        _record(experiment_id="other-exp", value=0.3),
    ]
    with _host_repository(records, [records[3]]) as repository:
        experiment = ExperimentIdentity("lab", "exp")
        assert len(repository.list_executions(experiment, include_invalidated=True)) == 5
        reference = repository.get(_key(records[4])).record.metric_results["r2"]
        assert [c.comparability.status for c in ExperimentMemory(repository).find_comparison_candidates(
            experiment, reference,
        )] == ["comparable", "incompatible", "unknown", "comparable"]
        yield repository


def _tools(api, repository, *, allowed=("exp",), allow_invalidated=False):
    provider = _Provider(repository)
    scope = api.MemoryAccessScope("lab", allowed, allow_invalidated=allow_invalidated)
    return api.MemoryTools(scope, provider), provider


def _success(data):
    return {"format_version": 1, "ok": True, "data": data}


def _error(code):
    return {"format_version": 1, "ok": False, "error": {"code": code}}


def _encoded(value):
    return json.dumps(value, ensure_ascii=False, allow_nan=False, sort_keys=True,
                      separators=(",", ":")).encode("utf-8")


def _identity(execution):
    return {"namespace": execution.experiment.namespace,
            "experiment_id": execution.experiment.experiment_id, "execution_id": execution.execution_id}


def _metric(metric):
    # Explicit expected transport projection, not a scoring/comparability implementation.
    return {
        "metric_name": metric.metric_name, "value": metric.value, "best_epoch": metric.best_epoch,
        "metric_provenance": {
            **{name: getattr(metric.provenance, name) for name in ("task", "target", "dataset_version", "split_id")},
            **{name: getattr(metric, name) for name in METRIC_FIELDS},
        },
        "artifacts": {name: "not_declared" if getattr(metric, name) is None else "withheld"
                      for name in ARTIFACTS},
    }


def _entry(stored):
    return {"execution": _identity(_key(stored.record)), "lifecycle": {"status": stored.status},
            "provenance": {name: getattr(stored.record.provenance, name) for name in FILTERS},
            "comparability": None}


def _lookup(stored):
    return {**_entry(stored), "metrics": {name: _metric(metric)
                                        for name, metric in stored.record.metric_results.items()}}


def _arguments(name, **changes):
    arguments = {"experiment_id": "exp"}
    if name == NAMES[0]:
        arguments["execution_id"] = "A-reference"
    if name == NAMES[2]:
        arguments.update(reference_execution_id="A-reference", metric_name="r2")
    return {**arguments, **changes}


def _forbidden_provider():
    pytest.fail("Memory provider must not be acquired for rejected requests")


def test_scope_is_immutable_and_authorization_precedes_provider_acquisition(history):
    api = _api()
    supplied = ["exp"]
    scope = api.MemoryAccessScope("lab", supplied)
    supplied.append("other-exp")
    assert scope.namespace == "lab" and scope.allow_invalidated is False
    assert frozenset(scope.allowed_experiment_ids) == frozenset({"exp"})
    for field, value in (("namespace", "other-lab"), ("allowed_experiment_ids", ("other-exp",)),
                         ("allow_invalidated", True)):
        with pytest.raises((AttributeError, TypeError)):
            setattr(scope, field, value)
    with pytest.raises((AttributeError, TypeError)):
        scope.allowed_experiment_ids.add("other-exp")
    calls = []

    def forbidden():
        calls.append(True)
        return _forbidden_provider()

    for current in (scope, api.MemoryAccessScope("lab", []), api.MemoryAccessScope("lab", ["*"])):
        tools = api.MemoryTools(current, forbidden)
        for name in NAMES:
            assert tools.invoke_tool(name, _arguments(name, experiment_id="other-exp")) == _error("unauthorized_scope")
        assert calls == []
    for name in NAMES:
        assert api.MemoryTools(api.MemoryAccessScope("lab", []), forbidden).invoke_tool(
            name, _arguments(name),
        ) == _error("unauthorized_scope")
    assert calls == []
    for field, value in (("namespace", ""), ("namespace", None),
                         ("allowed_experiment_ids", "exp"), ("allowed_experiment_ids", [""]),
                         ("allow_invalidated", 1)):
        options = {"namespace": "lab", "allowed_experiment_ids": ["exp"], "allow_invalidated": False}
        options[field] = value
        with pytest.raises((TypeError, ValueError)):
            api.MemoryAccessScope(**options)


def test_tool_definitions_are_exact_strict_isolated_and_read_only(history):
    api = _api()
    tools = api.MemoryTools(api.MemoryAccessScope("lab", ["exp"]), _forbidden_provider)
    definitions = tools.list_tools()
    assert [d["function"]["name"] for d in definitions] == list(NAMES)
    text = {"type": "string", "minLength": 1, "maxLength": 256, "pattern": r"\S"}
    invalidated = {"type": "boolean", "default": False}
    filters = {"type": "object", "properties": {
        name: {"type": "integer" if name == "realized_seed" else "string"} for name in FILTERS
    }, "additionalProperties": False}
    query = {"filters": filters, "limit": {"type": "integer", "minimum": 1, "maximum": 50, "default": 10},
             "include_invalidated": invalidated}
    properties = [
        {"experiment_id": text, "execution_id": text, "include_invalidated": invalidated},
        {"experiment_id": text, **query},
        {"experiment_id": text, "reference_execution_id": text, "metric_name": text, **query},
    ]
    required = [["experiment_id", "execution_id"], ["experiment_id"],
                ["experiment_id", "reference_execution_id", "metric_name"]]
    for definition, fields, needed in zip(definitions, properties, required):
        assert set(definition) == {"type", "function"} and definition["type"] == "function"
        function = definition["function"]
        assert set(function) == {"name", "description", "parameters"}
        assert isinstance(function["description"], str) and "read-only" in function["description"].lower()
        assert function["parameters"] == {"type": "object", "properties": fields,
                                            "required": needed, "additionalProperties": False}
    assert "recommend" not in definitions[2]["function"]["parameters"]["properties"]
    before = deepcopy(definitions)
    definitions[0]["function"]["parameters"]["properties"].clear()
    assert tools.list_tools() == before
    assert json.loads(_encoded(before)) == before
    for name in FORBIDDEN:
        assert not hasattr(tools, name)
        with pytest.raises(KeyError):
            tools.invoke_tool(name, {})


def test_model_arguments_are_strict_and_rejected_before_acquisition(history):
    api = _api()
    tools = api.MemoryTools(api.MemoryAccessScope("lab", ["exp"]), _forbidden_provider)
    for name in NAMES:
        for arguments in (None, [], "{}", {}):
            assert tools.invoke_tool(name, arguments) == _error("invalid_arguments")
        identity_fields = ["experiment_id"] + (["execution_id"] if name == NAMES[0] else
                          ["reference_execution_id", "metric_name"] if name == NAMES[2] else [])
        for field in identity_fields:
            missing = _arguments(name)
            del missing[field]
            assert tools.invoke_tool(name, missing) == _error("invalid_arguments")
            for value in (None, "", " \t", 1, True, [], "x" * 257):
                assert tools.invoke_tool(name, _arguments(name, **{field: value})) == _error("invalid_arguments")
        for value in (None, 0, 1, "false", []):
            assert tools.invoke_tool(name, _arguments(name, include_invalidated=value)) == _error("invalid_arguments")
        for option in ("namespace", "database_path", "sql", "provenance", "reference_metric_result",
                       "reference_value", "artifact_selector", "filesystem_path", "comparable_only",
                       "direction", "aggregation", "evaluation_protocol", "selection_protocol", "dataset_version", "split_id"):
            assert tools.invoke_tool(name, _arguments(name, **{option: "SECRET"})) == _error("unsupported_option")
    for name in NAMES[1:]:
        for limit in (0, -1, 51, True, False, 1.0, "10", None):
            assert tools.invoke_tool(name, _arguments(name, limit=limit)) == _error("invalid_arguments")
        for filters in (None, [], "all", {"other": "x"}, {"realized_seed": True}, {"realized_seed": 1.0},
                        {"realized_seed": "7"}, *({field: None} for field in FILTERS),
                        *({field: 42} for field in FILTERS if field != "realized_seed")):
            assert tools.invoke_tool(name, _arguments(name, filters=filters)) == _error("invalid_filter")


def test_exact_lookup_projects_safe_evidence_and_preserves_literal_identity(history):
    literals = [_record(" Execution ", experiment_id=" Exp "), _record("X" * 256, experiment_id="E" * 256)]
    with _host_repository(literals) as literal_repository:
        api = _api()
        tools, provider = _tools(api, history)
        key = ExecutionIdentity(ExperimentIdentity("lab", "exp"), "A-reference")
        expected = _success(_lookup(history.get(key)))
        result = tools.invoke_tool(NAMES[0], _arguments(NAMES[0]))
        assert result == expected and json.loads(_encoded(result)) == result
        assert b"SECRET" not in _encoded(result)
        assert result["data"]["metrics"]["r2"]["artifacts"] == {
            "checkpoint_ref": "withheld", "result_artifact_ref": "not_declared", "history_ref": "withheld",
        }
        assert provider.memories[0].calls == [("get", key)]
        result["data"]["metrics"]["r2"]["value"] = -999
        assert tools.invoke_tool(NAMES[0], _arguments(NAMES[0])) == expected
        assert len(provider.memories) == 2 and provider.memories[0] is not provider.memories[1]
        unknown = tools.invoke_tool(NAMES[0], _arguments(NAMES[0], execution_id="b-unknown"))
        assert unknown["data"]["provenance"]["dataset_version"] is None
        assert unknown["data"]["comparability"] is None
        exact_tools, _ = _tools(api, literal_repository, allowed=(" Exp ", "E" * 256))
        for record in literals:
            execution = _key(record)
            assert exact_tools.invoke_tool(NAMES[0], {
                "experiment_id": execution.experiment.experiment_id, "execution_id": execution.execution_id,
            }) == _success(_lookup(literal_repository.get(execution)))
        assert exact_tools.invoke_tool(NAMES[0], {"experiment_id": "Exp", "execution_id": "Execution"}) == _error("unauthorized_scope")
        assert tools.invoke_tool(NAMES[0], _arguments(NAMES[0], execution_id="a-reference")) == _error("execution_not_found")


def test_listing_preserves_exact_filters_order_unknowns_and_bounded_probe(history):
    with _host_repository([_record(f"run-{i:02d}", seed=i % 2, value=i / 100)
                           for i in reversed(range(55))]) as many:
        api = _api()
        tools, provider = _tools(api, history)
        experiment = ExperimentIdentity("lab", "exp")
        memory = ExperimentMemory(history)
        for filters in ({}, {"dataset_version": "dataset-v1"}, {"split_id": "validation"},
                        {"task": "regression"}, {"target": "target"}, {"source_revision": "revision-v1"},
                        {"realized_seed": 9}, {"dataset_version": "dataset-v1", "realized_seed": 9},
                        {"dataset_version": "DATASET-V1"}, {"task": ""}):
            supplied = deepcopy(filters)
            result = tools.invoke_tool(NAMES[1], _arguments(NAMES[1], filters=filters, limit=2))
            expected = memory.list_executions(experiment, filters=filters, limit=3)
            assert result == _success({"items": [_entry(s) for s in expected[:2]], "limit": 2, "has_more": len(expected) > 2})
            assert filters == supplied
            assert provider.memories[-1].calls == [("list", experiment, {
                "filters": filters, "include_invalidated": False, "limit": 3,
            })]
        unfiltered = tools.invoke_tool(NAMES[1], _arguments(NAMES[1]))["data"]
        assert [s["execution"]["execution_id"] for s in unfiltered["items"]] == [
            "A-reference", "a-incompatible", "b-unknown", "z-comparable",
        ]
        assert unfiltered["items"][2]["provenance"]["dataset_version"] is None
        bounded, probes = _tools(api, many)
        for options, limit in (({}, 10), ({"limit": 1}, 1), ({"limit": 50}, 50)):
            result = bounded.invoke_tool(NAMES[1], _arguments(NAMES[1], **options))
            assert result["data"]["limit"] == limit and result["data"]["has_more"] is True
            assert [s["execution"]["execution_id"] for s in result["data"]["items"]] == [f"run-{i:02d}" for i in range(limit)]
            assert probes.memories[-1].calls[0][2] == {"filters": None, "include_invalidated": False, "limit": limit + 1}


def test_invalidated_visibility_requires_both_host_permission_and_request(history):
    api = _api()
    for permitted in (False, True):
        tools, provider = _tools(api, history, allow_invalidated=permitted)
        hidden = _arguments(NAMES[0], execution_id="d-invalidated")
        assert tools.invoke_tool(NAMES[0], hidden) == _error("execution_not_found")
        assert tools.invoke_tool(NAMES[0], _arguments(NAMES[0], execution_id="absent")) == _error("execution_not_found")
        reference = _arguments(NAMES[2], reference_execution_id="d-invalidated")
        assert tools.invoke_tool(NAMES[2], reference) == _error("execution_not_found")
        assert [c[0] for c in provider.memories[-1].calls] == ["get"]
        for name in NAMES:
            args = hidden if name == NAMES[0] else reference if name == NAMES[2] else _arguments(name)
            before = len(provider.memories)
            result = tools.invoke_tool(name, {**args, "include_invalidated": True})
            if not permitted:
                assert result == _error("unauthorized_scope")
                assert len(provider.memories) == before
            else:
                assert result["ok"] is True
                if name == NAMES[0]:
                    assert result["data"]["lifecycle"] == {"status": "invalidated"}
                else:
                    item = next(s for s in result["data"]["items"] if s["execution"]["execution_id"] == "d-invalidated")
                    assert item["lifecycle"] == {"status": "invalidated"}
                    if name == NAMES[2]:
                        assert item["is_reference"] is True and item["comparability"]["status"] == "comparable"
                assert b"SECRET" not in _encoded(result)


def test_comparison_uses_one_acquired_memory_and_the_actual_stored_reference(history):
    api = _api()
    tools, provider = _tools(api, history)
    arguments = _arguments(NAMES[2], filters={"realized_seed": 9}, limit=1)
    result = tools.invoke_tool(NAMES[2], arguments)
    assert result["ok"] is True and len(provider.memories) == 1
    memory = provider.memories[0]
    assert [call[0] for call in memory.calls] == ["get", "candidates"]
    reference = memory.retrieved[0].record.metric_results["r2"]
    assert memory.calls[1][2] is reference
    assert memory.calls[0][1] == ExecutionIdentity(ExperimentIdentity("lab", "exp"), "A-reference")
    assert memory.calls[1][1] == ExperimentIdentity("lab", "exp")
    assert memory.calls[1][3] == {"filters": {"realized_seed": 9}, "limit": 2,
                                  "include_invalidated": False, "comparable_only": False}
    assert result["data"]["reference"] == {
        "execution": _identity(reference.provenance.execution), "lifecycle": {"status": "active"},
        "metric": _metric(reference),
    }
    assert result["data"]["limit"] == 1 and result["data"]["has_more"] is False
    assert result["data"]["items"][0]["execution"]["execution_id"] == "z-comparable"
    first_only = tools.invoke_tool(NAMES[2], _arguments(NAMES[2], limit=1))
    assert first_only["data"]["has_more"] is True
    assert len(first_only["data"]["items"]) == 1
    assert first_only["data"]["items"][0]["is_reference"] is True
    assert provider.memories[-1].calls[1][3]["limit"] == 2
    assert tools.invoke_tool(NAMES[2], _arguments(NAMES[2], reference_execution_id="absent")) == _error("execution_not_found")
    assert tools.invoke_tool(NAMES[2], _arguments(NAMES[2], metric_name="absent")) == _error("metric_not_found")
    assert all([c[0] for c in m.calls] == ["get"] for m in provider.memories[-2:])
    # Metric names are literal keys as well; no trim/case normalization or
    # narrower hidden bound than the declared 256-character maximum.
    for metric_name in (" R2 ", "M" * 256):
        record = _record()
        record = replace(record, summary={"validation_metrics": {
            metric_name: {**record.summary["validation_metrics"]["r2"], "metric_name": metric_name},
        }}, metric_results={metric_name: replace(record.metric_results["r2"], metric_name=metric_name)})
        with _host_repository([record]) as repository:
            literal, _ = _tools(api, repository)
            result = literal.invoke_tool(NAMES[2], _arguments(NAMES[2], metric_name=metric_name))
            assert result["ok"] is True
            assert result["data"]["reference"]["metric"]["metric_name"] == metric_name


def test_candidates_preserve_all_m11_states_reasons_policy_and_self_reference(history):
    api = _api()
    tools, _ = _tools(api, history)
    memory = ExperimentMemory(history)
    reference = memory.get(ExecutionIdentity(ExperimentIdentity("lab", "exp"), "A-reference"))
    metric = reference.record.metric_results["r2"]
    expected = memory.find_comparison_candidates(metric.provenance.execution.experiment, metric)
    result = tools.invoke_tool(NAMES[2], _arguments(NAMES[2]))
    assert result == _success({
        "reference": {"execution": _identity(_key(reference.record)), "lifecycle": {"status": "active"}, "metric": _metric(metric)},
        "items": [{"execution": _identity(_key(c.stored.record)), "lifecycle": {"status": c.stored.status},
                   "metric": _metric(c.metric_result), "comparability": {
                       **asdict(c.comparability), "reason_codes": list(c.comparability.reason_codes),
                   }, "is_reference": _key(c.stored.record) == _key(reference.record)} for c in expected],
        "limit": 10, "has_more": False,
    })
    assert [c["comparability"]["status"] for c in result["data"]["items"]] == ["comparable", "incompatible", "unknown", "comparable"]
    assert [c["is_reference"] for c in result["data"]["items"]] == [True, False, False, False]
    unknown = tools.invoke_tool(NAMES[2], _arguments(NAMES[2], reference_execution_id="b-unknown"))
    self_item = next(c for c in unknown["data"]["items"] if c["is_reference"])
    assert self_item["comparability"] == {
        "status": "unknown", "reason_codes": ["missing_task", "missing_target", "missing_dataset_version", "missing_split_id"],
        "policy_version": "metric-result-v1",
    }
    assert self_item["metric"]["metric_provenance"]["task"] is None


def test_repository_errors_are_sanitized_at_acquisition_and_each_read_boundary(history):
    api = _api()
    errors = [(RepositoryCorruptionError, "repository_corrupt"),
              (UnsupportedRepositoryVersionError, "unsupported_repository_version"),
              (RepositoryError, "repository_unavailable")]
    for error_type, code in errors:
        for boundary in ("provider", "get", "list_executions", "find_comparison_candidates"):
            count = []

            def fail(*args, **kwargs):
                count.append(True)
                raise error_type("SECRET sqlite=/private/history.db token=credential traceback")

            memory = _MemorySpy(history)
            if boundary == "provider":
                provider = fail
                name = NAMES[0]
            else:
                setattr(memory, boundary, fail)
                provider = lambda: memory
                name = {"get": NAMES[0], "list_executions": NAMES[1], "find_comparison_candidates": NAMES[2]}[boundary]
            tools = api.MemoryTools(api.MemoryAccessScope("lab", ["exp"]), provider)
            assert tools.invoke_tool(name, _arguments(name)) == _error(code)
            assert count == [True]


def test_timeout_control_and_programmer_exceptions_propagate_without_retry(history):
    api = _api()
    for exception in (TimeoutError("deadline"), KeyboardInterrupt(), SystemExit(7),
                      RuntimeError("programmer defect"), AssertionError("broken invariant")):
        for boundary in ("provider", "get", "list_executions", "find_comparison_candidates"):
            calls = []

            def fail(*args, **kwargs):
                calls.append(True)
                raise exception

            memory = _MemorySpy(history)
            if boundary == "provider":
                provider, name = fail, NAMES[0]
            else:
                setattr(memory, boundary, fail)
                provider = lambda: memory
                name = {"get": NAMES[0], "list_executions": NAMES[1], "find_comparison_candidates": NAMES[2]}[boundary]
            tools = api.MemoryTools(api.MemoryAccessScope("lab", ["exp"]), provider)
            with pytest.raises(type(exception)) as caught:
                tools.invoke_tool(name, _arguments(name))
            assert caught.value is exception and calls == [True]


def test_utf8_output_bound_accepts_boundary_and_rejects_whole_oversized_projection(history):
    prototype = _record(task="")
    with _host_repository([prototype]) as repository:
        fixed = len(_encoded(_success(_lookup(repository.get(_key(prototype))))))
    # task occurs in both record and metric provenance: each ASCII character costs two bytes.
    fill, extra = divmod(32768 - fixed, 2)
    boundary = _record(task="x" * fill)
    if extra:
        boundary = replace(boundary, metric_results={"r2": replace(boundary.metric_results["r2"], metric_definition="r2-v1x")})
    oversized = _record(task="界" * 12000)
    with _host_repository([boundary]) as exact, _host_repository([oversized]) as large:
        assert len(_encoded(_success(_lookup(exact.get(_key(boundary)))))) == 32768
        assert len(_encoded(_success(_lookup(large.get(_key(oversized)))))) > 32768
        api = _api()
        tools, _ = _tools(api, exact)
        result = tools.invoke_tool(NAMES[0], _arguments(NAMES[0]))
        assert result == _success(_lookup(exact.get(_key(boundary))))
        assert len(_encoded(result)) == 32768
        tools, _ = _tools(api, large)
        for name in NAMES:
            options = {} if name == NAMES[0] else {"limit": 1}
            assert tools.invoke_tool(name, _arguments(name, **options)) == _error("result_too_large")
        assert large.get(_key(oversized)).record == oversized


def test_tools_borrow_resources_use_only_memory_reads_and_acquire_on_calling_thread(history, monkeypatch):
    api = _api()
    tools, provider = _tools(api, history)
    experiment = ExperimentIdentity("lab", "exp")
    before = history.list_executions(experiment, include_invalidated=True)

    def forbidden(*args, **kwargs):
        pytest.fail("Memory tools must not construct storage, write, access artifacts or close host resources")

    with monkeypatch.context() as guard:
        for method in ("add", "invalidate", "delete", "close"):
            guard.setattr(ExperimentRepository, method, forbidden)
        guard.setattr(ExperimentRepository, "__init__", forbidden)
        guard.setattr(sqlite3, "connect", forbidden)
        guard.setattr("builtins.open", forbidden)
        for name in NAMES:
            assert tools.invoke_tool(name, _arguments(name))["ok"] is True
    assert len(provider.memories) == 3
    assert [[c[0] for c in memory.calls] for memory in provider.memories] == [["get"], ["list"], ["get", "candidates"]]
    assert history.list_executions(experiment, include_invalidated=True) == before
    assert not hasattr(tools, "close")

    # Tools are constructed on the main thread; the HOST prepares/closes each
    # SQLite connection on its worker. The provider merely lends current Memory.
    from threading import local
    state = local()
    acquisitions = []
    outcomes = []

    def current_memory():
        acquisitions.append((get_ident(), state.memory))
        return state.memory

    threaded = api.MemoryTools(api.MemoryAccessScope("lab", ["exp"]), current_memory)

    def worker(value):
        try:
            record = _record(value=value)
            with _host_repository([record]) as repository:
                state.memory = _MemorySpy(repository)
                first = threaded.invoke_tool(NAMES[0], _arguments(NAMES[0]))
                state.memory = _MemorySpy(repository)
                second = threaded.invoke_tool(NAMES[0], _arguments(NAMES[0]))
                assert first == second == _success(_lookup(repository.get(_key(record))))
                outcomes.append((True, get_ident()))
        except BaseException as error:
            outcomes.append((False, error))

    workers = [Thread(target=worker, args=(value,)) for value in (0.2, 0.9)]
    for thread in workers:
        thread.start()
    for thread in workers:
        thread.join(timeout=5)
        assert not thread.is_alive()
    assert len(outcomes) == 2 and all(ok for ok, _ in outcomes), outcomes
    assert len(acquisitions) == 4
    assert all(thread_id != get_ident() for thread_id, _ in acquisitions)
    assert len({id(memory) for _, memory in acquisitions}) == 4

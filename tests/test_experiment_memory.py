"""M12 Slice 3A RED contracts for read-only, experiment-scoped queries.

Future experiment_memory API (no storage implementation or global search):
  ExperimentMemory(repository) borrows an existing ExperimentRepository.
  get(execution) -> the repository value envelope; absence raises KeyError.
  list_executions(experiment, *, filters=None, include_invalidated=False,
                  limit=100) -> tuple of repository value envelopes.
  find_comparison_candidates(experiment, reference_metric_result, *,
      filters=None, include_invalidated=False, comparable_only=False,
      limit=100) -> tuple of candidates with stored, metric_result, comparability.
Candidate.stored preserves record/status/invalidation_reason. Its metric_result
is the stored result with the reference's metric_name; records without that
metric are omitted. comparability is the actual frozen M11 ComparabilityResult.
Reference identity is not implicitly excluded. A reference need not be stored.

Both queries require one explicit ExperimentIdentity and order by execution_id
case-sensitively, ascending, within that namespace/experiment. Apply lifecycle,
provenance, metric availability and explicit comparable-only filters BEFORE
the result limit. Limits bound returned results, not repository I/O: the frozen
repository already returns the complete tuple for one experiment.

filters is an AND mapping over exactly dataset_version, split_id, task, target,
realized_seed and source_revision. Values must be concrete strings, or an int
(not bool) for realized_seed. None never matches a requested known value; None
as a filter value is rejected, not interpreted as wildcard/equal unknown.
Unsupported fields/values and non-positive/non-int limits raise ValueError.
Unfiltered unknown evidence remains visible and unknown under M11 comparison.

Default candidates preserve comparable/incompatible/unknown, including reasons
and policy version, with no score ranking or automatic reuse recommendation.
Invalidation controls visibility only; it does not alter comparability.
All queries use repository reads, never writes, direct SQLite, artifact I/O,
inference, repairs or connection ownership. Returned mutations cannot rewrite
stored evidence. The host remains responsible for repository lifecycle.
"""

from contextlib import closing
from copy import deepcopy
from dataclasses import replace
import importlib
import importlib.util
from pathlib import Path
import sqlite3

import pytest

from experiment_comparability import ComparabilityResult, compare_metric_results
from experiment_identity import (
    ExecutionIdentity,
    ExperimentIdentity,
    ExperimentProvenance,
)
from experiment_record import build_experiment_record
from experiment_repository import ExperimentRepository


def _api():
    assert importlib.util.find_spec("experiment_memory") is not None, (
        "missing M12 ExperimentMemory capability: experiment_memory"
    )
    module = importlib.import_module("experiment_memory")
    assert callable(getattr(module, "ExperimentMemory", None)), (
        "missing M12 memory API: ExperimentMemory"
    )
    return module


def _record(execution_id, *, namespace="lab-a", experiment_id="experiment-x", seed=42, value=0.82):
    provenance = ExperimentProvenance(
        ExecutionIdentity(ExperimentIdentity(namespace, experiment_id), execution_id),
        configuration_ref="config:v1", task="regression", target="target-a",
        model_ref="model:v1", dataset_version="dataset:v1", split_id="split:v1",
        source_revision="repository:revision-1", realized_seed=seed,
        initialization_checkpoint_ref="initial-weights:v1",
    )
    parsed = {
        "experiment_name": "shared-display-name",
        "experiment_dir": "absent/dataset:v1/split:v1/20260921/checkpoint-3.pt",
        "summary": {
            "configuration": {"seed": seed, "feature_list": ["vision", "audio"]},
            "validation_metrics": {
                "r2": {"metric_name": "r2", "best_value": value, "best_epoch": 3},
                "racc": {"metric_name": "racc", "best_value": 0.91, "best_epoch": 7},
            },
        },
    }
    declarations = {name: {
        "metric_definition": f"{name}:v1", "direction": "maximize",
        "aggregation": "macro:v1", "evaluation_protocol": "evaluation:v1",
        "selection_protocol": "best-validation-per-metric:v1",
        "checkpoint_ref": f"absent/checkpoint-{epoch}.pt",
        "result_artifact_ref": f"result:{name}", "history_ref": None,
    } for name, epoch in (("r2", 3), ("racc", 7))}
    return build_experiment_record(parsed, provenance=provenance, metric_declarations=declarations)


def _with_provenance(record, provenance):
    return replace(record, provenance=provenance, metric_results={
        name: replace(metric, provenance=provenance)
        for name, metric in record.metric_results.items()
    })


def _with_protocol(record, protocol):
    return replace(record, metric_results={
        **record.metric_results,
        "r2": replace(record.metric_results["r2"], evaluation_protocol=protocol),
    })


def _key(record):
    return record.provenance.execution


def _ids(stored):
    return tuple(_key(item.record).execution_id for item in stored)


def _candidate_ids(candidates):
    return _ids(tuple(item.stored for item in candidates))


def _database_snapshot(path):
    with closing(sqlite3.connect(path)) as connection:
        return connection.execute("PRAGMA user_version").fetchone()[0], tuple(connection.iterdump())


@pytest.fixture
def history(tmp_path):
    """Real frozen storage, initialized before the missing-capability assertion."""
    unknown = _record("b-unknown")
    unknown = _with_provenance(unknown, ExperimentProvenance(_key(unknown)))
    mixed = _record("c-mixed")
    mixed = _with_provenance(mixed, replace(mixed.provenance, dataset_version=None))
    no_r2 = _record("e-no-r2")
    del no_r2.summary["validation_metrics"]["r2"]
    del no_r2.metric_results["r2"]
    # Metric-policy conflicts are valid stored evidence; changing a known
    # experiment-definition field would correctly be rejected by Repository.
    records = {
        "z": _record("z-comparable", seed=108, value=0.20),
        "unknown": unknown,
        "withdrawn": _record("d-withdrawn"),
        "incompatible": _with_protocol(_record("a-incompatible", value=0.99), "evaluation:v2"),
        "mixed": _with_protocol(mixed, "evaluation:v2"),
        "no_r2": no_r2,
        "A": _record("A-comparable", value=0.60),
        "namespace": _record("z-comparable", namespace="lab-b", seed=108),
        "experiment": _record("z-comparable", experiment_id="experiment-y", seed=108),
    }
    path = tmp_path / "history.sqlite"
    with closing(ExperimentRepository(path)) as repository:
        for record in records.values():
            assert repository.add(record).record == record
        repository.invalidate(_key(records["withdrawn"]), reason="withdrawn by host")
        assert len(repository.list_executions(_key(records["A"]).experiment, include_invalidated=True)) == 7
        yield path, repository, records


@pytest.fixture
def bounded_history(tmp_path):
    path = tmp_path / "bounded.sqlite"
    with closing(ExperimentRepository(path)) as repository:
        for index in reversed(range(105)):
            repository.add(_record(f"execution-{index:03d}", seed=42 if index % 2 == 0 else 108))
        experiment = ExperimentIdentity("lab-a", "experiment-x")
        assert len(repository.list_executions(experiment)) == 105
        yield repository, experiment


def test_memory_exact_lookup_preserves_identity_lifecycle_and_metric_evidence(history):
    api = _api()
    _, repository, records = history
    memory = api.ExperimentMemory(repository)
    for record in records.values():
        restored = memory.get(_key(record))
        assert restored == repository.get(_key(record))
        assert restored.record == record
    withdrawn = memory.get(_key(records["withdrawn"]))
    assert (withdrawn.status, withdrawn.invalidation_reason) == ("invalidated", "withdrawn by host")
    metrics = memory.get(_key(records["A"])).record.metric_results
    assert (metrics["r2"].best_epoch, metrics["racc"].best_epoch) == (3, 7)
    assert (metrics["r2"].checkpoint_ref, metrics["racc"].checkpoint_ref) == (
        "absent/checkpoint-3.pt", "absent/checkpoint-7.pt",
    )
    assert metrics["r2"].history_ref is None
    experiment = _key(records["A"]).experiment
    for absent_id in ("missing", records["A"].experiment_name, records["A"].experiment_dir, "42"):
        with pytest.raises(KeyError):
            memory.get(ExecutionIdentity(experiment, absent_id))


def test_memory_lists_exact_experiment_in_stable_identity_order_after_reopen(history):
    api = _api()
    path, repository, records = history
    experiment = _key(records["A"]).experiment
    memory = api.ExperimentMemory(repository)
    listed = memory.list_executions(experiment)
    assert type(listed) is tuple
    assert _ids(listed) == (
        "A-comparable", "a-incompatible", "b-unknown", "c-mixed", "e-no-r2", "z-comparable",
    )
    assert all(_key(item.record).experiment == experiment for item in listed)
    for outside in (records["namespace"], records["experiment"]):
        assert memory.list_executions(_key(outside).experiment) == (repository.get(_key(outside)),)
    assert memory.list_executions(ExperimentIdentity("lab-missing", "experiment-x")) == ()
    assert memory.list_executions(experiment, filters={}) == listed
    with closing(ExperimentRepository(path)) as reopened:
        assert api.ExperimentMemory(reopened).list_executions(experiment) == listed


def test_memory_filters_only_concrete_declared_provenance_with_and_semantics(history):
    api = _api()
    _, repository, records = history
    experiment = _key(records["A"]).experiment
    memory = api.ExperimentMemory(repository)
    declared = {
        "dataset_version": "dataset:v1", "split_id": "split:v1", "task": "regression",
        "target": "target-a", "realized_seed": 108, "source_revision": "repository:revision-1",
    }
    for field, value in declared.items():
        expected = tuple(item for item in repository.list_executions(experiment)
                         if getattr(item.record.provenance, field) == value)
        matches = memory.list_executions(experiment, filters={field: value})
        assert matches == expected and matches
        assert "b-unknown" not in _ids(matches)
    filters = {"dataset_version": "dataset:v1", "realized_seed": 108}
    before = deepcopy(filters)
    assert _ids(memory.list_executions(experiment, filters=filters)) == ("z-comparable",)
    assert filters == before
    for filters in (
        {"dataset_version": "dataset:v"}, {"dataset_version": "DATASET:v1"},
        {"dataset_version": "dataset:v99"}, {"realized_seed": 999},
        {"dataset_version": "dataset:v1", "split_id": "different"},
    ):
        assert memory.list_executions(experiment, filters=filters) == ()


def test_memory_preserves_unknowns_without_inference_or_unknown_equality(history):
    api = _api()
    _, repository, records = history
    memory = api.ExperimentMemory(repository)
    unknown = records["unknown"]
    restored = memory.get(_key(unknown)).record
    assert restored == unknown
    assert restored.provenance == ExperimentProvenance(_key(unknown))
    experiment = _key(unknown).experiment
    assert "b-unknown" in _ids(memory.list_executions(experiment))
    assert "b-unknown" not in _ids(memory.list_executions(
        experiment, filters={"dataset_version": "dataset:v1"},
    ))
    reference = restored.metric_results["r2"]
    candidates = memory.find_comparison_candidates(experiment, reference)
    same_execution = next(item for item in candidates if _key(item.stored.record) == _key(unknown))
    assert same_execution.metric_result.provenance.dataset_version is None
    assert same_execution.comparability == compare_metric_results(reference, reference)
    assert same_execution.comparability.status == "unknown"
    assert same_execution.comparability.reason_codes == (
        "missing_task", "missing_target", "missing_dataset_version", "missing_split_id",
    )


def test_memory_bounds_results_after_filters_and_rejects_unsupported_query_options(bounded_history):
    api = _api()
    repository, experiment = bounded_history
    memory = api.ExperimentMemory(repository)
    reference = _record("reference", experiment_id="reference").metric_results["r2"]
    expected = tuple(f"execution-{index:03d}" for index in range(100))
    assert _ids(memory.list_executions(experiment)) == expected
    assert _candidate_ids(memory.find_comparison_candidates(experiment, reference)) == expected
    for query in (
        lambda **options: memory.list_executions(experiment, **options),
        lambda **options: memory.find_comparison_candidates(experiment, reference, **options),
    ):
        result = query(filters={"realized_seed": 108}, limit=2)
        stored = tuple(item.stored for item in result) if hasattr(result[0], "stored") else result
        assert _ids(stored) == ("execution-001", "execution-003")
        for limit in (0, -1, True, 1.5, None, "2"):
            with pytest.raises(ValueError):
                query(limit=limit)
        for filters in (
            {"experiment_dir": "absent"}, {"score": 0.8}, {"dataset_version": None},
            {"dataset_version": 1}, {"realized_seed": "108"}, {"realized_seed": True},
        ):
            with pytest.raises(ValueError):
                query(filters=filters)


def test_memory_candidates_preserve_all_m11_states_reasons_and_metric_provenance(history):
    api = _api()
    _, repository, records = history
    memory = api.ExperimentMemory(repository)
    experiment = _key(records["A"]).experiment
    reference = _record("reference", experiment_id="reference", seed=999).metric_results["r2"]
    candidates = memory.find_comparison_candidates(experiment, reference)
    assert type(candidates) is tuple
    assert _candidate_ids(candidates) == (
        "A-comparable", "a-incompatible", "b-unknown", "c-mixed", "z-comparable",
    )
    assert tuple(item.comparability.status for item in candidates) == (
        "comparable", "incompatible", "unknown", "incompatible", "comparable",
    )
    for item in candidates:
        assert item.stored == repository.get(_key(item.stored.record))
        assert item.metric_result == item.stored.record.metric_results["r2"]
        assert type(item.comparability) is ComparabilityResult
        assert item.comparability == compare_metric_results(reference, item.metric_result)
        assert item.comparability.policy_version == "metric-result-v1"
    assert candidates[3].comparability.reason_codes == (
        "missing_dataset_version", "evaluation_protocol_mismatch",
    )
    # Same execution is not an implicit exclusion; reference storage is optional.
    stored_reference = records["A"].metric_results["r2"]
    assert _candidate_ids(memory.find_comparison_candidates(experiment, stored_reference)) == _candidate_ids(candidates)
    assert memory.find_comparison_candidates(experiment, replace(reference, metric_name="absent-metric")) == ()
    for outside in (records["namespace"], records["experiment"]):
        scoped = memory.find_comparison_candidates(_key(outside).experiment, reference)
        assert tuple(item.stored.record for item in scoped) == (outside,)


def test_memory_comparable_only_is_explicit_and_applied_before_limit(history):
    api = _api()
    _, repository, records = history
    memory = api.ExperimentMemory(repository)
    experiment = _key(records["A"]).experiment
    reference = records["A"].metric_results["r2"]
    all_states = memory.find_comparison_candidates(experiment, reference)
    explicit_all = memory.find_comparison_candidates(experiment, reference, comparable_only=False)
    assert explicit_all == all_states
    comparable = memory.find_comparison_candidates(experiment, reference, comparable_only=True, limit=2)
    assert _candidate_ids(comparable) == ("A-comparable", "z-comparable")
    assert comparable == tuple(item for item in all_states if item.comparability.status == "comparable")
    assert _candidate_ids(memory.find_comparison_candidates(
        experiment, reference, filters={"realized_seed": 108}, comparable_only=True, limit=1,
    )) == ("z-comparable",)
    # Selection is per metric: this record's racc remains comparable despite r2's conflict.
    racc = memory.find_comparison_candidates(experiment, records["A"].metric_results["racc"])
    no_r2 = next(item for item in racc if _key(item.stored.record).execution_id == "e-no-r2")
    incompatible_r2 = next(item for item in racc if _key(item.stored.record).execution_id == "a-incompatible")
    assert no_r2.metric_result == records["no_r2"].metric_results["racc"]
    assert incompatible_r2.comparability.status == "comparable"
    assert all(item.metric_result.metric_name == "racc" for item in racc)


def test_memory_invalidation_controls_visibility_without_rewriting_comparability(history):
    api = _api()
    _, repository, records = history
    memory = api.ExperimentMemory(repository)
    experiment = _key(records["A"]).experiment
    reference = records["A"].metric_results["r2"]
    assert "d-withdrawn" not in _ids(memory.list_executions(experiment))
    assert "d-withdrawn" not in _candidate_ids(memory.find_comparison_candidates(experiment, reference))
    listed = memory.list_executions(experiment, include_invalidated=True)
    assert _ids(listed) == (
        "A-comparable", "a-incompatible", "b-unknown", "c-mixed", "d-withdrawn", "e-no-r2", "z-comparable",
    )
    candidates = memory.find_comparison_candidates(experiment, reference, include_invalidated=True)
    assert _candidate_ids(candidates) == (
        "A-comparable", "a-incompatible", "b-unknown", "c-mixed", "d-withdrawn", "z-comparable",
    )
    withdrawn = candidates[4]
    assert withdrawn.stored == memory.get(_key(records["withdrawn"]))
    assert (withdrawn.stored.status, withdrawn.stored.invalidation_reason) == ("invalidated", "withdrawn by host")
    assert withdrawn.comparability == compare_metric_results(reference, records["withdrawn"].metric_results["r2"])
    assert withdrawn.comparability.status == "comparable"
    assert _candidate_ids(memory.find_comparison_candidates(
        experiment, reference, include_invalidated=True, comparable_only=True,
    )) == ("A-comparable", "d-withdrawn", "z-comparable")
    # A trusted host may change lifecycle; subsequent queries must observe it.
    repository.invalidate(_key(records["unknown"]), reason="host withdrew incomplete evidence")
    assert "b-unknown" not in _candidate_ids(memory.find_comparison_candidates(experiment, reference))
    unknown = next(item for item in memory.find_comparison_candidates(
        experiment, reference, include_invalidated=True,
    ) if _key(item.stored.record) == _key(records["unknown"]))
    assert unknown.stored.status == "invalidated"
    assert unknown.comparability.status == "unknown"


def test_memory_queries_are_read_only_borrow_repository_and_do_not_access_artifacts(history, monkeypatch):
    api = _api()
    path, repository, records = history
    experiment = _key(records["A"]).experiment
    reference = records["A"].metric_results["r2"]
    filters = {"dataset_version": "dataset:v1"}
    input_before = deepcopy((records, reference, filters))
    database_before = _database_snapshot(path)
    stored_before = repository.list_executions(experiment, include_invalidated=True)

    def forbidden(*args, **kwargs):
        pytest.fail("Memory must only read through the borrowed repository")

    with monkeypatch.context() as guard:
        for method in ("add", "invalidate", "delete", "close"):
            guard.setattr(repository, method, forbidden)
        guard.setattr(sqlite3, "connect", forbidden)
        guard.setattr("builtins.open", forbidden)
        for method in ("exists", "stat", "is_file", "is_dir", "open", "read_text", "read_bytes", "iterdir", "glob", "rglob"):
            guard.setattr(Path, method, forbidden)
        memory = api.ExperimentMemory(repository)
        for method in ("add", "invalidate", "delete"):
            assert not callable(getattr(memory, method, None)), "Memory exposes no mutation API"
        exact = memory.get(_key(records["A"]))
        listed = memory.list_executions(experiment, filters=filters, include_invalidated=True)
        candidates = memory.find_comparison_candidates(experiment, reference, include_invalidated=True)
        assert exact == repository.get(_key(records["A"]))
        assert listed and candidates
        memory.find_comparison_candidates(experiment, reference, filters=filters, comparable_only=True, limit=1)
        # Read results are values, not a write channel or mutable evidence cache.
        exact.record.summary["configuration"]["feature_list"].clear()
        listed[0].record.metric_results.clear()
        candidates[0].stored.record.summary["configuration"]["seed"] = 999
        assert memory.get(_key(records["A"])).record == records["A"]
        assert memory.list_executions(experiment, include_invalidated=True) == stored_before
    assert (records, reference, filters) == input_before
    assert repository.list_executions(experiment, include_invalidated=True) == stored_before
    assert _database_snapshot(path) == database_before
    with closing(ExperimentRepository(path)) as reopened:
        assert reopened.list_executions(experiment, include_invalidated=True) == stored_before

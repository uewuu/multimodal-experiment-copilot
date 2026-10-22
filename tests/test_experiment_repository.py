"""M12 Slice 1 RED contracts for a local SQLite evidence repository.

Future experiment_repository API:
  ExperimentRepository(database_path), close()
  add(record), get(execution), invalidate(execution, *, reason)
    -> value envelope with record, status ('active'/'invalidated'),
       invalidation_reason (None initially).
  list_executions(experiment, *, include_invalidated=False)
    -> tuple of envelopes, ordered by execution_id (case-sensitive).
  delete(execution) -> bool; get/invalidate of an absent key raise KeyError.
No general update/upsert/revision API is part of this contract.

RepositoryError is the storage error base. RepositoryConflictError exposes
reason_codes and conflicting_execution; same-execution evidence changes use
('execution_snapshot_mismatch',). Cross-execution conflicts use M11's reasons
against the first conflicting retained execution in execution_id order,
including invalidated executions. UnsupportedRepositoryVersionError and
RepositoryCorruptionError distinguish unsupported formats from invalid data.

The v1 on-disk boundary is deliberately explicit for independent fault fixtures:
PRAGMA user_version = 1 is the DATABASE schema version. experiment_records has
namespace, experiment_id, execution_id (the unique, non-null composite key),
record_encoding_version, record_json, status, invalidation_reason columns.
record_encoding_version = 1 describes the separate EVIDENCE encoding.
record_json is the complete dataclasses.asdict(ExperimentRecord) JSON shape;
lifecycle state is outside it. No pickle, artifact resolution, or inferred data.
These columns permit inspection without private connection/codec APIs.

Snapshot equality is structural (mapping order and JSON formatting irrelevant;
list order and scalar types significant), not proof of provenance truth.
Invariants include matching row/payload execution keys, per-metric provenance
equal to record provenance, and metric names/values/epochs matching the summary.
Unsupported versions and corrupt data must fail without migration or repair.
"""

from contextlib import closing
from copy import deepcopy
from dataclasses import asdict, replace
import importlib
import importlib.util
import json
from pathlib import Path
import sqlite3

import pytest

from experiment_identity import (
    ExecutionIdentity,
    ExperimentIdentity,
    ExperimentProvenance,
    check_identity_consistency,
)
from experiment_record import build_experiment_record
from metrics import evaluate_metric_history


def _api():
    assert importlib.util.find_spec("experiment_repository") is not None, (
        "missing M12 ExperimentRepository capability: experiment_repository"
    )
    api = importlib.import_module("experiment_repository")
    for name in (
        "ExperimentRepository", "RepositoryError", "RepositoryConflictError",
        "UnsupportedRepositoryVersionError", "RepositoryCorruptionError",
    ):
        assert callable(getattr(api, name, None)), f"missing M12 repository API: {name}"
    for name in (
        "RepositoryConflictError", "UnsupportedRepositoryVersionError",
        "RepositoryCorruptionError",
    ):
        assert issubclass(getattr(api, name), api.RepositoryError)
    return api


def _record(execution_id="execution-a", *, namespace="lab-a", seed=42):
    provenance = ExperimentProvenance(
        ExecutionIdentity(ExperimentIdentity(namespace, "experiment-x"), execution_id),
        configuration_ref="config:v1", task="regression", target="target-a",
        model_ref="model:v1", dataset_version="dataset:v1", split_id="split:v1",
        source_revision="repository:revision-1",
        initialization_checkpoint_ref="initial-weights:v1", realized_seed=seed,
    )
    histories = {
        "r2": [[1, 0.70], [3, 0.82], [7, 0.78]],
        "racc": [[1, 0.80], [3, 0.85], [7, 0.91]],
    }
    parsed = {
        "experiment_name": "display-name",
        "experiment_dir": "absent/dataset-v99/20260921/checkpoint-3.pt",
        "summary": {
            "configuration": {
                "seed": seed, "feature_list": ["vision", "audio"],
                "use_modality_token_fusion": False,
            },
            "validation_metrics": {
                name: {"metric_name": name, **evaluate_metric_history(history, "maximize")}
                for name, history in histories.items()
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
    return build_experiment_record(
        parsed, provenance=provenance, metric_declarations=declarations,
    )


def _with_provenance(record, provenance):
    return replace(record, provenance=provenance, metric_results={
        name: replace(metric, provenance=provenance)
        for name, metric in record.metric_results.items()
    })


def _key(record):
    return record.provenance.execution


def _reordered(value):
    if isinstance(value, dict):
        return {key: _reordered(item) for key, item in reversed(tuple(value.items()))}
    if isinstance(value, list):
        return [_reordered(item) for item in value]
    return value


def _sql(path, statement, parameters=()):
    with closing(sqlite3.connect(path)) as connection:
        with connection:
            return connection.execute(statement, parameters).fetchall()


def _database_snapshot(path):
    """Logical database contents, including any auxiliary tables, without repair."""
    with closing(sqlite3.connect(path)) as connection:
        return (
            connection.execute("PRAGMA user_version").fetchone()[0],
            tuple(connection.iterdump()),
        )


def test_repository_round_trip_preserves_explicit_identity_and_metric_provenance(
    tmp_path, monkeypatch,
):
    api = _api()
    path = tmp_path / "evidence.sqlite"
    original = _record()
    unknown = _record("execution-unknown")
    unknown = _with_provenance(unknown, ExperimentProvenance(_key(unknown)))
    unknown = replace(unknown, metric_results={
        name: replace(metric, checkpoint_ref=None, result_artifact_ref=None)
        for name, metric in unknown.metric_results.items()
    })
    other_namespace = _record(namespace="lab-b")
    before = deepcopy((original, unknown, other_namespace))
    # Database I/O is permitted; touching caller-declared evidence paths is not.
    def forbid_evidence_access(method):
        def checked(path, *args, **kwargs):
            assert "absent" not in Path(path).parts, "repository must not inspect evidence paths"
            return method(path, *args, **kwargs)
        return checked

    for name in (
        "exists", "is_file", "is_dir", "stat", "open", "read_text", "read_bytes",
        "iterdir", "glob", "rglob",
    ):
        monkeypatch.setattr(Path, name, forbid_evidence_access(getattr(Path, name)))
    with closing(api.ExperimentRepository(path)) as repository:
        for record in before:
            stored = repository.add(record)
            assert stored.record == record
            assert (stored.status, stored.invalidation_reason) == ("active", None)
    assert path.is_file()
    assert _sql(path, "PRAGMA user_version") == [(1,)]
    assert _sql(path, "SELECT count(*) FROM experiment_records") == [(3,)]
    with closing(api.ExperimentRepository(path)) as reopened:
        for record in before:
            restored = reopened.get(_key(record)).record
            assert restored == record
            assert type(restored) is type(record)
            assert _key(restored) == _key(record)
        restored = reopened.get(_key(original)).record
        assert (restored.metric_results["r2"].best_epoch,
                restored.metric_results["racc"].best_epoch) == (3, 7)
        assert restored.metric_results["r2"].checkpoint_ref != restored.metric_results["racc"].checkpoint_ref
        missing = reopened.get(_key(unknown)).record
        assert missing.provenance == ExperimentProvenance(_key(unknown))
        assert check_identity_consistency(missing.provenance, missing.provenance).status == "unknown"
        assert all(metric.checkpoint_ref is None for metric in missing.metric_results.values())
    assert (original, unknown, other_namespace) == before


def test_repository_isolates_snapshots_from_input_and_read_result_mutations(tmp_path):
    api = _api()
    original = _record()
    expected = deepcopy(original)
    path = tmp_path / "snapshot.sqlite"
    with closing(api.ExperimentRepository(path)) as repository:
        added = repository.add(original)
        assert original == expected
        original.summary["configuration"]["feature_list"].append("caller-change")
        original.metric_results.clear()
        added.record.summary["configuration"]["seed"] = 999
        first = repository.get(_key(expected))
        assert first.record == expected
        first.record.summary["configuration"]["feature_list"].clear()
        first.record.metric_results["r2"] = replace(
            first.record.metric_results["r2"], checkpoint_ref="caller-replacement",
        )
        assert repository.get(_key(expected)).record == expected
        assert repository.list_executions(_key(expected).experiment)[0].record == expected
    with closing(api.ExperimentRepository(path)) as reopened:
        assert reopened.get(_key(expected)).record == expected


def test_repository_duplicate_import_uses_structural_evidence_and_preserves_lifecycle(tmp_path):
    api = _api()
    path = tmp_path / "duplicate.sqlite"
    original = _record()
    same = replace(
        deepcopy(original), summary=_reordered(original.summary),
        metric_results=dict(reversed(tuple(original.metric_results.items()))),
    )
    assert same == original and same is not original
    with closing(api.ExperimentRepository(path)) as repository:
        repository.add(original)
        assert repository.add(same).record == original
        repository.invalidate(_key(original), reason="withdrawn evidence")
    # Independently reformat stored JSON: textual identity is not the contract.
    payload = _sql(path, "SELECT record_json FROM experiment_records")[0][0]
    _sql(path, "UPDATE experiment_records SET record_json = ?", (
        json.dumps(_reordered(json.loads(payload)), indent=4),
    ))
    with closing(api.ExperimentRepository(path)) as reopened:
        duplicate = reopened.add(same)
        assert duplicate.record == original
        assert (duplicate.status, duplicate.invalidation_reason) == ("invalidated", "withdrawn evidence")
        assert reopened.list_executions(_key(original).experiment) == ()
        assert len(reopened.list_executions(_key(original).experiment, include_invalidated=True)) == 1
    assert _sql(path, "SELECT count(*) FROM experiment_records") == [(1,)]


def test_repository_rejects_same_execution_changed_evidence_without_overwrite(tmp_path):
    api = _api()
    original = _record()
    metric_change = deepcopy(original)
    metric_change.summary["validation_metrics"]["r2"]["best_value"] = 0.99
    metric_change.metric_results["r2"] = replace(metric_change.metric_results["r2"], value=0.99)
    artifact_change = deepcopy(original)
    artifact_change.metric_results["r2"] = replace(
        artifact_change.metric_results["r2"], checkpoint_ref="replacement-checkpoint",
    )
    summary_change = deepcopy(original)
    summary_change.summary["configuration"]["feature_list"].reverse()
    scalar_change = deepcopy(original)
    scalar_change.summary["configuration"]["use_modality_token_fusion"] = 0
    provenance_change = _with_provenance(
        original, replace(original.provenance, dataset_version="dataset:v2"),
    )
    path = tmp_path / "conflicts.sqlite"
    with closing(api.ExperimentRepository(path)) as repository:
        repository.add(original)
        before = _database_snapshot(path)
        for changed in (
            metric_change, artifact_change, summary_change, scalar_change, provenance_change,
        ):
            with pytest.raises(api.RepositoryConflictError) as error:
                repository.add(changed)
            assert error.value.reason_codes == ("execution_snapshot_mismatch",)
            assert error.value.conflicting_execution == _key(original)
            assert repository.get(_key(original)).record == original
            assert _database_snapshot(path) == before


def test_repository_checks_all_retained_executions_using_m11_conflict_semantics(tmp_path):
    api = _api()
    known = _record("a-known")
    repeated = _record("b-repeated", seed=108)
    unknown = _record("z-unknown")
    unknown = _with_provenance(unknown, replace(unknown.provenance, dataset_version=None))
    path = tmp_path / "definition-conflicts.sqlite"
    with closing(api.ExperimentRepository(path)) as repository:
        for record in (known, repeated, unknown):
            repository.add(record)
        assert _key(known).experiment == _key(repeated).experiment
        assert repository.get(_key(repeated)).record.provenance.realized_seed == 108
        assert repository.get(_key(unknown)).record.provenance.dataset_version is None
        assert check_identity_consistency(known.provenance, unknown.provenance).status == "unknown"
        repository.invalidate(_key(known), reason="retained for evidence review")
        # The latest row has missing data; the earliest conflicting row is invalidated.
        fields = (
            "configuration_ref", "task", "target", "model_ref", "dataset_version",
            "split_id", "source_revision", "initialization_checkpoint_ref",
        )
        for changes in (
            *({name: "different"} for name in fields),
            {"configuration_ref": None, "dataset_version": "dataset:v2"},
        ):
            candidate = _record("candidate")
            candidate = _with_provenance(candidate, replace(candidate.provenance, **changes))
            expected = check_identity_consistency(known.provenance, candidate.provenance)
            assert expected.status == "conflict"
            with pytest.raises(api.RepositoryConflictError) as error:
                repository.add(candidate)
            assert error.value.conflicting_execution == _key(known)
            assert error.value.reason_codes == expected.reason_codes
            with pytest.raises(KeyError):
                repository.get(_key(candidate))
        assert tuple(_key(item.record).execution_id for item in repository.list_executions(
            _key(known).experiment, include_invalidated=True,
        )) == ("a-known", "b-repeated", "z-unknown")


def test_repository_invalidation_preserves_evidence_and_excludes_active_listing(tmp_path):
    api = _api()
    first, sibling = _record("a"), _record("b", seed=108)
    path = tmp_path / "lifecycle.sqlite"
    with closing(api.ExperimentRepository(path)) as repository:
        repository.add(first)
        repository.add(sibling)
        invalidated = repository.invalidate(_key(first), reason="artifact replaced externally")
        assert invalidated.record == first
        assert (invalidated.status, invalidated.invalidation_reason) == (
            "invalidated", "artifact replaced externally",
        )
        repository.invalidate(_key(first), reason="artifact replaced externally")
        assert tuple(item.record for item in repository.list_executions(_key(first).experiment)) == (sibling,)
        assert tuple(item.record for item in repository.list_executions(
            _key(first).experiment, include_invalidated=True,
        )) == (first, sibling)
        assert repository.add(deepcopy(first)).status == "invalidated"
    with closing(api.ExperimentRepository(path)) as reopened:
        stored = reopened.get(_key(first))
        assert stored.record == first
        assert (stored.status, stored.invalidation_reason) == (
            "invalidated", "artifact replaced externally",
        )
        assert reopened.get(_key(sibling)).status == "active"


def test_repository_hard_delete_removes_only_target_record_not_artifacts(tmp_path):
    api = _api()
    experiment_dir = tmp_path / "training-output"
    experiment_dir.mkdir()
    artifact = experiment_dir / "checkpoint.pt"
    artifact.write_bytes(b"caller-owned-checkpoint")
    original = replace(_record("a"), experiment_dir=str(experiment_dir))
    original.metric_results["r2"] = replace(
        original.metric_results["r2"], checkpoint_ref=str(artifact),
    )
    sibling = _record("b", seed=108)
    path = tmp_path / "delete.sqlite"
    with closing(api.ExperimentRepository(path)) as repository:
        repository.add(original)
        repository.add(sibling)
        assert repository.delete(_key(original)) is True
        assert repository.delete(_key(original)) is False
        with pytest.raises(KeyError):
            repository.get(_key(original))
        with pytest.raises(KeyError):
            repository.invalidate(_key(original), reason="already absent")
        assert repository.get(_key(sibling)).record == sibling
        assert tuple(item.record for item in repository.list_executions(
            _key(original).experiment, include_invalidated=True,
        )) == (sibling,)
    assert experiment_dir.is_dir()
    assert artifact.read_bytes() == b"caller-owned-checkpoint"
    with closing(api.ExperimentRepository(path)) as reopened:
        with pytest.raises(KeyError):
            reopened.get(_key(original))
        assert reopened.get(_key(sibling)).record == sibling


def test_repository_rejects_future_schema_and_record_versions_without_migration(tmp_path):
    api = _api()
    path = tmp_path / "versions.sqlite"
    original = _record()
    with closing(api.ExperimentRepository(path)) as repository:
        repository.add(original)
    assert _sql(path, "PRAGMA user_version") == [(1,)]
    assert _sql(path, "SELECT record_encoding_version FROM experiment_records") == [(1,)]
    _sql(path, "PRAGMA user_version = 99")
    before = _database_snapshot(path)
    with pytest.raises(api.UnsupportedRepositoryVersionError):
        with closing(api.ExperimentRepository(path)):
            pass
    assert _database_snapshot(path) == before

    _sql(path, "PRAGMA user_version = 1")
    _sql(path, "UPDATE experiment_records SET record_encoding_version = 99")
    before = _database_snapshot(path)
    with pytest.raises(api.UnsupportedRepositoryVersionError):
        with closing(api.ExperimentRepository(path)) as repository:
            repository.get(_key(original))
    assert _database_snapshot(path) == before
    _sql(path, "UPDATE experiment_records SET record_encoding_version = 1")
    with closing(api.ExperimentRepository(path)) as repository:
        assert repository.get(_key(original)).record == original


def test_repository_reports_corrupt_storage_without_skipping_repair_or_reset(tmp_path):
    api = _api()
    original = _record()
    wrong_key = asdict(original)
    wrong_key["provenance"]["execution"]["experiment"]["namespace"] = "different"
    wrong_metric = asdict(original)
    wrong_metric["metric_results"]["r2"]["provenance"]["execution"]["execution_id"] = "other"
    wrong_value = asdict(original)
    wrong_value["metric_results"]["r2"]["value"] = 0.01
    missing_provenance = asdict(original)
    del missing_provenance["provenance"]
    payloads = ["{not-json", *(json.dumps(value) for value in (
        wrong_key, wrong_metric, wrong_value, missing_provenance,
    ))]
    for index, payload in enumerate(payloads):
        path = tmp_path / f"corrupt-{index}.sqlite"
        with closing(api.ExperimentRepository(path)) as repository:
            repository.add(original)
        _sql(path, "UPDATE experiment_records SET record_json = ?", (payload,))
        before = _database_snapshot(path)
        for operation in ("get", "list"):
            with pytest.raises(api.RepositoryCorruptionError) as error:
                with closing(api.ExperimentRepository(path)) as repository:
                    if operation == "get":
                        repository.get(_key(original))
                    else:
                        repository.list_executions(_key(original).experiment)
            assert not isinstance(error.value, api.UnsupportedRepositoryVersionError)
            assert _database_snapshot(path) == before
    path = tmp_path / "not-a-database.sqlite"
    content = b"not SQLite storage\x00do not overwrite"
    path.write_bytes(content)
    with pytest.raises(api.RepositoryCorruptionError):
        with closing(api.ExperimentRepository(path)):
            pass
    assert path.read_bytes() == content


def test_repository_writes_roll_back_and_two_connections_cannot_bypass_conflicts(tmp_path):
    api = _api()
    path = tmp_path / "transactions.sqlite"
    original = _record("a")
    with closing(api.ExperimentRepository(path)) as first, closing(api.ExperimentRepository(path)) as second:
        assert second.list_executions(_key(original).experiment) == ()
        first.add(original)
        same_key_changed = replace(original, experiment_name="changed evidence")
        with pytest.raises(api.RepositoryConflictError):
            second.add(same_key_changed)
        different_execution = _record("b")
        conflict = _with_provenance(
            different_execution, replace(different_execution.provenance, dataset_version="dataset:v2"),
        )
        with pytest.raises(api.RepositoryConflictError):
            second.add(conflict)
        assert second.get(_key(original)).record == original
        with pytest.raises(KeyError):
            first.get(_key(conflict))

        invalid = deepcopy(different_execution)
        invalid.metric_results["r2"] = replace(invalid.metric_results["r2"], provenance=original.provenance)
        before = _database_snapshot(path)
        with pytest.raises(api.RepositoryError):
            second.add(invalid)
        assert _database_snapshot(path) == before

        # RAISE(FAIL) leaves the inserted row in the transaction unless the
        # repository rolls it back; this checks a genuine mid-write failure.
        _sql(path, """CREATE TRIGGER injected_write_failure
            AFTER INSERT ON experiment_records
            WHEN NEW.execution_id = 'b'
            BEGIN SELECT RAISE(FAIL, 'injected storage failure'); END""")
        before = _database_snapshot(path)
        with pytest.raises(api.RepositoryError):
            first.add(different_execution)
        assert _database_snapshot(path) == before
        with pytest.raises(KeyError):
            second.get(_key(different_execution))
        _sql(path, "DROP TRIGGER injected_write_failure")
        second.add(different_execution)
        assert first.get(_key(different_execution)).record == different_execution
    with closing(api.ExperimentRepository(path)) as reopened:
        assert tuple(item.record for item in reopened.list_executions(
            _key(original).experiment, include_invalidated=True,
        )) == (original, different_execution)
    assert _sql(path, "SELECT count(*) FROM experiment_records") == [(2,)]

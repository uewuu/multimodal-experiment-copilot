"""Local SQLite storage for explicit M11 evidence snapshots.

Schema and record encoding are independently versioned. Lifecycle metadata is
not evidence, and artifact references are never resolved or accessed here.
"""

from contextlib import contextmanager
from dataclasses import dataclass, fields
import json
import math
import sqlite3

from experiment_identity import (
    ExecutionIdentity,
    ExperimentIdentity,
    ExperimentProvenance,
    MetricResultProvenance,
    check_identity_consistency,
)
from experiment_record import ExperimentRecord


class RepositoryError(Exception):
    """Invalid repository input or a storage operation that could not complete."""


class RepositoryConflictError(RepositoryError):
    """Retained evidence prevents accepting this snapshot."""

    def __init__(self, reason_codes: tuple[str, ...], conflicting_execution: ExecutionIdentity):
        self.reason_codes = reason_codes
        self.conflicting_execution = conflicting_execution
        super().__init__(f"Conflicting execution: {', '.join(reason_codes)}")


class UnsupportedRepositoryVersionError(RepositoryError):
    """The database schema or record encoding is not supported."""


class RepositoryCorruptionError(RepositoryError):
    """Persisted storage cannot be reconstructed as valid evidence."""


@dataclass(frozen=True)
class _StoredRecord:
    record: ExperimentRecord
    status: str
    invalidation_reason: str | None


_SCHEMA_VERSION = 1
_RECORD_ENCODING_VERSION = 1
_COLUMNS = (
    "namespace, experiment_id, execution_id, record_encoding_version, "
    "record_json, status, invalidation_reason"
)
_KEY_WHERE = "namespace = ? AND experiment_id = ? AND execution_id = ?"


def _check_json_value(value):
    """Accept only lossless JSON values, without coercing keys or containers."""
    if value is None or type(value) in (str, bool, int):
        return
    if type(value) is float and math.isfinite(value):
        return
    if type(value) is list:
        for item in value:
            _check_json_value(item)
        return
    if type(value) is dict and all(type(key) is str for key in value):
        for item in value.values():
            _check_json_value(item)
        return
    raise ValueError("Evidence must contain only finite JSON values with string keys")


def _canonical_json(value):
    _check_json_value(value)
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("Duplicate JSON object key")
        result[key] = value
    return result


def _object_values(value, model):
    if type(value) is not model:
        raise ValueError(f"Expected {model.__name__}")
    return {field.name: getattr(value, field.name) for field in fields(model)}


def _require_fields(value, model):
    if type(value) is not dict or set(value) != {field.name for field in fields(model)}:
        raise ValueError(f"Invalid {model.__name__} fields")


def _require_text(value):
    if type(value) is not str:
        raise ValueError("Expected a string")


def _experiment_key(experiment):
    _object_values(experiment, ExperimentIdentity)
    _require_text(experiment.namespace)
    _require_text(experiment.experiment_id)
    return experiment.namespace, experiment.experiment_id


def _execution_key(execution):
    _object_values(execution, ExecutionIdentity)
    _require_text(execution.execution_id)
    return (*_experiment_key(execution.experiment), execution.execution_id)


def _provenance_payload(provenance):
    payload = _object_values(provenance, ExperimentProvenance)
    execution = _object_values(provenance.execution, ExecutionIdentity)
    execution["experiment"] = _object_values(provenance.execution.experiment, ExperimentIdentity)
    payload["execution"] = execution
    return payload


def _record_payload(record):
    payload = _object_values(record, ExperimentRecord)
    payload["provenance"] = _provenance_payload(record.provenance)
    if type(record.metric_results) is not dict:
        raise ValueError("Expected a metric result mapping")
    metrics = {}
    for name, metric in record.metric_results.items():
        _require_text(name)
        metrics[name] = _object_values(metric, MetricResultProvenance)
        metrics[name]["provenance"] = _provenance_payload(metric.provenance)
    payload["metric_results"] = metrics
    return payload


def _decode_provenance(payload):
    _require_fields(payload, ExperimentProvenance)
    execution = payload["execution"]
    _require_fields(execution, ExecutionIdentity)
    experiment = execution["experiment"]
    _require_fields(experiment, ExperimentIdentity)
    identity = ExecutionIdentity(ExperimentIdentity(**experiment), execution["execution_id"])
    _execution_key(identity)
    for name, value in payload.items():
        if name == "execution" or value is None:
            continue
        if name == "realized_seed":
            if type(value) is not int:
                raise ValueError("Realized seed must be an integer or None")
        else:
            _require_text(value)
    return ExperimentProvenance(**{**payload, "execution": identity})


def _decode_record(record_json):
    """Reconstruct only the frozen M11 models and validate evidence invariants."""
    _require_text(record_json)
    payload = json.loads(record_json, object_pairs_hook=_unique_object)
    _check_json_value(payload)
    _require_fields(payload, ExperimentRecord)
    _require_text(payload["experiment_name"])
    _require_text(payload["experiment_dir"])
    provenance = _decode_provenance(payload["provenance"])
    summary, metric_payloads = payload["summary"], payload["metric_results"]
    if type(summary) is not dict or type(metric_payloads) is not dict:
        raise ValueError("Summary and metric results must be objects")
    source_metrics = summary.get("validation_metrics")
    if type(source_metrics) is not dict or set(source_metrics) != set(metric_payloads):
        raise ValueError("Metric results must match the summary metrics")

    metrics = {}
    for name, metric in metric_payloads.items():
        _require_fields(metric, MetricResultProvenance)
        if metric["metric_name"] != name:
            raise ValueError("Metric result name mismatch")
        if type(metric["value"]) not in (int, float):
            raise ValueError("Metric value must be numeric")
        if metric["best_epoch"] is not None and type(metric["best_epoch"]) is not int:
            raise ValueError("Metric epoch must be an integer or None")
        for field in fields(MetricResultProvenance):
            if field.name not in ("value", "best_epoch", "provenance"):
                value = metric[field.name]
                if value is not None:
                    _require_text(value)
        metric_provenance = _decode_provenance(metric["provenance"])
        if metric_provenance != provenance:
            raise ValueError("Metric provenance must match record provenance")
        source = source_metrics[name]
        if type(source) is not dict or not {"best_value", "best_epoch"} <= source.keys():
            raise ValueError("Invalid summary metric structure")
        if "metric_name" in source and source["metric_name"] != name:
            raise ValueError("Summary metric name mismatch")
        for field, source_field in (("value", "best_value"), ("best_epoch", "best_epoch")):
            if _canonical_json(metric[field]) != _canonical_json(source[source_field]):
                raise ValueError("Metric result must match its summary value and epoch")
        metrics[name] = MetricResultProvenance(**{**metric, "provenance": metric_provenance})
    return ExperimentRecord(**{**payload, "provenance": provenance, "metric_results": metrics})


def _storage_error(error):
    code = getattr(error, "sqlite_errorcode", 0) & 0xFF
    if code in (sqlite3.SQLITE_CORRUPT, sqlite3.SQLITE_NOTADB):
        return RepositoryCorruptionError("Invalid SQLite repository storage")
    return RepositoryError(f"SQLite repository operation failed: {error}")


class ExperimentRepository:
    """One owned connection to a host-supplied SQLite path.

    Writes reserve the SQLite writer before checking retained evidence. There
    is no connection sharing, evidence cache, artifact I/O, migration or repair.
    """

    def __init__(self, database_path):
        try:
            self._connection = sqlite3.connect(database_path, isolation_level=None)
        except sqlite3.Error as error:
            raise _storage_error(error) from error
        self._connection.row_factory = sqlite3.Row
        try:
            with self._transaction(write=True, validate=False):
                version = self._connection.execute("PRAGMA user_version").fetchone()[0]
                objects = self._connection.execute("SELECT name FROM sqlite_schema").fetchall()
                if version == 0 and not objects:
                    self._connection.execute("""
                        CREATE TABLE experiment_records (
                            namespace TEXT NOT NULL,
                            experiment_id TEXT NOT NULL,
                            execution_id TEXT NOT NULL,
                            record_encoding_version INTEGER NOT NULL,
                            record_json TEXT NOT NULL,
                            status TEXT NOT NULL CHECK (status IN ('active', 'invalidated')),
                            invalidation_reason TEXT,
                            PRIMARY KEY (namespace, experiment_id, execution_id),
                            CHECK ((status = 'active' AND invalidation_reason IS NULL)
                                OR (status = 'invalidated' AND invalidation_reason IS NOT NULL))
                        )
                    """)
                    self._connection.execute("PRAGMA user_version = 1")
                self._validate_schema()
        except BaseException:
            self._connection.close()
            raise

    @contextmanager
    def _transaction(self, *, write=False, validate=True):
        try:
            self._connection.execute("BEGIN IMMEDIATE" if write else "BEGIN")
            try:
                if validate:
                    self._validate_schema()
                yield
                self._connection.commit()
            except BaseException:
                self._connection.rollback()
                raise
        except sqlite3.Error as error:
            raise _storage_error(error) from error

    def _validate_schema(self):
        version = self._connection.execute("PRAGMA user_version").fetchone()[0]
        if version != _SCHEMA_VERSION:
            raise UnsupportedRepositoryVersionError(f"Unsupported database schema version: {version}")
        columns = self._connection.execute("PRAGMA table_info(experiment_records)").fetchall()
        shape = tuple((row["name"], row["type"], row["notnull"], row["pk"]) for row in columns)
        if shape != (
            ("namespace", "TEXT", 1, 1), ("experiment_id", "TEXT", 1, 2),
            ("execution_id", "TEXT", 1, 3), ("record_encoding_version", "INTEGER", 1, 0),
            ("record_json", "TEXT", 1, 0), ("status", "TEXT", 1, 0),
            ("invalidation_reason", "TEXT", 0, 0),
        ):
            raise RepositoryCorruptionError("Invalid v1 repository schema")

    def _decode_row(self, row):
        if row["record_encoding_version"] != _RECORD_ENCODING_VERSION:
            raise UnsupportedRepositoryVersionError(
                f"Unsupported record encoding version: {row['record_encoding_version']}"
            )
        try:
            record = _decode_record(row["record_json"])
            if _execution_key(record.provenance.execution) != tuple(row[name] for name in (
                "namespace", "experiment_id", "execution_id",
            )):
                raise ValueError("Row key and evidence execution differ")
            status, reason = row["status"], row["invalidation_reason"]
            if not (
                (status == "active" and reason is None)
                or (status == "invalidated" and type(reason) is str)
            ):
                raise ValueError("Invalid lifecycle state")
            return _StoredRecord(record, status, reason)
        except (ValueError, TypeError, KeyError, RecursionError) as error:
            raise RepositoryCorruptionError("Invalid stored experiment evidence") from error

    def _find(self, execution):
        row = self._connection.execute(
            f"SELECT {_COLUMNS} FROM experiment_records WHERE {_KEY_WHERE}",
            _execution_key(execution),
        ).fetchone()
        return None if row is None else self._decode_row(row)

    def _retained(self, experiment):
        rows = self._connection.execute(
            f"SELECT {_COLUMNS} FROM experiment_records "
            "WHERE namespace = ? AND experiment_id = ? ORDER BY execution_id COLLATE BINARY",
            _experiment_key(experiment),
        ).fetchall()
        return tuple(self._decode_row(row) for row in rows)

    def add(self, record: ExperimentRecord) -> _StoredRecord:
        """Snapshot evidence once; reject replacement and M11 definition conflicts."""
        try:
            encoded = _canonical_json(_record_payload(record))
            snapshot = _decode_record(encoded)
        except (ValueError, TypeError, KeyError, RecursionError) as error:
            raise RepositoryError("Invalid experiment evidence snapshot") from error
        execution = snapshot.provenance.execution
        with self._transaction(write=True):
            retained = self._retained(execution.experiment)
            for stored in retained:
                if stored.record.provenance.execution == execution:
                    if _canonical_json(_record_payload(stored.record)) != encoded:
                        raise RepositoryConflictError(("execution_snapshot_mismatch",), execution)
                    return stored
            for stored in retained:
                assessment = check_identity_consistency(stored.record.provenance, snapshot.provenance)
                if assessment.status == "conflict":
                    raise RepositoryConflictError(
                        assessment.reason_codes, stored.record.provenance.execution,
                    )
            self._connection.execute(
                f"INSERT INTO experiment_records ({_COLUMNS}) VALUES (?, ?, ?, ?, ?, ?, ?)",
                (*_execution_key(execution), _RECORD_ENCODING_VERSION, encoded, "active", None),
            )
            return self._find(execution)

    def get(self, execution: ExecutionIdentity) -> _StoredRecord:
        """Return independent evidence and lifecycle state, including invalidation."""
        with self._transaction():
            stored = self._find(execution)
            if stored is None:
                raise KeyError(execution)
            return stored

    def list_executions(
        self, experiment: ExperimentIdentity, *, include_invalidated: bool = False,
    ) -> tuple[_StoredRecord, ...]:
        """List executions in case-sensitive order; active means not invalidated."""
        with self._transaction():
            retained = self._retained(experiment)
            return tuple(item for item in retained if include_invalidated or item.status == "active")

    def invalidate(self, execution: ExecutionIdentity, *, reason: str) -> _StoredRecord:
        """Store explicit invalidation without changing any evidence fields."""
        if type(reason) is not str:
            raise RepositoryError("Invalidation reason must be a string")
        with self._transaction(write=True):
            if self._find(execution) is None:
                raise KeyError(execution)
            self._connection.execute(
                "UPDATE experiment_records SET status = 'invalidated', invalidation_reason = ? "
                f"WHERE {_KEY_WHERE}", (reason, *_execution_key(execution)),
            )
            return self._find(execution)

    def delete(self, execution: ExecutionIdentity) -> bool:
        """Delete only the database record; never touch referenced artifacts."""
        with self._transaction(write=True):
            if self._find(execution) is None:
                return False
            self._connection.execute(
                f"DELETE FROM experiment_records WHERE {_KEY_WHERE}", _execution_key(execution),
            )
            return True

    def close(self) -> None:
        """Release this instance's connection."""
        try:
            self._connection.close()
        except sqlite3.Error as error:
            raise _storage_error(error) from error

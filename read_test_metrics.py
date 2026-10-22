"""Bounded, single-record CSV evidence selected by host-provided MetricSpec.

This reader reports file contents, not verified checkpoint or dataset provenance.
Path authorization belongs to the caller; no tool registration or model access
is provided here.
"""

from collections.abc import Sequence
import csv
import io
import json
import math
from pathlib import Path

from metrics import MetricSpec


_FILE_LIMIT = 1024 * 1024
_FIELD_LIMIT = 16 * 1024
_OUTPUT_LIMIT = 16 * 1024


class _InvalidCSV(ValueError):
    """Internal format failure carrying only a fixed, safe issue code."""


def _validate_specs(metric_specs: Sequence[MetricSpec]) -> tuple[MetricSpec, ...]:
    if isinstance(metric_specs, (str, bytes)) or not isinstance(metric_specs, Sequence):
        raise TypeError("metric_specs must be a sequence of MetricSpec")
    if not 1 <= len(metric_specs) <= 32:
        raise ValueError("select between 1 and 32 test metrics")
    specs = tuple(metric_specs)
    names = set()
    for spec in specs:
        if not isinstance(spec, MetricSpec):
            raise TypeError("each test metric must be a MetricSpec")
        if len(spec.path) != 1:
            raise ValueError("test metric paths must contain exactly one CSV column")
        if spec.name in names:
            raise ValueError("duplicate test metric name")
        names.add(spec.name)
    return specs


def _result(source_file: str, status: str, *, code: str | None = None,
            metrics: dict | None = None) -> dict:
    result = {
        "schema_version": "test-results-v1",
        "status": status,
        "evidence_kind": "csv_reported",
        "source": {
            "file": source_file,
            "data_row": 1 if status == "available" else None,
            "selection": "exactly_one_data_row",
        },
        "metrics": {} if metrics is None else metrics,
        "issues": [] if code is None else [{"code": code}],
        "provenance": {
            "status": "unknown",
            "checkpoint_identity": None,
            "checkpoint_binding": "not_verified",
            "dataset_split_identity": None,
            "recomputed_from_predictions": False,
        },
    }
    if _output_size(result) > _OUTPUT_LIMIT:
        result["status"] = "invalid"
        result["metrics"] = {}
        result["source"]["data_row"] = None
        result["issues"] = [{"code": "output_too_large"}]
        if _output_size(result) > _OUTPUT_LIMIT:
            # An invalid host filename must not produce an oversized response.
            raise ValueError("test result source metadata exceeds output limit")
    return result


def _output_size(result: dict) -> int:
    return len(json.dumps(result, ensure_ascii=False, allow_nan=False,
                          separators=(",", ":")).encode("utf-8"))


def _check_fields(row: list[str]) -> None:
    if any(len(field.encode("utf-8")) > _FIELD_LIMIT for field in row):
        raise _InvalidCSV("field_too_large")


def _parse_csv(raw: bytes, specs: tuple[MetricSpec, ...]) -> dict:
    text = raw.decode("utf-8-sig")
    if not text:
        raise _InvalidCSV("empty_file")
    # Explicit dialect options, no sniffing and no global field_size_limit change.
    reader = csv.reader(io.StringIO(text, newline=""), delimiter=",",
                        quotechar='"', doublequote=True, escapechar=None,
                        skipinitialspace=False, quoting=csv.QUOTE_MINIMAL, strict=True)
    header = next(reader, None)
    if not header or any(not name or name != name.strip() for name in header):
        raise _InvalidCSV("invalid_header")
    if len(header) > 64:
        raise _InvalidCSV("too_many_columns")
    _check_fields(header)
    if len(set(header)) != len(header):
        raise _InvalidCSV("duplicate_header")
    row = next(reader, None)
    if row is None:
        raise _InvalidCSV("no_data_row")
    if len(row) != len(header):
        raise _InvalidCSV("row_width_mismatch")
    _check_fields(row)
    if next(reader, None) is not None:
        raise _InvalidCSV("multiple_rows_not_supported")

    columns = dict(zip(header, row))  # widths and unique names validated above
    metrics = {}
    for spec in specs:
        column = spec.path[0]
        if column not in columns:
            raise _InvalidCSV("missing_column")
        cell = columns[column].strip()
        if not cell:
            raise _InvalidCSV("empty_value")
        try:
            value = float(cell)
        except ValueError:
            raise _InvalidCSV("invalid_number") from None
        if not math.isfinite(value):
            raise _InvalidCSV("non_finite_number")
        metrics[spec.name] = {
            "value": value,
            "source_column": column,
            "direction": spec.direction,
            "display_name": spec.display_name,
            "precision": spec.precision,
        }
    return metrics


def read_test_metrics(csv_path: Path, metric_specs: Sequence[MetricSpec]) -> dict:
    """Read one UTF-8/BOM comma-separated test record, without rounding values.

    ``MetricSpec.path`` selects one exact column; names, directions and display
    metadata come from the host. At most 1 MiB is accepted. Reads are bounded
    to that budget plus one overflow-detection byte, independent of stat().

    Missing files and ordinary OSError failures return safe status envelopes.
    Format/value failures return invalid with no partial metrics. Invalid host
    arguments and unexpected/control exceptions propagate. The caller handles
    optional input (not_requested) and authorizes paths before calling here.
    """
    if not isinstance(csv_path, Path):
        raise TypeError("csv_path must be a Path")
    specs = _validate_specs(metric_specs)
    source_file = csv_path.name
    raw = bytearray()
    try:
        # Unbuffered binary I/O avoids text decoding and buffered read-ahead
        # before the byte budget has been enforced. Handle short reads too.
        with csv_path.open("rb", buffering=0) as file:
            while len(raw) < _FILE_LIMIT + 1:
                chunk = file.read(min(65536, _FILE_LIMIT + 1 - len(raw)))
                if not chunk:
                    break
                raw.extend(chunk)
    except FileNotFoundError:
        return _result(source_file, "missing", code="file_not_found")
    except OSError:
        return _result(source_file, "unavailable", code="read_error")

    if len(raw) > _FILE_LIMIT:
        return _result(source_file, "invalid", code="file_too_large")
    try:
        metrics = _parse_csv(bytes(raw), specs)
    except UnicodeDecodeError:
        return _result(source_file, "invalid", code="invalid_encoding")
    except csv.Error:
        return _result(source_file, "invalid", code="invalid_csv")
    except _InvalidCSV as error:
        return _result(source_file, "invalid", code=error.args[0])
    return _result(source_file, "available", metrics=metrics)

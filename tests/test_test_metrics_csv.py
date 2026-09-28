"""M16 RED: read_test_metrics.read_test_metrics(Path, MetricSpec sequence).

One header and exactly one data record, comma dialect, UTF-8/BOM. The source
file is the basename at reader level (composition rebases it to experiment).
data_row=1 is a data-record index, not a physical line or epoch. Errors have
issues=[{code: ...}], empty metrics, and safe fixed codes; no raw cells/errors.
Host spec errors propagate. No first/last/best row fallback or partial metrics.
"""

import importlib
import importlib.util
import io
import json
import builtins
import os
from pathlib import Path

import pytest

from metrics import MetricSpec


class OpenProbe:
    """Observe three Python open entry points, not arbitrary OS/extension I/O.

    Nested Path.open -> io.open is recorded/wrapped once. Saved originals still
    perform real I/O. A callback may deny access or inject an I/O exception.
    """

    def __init__(self, monkeypatch, *, before=None, wrap=None):
        self.attempts = []
        self.active = False
        for owner, name in ((Path, "Path.open"), (builtins, "builtins.open"),
                            (io, "io.open")):
            original = owner.open

            def opened(target, *args, _original=original, _name=name, **kwargs):
                if self.active:
                    return _original(target, *args, **kwargs)
                resolved = None if isinstance(target, int) else Path(target).resolve()
                self.attempts.append((resolved, _name, kwargs.get("mode", args[0] if args else "r")))
                self.active = True
                try:
                    if before is not None:
                        before(resolved)
                    handle = _original(target, *args, **kwargs)
                    return wrap(resolved, handle) if wrap is not None else handle
                finally:
                    self.active = False

            monkeypatch.setattr(owner, "open", opened)


class ReadBudgetProbe:
    """Count delivered bytes across common stream methods, including iteration.

    The adversarial fixture is ASCII with LF, so text UTF-8 size equals file
    bytes. This is not a measurement of buffering, syscalls, mmap or native I/O.
    No unknown stream method is silently forwarded. buffer/raw views, if used,
    share the counter. A seek or reopen cannot reset the caller-owned budget.
    """

    def __init__(self, handle, budget):
        self.handle, self.budget = handle, budget

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.handle.close()

    def __getattr__(self, name):
        if name in ("buffer", "raw"):
            return ReadBudgetProbe(getattr(self.handle, name), self.budget)
        if name in ("close", "closed", "fileno", "tell", "seek", "seekable",
                    "readable", "writable", "encoding", "errors", "name", "mode"):
            return getattr(self.handle, name)
        raise AssertionError(f"uncovered stream interface: {name}")

    def account(self, data):
        count = len(data.encode("utf-8") if isinstance(data, str) else data)
        self.budget[0] += count
        assert self.budget[0] <= self.budget[1], "CSV read budget exceeded"
        return data

    def read(self, size=-1):
        # Unbounded read is safe only when the open handle proves the remainder
        # fits. Path.stat is deliberately untrustworthy in the adversarial test.
        if size is None or size < 0:
            remaining = os.fstat(self.handle.fileno()).st_size - self.handle.tell()
            assert remaining <= self.budget[1] - self.budget[0], "unbounded CSV read"
        return self.account(self.handle.read(size))

    def read1(self, size=-1):
        return self.account(self.handle.read1(size))

    def readline(self, size=-1):
        return self.account(self.handle.readline(size))

    def readlines(self, hint=-1):
        lines = self.handle.readlines(hint)
        for line in lines:
            self.account(line)
        return lines

    def readinto(self, buffer):
        count = self.handle.readinto(buffer)
        if count is not None:
            self.account(memoryview(buffer).cast("B")[:count])
        return count

    def readinto1(self, buffer):
        count = self.handle.readinto1(buffer)
        if count is not None:
            self.account(memoryview(buffer).cast("B")[:count])
        return count

    def __iter__(self):
        return self

    def __next__(self):
        return self.account(next(self.handle))


def underreport_size(monkeypatch, path):
    original = Path.stat

    def underreport(self, *args, **kwargs):
        result = original(self, *args, **kwargs)
        if self == path:
            values = list(result)
            values[6] = 1
            return type(result)(values)
        return result

    monkeypatch.setattr(Path, "stat", underreport)


@pytest.mark.parametrize("entry", ["path", "builtins", "io"])
def test_open_probe_records_and_denies_each_entry_once(tmp_path, monkeypatch, entry):
    path = csv_file(tmp_path, "score\n1\n")
    probe = OpenProbe(monkeypatch)
    def open_target():
        return {"path": lambda: path.open("rb"),
                "builtins": lambda: builtins.open(path, "rb"),
                "io": lambda: io.open(path, "rb")}[entry]()
    with open_target() as handle:
        assert handle.read() == b"score\n1\n"
    assert probe.attempts == [(path.resolve(),
        {"path": "Path.open", "builtins": "builtins.open", "io": "io.open"}[entry], "rb")]
    with monkeypatch.context() as patch:
        error = PermissionError("probe sentinel")
        def deny(target):
            if target == path.resolve():
                raise error
        denied = OpenProbe(patch, before=deny)
        with pytest.raises(PermissionError) as caught:
            open_target()
        assert caught.value is error
        assert len(denied.attempts) == 1
    assert len(probe.attempts) == 1  # denied before reaching the saved open


@pytest.mark.parametrize("method", ["read", "read1", "readline", "readlines",
                                    "readinto", "readinto1", "iteration"])
def test_read_budget_probe_counts_safe_and_excess_reads(method):
    def consume(probe):
        if method in ("readinto", "readinto1"):
            return getattr(probe, method)(bytearray(4))
        if method == "iteration":
            return next(probe)
        return getattr(probe, method)(4)
    for limit in (4, 3):
        budget = [0, limit]
        probe = ReadBudgetProbe(io.BytesIO(b"abc\n"), budget)
        if limit == 4:
            consume(probe)
        else:
            with pytest.raises(AssertionError, match="budget exceeded"):
                consume(probe)
        assert budget[0] == 4


def test_read_budget_probe_survives_false_stat_and_allows_early_refusal(tmp_path, monkeypatch):
    path = csv_file(tmp_path, b"12345678")
    underreport_size(monkeypatch, path)
    budget = [0, 4]
    OpenProbe(monkeypatch, wrap=lambda target, handle:
              ReadBudgetProbe(handle, budget) if target == path.resolve() else handle)
    assert path.stat().st_size == 1
    with path.open("rb") as handle:
        assert os.fstat(handle.fileno()).st_size > 4  # trusted early refusal
    assert budget[0] == 0
    with builtins.open(path, "rb") as handle:
        assert handle.readinto(bytearray(4)) == 4
        with pytest.raises(AssertionError, match="budget exceeded"):
            handle.readinto(bytearray(1))


def api():
    name = "read_test_metrics"
    assert importlib.util.find_spec(name) is not None, "missing M16 capability: read_test_metrics"
    fn = getattr(importlib.import_module(name), "read_test_metrics", None)
    assert callable(fn), "missing M16 capability: read_test_metrics.read_test_metrics"
    return fn


def spec(name="score", column="score", direction="maximize"):
    return MetricSpec(name, (column,), direction, name, 6)


def csv_file(tmp_path, content, filename="heldout.csv"):
    path = tmp_path / filename
    path.write_bytes(content if isinstance(content, bytes) else content.encode("utf-8"))
    return path


def assert_provenance(result):
    assert result["schema_version"] == "test-results-v1"
    assert result["evidence_kind"] == "csv_reported"
    provenance = result["provenance"]
    assert provenance["status"] == "unknown"
    assert provenance["checkpoint_identity"] is None
    assert provenance["checkpoint_binding"] == "not_verified"
    assert provenance["dataset_split_identity"] is None
    assert provenance["recomputed_from_predictions"] is False
    assert "best_epoch" not in json.dumps(result)


@pytest.mark.parametrize("bom", [False, True], ids=["utf8", "utf8-bom"])
def test_single_row_custom_columns_direction_and_unrounded_values(tmp_path, bom):
    read = api()
    raw = "误差,score,comment\n 0.123456789 ,0.91,DO_NOT_EXPOSE\n".encode("utf-8")
    path = csv_file(tmp_path, (b"\xef\xbb\xbf" if bom else b"") + raw)
    result = read(path, [spec("loss", "误差", "minimize"), spec(direction="minimize")])
    assert result["status"] == "available" and result["issues"] == []
    assert result["source"] == {"file": "heldout.csv", "data_row": 1,
                                 "selection": "exactly_one_data_row"}
    assert list(result["metrics"]) == ["loss", "score"]
    assert result["metrics"]["loss"]["value"] == 0.123456789
    assert result["metrics"]["loss"]["source_column"] == "误差"
    assert result["metrics"]["score"]["direction"] == "minimize"
    assert "DO_NOT_EXPOSE" not in json.dumps(result)
    assert_provenance(result)


@pytest.mark.parametrize("raw,code", [
    (b"", "empty_file"), (b"score\n", "no_data_row"),
    (b"score\n1\n2\n", "multiple_rows_not_supported"),
    (b"score,score\n1,2\n", "duplicate_header"),
    (b"score,\n1,2\n", "invalid_header"),
    (b" score\n1\n", "invalid_header"),
    (b"score,x\n1\n", "row_width_mismatch"),
    (b"score\n1,2\n", "row_width_mismatch"),
    (b"other\n1\n", "missing_column"),
    (b'score,x\n,2\n', "empty_value"),
    (b'score\n"  "\n', "empty_value"),
    (b"score\nSECRET-invalid-cell\n", "invalid_number"),
    (b"score\nNaN\n", "non_finite_number"),
    (b"score\nInfinity\n", "non_finite_number"),
    (b"score\n-Infinity\n", "non_finite_number"),
    (b"score\n1e999\n", "non_finite_number"),
    (b"score\n\xff\n", "invalid_encoding"),
    (b'score\n"unterminated', "invalid_csv"),
    (b"score;other\n1;2\n", "missing_column"),
], ids=["empty", "header-only", "multirow", "duplicate-header", "empty-header",
        "header-whitespace", "short-row", "long-row", "missing-column",
        "empty-cell", "blank-cell", "text", "nan", "inf", "negative-inf",
        "overflow", "encoding", "quotes", "no-dialect-sniffing"])
def test_invalid_csv_has_no_partial_metrics_or_raw_error(tmp_path, raw, code):
    read = api()
    result = read(csv_file(tmp_path, raw), [spec()])
    assert result["status"] == "invalid" and result["metrics"] == {}
    assert code in [i["code"] for i in result["issues"]]
    assert "SECRET-invalid-cell" not in json.dumps(result)


def test_one_bad_selected_metric_discards_all_test_values(tmp_path):
    read = api()
    result = read(csv_file(tmp_path, "score,loss\n0.8,nope\n"), [spec(), spec("loss", "loss")])
    assert result["status"] == "invalid" and result["metrics"] == {}


def test_missing_file_is_not_an_available_result(tmp_path):
    read = api()
    result = read(tmp_path / "missing.csv", [spec()])
    assert result["status"] == "missing" and result["metrics"] == {}
    assert_provenance(result)


@pytest.mark.parametrize("error", [PermissionError("SECRET-io"), OSError("SECRET-io")],
                         ids=["permission", "io"])
def test_known_io_failure_is_sanitized_unavailable(tmp_path, monkeypatch, error):
    read = api()
    path = csv_file(tmp_path, "score\n1\n")
    def fail(target):
        if target == path.resolve():
            raise error
    probe = OpenProbe(monkeypatch, before=fail)
    result = read(path, [spec()])
    assert any(target == path.resolve() for target, _, _ in probe.attempts)
    assert result["status"] == "unavailable" and result["metrics"] == {}
    assert "SECRET-io" not in json.dumps(result)


@pytest.mark.parametrize("specs", [[], [object()], [spec(), spec()],
    [MetricSpec("nested", ("a", "b"), "maximize", "Nested")],
    [spec(str(i), str(i)) for i in range(33)]],
    ids=["empty", "wrong-type", "duplicate-name", "nested-column", "metrics-33"])
def test_invalid_host_specs_raise(tmp_path, specs):
    read = api()
    with pytest.raises((TypeError, ValueError)):
        read(csv_file(tmp_path, "score\n1\n"), specs)


@pytest.mark.parametrize("columns,expected", [(64, "available"), (65, "invalid")])
def test_column_limit(tmp_path, columns, expected):
    read = api()
    names = [f"m{i}" for i in range(columns)]
    content = ",".join(names) + "\n" + ",".join(["1"] * columns) + "\n"
    result = read(csv_file(tmp_path, content), [spec(n, n) for n in names[:32]])
    assert result["status"] == expected
    if expected == "available":
        assert len(result["metrics"]) == 32
    else:
        assert result["metrics"] == {}


@pytest.mark.parametrize("size,expected", [(16384, "available"), (16385, "invalid")])
def test_field_utf8_limit_includes_unselected_fields(tmp_path, size, expected):
    read = api()
    text = "界" * (size // 3) + "x" * (size % 3)
    result = read(csv_file(tmp_path, "score,note\n1," + text + "\n"), [spec()])
    assert result["status"] == expected


def test_actual_read_limit_cannot_be_bypassed_by_small_stat(tmp_path, monkeypatch):
    read = api()
    path = csv_file(tmp_path, b"score\n" + b"x" * (1024 * 1024))
    underreport_size(monkeypatch, path)
    # Permit one overflow-detection byte, but never an unbounded read followed
    # by a size check. This is an I/O probe, not a replacement CSV parser.
    budget = [0, 1024 * 1024 + 1]
    OpenProbe(monkeypatch, wrap=lambda target, handle:
              ReadBudgetProbe(handle, budget) if target == path.resolve() else handle)
    result = read(path, [spec()])
    assert result["status"] == "invalid" and result["metrics"] == {}
    assert "file_too_large" in [i["code"] for i in result["issues"]]
    # A trustworthy handle size check may reject before any data read.
    assert budget[0] <= budget[1]


def test_exact_one_mib_csv_is_accepted_without_global_csv_settings_change(tmp_path):
    read = api()
    import csv
    limit_before = csv.field_size_limit()
    names = ["score"] + [f"note{i}" for i in range(63)]
    header = ",".join(names) + "\n"
    fields = ["1" + " " * 16383] + ["x" * 16384] * 62
    prefix = header + ",".join(fields) + ","
    remaining = 1024 * 1024 - len(prefix.encode()) - 1
    assert 0 < remaining <= 16384
    content = prefix + "x" * remaining + "\n"
    assert len(content.encode()) == 1024 * 1024
    result = read(csv_file(tmp_path, content), [spec()])
    assert result["status"] == "available"
    assert result["metrics"]["score"]["value"] == 1
    assert csv.field_size_limit() == limit_before


@pytest.mark.parametrize("error", [RuntimeError("programmer-error"), KeyboardInterrupt()],
                         ids=["programmer", "control"])
def test_unexpected_reader_errors_propagate(tmp_path, monkeypatch, error):
    read = api()
    path = csv_file(tmp_path, "score\n1\n")
    def fail(target):
        if target == path.resolve():
            raise error
    probe = OpenProbe(monkeypatch, before=fail)
    with pytest.raises(type(error)) as caught:
        read(path, [spec()])
    assert caught.value is error
    assert any(target == path.resolve() for target, _, _ in probe.attempts)


def test_test_output_limit_is_not_silent_truncation(tmp_path):
    read = api()
    specs = [MetricSpec(f"m{i}", ("score",), "maximize", "x" * 1024) for i in range(32)]
    result = read(csv_file(tmp_path, "score\n0.123456789\n"), specs)
    assert result["status"] == "invalid" and result["metrics"] == {}
    assert "output_too_large" in [i["code"] for i in result["issues"]]
    assert len(json.dumps(result, ensure_ascii=False, separators=(",", ":")).encode()) <= 16384

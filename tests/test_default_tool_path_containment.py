"""FCR-01B: real default calls, native links, calibrated content-open evidence.

RED-01..20 identify frozen scenario groups, not a promised node count.
Platform mechanisms are collected only on their native platform; no skips.
All data and clients are synthetic. Changes in the replacement tests happen
BEFORE the final read check, not in the explicitly excluded check/open race.
"""

import builtins
from contextlib import contextmanager
import copy
import ctypes
import inspect
import io
import json
import os
from pathlib import Path
import stat
import subprocess
import sys
from types import SimpleNamespace

import pytest

import compare_experiments as core
from copilot.session import CopilotSession
import llm_adapters.openai_tool_adapter as adapter
import read_config
import read_history
import read_metrics_config
import summarize_experiment
import tool_layer.experiment_tools as tools
from tool_layer import list_tools
from tool_layer.experiment_path_security import _build_experiment_path_policy


TOOLS = ("compare_experiments", "analyze_experiment")
OUTSIDE_VALUES = (0.314159, 0.271828)
DIRECTORY_MECHANISM = "windows-junction" if os.name == "nt" else "linux-symlink"


def _experiment(path, r2=0.11, racc=0.22):
    path.mkdir(parents=True)
    (path / "hparams.yaml").write_text("config:\n  batch_size: 73\n", encoding="utf-8")
    (path / "history.json").write_text(json.dumps({"valid": {"app": {
        "r2": [[0, r2]], "racc": [[0, racc]],
    }}}), encoding="utf-8")
    return path


def _metrics(path):
    path.write_text("metrics:\n  - name: r2\n    path: [valid, app, r2]\n"
                    "    direction: maximize\n    display_name: R2\n", encoding="utf-8")
    return path


def _dirlink(link, target):
    if os.name == "nt":
        result = subprocess.run(["cmd", "/c", "mklink", "/J", str(link), str(target)],
                                capture_output=True, text=True, check=False)
        assert result.returncode == 0, "required native junction capability unavailable"
        assert link.lstat().st_reparse_tag == stat.IO_REPARSE_TAG_MOUNT_POINT
    else:
        os.symlink(target, link, target_is_directory=True)
        assert link.is_symlink()
    assert link.samefile(target)
    assert link.resolve(strict=True) == target.resolve(strict=True)


def _filelink(link, target, *, relative=False):
    os.symlink(os.path.relpath(target, link.parent) if relative else target, link)
    assert link.is_symlink()
    assert link.samefile(target)


class _Client:
    def __init__(self, name, args):
        self.requests = []
        self.call = SimpleNamespace(id="containment-call", type="function",
                                    function=SimpleNamespace(name=name, arguments=json.dumps(args)))
        self.chat = SimpleNamespace(completions=self)

    def create(self, **kwargs):
        self.requests.append(copy.deepcopy(kwargs))
        message = SimpleNamespace(role="assistant", content=None, tool_calls=[self.call])
        if len(self.requests) > 1:
            message = SimpleNamespace(role="assistant", content="synthetic done", tool_calls=None)
        return SimpleNamespace(choices=[SimpleNamespace(message=message)])


@pytest.fixture(autouse=True)
def _no_network(monkeypatch):
    import socket

    def denied(*args, **kwargs):
        raise AssertionError("these tests must not use a real network provider")

    monkeypatch.setattr(socket, "create_connection", denied)
    monkeypatch.setattr(socket.socket, "connect", denied)


@pytest.fixture
def fs(tmp_path):
    # Canonical temp root avoids mistaking Windows TEMP short aliases for escape.
    base = tmp_path.resolve()
    host = base / "authorized"
    host.mkdir()
    outside = _experiment(base / "outside-experiment", *OUTSIDE_VALUES)
    return host, outside


class _ReadRecorder:
    def __init__(self, monkeypatch, outside):
        self.outside = outside.resolve()
        self.events = []
        self.all_events = []
        original_io, original_builtin = io.open, builtins.open

        def record(original, file, *args, **kwargs):
            mode = args[0] if args else kwargs.get("mode", "r")
            if not isinstance(file, int) and ("r" in mode or "+" in mode):
                target = Path(file).resolve()
                self.all_events.append(str(target))
                if target == self.outside or self.outside in target.parents:
                    self.events.append(str(target))
            return original(file, *args, **kwargs)

        monkeypatch.setattr(io, "open", lambda f, *a, **k: record(original_io, f, *a, **k))
        monkeypatch.setattr(builtins, "open", lambda f, *a, **k: record(original_builtin, f, *a, **k))
        assert (outside / "history.json").read_text(encoding="utf-8")
        assert len(self.events) == 1, "content recorder must detect a real calibration read"
        self.events.clear()
        self.all_events.clear()


def _call(host, *, tool="compare_experiments", selection=".", extra=None):
    field = "experiment_root" if tool == "compare_experiments" else "experiment_dir"
    client = _Client(tool, {field: selection, **(extra or {})})
    context = None if host is None else {"experiment_root": str(host)}
    session = CopilotSession(client, model="synthetic-no-network", experiment_context=context)
    return client, session


def _success(client, session):
    session.ask("Analyze only the authorized experiments.")
    assert len(client.requests) == 2 and len(session.history) == 1
    messages = [m for m in client.requests[-1]["messages"] if m["role"] == "tool"]
    assert len(messages) == 1
    return json.loads(messages[0]["content"])


def _failure(client, session, recorder, request, *, error_type=ValueError, private=False):
    error = None
    result = None
    try:
        result = session.ask("Analyze only the authorized experiments.")
    except Exception as caught:
        error = caught
    evidence = dict(error_type=None if error is None else type(error).__name__,
                    outside_content_opens=len(recorder.events),
                    provider_calls=len(client.requests), history_entries=len(session.history),
                    returned_success=result is not None)
    captured = json.dumps(client.requests, ensure_ascii=True)
    evidence["outside_metrics_propagated"] = any(str(v) in captured for v in OUTSIDE_VALUES)
    request.node.user_properties.append(("containment_evidence", evidence))
    assert recorder.events == [], evidence
    assert isinstance(error, error_type), evidence
    assert result is None and len(client.requests) == 1 and len(session.history) == 0, evidence
    assert not evidence["outside_metrics_propagated"], evidence
    if private:
        assert str(recorder.outside) not in str(error)
        assert "outside-experiment" not in str(error)


def test_red01_normal_exact_metrics_and_order(fs):
    host, _ = fs
    a, b = _experiment(host / "exp_a"), _experiment(host / "exp_b", 0.33, 0.44)
    expected = {"comparison_records": [
        {"experiment_name": b.name, "experiment_dir": str(b), "best_r2": 0.33,
         "best_r2_epoch": 0, "best_racc": 0.44, "best_racc_epoch": 0},
        {"experiment_name": a.name, "experiment_dir": str(a), "best_r2": 0.11,
         "best_r2_epoch": 0, "best_racc": 0.22, "best_racc_epoch": 0}],
        "descending": True, "experiment_counts": {"failed": 0, "successful": 2, "total": 2},
        "failed_experiments": [], "sort_by": "best_r2"}
    assert _success(*_call(host)) == expected
    assert _success(*_call(host)) == expected


def test_red02_direct_outside_root_stays_rejected(fs, monkeypatch, request):
    host, outside = fs
    _dirlink(host / "linked", outside)
    rec = _ReadRecorder(monkeypatch, outside)
    _failure(*_call(host, selection="linked"), rec, request)


# The collected node names explicitly identify actual platform mechanisms.
if os.name == "nt":
    def test_red03_windows_junction_scan_escape(fs, monkeypatch, request):
        host, outside = fs
        _dirlink(host / "linked", outside)
        rec = _ReadRecorder(monkeypatch, outside)
        _failure(*_call(host), rec, request, private=True)
else:
    def test_red04_linux_directory_symlink_scan_escape(fs, monkeypatch, request):
        host, outside = fs
        _dirlink(host / "linked", outside)
        rec = _ReadRecorder(monkeypatch, outside)
        _failure(*_call(host), rec, request, private=True)


@pytest.mark.parametrize("tool", TOOLS)
@pytest.mark.parametrize("name", ["hparams.yaml", "history.json"])
def test_red05_actual_external_file_symlink(fs, monkeypatch, request, tool, name):
    host, outside = fs
    exp = _experiment(host / "exp")
    (exp / name).unlink()
    _filelink(exp / name, outside / name)
    rec = _ReadRecorder(monkeypatch, outside)
    _failure(*_call(host, tool=tool, selection="." if tool == TOOLS[0] else "exp"),
             rec, request, private=True)


@pytest.mark.parametrize("kind", ["directory", "file"])
@pytest.mark.parametrize("relative", [False, True])
def test_red06_multihop_external_links(fs, monkeypatch, request, kind, relative):
    host, outside = fs
    if kind == "directory":
        # Actual directory symlinks here supplement native junction RED03.
        os.symlink(os.path.relpath(outside, host) if relative else outside,
                   host / "middle", target_is_directory=True)
        os.symlink("middle" if relative else host / "middle", host / "first", target_is_directory=True)
        assert (host / "first").samefile(outside)
    else:
        exp = _experiment(host / "exp")
        _filelink(host / "middle", outside / "history.json", relative=relative)
        (exp / "history.json").unlink()
        _filelink(exp / "history.json", host / "middle", relative=relative)
    rec = _ReadRecorder(monkeypatch, outside)
    _failure(*_call(host), rec, request, private=True)


@pytest.mark.parametrize("file_link", [False, True])
def test_red07_internal_links_use_host_root_not_selected_scan_root(fs, file_link):
    host, _ = fs
    scan = host / "scan"
    scan.mkdir()
    exp = _experiment(host / "sibling", 0.33, 0.44)
    _dirlink(scan / "alias", exp)
    if file_link:
        target = host / "authorized-history.json"
        (exp / "history.json").rename(target)
        _filelink(exp / "history.json", target)
        assert (scan / "alias" / "history.json").samefile(target)
    payload = _success(*_call(host, selection="scan"))
    assert payload["experiment_counts"] == {"failed": 0, "successful": 1, "total": 1}
    row = payload["comparison_records"][0]
    assert row["experiment_name"] == "alias" and row["experiment_dir"] == str(scan / "alias")
    assert row["best_r2"] == 0.33 and row["best_racc"] == 0.44


@pytest.mark.parametrize("kind", ["dangling-directory", "dangling-file", "cycle-directory", "cycle-file"])
def test_red08_broken_or_cyclic_links_fail_whole_call(fs, monkeypatch, request, kind):
    host, outside = fs
    if kind.endswith("file"):
        exp = _experiment(host / "exp")
        a = exp / "history.json"
        a.unlink()
    else:
        a = host / "linked"
    b = host / "missing-or-cycle"
    os.symlink(b, a, target_is_directory=kind.endswith("directory"))
    if kind.startswith("cycle"):
        os.symlink(a, b, target_is_directory=kind.endswith("directory"))
    rec = _ReadRecorder(monkeypatch, outside)
    _failure(*_call(host), rec, request, private=True)


@pytest.mark.parametrize("unsafe_first", [False, True])
def test_red09_mixed_batch_never_returns_partial_success(fs, monkeypatch, request, unsafe_first):
    host, outside = fs
    _experiment(host / ("z-safe" if unsafe_first else "a-safe"))
    _dirlink(host / ("a-unsafe" if unsafe_first else "z-unsafe"), outside)
    rec = _ReadRecorder(monkeypatch, outside)
    _failure(*_call(host), rec, request, private=True)


def test_red10_calibrated_reader_and_fake_provider_zero_propagation(fs, monkeypatch, request):
    host, outside = fs
    _dirlink(host / "linked", outside)
    rec = _ReadRecorder(monkeypatch, outside)
    _failure(*_call(host), rec, request)


@pytest.mark.parametrize("descending", [False, True])
@pytest.mark.parametrize("diagnostics", [False, True])
@pytest.mark.parametrize("configured", [False, True])
def test_red11_ordinary_options_and_ties_are_deterministic(fs, descending, diagnostics, configured):
    host, _ = fs
    _experiment(host / "a", 0.33, 0.44)
    _experiment(host / "b", 0.33, 0.44)
    kwargs = dict(descending=descending, include_diagnostics=diagnostics)
    if configured:
        kwargs["metrics_config"] = str(_metrics(host / "metrics.yaml"))
    # Real unscoped public core is a compatibility oracle, not a mocked result.
    expected = tools.compare_experiments(str(host), **kwargs)
    first = _success(*_call(host, extra=kwargs))
    assert first == expected == _success(*_call(host, extra=kwargs))
    assert [r["experiment_name"] for r in first["comparison_records"]] == ["a", "b"]
    assert first["experiment_counts"] == {"failed": 0, "successful": 2, "total": 2}


def test_red12_public_signature_registry_and_unscoped_readers(fs):
    host, outside = fs
    assert str(inspect.signature(tools.compare_experiments)) == (
        "(experiment_root: str, sort_by: str | None = None, descending: bool = True, "
        "*, metrics_config: str | None = None, include_diagnostics: bool = False) -> dict")
    assert str(inspect.signature(tools.analyze_experiment)) == (
        "(experiment_dir: str, *, metrics_config: str | None = None, include_diagnostics: bool = False) -> dict")
    definitions = list_tools()
    assert {t["function"]["name"] for t in definitions} == set(TOOLS)
    for t in definitions:
        properties = t["function"]["parameters"]["properties"]
        required = "experiment_root" if t["function"]["name"] == TOOLS[0] else "experiment_dir"
        assert set(properties) == ({required, "metrics_config", "include_diagnostics"}
                                   | ({"sort_by", "descending"} if required == "experiment_root" else set()))
        assert t["function"]["parameters"]["additionalProperties"] is False
    assert read_history.read_history(outside / "history.json")["valid"]["app"]["r2"][0][1] == OUTSIDE_VALUES[0]
    _dirlink(host / "linked", outside)
    assert tools.compare_experiments(str(host))["comparison_records"][0]["best_r2"] == OUTSIDE_VALUES[0]


@pytest.mark.parametrize("extra", ["authorized_root", "trusted_root", "policy", "disable_path_checks"])
def test_red12_model_json_cannot_inject_authority(fs, monkeypatch, request, extra):
    host, outside = fs
    rec = _ReadRecorder(monkeypatch, outside)
    _failure(*_call(host, extra={extra: str(outside)}), rec, request, error_type=(TypeError, ValueError))


def test_red13_component_prefix_is_not_authority(fs, monkeypatch, request):
    host, outside = fs
    sibling = host.with_name(host.name + "-extra")
    _experiment(sibling, *OUTSIDE_VALUES)
    _dirlink(host / "linked", sibling)
    rec = _ReadRecorder(monkeypatch, sibling)
    _failure(*_call(host), rec, request)


@pytest.mark.parametrize("target", [r"\\server\share\external", r"\\?\C:\external", "Z:\\external"])
def test_red13_derived_foreign_path_seam_rejected_without_network(fs, monkeypatch, request, target):
    host, outside = fs
    exp = _experiment(host / "exp")
    real = Path.resolve
    seen = []

    def resolve(path, *args, **kwargs):
        if path == exp:
            seen.append(str(path))
            return Path(target)
        return real(path, *args, **kwargs)

    rec = _ReadRecorder(monkeypatch, outside)
    monkeypatch.setattr(Path, "resolve", resolve)
    _failure(*_call(host), rec, request)
    assert seen, "derived resolution must actually be reached"


if os.name == "nt":
    def test_red13_windows_real_short_name_and_case_alias(fs):
        host, _ = fs
        _experiment(host / "experiment-with-long-name")
        kernel = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel.GetShortPathNameW.argtypes = [ctypes.c_wchar_p, ctypes.c_wchar_p, ctypes.c_uint]
        kernel.GetShortPathNameW.restype = ctypes.c_uint
        buf = ctypes.create_unicode_buffer(32768)
        n = kernel.GetShortPathNameW(str(host), buf, len(buf))
        assert 0 < n < len(buf), "required alias capability unavailable"
        alias = Path(buf.value)
        assert alias.samefile(host)
        assert _success(*_call(alias))["experiment_counts"]["successful"] == 1
        assert _success(*_call(Path(str(host).swapcase())))["experiment_counts"]["successful"] == 1
else:
    def test_red13_linux_distinct_case_sibling_is_outside(fs, monkeypatch, request):
        host, _ = fs
        other = _experiment(host.with_name(host.name.upper()), *OUTSIDE_VALUES)
        _dirlink(host / "linked", other)
        rec = _ReadRecorder(monkeypatch, other)
        _failure(*_call(host), rec, request)


@pytest.mark.parametrize("absolute", [False, True])
def test_red14_no_host_authority(fs, monkeypatch, request, absolute):
    host, outside = fs
    rec = _ReadRecorder(monkeypatch, outside)
    monkeypatch.chdir(host)
    _failure(*_call(None, selection=str(host) if absolute else "."), rec, request)


@pytest.mark.parametrize("name", ["hparams.yaml", "history.json"])
def test_red15_swap_after_scan_before_actual_read(fs, monkeypatch, request, name):
    host, outside = fs
    exp = _experiment(host / "exp")
    original = tools.find_experiment_dirs
    events = []

    def scan(root):
        found = original(root)
        assert found == [exp]
        (exp / name).unlink()
        _filelink(exp / name, outside / name)
        events.append("swapped-after-scan")
        return found

    monkeypatch.setattr(tools, "find_experiment_dirs", scan)
    rec = _ReadRecorder(monkeypatch, outside)
    _failure(*_call(host), rec, request, private=True)
    assert events == ["swapped-after-scan"]


@pytest.mark.parametrize("kind", ["missing-config", "missing-history", "directory-config", "directory-history", "nested", "ordinary-file", "parse-error", "empty"])
def test_red16_ordinary_exclusion_and_parse_errors_unchanged(fs, kind):
    host, _ = fs
    if kind != "empty":
        if kind == "nested":
            _experiment(host / "group" / "nested")
        elif kind == "ordinary-file":
            (host / "unrelated.txt").write_text("synthetic", encoding="utf-8")
        else:
            exp = _experiment(host / "exp")
            if kind == "parse-error":
                (exp / "history.json").write_text("not JSON", encoding="utf-8")
                _experiment(host / "valid")
            else:
                path = exp / ("hparams.yaml" if kind.endswith("config") else "history.json")
                path.unlink()
                if kind.startswith("directory"):
                    path.mkdir()
    expected = tools.compare_experiments(str(host))
    assert _success(*_call(host)) == expected
    assert expected["experiment_counts"] == ({"total": 2, "successful": 1, "failed": 1}
                                             if kind == "parse-error" else {"total": 0, "successful": 0, "failed": 0})


def test_red17_scope_restored_after_success_and_rejection(fs, monkeypatch, request):
    host, outside = fs
    _experiment(host / "valid")
    assert _success(*_call(host))["experiment_counts"]["successful"] == 1
    assert read_history.read_history(outside / "history.json")
    rec = _ReadRecorder(monkeypatch, outside)
    _failure(*_call(host, selection=str(outside)), rec, request)
    # Separate host permission may authorize precisely the formerly outside file.
    assert _success(*_call(outside, tool="analyze_experiment"))
    assert read_history.read_history(outside / "history.json")


def test_red17_nested_other_dispatch_masks_then_restores_scope(fs, monkeypatch, request):
    host, outside = fs
    _experiment(host / "valid")
    original = adapter.invoke_tool
    nested = []

    def invoke(name, args):
        if name == "synthetic_other":
            nested.append(read_history.read_history(outside / "history.json"))
            return {"ok": True}
        # Validation of synthetic dispatch is isolated; actual default dispatch
        # remains real. Existing private policy returns no descriptors for it.
        call = SimpleNamespace(id="nested", type="function", function=SimpleNamespace(
            name="synthetic_other", arguments="{}"))
        validate = adapter._validate_tool_call
        with monkeypatch.context() as patch:
            patch.setattr(adapter, "_validate_tool_call", lambda c, i: ("nested", "synthetic_other", {}))
            response = SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(tool_calls=[call]))])
            adapter._execute_tool_calls(response, None)
        assert adapter._validate_tool_call is validate
        return original(name, args)

    monkeypatch.setattr(adapter, "invoke_tool", invoke)
    assert _success(*_call(host))["experiment_counts"]["successful"] == 1
    assert len(nested) == 1
    assert adapter._TOOL_PATH_POLICY.get() is None


def test_red18_history_swap_after_config_inside_batch_catch(fs, monkeypatch, request):
    host, outside = fs
    exp = _experiment(host / "exp")
    original = summarize_experiment.read_config
    changed = []

    def config(path):
        data = original(path)
        (exp / "history.json").unlink()
        _filelink(exp / "history.json", outside / "history.json")
        changed.append(True)
        return data

    monkeypatch.setattr(summarize_experiment, "read_config", config)
    rec = _ReadRecorder(monkeypatch, outside)
    _failure(*_call(host), rec, request, private=True)
    assert changed == [True]


@pytest.mark.parametrize("kind", ["permission", "fifo", "socket", "device"] + (["unknown-reparse"] if os.name == "nt" else []))
def test_red19_resolution_and_special_types_fail_closed(fs, monkeypatch, request, kind):
    host, outside = fs
    exp = _experiment(host / "exp")
    target = exp / "history.json"
    original = Path.lstat
    calls = []

    def lstat(path, *args, **kwargs):
        if path == target:
            calls.append(kind)
            if kind == "permission":
                raise PermissionError("synthetic private target")
            mode = {"fifo": stat.S_IFIFO, "socket": stat.S_IFSOCK, "device": stat.S_IFCHR,
                    "unknown-reparse": stat.S_IFREG}[kind]
            return SimpleNamespace(st_mode=mode, st_file_attributes=1024 if kind == "unknown-reparse" else 0,
                                   st_reparse_tag=0x80000042)
        return original(path, *args, **kwargs)

    rec = _ReadRecorder(monkeypatch, outside)
    monkeypatch.setattr(Path, "lstat", lstat)
    _failure(*_call(host), rec, request, private=True)
    assert calls, "metadata seam must be consumed by authorization"
    assert str(target) not in rec.all_events, "unsupported objects must never be opened"


@pytest.mark.parametrize("tool", TOOLS)
def test_red20_host_alias_optional_config_success(fs, tool):
    host, _ = fs
    _experiment(host / "exp")
    config = _metrics(host / "metrics.yaml")
    alias = host.with_name("host-alias")
    _dirlink(alias, host)
    expected = getattr(tools, tool)(str(host if tool == TOOLS[0] else host / "exp"), metrics_config=str(config))
    assert _success(*_call(alias, tool=tool, selection="." if tool == TOOLS[0] else "exp",
                          extra={"metrics_config": "metrics.yaml"})) == expected


@pytest.mark.parametrize("tool", TOOLS)
def test_red20_optional_config_swap_before_reader(fs, monkeypatch, request, tool):
    host, outside = fs
    _experiment(host / "exp")
    config = _metrics(host / "metrics.yaml")
    external = _metrics(outside / "metrics.yaml")
    original = tools.read_metric_specs_config
    changes = []

    def read(path):
        assert path == config  # Initial model-argument policy has already run.
        config.unlink()
        _filelink(config, external)
        changes.append(True)
        return original(path)

    monkeypatch.setattr(tools, "read_metric_specs_config", read)
    rec = _ReadRecorder(monkeypatch, outside)
    _failure(*_call(host, tool=tool, selection="." if tool == TOOLS[0] else "exp",
                    extra={"metrics_config": "metrics.yaml"}), rec, request, private=True)
    assert changes == [True]


# Independent-review supplements: retain the original 62 Windows node IDs and
# assertions, and complete exact frozen-matrix obligations without mocks of IO.
def test_red10_outside_marker_never_reaches_provider(fs, monkeypatch, request):
    host, outside = fs
    _dirlink(host / "linked", outside)
    rec = _ReadRecorder(monkeypatch, outside)
    client, session = _call(host)
    _failure(client, session, rec, request)
    assert outside.name not in json.dumps(client.requests)
    assert outside.name not in repr(session.history)


if os.name == "nt":
    def test_red13_real_external_short_alias_is_not_authority(fs, monkeypatch, request):
        host, outside = fs
        kernel = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel.GetShortPathNameW.argtypes = [ctypes.c_wchar_p, ctypes.c_wchar_p, ctypes.c_uint]
        kernel.GetShortPathNameW.restype = ctypes.c_uint
        buf = ctypes.create_unicode_buffer(32768)
        size = kernel.GetShortPathNameW(str(outside), buf, len(buf))
        assert 0 < size < len(buf)
        alias = Path(buf.value)
        assert alias.samefile(outside)
        assert "~" in alias.name and alias.name.casefold() != outside.name.casefold(), (
            "real distinct 8.3 alias required; unchanged long spelling is not acceptance"
        )
        request.node.user_properties.append(("native_alias", {
            "long_name": outside.name, "short_name": alias.name, "same_object": True,
        }))
        _dirlink(host / "linked", alias)
        rec = _ReadRecorder(monkeypatch, outside)
        _failure(*_call(host), rec, request, private=True)


@pytest.mark.parametrize("nested_raises", [False, True])
def test_red17_nested_dispatch_restores_outer_guard_even_after_exception(fs, monkeypatch, request, nested_raises):
    host, outside = fs
    _experiment(host / "valid")
    observed = []

    def invoke(name, args):
        if name == "synthetic_other":
            observed.append("nested")
            if nested_raises:
                raise RuntimeError("synthetic nested failure")
            return {"ok": True}
        call = SimpleNamespace(id="nested", type="function", function=SimpleNamespace(
            name="synthetic_other", arguments="{}"))
        response = SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(tool_calls=[call]))])
        with monkeypatch.context() as patch:
            patch.setattr(adapter, "_validate_tool_call", lambda c, i: ("nested", "synthetic_other", {}))
            if nested_raises:
                with pytest.raises(RuntimeError, match="synthetic nested failure"):
                    adapter._execute_tool_calls(response, None)
            else:
                adapter._execute_tool_calls(response, None)
        observed.append("outer-read-after-nested")
        # This must still be under the original default-tool host authority.
        return read_history.read_history(outside / "history.json")

    monkeypatch.setattr(adapter, "invoke_tool", invoke)
    rec = _ReadRecorder(monkeypatch, outside)
    _failure(*_call(host), rec, request, private=True)
    assert observed == ["nested", "outer-read-after-nested"]
    assert adapter._TOOL_PATH_POLICY.get() is None
    # Direct host APIs remain unscoped after failure, rather than retaining H.
    assert read_history.read_history(outside / "history.json")


def test_red17_derived_rejection_cleans_up_scope(fs, monkeypatch, request):
    host, outside = fs
    _dirlink(host / "linked", outside)
    rec = _ReadRecorder(monkeypatch, outside)
    _failure(*_call(host), rec, request, private=True)
    assert adapter._TOOL_PATH_POLICY.get() is None
    assert read_history.read_history(outside / "history.json")
    assert _success(*_call(outside, tool="analyze_experiment"))


def test_red19_reader_guard_preserves_unexpected_programming_exception(fs, monkeypatch):
    host, _ = fs
    exp = _experiment(host / "valid")
    target = exp / "history.json"
    original = Path.resolve
    bug = RuntimeError("synthetic programming bug")
    touched = []

    def resolve(path, *args, **kwargs):
        if path == target:
            touched.append(True)
            raise bug
        return original(path, *args, **kwargs)

    def invoke(name, args):
        # Real Adapter first validates host/model paths and establishes scope.
        # Isolate the reader guard from the legacy analyzer's broad parse catch.
        with monkeypatch.context() as patch:
            patch.setattr(Path, "resolve", resolve)
            return read_history.read_history(target)

    monkeypatch.setattr(adapter, "invoke_tool", invoke)
    client, session = _call(host)
    with pytest.raises(RuntimeError) as caught:
        session.ask("Use only the authorized experiments.")
    assert caught.value is bug and touched == [True]
    assert len(client.requests) == 1 and len(session.history) == 0

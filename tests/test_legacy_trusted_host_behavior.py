"""A1 legacy guards: real APIs, harmless protocols, no behavior modification.

G7 characterizes this baseline only. Retained mutable aliases are a limitation,
not a safety feature or permanent API promise. Authorized future isolation or
rejection requires reviewing these characterization cases, not preserving risk.
"""
from collections.abc import Mapping, Sequence
from contextlib import contextmanager
from copy import deepcopy
import json
from pathlib import Path
import sys
from types import SimpleNamespace as NS

import pytest

from copilot.session import CopilotSession
from llm_adapters.tool_binding import BoundToolCollection, bound_tools
from tool_layer import experiment_tools as tools
from tool_layer import tool_registry as registry
from experiment_comparison import compare_experiment_evidence
from experiment_replicates import aggregate_replicate_evidence
from experiment_group_comparison import compare_replicate_groups
from read_config import read_config_evidence
from read_test_metrics import read_test_metrics
from m17_contract_helpers import arguments, evidence, options, spec, target, write_experiment
from m18_contract_helpers import bundle, input_envelope
from m19_contract_helpers import pair as group_pair


CASES = [
    "G1-m16", "G1-m17", "G2-m16-sequence", "G2-m17-evidence-mapping",
    "G2-m17-target-sequence", "G2-m17-options-mapping", "G3-m16", "G3-m17",
    "G4-m16", "G4-m17", "G5-m16", "G5-m17", "G6-m16", "G6-m17",
    "G7-m16", "G7-m17", "G8-m18", "G8-m19", "G9-default-registry",
    "G10-m16-root", "G10-m17-ordered-root", "G10-csv-overflow",
]
M16 = "analyze_experiment_evidence"
M17 = "compare_experiment_evidence"


class Counter:
    def __init__(self):
        self.phase = "construction"
        self.events = []

    def hit(self, protocol):
        self.events.append((self.phase, protocol))

    def count(self, phase, protocol=None):
        return sum(p == phase and (protocol is None or name == protocol)
                   for p, name in self.events)


class CountedSequence(Sequence):
    def __init__(self, data, counter, *, alias=False):
        self.data, self.counter, self.alias = data, counter, alias

    def __len__(self):
        self.counter.hit("len")
        return len(self.data)

    def __getitem__(self, key):
        self.counter.hit("getitem")
        return self.data[key]

    def __iter__(self):
        self.counter.hit("iter")
        return iter(self.data)

    def __deepcopy__(self, memo):
        self.counter.hit("deepcopy")
        if self.alias:
            return self
        result = CountedSequence(deepcopy(self.data, memo), self.counter)
        memo[id(self)] = result
        return result


class CountedMapping(Mapping):
    def __init__(self, data, counter):
        self.data, self.counter = data, counter

    def __len__(self):
        self.counter.hit("len")
        return len(self.data)

    def __iter__(self):
        self.counter.hit("iter")
        return iter(self.data)

    def __getitem__(self, key):
        self.counter.hit("getitem")
        return self.data[key]

    def __deepcopy__(self, memo):
        self.counter.hit("deepcopy")
        result = CountedMapping(deepcopy(self.data, memo), self.counter)
        memo[id(self)] = result
        return result


def _calibrate():
    c = Counter()
    seq = CountedSequence([["enabled"]], c)
    assert len(seq) == 1 and list(seq) == [["enabled"]]
    copied = deepcopy(seq)
    seq.data[0][0] = "changed"
    assert copied.data == [["enabled"]] and copied is not seq
    mapping = CountedMapping({"x": [1]}, c)
    assert dict(mapping) == {"x": [1]} and len(mapping) == 1
    detached = deepcopy(mapping)
    mapping.data["x"].append(2)
    assert detached.data == {"x": [1]}
    for protocol in ("len", "iter", "getitem", "deepcopy"):
        assert c.count("construction", protocol) > 0
    alias = CountedSequence([], c, alias=True)
    assert deepcopy(alias) is alias


class Provider:
    """Messages only; every Tool result is produced by the real stack."""
    def __init__(self, name, args):
        self.name, self.args, self.requests = name, args, []
        self.chat = NS(completions=self)

    def create(self, **kwargs):
        self.requests.append(deepcopy(kwargs))
        if len(self.requests) == 1:
            message = NS(role="assistant", content=None, tool_calls=[NS(
                id="legacy-call", type="function", function=NS(
                    name=self.name, arguments=json.dumps(self.args)))])
        else:
            assert len(self.requests) == 2, "unexpected retry"
            message = NS(role="assistant", content="offline result", tool_calls=None)
        return NS(choices=[NS(message=message)])


@contextmanager
def _trace():
    """Observe actual calls without replacing any production function."""
    functions = [tools.analyze_experiment_evidence, compare_experiment_evidence,
                 read_config_evidence, read_test_metrics]
    codes = {fn.__code__: fn.__name__ for fn in functions}
    calls = []
    previous = sys.getprofile()

    def observe(frame, event, arg):
        if event == "call" and frame.f_code in codes:
            calls.append(codes[frame.f_code])
        if previous is not None:
            previous(frame, event, arg)

    sys.setprofile(observe)
    try:
        yield calls
    finally:
        sys.setprofile(previous)


def _bind(kind, a, b, *, counter=None, alias=False):
    oa, ob, selected = options(), options(), [target()]
    if kind == "m16":
        host = CountedSequence(oa["config_paths"], counter, alias=alias) if counter else oa["config_paths"]
        oa["config_paths"] = host
        return tools.ExperimentEvidenceTools(str(a), **oa), host
    host = CountedSequence(selected, counter, alias=alias) if counter else selected
    return tools.ExperimentComparisonTools(str(a), str(b), options_a=oa,
                                          options_b=ob, targets=host), host


def _args(kind, a, b):
    return {"experiment_dir": str(a)} if kind == "m16" else arguments(a, b)


def _name(kind):
    return M16 if kind == "m16" else M17


def _assert_result(kind, result):
    if kind == "m16":
        assert result["test_results"]["status"] == "available"
        assert result["test_results"]["metrics"]["score"]["value"] == 0.5
        assert result["configuration_evidence"]["fields"][0]["path"] == ["enabled"]
        assert result["configuration_evidence"]["fields"][0]["value"] is False
    else:
        row = result["targets"]["quality"]
        assert row["calculation"]["status"] == "calculated"
        assert row["calculation"]["raw_delta"] == 0.25
        assert row["comparability"]["status"] == "comparable"


def _session(tool, name, args, root):
    client = Provider(name, args)
    context = {"experiment_root": str(root)} if root else None
    session = CopilotSession(client, model="offline", experiment_context=context)
    return client, session, BoundToolCollection(tool)


def _phase_record(record_property, counter):
    record_property("callback_phases", json.dumps(counter.events))


def _strict_guard(kind, record_property):
    from dataclasses import fields
    if kind == "m18":
        data = input_envelope(bundle())

        def call(value):
            return aggregate_replicate_evidence(**value)

        key = "group_declaration"
    else:
        a, b, mapping = group_pair()
        data = {"group_a": a, "group_b": b, "targets": mapping}

        def call(value):
            return compare_replicate_groups(**value)

        key = "group_a"
    expected = call(data)
    c = Counter()
    invalid = dict(data)
    invalid[key] = CountedMapping(data[key], c)
    with pytest.raises(TypeError, match="concrete|exact"):
        call(invalid)
    assert c.events == []
    # Actual project instance with caller-supplied metadata must not drive schema.
    identity = (data["group_declaration"]["experiment_identity"] if kind == "m18"
                else data["group_a"]["group_declaration"]["experiment_identity"])
    metadata = CountedMapping(dict(identity.__dataclass_fields__), c)
    object.__setattr__(identity, "__dataclass_fields__", metadata)
    try:
        assert fields(identity)  # positive calibration of metadata traversal
        assert c.events
        c.events.clear()
        assert call(data) == expected
        assert c.events == []
    finally:
        object.__delattr__(identity, "__dataclass_fields__")
    for payload in ("x" * (2 * 1024 * 1024 + 1),):
        oversized = deepcopy(data)
        if kind == "m18":
            oversized["executions"][0]["snapshot_metadata"]["extra"] = payload
        else:
            oversized["group_a"]["executions"][0]["snapshot_metadata"]["extra"] = payload
        with pytest.raises(ValueError, match="budget|2 MiB"):
            call(oversized)
    cyclic = deepcopy(data)
    holder = cyclic["executions"][0]["snapshot_metadata"] if kind == "m18" else cyclic["group_a"]["executions"][0]["snapshot_metadata"]
    holder["cycle"] = holder
    with pytest.raises(ValueError, match="cycle|depth"):
        call(cyclic)
    record_property("strict_boundary", "custom mapping rejected; fixed fields; byte/cycle gates preserved")


@pytest.mark.parametrize("case", CASES)
def test_legacy_behavior_guard(case, tmp_path, record_property):
    _calibrate()
    if case.startswith("G8-"):
        _strict_guard(case.split("-", 1)[1], record_property)
        return
    if case == "G9-default-registry":
        assert [t["function"]["name"] for t in registry.list_tools()] == [
            "analyze_experiment", "compare_experiments"]
        with _trace() as calls:
            for name in (M16, M17):
                with pytest.raises(KeyError):
                    registry.invoke_tool(name, {})
        assert calls == []
        return
    if case == "G10-csv-overflow":
        path = tmp_path / "artifact.csv"
        path.write_bytes(b"x" * (1024 * 1024 + 2))
        reads = []

        class ObservedFile:
            def __init__(self, wrapped):
                self.wrapped = wrapped

            def __enter__(self):
                self.wrapped.__enter__()
                return self

            def __exit__(self, *args):
                return self.wrapped.__exit__(*args)

            def read(self, count):
                value = self.wrapped.read(count)
                reads.append((count, len(value)))
                return value

        class ObservedPath(type(Path())):
            def open(self, *args, **kwargs):
                assert args == ("rb",) and kwargs == {"buffering": 0}
                return ObservedFile(super().open(*args, **kwargs))

        result = read_test_metrics(ObservedPath(path), [spec()])
        assert result["status"] == "invalid"
        assert result["issues"] == [{"code": "file_too_large"}]
        assert result["metrics"] == {}
        assert sum(n for _, n in reads) == 1024 * 1024 + 1
        assert all(0 < requested <= 65536 for requested, _ in reads)
        record_property("actual_read_bytes", sum(n for _, n in reads))
        return
    a = write_experiment(tmp_path, "a")
    b = write_experiment(tmp_path, "b", value=0.75)
    kind = "m16" if "m16" in case else "m17"
    name, args = _name(kind), _args(kind, a, b)
    if case.startswith("G1-"):
        tool, host = _bind(kind, a, b)
        before = deepcopy(host)
        if kind == "m16":
            host[0][0] = "changed"
        else:
            host[0]["b"]["metric_definition"] = "changed"
        with _trace() as calls:
            result = tool.invoke_tool(name, args)
        _assert_result(kind, result)
        assert calls.count("analyze_experiment_evidence") == (1 if kind == "m16" else 2)
        assert calls.count("compare_experiment_evidence") == (0 if kind == "m16" else 1)
        result.clear()
        _assert_result(kind, tool.invoke_tool(name, args))
        assert host != before
        return
    if case.startswith("G2-"):
        c = Counter()
        oa, ob, selected = options(), options(), [target()]
        if case.endswith("evidence-mapping"):
            raw = evidence()
            result = compare_experiment_evidence(CountedMapping(raw, c), evidence(0.75), targets=selected)
            assert result == compare_experiment_evidence(raw, evidence(0.75), targets=selected)
        else:
            if kind == "m16":
                host = CountedSequence(oa["config_paths"], c)
                oa["config_paths"] = host
                tool = tools.ExperimentEvidenceTools(str(a), **oa)
                host.data[0][0] = "mutated"
            elif case.endswith("target-sequence"):
                host = CountedSequence(selected, c)
                tool = tools.ExperimentComparisonTools(str(a), str(b), options_a=oa, options_b=ob, targets=host)
                host.data[0]["b"]["metric_definition"] = "mutated"
            else:
                host = CountedMapping(oa, c)
                tool = tools.ExperimentComparisonTools(str(a), str(b), options_a=host, options_b=ob, targets=selected)
                host.data["test_metrics_file"] = "mutated.csv"
                host.data["config_paths"][0][0] = "mutated"
            result = tool.invoke_tool(name, args)
            plain, _ = _bind(kind, a, b)
            assert result == plain.invoke_tool(name, args)
        _assert_result("m17" if kind == "m17" else "m16", result)
        assert c.events
        _phase_record(record_property, c)
        return
    c = Counter()
    tool, host = _bind(kind, a, b, counter=c, alias=case.startswith("G7-"))
    assert c.count("construction", "deepcopy") > 0
    c.phase = "invocation"
    if case.startswith("G7-"):
        _assert_result(kind, tool.invoke_tool(name, args))
        if kind == "m16":
            host.data[0][0] = "missing_field"
            result = tool.invoke_tool(name, args)
            field = result["configuration_evidence"]["fields"][0]
            assert field["path"] == ["missing_field"] and field["status"] == "missing"
        else:
            host.data[0]["b"]["metric_definition"] = "changed-definition"
            result = tool.invoke_tool(name, args)
            row = result["targets"]["quality"]["calculation"]
            assert row["status"] == "blocked" and row["raw_delta"] is None
        record_property("characterization", "mutable self-copy alias affects result; NOT permanent API guarantee")
        record_property("alias_result", json.dumps(result))
        _phase_record(record_property, c)
        return
    if case.startswith("G3-"):
        forbidden = ["config_paths", "targets", "options", "options_a", "options_b",
                     "test_metrics_file", "test_metric_specs", "metrics_config"]
        for key in forbidden:
            client, session, collection = _session(tool, name, {**args, key: []}, tmp_path)
            with _trace() as calls, bound_tools(collection):
                message = ("unsupported evidence tool arguments" if kind == "m16"
                           else "comparison requires only experiment_a and experiment_b")
                with pytest.raises(ValueError, match=message):
                    session.ask_with_observability("read")
            assert calls == [] and session.history == () and len(client.requests) == 1
        assert c.count("invocation") == 0
    elif case.startswith(("G4-", "G6-")):
        client, session, collection = _session(tool, name, args, None)
        with _trace() as calls, bound_tools(collection):
            with pytest.raises(ValueError, match="authoriz|context|root"):
                session.ask_with_observability("read")
        assert calls == [] and session.history == () and len(client.requests) == 1
        assert c.count("invocation") == 0
        assert c.count("construction", "deepcopy") > 0
    elif case.startswith("G5-"):
        client, session, collection = _session(tool, name, args, tmp_path)
        with _trace() as calls, bound_tools(collection):
            observed = session.ask_with_observability("read")
        payload = json.loads(observed.turn.tool_invocations[0].result_json)
        _assert_result(kind, payload)
        assert len(client.requests) == 2 and session.history == (observed.turn,)
        assert c.count("invocation") > 0
        assert calls.count("analyze_experiment_evidence") == (1 if kind == "m16" else 2)
        assert calls.count("compare_experiment_evidence") == (0 if kind == "m16" else 1)
    else:
        assert case in ("G10-m16-root", "G10-m17-ordered-root")
        wrong = {"experiment_dir": str(b)} if kind == "m16" else arguments(b, a)
        client, session, collection = _session(tool, name, wrong, tmp_path)
        with _trace() as calls, bound_tools(collection):
            with pytest.raises(ValueError, match="host-bound"):
                session.ask_with_observability("read")
        assert calls == [] and session.history == () and len(client.requests) == 1
        assert c.count("invocation") == 0
    _phase_record(record_property, c)

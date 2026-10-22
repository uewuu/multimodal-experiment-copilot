"""M18 frozen test vocabulary; builders/assertions only, never an aggregator.

PUBLIC: experiment_replicates.aggregate_replicate_evidence(executions, *,
group_declaration, targets); tool_layer.replicate_tools.ReplicateAggregationTools
has the same constructor arguments and list_tools()/invoke_tool(name, arguments).

INPUT: group_declaration has group_id, experiment_identity (M11),
expected_executions [{execution: ExecutionIdentity, expected_seed: int|None}],
training_protocol_ref (str|None), membership_selection, aggregation_policy.
Only replicate-descriptive-v1 is supported. Executions have execution,
provenance (M11), training_protocol_ref, status (available/unavailable), evidence
(M16 mapping/None), snapshot_metadata {snapshot_id, acquisition, binding}.
Binding is host_declared. Unknown seed is not an unknown experiment definition.
Targets have target_id, section (validation/test), members, with one selector
per expected execution. Each selector has execution, metric_name, metric_spec
(MetricSpec|None), metric_definition, aggregation, evaluation_protocol,
selection_protocol. Direction is the existing MetricSpec.direction, not a
second conflicting direction field. There is no caller-supplied metric value.

Malformed declarations/duplicate IDs/extra executions or selectors are rejected
with TypeError/ValueError. Different experiment identities block, not merge.
Identity/protocol conflicts block the group; unknown definitions make it
insufficient. Metric conflicts block the target; unknown semantics make it
insufficient. Missing/invalid observations retain their roster slots. Invalid
observation values are target-local exclusions, not whole-input exceptions.
Known declaration conflicts must not disappear when a metric is unavailable.
Check every distinct supplied member pair through public M11 primitives;
singletons still require complete declarations. Do not pass fabricated metric
values to M11 for absent observations. Missing snapshots do not fabricate
provenance, and do not prevent a partial summary of eligible supplied members.

OUTPUT: schema_version=replicate-aggregation-v1. group contains group_id,
experiment_identity (dataclass-shaped JSON), membership_source=host_declared,
group_status, identity_consistency {status, reason_codes}, aggregation_eligibility
{status, reason_codes}, coverage, seed_summary. Group status and eligibility:
eligible/blocked/insufficient. Per-target status: complete/partial/blocked/
insufficient. metric_comparability is M11 projection {status, reason_codes,
policy_version} or None when no numeric observation exists. Conflict wins over
unknown, without erasing missing reasons. Group/target eligibility blocks all
otherwise valid observations; never select a compatible subgroup.

Each target has included_executions [{execution: JSON identity, value, source}],
missing_executions/excluded_executions [{execution, reason_codes}], coverage,
statistics {count, mean, median, sample_std, min, max}, limitations. Included
source retains section/evidence_kind, metadata_source=host_declared, and Test
provenance; Validation retains best_epoch without claiming checkpoint identity.
No full pair matrix. All member lists sorted by (namespace, experiment_id,
execution_id); targets keyed in sorted target_id order. All reasons are stable
code strings. Input ordering has no effect on output.

Target observed_count counts finite observations before eligibility. Final
eligible_count=included_count. expected=included+missing+excluded. Missing
means absent/unavailable/not_requested, excluded means invalid or blocked.
On target/group block, valid observed members become excluded with reasons;
missing members stay missing. For numerical failure, target is blocked,
statistics count=0/others=None, observations become excluded, observed_count
is retained. Group usable_count is intersection of included execution sets
over requested targets; it does not restrict individual target statistics.
Zero observations: insufficient, count=0/others=None. One observation may be
complete or partial by coverage; sample_std=None plus insufficient_variability.
Std uses n-1; equal execution weights, never dataset sample weighting.

seed_summary: distinct_known_seed_count, unknown_seed_count, repeated_seeds
[{seed, executions:[JSON identity]}]. Distinct executions with repeated/unknown
seeds remain included. Only supplied members contribute seed_summary. Known
expected/realized seed disagreement blocks the group; unknown realized seed
does not get filled from expectation/config. Environment drift is a limitation.

Limits: 1..16 expected executions, 1..32 targets, <=120 distinct member pairs
per target; input <=2 MiB, output <=128 KiB compact UTF-8 JSON (ensure_ascii=False).
Input size is the complete {executions, group_declaration, targets} projection:
known dataclasses via asdict, tuples as arrays; byte counting may represent
invalid floats as JSON tokens, but OUTPUT must use allow_nan=False. Container
depth <=16 from that envelope (root depth 1); cycles/custom/lazy objects reject.
Budget checks cover unused metadata too, before unbounded recursion/copy.
Output overflow raises ValueError containing 'output'; never silently truncates.

Root limitations include statistical_inference_not_performed,
independence_not_verified, roster_host_declared_not_preregistered,
snapshot_currentness_not_verified. Target limitations retain
checkpoint_binding_not_verified, plus local missing/invalid/seed limitations.
No mean/delta group comparison and no inferred provenance or independence.

Tests use generic M16-shaped fixtures independently checked against real M16.
Provider doubles simulate only messages; no future implementation is faked.
I/O probes cover listed Python/public project entrances, not every OS I/O path.
"""

from contextlib import contextmanager
from copy import deepcopy
from dataclasses import asdict
import builtins
import importlib
import importlib.util
import io
import json
import os
from pathlib import Path
import socket
from types import SimpleNamespace as NS

from experiment_identity import ExperimentIdentity, ExecutionIdentity, ExperimentProvenance
from m17_contract_helpers import evidence, spec


TOOL = "aggregate_replicate_evidence"
CORE_MISSING = "missing M18 core: experiment_replicates.aggregate_replicate_evidence"
TOOL_MISSING = "missing M18 tool: tool_layer.replicate_tools.ReplicateAggregationTools"
PARAMETERS = {"type": "object", "properties": {}, "additionalProperties": False}
ROOT_LIMITATIONS = {
    "statistical_inference_not_performed", "independence_not_verified",
    "roster_host_declared_not_preregistered", "snapshot_currentness_not_verified",
}


def core_api():
    assert importlib.util.find_spec("experiment_replicates") is not None, CORE_MISSING
    function = getattr(importlib.import_module("experiment_replicates"), TOOL, None)
    assert callable(function), CORE_MISSING
    return function


def tool_class():
    assert importlib.util.find_spec("tool_layer.replicate_tools") is not None, TOOL_MISSING
    cls = getattr(importlib.import_module("tool_layer.replicate_tools"),
                  "ReplicateAggregationTools", None)
    assert callable(cls), TOOL_MISSING
    return cls


def identity(index):
    return ExecutionIdentity(ExperimentIdentity("synthetic-lab", "method-v1"), f"exec-{index:02d}")


def bundle(values=(1.0, 3.0, 5.0), *, seeds=None, section="test"):
    """Build data, not expected output or computed statistics."""
    seeds = list(range(42, 42 + len(values))) if seeds is None else list(seeds)
    assert len(seeds) == len(values)
    executions, roster, members = [], [], []
    for i, (value, seed) in enumerate(zip(values, seeds)):
        execution = identity(i)
        provenance = ExperimentProvenance(
            execution, configuration_ref="recipe-v1", task="regression", target="quality",
            model_ref="model-v1", dataset_version="dataset-v1", split_id="split-v1",
            source_revision="revision-v1", initialization_checkpoint_ref="initialization-v1",
            realized_seed=seed, environment_ref="environment-v1",
            location=f"opaque-location-{i}", recorded_at=f"declared-time-{i}",
        )
        executions.append({
            "execution": execution, "provenance": provenance,
            "training_protocol_ref": "training-v1", "status": "available",
            "evidence": evidence(value),
            "snapshot_metadata": {"snapshot_id": f"snapshot-{i}",
                                  "acquisition": "host supplied offline snapshot", "binding": "host_declared"},
        })
        roster.append({"execution": execution, "expected_seed": seed})
        members.append({
            "execution": execution, "metric_name": "score", "metric_spec": spec(section=section),
            "metric_definition": "quality-v1", "aggregation": "macro",
            "evaluation_protocol": "evaluation-v1", "selection_protocol": "selection-v1",
        })
    group = {"group_id": "group-v1", "experiment_identity": identity(0).experiment,
             "expected_executions": roster, "training_protocol_ref": "training-v1",
             "membership_selection": "host-selected executions; not preregistration",
             "aggregation_policy": "replicate-descriptive-v1"}
    return executions, group, [{"target_id": "quality", "section": section, "members": members}]


def invoke(function, data):
    executions, group, targets = data
    return function(executions, group_declaration=group, targets=targets)


def make_tools(cls, data):
    executions, group, targets = data
    return cls(executions, group_declaration=group, targets=targets)


def json_size(value):
    return len(json.dumps(value, default=asdict, ensure_ascii=False,
                          separators=(",", ":")).encode("utf-8"))


def input_envelope(data):
    return dict(zip(("executions", "group_declaration", "targets"), data))


def member_ids(rows):
    return [row["execution"]["execution_id"] for row in rows]


def assert_partition(row, expected_ids):
    included, missing, excluded = [member_ids(row[key]) for key in (
        "included_executions", "missing_executions", "excluded_executions")]
    flattened = included + missing + excluded
    assert len(flattened) == len(set(flattened))
    assert sorted(flattened) == sorted(expected_ids)
    assert all(ids == sorted(ids) for ids in (included, missing, excluded))
    coverage = row["coverage"]
    for key, ids in (("included_count", included), ("missing_count", missing), ("excluded_count", excluded)):
        assert coverage[key] == len(ids)
    assert coverage["expected_count"] == len(expected_ids)
    assert coverage["eligible_count"] == len(included)
    assert coverage["coverage_ratio"] == len(included) / len(expected_ids)
    assert row["statistics"]["count"] == len(included)
    for entry in row["missing_executions"] + row["excluded_executions"]:
        assert entry["reason_codes"] and all(type(x) is str for x in entry["reason_codes"])


def assert_empty_statistics(row):
    assert row["statistics"] == {
        "count": 0, "mean": None, "median": None, "sample_std": None, "min": None, "max": None}


class Client:
    """Provider messages only; real core/tool/Session supply all evidence."""
    def __init__(self, args=None, *, final="offline answer", error=None):
        self.args = {} if args is None else args
        self.final, self.error, self.requests = final, error, []
        self.chat = NS(completions=self)

    def create(self, **kwargs):
        self.requests.append(deepcopy(kwargs))
        if len(self.requests) == 1:
            message = NS(role="assistant", content=None, tool_calls=[NS(
                id="replicate-call", type="function", function=NS(name=TOOL, arguments=json.dumps(self.args)))])
        elif len(self.requests) == 2:
            if self.error is not None:
                raise self.error
            message = NS(role="assistant", content=self.final, tool_calls=None)
        else:
            raise AssertionError("unexpected provider retry")
        return NS(choices=[NS(message=message)])


@contextmanager
def deny_external_access(monkeypatch):
    """Preload before patching; scoped deny list, independently calibrated below."""
    from experiment_memory import ExperimentMemory
    from experiment_repository import ExperimentRepository
    from tool_layer import tool_registry, experiment_tools
    calls = []
    def deny(*args, **kwargs):
        calls.append((args, kwargs))
        raise AssertionError("unexpected external access")
    with monkeypatch.context() as patch:
        entrances = [(builtins, "open"), (io, "open"), (os, "open"), (Path, "open"),
                     (Path, "resolve"), (Path, "stat"), (socket, "create_connection"),
                     (socket.socket, "connect"), (ExperimentRepository, "__init__"),
                     (ExperimentMemory, "get"), (ExperimentMemory, "list_executions"),
                     (ExperimentMemory, "find_comparison_candidates"),
                     (tool_registry, "invoke_tool"), (tool_registry, "list_tools"),
                     (tool_registry, "_workspace_paths_for_tool"),
                     (experiment_tools, "analyze_experiment_evidence")]
        for owner, name in entrances:
            patch.setattr(owner, name, deny)
        yield calls

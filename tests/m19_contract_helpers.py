"""M19 v1 scope freeze: fixtures, probes and assertions, never a comparator.

PUBLIC: experiment_group_comparison.compare_replicate_groups(group_a, group_b,
*, targets); tool_layer.replicate_group_tools.ReplicateGroupComparisonTools
has the same constructor signature and list_tools()/invoke_tool(name, arguments).
Each group has exactly executions/group_declaration/targets, using M18 inputs.
Mapping entries have exactly target_id/a_target_id/b_target_id. Every submitted
source target on both sides maps once; no implicit name/order alignment.

DELEGATION: real public M18 aggregation twice per valid invocation; real public
M11 compare_metric_results for every supplied A x supplied B member per target.
Use original execution provenance and selectors, finite observation or None,
never a mean/fabricated zero. Absent snapshots contribute no provenance/pairs.
Known conflicts survive unavailable numeric observations. M18 owns statistics,
coverage, exclusions and group eligibility; M19 never recomputes common cases.

OUTPUT root has schema_version=replicate-group-comparison-v1,
comparison_policy=unpaired-descriptive-v1, groups={a,b}, targets, limitations.
Group projection: group_id, experiment_identity, aggregation_policy,
membership_source, training_protocol_ref, aggregation_eligibility, coverage,
seed_summary, expected_executions (declared execution/expected_seed records).
Target sides: source_target_id, section, target_status, aggregation_eligibility,
coverage, statistics, included_executions, missing_executions,
excluded_executions, limitations. Selected source evidence stays in included
member.source (no redundant raw envelope/full M18 result/pair matrix).
alignment={status: aligned|blocked, reason_codes}; section conflict blocks.
metric_comparability={status: comparable|incompatible|unknown, policy_version:
metric-result-v1, reason_codes, assessed_pair_count}; no supplied pairs means
unknown, count 0, reason no_supplied_cross_pairs. Comparison fields:
eligibility, reason_codes, calculation_status (calculated|blocked), raw_delta,
direction (maximize|minimize|None), direction_interpretation
(a_favorable|b_favorable|equal|None), coverage_relation.
coverage_relation is {a_complete: bool, b_complete: bool,
same_expected_count: bool, same_included_count: bool}.

POLICY: known block wins over insufficient; M18 blocked/section conflict/full
ExecutionIdentity overlap anywhere in expected/supplied rosters/M11 incompatible
block delta. M11 unknown or mean unavailable is insufficient. Otherwise partial
coverage, unequal expected sizes or n=1 makes limited; complete equal rosters
with n>=2 are eligible. Baseline provenance limitations alone do not downgrade
every comparison. Delta always B.mean-A.mean, independent of direction. No
finite subtraction result => blocked with numeric_overflow, never zero.
Reasons include section_mismatch, overlapping_execution_identity,
coverage_mismatch (different included/expected counts), incomplete_coverage,
expected_roster_size_mismatch, insufficient_variability, mean_unavailable;
retain M18/M11 reasons. Conflicting/unknown directions have no interpretation.
Unsupported aggregation policy and malformed inputs reject TypeError/ValueError.
Source statuses/provenance never upgraded. Limitations retain M18 limits;
sorted unique reasons/limitations, members sorted by full ExecutionIdentity,
targets sorted by comparison id. No winner, ranking, pairing, inference or
std_delta; no verified independence, currentness, stability or causal claims.

SAFETY: exact primitives/dict/list/tuple and four approved project dataclasses
only; declared fields only, extra instance attributes ignored. Both groups and
mapping safely projected BEFORE any M18 call, no caller copy/iteration protocols.
Detached constructor/delegate/return/cross-call data; real core on every call.
No filesystem, network, Reader, Memory, Registry or path acquisition in core/tool.
Tool schema empty object, additionalProperties=false, any nonempty args rejected
before core. Explicit binding only, no new path capability/default dispatch.

BOUNDS: 1..16 expected and 0..16 supplied/group, 1..32 mappings; each group
compact UTF8 JSON <=2MiB, mapping <=16KiB, whole {group_a,group_b,targets} <=4MiB.
Group container depth <=16 from group envelope root=1, mapping fixed shallow
shape. <=256 cross pairs/target. Output strict compact UTF8 JSON <=128KiB or
ValueError containing 'output', no truncation. Input invalid measurement floats
remain data for M18 local exclusion; never emit NaN/Infinity. Inclusive bounds.

Tests use generic offline fixtures; providers double messages only. I/O probes
cover the named Python/project entrances, not every operating-system I/O path.
Structured evaluation is not a universal natural-language claim detector.
"""

from copy import deepcopy
from dataclasses import asdict, replace
import importlib
import importlib.util
import json
from types import SimpleNamespace as NS

from experiment_identity import ExperimentIdentity, ExecutionIdentity
from m18_contract_helpers import bundle, input_envelope, json_size, ROOT_LIMITATIONS

TOOL = "compare_replicate_groups"
PARAMETERS = {"type": "object", "properties": {}, "additionalProperties": False}
CORE_MISSING = "missing M19 core: experiment_group_comparison.compare_replicate_groups"
TOOL_MISSING = "missing M19 tool: tool_layer.replicate_group_tools.ReplicateGroupComparisonTools"
SIDE_KEYS = {"source_target_id", "section", "target_status", "aggregation_eligibility",
             "coverage", "statistics", "included_executions", "missing_executions",
             "excluded_executions", "limitations"}
GROUP_KEYS = {"group_id", "experiment_identity", "aggregation_policy", "membership_source",
              "training_protocol_ref", "aggregation_eligibility", "coverage", "seed_summary",
              "expected_executions"}


def core_api():
    assert importlib.util.find_spec("experiment_group_comparison") is not None, CORE_MISSING
    function = getattr(importlib.import_module("experiment_group_comparison"), TOOL, None)
    assert callable(function), CORE_MISSING
    return function


def tool_class():
    assert importlib.util.find_spec("tool_layer.replicate_group_tools") is not None, TOOL_MISSING
    cls = getattr(importlib.import_module("tool_layer.replicate_group_tools"),
                  "ReplicateGroupComparisonTools", None)
    assert callable(cls), TOOL_MISSING
    return cls


def group(label, values=(1., 3., 5.), *, section="test", same_experiment=False):
    data = input_envelope(bundle(values, section=section))
    experiment = ExperimentIdentity("synthetic-lab", "shared" if same_experiment else label)
    declaration = data["group_declaration"]
    declaration.update(group_id=f"group-{label}", experiment_identity=experiment)
    for i, (snapshot, roster, member) in enumerate(zip(
            data["executions"], declaration["expected_executions"], data["targets"][0]["members"])):
        execution = ExecutionIdentity(experiment, f"{label}-{i:02d}")
        snapshot["execution"] = roster["execution"] = member["execution"] = execution
        snapshot["provenance"] = replace(snapshot["provenance"], execution=execution)
    data["targets"][0]["target_id"] = f"quality-{label}"
    return data


def pair(a_values=(1., 3., 5.), b_values=(2., 4., 6.), **kwargs):
    return (group("A", a_values, **kwargs), group("B", b_values, **kwargs),
            [{"target_id": "quality", "a_target_id": "quality-A", "b_target_id": "quality-B"}])


def invoke(function, data):
    return function(data[0], data[1], targets=data[2])


def aggregate(data):
    from experiment_replicates import aggregate_replicate_evidence
    return aggregate_replicate_evidence(data["executions"],
        group_declaration=data["group_declaration"], targets=data["targets"])


def many_targets(data, count):
    """Duplicate declarations with distinct ids, never compute expected results."""
    a, b, mapping = data
    mapping.clear()
    for side in (a, b):
        first = side["targets"][0]
        side["targets"] = [dict(deepcopy(first), target_id=f"t-{i:02d}") for i in range(count)]
    for i in range(count):
        mapping.append({"target_id": f"comparison-{i:02d}", "a_target_id": f"t-{i:02d}",
                        "b_target_id": f"t-{i:02d}"})
    return data


def set_direction(data, direction):
    for target in data["targets"]:
        for member in target["members"]:
            member["metric_spec"] = (None if direction is None else
                                     replace(member["metric_spec"], direction=direction))
    if direction is not None:
        for snapshot in data["executions"]:
            snapshot["evidence"]["test_results"]["metrics"]["score"]["direction"] = direction


def patch_public(monkeypatch, owner, name, wrapper, *consumers):
    """Probe public function plus existing direct-import aliases, never results."""
    original = getattr(owner, name)
    monkeypatch.setattr(owner, name, wrapper)
    for module in consumers:
        for alias, value in list(vars(module).items()):
            if value is original:
                monkeypatch.setattr(module, alias, wrapper)


def key(execution):
    return (execution.experiment.namespace, execution.experiment.experiment_id, execution.execution_id)


def assert_no_claims(value):
    forbidden = {"winner", "best", "rank", "leaderboard", "recommendation", "std_delta",
                 "paired_delta", "paired_mean", "paired_std", "paired_ci", "paired_significance",
                 "mean_paired_improvement", "p_value", "confidence_interval", "t_test",
                 "wilcoxon", "bootstrap", "power_analysis", "statistically_significant",
                 "causal_effect", "effective_sample_size"}
    if isinstance(value, dict):
        assert not forbidden.intersection(value)
        for item in value.values(): assert_no_claims(item)
    elif isinstance(value, list):
        for item in value: assert_no_claims(item)


class Client:
    """Offline provider messages only; real binding/core supply evidence."""
    def __init__(self, args=None, *, final="offline descriptive answer", error=None):
        self.args = {} if args is None else args
        self.final, self.error, self.requests = final, error, []
        self.chat = NS(completions=self)

    def create(self, **kwargs):
        self.requests.append(deepcopy(kwargs))
        if len(self.requests) == 1:
            message = NS(role="assistant", content=None, tool_calls=[NS(
                id="group-comparison-call", type="function",
                function=NS(name=TOOL, arguments=json.dumps(self.args)))])
        elif len(self.requests) == 2:
            if self.error is not None: raise self.error
            message = NS(role="assistant", content=self.final, tool_calls=None)
        else:
            raise AssertionError("unexpected provider retry")
        return NS(choices=[NS(message=message)])

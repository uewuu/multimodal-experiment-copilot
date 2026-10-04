"""Bounded descriptive comparison of two host-supplied replicate groups.

M18 owns aggregation; M11 owns metric comparability. Agreement of supplied
declarations neither verifies their provenance nor establishes independence.
"""

from dataclasses import fields
import json
import math

import experiment_comparability as m11
import experiment_replicates as m18
from experiment_identity import (
    ExperimentIdentity, ExecutionIdentity, ExperimentProvenance, MetricResultProvenance,
)
from metrics import MetricSpec


_DATA_TYPES = (ExperimentIdentity, ExecutionIdentity, ExperimentProvenance, MetricSpec)
_GROUP_LIMIT = 2 * 1024 * 1024
_MAPPING_LIMIT = 16 * 1024
_TOTAL_LIMIT = 4 * 1024 * 1024
_OUTPUT_LIMIT = 128 * 1024
_TARGET_FIELDS = (
    "target_status", "aggregation_eligibility", "coverage", "statistics",
    "included_executions", "missing_executions", "excluded_executions", "limitations",
)


def _data_snapshot(value, *, limit, max_depth=16):
    """Project exact data types, counting compact UTF-8 bytes incrementally.

    Never consult extra attributes or caller traversal/copy protocols. Invalid
    measurement floats remain data for M18's local exclusion policy. Counting
    encodes only validated primitive scalars, never arbitrary host objects.
    """
    used, ancestors = 0, set()

    def charge(size):
        nonlocal used
        used += size
        if used > limit:
            raise ValueError("comparison input exceeds byte budget")

    def scalar(item):
        if type(item) is str and len(item) > limit:
            raise ValueError("comparison input string exceeds budget")
        if type(item) is int and item.bit_length() > limit * 4:
            raise ValueError("comparison input integer exceeds budget")
        charge(len(json.dumps(item, ensure_ascii=False, separators=(",", ":")).encode("utf-8")))
        return item

    def visit(item, depth):
        kind = type(item)
        # Identity checks also avoid a custom metaclass's equality protocol.
        if item is None or any(kind is allowed for allowed in (str, int, float, bool)):
            return scalar(item)
        if not any(kind is allowed for allowed in (dict, list, tuple, *_DATA_TYPES)):
            raise TypeError("comparison input requires exact data types")
        if depth > max_depth or id(item) in ancestors:
            raise ValueError("comparison input exceeds depth or contains a cycle")
        ancestors.add(id(item))
        try:
            charge(2)
            if kind is list or kind is tuple:
                copied = []
                for index, child in enumerate(item):
                    charge(int(index > 0))
                    copied.append(visit(child, depth + 1))
                return tuple(copied) if kind is tuple else copied
            entries = item.items() if kind is dict else (
                (field.name, getattr(item, field.name)) for field in fields(item))
            copied = {}
            for index, (name, child) in enumerate(entries):
                if type(name) is not str:
                    raise TypeError("comparison input mapping keys must be strings")
                charge(1 + int(index > 0))
                scalar(name)
                copied[name] = visit(child, depth + 1)
            return copied if kind is dict else kind(**copied)
        finally:
            ancestors.remove(id(item))

    result = visit(value, 1)
    return result, used


def _snapshot_inputs(group_a, group_b, targets):
    a, size_a = _data_snapshot(group_a, limit=_GROUP_LIMIT)
    b, size_b = _data_snapshot(group_b, limit=_GROUP_LIMIT)
    mapping, size_mapping = _data_snapshot(targets, limit=_MAPPING_LIMIT, max_depth=3)
    wrapper_size = len(b'{"group_a":,"group_b":,"targets":}')
    if size_a + size_b + size_mapping + wrapper_size > _TOTAL_LIMIT:
        raise ValueError("total comparison input exceeds 4 MiB")
    return a, b, mapping


def _text(value):
    if type(value) is not str:
        raise TypeError("target identifiers must be strings")
    if not value or value != value.strip():
        raise ValueError("target identifiers must be nonblank without surrounding whitespace")


def _source_targets(group):
    if type(group) is not dict:
        raise TypeError("group envelope must be a dict")
    if group.keys() != {"executions", "group_declaration", "targets"}:
        raise ValueError("invalid group envelope fields")
    targets = group["targets"]
    if type(targets) not in (list, tuple) or not 1 <= len(targets) <= 32:
        raise ValueError("each group requires 1..32 source targets")
    result = {}
    for target in targets:
        if type(target) is not dict or target.keys() != {"target_id", "section", "members"}:
            raise ValueError("invalid source target fields")
        name = target["target_id"]
        _text(name)
        if name in result:
            raise ValueError("duplicate source target")
        result[name] = target
    return result


def _validate_mapping(a, b, targets):
    if type(targets) not in (list, tuple) or not 1 <= len(targets) <= 32:
        raise ValueError("comparison requires 1..32 target mappings")
    seen, used_a, used_b = set(), set(), set()
    for entry in targets:
        if type(entry) is not dict or entry.keys() != {"target_id", "a_target_id", "b_target_id"}:
            raise ValueError("invalid comparison mapping fields")
        for name in entry.values():
            _text(name)
        name, left, right = entry["target_id"], entry["a_target_id"], entry["b_target_id"]
        if name in seen or left in used_a or right in used_b:
            raise ValueError("duplicate comparison or source mapping")
        if left not in a or right not in b:
            raise ValueError("unknown source target")
        seen.add(name)
        used_a.add(left)
        used_b.add(right)
    if used_a != a.keys() or used_b != b.keys():
        raise ValueError("every source target must be mapped exactly once")
    return sorted(targets, key=lambda item: item["target_id"])


def _execution_key(execution):
    return (execution.experiment.namespace, execution.experiment.experiment_id, execution.execution_id)


def _execution_json(execution):
    return {"experiment": {"namespace": execution.experiment.namespace,
                           "experiment_id": execution.experiment.experiment_id},
            "execution_id": execution.execution_id}


def _group_projection(group, aggregate):
    declaration = group["group_declaration"]
    result = {name: aggregate["group"][name] for name in (
        "group_id", "experiment_identity", "membership_source", "aggregation_eligibility",
        "coverage", "seed_summary")}
    result.update(
        aggregation_policy=declaration["aggregation_policy"],
        training_protocol_ref=declaration["training_protocol_ref"],
        expected_executions=[{"execution": _execution_json(entry["execution"]),
                              "expected_seed": entry["expected_seed"]}
                             for entry in sorted(declaration["expected_executions"],
                                                 key=lambda entry: _execution_key(entry["execution"]))],
    )
    return result


def _finite(value):
    return type(value) is int or (type(value) is float and math.isfinite(value))


def _operand(snapshot, target, member):
    """Attach observed values to declarations, without applying metric policy.

    These values are only M11 operands, never inputs to group arithmetic. Even
    unavailable observations retain their supplied provenance and declarations.
    """
    value, epoch = None, None
    evidence = snapshot["evidence"]
    if snapshot["status"] == "available" and type(evidence) is dict:
        section = target["section"]
        meta = evidence.get("test_results" if section == "test" else "validation_evidence")
        if type(meta) is dict and (section == "validation" or meta.get("status") == "available"):
            metrics = meta.get("metrics") if section == "test" else evidence.get("validation_metrics")
            metric = metrics.get(member["metric_name"]) if type(metrics) is dict else None
            if type(metric) is dict:
                candidate = metric.get("value" if section == "test" else "best_value")
                if _finite(candidate):
                    value = candidate
                if section == "validation" and type(metric.get("best_epoch")) is int:
                    epoch = metric["best_epoch"]
    spec = member["metric_spec"]
    return MetricResultProvenance(
        metric_name=member["metric_name"], value=value, provenance=snapshot["provenance"],
        direction=None if spec is None else spec.direction, best_epoch=epoch,
        **{name: member[name] for name in (
            "metric_definition", "aggregation", "evaluation_protocol", "selection_protocol")},
    )


def _operands(group, target):
    members = {_execution_key(member["execution"]): member for member in target["members"]}
    return [_operand(snapshot, target, members[_execution_key(snapshot["execution"])])
            for snapshot in sorted(group["executions"], key=lambda item: _execution_key(item["execution"]))]


def _cross_assessment(left, right):
    statuses, reasons, count = set(), set(), 0
    for a in left:
        for b in right:
            assessment = m11.compare_metric_results(a, b)
            statuses.add(assessment.status)
            reasons.update(assessment.reason_codes)
            count += 1
    if count == 0:
        reasons.add("no_supplied_cross_pairs")
    status = ("incompatible" if "incompatible" in statuses else
              "unknown" if "unknown" in statuses or count == 0 else "comparable")
    return {"status": status, "policy_version": "metric-result-v1",
            "reason_codes": sorted(reasons), "assessed_pair_count": count}


def _comparison(a, b, metric, aligned, overlap, direction):
    reasons = set(metric["reason_codes"])
    blocked = not aligned or overlap or metric["status"] == "incompatible"
    insufficient = metric["status"] == "unknown"
    for side in (a, b):
        eligibility = side["aggregation_eligibility"]
        reasons.update(eligibility["reason_codes"])
        blocked |= eligibility["status"] == "blocked"
        insufficient |= eligibility["status"] == "insufficient"
    if not aligned:
        reasons.add("section_mismatch")
    if overlap:
        reasons.add("overlapping_execution_identity")

    ca, cb = a["coverage"], b["coverage"]
    coverage = {"a_complete": ca["included_count"] == ca["expected_count"],
                "b_complete": cb["included_count"] == cb["expected_count"],
                "same_expected_count": ca["expected_count"] == cb["expected_count"],
                "same_included_count": ca["included_count"] == cb["included_count"]}
    limited = False
    if not coverage["a_complete"] or not coverage["b_complete"]:
        limited = True
        reasons.add("incomplete_coverage")
    if not coverage["same_expected_count"]:
        limited = True
        reasons.add("expected_roster_size_mismatch")
    if not coverage["same_expected_count"] or not coverage["same_included_count"]:
        limited = True
        reasons.add("coverage_mismatch")
    if ca["included_count"] == 1 or cb["included_count"] == 1:
        limited = True
        reasons.add("insufficient_variability")

    mean_a, mean_b = a["statistics"]["mean"], b["statistics"]["mean"]
    if not _finite(mean_a) or not _finite(mean_b) or min(ca["included_count"], cb["included_count"]) == 0:
        insufficient = True
        reasons.add("mean_unavailable")
    delta, interpretation = None, None
    if not blocked and not insufficient:
        try:
            delta = mean_b - mean_a
            if not _finite(delta):
                raise ValueError("nonfinite subtraction")
        except (OverflowError, ValueError):
            blocked, delta = True, None
            reasons.add("numeric_overflow")
        if delta is not None and direction is not None:
            if delta == 0:
                interpretation = "equal"
            else:
                favorable = delta > 0 if direction == "maximize" else delta < 0
                interpretation = "b_favorable" if favorable else "a_favorable"
    eligibility = ("blocked" if blocked else "insufficient" if insufficient else
                   "limited" if limited else "eligible")
    return {"eligibility": eligibility, "reason_codes": sorted(reasons),
            "calculation_status": "calculated" if delta is not None else "blocked",
            "raw_delta": delta, "direction": direction, "direction_interpretation": interpretation,
            "coverage_relation": coverage}


def compare_replicate_groups(group_a, group_b, *, targets) -> dict:
    """Compare each group's own M18 means under explicit unpaired semantics.

    All inputs are safely detached before either aggregation call. Delegates
    receive separate snapshots so their mutation cannot change host data or the
    declarations used for cross-group assessment. No results are cached.
    """
    a, b, mapping = _snapshot_inputs(group_a, group_b, targets)
    sources_a, sources_b = _source_targets(a), _source_targets(b)
    mapping = _validate_mapping(sources_a, sources_b, mapping)
    aggregates = []
    for group in (a, b):
        delegated, _ = _data_snapshot(group, limit=_GROUP_LIMIT)
        aggregates.append(m18.aggregate_replicate_evidence(
            delegated["executions"], group_declaration=delegated["group_declaration"],
            targets=delegated["targets"]))
    # M18 has now validated the complete group declarations and selectors.
    rosters = [{_execution_key(entry["execution"])
                for entry in (*group["group_declaration"]["expected_executions"], *group["executions"])}
               for group in (a, b)]
    overlap = bool(rosters[0] & rosters[1])
    rows = {}
    for entry in mapping:
        ta, tb = sources_a[entry["a_target_id"]], sources_b[entry["b_target_id"]]
        sides = []
        for target, aggregate in zip((ta, tb), aggregates):
            source = aggregate["targets"][target["target_id"]]
            sides.append({"source_target_id": target["target_id"], "section": target["section"],
                          **{name: source[name] for name in _TARGET_FIELDS}})
        left, right = _operands(a, ta), _operands(b, tb)
        metric = _cross_assessment(left, right)
        directions = {operand.direction for operand in (*left, *right)}
        direction = next(iter(directions)) if len(directions) == 1 else None
        if direction not in ("maximize", "minimize"):
            direction = None
        aligned = ta["section"] == tb["section"]
        comparison = _comparison(*sides, metric, aligned, overlap, direction)
        rows[entry["target_id"]] = {
            "a": sides[0], "b": sides[1],
            "alignment": {"status": "aligned" if aligned else "blocked",
                          "reason_codes": [] if aligned else ["section_mismatch"]},
            "metric_comparability": metric, "comparison": comparison,
            "limitations": sorted(set(sides[0]["limitations"]) | set(sides[1]["limitations"]) |
                                  set(comparison["reason_codes"])),
        }
    result = {
        "schema_version": "replicate-group-comparison-v1",
        "comparison_policy": "unpaired-descriptive-v1",
        "groups": {"a": _group_projection(a, aggregates[0]), "b": _group_projection(b, aggregates[1])},
        "targets": rows,
        "limitations": sorted(set(aggregates[0]["limitations"]) | set(aggregates[1]["limitations"])),
    }
    encoded = json.dumps(result, ensure_ascii=False, allow_nan=False, separators=(",", ":"))
    if len(encoded.encode("utf-8")) > _OUTPUT_LIMIT:
        raise ValueError("comparison output exceeds 128 KiB")
    return result

"""Descriptive comparison of explicitly paired, host-declared executions.

M18 owns full-group eligibility and observation selection; M11 owns metric
comparability. Pair declarations do not establish experimental control.
"""

import json
import math
import statistics

import experiment_comparability as m11
import experiment_replicates as m18
from experiment_identity import (
    ExperimentIdentity, ExecutionIdentity, ExperimentProvenance, MetricResultProvenance,
)
from metrics import MetricSpec


_PROVENANCE = (
    "execution", "configuration_ref", "task", "target", "model_ref",
    "dataset_version", "split_id", "source_revision",
    "initialization_checkpoint_ref", "realized_seed", "environment_ref",
    "location", "recorded_at",
)
_DECLARATIONS = ("metric_definition", "aggregation", "evaluation_protocol", "selection_protocol")
_CLAIMS = (
    "same_seed_means_controlled", "paired_delta_is_causal",
    "positive_mean_is_significant", "all_positive_is_universally_robust",
    "host_factor_is_verified_only_change", "partial_coverage_is_complete",
    "missing_metric_means_zero", "unpaired_run_is_invalid",
    "explicit_pair_verifies_rationale", "full_coverage_is_unbiased",
    "artifact_equality_establishes_pairing",
)
_LIMITS = {
    "statistical_inference_not_performed", "independence_not_verified",
    "roster_host_declared_not_preregistered", "snapshot_currentness_not_verified",
    "pairing_host_declared_not_verified", "causal_attribution_not_performed",
    "controlled_factors_not_verified",
}


def _measurement_path(path):
    # The only nonfinite input exception is an actual M16 measurement slot.
    return (len(path) == 7 and path[0] == "executions" and type(path[1]) is int
            and path[2:5] == ("evidence", "test_results", "metrics")
            and path[-1] == "value") or (
        len(path) == 6 and path[0] == "executions" and type(path[1]) is int
        and path[2:4] == ("evidence", "validation_metrics") and path[-1] == "best_value")


def _snapshot(value, limit, depth_limit):
    """Validate before traversal/copy protocols, charging serialized bytes.

    Only exact builtins are traversed. In particular, test key types before
    hashing, and compare type objects by identity rather than metaclass equality.
    """
    used, ancestors = 0, set()

    def charge(count):
        nonlocal used
        used += count
        if used > limit:
            raise ValueError("paired comparison input exceeds byte budget")

    def scalar(item):
        if type(item) is str and len(item) > limit:
            raise ValueError("paired comparison input string exceeds budget")
        if type(item) is int and item.bit_length() > limit * 4:
            raise ValueError("paired comparison input integer exceeds budget")
        try:
            charge(len(json.dumps(item, ensure_ascii=False, separators=(",", ":")).encode("utf-8")))
        except (UnicodeError, ValueError):
            raise ValueError("paired comparison input encoding or budget invalid") from None
        return item

    def visit(item, depth, path):
        kind = type(item)
        if item is None or any(kind is allowed for allowed in (str, bool, int, float)):
            if kind is float and not math.isfinite(item) and not _measurement_path(path):
                raise ValueError("nonfinite input metadata")
            return scalar(item)
        if kind is not dict and kind is not list:
            raise TypeError("paired comparison requires exact plain data")
        if depth > depth_limit or id(item) in ancestors:
            raise ValueError("paired comparison input depth or cycle invalid")
        ancestors.add(id(item))
        try:
            charge(2)
            if kind is list:
                result = []
                for index, child in enumerate(item):
                    charge(int(index > 0))
                    result.append(visit(child, depth + 1, (*path, index)))
                return result
            result = {}
            for index, (name, child) in enumerate(item.items()):
                if type(name) is not str:
                    raise TypeError("paired comparison keys must be exact strings")
                charge(1 + int(index > 0))
                scalar(name)
                result[name] = visit(child, depth + 1, (*path, name))
            return result
        finally:
            ancestors.remove(id(item))

    return visit(value, 1, ()), used


def _clone(value):
    """Copy already validated plain data, never external protocol objects."""
    if type(value) is dict:
        return {k: _clone(v) for k, v in value.items()}
    if type(value) is list:
        return [_clone(v) for v in value]
    return value


def _record(value, required, optional=()):
    if type(value) is not dict:
        raise TypeError("record must be an exact dictionary")
    if not set(required) <= value.keys() or value.keys() - set(required) - set(optional):
        raise ValueError("invalid record fields")


def _sequence(value, minimum, maximum):
    if type(value) is not list:
        raise TypeError("sequence must be an exact list")
    if not minimum <= len(value) <= maximum:
        raise ValueError("invalid sequence count")


def _text(value, limit=128):
    if type(value) is not str:
        raise TypeError("identifier must be an exact string")
    if not value or value != value.strip() or len(value.encode("utf-8")) > limit:
        raise ValueError("invalid identifier or text bound")


def _experiment(value):
    _record(value, ("namespace", "experiment_id"))
    for item in value.values():
        _text(item)
    return ExperimentIdentity(**value)


def _identity(value):
    _record(value, ("experiment", "execution_id"))
    _text(value["execution_id"])
    return ExecutionIdentity(_experiment(value["experiment"]), value["execution_id"])


def _key(value):
    return (value["experiment"]["namespace"], value["experiment"]["experiment_id"], value["execution_id"])


def _bridge(group):
    """Construct project models only from detached, validated plain fields.

    M18 retains responsibility for group membership, selector and policy
    validation. This bridge does not invent missing provenance declarations.
    """
    group = _clone(group)
    _record(group, ("executions", "group_declaration", "targets"))
    declaration = group["group_declaration"]
    _record(declaration, ("group_id", "experiment_identity", "expected_executions",
                          "training_protocol_ref", "membership_selection", "aggregation_policy"))
    _text(declaration["group_id"])
    declaration["experiment_identity"] = _experiment(declaration["experiment_identity"])
    _sequence(declaration["expected_executions"], 1, 16)
    for entry in declaration["expected_executions"]:
        _record(entry, ("execution", "expected_seed"))
        entry["execution"] = _identity(entry["execution"])
    _sequence(group["executions"], 0, 16)
    for snapshot in group["executions"]:
        _record(snapshot, ("execution", "provenance", "training_protocol_ref", "status",
                           "evidence", "snapshot_metadata"))
        snapshot["execution"] = _identity(snapshot["execution"])
        provenance = snapshot["provenance"]
        _record(provenance, _PROVENANCE)
        provenance["execution"] = _identity(provenance["execution"])
        snapshot["provenance"] = ExperimentProvenance(**provenance)
    _sequence(group["targets"], 1, 32)
    for target in group["targets"]:
        _record(target, ("target_id", "section", "members"))
        _text(target["target_id"])
        _sequence(target["members"], 1, 16)
        for member in target["members"]:
            _record(member, ("execution", "metric_name", "metric_spec", *_DECLARATIONS))
            member["execution"] = _identity(member["execution"])
            spec = member["metric_spec"]
            if spec is not None:
                _record(spec, ("name", "path", "direction", "display_name", "precision"))
                if type(spec["path"]) is not list:
                    raise TypeError("metric path must be an exact list")
                # The original MetricSpec validates path semantics. Its list
                # already falls within the group snapshot's byte/depth bounds.
                spec["path"] = tuple(spec["path"])
                member["metric_spec"] = MetricSpec(**spec)
    return group


def _pairing(a, b, declaration):
    _record(declaration, ("schema_version", "pairing_source", "pairs"), ("variant_context",))
    if declaration["schema_version"] != "explicit-pairing-v1" or declaration["pairing_source"] != "host_declared":
        raise ValueError("unsupported pairing declaration")
    pairs = declaration["pairs"]
    _sequence(pairs, 1, 16)
    rosters = [{_key(e["execution"]): e["execution"] for e in g["group_declaration"]["expected_executions"]}
               for g in (a, b)]
    if rosters[0].keys() & rosters[1].keys():
        raise ValueError("overlapping group execution identities")
    ids, used_a, used_b = set(), set(), set()
    for pair in pairs:
        _record(pair, ("pair_id", "a_execution", "b_execution"), ("declared_pair_basis",))
        _text(pair["pair_id"])
        _identity(pair["a_execution"])
        _identity(pair["b_execution"])
        left, right = _key(pair["a_execution"]), _key(pair["b_execution"])
        if pair["pair_id"] in ids or left in used_a or right in used_b:
            raise ValueError("pairing must be one-to-one")
        if left not in rosters[0] or right not in rosters[1]:
            raise ValueError("pair endpoint outside expected side roster")
        if "declared_pair_basis" in pair:
            _text(pair["declared_pair_basis"], 256)
        ids.add(pair["pair_id"])
        used_a.add(left)
        used_b.add(right)
    context = declaration.get("variant_context")
    if "variant_context" in declaration:
        _record(context, (), ("baseline_label", "candidate_label", "host_declared_changed_factor"))
        for value in context.values():
            _text(value, 256)
        context = {**context, "source": "host_declared"}
    coverage = {"declared_pair_count": len(pairs)}
    for side, roster, used in zip(("a", "b"), rosters, (used_a, used_b)):
        coverage.update({side + "_expected_count": len(roster), side + "_referenced_count": len(used),
                         side + "_pairing_coverage": len(used) / len(roster),
                         "unpaired_" + side: [roster[k] for k in sorted(roster.keys() - used)]})
    return sorted(pairs, key=lambda p: p["pair_id"]), coverage, context


def _mapping(a, b, targets):
    _sequence(targets, 1, 32)
    source_a = {t["target_id"]: t for t in a["targets"]}
    source_b = {t["target_id"]: t for t in b["targets"]}
    seen, used_a, used_b = set(), set(), set()
    for entry in targets:
        _record(entry, ("target_id", "a_target_id", "b_target_id"))
        for value in entry.values():
            _text(value)
        name, left, right = (entry[k] for k in ("target_id", "a_target_id", "b_target_id"))
        if name in seen or left in used_a or right in used_b:
            raise ValueError("duplicate target mapping")
        seen.add(name)
        used_a.add(left)
        used_b.add(right)
    if used_a != source_a.keys() or used_b != source_b.keys():
        raise ValueError("mapping must cover each source target exactly once")
    return sorted(targets, key=lambda t: t["target_id"]), source_a, source_b


def _finite(value):
    return type(value) is int or (type(value) is float and math.isfinite(value))


def _operand(snapshot, target, member):
    """M11 receives original declarations, even if the measurement is absent.

    This extraction supplies M11 traceability only. Arithmetic uses exclusively
    M18's included observations, never this raw value or a group mean.
    """
    value, epoch = None, None
    evidence, section = snapshot["evidence"], target["section"]
    if snapshot["status"] == "available" and type(evidence) is dict:
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
    provenance = _clone(snapshot["provenance"])
    provenance["execution"] = _identity(provenance["execution"])
    spec = member["metric_spec"]
    return MetricResultProvenance(
        metric_name=member["metric_name"], value=value, best_epoch=epoch,
        provenance=ExperimentProvenance(**provenance),
        direction=None if spec is None else spec["direction"],
        **{name: member[name] for name in _DECLARATIONS},
    )


def _sides(target):
    result = {}
    for state in ("included", "missing", "excluded"):
        for item in target[state + "_executions"]:
            result[_key(item["execution"])] = (
                {"availability": state, "value": item.get("value"), "source": item.get("source")},
                item.get("reason_codes", []),
            )
    return result


def _statistics(values):
    result = dict(count=len(values), mean=None, median=None, sample_std=None, min=None, max=None,
                  positive_count=0, zero_count=0, negative_count=0)
    if values:
        ordered = sorted(values)
        n = len(values)
        result.update(mean=statistics.mean(values), median=statistics.mean(ordered[(n - 1) // 2:n // 2 + 1]),
                      sample_std=statistics.stdev(values) if n > 1 else None,
                      min=ordered[0], max=ordered[-1], positive_count=sum(v > 0 for v in values),
                      zero_count=sum(v == 0 for v in values), negative_count=sum(v < 0 for v in values))
        if any(v is not None and not _finite(v) for v in result.values()):
            raise ValueError("nonfinite paired statistic")
    return result


def _target(a, b, source_a, source_b, result_a, result_b, pairs, unpaired):
    sides_a, sides_b = _sides(result_a), _sides(result_b)
    snapshots_a = {_key(s["execution"]): s for s in a["executions"]}
    snapshots_b = {_key(s["execution"]): s for s in b["executions"]}
    members_a = {_key(m["execution"]): m for m in source_a["members"]}
    members_b = {_key(m["execution"]): m for m in source_b["members"]}
    reasons, limitations, statuses = set(), set(_LIMITS), set()
    for result in (result_a, result_b):
        statuses.add(result["aggregation_eligibility"]["status"])
        reasons.update(result["aggregation_eligibility"]["reason_codes"])
        limitations.update(result["limitations"])
    if source_a["section"] != source_b["section"]:
        statuses.add("blocked")
        reasons.add("section_mismatch")
    rows = {}
    for pair in pairs:
        left, right = _key(pair["a_execution"]), _key(pair["b_execution"])
        side_a, codes_a = sides_a[left]
        side_b, codes_b = sides_b[right]
        codes = set(codes_a) | set(codes_b)
        comparison = None
        if left in snapshots_a and right in snapshots_b:
            assessment = m11.compare_metric_results(
                _operand(snapshots_a[left], source_a, members_a[left]),
                _operand(snapshots_b[right], source_b, members_b[right]),
            )
            comparison = {"status": assessment.status, "reason_codes": sorted(set(assessment.reason_codes)),
                          "policy_version": assessment.policy_version}
            codes.update(assessment.reason_codes)
            reasons.update(assessment.reason_codes)
            if assessment.status == "incompatible":
                statuses.add("blocked")
            elif assessment.status == "unknown":
                statuses.add("insufficient")
        missing_a, missing_b = side_a["availability"] == "missing", side_b["availability"] == "missing"
        eligible = (side_a["availability"] == side_b["availability"] == "included"
                    and comparison is not None and comparison["status"] == "comparable")
        spec = members_a[left]["metric_spec"]
        rows[pair["pair_id"]] = {
            "a_execution": pair["a_execution"], "b_execution": pair["b_execution"],
            "a": side_a, "b": side_b, "metric_comparability": comparison,
            "partition": "missing" if missing_a or missing_b else "included" if eligible else "excluded",
            "missing_a": missing_a, "missing_b": missing_b,
            "calculation": {"status": "blocked", "raw_delta": None,
                            "direction": None if spec is None else spec["direction"],
                            "direction_interpretation": None},
            "reason_codes": sorted(codes),
        }
    blocked, insufficient = "blocked" in statuses, "insufficient" in statuses
    stats = _statistics([])
    if not blocked and not insufficient:
        try:
            deltas = []
            for row in rows.values():
                if row["partition"] != "included":
                    continue
                delta = row["b"]["value"] - row["a"]["value"]
                if not _finite(delta):
                    raise ValueError("nonfinite paired delta")
                calculation = row["calculation"]
                favorable = (delta > 0) == (calculation["direction"] == "maximize")
                calculation.update(status="calculated", raw_delta=delta,
                                   direction_interpretation="equal" if delta == 0 else
                                   "b_favorable" if favorable else "a_favorable")
                deltas.append(delta)
            stats = _statistics(deltas)
        except (ValueError, OverflowError):
            blocked = True
            reasons.add("numeric_error")
    if blocked or insufficient:
        stats = _statistics([])
        for row in rows.values():
            if row["partition"] == "included":
                row["partition"] = "excluded"
            row["calculation"].update(status="blocked", raw_delta=None, direction_interpretation=None)
            row["reason_codes"] = sorted(set(row["reason_codes"]) | reasons)
    count, total = stats["count"], len(pairs)
    if count < total:
        reasons.add("incomplete_pair_coverage")
    if unpaired:
        reasons.add("unpaired_executions")
    if count == 1:
        reasons.add("insufficient_variability")
    status = ("blocked" if blocked else "insufficient" if insufficient or count == 0 else
              "limited" if count < total or unpaired or count == 1 else "eligible")
    coverage = {"declared_pair_count": total, "coverage_ratio": count / total}
    for state in ("included", "missing", "excluded"):
        coverage[state + "_count"] = sum(row["partition"] == state for row in rows.values())
    for side in ("a", "b"):
        coverage["missing_" + side + "_count"] = sum(row["missing_" + side] for row in rows.values())
    return {"pair_results": rows, "coverage": coverage, "statistics": stats,
            "summary_eligibility": {"status": status, "reason_codes": sorted(reasons)},
            "limitations": sorted(limitations)}


def compare_paired_replicates(group_a, group_b, *, pairing_declaration, targets) -> dict:
    """Compare declared pairs using full-group gates and equal-pair statistics.

    Pure bounded data processing: no acquisition, automatic pairing or inference.
    Caller data is validated and detached before entering either frozen policy.
    """
    a, size_a = _snapshot(group_a, 2 * 1024 * 1024, 16)
    b, size_b = _snapshot(group_b, 2 * 1024 * 1024, 16)
    declaration, size_d = _snapshot(pairing_declaration, 16 * 1024, 8)
    mapping, size_m = _snapshot(targets, 16 * 1024, 8)
    wrapper = len(b'{"group_a":,"group_b":,"pairing_declaration":,"targets":}')
    if size_a + size_b + size_d + size_m + wrapper > 4 * 1024 * 1024:
        raise ValueError("total paired comparison input exceeds budget")
    # Each bridge owns its entire tree, separate from the retained cross-pair
    # data. Unpaired members remain present for every M18 eligibility decision.
    internal_a, internal_b = _bridge(a), _bridge(b)
    pairs, coverage, context = _pairing(a, b, declaration)
    mapping, source_a, source_b = _mapping(a, b, mapping)
    result_a = m18.aggregate_replicate_evidence(**internal_a)
    result_b = m18.aggregate_replicate_evidence(**internal_b)
    unpaired = bool(coverage["unpaired_a"] or coverage["unpaired_b"])
    rows = {m["target_id"]: _target(
        a, b, source_a[m["a_target_id"]], source_b[m["b_target_id"]],
        result_a["targets"][m["a_target_id"]], result_b["targets"][m["b_target_id"]], pairs, unpaired,
    ) for m in mapping}
    result = {
        "schema_version": "paired-replicate-comparison-v1",
        "comparison_policy": "explicit-paired-descriptive-v1", "pairing_source": "host_declared",
        "groups": {"a": result_a["group"], "b": result_b["group"]},
        "pairing_coverage": coverage, "targets": rows, "variant_context": context,
        "claim_boundaries": dict.fromkeys(_CLAIMS, False),
        "limitations": sorted(_LIMITS | set(result_a["limitations"]) | set(result_b["limitations"])),
    }
    encoded = json.dumps(result, ensure_ascii=False, allow_nan=False, sort_keys=True, separators=(",", ":"))
    if len(encoded.encode("utf-8")) > 128 * 1024:
        raise ValueError("paired comparison output exceeds 128 KiB")
    return json.loads(encoded)

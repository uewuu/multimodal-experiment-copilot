"""Bounded descriptive aggregation of one explicitly declared replicate group.

Inputs are offline snapshots, not requests to acquire evidence. M11 owns the
identity and metric policies; agreement of declarations verifies neither their
truth nor independence. Missing observations never become numeric stand-ins.
"""

from dataclasses import asdict, fields
from itertools import combinations
import json
import math
import statistics

import experiment_comparability as m11
import experiment_identity as identities
from experiment_identity import (
    ExperimentIdentity, ExecutionIdentity, ExperimentProvenance,
    MetricResultProvenance,
)
from metrics import MetricSpec


_DECLARATIONS = (
    "metric_definition", "aggregation", "evaluation_protocol", "selection_protocol",
)
_ROOT_LIMITATIONS = {
    "statistical_inference_not_performed", "independence_not_verified",
    "roster_host_declared_not_preregistered", "snapshot_currentness_not_verified",
}
_INPUT_LIMIT = 2 * 1024 * 1024
_OUTPUT_LIMIT = 128 * 1024
_DATA_TYPES = (ExperimentIdentity, ExecutionIdentity, ExperimentProvenance, MetricSpec)


def _check_input_budget(envelope):
    """Count the exact compact JSON projection before copying any input tree.

    Only concrete containers and known dataclasses participate. Counting is
    incremental, including unused metadata; recursion and allocation are bounded.
    Nonfinite measurement tokens can be counted here, but never enter the output.
    """
    total = 0
    ancestors = set()

    def charge(count):
        nonlocal total
        total += count
        if total > _INPUT_LIMIT:
            raise ValueError("replicate input exceeds 2 MiB")

    def scalar(value):
        if type(value) is str and len(value) > _INPUT_LIMIT:
            raise ValueError("replicate input exceeds 2 MiB")
        if type(value) is int and value.bit_length() > _INPUT_LIMIT * 4:
            raise ValueError("replicate input integer exceeds budget")
        charge(len(json.dumps(value, ensure_ascii=False,
                              separators=(",", ":")).encode("utf-8")))

    def visit(value, depth):
        kind = type(value)
        if value is None or kind in (str, bool, int, float):
            scalar(value)
            return
        if kind not in (dict, list, tuple, *_DATA_TYPES):
            raise TypeError("input requires concrete JSON containers or known dataclasses")
        if depth > 16 or id(value) in ancestors:
            raise ValueError("input exceeds structure depth or contains a cycle")
        ancestors.add(id(value))
        try:
            charge(2)
            if kind in (list, tuple):
                for index, item in enumerate(value):
                    charge(int(index > 0))
                    visit(item, depth + 1)
            else:
                entries = value.items() if kind is dict else (
                    (field.name, getattr(value, field.name)) for field in fields(value))
                for index, (key, item) in enumerate(entries):
                    if type(key) is not str:
                        raise TypeError("input mapping keys must be strings")
                    charge(1 + int(index > 0))
                    scalar(key)
                    visit(item, depth + 1)
        finally:
            ancestors.remove(id(value))

    visit(envelope, 1)


def _text(value, name, *, optional=False):
    if optional and value is None:
        return
    if type(value) is not str:
        raise TypeError(f"{name} must be a string")
    if not value or value != value.strip():
        raise ValueError(f"{name} must be nonblank without surrounding whitespace")


def _record(value, keys, name):
    if type(value) is not dict:
        raise TypeError(f"{name} must be a dict")
    if value.keys() != keys:
        raise ValueError(f"invalid {name} fields")


def _sequence(value, name, maximum, minimum=0):
    if type(value) not in (list, tuple):
        raise TypeError(f"{name} must be a concrete sequence")
    if not minimum <= len(value) <= maximum:
        raise ValueError(f"{name} count must be between {minimum} and {maximum}")


def _experiment(value):
    if type(value) is not ExperimentIdentity:
        raise TypeError("experiment_identity must be ExperimentIdentity")
    _text(value.namespace, "namespace")
    _text(value.experiment_id, "experiment_id")


def _execution(value):
    if type(value) is not ExecutionIdentity:
        raise TypeError("execution must be ExecutionIdentity")
    _experiment(value.experiment)
    _text(value.execution_id, "execution_id")
    return (value.experiment.namespace, value.experiment.experiment_id, value.execution_id)


def _seed(value):
    if value is not None and type(value) is not int:
        raise TypeError("seed must be an integer or None")


def _index(entries):
    result = {}
    for entry in entries:
        key = _execution(entry["execution"])
        if key in result:
            raise ValueError("duplicate ExecutionIdentity")
        result[key] = entry
    return dict(sorted(result.items()))


def _validate(executions, group, targets):
    _record(group, {"group_id", "experiment_identity", "expected_executions",
                    "training_protocol_ref", "membership_selection", "aggregation_policy"}, "group")
    for name in ("group_id", "membership_selection", "aggregation_policy"):
        _text(group[name], name)
    if group["aggregation_policy"] != "replicate-descriptive-v1":
        raise ValueError("unsupported aggregation policy")
    _experiment(group["experiment_identity"])
    _text(group["training_protocol_ref"], "training_protocol_ref", optional=True)
    _sequence(group["expected_executions"], "expected executions", 16, 1)
    for entry in group["expected_executions"]:
        _record(entry, {"execution", "expected_seed"}, "roster entry")
        _seed(entry["expected_seed"])
    roster = _index(group["expected_executions"])
    _sequence(executions, "executions", 16)
    for entry in executions:
        _record(entry, {"execution", "provenance", "training_protocol_ref", "status",
                        "evidence", "snapshot_metadata"}, "snapshot")
        _execution(entry["execution"])
        provenance = entry["provenance"]
        if type(provenance) is not ExperimentProvenance:
            raise TypeError("provenance must be ExperimentProvenance")
        if provenance.execution != entry["execution"]:
            raise ValueError("snapshot and provenance execution must match")
        for field in fields(provenance):
            value = getattr(provenance, field.name)
            if field.name == "realized_seed":
                _seed(value)
            elif field.name != "execution":
                _text(value, field.name, optional=True)
        _text(entry["training_protocol_ref"], "training_protocol_ref", optional=True)
        if entry["status"] not in ("available", "unavailable"):
            raise ValueError("invalid snapshot status")
        if entry["evidence"] is not None and type(entry["evidence"]) is not dict:
            raise TypeError("evidence must be a dict or None")
        metadata = entry["snapshot_metadata"]
        if type(metadata) is not dict:
            raise TypeError("snapshot_metadata must be a dict")
        for name in ("snapshot_id", "acquisition", "binding"):
            _text(metadata.get(name), name)
        if metadata["binding"] != "host_declared":
            raise ValueError("snapshot binding must be host_declared")
    snapshots = _index(executions)
    if snapshots.keys() - roster.keys():
        raise ValueError("unexpected execution outside roster")
    _sequence(targets, "targets", 32, 1)
    selected = {}
    for target in targets:
        _record(target, {"target_id", "section", "members"}, "target")
        _text(target["target_id"], "target_id")
        if target["target_id"] in selected:
            raise ValueError("duplicate target_id")
        if target["section"] not in ("validation", "test"):
            raise ValueError("section must be validation or test")
        _sequence(target["members"], "target members", 16, 1)
        for member in target["members"]:
            _record(member, {"execution", "metric_name", "metric_spec", *_DECLARATIONS}, "selector")
            _text(member["metric_name"], "metric_name")
            for name in _DECLARATIONS:
                _text(member[name], name, optional=True)
            spec = member["metric_spec"]
            if spec is not None:
                if type(spec) is not MetricSpec:
                    raise TypeError("metric_spec must be MetricSpec or None")
                if spec.name != member["metric_name"]:
                    raise ValueError("metric_spec name must match selector")
                if target["section"] == "test" and len(spec.path) != 1:
                    raise ValueError("Test selector must specify one column")
        members = _index(target["members"])
        if members.keys() != roster.keys():
            raise ValueError("selectors must match the entire expected roster")
        selected[target["target_id"]] = (target["section"], members)
    return roster, snapshots, dict(sorted(selected.items()))


def _pairs(items):
    return [(items[0], items[0])] if len(items) == 1 else combinations(items, 2)


def _assessment(status, reasons):
    return {"status": status, "reason_codes": sorted(set(reasons))}


def _member_reason(execution, reasons):
    return {"execution": asdict(execution), "reason_codes": sorted(set(reasons))}


def _group_assessment(group, roster, snapshots):
    reasons, identity_reasons, limitations = set(), set(), set(_ROOT_LIMITATIONS)
    identity_statuses = set()
    if any(entry["execution"].experiment != group["experiment_identity"] for entry in roster.values()):
        identity_statuses.add("conflict")
        identity_reasons.add("experiment_identity_mismatch")
    provenance = [entry["provenance"] for entry in snapshots.values()]
    for left, right in _pairs(provenance):
        if left.execution.experiment != right.execution.experiment:
            identity_statuses.add("conflict")
            identity_reasons.add("experiment_identity_mismatch")
            continue
        result = identities.check_identity_consistency(left, right)
        identity_statuses.add(result.status)
        identity_reasons.update(result.reason_codes)
    if not provenance:
        identity_statuses.add("unknown")
        identity_reasons.add("no_supplied_executions")
    identity_status = ("conflict" if "conflict" in identity_statuses else
                       "unknown" if "unknown" in identity_statuses else "consistent")
    blocked, insufficient = identity_status == "conflict", identity_status == "unknown"
    reasons.update(identity_reasons)
    protocols = [group["training_protocol_ref"], *(
        entry["training_protocol_ref"] for entry in snapshots.values())]
    if None in protocols:
        insufficient = True
        reasons.add("missing_training_protocol_ref")
    if len({value for value in protocols if value is not None}) > 1:
        blocked = True
        reasons.add("training_protocol_ref_mismatch")
    seeds, unknown = {}, 0
    for key, entry in snapshots.items():
        seed = entry["provenance"].realized_seed
        expected = roster[key]["expected_seed"]
        if seed is None:
            unknown += 1
        else:
            seeds.setdefault(seed, []).append(asdict(entry["execution"]))
            if expected is not None and expected != seed:
                blocked = True
                reasons.add("expected_seed_mismatch")
    repeated = [{"seed": seed, "executions": members} for seed, members in sorted(seeds.items())
                if len(members) > 1]
    if repeated:
        limitations.add("repeated_seed")
    if unknown:
        limitations.add("seed_unknown")
    environments = {item.environment_ref for item in provenance}
    if None in environments or len(environments) > 1:
        limitations.add("environment_not_controlled")
    available = {key for key, entry in snapshots.items() if _available(entry)}
    if not available:
        insufficient = True
        reasons.add("no_available_snapshots")
    status = "blocked" if blocked else "insufficient" if insufficient else "eligible"
    result = {
        "group_id": group["group_id"], "experiment_identity": asdict(group["experiment_identity"]),
        "membership_source": "host_declared", "group_status": status,
        "identity_consistency": _assessment(identity_status, identity_reasons),
        "aggregation_eligibility": _assessment(status, reasons),
        "coverage": {"expected_count": len(roster), "available_count": len(available),
                     "missing_replicates": [_member_reason(entry["execution"], ["snapshot_unavailable"])
                                            for key, entry in roster.items() if key not in available]},
        "seed_summary": {"execution_count": len(snapshots), "distinct_known_seed_count": len(seeds),
                         "unknown_seed_count": unknown, "repeated_seeds": repeated},
    }
    return result, limitations


def _available(snapshot):
    return snapshot is not None and snapshot["status"] == "available" and snapshot["evidence"] is not None


def _finite(value):
    return type(value) is int or (type(value) is float and math.isfinite(value))


def _observation(snapshot, section, selector):
    """Project one M16 observation; malformed data is a local exclusion."""
    observation = {"state": "missing", "value": None, "source": None,
                   "reasons": ["snapshot_unavailable"]}
    if not _available(snapshot):
        return observation
    evidence = snapshot["evidence"]
    meta = evidence.get("test_results" if section == "test" else "validation_evidence")
    if meta is None:
        observation["reasons"] = ["section_missing"]
        return observation
    observation.update(state="excluded", reasons=["invalid_evidence"])
    if type(meta) is not dict:
        return observation
    status = meta.get("status") if section == "test" else "available"
    if status in ("missing", "unavailable", "not_requested"):
        observation.update(state="missing", reasons=["observation_unavailable"])
        return observation
    if status != "available":
        return observation
    metrics = meta.get("metrics") if section == "test" else evidence.get("validation_metrics")
    if type(metrics) is not dict:
        return observation
    name = selector["metric_name"]
    if name not in metrics:
        observation.update(state="missing", reasons=["metric_missing"])
        return observation
    metric = metrics[name]
    if type(metric) is not dict:
        return observation
    value = metric.get("value" if section == "test" else "best_value")
    if not _finite(value):
        observation["reasons"] = ["invalid_numeric_observation"]
        return observation
    observation["value"] = value
    kind = "csv_reported" if section == "test" else "validation_history"
    if meta.get("evidence_kind") != kind:
        observation["reasons"] = ["evidence_kind_mismatch"]
        return observation
    spec = selector["metric_spec"]
    if section == "test" and spec is not None and (
            metric.get("source_column") != spec.path[0] or metric.get("direction") != spec.direction):
        observation["reasons"] = ["evidence_spec_mismatch"]
        return observation
    origin = meta.get("source")
    if type(origin) is not dict or type(origin.get("file")) is not str:
        return observation
    source = {"section": section, "metric_name": name, "evidence_kind": kind,
              "file": origin["file"], "metadata_source": "host_declared"}
    if section == "test":
        if type(meta.get("provenance")) is not dict:
            return observation
        source.update(provenance=meta["provenance"], data_row=origin.get("data_row"),
                      selection=origin.get("selection"))
    else:
        epoch = metric.get("best_epoch")
        if type(epoch) is not int or metric.get("metric_name") != name:
            return observation
        source["best_epoch"] = epoch
    try:
        # Selected metadata must itself be strict JSON. Invalid unselected
        # numeric evidence never contaminates this detached source projection.
        source = json.loads(json.dumps(source, ensure_ascii=False, allow_nan=False))
    except (ValueError, TypeError):
        return observation
    observation.update(state="included", source=source, reasons=[])
    return observation


def _metric_assessment(snapshots, members, observations):
    declarations = []
    for key, snapshot in snapshots.items():
        member, observation = members[key], observations[key]
        spec = member["metric_spec"]
        # M11 assesses declarations, not arithmetic. None explicitly represents
        # an absent/invalid observation here: never substitute 0 or another run's
        # value. This lets declared conflicts survive missing numeric evidence.
        declarations.append(MetricResultProvenance(
            metric_name=member["metric_name"], value=observation["value"],
            provenance=snapshot["provenance"], direction=None if spec is None else spec.direction,
            best_epoch=(observation["source"] or {}).get("best_epoch"),
            **{name: member[name] for name in _DECLARATIONS},
        ))
    statuses, reasons = set(), set()
    for left, right in _pairs(declarations):
        result = m11.compare_metric_results(left, right)
        statuses.add(result.status)
        reasons.update(result.reason_codes)
    status = ("incompatible" if "incompatible" in statuses else
              "unknown" if "unknown" in statuses or not declarations else "comparable")
    return {**_assessment(status, reasons), "policy_version": "metric-result-v1"}


def _statistics(values):
    result = dict(count=len(values), mean=None, median=None, sample_std=None, min=None, max=None)
    if not values:
        return result
    ordered = sorted(values)
    n = len(ordered)
    middle = ordered[(n - 1) // 2:n // 2 + 1]
    result.update(mean=statistics.mean(values), median=statistics.mean(middle),
                  sample_std=statistics.stdev(values) if n > 1 else None,
                  min=ordered[0], max=ordered[-1])
    if any(value is not None and not _finite(value) for value in result.values()):
        raise ValueError("nonfinite statistic")
    return result


def _target(section, members, roster, snapshots, group, root_limitations):
    observations = {key: _observation(snapshots.get(key), section, members[key]) for key in roster}
    observed = sum(item["value"] is not None for item in observations.values())
    metric = _metric_assessment(snapshots, members, observations)
    reasons = set(group["aggregation_eligibility"]["reason_codes"])
    reasons.update(metric["reason_codes"])
    blocked = group["group_status"] == "blocked" or metric["status"] == "incompatible"
    insufficient = group["group_status"] == "insufficient" or metric["status"] == "unknown"
    limitations = set(root_limitations)
    if section == "test":
        limitations.add("checkpoint_binding_not_verified")
    included, missing, excluded = [], [], []
    for key, observation in observations.items():
        execution = roster[key]["execution"]
        state, codes = observation["state"], observation["reasons"]
        if state == "included" and (blocked or insufficient):
            state, codes = "excluded", reasons or {"aggregation_ineligible"}
        if state == "included":
            included.append({"execution": asdict(execution), "value": observation["value"],
                             "source": observation["source"]})
        else:
            (missing if state == "missing" else excluded).append(_member_reason(execution, codes))
            limitations.update(codes)
    try:
        stats = _statistics([entry["value"] for entry in included])
    except (OverflowError, ValueError):
        blocked = True
        reasons.add("numerical_error")
        excluded.extend(_member_reason(roster[key]["execution"], ["numerical_error"])
                        for key, item in observations.items() if item["state"] == "included")
        excluded.sort(key=lambda item: (item["execution"]["experiment"]["namespace"],
                                       item["execution"]["experiment"]["experiment_id"],
                                       item["execution"]["execution_id"]))
        included, stats = [], _statistics([])
    if not included:
        insufficient = True
        reasons.add("no_eligible_observations")
    if len(included) == 1:
        limitations.add("insufficient_variability")
    eligibility = "blocked" if blocked else "insufficient" if insufficient else "eligible"
    status = eligibility if eligibility != "eligible" else (
        "complete" if len(included) == len(roster) else "partial")
    return {
        "target_status": status, "metric_comparability": metric if observed else None,
        "aggregation_eligibility": _assessment(eligibility, reasons),
        "included_executions": included, "missing_executions": missing, "excluded_executions": excluded,
        "coverage": {"expected_count": len(roster), "observed_count": observed,
                     "eligible_count": len(included), "included_count": len(included),
                     "missing_count": len(missing), "excluded_count": len(excluded),
                     "coverage_ratio": len(included) / len(roster)},
        "statistics": stats, "limitations": sorted(limitations),
    }


def aggregate_replicate_evidence(executions, *, group_declaration, targets) -> dict:
    """Summarize 1..16 declared executions over 1..32 explicit metric targets.

    All distinct supplied member pairs use M11. Conflicts and unknown semantics
    prevent aggregation; local missing/invalid values retain roster coverage.
    Only descriptive, equal-execution-weight statistics are returned. Inputs
    remain unchanged and returned data is detached, bounded, strict JSON.
    """
    _check_input_budget({"executions": executions, "group_declaration": group_declaration, "targets": targets})
    roster, snapshots, selected = _validate(executions, group_declaration, targets)
    group, limitations = _group_assessment(group_declaration, roster, snapshots)
    rows = {name: _target(section, members, roster, snapshots, group, limitations)
            for name, (section, members) in selected.items()}
    usable = set(roster)
    for row in rows.values():
        usable.intersection_update((item["execution"]["experiment"]["namespace"],
                                    item["execution"]["experiment"]["experiment_id"],
                                    item["execution"]["execution_id"])
                                   for item in row["included_executions"])
    group["coverage"].update(usable_count=len(usable), coverage_ratio=len(usable) / len(roster))
    result = {"schema_version": "replicate-aggregation-v1", "group": group,
              "targets": rows, "limitations": sorted(limitations)}
    encoded = json.dumps(result, ensure_ascii=False, allow_nan=False, separators=(",", ":"))
    if len(encoded.encode("utf-8")) > _OUTPUT_LIMIT:
        raise ValueError("replicate output exceeds 128 KiB")
    return result

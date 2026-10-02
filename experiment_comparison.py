"""Pure pairwise comparison of M16 evidence and explicit host declarations.

This module performs no acquisition or authorization. Values belong to the
supplied evidence; metric semantics belong to the host. M11 alone assesses
declared comparability. Neither a matching declaration nor a numerical delta
verifies checkpoint provenance, statistical significance, or causality.
"""

from collections.abc import Mapping, Sequence
from dataclasses import fields
import json
import math

import experiment_comparability as m11
from experiment_identity import (
    ExecutionIdentity, ExperimentIdentity, ExperimentProvenance, MetricResultProvenance,
)
from metrics import MetricSpec


_DECLARATIONS = (
    "metric_definition", "aggregation", "evaluation_protocol", "selection_protocol",
)
_SIDE_KEYS = {"section", "metric_name", "metric_spec", "provenance", *_DECLARATIONS}
_STATUSES = {"available", "missing", "invalid", "unavailable", "not_requested"}
_SEMANTIC_CONFLICTS = {
    "metric_definition_mismatch", "direction_mismatch", "aggregation_mismatch",
}


def _mapping(value, name):
    if not isinstance(value, Mapping):
        raise TypeError(f"{name} must be a mapping")
    if any(type(key) is not str for key in value):
        raise TypeError(f"{name} keys must be strings")
    return value


def _text(value, name):
    if type(value) is not str:
        raise TypeError(f"{name} must be a string")
    if not value or value != value.strip():
        raise ValueError(f"{name} must be nonblank without surrounding whitespace")
    return value


def _sequence(value, name):
    if isinstance(value, (str, bytes, bytearray)) or not isinstance(value, Sequence):
        raise TypeError(f"{name} must be a sequence")
    return value


def _required(mapping, key):
    if key not in mapping:
        raise ValueError(f"missing required field: {key}")
    return mapping[key]


def _finite(value):
    # Integers are finite without a lossy/overflowing conversion to float.
    return type(value) is int or (type(value) is float and math.isfinite(value))


def _json_copy(value):
    """Detach JSON data without silently coercing keys, tuples or custom values."""
    if value is None or type(value) in (str, bool, int):
        return value
    if type(value) is float:
        if not math.isfinite(value):
            raise ValueError("JSON evidence must contain finite values")
        return value
    if type(value) is list:
        return [_json_copy(item) for item in value]
    if isinstance(value, Mapping):
        _mapping(value, "JSON evidence")
        return {key: _json_copy(item) for key, item in value.items()}
    raise TypeError("evidence projection must contain only JSON values")


def _validate_provenance(provenance):
    if not isinstance(provenance, ExperimentProvenance):
        raise TypeError("provenance must be an ExperimentProvenance")
    execution = provenance.execution
    if not isinstance(execution, ExecutionIdentity):
        raise TypeError("execution must be an ExecutionIdentity")
    if not isinstance(execution.experiment, ExperimentIdentity):
        raise TypeError("experiment must be an ExperimentIdentity")
    for value in (execution.execution_id, execution.experiment.namespace,
                  execution.experiment.experiment_id):
        _text(value, "identity")
    # Validate declarations' types, not their agreement. Agreement belongs to M11.
    for field in fields(provenance):
        value = getattr(provenance, field.name)
        if field.name == "execution" or value is None:
            continue
        if field.name == "realized_seed":
            if type(value) is not int:
                raise TypeError("realized_seed must be an integer or None")
        else:
            _text(value, field.name)


def _validate_targets(targets):
    _sequence(targets, "targets")
    if not 1 <= len(targets) <= 32:
        raise ValueError("select between 1 and 32 comparison targets")
    seen = set()
    for target in targets:
        _mapping(target, "target")
        if target.keys() != {"target_id", "a", "b"}:
            raise ValueError("target must contain exactly target_id, a and b")
        key = _text(target["target_id"], "target_id")
        if key in seen:
            raise ValueError("duplicate target_id")
        seen.add(key)
        for label in ("a", "b"):
            side = _mapping(target[label], "target side")
            if side.keys() != _SIDE_KEYS:
                raise ValueError("invalid target side fields")
            if _text(side["section"], "section") not in ("validation", "test"):
                raise ValueError("section must be validation or test")
            _text(side["metric_name"], "metric_name")
            spec = side["metric_spec"]
            if spec is not None:
                if not isinstance(spec, MetricSpec):
                    raise TypeError("metric_spec must be a MetricSpec or None")
                if spec.name != side["metric_name"]:
                    raise ValueError("metric_spec must identify the selected metric")
                if _text(spec.direction, "direction") not in ("maximize", "minimize"):
                    raise ValueError("invalid metric direction")
                if side["section"] == "test" and len(spec.path) != 1:
                    raise ValueError("Test metric spec must select one column")
            _validate_provenance(side["provenance"])
            for name in _DECLARATIONS:
                if side[name] is not None:
                    _text(side[name], name)


def _observation(evidence, side):
    """Select a value without filling unavailable evidence from another source."""
    section, name = side["section"], side["metric_name"]
    if section == "validation":
        meta = _mapping(_required(evidence, "validation_evidence"), "validation_evidence")
        metrics = _mapping(_required(evidence, "validation_metrics"), "validation_metrics")
        status, value_key = "available", "best_value"
    else:
        meta = _mapping(_required(evidence, "test_results"), "test_results")
        status = _text(_required(meta, "status"), "test status")
        if status not in _STATUSES:
            raise ValueError("unsupported test status")
        metrics = _mapping(_required(meta, "metrics"), "test metrics")
        value_key = "value"

    # not_requested is deliberately sparse in M16; do not invent provenance.
    sparse = section == "test" and status == "not_requested"
    kind = None if sparse else _text(_required(meta, "evidence_kind"), "evidence_kind")
    origin = {} if sparse else _mapping(_required(meta, "source"), "source")
    source = {
        "section": section, "metric_name": name, "evidence_kind": kind,
        "file": None if sparse else _text(_required(origin, "file"), "source file"),
        "metadata_source": "host_declared",
    }
    if section == "test" and not sparse:
        source["provenance"] = _json_copy(_mapping(_required(meta, "provenance"), "test provenance"))
        source["data_row"] = _json_copy(origin.get("data_row"))
        source["selection"] = _json_copy(origin.get("selection"))

    if status != "available":
        return {"source": source, "status": status, "value": None, "metric": None}
    if name not in metrics:
        return {"source": source, "status": "missing", "value": None, "metric": None}
    metric = _mapping(metrics[name], "selected metric")
    value = _required(metric, value_key)
    if not _finite(value):
        raise ValueError("selected metric value must be a finite number, excluding bool")
    if section == "validation":
        epoch = _required(metric, "best_epoch")
        if type(epoch) is not int:
            raise TypeError("best_epoch must be an integer")
        source["best_epoch"] = epoch
    return {"source": source, "status": status, "value": value, "metric": metric}


def _alignment_reasons(a, b, left, right):
    reasons = []
    if left["section"] != right["section"]:
        reasons.append("section_mismatch")
    for label, observation, side in (("a", a, left), ("b", b, right)):
        expected_kind = "csv_reported" if side["section"] == "test" else "validation_history"
        if observation["source"]["evidence_kind"] != expected_kind:
            reasons.append(f"{label}_evidence_kind_mismatch")
        spec = side["metric_spec"]
        if spec is None or side["metric_definition"] is None or side["aggregation"] is None:
            reasons.append(f"{label}_metric_metadata_missing")
        metric = observation["metric"]
        if spec is not None and metric is not None and side["section"] == "test":
            if metric.get("source_column") != spec.path[0] or metric.get("direction") != spec.direction:
                reasons.append(f"{label}_evidence_spec_mismatch")
    return reasons


def _metric_result(observation, side):
    return MetricResultProvenance(
        metric_name=side["metric_name"], value=observation["value"],
        provenance=side["provenance"], direction=side["metric_spec"].direction,
        best_epoch=observation["source"].get("best_epoch"),
        **{name: side[name] for name in _DECLARATIONS},
    )


def _compare_target(evidence_a, evidence_b, target):
    left, right = target["a"], target["b"]
    a, b = _observation(evidence_a, left), _observation(evidence_b, right)
    reasons = _alignment_reasons(a, b, left, right)
    limitations = ["statistical_significance_not_assessed"]
    for observation in (a, b):
        provenance = observation["source"].get("provenance", {})
        if provenance.get("checkpoint_binding") == "not_verified":
            if "checkpoint_binding_not_verified" not in limitations:
                limitations.append("checkpoint_binding_not_verified")
    row = {
        "source_a": a["source"], "source_b": b["source"],
        "availability_a": a["status"], "availability_b": b["status"],
        "value_a": a["value"], "value_b": b["value"],
        "alignment": {"status": "blocked" if reasons else "aligned", "reason_codes": reasons},
        "comparability": None, "not_evaluated_reason": None,
        "calculation": {"status": "blocked", "raw_delta": None,
                        "direction": None, "direction_interpretation": None},
        "limitations": limitations,
    }
    if a["status"] != "available" or b["status"] != "available":
        row["not_evaluated_reason"] = "numeric_evidence_not_available"
        return row
    if reasons:
        row["not_evaluated_reason"] = "metric_alignment_incomplete"
        return row

    assessment = m11.compare_metric_results(_metric_result(a, left), _metric_result(b, right))
    row["comparability"] = {
        "status": assessment.status, "reason_codes": list(assessment.reason_codes),
        "policy_version": assessment.policy_version,
    }
    # Reuse M11's semantic conflicts; do not independently compare declarations.
    semantic_conflicts = [code for code in assessment.reason_codes if code in _SEMANTIC_CONFLICTS]
    if semantic_conflicts:
        row["alignment"] = {"status": "blocked", "reason_codes": semantic_conflicts}
    if assessment.status == "incompatible":
        return row
    if assessment.status == "unknown":
        limitations.extend(assessment.reason_codes)
        limitations.append("limited_numerical_observation")

    try:
        delta = b["value"] - a["value"]
    except OverflowError:
        delta = None
    if not _finite(delta):
        limitations.append("delta_non_finite")
        return row
    direction = left["metric_spec"].direction
    interpretation = "equal" if delta == 0 else (
        "b_favorable" if (delta > 0) == (direction == "maximize") else "a_favorable")
    row["calculation"] = {
        "status": "calculated", "raw_delta": delta, "direction": direction,
        "direction_interpretation": interpretation,
    }
    return row


def _config_fields(evidence):
    projection = _mapping(_required(evidence, "configuration_evidence"), "configuration_evidence")
    result = {}
    for field in _sequence(_required(projection, "fields"), "configuration fields"):
        _mapping(field, "configuration field")
        path = _required(field, "path")
        if type(path) is not list or not path:
            raise ValueError("configuration path must be a nonempty list")
        key = tuple(_text(part, "configuration path segment") for part in path)
        if key in result:
            raise ValueError("duplicate configuration path")
        status = _required(field, "status")
        if status not in ("present", "missing", "unavailable"):
            raise ValueError("unsupported configuration status")
        if status == "present":
            value = _required(field, "value")
            if value is not None and type(value) not in (str, bool, int, float):
                raise TypeError("projected configuration values must be scalars")
        elif "value" in field:
            raise ValueError("non-present configuration field cannot declare a value")
        result[key] = _json_copy(field)
    return result


def _configuration_context(a, b):
    left, right = _config_fields(a), _config_fields(b)
    rows = []
    for path in sorted(left.keys() | right.keys()):
        first, second = left.get(path), right.get(path)
        relation = "unknown"
        if first is not None and second is not None and first["status"] == second["status"] == "present":
            x, y = first["value"], second["value"]
            relation = "equal" if type(x) is type(y) and x == y else "different"
        rows.append({"path": list(path), "a": first, "b": second, "relation": relation})
    return rows


def compare_experiment_evidence(evidence_a, evidence_b, *, targets) -> dict:
    """Compare 1..32 explicit targets, returning at most 64 KiB of UTF-8 JSON.

    Malformed inputs raise TypeError/ValueError. Absent evidence and incomplete
    alignment retain target entries with blocked calculations. Complete metric
    semantics permit a delta under M11 unknown, with its missing declarations
    preserved as limitations. M11 incompatible always blocks calculation.

    The pair labels identify the supplied inputs, not inferred experiment IDs.
    No input is modified, no unavailable value is fabricated, and no artifact is
    read or bound to a checkpoint. Returned projections are detached JSON data.
    """
    _mapping(evidence_a, "evidence_a")
    _mapping(evidence_b, "evidence_b")
    _validate_targets(targets)
    result = {
        "schema_version": "experiment-comparison-v1",
        "pair": {"a": "evidence_a", "b": "evidence_b"},
        "targets": {target["target_id"]: _compare_target(evidence_a, evidence_b, target)
                    for target in targets},
        "configuration_differences": _configuration_context(evidence_a, evidence_b),
    }
    encoded = json.dumps(result, ensure_ascii=False, allow_nan=False,
                         separators=(",", ":")).encode("utf-8")
    if len(encoded) > 64 * 1024:
        raise ValueError("comparison output exceeds 64 KiB")
    return result

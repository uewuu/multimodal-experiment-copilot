"""Compare host-provided artifact descriptors without observing artifacts.

Digest equality describes supplied fields only. It establishes neither file
currentness nor producer authenticity, checkpoint binding or semantic validity.
"""

import json


__all__ = ("compare_artifact_descriptors",)

_INPUT_LIMIT = 128 * 1024
_ENTRY_LIMIT = 2 * 1024
_OUTPUT_LIMIT = 64 * 1024
_INTEGER_LIMIT = 2**63 - 1
_LIMITATIONS = (
    "artifact_currentness_not_checked",
    "checkpoint_binding_not_verified",
    "experiment_membership_not_verified",
    "filesystem_not_observed_by_copilot",
    "host_reported_only",
    "metric_recomputation_not_performed_by_copilot",
)
_DESCRIPTOR_FIELDS = ("locator", "byte_size", "sha256")
_STATUSES = ("observed", "missing", "unauthorized", "unreadable", "unobserved")


def _json_size(value):
    """Measure only already validated plain data, never raw host objects."""
    return len(json.dumps(value, ensure_ascii=False, allow_nan=False,
                          separators=(",", ":")).encode("utf-8"))


def _snapshot_inputs(declared_set, reported_observations):
    """Detach exact data while charging the raw input's compact JSON budget.

    Type identity checks avoid custom metaclass equality/hash. Ancestors detect
    cycles without rejecting shared acyclic data. No caller traversal, copy or
    attribute protocols are consulted. All legal strings fit within 512 bytes;
    bound primitives before encoding them or constructing intermediate copies.
    """
    used, ancestors = 0, set()

    def charge(count):
        nonlocal used
        used += count
        if used > _INPUT_LIMIT:
            raise ValueError("artifact comparison input exceeds byte budget")

    def visit(value, depth):
        kind = type(value)
        if value is None:
            charge(4)
            return None
        if kind is str:
            if len(value) > 512 or len(value.encode("utf-8")) > 512:
                raise ValueError("artifact input string exceeds byte budget")
            charge(_json_size(value))
            return value
        if kind is int:
            if value < 0 or value > _INTEGER_LIMIT:
                raise ValueError("artifact input integer is outside the supported range")
            charge(_json_size(value))
            return value
        if kind is not dict and kind is not list:
            raise TypeError("artifact comparison requires exact plain data types")
        if depth > 8 or id(value) in ancestors:
            raise ValueError("artifact input exceeds depth or contains a cycle")
        ancestors.add(id(value))
        try:
            charge(2)
            if kind is list:
                copied = []
                for index, child in enumerate(value):
                    charge(int(index > 0))
                    copied.append(visit(child, depth + 1))
                return copied
            copied = {}
            for index, (key, child) in enumerate(value.items()):
                if type(key) is not str:
                    raise TypeError("artifact input keys must be exact strings")
                charge(1 + int(index > 0))
                visit(key, depth + 1)
                copied[key] = visit(child, depth + 1)
            return copied
        finally:
            ancestors.remove(id(value))

    envelope = visit({"declared_set": declared_set,
                      "reported_observations": reported_observations}, 1)
    return envelope["declared_set"], envelope["reported_observations"]


def _record(value, required, optional=()):
    if type(value) is not dict:
        raise TypeError("artifact record must be an exact dict")
    keys = set(value)
    if not set(required) <= keys or keys - set(required) - set(optional):
        raise ValueError("invalid artifact record fields")


def _text(value, *, optional=False, locator=False):
    if optional and value is None:
        return None
    if type(value) is not str:
        raise TypeError("artifact label must be an exact string")
    limit = 512 if locator else 128
    if len(value.encode("utf-8")) > limit or (not locator and not value.strip()):
        raise ValueError("artifact label is blank or exceeds byte budget")
    return value


def _integer(value):
    if value is None:
        return None
    if type(value) is not int:
        raise TypeError("artifact size and mtime must be exact integers")
    if not 0 <= value <= _INTEGER_LIMIT:
        raise ValueError("artifact integer is outside the supported range")
    return value


def _sha256(value):
    if value is None:
        return None
    if type(value) is not str:
        raise TypeError("sha256 must be an exact string")
    if len(value) != 64 or any(char not in "0123456789abcdefABCDEF" for char in value):
        raise ValueError("sha256 must contain 64 ASCII hex characters")
    return value.lower()


def _descriptor(record):
    return {
        "locator": _text(record.get("locator"), optional=True, locator=True),
        "byte_size": _integer(record.get("byte_size")),
        "sha256": _sha256(record.get("sha256")),
    }


def _entries(value, minimum):
    if type(value) is not list:
        raise TypeError("artifact collections must be exact lists")
    if not minimum <= len(value) <= 32:
        raise ValueError("artifact collection exceeds cardinality bounds")
    for entry in value:
        if _json_size(entry) > _ENTRY_LIMIT:
            raise ValueError("artifact entry exceeds byte budget")


def _validate(declared, reports):
    """Validate the detached graph completely before comparing any fields."""
    _record(declared, ("schema_version", "set_id", "declared_by", "artifacts"),
            ("execution_ref", "data_snapshot_ref"))
    if declared["schema_version"] != "declared-artifact-set-v1":
        raise ValueError("unsupported declared artifact schema version")
    _text(declared["set_id"])
    _text(declared["declared_by"])
    _text(declared.get("data_snapshot_ref"), optional=True)
    execution = declared.get("execution_ref")
    if execution is not None:
        _record(execution, ("namespace", "experiment_id", "execution_id"))
        for value in execution.values():
            _text(value)
    _entries(declared["artifacts"], 1)
    _entries(reports, 0)
    artifacts, observations = {}, {}
    for entry in declared["artifacts"]:
        _record(entry, ("artifact_id", "logical_role"), _DESCRIPTOR_FIELDS)
        key = _text(entry["artifact_id"])
        role = _text(entry["logical_role"])
        if key in artifacts:
            raise ValueError("duplicate artifact_id")
        artifacts[key] = (role, _descriptor(entry))
    for entry in reports:
        _record(entry, ("artifact_id", "observation_id", "reported_by", "reported_status"),
                ("reported_at", "locator", "byte_size", "sha256", "mtime_ns"))
        key = _text(entry["artifact_id"])
        if key not in artifacts or key in observations:
            raise ValueError("unknown or duplicate observation artifact_id")
        status = _text(entry["reported_status"])
        if status not in _STATUSES:
            raise ValueError("unsupported reported status")
        projection = {
            "observation_id": _text(entry["observation_id"]),
            "reported_by": _text(entry["reported_by"]),
            "reported_at": _text(entry.get("reported_at"), optional=True),
            "reported_status": status,
            **_descriptor(entry),
            "mtime_ns": _integer(entry.get("mtime_ns")),
        }
        if status != "observed" and any(
            projection[name] is not None for name in ("byte_size", "sha256", "mtime_ns")
        ):
            raise ValueError("non-observed report contains integrity observations")
        observations[key] = projection
    return artifacts, observations


def _comparison(left, right, observed):
    if not observed:
        return "not_evaluated"
    if left is None or right is None:
        return "unknown"
    return "match" if left == right else "mismatch"


def _row(role, declared, reported, membership):
    status = "unobserved" if reported is None else reported["reported_status"]
    observed = status == "observed"
    other = {} if reported is None else reported
    sha = _comparison(declared["sha256"], other.get("sha256"), observed)
    size = _comparison(declared["byte_size"], other.get("byte_size"), observed)
    locator = _comparison(declared["locator"], other.get("locator"), observed)
    reasons = ["sha256_" + sha, "size_" + size, "locator_" + locator]
    if reported is None:
        reasons.append("observation_missing")
    elif not observed:
        reasons.append("reported_" + status)
    return {
        "logical_role": role,
        "declared": declared,
        "reported": reported,
        "identity_status": "missing_observation" if reported is None else "aligned_host_key",
        "reported_status": status,
        "integrity": {"sha256_comparison": sha, "size_comparison": size},
        "locator_comparison": {"match": "same", "mismatch": "different"}.get(locator, locator),
        "currentness_status": "not_checked",
        "experiment_membership_status": membership,
        "checkpoint_binding_status": "not_verified",
        "recomputation_status": "not_performed_by_copilot",
        "evidence_basis": "host_reported",
        "reason_codes": sorted(reasons),
        "limitations": list(_LIMITATIONS),
    }


def compare_artifact_descriptors(declared_set, reported_observations) -> dict:
    """Return independent descriptor comparisons with fixed provenance limits.

    All inputs are host supplied. No external source is consulted, and no
    agreement of fields upgrades currentness, membership or checkpoint claims.
    Invalid types, schemas or budgets raise TypeError/ValueError. Disagreements
    remain report data. Inputs and returned mutable data are never cached.
    """
    declared, reports = _snapshot_inputs(declared_set, reported_observations)
    artifacts, observations = _validate(declared, reports)
    execution = declared.get("execution_ref")
    membership = "unknown" if execution is None else "host_declared"
    result = {
        "schema_version": "artifact-verification-report-v1",
        "set_id": declared["set_id"],
        "declared_by": declared["declared_by"],
        "execution_ref": execution,
        "data_snapshot_ref": declared.get("data_snapshot_ref"),
        "evidence_basis": "host_reported",
        "artifacts": {key: _row(*artifacts[key], observations.get(key), membership)
                      for key in sorted(artifacts)},
        "limitations": list(_LIMITATIONS),
    }
    if _json_size(result) > _OUTPUT_LIMIT:
        raise ValueError("artifact comparison output exceeds byte budget")
    return result

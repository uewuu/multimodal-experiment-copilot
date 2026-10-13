"""M20 scope freeze: Evidence Artifact Verification -- Contract v1.

PUBLIC: experiment_artifact_verification.compare_artifact_descriptors(
declared_set, reported_observations); tool_layer.artifact_verification_tools.
ArtifactVerificationTools has the same two required positional-or-keyword
constructor parameters. Tool name compare_artifact_descriptors; empty schema.
No production implementation, live observation, Reader, file hashing, Memory,
path acquisition, provenance authentication, recomputation or old schema change.

INPUT: exact dict/list/str/int/None only; no tuple, float, bool, dataclass,
subclasses, custom protocols or arbitrary attribute/schema discovery. Required
set keys: schema_version='declared-artifact-set-v1', set_id, declared_by,
artifacts. Optional execution_ref/data_snapshot_ref default to None. execution_ref
is None or exactly {namespace, experiment_id, execution_id}, all identifiers.
Required artifact keys: artifact_id/logical_role; optional locator/byte_size/sha256.
Required observation keys: artifact_id/observation_id/reported_by/reported_status;
optional reported_at/locator/byte_size/sha256/mtime_ns. No observation envelope.
Unknown keys anywhere reject. Identifiers, logical roles, source labels and
reported_at are nonblank exact strings <=128 UTF8 bytes, preserved verbatim.
reported_at is an opaque host timestamp label, not parsed or system generated.
Locator is None or exact str <=512 UTF8 bytes, including empty/absolute/relative/
traversal-looking/NUL strings: opaque data, never resolved or normalized.
sha256 is None or 64 ASCII hex chars, normalized lowercase in projections and
comparisons. Size and mtime are None or exact int in 0..2**63-1; bool rejects.
Missing optional fields normalize to None. No field defaults to a trusted fact.

ALIGNMENT: 1..32 artifacts, 0..32 observations; ID equality is case-sensitive,
set-local, and the ONLY alignment rule. Duplicate artifact IDs, duplicate
observation artifact IDs or unknown artifact references reject. observation_id
is a report label, not an alignment key; the same label across different
artifacts is allowed. List position, role, locator and digest never align rows.
Status is observed|missing|unauthorized|unreadable|unobserved. Non-observed rows
may retain locator/reported_at but all of size/sha256/mtime must be None/absent.

OUTPUT root EXACTLY schema_version='artifact-verification-report-v1', set_id,
declared_by, execution_ref, data_snapshot_ref, evidence_basis='host_reported',
artifacts keyed by sorted artifact_id, limitations. Each row EXACTLY logical_role,
declared={locator,byte_size,sha256}, reported=None or {observation_id,reported_by,
reported_at,reported_status,locator,byte_size,sha256,mtime_ns}, identity_status,
reported_status, integrity={sha256_comparison,size_comparison}, locator_comparison,
currentness_status, experiment_membership_status, checkpoint_binding_status,
recomputation_status, evidence_basis, reason_codes, limitations.
identity_status=aligned_host_key for any supplied report, else missing_observation;
absent report projects unobserved, never missing. Membership host_declared iff
execution_ref supplied, else unknown (snapshot ref/locator do not establish it).
Currentness not_checked; checkpoint not_verified; recomputation
not_performed_by_copilot; evidence_basis host_reported, unconditionally.

For observed rows: digest/size compare match|mismatch iff BOTH fields non-null,
otherwise unknown, including null/null. Locator compares same|different iff
both non-null, else unknown; empty equals empty, no path normalization.
For all other statuses all three comparisons are not_evaluated. Independent
dimensions retain hash match + size mismatch. No overall status/verified flag.
Reasons sorted unique: sha256_<comparison>, size_<comparison>, and
locator_match|locator_mismatch|locator_unknown|locator_not_evaluated. Add
observation_missing for absent report; otherwise reported_<status> only for
non-observed report. Limitations root AND every row are exactly LIMITATIONS below.

SAFETY: bounded data-only validation/snapshot BEFORE comparison/core delegation.
Reject unsafe types with TypeError/ValueError without copy/reduce/iteration/
reflection/fspath callbacks. Cycle/depth/size failures ValueError, no truncation.
Constructor, invocation, result and schema isolated; each invoke calls real core.
Invalid model arguments reject before core. Unknown tool KeyError. Tool uses
existing explicit binding, Session transactions and Runtime events; no new
capability, no registration. Direct core/tool must not call Path Policy.
Session's existing policy envelope remains untouched (no M20 path capability).

BOUNDS inclusive: 1..32 artifacts; 0..32 reports; each raw supplied artifact and
observation <=2048 compact UTF8 JSON bytes; combined raw input envelope exactly
{declared_set:...,reported_observations:...} <=131072 bytes; container depth <=8
from that envelope at depth 1. Fixed schemas/field caps can impose tighter bounds
than these aggregate ceilings; no filler/unknown field is made legal to reach
them. Output <=65536 compact UTF8 bytes or ValueError containing 'output'.
No silent truncation. Per-file bytes read=0; total bytes hashed=0.

Helpers contain fixtures, handwritten oracles and independently calibrated
probes, never a future comparator. Structured scoring is not a general prose
claim classifier. Deny probes cover named Python/project entrances, not all OS I/O.
"""

from contextlib import contextmanager
from copy import deepcopy
import importlib
import json
from types import SimpleNamespace as NS

TOOL = "compare_artifact_descriptors"
PARAMETERS = {"type": "object", "properties": {}, "additionalProperties": False}
LIMITATIONS = sorted([
    "host_reported_only", "filesystem_not_observed_by_copilot",
    "artifact_currentness_not_checked", "experiment_membership_not_verified",
    "checkpoint_binding_not_verified", "metric_recomputation_not_performed_by_copilot",
])
ROOT_KEYS = {"schema_version", "set_id", "declared_by", "execution_ref",
             "data_snapshot_ref", "evidence_basis", "artifacts", "limitations"}
ROW_KEYS = {"logical_role", "declared", "reported", "identity_status", "reported_status",
            "integrity", "locator_comparison", "currentness_status",
            "experiment_membership_status", "checkpoint_binding_status",
            "recomputation_status", "evidence_basis", "reason_codes", "limitations"}
SHA_A, SHA_B = "ab" * 32, "cd" * 32  # Synthetic descriptor values, never file digests.


def core_api():
    try:
        from experiment_artifact_verification import compare_artifact_descriptors
    except ModuleNotFoundError as error:
        if error.name != "experiment_artifact_verification":
            raise
        raise AssertionError("missing M20 core") from error
    return compare_artifact_descriptors


def tool_class():
    try:
        from tool_layer.artifact_verification_tools import ArtifactVerificationTools
    except ModuleNotFoundError as error:
        if error.name != "tool_layer.artifact_verification_tools":
            raise
        raise AssertionError("missing M20 tool/integration") from error
    return ArtifactVerificationTools


def data():
    return ({"schema_version": "declared-artifact-set-v1", "set_id": "bundle-1",
             "declared_by": "host-A", "execution_ref": {
                 "namespace": "lab", "experiment_id": "generic", "execution_id": "run-1"},
             "data_snapshot_ref": "data-1", "artifacts": [{
                 "artifact_id": "metrics", "logical_role": "metrics", "locator": "metrics.csv",
                 "byte_size": 12, "sha256": SHA_A}]},
            [{"artifact_id": "metrics", "observation_id": "report-1", "reported_by": "host-B",
              "reported_at": "2026-01-01T00:00:00Z", "reported_status": "observed",
              "locator": "metrics.csv", "byte_size": 12, "sha256": SHA_A, "mtime_ns": 123}])


def size(value):
    return len(json.dumps(value, ensure_ascii=False, allow_nan=False,
                          separators=(",", ":")).encode("utf-8"))


def assert_limits(report):
    assert set(report) == ROOT_KEYS
    assert report["schema_version"] == "artifact-verification-report-v1"
    assert report["evidence_basis"] == "host_reported"
    assert report["limitations"] == LIMITATIONS
    assert list(report["artifacts"]) == sorted(report["artifacts"])
    for row in report["artifacts"].values():
        assert set(row) == ROW_KEYS
        assert row["currentness_status"] == "not_checked"
        assert row["checkpoint_binding_status"] == "not_verified"
        assert row["recomputation_status"] == "not_performed_by_copilot"
        assert row["evidence_basis"] == "host_reported"
        assert row["limitations"] == LIMITATIONS
        assert row["reason_codes"] == sorted(set(row["reason_codes"]))
    assert size(report) <= 65536


def patch_core(monkeypatch, replacement):
    owner = importlib.import_module("experiment_artifact_verification")
    consumer = importlib.import_module("tool_layer.artifact_verification_tools")
    original = getattr(owner, TOOL)
    monkeypatch.setattr(owner, TOOL, replacement)
    for name, value in list(vars(consumer).items()):
        if value is original:
            monkeypatch.setattr(consumer, name, replacement)


class Client:
    """Offline provider messages; never manufactures tool evidence."""
    def __init__(self, args=None, *, final="offline answer", error=None):
        self.args = {} if args is None else args
        self.final, self.error, self.requests = final, error, []
        self.chat = NS(completions=self)

    def create(self, **kwargs):
        self.requests.append(deepcopy(kwargs))
        if len(self.requests) == 1:
            message = NS(role="assistant", content=None, tool_calls=[NS(
                id="artifact-call", type="function", function=NS(name=TOOL, arguments=json.dumps(self.args)))])
        elif len(self.requests) == 2:
            if self.error is not None:
                raise self.error
            message = NS(role="assistant", content=self.final, tool_calls=None)
        else:
            raise AssertionError("unexpected provider retry")
        return NS(choices=[NS(message=message)], usage=NS(
            prompt_tokens=10, completion_tokens=3, total_tokens=13))


@contextmanager
def deny_external(monkeypatch):
    import builtins
    import hashlib
    import io
    import os
    from pathlib import Path
    import socket
    import read_config
    import read_history
    import read_test_metrics
    from experiment_memory import ExperimentMemory
    from experiment_repository import ExperimentRepository
    from tool_layer import experiment_tools, tool_registry, experiment_path_security
    calls = []

    def deny(*args, **kwargs):
        calls.append(True)
        raise AssertionError("external access sentinel")

    entrances = [(builtins, "open"), (io, "open"), (os, "open"), (os, "stat"),
                 (os, "fstat"), (Path, "open"), (Path, "stat"), (Path, "resolve"),
                 (hashlib, "sha256"), (hashlib, "new"), (socket, "create_connection"),
                 (ExperimentMemory, "get"), (ExperimentMemory, "list_executions"),
                 (ExperimentRepository, "__init__"), (read_config, "read_config_evidence"),
                 (read_history, "read_history"), (read_test_metrics, "read_test_metrics"),
                 (experiment_tools, "analyze_experiment_evidence"),
                 (tool_registry, "_workspace_paths_for_tool"),
                 (experiment_path_security, "_build_experiment_path_policy")]
    with monkeypatch.context() as patch:
        for owner, name in entrances:
            original = getattr(owner, name)
            patch.setattr(owner, name, deny)
            # Patch direct-import aliases after modules have been loaded.
            import sys
            for module_name in ("experiment_artifact_verification", "tool_layer.artifact_verification_tools"):
                module = sys.modules.get(module_name)
                if module is not None:
                    for alias, value in list(vars(module).items()):
                        if value is original:
                            patch.setattr(module, alias, deny)
        yield calls

# Python API reference

These are current source interfaces for Python 3.11, not a promise of full SemVer API stability. Run from the repository root. Internal underscore helpers and the demo's fixture/provider seams are not public APIs.

## Session, service and binding

```python
from copilot import CopilotService, CopilotSession
from copilot.tool_binding import BoundCopilotService, BoundCopilotSession
from llm_adapters.tool_binding import BoundToolCollection, bound_tools
```

| Entry | Signature / result | Host responsibility and errors |
| --- | --- | --- |
| Service | `CopilotService(client, *, model)` | Borrowed compatible client; host owns closing it |
| One-shot | `service.run(question, *, experiment_context=None, turn_timeout_seconds=None, on_failure=None, **request_options) -> CopilotObservedResult` | Supply trusted context; validation, provider and tool exceptions propagate |
| Session creation | `service.create_session(*, experiment_context=None, max_turns=8, turn_timeout_seconds=None, **request_options) -> CopilotSession` | Independent bounded history |
| Session | `CopilotSession(client, *, model, experiment_context=None, max_turns=8, turn_timeout_seconds=None, **request_options)` | Invalid configuration rejects before execution |
| Ask | `session.ask(question) -> str`; `ask_with_result(question) -> CopilotTurn`; `ask_with_observability(question, *, on_failure=None) -> CopilotObservedResult` | Successful turns commit history; failed turns do not |
| Collection | `BoundToolCollection(*tool_sets)` | Adds default tools plus explicit sets; duplicate names reject with `ValueError` |
| Bound facades | `BoundCopilotSession(session, tools)` / `BoundCopilotService(service, tools)` | Call-scoped binding; previous binding restored on exit |
| Context | `with bound_tools(collection): ...` | Same binding behavior for direct calls |

`CopilotObservedResult.turn` contains `answer` and `tool_invocations`. Each invocation records `tool_name`, `arguments_json` and `result_json`. `.metrics` has `provider_request_count`, `tool_invocation_count` and `elapsed_seconds`; `.run` holds optional run metadata. A Session exposes `history`, `turn_count`, `export_history()` and `reset()`. Reset clears conversation state, not external application state.

Example integration shape (supply `client` and a validated host Tool):

```python
session = CopilotSession(client, model="your-model", experiment_context=host_context)
bound = BoundCopilotSession(session, BoundToolCollection(selected_tool))
observed = bound.ask_with_observability("Report the selected evidence and its limitations.")
```

For a complete credential-free invocation use `run_scenario` in [the executable example](../examples/v1_bound_workflows_demo.py). Its fake Provider supports `client.chat.completions.create`; the example returns an envelope rather than the raw `CopilotObservedResult`.

## Advanced Tools

All classes below expose `list_tools() -> list[dict]` and `invoke_tool(tool_name, arguments) -> dict`. Construction freezes host selections; it does not grant Session path authority. Use the formal Adapter/Session path for model requests. Calling file Tools directly makes authorization the caller's responsibility.

```python
from tool_layer.experiment_tools import ExperimentEvidenceTools, ExperimentComparisonTools
from tool_layer.replicate_tools import ReplicateAggregationTools
from tool_layer.replicate_group_tools import ReplicateGroupComparisonTools
from tool_layer.paired_experiment_tools import PairedReplicateComparisonTools
from tool_layer.artifact_verification_tools import ArtifactVerificationTools
from tool_layer.artifact_observation_tools import ArtifactObservationTools
```

| Tool constructor | Model tool / arguments | Host selection |
| --- | --- | --- |
| `ExperimentEvidenceTools(experiment_dir, *, config_paths=(), test_metrics_file=None, test_metric_specs=(), metrics_config=None)` | `analyze_experiment_evidence`: `experiment_dir`, optional `include_diagnostics` | One root, relative selected files, metric specs |
| `ExperimentComparisonTools(experiment_a, experiment_b, *, options_a=None, options_b=None, targets)` | `compare_experiment_evidence`: `experiment_a`, `experiment_b` | Ordered roots, per-side M16 options, explicit target sides |
| `ReplicateAggregationTools(executions, *, group_declaration, targets)` | `aggregate_replicate_evidence`: `{}` | Expected roster, supplied snapshots, selectors |
| `ReplicateGroupComparisonTools(group_a, group_b, *, targets)` | `compare_replicate_groups`: `{}` | Two complete M18 groups, one-to-one target map |
| `PairedReplicateComparisonTools(group_a, group_b, *, pairing_declaration, targets)` | `compare_paired_replicates`: `{}` | Plain-data groups, explicit pair endpoints and target map |
| `ArtifactVerificationTools(declared_set, reported_observations)` | `compare_artifact_descriptors`: `{}` | Two sets of host claims; no file reading |
| `ArtifactObservationTools(binding)` | `observe_artifact`: `{"experiment_dir":"."}` | Exact Windows root, direct-child file selection and fixed budgets |

Unknown names reject with `KeyError`; malformed arguments generally raise `TypeError`/`ValueError`. Underlying computation errors propagate. M16 data-unavailability and M21 observation statuses are report data where their contracts specify that behavior. A report is not necessarily numerically eligible merely because invocation returned normally.

`options_a`/`options_b` permit only `config_paths`, `test_metrics_file`, `test_metric_specs`, `metrics_config`. The Adapter resolves model paths using trusted `experiment_context`; the bound Tool then checks the exact selected roots. Snapshot Tools accept no extra model options. Default Registry behavior remains unchanged.

## Public Core / Reader entry points

```python
from tool_layer.experiment_tools import analyze_experiment_evidence
from experiment_comparison import compare_experiment_evidence
from experiment_replicates import aggregate_replicate_evidence
from experiment_group_comparison import compare_replicate_groups
from experiment_paired_comparison import compare_paired_replicates
from experiment_artifact_verification import compare_artifact_descriptors
from artifact_observation_reader import observe_artifact
from artifact_observation import validate_artifact_binding, project_artifact_observation
```

| Function | Inputs and return | Trust / failure semantics |
| --- | --- | --- |
| `analyze_experiment_evidence(experiment_dir, *, config_paths=(), test_metrics_file=None, test_metric_specs=(), metrics_config=None, include_diagnostics=False) -> dict` | M16 composition: configuration, validation and optional Test report | Performs selected reads; unavailable CSV evidence retains Reader status; argument/path/unexpected errors propagate |
| `compare_experiment_evidence(evidence_a, evidence_b, *, targets) -> dict` | M17 `targets`, alignment, comparability, `calculation`, configuration differences | Pure comparison; M11 unknown can permit a limited numerical observation, incompatibility blocks |
| `aggregate_replicate_evidence(executions, *, group_declaration, targets) -> dict` | M18 `group`, `targets`, coverage and descriptive statistics | Full-group eligibility, no subgroup salvage or acquisition |
| `compare_replicate_groups(group_a, group_b, *, targets) -> dict` | M19 `groups`, target sides and `comparison.raw_delta` | Real M18 twice; unpaired B.mean minus A.mean |
| `compare_paired_replicates(group_a, group_b, *, pairing_declaration, targets) -> dict` | M22 `pairing_coverage`, `pair_results`, statistics and claim boundaries | Exact plain data; full execution identity, explicit pairs; no automatic pairing or inference |
| `compare_artifact_descriptors(declared_set, reported_observations) -> dict` | M20 per-artifact dimensions and reasons | Host-reported fields only; mismatches are data, malformed structures/budgets reject |
| `observe_artifact(binding) -> dict` | M21 status, byte count, digest, metadata, observation basis and limitations | Actual bounded I/O; invalid binding raises; classified OS failures become statuses; unexpected bugs propagate |
| `validate_artifact_binding(binding) -> dict` | Detached validated binding | Pure validation; exact plain fields only |
| `project_artifact_observation(binding, capture) -> dict` | Fixed observation schema from validated capture | Pure projection; calling it with host data is not evidence that a Reader ran |

The pure comparison/aggregation functions do not access filesystem, Memory or network. Schema, size or type violations raise rather than silently truncate. Read complete input builders in [the fixture module](../examples/v1_workflow_fixtures.py); those builders supply inputs, not computed results.

### Input shapes

- M17 target: `target_id`, sides `a`/`b`; each side declares section, metric name/spec, provenance, metric definition, aggregation, evaluation and selection protocols. Values come from evidence, not target declarations.
- M18 group: `executions`, `group_declaration`, `targets`. The declaration includes experiment identity, complete `expected_executions`, training protocol, membership selection and `replicate-descriptive-v1` policy. A supplied execution carries provenance, availability, evidence and host-bound snapshot metadata. Target members explicitly select metrics for expected executions.
- M19/M22 mapping: `target_id`, `a_target_id`, `b_target_id`. M22 pairing has `schema_version="explicit-pairing-v1"`, `pairing_source="host_declared"`, and `pairs` containing unique `pair_id`, `a_execution`, `b_execution`. Endpoint identity includes namespace, experiment ID and execution ID. M22 uses plain structures; M18/M19 also support their frozen approved project dataclasses.
- M20 declaration: `schema_version="declared-artifact-set-v1"`, `set_id`, `declared_by`, `artifacts`; each artifact has `artifact_id`, `logical_role` and optional descriptor fields. Reports supply matching host keys, `observation_id`, `reported_by`, `reported_status` and optional locator/size/digest/time. No agreement upgrades checkpoint or currentness status.
- M21 binding: `schema_version="artifact-observation-binding-v1"`, `artifact_id`, `logical_role`, `artifact_ref`, `file_name`, `authorized_root`, `max_bytes=1048576`, `chunk_bytes=65536`. Paths are exact strings, filename is one component. Root length <=1024 UTF-8 bytes; filename <=255; identity/role/ref <=128; binding <=4 KiB and report <=8 KiB. Only a successful full observation publishes the digest/byte count.

## Foundations M11–M15

```python
from experiment_identity import ExperimentIdentity, ExecutionIdentity, ExperimentProvenance, MetricResultProvenance
from experiment_comparability import compare_metric_results
from experiment_repository import ExperimentRepository
from experiment_memory import ExperimentMemory
from copilot.evaluation_suite import EvaluationSuite, EvaluationScenario
from copilot.evaluation_batch import BatchEvaluationRunner
from tool_layer.memory_tools import MemoryTools, MemoryAccessScope
from copilot.controlled_workflow import WorkflowController, WorkflowScope, ActionSpec
from tool_layer.workflow_tools import WorkflowTools
```

| Entry | Host contract | Result / boundary |
| --- | --- | --- |
| `ExperimentIdentity(namespace, experiment_id)`, `ExecutionIdentity(experiment, execution_id)` | Explicit keys; no path-derived identity | Dataclasses for declarations |
| `compare_metric_results(left, right)` | Two `MetricResultProvenance` values | `ComparabilityResult`: comparable, incompatible or unknown; no truth verification |
| `ExperimentRepository(database_path)` | Host SQLite location and lifecycle | Owned connection; explicit storage writes exist; close with `close()`; repository errors propagate |
| `ExperimentMemory(repository)` | Borrowed repository | `get(execution)`, `list_executions(experiment, *, filters=None, include_invalidated=False, limit=100)`; historical records, no artifact verification |
| `BatchEvaluationRunner(session_factory)` | Factory of real Sessions and explicit evaluation definitions | Structured batch evaluation; no universal natural-language claim detector |
| `MemoryAccessScope(namespace, allowed_experiment_ids, *, allow_invalidated=False)`; `MemoryTools(scope, memory_provider)` | Host scope and callable returning Memory | Explicit read tools; provider lifetime stays with host |
| `WorkflowController(scope, *, target_provider, clock, simulator=None)`; `WorkflowTools(scope, controller, *, identity_provider, approval_provider)` | Host scope, identity and approval callbacks | Simulation-only lifecycle; `WorkflowError` codes and designated simulation failures, unexpected errors propagate |

These foundations are reused as implemented. No example automatically retrieves Memory, grants approval, executes training or changes a configuration.

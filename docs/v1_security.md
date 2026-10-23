# Security, trust and platform boundaries

## Host authority

The host owns client configuration, Tool binding, experiment roots, expected rosters, target mappings, pairing declarations and artifact selection. Model arguments cannot enlarge that authority. On the formal Session/Adapter path, Path Policy and exact binding checks execute before live Reader invocation. Direct Python callers must establish their own applicable authority. Default Registry exposure remains two tools; advanced Tools are opt-in.

M21 additionally validates the opened object through a Windows handle. Lexical/path authorization alone is not opened-object identity. Its model schema exposes only `experiment_dir` with the single value `"."`; the host binds the filename and root. Snapshot Tools C–F accept an empty object and cannot use model arguments to replace host data.

## Default tool path containment

On the host-authorized Adapter path for `analyze_experiment` and `compare_experiments`, the derived-path scope checks experiment directories and required `hparams.yaml` / `history.json` files during discovery, then rechecks directories and files before their reads. Optional `metrics_config` is also checked immediately before reading. Checks resolve targets against the host-authorized root and inspect link metadata and required directory/ordinary-file types. An internal symlink or Windows junction is allowed when its resolved target remains within that authority and passes these checks; the target may be outside a selected scan subdirectory but still inside the host root. External targets, broken/cyclic links, unknown reparse points and unsupported object types fail closed.

Ordinary per-experiment parsing errors are collected as failed experiments so comparison can continue. A derived-path authorization violation instead propagates and fails the entire tool call, including a mixed batch; no partial successful comparison is returned. Missing ordinary required files remain discovery exclusions. This scope is installed only when an Adapter path policy is present for the two default tools; direct Core/Reader calls do not acquire missing authority automatically. See [the scope and checks](../experiment_path_access.py), [comparison error handling](../compare_experiments.py) and [containment regressions](../tests/test_default_tool_path_containment.py).

These checks assume a trusted, host-managed filesystem and stable host authority. Read-time rechecks catch changes after scanning, but validation and the later path-based open are separate operations: a malicious concurrent namespace change can still race them (TOCTOU). This is not an atomic open, complete sandbox or general defense against a hostile filesystem. It does not inherit [M21's native handle guarantees](#m21-native-boundary), where metadata and bytes come from the validated Windows handle; M21 itself retains the residual risks described below.

## Claims that remain separate

| Evidence | Supports | Does not establish |
| --- | --- | --- |
| M11 comparable declarations | Required declared dimensions agree | Experimental truth, causality or safe substitution |
| M16 reported CSV metric | Value read from the selected report | Checkpoint origin or recomputation |
| M18/M19 supplied snapshots | Descriptive statistics under roster/comparability gates | Independent sampling or current artifacts |
| M20 matching descriptors | Host-reported fields agree | System observation, authenticity or checkpoint binding |
| M21 computed SHA-256 | Digest of this observed byte stream | Trusted producer, latest/current status or semantic correctness |
| M22 explicit pairs | Host-declared paired descriptive deltas | Verified control, causal effects, significance or a winner |

Same locator is not same bytes. File existence is not artifact validity. Same mtime does not prove historical immutability. `no_change_detected` does not prove an atomic snapshot. A new M21 observation does not prove an older M16 parser consumed those same bytes. Neither M20 nor M21 automatically composes with the other.

## M21 native boundary

Formal native support is Windows/local NTFS, one host-selected direct-child ordinary file, at most 1 MiB. The backend checks handle type, reparse status, final path and identity before content reads. The same handle supplies metadata, fixed-chunk streaming bytes and post-read metadata. The Reader computes actual byte count and incremental SHA-256; an overflow or detected change discards the successful digest.

Unsupported platforms fail closed. No POSIX safe-open fallback is claimed. No directory scanning, directory arrays, nested selections, retry with relaxed sharing, parallel hashing, large-checkpoint exemption or automatic repair is provided. Observations distinguish missing, unauthorized, unreadable, unsupported type/platform, oversized, changed during read and I/O error. Invalid bindings are call errors, not successful observation reports.

This is best-effort change detection, not complete TOCTOU elimination or a race-free namespace. The program does not intentionally modify observed files; OS access-time behavior remains outside that guarantee. Old M16 YAML/history reading does not inherit M21's uniform byte bound.

## Data and callbacks

M18/M19 and the other separately specified snapshot boundaries retain their exact-type/callback safety contracts. M16/M17 legacy Python inputs have the explicitly scoped contract below; they do not acquire newer arbitrary-object rejection guarantees. The example builds known generic data; it is not a new arbitrary-object deserializer. Host callbacks are executable application code whose provenance and protocols require host review. M15 stays simulation-only; its approval contract does not authorize real training, HPO, checkpoint loading or deployment.

Memory discovery is separate from artifact verification. Repository writes exist for explicit host lifecycle operations; the read-only Memory interface does not make the entire repository layer read-only. The demo does not open Memory or a production database.

<a id="m16-m17-legacy-trusted-host-python-v1"></a>
## M16/M17 Legacy Trusted-Host Python API (v1)

This contract defines the legacy Python object boundary for repository implementations and examples. Its disclosed residual risks were approved for this limited scope. External custom hosts, plugins and SDK extensions remain **NOT AUDITED / HOST RESPONSIBILITY**. This documentation alignment adds no code-level isolation or rejection mechanism and does not establish release acceptance.

<!-- LTH-V1:R01 -->
M16/M17 Legacy Trusted-Host Python API v1 applies to analyze_experiment_evidence, compare_experiment_evidence, ExperimentEvidenceTools and ExperimentComparisonTools. Host trust does not establish automatic object safety.
<!-- /LTH-V1:R01 -->

### Supported API Surface

- M16 composition: tool_layer.experiment_tools.analyze_experiment_evidence(experiment_dir, *, config_paths=(), test_metrics_file=None, test_metric_specs=(), metrics_config=None, include_diagnostics=False).
- M16 binding: ExperimentEvidenceTools(experiment_dir, *, config_paths=(), test_metrics_file=None, test_metric_specs=(), metrics_config=None), list_tools and invoke_tool.
- M17 core: experiment_comparison.compare_experiment_evidence(evidence_a, evidence_b, *, targets).
- M17 binding: ExperimentComparisonTools(experiment_a, experiment_b, *, options_a=None, options_b=None, targets), list_tools and invoke_tool.
- Supporting evidence readers: read_config_evidence(config_path, field_paths), read_test_metrics(csv_path, metric_specs); contained metrics configuration loading uses read_metric_specs_config. Direct reader calls retain their actual signatures and caller authorization responsibility.
- BoundCopilotSession/BoundCopilotService, bound_tools and BoundToolCollection activate these tools. They do not transform arbitrary host options into safe inert objects.

Existing signatures, result schema identifiers, target semantics, bounds and exceptions remain authoritative; this document adds an explicit trust scope, not a new result-schema version or API implementation.

### Trust Actors

The host owns Python application code, provider client selection, tool implementations, context and object construction. The model supplies JSON tool arguments through the Adapter. Files contain reported data selected by the host. Python extensions and SDK response objects are executable host code, even when their data originated remotely.

REVIEWED_HOST_OWNED describes the inspected repository fixture builders, not every value passed by a host. INTERNAL_PROJECT_OBJECT describes known project classes, not truth of their declarations or immunity to altered instances. HOST_SUPPLIED_UNVERIFIED and UNKNOWN must not be relabeled trusted without review. JSON_DATA_ONLY is a representation classification, not semantic validity or unlimited resource safety.

### Supported Input Domain

Valid ordinary dict/list/tuple structures in the roles accepted by each API; valid strings/scalars; MetricSpec, ExperimentIdentity, ExecutionIdentity and ExperimentProvenance as required by the current schema. A tuple is not universally accepted at every nested JSON-only position.

Reviewed cooperative custom Mapping/Sequence implementations remain compatible where the existing API admits them. They must terminate, yield stable schema-valid content, avoid unapproved side effects, and provide copy behavior consistent with the host's lifetime/stability needs. The host must review nested values and protocol implementations, not merely isinstance/ABC membership. Type admission and source review are distinct.

The repository's V1 A/B examples use concrete containers and project objects built by examples/v1_workflow_fixtures.py. External custom-container examples in tests/probes demonstrate behavior; they do not certify real external integrations.

### Unsupported Input Domain

Unsupported for the reviewed safe domain: unreviewed plugin/proxy/lazy/dynamic objects, unpredictable protocols, cyclic or uncontrolled resource graphs, unauthorized files, and input whose selections can change during use. Such objects may be accepted or execute code under the legacy implementation; this exclusion is a host obligation, NOT a promise of automatic fail-closed rejection.

<!-- LTH-V1:R10 -->
Unreviewed plugin, proxy, lazy-provider, custom SDK or other dynamic objects are outside the reviewed safe input domain. External Python extension interfaces remain usable under host responsibility, but this V1 acceptance does not certify external integrations or arbitrary-object safety.
<!-- /LTH-V1:R10 -->

### Host Responsibilities

| ID | Obligation | Applicable inputs and entry points | Repository evidence / external status |
|---|---|---|---|
| H1 | Record provenance before use | All options, evidence, targets, clients, Tool sets | Fixture builders reviewed; external object history UNVERIFIED |
| H2 | Review types and nested values | M16 options; M17 evidence/targets | Concrete fixture/project construction reviewed; arbitrary custom instances UNVERIFIED |
| H3 | Do not forward unreviewed plugins/lazy/proxies | Constructors and direct Core/Reader calls | No automatic plugin-to-M16/M17 loader identified; external forwarding UNVERIFIED |
| H4 | Review iteration, length, lookup and mapping methods | config_paths/specs/options/evidence/targets | Current code traverses ABCs; responsibility applies before invocation |
| H5 | Review copy/reduce/state hooks and aliases | Both constructors, M17 invocation | deepcopy used; host must verify effective detachment or safe shared immutability |
| H6 | Review project objects, metadata and dynamic attributes | MetricSpec and provenance graph | Plain fixture instances reviewed; altered metadata/subclasses UNVERIFIED |
| H7 | Keep selections and targets stable for their lifetime | Bound Tools and collection | Ordinary fixtures stable; no concurrent mutation in inspected example |
| H8 | Bound input graph and file sizes for deployment | Direct inputs, YAML/history/metrics files | Fixtures tiny/finite; no general legacy input-envelope guarantee |
| H9 | Authorize construction separately from requests | Host constructors; Session context | Constructor executes before Session; invocation authority cannot retroactively protect it |
| H10 | Own side effects and cleanup | All callback-capable paths | History rollback is not callback transaction rollback |

Do not attempt to make an unreviewed custom object safe by first traversing or deep-copying it inside the same trust boundary; those operations themselves may execute its protocols. Prefer known concrete data assembled by reviewed code. A need to accept uncontrolled executable objects requires a separate security design, not a contract disclaimer alone.

### Constructor-Time Behavior

<!-- LTH-V1:R03 -->
Before any Session invocation, M16 Tool construction may call deepcopy and M17 Tool construction may call mapping methods, dict(options) and deepcopy. The host must review these protocols before constructing the Tool.
<!-- /LTH-V1:R03 -->

M16 validates/resolves its root, then copies host selections. M17 validates roots/options shape, traverses options and copies options/targets. BoundToolCollection also reads supplied Tool attributes and calls list_tools before model execution. These are host initialization activities, not isolated or authorized by a subsequent Session request.

### Invocation-Time Behavior

<!-- LTH-V1:R06 -->
An authorized model request may invoke protocols of objects previously bound by the host. This is host-object reachability, not direct model injection of those Python objects, and it does not establish their safety.
<!-- /LTH-V1:R06 -->

Formal Session dispatch first applies its argument/path policy, followed by the bound Tool's root/ordered-pair and allowed-key checks. M16 then traverses its stored selections and reads the chosen evidence. M17 copies its stored choices, reads both sides through M16, supplements validation metric specs and runs the actual comparison core. Direct Core entry has no Session authority gate.

### Copy and Alias Semantics

<!-- LTH-V1:R04 -->
Deepcopy is cooperative copying, not an object-isolation security boundary. A custom deepcopy implementation can return the original object or retain aliases; the host must ensure bound selections remain stable and must not assume arbitrary objects were detached.
<!-- /LTH-V1:R04 -->

Cooperative plain selections are detached by the existing constructors. For a mutable custom object returning itself, later host mutation can affect results; M17 supplementation may also write through retained target aliases. Reviewed custom inputs must provide real required detachment or demonstrably safe immutability/stability. Frozen dataclass syntax alone is not an adversarial isolation guarantee. No automatic code-level restriction is introduced by this contract.

### Callback Side Effects

<!-- LTH-V1:R02 -->
M16/M17 do not guarantee callback-free rejection of arbitrary Python objects. Type checks and semantic validation may invoke supplied object protocols before accepting or rejecting an input.
<!-- /LTH-V1:R02 -->

Possible protocols include deepcopy/copy-related hooks, iteration/length/lookup, properties and dataclass metadata, equality/hash and formatting on applicable direct Python paths. Review all reachable implementations. Neither successful validation nor a later exception proves no earlier side effect occurred. Ordinary model JSON cannot serialize executable Python implementations.

### Input Resource Limitations

M16 config selections retain 32 paths, depth 8, 64 UTF-8-byte segments, 512-byte selected string values and 16 KiB report limits. CSV retains 1 MiB acceptance with bounded overflow detection, 32 specs and 16 KiB output. M17 retains 1..32 targets and 64 KiB final strict JSON output. These do not bound every prior custom traversal, recursive graph, YAML/history load, callback duration or allocation. A controlled M17 cycle reaches RecursionError. The host must impose relevant limits before using the legacy APIs; no hard callback timeout or universal safe parser claim is made.

### Model JSON Boundary

<!-- LTH-V1:R05 -->
Ordinary model tool arguments are decoded as JSON data and cannot directly instantiate arbitrary Python classes or replace host-bound config_paths, test_metric_specs, targets or options. Python SDK response objects and host extensions remain a separate executable trust surface.
<!-- /LTH-V1:R05 -->

M16 model keys remain experiment_dir and optional include_diagnostics. M17 keys remain experiment_a and experiment_b. Extra hidden-selection arguments reject. Class-like keys in JSON remain data; there is no standard Adapter object_hook that instantiates them. Content trust, parser resource limits and object-protocol trust are different concerns.

### Session Authorization Boundary

<!-- LTH-V1:R07 -->
Session path authorization applies to invocation and precedes the relevant bound Tool dispatch. It does not protect earlier constructor protocol execution, and Session history rollback cannot undo Python callback side effects.
<!-- /LTH-V1:R07 -->

An unauthorized Session test must record constructor_count and invocation_count separately. It may assert zero relevant bound-object callbacks during denied invocation, not zero callbacks over the whole lifetime. Provider/SDK machinery is a separate counter domain; do not claim it was never called.

### Direct Python Caller Responsibility

<!-- LTH-V1:R09 -->
Direct Python Core, Reader, Tool and Adapter callers must establish their own applicable authority and input trust. A direct Adapter call does not create a missing Session path policy. Host-selected filesystem authority is separate from Python object-protocol trust.
<!-- /LTH-V1:R09 -->

Direct Reader calls add no experiment authority. Core comparison acquires no files itself, but supplied protocols may have their own effects. Direct Tool root matching does not substitute for a host's authorization decision. Selecting a filesystem workspace does not audit its contents or any custom Python object.

### Compatibility Policy

This is documentation and acceptance alignment only. Keep existing public signatures, ordinary data behavior, cooperative custom container support and old frozen tests. Objects excluded from the reviewed safe domain are not newly rejected by code. No blanket acceptance of arbitrary ABC instances is promised. Intentional rejection hardening would require a separately authorized follow-up with compatibility decisions and new RED/GREEN evidence.

### Security Non-Guarantees

No arbitrary-object zero-callback rejection; no unconditional deepcopy isolation; no uniform pre-traversal input bound; no protocol side-effect sandbox; no hard cancellation; no rollback of external effects; no certification of unreviewed extensions. M16/M17 reported metrics/provenance do not establish checkpoint origin, currentness, causality, significance or an automatic winner. Existing path/reader defenses remain required, not waived by host trust.

### Versioned M18/M19 Distinction

<!-- LTH-V1:R08 -->
M18 and M19 have separately scoped exact-type, fixed-field and bounded snapshot contracts. M16/M17 do not inherit those callback-rejection or input-bound guarantees; a bounded output does not establish a uniform pre-traversal input bound.
<!-- /LTH-V1:R08 -->

The current M18/R4b and M19 snapshots admit exact concrete types and approved fixed project fields with their own byte/depth/cycle checks. Keep those guarantees and rejection tests unchanged. These APIs assume trusted class implementations; they do not isolate arbitrary modifications to the whole Python process. No M20/M21/M22 or R4a/R4b/R4c2a contract is altered.

### Residual Risks

Host review can fail; a reviewed custom object can retain aliases or grow/change unexpectedly; YAML/history inputs lack a uniform legacy read bound; direct callers can omit authorization; SDK/custom Tool code executes in the host process. External integrations and dependency implementations are not comprehensively certified by this repository audit. This contract cannot justify a supported untrusted executable-object ingestion path if one is later identified.

## Runtime and transport

Provider requests and Tool results have existing runtime limits. Deadlines are cooperative checkpoints, not hard preemptive cancellation of a blocking filesystem read or arbitrary callback. Session history rollback does not undo external side effects already performed by host code.

An injected real provider may receive conversation and Tool results. Transcripts and HTTP exports are not automatically safe to publish. Configure redaction, retention and access controls in the host application. The offline examples instantiate no real provider, need no keys, make no external requests and execute no subprocesses.

The HTTP adapter is intended for an injected, single-process/single-worker host. It provides no complete multi-user authentication or identity isolation, persistence, CORS policy or server launcher. Public deployment requires additional host controls and review.

## Example errors and fixtures

A/B/G create tiny files in an owned temporary directory before Tool execution. Their fixture preparation and cleanup are writes; the demo does not claim zero filesystem writes. The analysis/observation Tool operations remain read-only. C–F use in-memory snapshots.

At the CLI presentation boundary, exceptions produce a stable `execution_failed` envelope without raw exception strings or private paths. Errors do not become fabricated successful reports. Argument errors exit 2; unsupported G exits 3. Tests exercise network denial, protected dispatch, binding restoration and history rollback. These named Python probes are regression evidence, not an OS sandbox proof or a guarantee against arbitrary injected host code.

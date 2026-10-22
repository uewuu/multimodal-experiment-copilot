# V1 architecture

The host selects data and authority. Deterministic code evaluates evidence; an optional provider chooses advertised tools and explains their structured results. The offline example substitutes provider messages only.

```mermaid
flowchart TD
    Host[Host: data selection and authorization] --> Binding[BoundToolCollection]
    Host --> Session[CopilotSession: context and history]
    Binding --> Bound[BoundCopilotSession]
    Session --> Bound
    Bound --> Runtime[Runtime and Adapter]
    Runtime <--> Provider[Injected Provider or offline fake]
    Runtime --> Policy[Argument and Path Policy]
    Policy --> Tool[Explicitly bound Tool]
    Tool --> Core[Deterministic Core]
    Tool --> Reader[Selected Reader when needed]
    Reader --> Files[Authorized filesystem observation]
    Core --> Result[Structured Tool result]
    Reader --> Result
    Result --> Runtime
    Runtime --> History[Commit history after successful turn]
```

This shows a call boundary, not an automatic sequence of milestones. Each A–G workflow is selected independently. M20 is never automatically combined with M21.

## Workflow selection

| Layer | Responsibility | Evidence boundary |
| --- | --- | --- |
| M11 identity/comparability | Explicit experiment/execution keys and policy comparison | Agreement of declarations, not their truth |
| M12 Memory/Repository | Historical storage, lifecycle and queries | An active record is not a current artifact |
| M13 evaluation | Structured expected facts, scenarios and baseline comparisons | Scored structured contracts, not universal prose verification |
| M14 Memory tools | Explicit scope/provider binding | Host controls discovery scope |
| M15 controlled workflow | Approval, lifecycle and idempotency for simulation | No real training or deployment action |
| M16 evidence | Read selected configuration, history and CSV evidence | File-reported metrics retain unknown checkpoint binding |
| M17 pairwise evidence | Two experiments, explicit metric targets, B minus A | Alignment and M11 comparability determine allowed observation |
| M18 aggregation | Full expected roster, eligibility, coverage, descriptive statistics | Snapshots and membership are host-declared |
| M19 group comparison | Real M18 twice plus M11 cross-group comparison | Unpaired difference of group means |
| M20 descriptors | Compare declared and reported fields in memory | `host_reported`; no file observation |
| M21 observation | Authorized handle, bounded bytes, incremental digest | Observed stream; no checkpoint/currentness proof |
| M22 explicit pairs | One-to-one declarations, full-group gates, pair deltas | Host-declared paired descriptive statistics |

M17 compares two evidence bundles and does not define replicate pairing. M19 uses each group's aggregation and does not infer pairs. M22 retains full execution identities, declared-pair denominators, missing/excluded/unpaired partitions and per-pair values. It still sends both complete groups to M18: dropping an unpaired conflicting member to obtain a favorable statistic would violate the contract. Summary precedence remains blocked, insufficient, limited, eligible.

## Entry points and lifetime

`CopilotService` borrows a provider client for one-shot calls and Session creation. `CopilotSession` owns bounded in-memory conversation history. `BoundCopilotService` and `BoundCopilotSession` temporarily activate a stable collection and restore the previous binding on both success and failure. No second Session framework is introduced by the demo.

The default Registry contains exactly `analyze_experiment` and `compare_experiments`. `BoundToolCollection` copies schemas, rejects duplicate names and adds only host-selected advanced tools. Snapshot selection is not delegated to the provider. On the formal Session/Adapter path, live file tools pass through Path Policy and their own binding checks; direct Python callers own applicable authorization.

On a successful demo turn the Adapter dispatches one tool, sends its actual JSON result to the fake provider, and receives a facts projection. Session history commits only when the turn succeeds. A failure restores binding and leaves history uncommitted; this cannot undo arbitrary host callback side effects.

### Host construction versus model invocation

The [M16/M17 Legacy Trusted-Host Python API (v1)](v1_security.md#m16-m17-legacy-trusted-host-python-v1) separates Python object trust from filesystem authority. The host constructs Tools before a Session invocation: M16 may call `deepcopy`, and M17 may traverse mappings and copy options/targets at that point. `BoundToolCollection` also accesses Tool interfaces and copies schemas. `BoundCopilotSession` activates the selected collection for a call; it does not sanitize hidden host objects or authorize earlier construction.

The Adapter decodes ordinary model arguments as JSON data, then the formal Session path policy and bound Tool checks constrain invocation. The model cannot replace hidden selections with arbitrary Python instances through that JSON schema. An authorized request can nevertheless invoke protocols of the objects the host previously bound. SDK response objects and custom Tool/client implementations are executable host code, distinct from decoded JSON. External plugins, custom hosts and SDK extensions remain **NOT AUDITED / HOST RESPONSIBILITY**.

Host review must precede construction and cover nested protocols, resource size and stable object lifetime. Cooperative copying preserves ordinary snapshots; custom copying can retain aliases. Neither history rollback nor binding restoration reverses callback side effects. Direct Core/Reader/Tool/Adapter calls do not gain missing Session authority automatically.

M18/M19 retain separate exact-type, fixed-field and bounded snapshot contracts. Those guarantees do not transfer to M16/M17, and bounded legacy output does not bound every earlier traversal, YAML/history load or callback duration. No runtime hardening or broader extension certification is introduced by the documentation contract.

## I/O separation

M18/M19/M20/M22 consume supplied structures; they do not discover files or retrieve Memory. M16/M17 use selected readers. M21 separates pure binding/projection (`artifact_observation.py`), authorized Windows handles (`artifact_observation_io.py`) and bounded streaming (`artifact_observation_reader.py`). Its Tool invokes the Reader afresh without caching. The examples perform fixture preparation outside Tool dispatch.

The source delivery remains Python 3.11. The existing CLIs and injected HTTP adapter share the runtime, but the new advanced workflows are demonstrated through explicit Python binding. Existing CLI/HTTP exposure is not silently expanded.

# Security, trust and platform boundaries

## Host authority

The host owns client configuration, Tool binding, experiment roots, expected rosters, target mappings, pairing declarations and artifact selection. Model arguments cannot enlarge that authority. Path Policy and exact binding checks execute before live Reader invocation. Default Registry exposure remains two tools; advanced Tools are opt-in.

M21 additionally validates the opened object through a Windows handle. Lexical/path authorization alone is not opened-object identity. Its model schema exposes only `experiment_dir` with the single value `"."`; the host binds the filename and root. Snapshot Tools C–F accept an empty object and cannot use model arguments to replace host data.

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

Frozen snapshot boundaries reject unsupported objects and retain exact-type/callback safety contracts. The example builds known generic data; it is not a new arbitrary-object deserializer. Host callbacks remain trusted application code. M15 stays simulation-only; its approval contract does not authorize real training, HPO, checkpoint loading or deployment.

Memory discovery is separate from artifact verification. Repository writes exist for explicit host lifecycle operations; the read-only Memory interface does not make the entire repository layer read-only. The demo does not open Memory or a production database.

## Runtime and transport

Provider requests and Tool results have existing runtime limits. Deadlines are cooperative checkpoints, not hard preemptive cancellation of a blocking filesystem read or arbitrary callback. Session history rollback does not undo external side effects already performed by host code.

An injected real provider may receive conversation and Tool results. Transcripts and HTTP exports are not automatically safe to publish. Configure redaction, retention and access controls in the host application. The offline examples instantiate no real provider, need no keys, make no external requests and execute no subprocesses.

The HTTP adapter is intended for an injected, single-process/single-worker host. It provides no complete multi-user authentication or identity isolation, persistence, CORS policy or server launcher. Public deployment requires additional host controls and review.

## Example errors and fixtures

A/B/G create tiny files in an owned temporary directory before Tool execution. Their fixture preparation and cleanup are writes; the demo does not claim zero filesystem writes. The analysis/observation Tool operations remain read-only. C–F use in-memory snapshots.

At the CLI presentation boundary, exceptions produce a stable `execution_failed` envelope without raw exception strings or private paths. Errors do not become fabricated successful reports. Argument errors exit 2; unsupported G exits 3. Tests exercise network denial, protected dispatch, binding restoration and history rollback. These named Python probes are regression evidence, not an OS sandbox proof or a guarantee against arbitrary injected host code.

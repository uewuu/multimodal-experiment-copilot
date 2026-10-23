# Testing and release evidence

## Status and evidence scope

M9–M23 V1 is integrated into `main` by [PR #59](https://github.com/uewuu/multimodal-experiment-copilot/pull/59). The verified M9–M23 V1 integration baseline is commit `e437795e041247bba9ae9c83b7af37a77ae0f706`, tree `e3e4c4e2acd65db4fed8b64570bc41d868ec3516`, with ordered merge parents `6afec267c11925183be41008f7021ee85d7ff96d` and `b34f4161903ca963c18b74793418d980054b1955`.

[Main Run #70 (39757399767)](https://github.com/uewuu/multimodal-experiment-copilot/actions/runs/39757399767) is the successful post-merge `push` acceptance on that exact commit. Earlier PR CI and temporary-branch CI are historical evidence, distinct from this main run.

| Current main lane | Files | Global / per-file / executed (G/F/E) | PASS | FAIL / ERROR / SKIP / XFAIL / XPASS |
| --- | ---: | --- | ---: | --- |
| [Windows Full](https://github.com/uewuu/multimodal-experiment-copilot/actions/runs/39757399767/job/119752783351) | 91 | 4891 / 4891 / 4891 | 4891 | 0 / 0 / 0 / 0 / 0 |
| [Linux Portable](https://github.com/uewuu/multimodal-experiment-copilot/actions/runs/39757399767/job/119752783794) | 23 | 2030 / 2030 / 2030 | 2030 | 0 / 0 / 0 / 0 / 0 |

G/F/E reconciliation compares exact node-ID lists, not only counts. The platform selections overlap and differ; adding the totals would not yield a unique test count. These outcomes are not code-coverage percentages. Current [requirements](../requirements.txt) require pytest `>=9.0.3,<10`, and [release constraints](../constraints-release-py311.txt) pin `9.0.3`, as installed in both Run #70 jobs. CI uses Python 3.11, base dependencies only and no optional provider SDK.

### Historical M23 evidence

The following Slice 1–3 records predate the final integration and audit fixes; their counts and environment versions are retained as historical evidence.

| Stage | Evidence | Meaning |
| --- | --- | --- |
| M23 planning, historical | 83 test files / 4576 collected node IDs | Collection inventory, not execution success |
| Slice 1, historical | 86 files / 4647 collected; M23 71 collected, 11 PASS, 60 expected FAIL | Acceptance contracts frozen; 30 Demo and 30 Runner failures |
| Slice 2, historical | Demo + safety: 41 PASS | Real bound workflows, native G, offline/authorization/rollback contracts |
| Slice 2, full M23 | 71 collected, 41 PASS, 30 expected FAIL | Remaining failures are exclusively missing Runner; unexpected/error/skip/xfail = 0 |
| Slice 2, collection | 86 files / 4647 node IDs | Exact inventory preserved; no tests added or modified |
| Slice 2, specified frozen regressions | 2276 PASS | Independent suite processes, listed below; not a full-repository execution claim |
| Slice 3, Windows full | 86 files / 4647 collected / 4647 executed / 4647 PASS | Exact global/per-file/executed IDs match; no duplicate, extra or missing IDs |
| Slice 3, M23 | 71 / 71 PASS | Runner implemented; remaining RED = 0 |
| Slice 3, native Windows | 8 / 8 native tests PASS | Real safe-open/observation/authorization; local fixed NTFS |
| Slice 3, fresh environment | Install, pip check, imports, existing demo and A-G PASS | Separate Python 3.11.15 venv, user site disabled, no global dependency changes |
| Slice 3, fresh base full matrix | 86 files / 4647 collected / 4647 executed / 4647 PASS | SDK-free base dependencies; final CI configuration; exact G/F/E match |
| Slice 3, portable command on Windows | 22 files / 1964 collected / executed / PASS | Local CLI/allowlist validation only; NOT Linux execution |
| Slice 3, Linux | 22 files / 1964 collected / 1964 executed / 1964 PASS | Actual Ubuntu 24.04 / Python 3.11.17; exact G/F/E; all non-pass counts zero |
| Slice 3, CI | Temporary Linux validation SUCCESS | Run 38114641921; remote Windows CI was pending at that stage |

Historical Slice 3 Windows evidence used Python 3.11.15, pytest 8.3.4 and the dependency pins recorded at that stage; the current constraints now pin pytest 9.0.3. The workspace and temporary volume were checked with `GetVolumePathNameW`, `GetVolumeInformationW` and `GetDriveTypeW`: both report NTFS and local fixed drive type 3. Native M21 tests actually executed; seam-only results are not substituted for them. The historical full run had zero FAIL/ERROR/SKIP/XFAIL/XPASS, collection failures, process failures or timeouts.

All 30 previously failing Runner contracts now pass. They cover dynamic collection (1), corrupt/incomplete child evidence (19), non-pass outcomes (5), forbidden partial full-lane selection (1), portable scope reporting (1), stable reports (1), real child isolation (1), and setup/teardown outcome reconciliation (1). A separate two-file parameterized temporary project also produced exact G/F/E equality for all four executed node IDs.

## Runner and evidence commands

From the source root, with Python 3.11 and documented dependencies:

```bash
python -B -m pytest -p no:cacheprovider -q tests/test_v1_bound_demo_contract.py tests/test_v1_release_safety_contract.py
python -B -m pytest -p no:cacheprovider -q tests/test_v1_bound_demo_contract.py tests/test_v1_release_runner_contract.py tests/test_v1_release_safety_contract.py
python -B -m pytest -p no:cacheprovider --collect-only -q tests
python -B scripts/run_release_checks.py --lane windows-full
# Portable subset only; actual Ubuntu execution verified separately:
python -B scripts/run_release_checks.py --lane portable --test-list scripts/release-portable-files.json
```

The first two commands now pass. The third is collection only, never execution evidence. Full acceptance uses the Runner. JSON goes to stdout; pytest diagnostics, warnings and a summary go to stderr. Redirect both streams to an explicitly chosen directory outside the checkout when retaining evidence; PowerShell 5 redirection produces UTF-16, whereas Python UTF-8 writers and modern shells can produce UTF-8. Do not commit reports or virtual environments.

`run_checks(root, *, lane, test_files=None)` uses the current interpreter for one global collection and one execution subprocess per selected file. `windows-full` discovers `tests/**/test_*.py` dynamically and refuses selection overrides. `portable` requires explicit files (`--test-list` JSON or repeated `--test-file`) and records every exclusion. Stable JSON includes all three ID lists, per-file exits/outcomes, counts, environment and reason codes. Exit 0 means passed; 1 means blocked; invalid CLI arguments return 2.

Children use `-B`, `PYTHONDONTWRITEBYTECODE=1`, disabled pytest cache/plugin autoload, no shell and no retry. Each child has a 300-second subprocess timeout. Timeout/interruption blocks acceptance; subprocess cleanup handles the immediate child. This is not a guarantee of preempting arbitrary descendant processes or blocking filesystem I/O. Setup/teardown failures collapse to one terminal error per node. Every non-pass outcome blocks acceptance. Diagnostics are preserved separately rather than parsed as test-count evidence.

### Portable selection rationale

The current [portable allowlist](../scripts/release-portable-files.json) explicitly selects 23 files (68 of the 91 Windows files are excluded from this lane), including the default-tool path-containment regressions. It covers M11/M17-M20/M22 pure-memory analysis and snapshot safety, Session/Runtime/Adapter, CLI/HTTP, the existing demo, A-F, and G's unsupported-platform behavior. The M21 I/O file exercises controlled native seams and unsupported-platform rejection; it is not native Linux safe-open acceptance. The historical Slice 3 list selected 22 files and excluded 64.

Windows-only native acceptance is excluded. Runner contracts that deliberately assert successful `windows-full` operation are excluded from Linux. Other files remain covered by Windows full acceptance and are not certified by this deliberately narrower portable list. No failing test was removed from Windows full selection. Linux CI additionally invokes real G on Linux and requires exit 3, `unsupported_platform`, and no successful digest. Main Run #70 passed the current portable scope and G check; the temporary Ubuntu validation run passed the historical scope. Neither establishes full Linux repository or POSIX native-reader acceptance.

### Historical GitHub Actions Linux evidence

[Run 38114641921](https://github.com/uewuu/multimodal-experiment-copilot/actions/runs/38114641921), job `linux-portable` (114397003867), completed successfully on 2026-10-11. The temporary branch is `validation/m23-linux-portable-20261010`, commit `f64af88def71788fccd27e4e3391ae662c8cf441`, parent `015839f8cbb3779a95e2b6e5648b7b4feab16420`. The temporary workflow belongs only to that validation branch; it is not part of the formal Slice 3 delivery.

The historical platform report records Linux / x86_64, Ubuntu 24.04 runner, Python 3.11.17, pytest 8.3.4 and pip 26.2.1. The base dependency versions matched the constraints at that stage (including FastAPI 0.141.1, Pydantic 2.13.5, httpx 0.28.1 and Starlette 1.6.0); the optional OpenAI SDK was absent. `pip check` passed. All 22 files ran in independent subprocesses: 1964 collected, 1964 executed and 1964 passed, with zero FAIL/ERROR/SKIP/XFAIL/XPASS or process/collection failures. Downloaded raw job logs supplied complete G/F/E node-ID lists, independently reconciled locally with no duplicates, extras or omissions. This historical pytest version is not the current installation constraint.

The same Ubuntu job executed the existing deterministic demo and A-F with exit 0 and checked their frozen golden facts. G returned exit 3, `unsupported_platform`, and no digest. The run's head SHA was verified through the REST API rather than inferred from a green checkmark.

Source manifest SHA-256: `0895cdec9b675e000570778eebfaa73fc4bb949063005420dddb570b2399bfc5`. Before status-document updates, all seven original worktree file hashes exactly matched the temporary worktree, committed blobs and cloud-observed hashes, including original line endings. Runner, allowlist, constraints, formal CI configuration, frozen analysis code and tests remain unchanged after validation. Only README/Quickstart/testing-document evidence statements were subsequently updated; executable commands were preserved. Windows evidence is reused from its recorded runs, not described as newly executed Linux evidence.

### Fresh environment reproduction and historical evidence

Use an isolated Python 3.11 venv. In historical Slice 3 checks, two environments outside the checkout were verified: base dependencies only, and base plus optional OpenAI. Both had `sys.prefix != sys.base_prefix`, `include-system-site-packages = false` and `site.ENABLE_USER_SITE = false`. The interpreter came from agent311, but its installed packages were not inherited or changed. Both installs used pip 24.0 and passed imports, pip check and all eight demo commands. The final base-only environment also passed the then-complete 4647-test matrix. Optional OpenAI was checked for import compatibility only; no real provider was called. Default CI installs only base dependencies. The commands below use the current constraints, not the historical pytest pin.

Windows PowerShell (choose a new temporary path):

```powershell
$releaseVenv = Join-Path $env:TEMP ('copilot-release-' + [guid]::NewGuid().ToString('N'))
python -m venv $releaseVenv
& "$releaseVenv\Scripts\python.exe" -m pip install -c constraints-release-py311.txt -r requirements.txt
& "$releaseVenv\Scripts\python.exe" -m pip check
& "$releaseVenv\Scripts\python.exe" -B examples/deterministic_copilot_demo.py
foreach ($scenario in 'A','B','C','D','E','F','G') {
    & "$releaseVenv\Scripts\python.exe" -B examples/v1_bound_workflows_demo.py --scenario $scenario
    if ($LASTEXITCODE -ne 0) { throw "Demo failed: $scenario" }
}
& "$releaseVenv\Scripts\python.exe" -B scripts/run_release_checks.py --lane windows-full
```

The historical direct combination was PyYAML 6.0.3, pytest 8.3.4, FastAPI 0.141.1, Pydantic 2.13.5, httpx 0.28.1, Starlette 1.6.0 and optional openai 2.54.0. Current constraints retain these direct versions except pytest, now 9.0.3; they also record resolved dependencies. Historical `pip check`, base imports and all eight demo commands passed. Run #70 supplies current constrained base-dependency acceptance on Windows and the Linux portable subset; it does not establish real-provider behavior. Pins are not a cross-platform lock or package-authenticity guarantee. The known Starlette/httpx deprecation warning remains visible.

Optional SDK installation remains a separate command: `python -m pip install -c constraints-release-py311.txt -r requirements-openai.txt`. The constraints file alone does not install the SDK. An intermediate full run detected that the initial CI edit included optional SDK installation (4646 PASS / 1 FAIL in the frozen SDK-free CI contract). That CI configuration was corrected without modifying the frozen test; its 79-test file and the final fresh-base full matrix passed. The failing run remains diagnostic evidence, not final acceptance.

## Historical specified regression commands

Run each regression command as a separate process, especially Adapter. These are the exact specified suites, not an implicit full-repository selection:

```bash
python -B -m pytest -p no:cacheprovider -q tests/test_paired_replicate_comparison.py tests/test_paired_replicate_comparison_bounds.py tests/test_paired_replicate_comparison_snapshot_safety.py tests/test_paired_replicate_comparison_tools.py tests/test_paired_replicate_comparison_evaluation.py
python -B -m pytest -p no:cacheprovider -q tests/test_artifact_observation.py tests/test_artifact_observation_io.py tests/test_artifact_observation_reader.py tests/test_artifact_observation_tools.py tests/test_artifact_observation_snapshot_safety.py tests/test_artifact_observation_evaluation.py tests/test_artifact_observation_windows.py
python -B -m pytest -p no:cacheprovider -q tests/test_artifact_verification.py tests/test_artifact_verification_tools.py tests/test_artifact_verification_snapshot_safety.py tests/test_artifact_verification_evaluation.py
python -B -m pytest -p no:cacheprovider -q tests/test_replicate_group_comparison.py tests/test_replicate_group_comparison_tools.py tests/test_replicate_group_comparison_snapshot_safety.py tests/test_replicate_group_comparison_evaluation.py tests/test_replicate_group_comparison_dataclass_metadata_safety.py
python -B -m pytest -p no:cacheprovider -q tests/test_replicate_aggregation.py tests/test_replicate_aggregation_tools.py tests/test_replicate_aggregation_snapshot_safety.py tests/test_replicate_aggregation_evaluation.py
python -B -m pytest -p no:cacheprovider -q tests/test_copilot_memory_integration.py
python -B -m pytest -p no:cacheprovider -q tests/test_openai_tool_adapter.py
python -B -m pytest -p no:cacheprovider -q tests/test_copilot_runtime_observability.py tests/test_copilot_runtime_failure_observability.py
```

| Suite | PASS |
| --- | ---: |
| M22 | 317 |
| M21 | 501 |
| M20 | 655 |
| M19 | 401 |
| M18 | 211 |
| Memory integration | 10 |
| Adapter, isolated | 82 |
| Runtime observability | 99 |
| Total | 2276 |

These Slice 2 suite totals remain historical evidence. Slice 3 uses the dynamically discovered full repository instead of treating this 2276-test subset as full coverage.

## Integration acceptance and remaining publication gates

PR #59 is merged and main Run #70 passed formal Windows full and Linux portable acceptance at the exact main SHA above. The historical Slice 3 records remain supporting evidence; pre-merge PR CI is not substituted for post-merge main acceptance.

The acceptance matrix remains defined in `tests/m23_release_contract_helpers.py`. Functional/offline/integration and Windows node-ID coverage/isolation/native gates have historical local evidence and the post-merge cloud acceptance above; Linux has only portable acceptance. Final version authorization, a new tag and Release publication remain separate from completed main integration and CI acceptance.

The [formal workflow](../.github/workflows/tests.yml) installs constrained dependencies on Windows and Linux. Windows probes workspace and temporary filesystems, fails closed unless both are local NTFS, runs the full Runner and checks that real native test outcomes exist and all passed. Linux uses only the explicit allowlist and a real unsupported-platform check. Both formal jobs passed in main Run #70. Linux has no accepted M21 POSIX safe-open backend.

Any later release review must bind evidence to the exact candidate SHA and cumulative scope. A new tag or Release requires separate authorization. No remote or release action is performed by the demos.

## Version and delivery

The repository has an annotated `v0.1.0` tag and a historical [v0.1.0 Release](https://github.com/uewuu/multimodal-experiment-copilot/releases/tag/v0.1.0), separate from the M9–M23 V1 integration. `v0.2.0` remains a planning candidate; no new tag or Release has been published for this integration. V1 delivery targets source-based Python 3.11 use; editable packaging, wheels, PyPI publication and full SemVer API stability are not promised. Main integration, CI acceptance and release publication remain separate claims.

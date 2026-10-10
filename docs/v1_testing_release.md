# Testing and release evidence

## Status and evidence scope

| Stage | Evidence | Meaning |
| --- | --- | --- |
| M23 planning, historical | 83 test files / 4576 collected node IDs | Collection inventory, not execution success |
| Slice 1, historical | 86 files / 4647 collected; M23 71 collected, 11 PASS, 60 expected FAIL | Acceptance contracts frozen; 30 Demo and 30 Runner failures |
| Slice 2, current work | Demo + safety: 41 PASS | Real bound workflows, native G, offline/authorization/rollback contracts |
| Slice 2, full M23 | 71 collected, 41 PASS, 30 expected FAIL | Remaining failures are exclusively missing Runner; unexpected/error/skip/xfail = 0 |
| Slice 2, collection | 86 files / 4647 node IDs | Exact inventory preserved; no tests added or modified |
| Slice 2, specified frozen regressions | 2276 PASS | Independent suite processes, listed below; not a full-repository execution claim |

The completed evidence was obtained on the existing Windows Python 3.11 environment. Initial sandbox restrictions prevented temporary-directory access; verification used permitted execution outside that sandbox. This is not a clean-environment installation result or CI acceptance.

The remaining 30 Runner cases are ordinary FAIL outcomes. They cover dynamic collection (1), corrupt/incomplete child evidence (19), non-pass outcomes (5), forbidden partial full-lane selection (1), portable scope reporting (1), stable reports (1), real child isolation (1), and setup/teardown outcome reconciliation (1). No skips or xfails hide missing delivery work.

## Reproduce completed checks

From the source root, with Python 3.11 and documented dependencies:

```bash
python -B -m pytest -p no:cacheprovider -q tests/test_v1_bound_demo_contract.py tests/test_v1_release_safety_contract.py
python -B -m pytest -p no:cacheprovider -q tests/test_v1_bound_demo_contract.py tests/test_v1_release_runner_contract.py tests/test_v1_release_safety_contract.py
python -B -m pytest -p no:cacheprovider --collect-only -q tests
```

The second command currently exits nonzero for the expected Runner failures. The third command is collection only. A successful collection does not demonstrate that every test passed.

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

The known Starlette/httpx deprecation warning appears during HTTP-related collection/testing. It is retained in evidence; changing dependency constraints is outside Slice 2.

## Remaining release gates

Slice 3 must implement and validate the release Runner, correct CI orchestration, install dependencies in fresh environments, execute Windows native full regression, execute an explicit Linux portable subset and record resolved dependency versions. None of those steps is certified by the tables above.

The frozen acceptance design requires dynamic discovery, exact collected/executed node-ID reconciliation and one execution subprocess per test file using the current interpreter. Missing/duplicate/extra IDs, malformed child reports, errors, skips, xfails or xpasses must block full acceptance. A partial portable lane must list exclusions and cannot claim Windows native acceptance. These are future implementation requirements; `scripts/run_release_checks.py` is not implemented in Slice 2.

The existing Ubuntu workflow runs a combined suite. Its presence is not Windows acceptance or proof of test-file isolation. Linux has no accepted M21 POSIX safe-open backend; explicit unsupported behavior is distinct from native success.

Subsequent independent review must bind evidence to the exact release candidate SHA and cumulative scope. Remote synchronization and any merge/tag/PR/Release require their separately authorized gates. No remote or release action is performed by the demos.

## Version and delivery

The repository has an annotated `v0.1.0` tag. `v0.2.0` is a planning candidate for source delivery, not a newly assigned version or tag. V1 delivery targets source-based Python 3.11 use; editable packaging, wheels, PyPI publication and full SemVer API stability are not promised. Implementation frozen, remote synced, release acceptance complete and released remain separate claims.

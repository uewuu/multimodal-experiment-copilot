# V1 source quickstart

Use Python 3.11 and run from the repository root. This is source delivery; there is no supported `pip install -e .`, wheel or PyPI package. Fresh-environment validation is still a Slice 3 gate. The commands below describe the existing files and entry points; they are not a claim that a clean installation has been accepted.

## Get the source and dependencies

```bash
git clone https://github.com/uewuu/multimodal-experiment-copilot.git
cd multimodal-experiment-copilot
```

Use a maintainer-provided revision containing the M23 examples. Until that work is published, a default-branch clone may not contain this document or the new demo. No release tag is assigned by this guide.

Windows PowerShell:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

Linux shell:

```bash
python3.11 -m venv .venv
. .venv/bin/activate
python -m pip install -r requirements.txt
```

The base requirements include PyYAML, pytest and the HTTP adapter dependencies. The demos need no API keys or provider SDK. Only for real OpenAI-compatible provider use:

```bash
python -m pip install -r requirements-openai.txt
```

The real provider may receive prompts and tool results. Review [security](v1_security.md) before supplying private experiment data.

## Run the existing demo

```bash
python examples/deterministic_copilot_demo.py
```

It creates temporary synthetic experiments, calls the real default comparison tool through a fake provider, and prints JSON. Its output shape predates the new scenario envelope.

## Run the seven independent bound workflows

```bash
python examples/v1_bound_workflows_demo.py --scenario A
python examples/v1_bound_workflows_demo.py --scenario B
python examples/v1_bound_workflows_demo.py --scenario C
python examples/v1_bound_workflows_demo.py --scenario D
python examples/v1_bound_workflows_demo.py --scenario E
python examples/v1_bound_workflows_demo.py --scenario F
python examples/v1_bound_workflows_demo.py --scenario G
```

| Scenario | Host input | Actual computation and compact facts |
| --- | --- | --- |
| A | One temporary experiment, explicit CSV column and metric spec | M16 file evidence; `test_value=2.0` |
| B | Two temporary experiments and explicit target mapping | M16 + M17; `raw_delta=1.0`, B minus A |
| C | Complete expected roster and snapshots `[1,3]` | M18; `count=2`, `mean=2.0` |
| D | Groups A `[1,3]`, B `[2,7]` | M19 unpaired mean difference `2.5` |
| E | Same groups; explicit pairs A0–B1 and A1–B0 | M22 paired deltas `[6.0,-1.0]`, mean `2.5` |
| F | Matching synthetic host-reported descriptor fields | M20; SHA comparison `match`, basis `host_reported` |
| G | Authorized root and direct-child `artifact.bin` containing `abc` | M21 native observation; actual byte count 3 and SHA-256 |

These values are computed or projected from real tool responses. The provider never supplies a replacement report. D and E happen to have equal summary values here, but E also retains individual pair endpoints, deltas, declared-pair coverage and unpaired members. No pairing is inferred from seeds or input order.

G requires Windows with local NTFS temporary storage. It emits `unsupported` on unsupported platforms; this is not native success. Linux execution evidence remains pending. A/B/G write only their temporary fixture inputs before tool execution and clean them up afterwards. C–F do not acquire files, Memory records or network data.

## Inspect output and exit codes

Each new demo invocation emits one strict JSON object:

- `schema_version`: `v1-bound-demo-v1`.
- `scenario`, `tool_name`, `status`: workflow identity and `ok`, `unsupported` or `error`.
- `result`: the full actual report with display-only metadata normalization; `None`/JSON `null` on execution errors.
- `facts`: a compact projection grounded in that report, not an independent truth claim.
- `limitations`: demo limitations and available root report limitations; also inspect limitations inside `result`.
- `runtime`: actual provider-request and tool-invocation counts (2 and 1 for a completed scenario).
- `error_code`: safe `execution_failed` on execution error; no raw exception or private path.

Display normalization replaces run/time identifiers, zeroes displayed observation mtimes, and reduces experiment path fields to basenames. It operates on a detached report and does not modify Tool/Core results. The demo output is suitable for illustration, not an archival raw-observation record.

| Exit | Meaning |
| --- | --- |
| 0 | Scenario completed |
| 1 | Execution failed; inspect the safe error envelope |
| 2 | Invalid CLI arguments; argparse diagnostic on stderr, no scenario dispatch |
| 3 | G platform unsupported; no successful observation digest |

Use `--help` for CLI syntax. Never interpret `ok` as verified provenance, causality, significance or a release acceptance result.

## Adapt the host binding

Read [the fixture builders](../examples/v1_workflow_fixtures.py) for complete generic input shapes and [the demo](../examples/v1_bound_workflows_demo.py) for the integration path. They import no test helpers. Replace synthetic inputs in host code, then construct the relevant public Tool and a `BoundToolCollection`. The Session separately needs trusted `experiment_context` for file access. Binding a Tool does not itself grant filesystem authorization.

Default tools remain advertised. Only the selected advanced Tool is added for each scenario. Snapshot tools take no model arguments; the host supplies rosters, targets, descriptors and pairing. The G model argument `experiment_dir="."` refers to the already-authorized root, never to a model-selected filename.

"""Seven independent offline workflows through the real bound Copilot stack.

Only provider messages are simulated. Host fixtures, selected tools, Session,
Adapter and computations use the existing APIs. Run --scenario A through G.
"""

import argparse
from contextlib import ExitStack
import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
from types import SimpleNamespace as NS

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from copilot import CopilotSession
from copilot.tool_binding import BoundCopilotSession
from llm_adapters.tool_binding import BoundToolCollection
from tool_layer.experiment_tools import ExperimentEvidenceTools, ExperimentComparisonTools
from tool_layer.replicate_tools import ReplicateAggregationTools
from tool_layer.replicate_group_tools import ReplicateGroupComparisonTools
from tool_layer.paired_experiment_tools import PairedReplicateComparisonTools
from tool_layer.artifact_verification_tools import ArtifactVerificationTools
from tool_layer.artifact_observation_tools import ArtifactObservationTools
from examples import v1_workflow_fixtures as fixtures


_TOOLS = dict(zip('ABCDEFG', (
    'analyze_experiment_evidence', 'compare_experiment_evidence',
    'aggregate_replicate_evidence', 'compare_replicate_groups',
    'compare_paired_replicates', 'compare_artifact_descriptors', 'observe_artifact',
)))


def _facts(scenario, result):
    """Project facts from a REAL tool response; do not recompute statistics."""
    if scenario == 'A':
        return {'test_value': result['test_results']['metrics']['score']['value']}
    if scenario == 'B':
        return {'raw_delta': result['targets']['quality']['calculation']['raw_delta']}
    if scenario == 'C':
        stats = result['targets']['quality-A']['statistics']
        return {'count': stats['count'], 'mean': stats['mean']}
    if scenario == 'D':
        return {'comparison_policy': result['comparison_policy'],
                'raw_delta': result['targets']['quality']['comparison']['raw_delta']}
    if scenario == 'E':
        target = result['targets']['quality']
        return {'comparison_policy': result['comparison_policy'],
                'pairing_source': result['pairing_source'],
                'pair_deltas': [target['pair_results'][key]['calculation']['raw_delta']
                                for key in sorted(target['pair_results'])],
                'mean': target['statistics']['mean']}
    if scenario == 'F':
        row = result['artifacts']['metrics']
        return {'basis': result['evidence_basis'],
                'sha256_comparison': row['integrity']['sha256_comparison'],
                'currentness_status': row['currentness_status'],
                'checkpoint_binding_status': row['checkpoint_binding_status']}
    if result['status'] != 'observed':
        return {'status': result['status']}
    return {key: result[key] for key in ('status', 'sha256', 'byte_size', 'consistency_status')}


class _Completions:
    """Provider-only double. It requests one tool, then reads its actual reply."""

    def __init__(self, scenario):
        self.scenario = scenario
        self.count = 0

    def create(self, **request):
        self.count += 1
        if self.count == 1:
            arguments = {'A': {'experiment_dir': '.'},
                         'B': {'experiment_a': 'a', 'experiment_b': 'b'},
                         'G': {'experiment_dir': '.'}}.get(self.scenario, {})
            message = NS(role='assistant', content=None, tool_calls=[NS(
                id='demo-call', type='function', function=NS(
                    name=_TOOLS[self.scenario], arguments=json.dumps(arguments)))])
        elif self.count == 2:
            tool_message = request['messages'][-1]
            if tool_message['role'] != 'tool':
                raise ValueError('expected actual tool response')
            result = json.loads(tool_message['content'])
            message = NS(role='assistant', tool_calls=None,
                         content=json.dumps({'facts': _facts(self.scenario, result)}, allow_nan=False))
        else:
            raise RuntimeError('unexpected provider request')
        return NS(choices=[NS(message=message)])


def _make_client(scenario):
    return NS(chat=NS(completions=_Completions(scenario)))


def _host_binding(scenario, stack):
    """Select one tool; create files only for A/B/G before tool execution."""
    if scenario in 'ABG':
        workspace = Path(stack.enter_context(TemporaryDirectory(prefix='copilot-v1-'))).resolve(strict=True)
        if scenario == 'A':
            root = fixtures.write_experiment(workspace, 'experiment', 2.0)
            return ExperimentEvidenceTools(str(root), **fixtures.evidence_options()), {'experiment_dir': str(root)}
        if scenario == 'B':
            a = fixtures.write_experiment(workspace, 'a', 2.0)
            b = fixtures.write_experiment(workspace, 'b', 3.0)
            tool = ExperimentComparisonTools(str(a), str(b), options_a=fixtures.evidence_options(),
                                             options_b=fixtures.evidence_options(), targets=fixtures.evidence_targets())
            return tool, {'experiment_root': str(workspace)}
        (workspace / 'artifact.bin').write_bytes(b'abc')
        binding = dict(schema_version='artifact-observation-binding-v1',
                       artifact_id='artifact-1', logical_role='metrics', artifact_ref='host-artifact-1',
                       file_name='artifact.bin', authorized_root=str(workspace),
                       max_bytes=1048576, chunk_bytes=65536)
        return ArtifactObservationTools(binding), {'experiment_dir': str(workspace)}
    if scenario == 'C':
        return ReplicateAggregationTools(**fixtures.group('A', (1.0, 3.0))), None
    if scenario == 'D':
        return ReplicateGroupComparisonTools(fixtures.group('A', (1.0, 3.0)),
            fixtures.group('B', (2.0, 7.0)), targets=fixtures.group_targets()), None
    if scenario == 'E':
        a, b = fixtures.plain_group('A', (1.0, 3.0)), fixtures.plain_group('B', (2.0, 7.0))
        return PairedReplicateComparisonTools(a, b, pairing_declaration=fixtures.crossed_pairing(a, b),
                                              targets=fixtures.group_targets()), None
    return ArtifactVerificationTools(*fixtures.artifact_descriptors()), None


def _display(value, key=None):
    """Detach display metadata only; preserve every computation and limitation."""
    if key in ('started_at', 'completed_at', 'run_id') and value is not None:
        return '<normalized>'
    if key in ('mtime_ns_before', 'mtime_ns_after') and value is not None:
        return 0
    if key in ('experiment_dir', 'experiment_root') and type(value) is str:
        return Path(value).name
    if type(value) is dict:
        return {k: _display(v, k) for k, v in value.items()}
    if type(value) is list:
        return [_display(v) for v in value]
    return value


def run_scenario(name):
    """Return a fresh strict-JSON envelope; errors never expose exception text."""
    if type(name) is not str or name not in _TOOLS:
        raise ValueError('scenario must be A through G')
    report = dict(schema_version='v1-bound-demo-v1', scenario=name, status='error',
                  tool_name=_TOOLS[name], result=None, facts={},
                  limitations=['synthetic_demo_not_real_experiment_verification'],
                  runtime={'provider_request_count': 0, 'tool_invocation_count': 0},
                  error_code='execution_failed')
    if name == 'G' and sys.platform != 'win32':
        report.update(status='unsupported', facts={'status': 'unsupported_platform'},
                      error_code=None)
        return report
    failures = []
    try:
        with ExitStack() as stack:
            tool, context = _host_binding(name, stack)
            session = CopilotSession(_make_client(name), model='offline-demo', experiment_context=context)
            observed = BoundCopilotSession(session, BoundToolCollection(tool)).ask_with_observability(
                'Report the selected workflow facts and preserve its limitations.', on_failure=failures.append)
            if len(observed.turn.tool_invocations) != 1:
                raise ValueError('expected one tool invocation')
            result = json.loads(observed.turn.tool_invocations[0].result_json)
            facts = json.loads(observed.turn.answer)['facts']
            if facts != _facts(name, result):
                raise ValueError('provider facts do not match the tool response')
            report['runtime'] = dict(provider_request_count=observed.metrics.provider_request_count,
                                    tool_invocation_count=observed.metrics.tool_invocation_count)
            if name == 'G' and result['status'] not in ('observed', 'unsupported_platform'):
                return report
            report.update(status='unsupported' if name == 'G' and result['status'] == 'unsupported_platform' else 'ok',
                          result=_display(result), facts=facts, error_code=None)
            report['limitations'].extend(result.get('limitations', []))
            json.dumps(report, allow_nan=False)
        return report
    except Exception:
        # Presentation boundary only: Session/Tools keep their own error semantics.
        # Never format an exception or return a success-shaped partial report.
        report.update(status='error', result=None, facts={}, error_code='execution_failed')
        if failures:
            report['runtime'] = dict(provider_request_count=failures[-1].provider_request_count,
                                    tool_invocation_count=failures[-1].tool_invocation_count)
        return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--scenario', required=True, choices=list(_TOOLS))
    args = parser.parse_args(argv)
    report = run_scenario(args.scenario)
    print(json.dumps(report, ensure_ascii=False, allow_nan=False, sort_keys=True, indent=2))
    return {'ok': 0, 'error': 1, 'unsupported': 3}[report['status']]


if __name__ == '__main__':
    raise SystemExit(main())

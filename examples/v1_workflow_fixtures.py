"""Small, generic host inputs for the offline demo; never computed reports.

Snapshot scenarios use declared data, independently of the file fixtures.
These builders are example code, not public acquisition or validation APIs.
"""

from dataclasses import asdict
import json

import yaml

from experiment_identity import ExperimentIdentity, ExecutionIdentity, ExperimentProvenance
from metrics import MetricSpec


def metric_spec():
    return MetricSpec('score', ('measurement',), 'maximize', 'Score', 6)


def provenance(execution, seed=42):
    return ExperimentProvenance(
        execution, configuration_ref='recipe-v1', task='regression', target='quality',
        model_ref='model-v1', dataset_version='dataset-v1', split_id='heldout-v1',
        source_revision='revision-v1', initialization_checkpoint_ref='initialization-v1',
        realized_seed=seed, environment_ref='environment-v1',
    )


def write_experiment(parent, name, value):
    """Prepare temporary input files BEFORE invoking the read-only tools."""
    root = parent / name
    root.mkdir()
    (root / 'hparams.yaml').write_text('config:\n  enabled: false\n', encoding='utf-8')
    (root / 'history.json').write_text(json.dumps({
        'validation': {'measurement': [[0, value], [1, value]]},
    }), encoding='utf-8')
    (root / 'metrics.yaml').write_text(yaml.safe_dump({'metrics': [{
        'name': 'score', 'path': ['validation', 'measurement'],
        'direction': 'maximize', 'display_name': 'Score', 'precision': 6,
    }]}), encoding='utf-8')
    (root / 'heldout.csv').write_text(f'measurement\n{value}\n', encoding='utf-8')
    return root


def evidence_options():
    return dict(config_paths=[['enabled']], metrics_config='metrics.yaml',
                test_metrics_file='heldout.csv', test_metric_specs=[metric_spec()])


def evidence_targets():
    def side(label):
        execution = ExecutionIdentity(ExperimentIdentity('demo-lab', label), 'run-1')
        return dict(section='test', metric_name='score', metric_spec=metric_spec(),
                    provenance=provenance(execution), metric_definition='quality-v1',
                    aggregation='macro', evaluation_protocol='evaluation-v1',
                    selection_protocol='selection-v1')
    return [dict(target_id='quality', a=side('A'), b=side('B'))]


def snapshot(value):
    """Host-supplied M16-shaped evidence, with explicitly unverified origin."""
    return {
        'configuration': {},
        'configuration_evidence': {
            'schema_version': 'configuration-evidence-v1', 'source_file': 'hparams.yaml',
            'root_path': ['config'], 'fields': [],
        },
        'validation_evidence': {
            'evidence_kind': 'validation_history', 'source': {'file': 'history.json'},
        },
        'validation_metrics': {},
        'test_results': {
            'schema_version': 'test-results-v1', 'status': 'available',
            'evidence_kind': 'csv_reported',
            'source': {'file': 'heldout.csv', 'data_row': 1,
                       'selection': 'exactly_one_data_row'},
            'metrics': {'score': {'value': value, 'source_column': 'measurement',
                                  'direction': 'maximize', 'display_name': 'Score', 'precision': 6}},
            'issues': [],
            'provenance': {'status': 'unknown', 'checkpoint_identity': None,
                           'checkpoint_binding': 'not_verified', 'dataset_split_identity': None,
                           'recomputed_from_predictions': False},
        },
    }


def group(label, values):
    experiment = ExperimentIdentity('demo-lab', label)
    executions, roster, members = [], [], []
    for index, value in enumerate(values):
        execution = ExecutionIdentity(experiment, f'{label}-{index}')
        seed = 42 + index
        executions.append(dict(
            execution=execution, provenance=provenance(execution, seed),
            training_protocol_ref='training-v1', status='available', evidence=snapshot(value),
            snapshot_metadata=dict(snapshot_id=f'{label}-snapshot-{index}',
                                   acquisition='synthetic host snapshot', binding='host_declared'),
        ))
        roster.append(dict(execution=execution, expected_seed=seed))
        members.append(dict(
            execution=execution, metric_name='score', metric_spec=metric_spec(),
            metric_definition='quality-v1', aggregation='macro',
            evaluation_protocol='evaluation-v1', selection_protocol='selection-v1',
        ))
    return dict(
        executions=executions,
        group_declaration=dict(
            group_id=f'group-{label}', experiment_identity=experiment,
            expected_executions=roster, training_protocol_ref='training-v1',
            membership_selection='host-selected; not preregistered',
            aggregation_policy='replicate-descriptive-v1',
        ),
        targets=[dict(target_id=f'quality-{label}', section='test', members=members)],
    )


def group_targets():
    return [dict(target_id='quality', a_target_id='quality-A', b_target_id='quality-B')]


def plain_group(label, values):
    # Only our own known dataclasses are converted. M22 accepts plain data.
    return json.loads(json.dumps(group(label, values), default=asdict, allow_nan=False))


def crossed_pairing(a, b):
    """The host explicitly chooses crossed pairs; no inferred matching."""
    return {
        'schema_version': 'explicit-pairing-v1', 'pairing_source': 'host_declared',
        'pairs': [
            {'pair_id': 'pair-0', 'a_execution': a['executions'][0]['execution'],
             'b_execution': b['executions'][1]['execution']},
            {'pair_id': 'pair-1', 'a_execution': a['executions'][1]['execution'],
             'b_execution': b['executions'][0]['execution']},
        ],
    }


def artifact_descriptors():
    # Synthetic HOST claims, not a digest computed from any file.
    descriptor = dict(locator='metrics.csv', byte_size=12, sha256='ab' * 32)
    declared = dict(
        schema_version='declared-artifact-set-v1', set_id='bundle-1', declared_by='demo-host-A',
        artifacts=[dict(artifact_id='metrics', logical_role='metrics', **descriptor)],
    )
    reports = [dict(artifact_id='metrics', observation_id='report-1', reported_by='demo-host-B',
                    reported_status='observed', reported_at='2026-01-01T00:00:00Z',
                    **descriptor)]
    return declared, reports

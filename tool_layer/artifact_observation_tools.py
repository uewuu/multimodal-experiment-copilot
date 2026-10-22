"""Explicitly bound live observation of one host-selected artifact."""

import ntpath

import artifact_observation as _core
import artifact_observation_reader as _reader

__all__ = ('ArtifactObservationTools',)


class ArtifactObservationTools:
    """Bind plain host selections without reading or granting Session access.

    Formal Path Policy resolves the model's '.' before dispatch. This boundary
    checks the resulting absolute root against the host selection; the Reader
    owns opened-object authorization and fresh bounded observation.
    """

    def __init__(self, binding):
        self._binding = _core.validate_artifact_binding(binding)

    def list_tools(self) -> list[dict]:
        """Return a fresh schema containing no host file selection."""
        return [{
            'type': 'function',
            'function': {
                'name': 'observe_artifact',
                'description': (
                    'Observe bytes and metadata of one host-selected small artifact '
                    'under the authorized experiment root. Reports do not establish '
                    'producer provenance, checkpoint binding, or currentness.'
                ),
                'parameters': {
                    'type': 'object',
                    'properties': {'experiment_dir': {'type': 'string', 'enum': ['.']}},
                    'required': ['experiment_dir'],
                    'additionalProperties': False,
                },
            },
        }]

    def invoke_tool(self, tool_name: str, arguments: dict) -> dict:
        """Validate secured arguments, then delegate once without caching."""
        if type(tool_name) is not str:
            raise TypeError('tool_name must be an exact string')
        if tool_name != 'observe_artifact':
            raise KeyError('unknown artifact observation tool')
        if type(arguments) is not dict:
            raise TypeError('arguments must be an exact dict')
        if len(arguments) != 1:
            raise ValueError('exactly experiment_dir is required')
        for key in arguments:
            if type(key) is not str:
                raise TypeError('argument names must be exact strings')
            if key != 'experiment_dir':
                raise ValueError('unexpected argument')
        # Reuse pure binding validation for path types/forms. In particular '.'
        # is not a grant and cannot pass as an absolute authorized root here.
        candidate = {**self._binding, 'authorized_root': arguments['experiment_dir']}
        secured = _core.validate_artifact_binding(candidate)
        if ntpath.normcase(secured['authorized_root']) != ntpath.normcase(self._binding['authorized_root']):
            raise ValueError('experiment root does not match the host binding')
        # Give the delegate a detached binding; retain neither results nor state
        # changes made by a delegate. The frozen Reader returns a fresh report.
        return _reader.observe_artifact(_core.validate_artifact_binding(self._binding))

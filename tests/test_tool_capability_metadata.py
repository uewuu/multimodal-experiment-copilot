import inspect
from collections.abc import Mapping
from pathlib import Path

import pytest

import copilot.failure_observability as failure_observability
import llm_adapters.tool_result_governance as result_governance
import tool_layer
import tool_layer.experiment_path_security as path_security
import tool_layer.tool_registry as registry


EXPECTED_TOOL_NAMES = (
    "analyze_experiment",
    "compare_experiments",
)
EXPECTED_WORKSPACE_PATHS = {
    "analyze_experiment": (
        ("experiment_dir", "directory"),
        ("metrics_config", "file"),
    ),
    "compare_experiments": (
        ("experiment_root", "directory"),
        ("metrics_config", "file"),
    ),
}
EXPECTED_PROVIDER_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "analyze_experiment",
            "description": (
                "Analyze one machine-learning experiment directory."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "experiment_dir": {"type": "string"},
                    "metrics_config": {
                        "type": ["string", "null"],
                        "default": None,
                    },
                    "include_diagnostics": {
                        "type": "boolean",
                        "default": False,
                    },
                },
                "required": ["experiment_dir"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "compare_experiments",
            "description": (
                "Compare multiple machine-learning experiment directories."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "experiment_root": {"type": "string"},
                    "sort_by": {
                        "type": ["string", "null"],
                        "default": None,
                    },
                    "descending": {
                        "type": "boolean",
                        "default": True,
                    },
                    "metrics_config": {
                        "type": ["string", "null"],
                        "default": None,
                    },
                    "include_diagnostics": {
                        "type": "boolean",
                        "default": False,
                    },
                },
                "required": ["experiment_root"],
                "additionalProperties": False,
            },
        },
    },
]


def _definition_name(definition: object) -> object:
    name = getattr(definition, "name", None)
    if name is not None:
        return name
    if type(definition) is dict:
        function = definition.get("function")
        if type(function) is dict:
            return function.get("name")
    return None


def _field(value: object, name: str) -> object:
    if isinstance(value, Mapping):
        assert name in value, f"tool definition must contain {name}"
        return value[name]
    assert hasattr(value, name), f"tool definition must own {name}"
    return getattr(value, name)


def _registered_definitions() -> tuple[object, ...]:
    candidates: list[tuple[int, tuple[object, ...]]] = []
    for value in vars(registry).values():
        if isinstance(value, Mapping):
            items = tuple(value.values())
        elif type(value) in (list, tuple):
            items = tuple(value)
        else:
            continue
        if tuple(_definition_name(item) for item in items) != (
            EXPECTED_TOOL_NAMES
        ):
            continue
        score = sum(
            hasattr(item, "capabilities")
            for item in items
        )
        candidates.append((score, items))

    assert candidates, (
        "the Registry must own one ordered collection of complete "
        "tool definitions"
    )
    return max(candidates, key=lambda candidate: candidate[0])[1]


def _definitions_by_name() -> dict[str, object]:
    return {
        _definition_name(definition): definition
        for definition in _registered_definitions()
    }


def _workspace_paths(definition: object) -> tuple[object, ...]:
    capabilities = _field(definition, "capabilities")
    workspace_paths = _field(capabilities, "workspace_paths")
    assert type(workspace_paths) is tuple
    return workspace_paths


def _descriptor_pairs(definition: object) -> tuple[tuple[str, str], ...]:
    pairs: list[tuple[str, str]] = []
    for descriptor in _workspace_paths(definition):
        parameter_name = _field(descriptor, "parameter_name")
        path_kind = _field(descriptor, "path_kind")
        assert isinstance(parameter_name, str)
        assert path_kind in ("directory", "file")
        pairs.append((parameter_name, path_kind))
    return tuple(pairs)


def _assert_attribute_is_immutable(
    value: object,
    attribute: str,
) -> None:
    original = _field(value, attribute)
    if isinstance(value, Mapping):
        with pytest.raises(TypeError):
            value[attribute] = original
    else:
        with pytest.raises((AttributeError, TypeError)):
            setattr(value, attribute, original)


def test_provider_tool_descriptions_remain_exactly_compatible() -> None:
    assert registry.list_tools() == EXPECTED_PROVIDER_TOOLS


def test_private_capabilities_are_not_provider_visible() -> None:
    provider_text = repr(registry.list_tools()).lower()

    assert "capabilities" not in provider_text
    assert "workspace_paths" not in provider_text
    assert "path_kind" not in provider_text


def test_provider_descriptions_remain_deeply_isolated() -> None:
    mutated = registry.list_tools()
    mutated[0]["function"]["parameters"]["properties"].clear()
    mutated[1]["function"]["name"] = "mutated"

    assert registry.list_tools() == EXPECTED_PROVIDER_TOOLS


def test_registry_owns_atomic_complete_tool_definitions() -> None:
    for definition in _registered_definitions():
        assert isinstance(_field(definition, "name"), str)
        assert isinstance(_field(definition, "description"), str)
        assert type(_field(definition, "parameters")) is dict
        assert callable(_field(definition, "function"))
        assert _field(definition, "capabilities") is not None


def test_registered_definition_names_are_unique_and_ordered() -> None:
    names = tuple(
        _definition_name(definition)
        for definition in _registered_definitions()
    )

    assert names == EXPECTED_TOOL_NAMES
    assert len(names) == len(set(names))


def test_definition_names_match_provider_schema_names() -> None:
    definition_names = tuple(
        _definition_name(definition)
        for definition in _registered_definitions()
    )
    provider_names = tuple(
        tool["function"]["name"]
        for tool in registry.list_tools()
    )

    assert definition_names == provider_names


def test_definition_schema_data_matches_provider_output() -> None:
    providers = {
        tool["function"]["name"]: tool["function"]
        for tool in registry.list_tools()
    }
    for name, definition in _definitions_by_name().items():
        assert _field(definition, "description") == (
            providers[name]["description"]
        )
        assert _field(definition, "parameters") == (
            providers[name]["parameters"]
        )


@pytest.mark.parametrize("tool_name", EXPECTED_TOOL_NAMES)
def test_each_definition_owns_a_callable(tool_name: str) -> None:
    definition = _definitions_by_name()[tool_name]

    assert callable(_field(definition, "function"))


@pytest.mark.parametrize("tool_name", EXPECTED_TOOL_NAMES)
def test_workspace_path_capabilities_are_exact(tool_name: str) -> None:
    definition = _definitions_by_name()[tool_name]

    assert _descriptor_pairs(definition) == (
        EXPECTED_WORKSPACE_PATHS[tool_name]
    )


def test_tool_definitions_and_capabilities_are_immutable() -> None:
    for definition in _registered_definitions():
        _assert_attribute_is_immutable(definition, "name")
        _assert_attribute_is_immutable(definition, "capabilities")
        _assert_attribute_is_immutable(
            _field(definition, "capabilities"),
            "workspace_paths",
        )


def test_workspace_path_descriptors_are_immutable() -> None:
    for definition in _registered_definitions():
        for descriptor in _workspace_paths(definition):
            _assert_attribute_is_immutable(
                descriptor,
                "parameter_name",
            )
            _assert_attribute_is_immutable(descriptor, "path_kind")


def test_model_arguments_cannot_mutate_capability_metadata() -> None:
    before = {
        name: _descriptor_pairs(definition)
        for name, definition in _definitions_by_name().items()
    }
    provider_tools = registry.list_tools()
    provider_tools[0]["capabilities"] = {
        "workspace_paths": [
            {"parameter_name": "outside", "path_kind": "directory"}
        ]
    }
    arguments = {
        "experiment_dir": "experiment",
        "capabilities": provider_tools[0]["capabilities"],
    }

    assert arguments["capabilities"]["workspace_paths"][0][
        "parameter_name"
    ] == "outside"
    assert {
        name: _descriptor_pairs(definition)
        for name, definition in _definitions_by_name().items()
    } == before


def test_invoke_tool_dispatch_remains_monkeypatch_friendly(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sentinel = {"identity": object()}
    calls: list[dict] = []

    def fake_analyze(**arguments: object) -> dict:
        calls.append(arguments)
        return sentinel

    monkeypatch.setattr(registry, "analyze_experiment", fake_analyze)
    result = registry.invoke_tool(
        "analyze_experiment",
        {"experiment_dir": "experiment"},
    )

    assert result is sentinel
    assert calls == [{"experiment_dir": "experiment"}]


def test_invoke_tool_preserves_callable_exception_identity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    expected = RuntimeError("tool failed")

    def fail(**arguments: object) -> dict:
        raise expected

    monkeypatch.setattr(registry, "compare_experiments", fail)
    with pytest.raises(RuntimeError) as caught:
        registry.invoke_tool(
            "compare_experiments",
            {"experiment_root": "experiments"},
        )

    assert caught.value is expected


def test_invoke_tool_dispatch_comes_from_atomic_definitions() -> None:
    source = inspect.getsource(registry.invoke_tool)

    assert 'tool_name == "analyze_experiment"' not in source
    assert 'tool_name == "compare_experiments"' not in source


def test_path_security_has_no_duplicated_tool_path_field_map() -> None:
    source = Path(path_security.__file__).read_text(encoding="utf-8")

    assert "_TOOL_PATH_FIELDS" not in source
    assert '"analyze_experiment"' not in source
    assert '"compare_experiments"' not in source


def test_path_security_consumes_registry_owned_metadata() -> None:
    source = Path(path_security.__file__).read_text(encoding="utf-8")

    assert "tool_registry" in source
    assert "workspace_paths" in source


def test_unknown_tool_is_not_granted_filesystem_capability(
    tmp_path: Path,
) -> None:
    root = tmp_path / "workspace"
    outside = tmp_path / "outside"
    root.mkdir()
    outside.mkdir()
    arguments = {"experiment_dir": str(outside)}
    policy = path_security._build_experiment_path_policy(
        {"experiment_root": str(root)}
    )

    assert policy("unknown_tool", arguments) is arguments


def test_request_capability_still_allows_only_resolved_inside_path(
    tmp_path: Path,
) -> None:
    root = tmp_path / "workspace"
    inside = root / "experiment"
    outside = tmp_path / "outside"
    inside.mkdir(parents=True)
    outside.mkdir()
    policy = path_security._build_experiment_path_policy(
        {"experiment_root": str(root)}
    )

    secured = policy(
        "analyze_experiment",
        {"experiment_dir": "experiment"},
    )
    assert secured == {"experiment_dir": str(inside.resolve())}
    with pytest.raises(ValueError):
        policy(
            "analyze_experiment",
            {"experiment_dir": str(outside)},
        )


def test_public_tool_api_and_exports_remain_unchanged() -> None:
    assert tool_layer.__all__ == [
        "analyze_experiment",
        "compare_experiments",
        "list_tools",
        "invoke_tool",
    ]
    assert str(inspect.signature(registry.list_tools)) == (
        "() -> list[dict]"
    )
    assert str(inspect.signature(registry.invoke_tool)) == (
        "(tool_name: str, arguments: dict) -> dict"
    )


def test_failure_observability_keeps_exact_eight_stages() -> None:
    progress = failure_observability._ProgressState()
    stages = {progress.stage}
    for event in (
        "provider_request_started",
        "provider_response_received",
        "tool_call_validation",
        "tool_execution",
        "tool_result_serialization",
        "provider_request_started",
        "provider_response_received",
    ):
        progress.update(event)
        stages.add(progress.stage)

    assert stages == {
        "input_validation",
        "first_provider_request",
        "first_provider_response_validation",
        "tool_call_validation",
        "tool_execution",
        "tool_result_serialization",
        "second_provider_request",
        "final_response_validation",
    }


def test_m3_result_governance_remains_metadata_independent() -> None:
    source = Path(result_governance.__file__).read_text(encoding="utf-8")

    assert result_governance.MAX_TOOL_RESULT_BYTES == 256 * 1024
    assert (
        result_governance.MAX_TOOL_RESULTS_PER_CYCLE_BYTES
        == 512 * 1024
    )
    assert "capabil" not in source.lower()

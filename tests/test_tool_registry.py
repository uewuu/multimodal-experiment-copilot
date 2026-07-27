import importlib
import inspect
import json
import math
import socket
from collections import UserDict
from pathlib import Path
from types import MappingProxyType

import pytest

import tool_layer


EXPECTED_TOOL_NAMES = [
    "analyze_experiment",
    "compare_experiments",
]


def _registry_module():
    try:
        return importlib.import_module("tool_layer.tool_registry")
    except ModuleNotFoundError as error:
        if error.name == "tool_layer.tool_registry":
            pytest.fail(
                "tool_layer.tool_registry must implement Issue #14",
                pytrace=False,
            )
        raise


def _registry_api(name: str):
    module = _registry_module()
    assert hasattr(module, name), (
        f"tool_layer.tool_registry must define {name}"
    )
    value = getattr(module, name)
    assert callable(value), f"{name} must be callable"
    return value


def _list_tools() -> list[dict]:
    result = _registry_api("list_tools")()
    assert type(result) is list, "list_tools must return an actual list"
    return result


def _tools_by_name() -> dict[str, dict]:
    tools = _list_tools()
    return {
        tool["function"]["name"]: tool
        for tool in tools
    }


def _parameters_for(tool_name: str) -> dict:
    tools = _tools_by_name()
    assert tool_name in tools, f"{tool_name} must be registered"
    return tools[tool_name]["function"]["parameters"]


def _assert_nullable_string(schema: dict) -> None:
    if isinstance(schema.get("type"), list):
        assert set(schema["type"]) == {"string", "null"}
        return

    alternatives = schema.get("anyOf", schema.get("oneOf"))
    assert isinstance(alternatives, list), (
        "nullable string must use a standard JSON Schema representation"
    )
    assert {
        alternative.get("type")
        for alternative in alternatives
    } == {"string", "null"}


def _assert_json_native(value: object, path: str = "$") -> None:
    assert not callable(value), f"{path} must not expose a callable"
    assert not isinstance(value, Path), f"{path} must not expose Path"
    assert not isinstance(value, set), f"{path} must not expose set"
    assert not isinstance(value, bytes), f"{path} must not expose bytes"

    if isinstance(value, float):
        assert math.isfinite(value), f"{path} must contain finite numbers"
    elif isinstance(value, dict):
        for key, nested_value in value.items():
            assert isinstance(key, str), f"{path} keys must be strings"
            _assert_json_native(nested_value, f"{path}.{key}")
    elif isinstance(value, list):
        for index, nested_value in enumerate(value):
            _assert_json_native(nested_value, f"{path}[{index}]")
    else:
        assert value is None or isinstance(
            value,
            (str, int, bool),
        ), f"{path} contains non-JSON-native {type(value).__name__}"


def _patch_registry_tools(
    monkeypatch: pytest.MonkeyPatch,
    *,
    analyze,
    compare,
):
    module = _registry_module()
    monkeypatch.setattr(
        module,
        "analyze_experiment",
        analyze,
        raising=False,
    )
    monkeypatch.setattr(
        module,
        "compare_experiments",
        compare,
        raising=False,
    )
    return module


def test_tool_registry_module_is_importable() -> None:
    module = _registry_module()

    assert module.__name__ == "tool_layer.tool_registry"


def test_tool_layer_publicly_exports_registry_functions() -> None:
    for name in ("list_tools", "invoke_tool"):
        assert hasattr(tool_layer, name), (
            f"tool_layer must publicly export {name}"
        )
        assert callable(getattr(tool_layer, name))


def test_tool_layer_all_includes_registry_functions() -> None:
    assert "list_tools" in tool_layer.__all__
    assert "invoke_tool" in tool_layer.__all__


def test_public_exports_reference_registry_functions() -> None:
    registry = _registry_module()

    assert tool_layer.list_tools is registry.list_tools
    assert tool_layer.invoke_tool is registry.invoke_tool


def test_list_tools_has_exact_signature() -> None:
    signature = inspect.signature(_registry_api("list_tools"))

    assert list(signature.parameters.values()) == []
    assert signature.return_annotation == list[dict]


def test_invoke_tool_has_exact_signature() -> None:
    signature = inspect.signature(_registry_api("invoke_tool"))
    parameters = list(signature.parameters.values())

    assert [parameter.name for parameter in parameters] == [
        "tool_name",
        "arguments",
    ]
    assert all(
        parameter.kind is inspect.Parameter.POSITIONAL_OR_KEYWORD
        for parameter in parameters
    )
    assert all(
        parameter.default is inspect.Parameter.empty
        for parameter in parameters
    )
    assert parameters[0].annotation is str
    assert parameters[1].annotation is dict
    assert signature.return_annotation is dict


def test_list_tools_returns_actual_list() -> None:
    assert type(_list_tools()) is list


def test_list_tools_registers_exactly_two_tools() -> None:
    assert len(_list_tools()) == 2


def test_list_tools_preserves_registration_order() -> None:
    names = [
        tool["function"]["name"]
        for tool in _list_tools()
    ]

    assert names == EXPECTED_TOOL_NAMES


def test_tool_descriptions_have_exact_top_level_keys() -> None:
    for tool in _list_tools():
        assert list(tool) == ["type", "function"]
        assert tool["type"] == "function"


def test_function_descriptions_have_exact_keys() -> None:
    for tool in _list_tools():
        assert list(tool["function"]) == [
            "name",
            "description",
            "parameters",
        ]


def test_tool_descriptions_are_nonempty_and_distinct() -> None:
    tools = _tools_by_name()
    analyze_description = tools["analyze_experiment"]["function"][
        "description"
    ]
    compare_description = tools["compare_experiments"]["function"][
        "description"
    ]

    assert isinstance(analyze_description, str)
    assert analyze_description.strip()
    assert isinstance(compare_description, str)
    assert compare_description.strip()
    assert analyze_description != compare_description
    assert any(
        word in analyze_description.lower()
        for word in ("single", "one")
    )
    assert any(
        word in compare_description.lower()
        for word in ("multiple", "compare")
    )


def test_each_parameters_schema_has_required_object_shape() -> None:
    for tool_name in EXPECTED_TOOL_NAMES:
        parameters = _parameters_for(tool_name)

        assert parameters["type"] == "object"
        assert type(parameters["properties"]) is dict
        assert type(parameters["required"]) is list
        assert parameters["additionalProperties"] is False


def test_analyze_schema_property_order_matches_signature() -> None:
    properties = _parameters_for("analyze_experiment")["properties"]

    assert list(properties) == [
        "experiment_dir",
        "metrics_config",
        "include_diagnostics",
    ]


def test_analyze_schema_has_exact_required_parameters() -> None:
    parameters = _parameters_for("analyze_experiment")

    assert parameters["required"] == ["experiment_dir"]


def test_analyze_experiment_dir_schema_is_required_string() -> None:
    parameters = _parameters_for("analyze_experiment")
    schema = parameters["properties"]["experiment_dir"]

    assert schema["type"] == "string"
    assert "experiment_dir" in parameters["required"]
    assert "default" not in schema


def test_analyze_metrics_config_schema_is_optional_nullable_string() -> None:
    parameters = _parameters_for("analyze_experiment")
    schema = parameters["properties"]["metrics_config"]

    _assert_nullable_string(schema)
    assert "metrics_config" not in parameters["required"]
    assert "default" not in schema or schema["default"] is None


def test_analyze_diagnostics_schema_has_false_default() -> None:
    parameters = _parameters_for("analyze_experiment")
    schema = parameters["properties"]["include_diagnostics"]

    assert schema["type"] == "boolean"
    assert schema["default"] is False
    assert "include_diagnostics" not in parameters["required"]


def test_compare_schema_property_order_matches_signature() -> None:
    properties = _parameters_for("compare_experiments")["properties"]

    assert list(properties) == [
        "experiment_root",
        "sort_by",
        "descending",
        "metrics_config",
        "include_diagnostics",
    ]


def test_compare_schema_has_exact_required_parameters() -> None:
    parameters = _parameters_for("compare_experiments")

    assert parameters["required"] == ["experiment_root"]


def test_compare_experiment_root_schema_is_required_string() -> None:
    parameters = _parameters_for("compare_experiments")
    schema = parameters["properties"]["experiment_root"]

    assert schema["type"] == "string"
    assert "experiment_root" in parameters["required"]
    assert "default" not in schema


def test_compare_sort_by_schema_is_optional_nullable_string() -> None:
    parameters = _parameters_for("compare_experiments")
    schema = parameters["properties"]["sort_by"]

    _assert_nullable_string(schema)
    assert "sort_by" not in parameters["required"]
    assert "default" not in schema or schema["default"] is None


def test_compare_descending_schema_has_true_default() -> None:
    parameters = _parameters_for("compare_experiments")
    schema = parameters["properties"]["descending"]

    assert schema["type"] == "boolean"
    assert schema["default"] is True
    assert "descending" not in parameters["required"]


def test_compare_metrics_config_schema_is_optional_nullable_string() -> None:
    parameters = _parameters_for("compare_experiments")
    schema = parameters["properties"]["metrics_config"]

    _assert_nullable_string(schema)
    assert "metrics_config" not in parameters["required"]
    assert "default" not in schema or schema["default"] is None


def test_compare_diagnostics_schema_has_false_default() -> None:
    parameters = _parameters_for("compare_experiments")
    schema = parameters["properties"]["include_diagnostics"]

    assert schema["type"] == "boolean"
    assert schema["default"] is False
    assert "include_diagnostics" not in parameters["required"]


def test_list_tools_is_strictly_json_serializable() -> None:
    tools = _list_tools()

    json.dumps(
        tools,
        ensure_ascii=False,
        allow_nan=False,
    )


def test_list_tools_does_not_expose_non_json_native_objects() -> None:
    _assert_json_native(_list_tools())


def test_list_tools_returns_distinct_top_level_lists() -> None:
    first = _list_tools()
    second = _list_tools()

    assert first is not second


def test_list_tools_returns_distinct_nested_objects() -> None:
    first = _list_tools()
    second = _list_tools()

    assert first[0] is not second[0]
    assert first[0]["function"] is not second[0]["function"]
    assert (
        first[0]["function"]["parameters"]
        is not second[0]["function"]["parameters"]
    )
    assert (
        first[0]["function"]["parameters"]["properties"]
        is not second[0]["function"]["parameters"]["properties"]
    )


def test_list_tools_mutation_does_not_affect_another_result() -> None:
    first = _list_tools()
    second = _list_tools()
    expected_second = json.loads(json.dumps(second))

    first.append({"type": "mutated"})
    first[0]["type"] = "mutated"
    first[0]["function"]["description"] = "mutated"
    first[0]["function"]["parameters"]["type"] = "mutated"
    first[0]["function"]["parameters"]["properties"].clear()

    assert second == expected_second


def test_list_tools_mutation_does_not_affect_future_calls() -> None:
    first = _list_tools()
    canonical = _list_tools()

    first[0]["function"]["name"] = "mutated"
    first[0]["function"]["parameters"]["required"].append("mutated")

    assert _list_tools() == canonical


@pytest.mark.parametrize(
    "invalid_name",
    [None, 1, True, [], {}],
)
def test_invoke_tool_rejects_non_string_tool_name(
    invalid_name: object,
) -> None:
    invoke_tool = _registry_api("invoke_tool")

    with pytest.raises(TypeError) as error_info:
        invoke_tool(invalid_name, {})  # type: ignore[arg-type]

    assert "tool" in str(error_info.value).lower()


@pytest.mark.parametrize(
    "invalid_name",
    ["", " ", "\t", "\n"],
)
def test_invoke_tool_rejects_blank_tool_name(
    invalid_name: str,
) -> None:
    invoke_tool = _registry_api("invoke_tool")

    with pytest.raises(ValueError) as error_info:
        invoke_tool(invalid_name, {})

    assert "tool" in str(error_info.value).lower()


@pytest.mark.parametrize(
    "invalid_arguments",
    [
        None,
        [],
        (),
        "value",
        1,
        True,
        UserDict(),
        MappingProxyType({}),
    ],
)
def test_invoke_tool_rejects_non_dict_arguments(
    invalid_arguments: object,
) -> None:
    invoke_tool = _registry_api("invoke_tool")

    with pytest.raises(TypeError) as error_info:
        invoke_tool(
            "analyze_experiment",
            invalid_arguments,  # type: ignore[arg-type]
        )

    assert "arguments" in str(error_info.value).lower()


def test_invoke_tool_rejects_dict_subclasses() -> None:
    class CustomDict(dict):
        pass

    with pytest.raises(TypeError):
        _registry_api("invoke_tool")(
            "analyze_experiment",
            CustomDict(experiment_dir="experiment"),
        )


def test_invoke_tool_rejects_unknown_tool_name() -> None:
    unknown_name = "unknown_tool"

    with pytest.raises(KeyError) as error_info:
        _registry_api("invoke_tool")(unknown_name, {})

    assert unknown_name in str(error_info.value)


@pytest.mark.parametrize(
    "invalid_name",
    [
        " analyze_experiment",
        "analyze_experiment ",
        "Analyze_Experiment",
    ],
)
def test_invoke_tool_does_not_normalize_nonexact_names(
    invalid_name: str,
) -> None:
    with pytest.raises(KeyError) as error_info:
        _registry_api("invoke_tool")(invalid_name, {})

    assert invalid_name in str(error_info.value)


def test_invoke_tool_dispatches_analyze_experiment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[dict] = []
    sentinel = {"result": "analyze"}

    def fake_analyze(**kwargs: object) -> dict:
        calls.append(kwargs)
        return sentinel

    def fail_compare(**kwargs: object) -> dict:
        pytest.fail("compare_experiments must not be called")

    registry = _patch_registry_tools(
        monkeypatch,
        analyze=fake_analyze,
        compare=fail_compare,
    )

    result = registry.invoke_tool(
        "analyze_experiment",
        {"experiment_dir": "experiment"},
    )

    assert result is sentinel
    assert calls == [{"experiment_dir": "experiment"}]


def test_invoke_tool_dispatches_compare_experiments(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[dict] = []
    sentinel = {"result": "compare"}

    def fail_analyze(**kwargs: object) -> dict:
        pytest.fail("analyze_experiment must not be called")

    def fake_compare(**kwargs: object) -> dict:
        calls.append(kwargs)
        return sentinel

    registry = _patch_registry_tools(
        monkeypatch,
        analyze=fail_analyze,
        compare=fake_compare,
    )

    result = registry.invoke_tool(
        "compare_experiments",
        {"experiment_root": "experiments"},
    )

    assert result is sentinel
    assert calls == [{"experiment_root": "experiments"}]


def test_invoke_tool_forwards_exact_keyword_arguments(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    received: list[dict] = []
    arguments = {
        "experiment_root": "experiments",
        "sort_by": "best_racc",
        "descending": False,
        "metrics_config": "metrics.yaml",
        "include_diagnostics": True,
    }

    def fake_compare(**kwargs: object) -> dict:
        received.append(kwargs)
        return {}

    registry = _patch_registry_tools(
        monkeypatch,
        analyze=lambda **kwargs: {},
        compare=fake_compare,
    )

    registry.invoke_tool("compare_experiments", arguments)

    assert received == [arguments]
    assert list(received[0]) == list(arguments)


def test_invoke_tool_does_not_modify_arguments(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    arguments = {
        "experiment_dir": "experiment",
        "include_diagnostics": True,
    }
    original = arguments.copy()
    registry = _patch_registry_tools(
        monkeypatch,
        analyze=lambda **kwargs: {},
        compare=lambda **kwargs: {},
    )

    registry.invoke_tool("analyze_experiment", arguments)

    assert arguments == original


def test_invoke_tool_does_not_inject_omitted_defaults(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    received: list[dict] = []

    def fake_analyze(**kwargs: object) -> dict:
        received.append(kwargs)
        return {}

    registry = _patch_registry_tools(
        monkeypatch,
        analyze=fake_analyze,
        compare=lambda **kwargs: {},
    )

    registry.invoke_tool(
        "analyze_experiment",
        {"experiment_dir": "experiment"},
    )

    assert received == [{"experiment_dir": "experiment"}]


def test_invoke_tool_returns_original_result_identity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sentinel = {"nested": {"value": []}}
    registry = _patch_registry_tools(
        monkeypatch,
        analyze=lambda **kwargs: sentinel,
        compare=lambda **kwargs: {},
    )

    result = registry.invoke_tool(
        "analyze_experiment",
        {"experiment_dir": "experiment"},
    )

    assert result is sentinel


def test_invoke_tool_does_not_add_result_envelope(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sentinel = {"summary": "original"}
    registry = _patch_registry_tools(
        monkeypatch,
        analyze=lambda **kwargs: sentinel,
        compare=lambda **kwargs: {},
    )

    result = registry.invoke_tool(
        "analyze_experiment",
        {"experiment_dir": "experiment"},
    )

    assert result == {"summary": "original"}
    assert "status" not in result
    assert "message" not in result
    assert "data" not in result
    assert "tool_name" not in result


def test_invoke_tool_propagates_underlying_exception_identity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    expected_error = ValueError("analysis failed")

    def fail(**kwargs: object) -> dict:
        raise expected_error

    registry = _patch_registry_tools(
        monkeypatch,
        analyze=fail,
        compare=lambda **kwargs: {},
    )

    with pytest.raises(ValueError) as error_info:
        registry.invoke_tool(
            "analyze_experiment",
            {"experiment_dir": "experiment"},
        )

    assert error_info.value is expected_error


def test_invoke_tool_propagates_underlying_type_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    expected_error = TypeError("invalid argument")

    def fail(**kwargs: object) -> dict:
        raise expected_error

    registry = _patch_registry_tools(
        monkeypatch,
        analyze=lambda **kwargs: {},
        compare=fail,
    )

    with pytest.raises(TypeError) as error_info:
        registry.invoke_tool(
            "compare_experiments",
            {"experiment_root": "experiments"},
        )

    assert error_info.value is expected_error


def test_invoke_tool_calls_registered_tool_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    call_count = 0

    def fake_analyze(**kwargs: object) -> dict:
        nonlocal call_count
        call_count += 1
        return {}

    registry = _patch_registry_tools(
        monkeypatch,
        analyze=fake_analyze,
        compare=lambda **kwargs: {},
    )

    registry.invoke_tool(
        "analyze_experiment",
        {"experiment_dir": "experiment"},
    )

    assert call_count == 1


def test_registry_references_real_public_tools() -> None:
    registry = _registry_module()

    assert registry.analyze_experiment is tool_layer.analyze_experiment
    assert registry.compare_experiments is tool_layer.compare_experiments


def test_list_tools_does_not_print(
    capsys: pytest.CaptureFixture[str],
) -> None:
    _list_tools()

    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err == ""


def test_invoke_tool_does_not_add_output(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    registry = _patch_registry_tools(
        monkeypatch,
        analyze=lambda **kwargs: {},
        compare=lambda **kwargs: {},
    )

    registry.invoke_tool(
        "analyze_experiment",
        {"experiment_dir": "experiment"},
    )

    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err == ""


def test_list_tools_does_not_create_files_or_directories(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.chdir(tmp_path)
    paths_before = sorted(tmp_path.rglob("*"))

    _list_tools()

    assert sorted(tmp_path.rglob("*")) == paths_before


def test_registry_import_does_not_create_files_or_directories(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.chdir(tmp_path)
    paths_before = sorted(tmp_path.rglob("*"))

    _registry_module()

    assert sorted(tmp_path.rglob("*")) == paths_before


def test_registry_does_not_use_forbidden_framework_imports() -> None:
    registry = _registry_module()
    source = Path(registry.__file__).read_text(encoding="utf-8")
    forbidden_imports = (
        "import openai",
        "from openai",
        "import pydantic",
        "from pydantic",
        "import langchain",
        "from langchain",
        "import langgraph",
        "from langgraph",
        "import fastapi",
        "from fastapi",
    )

    for forbidden_import in forbidden_imports:
        assert forbidden_import not in source


def test_list_tools_does_not_access_network(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail_network(*args: object, **kwargs: object) -> object:
        pytest.fail("list_tools must not access the network")

    monkeypatch.setattr(socket, "create_connection", fail_network)

    _list_tools()

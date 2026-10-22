"""M16 RED: read_config.read_config_evidence(Path, field_paths) -> dict.

Paths are ordered, unique sequences of nonblank string segments; no wildcards.
Selections are relative to the normalized config root. Only JSON scalars are
projected. Missing values omit `value`; explicit null retains it. Non-scalar,
non-finite and oversized values are unavailable, never truncated. Invalid host
selections and a projection exceeding 16 KiB raise ValueError/TypeError.
Byte limits use compact ensure_ascii=False, allow_nan=False JSON in UTF-8.
This records declarations, not proof that training executed a component.
"""

import importlib
import json
from copy import deepcopy

import pytest
import yaml


def api():
    fn = getattr(importlib.import_module("read_config"), "read_config_evidence", None)
    assert callable(fn), "missing M16 capability: read_config.read_config_evidence"
    return fn


def encoded(value):
    return json.dumps(value, ensure_ascii=False, allow_nan=False,
                      separators=(",", ":")).encode("utf-8")


def config_file(tmp_path, data):
    path = tmp_path / "hparams.yaml"
    path.write_text(yaml.safe_dump(data, allow_unicode=True), encoding="utf-8")
    return path


@pytest.mark.parametrize("wrapped", [False, True], ids=["root", "config-root"])
def test_normalization_source_paths_and_non_fi_nested_selection(tmp_path, wrapped):
    read = api()
    data = {"optimizer": {"base_lr": 0.003}, "early_stopping": {"monitor": "loss"},
            "optimizer.base_lr": 0.9, "unselected": "DO_NOT_EXPOSE"}
    path = config_file(tmp_path, {"config": data} if wrapped else data)
    paths = [("early_stopping", "monitor"), ("optimizer", "base_lr"),
             ("optimizer.base_lr",)]
    original = deepcopy(paths)
    result = read(path, paths)
    assert paths == original
    assert result["schema_version"] == "configuration-evidence-v1"
    assert result["source_file"] == "hparams.yaml"
    assert result["root_path"] == (["config"] if wrapped else [])
    assert [f["path"] for f in result["fields"]] == [list(p) for p in paths]
    assert [f["value"] for f in result["fields"]] == ["loss", 0.003, 0.9]
    assert all(f["status"] == "present" and f["evidence_kind"] ==
               "configuration_declared" for f in result["fields"])
    assert "DO_NOT_EXPOSE" not in encoded(result).decode()
    result["fields"][0]["value"] = "changed"
    assert read(path, paths)["fields"][0]["value"] == "loss"


def assert_distinct_fields(fields):
    assert len(fields) == 5
    assert [field["path"] for field in fields] == [
        ["absent"], ["off"], ["zero"], ["nil"], ["text"]]
    assert fields[0]["status"] == "missing" and "value" not in fields[0]
    for index, value in enumerate((False, 0, None, ""), start=1):
        field = fields[index]
        assert field["status"] == "present"
        assert "value" in field
        assert type(field["value"]) is type(value) and field["value"] == value


def test_distinct_fields_assertion_probe():
    missing = {"path": ["absent"], "status": "missing"}
    with pytest.raises(AssertionError):
        assert_distinct_fields([missing])
    complete = [missing] + [
        {"path": [name], "status": "present", "value": value}
        for name, value in (("off", False), ("zero", 0), ("nil", None), ("text", ""))]
    assert_distinct_fields(complete)
    wrong_type = deepcopy(complete)
    wrong_type[1]["value"] = 0
    with pytest.raises(AssertionError):
        assert_distinct_fields(wrong_type)


def test_missing_null_false_zero_empty_string_are_distinct(tmp_path):
    read = api()
    path = config_file(tmp_path, {"off": False, "zero": 0, "nil": None, "text": ""})
    fields = read(path, [(k,) for k in ("absent", "off", "zero", "nil", "text")])["fields"]
    assert_distinct_fields(fields)


def test_non_mapping_parent_is_not_missing_or_recursively_expanded(tmp_path):
    read = api()
    fields = read(config_file(tmp_path, {"x": 3}), [("x", "y")])["fields"]
    assert fields[0]["status"] == "unavailable"
    assert fields[0]["reason"] == "non_mapping_parent"
    assert "value" not in fields[0]


@pytest.mark.parametrize("value", [{"secret": "UNSELECTED"}, [1, 2],
                                      float("nan"), float("inf"), float("-inf")],
                         ids=["mapping", "list", "nan", "inf", "negative-inf"])
def test_unsafe_values_are_unavailable_without_payload(tmp_path, value):
    read = api()
    field = read(config_file(tmp_path, {"x": value}), [("x",)])["fields"][0]
    assert field["status"] == "unavailable" and "value" not in field
    assert field["reason"] in ("unsupported_value_type", "non_finite_value")
    assert "UNSELECTED" not in encoded(field).decode()


@pytest.mark.parametrize("paths", ["x", [("",)], [(" x",)], [("*",)],
    [("**",)], [()], [(1,)], [("x",), ("x",)], [("x",) * 9],
    [("x" * 65,)], [("界" * 22,)], [(str(i),) for i in range(33)]],
    ids=["not-sequence", "empty-key", "whitespace", "wildcard", "recursive",
         "empty-path", "non-string-key", "duplicate", "depth-9", "bytes-65",
         "unicode-key-over-limit", "fields-33"])
def test_invalid_host_selections_are_rejected(tmp_path, paths):
    read = api()
    with pytest.raises((TypeError, ValueError)):
        read(config_file(tmp_path, {"x": 1}), paths)


def test_inclusive_field_depth_key_and_string_limits(tmp_path):
    read = api()
    key = "界" * 21 + "x"  # exactly 64 bytes
    deep = (key,) + tuple(f"d{i}" for i in range(7))
    value = "界" * 170 + "ab"  # exactly 512 bytes
    data = {f"f{i}": i for i in range(31)}
    node = data
    for part in deep[:-1]:
        node = node.setdefault(part, {})
    node[deep[-1]] = value
    paths = [(f"f{i}",) for i in range(31)] + [deep]
    result = read(config_file(tmp_path, data), paths)
    assert len(result["fields"]) == 32
    assert result["fields"][-1]["value"] == value
    assert len(encoded(result)) <= 16384


def test_513_utf8_bytes_is_unavailable_not_truncated(tmp_path):
    read = api()
    field = read(config_file(tmp_path, {"x": "界" * 171}), [("x",)])["fields"][0]
    assert field["status"] == "unavailable"
    assert field["reason"] == "value_too_large" and "value" not in field


def test_total_projection_limit_rejects_without_partial_result(tmp_path):
    read = api()
    data = {f"field{i}": "x" * 512 for i in range(32)}
    with pytest.raises(ValueError):
        read(config_file(tmp_path, data), [(k,) for k in data])


def test_empty_selection_does_not_auto_select_config(tmp_path):
    read = api()
    result = read(config_file(tmp_path, {"secret": "NOT_SELECTED"}), ())
    assert result["fields"] == []
    assert "NOT_SELECTED" not in encoded(result).decode()

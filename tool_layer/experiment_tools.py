"""单实验分析 Tool Layer。"""

from collections.abc import Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass
import json
from pathlib import Path, PureWindowsPath

from compare_experiments import (
    analyze_experiment_dirs,
    build_comparison_payload,
    find_experiment_dirs,
)
from generate_report import resolve_experiment_paths
import experiment_comparison
from metrics import MetricSpec
from read_config import read_config_evidence
from read_metrics_config import read_metric_specs_config
from read_test_metrics import read_test_metrics
from summarize_experiment import build_experiment_summary


def analyze_experiment(
    experiment_dir: str,
    *,
    metrics_config: str | None = None,
    include_diagnostics: bool = False,
) -> dict:
    """分析单个实验目录并返回严格 JSON 友好的结构化摘要。

    Args:
        experiment_dir: 包含 hparams.yaml 和 history.json 的目录路径。
        metrics_config: 可选的独立指标 YAML 配置文件路径。
        include_diagnostics: 是否在摘要中包含规则诊断和建议。

    Returns:
        ``build_experiment_summary`` 返回的原始摘要字典。

    Raises:
        TypeError: 参数类型、摘要类型或 JSON 值类型无效。
        ValueError: 字符串参数为空，或摘要包含非有限数值。
        OSError: 实验文件或指标配置文件无法访问。
        Exception: 其他底层解析、指标和诊断异常原样传播。

    此函数不打印业务结果，不创建输出目录，也不写入报告文件。
    """
    if not isinstance(experiment_dir, str):
        raise TypeError("experiment_dir must be a string")
    if not experiment_dir.strip():
        raise ValueError(
            "experiment_dir must not be empty or whitespace"
        )

    if metrics_config is not None:
        if not isinstance(metrics_config, str):
            raise TypeError(
                "metrics_config must be a string or None"
            )
        if not metrics_config.strip():
            raise ValueError(
                "metrics_config must not be empty or whitespace"
            )

    if not isinstance(include_diagnostics, bool):
        raise TypeError("include_diagnostics must be a boolean")

    config_path, history_path = resolve_experiment_paths(
        Path(experiment_dir)
    )

    if metrics_config is None:
        result = build_experiment_summary(
            config_path=config_path,
            history_path=history_path,
            include_diagnostics=include_diagnostics,
        )
    else:
        metric_specs = read_metric_specs_config(
            Path(metrics_config)
        )
        result = build_experiment_summary(
            config_path=config_path,
            history_path=history_path,
            metric_specs=metric_specs,
            include_diagnostics=include_diagnostics,
        )

    if not isinstance(result, dict):
        raise TypeError(
            "build_experiment_summary must return a dict"
        )

    json.dumps(
        result,
        ensure_ascii=False,
        allow_nan=False,
    )
    return result


def _evidence_file(root: Path, value: str) -> tuple[Path, str]:
    """Authorize an experiment-relative file before any reader opens it.

    Missing inside files remain valid candidates for the reader's status
    handling. Windows path forms are checked even on non-Windows hosts.
    """
    if not isinstance(value, str):
        raise TypeError("evidence file path must be a string")
    if not value.strip() or "\x00" in value:
        raise ValueError("evidence file path must be nonblank and valid")
    relative = Path(value)
    windows = PureWindowsPath(value)
    if (relative.is_absolute() or windows.drive or windows.root
            or ".." in relative.parts or ".." in windows.parts
            or not relative.parts or windows.is_reserved()
            or ":" in value):
        raise ValueError("evidence file must be an experiment-relative path")
    candidate = (root / relative).resolve(strict=False)
    if not candidate.is_relative_to(root) or candidate == root:
        raise ValueError("evidence file is outside the experiment directory")
    return candidate, relative.as_posix()


def analyze_experiment_evidence(
    experiment_dir: str,
    *,
    config_paths: Sequence[Sequence[str]] = (),
    test_metrics_file: str | None = None,
    test_metric_specs: Sequence[MetricSpec] = (),
    metrics_config: str | None = None,
    include_diagnostics: bool = False,
) -> dict:
    """Compose legacy validation with explicitly selected file evidence.

    File options are relative to this host-selected experiment. All child
    paths are authorized before reading; this does not replace Agent-level
    authorization of the experiment itself. No Test file is discovered by
    default. CSV data failures retain the Reader's status; argument, path and
    unexpected exceptions propagate. No checkpoint provenance is inferred.
    """
    if not isinstance(experiment_dir, str):
        raise TypeError("experiment_dir must be a string")
    if not experiment_dir.strip():
        raise ValueError("experiment_dir must not be empty or whitespace")
    if not isinstance(include_diagnostics, bool):
        raise TypeError("include_diagnostics must be a boolean")
    if (isinstance(test_metric_specs, (str, bytes, bytearray))
            or not isinstance(test_metric_specs, Sequence)):
        raise TypeError("test_metric_specs must be a sequence")
    if (test_metrics_file is not None) != bool(test_metric_specs):
        raise ValueError("test_metrics_file and test_metric_specs must be supplied together")

    root = Path(experiment_dir).resolve(strict=True)
    if not root.is_dir():
        raise NotADirectoryError("experiment_dir must be a directory")
    _evidence_file(root, "hparams.yaml")
    _evidence_file(root, "history.json")
    metric_path = (
        None if metrics_config is None else _evidence_file(root, metrics_config)[0]
    )
    test_path = None
    if test_metrics_file is not None:
        test_path, test_source = _evidence_file(root, test_metrics_file)

    # Keep the authorized input name, as the legacy analysis does, so the
    # Reader's source metadata and bounded projection remain unchanged.
    configuration_evidence = read_config_evidence(root / "hparams.yaml", config_paths)
    legacy = analyze_experiment(
        str(root),
        metrics_config=None if metric_path is None else str(metric_path),
        include_diagnostics=include_diagnostics,
    )
    test_results = {"status": "not_requested", "metrics": {}}
    if test_path is not None:
        test_results = read_test_metrics(test_path, test_metric_specs)
        test_results["source"]["file"] = test_source
        # Rebasing source metadata must not bypass the Reader's output bound.
        encoded = json.dumps(test_results, ensure_ascii=False, allow_nan=False,
                             separators=(",", ":")).encode("utf-8")
        if len(encoded) > 16 * 1024:
            raise ValueError("experiment-relative test evidence exceeds 16 KiB")
    return {
        **legacy,
        "configuration_evidence": configuration_evidence,
        "validation_evidence": {
            "evidence_kind": "validation_history",
            "source": {"file": "history.json"},
        },
        "test_results": test_results,
    }


@dataclass(frozen=True, slots=True, init=False, eq=False, repr=False)
class ExperimentEvidenceTools:
    """Opt-in evidence analysis bound to one host-selected experiment.

    Host selections are detached snapshots. Session path authorization still
    runs before dispatch; this set additionally requires the exact bound root.
    Evidence reading and result semantics remain with the composition function.
    """

    _experiment_dir: Path
    _config_paths: Sequence[Sequence[str]]
    _test_metrics_file: str | None
    _test_metric_specs: Sequence[MetricSpec]
    _metrics_config: str | None

    def __init__(
        self,
        experiment_dir: str,
        *,
        config_paths: Sequence[Sequence[str]] = (),
        test_metrics_file: str | None = None,
        test_metric_specs: Sequence[MetricSpec] = (),
        metrics_config: str | None = None,
    ) -> None:
        if not isinstance(experiment_dir, str):
            raise TypeError("experiment_dir must be a string")
        if not experiment_dir.strip():
            raise ValueError("experiment_dir must not be empty or whitespace")
        root = Path(experiment_dir).resolve(strict=True)
        if not root.is_dir():
            raise NotADirectoryError("experiment_dir must be a directory")
        object.__setattr__(self, "_experiment_dir", root)
        object.__setattr__(self, "_config_paths", deepcopy(config_paths))
        object.__setattr__(self, "_test_metrics_file", deepcopy(test_metrics_file))
        object.__setattr__(self, "_test_metric_specs", deepcopy(test_metric_specs))
        object.__setattr__(self, "_metrics_config", deepcopy(metrics_config))

    def list_tools(self) -> list[dict]:
        """Return a fresh description exposing only model-owned arguments."""
        return [{
            "type": "function",
            "function": {
                "name": "analyze_experiment_evidence",
                "description": (
                    "Read selected configuration, validation history and optional "
                    "test evidence for the host-bound experiment. File-reported "
                    "metrics do not establish checkpoint provenance."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "experiment_dir": {"type": "string"},
                        "include_diagnostics": {"type": "boolean", "default": False},
                    },
                    "required": ["experiment_dir"],
                    "additionalProperties": False,
                },
            },
        }]

    def invoke_tool(self, tool_name: str, arguments: dict) -> dict:
        """Validate the model boundary, then delegate with host selections."""
        if not isinstance(tool_name, str):
            raise TypeError("tool_name must be a string")
        if not tool_name.strip():
            raise ValueError("tool_name must not be empty or whitespace")
        if tool_name != "analyze_experiment_evidence":
            raise KeyError("unknown evidence tool")
        if type(arguments) is not dict:
            raise TypeError("arguments must be a dict")
        if arguments.keys() - {"experiment_dir", "include_diagnostics"}:
            raise ValueError("unsupported evidence tool arguments")
        include_diagnostics = arguments.get("include_diagnostics", False)
        if type(include_diagnostics) is not bool:
            raise TypeError("include_diagnostics must be a boolean")
        experiment_dir = arguments.get("experiment_dir")
        if not isinstance(experiment_dir, str):
            raise TypeError("experiment_dir must be a string")
        if not experiment_dir.strip():
            raise ValueError("experiment_dir must not be empty or whitespace")
        if Path(experiment_dir).resolve(strict=False) != self._experiment_dir:
            raise ValueError("experiment_dir must match the host-bound experiment")
        return analyze_experiment_evidence(
            str(self._experiment_dir),
            config_paths=self._config_paths,
            test_metrics_file=self._test_metrics_file,
            test_metric_specs=self._test_metric_specs,
            metrics_config=self._metrics_config,
            include_diagnostics=include_diagnostics,
        )


@dataclass(frozen=True, slots=True, init=False, eq=False, repr=False)
class ExperimentComparisonTools:
    """Explicit comparison binding for one ordered pair of experiment roots.

    Formal Session authorization remains upstream of dispatch. This binding
    checks both identities before either composition, and keeps host metadata
    separate from file evidence. Only the frozen core interprets comparisons.
    """

    _experiment_a: Path
    _experiment_b: Path
    _options_a: dict
    _options_b: dict
    _targets: Sequence[Mapping]

    def __init__(
        self,
        experiment_a: str,
        experiment_b: str,
        *,
        options_a: Mapping | None = None,
        options_b: Mapping | None = None,
        targets: Sequence[Mapping],
    ) -> None:
        for label, value, options in (
            ("a", experiment_a, options_a), ("b", experiment_b, options_b),
        ):
            self._validate_directory_argument(value)
            root = Path(value).resolve(strict=True)
            if not root.is_dir():
                raise NotADirectoryError("bound experiment must be a directory")
            if options is None:
                options = {}
            if not isinstance(options, Mapping):
                raise TypeError("evidence options must be a mapping or None")
            if options.keys() - {
                "config_paths", "test_metrics_file", "test_metric_specs", "metrics_config",
            }:
                raise ValueError("unsupported comparison evidence options")
            object.__setattr__(self, "_experiment_" + label, root)
            object.__setattr__(self, "_options_" + label, deepcopy(dict(options)))
        if (isinstance(targets, (str, bytes, bytearray))
                or not isinstance(targets, Sequence)):
            raise TypeError("targets must be a sequence")
        object.__setattr__(self, "_targets", deepcopy(targets))

    @staticmethod
    def _validate_directory_argument(value: object) -> None:
        if not isinstance(value, str):
            raise TypeError("experiment directory must be a string")
        if not value.strip():
            raise ValueError("experiment directory must not be empty or whitespace")

    def list_tools(self) -> list[dict]:
        """Return an independent schema without host-only declarations."""
        return [{
            "type": "function",
            "function": {
                "name": "compare_experiment_evidence",
                "description": (
                    "Compare selected evidence for the host-bound ordered experiment "
                    "pair. Report numerical differences with comparability and "
                    "provenance limits; no checkpoint verification is implied."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "experiment_a": {"type": "string"},
                        "experiment_b": {"type": "string"},
                    },
                    "required": ["experiment_a", "experiment_b"],
                    "additionalProperties": False,
                },
            },
        }]

    @staticmethod
    def _supplement_validation_specs(targets, label, root, options) -> None:
        """Supplement detached declarations from a contained host-selected config.

        No default metric semantics are inferred and no M16 result is modified.
        The existing config reader and M16 child-path checks remain authoritative.
        """
        selected = [target[label] for target in targets
                    if target[label]["section"] == "validation"
                    and target[label]["metric_spec"] is None]
        config = options.get("metrics_config")
        if not selected or config is None:
            return
        path, _ = _evidence_file(root, config)
        specs = {spec.name: spec for spec in read_metric_specs_config(path)}
        for side in selected:
            side["metric_spec"] = specs.get(side["metric_name"])

    def invoke_tool(self, tool_name: str, arguments: dict) -> dict:
        """Validate the ordered pair, compose both sides, then call the core."""
        if not isinstance(tool_name, str):
            raise TypeError("tool_name must be a string")
        if not tool_name.strip():
            raise ValueError("tool_name must not be empty or whitespace")
        if tool_name != "compare_experiment_evidence":
            raise KeyError("unknown comparison tool")
        if type(arguments) is not dict:
            raise TypeError("arguments must be a dict")
        if arguments.keys() != {"experiment_a", "experiment_b"}:
            raise ValueError("comparison requires only experiment_a and experiment_b")
        for key, bound in (
            ("experiment_a", self._experiment_a), ("experiment_b", self._experiment_b),
        ):
            value = arguments[key]
            self._validate_directory_argument(value)
            if Path(value).resolve(strict=False) != bound:
                raise ValueError("experiment pair must match the host-bound ordered pair")

        options_a, options_b = deepcopy(self._options_a), deepcopy(self._options_b)
        targets = deepcopy(self._targets)
        evidence_a = analyze_experiment_evidence(str(self._experiment_a), **options_a)
        evidence_b = analyze_experiment_evidence(str(self._experiment_b), **options_b)
        self._supplement_validation_specs(targets, "a", self._experiment_a, options_a)
        self._supplement_validation_specs(targets, "b", self._experiment_b, options_b)
        return experiment_comparison.compare_experiment_evidence(
            evidence_a, evidence_b, targets=targets,
        )


def compare_experiments(
    experiment_root: str,
    sort_by: str | None = None,
    descending: bool = True,
    *,
    metrics_config: str | None = None,
    include_diagnostics: bool = False,
) -> dict:
    """Compare experiment directories and return a JSON-safe payload."""
    if not isinstance(experiment_root, str):
        raise TypeError("experiment_root must be a string")
    if not experiment_root.strip():
        raise ValueError(
            "experiment_root must not be empty or whitespace"
        )

    if sort_by is not None:
        if not isinstance(sort_by, str):
            raise TypeError("sort_by must be a string or None")
        if not sort_by.strip():
            raise ValueError(
                "sort_by must not be empty or whitespace"
            )

    if metrics_config is not None:
        if not isinstance(metrics_config, str):
            raise TypeError(
                "metrics_config must be a string or None"
            )
        if not metrics_config.strip():
            raise ValueError(
                "metrics_config must not be empty or whitespace"
            )

    if not isinstance(descending, bool):
        raise TypeError("descending must be a boolean")
    if not isinstance(include_diagnostics, bool):
        raise TypeError("include_diagnostics must be a boolean")

    root_path = Path(experiment_root)
    metric_specs = (
        None
        if metrics_config is None
        else read_metric_specs_config(Path(metrics_config))
    )
    experiment_dirs = find_experiment_dirs(root_path)
    batch_result = analyze_experiment_dirs(
        experiment_dirs,
        metric_specs=metric_specs,
    )

    if sort_by is not None:
        resolved_sort_by = sort_by
    elif metric_specs is None:
        resolved_sort_by = "best_r2"
    else:
        resolved_sort_by = metric_specs[0].name

    payload = build_comparison_payload(
        batch_result,
        sort_by=resolved_sort_by,
        descending=descending,
        metric_specs=metric_specs,
        include_diagnostics=include_diagnostics,
    )
    json.dumps(
        payload,
        ensure_ascii=False,
        allow_nan=False,
    )
    return payload

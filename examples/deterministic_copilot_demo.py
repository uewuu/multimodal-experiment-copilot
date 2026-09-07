"""Run a credential-free, deterministic experiment Copilot demo."""

from __future__ import annotations

import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
from types import SimpleNamespace


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from copilot import CopilotService


_TOOL_ARGUMENTS = {
    "experiment_root": ".",
    "include_diagnostics": True,
}


def _response(
    content: str | None,
    tool_calls: list[object] | None,
) -> SimpleNamespace:
    return SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(
                    role="assistant",
                    content=content,
                    tool_calls=tool_calls,
                )
            )
        ]
    )


class _DeterministicCompletions:
    """Return one comparison request followed by a grounded answer."""

    def __init__(self) -> None:
        self._call_count = 0

    def create(self, **request: object) -> SimpleNamespace:
        self._call_count += 1
        if self._call_count == 1:
            tool_call = SimpleNamespace(
                id="deterministic-comparison",
                type="function",
                function=SimpleNamespace(
                    name="compare_experiments",
                    arguments=json.dumps(
                        _TOOL_ARGUMENTS,
                        sort_keys=True,
                        separators=(",", ":"),
                    ),
                ),
            )
            return _response(None, [tool_call])

        if self._call_count == 2:
            messages = request["messages"]
            if type(messages) is not list or not messages:
                raise TypeError("follow-up messages must be a nonempty list")
            tool_message = messages[-1]
            if type(tool_message) is not dict:
                raise TypeError("final follow-up message must be a dict")
            comparison = json.loads(tool_message["content"])
            names = [
                record["experiment_name"]
                for record in comparison["comparison_records"]
            ]
            diagnostic_codes = [
                item["code"]
                for item in comparison["diagnostics"]["diagnostics"]
            ]
            answer = (
                f"Compared {len(names)} experiments successfully: "
                f"{', '.join(names)}. Diagnostics: "
                f"{', '.join(diagnostic_codes)}."
            )
            return _response(answer, [])

        raise AssertionError("unexpected extra provider request")


class _DeterministicClient:
    """Expose the attribute shape consumed by the existing adapter."""

    def __init__(self) -> None:
        completions = _DeterministicCompletions()
        self.chat = SimpleNamespace(completions=completions)


def _write_experiment(
    experiment_dir: Path,
    *,
    model_name: str,
    racc_values: list[float],
) -> None:
    experiment_dir.mkdir()
    (experiment_dir / "hparams.yaml").write_text(
        "config:\n"
        f"  model_name: {model_name}\n"
        "  batch_size: 16\n",
        encoding="utf-8",
    )
    history = {
        "valid": {
            "app": {
                "r2": [[0, 0.70], [1, 0.82]],
                "racc": [
                    [epoch, value]
                    for epoch, value in enumerate(racc_values)
                ],
            }
        }
    }
    (experiment_dir / "history.json").write_text(
        json.dumps(history, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _stable_comparison(result_json: str) -> dict:
    comparison = json.loads(result_json)
    for record in comparison["comparison_records"]:
        record["experiment_dir"] = (
            f"<temporary-workspace>/{record['experiment_name']}"
        )
    for failure in comparison["failed_experiments"]:
        failure["experiment_dir"] = (
            f"<temporary-workspace>/{failure['experiment_name']}"
        )
    return comparison


def build_demo_result() -> dict:
    """Run the real Copilot stack and return stable JSON-native output."""
    with TemporaryDirectory(prefix="experiment-copilot-demo-") as temp_dir:
        experiment_root = Path(temp_dir)
        _write_experiment(
            experiment_root / "baseline",
            model_name="linear_baseline",
            racc_values=[0.68, 0.78],
        )
        _write_experiment(
            experiment_root / "candidate",
            model_name="regularized_candidate",
            racc_values=[0.76, 0.91],
        )

        service = CopilotService(
            _DeterministicClient(),
            model="deterministic-demo-provider",
        )
        observed = service.run(
            "Compare the experiments and include diagnostics.",
            experiment_context={
                "experiment_root": str(experiment_root),
            },
        )

        if len(observed.turn.tool_invocations) != 1:
            raise RuntimeError("demo expected exactly one tool invocation")
        invocation = observed.turn.tool_invocations[0]
        comparison = _stable_comparison(invocation.result_json)
        return {
            "answer": observed.turn.answer,
            "comparison": comparison,
            "runtime": {
                "provider_request_count": (
                    observed.metrics.provider_request_count
                ),
                "tool_invocation_count": (
                    observed.metrics.tool_invocation_count
                ),
            },
            "tool_call": {
                "arguments": json.loads(invocation.arguments_json),
                "name": invocation.tool_name,
            },
        }


def main() -> None:
    """Print one stable JSON document for people and automation."""
    print(
        json.dumps(
            build_demo_result(),
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()

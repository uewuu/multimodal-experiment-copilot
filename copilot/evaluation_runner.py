"""Run one isolated evaluation scenario through public Session observations."""

from collections.abc import Callable

from . import evaluation
from .evaluation import EvaluationCase, EvaluationResult
from .failure_observability import CopilotFailureObservation
from .session import CopilotSession


__all__ = ("run_evaluation_case",)


def run_evaluation_case(
    case: EvaluationCase,
    session_factory: Callable[[], CopilotSession],
    *,
    context_prompts: tuple[str, ...] = (),
) -> EvaluationResult:
    """Execute context turns followed by case.prompt, then score the last turn.

    The factory must supply a fresh Session on each call and retains client
    ownership. Session alone manages history, including rollback on failure.
    Execution exceptions become sanitized evaluation failures; factory and
    scoring errors, as well as process-control exceptions, propagate unchanged.
    """
    session = session_factory()
    failure: CopilotFailureObservation | None = None

    def on_failure(observation: CopilotFailureObservation) -> None:
        nonlocal failure
        failure = observation

    for prompt in (*context_prompts, case.prompt):
        # Only this turn's callback may supply its failure identity.
        failure = None
        try:
            observed = session.ask_with_observability(
                prompt,
                on_failure=on_failure,
            )
        except Exception:
            run = None if failure is None else failure.run
            return EvaluationResult(
                case_id=case.case_id,
                scenario_version=case.scenario_version,
                passed=False,
                score=0.0,
                failure_reasons=("execution_failure",),
                run_id=None if run is None else run.run_id,
            )

    return evaluation.evaluate(case, observed)

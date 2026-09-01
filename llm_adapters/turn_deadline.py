"""Request-scoped monotonic deadline for one Copilot turn."""

from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
import math
from time import perf_counter as _perf_counter


_TURN_DEADLINE: ContextVar[float | None] = ContextVar(
    "copilot_turn_deadline",
    default=None,
)
_DEADLINE_EXCEEDED_MESSAGE = "Copilot turn deadline exceeded"


def _validate_turn_timeout_seconds(
    turn_timeout_seconds: object,
) -> float | None:
    if turn_timeout_seconds is None:
        return None
    if isinstance(turn_timeout_seconds, bool) or not isinstance(
        turn_timeout_seconds,
        (int, float),
    ):
        raise TypeError(
            "turn_timeout_seconds must be an int, float, or None"
        )

    try:
        validated_timeout = float(turn_timeout_seconds)
    except OverflowError as error:
        raise ValueError(
            "turn_timeout_seconds must be finite and greater than zero"
        ) from error
    if not math.isfinite(validated_timeout) or validated_timeout <= 0:
        raise ValueError(
            "turn_timeout_seconds must be finite and greater than zero"
        )
    return validated_timeout


def _validate_turn_deadline_options(
    turn_timeout_seconds: object,
    request_options: dict[str, object],
) -> float | None:
    validated_timeout = _validate_turn_timeout_seconds(
        turn_timeout_seconds
    )
    if validated_timeout is not None and "timeout" in request_options:
        raise TypeError(
            "timeout cannot be combined with turn_timeout_seconds"
        )
    return validated_timeout


@contextmanager
def _turn_deadline_scope(
    turn_timeout_seconds: float | None,
) -> Iterator[None]:
    expiry = (
        None
        if turn_timeout_seconds is None
        else _perf_counter() + turn_timeout_seconds
    )
    token = _TURN_DEADLINE.set(expiry)
    try:
        yield
    finally:
        _TURN_DEADLINE.reset(token)


def _remaining_turn_seconds() -> float | None:
    expiry = _TURN_DEADLINE.get()
    if expiry is None:
        return None
    remaining = expiry - _perf_counter()
    if remaining <= 0:
        raise TimeoutError(_DEADLINE_EXCEEDED_MESSAGE)
    return remaining


def _check_turn_deadline() -> None:
    _remaining_turn_seconds()

"""Bounded in-memory membership for Copilot sessions."""

from collections.abc import Callable
from uuid import uuid4

from .service import CopilotService
from .session import CopilotSession


__all__ = ("CopilotSessionRepository",)


def _default_id_factory() -> str:
    return str(uuid4())


class CopilotSessionRepository:
    """Manage opaque IDs for Sessions created by a borrowed Service."""

    def __init__(
        self,
        service: CopilotService,
        *,
        max_sessions: int,
        id_factory: Callable[[], str] | None = None,
    ) -> None:
        if isinstance(max_sessions, bool) or not isinstance(
            max_sessions,
            int,
        ):
            raise TypeError("max_sessions must be an integer")
        if max_sessions < 1:
            raise ValueError("max_sessions must be at least 1")
        if id_factory is not None and not callable(id_factory):
            raise TypeError("id_factory must be callable or None")

        self._service = service
        self._max_sessions = max_sessions
        self._id_factory = (
            _default_id_factory
            if id_factory is None
            else id_factory
        )
        self._sessions: dict[str, CopilotSession] = {}

    def create(
        self,
        *,
        experiment_context: dict[str, object] | None = None,
        max_turns: int = 8,
        turn_timeout_seconds: float | None = None,
        **request_options: object,
    ) -> str:
        """Create and store one Session under a new opaque ID."""
        if len(self._sessions) >= self._max_sessions:
            raise RuntimeError("session repository is full")

        session_id = self._id_factory()
        if not isinstance(session_id, str):
            raise TypeError("id_factory must return a string")
        if not session_id.strip():
            raise ValueError(
                "id_factory must return a non-empty session ID"
            )
        if session_id in self._sessions:
            raise RuntimeError("generated session ID already exists")

        session = self._service.create_session(
            experiment_context=experiment_context,
            max_turns=max_turns,
            turn_timeout_seconds=turn_timeout_seconds,
            **request_options,
        )
        self._sessions[session_id] = session
        return session_id

    def get(self, session_id: str) -> CopilotSession:
        """Return the exact Session stored under an opaque ID."""
        return self._sessions[session_id]

    def delete(self, session_id: str) -> None:
        """Remove one Session membership without owning its lifecycle."""
        del self._sessions[session_id]

from __future__ import annotations

from collections.abc import Awaitable, Callable
import inspect
from pathlib import Path
from typing import Protocol

from predict_rlm.trace import extract_trace_from_exc

from .agent.schema import FractalResult
from .agent.service import FractalAgent, coerce_trace
from .session import FractalSession, SessionHistoryTurn, SummaryTurn, session_path


class FractalAgentLike(Protocol):
    async def aforward(
        self,
        workspace_path: str | Path,
        user_message: str,
        rendered_session_summary: str = "",
        session_history: list[SessionHistoryTurn] | None = None,
    ) -> FractalResult: ...


class FractalRuntime:
    """Coordinates non-UI Fractal turn execution."""

    def __init__(
        self,
        *,
        workspace_path: str | Path,
        session: FractalSession,
        agent: FractalAgentLike,
    ) -> None:
        self.workspace_path = Path(workspace_path).resolve()
        self.session = session
        self.agent = agent

    @classmethod
    def create(
        cls,
        *,
        workspace_path: str | Path,
        lm: str | None,
        sub_lm: str | None,
        max_iterations: int,
        verbose: bool,
        debug: bool,
        session_id: str | None = None,
    ) -> "FractalRuntime":
        workspace = Path(workspace_path).resolve()
        runtime = cls(
            workspace_path=workspace,
            session=FractalSession.load(workspace),
            agent=FractalAgent(
                lm=lm,
                sub_lm=sub_lm,
                max_iterations=max_iterations,
                verbose=verbose,
                debug=debug,
            ),
        )
        if session_id is not None:
            runtime.resume(session_id)
        return runtime

    def resume(self, session_id: str) -> None:
        if not session_path(self.workspace_path, session_id).exists():
            raise FileNotFoundError(f"No Fractal session found for id {session_id!r}.")
        self.session = FractalSession.load(self.workspace_path, session_id=session_id)

    @property
    def session_id(self) -> str:
        return self.session.session_id

    @property
    def turns(self) -> list[SummaryTurn]:
        return self.session.turns

    async def submit(
        self,
        user_message: str,
        *,
        on_pending: Callable[[], Awaitable[None] | None] | None = None,
    ) -> FractalResult:
        turn_id = self.session.add_user_message(user_message)
        self.session.save(self.workspace_path)
        if on_pending is not None:
            pending_result = on_pending()
            if inspect.isawaitable(pending_result):
                await pending_result

        try:
            result = await self.agent.aforward(
                workspace_path=self.workspace_path,
                user_message=user_message,
                rendered_session_summary=self.session.summary(),
                session_history=self.session.session_history_payload(),
            )
        except Exception as exc:
            # Failed turns still need durable context for the next turn; the UI
            # can decide how loudly to surface the exception after it is saved.
            self.session.add_agent_failure(
                str(exc),
                trace=coerce_trace(extract_trace_from_exc(exc)),
                turn_id=turn_id,
            )
            self.session.save(self.workspace_path)
            raise

        self.session.add_agent_response(
            result.response,
            result.changed_files,
            trace=result.trace,
            turn_id=turn_id,
        )
        self.session.save(self.workspace_path)
        return result

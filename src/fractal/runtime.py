from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
import inspect
from pathlib import Path
from typing import Protocol

from predict_rlm.trace import extract_trace_from_exc

from .agent.schema import FractalResult
from .agent.service import FractalAgent, coerce_trace, create_sbx_interpreter
from .session import (
    INTERRUPTED_ERROR,
    MAX_ITERATIONS_ERROR,
    FractalSession,
    SessionHistoryTurn,
    SummaryTurn,
    session_path,
)


class FractalAgentLike(Protocol):
    async def aforward(
        self,
        workspace_path: str | Path,
        user_message: str,
        rendered_session_summary: str = "",
        session_history: list[SessionHistoryTurn] | None = None,
        included_paths: list[Path] | None = None,
    ) -> FractalResult: ...

    def close(self) -> None: ...

    def prewarm(self) -> None: ...


class FractalRuntime:
    """Coordinates non-UI Fractal turn execution."""

    def __init__(
        self,
        *,
        workspace_path: str | Path,
        included_paths: list[str | Path] | None = None,
        session: FractalSession,
        agent: FractalAgentLike,
    ) -> None:
        self.workspace_path = Path(workspace_path).resolve()
        self.included_paths = [Path(path).resolve() for path in included_paths or []]
        self.session = session
        self.agent = agent

    @classmethod
    def create(
        cls,
        *,
        workspace_path: str | Path,
        included_paths: list[str | Path] | None = None,
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
            included_paths=included_paths,
            session=FractalSession.load(workspace),
            agent=FractalAgent(
                lm=lm,
                sub_lm=sub_lm,
                max_iterations=max_iterations,
                verbose=verbose,
                debug=debug,
                interpreter=create_sbx_interpreter(workspace, included_paths),
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

    def close(self) -> None:
        close = getattr(self.agent, "close", None)
        if close is not None:
            close()

    def prewarm(self) -> None:
        prewarm = getattr(self.agent, "prewarm", None)
        if prewarm is not None:
            prewarm()

    async def submit(
        self,
        user_message: str,
        *,
        on_pending: Callable[[], Awaitable[None] | None] | None = None,
        interrupt_requested: Callable[[], bool] | None = None,
    ) -> FractalResult:
        turn_id = self.session.add_user_message(user_message)
        self.session.save(self.workspace_path)
        if on_pending is not None:
            pending_result = on_pending()
            if inspect.isawaitable(pending_result):
                await pending_result

        if interrupt_requested is not None and interrupt_requested():
            self.session.add_agent_turn(
                status="interrupted",
                error=INTERRUPTED_ERROR,
                turn_id=turn_id,
            )
            self.session.save(self.workspace_path)
            raise asyncio.CancelledError(INTERRUPTED_ERROR)

        try:
            result = await self.agent.aforward(
                workspace_path=self.workspace_path,
                user_message=user_message,
                rendered_session_summary=self.session.summary(),
                session_history=self.session.session_history_payload(),
                included_paths=self.included_paths,
            )
        except asyncio.CancelledError as exc:
            if interrupt_requested is None or not interrupt_requested():
                raise
            # Ctrl-C cancels the active turn. Persist it distinctly so the next
            # prompt and future resumes know this was user-initiated, not a
            # model/tool failure.
            self.session.add_agent_turn(
                status="interrupted",
                error=INTERRUPTED_ERROR,
                trace=coerce_trace(extract_trace_from_exc(exc)),
                turn_id=turn_id,
            )
            self.session.save(self.workspace_path)
            raise
        except Exception as exc:
            if interrupt_requested is not None and interrupt_requested():
                self.session.add_agent_turn(
                    status="interrupted",
                    error=INTERRUPTED_ERROR,
                    trace=coerce_trace(extract_trace_from_exc(exc)),
                    turn_id=turn_id,
                )
                self.session.save(self.workspace_path)
                raise asyncio.CancelledError(INTERRUPTED_ERROR) from exc
            # Failed turns still need durable context for the next turn; the UI
            # can decide how loudly to surface the exception after it is saved.
            self.session.add_agent_turn(
                status="failed",
                error=str(exc),
                trace=coerce_trace(extract_trace_from_exc(exc)),
                turn_id=turn_id,
            )
            self.session.save(self.workspace_path)
            raise

        # PredictRLM returns fallback output, not an exception, when the REPL
        # loop exhausts its budget. Preserve that output, but do not mark the
        # turn as a normal success.
        if result.trace is not None and result.trace.status == "max_iterations":
            self.session.add_agent_turn(
                status="max_iterations",
                response=result.response,
                changed_files=result.changed_files,
                trace=result.trace,
                turn_id=turn_id,
                error=MAX_ITERATIONS_ERROR,
            )
        else:
            self.session.add_agent_turn(
                status="succeeded",
                response=result.response,
                changed_files=result.changed_files,
                trace=result.trace,
                turn_id=turn_id,
            )
        self.session.save(self.workspace_path)
        return result

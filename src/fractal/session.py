from __future__ import annotations

import json
import shutil
import uuid
import warnings
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field
from predict_rlm import RunTrace


SESSION_DIR = ".fractal"
SESSIONS_DIR = "sessions"
SCHEMA_VERSION = 1
MAX_HISTORY_TURNS = 20
MAX_ITERATIONS_ERROR = (
    "Reached max iterations before SUBMIT; response came from fallback extraction."
)


def _new_session_id() -> str:
    return uuid.uuid4().hex


class UserTurn(BaseModel):
    message: str


class AgentTurn(BaseModel):
    status: Literal["succeeded", "failed", "max_iterations"]
    response: str = ""
    files_read: list[str] = Field(default_factory=list)
    files_modified: list[str] = Field(default_factory=list)
    error: str | None = None


class SummaryTurn(BaseModel):
    turn_id: str
    user: UserTurn
    agent: AgentTurn | None = None


class SessionSummary(BaseModel):
    turns: list[SummaryTurn] = Field(default_factory=list)


class SessionHistoryTurn(BaseModel):
    turn_id: str
    user_message: str
    status: Literal["pending", "succeeded", "failed", "max_iterations"]
    trace: RunTrace | None = None
    error: str | None = None
    created_at: str
    updated_at: str


class SessionState(BaseModel):
    schema_version: Literal[1] = SCHEMA_VERSION
    session_id: str = Field(default_factory=_new_session_id)
    summary: SessionSummary = Field(default_factory=SessionSummary)
    history: list[SessionHistoryTurn] = Field(default_factory=list)


class FractalSession:
    """Workspace-local Fractal session state."""

    def __init__(
        self, state: SessionState | None = None, *, session_id: str | None = None
    ) -> None:
        self.state = state or SessionState(session_id=session_id or _new_session_id())

    @property
    def session_id(self) -> str:
        return self.state.session_id

    @property
    def summary_model(self) -> SessionSummary:
        return self.state.summary

    @property
    def history(self) -> list[SessionHistoryTurn]:
        return self.state.history

    @property
    def turns(self) -> list[SummaryTurn]:
        return self.state.summary.turns

    @classmethod
    def load(
        cls, workspace_path: str | Path, *, session_id: str | None = None
    ) -> "FractalSession":
        if session_id is None:
            # Multi-session storage exists before resume selection does. Until
            # the CLI has an explicit selector, each process gets a fresh ID
            # instead of silently choosing the wrong prior conversation.
            return cls()

        path = session_path(workspace_path, session_id)
        if not path.exists():
            return cls(session_id=session_id)
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            backup_path = _backup_bad_session(path)
            backup_note = (
                f"Preserved a backup at {backup_path}."
                if backup_path is not None
                else "Could not preserve a backup."
            )
            warnings.warn(
                f"Ignoring unreadable Fractal session at {path}: {exc}. "
                f"{backup_note}",
                RuntimeWarning,
                stacklevel=2,
            )
            return cls(session_id=session_id)

        if not isinstance(data, dict):
            warnings.warn(
                f"Ignoring malformed Fractal session at {path}: expected a JSON object.",
                RuntimeWarning,
                stacklevel=2,
            )
            return cls(session_id=session_id)

        if data.get("schema_version") == SCHEMA_VERSION:
            try:
                state = SessionState.model_validate(data)
            except ValueError as exc:
                warnings.warn(
                    f"Ignoring malformed Fractal session at {path}: {exc}.",
                    RuntimeWarning,
                    stacklevel=2,
                )
                return cls(session_id=session_id)
            if state.session_id != session_id:
                warnings.warn(
                    f"Ignoring Fractal session at {path}: embedded session_id "
                    f"{state.session_id!r} does not match requested session_id "
                    f"{session_id!r}.",
                    RuntimeWarning,
                    stacklevel=2,
                )
                return cls(session_id=session_id)
            return cls(state)

        warnings.warn(
            f"Ignoring unsupported Fractal session format at {path}: "
            f"expected schema_version={SCHEMA_VERSION}.",
            RuntimeWarning,
            stacklevel=2,
        )
        return cls(session_id=session_id)

    def save(self, workspace_path: str | Path) -> None:
        self._enforce_history_limit()
        path = session_path(workspace_path, self.session_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = self.state.model_dump(mode="json")
        path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    def summary(self) -> str:
        return render_session_summary(self.state.summary)

    def session_history_payload(self) -> list[SessionHistoryTurn]:
        # Full traces can be large. The structured summary is the durable
        # compressed trajectory, so history is bounded to exact-recall data.
        self._enforce_history_limit()
        return list(self.state.history)

    def add_user_message(self, content: str) -> str:
        turn_id = f"turn-{uuid.uuid4().hex}"
        now = _utc_now()
        self.state.summary.turns.append(
            SummaryTurn(turn_id=turn_id, user=UserTurn(message=content))
        )
        self.state.history.append(
            SessionHistoryTurn(
                turn_id=turn_id,
                user_message=content,
                status="pending",
                created_at=now,
                updated_at=now,
            )
        )
        self._enforce_history_limit()
        return turn_id

    def add_agent_response(
        self,
        content: str,
        changed_files: list[str],
        *,
        trace: RunTrace | None = None,
        turn_id: str | None = None,
    ) -> None:
        summary_turn = self._find_summary_turn(turn_id)
        if summary_turn is None:
            generated_id = self.add_user_message("")
            summary_turn = self._find_summary_turn(generated_id)
            turn_id = generated_id
        assert summary_turn is not None
        files_modified = _coerce_string_list(changed_files)
        summary_turn.agent = AgentTurn(
            status="succeeded",
            response=content,
            files_modified=files_modified,
        )
        history_turn = self._find_history_turn(turn_id or summary_turn.turn_id)
        if history_turn is not None:
            history_turn.status = "succeeded"
            history_turn.trace = trace
            history_turn.error = None
            history_turn.updated_at = _utc_now()
        self._enforce_history_limit()

    def add_agent_failure(
        self,
        error: str,
        *,
        trace: RunTrace | None = None,
        turn_id: str | None = None,
    ) -> None:
        summary_turn = self._find_summary_turn(turn_id)
        if summary_turn is None:
            generated_id = self.add_user_message("")
            summary_turn = self._find_summary_turn(generated_id)
            turn_id = generated_id
        assert summary_turn is not None
        summary_turn.agent = AgentTurn(status="failed", error=error)
        history_turn = self._find_history_turn(turn_id or summary_turn.turn_id)
        if history_turn is not None:
            history_turn.status = "failed"
            history_turn.trace = trace
            history_turn.error = error
            history_turn.updated_at = _utc_now()
        self._enforce_history_limit()

    def add_agent_max_iterations(
        self,
        content: str,
        changed_files: list[str],
        *,
        trace: RunTrace | None = None,
        turn_id: str | None = None,
        error: str | None = None,
    ) -> None:
        summary_turn = self._find_summary_turn(turn_id)
        if summary_turn is None:
            generated_id = self.add_user_message("")
            summary_turn = self._find_summary_turn(generated_id)
            turn_id = generated_id
        assert summary_turn is not None
        files_modified = _coerce_string_list(changed_files)
        summary_turn.agent = AgentTurn(
            status="max_iterations",
            response=content,
            files_modified=files_modified,
            error=error,
        )
        history_turn = self._find_history_turn(turn_id or summary_turn.turn_id)
        if history_turn is not None:
            history_turn.status = "max_iterations"
            history_turn.trace = trace
            history_turn.error = error
            history_turn.updated_at = _utc_now()
        self._enforce_history_limit()

    def _find_summary_turn(self, turn_id: str | None) -> SummaryTurn | None:
        if turn_id is not None:
            for turn in reversed(self.state.summary.turns):
                if turn.turn_id == turn_id:
                    return turn
            return None
        return self.state.summary.turns[-1] if self.state.summary.turns else None

    def _find_history_turn(self, turn_id: str) -> SessionHistoryTurn | None:
        for turn in reversed(self.state.history):
            if turn.turn_id == turn_id:
                return turn
        return None

    def _enforce_history_limit(self) -> None:
        self.state.history = self.state.history[-MAX_HISTORY_TURNS:]


def render_session_summary(summary: SessionSummary) -> str:
    if not summary.turns:
        return "No prior Fractal session context."

    # The summary is rendered into prompt text, but the source of truth stays
    # structured so future context builders can reason over the same artifact.
    lines = ["## Session Summary"]
    for index, turn in enumerate(summary.turns, start=1):
        lines.append("")
        lines.append(f"Turn {index} ({turn.turn_id})")
        lines.append(f"User: {turn.user.message}")
        if turn.agent is None:
            lines.append("Agent status: pending")
            continue
        lines.append(f"Agent status: {turn.agent.status}")
        if turn.agent.response:
            lines.append(f"Agent response: {turn.agent.response}")
        if turn.agent.files_read:
            lines.append(f"Files read: {', '.join(turn.agent.files_read)}")
        if turn.agent.files_modified:
            lines.append(f"Files modified: {', '.join(turn.agent.files_modified)}")
        if turn.agent.error:
            lines.append(f"Error: {turn.agent.error}")
    return "\n".join(lines)


def sessions_dir_path(workspace_path: str | Path) -> Path:
    return Path(workspace_path) / SESSION_DIR / SESSIONS_DIR


def session_path(workspace_path: str | Path, session_id: str) -> Path:
    _validate_session_id(session_id)
    return sessions_dir_path(workspace_path) / f"{session_id}.json"


def _validate_session_id(session_id: str) -> None:
    # Session IDs eventually become user-provided resume selectors. Keeping
    # them as plain filenames avoids turning resume support into path access.
    if not session_id or Path(session_id).name != session_id or session_id.endswith(
        ".json"
    ):
        raise ValueError(f"Invalid Fractal session id: {session_id!r}")


def _backup_bad_session(path: Path) -> Path | None:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S")
    backup_path = path.with_suffix(f"{path.suffix}.bad-{stamp}")
    counter = 1
    while backup_path.exists():
        backup_path = path.with_suffix(f"{path.suffix}.bad-{stamp}-{counter}")
        counter += 1
    try:
        shutil.copy2(path, backup_path)
    except OSError:
        return None
    return backup_path


def _coerce_string_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    if isinstance(value, (list, tuple, set)):
        return [str(path) for path in value]
    return [str(value)]


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()

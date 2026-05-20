from __future__ import annotations

import asyncio
from pathlib import Path

import pytest


pytest.importorskip(
    "predict_rlm",
    reason="predict-rlm is required for Fractal runtime tests",
)


def test_runtime_submit_persists_success_and_exposes_pending_state(tmp_path: Path) -> None:
    from fractal.agent.schema import FractalResult
    from fractal.runtime import FractalRuntime
    from fractal.session import FractalSession

    pending_seen: list[str] = []
    calls: list[dict[str, object]] = []

    class FakeAgent:
        async def aforward(self, **kwargs: object) -> FractalResult:
            calls.append(kwargs)
            return FractalResult(response="updated docs", changed_files=["README.md"])

    session = FractalSession()
    runtime = FractalRuntime(
        workspace_path=tmp_path,
        session=session,
        agent=FakeAgent(),
    )

    async def on_pending() -> None:
        pending_seen.append(session.turns[-1].agent.status if session.turns[-1].agent else "pending")

    result = asyncio.run(runtime.submit("update docs", on_pending=on_pending))

    assert result.response == "updated docs"
    assert pending_seen == ["pending"]
    assert session.turns[-1].user.message == "update docs"
    assert session.turns[-1].agent is not None
    assert session.turns[-1].agent.response == "updated docs"
    assert session.turns[-1].agent.files_modified == ["README.md"]
    assert calls[0]["workspace_path"] == tmp_path
    assert calls[0]["user_message"] == "update docs"
    assert "update docs" in str(calls[0]["rendered_session_summary"])
    assert session.history[-1].status == "succeeded"


def test_runtime_submit_persists_failure_before_reraising(tmp_path: Path) -> None:
    from fractal.runtime import FractalRuntime
    from fractal.session import FractalSession

    class FailingAgent:
        async def aforward(self, **kwargs: object) -> object:
            raise RuntimeError("model failed")

    session = FractalSession()
    runtime = FractalRuntime(
        workspace_path=tmp_path,
        session=session,
        agent=FailingAgent(),
    )

    with pytest.raises(RuntimeError, match="model failed"):
        asyncio.run(runtime.submit("run tests"))

    assert session.turns[-1].agent is not None
    assert session.turns[-1].agent.status == "failed"
    assert session.turns[-1].agent.error == "model failed"
    assert session.history[-1].status == "failed"
    assert session.history[-1].error == "model failed"


def test_runtime_create_and_resume_load_session_ids(tmp_path: Path) -> None:
    from fractal.runtime import FractalRuntime
    from fractal.session import FractalSession

    class FakeAgent:
        async def aforward(self, **kwargs: object) -> object:
            raise AssertionError("agent should not run")

    existing = FractalSession(session_id="existing")
    existing.add_user_message("prior work")
    existing.save(tmp_path)

    runtime = FractalRuntime(
        workspace_path=tmp_path,
        session=FractalSession(),
        agent=FakeAgent(),
    )
    runtime.resume("existing")

    assert runtime.session_id == "existing"
    assert runtime.turns[-1].user.message == "prior work"

    created = FractalRuntime.create(
        workspace_path=tmp_path,
        lm=None,
        sub_lm=None,
        max_iterations=1,
        verbose=False,
        debug=False,
        session_id="existing",
    )

    assert created.session_id == "existing"
    assert created.turns[-1].user.message == "prior work"


def test_runtime_resume_requires_existing_session(tmp_path: Path) -> None:
    from fractal.runtime import FractalRuntime
    from fractal.session import FractalSession

    class FakeAgent:
        async def aforward(self, **kwargs: object) -> object:
            raise AssertionError("agent should not run")

    runtime = FractalRuntime(
        workspace_path=tmp_path,
        session=FractalSession(session_id="current"),
        agent=FakeAgent(),
    )

    with pytest.raises(FileNotFoundError, match="missing"):
        runtime.resume("missing")

    with pytest.raises(FileNotFoundError, match="missing"):
        FractalRuntime.create(
            workspace_path=tmp_path,
            lm=None,
            sub_lm=None,
            max_iterations=1,
            verbose=False,
            debug=False,
            session_id="missing",
        )

    assert runtime.session_id == "current"

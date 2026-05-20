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

from __future__ import annotations

import json
from pathlib import Path

import pytest


pytest.importorskip(
    "predict_rlm",
    reason="predict-rlm is required for Fractal session tests",
)


def test_session_round_trip(tmp_path: Path) -> None:
    from predict_rlm import RunTrace

    from fractal.session import FractalSession, SCHEMA_VERSION, session_path

    session = FractalSession()
    turn_id = session.add_user_message("change the README")
    trace = RunTrace(
        status="completed",
        model="test-model",
        iterations=1,
        max_iterations=3,
        duration_ms=10,
    )
    session.add_agent_response(
        "updated",
        ["README.md"],
        trace=trace,
        turn_id=turn_id,
    )
    session.save(tmp_path)

    loaded = FractalSession.load(tmp_path, session_id=session.session_id)
    payload = json.loads(
        session_path(tmp_path, session.session_id).read_text(encoding="utf-8")
    )

    assert payload["schema_version"] == SCHEMA_VERSION
    assert payload["session_id"] == session.session_id
    assert session_path(tmp_path, session.session_id).name == f"{session.session_id}.json"
    assert loaded.session_id == session.session_id
    assert "change the README" in loaded.summary()
    assert loaded.turns[-1].user.message == "change the README"
    assert loaded.turns[-1].agent is not None
    assert loaded.turns[-1].agent.response == "updated"
    assert loaded.turns[-1].agent.files_modified == ["README.md"]
    assert loaded.history[-1].trace == trace


def test_load_without_session_id_starts_new_session(tmp_path: Path) -> None:
    from fractal.session import FractalSession

    existing = FractalSession()
    existing.add_user_message("old context")
    existing.save(tmp_path)

    loaded = FractalSession.load(tmp_path)

    assert loaded.session_id != existing.session_id
    assert loaded.turns == []
    assert loaded.history == []


def test_session_load_ignores_unsupported_format(tmp_path: Path) -> None:
    from fractal.session import FractalSession, session_path

    session_id = "legacy-placeholder"
    session_path(tmp_path, session_id).parent.mkdir(parents=True)
    session_path(tmp_path, session_id).write_text(
        json.dumps(
            {
                "turns": [
                    {"role": "user", "content": "change docs", "changed_files": []},
                    {
                        "role": "assistant",
                        "content": "done",
                        "changed_files": "README.md",
                    },
                ]
            }
        ),
        encoding="utf-8",
    )

    with pytest.warns(RuntimeWarning, match="unsupported Fractal session format"):
        loaded = FractalSession.load(tmp_path, session_id=session_id)

    assert loaded.session_id == session_id
    assert loaded.turns == []
    assert loaded.history == []


def test_session_load_resets_malformed_file_under_requested_id(tmp_path: Path) -> None:
    from fractal.session import FractalSession, session_path

    session_id = "malformed"
    session_path(tmp_path, session_id).parent.mkdir(parents=True)
    session_path(tmp_path, session_id).write_text("[]", encoding="utf-8")

    with pytest.warns(RuntimeWarning, match="expected a JSON object"):
        loaded = FractalSession.load(tmp_path, session_id=session_id)

    assert loaded.session_id == session_id
    assert loaded.turns == []


def test_session_can_store_failed_turn(tmp_path: Path) -> None:
    from predict_rlm import RunTrace

    from fractal.session import FractalSession

    session = FractalSession()
    turn_id = session.add_user_message("run tests")
    trace = RunTrace(
        status="error",
        model="test-model",
        iterations=1,
        max_iterations=3,
        duration_ms=10,
    )
    session.add_agent_failure(
        "pytest failed",
        trace=trace,
        turn_id=turn_id,
    )
    session.save(tmp_path)

    loaded = FractalSession.load(tmp_path, session_id=session.session_id)

    assert loaded.turns[-1].agent is not None
    assert loaded.turns[-1].agent.status == "failed"
    assert loaded.turns[-1].agent.error == "pytest failed"
    assert loaded.history[-1].status == "failed"
    assert loaded.history[-1].trace == trace


def test_session_can_store_max_iteration_turn(tmp_path: Path) -> None:
    from predict_rlm import RunTrace

    from fractal.session import FractalSession

    session = FractalSession()
    turn_id = session.add_user_message("finish task")
    trace = RunTrace(
        status="max_iterations",
        model="test-model",
        iterations=3,
        max_iterations=3,
        duration_ms=10,
    )
    from fractal.session import MAX_ITERATIONS_ERROR

    session.add_agent_max_iterations(
        "fallback answer",
        ["README.md"],
        trace=trace,
        turn_id=turn_id,
        error=MAX_ITERATIONS_ERROR,
    )
    session.save(tmp_path)

    loaded = FractalSession.load(tmp_path, session_id=session.session_id)

    assert loaded.turns[-1].agent is not None
    assert loaded.turns[-1].agent.status == "max_iterations"
    assert loaded.turns[-1].agent.response == "fallback answer"
    assert loaded.turns[-1].agent.files_modified == ["README.md"]
    assert loaded.history[-1].status == "max_iterations"
    assert loaded.history[-1].trace == trace
    assert "Agent status: max_iterations" in loaded.summary()


def test_load_rejects_mismatched_embedded_session_id(tmp_path: Path) -> None:
    from fractal.session import FractalSession, session_path

    session = FractalSession(session_id="first")
    session.save(tmp_path)
    session_path(tmp_path, "second").parent.mkdir(parents=True, exist_ok=True)
    session_path(tmp_path, "second").write_text(
        session_path(tmp_path, "first").read_text(encoding="utf-8"),
        encoding="utf-8",
    )

    with pytest.warns(RuntimeWarning, match="does not match requested session_id"):
        loaded = FractalSession.load(tmp_path, session_id="second")

    assert loaded.session_id == "second"
    assert loaded.turns == []


def test_session_id_cannot_address_paths(tmp_path: Path) -> None:
    from fractal.session import session_path

    with pytest.raises(ValueError, match="Invalid Fractal session id"):
        session_path(tmp_path, "../outside")

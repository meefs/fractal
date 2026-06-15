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

    from fractal.session import SCHEMA_VERSION, FractalSession, session_path

    session = FractalSession()
    turn_id = session.add_user_message("change the README")
    trace = RunTrace(
        status="completed",
        model="test-model",
        iterations=1,
        max_iterations=3,
        duration_ms=10,
    )
    session.add_agent_turn(
        status="succeeded",
        response="updated",
        files_read=["src/app.py"],
        changed_files=["README.md"],
        commands_run=["uv run pytest"],
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
    assert loaded.turns[-1].agent.files_read_count == 1
    assert loaded.turns[-1].agent.files_changed_count == 1
    assert loaded.turns[-1].agent.commands_run_count == 1
    assert loaded.history[-1].files_read == ["src/app.py"]
    assert loaded.history[-1].files_modified == ["README.md"]
    assert loaded.history[-1].commands_run == ["uv run pytest"]
    assert loaded.history[-1].trace == trace
    summary_agent = payload["summary"]["turns"][-1]["agent"]
    assert summary_agent["files_read_count"] == 1
    assert summary_agent["files_changed_count"] == 1
    assert summary_agent["commands_run_count"] == 1
    assert "files_read" not in summary_agent
    assert "files_modified" not in summary_agent
    assert "commands_run" not in summary_agent
    assert payload["history"][-1]["files_read"] == ["src/app.py"]
    assert payload["history"][-1]["files_modified"] == ["README.md"]
    assert payload["history"][-1]["commands_run"] == ["uv run pytest"]
    assert "files_read_count: 1" in loaded.summary()
    assert "files_changed_count: 1" in loaded.summary()
    assert "commands_run_count: 1" in loaded.summary()
    assert "Files read:" not in loaded.summary()
    assert "Files modified:" not in loaded.summary()
    assert "Commands run:" not in loaded.summary()


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
    session.add_agent_turn(
        status="failed",
        error="pytest failed",
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

    session.add_agent_turn(
        status="max_iterations",
        response="fallback answer",
        changed_files=["README.md"],
        trace=trace,
        turn_id=turn_id,
        error=MAX_ITERATIONS_ERROR,
    )
    session.save(tmp_path)

    loaded = FractalSession.load(tmp_path, session_id=session.session_id)

    assert loaded.turns[-1].agent is not None
    assert loaded.turns[-1].agent.status == "max_iterations"
    assert loaded.turns[-1].agent.response == "fallback answer"
    assert loaded.turns[-1].agent.files_changed_count == 1
    assert loaded.history[-1].files_modified == ["README.md"]
    assert loaded.history[-1].status == "max_iterations"
    assert loaded.history[-1].trace == trace
    assert "Agent status: max_iterations" in loaded.summary()


def test_session_can_store_interrupted_turn(tmp_path: Path) -> None:
    from predict_rlm import RunTrace

    from fractal.session import INTERRUPTED_ERROR, FractalSession

    session = FractalSession()
    turn_id = session.add_user_message("long task")
    trace = RunTrace(
        status="error",
        model="test-model",
        iterations=1,
        max_iterations=3,
        duration_ms=10,
    )
    session.add_agent_turn(
        status="interrupted",
        error=INTERRUPTED_ERROR,
        trace=trace,
        turn_id=turn_id,
    )
    session.save(tmp_path)

    loaded = FractalSession.load(tmp_path, session_id=session.session_id)

    assert loaded.turns[-1].agent is not None
    assert loaded.turns[-1].agent.status == "interrupted"
    assert loaded.turns[-1].agent.error == INTERRUPTED_ERROR
    assert loaded.history[-1].status == "interrupted"
    assert loaded.history[-1].trace == trace
    assert "Agent status: interrupted" in loaded.summary()


def test_session_requires_user_turn_before_agent_turn() -> None:
    from fractal.session import FractalSession

    session = FractalSession()

    with pytest.raises(ValueError, match="before a user turn"):
        session.add_agent_turn(status="succeeded")

    session.add_user_message("hello")

    with pytest.raises(ValueError, match="missing"):
        session.add_agent_turn(status="succeeded", turn_id="missing")


def test_session_rejects_non_list_changed_files() -> None:
    from fractal.session import FractalSession

    session = FractalSession()
    turn_id = session.add_user_message("change docs")

    with pytest.raises(TypeError, match="changed_files"):
        session.add_agent_turn(
            status="succeeded",
            changed_files="README.md",
            turn_id=turn_id,
        )


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


def test_turn_usage_from_trace_records_tokens_and_context() -> None:
    from predict_rlm.trace import (
        IterationStep,
        LMUsage,
        RunTrace,
        TokenUsage,
    )

    from fractal.session import turn_usage_from_trace

    trace = RunTrace(
        status="completed",
        model="test-model",
        iterations=2,
        max_iterations=5,
        duration_ms=1234,
        usage=LMUsage(
            main=TokenUsage(input_tokens=7000, output_tokens=300, cost=0.04),
            sub=TokenUsage(input_tokens=1200, output_tokens=100, cost=0.01),
        ),
        steps=[
            IterationStep(
                iteration=1,
                reasoning="r",
                code="x",
                output="o",
                untruncated_output="o",
                duration_ms=10,
                usage=LMUsage(main=TokenUsage(input_tokens=3000)),
            ),
            IterationStep(
                iteration=2,
                reasoning="r",
                code="x",
                output="o",
                untruncated_output="o",
                duration_ms=10,
                usage=LMUsage(main=TokenUsage(input_tokens=4100)),
            ),
        ],
    )

    usage = turn_usage_from_trace(trace)

    assert usage is not None
    assert usage.input_tokens == 8200
    assert usage.output_tokens == 400
    assert usage.cost == pytest.approx(0.05)
    assert usage.duration_ms == 1234
    assert usage.iterations == 2
    assert usage.context_tokens == 4100


def test_turn_usage_from_trace_handles_missing_trace() -> None:
    from fractal.session import turn_usage_from_trace

    assert turn_usage_from_trace(None) is None


def test_usage_totals_aggregates_across_turns(tmp_path: Path) -> None:
    from fractal.session import FractalSession, TurnUsage

    session = FractalSession()
    for context in (3000, 4100):
        turn_id = session.add_user_message("do it")
        session.add_agent_turn(status="succeeded", response="ok", turn_id=turn_id)
        turn = session.turns[-1]
        assert turn.agent is not None
        turn.agent.usage = TurnUsage(
            input_tokens=1000,
            output_tokens=100,
            cost=0.01,
            duration_ms=500,
            iterations=2,
            context_tokens=context,
        )
    session.save(tmp_path)
    reloaded = FractalSession.load(tmp_path, session_id=session.session_id)

    totals = reloaded.usage_totals()

    assert totals.input_tokens == 2000
    assert totals.output_tokens == 200
    assert totals.cost == pytest.approx(0.02)
    assert totals.duration_ms == 1000
    assert totals.iterations == 4
    assert totals.context_tokens == 4100


def test_list_sessions_orders_and_skips_invalid(tmp_path: Path) -> None:
    import os

    from fractal.session import FractalSession, list_sessions, sessions_dir_path

    older = FractalSession()
    older.add_user_message("first request")
    older.save(tmp_path)
    newer = FractalSession()
    newer.add_user_message("second request")
    newer.save(tmp_path)
    older_path = sessions_dir_path(tmp_path) / f"{older.session_id}.json"
    newer_path = sessions_dir_path(tmp_path) / f"{newer.session_id}.json"
    os.utime(older_path, (1_000_000, 1_000_000))
    os.utime(newer_path, (2_000_000, 2_000_000))
    (sessions_dir_path(tmp_path) / "garbage.json").write_text("not json")

    sessions = list_sessions(tmp_path)

    assert [info.session_id for info in sessions] == [
        newer.session_id,
        older.session_id,
    ]
    assert sessions[0].first_message == "second request"
    assert sessions[0].turn_count == 1


def test_list_sessions_empty_workspace(tmp_path: Path) -> None:
    from fractal.session import list_sessions

    assert list_sessions(tmp_path) == []

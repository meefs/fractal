from __future__ import annotations

import asyncio
from io import StringIO
from pathlib import Path

import pytest
from rich.console import Console


pytest.importorskip(
    "predict_rlm",
    reason="predict-rlm is required for Fractal TUI session models",
)
pytest.importorskip("prompt_toolkit", reason="prompt_toolkit is required for input")


class FakeRuntime:
    def __init__(
        self,
        tmp_path: Path,
        *,
        fail: bool = False,
        include_trace: bool = False,
    ) -> None:
        from fractal.session import FractalSession

        self.workspace_path = tmp_path
        self.session = FractalSession(session_id="test-session")
        self.submitted: list[str] = []
        self.fail = fail
        self.include_trace = include_trace

    @property
    def session_id(self) -> str:
        return self.session.session_id

    async def submit(self, user_message: str, **kwargs: object) -> object:
        from fractal.agent.schema import FractalResult

        self.submitted.append(user_message)
        turn_id = self.session.add_user_message(user_message)
        on_pending = kwargs.get("on_pending")
        if on_pending is not None:
            pending_result = on_pending()
            if pending_result is not None:
                await pending_result
        if self.fail:
            self.session.add_agent_failure("model failed", turn_id=turn_id)
            raise RuntimeError("model failed")
        self.session.add_agent_response(
            f"response to {user_message}",
            [],
            trace=self._trace() if self.include_trace else None,
            turn_id=turn_id,
        )
        return FractalResult(
            response=f"response to {user_message}",
            trace=self._trace() if self.include_trace else None,
        )

    def _trace(self) -> object:
        from predict_rlm.trace import IterationStep, RunTrace

        return RunTrace(
            status="completed",
            model="test-model",
            iterations=2,
            max_iterations=3,
            duration_ms=25,
            steps=[
                IterationStep(
                    iteration=1,
                    reasoning=(
                        "Inspect the files first and keep the reasoning long enough "
                        "to wrap in a narrow terminal."
                    ),
                    code="print('a')\nprint('b')",
                    output="hello",
                    untruncated_output="hello world",
                    duration_ms=10,
                ),
                IterationStep(
                    iteration=2,
                    reasoning="Apply the edit.",
                    code="x = 1",
                    output="done",
                    untruncated_output="done",
                    duration_ms=15,
                ),
            ],
        )


class FakePromptSession:
    def __init__(self, messages: list[str]) -> None:
        self.messages = messages
        self.prompts: list[object] = []

    async def prompt_async(self, prompt: object) -> str:
        self.prompts.append(prompt)
        if not self.messages:
            raise EOFError
        return self.messages.pop(0)


def capture_console() -> tuple[Console, StringIO]:
    output = StringIO()
    return (
        Console(
            file=output,
            force_terminal=True,
            color_system=None,
            width=56,
            legacy_windows=False,
        ),
        output,
    )


def test_terminal_tui_renders_summary_as_native_output(tmp_path: Path) -> None:
    from fractal.tui.app import render_summary

    runtime = FakeRuntime(tmp_path)
    turn_id = runtime.session.add_user_message("hello fractal")
    runtime.session.add_agent_response("hello human", [], turn_id=turn_id)
    console, output = capture_console()

    console.print(render_summary(runtime.session.summary_model))

    text = output.getvalue()
    assert "You" in text
    assert "RLM" in text
    assert "hello fractal" in text
    assert "hello human" in text
    assert "─" in text


def test_terminal_tui_renders_compact_trace_summary(tmp_path: Path) -> None:
    from fractal.tui.app import render_trace_summary

    runtime = FakeRuntime(tmp_path, include_trace=True)
    trace = runtime._trace()
    console, output = capture_console()

    console.print(render_trace_summary(trace))

    text = output.getvalue()
    assert "RLM turn 1/3" in text
    assert "reasoning: Inspect the files first" in text
    assert "\n             reasoning long enough to wrap in a narrow" in text
    assert "python: 2 lines" in text
    assert "output: 11 chars" in text
    assert "\n\nRLM turn 2/3" in text
    assert "RLM turn 2/3" in text
    assert "python: 1 lines" in text
    assert "print('a')" not in text


def test_terminal_tui_run_submits_and_prints_to_scrollback(tmp_path: Path) -> None:
    from fractal.tui import TerminalFractalApp

    runtime = FakeRuntime(tmp_path, include_trace=True)
    console, output = capture_console()
    app = TerminalFractalApp(
        runtime,
        console=console,
        input_stream=StringIO("fix\n/exit\n"),
    )

    asyncio.run(app.run())

    text = output.getvalue()
    assert runtime.submitted == ["fix"]
    assert "fractal> " in text
    assert "fix" in text
    assert "You" not in text
    assert "Running..." not in text
    assert "✓ complete" in text
    assert "RLM turn 1/3" in text
    assert "python: 2 lines" in text
    assert "output: 11 chars" in text
    assert "response to fix" in text


def test_terminal_tui_failed_submit_renders_error_and_continues(tmp_path: Path) -> None:
    from fractal.tui import TerminalFractalApp

    runtime = FakeRuntime(tmp_path, fail=True)
    console, output = capture_console()
    app = TerminalFractalApp(
        runtime,
        console=console,
        input_stream=StringIO("fail\n/exit\n"),
    )

    asyncio.run(app.run())

    text = output.getvalue()
    assert runtime.submitted == ["fail"]
    assert "✗ failed" in text
    assert "model failed" in text
    assert "You" not in text
    assert "Running..." not in text
    assert text.count("fractal> ") == 2


def test_terminal_tui_uses_prompt_toolkit_session_for_live_input(tmp_path: Path) -> None:
    from fractal.tui import TerminalFractalApp

    runtime = FakeRuntime(tmp_path)
    console, output = capture_console()
    prompt_session = FakePromptSession(["fix", "/exit"])
    app = TerminalFractalApp(
        runtime,
        console=console,
        prompt_session=prompt_session,
    )

    asyncio.run(app.run())

    text = output.getvalue()
    assert runtime.submitted == ["fix"]
    assert prompt_session.prompts
    assert "fractal" in str(prompt_session.prompts[0])
    assert "fractal> " not in text
    assert "response to fix" in text

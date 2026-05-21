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
        max_iterations: bool = False,
    ) -> None:
        from fractal.session import FractalSession

        self.workspace_path = tmp_path
        self.session = FractalSession(session_id="test-session")
        self.submitted: list[str] = []
        self.resumed: list[str] = []
        self.fail = fail
        self.include_trace = include_trace
        self.max_iterations = max_iterations

    @property
    def session_id(self) -> str:
        return self.session.session_id

    def resume(self, session_id: str) -> None:
        from fractal.session import FractalSession
        from fractal.session import session_path

        self.resumed.append(session_id)
        if not session_path(self.workspace_path, session_id).exists():
            raise FileNotFoundError(f"No Fractal session found for id {session_id!r}.")
        self.session = FractalSession.load(self.workspace_path, session_id=session_id)

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
        response = self._response(user_message)
        trace = self._trace() if self.include_trace or self.max_iterations else None
        if self.max_iterations:
            from fractal.session import MAX_ITERATIONS_ERROR

            self.session.add_agent_max_iterations(
                response,
                [],
                trace=trace,
                turn_id=turn_id,
                error=MAX_ITERATIONS_ERROR,
            )
        else:
            self.session.add_agent_response(
                response,
                [],
                trace=trace,
                turn_id=turn_id,
            )
        return FractalResult(
            response=response,
            trace=trace,
        )

    def _response(self, user_message: str) -> str:
        return f"response to {user_message}"

    def _trace(self) -> object:
        from predict_rlm.trace import IterationStep, RunTrace

        return RunTrace(
            status="max_iterations" if self.max_iterations else "completed",
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
    assert text.startswith("\nfractal› hello fractal\n")
    assert "fractal›" in text
    assert "You" not in text
    assert "RLM" in text
    assert "hello fractal" in text
    assert "hello human" in text
    assert "─" in text


def test_terminal_tui_renders_final_response_as_markdown(tmp_path: Path) -> None:
    from rich.markdown import Markdown

    from fractal.tui.app import FractalMarkdown
    from fractal.tui.app import MARKDOWN_STYLE_OVERRIDES
    from fractal.tui.app import render_agent_message

    runtime = FakeRuntime(tmp_path)
    turn_id = runtime.session.add_user_message("format response")
    runtime.session.add_agent_response(
        "**Done**\n\n- updated `README.md`",
        [],
        turn_id=turn_id,
    )
    turn = runtime.session.turns[-1]

    panel = render_agent_message(turn)

    assert isinstance(panel.renderable, Markdown)
    assert isinstance(panel.renderable, FractalMarkdown)
    assert MARKDOWN_STYLE_OVERRIDES["markdown.code"] == "bold cyan"


def test_terminal_tui_renders_max_iteration_response_as_incomplete(tmp_path: Path) -> None:
    from rich.console import Group

    from fractal.tui.app import render_agent_message

    runtime = FakeRuntime(tmp_path)
    turn_id = runtime.session.add_user_message("finish task")
    from fractal.session import MAX_ITERATIONS_ERROR

    runtime.session.add_agent_max_iterations(
        "fallback response",
        [],
        turn_id=turn_id,
        error=MAX_ITERATIONS_ERROR,
    )
    turn = runtime.session.turns[-1]

    panel = render_agent_message(turn)

    assert panel.border_style == "yellow"
    assert isinstance(panel.renderable, Group)


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
    assert "fractal› " in text
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
    assert text.count("fractal› ") == 2


def test_terminal_tui_max_iterations_is_not_rendered_as_complete(tmp_path: Path) -> None:
    from fractal.tui import TerminalFractalApp

    runtime = FakeRuntime(tmp_path, max_iterations=True)
    console, output = capture_console()
    app = TerminalFractalApp(
        runtime,
        console=console,
        input_stream=StringIO("finish\n/exit\n"),
    )

    asyncio.run(app.run())

    text = output.getvalue()
    assert runtime.submitted == ["finish"]
    assert "! max iterations" in text
    assert "✓ complete" not in text
    assert "Reached max iterations; showing fallback response." in text
    assert "response to finish" in text


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
    assert "fractal› " not in text
    assert "response to fix" in text


def test_terminal_tui_resume_command_switches_sessions(tmp_path: Path) -> None:
    from fractal.session import FractalSession
    from fractal.tui import TerminalFractalApp

    existing = FractalSession(session_id="existing")
    turn_id = existing.add_user_message("prior request")
    existing.add_agent_response("prior response", [], turn_id=turn_id)
    existing.save(tmp_path)

    runtime = FakeRuntime(tmp_path)
    console, output = capture_console()
    app = TerminalFractalApp(
        runtime,
        console=console,
        input_stream=StringIO("/resume existing\n/exit\n"),
    )

    asyncio.run(app.run())

    text = output.getvalue()
    assert runtime.resumed == ["existing"]
    assert runtime.session_id == "existing"
    assert "resumed session existing" in text
    assert "fractal› prior request" in text
    assert "prior request" in text
    assert "prior response" in text


def test_terminal_tui_resume_command_requires_session_id(tmp_path: Path) -> None:
    from fractal.tui import TerminalFractalApp

    runtime = FakeRuntime(tmp_path)
    console, output = capture_console()
    app = TerminalFractalApp(
        runtime,
        console=console,
        input_stream=StringIO("/resume\n/exit\n"),
    )

    asyncio.run(app.run())

    assert runtime.resumed == []
    assert "usage: /resume <session-id>" in output.getvalue()


def test_terminal_tui_resume_command_errors_for_missing_session(tmp_path: Path) -> None:
    from fractal.tui import TerminalFractalApp

    runtime = FakeRuntime(tmp_path)
    console, output = capture_console()
    app = TerminalFractalApp(
        runtime,
        console=console,
        input_stream=StringIO("/resume missing\n/exit\n"),
    )

    asyncio.run(app.run())

    text = output.getvalue()
    assert runtime.resumed == ["missing"]
    assert runtime.session_id == "test-session"
    assert "No Fractal session found for id 'missing'." in text


def test_slash_command_completer_lists_commands() -> None:
    from prompt_toolkit.document import Document

    from fractal.tui.app import SlashCommandCompleter

    completer = SlashCommandCompleter()

    resume = list(completer.get_completions(Document("/r"), None))
    none_after_space = list(completer.get_completions(Document("/resume "), None))

    assert [completion.text for completion in resume] == ["/resume"]
    assert none_after_space == []

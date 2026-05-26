from __future__ import annotations

import asyncio
from io import StringIO
from pathlib import Path
import signal

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
        interrupt: bool = False,
        interrupt_count: int = 1,
    ) -> None:
        from fractal.session import FractalSession

        self.workspace_path = tmp_path
        self.session = FractalSession(session_id="test-session")
        self.submitted: list[str] = []
        self.resumed: list[str] = []
        self.fail = fail
        self.include_trace = include_trace
        self.max_iterations = max_iterations
        self.interrupt_count = interrupt_count if interrupt else 0

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
        interrupt_requested = kwargs.get("interrupt_requested")
        if interrupt_requested is not None and interrupt_requested():
            self.session.add_agent_turn(status="interrupted", turn_id=turn_id)
            raise asyncio.CancelledError
        if len(self.submitted) <= self.interrupt_count:
            signal.raise_signal(signal.SIGINT)
            try:
                await asyncio.sleep(0)
            except asyncio.CancelledError:
                self.session.add_agent_turn(status="interrupted", turn_id=turn_id)
                raise
        if self.fail:
            self.session.add_agent_turn(
                status="failed",
                error="model failed",
                turn_id=turn_id,
            )
            raise RuntimeError("model failed")
        response = self._response(user_message)
        trace = self._trace() if self.include_trace or self.max_iterations else None
        if self.max_iterations:
            from fractal.session import MAX_ITERATIONS_ERROR

            self.session.add_agent_turn(
                status="max_iterations",
                response=response,
                changed_files=[],
                trace=trace,
                turn_id=turn_id,
                error=MAX_ITERATIONS_ERROR,
            )
        else:
            self.session.add_agent_turn(
                status="succeeded",
                response=response,
                changed_files=[],
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

    async def prompt_async(self, prompt: object, **kwargs: object) -> str:
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


def normalized_output(text: str) -> str:
    return " ".join(text.split())


def agent_statuses(runtime: FakeRuntime) -> list[str | None]:
    return [
        turn.agent.status if turn.agent is not None else None
        for turn in runtime.session.turns
    ]


def test_terminal_tui_renders_summary_as_native_output(tmp_path: Path) -> None:
    from rich.padding import Padding

    from fractal.tui.app import render_summary
    from fractal.tui.app import render_agent_response
    from fractal.tui.app import render_user_message

    runtime = FakeRuntime(tmp_path)
    turn_id = runtime.session.add_user_message("hello fractal")
    runtime.session.add_agent_turn(
        status="succeeded",
        response="hello human",
        changed_files=[],
        turn_id=turn_id,
    )
    console, output = capture_console()

    console.print(render_summary(runtime.session.summary_model))

    text = output.getvalue()
    assert "hello fractal" in text
    assert "hello human" in text
    user_message = render_user_message("hello fractal")
    assert user_message.renderables[1].spans[-1].style == "bold"
    agent_response = render_agent_response(runtime.session.turns[-1])
    assert isinstance(agent_response, Padding)
    assert agent_response.left == 2


def test_terminal_tui_prompt_label_is_purple() -> None:
    from fractal.tui.app import PROMPT_STYLE
    from fractal.tui.app import render_prompt_label

    label = render_prompt_label()

    assert label.spans[0].style == "bold #8b5cf6"
    assert PROMPT_STYLE.style_rules[0] == ("prompt", "bold #8b5cf6")


def test_terminal_tui_renders_final_response_as_markdown(tmp_path: Path) -> None:
    from rich.markdown import Markdown

    from fractal.tui.app import FractalMarkdown
    from fractal.tui.app import render_agent_message

    runtime = FakeRuntime(tmp_path)
    turn_id = runtime.session.add_user_message("format response")
    runtime.session.add_agent_turn(
        status="succeeded",
        response="**Done**\n\n- updated `README.md`",
        changed_files=[],
        turn_id=turn_id,
    )
    turn = runtime.session.turns[-1]

    message = render_agent_message(turn)

    assert isinstance(message, Markdown)
    assert isinstance(message, FractalMarkdown)


def test_terminal_tui_renders_max_iteration_response_as_incomplete(tmp_path: Path) -> None:
    from fractal.tui.app import render_agent_message

    runtime = FakeRuntime(tmp_path)
    turn_id = runtime.session.add_user_message("finish task")
    from fractal.session import MAX_ITERATIONS_ERROR

    runtime.session.add_agent_turn(
        status="max_iterations",
        response="fallback response",
        changed_files=[],
        turn_id=turn_id,
        error=MAX_ITERATIONS_ERROR,
    )
    turn = runtime.session.turns[-1]

    panel = render_agent_message(turn)
    console, output = capture_console()
    console.print(panel)

    text = output.getvalue()
    assert "Reached max iterations; showing fallback response." in text
    assert "fallback response" in text


def test_terminal_tui_renders_interrupted_response(tmp_path: Path) -> None:
    from fractal.tui.app import render_agent_message

    runtime = FakeRuntime(tmp_path)
    turn_id = runtime.session.add_user_message("long task")
    runtime.session.add_agent_turn(status="interrupted", turn_id=turn_id)
    turn = runtime.session.turns[-1]

    rendered = render_agent_message(turn)

    assert "interrupted" in str(rendered)


def test_terminal_tui_renders_compact_trace_summary(tmp_path: Path) -> None:
    from fractal.tui.app import render_trace_summary

    runtime = FakeRuntime(tmp_path, include_trace=True)
    trace = runtime._trace()
    console, output = capture_console()

    console.print(render_trace_summary(trace))

    text = output.getvalue()
    normalized_text = normalized_output(text)
    assert "RLM turn 1/3" in text
    assert "reasoning: Inspect the files first" in text
    assert (
        "Inspect the files first and keep the reasoning long enough to wrap "
        "in a narrow terminal."
    ) in normalized_text
    assert "python: 2 lines" in text
    assert "output: 11 chars" in text
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
    assert "Running..." not in text


def test_terminal_tui_interrupted_submit_renders_and_continues(tmp_path: Path) -> None:
    from fractal.tui import TerminalFractalApp

    runtime = FakeRuntime(tmp_path, interrupt=True)
    console, output = capture_console()
    app = TerminalFractalApp(
        runtime,
        console=console,
        input_stream=StringIO("stop\nnext\n/exit\n"),
    )

    asyncio.run(app.run())

    text = output.getvalue()
    assert runtime.submitted == ["stop", "next"]
    assert "! turn interrupted" not in text
    assert "Turn interrupted by user." in text
    assert "response to next" in text
    assert "✓ complete" in text


def test_terminal_tui_handles_interrupt_before_submit_task_is_active(tmp_path: Path) -> None:
    from fractal.tui import TerminalFractalApp

    runtime = FakeRuntime(tmp_path)
    console, output = capture_console()
    app = TerminalFractalApp(
        runtime,
        console=console,
        input_stream=StringIO(""),
    )

    app._sigint_mode = "turn"
    app._handle_sigint(signal.SIGINT, None)
    result = asyncio.run(app.run_turn("early stop"))

    text = output.getvalue()
    assert result is None
    assert runtime.submitted == ["early stop"]
    assert runtime.session.turns[-1].agent is not None
    assert runtime.session.turns[-1].agent.status == "interrupted"
    assert "! turn interrupted" not in text
    assert "Turn interrupted by user." in text


def test_terminal_tui_run_turn_propagates_external_cancellation(tmp_path: Path) -> None:
    from fractal.tui import TerminalFractalApp

    class BlockingRuntime(FakeRuntime):
        async def submit(self, user_message: str, **kwargs: object) -> object:
            self.submitted.append(user_message)
            self.session.add_user_message(user_message)
            await asyncio.Event().wait()

    runtime = BlockingRuntime(tmp_path)
    console, _ = capture_console()
    app = TerminalFractalApp(
        runtime,
        console=console,
        input_stream=StringIO(""),
    )

    async def cancel_run_turn() -> None:
        task = asyncio.create_task(app.run_turn("shutdown"))
        for _ in range(10):
            if runtime.submitted:
                break
            await asyncio.sleep(0)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

    asyncio.run(cancel_run_turn())

    assert runtime.submitted == ["shutdown"]
    assert runtime.session.turns[-1].agent is None
    assert runtime.session.history[-1].status == "pending"


def test_terminal_tui_sigint_updates_active_status_immediately(tmp_path: Path) -> None:
    from fractal.tui import TerminalFractalApp
    from fractal.tui.app import INTERRUPTING_STATUS

    class FakeStatus:
        def __init__(self) -> None:
            self.updates: list[str] = []

        def update(self, value: str) -> None:
            self.updates.append(value)

    runtime = FakeRuntime(tmp_path)
    app = TerminalFractalApp(
        runtime,
        input_stream=StringIO(""),
    )
    status = FakeStatus()

    app._sigint_mode = "turn"
    app._active_status = status
    app._handle_sigint(signal.SIGINT, None)

    assert app._turn_interrupt_requested is True
    assert status.updates == [INTERRUPTING_STATUS]


def test_terminal_tui_prompt_mode_sigint_does_not_escape_handler(tmp_path: Path) -> None:
    from fractal.tui import TerminalFractalApp

    runtime = FakeRuntime(tmp_path)
    app = TerminalFractalApp(
        runtime,
        input_stream=StringIO(""),
    )

    app._sigint_mode = "prompt"
    app._handle_sigint(signal.SIGINT, None)

    assert app._turn_interrupt_requested is False


def test_terminal_tui_two_interrupted_submits_do_not_exit(tmp_path: Path) -> None:
    from fractal.tui import TerminalFractalApp

    runtime = FakeRuntime(tmp_path, interrupt=True, interrupt_count=2)
    console, output = capture_console()
    app = TerminalFractalApp(
        runtime,
        console=console,
        input_stream=StringIO("stop\nagain\nfinal\n/exit\n"),
    )

    asyncio.run(app.run())

    text = output.getvalue()
    assert runtime.submitted == ["stop", "again", "final"]
    assert "! turn interrupted" not in text
    assert agent_statuses(runtime) == ["interrupted", "interrupted", "succeeded"]
    assert "response to final" in text
    assert "✓ complete" in text


def test_terminal_tui_late_prompt_sigint_after_interrupt_does_not_exit(
    tmp_path: Path,
) -> None:
    from fractal.tui import TerminalFractalApp

    class LateSigintApp(TerminalFractalApp):
        def __init__(self, *args: object, **kwargs: object) -> None:
            super().__init__(*args, **kwargs)
            self.messages = ["stop", "again", "final", "/exit"]
            self.injected_late_sigints = 0

        async def read_message(self) -> str | None:
            if len(runtime.submitted) in {1, 2} and self.injected_late_sigints < 2:
                self.injected_late_sigints += 1
                self._handle_sigint(signal.SIGINT, None)
            return self.messages.pop(0)

    runtime = FakeRuntime(tmp_path, interrupt=True, interrupt_count=2)
    console, output = capture_console()
    app = LateSigintApp(runtime, console=console)

    asyncio.run(app.run())

    text = output.getvalue()
    assert runtime.submitted == ["stop", "again", "final"]
    assert app.injected_late_sigints == 2
    assert agent_statuses(runtime) == ["interrupted", "interrupted", "succeeded"]
    assert "response to final" in text


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
    assert "response to fix" in text


def test_terminal_tui_resume_command_switches_sessions(tmp_path: Path) -> None:
    from fractal.session import FractalSession
    from fractal.tui import TerminalFractalApp

    existing = FractalSession(session_id="existing")
    turn_id = existing.add_user_message("prior request")
    existing.add_agent_turn(
        status="succeeded",
        response="prior response",
        changed_files=[],
        turn_id=turn_id,
    )
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

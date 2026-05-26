from __future__ import annotations

import asyncio
from pathlib import Path
import signal
from typing import Protocol, TextIO

from prompt_toolkit import PromptSession
from prompt_toolkit.completion import Completer, Completion
from prompt_toolkit.document import Document
from prompt_toolkit.formatted_text import HTML
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.styles import Style
from rich.console import Console, Group
from rich.markdown import Markdown
from rich.padding import Padding
from rich.rule import Rule
from rich.table import Table
from rich.text import Text
from rich.theme import Theme

from fractal.agent.schema import FractalResult
from fractal.session import SessionSummary, SummaryTurn
from predict_rlm import RunTrace


PROMPT_STYLE = Style.from_dict(
    {
        "prompt": "bold #8b5cf6",
        "session": "ansibrightblack",
    }
)
SLASH_COMMANDS = {
    "/resume": "Resume an existing session by id",
    "/exit": "Exit Fractal",
    "/quit": "Exit Fractal",
}
RUNNING_STATUS = "[dim]running RLM... (Ctrl-C to interrupt)[/dim]"
INTERRUPTING_STATUS = "[yellow]interrupting RLM...[/yellow] [dim](waiting for shutdown)[/dim]"
MARKDOWN_STYLE_OVERRIDES = {
    # Rich defaults inline code to "cyan on black", which reads as a
    # highlight block inside Fractal's already framed response panel.
    # Keep Markdown emphasis visible without adding another background.
    "markdown.code": "bold cyan",
    "markdown.code_block": "cyan",
    "markdown.strong": "bold",
    "markdown.item.bullet": "bright_black",
    "markdown.list": "none",
}
MARKDOWN_THEME = Theme(MARKDOWN_STYLE_OVERRIDES)


class FractalMarkdown(Markdown):
    def __rich_console__(self, console: Console, options: object) -> object:
        with console.use_theme(MARKDOWN_THEME):
            yield from super().__rich_console__(console, options)


class SlashCommandCompleter(Completer):
    def get_completions(self, document: Document, complete_event: object) -> object:
        text = document.text_before_cursor
        if not text.startswith("/") or " " in text:
            return
        for command, description in SLASH_COMMANDS.items():
            if command.startswith(text):
                yield Completion(
                    command,
                    start_position=-len(text),
                    display_meta=description,
                )


def slash_command_key_bindings() -> KeyBindings:
    bindings = KeyBindings()

    @bindings.add("enter")
    def _(event: object) -> None:
        buffer = event.current_buffer
        complete_state = buffer.complete_state
        completion = (
            complete_state.current_completion if complete_state is not None else None
        )
        if completion is not None:
            buffer.apply_completion(completion)
            if not buffer.document.text_before_cursor.endswith(" "):
                buffer.insert_text(" ")
            return
        buffer.validate_and_handle()

    return bindings


class SessionLike(Protocol):
    @property
    def summary_model(self) -> SessionSummary: ...


class FractalRuntimeLike(Protocol):
    workspace_path: Path

    @property
    def session_id(self) -> str: ...

    @property
    def session(self) -> SessionLike: ...

    def resume(self, session_id: str) -> None: ...

    async def submit(self, user_message: str, **kwargs: object) -> FractalResult: ...


class TerminalFractalApp:
    """Terminal-native Fractal interface using the user's normal scrollback."""

    def __init__(
        self,
        runtime: FractalRuntimeLike,
        *,
        console: Console | None = None,
        input_stream: TextIO | None = None,
        prompt_session: PromptSession[str] | None = None,
    ) -> None:
        self.runtime = runtime
        self.console = console or Console()
        self.input_stream = input_stream
        self.prompt_session = prompt_session or PromptSession(
            style=PROMPT_STYLE,
            completer=SlashCommandCompleter(),
            complete_while_typing=True,
            key_bindings=slash_command_key_bindings(),
        )
        self._rendered_turn_ids: set[str] = set()
        self._pending_turn_ids: set[str] = set()
        self._prompt_echo_turn_ids: set[str] = set()
        self._sigint_mode = "prompt"
        self._active_submit_task: asyncio.Task[FractalResult] | None = None
        self._turn_interrupt_requested = False
        self._active_status: object | None = None

    async def run(self) -> None:
        previous_sigint_handler = signal.getsignal(signal.SIGINT)
        signal.signal(signal.SIGINT, self._handle_sigint)
        try:
            self.render_header()
            self.render_new_turns()

            while True:
                self._sigint_mode = "prompt"
                self._active_submit_task = None
                self._turn_interrupt_requested = False

                message = await self.read_message()
                if message is None or message in {"/exit", "/quit"}:
                    return
                if not message:
                    continue
                if self.handle_slash_command(message):
                    self._sigint_mode = "prompt"
                    continue

                self._sigint_mode = "turn"
                result = await self.run_turn(message)
                if result is None:
                    continue
                if result.trace is not None and result.trace.status == "max_iterations":
                    self.console.print(Text("! max iterations", style="yellow"))
                else:
                    self.console.print(Text("✓ complete"))
                if result.trace is not None:
                    self.console.print(render_trace_summary(result.trace))
                self.render_new_turns()
        finally:
            signal.signal(signal.SIGINT, previous_sigint_handler)

    async def run_turn(self, message: str) -> FractalResult | None:
        def mark_pending() -> None:
            self.mark_latest_turn_as_prompt_echoed()

        status = self.console.status(RUNNING_STATUS, spinner="dots")
        status.start()
        self._active_status = status
        status_running = True
        if self._turn_interrupt_requested:
            self._show_interrupting_status()

        def stop_status() -> None:
            nonlocal status_running
            if status_running:
                status.stop()
                status_running = False

        submit_task = asyncio.create_task(
            self.runtime.submit(
                message,
                on_pending=mark_pending,
                interrupt_requested=lambda: self._turn_interrupt_requested,
            )
        )
        self._active_submit_task = submit_task
        try:
            result = await submit_task
        except asyncio.CancelledError:
            if not self._turn_interrupt_requested:
                raise
            stop_status()
            self.render_new_turns()
            return None
        except Exception:
            stop_status()
            self.console.print(Text("✗ failed", style="red"))
            self.render_new_turns()
            return None
        finally:
            self._active_submit_task = None
            self._active_status = None
            self._sigint_mode = "prompt"
            stop_status()
        return result

    def _handle_sigint(self, signum: int, frame: object) -> None:
        if self._sigint_mode != "turn":
            # A second Ctrl-C can arrive after the interrupted turn has already
            # returned control to the prompt. Raising from the process signal
            # handler escapes prompt_toolkit/asyncio and crashes the CLI.
            return
        self._turn_interrupt_requested = True
        self._show_interrupting_status()
        task = self._active_submit_task
        if task is not None and not task.done():
            task.cancel()

    def _show_interrupting_status(self) -> None:
        status = self._active_status
        if status is None:
            return
        update = getattr(status, "update", None)
        if update is not None:
            update(INTERRUPTING_STATUS)

    def render_header(self) -> None:
        self.console.print(
            Text.assemble(
                ("Fractal", "bold"),
                " | ",
                (str(self.runtime.workspace_path), "dim"),
                " | session ",
                (self.runtime.session_id, "cyan"),
            )
        )
        self.console.print(Text("Type /exit or /quit to quit.", style="dim"))

    def handle_slash_command(self, message: str) -> bool:
        command, _, rest = message.partition(" ")
        if command != "/resume":
            return False
        session_id = rest.strip()
        if not session_id:
            self.console.print(Text("usage: /resume <session-id>", style="yellow"))
            return True
        try:
            self.runtime.resume(session_id)
        except FileNotFoundError as exc:
            self.console.print(Text(str(exc), style="red"))
            return True
        self._rendered_turn_ids.clear()
        self._pending_turn_ids.clear()
        self._prompt_echo_turn_ids.clear()
        self.console.print(Text(f"resumed session {self.runtime.session_id}", style="dim"))
        self.render_new_turns()
        return True

    def render_new_turns(self) -> None:
        for turn in self.runtime.session.summary_model.turns:
            if turn.turn_id in self._rendered_turn_ids:
                continue
            if turn.agent is None:
                if turn.turn_id not in self._pending_turn_ids:
                    if turn.turn_id not in self._prompt_echo_turn_ids:
                        self.render_turn(turn, pending=True)
                    self._pending_turn_ids.add(turn.turn_id)
                continue
            if turn.turn_id in self._pending_turn_ids:
                self.console.print(Rule(style="dim"))
                self.console.print(render_agent_response(turn))
                self._pending_turn_ids.remove(turn.turn_id)
            elif turn.turn_id in self._prompt_echo_turn_ids:
                self.console.print(Rule(style="dim"))
                self.console.print(render_agent_response(turn))
            else:
                self.render_turn(turn)
            self._rendered_turn_ids.add(turn.turn_id)

    def render_turn(self, turn: SummaryTurn, *, pending: bool = False) -> None:
        self.console.print(render_user_message(turn.user.message))
        self.console.print(Rule(style="dim"))
        self.console.print(render_agent_response(turn, pending=pending))

    def mark_latest_turn_as_prompt_echoed(self) -> None:
        if self.runtime.session.summary_model.turns:
            self._prompt_echo_turn_ids.add(
                self.runtime.session.summary_model.turns[-1].turn_id
            )

    async def read_message(self) -> str | None:
        self.console.print()
        if self.input_stream is None:
            try:
                message = await self.prompt_session.prompt_async(
                    HTML("<prompt>fractal</prompt><session>›</session> "),
                    handle_sigint=False,
                )
            except (EOFError, KeyboardInterrupt):
                self.console.print()
                return None
            message = message.strip()
            if _will_submit_turn(message):
                self._sigint_mode = "turn"
            return message

        try:
            message = await asyncio.to_thread(self._readline)
        except EOFError:
            self.console.print()
            return None
        message = message.strip()
        if _will_submit_turn(message):
            self._sigint_mode = "turn"
        return message

    def _readline(self) -> str:
        assert self.input_stream is not None
        self.console.print(render_prompt_label(), end="")
        line = self.input_stream.readline()
        if line == "":
            raise EOFError
        return line


def render_summary(summary: SessionSummary) -> Group:
    rendered: list[object] = []
    for index, turn in enumerate(summary.turns):
        if index > 0:
            rendered.append(Rule(style="dim"))
        rendered.append(render_user_message(turn.user.message))
        rendered.append(Rule(style="dim"))
        rendered.append(render_agent_response(turn))
    return Group(*rendered)


def render_trace_summary(trace: RunTrace) -> Group:
    rendered: list[object] = []
    if trace.steps:
        rendered.append("")
    for index, step in enumerate(trace.steps):
        if index > 0:
            rendered.append("")
        rendered.append(render_trace_step(trace, step))
    if not rendered:
        rendered.append(Text("No RLM iteration trace captured.", style="dim italic"))
    return Group(*rendered)


def render_trace_step(trace: RunTrace, step: object) -> Group:
    code = str(getattr(step, "code", "") or "")
    output = str(
        getattr(step, "untruncated_output", None) or getattr(step, "output", "") or ""
    )
    reasoning = str(getattr(step, "reasoning", "") or "").strip()
    iteration = int(getattr(step, "iteration", len(trace.steps)))
    status = "error" if bool(getattr(step, "error", False)) else "ok"

    text = Text()
    text.append(
        f"RLM turn {iteration}/{trace.max_iterations} ",
        style="bold bright_black",
    )
    text.append(f"({status})", style="red" if status == "error" else "dim")
    rendered: list[object] = [text]
    if reasoning:
        rendered.append(render_reasoning(reasoning))
    rendered.append(
        Padding(
            Text(
                f"python: {_line_count(code)} lines\noutput: {len(output)} chars",
                style="dim",
            ),
            (0, 0, 0, 2),
        )
    )
    return Group(*rendered)


def render_reasoning(reasoning: str) -> Padding:
    table = Table.grid(expand=True, padding=(0, 1))
    table.add_column(no_wrap=True)
    table.add_column(ratio=1)
    table.add_row(
        Text("reasoning:", style="dim italic"),
        Text(reasoning, style="dim italic"),
    )
    return Padding(table, (0, 0, 0, 2))


def render_user_message(message: str) -> Group:
    return Group(
        "",
        Text.assemble(render_prompt_label(), (message, "bold")),
    )


def render_prompt_label() -> Text:
    return Text.assemble(("fractal", "bold #8b5cf6"), ("›", "bright_black"), " ")


def _will_submit_turn(message: str) -> bool:
    if not message or message in {"/exit", "/quit"}:
        return False
    command, _, _ = message.partition(" ")
    return command != "/resume"


def render_agent_message(turn: SummaryTurn, *, pending: bool = False) -> object:
    if pending or turn.agent is None:
        return Text("Running...", style="italic dim")
    elif turn.agent.status == "failed":
        return Text(turn.agent.error or "Turn failed.", style="red")
    elif turn.agent.status == "interrupted":
        return Text(turn.agent.error or "Turn interrupted by user.", style="yellow")
    elif turn.agent.status == "max_iterations":
        # PredictRLM's fallback can contain useful work, but the agent did not
        # explicitly SUBMIT it. Make that state visible in scrollback.
        response: Text | Markdown
        if turn.agent.response:
            response = FractalMarkdown(turn.agent.response)
        else:
            response = Text("No fallback response.", style="dim")
        body = Group(
            Text("Reached max iterations; showing fallback response.", style="yellow"),
            "",
            response,
        )
        return body
    else:
        return FractalMarkdown(turn.agent.response)


def render_agent_response(turn: SummaryTurn, *, pending: bool = False) -> Padding:
    return Padding(render_agent_message(turn, pending=pending), (0, 0, 0, 2))


def _line_count(text: str) -> int:
    return len(text.splitlines()) if text else 0

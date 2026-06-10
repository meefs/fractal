from __future__ import annotations

import argparse
import asyncio
from pathlib import Path
import sys
from typing import Any, TextIO

from .config_commands import run_config_command
from .runtime_lms import resolve_runtime_lms


MAX_STDIN_BYTES = 10 * 1024 * 1024
MAX_ITERATIONS_EXIT_CODE = 2
DEFAULT_MAX_ITERATIONS = 30


FRACTAL_BANNER = r"""
 ______              _        _
|  ____|            | |      | |
| |__ _ __ __ _  ___| |_ __ _| |
|  __| '__/ _` |/ __| __/ _` | |
| |  | | | (_| | (__| || (_| | |
|_|  |_|  \__,_|\___|\__\__,_|_|
""".strip("\n")


def _print_startup_banner(output: TextIO) -> None:
    print(FRACTAL_BANNER, file=output)
    print(file=output)


def _effective_max_iterations(args: argparse.Namespace, lm_config: Any) -> int:
    if args.max_iterations is not None:
        return args.max_iterations
    defaults = getattr(lm_config, "defaults", None)
    if defaults is not None and defaults.max_iterations is not None:
        return defaults.max_iterations
    return DEFAULT_MAX_ITERATIONS


def _effective_verbose(args: argparse.Namespace, lm_config: Any) -> bool:
    if getattr(args, "verbose", False):
        return True
    defaults = getattr(lm_config, "defaults", None)
    return bool(defaults is not None and defaults.verbose)


def include_path(value: str) -> Path:
    path = Path(value)
    if path.is_symlink():
        raise argparse.ArgumentTypeError(f"included path cannot be a symlink: {value}")
    resolved = path.resolve()
    if not resolved.exists():
        raise argparse.ArgumentTypeError(f"included path does not exist: {value}")
    if not resolved.is_dir():
        raise argparse.ArgumentTypeError(f"included path is not a directory: {value}")
    return resolved


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="fractal")
    parser.add_argument(
        "--workspace",
        type=Path,
        default=Path.cwd(),
        help="workspace directory to edit; defaults to the current directory",
    )
    parser.add_argument("--lm", help="override the configured main DSPy LM")
    parser.add_argument(
        "--include",
        action="append",
        default=[],
        type=include_path,
        help=(
            "additional directory to mount into the sandbox at its absolute path; "
            "may be passed multiple times"
        ),
    )
    parser.add_argument("--sub-lm", help="override the configured DSPy sub-LM")
    parser.add_argument(
        "--max-iterations",
        type=int,
        default=None,
        help="max RLM iterations per turn; defaults to the configured value or 30",
    )
    parser.add_argument(
        "--resume",
        metavar="SESSION_ID",
        help="resume an existing workspace-local session by id",
    )
    parser.add_argument(
        "--quiet", action="store_true", help="reserved for quieter terminal output"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="show generated code and model-visible output for each RLM iteration",
    )
    parser.add_argument(
        "--debug", action="store_true", help="enable PredictRLM debug mode"
    )
    parser.add_argument(
        "-p",
        "--prompt",
        help=(
            "run one Fractal turn non-interactively with this prompt; use '-' "
            "to read the full prompt from stdin"
        ),
    )

    subparsers = parser.add_subparsers(dest="command")
    config_parser = subparsers.add_parser(
        "config",
        help="inspect or repair global Fractal provider/model configuration",
    )
    config_subparsers = config_parser.add_subparsers(
        dest="config_command",
        required=True,
    )
    config_subparsers.add_parser(
        "show",
        help="show effective global config with credential references redacted",
    )
    status_parser = config_subparsers.add_parser(
        "status",
        help="validate configured provider, model, and auth availability",
    )
    status_parser.add_argument(
        "--offline",
        action="store_true",
        help="skip the live provider connectivity check",
    )
    setup_parser = config_subparsers.add_parser(
        "setup",
        help="run interactive global provider/model/auth setup",
    )
    setup_parser.add_argument(
        "--offline",
        action="store_true",
        help="skip the live provider connectivity check after setup",
    )
    get_parser = config_subparsers.add_parser(
        "get",
        help="print one effective config value by dotted key",
    )
    get_parser.add_argument("key", help="dotted key, e.g. defaults.max_iterations")
    set_parser = config_subparsers.add_parser(
        "set",
        help="set one config value by dotted key",
    )
    set_parser.add_argument("key", help="dotted key, e.g. active_model")
    set_parser.add_argument("value", help="TOML literal or bare string")
    set_parser.add_argument(
        "--project",
        action="store_true",
        help="write to the workspace project config instead of the global config",
    )
    unset_parser = config_subparsers.add_parser(
        "unset",
        help="remove one config value by dotted key",
    )
    unset_parser.add_argument("key", help="dotted key, e.g. active_sub_model")
    unset_parser.add_argument(
        "--project",
        action="store_true",
        help="edit the workspace project config instead of the global config",
    )
    return parser


def run_tui(args: argparse.Namespace) -> int:
    from rich.console import Console

    console = Console()
    lm_config = resolve_runtime_lms(
        args,
        stdin=sys.stdin,
        stdout=sys.stdout,
        stderr=sys.stderr,
        auto_setup=_stdin_is_tty(sys.stdin),
    )
    if lm_config is None:
        return 1

    from .runtime import FractalRuntime
    from .tui import TerminalFractalApp

    workspace = args.workspace.resolve()
    display_verbose = _effective_verbose(args, lm_config)
    runtime = FractalRuntime.create(
        workspace_path=workspace,
        included_paths=args.include,
        lm=lm_config.lm,
        sub_lm=lm_config.sub_lm,
        max_iterations=_effective_max_iterations(args, lm_config),
        verbose=False,
        debug=args.debug,
        session_id=args.resume,
        provider_selection=lm_config.provider_selection,
        sub_lm_follows_main=lm_config.sub_lm_follows_main,
        sub_model=lm_config.sub_model,
    )
    try:
        with console.status("[dim]starting sandbox...[/dim]", spinner="dots"):
            runtime.prewarm()
        asyncio.run(
            TerminalFractalApp(
                runtime,
                console=console,
                verbose_iterations=display_verbose,
                banner=FRACTAL_BANNER,
            ).run()
        )
    finally:
        try:
            with console.status(
                "[dim]shutting down sandbox... press Ctrl-C again to force exit without cleaning up the sandbox[/dim]",
                spinner="dots",
            ):
                runtime.close()
        except KeyboardInterrupt:
            console.print(
                "sandbox shutdown interrupted; a sandbox may still be running. "
                "Run `sbx ls` and `sbx rm --force <name>` to clean it up.",
                style="yellow",
            )
            return 130
    return 0


def run_non_interactive(
    args: argparse.Namespace,
    *,
    stdin: TextIO | None = None,
    stdout: TextIO | None = None,
    stderr: TextIO | None = None,
) -> int:
    from rich.console import Console

    from .runtime import FractalRuntime
    from .tui.app import render_iteration_event_log, render_trace_summary

    stdin = stdin or sys.stdin
    stdout = stdout or sys.stdout
    stderr = stderr or sys.stderr

    try:
        stdin_text = read_non_interactive_stdin(args.prompt, stdin)
        message = build_non_interactive_message(args.prompt, stdin_text)
    except ValueError as exc:
        print(f"fractal: {exc}", file=stderr)
        return 1

    lm_config = resolve_runtime_lms(
        args,
        stdin=stdin,
        stdout=stdout,
        stderr=stderr,
        auto_setup=_stdin_is_tty(stdin),
    )
    if lm_config is None:
        return 1

    display_verbose = _effective_verbose(args, lm_config)
    workspace = args.workspace.resolve()
    try:
        runtime = FractalRuntime.create(
            workspace_path=workspace,
            included_paths=args.include,
            lm=lm_config.lm,
            sub_lm=lm_config.sub_lm,
            max_iterations=_effective_max_iterations(args, lm_config),
            verbose=False,
            debug=args.debug,
            session_id=args.resume,
            provider_selection=lm_config.provider_selection,
            sub_lm_follows_main=lm_config.sub_lm_follows_main,
            sub_model=lm_config.sub_model,
        )
    except Exception as exc:
        print(f"fractal: {exc}", file=stderr)
        return 1

    if not args.quiet:
        _print_startup_banner(stderr)
        print(f"fractal: workspace {runtime.workspace_path}", file=stderr)
        print(f"fractal: session {runtime.session_id}", file=stderr)
        print("fractal: running RLM...", file=stderr)

    def print_runtime_event(event: Any) -> None:
        if not args.quiet:
            print(f"fractal: {event.message}", file=stderr)

    trace_console = Console(file=stderr, force_terminal=False, color_system=None)
    live_iteration_events_seen = 0

    def print_iteration_event(event: object) -> None:
        nonlocal live_iteration_events_seen
        if args.quiet or not display_verbose:
            return
        live_iteration_events_seen += 1
        trace_console.print(render_iteration_event_log(event, verbose=True))

    try:
        result = asyncio.run(
            runtime.submit(
                message,
                on_runtime_event=print_runtime_event if not args.quiet else None,
                on_iteration_event=(
                    print_iteration_event
                    if display_verbose and not args.quiet
                    else None
                ),
            )
        )
    except KeyboardInterrupt:
        print("fractal: interrupted", file=stderr)
        return 130
    except Exception as exc:
        print(f"fractal: failed: {exc}", file=stderr)
        return 1

    if (
        display_verbose
        and not args.quiet
        and live_iteration_events_seen == 0
        and result.trace is not None
    ):
        trace_console.print(render_trace_summary(result.trace, verbose=True))

    if result.response:
        stdout.write(result.response)
        if not result.response.endswith("\n"):
            stdout.write("\n")

    if result.changed_files and not args.quiet:
        print(
            "fractal: changed files " + ", ".join(result.changed_files),
            file=stderr,
        )

    if not args.quiet:
        from .session import turn_usage_from_trace

        usage = turn_usage_from_trace(result.trace)
        if usage is not None and (usage.input_tokens or usage.output_tokens):
            print(
                f"fractal: usage {usage.input_tokens:,} in / "
                f"{usage.output_tokens:,} out tokens, ${usage.cost:.4f}",
                file=stderr,
            )

    if result.trace is not None and result.trace.status == "max_iterations":
        print("fractal: max iterations reached before completion", file=stderr)
        return MAX_ITERATIONS_EXIT_CODE

    if not args.quiet:
        print("fractal: complete", file=stderr)
    return 0


def read_non_interactive_stdin(prompt: str, stdin: TextIO) -> str | None:
    if prompt == "-":
        stdin_text = _read_limited_stdin(stdin)
        if not stdin_text:
            raise ValueError("-p - requires stdin input")
        return stdin_text

    if _stdin_is_tty(stdin):
        return None
    stdin_text = _read_limited_stdin(stdin)
    return stdin_text or None


def build_non_interactive_message(prompt: str, stdin_text: str | None) -> str:
    if prompt == "-":
        return stdin_text or ""
    if stdin_text is None:
        return prompt
    return (
        f"{prompt}\n\n<Fractal stdin context>\n{stdin_text}\n</Fractal stdin context>"
    )


def _read_limited_stdin(stdin: TextIO) -> str:
    chunks: list[str] = []
    total_bytes = 0
    while True:
        chunk = stdin.read(8192)
        if chunk == "":
            break
        total_bytes += len(chunk.encode("utf-8"))
        if total_bytes > MAX_STDIN_BYTES:
            raise ValueError(
                f"stdin input exceeds {MAX_STDIN_BYTES // (1024 * 1024)} MiB limit"
            )
        chunks.append(chunk)
    return "".join(chunks)


def _stdin_is_tty(stdin: TextIO) -> bool:
    try:
        return stdin.isatty()
    except AttributeError:
        return False


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command == "config":
        return run_config_command(args)
    if args.prompt is not None:
        return run_non_interactive(args)
    return run_tui(args)


if __name__ == "__main__":
    raise SystemExit(main())

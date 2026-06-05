from __future__ import annotations

import argparse
import asyncio
from dataclasses import dataclass
from pathlib import Path
import sys
from typing import Any, TextIO


MAX_STDIN_BYTES = 10 * 1024 * 1024
MAX_ITERATIONS_EXIT_CODE = 2


@dataclass(frozen=True)
class RuntimeLMConfig:
    lm: Any
    sub_lm: Any


class SetupInputError(ValueError):
    """Raised when interactive setup cannot collect a required answer."""


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
    parser.add_argument("--max-iterations", type=int, default=30)
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
    config_subparsers.add_parser(
        "status",
        help="validate configured provider, model, and auth availability",
    )
    config_subparsers.add_parser(
        "setup",
        help="run interactive global provider/model/auth setup",
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
    display_verbose = bool(getattr(args, "verbose", False))
    runtime = FractalRuntime.create(
        workspace_path=workspace,
        included_paths=args.include,
        lm=lm_config.lm,
        sub_lm=lm_config.sub_lm,
        max_iterations=args.max_iterations,
        verbose=False,
        debug=args.debug,
        session_id=args.resume,
    )
    try:
        with console.status("[dim]starting sandbox...[/dim]", spinner="dots"):
            runtime.prewarm()
        asyncio.run(
            TerminalFractalApp(
                runtime,
                console=console,
                verbose_iterations=display_verbose,
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
    display_verbose = bool(getattr(args, "verbose", False))

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

    workspace = args.workspace.resolve()
    try:
        runtime = FractalRuntime.create(
            workspace_path=workspace,
            lm=lm_config.lm,
            sub_lm=lm_config.sub_lm,
            max_iterations=args.max_iterations,
            verbose=False,
            debug=args.debug,
            session_id=args.resume,
        )
    except Exception as exc:
        print(f"fractal: {exc}", file=stderr)
        return 1

    if not args.quiet:
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

    if result.trace is not None and result.trace.status == "max_iterations":
        print("fractal: max iterations reached before completion", file=stderr)
        return MAX_ITERATIONS_EXIT_CODE

    if not args.quiet:
        print("fractal: complete", file=stderr)
    return 0


def run_config_command(
    args: argparse.Namespace,
    *,
    stdin: TextIO | None = None,
    stdout: TextIO | None = None,
    stderr: TextIO | None = None,
) -> int:
    stdin = stdin or sys.stdin
    stdout = stdout or sys.stdout
    stderr = stderr or sys.stderr

    if args.config_command == "show":
        return config_show(stdout=stdout, stderr=stderr)
    if args.config_command == "status":
        return config_status(stdout=stdout, stderr=stderr)
    if args.config_command == "setup":
        return config_setup(stdin=stdin, stdout=stdout, stderr=stderr)
    print(f"fractal config: unknown command {args.config_command!r}", file=stderr)
    return 1


def config_show(*, stdout: TextIO, stderr: TextIO) -> int:
    from .config import FractalConfigError, load_config, render_config

    try:
        result = load_config()
    except FractalConfigError as exc:
        print(f"fractal config: {exc}", file=stderr)
        return 1
    if result.config is None:
        print(f"fractal config: no config found at {result.path}", file=stderr)
        print("Run `fractal config setup`.", file=stderr)
        return 1
    print(render_config(result.config, path=result.path), file=stdout)
    return 0


def config_status(*, stdout: TextIO, stderr: TextIO) -> int:
    from .config import FractalConfigError, load_config, render_config
    from .providers import ProviderError, validate_provider_selection

    try:
        result = load_config()
    except FractalConfigError as exc:
        print("Fractal config status: invalid", file=stdout)
        print(f"fractal config: {exc}", file=stderr)
        print("Run `fractal config setup` after fixing the config.", file=stderr)
        return 1
    if result.config is None:
        print("Fractal config status: not configured", file=stdout)
        print(f"path: {result.path}", file=stdout)
        print("Run `fractal config setup`.", file=stdout)
        return 1

    selection = _selection_from_config(result.config, path=result.path)
    try:
        validate_provider_selection(selection)
    except ProviderError as exc:
        print("Fractal config status: invalid", file=stdout)
        print(render_config(result.config, path=result.path), file=stdout)
        print(f"auth/provider check failed: {exc}", file=stderr)
        print(
            "Run `fractal config setup` or fix the configured auth source.",
            file=stderr,
        )
        return 1

    print("Fractal config status: ok", file=stdout)
    print(render_config(result.config, path=result.path), file=stdout)
    return 0


def config_setup(*, stdin: TextIO, stdout: TextIO, stderr: TextIO) -> int:
    from .config import FractalConfigError, write_config
    from .providers import ProviderError, validate_provider_selection

    try:
        config = _prompt_for_config(stdin=stdin, stdout=stdout)
        selection = _selection_from_config(config)
        validate_provider_selection(selection)
        path = write_config(config)
    except (FractalConfigError, ProviderError, SetupInputError, ValueError) as exc:
        print(f"fractal config setup: {exc}", file=stderr)
        print(
            "No config was written. Fix the issue, then run "
            "`fractal config setup` again.",
            file=stderr,
        )
        return 1

    print(f"Fractal config written to {path}", file=stdout)
    return 0


def resolve_runtime_lms(
    args: argparse.Namespace,
    *,
    stdin: TextIO,
    stdout: TextIO,
    stderr: TextIO,
    auto_setup: bool,
) -> RuntimeLMConfig | None:
    from .config import FractalConfigError, load_config
    from .providers import ProviderError, build_lm

    if args.lm is not None:
        return RuntimeLMConfig(lm=args.lm, sub_lm=args.sub_lm)

    try:
        result = load_config()
    except FractalConfigError as exc:
        print(f"fractal config: {exc}", file=stderr)
        print("Run `fractal config setup` after fixing the config.", file=stderr)
        return None

    if result.config is None:
        if not auto_setup:
            print(f"fractal: no global config found at {result.path}", file=stderr)
            print("Run `fractal config setup` or pass `--lm` explicitly.", file=stderr)
            return None
        print("fractal: no global config found; starting setup.", file=stderr)
        if config_setup(stdin=stdin, stdout=stdout, stderr=stderr) != 0:
            return None
        try:
            result = load_config()
        except FractalConfigError as exc:
            print(f"fractal config: {exc}", file=stderr)
            return None
        if result.config is None:
            print(
                "fractal: setup completed but config could not be loaded", file=stderr
            )
            return None

    selection = _selection_from_config(result.config, path=result.path)
    try:
        lm = build_lm(selection)
    except ProviderError as exc:
        print(f"fractal config: {exc}", file=stderr)
        print(
            "Run `fractal config status` for details or "
            "`fractal config setup` to repair setup.",
            file=stderr,
        )
        return None
    sub_lm = args.sub_lm if args.sub_lm is not None else lm
    return RuntimeLMConfig(lm=lm, sub_lm=sub_lm)


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


def _prompt_for_config(*, stdin: TextIO, stdout: TextIO) -> Any:
    from .config import FractalConfig
    from .providers import list_providers

    providers = list_providers()
    print("Fractal global config setup", file=stdout)
    print("Choose a provider:", file=stdout)
    for index, provider in enumerate(providers, start=1):
        examples = ", ".join(provider.model_examples)
        print(
            f"{index}. {provider.display_name} ({provider.id})",
            file=stdout,
        )
        print(f"   default model: {provider.default_model}", file=stdout)
        if examples:
            print(f"   model examples: {examples}", file=stdout)

    provider = _prompt_provider(stdin=stdin, stdout=stdout, providers=providers)
    model = _prompt(
        stdin=stdin,
        stdout=stdout,
        label=f"Model for {provider.display_name}",
        default=provider.default_model,
    )
    provider_config = _prompt_provider_settings(
        stdin=stdin,
        stdout=stdout,
        provider_id=provider.id,
    )
    return FractalConfig(
        active_provider=provider.id,
        active_model=model,
        providers={provider.id: provider_config},
    )


def _prompt_provider(*, stdin: TextIO, stdout: TextIO, providers: list[Any]) -> Any:
    from .providers import get_provider

    provider_by_index = {
        str(index): provider for index, provider in enumerate(providers, start=1)
    }
    provider_ids = {provider.id for provider in providers}
    while True:
        answer = _prompt(
            stdin=stdin,
            stdout=stdout,
            label="Provider number or id",
            default=providers[0].id,
        )
        if answer in provider_by_index:
            return provider_by_index[answer]
        if answer in provider_ids:
            return get_provider(answer)
        print(
            f"Unknown provider {answer!r}. Choose one of the listed providers.",
            file=stdout,
        )


def _prompt_provider_settings(
    *,
    stdin: TextIO,
    stdout: TextIO,
    provider_id: str,
) -> Any:
    from .config import ProviderConfig
    from .providers import (
        CUSTOM_OPENAI_COMPATIBLE,
        OPENAI_CODEX,
        get_provider,
    )

    provider = get_provider(provider_id)
    if provider.id == OPENAI_CODEX:
        print("OpenAI Codex uses the official Codex CLI auth store.", file=stdout)
        print(
            "If needed, run `codex login --device-auth` before continuing.",
            file=stdout,
        )
        return ProviderConfig(auth_source="codex-cli")

    if provider.id == CUSTOM_OPENAI_COMPATIBLE:
        base_url = _prompt_required(
            stdin=stdin,
            stdout=stdout,
            label="OpenAI-compatible base URL",
        )
        api_key_env = _prompt(
            stdin=stdin,
            stdout=stdout,
            label="API key environment variable",
            default="CUSTOM_OPENAI_API_KEY",
        )
        return ProviderConfig(
            auth_source="env",
            api_key_env=api_key_env,
            base_url=base_url,
        )

    if provider.auth_type == "api_key_env":
        api_key_env = _prompt(
            stdin=stdin,
            stdout=stdout,
            label="API key environment variable",
            default=provider.default_api_key_env,
        )
        return ProviderConfig(auth_source="env", api_key_env=api_key_env)

    raise SetupInputError(f"provider {provider.id!r} is not supported by setup")


def _prompt(
    *,
    stdin: TextIO,
    stdout: TextIO,
    label: str,
    default: str | None = None,
) -> str:
    suffix = f" [{default}]" if default else ""
    print(f"{label}{suffix}: ", end="", flush=True, file=stdout)
    answer = stdin.readline()
    if answer == "":
        raise SetupInputError("setup requires interactive input")
    value = answer.strip()
    if value:
        return value
    if default:
        return default
    return ""


def _prompt_required(*, stdin: TextIO, stdout: TextIO, label: str) -> str:
    value = _prompt(stdin=stdin, stdout=stdout, label=label)
    if not value:
        raise SetupInputError(f"{label} is required")
    return value


def _selection_from_config(config: Any, *, path: str | Path | None = None) -> Any:
    from .config import resolve_effective_config
    from .providers import ProviderSelection

    effective = resolve_effective_config(config, path=path)
    provider_config = effective.provider_config
    return ProviderSelection(
        provider=effective.provider,
        model=effective.model,
        api_key_env=provider_config.api_key_env,
        base_url=provider_config.base_url,
        auth_source=provider_config.auth_source,
    )


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

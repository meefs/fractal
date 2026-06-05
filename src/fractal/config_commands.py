from __future__ import annotations

import sys
from typing import Any, TextIO

from .onboarding import SetupInputError, prompt_for_config
from .runtime_lms import selection_from_config


def run_config_command(
    args: Any,
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

    selection = selection_from_config(result.config, path=result.path)
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
        config = prompt_for_config(stdin=stdin, stdout=stdout)
        selection = selection_from_config(config)
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

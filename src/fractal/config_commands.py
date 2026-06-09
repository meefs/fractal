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

    offline = bool(getattr(args, "offline", False))
    if args.config_command == "show":
        return config_show(stdout=stdout, stderr=stderr)
    if args.config_command == "status":
        return config_status(stdout=stdout, stderr=stderr, offline=offline)
    if args.config_command == "setup":
        return config_setup(stdin=stdin, stdout=stdout, stderr=stderr, offline=offline)
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


def config_status(*, stdout: TextIO, stderr: TextIO, offline: bool = False) -> int:
    from .config import FractalConfigError, load_config, render_config
    from .connectivity import ProviderConnectivityError, check_provider_connectivity
    from .providers import ProviderError, check_provider_readiness

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
        check_provider_readiness(selection)
    except ProviderError as exc:
        print("Fractal config status: invalid", file=stdout)
        print(render_config(result.config, path=result.path), file=stdout)
        print(f"auth/provider check failed: {exc}", file=stderr)
        print(
            "Run `fractal config setup` or fix the configured auth source.",
            file=stderr,
        )
        return 1

    connectivity_note = "connectivity: skipped (--offline)"
    if not offline:
        try:
            checked = check_provider_connectivity(selection)
        except ProviderConnectivityError as exc:
            print("Fractal config status: unreachable", file=stdout)
            print(render_config(result.config, path=result.path), file=stdout)
            print(f"connectivity check failed: {exc}", file=stderr)
            print(
                "Fix the credential or network, or re-run with `--offline`.",
                file=stderr,
            )
            return 1
        connectivity_note = (
            "connectivity: verified"
            if checked
            else "connectivity: not checked for this provider"
        )

    print("Fractal config status: ok", file=stdout)
    print(render_config(result.config, path=result.path), file=stdout)
    print(connectivity_note, file=stdout)
    return 0


def config_setup(
    *,
    stdin: TextIO,
    stdout: TextIO,
    stderr: TextIO,
    offline: bool = False,
) -> int:
    from .config import FractalConfigError, write_config
    from .connectivity import ProviderConnectivityError, check_provider_connectivity
    from .providers import (
        MissingProviderCredentialError,
        ProviderError,
        check_provider_readiness,
    )

    # A missing credential should not discard the user's setup answers: the
    # config holds only non-secret references, so write it and tell the user
    # exactly what to provide. Any other failure means the setup itself is
    # wrong and nothing should be written.
    credential_warning: str | None = None
    try:
        config = prompt_for_config(stdin=stdin, stdout=stdout)
        selection = selection_from_config(config)
        try:
            check_provider_readiness(selection)
        except MissingProviderCredentialError as exc:
            credential_warning = str(exc)
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
    if credential_warning is not None:
        print(f"warning: {credential_warning}", file=stderr)
        print(
            "Provide the credential, then run `fractal config status` to verify.",
            file=stderr,
        )
        return 0

    # Verify the credential actually works, not just that it exists. Network
    # problems should not undo a finished setup, so failures only warn.
    if not offline:
        try:
            if check_provider_connectivity(selection):
                print("Provider connectivity verified.", file=stdout)
        except ProviderConnectivityError as exc:
            print(f"warning: {exc}", file=stderr)
            print(
                "The config was written. Fix the credential or network, then "
                "run `fractal config status` to verify.",
                file=stderr,
            )
    return 0

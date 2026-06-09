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
    workspace = getattr(args, "workspace", None)
    if args.config_command == "show":
        return config_show(stdout=stdout, stderr=stderr, workspace=workspace)
    if args.config_command == "status":
        return config_status(
            stdout=stdout,
            stderr=stderr,
            offline=offline,
            workspace=workspace,
        )
    if args.config_command == "setup":
        return config_setup(stdin=stdin, stdout=stdout, stderr=stderr, offline=offline)
    if args.config_command == "get":
        return config_get(args.key, stdout=stdout, stderr=stderr, workspace=workspace)
    if args.config_command == "set":
        return config_set(
            args.key,
            args.value,
            stdout=stdout,
            stderr=stderr,
            project=bool(getattr(args, "project", False)),
            workspace=workspace,
        )
    if args.config_command == "unset":
        return config_unset(
            args.key,
            stdout=stdout,
            stderr=stderr,
            project=bool(getattr(args, "project", False)),
            workspace=workspace,
        )
    print(f"fractal config: unknown command {args.config_command!r}", file=stderr)
    return 1


def config_show(
    *,
    stdout: TextIO,
    stderr: TextIO,
    workspace: Any | None = None,
) -> int:
    from .config import FractalConfigError, load_layered_config, render_config

    try:
        result = load_layered_config(workspace=workspace)
    except FractalConfigError as exc:
        print(f"fractal config: {exc}", file=stderr)
        return 1
    if result.config is None:
        print(f"fractal config: no config found at {result.path}", file=stderr)
        print("Run `fractal config setup`.", file=stderr)
        return 1
    print(render_config(result.config, path=result.path), file=stdout)
    _print_layer_notes(result, stdout=stdout)
    return 0


def _print_layer_notes(result: Any, *, stdout: TextIO) -> None:
    if result.project_path is not None:
        print(f"project overrides: {result.project_path}", file=stdout)
    if result.env_overrides:
        print("env overrides: " + ", ".join(result.env_overrides), file=stdout)


def config_status(
    *,
    stdout: TextIO,
    stderr: TextIO,
    offline: bool = False,
    workspace: Any | None = None,
) -> int:
    from .config import FractalConfigError, load_layered_config, render_config
    from .connectivity import ProviderConnectivityError, check_provider_connectivity
    from .providers import ProviderError, check_provider_readiness

    try:
        result = load_layered_config(workspace=workspace)
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
    _print_layer_notes(result, stdout=stdout)
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


def config_get(
    key: str,
    *,
    stdout: TextIO,
    stderr: TextIO,
    workspace: Any | None = None,
) -> int:
    from .config import FractalConfigError, load_layered_config

    try:
        result = load_layered_config(workspace=workspace)
    except FractalConfigError as exc:
        print(f"fractal config: {exc}", file=stderr)
        return 1
    if result.config is None:
        print("fractal config: not configured; run `fractal config setup`.", file=stderr)
        return 1

    data = result.config.model_dump(mode="python", exclude_none=True)
    found, value = _walk_config_path(data, key)
    if not found:
        print(f"fractal config: {key} is not set", file=stderr)
        return 1
    print(_format_config_value(value), file=stdout)
    return 0


def config_set(
    key: str,
    raw_value: str,
    *,
    stdout: TextIO,
    stderr: TextIO,
    project: bool = False,
    workspace: Any | None = None,
) -> int:
    from pathlib import Path

    from .config import (
        FractalConfig,
        FractalConfigError,
        ProjectFractalConfig,
        load_config,
        load_project_config,
        write_config,
        write_project_config,
    )

    value = _parse_config_value(raw_value)
    try:
        if project:
            target_workspace = Path(workspace) if workspace is not None else Path.cwd()
            current = load_project_config(target_workspace) or ProjectFractalConfig()
            data = current.model_dump(mode="python", exclude_none=True)
            _set_config_path(data, key, value)
            updated = ProjectFractalConfig.model_validate(data)
            path = write_project_config(updated, target_workspace)
        else:
            result = load_config()
            if result.config is None:
                print(
                    "fractal config: not configured; run `fractal config setup` "
                    "first or use `--project`.",
                    file=stderr,
                )
                return 1
            data = result.config.model_dump(mode="python", exclude_none=True)
            _set_config_path(data, key, value)
            updated = FractalConfig.model_validate(data)
            path = write_config(updated, path=result.path)
    except FractalConfigError as exc:
        print(f"fractal config: {exc}", file=stderr)
        return 1
    except ValueError as exc:
        print(f"fractal config: cannot set {key}: {exc}", file=stderr)
        return 1

    print(f"set {key} = {_format_config_value(value)} in {path}", file=stdout)
    return 0


def config_unset(
    key: str,
    *,
    stdout: TextIO,
    stderr: TextIO,
    project: bool = False,
    workspace: Any | None = None,
) -> int:
    from pathlib import Path

    from .config import (
        FractalConfig,
        FractalConfigError,
        ProjectFractalConfig,
        load_config,
        load_project_config,
        write_config,
        write_project_config,
    )

    try:
        if project:
            target_workspace = Path(workspace) if workspace is not None else Path.cwd()
            current = load_project_config(target_workspace)
            if current is None:
                print("fractal config: no project config found", file=stderr)
                return 1
            data = current.model_dump(mode="python", exclude_none=True)
            if not _unset_config_path(data, key):
                print(f"fractal config: {key} is not set", file=stderr)
                return 1
            updated = ProjectFractalConfig.model_validate(data)
            path = write_project_config(updated, target_workspace)
        else:
            result = load_config()
            if result.config is None:
                print("fractal config: not configured", file=stderr)
                return 1
            data = result.config.model_dump(mode="python", exclude_none=True)
            if not _unset_config_path(data, key):
                print(f"fractal config: {key} is not set", file=stderr)
                return 1
            updated = FractalConfig.model_validate(data)
            path = write_config(updated, path=result.path)
    except FractalConfigError as exc:
        print(f"fractal config: {exc}", file=stderr)
        return 1
    except ValueError as exc:
        print(f"fractal config: cannot unset {key}: {exc}", file=stderr)
        return 1

    print(f"unset {key} in {path}", file=stdout)
    return 0


def _parse_config_value(raw: str) -> Any:
    import tomllib

    try:
        return tomllib.loads(f"value = {raw}")["value"]
    except tomllib.TOMLDecodeError:
        return raw


def _format_config_value(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, dict):
        import tomli_w

        return tomli_w.dumps(value).strip()
    return str(value)


def _walk_config_path(data: Any, key: str) -> tuple[bool, Any]:
    node = data
    for part in key.split("."):
        if not isinstance(node, dict) or part not in node:
            return False, None
        node = node[part]
    return True, node


def _set_config_path(data: dict[str, Any], key: str, value: Any) -> None:
    parts = key.split(".")
    node: Any = data
    for part in parts[:-1]:
        if part not in node or not isinstance(node[part], dict):
            node[part] = {}
        node = node[part]
    if not isinstance(node, dict):
        raise ValueError(f"{key} does not address a config table")
    node[parts[-1]] = value


def _unset_config_path(data: dict[str, Any], key: str) -> bool:
    parts = key.split(".")
    node: Any = data
    for part in parts[:-1]:
        if not isinstance(node, dict) or part not in node:
            return False
        node = node[part]
    if not isinstance(node, dict) or parts[-1] not in node:
        return False
    del node[parts[-1]]
    return True

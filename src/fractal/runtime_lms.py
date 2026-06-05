from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, TextIO


@dataclass(frozen=True)
class RuntimeLMConfig:
    lm: Any
    sub_lm: Any


def resolve_runtime_lms(
    args: Any,
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
        from .config_commands import config_setup

        if config_setup(stdin=stdin, stdout=stdout, stderr=stderr) != 0:
            return None
        try:
            result = load_config()
        except FractalConfigError as exc:
            print(f"fractal config: {exc}", file=stderr)
            return None
        if result.config is None:
            print("fractal: setup completed but config could not be loaded", file=stderr)
            return None

    selection = selection_from_config(result.config, path=result.path)
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


def selection_from_config(config: Any, *, path: str | Path | None = None) -> Any:
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

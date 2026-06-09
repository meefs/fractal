from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from typing import Protocol, TextIO

from .config import DefaultsConfig, FractalConfig
from .lm_types import RuntimeLM
from .providers import ProviderSelection


@dataclass(frozen=True)
class RuntimeLMConfig:
    lm: RuntimeLM
    sub_lm: RuntimeLM | None
    provider_selection: ProviderSelection | None = None
    sub_lm_follows_main: bool = True
    defaults: DefaultsConfig | None = None


class RuntimeLMArgs(Protocol):
    lm: RuntimeLM | None
    sub_lm: RuntimeLM | None
    workspace: Path | None


def resolve_runtime_lms(
    args: RuntimeLMArgs,
    *,
    stdin: TextIO,
    stdout: TextIO,
    stderr: TextIO,
    auto_setup: bool,
) -> RuntimeLMConfig | None:
    from .config import FractalConfigError, load_layered_config
    from .providers import ProviderError, build_lm

    if args.lm is not None:
        return RuntimeLMConfig(
            lm=args.lm,
            sub_lm=args.sub_lm,
            provider_selection=None,
            sub_lm_follows_main=args.sub_lm is None,
        )

    workspace = getattr(args, "workspace", None)
    try:
        result = load_layered_config(workspace=workspace)
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
            result = load_layered_config(workspace=workspace)
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
    sub_lm = args.sub_lm
    sub_lm_follows_main = args.sub_lm is None
    if sub_lm is None and result.config.active_sub_model is not None:
        try:
            sub_lm = build_lm(replace(selection, model=result.config.active_sub_model))
        except ProviderError as exc:
            print(f"fractal config: sub model: {exc}", file=stderr)
            print(
                "Fix `active_sub_model` in the config or remove it.",
                file=stderr,
            )
            return None
        sub_lm_follows_main = False
    if sub_lm is None:
        sub_lm = lm
    return RuntimeLMConfig(
        lm=lm,
        sub_lm=sub_lm,
        provider_selection=selection,
        sub_lm_follows_main=sub_lm_follows_main,
        defaults=result.config.defaults,
    )


def selection_from_config(
    config: FractalConfig,
    *,
    path: str | Path | None = None,
) -> ProviderSelection:
    from .config import resolve_effective_config

    effective = resolve_effective_config(config, path=path)
    provider_config = effective.provider_config
    return ProviderSelection(
        provider=effective.provider,
        model=effective.model,
        api_key_env=provider_config.api_key_env,
        base_url=provider_config.base_url,
        auth_source=provider_config.auth_source,
    )

from __future__ import annotations

from typing import Any, TextIO


class SetupInputError(ValueError):
    """Raised when interactive setup cannot collect a required answer."""


def prompt_for_config(*, stdin: TextIO, stdout: TextIO) -> Any:
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

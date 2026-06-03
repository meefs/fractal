from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any
from urllib.parse import urlparse


OPENAI_CODEX = "openai-codex"
OPENAI_API = "openai-api"
ANTHROPIC = "anthropic"
OPENROUTER = "openrouter"
CUSTOM_OPENAI_COMPATIBLE = "custom-openai-compatible"


class ProviderError(ValueError):
    """Base class for provider registry and LM factory errors."""


class UnknownProviderError(ProviderError):
    """Raised when config references a provider Fractal does not know."""


class ProviderConfigError(ProviderError):
    """Raised when a provider selection is incomplete or inconsistent."""


class UnsupportedProviderModelError(ProviderError):
    """Raised when a provider cannot use the requested model."""


class MissingProviderCredentialError(ProviderConfigError):
    """Raised when a provider's configured credential source is unavailable."""


@dataclass(frozen=True)
class ProviderDefinition:
    id: str
    display_name: str
    auth_type: str
    default_model: str
    model_examples: tuple[str, ...] = ()
    default_api_key_env: str | None = None
    model_prefix: str | None = None
    supports_base_url: bool = False


@dataclass(frozen=True)
class ProviderSelection:
    provider: str
    model: str | None = None
    api_key_env: str | None = None
    base_url: str | None = None
    auth_source: str | None = None


_PROVIDERS: dict[str, ProviderDefinition] = {
    OPENAI_CODEX: ProviderDefinition(
        id=OPENAI_CODEX,
        display_name="OpenAI Codex",
        auth_type="codex_cli",
        default_model="gpt-5.3-codex",
        model_examples=(
            "gpt-5.3-codex",
            "gpt-5.4-mini",
            "gpt-5.5",
        ),
    ),
    OPENAI_API: ProviderDefinition(
        id=OPENAI_API,
        display_name="OpenAI API",
        auth_type="api_key_env",
        default_model="gpt-5.5",
        model_examples=("gpt-5.5", "gpt-5.4-mini"),
        default_api_key_env="OPENAI_API_KEY",
        model_prefix="openai",
    ),
    ANTHROPIC: ProviderDefinition(
        id=ANTHROPIC,
        display_name="Anthropic",
        auth_type="api_key_env",
        default_model="claude-sonnet-4-5",
        model_examples=("claude-sonnet-4-5", "claude-haiku-4-5"),
        default_api_key_env="ANTHROPIC_API_KEY",
        model_prefix="anthropic",
    ),
    OPENROUTER: ProviderDefinition(
        id=OPENROUTER,
        display_name="OpenRouter",
        auth_type="api_key_env",
        default_model="openai/gpt-5.5",
        model_examples=("openai/gpt-5.5", "anthropic/claude-sonnet-4-5"),
        default_api_key_env="OPENROUTER_API_KEY",
        model_prefix="openrouter",
    ),
    CUSTOM_OPENAI_COMPATIBLE: ProviderDefinition(
        id=CUSTOM_OPENAI_COMPATIBLE,
        display_name="Custom OpenAI-compatible",
        auth_type="api_key_env",
        default_model="model-name",
        model_examples=("gpt-oss-120b", "qwen3-coder"),
        supports_base_url=True,
        model_prefix="openai",
    ),
}


def provider_registry() -> Mapping[str, ProviderDefinition]:
    return MappingProxyType(_PROVIDERS)


def list_providers() -> list[ProviderDefinition]:
    return list(_PROVIDERS.values())


def get_provider(provider_id: str) -> ProviderDefinition:
    try:
        return _PROVIDERS[provider_id]
    except KeyError as exc:
        raise UnknownProviderError(f"unknown provider {provider_id!r}") from exc


def resolve_lm(
    explicit_lm: Any | None,
    selection: ProviderSelection | None,
    *,
    env: Mapping[str, str] | None = None,
) -> Any:
    if explicit_lm is not None:
        return explicit_lm
    if selection is None:
        return None
    return build_lm(selection, env=env)


def build_lm(
    selection: ProviderSelection,
    *,
    env: Mapping[str, str] | None = None,
) -> Any:
    definition = get_provider(selection.provider)
    if definition.id == OPENAI_CODEX:
        return _build_codex_lm(selection, definition)
    if definition.id == CUSTOM_OPENAI_COMPATIBLE:
        return _build_custom_openai_lm(selection, definition, env=env)
    validate_provider_selection(selection, env=env)
    return _normalize_model(_selection_model(selection, definition), definition)


def validate_provider_selection(
    selection: ProviderSelection,
    *,
    env: Mapping[str, str] | None = None,
) -> None:
    definition = get_provider(selection.provider)
    _selection_model(selection, definition)
    if definition.id == CUSTOM_OPENAI_COMPATIBLE:
        _validate_custom_openai_selection(selection, env=env)
        return
    if definition.auth_type == "api_key_env":
        _require_api_key_env(selection, definition, env=env)


def _selection_model(
    selection: ProviderSelection,
    definition: ProviderDefinition,
) -> str:
    model = selection.model or definition.default_model
    if not model:
        raise ProviderConfigError(f"provider {definition.id!r} requires a model")
    return model


def _normalize_model(model: str, definition: ProviderDefinition) -> str:
    prefix = definition.model_prefix
    if prefix is None or model.startswith(f"{prefix}/"):
        return model
    return f"{prefix}/{model}"


def _api_key_env_name(
    selection: ProviderSelection,
    definition: ProviderDefinition,
) -> str:
    env_name = selection.api_key_env or definition.default_api_key_env
    if not env_name:
        raise ProviderConfigError(
            f"provider {definition.id!r} requires an API key env var name"
        )
    return env_name


def _require_api_key_env(
    selection: ProviderSelection,
    definition: ProviderDefinition,
    *,
    env: Mapping[str, str] | None,
) -> str:
    env_name = _api_key_env_name(selection, definition)
    values = os.environ if env is None else env
    if not values.get(env_name):
        raise MissingProviderCredentialError(
            f"provider {definition.id!r} requires environment variable "
            f"{env_name} to be set"
        )
    return env_name


def _build_codex_lm(selection: ProviderSelection, definition: ProviderDefinition) -> Any:
    try:
        from dspy_codex_lm import CodexLM
        from dspy_codex_lm.cli import CodexLMUnsupportedModelError, resolve_codex_model
    except ImportError as exc:
        raise ProviderConfigError(
            "openai-codex requires PredictRLM's dspy_codex_lm integration"
        ) from exc

    model = _selection_model(selection, definition)
    try:
        codex_model = resolve_codex_model(model)
    except CodexLMUnsupportedModelError as exc:
        raise UnsupportedProviderModelError(str(exc)) from exc
    return CodexLM(model=codex_model)


def _build_custom_openai_lm(
    selection: ProviderSelection,
    definition: ProviderDefinition,
    *,
    env: Mapping[str, str] | None,
) -> Any:
    base_url = _validate_custom_openai_selection(selection, env=env)
    api_key = (os.environ if env is None else env)[selection.api_key_env or ""]

    try:
        import dspy
    except ImportError as exc:
        raise ProviderConfigError("custom OpenAI-compatible provider requires DSPy") from exc

    return dspy.LM(
        model=_normalize_model(_selection_model(selection, definition), definition),
        api_base=base_url,
        api_key=api_key,
    )


def _validate_custom_openai_selection(
    selection: ProviderSelection,
    *,
    env: Mapping[str, str] | None,
) -> str:
    definition = get_provider(CUSTOM_OPENAI_COMPATIBLE)
    if not selection.model:
        raise ProviderConfigError("custom OpenAI-compatible provider requires model")
    if not selection.base_url:
        raise ProviderConfigError("custom OpenAI-compatible provider requires base_url")
    base_url = selection.base_url.strip()
    parsed = urlparse(base_url)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise ProviderConfigError(
            "custom OpenAI-compatible provider requires base_url to be an "
            "HTTP(S) URL"
        )
    if not selection.api_key_env:
        raise ProviderConfigError(
            "custom OpenAI-compatible provider requires api_key_env"
        )
    _require_api_key_env(selection, definition, env=env)
    return base_url

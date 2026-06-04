from __future__ import annotations

import os
import shutil
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any, Literal, Protocol
from urllib.parse import urlparse


OPENAI_CODEX = "openai-codex"
OPENAI_API = "openai-api"
ANTHROPIC = "anthropic"
OPENROUTER = "openrouter"
CUSTOM_OPENAI_COMPATIBLE = "custom-openai-compatible"
ProviderAuthType = Literal["api_key_env", "codex_cli"]
ProviderAuthSource = Literal["env", "codex-cli"]


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


class MissingCodexCliError(ProviderConfigError):
    """Raised when the official Codex CLI is required but unavailable."""


class MissingCodexAuthError(MissingProviderCredentialError):
    """Raised when official Codex CLI auth is missing or unusable."""


@dataclass(frozen=True)
class ProviderSelection:
    provider: str
    model: str | None = None
    api_key_env: str | None = None
    base_url: str | None = None
    auth_source: str | None = None


class ProviderBehavior(Protocol):
    def validate_selection(
        self,
        selection: ProviderSelection,
        definition: "ProviderDefinition",
        *,
        env: Mapping[str, str] | None,
    ) -> None: ...

    def build_lm(
        self,
        selection: ProviderSelection,
        definition: "ProviderDefinition",
        *,
        env: Mapping[str, str] | None,
    ) -> Any: ...


@dataclass(frozen=True)
class ProviderDefinition:
    id: str
    display_name: str
    auth_type: ProviderAuthType
    auth_source: ProviderAuthSource
    default_model: str
    behavior: ProviderBehavior
    model_examples: tuple[str, ...] = ()
    default_api_key_env: str | None = None
    model_prefix: str | None = None
    supports_base_url: bool = False
    base_url_label: str | None = None
    setup_messages: tuple[str, ...] = ()

    def validate_selection(
        self,
        selection: ProviderSelection,
        *,
        env: Mapping[str, str] | None = None,
    ) -> None:
        self.behavior.validate_selection(selection, self, env=env)

    def build_lm(
        self,
        selection: ProviderSelection,
        *,
        env: Mapping[str, str] | None = None,
    ) -> Any:
        return self.behavior.build_lm(selection, self, env=env)


class ApiKeyStringLMBehavior:
    def validate_selection(
        self,
        selection: ProviderSelection,
        definition: ProviderDefinition,
        *,
        env: Mapping[str, str] | None,
    ) -> None:
        _selection_model(selection, definition)
        _require_api_key_env(selection, definition, env=env)

    def build_lm(
        self,
        selection: ProviderSelection,
        definition: ProviderDefinition,
        *,
        env: Mapping[str, str] | None,
    ) -> str:
        self.validate_selection(selection, definition, env=env)
        return _normalize_model(_selection_model(selection, definition), definition)


class CodexCliLMBehavior:
    def validate_selection(
        self,
        selection: ProviderSelection,
        definition: ProviderDefinition,
        *,
        env: Mapping[str, str] | None,
    ) -> None:
        _selection_model(selection, definition)
        _resolve_codex_model(selection, definition)
        _codex_cli_auth_path(selection)

    def build_lm(
        self,
        selection: ProviderSelection,
        definition: ProviderDefinition,
        *,
        env: Mapping[str, str] | None,
    ) -> Any:
        codex_model = _resolve_codex_model(selection, definition)
        auth_path = _codex_cli_auth_path(selection)
        try:
            from dspy_codex_lm import CodexLM
        except ImportError as exc:
            raise ProviderConfigError(
                "openai-codex requires PredictRLM's dspy_codex_lm integration"
            ) from exc

        return CodexLM(model=codex_model, auth_path=auth_path)


class CustomOpenAICompatibleBehavior:
    def validate_selection(
        self,
        selection: ProviderSelection,
        definition: ProviderDefinition,
        *,
        env: Mapping[str, str] | None,
    ) -> None:
        _validate_custom_openai_selection(selection, definition, env=env)

    def build_lm(
        self,
        selection: ProviderSelection,
        definition: ProviderDefinition,
        *,
        env: Mapping[str, str] | None,
    ) -> Any:
        base_url = _validate_custom_openai_selection(selection, definition, env=env)
        api_key = (os.environ if env is None else env)[selection.api_key_env or ""]

        try:
            import dspy
        except ImportError as exc:
            raise ProviderConfigError(
                "custom OpenAI-compatible provider requires DSPy"
            ) from exc

        return dspy.LM(
            model=_normalize_model(_selection_model(selection, definition), definition),
            api_base=base_url,
            api_key=api_key,
        )


_API_KEY_STRING_LM = ApiKeyStringLMBehavior()
_CODEX_CLI_LM = CodexCliLMBehavior()
_CUSTOM_OPENAI_COMPATIBLE_LM = CustomOpenAICompatibleBehavior()


_PROVIDERS: dict[str, ProviderDefinition] = {
    OPENAI_CODEX: ProviderDefinition(
        id=OPENAI_CODEX,
        display_name="OpenAI Codex",
        auth_type="codex_cli",
        auth_source="codex-cli",
        default_model="gpt-5.3-codex",
        behavior=_CODEX_CLI_LM,
        model_examples=(
            "gpt-5.3-codex",
            "gpt-5.4-mini",
            "gpt-5.5",
        ),
        setup_messages=(
            "OpenAI Codex uses the official Codex CLI auth store.",
            "If needed, run `codex login --device-auth` before continuing.",
        ),
    ),
    OPENAI_API: ProviderDefinition(
        id=OPENAI_API,
        display_name="OpenAI API",
        auth_type="api_key_env",
        auth_source="env",
        default_model="gpt-5.5",
        behavior=_API_KEY_STRING_LM,
        model_examples=("gpt-5.5", "gpt-5.4-mini"),
        default_api_key_env="OPENAI_API_KEY",
        model_prefix="openai",
    ),
    ANTHROPIC: ProviderDefinition(
        id=ANTHROPIC,
        display_name="Anthropic",
        auth_type="api_key_env",
        auth_source="env",
        default_model="claude-sonnet-4-5",
        behavior=_API_KEY_STRING_LM,
        model_examples=("claude-sonnet-4-5", "claude-haiku-4-5"),
        default_api_key_env="ANTHROPIC_API_KEY",
        model_prefix="anthropic",
    ),
    OPENROUTER: ProviderDefinition(
        id=OPENROUTER,
        display_name="OpenRouter",
        auth_type="api_key_env",
        auth_source="env",
        default_model="openai/gpt-5.5",
        behavior=_API_KEY_STRING_LM,
        model_examples=("openai/gpt-5.5", "anthropic/claude-sonnet-4-5"),
        default_api_key_env="OPENROUTER_API_KEY",
        model_prefix="openrouter",
    ),
    CUSTOM_OPENAI_COMPATIBLE: ProviderDefinition(
        id=CUSTOM_OPENAI_COMPATIBLE,
        display_name="Custom OpenAI-compatible",
        auth_type="api_key_env",
        auth_source="env",
        default_model="model-name",
        behavior=_CUSTOM_OPENAI_COMPATIBLE_LM,
        model_examples=("gpt-oss-120b", "qwen3-coder"),
        default_api_key_env="CUSTOM_OPENAI_API_KEY",
        supports_base_url=True,
        base_url_label="OpenAI-compatible base URL",
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
    return get_provider(selection.provider).build_lm(selection, env=env)


def validate_provider_selection(
    selection: ProviderSelection,
    *,
    env: Mapping[str, str] | None = None,
) -> None:
    get_provider(selection.provider).validate_selection(selection, env=env)


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


def _resolve_codex_model(
    selection: ProviderSelection,
    definition: ProviderDefinition,
) -> str:
    try:
        from dspy_codex_lm.cli import CodexLMUnsupportedModelError, resolve_codex_model
    except ImportError as exc:
        raise ProviderConfigError(
            "openai-codex requires PredictRLM's dspy_codex_lm integration"
        ) from exc

    model = _selection_model(selection, definition)
    try:
        return resolve_codex_model(model)
    except CodexLMUnsupportedModelError as exc:
        raise UnsupportedProviderModelError(str(exc)) from exc


def _codex_cli_auth_path(selection: ProviderSelection) -> Path:
    if selection.auth_source not in {None, "codex-cli"}:
        raise ProviderConfigError(
            "openai-codex requires auth_source='codex-cli'"
        )
    if shutil.which("codex") is None:
        raise MissingCodexCliError(
            "openai-codex requires the official `codex` CLI. "
            "Install it, then run `codex login --device-auth`."
        )
    try:
        from dspy_codex_lm.auth import codex_auth_path, load_codex_auth
    except ImportError as exc:
        raise ProviderConfigError(
            "openai-codex requires PredictRLM's dspy_codex_lm integration"
        ) from exc

    auth_path = codex_auth_path()
    try:
        load_codex_auth(auth_path)
    except FileNotFoundError as exc:
        raise MissingCodexAuthError(
            f"openai-codex could not find Codex CLI auth at {auth_path}. "
            "Run `codex login --device-auth`."
        ) from exc
    except Exception as exc:
        raise MissingCodexAuthError(
            f"openai-codex found unusable Codex CLI auth at {auth_path}. "
            "Run `codex login --device-auth` again."
        ) from exc
    return auth_path


def _validate_custom_openai_selection(
    selection: ProviderSelection,
    definition: ProviderDefinition,
    *,
    env: Mapping[str, str] | None,
) -> str:
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

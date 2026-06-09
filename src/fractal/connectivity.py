from __future__ import annotations

import urllib.error
import urllib.request
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field

from .providers import (
    ANTHROPIC,
    CUSTOM_OPENAI_COMPATIBLE,
    DEEPSEEK,
    GEMINI,
    GROQ,
    MISTRAL,
    OLLAMA,
    OPENAI_API,
    OPENROUTER,
    XAI,
    ProviderDefinition,
    ProviderError,
    ProviderSelection,
    get_provider,
    resolve_api_key,
)


DEFAULT_TIMEOUT_SECONDS = 5.0

Opener = Callable[[urllib.request.Request, float], int]


class ProviderConnectivityError(ProviderError):
    """Raised when a provider's API cannot be reached or rejects the credential."""


@dataclass(frozen=True)
class _Endpoint:
    url: str
    headers: Mapping[str, str] = field(default_factory=dict)
    unreachable_hint: str | None = None


def check_provider_connectivity(
    selection: ProviderSelection,
    *,
    env: Mapping[str, str] | None = None,
    timeout: float = DEFAULT_TIMEOUT_SECONDS,
    opener: Opener | None = None,
) -> bool:
    """Make one cheap authenticated request against the configured provider.

    Returns True when the provider answered, False when the provider has no
    connectivity check (e.g. Codex CLI auth is validated locally). Raises
    ProviderConnectivityError when the provider is unreachable or rejects
    the credential.
    """
    definition = get_provider(selection.provider)
    definition.validate_shape(selection)
    api_key = resolve_api_key(selection, env=env)
    endpoint = _endpoint_for(selection, definition, api_key)
    if endpoint is None:
        return False

    request = urllib.request.Request(
        endpoint.url,
        headers=dict(endpoint.headers),
        method="GET",
    )
    open_status = opener or _urlopen_status
    try:
        status = open_status(request, timeout)
    except urllib.error.HTTPError as exc:
        if exc.code in {401, 403}:
            raise ProviderConnectivityError(
                f"provider {definition.id!r} rejected the credential "
                f"(HTTP {exc.code}). Check the configured API key."
            ) from exc
        raise ProviderConnectivityError(
            f"provider {definition.id!r} connectivity check failed "
            f"(HTTP {exc.code} from {endpoint.url})."
        ) from exc
    except urllib.error.URLError as exc:
        hint = f" {endpoint.unreachable_hint}" if endpoint.unreachable_hint else ""
        raise ProviderConnectivityError(
            f"provider {definition.id!r} is unreachable at {endpoint.url}: "
            f"{exc.reason}.{hint}"
        ) from exc
    if status >= 400:
        raise ProviderConnectivityError(
            f"provider {definition.id!r} connectivity check failed "
            f"(HTTP {status} from {endpoint.url})."
        )
    return True


def _urlopen_status(request: urllib.request.Request, timeout: float) -> int:
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return response.status


def _endpoint_for(
    selection: ProviderSelection,
    definition: ProviderDefinition,
    api_key: str | None,
) -> _Endpoint | None:
    bearer = {"Authorization": f"Bearer {api_key}"}
    if definition.id == OPENAI_API:
        return _Endpoint("https://api.openai.com/v1/models", bearer)
    if definition.id == ANTHROPIC:
        return _Endpoint(
            "https://api.anthropic.com/v1/models",
            {"x-api-key": api_key or "", "anthropic-version": "2023-06-01"},
        )
    if definition.id == GEMINI:
        return _Endpoint(
            "https://generativelanguage.googleapis.com/v1beta/models",
            {"x-goog-api-key": api_key or ""},
        )
    if definition.id == XAI:
        return _Endpoint("https://api.x.ai/v1/models", bearer)
    if definition.id == DEEPSEEK:
        return _Endpoint("https://api.deepseek.com/models", bearer)
    if definition.id == MISTRAL:
        return _Endpoint("https://api.mistral.ai/v1/models", bearer)
    if definition.id == GROQ:
        return _Endpoint("https://api.groq.com/openai/v1/models", bearer)
    if definition.id == OPENROUTER:
        # /key validates the credential itself; /models is unauthenticated.
        return _Endpoint("https://openrouter.ai/api/v1/key", bearer)
    if definition.id == OLLAMA:
        base_url = _base_url(selection, definition)
        return _Endpoint(
            f"{base_url}/api/tags",
            unreachable_hint="Is the Ollama server running? Start it with `ollama serve`.",
        )
    if definition.id == CUSTOM_OPENAI_COMPATIBLE:
        base_url = _base_url(selection, definition)
        return _Endpoint(f"{base_url}/models", bearer)
    return None


def _base_url(
    selection: ProviderSelection,
    definition: ProviderDefinition,
) -> str:
    base_url = selection.base_url or definition.default_base_url or ""
    return base_url.rstrip("/")

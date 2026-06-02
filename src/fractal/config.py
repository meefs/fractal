from __future__ import annotations

import os
import re
import tempfile
import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator


CONFIG_SCHEMA_VERSION = 1
CONFIG_DIR_NAME = "fractal"
CONFIG_FILE_NAME = "config.toml"
REDACTED = "<redacted>"
_ENV_VAR_NAME = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_PROVIDER_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")
_RAW_SECRET_FIELD_NAMES = frozenset({
    "api_key",
    "access_token",
    "refresh_token",
    "token",
    "secret",
    "password",
    "credential",
    "credentials",
})


class FractalConfigError(Exception):
    """Base class for global Fractal config errors."""

    def __init__(self, path: str | Path, message: str) -> None:
        self.path = Path(path)
        self.message = message
        super().__init__(f"{self.path}: {message}")


class FractalConfigParseError(FractalConfigError):
    """Raised when the config file is not valid TOML."""


class FractalConfigSchemaError(FractalConfigError):
    """Raised when valid TOML does not match Fractal's config schema."""


class ProviderConfig(BaseModel):
    """Non-secret provider settings stored in global Fractal config."""

    model_config = ConfigDict(extra="forbid")

    auth_source: Literal["env", "codex-cli"] | None = None
    api_key_env: str | None = None
    base_url: str | None = None

    @model_validator(mode="before")
    @classmethod
    def reject_raw_secret_fields(cls, value: object) -> object:
        _reject_raw_secret_fields(value)
        return value

    @model_validator(mode="after")
    def validate_non_secret_fields(self) -> "ProviderConfig":
        if self.api_key_env is not None and not _ENV_VAR_NAME.fullmatch(
            self.api_key_env
        ):
            raise ValueError("api_key_env must be an environment variable name.")
        if self.base_url is not None and not self.base_url.strip():
            raise ValueError("base_url must not be blank when provided.")
        return self


class FractalConfig(BaseModel):
    """Versioned global Fractal config loaded from TOML."""

    model_config = ConfigDict(extra="forbid")

    schema_version: Literal[1] = CONFIG_SCHEMA_VERSION
    active_provider: str = Field(min_length=1)
    active_model: str = Field(min_length=1)
    providers: dict[str, ProviderConfig] = Field(default_factory=dict)

    @model_validator(mode="before")
    @classmethod
    def reject_raw_secret_fields(cls, value: object) -> object:
        _reject_raw_secret_fields(value)
        return value

    @model_validator(mode="after")
    def validate_active_provider(self) -> "FractalConfig":
        invalid_provider_ids = [
            provider_id
            for provider_id in self.providers
            if not _PROVIDER_ID.fullmatch(provider_id)
        ]
        if invalid_provider_ids:
            raise ValueError(
                "provider ids must contain only letters, numbers, dots, "
                "underscores, or hyphens."
            )
        if self.active_provider not in self.providers:
            raise ValueError("active_provider must reference a configured provider.")
        return self


class EffectiveFractalConfig(BaseModel):
    """Resolved config object consumed by future runtime and onboarding code."""

    model_config = ConfigDict(frozen=True)

    schema_version: Literal[1] = CONFIG_SCHEMA_VERSION
    provider: str
    model: str
    provider_config: ProviderConfig
    config_path: Path | None = None


@dataclass(frozen=True)
class ConfigLoadResult:
    path: Path
    config: FractalConfig | None

    @property
    def missing(self) -> bool:
        return self.config is None

    @property
    def loaded(self) -> bool:
        return self.config is not None


def default_config_path(
    *,
    env: dict[str, str] | None = None,
    home: str | Path | None = None,
) -> Path:
    environment = os.environ if env is None else env
    xdg_config_home = environment.get("XDG_CONFIG_HOME")
    if xdg_config_home:
        return Path(xdg_config_home).expanduser() / CONFIG_DIR_NAME / CONFIG_FILE_NAME
    home_path = Path.home() if home is None else Path(home).expanduser()
    return home_path / ".config" / CONFIG_DIR_NAME / CONFIG_FILE_NAME


def load_config(path: str | Path | None = None) -> ConfigLoadResult:
    config_path = Path(path) if path is not None else default_config_path()
    if not config_path.exists():
        return ConfigLoadResult(path=config_path, config=None)
    try:
        raw_text = config_path.read_text(encoding="utf-8")
    except OSError as exc:
        raise FractalConfigError(config_path, f"could not read config: {exc}") from exc
    try:
        data = tomllib.loads(raw_text)
    except tomllib.TOMLDecodeError as exc:
        raise FractalConfigParseError(config_path, str(exc)) from exc
    try:
        config = FractalConfig.model_validate(data)
    except (ValidationError, ValueError) as exc:
        raise FractalConfigSchemaError(config_path, str(exc)) from exc
    return ConfigLoadResult(path=config_path, config=config)


def resolve_effective_config(
    config: FractalConfig,
    *,
    path: str | Path | None = None,
) -> EffectiveFractalConfig:
    return EffectiveFractalConfig(
        provider=config.active_provider,
        model=config.active_model,
        provider_config=config.providers[config.active_provider],
        config_path=Path(path) if path is not None else None,
    )


def write_config(config: FractalConfig, path: str | Path | None = None) -> Path:
    config_path = Path(path) if path is not None else default_config_path()
    validated = FractalConfig.model_validate(config.model_dump(mode="python"))
    payload = validated.model_dump(mode="python", exclude_none=True)

    _ensure_config_dir(config_path.parent)
    try:
        import tomli_w

        toml_text = tomli_w.dumps(payload)
        tmp_path = _write_temp_config(config_path.parent, config_path.name, toml_text)
        try:
            os.replace(tmp_path, config_path)
            _chmod_posix(config_path, 0o600)
        finally:
            if tmp_path.exists():
                tmp_path.unlink()
    except OSError as exc:
        raise FractalConfigError(config_path, f"could not write config: {exc}") from exc
    return config_path


def render_effective_config(config: EffectiveFractalConfig) -> str:
    provider_config = config.provider_config
    lines = ["Fractal config"]
    if config.config_path is not None:
        lines.append(f"path: {config.config_path}")
    lines.extend([
        f"schema_version: {config.schema_version}",
        f"provider: {config.provider}",
        f"model: {config.model}",
    ])
    if provider_config.auth_source is not None:
        lines.append(f"auth_source: {provider_config.auth_source}")
    if provider_config.api_key_env is not None:
        lines.append(f"api_key_env: {REDACTED}")
    if provider_config.base_url is not None:
        lines.append(f"base_url: {provider_config.base_url}")
    return "\n".join(lines)


def render_config(config: FractalConfig, *, path: str | Path | None = None) -> str:
    return render_effective_config(resolve_effective_config(config, path=path))


def _ensure_config_dir(path: Path) -> None:
    created = not path.exists()
    path.mkdir(parents=True, mode=0o700, exist_ok=True)
    if created:
        _chmod_posix(path, 0o700)


def _write_temp_config(directory: Path, filename: str, toml_text: str) -> Path:
    fd, tmp_name = tempfile.mkstemp(
        prefix=f".{filename}.",
        suffix=".tmp",
        dir=directory,
        text=True,
    )
    tmp_path = Path(tmp_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as tmp_file:
            tmp_file.write(toml_text)
            tmp_file.flush()
            os.fsync(tmp_file.fileno())
        _chmod_posix(tmp_path, 0o600)
    except BaseException:
        try:
            tmp_path.unlink()
        finally:
            raise
    return tmp_path


def _chmod_posix(path: Path, mode: int) -> None:
    if os.name == "posix":
        path.chmod(mode)


def _reject_raw_secret_fields(value: object) -> None:
    if isinstance(value, dict):
        for key, nested in value.items():
            if isinstance(key, str) and key in _RAW_SECRET_FIELD_NAMES:
                raise ValueError(
                    f"{key!r} is not allowed in Fractal config; store only "
                    "non-secret credential references."
                )
            _reject_raw_secret_fields(nested)
    elif isinstance(value, list):
        for item in value:
            _reject_raw_secret_fields(item)

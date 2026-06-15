from __future__ import annotations

import os
import tomllib
from pathlib import Path

from .config import (
    FractalConfigError,
    _chmod_posix,
    _ensure_config_dir,
    _write_temp_config,
    default_config_path,
)

CREDENTIALS_FILE_NAME = "credentials.toml"
_KEYS_TABLE = "api_keys"


class FractalCredentialsError(FractalConfigError):
    """Raised when the local credentials file cannot be read or written."""


def default_credentials_path(
    *,
    env: dict[str, str] | None = None,
    home: str | Path | None = None,
) -> Path:
    return default_config_path(env=env, home=home).parent / CREDENTIALS_FILE_NAME


def load_stored_credentials(path: str | Path | None = None) -> dict[str, str]:
    credentials_path = Path(path) if path is not None else default_credentials_path()
    if not credentials_path.exists():
        return {}
    try:
        raw_text = credentials_path.read_text(encoding="utf-8")
        data = tomllib.loads(raw_text)
    except OSError as exc:
        raise FractalCredentialsError(
            credentials_path, f"could not read credentials: {exc}"
        ) from exc
    except tomllib.TOMLDecodeError as exc:
        raise FractalCredentialsError(credentials_path, str(exc)) from exc
    keys = data.get(_KEYS_TABLE, {})
    if not isinstance(keys, dict) or not all(
        isinstance(provider, str) and isinstance(key, str)
        for provider, key in keys.items()
    ):
        raise FractalCredentialsError(
            credentials_path,
            f"[{_KEYS_TABLE}] must map provider ids to API key strings.",
        )
    return dict(keys)


def get_stored_credential(
    provider_id: str,
    path: str | Path | None = None,
) -> str | None:
    return load_stored_credentials(path).get(provider_id)


def store_credential(
    provider_id: str,
    api_key: str,
    path: str | Path | None = None,
) -> Path:
    if not api_key.strip():
        raise ValueError("API key must not be blank.")
    keys = load_stored_credentials(path)
    keys[provider_id] = api_key
    return _write_credentials(keys, path)


def delete_credential(provider_id: str, path: str | Path | None = None) -> bool:
    keys = load_stored_credentials(path)
    if provider_id not in keys:
        return False
    del keys[provider_id]
    _write_credentials(keys, path)
    return True


def _write_credentials(keys: dict[str, str], path: str | Path | None) -> Path:
    credentials_path = Path(path) if path is not None else default_credentials_path()
    _ensure_config_dir(credentials_path.parent)
    try:
        import tomli_w

        toml_text = tomli_w.dumps({_KEYS_TABLE: keys})
        tmp_path = _write_temp_config(
            credentials_path.parent, credentials_path.name, toml_text
        )
        try:
            os.replace(tmp_path, credentials_path)
            _chmod_posix(credentials_path, 0o600)
        finally:
            if tmp_path.exists():
                tmp_path.unlink()
    except OSError as exc:
        raise FractalCredentialsError(
            credentials_path, f"could not write credentials: {exc}"
        ) from exc
    return credentials_path

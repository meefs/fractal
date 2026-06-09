from __future__ import annotations

import os
import sys
import tomllib
from io import StringIO
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest


def test_credentials_round_trip_with_restrictive_permissions(tmp_path: Path) -> None:
    from fractal.credentials import (
        delete_credential,
        get_stored_credential,
        load_stored_credentials,
        store_credential,
    )

    path = tmp_path / "credentials.toml"

    store_credential("anthropic", "sk-ant-secret", path)
    store_credential("openai-api", "sk-oa-secret", path)

    assert get_stored_credential("anthropic", path) == "sk-ant-secret"
    assert load_stored_credentials(path) == {
        "anthropic": "sk-ant-secret",
        "openai-api": "sk-oa-secret",
    }
    if os.name == "posix":
        assert (path.stat().st_mode & 0o777) == 0o600

    assert delete_credential("anthropic", path) is True
    assert delete_credential("anthropic", path) is False
    assert get_stored_credential("anthropic", path) is None


def test_store_credential_rejects_blank_key(tmp_path: Path) -> None:
    from fractal.credentials import store_credential

    with pytest.raises(ValueError, match="blank"):
        store_credential("anthropic", "   ", tmp_path / "credentials.toml")


def test_default_credentials_path_lives_beside_config(tmp_path: Path) -> None:
    from fractal.credentials import default_credentials_path

    path = default_credentials_path(env={"XDG_CONFIG_HOME": str(tmp_path)})

    assert path == tmp_path / "fractal" / "credentials.toml"


def test_stored_auth_builds_lm_with_key_from_credentials_file(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    from fractal.credentials import store_credential
    from fractal.providers import ANTHROPIC, ProviderSelection, build_lm

    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))
    store_credential(ANTHROPIC, "sk-ant-secret")

    created: dict[str, object] = {}

    def fake_lm(**kwargs: object) -> object:
        created.update(kwargs)
        return SimpleNamespace(kind="lm")

    dspy_module = ModuleType("dspy")
    dspy_module.LM = fake_lm
    monkeypatch.setitem(sys.modules, "dspy", dspy_module)

    selection = ProviderSelection(
        ANTHROPIC,
        model="claude-sonnet-4-6",
        auth_source="stored",
    )
    lm = build_lm(selection, env={})

    assert lm.kind == "lm"
    assert created == {
        "model": "anthropic/claude-sonnet-4-6",
        "api_key": "sk-ant-secret",
    }


def test_stored_auth_reports_missing_key_without_requiring_env(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    from fractal.providers import (
        ANTHROPIC,
        MissingProviderCredentialError,
        ProviderSelection,
        check_provider_readiness,
        validate_provider_selection,
    )

    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))
    selection = ProviderSelection(
        ANTHROPIC,
        model="claude-sonnet-4-6",
        auth_source="stored",
    )
    validate_provider_selection(selection)

    with pytest.raises(MissingProviderCredentialError) as excinfo:
        check_provider_readiness(selection, env={})

    assert "no stored API key" in str(excinfo.value)
    assert "fractal config setup" in str(excinfo.value)


def test_custom_openai_compatible_supports_stored_keys(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    from fractal.credentials import store_credential
    from fractal.providers import (
        CUSTOM_OPENAI_COMPATIBLE,
        ProviderSelection,
        build_lm,
    )

    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))
    store_credential(CUSTOM_OPENAI_COMPATIBLE, "endpoint-secret")

    created: dict[str, object] = {}

    def fake_lm(**kwargs: object) -> object:
        created.update(kwargs)
        return SimpleNamespace(kind="lm")

    dspy_module = ModuleType("dspy")
    dspy_module.LM = fake_lm
    monkeypatch.setitem(sys.modules, "dspy", dspy_module)

    lm = build_lm(
        ProviderSelection(
            CUSTOM_OPENAI_COMPATIBLE,
            model="custom-model",
            base_url="https://llm.example.test/v1",
            auth_source="stored",
        ),
        env={},
    )

    assert lm.kind == "lm"
    assert created["api_key"] == "endpoint-secret"


def test_setup_paste_flow_stores_key_outside_config(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    from fractal import cli

    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    args = cli.build_parser().parse_args(["config", "setup"])
    stdout = StringIO()
    stderr = StringIO()

    exit_code = cli.run_config_command(
        args,
        stdin=StringIO("anthropic\nclaude-sonnet-4-6\n1\nsk-ant-pasted\n"),
        stdout=stdout,
        stderr=stderr,
    )

    assert exit_code == 0

    config_path = tmp_path / "fractal" / "config.toml"
    config_text = config_path.read_text(encoding="utf-8")
    data = tomllib.loads(config_text)
    assert data["providers"]["anthropic"] == {"auth_source": "stored"}
    assert "sk-ant-pasted" not in config_text

    credentials_path = tmp_path / "fractal" / "credentials.toml"
    credentials = tomllib.loads(credentials_path.read_text(encoding="utf-8"))
    assert credentials["api_keys"]["anthropic"] == "sk-ant-pasted"
    assert "sk-ant-pasted" not in stdout.getvalue()
    assert "sk-ant-pasted" not in stderr.getvalue()
    assert str(credentials_path) in stdout.getvalue()


def test_config_status_validates_stored_credentials(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    from fractal import cli
    from fractal.credentials import store_credential

    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))
    config_path = tmp_path / "fractal" / "config.toml"
    config_path.parent.mkdir(parents=True)
    config_path.write_text(
        """
schema_version = 1
active_provider = "anthropic"
active_model = "claude-sonnet-4-6"

[providers.anthropic]
auth_source = "stored"
""".strip(),
        encoding="utf-8",
    )
    args = cli.build_parser().parse_args(["config", "status"])

    stdout = StringIO()
    exit_code = cli.run_config_command(args, stdout=stdout, stderr=StringIO())
    assert exit_code == 1
    assert "Fractal config status: invalid" in stdout.getvalue()

    store_credential("anthropic", "sk-ant-secret")

    stdout = StringIO()
    exit_code = cli.run_config_command(args, stdout=stdout, stderr=StringIO())
    assert exit_code == 0
    output = stdout.getvalue()
    assert "Fractal config status: ok" in output
    assert "stored in local credentials file" in output
    assert "sk-ant-secret" not in output

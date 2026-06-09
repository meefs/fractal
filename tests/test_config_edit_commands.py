from __future__ import annotations

import tomllib
from io import StringIO
from pathlib import Path

import pytest

def write_api_config(config_home: Path, *, api_key_env: str = "OPENAI_API_KEY") -> Path:
    path = config_home / "fractal" / "config.toml"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        f"""
schema_version = 1
active_provider = "openai-api"
active_model = "gpt-5.5"

[providers.openai-api]
auth_source = "env"
api_key_env = "{api_key_env}"
""".strip(),
        encoding="utf-8",
    )
    return path


def run_config(args_list: list[str]) -> tuple[int, str, str]:
    from fractal import cli

    args = cli.build_parser().parse_args(args_list)
    stdout = StringIO()
    stderr = StringIO()
    exit_code = cli.run_config_command(args, stdout=stdout, stderr=stderr)
    return exit_code, stdout.getvalue(), stderr.getvalue()


def test_config_get_reads_effective_values(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))
    write_api_config(tmp_path)

    exit_code, stdout, _ = run_config(["config", "get", "active_model"])
    assert exit_code == 0
    assert stdout.strip() == "gpt-5.5"

    exit_code, _, stderr = run_config(["config", "get", "defaults.max_iterations"])
    assert exit_code == 1
    assert "is not set" in stderr


def test_config_set_updates_global_config_with_typed_values(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))
    config_path = write_api_config(tmp_path)

    exit_code, stdout, _ = run_config(["config", "set", "active_model", "gpt-5.4-mini"])
    assert exit_code == 0
    assert str(config_path) in stdout

    exit_code, _, _ = run_config(["config", "set", "defaults.max_iterations", "12"])
    assert exit_code == 0
    exit_code, _, _ = run_config(["config", "set", "defaults.verbose", "true"])
    assert exit_code == 0

    data = tomllib.loads(config_path.read_text(encoding="utf-8"))
    assert data["active_model"] == "gpt-5.4-mini"
    assert data["defaults"] == {"max_iterations": 12, "verbose": True}


def test_config_set_rejects_secrets_and_unknown_keys(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))
    config_path = write_api_config(tmp_path)
    original = config_path.read_text(encoding="utf-8")

    exit_code, _, stderr = run_config(
        ["config", "set", "providers.openai-api.api_key", "sk-leaked"]
    )
    assert exit_code == 1
    assert "not allowed" in stderr
    assert "sk-leaked" not in config_path.read_text(encoding="utf-8")

    exit_code, _, stderr = run_config(["config", "set", "no_such_key", "x"])
    assert exit_code == 1

    exit_code, _, stderr = run_config(
        ["config", "set", "defaults.max_iterations", "0"]
    )
    assert exit_code == 1

    assert config_path.read_text(encoding="utf-8") == original


def test_config_unset_removes_value(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))
    config_path = write_api_config(tmp_path)
    run_config(["config", "set", "active_sub_model", "gpt-5.4-mini"])

    exit_code, stdout, _ = run_config(["config", "unset", "active_sub_model"])
    assert exit_code == 0
    assert "unset active_sub_model" in stdout
    assert "active_sub_model" not in config_path.read_text(encoding="utf-8")

    exit_code, _, stderr = run_config(["config", "unset", "active_sub_model"])
    assert exit_code == 1
    assert "is not set" in stderr


def test_config_set_project_creates_partial_project_file(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "home"))
    write_api_config(tmp_path / "home")
    workspace = tmp_path / "repo"
    workspace.mkdir()

    exit_code, stdout, _ = run_config(
        [
            "--workspace",
            str(workspace),
            "config",
            "set",
            "active_model",
            "gpt-5.4",
            "--project",
        ]
    )
    assert exit_code == 0

    project_path = workspace / ".fractal" / "config.toml"
    data = tomllib.loads(project_path.read_text(encoding="utf-8"))
    assert data["active_model"] == "gpt-5.4"
    assert "active_provider" not in data

    # The effective value reflects the project layer.
    exit_code, stdout, _ = run_config(
        ["--workspace", str(workspace), "config", "get", "active_model"]
    )
    assert exit_code == 0
    assert stdout.strip() == "gpt-5.4"


def test_config_set_without_global_config_requires_setup(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))

    exit_code, _, stderr = run_config(["config", "set", "active_model", "gpt-5.5"])

    assert exit_code == 1
    assert "fractal config setup" in stderr

from __future__ import annotations

import urllib.error
import urllib.request
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


def _http_error(url: str, code: int) -> urllib.error.HTTPError:
    return urllib.error.HTTPError(url, code, "error", hdrs=None, fp=None)


def test_connectivity_sends_credential_and_accepts_success() -> None:
    from fractal.providers import ANTHROPIC, ProviderSelection
    from fractal.connectivity import check_provider_connectivity

    requests: list[urllib.request.Request] = []

    def opener(request: urllib.request.Request, timeout: float) -> int:
        requests.append(request)
        return 200

    checked = check_provider_connectivity(
        ProviderSelection(ANTHROPIC, model="claude-sonnet-4-6"),
        env={"ANTHROPIC_API_KEY": "sk-ant-secret"},
        opener=opener,
    )

    assert checked is True
    request = requests[0]
    assert request.full_url == "https://api.anthropic.com/v1/models"
    assert request.get_header("X-api-key") == "sk-ant-secret"


def test_connectivity_reports_rejected_credential_without_leaking_it() -> None:
    from fractal.providers import OPENAI_API, ProviderSelection
    from fractal.connectivity import (
        ProviderConnectivityError,
        check_provider_connectivity,
    )

    def opener(request: urllib.request.Request, timeout: float) -> int:
        raise _http_error(request.full_url, 401)

    with pytest.raises(ProviderConnectivityError) as excinfo:
        check_provider_connectivity(
            ProviderSelection(OPENAI_API, model="gpt-5.5"),
            env={"OPENAI_API_KEY": "sk-bad-secret"},
            opener=opener,
        )

    message = str(excinfo.value)
    assert "rejected the credential" in message
    assert "401" in message
    assert "sk-bad-secret" not in message


def test_connectivity_reports_unreachable_ollama_with_hint() -> None:
    from fractal.providers import OLLAMA, ProviderSelection
    from fractal.connectivity import (
        ProviderConnectivityError,
        check_provider_connectivity,
    )

    def opener(request: urllib.request.Request, timeout: float) -> int:
        raise urllib.error.URLError("connection refused")

    with pytest.raises(ProviderConnectivityError) as excinfo:
        check_provider_connectivity(
            ProviderSelection(OLLAMA, model="qwen3-coder"),
            env={},
            opener=opener,
        )

    message = str(excinfo.value)
    assert "http://localhost:11434/api/tags" in message
    assert "ollama serve" in message


def test_connectivity_skips_codex_cli_provider() -> None:
    from fractal.providers import OPENAI_CODEX, ProviderSelection
    from fractal.connectivity import check_provider_connectivity

    def opener(request: urllib.request.Request, timeout: float) -> int:
        raise AssertionError("codex must not trigger a network call")

    checked = check_provider_connectivity(
        ProviderSelection(OPENAI_CODEX, model="gpt-5.5"),
        env={},
        opener=opener,
    )

    assert checked is False


def test_config_status_runs_connectivity_check(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    from fractal import cli

    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))
    monkeypatch.setenv("OPENAI_API_KEY", "sk-secret-value")
    write_api_config(tmp_path)
    args = cli.build_parser().parse_args(["config", "status"])
    stdout = StringIO()

    exit_code = cli.run_config_command(args, stdout=stdout, stderr=StringIO())

    assert exit_code == 0
    assert "Fractal config status: ok" in stdout.getvalue()
    assert "connectivity: verified" in stdout.getvalue()


def test_config_status_reports_unreachable_provider(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    from fractal import cli

    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))
    monkeypatch.setenv("OPENAI_API_KEY", "sk-secret-value")
    write_api_config(tmp_path)

    def failing_opener(request: urllib.request.Request, timeout: float) -> int:
        raise _http_error(request.full_url, 401)

    monkeypatch.setattr("fractal.connectivity._urlopen_status", failing_opener)
    args = cli.build_parser().parse_args(["config", "status"])
    stdout = StringIO()
    stderr = StringIO()

    exit_code = cli.run_config_command(args, stdout=stdout, stderr=stderr)

    assert exit_code == 1
    assert "Fractal config status: unreachable" in stdout.getvalue()
    assert "rejected the credential" in stderr.getvalue()
    assert "sk-secret-value" not in stderr.getvalue()


def test_config_status_offline_skips_connectivity(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    from fractal import cli

    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))
    monkeypatch.setenv("OPENAI_API_KEY", "sk-secret-value")
    write_api_config(tmp_path)

    def failing_opener(request: urllib.request.Request, timeout: float) -> int:
        raise AssertionError("offline mode must not touch the network")

    monkeypatch.setattr("fractal.connectivity._urlopen_status", failing_opener)
    args = cli.build_parser().parse_args(["config", "status", "--offline"])
    stdout = StringIO()

    exit_code = cli.run_config_command(args, stdout=stdout, stderr=StringIO())

    assert exit_code == 0
    assert "Fractal config status: ok" in stdout.getvalue()
    assert "connectivity: skipped (--offline)" in stdout.getvalue()


def test_config_setup_warns_but_writes_when_connectivity_fails(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    from fractal import cli

    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-secret")

    def failing_opener(request: urllib.request.Request, timeout: float) -> int:
        raise urllib.error.URLError("network down")

    monkeypatch.setattr("fractal.connectivity._urlopen_status", failing_opener)
    args = cli.build_parser().parse_args(["config", "setup"])
    stdout = StringIO()
    stderr = StringIO()

    exit_code = cli.run_config_command(
        args,
        stdin=StringIO("anthropic\nclaude-sonnet-4-6\n2\n\n"),
        stdout=stdout,
        stderr=stderr,
    )

    assert exit_code == 0
    assert (tmp_path / "fractal" / "config.toml").exists()
    assert "warning" in stderr.getvalue()
    assert "unreachable" in stderr.getvalue()
    assert "fractal config status" in stderr.getvalue()


def test_config_setup_verifies_connectivity_on_success(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    from fractal import cli

    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-secret")
    args = cli.build_parser().parse_args(["config", "setup"])
    stdout = StringIO()

    exit_code = cli.run_config_command(
        args,
        stdin=StringIO("anthropic\nclaude-sonnet-4-6\n2\n\n"),
        stdout=stdout,
        stderr=StringIO(),
    )

    assert exit_code == 0
    assert "Provider connectivity verified." in stdout.getvalue()


def test_list_ollama_models_parses_tags_and_dedupes() -> None:
    from fractal.connectivity import list_ollama_models

    requests: list[urllib.request.Request] = []

    def opener(request: urllib.request.Request, timeout: float) -> object:
        requests.append(request)
        return {
            "models": [
                {"name": "qwen3-coder:latest"},
                {"name": "qwen3-coder"},
                {"name": "gpt-oss:20b"},
                {"size": 123},
            ]
        }

    models = list_ollama_models(opener=opener)

    assert models == ["qwen3-coder", "gpt-oss:20b"]
    assert requests[0].full_url == "http://localhost:11434/api/tags"


def test_list_ollama_models_falls_back_to_empty_on_failure() -> None:
    from fractal.connectivity import list_ollama_models

    def opener(request: urllib.request.Request, timeout: float) -> object:
        raise urllib.error.URLError("connection refused")

    assert list_ollama_models(opener=opener) == []


def test_ollama_setup_lists_installed_models_first(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    from fractal.onboarding import prompt_for_config

    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))
    monkeypatch.setattr(
        "fractal.connectivity.list_ollama_models",
        lambda *args, **kwargs: ["smollm2", "qwen3-coder"],
    )
    stdout = StringIO()

    config = prompt_for_config(
        stdin=StringIO("ollama\n1\n\n"),
        stdout=stdout,
    )

    assert config.active_provider == "ollama"
    # Choice 1 is the first installed model, ahead of the static suggestions.
    assert config.active_model == "smollm2"
    assert "1. smollm2 (installed)" in stdout.getvalue()


def test_ollama_setup_degrades_to_static_models_when_server_down(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    from fractal.onboarding import prompt_for_config

    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))

    config = prompt_for_config(
        stdin=StringIO("ollama\n1\n\n"),
        stdout=StringIO(),
    )

    assert config.active_model == "qwen3-coder"

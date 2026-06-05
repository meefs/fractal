from __future__ import annotations

from io import StringIO
from pathlib import Path
import sys
import tomllib
from types import ModuleType, SimpleNamespace

import pytest


def write_api_config(config_home: Path, *, api_key_env: str = "OPENAI_API_KEY") -> Path:
    path = config_home / "fractal" / "config.toml"
    path.parent.mkdir(parents=True)
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


def install_fake_codex_modules(monkeypatch: pytest.MonkeyPatch) -> None:
    class FakeUnsupportedModelError(RuntimeError):
        pass

    codex_module = ModuleType("dspy_codex_lm")
    cli_module = ModuleType("dspy_codex_lm.cli")
    cli_module.CodexLMUnsupportedModelError = FakeUnsupportedModelError
    cli_module.resolve_codex_model = lambda model: model
    auth_module = ModuleType("dspy_codex_lm.auth")
    auth_module.codex_auth_path = lambda: "/tmp/codex-auth.json"
    auth_module.load_codex_auth = lambda path: ("secret-token", "acct-123")

    monkeypatch.setitem(sys.modules, "dspy_codex_lm", codex_module)
    monkeypatch.setitem(sys.modules, "dspy_codex_lm.cli", cli_module)
    monkeypatch.setitem(sys.modules, "dspy_codex_lm.auth", auth_module)
    monkeypatch.setattr("fractal.providers.shutil.which", lambda name: "/bin/codex")


def test_cli_parser_accepts_config_commands() -> None:
    from fractal.cli import build_parser

    args = build_parser().parse_args(["config", "status"])

    assert args.command == "config"
    assert args.config_command == "status"


def test_config_show_redacts_credential_reference(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    from fractal import cli

    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))
    monkeypatch.setenv("OPENAI_API_KEY", "sk-secret-value")
    write_api_config(tmp_path)
    args = cli.build_parser().parse_args(["config", "show"])
    stdout = StringIO()
    stderr = StringIO()

    exit_code = cli.run_config_command(args, stdout=stdout, stderr=stderr)

    assert exit_code == 0
    output = stdout.getvalue()
    assert "provider: openai-api" in output
    assert "model: gpt-5.5" in output
    assert "api_key_env: <redacted>" in output
    assert "OPENAI_API_KEY" not in output
    assert "sk-secret-value" not in output
    assert stderr.getvalue() == ""


def test_config_status_reports_missing_env_without_secret_leak(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    from fractal import cli

    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    write_api_config(tmp_path)
    args = cli.build_parser().parse_args(["config", "status"])
    stdout = StringIO()
    stderr = StringIO()

    exit_code = cli.run_config_command(args, stdout=stdout, stderr=stderr)

    assert exit_code == 1
    assert "Fractal config status: invalid" in stdout.getvalue()
    assert "api_key_env: <redacted>" in stdout.getvalue()
    error = stderr.getvalue()
    assert "OPENAI_API_KEY" in error
    assert "sk-secret-value" not in error
    assert "fractal config setup" in error


def test_config_setup_api_provider_writes_non_secret_toml(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    from fractal import cli

    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))
    monkeypatch.setenv("OPENAI_API_KEY", "sk-secret-value")
    args = cli.build_parser().parse_args(["config", "setup"])
    stdout = StringIO()
    stderr = StringIO()

    exit_code = cli.run_config_command(
        args,
        stdin=StringIO("openai-api\ngpt-5.5\n\n"),
        stdout=stdout,
        stderr=stderr,
    )

    assert exit_code == 0
    path = tmp_path / "fractal" / "config.toml"
    data = tomllib.loads(path.read_text(encoding="utf-8"))
    assert data == {
        "schema_version": 1,
        "active_provider": "openai-api",
        "active_model": "gpt-5.5",
        "providers": {
            "openai-api": {
                "auth_source": "env",
                "api_key_env": "OPENAI_API_KEY",
            }
        },
    }
    assert "sk-secret-value" not in path.read_text(encoding="utf-8")
    assert str(path) in stdout.getvalue()
    assert stderr.getvalue() == ""


def test_line_setup_accepts_numbered_provider_and_model_choices() -> None:
    from fractal.onboarding import prompt_for_config

    config = prompt_for_config(
        stdin=StringIO("2\n2\n\n"),
        stdout=StringIO(),
    )

    assert config.active_provider == "openai-api"
    assert config.active_model == "gpt-5.4"
    assert config.providers["openai-api"].api_key_env == "OPENAI_API_KEY"


def test_inline_menu_space_selects_and_enter_confirms() -> None:
    from fractal.onboarding import InlineMenuState, MenuChoice

    state = InlineMenuState.create(
        choices=[
            MenuChoice(value="first", label="First"),
            MenuChoice(value="second", label="Second"),
        ],
        default="first",
    )

    state.move_down()
    assert state.confirmed_value() == "first"

    state.select_active()

    assert state.confirmed_value() == "second"


def test_inline_menu_key_bindings_include_space_and_enter() -> None:
    from fractal.onboarding import (
        InlineMenuState,
        MenuChoice,
        _inline_menu_key_bindings,
    )

    state = InlineMenuState.create(
        choices=[MenuChoice(value="first", label="First")],
        default="first",
    )
    bindings = _inline_menu_key_bindings(state)

    key_names = {tuple(str(key) for key in binding.keys) for binding in bindings.bindings}
    assert (" ",) in key_names
    assert ("Keys.ControlM",) in key_names


def test_inline_setup_uses_provider_and_model_menus(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from fractal import onboarding

    calls: list[dict[str, object]] = []

    def fake_choose_from_menu(**kwargs: object) -> str:
        calls.append(kwargs)
        if kwargs["title"] == "Fractal setup":
            return "openai-api"
        return "gpt-5.4-mini"

    monkeypatch.setattr(
        onboarding,
        "_should_use_inline_menu",
        lambda **kwargs: True,
    )
    monkeypatch.setattr(onboarding, "_choose_from_menu", fake_choose_from_menu)
    monkeypatch.setattr(
        onboarding,
        "_prompt_text_interactive",
        lambda **kwargs: kwargs.get("default", ""),
    )

    config = onboarding.prompt_for_config(stdin=StringIO(), stdout=StringIO())

    assert config.active_provider == "openai-api"
    assert config.active_model == "gpt-5.4-mini"
    assert [call["title"] for call in calls] == [
        "Fractal setup",
        "OpenAI API model",
    ]
    provider_values = [choice.value for choice in calls[0]["choices"]]
    model_values = [choice.value for choice in calls[1]["choices"]]
    assert "openai-api" in provider_values
    assert model_values == [
        "gpt-5.5",
        "gpt-5.4",
        "gpt-5.4-mini",
        "gpt-5.4-nano",
    ]


def test_inline_setup_allows_custom_openai_model_entry(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from fractal import onboarding

    model_choices: list[str] = []

    def fake_choose_from_menu(**kwargs: object) -> str:
        if kwargs["title"] == "Fractal setup":
            return "custom-openai-compatible"
        model_choices.extend(choice.value for choice in kwargs["choices"])
        return onboarding.CUSTOM_MODEL_SENTINEL

    def fake_prompt_text_interactive(**kwargs: object) -> str:
        if kwargs["label"] == "Model id":
            return "endpoint-specific-model"
        if kwargs["label"] == "OpenAI-compatible base URL":
            return "https://llm.example.test/v1"
        return "CUSTOM_KEY"

    monkeypatch.setattr(
        onboarding,
        "_should_use_inline_menu",
        lambda **kwargs: True,
    )
    monkeypatch.setattr(onboarding, "_choose_from_menu", fake_choose_from_menu)
    monkeypatch.setattr(
        onboarding,
        "_prompt_text_interactive",
        fake_prompt_text_interactive,
    )

    config = onboarding.prompt_for_config(stdin=StringIO(), stdout=StringIO())

    assert onboarding.CUSTOM_MODEL_SENTINEL in model_choices
    assert config.active_provider == "custom-openai-compatible"
    assert config.active_model == "endpoint-specific-model"
    assert (
        config.providers["custom-openai-compatible"].base_url
        == "https://llm.example.test/v1"
    )


def test_config_setup_custom_invalid_url_does_not_write_config(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    from fractal import cli

    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))
    monkeypatch.setenv("CUSTOM_OPENAI_API_KEY", "secret-value")
    args = cli.build_parser().parse_args(["config", "setup"])
    stderr = StringIO()

    exit_code = cli.run_config_command(
        args,
        stdin=StringIO(
            "custom-openai-compatible\ncustom-model\nnot-a-url\nCUSTOM_OPENAI_API_KEY\n"
        ),
        stdout=StringIO(),
        stderr=stderr,
    )

    assert exit_code == 1
    assert not (tmp_path / "fractal" / "config.toml").exists()
    assert "HTTP(S) URL" in stderr.getvalue()
    assert "secret-value" not in stderr.getvalue()


def test_config_setup_codex_provider_writes_codex_cli_source(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    from fractal import cli

    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))
    install_fake_codex_modules(monkeypatch)
    args = cli.build_parser().parse_args(["config", "setup"])

    exit_code = cli.run_config_command(
        args,
        stdin=StringIO("openai-codex\n\n"),
        stdout=StringIO(),
        stderr=StringIO(),
    )

    assert exit_code == 0
    data = tomllib.loads(
        (tmp_path / "fractal" / "config.toml").read_text(encoding="utf-8")
    )
    assert data["active_provider"] == "openai-codex"
    assert data["active_model"] == "gpt-5.5"
    assert data["providers"]["openai-codex"] == {"auth_source": "codex-cli"}


def test_resolve_runtime_lms_uses_global_config(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    from fractal import cli

    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))
    monkeypatch.setenv("OPENAI_API_KEY", "sk-secret-value")
    write_api_config(tmp_path)
    args = SimpleNamespace(lm=None, sub_lm=None)

    lm_config = cli.resolve_runtime_lms(
        args,
        stdin=StringIO(),
        stdout=StringIO(),
        stderr=StringIO(),
        auto_setup=False,
    )

    assert lm_config is not None
    assert lm_config.lm == "openai/gpt-5.5"
    assert lm_config.sub_lm == "openai/gpt-5.5"


def test_resolve_runtime_lms_auto_setup_on_missing_config(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    from fractal import cli

    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))
    monkeypatch.setenv("ANTHROPIC_API_KEY", "secret-value")
    args = SimpleNamespace(lm=None, sub_lm=None)
    stderr = StringIO()

    lm_config = cli.resolve_runtime_lms(
        args,
        stdin=StringIO("anthropic\nclaude-sonnet-4-6\n\n"),
        stdout=StringIO(),
        stderr=stderr,
        auto_setup=True,
    )

    assert lm_config is not None
    assert lm_config.lm == "anthropic/claude-sonnet-4-6"
    assert "starting setup" in stderr.getvalue()
    assert (tmp_path / "fractal" / "config.toml").exists()


def test_run_non_interactive_without_config_does_not_enter_runtime(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    from fractal import cli

    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))
    args = cli.build_parser().parse_args(["--workspace", str(tmp_path), "-p", "hello"])
    stderr = StringIO()

    exit_code = cli.run_non_interactive(
        args,
        stdin=StringIO(),
        stdout=StringIO(),
        stderr=stderr,
    )

    assert exit_code == 1
    assert "no global config found" in stderr.getvalue()
    assert "fractal config setup" in stderr.getvalue()

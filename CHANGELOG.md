# Changelog

All notable changes to Fractal are documented here. The format is based on
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project
adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- `install.sh` one-line installer (`curl -LsSf … | sh`) that bootstraps uv,
  installs Fractal as an isolated tool, and checks Docker/`sbx` prerequisites.
- Documentation for release: requirements (Docker + `sbx`), installation,
  quickstart, a full command-line option reference, and a headless / CI guide.
- `CONTRIBUTING.md` and this changelog.

## [0.1.0] - Unreleased

Initial pre-release of Fractal — an agentic CLI powered by predict-rlm.

### Added
- Agentic CLI (`fractal`) powered by predict-rlm's self-harnessed Recursive
  Language Model runtime, with each turn run over a direct workspace in a
  Docker Sandbox.
- Non-interactive single-turn mode via `-p`/`--prompt`, including stdin input
  and distinct exit codes (`2` for max-iterations reached).
- Layered TOML configuration (global → per-workspace → environment → CLI flags)
  with `config setup/status/show/get/set/unset/reset`.
- Provider/auth abstraction for OpenAI (API and Codex), Anthropic, Gemini, xAI,
  DeepSeek, Mistral, Groq, OpenRouter, Ollama, and custom OpenAI-compatible
  endpoints. Secrets are stored as references or in a separate `0600`
  credentials file, never in config.
- A configurable cheaper sub-model, optionally on a different provider.
- Durable session state under `.fractal/sessions/`, resumable via `--resume`
  and the `/resume` slash command, with per-turn and session usage reporting.
- Interactive slash commands: `/help`, `/sessions`, `/resume`, `/new`,
  `/model`, `/provider`, `/usage`, `/verbose`, `/exit`.

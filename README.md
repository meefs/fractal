Fractal
=======

Fractal is an interactive coding-agent CLI built around a Recursive Language
Model. Each user turn is one RLM call over a direct `Workspace` input mounted
into a Docker Sandbox through predict-rlm's SBX backend, so Python subprocesses
and project commands operate on the real workspace path.

Configuration
-------------

Fractal uses a global TOML config for non-secret provider and model settings.
On first interactive run, if no global config exists, Fractal starts setup
automatically:

```bash
uv run fractal
```

You can also run setup directly:

```bash
uv run fractal config setup
uv run fractal config status
uv run fractal config show
```

The default config path is `~/.config/fractal/config.toml`, or
`$XDG_CONFIG_HOME/fractal/config.toml` when `XDG_CONFIG_HOME` is set. The config
stores provider ids, model names, auth source metadata, API-key environment
variable names, and custom OpenAI-compatible base URLs. It must not store raw
API keys, OAuth tokens, or other secrets.

Supported MVP providers:

| Provider | Auth source | Default credential reference |
| --- | --- | --- |
| `openai-codex` | Official Codex CLI login | `codex login --device-auth` |
| `openai-api` | Environment variable | `OPENAI_API_KEY` |
| `anthropic` | Environment variable | `ANTHROPIC_API_KEY` |
| `openrouter` | Environment variable | `OPENROUTER_API_KEY` |
| `custom-openai-compatible` | Environment variable plus base URL | User-selected env var |

`openai-codex` requires the official `codex` CLI and an existing Codex login.
Fractal reads Codex CLI auth through PredictRLM's `dspy_codex_lm.CodexLM`
adapter and does not copy Codex OAuth tokens into Fractal config.

For one-off runs or tests, `--lm` bypasses global config resolution:

```bash
uv run fractal --lm openai/gpt-5.5 -p "summarize this repo"
```

Development
-----------

This project depends on a local editable checkout of predict-rlm:

```bash
uv run fractal --help
uv run pytest
```

The RLM-facing imports require `predict_rlm.WorkspaceMode` and direct workspace
support from the local `/Users/emile/git/predict-rlm` checkout.

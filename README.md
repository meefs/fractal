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
automatically. Setup uses inline keyboard menus for provider and model
selection:

```bash
uv run fractal
```

You can also run setup directly. Use Up/Down to move through highlighted
choices, Space to select, and Enter to confirm:

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
adapter and does not copy Codex OAuth tokens into Fractal config. Fractal only
offers the Codex `gpt-5.5` family during setup right now.

Setup model menus are curated starting points, not exhaustive provider
catalogs. OpenAI API setup offers the current GPT-5.5/GPT-5.4 text-output
models that fit Fractal's coding-agent path. Anthropic setup offers Claude
Sonnet 4.6, Opus 4.8, and Haiku 4.5. OpenRouter setup offers a small current
coding-focused subset from its live catalog, while explicit config can still
use another valid OpenRouter model id. Custom OpenAI-compatible endpoints keep
a custom model entry because Fractal cannot know an endpoint-local catalog
before the endpoint is configured.

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

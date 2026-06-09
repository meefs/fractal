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
| `gemini` | Environment variable | `GEMINI_API_KEY` |
| `xai` | Environment variable | `XAI_API_KEY` |
| `deepseek` | Environment variable | `DEEPSEEK_API_KEY` |
| `mistral` | Environment variable | `MISTRAL_API_KEY` |
| `groq` | Environment variable | `GROQ_API_KEY` |
| `openrouter` | Environment variable | `OPENROUTER_API_KEY` |
| `ollama` | Local server, no credential | `http://localhost:11434` |
| `custom-openai-compatible` | Environment variable plus base URL | User-selected env var |

`openai-codex` requires the official `codex` CLI and an existing Codex login.
Fractal reads Codex CLI auth through PredictRLM's `dspy_codex_lm.CodexLM`
adapter and does not copy Codex OAuth tokens into Fractal config. Fractal only
offers the Codex `gpt-5.5` family during setup right now.

Setup model menus are curated starting points, not exhaustive provider
catalogs. Every provider except `openai-codex` also accepts a free-form model
id (the "Custom model..." entry in menus), so newly released models work
without a Fractal update. `ollama` talks to a local Ollama server and needs no
API key; setup asks for the server URL and defaults to
`http://localhost:11434`.

For API-key providers, setup asks how to provide the key: paste it directly
(the default), or reference an environment variable. Pasted keys are stored in
`~/.config/fractal/credentials.toml` with `0600` permissions, next to the
config but never inside it; the config records only `auth_source = "stored"`.

If setup uses an environment variable that is currently unset, it still writes
the config (which never contains secrets) and prints the exact variable to
export; `fractal config status` verifies readiness afterwards.

Setup and `config status` also make one cheap authenticated request against
the provider (a models-list endpoint, or `/api/tags` for Ollama) so a typo'd
or revoked key is caught immediately instead of on the first agent turn. Pass
`--offline` to skip the live check; network failures during setup only warn
and never discard a finished setup.

Config is resolved in layers: the global file, then per-workspace overrides in
`<workspace>/.fractal/config.toml` (same schema, every field optional), then
`FRACTAL_PROVIDER` / `FRACTAL_MODEL` / `FRACTAL_SUB_MODEL` /
`FRACTAL_MAX_ITERATIONS` / `FRACTAL_VERBOSE` environment variables, with CLI
flags on top. A repo can pin its model without touching anyone's global
config, and CI can override via env. `fractal config show` lists which layers
contributed. Environment overrides apply only once some config file exists,
so first-run onboarding still triggers.

Beyond the active provider and model, the config supports:

```toml
# optional: a cheaper model for RLM sub-calls; defaults to the main model
active_sub_model = "gpt-5.4-mini"

[defaults]            # optional run defaults, overridden by CLI flags
max_iterations = 30   # --max-iterations
verbose = false       # --verbose
```

Inside the interactive session, `/provider` re-runs provider setup, `/model`
switches the model for the configured provider, and `/verbose` toggles trace
display.

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

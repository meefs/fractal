Fractal
=======

Fractal is an agentic CLI powered by
[predict-rlm](https://github.com/Trampoline-AI/predict-rlm), Trampoline's
self-harnessed Recursive Language Model runtime. It is an early, intentionally
bare-bones proof of concept — a thin UI that turns predict-rlm into a usable
terminal agent — released to see what people build with it.

What makes it different
-----------------------

Most agent harnesses are hand-engineered: humans write the control flow, the
context management, and the tool routing. **predict-rlm is self-harnessed** —
the model writes and runs its own code, calls a sub-model when it needs to, and
manages its own context as it works, so capability scales with the underlying
model instead of with harness engineering, and
[without context rot](https://github.com/Trampoline-AI/predict-rlm). (It's an
implementation of the
[Recursive Language Models](https://arxiv.org/abs/2512.24601v1) work from MIT
CSAIL.)

Fractal adds exactly one thing on top: **session management** — multi-turn
conversation and history, which predict-rlm doesn't do on its own. That is the
whole product. Each turn is a single RLM call over your workspace, mounted into
a Docker sandbox so the model's own code and project commands run against the
real files.

What it's good for
------------------

Because the RLM reasons over context programmatically instead of stuffing
everything into one prompt, Fractal is strongest on analysis- and
context-heavy work: reading across a large or deep codebase, synthesizing an
answer from many files, audits, and open-ended investigation. Two ways to use
it:

- **Directly**, as your own terminal agent — ask questions, edit code, run
  tasks.
- **As a tool other agents defer to** — your main agent (Claude Code, Cursor,
  etc.) can hand a heavy analysis or large-context job to Fractal in
  [headless mode](#headless--ci-use) and get back a distilled answer. The
  bundled [`fractal-headless` skill](.agents/skills/fractal-headless/SKILL.md)
  teaches an agent when and how to do this.

Fractal is not trying to replace your daily coding agent — more mature tools
exist for that. It's a window onto what a self-harnessed RLM can do as an agent.

Requirements
------------

- **Python 3.11+**.
- **[uv](https://docs.astral.sh/uv/)** to install and run Fractal.
- **Docker**, running. Every Fractal turn executes generated code inside a
  Docker Sandbox, so the Docker daemon must be up.
- **The `sbx` CLI, logged in.** Fractal uses predict-rlm's `sbx` (Docker
  Sandboxes) backend for code execution:

  ```bash
  brew install docker/tap/sbx
  sbx login
  ```

  If Docker is not running or `sbx` is not logged in, the first turn fails. You
  can verify the rest of your setup (provider, model, auth) ahead of time with
  `fractal config status`.
- **A model provider.** One of the providers in the table below, with its API
  key available (or `codex login` for `openai-codex`, or a local Ollama
  server). Setup walks you through this on first run.

Installation
------------

The quickest way to install the `fractal` command:

```bash
curl -LsSf https://fractal.trampoline.ai/install.sh | sh
```

The script installs [uv](https://docs.astral.sh/uv/) if needed, installs
Fractal as an isolated tool, and checks your Docker/`sbx` prerequisites. To
pin a version, set `FRACTAL_VERSION`.

If you already use uv or pipx, install the tool directly instead:

```bash
uv tool install fractal      # or: pipx install fractal
fractal --help
```

Both create an isolated environment and put `fractal` on your PATH, so it is
callable from any directory.

To work on Fractal itself, clone the repository and use uv:

```bash
git clone git@github.com:Trampoline-AI/fractal.git
cd fractal
uv sync
uv run fractal --help
```

When running from a checkout, prefix the commands below with `uv run`
(e.g. `uv run fractal`); an installed tool just uses `fractal`.

Quickstart
----------

```bash
cd your-project
fractal                       # first run launches provider/model setup, then a session
```

On first interactive run with no global config, Fractal runs setup
automatically: pick a provider, a model, an optional cheaper sub-model, and how
to supply the API key. After setup you land in an interactive session in the
current directory. Type a request, and Fractal edits the workspace and reports
what it changed. Use `/help` to list slash commands and `/exit` to quit.

Usage
-----

```bash
fractal                       # interactive session in the current directory
fractal -p "fix the tests"    # one non-interactive turn
fractal --resume <session-id> # resume a stored workspace session
```

Interactive slash commands: `/help`, `/sessions`, `/resume <id>`, `/new`,
`/model`, `/provider`, `/usage`, `/verbose`, `/exit`. The header always shows
both the main model and the sub-model.

### Command-line options

| Flag | Description |
| --- | --- |
| `--workspace DIR` | Workspace directory to edit; defaults to the current directory. |
| `--include DIR` | Additional directory to mount into the sandbox at its absolute path. Repeatable. |
| `-p`, `--prompt TEXT` | Run one turn non-interactively with `TEXT`; use `-` to read the prompt from stdin. |
| `--resume SESSION_ID` | Resume an existing workspace-local session by id. |
| `--max-iterations N` | Max RLM iterations per turn; defaults to the configured value or 30. |
| `--lm MODEL` | Override the configured main model for this run (bypasses config resolution). |
| `--sub-lm MODEL` | Override the configured sub-model for this run. |
| `--verbose` | Show generated code and model-visible output for each RLM iteration. |
| `--quiet` | Suppress progress chatter (non-interactive runs). |
| `--debug` | Enable PredictRLM debug mode. |

Subcommands: `fractal config <show|status|setup|get|set|unset|reset>` manage
configuration (see [Configuration](#configuration)).

### Headless / CI use

`-p`/`--prompt` runs a single turn without the interactive UI, which is the
mode to use from scripts, hooks, and CI:

```bash
fractal -p "fix the failing tests"          # one turn, prompt as an argument
git diff | fractal -p -                      # read the entire prompt from stdin
echo "summarize recent changes" | fractal -p "review this diff"  # prompt + stdin context
```

How non-interactive runs behave:

- The agent's reply is written to **stdout**; progress, the workspace/session
  banner, changed-file lists, and usage go to **stderr**. This lets you pipe
  the response cleanly while still seeing diagnostics.
- Add `--quiet` to silence everything but the final stdout response.
- Stdin input is capped at 10 MiB.
- Exit codes: `0` success, `1` error (bad input, setup/runtime failure),
  `2` the turn hit `--max-iterations` before completing, `130` interrupted.
- A provider must already be configured. Headless runs do **not** trigger
  interactive setup when stdin is not a TTY; configure first with
  `fractal config setup`, or pin a model inline with `--lm`. Environment
  variables (`FRACTAL_PROVIDER`, `FRACTAL_MODEL`, …) are convenient for CI —
  see [Configuration](#configuration).
- Docker must be running and `sbx` logged in on the runner, exactly as for
  interactive use.

After each turn Fractal shows host-recorded facts: iterations, wall time,
tokens in/out, the current RLM context size, billed cost, and changed files.
Because the RLM loop re-summarizes between turns, "context" is the prompt size
of the latest main-LM call rather than a cumulative count. `/usage` reports
session totals, which persist in `.fractal/sessions/<session-id>.json` and
survive `--resume`.

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

For scripts and quick edits there is non-interactive dotted-key access. `set`
parses TOML literals (`12`, `true`) and falls back to strings; values are
validated against the schema before anything is written, and raw secrets are
rejected. `--project` targets the workspace config instead of the global one:

```bash
uv run fractal config get active_model
uv run fractal config set active_model gpt-5.4-mini
uv run fractal config set defaults.max_iterations 12
uv run fractal config set active_model gpt-5.4 --project
uv run fractal config unset active_sub_model
```

Setting `active_model` or `active_sub_model` warns when the model is not in
the provider's known catalog, and refuses ids the provider restricts.

To start over, `config reset` deletes the global config after confirmation
(`--yes` skips the prompt); add `--credentials` to also delete locally stored
API keys. Project configs are never touched by reset:

```bash
uv run fractal config reset
uv run fractal config reset --credentials --yes
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
API key; setup asks for the server URL (default `http://localhost:11434`) and
queries `/api/tags` so models you have actually pulled are listed first,
marked "(installed)", falling back to static suggestions when the server is
not running.

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
`FRACTAL_PROVIDER` / `FRACTAL_MODEL` / `FRACTAL_SUB_PROVIDER` /
`FRACTAL_SUB_MODEL` / `FRACTAL_MAX_ITERATIONS` / `FRACTAL_VERBOSE`
environment variables, with CLI
flags on top. A repo can pin its model without touching anyone's global
config, and CI can override via env. `fractal config show` lists which layers
contributed. Environment overrides apply only once some config file exists,
so first-run onboarding still triggers.

Beyond the active provider and model, the config supports:

```toml
# optional: a cheaper model for RLM sub-calls; chosen during setup and /model,
# defaults to the main model
active_sub_model = "gpt-5.4-mini"
# optional: run the sub-model on a different provider (its auth is collected
# during setup too); defaults to the main provider
active_sub_provider = "groq"

[defaults]            # optional run defaults, overridden by CLI flags
max_iterations = 30   # --max-iterations
verbose = false       # --verbose
```

The config can hold several provider profiles at once. Setup merges into the
existing `providers` table instead of replacing it, marks already-configured
providers in the menu, defaults to the active one, and offers to keep their
saved auth — so switching back to a configured provider is just two prompts
(provider, model), and `fractal config set active_provider <id>` switches
non-interactively. Switching providers clears `active_sub_model`; run
defaults are preserved.

Inside the interactive session, `/provider` re-runs provider setup and
`/model` switches models for the configured providers, and `/verbose`
toggles trace display. Setup walks main provider → main model → sub-model
provider (defaulting to "same as main provider") → sub-model, then collects
auth for each distinct provider; `/model` changes only the two models within
their providers.

For one-off runs or tests, `--lm` bypasses global config resolution:

```bash
uv run fractal --lm openai/gpt-5.5 -p "summarize this repo"
```

Troubleshooting
---------------

- **A turn fails immediately / "sandbox" errors.** Docker is not running or
  `sbx` is not logged in. Start Docker, run `sbx login`, then retry. After an
  interrupted shutdown a sandbox can be left behind — list with `sbx ls` and
  remove with `sbx rm --force <name>`.
- **`fractal config status` reports the provider isn't ready.** The API key is
  missing or invalid. Re-run `fractal config setup`, or export the environment
  variable it names. Use `--offline` to skip the live provider check.
- **First run doesn't start setup.** Setup auto-runs only on an interactive
  (TTY) first run with no global config. In a non-interactive context, run
  `fractal config setup` explicitly or pass `--lm`.
- **A newly released model is rejected.** Most providers accept a free-form id
  via the "Custom model..." menu entry or `fractal config set active_model
  <id>`; only providers with `restricted_models` refuse unknown ids.

Development
-----------

```bash
uv sync                # install dependencies
uv run fractal --help
uv run pytest          # 200+ tests
```

See [CONTRIBUTING.md](CONTRIBUTING.md) for the contribution workflow and
[CHANGELOG.md](CHANGELOG.md) for release notes.

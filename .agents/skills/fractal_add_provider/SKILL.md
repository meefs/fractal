---
name: fractal_add_provider
description: Guide contributors through adding or updating Fractal model providers, provider auth wiring, and provider model options. Use when adding a Fractal provider, changing provider defaults/model_options/restricted_models, updating setup model menus, or preparing provider-related commits and PRs.
---

# Fractal Provider Changes

## Start Here

1. Inspect the current branch and worktree with `git status --short`; preserve unrelated user changes.
2. Read `src/fractal/providers.py`, `src/fractal/onboarding.py`, `src/fractal/config.py`, `tests/test_providers.py`, `tests/test_cli_config.py`, and `README.md`.
3. For model IDs, check current provider documentation or catalogs before editing. Do not rely on memory for active model names.
4. Decide whether the task is only a model-menu update or a new provider/runtime behavior.

## Provider Registry Pattern

Most provider changes live in `src/fractal/providers.py`.

- Add a provider id constant near the existing provider constants.
- Add the provider to `_PROVIDERS` with a `ProviderDefinition`.
- Use `default_model` for the default menu selection.
- Use `model_options` for curated alternatives only; do not duplicate `default_model`.
- Use `restricted_models` only when Fractal must reject all other model IDs at runtime.
- Set `model_prefix` to the DSPy/LiteLLM provider prefix when runtime strings need normalization, such as `openai`, `anthropic`, or `openrouter`.
- Keep `model_options` curated. It is a setup menu, not an exhaustive provider catalog.

Provider behavior choices:

- API key plus LiteLLM-compatible string: reuse `ApiKeyStringLMBehavior`.
- OpenAI-compatible custom endpoint: follow `CustomOpenAICompatibleBehavior`.
- CLI/OAuth-backed provider: add a dedicated behavior class that validates shape, checks readiness without leaking secrets, and builds the runtime LM object.

Secrets must not be stored in Fractal config. Store references only: env var names, auth source names, or paths to external auth stores.

## Model ID Rules

- OpenAI API options should be OpenAI model IDs without `openai/`; `_normalize_model` adds the prefix.
- Anthropic options should be Claude API IDs without `anthropic/`; `_normalize_model` adds the prefix.
- OpenRouter options should be OpenRouter catalog IDs like `openai/gpt-5.5` or `anthropic/claude-sonnet-4.6`; `_normalize_model` adds `openrouter/`.
- Custom OpenAI-compatible endpoints cannot have a complete static model catalog. Keep a custom model entry in onboarding.
- If a provider has a dynamic catalog, do not hardcode the full live response into `model_options`; pick stable, coding-relevant defaults and document that the list is curated.

## Onboarding And Config

New providers appear in setup through `list_providers()`. Update onboarding only when the provider needs extra inputs beyond base URL and API-key env var.

- Provider/model menus use `model_choices()`.
- The line-mode fallback must keep working for non-TTY tests and automation.
- For any new config fields, update `ProviderConfig`, redaction/rendering, schema validation, and tests.
- Never add raw secret fields such as tokens, API keys, passwords, or credentials to config.

## Tests

Add or update focused tests before broad tests:

- `tests/test_providers.py`: registry membership, `model_choices`, model normalization, shape validation, credential readiness, and unsupported model errors.
- `tests/test_cli_config.py`: setup output/input flow, model selection, config writes, and secret redaction.
- If config schema changes, update `tests/test_config.py`.
- If runtime behavior changes outside provider resolution, add narrow smoke/runtime coverage.

Validation commands:

```bash
.venv/bin/python -m pytest tests/test_providers.py tests/test_cli_config.py
.venv/bin/python -m pytest
```

## Docs

Update `README.md` when a provider, auth source, default credential reference, or setup model menu changes. State when a menu is curated rather than exhaustive.

## Commit And PR Workflow

Only commit when the user asks.

1. Re-check `git status --short` and stage only files changed for this task.
2. Commit with a provider-scoped message, for example `Add <provider> provider support` or `Update <provider> model options`.
3. Push the current branch only when the user asks.
4. Open a PR only when the user asks, using the repo's normal CLI if available, usually `gh pr create --fill`.
5. In the PR body, include provider behavior, auth/secrets handling, model sources, and tests run.

If current branch ownership is unclear, ask before creating, switching, pushing, or opening a PR.

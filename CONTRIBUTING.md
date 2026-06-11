# Contributing to Fractal

Thanks for helping build Fractal. This guide covers the local setup and the
conventions we hold to.

## Local setup

```bash
git clone git@github.com:Trampoline-AI/fractal.git
cd fractal
uv sync
```

You also need Docker running and the `sbx` CLI logged in to exercise agent
turns — see the [README requirements](README.md#requirements).

## Running checks

```bash
uv run pytest                 # full suite
uv run pytest tests/test_providers.py   # a focused module
uv run ruff check .           # lint
```

Tests that require `predict_rlm.Workspace` skip cleanly when that interface is
unavailable, so a partial environment still runs the rest of the suite.

## Conventions

These mirror `AGENTS.md`, which is the source of truth for engineering
guidelines:

- **Keep changes narrowly scoped.** The repo is small; avoid broad
  abstractions before the behavior needs them.
- **Push fixes to the right layer.** If a problem is caused by a PredictRLM
  interface, sandbox, serialization, or trace limitation, fix `predict-rlm`
  directly instead of adding a Fractal workaround. Record observed predict-rlm
  issues in `docs/predict-rlm-notes.md`.
- **Prefer host-side truth** over model-reported truth for state, changed
  files, commands run, and verification status.
- **Never store secrets in config.** Config holds references only (env var
  names, auth-source names, paths). Pasted keys go to
  `~/.config/fractal/credentials.toml` at `0600`.
- **Test behavior, not presentation.** Avoid assertions on exact terminal
  wrapping, whitespace, box-drawing, or color internals; check semantic content
  or normalized text instead.
- **Comment the why,** not the mechanics — especially around session state,
  prompt wiring, persistence, recovery, and safety.

## Commits and pull requests

- Commit only when it is part of the task you were asked to do.
- Use focused, scoped commit messages (e.g. `Add <provider> provider support`,
  `Document headless usage`).
- In PR descriptions, note behavior changes, any secrets/auth handling, and the
  tests you ran.
- Update `README.md` and `CHANGELOG.md` when user-facing behavior, providers,
  flags, or configuration change.

## Adding a provider

See the `fractal_add_provider` skill under
`.agents/skills/fractal_add_provider/SKILL.md` for the full registry pattern,
model-id rules, and the tests to update.

---
name: fractal-headless
description: Delegate analysis- and context-heavy work to Fractal, an agentic CLI powered by a self-harnessed Recursive Language Model (predict-rlm), by running it non-interactively (fractal -p). Reach for it when a task needs reasoning over a large or deep codebase, synthesizing an answer across many files, auditing, or open-ended investigation — work that would otherwise flood your own context. The RLM reasons over context programmatically (no context rot) and returns a distilled answer. Use when asked to run Fractal headless, script it, call it from CI or another agent, or offload a heavy analysis/large-context task. Less suited to trivial single-file edits you can do directly.
---

# Fractal headless mode

Fractal is an agentic CLI powered by [predict-rlm](https://github.com/Trampoline-AI/predict-rlm),
a self-harnessed Recursive Language Model runtime. Each invocation runs one RLM
turn: it mounts the workspace into a Docker sandbox, lets the model write and
run its own code to read/edit files and run commands, then prints a final
answer. Headless mode is `fractal -p "<task>"`.

## When to reach for it

An RLM reasons over context *programmatically* — it greps, reads, and slices
files in code instead of loading everything into one prompt — so it handles
large or deep inputs without context rot. Defer to Fractal when a task is
analysis- or context-heavy and would otherwise flood your own context:

- "Trace how a request flows through this 200-file service, end to end."
- "Audit this repo for `X` across every file and summarize the findings."
- "Read these logs / this diff / these docs and tell me the root cause."
- Synthesis or Q&A over a directory too big to pull into your own window.

It also works as a general coding agent (fix a bug, make tests pass) — use that
when convenient. But skip it for trivial, local edits you can do yourself in a
file read or two: each turn is slow (minutes) and separately billed, so the
payoff is in the heavy jobs.

## Quick start

```bash
fractal -p "Trace how a request flows through this service, end to end" --workspace /path/to/project
```

- **stdout** = the agent's final response text, nothing else.
- **stderr** = banner, progress, changed files, usage/cost, completion status.
- From a source checkout, use `uv run --project /path/to/fractal fractal ...` so
  the invoking cwd doesn't matter. Beware: if uv fails to spawn (wrong dir, no
  venv) it exits 2, which collides with Fractal's own max-iterations exit code —
  distinguish by checking stderr for `error: Failed to spawn`.

## Preflight (do once per machine/session)

```bash
docker info >/dev/null || echo "start Docker first"   # sandbox needs the daemon
fractal config status                                  # provider/model/auth ok?
```

If config is missing, `fractal config setup` is interactive — in automation use
`fractal config set ...` or the `FRACTAL_PROVIDER` / `FRACTAL_MODEL` env vars.

## Output contract

| Channel | Content |
| --- | --- |
| stdout | Final response text only (trailing newline guaranteed) |
| stderr | `fractal: session <id>`, progress events, `fractal: changed files a.py, b.py`, `fractal: usage N in / M out tokens, $X`, `fractal: complete` |
| exit 0 | Turn completed |
| exit 1 | Setup/runtime error (`fractal: failed: ...` on stderr) |
| exit 2 | Hit `--max-iterations`; best-effort response still on stdout |
| exit 130 | Interrupted (Ctrl-C / SIGINT) |

For scripting, capture streams separately: `out=$(fractal -p "..." 2>err.log)`.

## Key flags

- `--workspace DIR` — directory the agent edits. **Always pass it explicitly**; default is the cwd.
- `--quiet` — suppress all stderr. Stdout-only, but you lose progress, the session id, changed-files, and cost. Prefer capturing stderr to a file over `--quiet` when you need to resume or audit.
- `--verbose` — full per-iteration trace (reasoning, code, output) on stderr. Use when debugging why a turn did something.
- `-p -` — read the whole prompt from stdin (fails fast if stdin is empty). Piping stdin alongside a normal `-p "..."` instead appends it as context: `git diff | fractal -p "review this diff"`.
- `--resume SESSION_ID` — continue a prior session (multi-turn memory works headless). Get the id from the stderr line `fractal: session <id>`, or the newest file in `<workspace>/.fractal/sessions/`.
- `--include DIR` — mount extra read/write dirs into the sandbox (repeatable).
- `--lm` / `--sub-lm` / `--max-iterations` — per-run overrides; env equivalents `FRACTAL_MODEL`, `FRACTAL_SUB_MODEL`, `FRACTAL_MAX_ITERATIONS`.

## Gotchas (battle-tested)

- **Redirect stdin (`< /dev/null`) whenever you aren't piping data in.** Recent
  builds wait at most 1s before ignoring a silent non-TTY stdin, but older
  builds block forever reading it to EOF (CI runners, process managers, agent
  harnesses keep stdin open) — and with `--quiet` that hang is silent. The
  redirect is the safe habit on every version:

  ```bash
  fractal -p "task" --workspace "$ws" < /dev/null
  ```

  Implicit piped context must start arriving within ~1s of launch; if your
  producer is slower, buffer it first (`fractal -p "..." <<<"$(slow_cmd)"`).

- **Turns are slow and billed.** A trivial turn ≈ 1–2 min and ~$0.04; real tasks run several minutes and tens of cents. Wrap calls in your own timeout (≥10 min) — there is no built-in one, and with `--quiet` a hang is indistinguishable from a slow run.
- **Don't send an empty prompt.** `-p ""` still makes a full billed LLM call just to answer "no request provided".
- `fractal: command failed: ...` lines on stderr are the agent probing (e.g. trying `pytest` variants) — not fatal; only trust the exit code and `fractal: failed:`.
- A bad `--lm` id fails with exit 1 only after ~3 LiteLLM retries (slow); validate model ids beforehand with `fractal config status`.
- No JSON output mode exists. Parse changed files from the stderr `fractal: changed files ...` line, or diff the workspace with git yourself (recommended: run in a clean git tree).

See [RECIPES.md](RECIPES.md) for copy-paste patterns: CI gate, multi-turn driver, diff review, structured-ish output.

# Fractal headless recipes

All examples assume `fractal` is on PATH (else
`uv run --project /path/to/fractal fractal`) and Docker is running.

## Delegate a task and capture everything

```bash
ws=/path/to/project
out=$(fractal -p "Fix the bug in calc.py and make tests pass" \
        --workspace "$ws" </dev/null 2>fractal-err.log)
code=$?
session=$(grep -m1 '^fractal: session ' fractal-err.log | awk '{print $3}')
changed=$(grep -m1 '^fractal: changed files ' fractal-err.log | sed 's/^fractal: changed files //')
echo "exit=$code session=$session changed=$changed"
echo "$out"
```

## Multi-turn conversation (resume)

```bash
fractal -p "Audit auth.py for security issues" --workspace "$ws" </dev/null 2>err1.log
sid=$(grep -m1 '^fractal: session ' err1.log | awk '{print $3}')
fractal -p "Now fix the worst issue you found" --resume "$sid" --workspace "$ws" </dev/null
```

Session state lives in `<workspace>/.fractal/sessions/<sid>.json` and includes
turn history, token totals, and cost — readable JSON if you need an audit trail.

## Pipe context in (logs, diffs, issues)

```bash
git diff main | fractal -p "Review this diff; list real bugs only" --workspace "$ws"
tail -200 app.log | fractal -p "Why is this service crashing?" --workspace "$ws"
```

For a fully dynamic prompt, use `-p -`:

```bash
printf 'Rename every occurrence of %s to %s\n' "$old" "$new" | fractal -p - --workspace "$ws"
```

## CI gate

```bash
fractal -p "Run the test suite; fix any failures you find" \
  --workspace "$PWD" --max-iterations 20 </dev/null 2>fractal.log
case $? in
  0) git diff --exit-code || open_pr_with_changes ;;
  2) echo "::warning::Fractal ran out of iterations"; cat fractal.log ;;
  *) echo "::error::Fractal failed"; cat fractal.log; exit 1 ;;
esac
```

Exit 2 still prints a best-effort response on stdout — decide per-pipeline
whether to treat it as soft-fail.

## Structured-ish output

There is no `--json` flag. Two reliable options:

1. Ask for it in the prompt and parse stdout (works well in practice):

   ```bash
   fractal --quiet --workspace "$ws" -p 'List every TODO in this repo.
   Respond with ONLY a JSON array of {"file": ..., "line": ..., "text": ...}.' | jq .
   ```

2. Trust the filesystem, not the prose: run in a clean git tree and use
   `git -C "$ws" diff` / `git status --porcelain` after the run to see exactly
   what changed.

## Read-only questions

Fractal will edit files if the prompt implies it. For pure questions, say so:

```bash
fractal --quiet --workspace "$ws" \
  -p "How does session persistence work in this repo? Do not modify any files."
```

## Tuning

- `--max-iterations N` — cap runaway turns (default 30). Exit 2 when hit.
- `--lm provider/model`, `--sub-lm provider/model` — per-run model override,
  e.g. a cheaper sub-LM for summarization.
- `--include /other/repo` — give the agent a second directory (mounted at its
  real absolute path, writable).
- `--verbose` — full reasoning/code/output trace per iteration on stderr; the
  only way to see *why* the agent did something without replaying the session
  JSON.

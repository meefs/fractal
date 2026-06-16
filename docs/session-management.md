# Session Management

Fractal stores workspace-local session state under `.fractal/sessions/`.

Each new session gets a random `session_id` and is saved as
`.fractal/sessions/<session_id>.json`. For now the CLI starts a new session when
it starts; later resume support can select an existing session ID and load that
specific file.

The session has two memory layers:

1. Structured session summary
2. Full session history

## Structured Session Summary

The summary is a compressed trajectory artifact, not prose memory. It preserves
the ordered user and agent turns from the session:

- user message
- agent status
- agent response
- files read from runtime hook events, when available
- files modified
- commands run from runtime hook events, when available
- error, when the turn failed

For each Fractal turn, the summary is rendered into the dynamic PredictRLM
signature instructions. It is intentionally not a DSPy input field. PredictRLM
currently exposes input fields primarily as Python REPL variables with prompt
previews; the summary needs to be always visible to the main model, so Fractal
embeds the rendered summary in the prompt text.

## Full Session History

The full history stores prior PredictRLM traces. These traces include the REPL
reasoning, generated code, output, tool calls, predict calls, usage, and status.

Full history is passed to PredictRLM as `session_history`, a normal input
variable. The model can inspect it from Python when it needs exact prior code,
outputs, failed attempts, or tool call details.

## Turn Lifecycle

Before each RLM call, Fractal appends a pending summary turn and pending history
turn, then saves the session.

On success, Fractal stores the assistant response, changed files, and full trace,
then marks both records as succeeded.

On failure, Fractal stores the error and any trace attached to the exception,
then marks both records as failed.

## Known Limits

- File reads and commands are tracked only when the active PredictRLM backend
  supports runtime hook events.
- History is bounded and passed directly as structured data for now.
- PredictRLM does not yet have a first-class prompt-only context field separate
  from REPL variables, so Fractal uses dynamic signature instructions for
  always-visible memory.

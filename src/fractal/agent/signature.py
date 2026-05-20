from __future__ import annotations

import dspy
from predict_rlm import Workspace

from fractal.session import SessionHistoryTurn


BASE_EDIT_WORKSPACE_INSTRUCTIONS = """Act as a focused coding agent over the mounted workspace.

You receive:
- `workspace`: mutable project workspace mounted at /sandbox/workspace.
- `user_message`: the user's current request.
- `session_history`: detailed prior Fractal turn history with PredictRLM traces.

Inspect and edit files only through Python code in /sandbox/workspace. Prefer
pathlib/os operations rooted at /sandbox/workspace, and prefer os.open with
dir_fd/root_fd, os.pread/os.pwrite/os.ftruncate, and temp-file plus os.replace
patterns when they make edits safer.

Keep changes focused on the current user request. Inspect files before modifying
them, preserve unrelated content, and verify important edits. Return only a
concise user-facing response and a list of relative changed file paths.
"""


def build_edit_workspace_signature(
    rendered_session_summary: str,
) -> type[dspy.Signature]:
    """Build the per-turn Fractal coding-agent signature."""

    # The summary must be visible before the RLM chooses to inspect variables,
    # so it is baked into the instructions for this specific turn.
    summary = rendered_session_summary.strip() or "No prior Fractal session context."
    instructions = f"""{BASE_EDIT_WORKSPACE_INSTRUCTIONS}

## Always-visible session summary

{summary}

The summary above is compressed structured trajectory context and is always
visible. It preserves prior user messages and compressed agent results. For
exact prior REPL reasoning, code, outputs, tool calls, or predict calls, inspect
`session_history` from Python.
"""

    # The rendered session summary is intentionally embedded in the signature
    # instructions instead of declared as an InputField. PredictRLM currently
    # exposes InputFields primarily as REPL variables with prompt previews; for
    # always-visible memory we need prompt text. A future PredictRLM API may
    # support explicit prompt-only context fields separate from REPL variables.
    class EditWorkspaceWithSession(dspy.Signature):
        __doc__ = instructions

        workspace: Workspace = dspy.InputField(
            desc=(
                "Project workspace mounted at /sandbox/workspace and synced "
                "after code blocks."
            )
        )
        user_message: str = dspy.InputField(
            desc="The user's current request for this turn."
        )
        session_history: list[SessionHistoryTurn] = dspy.InputField(
            desc=(
                "Full prior Fractal turn history, including PredictRLM traces, "
                "for exact recall from Python."
            )
        )

        response: str = dspy.OutputField(
            desc=(
                "Concise Markdown-formatted response to show in the CLI. Use "
                "Markdown for bullets, code spans, and short code blocks when helpful."
            )
        )
        changed_files: list[str] = dspy.OutputField(
            desc="Relative paths of files changed in /sandbox/workspace."
        )

    return EditWorkspaceWithSession


EditWorkspace = build_edit_workspace_signature("No prior Fractal session context.")

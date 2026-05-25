from __future__ import annotations

from pathlib import Path
from typing import Any

import dspy
from predict_rlm import PredictRLM, RunTrace, Workspace, WorkspaceMode

from .schema import FractalResult
from .signature import build_edit_workspace_signature
from .skills import filesystem_coding_skill
from ..session import SessionHistoryTurn


class FractalAgent(dspy.Module):
    """Thin DSPy module wrapping Fractal's workspace-editing RLM."""

    def __init__(
        self,
        lm: dspy.LM | str | None = None,
        sub_lm: dspy.LM | str | None = None,
        max_iterations: int = 30,
        verbose: bool = True,
        debug: bool = False,
    ) -> None:
        self.lm = lm
        self.sub_lm = sub_lm
        self.max_iterations = max_iterations
        self.verbose = verbose
        self.debug = debug

    async def aforward(
        self,
        workspace_path: str | Path,
        user_message: str,
        rendered_session_summary: str = "",
        session_history: list[SessionHistoryTurn] | None = None,
        included_paths: list[str | Path] | None = None,
    ) -> FractalResult:
        workspace = Workspace(
            path=str(Path(workspace_path).resolve()),
            mode=WorkspaceMode.DIRECT,
        )
        if ".fractal" not in workspace.exclude:
            workspace.exclude = [*workspace.exclude, ".fractal"]
        included_workspaces = [
            Workspace(
                path=str(Path(path).resolve()),
                mode=WorkspaceMode.DIRECT,
            )
            for path in included_paths or []
        ]

        signature = build_edit_workspace_signature(rendered_session_summary)
        predictor = PredictRLM(
            signature,
            lm=self.lm,
            sub_lm=self.sub_lm,
            skills=[filesystem_coding_skill],
            max_iterations=self.max_iterations,
            verbose=self.verbose,
            debug=self.debug,
            sandbox_backend="sbx",
        )
        result = await predictor.acall(
            workspace=workspace,
            included_paths=included_workspaces or None,
            user_message=user_message,
            session_history=session_history or [],
        )
        return _coerce_result(result)


def _coerce_result(prediction: Any) -> FractalResult:
    response = str(getattr(prediction, "response", "") or "")
    changed_files = _coerce_changed_files(getattr(prediction, "changed_files", None))
    return FractalResult(
        response=response,
        changed_files=changed_files,
        trace=coerce_trace(getattr(prediction, "trace", None)),
    )


def coerce_trace(trace: Any) -> RunTrace | None:
    # Fractal persists PredictRLM's trace as typed state. Normalize at the
    # service boundary so session code never handles an untyped trace blob.
    if trace is None:
        return None
    if isinstance(trace, RunTrace):
        return trace
    if isinstance(trace, dict):
        return RunTrace.model_validate(trace)
    if hasattr(trace, "model_dump"):
        return RunTrace.model_validate(trace.model_dump(mode="python"))
    raise TypeError(f"Unsupported PredictRLM trace type: {type(trace).__name__}")


def _coerce_changed_files(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    if isinstance(value, (list, tuple, set)):
        return [str(path) for path in value]
    if isinstance(value, dict):
        raise TypeError(
            "changed_files must be a string or a sequence of paths, not a dict"
        )
    try:
        return [str(path) for path in list(value)]
    except TypeError:
        return [str(value)]

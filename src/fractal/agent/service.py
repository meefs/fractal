from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Protocol, cast

import dspy
from dspy.utils.callback import BaseCallback
from predict_rlm import PredictRLM, RunTrace, Workspace, WorkspaceMode
from predict_rlm.interpreters import PredictRLMInterpreter, SbxInterpreter
from predict_rlm.workspace import DirectWorkspaceMount

from .schema import FractalIterationEvent, FractalResult
from .signature import build_edit_workspace_signature
from .skills import filesystem_coding_skill
from ..events import build_predict_runtime_hooks
from ..lm_types import RuntimeLM
from ..session import SessionHistoryTurn


class FractalInterpreter(PredictRLMInterpreter, Protocol):
    def prewarm(self) -> None: ...


class FractalAgent(dspy.Module):
    """Thin DSPy module wrapping Fractal's workspace-editing RLM."""

    def __init__(
        self,
        lm: RuntimeLM | None = None,
        sub_lm: RuntimeLM | None = None,
        max_iterations: int = 30,
        verbose: bool = True,
        debug: bool = False,
        interpreter: FractalInterpreter | None = None,
    ) -> None:
        self.lm = lm
        self.sub_lm = sub_lm
        self.max_iterations = max_iterations
        self.verbose = verbose
        self.debug = debug
        self.interpreter = interpreter

    async def aforward(
        self,
        workspace_path: str | Path,
        user_message: str,
        rendered_session_summary: str = "",
        session_history: list[SessionHistoryTurn] | None = None,
        included_paths: list[str | Path] | None = None,
        on_runtime_event: Callable[[object], object] | None = None,
        on_iteration_event: Callable[[FractalIterationEvent], object] | None = None,
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
        predictor_kwargs: dict[str, object] = {
            "lm": self.lm,
            "sub_lm": self.sub_lm,
            "skills": [filesystem_coding_skill],
            "max_iterations": self.max_iterations,
            "verbose": self.verbose,
            "debug": self.debug,
        }
        if self.interpreter is None:
            predictor_kwargs["sandbox_backend"] = "sbx"
        else:
            predictor_kwargs["interpreter"] = self.interpreter
        runtime_hooks = build_predict_runtime_hooks()
        if runtime_hooks:
            predictor_kwargs["runtime_hooks"] = runtime_hooks
            if on_runtime_event is not None:
                predictor_kwargs["on_runtime_hook_event"] = on_runtime_event

        predictor = PredictRLM(signature, **predictor_kwargs)
        if on_iteration_event is not None:
            predictor.callbacks = [
                *list(getattr(predictor, "callbacks", []) or []),
                _FractalIterationCallback(
                    max_iterations=self.max_iterations,
                    on_iteration_event=on_iteration_event,
                ),
            ]
        result = cast(
            dspy.Prediction,
            await predictor.acall(
                workspace=workspace,
                included_paths=included_workspaces or None,
                user_message=user_message,
                session_history=session_history or [],
            ),
        )
        return _prediction_to_result(result)

    def close(self) -> None:
        if self.interpreter is not None:
            self.interpreter.shutdown()

    def prewarm(self) -> None:
        if self.interpreter is not None:
            self.interpreter.prewarm()


def create_sbx_interpreter(
    workspace_path: str | Path,
    included_paths: list[str | Path] | None = None,
) -> SbxInterpreter:
    return SbxInterpreter(
        direct_workspace_mounts=build_direct_workspace_mounts(
            workspace_path,
            included_paths,
        )
    )


def build_direct_workspace_mounts(
    workspace_path: str | Path,
    included_paths: list[str | Path] | None = None,
) -> list[DirectWorkspaceMount]:
    paths = [Path(workspace_path), *[Path(path) for path in included_paths or []]]
    return [
        DirectWorkspaceMount(
            host_path=str(path.resolve()),
            sandbox_path=str(path.resolve()),
        )
        for path in paths
    ]


def _prediction_to_result(prediction: dspy.Prediction) -> FractalResult:
    response = prediction.response
    if not isinstance(response, str):
        raise TypeError("PredictRLM response must be a string.")

    changed_files = _require_string_list(
        prediction.changed_files,
        "PredictRLM changed_files",
    )
    trace = prediction.trace
    if trace is not None and not isinstance(trace, RunTrace):
        raise TypeError(f"PredictRLM trace must be RunTrace, not {type(trace).__name__}.")

    return FractalResult(
        response=response,
        changed_files=changed_files,
        trace=trace,
    )


def _require_string_list(value: object, field_name: str) -> list[str]:
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        raise TypeError(f"{field_name} must be list[str].")
    return value


class _FractalIterationCallback(BaseCallback):
    def __init__(
        self,
        *,
        max_iterations: int,
        on_iteration_event: Callable[[FractalIterationEvent], object],
    ) -> None:
        self.max_iterations = max_iterations
        self.on_iteration_event = on_iteration_event

    def on_rlm_iteration_end(
        self,
        *,
        step: object,
        is_final: bool,
        exception: BaseException | None,
        **_: object,
    ) -> None:
        if step is None or exception is not None:
            return
        try:
            self.on_iteration_event(
                FractalIterationEvent(
                    step=step,
                    max_iterations=self.max_iterations,
                    is_final=is_final,
                )
            )
        except Exception:
            pass

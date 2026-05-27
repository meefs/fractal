from __future__ import annotations

import os
import shlex
from dataclasses import dataclass, field
from typing import Any, Literal


RuntimeEventKind = Literal["file_read", "file_write", "command"]

FILE_HOOK_PHASES = {"before", "after", "error"}
COMMAND_HOOK_PHASES = {"before", "after", "error"}
MAX_COMMAND_DISPLAY_CHARS = 160
TRUNCATION_MARKER = "..."

FILE_HOOK_TARGETS: tuple[str, ...] = (
    "builtins.open",
    "pathlib.Path.open",
    "pathlib.Path.read_text",
    "pathlib.Path.read_bytes",
    "pathlib.Path.write_text",
    "pathlib.Path.write_bytes",
    "os.open",
    "os.pread",
    "os.pwrite",
    "os.ftruncate",
    "os.replace",
    "os.unlink",
)
OPEN_FILE_HOOK_TARGETS = {"builtins.open", "pathlib.Path.open", "os.open"}
PATH_METHOD_HOOK_TARGETS = {
    "builtins.open",
    "pathlib.Path.open",
    "pathlib.Path.read_text",
    "pathlib.Path.read_bytes",
    "pathlib.Path.write_text",
    "pathlib.Path.write_bytes",
}
# These methods call Path.open internally. Fractal surfaces the compound action
# once while still using the after hook to persist the durable file fact.
COMPOUND_PATH_HOOK_TARGETS = {
    "pathlib.Path.read_text",
    "pathlib.Path.read_bytes",
    "pathlib.Path.write_text",
    "pathlib.Path.write_bytes",
}
COMMAND_HOOK_TARGETS: tuple[str, ...] = (
    "subprocess.run",
    "subprocess.Popen",
    "subprocess.call",
    "subprocess.check_call",
    "subprocess.check_output",
)
COMPOUND_COMMAND_HOOK_TARGETS = {
    "subprocess.run",
    "subprocess.call",
    "subprocess.check_call",
    "subprocess.check_output",
}


@dataclass(slots=True)
class FractalRuntimeEvent:
    kind: RuntimeEventKind  # Broad event category used for styling and summary grouping.
    target: str  # Dotted Python function or method name that produced the hook event.
    phase: str  # Hook lifecycle phase: before, after, or error.
    message: str  # Operator-facing status line rendered by the CLI/TUI.
    path: str | None = None  # File path associated with file activity, when available.
    command: str | None = None  # Shell-style command string for subprocess activity.


@dataclass(slots=True)
class RuntimeHookSnapshot:
    """Fractal's normalized view of a PredictRLM runtime hook event."""

    target: str
    phase: str
    args: list[Any] = field(default_factory=list)
    kwargs: dict[str, Any] = field(default_factory=dict)
    result: Any = None
    error: str | None = None


@dataclass(slots=True)
class RuntimeEventTracker:
    """Turn-local reducer for low-level PredictRLM runtime hook events."""

    files_read: list[str] = field(default_factory=list)
    files_modified: list[str] = field(default_factory=list)
    commands_run: list[str] = field(default_factory=list)
    _fd_paths: dict[int, str] = field(default_factory=dict)
    _active_compound_paths: dict[str, int] = field(default_factory=dict)
    _active_compound_commands: dict[str, list[str]] = field(default_factory=dict)

    def observe(self, raw_event: object) -> FractalRuntimeEvent | None:
        event = adapt_runtime_hook_event(raw_event)
        target = event.target
        phase = event.phase
        args = event.args
        kwargs = event.kwargs
        result = event.result

        if target in COMMAND_HOOK_TARGETS:
            command = _command_from_args(args)
            if not command:
                command = target.removeprefix("subprocess.")
            if self._is_nested_compound_command(target, command):
                return None
            if phase == "before":
                _append_unique(self.commands_run, command)
            if target in COMPOUND_COMMAND_HOOK_TARGETS:
                if phase == "before":
                    self._begin_compound_command(command, target)
                try:
                    return self._command_event(target, phase, command)
                finally:
                    if phase in {"after", "error"}:
                        self._end_compound_command(command, target)
            return self._command_event(target, phase, command)

        if target == "os.open":
            path = _path_from_args(args)
            if self._is_nested_compound_path(target, path):
                return None
            if phase == "after" and path is not None:
                fd = _coerce_int(result)
                if fd is not None:
                    self._fd_paths[fd] = path
            mode = _os_open_mode(args)
            return self._file_event(
                target,
                phase,
                path,
                mode,
                surface=phase == "before",
            )

        if target in {"os.pread", "os.pwrite", "os.ftruncate"}:
            fd = _coerce_int(args[0]) if args else None
            path = self._fd_paths.get(fd) if fd is not None else None
            mode: Literal["read", "write"] = (
                "read" if target == "os.pread" else "write"
            )
            return self._file_event(
                target,
                phase,
                path,
                mode,
                surface=phase == "before",
            )

        if target in {"os.replace", "os.unlink"}:
            path = _path_from_args(args[1:2] if target == "os.replace" else args)
            return self._file_event(
                target,
                phase,
                path,
                "write",
                surface=phase == "before",
            )

        if target in PATH_METHOD_HOOK_TARGETS:
            path = _path_from_args(args)
            if self._is_nested_compound_path(target, path):
                return None
            mode = _path_target_mode(target, args, kwargs)
            if target in COMPOUND_PATH_HOOK_TARGETS:
                if path is not None and phase == "before":
                    self._begin_compound_path(path)
                try:
                    return self._file_event(
                        target,
                        phase,
                        path,
                        mode,
                        surface=phase == "before",
                    )
                finally:
                    if path is not None and phase in {"after", "error"}:
                        self._end_compound_path(path)
            return self._file_event(
                target,
                phase,
                path,
                mode,
                surface=phase == "before",
            )

        return None

    def _command_event(
        self,
        target: str,
        phase: str,
        command: str,
    ) -> FractalRuntimeEvent | None:
        if phase == "after":
            return None
        return FractalRuntimeEvent(
            kind="command",
            target=target,
            phase=phase,
            command=command,
            message=_command_message(phase, command),
        )

    def _file_event(
        self,
        target: str,
        phase: str,
        path: str | None,
        mode: Literal["read", "write", "read_write"],
        *,
        surface: bool,
    ) -> FractalRuntimeEvent | None:
        if path and phase == "after":
            if mode in {"read", "read_write"}:
                _append_unique(self.files_read, path)
            if mode in {"write", "read_write"}:
                _append_unique(self.files_modified, path)
        if not surface:
            return None
        kind: RuntimeEventKind = "file_read" if mode == "read" else "file_write"
        display_path = path or "file"
        if kind == "file_read" and target in OPEN_FILE_HOOK_TARGETS:
            verb = "opening"
        else:
            verb = "reading" if kind == "file_read" else "editing"
        return FractalRuntimeEvent(
            kind=kind,
            target=target,
            phase=phase,
            path=path,
            message=f"{verb} {display_path}",
        )

    def _begin_compound_path(self, path: str) -> None:
        self._active_compound_paths[path] = (
            self._active_compound_paths.get(path, 0) + 1
        )

    def _end_compound_path(self, path: str) -> None:
        depth = self._active_compound_paths.get(path, 0)
        if depth <= 1:
            self._active_compound_paths.pop(path, None)
        else:
            self._active_compound_paths[path] = depth - 1

    def _is_nested_compound_path(self, target: str, path: str | None) -> bool:
        return (
            path is not None
            and target not in COMPOUND_PATH_HOOK_TARGETS
            and self._active_compound_paths.get(path, 0) > 0
        )

    def _begin_compound_command(self, command: str, target: str) -> None:
        self._active_compound_commands.setdefault(command, []).append(target)

    def _end_compound_command(self, command: str, target: str) -> None:
        stack = self._active_compound_commands.get(command)
        if not stack:
            return
        if stack[-1] == target:
            stack.pop()
        else:
            try:
                stack.remove(target)
            except ValueError:
                pass
        if not stack:
            self._active_compound_commands.pop(command, None)

    def _is_nested_compound_command(self, target: str, command: str) -> bool:
        stack = self._active_compound_commands.get(command)
        return bool(stack and stack[-1] != target)


def build_predict_runtime_hooks() -> list[Any]:
    try:
        from predict_rlm import RuntimeHook
    except ImportError:
        return []

    return [
        *[
            RuntimeHook(target=target, phases=FILE_HOOK_PHASES)
            for target in FILE_HOOK_TARGETS
        ],
        *[
            RuntimeHook(target=target, phases=COMMAND_HOOK_PHASES)
            for target in COMMAND_HOOK_TARGETS
        ],
    ]


def adapt_runtime_hook_event(raw_event: object) -> RuntimeHookSnapshot:
    if isinstance(raw_event, RuntimeHookSnapshot):
        return raw_event
    try:
        from predict_rlm import RuntimeHookEvent
    except ImportError as exc:
        raise TypeError("PredictRLM RuntimeHookEvent is unavailable.") from exc

    event = (
        raw_event
        if isinstance(raw_event, RuntimeHookEvent)
        else RuntimeHookEvent.model_validate(raw_event)
    )
    return RuntimeHookSnapshot(
        target=event.target,
        phase=event.phase,
        args=list(event.args),
        kwargs=dict(event.kwargs),
        result=event.result,
        error=event.error,
    )


def _path_from_args(args: list[Any]) -> str | None:
    if not args:
        return None
    value = args[0]
    if isinstance(value, str):
        return value
    if isinstance(value, os.PathLike):
        return os.fspath(value)
    if isinstance(value, dict):
        for key in ("path", "value", "repr"):
            nested = value.get(key)
            if isinstance(nested, str):
                return nested
    return str(value) if value is not None else None


def _path_target_mode(
    target: str,
    args: list[Any],
    kwargs: dict[str, Any],
) -> Literal["read", "write", "read_write"]:
    if target.endswith(("read_text", "read_bytes")):
        return "read"
    if target.endswith(("write_text", "write_bytes")):
        return "write"
    mode = kwargs.get("mode")
    if mode is None and len(args) > 1:
        mode = args[1]
    if not isinstance(mode, str):
        mode = "r"
    readable = mode.startswith("r") or "+" in mode
    writable = any(flag in mode for flag in ("w", "a", "x", "+"))
    if readable and writable:
        return "read_write"
    return "write" if writable else "read"


def _os_open_mode(args: list[Any]) -> Literal["read", "write", "read_write"]:
    flags = _coerce_int(args[1]) if len(args) > 1 else None
    if flags is None:
        return "read"
    write_only = bool(flags & os.O_WRONLY)
    read_write = bool(flags & os.O_RDWR)
    write_like = bool(flags & (os.O_CREAT | os.O_TRUNC | os.O_APPEND))
    if read_write:
        return "read_write"
    return "write" if write_only or write_like else "read"


def _command_from_args(args: list[Any]) -> str | None:
    if not args:
        return None
    command = args[0]
    if isinstance(command, str):
        return command
    if isinstance(command, (list, tuple)):
        return " ".join(shlex.quote(str(part)) for part in command)
    if isinstance(command, dict):
        value = command.get("value") or command.get("repr")
        if isinstance(value, str):
            return value
    return str(command)


def _command_message(phase: str, command: str) -> str:
    display_command = _truncate_middle(command, MAX_COMMAND_DISPLAY_CHARS)
    if phase == "after":
        return f"command finished: {display_command}"
    if phase == "error":
        return f"command failed: {display_command}"
    return f"running {display_command}"


def _truncate_middle(value: str, max_chars: int) -> str:
    if len(value) <= max_chars:
        return value
    if max_chars <= len(TRUNCATION_MARKER):
        return TRUNCATION_MARKER[:max_chars]
    remaining = max_chars - len(TRUNCATION_MARKER)
    head = (remaining + 1) // 2
    tail = remaining // 2
    return f"{value[:head]}{TRUNCATION_MARKER}{value[-tail:]}"


def _coerce_int(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, str):
        try:
            return int(value)
        except ValueError:
            return None
    if isinstance(value, dict):
        for key in ("value", "repr"):
            coerced = _coerce_int(value.get(key))
            if coerced is not None:
                return coerced
    return None


def _append_unique(values: list[str], value: str) -> None:
    if value not in values:
        values.append(value)

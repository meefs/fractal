import shlex
import subprocess
from collections.abc import Iterator
from pathlib import Path

SBX_DOCKER_AUTH_MESSAGE = (
    "Your sbx CLI is not logged in to Docker. "
    "Run `sbx login`, then try Fractal again."
)


def user_facing_error(exc: BaseException) -> str:
    """Return the concise CLI message Fractal should show for a failure."""
    if _is_sbx_docker_auth_error(exc):
        return SBX_DOCKER_AUTH_MESSAGE
    return str(exc)


def _is_sbx_docker_auth_error(exc: BaseException) -> bool:
    saw_sbx_command = False
    fragments: list[str] = []

    for current in _exception_chain(exc):
        fragments.append(str(current))
        if _is_called_sbx_command(current):
            saw_sbx_command = True
        if isinstance(current, subprocess.CalledProcessError):
            fragments.extend(
                _output_text(output)
                for output in (current.stdout, current.stderr, current.output)
            )

    text = "\n".join(fragment for fragment in fragments if fragment).lower()
    if not saw_sbx_command and "sbx" not in text:
        return False

    auth_markers = (
        "401 unauthorized",
        "not authenticated to docker",
        "no valid user session",
        "please sign in to docker",
        "secret not found",
    )
    return any(marker in text for marker in auth_markers)


def _exception_chain(exc: BaseException) -> Iterator[BaseException]:
    pending = [exc]
    seen: set[int] = set()
    while pending:
        current = pending.pop()
        identity = id(current)
        if identity in seen:
            continue
        seen.add(identity)
        yield current
        if current.__context__ is not None:
            pending.append(current.__context__)
        if current.__cause__ is not None:
            pending.append(current.__cause__)


def _is_called_sbx_command(exc: BaseException) -> bool:
    cmd = getattr(exc, "cmd", None)
    if isinstance(cmd, str):
        try:
            parts = shlex.split(cmd)
        except ValueError:
            parts = cmd.split()
    elif isinstance(cmd, (list, tuple)):
        parts = [str(part) for part in cmd]
    else:
        return False
    if not parts:
        return False
    return Path(parts[0]).name == "sbx"


def _output_text(output: object) -> str:
    if output is None:
        return ""
    if isinstance(output, bytes):
        return output.decode("utf-8", errors="replace")
    if isinstance(output, str):
        return output
    return str(output)

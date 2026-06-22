from __future__ import annotations

import subprocess


def test_user_facing_error_suggests_sbx_login_from_wrapped_stderr() -> None:
    from fractal.errors import user_facing_error

    called_process_error = subprocess.CalledProcessError(
        1,
        ["sbx", "create", "shell", "/tmp/sbx", "/workspace"],
        stderr=(
            "ERROR: list sandboxes: request failed: 401 Unauthorized: "
            "user is not authenticated to Docker\n"
            "no valid user session found, please sign in to Docker to proceed"
        ),
    )

    try:
        raise RuntimeError("Failed to create sbx sandbox") from called_process_error
    except RuntimeError as exc:
        assert user_facing_error(exc) == (
            "Your sbx CLI is not logged in to Docker. "
            "Run `sbx login`, then try Fractal again."
        )


def test_user_facing_error_keeps_generic_exception_text() -> None:
    from fractal.errors import user_facing_error

    assert user_facing_error(RuntimeError("model failed")) == "model failed"

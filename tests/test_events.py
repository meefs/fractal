from __future__ import annotations

import os


def hook_event(
    target: str,
    phase: str,
    *,
    args: list[object] | None = None,
    kwargs: dict[str, object] | None = None,
    result: object = None,
    error: str | None = None,
) -> object:
    from predict_rlm import RuntimeHookEvent

    return RuntimeHookEvent(
        target=target,
        phase=phase,
        args=args or [],
        kwargs=kwargs or {},
        result=result,
        error=error,
        timestamp=0.0,
    )


def test_runtime_event_tracker_records_file_reads_and_writes() -> None:
    from fractal.events import RuntimeEventTracker

    tracker = RuntimeEventTracker()

    opened = tracker.observe(
        hook_event("builtins.open", "before", args=["README.md", "r"])
    )
    tracker.observe(hook_event("builtins.open", "after", args=["README.md", "r"]))
    write = tracker.observe(
        hook_event(
            "pathlib.Path.write_text",
            "before",
            args=["README.md", "updated"],
        )
    )
    tracker.observe(
        hook_event(
            "pathlib.Path.write_text",
            "after",
            args=["README.md", "updated"],
        )
    )

    assert opened is not None
    assert opened.message == "opening README.md"
    assert write is not None
    assert write.message == "editing README.md"
    assert tracker.files_read == ["README.md"]
    assert tracker.files_modified == ["README.md"]


def test_runtime_event_tracker_suppresses_nested_path_open_events() -> None:
    from fractal.events import RuntimeEventTracker

    tracker = RuntimeEventTracker()

    events = [
        tracker.observe(
            hook_event("pathlib.Path.read_text", "before", args=["README.md"])
        ),
        tracker.observe(
            hook_event("pathlib.Path.open", "before", args=["README.md", "r"])
        ),
        tracker.observe(
            hook_event("pathlib.Path.open", "after", args=["README.md", "r"])
        ),
        tracker.observe(
            hook_event("pathlib.Path.read_text", "after", args=["README.md"])
        ),
    ]

    assert [event.message for event in events if event is not None] == [
        "reading README.md"
    ]
    assert tracker.files_read == ["README.md"]


def test_runtime_event_tracker_maps_os_file_descriptors() -> None:
    from fractal.events import RuntimeEventTracker

    tracker = RuntimeEventTracker()

    tracker.observe(
        hook_event("os.open", "after", args=["src/app.py", os.O_RDWR], result=7)
    )
    write = tracker.observe(hook_event("os.pwrite", "before", args=[7, "data", 0]))
    tracker.observe(hook_event("os.pwrite", "after", args=[7, "data", 0]))

    assert write is not None
    assert write.message == "editing src/app.py"
    assert tracker.files_modified == ["src/app.py"]


def test_runtime_event_tracker_records_subprocess_commands() -> None:
    from fractal.events import RuntimeEventTracker

    tracker = RuntimeEventTracker()

    event = tracker.observe(
        hook_event("subprocess.run", "before", args=[["uv", "run", "pytest"]])
    )

    assert event is not None
    assert event.message == "running uv run pytest"
    assert tracker.commands_run == ["uv run pytest"]


def test_runtime_event_tracker_suppresses_nested_subprocess_popen_events() -> None:
    from fractal.events import RuntimeEventTracker

    tracker = RuntimeEventTracker()

    events = [
        tracker.observe(
            hook_event(
                "subprocess.run",
                "before",
                args=[["git", "status", "--short"]],
            )
        ),
        tracker.observe(
            hook_event(
                "subprocess.Popen",
                "before",
                args=[["git", "status", "--short"]],
            )
        ),
        tracker.observe(
            hook_event(
                "subprocess.Popen",
                "after",
                args=[["git", "status", "--short"]],
            )
        ),
        tracker.observe(
            hook_event(
                "subprocess.run",
                "after",
                args=[["git", "status", "--short"]],
            )
        ),
    ]

    assert [event.message for event in events if event is not None] == [
        "running git status --short"
    ]
    assert tracker.commands_run == ["git status --short"]


def test_runtime_event_tracker_surfaces_direct_subprocess_popen_events() -> None:
    from fractal.events import RuntimeEventTracker

    tracker = RuntimeEventTracker()

    started = tracker.observe(
        hook_event(
            "subprocess.Popen",
            "before",
            args=[["git", "status", "--short"]],
        )
    )
    finished = tracker.observe(
        hook_event(
            "subprocess.Popen",
            "after",
            args=[["git", "status", "--short"]],
        )
    )

    assert started is not None
    assert started.message == "running git status --short"
    assert finished is None
    assert tracker.commands_run == ["git status --short"]


def test_runtime_event_tracker_surfaces_subprocess_failures() -> None:
    from fractal.events import RuntimeEventTracker

    tracker = RuntimeEventTracker()

    failed = tracker.observe(
        hook_event(
            "subprocess.run",
            "error",
            args=[["git", "status", "--short"]],
        )
    )

    assert failed is not None
    assert failed.message == "command failed: git status --short"


def test_runtime_event_tracker_truncates_long_command_messages() -> None:
    from fractal.events import MAX_COMMAND_DISPLAY_CHARS, RuntimeEventTracker

    tracker = RuntimeEventTracker()
    long_arg = "x" * 220
    full_command = f"python -c {long_arg} -- src/generated/output.txt"

    event = tracker.observe(
        hook_event(
            "subprocess.run",
            "before",
            args=[["python", "-c", long_arg, "--", "src/generated/output.txt"]],
        )
    )

    assert event is not None
    assert event.command == full_command
    assert tracker.commands_run == [full_command]
    assert event.message.startswith("running python -c ")
    assert event.message.endswith("src/generated/output.txt")
    assert "..." in event.message
    assert len(event.message) == len("running ") + MAX_COMMAND_DISPLAY_CHARS

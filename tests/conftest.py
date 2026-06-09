from __future__ import annotations

import pytest


@pytest.fixture(autouse=True)
def _clear_fractal_env_overrides(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in (
        "FRACTAL_PROVIDER",
        "FRACTAL_MODEL",
        "FRACTAL_SUB_MODEL",
        "FRACTAL_MAX_ITERATIONS",
        "FRACTAL_VERBOSE",
    ):
        monkeypatch.delenv(name, raising=False)


@pytest.fixture(autouse=True)
def _stub_provider_connectivity(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep tests offline: pretend every connectivity probe answered HTTP 200.

    Tests that exercise failure handling monkeypatch the opener themselves or
    pass an explicit opener to check_provider_connectivity.
    """
    monkeypatch.setattr(
        "fractal.connectivity._urlopen_status",
        lambda request, timeout: 200,
    )

    def _no_json(request: object, timeout: float) -> object:
        import urllib.error

        raise urllib.error.URLError("network disabled in tests")

    monkeypatch.setattr("fractal.connectivity._urlopen_json", _no_json)

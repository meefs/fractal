from __future__ import annotations

import pytest


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

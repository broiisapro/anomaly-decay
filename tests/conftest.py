"""Shared fixtures.

The whole test suite is offline: an autouse fixture replaces ``socket.socket`` and
``socket.create_connection`` with a guard that raises, so any accidental network access
fails loudly. Tests that genuinely need the network must opt in with ``@pytest.mark.network``
(none do).
"""

from __future__ import annotations

import socket
from collections.abc import Iterator

import pytest


@pytest.fixture(autouse=True)
def block_network(
    request: pytest.FixtureRequest, monkeypatch: pytest.MonkeyPatch
) -> Iterator[None]:
    """Fail any test that attempts a real socket connection."""
    if request.node.get_closest_marker("network") is not None:
        yield
        return

    def guard(*args: object, **kwargs: object) -> None:
        raise RuntimeError("network access attempted in an offline test")

    monkeypatch.setattr(socket, "socket", guard)
    monkeypatch.setattr(socket, "create_connection", guard)
    yield

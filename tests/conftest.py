"""Suite-wide autouse network guard.

Every pytest in this repo must make ZERO real network calls (Global
Constraints): providers are exercised through ``httpx.MockTransport`` and
LLM agents through fake providers, so no test should ever open a TCP socket.
This autouse fixture enforces that structurally — it monkeypatches
``socket.socket.connect`` / ``connect_ex`` to raise the instant any code
tries to connect an ``AF_INET`` / ``AF_INET6`` socket. ``AF_UNIX`` (and any
other family) is left alone, so local IPC / self-pipes used by asyncio and
the test machinery keep working.

A leaked real HTTP call therefore fails loudly with ``NetworkBlocked``
rather than silently hitting the network (or hanging).
"""

import socket

import pytest


class NetworkBlocked(RuntimeError):
    """Raised when a test attempts to open a real network connection."""


_BLOCKED_FAMILIES = {socket.AF_INET, socket.AF_INET6}
_real_connect = socket.socket.connect
_real_connect_ex = socket.socket.connect_ex


@pytest.fixture(autouse=True)
def _no_network(monkeypatch):
    def _guarded(real):
        def wrapper(self, address, *args, **kwargs):
            if self.family in _BLOCKED_FAMILIES:
                raise NetworkBlocked(
                    f"network access blocked in tests (family={self.family!r}, "
                    f"address={address!r}); use httpx.MockTransport / a fake "
                    "provider instead"
                )
            return real(self, address, *args, **kwargs)

        return wrapper

    monkeypatch.setattr(socket.socket, "connect", _guarded(_real_connect))
    monkeypatch.setattr(socket.socket, "connect_ex", _guarded(_real_connect_ex))
    yield

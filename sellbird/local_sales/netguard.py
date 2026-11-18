"""Network guard used by the offline acceptance test.

block_external_network() patches the socket layer so any attempt to reach a
non-loopback address raises immediately and is recorded. This is a *proof
aid* that complements (not replaces) physically disconnecting the network.
"""
from __future__ import annotations

import ipaddress
import json
import socket
import time

from sellbird.paths import PROJECT_ROOT

ATTEMPTS_LOG = PROJECT_ROOT / "logs" / "network_attempts.jsonl"
_state = {"active": False, "blocked": [], "allowed": 0}
_orig = {}


def _is_loopback(host) -> bool:
    if isinstance(host, bytes):
        host = host.decode()
    if host in ("localhost", "", None):
        return True
    try:
        ip = ipaddress.ip_address(host)
        return ip.is_loopback
    except ValueError:
        return False


def _record(kind: str, target) -> None:
    entry = {"ts": time.time(), "kind": kind, "target": str(target)}
    _state["blocked"].append(entry)
    ATTEMPTS_LOG.parent.mkdir(parents=True, exist_ok=True)
    with ATTEMPTS_LOG.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(entry) + "\n")


def block_external_network() -> None:
    if _state["active"]:
        return
    _orig["connect"] = socket.socket.connect
    _orig["getaddrinfo"] = socket.getaddrinfo

    def guarded_connect(self, address):
        host = address[0] if isinstance(address, tuple) else address
        if self.family in (socket.AF_INET, socket.AF_INET6) and not _is_loopback(host):
            _record("connect", address)
            raise OSError(f"[netguard] external connection blocked: {address}")
        _state["allowed"] += 1
        return _orig["connect"](self, address)

    def guarded_getaddrinfo(host, *a, **kw):
        if not _is_loopback(host):
            _record("dns", host)
            raise socket.gaierror(f"[netguard] DNS lookup blocked: {host}")
        return _orig["getaddrinfo"](host, *a, **kw)

    socket.socket.connect = guarded_connect
    socket.getaddrinfo = guarded_getaddrinfo
    _state["active"] = True


def unblock() -> None:
    if _state["active"]:
        socket.socket.connect = _orig["connect"]
        socket.getaddrinfo = _orig["getaddrinfo"]
        _state["active"] = False


def report() -> dict:
    return {"active": _state["active"], "blocked_attempts": list(_state["blocked"]),
            "loopback_connections": _state["allowed"]}

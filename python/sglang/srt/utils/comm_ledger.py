"""Process-local semantic communication counters for link validation."""

from __future__ import annotations

import atexit
import json
import logging
import os
import socket
import threading
from collections import defaultdict

logger = logging.getLogger(__name__)
_MARKER = "AFLEX_COMM_LEDGER"
_LOCK = threading.Lock()
_COUNTERS = defaultdict(
    lambda: {
        "logical_tx_bytes": 0,
        "logical_rx_bytes": 0,
        "expected_d2h_bytes": 0,
        "expected_h2d_bytes": 0,
        "calls": 0,
        "tx_calls": 0,
        "rx_calls": 0,
    }
)


def _should_emit(calls: int) -> bool:
    return calls <= 8 or (calls > 0 and calls & (calls - 1) == 0)


def _snapshot(backend: str, final: bool = False) -> dict:
    return {
        "schema_version": 1,
        "pid": os.getpid(),
        "process_id": f"{socket.gethostname()}:{os.getpid()}",
        "backend": backend,
        **_COUNTERS[backend],
        "final": final,
    }


def _emit(payload: dict, *, direct: bool = False) -> None:
    line = f"{_MARKER} {json.dumps(payload, sort_keys=True, separators=(',', ':'))}"
    if direct:
        print(line, flush=True)
    else:
        logger.info(line)


def record_comm(
    backend: str,
    *,
    logical_tx_bytes: int = 0,
    logical_rx_bytes: int = 0,
    expected_d2h_bytes: int = 0,
    expected_h2d_bytes: int = 0,
    tx_calls: int = 0,
    rx_calls: int = 0,
) -> None:
    """Accumulate one successful transfer and occasionally emit a stable marker."""
    values = {
        "logical_tx_bytes": logical_tx_bytes,
        "logical_rx_bytes": logical_rx_bytes,
        "expected_d2h_bytes": expected_d2h_bytes,
        "expected_h2d_bytes": expected_h2d_bytes,
        "tx_calls": tx_calls,
        "rx_calls": rx_calls,
    }
    if any(int(value) < 0 for value in values.values()):
        raise ValueError("communication ledger deltas must be non-negative")
    with _LOCK:
        counter = _COUNTERS[backend]
        for key, value in values.items():
            counter[key] += int(value)
        counter["calls"] += int(tx_calls) + int(rx_calls)
        payload = _snapshot(backend) if _should_emit(counter["calls"]) else None
    if payload is not None:
        _emit(payload)


def emit_final_comm_ledgers() -> None:
    """Best-effort final snapshots; periodic markers remain authoritative on crashes."""
    with _LOCK:
        payloads = [
            _snapshot(backend, final=True)
            for backend in sorted(_COUNTERS)
            if _COUNTERS[backend]["calls"]
        ]
    for payload in payloads:
        _emit(payload, direct=True)


atexit.register(emit_final_comm_ledgers)

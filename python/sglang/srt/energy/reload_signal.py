"""Tier1 reload signal protocol — shared between scheduler, orchestrator, and benchmark.

Signal file: tier1_reload_signal.json
Location: same directory as tier1_stats_path (or /tmp/tier1_shared/)

Statuses:
  - "idle": no reload in progress (or file doesn't exist)
  - "draining": old modules are draining inflight requests
  - "starting": new modules are starting up
  - "switching": router is switching traffic to new modules
  - "reloading": legacy full reload in progress (kill-all mode)
  - "ready": reload complete, benchmark can resume
  - "error": reload failed
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path


SIGNAL_FILENAME = "tier1_reload_signal.json"

# Valid status transitions for graceful reload
GRACEFUL_STATUSES = ("draining", "starting", "switching")
TERMINAL_STATUSES = ("ready", "idle", "error")


def get_signal_path(stats_path: str = None) -> str:
    if stats_path:
        return os.path.join(os.path.dirname(stats_path), SIGNAL_FILENAME)
    return f"/tmp/tier1_shared/{SIGNAL_FILENAME}"


def read_signal(signal_path: str) -> dict:
    """Read the current reload signal. Returns {"status": "idle"} if not found."""
    try:
        if os.path.exists(signal_path):
            with open(signal_path) as f:
                return json.load(f)
    except (json.JSONDecodeError, OSError):
        pass
    return {"status": "idle"}


def is_reloading(signal_path: str) -> bool:
    """Check if a reload is currently in progress (any non-terminal state)."""
    sig = read_signal(signal_path)
    status = sig.get("status", "idle")
    return status in ("reloading", "draining", "starting", "switching")


def is_ready(signal_path: str) -> bool:
    """Check if reload is complete and system is ready."""
    sig = read_signal(signal_path)
    return sig.get("status") in ("ready", "idle")


def wait_until_ready(signal_path: str, timeout: float = 300.0,
                     poll_interval: float = 2.0) -> bool:
    """Block until reload completes or timeout. Returns True if ready."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        sig = read_signal(signal_path)
        status = sig.get("status", "idle")
        if status in ("ready", "idle"):
            return True
        if status == "error":
            return False
        time.sleep(poll_interval)
    return False


def write_signal(signal_path: str, status: str, extra: dict = None):
    """Write a signal status update to the signal file."""
    data = {"status": status, "timestamp": time.time()}
    if extra:
        data.update(extra)
    Path(signal_path).parent.mkdir(parents=True, exist_ok=True)
    with open(signal_path, "w") as f:
        json.dump(data, f)


def clear_signal(signal_path: str):
    """Reset signal to idle state."""
    write_signal(signal_path, "idle")

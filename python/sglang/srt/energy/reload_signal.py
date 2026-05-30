"""Tier1 reload signal protocol — shared between scheduler, orchestrator, and benchmark.

Signal file: tier1_reload_signal.json
Location: same directory as tier1_stats_path (or /tmp/tier1_shared/)

Statuses:
  - "idle": no reload in progress (or file doesn't exist)
  - "reloading": reload in progress, benchmark should pause
  - "ready": reload complete, benchmark can resume
  - "error": reload failed
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path


SIGNAL_FILENAME = "tier1_reload_signal.json"


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
    """Check if a reload is currently in progress."""
    sig = read_signal(signal_path)
    return sig.get("status") == "reloading"


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


def clear_signal(signal_path: str):
    """Reset signal to idle state."""
    Path(signal_path).parent.mkdir(parents=True, exist_ok=True)
    with open(signal_path, "w") as f:
        json.dump({"status": "idle", "timestamp": time.time()}, f)

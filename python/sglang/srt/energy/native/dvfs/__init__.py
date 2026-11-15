"""Optional NVML native library sources and CPU-safe build utilities."""

import os
from pathlib import Path

LIBRARY_NAME = "libdvfs_ctrl.so"


def default_library_path() -> Path:
    """Return the writable per-user build location (never the source tree)."""
    cache = Path(os.environ.get("XDG_CACHE_HOME", "~/.cache")).expanduser()
    return cache / "sglang" / "dvfs" / LIBRARY_NAME

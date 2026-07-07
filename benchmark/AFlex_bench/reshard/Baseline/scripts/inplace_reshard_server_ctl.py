#!/usr/bin/env python3
"""Start/stop helper for in-place reshard benchmark server (port 31700)."""
from __future__ import annotations

import argparse
import subprocess
import time


def stop_inplace_reshard_server(port: int = 31700) -> None:
    """Terminate sglang server and scheduler workers for the benchmark port."""
    pat = f"sglang.launch_server.*{port}"
    subprocess.run(
        f"pkill -TERM -f '{pat}' 2>/dev/null || true; "
        "pkill -TERM -f 'sglang::' 2>/dev/null || true; "
        "sleep 2; "
        f"pkill -KILL -f '{pat}' 2>/dev/null || true; "
        "pkill -KILL -f 'sglang::' 2>/dev/null || true; "
        "pkill -KILL -f 'sglang::detokenizer' 2>/dev/null || true; "
        "pkill -KILL -f 'sglang::scheduler' 2>/dev/null || true",
        shell=True,
        check=False,
    )
    time.sleep(2)
    print(f"[cleanup] stopped sglang processes (port {port})")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=31700)
    args = ap.parse_args()
    stop_inplace_reshard_server(args.port)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

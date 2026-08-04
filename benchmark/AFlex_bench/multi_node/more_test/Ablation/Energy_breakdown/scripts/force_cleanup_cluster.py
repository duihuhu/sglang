#!/usr/bin/env python3
"""Force-clean benchmark ports on node3/node4 before rerunning tests."""
from __future__ import annotations

import logging
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import bench_no_dvfs_common as BNC

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


def main() -> int:
    BNC.ensure_cluster_clean()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

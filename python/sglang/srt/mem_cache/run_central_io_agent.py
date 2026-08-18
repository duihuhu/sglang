#!/usr/bin/env python3
"""Start one LatticeKV Central I/O agent for SGLang model processes."""

from __future__ import annotations

import argparse

from sglang.srt.mem_cache.central_io import CentralIOAgent


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--socket", required=True)
    parser.add_argument("--pool-gib", required=True, type=float)
    args = parser.parse_args()
    if args.pool_gib <= 0:
        parser.error("--pool-gib must be positive")
    agent = CentralIOAgent(args.socket, int(args.pool_gib * 1024**3))
    agent.warmup_restore_operator()
    agent.serve_forever()


if __name__ == "__main__":
    main()

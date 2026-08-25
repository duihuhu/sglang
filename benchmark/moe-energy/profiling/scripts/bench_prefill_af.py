#!/usr/bin/env python3
import sys

if "--help" in sys.argv or "-h" in sys.argv:
    print("Qwen3-30B-A3B prefill component profiler")
    print("Required profiler arguments: --component {A,F} --parallel-mode {attn_tp,moe_tp,moe_ep} --output PATH")
    print("Matrix arguments: --freqs ... --lengths ... --batch-sizes ... --warmup N --repeat N --quick")
    print("Capacity arguments: --shape-token-limit N (0=runner limit) --stop-on-shape-failure")
    print("SGLang arguments include: --model-path PATH --tp-size N --ep-size N --dist-timeout N --moe-runner-backend triton --moe-a2a-backend none")
    raise SystemExit(0)

from profile_utils import profile_main

if __name__ == "__main__":
    profile_main("prefill")

#!/usr/bin/env python3
import sys

if "--help" in sys.argv or "-h" in sys.argv:
    print("Qwen3 MoE kernel-only prefill profiler (run_moe_core CUDA segment)")
    print("Required: --forced-routing {balanced,middle_rank0,skewed_rank0} --output PATH")
    print("SGLang: --model-path PATH --tp-size N --ep-size N --moe-runner-backend triton --moe-a2a-backend none")
    raise SystemExit(0)

from bench_kernel_ep import profile_kernel_main

if __name__ == "__main__":
    profile_kernel_main("prefill")

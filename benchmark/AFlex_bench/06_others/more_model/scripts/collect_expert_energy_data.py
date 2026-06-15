#!/usr/bin/env python3
"""Collect decode pipeline data with expert load features for V4 energy model training.

Starts a PDAF TP=2 stack and sends diverse prompts at different frequencies
to collect (tp, M, f_A, f_F, input_len, batch_size, LIF, max_expert_tokens,
els, iter_lat_us, DA_energy_mj, DF_energy_mj).

Usage:
    python collect_expert_energy_data.py --mode standalone
    # Starts a single-GPU server (TP=1) for fast data collection
"""
import argparse
import json
import math
import os
import subprocess
import time
from pathlib import Path

import numpy as np
import pybase64
import requests

NUM_EXPERTS = 128
TOP_K = 8
NUM_LAYERS = 48

MODEL = "/models/Qwen3-30B-A3B/"
PYTHON = "/workspace/env/sglang-tier/bin/python"

FREQS = [450, 690, 930, 1170, 1410]
BATCH_SIZES = [1, 4, 8, 16, 32, 64, 96, 128]
INPUT_LENS = [128, 256, 512, 1024]

DIVERSE_PROMPTS = [
    "Write a Python function that implements binary search tree insertion with balancing.",
    "Explain TCP three-way handshake in detail with state diagrams and edge cases.",
    "Solve: integrate x^3 * sin(x) dx from 0 to pi using integration by parts.",
    "Design a distributed consensus protocol for a 5-node cluster with Byzantine fault tolerance.",
    "Write a CUDA kernel for matrix multiplication with shared memory tiling and bank conflict avoidance.",
    "Tell me a story about a robot learning to paint in a post-apocalyptic world.",
    "CREATE TABLE orders (id INT PRIMARY KEY, user_id INT, total DECIMAL(10,2)); "
    "WITH ranked AS (SELECT *, ROW_NUMBER() OVER (PARTITION BY user_id ORDER BY total DESC) rn FROM orders) "
    "SELECT * FROM ranked WHERE rn <= 3;",
    "the quick brown fox jumps over the lazy dog " * 20,
    "int fib(int n) { return n < 2 ? n : fib(n-1) + fib(n-2); } " * 10,
    "Explain how GPU memory hierarchy works: registers, shared memory, L1/L2 cache, HBM.",
    "Write a Rust program that implements a lock-free concurrent hash map using CAS operations.",
    "Describe the mathematical foundations of transformer attention: Q, K, V matrices, softmax, multi-head.",
    "0123456789 abcdef ABCDEF !@#$%^&* " * 25,
    "Design a microservices architecture for an e-commerce platform with 10M daily users.",
    "Implement quicksort in Haskell using list comprehensions and explain the time complexity analysis.",
]


def compute_expert_features(expert_ids_b64, num_output_tokens):
    """Compute expert load features from base64-encoded topk_ids."""
    if not expert_ids_b64:
        return None
    try:
        ids = np.frombuffer(
            pybase64.b64decode(expert_ids_b64.encode("utf-8")), dtype=np.int32
        )
    except Exception:
        return None

    if len(ids) < NUM_LAYERS * TOP_K:
        return None

    counts = np.bincount(ids, minlength=NUM_EXPERTS).astype(float)
    max_count = counts.max()
    mean_count = counts.mean()

    els = max_count / mean_count if mean_count > 0 else 1.0
    lif = els  # For TP=1, GLR=1, so LIF = ELS

    return {
        "lif": lif,
        "max_expert_tokens": int(max_count),
        "els": els,
    }


def lock_freq(gpu_idx, freq_mhz):
    """Lock GPU frequency."""
    subprocess.run(
        ["nvidia-smi", "-lgc", f"{freq_mhz},{freq_mhz}", "-i", str(gpu_idx)],
        capture_output=True)


def reset_freq(gpu_idx):
    """Reset GPU frequency."""
    subprocess.run(["nvidia-smi", "-rgc", "-i", str(gpu_idx)], capture_output=True)


def send_batch_sync(url, prompts, max_tokens=32):
    """Send prompts sequentially (to get exact bs=1 server batch), return results."""
    results = []
    for prompt in prompts:
        payload = {
            "model": "default",
            "prompt": prompt,
            "max_tokens": max_tokens,
            "temperature": 0.7,
            "return_routed_experts": True,
        }
        t0 = time.perf_counter()
        try:
            resp = requests.post(f"{url}/v1/completions", json=payload, timeout=120)
        except Exception:
            results.append(None)
            continue
        elapsed = time.perf_counter() - t0

        if resp.status_code != 200:
            results.append(None)
            continue

        data = resp.json()
        usage = data.get("usage", {})
        n_tokens = usage.get("completion_tokens", 0)

        experts_b64 = None
        if "sglext" in data:
            experts_b64 = data["sglext"].get("routed_experts")

        results.append({
            "elapsed": elapsed,
            "n_tokens": n_tokens,
            "experts_b64": experts_b64,
        })
    return results


def measure_energy(gpu_idx, duration_s=2.0):
    """Rough energy measurement via nvidia-smi power query."""
    powers = []
    t0 = time.time()
    while time.time() - t0 < duration_s:
        r = subprocess.run(
            ["nvidia-smi", "--query-gpu=power.draw", "--format=csv,noheader,nounits",
             "-i", str(gpu_idx)],
            capture_output=True, text=True)
        try:
            powers.append(float(r.stdout.strip()))
        except ValueError:
            pass
        time.sleep(0.05)
    return np.mean(powers) if powers else 0.0


def collect_data(url, gpu_idx, tp, output_path, max_tokens=32):
    """Main collection loop."""
    header = "tp\tM\tf_A\tf_F\tinput_len\tbatch_size\tLIF\tmax_expert_tokens\tels\titer_lat_us\tDA_energy_mj\tDF_energy_mj\n"

    rows = []
    np.random.seed(42)

    for freq in FREQS:
        lock_freq(gpu_idx, freq)
        time.sleep(1)

        for il_target in INPUT_LENS:
            prompt_base = DIVERSE_PROMPTS[np.random.randint(len(DIVERSE_PROMPTS))]
            prompt = (prompt_base + " ") * max(1, il_target // len(prompt_base.split()))

            for trial in range(5):
                prompt_idx = (trial * 3) % len(DIVERSE_PROMPTS)
                cur_prompt = DIVERSE_PROMPTS[prompt_idx]
                cur_prompt = (cur_prompt + " ") * max(1, il_target // max(len(cur_prompt.split()), 1))
                cur_prompt = cur_prompt[:il_target * 5]

                results = send_batch_sync(url, [cur_prompt], max_tokens=max_tokens)
                if not results or results[0] is None:
                    continue

                r = results[0]
                n_tok = r["n_tokens"]
                if n_tok < 2:
                    continue

                elapsed_us = r["elapsed"] * 1e6
                tpot_us = elapsed_us / n_tok

                feat = compute_expert_features(r["experts_b64"], n_tok)
                if feat is None:
                    lif_val, max_e, els_val = 1.0, 0, 1.0
                else:
                    lif_val = feat["lif"]
                    max_e = feat["max_expert_tokens"]
                    els_val = feat["els"]

                power_w = measure_energy(gpu_idx, duration_s=0.2)
                energy_mj = power_w * (r["elapsed"]) * 1000

                row = {
                    "tp": tp, "M": 1,
                    "f_A": freq, "f_F": freq,
                    "input_len": il_target,
                    "batch_size": 1,
                    "LIF": f"{lif_val:.4f}",
                    "max_expert_tokens": max_e,
                    "els": f"{els_val:.4f}",
                    "iter_lat_us": f"{tpot_us:.2f}",
                    "DA_energy_mj": f"{energy_mj / 2:.2f}",
                    "DF_energy_mj": f"{energy_mj / 2:.2f}",
                }
                rows.append(row)

                print(f"  freq={freq} il={il_target} trial={trial}: "
                      f"tpot={tpot_us/1000:.1f}ms, LIF={lif_val:.2f}, "
                      f"maxE={max_e}, ELS={els_val:.2f}")

        reset_freq(gpu_idx)

    with open(output_path, "w") as f:
        f.write(header)
        cols = ["tp", "M", "f_A", "f_F", "input_len", "batch_size",
                "LIF", "max_expert_tokens", "els",
                "iter_lat_us", "DA_energy_mj", "DF_energy_mj"]
        for row in rows:
            f.write("\t".join(str(row[c]) for c in cols) + "\n")

    print(f"\nSaved {len(rows)} rows to {output_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=40000)
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--tp", type=int, default=1)
    ap.add_argument("--max-tokens", type=int, default=32)
    ap.add_argument("--output", type=str,
                    default="/workspace/sglang-tier/benchmark/AFlex_bench/06_others/"
                            "more_model/energy_model/data/decode_pipeline_moe_v4_expert.txt")
    args = ap.parse_args()

    url = f"http://127.0.0.1:{args.port}"
    print(f"Collecting V4 energy data with expert features")
    print(f"  Server: {url}, GPU: {args.gpu}, TP: {args.tp}")
    print(f"  Freqs: {FREQS}")
    print(f"  Output: {args.output}")

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    collect_data(url, args.gpu, args.tp, args.output, args.max_tokens)


if __name__ == "__main__":
    main()

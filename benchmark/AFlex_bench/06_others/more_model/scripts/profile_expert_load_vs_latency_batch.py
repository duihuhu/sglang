#!/usr/bin/env python3
"""Profile MoE expert load vs latency with controlled batch sizes.

Strategy: Send concurrent requests to force the server to batch them together,
then measure per-token latency under different batch sizes and diverse prompts.

For each batch of requests:
  - Send `bs` concurrent requests simultaneously
  - Each request gets `return_routed_experts=True`
  - Measure the average TPOT across the batch
  - Aggregate topk_ids from all requests to compute batch-level features

Output: TSV file with one row per batch observation.
"""
import argparse
import asyncio
import json
import math
import time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pybase64
import requests

NUM_EXPERTS = 128
TOP_K = 8
NUM_LAYERS = 48

PROMPT_TEMPLATES = {
    "code_python": "Write a Python function that implements {topic}. Be detailed and thorough.",
    "code_cpp": "Write a C++ class that implements {topic} with proper RAII and templates.",
    "math": "Solve the following math problem step by step with equations: {topic}",
    "conversation": "Tell me about {topic} in a casual, conversational way with examples.",
    "technical": "Explain the technical details of {topic} for a senior systems engineer.",
    "creative": "Write a short story about {topic} with vivid descriptions and dialogue.",
    "random_tokens": "{topic}",
}

TOPICS = [
    "binary search tree", "matrix multiplication", "quicksort algorithm",
    "neural network backpropagation", "TCP three-way handshake",
    "memory allocation in Linux kernel", "GPU CUDA kernel programming",
    "distributed consensus Raft protocol", "B-tree database indexing",
    "garbage collection algorithms", "HTTP/2 protocol multiplexing",
    "quantum computing qubits", "reinforcement learning Q-table",
    "compiler optimization passes", "filesystem journaling ext4",
    "CPU cache coherence MESI protocol", "TLS 1.3 handshake process",
    "graph shortest path Dijkstra algorithm", "hash table collision resolution",
    "operating system CPU scheduling", "load balancing algorithms",
    "MapReduce parallel processing framework", "genetic algorithm optimization",
    "blockchain proof of work consensus", "real-time scheduling EDF",
    "FFT signal processing", "computer vision convolution neural network",
    "natural language parsing transformers", "microservices service mesh",
    "container orchestration Kubernetes pods", "red-black tree rebalancing",
    "AES encryption round operations", "virtual memory page tables",
    "network congestion control BBR", "database MVCC isolation levels",
    "GPU memory hierarchy shared local global", "SIMD vectorization techniques",
    "graph neural network message passing", "distributed file system GFS",
    "lock-free concurrent data structures",
]

RANDOM_TEXTS = [
    "asdf jkl; qwer uiop zxcv bnm, The matrix has you. " * 15,
    "the quick brown fox jumps over the lazy dog while computing hashes " * 10,
    "0123456789 abcdef ABCDEF !@#$%^&*() SELECT INSERT UPDATE " * 12,
    "int x = 0; for(int i=0;i<n;i++){x+=arr[i];} printf('%d',x); " * 8,
    "FROM node:18 AS builder WORKDIR /app COPY package*.json ./ RUN npm ci " * 8,
    "def fib(n): return n if n<2 else fib(n-1)+fib(n-2) # recursive " * 10,
    "CREATE TABLE users (id SERIAL PRIMARY KEY, name TEXT, age INT); " * 8,
    "import torch; x = torch.randn(32,128); y = x @ x.T; print(y.shape) " * 8,
]


def build_all_prompts():
    """Generate full pool of diverse prompts."""
    prompts = []
    for ptype, template in PROMPT_TEMPLATES.items():
        if ptype == "random_tokens":
            for txt in RANDOM_TEXTS:
                prompts.append({"type": ptype, "text": txt})
        else:
            for topic in TOPICS:
                prompts.append({"type": ptype, "text": template.format(topic=topic)})
    return prompts


def send_one_request(url, prompt_text, max_tokens=32):
    """Send a single request, return (elapsed_s, n_tokens, expert_ids_np)."""
    payload = {
        "model": "default",
        "prompt": prompt_text,
        "max_tokens": max_tokens,
        "temperature": 0.7,
        "return_routed_experts": True,
    }
    t0 = time.perf_counter()
    try:
        resp = requests.post(f"{url}/v1/completions", json=payload, timeout=180)
    except Exception:
        return None, None, None
    elapsed = time.perf_counter() - t0

    if resp.status_code != 200:
        return None, None, None

    data = resp.json()
    choice = data["choices"][0]
    usage = data.get("usage", {})
    n_tokens = usage.get("completion_tokens", len(choice.get("text", "").split()))

    routed_experts_b64 = None
    if "sglext" in data and data["sglext"].get("routed_experts"):
        routed_experts_b64 = data["sglext"]["routed_experts"]
    elif "meta_info" in choice and choice["meta_info"].get("routed_experts"):
        routed_experts_b64 = choice["meta_info"]["routed_experts"]

    if routed_experts_b64 is None:
        return elapsed, n_tokens, None

    expert_ids = np.frombuffer(
        pybase64.b64decode(routed_experts_b64.encode("utf-8")), dtype=np.int32
    )
    return elapsed, n_tokens, expert_ids


def send_batch_concurrent(url, prompts_batch, max_tokens, executor):
    """Send a batch of requests concurrently, return list of results."""
    futures = []
    for p in prompts_batch:
        futures.append(executor.submit(send_one_request, url, p["text"], max_tokens))
    results = [f.result() for f in futures]
    return results


def compute_batch_features(all_expert_ids_list, tp=1):
    """Compute aggregated expert distribution features for an entire batch.

    all_expert_ids_list: list of numpy arrays, one per request in the batch.
    Each array has shape (n_tokens * NUM_LAYERS * TOP_K,).
    """
    valid_ids = [ids for ids in all_expert_ids_list if ids is not None and len(ids) > 0]
    if not valid_ids:
        return None

    combined = np.concatenate(valid_ids)
    counts = np.bincount(combined, minlength=NUM_EXPERTS).astype(float)

    max_count = counts.max()
    mean_count = counts.mean()
    std_count = counts.std()
    num_active = (counts > 0).sum()
    total_pairs = combined.shape[0]

    els = max_count / mean_count if mean_count > 0 else 1.0
    top1_share = max_count / total_pairs if total_pairs > 0 else 0.0

    sorted_counts = np.sort(counts)
    n = len(sorted_counts)
    cum_sum = sorted_counts.sum()
    if cum_sum > 0:
        gini = (2.0 * np.sum((np.arange(1, n+1) * sorted_counts)) / (n * cum_sum) - (n+1)/n)
    else:
        gini = 0.0

    glr = 1.0
    if tp > 1:
        experts_per_gpu = NUM_EXPERTS // tp
        gpu_loads = np.array([
            counts[g * experts_per_gpu: (g+1) * experts_per_gpu].sum()
            for g in range(tp)
        ])
        mean_load = gpu_loads.mean()
        glr = gpu_loads.max() / mean_load if mean_load > 0 else 1.0

    return {
        "max_tokens_per_expert": int(max_count),
        "els": els,
        "std_tokens_per_expert": std_count,
        "num_active_experts": int(num_active),
        "top1_share": top1_share,
        "glr": glr,
        "gini": gini,
    }


def run_profiling(url, tp, output_path, max_tokens=32, batch_sizes=None,
                  trials_per_bs=20):
    """Main profiling loop: for each batch size, run multiple trials."""
    if batch_sizes is None:
        batch_sizes = [1, 4, 8, 16, 32, 64, 128]

    all_prompts = build_all_prompts()
    print(f"Prompt pool: {len(all_prompts)} prompts")
    np.random.seed(42)
    perm = np.random.permutation(len(all_prompts))

    header = [
        "batch_size", "trial", "prompt_types", "avg_prompt_len",
        "num_output_tokens_avg", "total_time_s_max", "tpot_ms_avg",
        "tpot_ms_p50", "tpot_ms_p95",
        "max_tokens_per_expert", "els", "std_tokens_per_expert",
        "num_active_experts", "top1_share", "glr", "gini",
    ]

    results = []
    executor = ThreadPoolExecutor(max_workers=max(batch_sizes) + 4)
    prompt_idx = 0

    for bs in batch_sizes:
        print(f"\n{'='*60}")
        print(f"  Batch size = {bs}, {trials_per_bs} trials")
        print(f"{'='*60}")

        for trial in range(trials_per_bs):
            batch_prompts = []
            for _ in range(bs):
                batch_prompts.append(all_prompts[perm[prompt_idx % len(perm)]])
                prompt_idx += 1

            batch_results = send_batch_concurrent(url, batch_prompts, max_tokens, executor)

            elapsed_list = []
            n_tokens_list = []
            expert_ids_list = []
            prompt_types = set()

            for (elapsed, n_tok, eids), p in zip(batch_results, batch_prompts):
                if elapsed is None:
                    continue
                elapsed_list.append(elapsed)
                n_tokens_list.append(n_tok if n_tok else 0)
                expert_ids_list.append(eids)
                prompt_types.add(p["type"])

            if not elapsed_list:
                print(f"  Trial {trial+1}: ALL FAILED")
                continue

            tpot_list = [(e / max(n, 1)) * 1000 for e, n in zip(elapsed_list, n_tokens_list)]
            avg_tpot = np.mean(tpot_list)
            p50_tpot = np.percentile(tpot_list, 50)
            p95_tpot = np.percentile(tpot_list, 95)
            max_elapsed = max(elapsed_list)
            avg_n_tokens = np.mean(n_tokens_list)
            avg_prompt_len = np.mean([len(p["text"]) for p in batch_prompts])

            feat = compute_batch_features(expert_ids_list, tp)
            if feat is None:
                print(f"  Trial {trial+1}: no expert data")
                continue

            row = {
                "batch_size": bs,
                "trial": trial,
                "prompt_types": "+".join(sorted(prompt_types)),
                "avg_prompt_len": f"{avg_prompt_len:.0f}",
                "num_output_tokens_avg": f"{avg_n_tokens:.1f}",
                "total_time_s_max": f"{max_elapsed:.4f}",
                "tpot_ms_avg": f"{avg_tpot:.2f}",
                "tpot_ms_p50": f"{p50_tpot:.2f}",
                "tpot_ms_p95": f"{p95_tpot:.2f}",
                **{k: f"{v:.4f}" if isinstance(v, float) else str(v) for k, v in feat.items()},
            }
            results.append(row)

            print(f"  Trial {trial+1}/{trials_per_bs}: "
                  f"tpot_avg={avg_tpot:.1f}ms, tpot_p95={p95_tpot:.1f}ms, "
                  f"ELS={feat['els']:.2f}, maxE={feat['max_tokens_per_expert']}, "
                  f"gini={feat['gini']:.3f}")

    executor.shutdown(wait=False)

    with open(output_path, "w") as f:
        f.write("\t".join(header) + "\n")
        for r in results:
            f.write("\t".join(str(r.get(h, "")) for h in header) + "\n")

    print(f"\nSaved {len(results)} rows to {output_path}")
    return results


def main():
    ap = argparse.ArgumentParser(description="Profile MoE expert load vs latency (batch)")
    ap.add_argument("--port", type=int, default=40000)
    ap.add_argument("--host", type=str, default="127.0.0.1")
    ap.add_argument("--tp", type=int, default=1)
    ap.add_argument("--max-tokens", type=int, default=32)
    ap.add_argument("--batch-sizes", type=str, default="1,4,8,16,32,64,128")
    ap.add_argument("--trials-per-bs", type=int, default=20)
    ap.add_argument("--output", type=str,
                    default="/workspace/sglang-tier/benchmark/AFlex_bench/06_others/"
                            "more_model/data/expert_load_vs_latency_batch.tsv")
    args = ap.parse_args()

    url = f"http://{args.host}:{args.port}"
    batch_sizes = [int(x) for x in args.batch_sizes.split(",")]

    print(f"MoE Expert Load vs Latency Profiling (Batch Mode)")
    print(f"  Server: {url}")
    print(f"  TP: {args.tp}")
    print(f"  Max tokens: {args.max_tokens}")
    print(f"  Batch sizes: {batch_sizes}")
    print(f"  Trials per bs: {args.trials_per_bs}")
    print(f"  Output: {args.output}")

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)

    # Warmup
    print("\nWarming up...")
    for _ in range(3):
        send_one_request(url, "Hello, explain sorting.", max_tokens=5)
    print("Warmup done.\n")

    run_profiling(url, args.tp, args.output,
                  max_tokens=args.max_tokens,
                  batch_sizes=batch_sizes,
                  trials_per_bs=args.trials_per_bs)


if __name__ == "__main__":
    main()

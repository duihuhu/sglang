#!/usr/bin/env python3
"""Profile MoE expert load distribution vs per-token latency.

Sends diverse prompts to a running SGLang server (Qwen3-30B-A3B) with
`return_routed_experts=True`, then collects:
  - topk_ids per decode step (expert routing decisions)
  - per-token latency (TPOT)

Computes expert distribution features and saves to TSV for regression analysis.

Usage:
    # Start server first (TP=1 or TP=2):
    python -m sglang.launch_server --model-path /models/Qwen/Qwen3-30B-A3B/ \
        --tp 1 --port 40000 --enable-return-routed-experts \
        --disable-cuda-graph --disable-radix-cache --mem-fraction-static 0.85

    # Then run profiling:
    python profile_expert_load_vs_latency.py --port 40000 --tp 1 \
        --output expert_load_vs_latency.tsv
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import pybase64
import requests

NUM_EXPERTS = 128
TOP_K = 8
NUM_LAYERS = 48

PROMPT_TEMPLATES = {
    "code_python": "Write a Python function that implements {topic}. Be detailed.",
    "code_cpp": "Write a C++ class that implements {topic} with proper RAII.",
    "math": "Solve the following math problem step by step: {topic}",
    "conversation": "Tell me about {topic} in a casual, conversational way.",
    "technical": "Explain the technical details of {topic} for a systems engineer.",
    "creative": "Write a short story about {topic} with vivid descriptions.",
    "random_tokens": "{topic}",
}

TOPICS = [
    "binary search tree", "matrix multiplication", "quicksort",
    "neural network backpropagation", "TCP three-way handshake",
    "memory allocation in Linux", "GPU CUDA kernel programming",
    "distributed consensus (Raft)", "B-tree database indexing",
    "garbage collection algorithms", "HTTP/2 protocol flow",
    "quantum computing basics", "reinforcement learning Q-table",
    "compiler optimization passes", "filesystem journaling",
    "CPU cache coherence MESI", "TLS 1.3 handshake",
    "graph shortest path Dijkstra", "hash table collision resolution",
    "operating system scheduling", "load balancing algorithms",
    "MapReduce parallel processing", "genetic algorithm optimization",
    "blockchain consensus mechanisms", "real-time operating systems",
    "signal processing FFT", "computer vision convolution",
    "natural language parsing", "microservices architecture",
    "container orchestration Kubernetes",
]

RANDOM_TEXTS = [
    "asdf jkl; qwer uiop zxcv bnm, " * 20,
    "the quick brown fox jumps over the lazy dog " * 15,
    "0123456789 abcdef ABCDEF !@#$%^&*() " * 18,
    "int x = 0; for(int i=0;i<n;i++){x+=arr[i];} return x; " * 10,
    "SELECT * FROM users WHERE id IN (1,2,3) ORDER BY created_at DESC; " * 8,
]


def build_prompts(num_per_type=5):
    """Generate diverse prompts to create varied expert routing patterns."""
    prompts = []
    for ptype, template in PROMPT_TEMPLATES.items():
        if ptype == "random_tokens":
            for txt in RANDOM_TEXTS[:num_per_type]:
                prompts.append({"type": ptype, "text": txt})
        else:
            for topic in TOPICS[:num_per_type]:
                prompts.append({"type": ptype, "text": template.format(topic=topic)})
    return prompts


def send_request(url, prompt_text, max_tokens=32):
    """Send a single synchronous request and get response with routed_experts."""
    payload = {
        "model": "default",
        "prompt": prompt_text,
        "max_tokens": max_tokens,
        "temperature": 0.7,
        "return_routed_experts": True,
    }
    t0 = time.perf_counter()
    resp = requests.post(f"{url}/v1/completions", json=payload, timeout=120)
    elapsed = time.perf_counter() - t0

    if resp.status_code != 200:
        return None, None, None

    data = resp.json()
    choice = data["choices"][0]
    usage = data.get("usage", {})
    num_output_tokens = usage.get("completion_tokens",
                        len(choice.get("text", "").split()))

    routed_experts_b64 = None
    if "sglext" in data and data["sglext"].get("routed_experts"):
        routed_experts_b64 = data["sglext"]["routed_experts"]
    elif "meta_info" in choice and choice["meta_info"].get("routed_experts"):
        routed_experts_b64 = choice["meta_info"]["routed_experts"]
    elif "routed_experts" in data:
        routed_experts_b64 = data["routed_experts"]

    if routed_experts_b64 is None:
        return elapsed, num_output_tokens, None

    expert_ids = np.frombuffer(
        pybase64.b64decode(routed_experts_b64.encode("utf-8")), dtype=np.int32
    )
    return elapsed, num_output_tokens, expert_ids


def compute_features(expert_ids, num_tokens, tp=1):
    """Compute expert distribution features from topk_ids.

    expert_ids shape: (num_tokens * num_layers * top_k,) flattened
    We analyze the LAST decode step's routing across all layers.
    """
    total_elements = len(expert_ids)
    expected_per_token = NUM_LAYERS * TOP_K

    if total_elements < expected_per_token:
        return None

    last_step_ids = expert_ids[-expected_per_token:]
    counts = np.bincount(last_step_ids, minlength=NUM_EXPERTS).astype(float)

    max_count = counts.max()
    mean_count = counts.mean()
    std_count = counts.std()
    num_active = (counts > 0).sum()
    total_pairs = last_step_ids.shape[0]

    els = max_count / mean_count if mean_count > 0 else 1.0
    top1_share = max_count / total_pairs if total_pairs > 0 else 0.0

    sorted_counts = np.sort(counts)
    n = len(sorted_counts)
    cum = np.cumsum(sorted_counts)
    gini = (2.0 * np.sum((np.arange(1, n+1) * sorted_counts)) / (n * cum[-1]) - (n+1)/n) if cum[-1] > 0 else 0.0

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


def compute_all_steps_features(expert_ids, num_output_tokens, tp=1):
    """Compute features aggregated over ALL decode steps (not just last)."""
    expected_per_token = NUM_LAYERS * TOP_K
    total_elements = len(expert_ids)
    n_steps = total_elements // expected_per_token

    if n_steps < 1:
        return None

    all_ids = expert_ids[:n_steps * expected_per_token].reshape(n_steps, -1)
    counts = np.zeros(NUM_EXPERTS, dtype=float)
    for step in range(n_steps):
        counts += np.bincount(all_ids[step], minlength=NUM_EXPERTS).astype(float)

    max_count = counts.max()
    mean_count = counts.mean()
    std_count = counts.std()
    num_active = (counts > 0).sum()
    total_pairs = all_ids.size

    els = max_count / mean_count if mean_count > 0 else 1.0
    top1_share = max_count / total_pairs if total_pairs > 0 else 0.0

    sorted_counts = np.sort(counts)
    n = len(sorted_counts)
    cum = np.cumsum(sorted_counts)
    gini = (2.0 * np.sum((np.arange(1, n+1) * sorted_counts)) / (n * cum[-1]) - (n+1)/n) if cum[-1] > 0 else 0.0

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
        "max_tokens_per_expert_agg": int(max_count),
        "els_agg": els,
        "std_agg": std_count,
        "num_active_agg": int(num_active),
        "top1_share_agg": top1_share,
        "glr_agg": glr,
        "gini_agg": gini,
    }


def run_profiling(url, tp, output_path, max_tokens=32, num_prompts_per_type=5):
    """Main profiling loop."""
    prompts = build_prompts(num_prompts_per_type)
    print(f"Generated {len(prompts)} prompts across {len(PROMPT_TEMPLATES)} types")

    header = [
        "prompt_type", "prompt_len_chars", "num_output_tokens",
        "total_time_s", "tpot_ms",
        "max_tokens_per_expert", "els", "std_tokens_per_expert",
        "num_active_experts", "top1_share", "glr", "gini",
        "max_tokens_per_expert_agg", "els_agg", "std_agg",
        "num_active_agg", "top1_share_agg", "glr_agg", "gini_agg",
    ]

    results = []
    total = len(prompts)

    for i, p in enumerate(prompts):
        print(f"  [{i+1}/{total}] type={p['type']}, len={len(p['text'])}...", end=" ")

        elapsed, n_tokens, expert_ids = send_request(url, p["text"], max_tokens)
        if elapsed is None or n_tokens is None:
            print("FAILED")
            continue

        tpot_ms = (elapsed / max(n_tokens, 1)) * 1000.0

        feat_last = compute_features(expert_ids, n_tokens, tp) if expert_ids is not None else None
        feat_agg = compute_all_steps_features(expert_ids, n_tokens, tp) if expert_ids is not None else None

        if feat_last is None:
            print(f"tokens={n_tokens}, no expert data")
            continue

        row = {
            "prompt_type": p["type"],
            "prompt_len_chars": len(p["text"]),
            "num_output_tokens": n_tokens,
            "total_time_s": f"{elapsed:.4f}",
            "tpot_ms": f"{tpot_ms:.2f}",
            **{k: f"{v:.4f}" if isinstance(v, float) else str(v) for k, v in feat_last.items()},
        }
        if feat_agg:
            row.update({k: f"{v:.4f}" if isinstance(v, float) else str(v) for k, v in feat_agg.items()})
        else:
            row.update({k: "" for k in ["max_tokens_per_expert_agg", "els_agg", "std_agg",
                                         "num_active_agg", "top1_share_agg", "glr_agg", "gini_agg"]})

        results.append(row)
        print(f"tokens={n_tokens}, tpot={tpot_ms:.1f}ms, ELS={feat_last['els']:.2f}, "
              f"maxE={feat_last['max_tokens_per_expert']}")

    with open(output_path, "w") as f:
        f.write("\t".join(header) + "\n")
        for r in results:
            f.write("\t".join(str(r.get(h, "")) for h in header) + "\n")

    print(f"\nSaved {len(results)} rows to {output_path}")
    return results


def main():
    ap = argparse.ArgumentParser(description="Profile MoE expert load vs latency")
    ap.add_argument("--port", type=int, default=40000)
    ap.add_argument("--host", type=str, default="127.0.0.1")
    ap.add_argument("--tp", type=int, default=1)
    ap.add_argument("--max-tokens", type=int, default=32,
                    help="Output tokens per request (controls decode steps)")
    ap.add_argument("--num-prompts-per-type", type=int, default=5)
    ap.add_argument("--output", type=str,
                    default="/workspace/sglang-tier/benchmark/AFlex_bench/06_others/"
                            "more_model/data/expert_load_vs_latency.tsv")
    args = ap.parse_args()

    url = f"http://{args.host}:{args.port}"
    print(f"Profiling MoE expert load vs latency")
    print(f"  Server: {url}")
    print(f"  TP: {args.tp}")
    print(f"  Max tokens: {args.max_tokens}")
    print(f"  Output: {args.output}")

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)

    # Warmup
    print("\nWarming up...")
    send_request(url, "Hello, how are you?", max_tokens=5)
    send_request(url, "Hello, how are you?", max_tokens=5)
    print("Warmup done.\n")

    run_profiling(url, args.tp, args.output,
                  max_tokens=args.max_tokens,
                  num_prompts_per_type=args.num_prompts_per_type)


if __name__ == "__main__":
    main()

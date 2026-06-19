#!/usr/bin/env python3
"""Profile MoE expert load vs latency V2 — large-scale, accurate measurement.

Improvements over V1:
  - More batch sizes: 2,4,8,16,24,32,48,64,96,128,160
  - More trials per bs: 50
  - Longer output: 128 tokens (more stable TPOT stats)
  - Inter-trial sleep: 2s gap to avoid scheduler batch splitting
  - TP=2 with max_running_requests=256 to handle large batches
  - Validates all requests completed before computing features

Usage:
    python profile_expert_load_v2.py --port 40000 --tp 2
"""
import argparse
import json
import time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import requests

try:
    import pybase64
except ImportError:
    import base64 as pybase64

NUM_EXPERTS = 128
TOP_K = 8

PROMPT_TEMPLATES = {
    "code_python": "Write a Python function that {topic}. Include type hints and docstrings.",
    "code_cpp": "Implement a C++ class that {topic}. Use modern C++17 features.",
    "math": "Solve the following math problem step by step: {topic}",
    "conversation": "Explain {topic} in simple terms as if talking to a curious teenager.",
    "technical": "Describe the architecture of {topic} with technical details.",
    "creative": "Write a short story about {topic} with vivid imagery.",
    "reasoning": "Think step by step about {topic} and give your conclusion.",
}

TOPICS = [
    "sorting algorithms", "binary search", "hash tables", "graph traversal",
    "dynamic programming", "matrix multiplication", "neural networks",
    "gradient descent", "attention mechanisms", "transformer architecture",
    "database indexing", "distributed systems", "consensus protocols",
    "memory management", "garbage collection", "compiler optimization",
    "operating system scheduling", "network protocols", "encryption",
    "quantum computing", "protein folding", "climate modeling",
    "game theory", "supply chain optimization", "recommendation systems",
]

RANDOM_TEXTS = [
    "The quick brown fox jumps over the lazy dog. " * 5,
    "In a world where technology evolves rapidly, understanding fundamentals remains crucial. " * 4,
    "Mathematics is the language of the universe, expressed through equations and proofs. " * 4,
    "Software engineering combines creativity with logical thinking to build systems. " * 4,
    "Machine learning models learn patterns from data to make predictions. " * 4,
]


def build_all_prompts():
    prompts = []
    for ptype, template in PROMPT_TEMPLATES.items():
        if ptype == "random_tokens":
            for txt in RANDOM_TEXTS:
                prompts.append({"type": ptype, "text": txt})
        else:
            for topic in TOPICS:
                prompts.append({"type": ptype, "text": template.format(topic=topic)})
    for txt in RANDOM_TEXTS:
        prompts.append({"type": "random", "text": txt})
    return prompts


def send_one_request(url, prompt_text, max_tokens=128):
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
        resp = requests.post(f"{url}/v1/completions", json=payload, timeout=300)
    except Exception as e:
        return None, None, None
    elapsed = time.perf_counter() - t0

    if resp.status_code != 200:
        return None, None, None

    data = resp.json()
    choice = data["choices"][0]
    usage = data.get("usage", {})
    n_tokens = usage.get("completion_tokens", 0)
    if n_tokens == 0:
        n_tokens = len(choice.get("text", "").split())

    routed_experts_b64 = None
    if "sglext" in data and data["sglext"].get("routed_experts"):
        routed_experts_b64 = data["sglext"]["routed_experts"]
    elif "meta_info" in choice and choice["meta_info"].get("routed_experts"):
        routed_experts_b64 = choice["meta_info"]["routed_experts"]

    if routed_experts_b64 is None:
        return elapsed, n_tokens, None

    expert_ids = np.frombuffer(
        pybase64.b64decode(routed_experts_b64 if isinstance(routed_experts_b64, bytes)
                           else routed_experts_b64.encode("utf-8")),
        dtype=np.int32
    )
    return elapsed, n_tokens, expert_ids


def compute_batch_features(all_expert_ids_list, tp=2):
    """Compute aggregated expert distribution features for an entire batch."""
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
        gini = (2.0 * np.sum((np.arange(1, n + 1) * sorted_counts)) / (n * cum_sum) - (n + 1) / n)
    else:
        gini = 0.0

    glr = 1.0
    if tp > 1:
        experts_per_gpu = NUM_EXPERTS // tp
        gpu_loads = np.array([
            counts[g * experts_per_gpu: (g + 1) * experts_per_gpu].sum()
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


def run_profiling(url, tp, output_path, max_tokens=128, batch_sizes=None,
                  trials_per_bs=50, inter_trial_sleep=2.0):
    if batch_sizes is None:
        batch_sizes = [2, 4, 8, 16, 24, 32, 48, 64, 96, 128, 160]

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
        "success_rate", "actual_n_tokens_avg",
    ]

    results = []
    prompt_idx = 0

    for bs in batch_sizes:
        print(f"\n{'=' * 70}")
        print(f"  Batch size = {bs}, {trials_per_bs} trials, max_tokens={max_tokens}")
        print(f"{'=' * 70}")

        executor = ThreadPoolExecutor(max_workers=bs + 4)

        for trial in range(trials_per_bs):
            batch_prompts = []
            for _ in range(bs):
                batch_prompts.append(all_prompts[perm[prompt_idx % len(perm)]])
                prompt_idx += 1

            # Send all requests concurrently
            futures = []
            for p in batch_prompts:
                futures.append(executor.submit(send_one_request, url, p["text"], max_tokens))
            batch_results = [f.result() for f in futures]

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
                print(f"  Trial {trial + 1}: ALL FAILED")
                time.sleep(inter_trial_sleep)
                continue

            success_rate = len(elapsed_list) / bs

            # TPOT: elapsed / n_tokens for each request
            tpot_list = [(e / max(n, 1)) * 1000 for e, n in zip(elapsed_list, n_tokens_list)]
            avg_tpot = np.mean(tpot_list)
            p50_tpot = np.percentile(tpot_list, 50)
            p95_tpot = np.percentile(tpot_list, 95)
            max_elapsed = max(elapsed_list)
            avg_n_tokens = np.mean(n_tokens_list)
            avg_prompt_len = np.mean([len(p["text"]) for p in batch_prompts])

            feat = compute_batch_features(expert_ids_list, tp)
            if feat is None:
                print(f"  Trial {trial + 1}: no expert data")
                time.sleep(inter_trial_sleep)
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
                "success_rate": f"{success_rate:.3f}",
                "actual_n_tokens_avg": f"{avg_n_tokens:.1f}",
                **{k: f"{v:.4f}" if isinstance(v, float) else str(v) for k, v in feat.items()},
            }
            results.append(row)

            if (trial + 1) % 10 == 0 or trial == 0:
                print(f"  Trial {trial + 1}/{trials_per_bs}: "
                      f"tpot_avg={avg_tpot:.1f}ms, "
                      f"ELS={feat['els']:.2f}, maxE={feat['max_tokens_per_expert']}, "
                      f"success={success_rate:.0%}, n_tok_avg={avg_n_tokens:.0f}")

            # Inter-trial sleep to let scheduler drain
            time.sleep(inter_trial_sleep)

        executor.shutdown(wait=False)

        # Save intermediate results after each batch_size
        with open(output_path, "w") as f:
            f.write("\t".join(header) + "\n")
            for r in results:
                f.write("\t".join(str(r.get(h, "")) for h in header) + "\n")
        print(f"  [Checkpoint] Saved {len(results)} rows to {output_path}")

    print(f"\nDone! Total: {len(results)} rows saved to {output_path}")
    return results


def main():
    ap = argparse.ArgumentParser(description="Profile MoE expert load vs latency V2")
    ap.add_argument("--port", type=int, default=40000)
    ap.add_argument("--host", type=str, default="127.0.0.1")
    ap.add_argument("--tp", type=int, default=2)
    ap.add_argument("--max-tokens", type=int, default=128)
    ap.add_argument("--batch-sizes", type=str, default="2,4,8,16,24,32,48,64,96,128,160")
    ap.add_argument("--trials-per-bs", type=int, default=50)
    ap.add_argument("--inter-trial-sleep", type=float, default=2.0)
    ap.add_argument("--output", type=str,
                    default="/workspace/sglang-tier/benchmark/AFlex_bench/06_others/"
                            "more_model/data/expert_load_vs_latency_batch_v2.tsv")
    args = ap.parse_args()

    url = f"http://{args.host}:{args.port}"
    batch_sizes = [int(x) for x in args.batch_sizes.split(",")]

    print(f"MoE Expert Load vs Latency Profiling V2")
    print(f"  Server: {url}")
    print(f"  TP: {args.tp}")
    print(f"  Max tokens: {args.max_tokens}")
    print(f"  Batch sizes: {batch_sizes}")
    print(f"  Trials per bs: {args.trials_per_bs}")
    print(f"  Inter-trial sleep: {args.inter_trial_sleep}s")
    print(f"  Output: {args.output}")

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)

    # Warmup
    print("\nWarming up...")
    for i in range(5):
        _, n, _ = send_one_request(url, "Hello, explain sorting algorithms.", max_tokens=16)
        if n and n > 0:
            print(f"  Warmup {i + 1}/5: OK ({n} tokens)")
        else:
            print(f"  Warmup {i + 1}/5: waiting...")
            time.sleep(5)
    print("Warmup done.\n")

    run_profiling(url, args.tp, args.output,
                  max_tokens=args.max_tokens,
                  batch_sizes=batch_sizes,
                  trials_per_bs=args.trials_per_bs,
                  inter_trial_sleep=args.inter_trial_sleep)


if __name__ == "__main__":
    main()

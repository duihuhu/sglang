#!/usr/bin/env python3
"""Analyze Azure LLM trace in 30s windows and compute Tier 1 workload parameters.

Usage:
    python workload_analysis.py
"""

import csv
from collections import defaultdict
from datetime import datetime, timezone
from statistics import mean, median

TRACE_PATH = "AzurePublicDataset/AzureLLMInferenceTrace_conv_1week.csv"
WINDOW_S = 30.0

# System performance assumptions (for n_active_decode simulation)
PREFILL_TOK_PER_S = 5000   # tokens/s per GPU (conservative for Llama 8B)
DECODE_TOK_PER_S = 50      # tokens/s per GPU (~20ms/token)


def parse_trace(path):
    """Parse CSV, return list of (timestamp_s, ctx_tokens, gen_tokens)."""
    records = []
    with open(path) as f:
        reader = csv.reader(f)
        header = next(reader)
        base_ts = None
        for row in reader:
            if len(row) < 3:
                continue
            ts_str, ctx_str, gen_str = row[0], row[1], row[2]
            try:
                ts = datetime.fromisoformat(ts_str)
                ctx = int(ctx_str)
                gen = int(gen_str)
            except (ValueError, IndexError):
                continue
            if base_ts is None:
                base_ts = ts
            elapsed = (ts - base_ts).total_seconds()
            records.append((elapsed, ctx, gen, ts))
    return records, base_ts


def simulate_concurrency(records, window_start, window_end):
    """Simulate a FIFO server to estimate how many requests are in decode phase
    during the window. Returns avg_active_decode (mean concurrent decodes)."""
    prefill_tput = PREFILL_TOK_PER_S
    decode_tput = DECODE_TOK_PER_S

    events = []  # (time, delta)  delta=+1 when enters decode, -1 when finishes
    for ts_arrival, ctx_tok, gen_tok, _ in records:
        # Request enters decode phase after prefill completes
        prefill_time = ctx_tok / prefill_tput
        ts_decode_start = ts_arrival + prefill_time
        total_decode_time = gen_tok / decode_tput
        ts_decode_end = ts_decode_start + total_decode_time

        # Only consider events that overlap with the window
        if ts_decode_end < window_start or ts_decode_start > window_end:
            continue

        events.append((max(ts_decode_start, window_start), +1))
        events.append((min(ts_decode_end, window_end), -1))

    if not events:
        return 0.0

    events.sort()
    active = 0
    prev_t = events[0][0]
    total_active_time = 0.0

    for t, delta in events:
        dt = t - prev_t
        total_active_time += active * dt
        active += delta
        prev_t = t

    return total_active_time / (window_end - window_start)


def estimate_n_active_decode_ltl(records_in_window):
    """Estimate n_active_decode using Little's Law:
    n_active_decode = lambda_prefill * avg_decode_time
    where avg_decode_time ≈ mean(GeneratedTokens) / DECODE_TOK_PER_S
    """
    if not records_in_window:
        return 0.0
    lam = len(records_in_window) / WINDOW_S
    avg_gen = mean(r[2] for r in records_in_window)
    avg_decode_time = avg_gen / DECODE_TOK_PER_S
    return lam * avg_decode_time


def estimate_bs_avg_p(records_in_window):
    """Estimate prefill batch size.
    With continuous batching, after a prefill completes, we batch all
    requests that arrived during the prefill. So bs_avg_p ≈
    lambda_prefill * avg_prefill_time, where avg_prefill_time is
    the time to process avg ContextTokens.
    """
    if not records_in_window:
        return 0.0
    lam = len(records_in_window) / WINDOW_S
    avg_ctx = mean(r[1] for r in records_in_window)
    avg_prefill_time = avg_ctx / PREFILL_TOK_PER_S
    return lam * avg_prefill_time


def analyze():
    records, base_ts = parse_trace(TRACE_PATH)
    print(f"Total records: {len(records)}")
    print(f"Time range: {base_ts} → {records[-1][3]}")
    total_duration = records[-1][0]
    print(f"Total duration: {total_duration:.1f}s ({total_duration/60:.1f} min)")
    print(f"Overall arrival rate: {len(records)/total_duration:.2f} req/s")
    print()

    # Partition into 30s windows
    max_window = int(total_duration // WINDOW_S) + 1
    windows = [[] for _ in range(max_window)]
    for r in records:
        elapsed = r[0]
        wi = int(elapsed // WINDOW_S)
        if wi < max_window:
            windows[wi].append(r)

    print(f"{'Window':<12} {'Count':<8} {'λ_prefill':<12} {'il_rep_p':<10} {'bs_avg_p':<10} {'il_rep_d':<10} {'ol_rep_d':<10} {'bs_avg_d':<10} {'n_act_d':<10}")
    print("=" * 110)

    for wi, w_records in enumerate(windows):
        w_start = wi * WINDOW_S
        w_end = min((wi + 1) * WINDOW_S, total_duration)
        w_label = f"{w_start:.0f}s-{w_end:.0f}s"

        if not w_records:
            print(f"{w_label:<12} {'0':<8} {'0.00':<12} {'N/A':<10} {'N/A':<10} {'N/A':<10} {'N/A':<10} {'N/A':<10} {'0.00':<10}")
            continue

        count = len(w_records)
        lam = count / WINDOW_S

        # il_rep_p: representative prefill input length
        ctx_tokens = [r[1] for r in w_records]
        il_rep_p_mean = mean(ctx_tokens)
        il_rep_p_med = median(ctx_tokens)

        # ol_rep_d: representative decode output length
        gen_tokens = [r[2] for r in w_records]
        ol_rep_d_mean = mean(gen_tokens)
        ol_rep_d_med = median(gen_tokens)

        # il_rep_d is always 1 for decode
        il_rep_d = 1

        # bs_avg_p: expected prefill batch size
        bs_avg_p = estimate_bs_avg_p(w_records)

        # n_active_decode using Little's Law
        n_active_ltl = estimate_n_active_decode_ltl(w_records)
        # Also via simulation
        n_active_sim = simulate_concurrency(records, w_start, w_end)

        # bs_avg_d: expected decode batch size ≈ n_active_decode per pair
        bs_avg_d = max(1, round(n_active_ltl))

        print(f"{w_label:<12} {count:<8} {lam:<12.2f} "
              f"{il_rep_p_mean:<10.0f} "
              f"{bs_avg_p:<10.1f} "
              f"{il_rep_d:<10} "
              f"{ol_rep_d_mean:<10.0f} "
              f"{bs_avg_d:<10} "
              f"{n_active_ltl:<10.1f}")

    # Summary: aggregate stats per parameter
    print()
    print("=" * 70)
    print("AGGREGATE STATISTICS (across all windows)")
    print("=" * 70)

    all_ctx = [r[1] for r in records]
    all_gen = [r[2] for r in records]

    print(f"\nContextTokens (il_rep_p):")
    print(f"  mean={mean(all_ctx):.0f}  median={median(all_ctx):.0f}  "
          f"min={min(all_ctx)}  max={max(all_ctx)}")
    all_ctx_sorted = sorted(all_ctx)
    for p in [25, 50, 75, 90, 95, 99]:
        idx = int(len(all_ctx_sorted) * p / 100)
        print(f"  p{p:02d}={all_ctx_sorted[idx]}")

    print(f"\nGeneratedTokens (ol_rep_d):")
    print(f"  mean={mean(all_gen):.0f}  median={median(all_gen):.0f}  "
          f"min={min(all_gen)}  max={max(all_gen)}")
    all_gen_sorted = sorted(all_gen)
    for p in [25, 50, 75, 90, 95, 99]:
        idx = int(len(all_gen_sorted) * p / 100)
        print(f"  p{p:02d}={all_gen_sorted[idx]}")

    # Overall recommended values
    print()
    print("=" * 70)
    print("RECOMMENDED CONFIG VALUES")
    print("=" * 70)
    il_rep_p_rec = round(mean(all_ctx))
    ol_rep_d_rec = round(mean(all_gen))

    # n_active_decode overall
    total_lam = len(records) / total_duration
    overall_n_active = total_lam * ol_rep_d_rec / DECODE_TOK_PER_S

    print(f"""
    "gpu_count": 4,
    "lambda_prefill": {total_lam:.1f},
    "n_active_decode": {overall_n_active:.0f},
    "il_rep_p": {il_rep_p_rec},
    "bs_avg_p": {max(1, round(total_lam * il_rep_p_rec / PREFILL_TOK_PER_S))},
    "il_rep_d": 1,
    "ol_rep_d": {ol_rep_d_rec},
    "bs_avg_d": {max(1, round(overall_n_active))},
    "monitor_window_s": 30.0,
""")

    print(f"Assumptions: prefill_throughput={PREFILL_TOK_PER_S} tok/s, "
          f"decode_throughput={DECODE_TOK_PER_S} tok/s ({1000/DECODE_TOK_PER_S:.0f}ms/tok)")


if __name__ == "__main__":
    analyze()

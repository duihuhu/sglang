#!/usr/bin/env python3
"""
T4-1: Azure LLM Trace preprocessing for energy-aware evaluation.

Reads DynamoLLM (HPCA'25) Azure traces and produces:
  1. Summary statistics (arrival rate, length distributions)
  2. Bucketed request types aligned with profiling data grid
  3. Multiple load levels (0.5x, 1x, 2x) for evaluation
  4. Processed trace files ready for simulator input

Input:  AzurePublicDataset/data/AzureLLMInferenceTrace_{code,conv}.csv
Output: trace_processed/ directory with:
        - {code,conv}_summary.json       (statistics)
        - {code,conv}_bucketed.csv       (bucketed il/ol + arrival)
        - {code,conv}_{low,mid,high}.csv (scaled load levels)

Usage:
    python prepare_trace.py
    python prepare_trace.py --trace-dir AzurePublicDataset/data
    python prepare_trace.py --plot  # generate distribution plots
"""

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_TRACE_DIR = SCRIPT_DIR / "AzurePublicDataset" / "data"
DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "trace_processed"

# Profiling data grid points for snapping
DECODE_IL_GRID = [128, 256, 512, 1024, 2048, 4096]
DECODE_OL_GRID = [64, 128, 256, 512, 1024, 2048, 4096]
PREFILL_IL_GRID = [128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32000]

# Load scaling factors
LOAD_SCALES = {"low": 0.5, "mid": 1.0, "high": 2.0}


def snap_to_grid(values: np.ndarray, grid: list[int]) -> np.ndarray:
    """Snap each value to the nearest grid point."""
    grid = np.array(grid)
    idx = np.searchsorted(grid, values, side="left")
    idx = np.clip(idx, 1, len(grid) - 1)
    left = grid[idx - 1]
    right = grid[idx]
    snapped = np.where(values - left <= right - values, left, right)
    return snapped


def load_trace(path: str) -> pd.DataFrame:
    """Load Azure trace CSV."""
    df = pd.read_csv(path)
    df.columns = df.columns.str.strip()
    df["TIMESTAMP"] = pd.to_datetime(df["TIMESTAMP"])
    df = df.sort_values("TIMESTAMP").reset_index(drop=True)
    return df


def compute_summary(df: pd.DataFrame, name: str) -> dict:
    """Compute trace summary statistics."""
    ts = df["TIMESTAMP"]
    duration_s = (ts.max() - ts.min()).total_seconds()
    n = len(df)

    ctx = df["ContextTokens"].values
    gen = df["GeneratedTokens"].values

    # Per-second arrival rate
    df_sec = df.set_index("TIMESTAMP").resample("1s").size()
    rps = df_sec.values

    summary = {
        "name": name,
        "num_requests": n,
        "duration_s": round(duration_s, 1),
        "avg_rps": round(n / duration_s, 2),
        "rps_p50": float(np.percentile(rps[rps > 0], 50)),
        "rps_p90": float(np.percentile(rps[rps > 0], 90)),
        "rps_p99": float(np.percentile(rps[rps > 0], 99)),
        "rps_max": int(rps.max()),
        "context_tokens": {
            "mean": round(ctx.mean(), 1),
            "p25": int(np.percentile(ctx, 25)),
            "p50": int(np.percentile(ctx, 50)),
            "p75": int(np.percentile(ctx, 75)),
            "p90": int(np.percentile(ctx, 90)),
            "p99": int(np.percentile(ctx, 99)),
            "max": int(ctx.max()),
        },
        "generated_tokens": {
            "mean": round(gen.mean(), 1),
            "p25": int(np.percentile(gen, 25)),
            "p50": int(np.percentile(gen, 50)),
            "p75": int(np.percentile(gen, 75)),
            "p90": int(np.percentile(gen, 90)),
            "p99": int(np.percentile(gen, 99)),
            "max": int(gen.max()),
        },
        "decode_il_out_of_range": int((ctx > max(DECODE_IL_GRID)).sum()),
        "decode_il_out_of_range_pct": round(
            (ctx > max(DECODE_IL_GRID)).mean() * 100, 2),
    }
    return summary


def bucket_trace(df: pd.DataFrame) -> pd.DataFrame:
    """Snap il/ol to profiling grid and compute inter-arrival times."""
    out = df.copy()

    # Snap to Decode grid (used for Decode phase modeling)
    out["il_decode"] = snap_to_grid(
        df["ContextTokens"].values, DECODE_IL_GRID)
    out["ol_decode"] = snap_to_grid(
        df["GeneratedTokens"].values, DECODE_OL_GRID)

    # Snap to Prefill grid (used for Prefill phase modeling)
    out["il_prefill"] = snap_to_grid(
        df["ContextTokens"].values, PREFILL_IL_GRID)

    # Inter-arrival time in ms
    ts = df["TIMESTAMP"]
    iat_ms = ts.diff().dt.total_seconds() * 1000
    iat_ms.iloc[0] = 0
    out["iat_ms"] = iat_ms.values

    # Relative timestamp in seconds from trace start
    out["t_s"] = (ts - ts.iloc[0]).dt.total_seconds()

    cols = ["t_s", "iat_ms", "ContextTokens", "GeneratedTokens",
            "il_prefill", "il_decode", "ol_decode"]
    return out[cols]


def scale_trace(df_bucketed: pd.DataFrame, scale: float) -> pd.DataFrame:
    """Scale arrival rate by adjusting inter-arrival times."""
    out = df_bucketed.copy()
    if scale != 1.0:
        out["iat_ms"] = out["iat_ms"] / scale
        out["t_s"] = out["iat_ms"].cumsum() / 1000.0
    return out


def print_bucket_distribution(df_bucketed: pd.DataFrame, name: str):
    """Print (il, ol) bucket distribution."""
    print(f"\n  {name} — Decode bucket distribution (il_decode × ol_decode):")
    ct = pd.crosstab(df_bucketed["il_decode"], df_bucketed["ol_decode"],
                     margins=True)
    print(ct.to_string(col_space=8))


def plot_distributions(df: pd.DataFrame, name: str, output_dir: Path):
    """Generate distribution plots."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("  [WARN] matplotlib not installed, skipping plots")
        return

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f"Azure Trace: {name}", fontsize=14)

    # 1. Arrival rate over time
    ax = axes[0, 0]
    ts = df["TIMESTAMP"]
    t_rel = (ts - ts.iloc[0]).dt.total_seconds()
    bins = np.arange(0, t_rel.max() + 60, 60)
    ax.hist(t_rel, bins=bins, edgecolor="black", alpha=0.7)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Requests per minute")
    ax.set_title("Arrival Rate")

    # 2. ContextTokens distribution
    ax = axes[0, 1]
    ax.hist(df["ContextTokens"], bins=50, edgecolor="black", alpha=0.7)
    ax.set_xlabel("ContextTokens (input length)")
    ax.set_ylabel("Count")
    ax.set_title("Input Length Distribution")
    ax.axvline(x=4096, color="r", linestyle="--", label="Decode il max=4096")
    ax.legend()

    # 3. GeneratedTokens distribution
    ax = axes[1, 0]
    ax.hist(df["GeneratedTokens"], bins=50, edgecolor="black", alpha=0.7)
    ax.set_xlabel("GeneratedTokens (output length)")
    ax.set_ylabel("Count")
    ax.set_title("Output Length Distribution")

    # 4. Scatter: il vs ol
    ax = axes[1, 1]
    ax.scatter(df["ContextTokens"], df["GeneratedTokens"],
               alpha=0.1, s=2)
    ax.set_xlabel("ContextTokens")
    ax.set_ylabel("GeneratedTokens")
    ax.set_title("Input vs Output Length")

    plt.tight_layout()
    out_path = output_dir / f"trace_{name}_dist.png"
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"  Plot saved: {out_path}")


def main():
    parser = argparse.ArgumentParser(
        description="T4-1: Azure trace preprocessing")
    parser.add_argument("--trace-dir", type=str,
                        default=str(DEFAULT_TRACE_DIR))
    parser.add_argument("--output-dir", type=str,
                        default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--plot", action="store_true",
                        help="Generate distribution plots")
    args = parser.parse_args()

    trace_dir = Path(args.trace_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    traces = {
        "code": trace_dir / "AzureLLMInferenceTrace_code.csv",
        "conv": trace_dir / "AzureLLMInferenceTrace_conv.csv",
    }

    print("=" * 70)
    print(" T4-1: Azure Trace Preprocessing")
    print("=" * 70)

    for name, path in traces.items():
        if not path.exists():
            print(f"\n  [SKIP] {path} not found")
            continue

        print(f"\n{'─' * 70}")
        print(f"  Processing: {name} ({path.name})")
        print(f"{'─' * 70}")

        # Load
        df = load_trace(str(path))
        print(f"  Loaded {len(df)} requests")

        # Summary
        summary = compute_summary(df, name)
        summary_path = output_dir / f"{name}_summary.json"
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2)
        print(f"  Summary saved: {summary_path}")
        print(f"    Duration: {summary['duration_s']}s, "
              f"Avg RPS: {summary['avg_rps']}, "
              f"Max RPS: {summary['rps_max']}")
        print(f"    ContextTokens:   p50={summary['context_tokens']['p50']}, "
              f"p90={summary['context_tokens']['p90']}, "
              f"max={summary['context_tokens']['max']}")
        print(f"    GeneratedTokens: p50={summary['generated_tokens']['p50']}, "
              f"p90={summary['generated_tokens']['p90']}, "
              f"max={summary['generated_tokens']['max']}")
        print(f"    Decode il out-of-range (>4096): "
              f"{summary['decode_il_out_of_range']} "
              f"({summary['decode_il_out_of_range_pct']}%)")

        # Bucket
        df_bucketed = bucket_trace(df)
        bucketed_path = output_dir / f"{name}_bucketed.csv"
        df_bucketed.to_csv(bucketed_path, index=False)
        print(f"  Bucketed trace saved: {bucketed_path}")
        print_bucket_distribution(df_bucketed, name)

        # Scaled load levels
        for level, scale in LOAD_SCALES.items():
            df_scaled = scale_trace(df_bucketed, scale)
            scaled_path = output_dir / f"{name}_{level}.csv"
            df_scaled.to_csv(scaled_path, index=False)
            effective_rps = summary["avg_rps"] * scale
            print(f"  {level} ({scale}x): {scaled_path.name} "
                  f"(effective RPS ≈ {effective_rps:.1f})")

        # Plots
        if args.plot:
            plot_distributions(df, name, output_dir)

    # Combined summary
    print(f"\n{'=' * 70}")
    print(f" Output directory: {output_dir}")
    print(f"{'=' * 70}")
    for f in sorted(output_dir.iterdir()):
        size_kb = f.stat().st_size / 1024
        print(f"  {f.name:<40} {size_kb:>8.1f} KB")


if __name__ == "__main__":
    main()

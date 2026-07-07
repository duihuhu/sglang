#!/usr/bin/env python3
"""Plot NVLink vs Mooncake RDMA Write comparison charts.

Reads results from results/ directory and generates:
  1. Throughput (GB/s) vs Message Size — all configs
  2. Latency (us) vs Message Size — all configs
  3. AFD-specific analysis: transfer time for realistic tensor sizes

Usage:
    python3 plot_comm_comparison.py [--results-dir results/] [--prefix <timestamp>]
    python3 plot_comm_comparison.py --use-embedded   # use reference data from read.md
"""

import argparse
import glob
import json
import os
import sys
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update({
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 12,
    "legend.fontsize": 10,
    "figure.dpi": 150,
})

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(SCRIPT_DIR, "results")
CHARTS_DIR = os.path.join(SCRIPT_DIR, "charts")

# Reference data from read.md (ib_write_bw single card, NVLink)
REF_RDMA_SINGLE = {
    2: 0.011, 4: 0.023, 8: 0.044, 16: 0.088, 32: 0.174, 64: 0.351,
    128: 0.701, 256: 1.403, 512: 2.785, 1024: 5.576, 2048: 10.598,
    4096: 16.986, 8192: 24.071, 16384: 24.545, 32768: 24.579,
    65536: 24.591, 131072: 24.596, 262144: 24.598, 524288: 24.599,
    1048576: 24.600, 2097152: 24.601, 4194304: 24.601, 8388608: 24.600,
}

REF_RDMA_4PORT = {
    2: 0.05, 4: 0.09, 8: 0.19, 16: 0.38, 32: 0.76, 64: 1.53,
    128: 3.07, 256: 6.11, 512: 11.97, 1024: 23.6, 2048: 43.7,
    4096: 69.7, 8192: 90.1, 16384: 91.5, 32768: 91.6,
    65536: 91.6, 131072: 91.6, 262144: 91.6, 524288: 91.6,
    1048576: 91.6, 2097152: 91.3, 4194304: 91.6, 8388608: 87.9,
}

REF_RDMA_LATENCY = {
    2: 1.82, 4: 1.82, 8: 1.83, 16: 1.82, 32: 1.86, 64: 1.86,
    128: 1.93, 256: 2.70, 512: 2.74, 1024: 2.90, 2048: 2.95,
    4096: 3.55, 8192: 3.78, 16384: 4.13, 32768: 4.82,
    65536: 6.15, 131072: 8.81, 262144: 14.12, 524288: 24.76,
    1048576: 46.73, 2097152: 89.44, 4194304: 174.69, 8388608: 345.19,
}

# NVLink data from read.md (reconstructed)
REF_NVLINK = {}
nvlink_raw = [
    (1, 1, 41.96, 0.24), (1, 2, 36.59, 0.56), (1, 4, 36.55, 1.12),
    (1, 8, 37.44, 2.19), (1, 16, 36.77, 4.46), (1, 32, 36.74, 8.92),
    (1, 64, 36.91, 17.76), (1, 128, 37.08, 35.34), (1, 256, 36.80, 71.23),
    (128, 1, 37.05, 35.38), (128, 2, 37.11, 70.63), (128, 4, 48.64, 107.79),
    (128, 8, 81.49, 128.67), (128, 16, 146.05, 143.60),
    (128, 32, 282.52, 148.46), (128, 64, 549.31, 152.71),
    (128, 128, 965.07, 173.84), (128, 256, 1909.54, 175.72),
    (256, 256, 3801.44, 176.54), (512, 256, 7590.20, 176.83),
    (1024, 256, 15163.62, 177.03), (2048, 256, 30335.97, 176.98),
    (4096, 256, 60633.29, 177.09), (8192, 256, 121238.55, 177.13),
]
for seq, bs, lat_us, tp_GBs in nvlink_raw:
    size = int(tp_GBs * 1e9 * lat_us * 1e-6)
    REF_NVLINK[size] = {"bw_GBs": tp_GBs, "lat_us": lat_us}


def load_json_results(filepath: str) -> Dict:
    with open(filepath) as f:
        return json.load(f)


def find_latest_results(results_dir: str, prefix: str = None) -> Dict[str, str]:
    """Find the latest result files for each test type."""
    found = {}
    patterns = {
        "nvlink": "nvlink_*.json",
        "rdma_single": "rdma_single_*.json",
        "rdma_4nic": "rdma_4nic_*.json",
        "rdma_batch64": "rdma_batch64_*.json",
        "rdma_afd": "rdma_afd_*.json",
    }
    for key, pat in patterns.items():
        files = sorted(glob.glob(os.path.join(results_dir, pat)))
        if prefix:
            files = [f for f in files if prefix in f]
        if files:
            found[key] = files[-1]
    return found


def extract_bw_lat(data: Dict) -> Tuple[List[int], List[float], List[float]]:
    """Extract sizes, bandwidths, latencies from result dict."""
    sizes, bws, lats = [], [], []
    for key, val in sorted(data.items(), key=lambda x: int(x[0])):
        s = int(key)
        sizes.append(s)
        bws.append(val.get("bw_GBs", 0))
        lats.append(val.get("p50_us", 0))
    return sizes, bws, lats


def plot_comparison(results_files: Dict[str, str], use_embedded: bool = False):
    """Generate comparison plots."""
    os.makedirs(CHARTS_DIR, exist_ok=True)

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # Collect data
    datasets = {}

    if use_embedded or not results_files:
        # Use reference data
        sizes_ref = sorted(REF_RDMA_SINGLE.keys())
        datasets["ib_write (1×, ref)"] = {
            "sizes": sizes_ref,
            "bw": [REF_RDMA_SINGLE[s] for s in sizes_ref],
            "lat": [REF_RDMA_LATENCY.get(s, 0) for s in sizes_ref],
            "color": "#90CAF9", "ls": "--", "marker": ".",
        }
        sizes_4p = sorted(REF_RDMA_4PORT.keys())
        datasets["ib_write (4×, ref)"] = {
            "sizes": sizes_4p,
            "bw": [REF_RDMA_4PORT[s] for s in sizes_4p],
            "lat": None,
            "color": "#42A5F5", "ls": "--", "marker": ".",
        }
        nvlink_sorted = sorted(REF_NVLINK.items(), key=lambda x: x[0])
        datasets["NVLink (ref)"] = {
            "sizes": [x[0] for x in nvlink_sorted],
            "bw": [x[1]["bw_GBs"] for x in nvlink_sorted],
            "lat": [x[1]["lat_us"] for x in nvlink_sorted],
            "color": "#FFAB91", "ls": "--", "marker": ".",
        }

    # Load actual results
    if "nvlink" in results_files:
        data = load_json_results(results_files["nvlink"])
        if "nvlink" in data:
            data = data["nvlink"]
        sizes, bws, lats = extract_bw_lat(data)
        datasets["NVLink (measured)"] = {
            "sizes": sizes, "bw": bws, "lat": lats,
            "color": "#FF5722", "ls": "-", "marker": "s",
        }

    if "rdma_single" in results_files:
        data = load_json_results(results_files["rdma_single"])
        sizes, bws, lats = extract_bw_lat(data)
        datasets["Mooncake 1×NIC"] = {
            "sizes": sizes, "bw": bws, "lat": lats,
            "color": "#2196F3", "ls": "-", "marker": "o",
        }

    if "rdma_4nic" in results_files:
        data = load_json_results(results_files["rdma_4nic"])
        if "aggregate" in data:
            agg = data["aggregate"]
            sizes = sorted([int(k) for k in agg.keys()])
            bws = [agg[str(s)]["aggregate_bw_GBs"] for s in sizes]
            lats = [agg[str(s)].get("max_p50_us", 0) for s in sizes]
            datasets["Mooncake 4×NIC"] = {
                "sizes": sizes, "bw": bws, "lat": lats,
                "color": "#1565C0", "ls": "-", "marker": "D",
            }

    if "rdma_batch64" in results_files:
        data = load_json_results(results_files["rdma_batch64"])
        sizes, bws, lats = extract_bw_lat(data)
        datasets["Mooncake batch(64)"] = {
            "sizes": sizes, "bw": bws, "lat": lats,
            "color": "#4CAF50", "ls": "-", "marker": "^",
        }

    # ── Plot 1: Throughput ──
    ax = axes[0, 0]
    for label, d in datasets.items():
        if d["bw"]:
            ax.semilogx(d["sizes"], d["bw"], marker=d["marker"], linestyle=d["ls"],
                        color=d["color"], linewidth=2, markersize=5, label=label)
    ax.set_xlabel("Message Size (Bytes)")
    ax.set_ylabel("Throughput (GB/s)")
    ax.set_title("Throughput vs Message Size")
    ax.legend(loc="upper left", fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 200)
    ax.axhline(y=177, color="#FF5722", linestyle=":", alpha=0.4)
    ax.axhline(y=24.6, color="#2196F3", linestyle=":", alpha=0.4)
    ax.axhline(y=91.6, color="#1565C0", linestyle=":", alpha=0.4)
    ax.text(2, 180, "177 GB/s NVLink peak", fontsize=8, color="#FF5722", alpha=0.6)
    ax.text(2, 27, "24.6 GB/s 1×RDMA peak", fontsize=8, color="#2196F3", alpha=0.6)
    ax.text(2, 94, "91.6 GB/s 4×RDMA peak", fontsize=8, color="#1565C0", alpha=0.6)

    # ── Plot 2: Latency ──
    ax = axes[0, 1]
    for label, d in datasets.items():
        if d.get("lat") and any(x > 0 for x in d["lat"]):
            ax.loglog(d["sizes"], d["lat"], marker=d["marker"], linestyle=d["ls"],
                      color=d["color"], linewidth=2, markersize=5, label=label)
    ax.set_xlabel("Message Size (Bytes)")
    ax.set_ylabel("Latency (μs)")
    ax.set_title("Latency vs Message Size")
    ax.legend(loc="upper left", fontsize=9)
    ax.grid(True, alpha=0.3)

    # ── Plot 3: Effective bandwidth for KV transfer scenarios ──
    ax = axes[1, 0]
    # Qwen3-32B KV transfer sizes: tokens × layers × per_token_per_layer
    per_token_per_layer = 4096  # bytes (8 heads × 128 dim × 2 KV × bf16)
    num_layers = 64
    token_counts = [1, 4, 16, 64, 128, 256, 512, 1024, 2048]
    transfer_sizes = [t * num_layers * per_token_per_layer for t in token_counts]

    # Compute transfer time for each config
    for label, d in datasets.items():
        if not d["bw"] or not d["sizes"]:
            continue
        # Interpolate BW for each transfer size
        times_ms = []
        for ts in transfer_sizes:
            # Find closest size with BW data
            closest_bw = None
            for s, bw in zip(d["sizes"], d["bw"]):
                if bw > 0:
                    if closest_bw is None or abs(s - ts) < abs(closest_bw[0] - ts):
                        closest_bw = (s, bw)
            if closest_bw and closest_bw[1] > 0:
                time_ms = ts / (closest_bw[1] * 1e9) * 1000
                times_ms.append(time_ms)
            else:
                times_ms.append(None)

        valid = [(t, tm) for t, tm in zip(token_counts, times_ms) if tm is not None]
        if valid:
            ax.semilogy([v[0] for v in valid], [v[1] for v in valid],
                        marker=d["marker"], linestyle=d["ls"], color=d["color"],
                        linewidth=2, markersize=5, label=label)

    ax.set_xlabel("Tokens to Transfer")
    ax.set_ylabel("Transfer Time (ms)")
    ax.set_title("KV Cache Transfer Time (Qwen3-32B, 64 layers)")
    ax.legend(loc="upper left", fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_xticks(token_counts)
    ax.set_xticklabels([str(t) for t in token_counts], rotation=45)

    # ── Plot 4: Bandwidth ratio (NVLink / RDMA) ──
    ax = axes[1, 1]
    # Compare with reference data
    if "NVLink (measured)" in datasets and "Mooncake 1×NIC" in datasets:
        nvl = datasets["NVLink (measured)"]
        rdma1 = datasets["Mooncake 1×NIC"]
    elif "NVLink (ref)" in datasets and "ib_write (1×, ref)" in datasets:
        nvl = datasets["NVLink (ref)"]
        rdma1 = datasets["ib_write (1×, ref)"]
    else:
        nvl, rdma1 = None, None

    if nvl and rdma1:
        common_sizes = sorted(set(nvl["sizes"]) & set(rdma1["sizes"]))
        if common_sizes:
            nvl_dict = dict(zip(nvl["sizes"], nvl["bw"]))
            rdma_dict = dict(zip(rdma1["sizes"], rdma1["bw"]))
            ratios = []
            for s in common_sizes:
                if nvl_dict.get(s, 0) > 0 and rdma_dict.get(s, 0) > 0:
                    ratios.append(nvl_dict[s] / rdma_dict[s])
                else:
                    ratios.append(0)
            ax.semilogx(common_sizes, ratios, "o-", color="#9C27B0", linewidth=2)
            ax.axhline(y=1, color="gray", linestyle="--", alpha=0.5)
            ax.set_xlabel("Message Size (Bytes)")
            ax.set_ylabel("NVLink / RDMA (1×) Ratio")
            ax.set_title("NVLink vs RDMA Bandwidth Ratio")
            ax.grid(True, alpha=0.3)

    # If 4-NIC data available, add ratio line
    if nvl:
        rdma4_key = next((k for k in datasets if "4" in k and "NIC" in k), None)
        if rdma4_key:
            rdma4 = datasets[rdma4_key]
            common = sorted(set(nvl["sizes"]) & set(rdma4["sizes"]))
            if common:
                nvl_d = dict(zip(nvl["sizes"], nvl["bw"]))
                r4_d = dict(zip(rdma4["sizes"], rdma4["bw"]))
                ratios4 = [nvl_d.get(s, 0) / r4_d[s] if r4_d.get(s, 0) > 0 else 0
                           for s in common]
                ax.semilogx(common, ratios4, "D-", color="#E91E63", linewidth=2,
                            label="NVLink / RDMA(4×)")
                ax.legend(fontsize=9)

    plt.tight_layout()
    out_png = os.path.join(CHARTS_DIR, "mooncake_rdma_vs_nvlink.png")
    out_pdf = os.path.join(CHARTS_DIR, "mooncake_rdma_vs_nvlink.pdf")
    plt.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.savefig(out_pdf, bbox_inches="tight")
    print(f"Charts saved to:\n  {out_png}\n  {out_pdf}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Plot NVLink vs RDMA comparison")
    parser.add_argument("--results-dir", default=RESULTS_DIR)
    parser.add_argument("--prefix", default=None, help="Timestamp prefix to filter results")
    parser.add_argument("--use-embedded", action="store_true",
                        help="Use reference data from read.md (no actual measurements needed)")
    args = parser.parse_args()

    results_dir = args.results_dir

    results_files = find_latest_results(results_dir, args.prefix)
    if not results_files and not args.use_embedded:
        print("No result files found. Using embedded reference data.")
        args.use_embedded = True

    if results_files:
        print("Found result files:")
        for k, v in results_files.items():
            print(f"  {k}: {os.path.basename(v)}")

    plot_comparison(results_files, use_embedded=args.use_embedded)


if __name__ == "__main__":
    main()

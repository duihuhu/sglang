#!/usr/bin/env python3
"""Compare single-node vs cross-node scalability for Qwen3-32B.

Combines:
  - Single-node (1-node) data from retesting/micro_benchmark/{4,8}gpu/results/
  - Cross-node (2-node) data from node_scalibility/results/scal_{4,8}card_*.json

Produces comparison tables + charts over:
  6 schemes (native_dp/pd_dp/pdaf x baseline/tier)
  2 deployments (1-node, 2-node)
  2 GPU totals (4-card, 8-card)
  4 datasets (chatbot/qa/rag/summary)
  multiple QPS
Metrics: TTFT, TPOT, total energy, energy/token, throughput.

Usage:
    python3 analyze_scalability.py
"""
import json
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
RESULTS_DIR = HERE / "results"
CHARTS_DIR = HERE / "charts"
SINGLE_BASE = Path("/mnt/workspace/lt/sglang/benchmark/AFlex_bench/retesting/micro_benchmark")

SCENARIOS = ["chatbot", "qa", "rag", "summary"]
SCHEMES = ["native_dp", "pd_dp", "pdaf"]
MODES = ["baseline", "tier"]

# Metrics of interest (key in JSON -> display label)
METRICS = {
    "throughput_tok_s": "Throughput (tok/s)",
    "ttft_proc_avg_ms": "TTFT (ms)",
    "tpot_avg_ms": "TPOT (ms)",
    "total_energy_j": "Total Energy (J)",
    "energy_per_token_mj": "Energy/token (mJ)",
}


def _latest(globpat, directory):
    files = sorted(directory.glob(globpat))
    return files[-1] if files else None


def _merge_into(dst, src):
    """Merge {scheme_mode: {wl: metrics}} src into dst (src wins on conflict)."""
    for sk, wl in src.items():
        if not isinstance(wl, dict):
            continue
        d = dst.setdefault(sk, {})
        for w, m in wl.items():
            if isinstance(m, dict) and m.get("status") == "PASS":
                d[w] = m


def load_single_node(ngpu):
    """Load 1-node data for given card count. Returns {scheme_mode: {wl: metrics}}."""
    out = {}
    rdir = SINGLE_BASE / f"{ngpu}gpu/results"
    if not rdir.is_dir():
        return out
    # Merge all result files (later timestamps win). Tier files carry _tier suffix keys.
    for f in sorted(rdir.glob(f"micro_{ngpu}gpu*.json")):
        try:
            data = json.load(open(f))
        except Exception:
            continue
        if not isinstance(data, dict) or not data:
            continue
        _merge_into(out, data)
    return out


def load_cross_node(ngpu):
    """Load 2-node data for given card count. Returns {scheme_mode: {wl: metrics}}."""
    out = {}
    for f in sorted(RESULTS_DIR.glob(f"scal_{ngpu}card_*.json")):
        try:
            data = json.load(open(f))
        except Exception:
            continue
        res = data.get("results", {})
        _merge_into(out, res)
    return out


def get_metric(store, scheme, mode, scenario, qps, key):
    sk = scheme if mode == "baseline" else f"{scheme}_tier"
    wl = store.get(sk, {})
    m = wl.get(f"{scenario}_qps{qps}")
    if not m or m.get("status") != "PASS":
        return None
    return m.get(key)


def discover_qps(stores):
    qps = set()
    for store in stores:
        for wl in store.values():
            for k in wl:
                mm = re.match(r".+_qps(\d+)$", k)
                if mm:
                    qps.add(int(mm.group(1)))
    return sorted(qps)


# ============================================================
# Markdown table
# ============================================================

def build_markdown(single4, cross4, single8, cross8):
    lines = []
    lines.append("# 跨节点 vs 单机可扩展性对比 (Qwen3-32B)\n")
    lines.append("对比 6 种方案（native_dp/pd_dp/pdaf × baseline/tier）× "
                 "2 种部署（单机1节点/跨机2节点）× 2 种 GPU 总数（4卡/8卡）"
                 "× 4 数据集 × 多 QPS。\n")
    lines.append("- **单机**: 单节点 N 卡（10.252.129.35 的后 N 卡）")
    lines.append("- **跨机**: 两节点各 N/2 卡（NIC 亲和布局，PDAF 8卡 PA 用 GPU4,6 "
                 "→ mlx5_4+mlx5_5 两张网卡）")
    lines.append("- **指标**: TTFT(ms) / TPOT(ms) / 吞吐(tok/s) / 总能耗(J) / energy-per-token(mJ)\n")

    configs = [(4, single4, cross4), (8, single8, cross8)]
    for ngpu, single, cross in configs:
        lines.append(f"\n## {ngpu} 卡\n")
        qps_list = discover_qps([single, cross])
        for mode in MODES:
            lines.append(f"\n### {ngpu}卡 — {mode}\n")
            hdr = ("| 方案 | 部署 | 数据集 | QPS | 吞吐(tok/s) | TTFT(ms) | "
                   "TPOT(ms) | 总能耗(J) | mJ/tok |")
            lines.append(hdr)
            lines.append("|---|---|---|---|---|---|---|---|---|")
            for scheme in SCHEMES:
                for scenario in SCENARIOS:
                    for qps in qps_list:
                        for dep_name, store in [("单机", single), ("跨机", cross)]:
                            thpt = get_metric(store, scheme, mode, scenario, qps, "throughput_tok_s")
                            if thpt is None:
                                continue
                            ttft = get_metric(store, scheme, mode, scenario, qps, "ttft_proc_avg_ms")
                            tpot = get_metric(store, scheme, mode, scenario, qps, "tpot_avg_ms")
                            etot = get_metric(store, scheme, mode, scenario, qps, "total_energy_j")
                            ept = get_metric(store, scheme, mode, scenario, qps, "energy_per_token_mj")
                            lines.append(
                                f"| {scheme} | {dep_name} | {scenario} | {qps} | "
                                f"{thpt:.1f} | {ttft:.1f} | {tpot:.1f} | "
                                f"{etot:.0f} | {ept:.1f} |")
    return "\n".join(lines)


# ============================================================
# Charts: single vs cross-node, per metric, per scheme
# ============================================================

def plot_metric_grid(ngpu, single, cross, metric_key, metric_label):
    qps_list = discover_qps([single, cross])
    if not qps_list:
        return
    fig, axes = plt.subplots(len(SCHEMES), len(SCENARIOS),
                             figsize=(5 * len(SCENARIOS), 4 * len(SCHEMES)),
                             squeeze=False)
    fig.suptitle(f"{ngpu}-card: {metric_label} — single-node vs cross-node (baseline)",
                 fontsize=14, fontweight="bold")
    for si, scheme in enumerate(SCHEMES):
        for ci, scenario in enumerate(SCENARIOS):
            ax = axes[si][ci]
            for dep_name, store, color, mk in [
                ("1-node", single, "#2196F3", "o-"),
                ("2-node", cross, "#FF5722", "s--")]:
                xs, ys = [], []
                for qps in qps_list:
                    v = get_metric(store, scheme, "baseline", scenario, qps, metric_key)
                    if v is not None:
                        xs.append(qps)
                        ys.append(v)
                if xs:
                    ax.plot(xs, ys, mk, color=color, label=dep_name, linewidth=2)
            ax.set_title(f"{scheme} / {scenario}", fontsize=10)
            ax.set_xlabel("QPS")
            ax.set_ylabel(metric_label)
            ax.grid(True, alpha=0.3)
            ax.legend(fontsize=8)
    plt.tight_layout()
    CHARTS_DIR.mkdir(parents=True, exist_ok=True)
    out = CHARTS_DIR / f"{ngpu}card_{metric_key}_single_vs_cross.png"
    fig.savefig(out, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {out}")


def main():
    single4 = load_single_node(4)
    cross4 = load_cross_node(4)
    single8 = load_single_node(8)
    cross8 = load_cross_node(8)

    print("Loaded stores:")
    for name, s in [("single4", single4), ("cross4", cross4),
                    ("single8", single8), ("cross8", cross8)]:
        keys = {k: len(v) for k, v in s.items()}
        print(f"  {name}: {keys}")

    # Markdown report
    md = build_markdown(single4, cross4, single8, cross8)
    out_md = HERE / "scalability_comparison.md"
    out_md.write_text(md)
    print(f"Markdown report: {out_md}")

    # Charts
    for ngpu, single, cross in [(4, single4, cross4), (8, single8, cross8)]:
        for mk, ml in METRICS.items():
            plot_metric_grid(ngpu, single, cross, mk, ml)


if __name__ == "__main__":
    main()

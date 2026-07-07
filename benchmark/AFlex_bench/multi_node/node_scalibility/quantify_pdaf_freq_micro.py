#!/usr/bin/env python3
"""Quantify PDAF+Tier decode frequency trajectories for micro workloads.

Runs 16-card PDAF+Tier with AFD_DVFS_DECISION_LOG enabled, replays one
workload per dataset, and compares decode (f_a, f_f) vs obs_iter_us.

Usage:
    python3 quantify_pdaf_freq_micro.py --scenario qa_lpld,chatbot_lphd,summary_hphd --qps 8
"""
from __future__ import annotations

import argparse
import asyncio
import json
import logging
import statistics
import sys
import time
from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import run_node_scalability as RS  # noqa: E402

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s",
                    datefmt="%H:%M:%S")
log = logging.getLogger("quantify_freq")

OUT_BASE = HERE / "results" / "quantify_pdaf_freq"
CONTAINER_LOG = (
    "/workspace/sglang/benchmark/AFlex_bench/multi_node/"
    "node_scalibility/results/quantify_pdaf_freq"
)
MAX_GPU_FREQ = 1410

_orig_afd_env = RS._afd_env


def _afd_env_with_log(role, gpus, tp, attn_gpus, ffn_gpus, session: str):
    base = _orig_afd_env(role, gpus, tp, attn_gpus, ffn_gpus)
    cpath = (
        f"{CONTAINER_LOG}/{session}/"
        "dvfs_decisions_{persp}_{disagg}_gpu{gpu}.jsonl"
    )
    return base.replace(";", f" AFD_DVFS_DECISION_LOG='{cpath}';", 1)


def parse_decode_logs(log_dir: Path) -> list[dict]:
    rows = []
    for f in sorted(log_dir.glob("dvfs_decisions_*decode*.jsonl")):
        for line in f.read_text().splitlines():
            if line.strip():
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if rec.get("phase") == "decode":
                    rows.append(rec)
    rows.sort(key=lambda r: r.get("t", 0))
    return rows


def summarize_decode(rows: list[dict]) -> dict:
    if not rows:
        return {}
    fa = [r.get("cur_f_a") or r.get("sel_f_a", 0) for r in rows]
    ff = [r.get("cur_f_f") or r.get("sel_f_f", 0) for r in rows]
    obs = [r.get("obs_iter_us", 0) for r in rows if r.get("obs_iter_us")]
    pred = [r.get("pred_iter_cur_us", 0) for r in rows if r.get("pred_iter_cur_us")]
    slo = rows[0].get("slo_tpot_us", 0)
    pair = Counter((a, f) for a, f in zip(fa, ff))
    max_freq = MAX_GPU_FREQ
    at_max = sum(1 for a, f in zip(fa, ff) if a >= max_freq - 30 and f >= max_freq - 30)
    return {
        "n_decode_windows": len(rows),
        "f_a_mean": round(statistics.mean(fa), 0),
        "f_f_mean": round(statistics.mean(ff), 0),
        "f_a_p50": round(float(np.percentile(fa, 50)), 0),
        "f_f_p50": round(float(np.percentile(ff, 50)), 0),
        "pct_at_max_freq": round(at_max / len(rows) * 100, 1),
        "freq_pair_top5": pair.most_common(5),
        "obs_iter_us_mean": round(statistics.mean(obs), 1) if obs else 0,
        "obs_iter_us_p50": round(float(np.percentile(obs, 50)), 1) if obs else 0,
        "pred_iter_us_mean": round(statistics.mean(pred), 1) if pred else 0,
        "slo_tpot_us": slo,
        "slo_util_pct": round(statistics.mean(obs) / slo * 100, 1) if obs and slo else 0,
        "bs_mean": round(statistics.mean([r.get("bs", 0) for r in rows]), 1),
        "ol_mean": round(statistics.mean([r.get("ol", 0) for r in rows if r.get("ol")]), 0),
    }


def run_one(session: str, scenario: str, qps: int, max_run_s: int) -> dict:
    log_dir = OUT_BASE / session
    log_dir.mkdir(parents=True, exist_ok=True)
    gpus = RS.card_gpus(16)

    RS.cleanup_all()
    RS._afd_env = lambda *a, **kw: _afd_env_with_log(*a, **kw, session=session)
    try:
        url = RS.start_pdaf(16, tier=True)
    finally:
        RS._afd_env = _orig_afd_env

    if url is None:
        return {"status": "DEPLOY_FAILED"}

    RS.lock_freq_both(gpus, MAX_GPU_FREQ)
    if not RS.test_generate(url):
        RS.cleanup_all()
        return {"status": "WARMUP_FAILED"}

    t_start = time.time()
    wl_key, metrics = RS.run_one_workload(url, scenario, qps, gpus, gpus, max_run_s)
    t_end = time.time()

    decode_rows = parse_decode_logs(log_dir)
    summary = summarize_decode(decode_rows)
    RS.unlock_freq_both(gpus)
    RS.cleanup_all()

    return {
        "status": "PASS",
        "scenario": scenario,
        "qps": qps,
        "wl_key": wl_key,
        "metrics": metrics,
        "dvfs_decode": summary,
        "n_decode_records": len(decode_rows),
        "t_start": t_start,
        "t_end": t_end,
        "_decode_rows": decode_rows,
    }


def plot_results(all_results: dict, out_dir: Path):
    scenarios = list(all_results.keys())
    labels = {
        "qa_lpld": "QA (128/64)",
        "chatbot_lphd": "Chatbot (128/1024)",
        "summary_hphd": "Summary (4096/1024)",
    }

    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    fig.suptitle("PDAF+Tier Decode Frequency Verification | 16-card QPS=8",
                 fontsize=14, fontweight="bold")

    # (a) Mean f_a / f_f bar chart
    ax = axes[0, 0]
    x = np.arange(len(scenarios))
    w = 0.35
    fa_vals = [all_results[s]["dvfs_decode"].get("f_a_mean", 0) for s in scenarios]
    ff_vals = [all_results[s]["dvfs_decode"].get("f_f_mean", 0) for s in scenarios]
    ax.bar(x - w / 2, fa_vals, w, label="f_A (Attn)", color="#1f77b4")
    ax.bar(x + w / 2, ff_vals, w, label="f_F (FFN)", color="#ff7f0e")
    ax.axhline(MAX_GPU_FREQ, color="red", ls="--", alpha=0.5, label="Max 1410MHz")
    ax.set_xticks(x)
    ax.set_xticklabels([labels.get(s, s) for s in scenarios], fontsize=9)
    ax.set_ylabel("Frequency (MHz)")
    ax.set_title("(a) Mean Decode Frequency")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3, axis="y")

    # (b) % time at max freq
    ax = axes[0, 1]
    pct_max = [all_results[s]["dvfs_decode"].get("pct_at_max_freq", 0) for s in scenarios]
    colors = ["#2ca02c", "#ff7f0e", "#d62728"]
    ax.bar([labels.get(s, s) for s in scenarios], pct_max, color=colors[: len(scenarios)])
    ax.set_ylabel("% decode windows at max freq")
    ax.set_title("(b) Frequency Headroom Used")
    ax.set_ylim(0, 105)
    ax.grid(True, alpha=0.3, axis="y")

    # (c) obs_iter_us vs SLO
    ax = axes[1, 0]
    obs_mean = [all_results[s]["dvfs_decode"].get("obs_iter_us_mean", 0) / 1000 for s in scenarios]
    slo = all_results[scenarios[0]]["dvfs_decode"].get("slo_tpot_us", 50000) / 1000
    ax.bar([labels.get(s, s) for s in scenarios], obs_mean, color=colors[: len(scenarios)],
           label="obs iter")
    ax.axhline(slo, color="red", ls="--", label=f"SLO TPOT={slo:.0f}ms")
    ax.set_ylabel("Decode iteration (ms)")
    ax.set_title("(c) Observed Decode Iter Time vs SLO")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3, axis="y")

    # (d) freq trajectory over time (decode phase, DA log)
    ax = axes[1, 1]
    for s in scenarios:
        rows = all_results[s].get("_decode_rows", [])
        if not rows:
            continue
        t0 = rows[0].get("t", 0)
        ts = [(r.get("t", 0) - t0) for r in rows]
        fa = [r.get("cur_f_a") or r.get("sel_f_a", 0) for r in rows]
        ax.plot(ts, fa, label=labels.get(s, s), linewidth=1.2, alpha=0.85)
    ax.set_xlabel("Time since decode start (s)")
    ax.set_ylabel("f_A (MHz)")
    ax.set_title("(d) f_A trajectory during decode")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    out = out_dir / "pdaf_tier_freq_verification.png"
    fig.savefig(out, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {out}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenario", default="qa_lpld,chatbot_lphd,summary_hphd")
    parser.add_argument("--qps", type=int, default=8)
    parser.add_argument("--max-run-s", type=int, default=400)
    args = parser.parse_args()

    scenarios = args.scenario.split(",")
    OUT_BASE.mkdir(parents=True, exist_ok=True)
    all_results = {}

    for sc in scenarios:
        session = f"{sc}_qps{args.qps}"
        log.info("=" * 60)
        log.info("Running %s QPS=%d with DVFS logging", sc, args.qps)
        res = run_one(session, sc, args.qps, args.max_run_s)
        all_results[sc] = res
        if res.get("status") != "PASS":
            log.error("  FAILED: %s", res.get("status"))
            continue
        d = res["dvfs_decode"]
        m = res["metrics"]
        log.info("  Energy=%.1f mJ/tok TPOT_p50=%.1fms",
                 m.get("energy_per_token_mj", 0), m.get("tpot_p50_ms", 0))
        log.info("  Decode: f_a=%.0f f_f=%.0f at_max=%.1f%% obs_iter=%.1fms SLO_util=%.1f%%",
                 d.get("f_a_mean", 0), d.get("f_f_mean", 0),
                 d.get("pct_at_max_freq", 0),
                 d.get("obs_iter_us_mean", 0) / 1000,
                 d.get("slo_util_pct", 0))

    # strip raw rows for json save
    save = {}
    for sc, res in all_results.items():
        save[sc] = {k: v for k, v in res.items() if not k.startswith("_")}

    out_json = OUT_BASE / f"freq_verify_qps{args.qps}.json"
    json.dump(save, open(out_json, "w"), indent=2)
    print(f"saved {out_json}")

    ok = {k: v for k, v in all_results.items() if v.get("status") == "PASS"}
    if ok:
        plot_results(all_results, OUT_BASE / "charts")
        charts_dir = HERE / "charts"
        charts_dir.mkdir(exist_ok=True)
        import shutil
        src = OUT_BASE / "charts" / "pdaf_tier_freq_verification.png"
        dst = charts_dir / "pdaf_tier_freq_verification.png"
        shutil.copy(src, dst)
        print(f"copied to {dst}")


if __name__ == "__main__":
    main()

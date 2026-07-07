#!/usr/bin/env python3
"""Quantify PDAF+Tier TTFT and energy model accuracy (Phase 1: idle overhead).

Runs cross-node 8-card PDAF+Tier (node1 prefill, node2 decode) with
AFD_DVFS_DECISION_LOG, replays macro workloads, and compares predicted
compute+bubble+idle energy vs NVML.

Usage (on node1 host, orchestrates node2):
    python3 quantify_pdaf_tier_ttft.py --qps 1,8 --dataset conv --n-requests 40
"""
from __future__ import annotations

import argparse
import asyncio
import glob
import json
import logging
import os
import statistics
import sys
import time
from collections import Counter
from pathlib import Path

import aiohttp
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import run_macro_benchmark as MN  # noqa: E402

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s",
                    datefmt="%H:%M:%S")
log = logging.getLogger("quantify_pdaf")

WORKLOAD_DIR = HERE / "workloads"
OUT_DIR = HERE / "results" / "quantify_pdaf_tier"
NGPU = 8
# 8-card affine layout: each node uses GPU [4,5,6,7]
NODE1_GPUS = MN.card_gpus(NGPU)
NODE2_GPUS = MN.card_gpus(NGPU)
CONTAINER_DVFS_LOG = (
    "/workspace/sglang/benchmark/AFlex_bench/multi_node/"
    "node_scalibility_macro/results/quantify_pdaf_tier"
)

_orig_afd_env = MN._afd_env


def _afd_env_with_dvfs_log(role, gpus, tp, attn_gpus, ffn_gpus, log_dir: Path):
    base = _orig_afd_env(role, gpus, tp, attn_gpus, ffn_gpus)
    # Path inside container (shared mount on both nodes).
    cpath = f"{CONTAINER_DVFS_LOG}/{log_dir.name}/dvfs_decisions_{{persp}}_{{disagg}}_gpu{{gpu}}.jsonl"
    return base.replace(";", f" AFD_DVFS_DECISION_LOG='{cpath}';", 1)


def start_pdaf_tier(log_dir: Path) -> str | None:
    """Launch 8-card cross-node PDAF+Tier with DVFS decision logging."""
    log_dir.mkdir(parents=True, exist_ok=True)
    MN._afd_env = lambda *a, **kw: _afd_env_with_dvfs_log(*a, **kw, log_dir=log_dir)
    try:
        url = MN.start_pdaf(NGPU, tier=True)
    finally:
        MN._afd_env = _orig_afd_env
    return url


def cleanup_both():
    MN.cleanup_all()


def load_workload(dataset: str, qps: int, n_requests: int | None) -> list[dict]:
    path = WORKLOAD_DIR / f"macro_{dataset}_qps{qps}.jsonl"
    if not path.exists():
        raise FileNotFoundError(path)
    rows = [json.loads(l) for l in path.read_text().splitlines() if l.strip()]
    if n_requests and n_requests < len(rows):
        rows = rows[:n_requests]
    return rows


async def run_workload(reqs, url, max_run_s=600):
    e1s = MN.get_energy_local(NODE1_GPUS)
    e2s = MN.get_energy_remote(NODE2_GPUS)
    results = []
    timeout = aiohttp.ClientTimeout(total=max_run_s + 60)
    async with aiohttp.ClientSession(timeout=timeout) as session:
        base_time = time.monotonic()
        tasks = [
            asyncio.create_task(MN.send_one(session, url + "/generate", r, base_time, results))
            for r in reqs
        ]
        if tasks:
            try:
                await asyncio.wait_for(
                    asyncio.gather(*tasks, return_exceptions=True),
                    timeout=max_run_s,
                )
            except asyncio.TimeoutError:
                log.warning("Timed out after %ds", max_run_s)
    duration_s = time.monotonic() - base_time
    e1e = MN.get_energy_local(NODE1_GPUS)
    e2e = MN.get_energy_remote(NODE2_GPUS)
    prefill_j = sum((e1e.get(i, 0) - e1s.get(i, 0)) / 1000.0 for i in NODE1_GPUS)
    decode_j = sum((e2e.get(i, 0) - e2s.get(i, 0)) / 1000.0 for i in NODE2_GPUS)
    total_j = prefill_j + decode_j

    ok = [r for r in results if r.get("success")]
    ttfts = [r["ttft_proc_ms"] for r in ok if r.get("ttft_proc_ms", 0) > 0]
    total_tokens = sum(r.get("completion_tokens", 0) for r in ok)
    return {
        "total": len(reqs),
        "ok": len(ok),
        "duration_s": round(duration_s, 1),
        "ttft_proc_avg": round(float(np.mean(ttfts)), 1) if ttfts else 0,
        "ttft_proc_p50": round(float(np.percentile(ttfts, 50)), 1) if ttfts else 0,
        "ttft_proc_p99": round(float(np.percentile(ttfts, 99)), 1) if ttfts else 0,
        "energy_total_j": round(total_j, 1),
        "energy_prefill_j": round(prefill_j, 1),
        "energy_decode_j": round(decode_j, 1),
        "energy_per_token_mj": round(total_j * 1000 / total_tokens, 1)
        if total_tokens > 0 else 0,
    }


def parse_dvfs_logs(log_dir: Path, t_start: float | None = None,
                    t_end: float | None = None) -> dict:
    prefill_rows = []
    decode_rows = []
    for f in sorted(log_dir.glob("dvfs_decisions_*.jsonl")):
        for line in f.read_text().splitlines():
            if not line.strip():
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            ts = rec.get("t")
            if t_start is not None and ts is not None and ts < t_start:
                continue
            if t_end is not None and ts is not None and ts > t_end:
                continue
            if rec.get("phase") == "prefill":
                prefill_rows.append(rec)
            elif rec.get("phase") == "decode":
                decode_rows.append(rec)

    def _sum_key(rows, key):
        return sum(r.get(key, 0) for r in rows if r.get(key))

    def _energy_summary(rows, pred_key, obs_j=None):
        pred_mj = [r.get(pred_key, 0) for r in rows if r.get(pred_key)]
        obs_lat = [r.get("obs_lat_us", 0) for r in rows if r.get("obs_lat_us")]
        compute_mj = _sum_key(rows, "pred_energy_compute_mj")
        bubble_mj = _sum_key(rows, "pred_energy_bubble_mj")
        idle_mj = _sum_key(rows, "pred_energy_idle_mj")
        out = {
            "pred_energy_total_j": round(sum(pred_mj) / 1000.0, 2) if pred_mj else 0,
            "pred_energy_compute_j": round(compute_mj / 1000.0, 2) if compute_mj else 0,
            "pred_energy_bubble_j": round(bubble_mj / 1000.0, 2) if bubble_mj else 0,
            "pred_energy_idle_j": round(idle_mj / 1000.0, 2) if idle_mj else 0,
            "pred_energy_avg_mj": round(statistics.mean(pred_mj), 1) if pred_mj else 0,
            "n_energy_samples": len(pred_mj),
        }
        if obs_j is not None and obs_j > 0 and pred_mj:
            out["obs_energy_total_j"] = round(obs_j, 2)
            out["energy_pred_err_pct"] = round(
                (out["pred_energy_total_j"] - obs_j) / obs_j * 100, 1)
            out["energy_pred_over_obs_ratio"] = round(
                out["pred_energy_total_j"] / obs_j, 2)
        if obs_j is not None and obs_lat and sum(obs_lat) > 0 and pred_mj:
            obs_mj_list = [obs_j * 1000.0 * lat / sum(obs_lat) for lat in obs_lat]
            batch_err = [(p - o) / o * 100 for p, o in zip(pred_mj, obs_mj_list) if o > 0]
            out["energy_batch_err_avg_pct"] = round(statistics.mean(batch_err), 1) if batch_err else None
        return out

    def summarize_prefill(rows, obs_j=None):
        if not rows:
            return {}
        bs = [r["bs"] for r in rows if "bs" in r]
        il = [r["il"] for r in rows if "il" in r]
        fa = [r.get("f_a", 0) for r in rows]
        ff = [r.get("f_f", 0) for r in rows]
        slack = [r.get("slack_us", 0) for r in rows if r.get("slack_us")]
        pred = [r.get("pred_lat_us", 0) for r in rows if r.get("pred_lat_us")]
        obs = [r.get("obs_lat_us", 0) for r in rows if r.get("obs_lat_us")]
        err = [r.get("pred_err_pct", 0) for r in rows if r.get("pred_err_pct") is not None]
        inter = [r.get("inter_arrival_us", 0) for r in rows if r.get("inter_arrival_us")]
        freq_pairs = Counter((r.get("f_a"), r.get("f_f")) for r in rows)
        summary = {
            "n_batches": len(rows),
            "bs_avg": round(statistics.mean(bs), 1) if bs else 0,
            "bs_max": max(bs) if bs else 0,
            "il_max": max(il) if il else 0,
            "f_a_mode": Counter(fa).most_common(1)[0] if fa else (0, 0),
            "f_f_mode": Counter(ff).most_common(1)[0] if ff else (0, 0),
            "freq_pair_top3": freq_pairs.most_common(3),
            "inter_arrival_avg_ms": round(statistics.mean(inter) / 1000, 1) if inter else 0,
            "pred_lat_avg_ms": round(statistics.mean(pred) / 1000, 1) if pred else 0,
            "obs_lat_avg_ms": round(statistics.mean(obs) / 1000, 1) if obs else 0,
            "pred_err_avg_pct": round(statistics.mean(err), 1) if err else None,
            "obs_over_pred_ratio": round(statistics.mean(obs) / statistics.mean(pred), 2)
            if pred and obs and statistics.mean(pred) > 0 else None,
        }
        summary.update(_energy_summary(rows, "pred_energy_mj", obs_j))
        return summary

    def summarize_decode(rows, obs_j=None):
        if not rows:
            return {}
        pred_e = [r.get("pred_sel_energy_mj", 0) for r in rows if r.get("pred_sel_energy_mj")]
        summary = {
            "n_windows": len(rows),
            "pred_energy_total_j": round(sum(pred_e) / 1000.0, 2) if pred_e else 0,
        }
        if obs_j is not None and obs_j > 0 and pred_e:
            summary["obs_energy_total_j"] = round(obs_j, 2)
            summary["energy_pred_err_pct"] = round(
                (summary["pred_energy_total_j"] - obs_j) / obs_j * 100, 1)
        return summary

    return {
        "prefill": summarize_prefill(prefill_rows),
        "decode": summarize_decode(decode_rows),
        "_prefill_rows": prefill_rows,
        "_decode_rows": decode_rows,
    }


def finalize_energy_comparison(dvfs: dict, metrics: dict) -> dict:
    rows = dvfs.pop("_prefill_rows", [])
    dec_rows = dvfs.pop("_decode_rows", [])
    prefill_j = metrics.get("energy_prefill_j", 0)
    decode_j = metrics.get("energy_decode_j", 0)

    def _energy_summary(rows, pred_key, obs_j):
        pred_mj = [r.get(pred_key, 0) for r in rows if r.get(pred_key)]
        compute_mj = sum(r.get("pred_energy_compute_mj", 0) for r in rows)
        bubble_mj = sum(r.get("pred_energy_bubble_mj", 0) for r in rows)
        idle_mj = sum(r.get("pred_energy_idle_mj", 0) for r in rows)
        obs_lat = [r.get("obs_lat_us", 0) for r in rows if r.get("obs_lat_us")]
        out = {
            "pred_energy_total_j": round(sum(pred_mj) / 1000.0, 2) if pred_mj else 0,
            "pred_energy_compute_j": round(compute_mj / 1000.0, 2),
            "pred_energy_bubble_j": round(bubble_mj / 1000.0, 2),
            "pred_energy_idle_j": round(idle_mj / 1000.0, 2),
            "pred_energy_avg_mj": round(statistics.mean(pred_mj), 1) if pred_mj else 0,
            "n_energy_samples": len(pred_mj),
        }
        if obs_j > 0 and pred_mj:
            out["obs_energy_total_j"] = round(obs_j, 2)
            out["energy_pred_err_pct"] = round(
                (out["pred_energy_total_j"] - obs_j) / obs_j * 100, 1)
            out["energy_pred_over_obs_ratio"] = round(
                out["pred_energy_total_j"] / obs_j, 2)
        if obs_j > 0 and obs_lat and sum(obs_lat) > 0 and pred_mj:
            obs_mj_list = [obs_j * 1000.0 * lat / sum(obs_lat) for lat in obs_lat]
            batch_err = [(p - o) / o * 100 for p, o in zip(pred_mj, obs_mj_list) if o > 0]
            out["energy_batch_err_avg_pct"] = round(statistics.mean(batch_err), 1) if batch_err else None
        return out

    if rows:
        dvfs["prefill"].update(_energy_summary(rows, "pred_energy_mj", prefill_j))
    if dec_rows:
        dvfs["decode"].update(_energy_summary(dec_rows, "pred_sel_energy_mj", decode_j))
    return dvfs


def run_one_qps(qps: int, dataset: str, n_requests: int, max_run_s: int,
                keep_server: bool, server_url: str | None,
                session_log_dir: Path) -> tuple[dict, str | None]:
    result_dir = OUT_DIR / f"qps{qps}_{dataset}"
    result_dir.mkdir(parents=True, exist_ok=True)
    session_log_dir.mkdir(parents=True, exist_ok=True)

    if server_url is None:
        cleanup_both()
        server_url = start_pdaf_tier(session_log_dir)
        if server_url is None:
            raise RuntimeError("PDAF+Tier cross-node deployment failed")

    reqs = load_workload(dataset, qps, n_requests)
    last_arrival = max((r["arrival_time_s"] for r in reqs), default=0)
    run_s = int(min(max(max_run_s, last_arrival + 90), 900))
    log.info("QPS=%d: %d reqs, window=%ds, node1=%s node2=%s",
             qps, len(reqs), run_s, MN.NODE1_IP, MN.NODE2_IP)

    t_start = time.time()
    metrics = asyncio.run(run_workload(reqs, server_url, run_s))
    t_end = time.time()

    dvfs = parse_dvfs_logs(session_log_dir, t_start=t_start, t_end=t_end)
    dvfs = finalize_energy_comparison(dvfs, metrics)

    out = {
        "qps": qps,
        "dataset": dataset,
        "deploy": "pdaf_8card_xnode_tier",
        "node1_ip": MN.NODE1_IP,
        "node2_ip": MN.NODE2_IP,
        "n_requests": len(reqs),
        "wall_time": {"start": t_start, "end": t_end},
        "metrics": metrics,
        "dvfs": dvfs,
        "log_dir": str(session_log_dir),
    }
    (result_dir / "summary.json").write_text(json.dumps(out, indent=2))

    if not keep_server:
        cleanup_both()
        server_url = None
    return out, server_url


def print_report(results: list[dict]):
    print("\n" + "=" * 100)
    print("  PDAF+Tier Phase-1 Validation (8-card cross-node, idle energy in model)")
    print(f"  node1={MN.NODE1_IP} (prefill)  node2={MN.NODE2_IP} (decode)")
    print("=" * 100)

    print("\n[Prefill DVFS — latency]")
    hdr = (f"{'QPS':>4} {'TTFTavg':>8} {'bs_avg':>6} {'bs_max':>6} "
           f"{'f_a':>5} {'f_f':>5} {'pred_ms':>8} {'obs_ms':>8} {'obs/pred':>8}")
    print(hdr)
    print("-" * 80)
    for r in sorted(results, key=lambda x: x["qps"]):
        m, p = r["metrics"], r["dvfs"].get("prefill", {})
        ratio = p.get("obs_over_pred_ratio")
        print(f"{r['qps']:>4} {m['ttft_proc_avg']:>8.1f} {p.get('bs_avg', 0):>6.1f} "
              f"{p.get('bs_max', 0):>6} {p.get('f_a_mode', (0, 0))[0]:>5} "
              f"{p.get('f_f_mode', (0, 0))[0]:>5} "
              f"{p.get('pred_lat_avg_ms', 0):>8.1f} {p.get('obs_lat_avg_ms', 0):>8.1f} "
              f"{ratio if ratio else 'n/a':>8}")

    print("\n[Prefill ENERGY — compute + bubble + idle vs NVML]")
    hdr_e = (f"{'QPS':>4} {'NVML(J)':>9} {'Pred(J)':>9} {'Cmp(J)':>8} "
             f"{'Bub(J)':>8} {'Idle(J)':>8} {'Ratio':>7} {'Err%':>8}")
    print(hdr_e)
    print("-" * 80)
    for r in sorted(results, key=lambda x: x["qps"]):
        m, p = r["metrics"], r["dvfs"].get("prefill", {})
        err = p.get("energy_pred_err_pct")
        print(f"{r['qps']:>4} {m.get('energy_prefill_j', 0):>9.1f} "
              f"{p.get('pred_energy_total_j', 0):>9.2f} "
              f"{p.get('pred_energy_compute_j', 0):>8.2f} "
              f"{p.get('pred_energy_bubble_j', 0):>8.2f} "
              f"{p.get('pred_energy_idle_j', 0):>8.2f} "
              f"{p.get('energy_pred_over_obs_ratio', 0):>7.2f} "
              f"{err if err is not None else 'n/a':>8}")
    print("=" * 100)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--qps", default="1,8")
    parser.add_argument("--dataset", default="conv", choices=["conv", "code"])
    parser.add_argument("--n-requests", type=int, default=40)
    parser.add_argument("--max-run-s", type=int, default=300)
    args = parser.parse_args()

    qps_list = [int(q) for q in args.qps.split(",")]
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    session_log_dir = OUT_DIR / f"session_{args.dataset}_xnode"

    results = []
    server_url = None
    try:
        for i, qps in enumerate(qps_list):
            keep = i < len(qps_list) - 1
            out, server_url = run_one_qps(
                qps, args.dataset, args.n_requests, args.max_run_s,
                keep_server=keep, server_url=server_url,
                session_log_dir=session_log_dir,
            )
            results.append(out)
            p = out["dvfs"].get("prefill", {})
            log.info(
                "QPS=%d: TTFT=%.1fms | NVML=%.1fJ pred=%.2fJ "
                "(cmp=%.2f bub=%.2f idle=%.2f) err=%s%%",
                qps, out["metrics"]["ttft_proc_avg"],
                out["metrics"].get("energy_prefill_j", 0),
                p.get("pred_energy_total_j", 0),
                p.get("pred_energy_compute_j", 0),
                p.get("pred_energy_bubble_j", 0),
                p.get("pred_energy_idle_j", 0),
                p.get("energy_pred_err_pct", "n/a"),
            )
    finally:
        if server_url:
            cleanup_both()

    report_path = OUT_DIR / f"quantify_{args.dataset}_xnode_{'-_'.join(str(q) for q in qps_list)}.json"
    report_path.write_text(json.dumps({"live": results}, indent=2))
    print_report(results)
    print(f"\nFull report: {report_path}")


if __name__ == "__main__":
    main()

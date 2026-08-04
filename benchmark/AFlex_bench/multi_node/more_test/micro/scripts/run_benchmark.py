#!/usr/bin/env python3
"""Micro 4-dataset benchmark: 4 baselines + migrated AFlex.

Usage:
  python3 migrate_aflex.py
  python3 run_benchmark.py run --datasets qa_lpld,chatbot_lphd
  python3 run_benchmark.py parallel
  python3 run_benchmark.py plot
"""
from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import subprocess
import sys
import time
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("micro_run")

MICRO_ROOT = Path(__file__).resolve().parent
MICRO_DIR = MICRO_ROOT.parent
sys.path.insert(0, str(MICRO_ROOT))
sys.path.insert(0, str(MICRO_DIR))
sys.path.insert(0, str(MICRO_DIR.parent / "macro" / "scripts"))

import bench_common as BC
import deploy_schemes as DS
import run_macro_benchmark as RMB

os.environ.setdefault("MN_NODE1_IP", "10.252.129.36")
os.environ.setdefault("MN_NODE2_IP", "10.252.129.35")

BASELINE_SCHEMES = [
    "native_tp1_baseline",
    "native_tp1_tier",
    "pd_hetero_baseline",
    "pd_hetero_tier_biscale",
]
GPUS = list(range(8))


def _is_pass(entry: dict | None) -> bool:
    return isinstance(entry, dict) and entry.get("status") in ("PASS", "PARTIAL_TIMEOUT")


def _run_point(url: str, dataset: str, qps: int) -> dict:
    reqs = BC.load_workload(dataset, qps)
    run_s = BC.run_window_s(reqs)
    log.info("  %s (%d reqs, run_window=%ds)", BC.wl_key(dataset, qps), len(reqs), run_s)
    summary = asyncio.run(
        BC.run_workload_with_requests(
            reqs, url + "/generate",
            RMB.get_energy_local, RMB.get_energy_remote,
            GPUS, GPUS, run_s,
        )
    )
    if summary.get("status") in ("PASS", "PARTIAL_TIMEOUT"):
        log.info(
            "  %s: thpt=%.1f TTFT p50/p90=%.1f/%.1f TPOT p50/p90=%.1f/%.1f E/tok=%.1fmJ",
            summary["status"],
            summary["throughput_tok_s"],
            summary.get("ttft_proc_p50_ms", 0), summary.get("ttft_proc_p90_ms", 0),
            summary.get("tpot_p50_ms", 0), summary.get("tpot_p90_ms", 0),
            summary.get("energy_per_token_mj", 0),
        )
    else:
        log.error("  FAIL: %s", summary.get("status"))
    return summary


def run_benchmark(args: argparse.Namespace) -> None:
    DS._sync_nodes()
    schemes = BASELINE_SCHEMES if args.schemes == "all" else [s.strip() for s in args.schemes.split(",")]
    datasets = [d.strip() for d in args.datasets.split(",") if d.strip()]
    qps_list = [int(x) for x in args.qps.split(",") if x.strip()]

    meta = {
        "node1": RMB.NODE1_IP,
        "node2": RMB.NODE2_IP,
        "group": args.group or None,
        "datasets": datasets,
        "qps": qps_list,
        "schemes": schemes,
    }

    if args.resume:
        resume_data = json.loads(Path(args.resume).read_text())
        all_results = BC.flatten_results(resume_data) or resume_data.get("results", {})
    else:
        all_results = BC.load_resume()

    for scheme in schemes:
        scheme_bucket = all_results.setdefault(scheme, {})
        for dataset in datasets:
            for qps in qps_list:
                key = BC.wl_key(dataset, qps)
                if BC.workload_file(dataset, qps) is None:
                    continue
                if not args.force and _is_pass(scheme_bucket.get(key)):
                    log.info("Skip PASS %s %s", scheme, key)
                    continue

                label = BC.SCHEME_LABELS.get(scheme, scheme)
                log.info("\n" + "#" * 72)
                log.info("RUN %s | %s | QPS=%d", label, dataset, qps)
                log.info("#" * 72)

                url, _ = DS.deploy_scheme(scheme)
                if url is None:
                    scheme_bucket[key] = {"status": "DEPLOY_FAILED"}
                    BC.save_partial(all_results, meta, datasets=datasets)
                    continue

                DS.post_deploy_lock(scheme)
                if not DS.warmup_url(url, scheme):
                    scheme_bucket[key] = {"status": "WARMUP_FAILED"}
                    DS.unlock_after_run(scheme)
                    DS.cleanup()
                    BC.save_partial(all_results, meta, datasets=datasets)
                    continue

                time.sleep(3)
                scheme_bucket[key] = _run_point(url, dataset, qps)
                DS.unlock_after_run(scheme)
                DS.cleanup()
                BC.save_partial(all_results, meta, datasets=datasets)
                time.sleep(5)

    out = BC.save_all(all_results, meta, datasets=datasets)
    log.info("Benchmark complete: %s", ", ".join(str(p.name) for p in out))


def merge_partials(prefixes: list[str]) -> dict:
    merged: dict = {}
    for prefix in prefixes:
        files = sorted(BC.DATA_DIR.glob(f"{prefix}_partial_*.json"))
        path = files[-1] if files else None
        if path is None:
            log.warning("No partial for prefix %s", prefix)
            continue
        data = json.loads(path.read_text())
        for scheme, bucket in data.get("results", {}).items():
            merged.setdefault(scheme, {}).update(bucket)
    return merged


def run_parallel(args: argparse.Namespace) -> None:
    node1_a = os.environ.get("MN_NODE1_IP_A", "10.252.129.36")
    node2_a = os.environ.get("MN_NODE2_IP_A", "10.252.129.35")
    node1_b = os.environ.get("MN_NODE1_IP_B", "10.252.129.34")
    node2_b = os.environ.get("MN_NODE2_IP_B", "10.252.129.33")
    script = str(MICRO_ROOT / "run_benchmark.py")  # self-reference
    schemes = ",".join(BASELINE_SCHEMES)

    env_a = {**os.environ, "MN_NODE1_IP": node1_a, "MN_NODE2_IP": node2_a}
    env_b = {**os.environ, "MN_NODE1_IP": node1_b, "MN_NODE2_IP": node2_b}

    cmd_a = [
        sys.executable, script, "run",
        "--datasets", "qa_lpld,chatbot_lphd",
        "--schemes", schemes,
        "--group", "micro_gA",
        "--prefix", "micro_e2e_gA",
    ]
    cmd_b = [
        sys.executable, script, "run",
        "--datasets", "rag_hpld,summary_hphd",
        "--schemes", schemes,
        "--group", "micro_gB",
        "--prefix", "micro_e2e_gB",
    ]
    if args.force:
        cmd_a.append("--force")
        cmd_b.append("--force")

    log.info("Group A (%s+%s): qa_lpld, chatbot_lphd", node1_a, node2_a)
    pa = subprocess.Popen(cmd_a, env=env_a, cwd=str(MICRO_ROOT))
    log.info("Group B (%s+%s): rag_hpld, summary_hphd", node1_b, node2_b)
    pb = subprocess.Popen(cmd_b, env=env_b, cwd=str(MICRO_ROOT))

    ra, rb = pa.wait(), pb.wait()
    log.info("Group A exit=%d, Group B exit=%d", ra, rb)

    merged_baselines = merge_partials(["micro_e2e_gA", "micro_e2e_gB"])
    existing = BC.load_resume()
    for scheme, bucket in merged_baselines.items():
        existing.setdefault(scheme, {}).update(bucket)

    meta = {
        "nodes": {"group_a": [node1_a, node2_a], "group_b": [node1_b, node2_b]},
        "merged_from": ["micro_e2e_gA", "micro_e2e_gB"],
        "merged_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    out = BC.save_all(existing, meta)
    log.info("Merged -> %s", ", ".join(str(p.name) for p in out))
    subprocess.run(
        [sys.executable, str(MICRO_DIR / "plot_micro_dashboard.py")],
        check=False,
    )


def plot_charts(args: argparse.Namespace) -> None:
    cmd = [sys.executable, str(MICRO_DIR / "plot_micro_dashboard.py")]
    if args.all_percentiles:
        cmd.append("--all-percentiles")
    if args.input:
        cmd.extend(["--input", str(args.input)])
    subprocess.run(cmd, check=False)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    p_run = sub.add_parser("run")
    p_run.add_argument("--schemes", default="all")
    p_run.add_argument("--datasets", default="qa_lpld,chatbot_lphd,rag_hpld,summary_hphd")
    p_run.add_argument("--qps", default="2,4,8,16")
    p_run.add_argument("--resume", default=None)
    p_run.add_argument("--force", action="store_true")
    p_run.add_argument("--group", default="")
    p_run.add_argument("--prefix", default="micro_e2e")
    p_run.set_defaults(func=run_benchmark)

    p_par = sub.add_parser("parallel")
    p_par.add_argument("--force", action="store_true")
    p_par.set_defaults(func=run_parallel)

    p_plot = sub.add_parser("plot", help="Generate charts from micro_e2e_*.json")
    p_plot.add_argument("--input", type=Path, default=None)
    p_plot.add_argument("--all-percentiles", action="store_true")
    p_plot.set_defaults(func=plot_charts)

    args = parser.parse_args()
    if args.command == "run" and hasattr(args, "prefix"):
        BC.RESULT_PREFIX = args.prefix
    args.func(args)


if __name__ == "__main__":
    main()

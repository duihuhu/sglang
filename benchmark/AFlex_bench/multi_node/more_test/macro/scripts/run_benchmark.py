#!/usr/bin/env python3
"""Macro end-to-end benchmark: 5 schemes x code/conv x QPS 2/4/8/16.

Usage:
  # Run all schemes on current node pair:
  python3 run_benchmark.py run

  # Four-node parallel (Group A: SGLang/Dynamo/DistServe, Group B: BiScale/AFlex):
  python3 run_benchmark.py parallel

  # Regenerate charts from data/macro_e2e_all.json:
  python3 run_benchmark.py plot

  # Repack partial results into macro_e2e_all.json:
  python3 run_benchmark.py pack --input data/macro_e2e_partial_*.json
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
log = logging.getLogger("run_benchmark")

MACRO_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(MACRO_ROOT))
sys.path.insert(0, str(MACRO_ROOT.parent))

import bench_common as BC
import deploy_schemes as DS
import run_macro_benchmark as RMB

os.environ.setdefault("MN_NODE1_IP", "10.252.129.36")
os.environ.setdefault("MN_NODE2_IP", "10.252.129.35")

ALL_SCHEMES = list(BC.SCHEME_DEPLOY.keys())
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
            summary["ttft_proc_p50_ms"], summary["ttft_proc_p90_ms"],
            summary["tpot_p50_ms"], summary["tpot_p90_ms"],
            summary.get("energy_per_token_mj", 0),
        )
    else:
        log.error("  FAIL: %s", summary.get("status"))
    return summary


def _run_aflex(scheme_bucket, datasets, qps_list, all_results, meta, *, force=False):
    for dataset in datasets:
        for qps in qps_list:
            key = BC.wl_key(dataset, qps)
            if BC.workload_file(dataset, qps) is None:
                continue
            if not force and _is_pass(scheme_bucket.get(key)):
                log.info("Skip AFlex PASS %s", key)
                continue

            log.info("\n" + "#" * 72)
            log.info("RUN AFlex | %s | QPS=%d", dataset, qps)
            log.info("#" * 72)

            urls, deploy_extra = DS.deploy_aflex_e2e_point(dataset, qps)
            if urls is None:
                scheme_bucket[key] = {"status": "DEPLOY_FAILED"}
                BC.save_partial(all_results, meta)
                continue

            summary = dict(DS.run_aflex_e2e_benchmark(urls, dataset, qps))
            summary.update(DS.aflex_result_extra(dataset, qps))
            if deploy_extra:
                summary["config"] = deploy_extra.get("config")
            scheme_bucket[key] = summary
            DS.teardown_aflex_e2e_point()
            BC.save_partial(all_results, meta)
            time.sleep(5)


def run_benchmark(args: argparse.Namespace) -> None:
    DS._sync_sweep_nodes()
    schemes = ALL_SCHEMES if args.schemes == "all" else [s.strip() for s in args.schemes.split(",")]
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
    all_results = BC.load_resume() if not args.resume else json.loads(
        Path(args.resume).read_text()
    ).get("results", BC.flatten_results(json.loads(Path(args.resume).read_text())))

    for scheme in schemes:
        scheme_bucket = all_results.setdefault(scheme, {})
        if scheme == "aflex_tier1":
            _run_aflex(scheme_bucket, datasets, qps_list, all_results, meta, force=args.force)
            continue

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
                    BC.save_partial(all_results, meta)
                    continue

                DS.post_deploy_lock(scheme)
                if not DS.warmup_url(url, scheme):
                    scheme_bucket[key] = {"status": "WARMUP_FAILED"}
                    DS.unlock_after_run(scheme)
                    DS.cleanup()
                    BC.save_partial(all_results, meta)
                    continue

                time.sleep(3)
                scheme_bucket[key] = _run_point(url, dataset, qps)
                DS.unlock_after_run(scheme)
                DS.cleanup()
                BC.save_partial(all_results, meta)
                time.sleep(5)

    out = BC.save_all(all_results, meta)
    log.info("Benchmark complete: %s", out)

    if not args.skip_plot:
        subprocess.run([sys.executable, str(MACRO_ROOT / "plot_e2e_dashboard.py")], check=False)


def merge_partials(prefixes: list[str]) -> dict:
    merged: dict = {}
    for prefix in prefixes:
        files = sorted(BC.DATA_DIR.glob(f"{prefix}_partial_*.json"))
        if not files:
            final = sorted(BC.DATA_DIR.glob(f"{prefix}_final_*.json"))
            path = final[-1] if final else None
        else:
            path = files[-1]
        if path is None:
            log.warning("No results for prefix %s", prefix)
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
    script = str(MACRO_ROOT / "run_benchmark.py")

    env_a = {**os.environ, "MN_NODE1_IP": node1_a, "MN_NODE2_IP": node2_a}
    env_b = {**os.environ, "MN_NODE1_IP": node1_b, "MN_NODE2_IP": node2_b}

    cmd_a = [
        sys.executable, script, "run",
        "--schemes", "native_tp1_baseline,native_tp1_tier,pd_hetero_baseline",
        "--group", "group_a", "--prefix", "macro_e2e_gA", "--skip-plot",
    ]
    cmd_b = [
        sys.executable, script, "run",
        "--schemes", "pd_hetero_tier_biscale,aflex_tier1",
        "--group", "group_b", "--prefix", "macro_e2e_gB", "--skip-plot",
    ]
    if args.force:
        cmd_a.append("--force")
        cmd_b.append("--force")

    log.info("Launch Group A on %s+%s", node1_a, node2_a)
    pa = subprocess.Popen(cmd_a, env=env_a, cwd=str(MACRO_ROOT))
    log.info("Launch Group B on %s+%s", node1_b, node2_b)
    pb = subprocess.Popen(cmd_b, env=env_b, cwd=str(MACRO_ROOT))

    ra = pa.wait()
    rb = pb.wait()
    log.info("Group A exit=%d, Group B exit=%d", ra, rb)

    merged = merge_partials(["macro_e2e_gA", "macro_e2e_gB"])
    if not merged:
        raise SystemExit("No group results to merge")

    meta = {
        "nodes": {"group_a": [node1_a, node2_a], "group_b": [node1_b, node2_b]},
        "merged_from": ["macro_e2e_gA", "macro_e2e_gB"],
        "merged_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    BC.save_all(merged, meta)
    subprocess.run([sys.executable, str(MACRO_ROOT / "plot_e2e_dashboard.py")], check=False)


def pack_data(args: argparse.Namespace) -> None:
    path = Path(args.input) if args.input else sorted(BC.DATA_DIR.glob("macro_e2e_partial_*.json"))[-1]
    data = json.loads(path.read_text())
    flat = BC.flatten_results(data)
    BC.save_all(flat, data.get("meta", {}))
    log.info("Packed %s -> %s", path, BC.ALL_DATA_FILE)


def plot_charts(args: argparse.Namespace) -> None:
    cmd = [sys.executable, str(MACRO_ROOT / "plot_e2e_dashboard.py")]
    if args.input:
        cmd.extend(["--input", str(args.input)])
    subprocess.run(cmd, check=False)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    p_run = sub.add_parser("run", help="Run benchmark on current node pair")
    p_run.add_argument("--schemes", default="all")
    p_run.add_argument("--datasets", default="code,conv")
    p_run.add_argument("--qps", default="2,4,8,16")
    p_run.add_argument("--resume", default=None)
    p_run.add_argument("--force", action="store_true")
    p_run.add_argument("--skip-plot", action="store_true")
    p_run.add_argument("--group", default="")
    p_run.add_argument("--prefix", default="macro_e2e")
    p_run.set_defaults(func=run_benchmark)

    p_par = sub.add_parser("parallel", help="Four-node parallel benchmark")
    p_par.add_argument("--force", action="store_true")
    p_par.set_defaults(func=run_parallel)

    p_pack = sub.add_parser("pack", help="Pack flat results into macro_e2e_all.json")
    p_pack.add_argument("--input", default=None)
    p_pack.set_defaults(func=pack_data)

    p_plot = sub.add_parser("plot", help="Generate charts")
    p_plot.add_argument("--input", type=Path, default=None)
    p_plot.set_defaults(func=plot_charts)

    args = parser.parse_args()
    if args.command == "run" and hasattr(args, "prefix"):
        BC.RESULT_PREFIX = args.prefix
    args.func(args)


if __name__ == "__main__":
    main()

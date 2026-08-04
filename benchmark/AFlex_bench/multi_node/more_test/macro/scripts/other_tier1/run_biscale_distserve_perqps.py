#!/usr/bin/env python3
"""Re-run BiScale and DistServe with per-QPS restart, code dataset, node3+node4.

For each QPS, cleanup → deploy → warmup → run workload → cleanup.
This gives more accurate per-QPS measurements vs. single-deploy continuous mode.
"""
from __future__ import annotations

import asyncio, json, logging, os, sys, time
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger("rerun_biscale_distserve")

os.environ.setdefault("MN_NODE1_IP", "10.252.129.34")
os.environ.setdefault("MN_NODE2_IP", "10.252.129.33")

HERE = Path(__file__).resolve().parent
MACRO_DIR = HERE.parent.parent  # node_scalibility_macro/
sys.path.insert(0, str(MACRO_DIR))
sys.path.insert(0, str(HERE.parent))  # more_trying/ (for run_biscale_pd_hetero_code)
import run_macro_benchmark as RMB

RMB.TTFT_SLO_MS = 5000.0
RMB.TPOT_SLO_MS = 300.0

QPS_LIST = [2, 4, 6, 8, 12, 16]
DATASET = "code"
NGPU = 16
MAX_RUN_S = 400
RESULTS_DIR = HERE / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
MACRO_WL_DIR = MACRO_DIR / "workloads"


def _workload_file(qps: int) -> Path | None:
    path = MACRO_WL_DIR / f"macro_{DATASET}_qps{qps}.jsonl"
    return path if path.exists() else None


def run_one_qps(url: str, qps: int, gpus: list[int]):
    wl_file = _workload_file(qps)
    if wl_file is None:
        log.error("Workload missing: code_qps%d", qps)
        return None
    with open(wl_file) as f:
        reqs = [json.loads(line) for line in f]
    last_arrival = max((r["arrival_time_s"] for r in reqs), default=0)
    run_s = int(min(max(MAX_RUN_S, last_arrival + 150), 900))
    log.info("  Workload: %d reqs, run_window=%ds", len(reqs), run_s)
    summary = asyncio.run(
        RMB.run_workload(reqs, url + "/generate", gpus, gpus, run_s)
    )
    if summary.get("status") == "PASS":
        log.info("  PASS: Thpt=%.1f | TTFT=%.1fms | TPOT=%.1fms | E/tok=%.1fmJ | SLO=%.1f%%",
                 summary["throughput_tok_s"], summary["ttft_proc_avg_ms"],
                 summary["tpot_avg_ms"], summary["energy_per_token_mj"],
                 summary["slo_violation_rate"])
    else:
        log.error("  FAIL: %s", summary)
    return summary


# ── BiScale deploy (from run_biscale_pd_hetero_code.py) ──
def deploy_biscale():
    """P=2×TP4 + D=4×TP2 with BiScale DVFS."""
    from run_biscale_pd_hetero_code import start_pd_hetero_biscale
    return start_pd_hetero_biscale(NGPU, tier=True)


# ── DistServe deploy ──
def deploy_distserve():
    """P=4×TP2 + D=2×TP4, locked at max freq (no DVFS)."""
    url = RMB.start_pd_hetero(NGPU, tier=False)
    if url:
        gpus = RMB.card_gpus(NGPU)
        RMB.lock_freq_both(gpus, RMB.MAX_GPU_FREQ)
    return url


def run_scheme(scheme_name: str, deploy_fn, qps_list: list[int]):
    """For each QPS: cleanup → deploy → warmup → run → cleanup."""
    gpus = RMB.card_gpus(NGPU)
    results = {}

    for qps in qps_list:
        log.info("\n" + "=" * 60)
        log.info("[%s] QPS=%d — deploying...", scheme_name, qps)
        log.info("=" * 60)

        RMB.cleanup_all()
        time.sleep(8)

        url = deploy_fn()
        if url is None:
            log.error("[%s] QPS=%d DEPLOY_FAILED", scheme_name, qps)
            results[f"{DATASET}_qps{qps}"] = {"status": "DEPLOY_FAILED"}
            continue

        if not RMB.test_generate(url):
            log.error("[%s] QPS=%d WARMUP_FAILED", scheme_name, qps)
            results[f"{DATASET}_qps{qps}"] = {"status": "WARMUP_FAILED"}
            RMB.cleanup_all()
            continue

        time.sleep(3)
        summary = run_one_qps(url, qps, gpus)
        results[f"{DATASET}_qps{qps}"] = summary if summary else {"status": "ERROR"}

        RMB.unlock_freq_both(gpus)
        RMB.cleanup_all()
        time.sleep(5)

    return results


def _save(scheme: str, results: dict) -> Path:
    ts = time.strftime("%Y%m%d_%H%M%S")
    out = RESULTS_DIR / f"{scheme}_perqps_restart_{ts}.json"
    payload = {
        "meta": {
            "scheme": scheme,
            "dataset": DATASET,
            "qps_list": QPS_LIST,
            "mode": "per_qps_restart",
            "ttft_slo_ms": RMB.TTFT_SLO_MS,
            "tpot_slo_ms": RMB.TPOT_SLO_MS,
            "node1": RMB.NODE1_IP, "node2": RMB.NODE2_IP,
        },
        "results": results,
    }
    out.write_text(json.dumps(payload, indent=2))
    log.info("Saved %s", out)
    return out


def main():
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--schemes", default="distserve,biscale",
                        help="Comma-separated: distserve,biscale")
    parser.add_argument("--qps", default=None, help="Override QPS list (e.g. 2,4,8)")
    args = parser.parse_args()

    schemes = [s.strip() for s in args.schemes.split(",")]
    qps_list = [int(x) for x in args.qps.split(",")] if args.qps else QPS_LIST

    all_results = {}
    for scheme in schemes:
        log.info("\n" + "#" * 72)
        log.info("SCHEME: %s (per-QPS restart)", scheme)
        log.info("#" * 72)

        if scheme == "biscale":
            results = run_scheme("BiScale", deploy_biscale, qps_list)
        elif scheme == "distserve":
            results = run_scheme("DistServe", deploy_distserve, qps_list)
        else:
            log.error("Unknown scheme: %s", scheme)
            continue

        all_results[scheme] = results
        _save(scheme, results)

    # Print summary
    print(f"\n{'='*72}")
    print("PER-QPS RESTART RESULTS (code dataset)")
    print(f"{'='*72}")
    for scheme, results in all_results.items():
        print(f"\n  [{scheme}]")
        for key, r in sorted(results.items()):
            if isinstance(r, dict) and r.get("status") == "PASS":
                print(f"    {key}: thpt={r['throughput_tok_s']:.1f} TTFT={r['ttft_proc_avg_ms']:.1f}ms "
                      f"TPOT={r['tpot_avg_ms']:.1f}ms E/tok={r['energy_per_token_mj']:.1f}mJ "
                      f"SLO={r['slo_violation_rate']:.1f}%")
            else:
                status = r.get("status", "UNKNOWN") if isinstance(r, dict) else "ERROR"
                print(f"    {key}: {status}")


if __name__ == "__main__":
    main()

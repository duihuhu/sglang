#!/usr/bin/env python3
"""Test Tier1-optimal configs using proven deploy_tier1_layout (k_P=1 only).

对比两个方案在 QPS=4 code 数据集上的实测能耗:
  A (10 GPU, 低能耗): tp=(4,4)@930MHz, k_D=1
  B (14 GPU, 对照):   tp=(4,4)@1410MHz, k_D=3  (原 fallback)
"""
from __future__ import annotations

import asyncio, json, logging, os, sys, time
from dataclasses import dataclass
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger("bench_tier1_kp1")

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent.parent))  # node_scalibility_macro/
sys.path.insert(0, str(HERE.parent))          # more_trying/

import deploy_tier1_layout as DTL
import run_macro_benchmark as RMB
from freq_timeline_utils import FreqTimelineSession, ensure_container_log_dir
from run_fixed_6scheme_7dataset import _wl_key, _workload_file, MAX_RUN_S

# ── Node3 + Node4 ──
RMB.NODE1_IP = os.environ.get("MN_NODE3_IP", "10.252.129.34")
RMB.NODE2_IP = os.environ.get("MN_NODE4_IP", "10.252.129.33")
RMB.TTFT_SLO_MS = 2000.0
RMB.TPOT_SLO_MS = 100.0

# Monkey-patch: DTL internally uses lock_freq_map_on_host which may not be
# available.  Replace _lock_node_gpus with a no-op; we handle freq ourselves.
DTL._lock_node_gpus = lambda host, gpu_freq: log.info(
    "  [monkey-patch] freq lock skipped for %s (handled externally)", host.split(".")[-1]
)

RESULTS_DIR = HERE / "results"
CHARTS_DIR = HERE / "charts"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
CHARTS_DIR.mkdir(parents=True, exist_ok=True)

QPS = 4
DATASET = "code"

# ── Test configs ──
CONFIGS = [
    {
        "name": "tier1_10g_930",
        "k_d": 1, "tp_pa": 4, "tp_pf": 4, "tp_da": 1, "tp_df": 1,
        "f_pa": 930, "f_pf": 930, "f_da": 930, "f_df": 930,
        "tier": True,  # AFlex T2 DVFS
    },
    {
        "name": "tier1_14g_1410",
        "k_d": 3, "tp_pa": 4, "tp_pf": 4, "tp_da": 1, "tp_df": 1,
        "f_pa": 1410, "f_pf": 1410, "f_da": 1410, "f_df": 1410,
        "tier": False,  # baseline: lock at max freq
    },
]


def run_test(cfg_dict):
    cfg = DTL.Tier1Layout(
        name=cfg_dict["name"],
        k_d=cfg_dict["k_d"],
        tp_pa=cfg_dict["tp_pa"], tp_pf=cfg_dict["tp_pf"],
        tp_da=cfg_dict["tp_da"], tp_df=cfg_dict["tp_df"],
        f_pa=cfg_dict["f_pa"], f_pf=cfg_dict["f_pf"],
        f_da=cfg_dict["f_da"], f_df=cfg_dict["f_df"],
        tier=cfg_dict["tier"],
    )
    total_gpu = (cfg.tp_pa + cfg.tp_pf) + cfg.k_d * (cfg.tp_da + cfg.tp_df)

    log.info("#" * 60)
    log.info("TEST: %s (%d GPU, k_D=%d tier=%s)", cfg.name, total_gpu, cfg.k_d, cfg.tier)
    log.info("#" * 60)

    # Deploy
    RMB.cleanup_all()
    time.sleep(8)

    if cfg.tier:
        # Tier mode: unlock freq, let DVFS control
        for host in [RMB.NODE1_IP, RMB.NODE2_IP]:
            try:
                py = "from sglang.srt.layers.dvfs import unlock_gpus; unlock_gpus(list(range(8)))"
                cmd = f"docker exec {RMB.CONTAINER} bash -lc 'python3 -c \"{py}\"'"
                if host == RMB.NODE1_IP:
                    subprocess.run(["docker", "exec", RMB.CONTAINER, "bash", "-lc", f'python3 -c "{py}"'], check=False)
                else:
                    subprocess.run(RMB._ssh(host, cmd), check=False)
            except: pass
    else:
        # Baseline mode: no DVFS — lock happens in harness after deploy
        pass

    url = DTL.deploy_tier1_layout(cfg)
    if url is None:
        log.error("DEPLOY FAILED"); return {"status": "DEPLOY_FAILED"}

    # For baseline: lock frequencies after deploy
    if not cfg.tier:
        gpus = list(range(8))
        RMB.lock_freq_both(gpus, RMB.MAX_GPU_FREQ)

    if not RMB.test_generate(url):
        log.error("WARMUP FAILED"); RMB.cleanup_all(); return {"status": "WARMUP_FAILED"}
    time.sleep(3)

    # Load workload
    wl_file = _workload_file(DATASET, QPS)
    if wl_file is None: return {"status": "NO_WORKLOAD"}
    with open(wl_file) as f:
        reqs = [json.loads(line) for line in f]
    last_arrival = max((r["arrival_time_s"] for r in reqs), default=0)
    run_s = int(min(max(MAX_RUN_S, last_arrival + 150), 900))
    log.info("Workload: %d reqs, run_window=%ds", len(reqs), run_s)

    # Run with freq timeline
    gpus = list(range(8))
    ensure_container_log_dir(cfg.name)
    sess = FreqTimelineSession(cfg.name, DATASET, QPS)
    sess.start()
    try:
        summary = asyncio.run(RMB.run_workload(reqs, url + "/generate", gpus, gpus, run_s))
    finally:
        freq_meta = sess.stop()

    if isinstance(summary, dict) and summary.get("status") == "PASS":
        summary = dict(summary)
        summary["freq_timeline"] = freq_meta
        summary["config"] = cfg_dict
        log.info("PASS: thpt=%.1f TTFT=%.1fms TPOT=%.1fms E/tok=%.1fmJ SLO=%.1f%%",
                 summary["throughput_tok_s"], summary["ttft_proc_avg_ms"],
                 summary["tpot_avg_ms"], summary.get("energy_per_token_mj", 0),
                 summary["slo_violation_rate"])
    else:
        log.error("FAIL: %s", summary)

    RMB.unlock_freq_both(gpus)
    RMB.cleanup_all()
    time.sleep(5)
    return summary


def main():
    import argparse, subprocess
    # ── Pre-clean: kill old detokenizer/router processes ──
    for host in [RMB.NODE1_IP, RMB.NODE2_IP]:
        inner = "pkill -9 -f 'sglang_router|sglang.launch_server|launch_router|sglang::router' 2>/dev/null; true"
        try:
            if host == RMB.NODE1_IP:
                subprocess.run(["docker", "exec", RMB.CONTAINER, "bash", "-lc", inner],
                               check=False, timeout=10)
            else:
                cmd = f"docker exec {RMB.CONTAINER} bash -lc '{inner}'"
                subprocess.run(RMB._ssh(host, cmd), check=False, timeout=15)
        except Exception:
            pass
    log.info("Pre-clean done")
    parser.add_argument("--configs", nargs="*", choices=[c["name"] for c in CONFIGS])
    args = parser.parse_args()

    targets = [c for c in CONFIGS if args.configs is None or c["name"] in args.configs]
    results = {}

    for cfg_dict in targets:
        results[cfg_dict["name"]] = run_test(cfg_dict)

    ts = time.strftime("%Y%m%d_%H%M%S")
    out = RESULTS_DIR / f"tier1_kp1_{ts}.json"
    meta = {"qps": QPS, "dataset": DATASET, "ttft_slo_ms": RMB.TTFT_SLO_MS,
            "tpot_slo_ms": RMB.TPOT_SLO_MS, "n3": RMB.NODE1_IP, "n4": RMB.NODE2_IP}
    out.write_text(json.dumps({"meta": meta, "results": results}, indent=2))
    log.info("Saved %s", out)

    print(f"\n{'='*60}")
    for name, r in results.items():
        if r.get("status") == "PASS":
            print(f"  {name}: thpt={r['throughput_tok_s']:.1f} TTFT={r['ttft_proc_avg_ms']:.1f}ms "
                  f"TPOT={r['tpot_avg_ms']:.1f}ms E/tok={r['energy_per_token_mj']:.1f}mJ SLO={r['slo_violation_rate']:.1f}%")
        else:
            print(f"  {name}: {r.get('status')}")


if __name__ == "__main__":
    import subprocess
    main()

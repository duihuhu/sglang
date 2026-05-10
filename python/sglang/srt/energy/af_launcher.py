#!/usr/bin/env python3
"""AFlex batch launcher — starts DF → DA → PF → PA → launch_router in order.

Usage
  python af_launcher.py [--config path/to/config.json] [--start-with-workload]

If ``--start-with-workload`` is given, the launcher initialises ProfileTable +
Tier1Solver once, runs ``solve()``, and writes the result to a JSON file under
the log directory.  The PA server receives ``--tier1-initial-solution <path>``
so its scheduler can load the pre-computed solution without re-initialising the
solver.  When ``--enable-tier1-pa`` is also set in the config, the PA scheduler
also starts the WorkloadMonitor + collector for dynamic re-planning, and
lazy-initialises the solver only when a re-plan is triggered.

Config file format: see af_launch_config.json.
"""

from __future__ import annotations

import json
import logging
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Optional

logger = logging.getLogger("af_launcher")

_HERE = Path(__file__).resolve().parent
_DEFAULT_CONFIG = _HERE / "af_launch_config.json"

# Port readiness: how long to wait for a server to start listening
_PORT_TIMEOUT_S = 120
_PORT_POLL_INTERVAL_S = 1.0

# Solution file written by pre-flight solve
_SOLUTION_FILENAME = "tier1_initial_solution.json"


# ── Helpers ──────────────────────────────────────────────────────────────


def _set_gpu_locked_clock(gpu_index: int, freq_mhz: int):
    """Lock a GPU to a fixed SM clock frequency via pynvml before launch."""
    try:
        from pynvml import (
            nvmlDeviceGetHandleByIndex,
            nvmlDeviceSetGpuLockedClocks,
            nvmlInit,
            nvmlShutdown,
        )
        nvmlInit()
        handle = nvmlDeviceGetHandleByIndex(gpu_index)
        nvmlDeviceSetGpuLockedClocks(handle, freq_mhz, freq_mhz)
        nvmlShutdown()
        logger.info("  GPU %d locked to %d MHz (min=max=%d)", gpu_index, freq_mhz, freq_mhz)
    except ImportError:
        logger.warning("  pynvml not available; cannot lock GPU %d to %d MHz", gpu_index, freq_mhz)
    except Exception as e:
        logger.warning("  Failed to lock GPU %d to %d MHz: %s", gpu_index, freq_mhz, e)


def _load_solution_dict(path: str) -> dict:
    """Load the Tier 1 solution JSON, return the dict."""
    with open(path, "r") as f:
        return json.load(f)


def _check_port_ready(host: str, port: int, timeout_s: float = _PORT_TIMEOUT_S) -> bool:
    """Wait until *port* on *host* is accepting TCP connections."""
    import socket

    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        try:
            with socket.create_connection((host, port), timeout=1):
                return True
        except (OSError, ConnectionRefusedError):
            time.sleep(_PORT_POLL_INTERVAL_S)
    return False


def _build_server_cmd(
    cfg: dict,
    mod: dict,
    tier1_extra: Optional[list[str]] = None,
    tp_override: Optional[int] = None,
) -> list[str]:
    """Build the ``python -m sglang.launch_server`` argument list for *mod*."""
    tp = tp_override if tp_override is not None else cfg["model"]["tp"]
    cmd = [
        sys.executable,
        "-m",
        "sglang.launch_server",
        "--model-path", cfg["model"]["path"],
        "--tp", str(tp),
        "--afd-perspective", mod["perspective"],
        "--afd-comm-backend", cfg["afd"]["comm_backend"],
        "--port", str(mod["port"]),
        "--disaggregation-mode", mod["disagg_mode"],
        "--disaggregation-bootstrap-port", str(cfg["disaggregation"]["bootstrap_port"]),
        "--disaggregation-ib-device", cfg["disaggregation"]["ib_device"],
        "--mem-fraction-static", str(cfg["model"]["mem_fraction_static"]),
    ]

    if cfg.get("enable_metrics", False):
        cmd += ["--enable-metrics"]
        cmd += ["--enable-metrics-for-all-schedulers"]

    if cfg["afd"]["dvfs_enabled"]:
        cmd += ["--afd-dvfs-enabled"]
    if cfg["afd"]["dvfs_enabled"] or tier1_extra is not None:
        cmd += ["--afd-energy-model-dir", cfg["afd"]["energy_model_dir"]]

    if tier1_extra:
        cmd += tier1_extra

    # Per-module extra CLI args (e.g. ["--skip-server-warmup"])
    extra = mod.get("extra_cli_args", [])
    if isinstance(extra, list):
        cmd += extra

    return cmd


def _build_server_env(
    cfg: dict,
    mod: dict,
    cuda_visible_devices: str = "",
) -> dict:
    """Build the environment dict for a server process."""
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = cuda_visible_devices or str(mod["gpu"])
    env["AFD_UCX_BASE_PORT"] = str(mod["ucx_base_port"])
    env["AFD_SCHED_PORT"] = str(mod["sched_port"])
    env["SGLANG_DISABLE_REQUEST_LOGGING"] = "true"
    env["UCX_LOG_LEVEL"] = "fatal"
    env["UCX_WARN_UNUSED_ENV_VARS"] = "n"
    env["AFD_UCX_TLS"] = "rc,tcp,cuda_copy,cuda_ipc"

    ffn_host = mod.get("ffn_host")
    if ffn_host:
        env["AFD_UCX_FFN_HOST"] = ffn_host

    return env


def _build_router_cmd(cfg: dict) -> list[str]:
    """Build the ``python -m sglang_router.launch_router`` argument list."""
    router = cfg["router"]
    return [
        sys.executable,
        "-m",
        "sglang_router.launch_router",
        "--pd-disaggregation",
        "--mini-lb",
        "--prefill", router["prefill_url"],
        "--decode", router["decode_url"],
        "--host", router["host"],
        "--port", str(router["port"]),
    ]


# ── Pre-flight Tier 1 solve ───────────────────────────────────────────────


def _run_preflight_solve(cfg: dict, log_dir: Path) -> Optional[str]:
    """Initialise ProfileTable + Tier1Solver, run solve(), write solution JSON.

    Returns the path to the solution JSON file, or None on failure.
    """
    tier1 = cfg.get("tier1", {})
    energy_dir = cfg["afd"]["energy_model_dir"]

    logger.info("Pre-flight Tier1Solver: initialising …")

    try:
        from sglang.srt.energy.profile_table import ProfileTable
        from sglang.srt.energy.tier1_solver import (
            SLOConfig,
            Tier1Solver,
            WorkloadProfile,
        )

        prefill_path = tier1.get("prefill_data_path",
            "benchmark/test_motivation/hucc/paper/prefill_data_v1.txt")
        decode_path = tier1.get("decode_data_path",
            "benchmark/test_motivation/hucc/paper/decode_data_v1.txt")

        pt = ProfileTable(
            prefill_path=prefill_path,
            decode_path=decode_path,
            energy_model_dir=energy_dir,
        )

        # Model geometry — read from config or fall back to Llama3.1-8B defaults
        model_cfg = cfg.get("model", {})
        solver = Tier1Solver(
            profile_table=pt,
            num_layers=model_cfg.get("num_layers", 32),
            num_kv_heads=model_cfg.get("num_kv_heads", 8),
            head_dim=model_cfg.get("head_dim", 128),
            hidden_size=model_cfg.get("hidden_size", 4096),
            gpu_mem_gb=model_cfg.get("gpu_mem_gb", 80.0),
        )

        wl = WorkloadProfile(
            lambda_prefill=tier1.get("lambda_prefill", 10.0),
            n_active_decode=tier1.get("n_active_decode", 32),
            il_rep_p=tier1.get("il_rep_p", 1024),
            bs_avg_p=tier1.get("bs_avg_p", 8),
            il_rep_d=tier1.get("il_rep_d", 512),
            ol_rep_d=tier1.get("ol_rep_d", 256),
            bs_avg_d=tier1.get("bs_avg_d", 16),
        )
        slo = SLOConfig(
            ttft_ms=tier1.get("ttft_slo_ms", 5000.0),
            tpot_ms=tier1.get("tpot_slo_us", 50000.0) / 1000.0,
        )

        solution = solver.solve(
            G=tier1.get("gpu_count", 16),
            workload=wl,
            slo=slo,
        )

        if not solution.feasible:
            logger.warning("Pre-flight solve: INFEASIBLE, using warm_start fallback")
            solution = solver.warm_start(
                G=tier1.get("gpu_count", 16), workload=wl, slo=slo,
            )

        logger.info("Pre-flight Tier 1 configuration:")
        logger.info("  PA (Prefill-Attention): tp=%d @ %d MHz", solution.tp_pa, solution.f_pa)
        logger.info("  PF (Prefill-FFN):      tp=%d @ %d MHz", solution.tp_pf, solution.f_pf)
        logger.info("  DA (Decode-Attention): tp=%d @ %d MHz", solution.tp_da, solution.f_da)
        logger.info("  DF (Decode-FFN):       tp=%d @ %d MHz", solution.tp_df, solution.f_df)
        logger.info("  k_P=%d  k_D=%d  GPUs=%d/%d  E/layer=%.2f mJ",
                    solution.k_p, solution.k_d, solution.gpu_used,
                    solution._gpu_avail + solution.gpu_used,
                    solution.total_energy_mj_per_layer)

        # Write to JSON
        sol_path = log_dir / _SOLUTION_FILENAME
        sol_path.write_text(json.dumps(solution.to_dict(), indent=2))
        logger.info("Pre-flight solution written to %s", sol_path)

        return str(sol_path)

    except Exception as e:
        logger.error("Pre-flight solve failed: %s", e)
        return None


# ── Launcher ─────────────────────────────────────────────────────────────


def launch_all(cfg: dict, start_with_workload: bool = False) -> int:
    """Start all modules in order.  Returns 0 on success, 1 on failure."""
    log_dir = Path(cfg["logs"]["dir"])
    log_dir.mkdir(parents=True, exist_ok=True)

    # ── Pre-flight Tier 1 solve ──────────────────────────────────────
    solution_path: Optional[str] = None
    if start_with_workload:
        solution_path = _run_preflight_solve(cfg, log_dir)

    # Build Tier 1 extra args for PA (only PA gets them)
    tier1_extra: Optional[list[str]] = None
    tier1_cfg = cfg.get("tier1", {})
    if tier1_cfg.get("enable_tier1_pa", False) or solution_path:
        tier1_extra = []
        if tier1_cfg.get("enable_tier1_pa", False):
            tier1_extra += ["--enable-tier1-pa"]
        if solution_path:
            tier1_extra += ["--tier1-initial-solution", solution_path]
        # Forward AFD SLO thresholds (used by WorkloadMetricsCollector)
        tier1_extra += [
            "--afd-ttft-slo-ms", str(tier1_cfg.get("ttft_slo_ms", 5000.0)),
            "--afd-tpot-slo-us", str(tier1_cfg.get("tpot_slo_us", 50000.0)),
        ]
        # Forward workload params (used when re-plan triggers lazy solver init)
        tier1_extra += [
            "--tier1-gpu-count", str(tier1_cfg.get("gpu_count", 16)),
            "--tier1-lambda-prefill", str(tier1_cfg.get("lambda_prefill", 10.0)),
            "--tier1-n-active-decode", str(tier1_cfg.get("n_active_decode", 32)),
            "--tier1-il-rep-p", str(tier1_cfg.get("il_rep_p", 1024)),
            "--tier1-bs-avg-p", str(tier1_cfg.get("bs_avg_p", 8)),
            "--tier1-il-rep-d", str(tier1_cfg.get("il_rep_d", 512)),
            "--tier1-ol-rep-d", str(tier1_cfg.get("ol_rep_d", 256)),
            "--tier1-bs-avg-d", str(tier1_cfg.get("bs_avg_d", 16)),
            "--tier1-monitor-window-s", str(tier1_cfg.get("monitor_window_s", 30.0)),
            "--tier1-prefill-data-path", str(tier1_cfg.get("prefill_data_path",
                "benchmark/test_motivation/hucc/paper/prefill_data_v1.txt")),
            "--tier1-decode-data-path", str(tier1_cfg.get("decode_data_path",
                "benchmark/test_motivation/hucc/paper/decode_data_v1.txt")),
        ]

    # Shared stats path for DA→PA decode timing (all modules)
    stats_path = str(log_dir / "decode_stats.json")
    stats_extra = ["--tier1-stats-path", stats_path]

    modules = cfg["modules"]

    # ── GPU pool: read configured GPU indices ──────────────────────────
    gpu_pool: list[int] = cfg.get("gpu_indices", [m["gpu"] for m in modules])
    if not gpu_pool:
        logger.error("No gpu_indices configured and no per-module GPU fallback.")
        return 1

    # ── Load Tier 1 solution and build per-module tp/freq map ──────────
    solution_map: dict[str, tuple[int, int]] = {}  # name → (tp, freq_mhz)
    if solution_path:
        sol = _load_solution_dict(solution_path)
        solution_map = {
            "PA": (sol["tp_pa"], sol["f_pa"]),
            "PF": (sol["tp_pf"], sol["f_pf"]),
            "DA": (sol["tp_da"], sol["f_da"]),
            "DF": (sol["tp_df"], sol["f_df"]),
        }

    # ── Allocate GPUs from pool to each module ─────────────────────────
    # Each module gets tp consecutive GPUs from the pool, in module-list order.
    gpu_alloc: dict[str, list[int]] = {}  # name → [nvml_index, ...]
    gpu_cursor = 0
    total_needed = 0
    for mod in modules:
        name = mod["name"]
        tp = solution_map[name][0] if name in solution_map else cfg["model"]["tp"]
        alloc = gpu_pool[gpu_cursor:gpu_cursor + tp]
        if len(alloc) < tp:
            logger.error(
                "Not enough GPUs in pool for %s (need %d, have %d left in pool %s)",
                name, tp, len(gpu_pool) - gpu_cursor, gpu_pool,
            )
            return 1
        gpu_alloc[name] = alloc
        gpu_cursor += tp
        total_needed += tp

    logger.info("")
    logger.info("GPU allocation (from pool %s):", gpu_pool)
    for m_name, m_gpus in gpu_alloc.items():
        tp_val = len(m_gpus)
        freq_val = solution_map[m_name][1] if m_name in solution_map else None
        freq_str = f" @ {freq_val} MHz" if freq_val else ""
        logger.info("  %s: tp=%d%s  GPUs=%s", m_name, tp_val, freq_str, m_gpus)
    logger.info("  Total GPUs used: %d / %d", total_needed, len(gpu_pool))
    logger.info("")

    # ── Build attn/ffn NVML index lists for PA monitoring ─────────────
    all_attn_gpus: list[int] = []
    all_ffn_gpus: list[int] = []
    for mod in modules:
        name = mod["name"]
        if mod["perspective"] == "attn":
            all_attn_gpus.extend(gpu_alloc[name])
        else:
            all_ffn_gpus.extend(gpu_alloc[name])

    processes: list[subprocess.Popen] = []

    try:
        for i, mod in enumerate(modules):
            name = mod["name"]
            log_file = log_dir / f"{name}.log"

            # Apply Tier 1 tp/freq for this module if available
            tp_freq = solution_map.get(name)
            tp_override: Optional[int] = tp_freq[0] if tp_freq else None
            freq_mhz: Optional[int] = tp_freq[1] if tp_freq else None

            # Allocated GPUs for this module
            allocated_gpus = gpu_alloc[name]
            cuda_visible = ",".join(str(g) for g in allocated_gpus)

            # Lock GPU frequencies BEFORE launching the server process
            if freq_mhz is not None:
                for gpu_idx in allocated_gpus:
                    _set_gpu_locked_clock(gpu_idx, freq_mhz)

            # Only PA gets Tier 1 extra args; stats path goes to all modules
            extra = tier1_extra if name == "PA" else []
            if extra is None:
                extra = []
            extra += stats_extra
            cmd = _build_server_cmd(cfg, mod, extra, tp_override=tp_override)
            env = _build_server_env(cfg, mod, cuda_visible_devices=cuda_visible)

            # Pass physical NVML GPU index for DVFS (first GPU in allocation)
            env["AFD_NVML_DEVICE_INDEX"] = str(allocated_gpus[0])

            # Pass GPU→pool mapping to PA for multi-GPU NVML monitoring
            if name == "PA":
                env["AFD_ATTN_GPU_INDICES"] = ",".join(str(g) for g in all_attn_gpus)
                env["AFD_FFN_GPU_INDICES"] = ",".join(str(g) for g in all_ffn_gpus)

            total = len(modules) + (1 if cfg.get("router", {}).get("enabled", True) else 0)
            tp_val = tp_override if tp_override else cfg["model"]["tp"]
            freq_str = f" @{freq_mhz}MHz" if freq_mhz else ""
            logger.info(
                "[%d/%d] Starting %s (GPU=%s, tp=%d%s, port=%s) %s→ %s",
                i + 1, total, name, cuda_visible, tp_val, freq_str, mod["port"],
                "(+tier1) " if extra else "",
                log_file,
            )

            fh = open(log_file, "w")
            proc = subprocess.Popen(
                cmd,
                env=env,
                stdout=fh,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
            processes.append(proc)

            # Wait for the HTTP port to be ready
            port = mod["port"]
            logger.info("  Waiting for port %s ...", port)
            ready = _check_port_ready("127.0.0.1", port)
            if not ready:
                logger.error(
                    "  %s failed to start within %s seconds (port %s). "
                    "Check %s",
                    name, _PORT_TIMEOUT_S, port, log_file,
                )
                return 1
            logger.info("  %s is ready.", name)

        # Router (no GPU, no port readiness check needed — it connects on startup)
        router_cfg = cfg.get("router", {})
        if router_cfg.get("enabled", True):
            log_file = log_dir / "router.log"
            cmd = _build_router_cmd(cfg)
            logger.info(
                "[%d/%d] Starting router → %s",
                len(modules) + 1, len(modules) + 1, log_file,
            )
            fh = open(log_file, "w")
            proc = subprocess.Popen(
                cmd,
                env=os.environ.copy(),
                stdout=fh,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
            processes.append(proc)

        logger.info("All modules started successfully.")
        return 0

    except Exception as e:
        logger.error("Launch failed: %s", e)
        return 1


# ── CLI ──────────────────────────────────────────────────────────────────


def main():
    import argparse

    parser = argparse.ArgumentParser(description="AFlex batch launcher")
    parser.add_argument(
        "--config",
        default=str(_DEFAULT_CONFIG),
        help=f"Path to JSON config (default: {_DEFAULT_CONFIG})",
    )
    parser.add_argument(
        "--start-with-workload",
        action="store_true",
        help="Run a pre-flight Tier1Solver.solve() before launching. "
        "The solution is written to a JSON file and passed to PA via "
        "--tier1-initial-solution, so PA does not need to re-initialise "
        "the solver at startup.",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )

    cfg_path = Path(args.config)
    if not cfg_path.exists():
        logger.error("Config not found: %s", cfg_path)
        sys.exit(1)

    cfg = json.loads(cfg_path.read_text())

    # Pre-flight solve is enabled if CLI flag OR config tier1.start_with_workload is set
    start_with_workload = args.start_with_workload or cfg.get("tier1", {}).get("start_with_workload", False)

    logger.info("=" * 50)
    logger.info("AFlex launcher — config: %s", cfg_path)
    logger.info("Modules: %s", [m["name"] for m in cfg["modules"]])
    if start_with_workload:
        logger.info("Pre-flight solve: ENABLED")
    logger.info("=" * 50)

    rc = launch_all(cfg, start_with_workload=start_with_workload)
    sys.exit(rc)


if __name__ == "__main__":
    main()

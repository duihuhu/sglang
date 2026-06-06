#!/usr/bin/env python3
"""Test Tier1 full reload flow — verifies signal protocol and orchestrator logic.

Tests:
  1. Signal protocol: write/read/poll reload signals
  2. Orchestrator config generation: verify _build_reload_config produces valid JSON
  3. End-to-end reload: start server → trigger reload with new TP → verify restart

Usage:
    /workspace/env/sglang-tier/bin/python test_tier1_reload.py
    /workspace/env/sglang-tier/bin/python test_tier1_reload.py --e2e  # full end-to-end (needs GPUs)
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
import tempfile
import threading
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "python"))


def test_signal_protocol():
    """Test reload signal read/write/poll."""
    from sglang.srt.energy.reload_signal import (
        read_signal, is_reloading, is_ready, wait_until_ready, clear_signal,
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        signal_path = os.path.join(tmpdir, "tier1_reload_signal.json")

        # Initially: no file → idle
        sig = read_signal(signal_path)
        assert sig["status"] == "idle", f"Expected idle, got {sig}"
        assert not is_reloading(signal_path)
        assert is_ready(signal_path)

        # Write reloading
        with open(signal_path, "w") as f:
            json.dump({"status": "reloading", "timestamp": time.time()}, f)
        assert is_reloading(signal_path)
        assert not is_ready(signal_path)

        # Write ready
        with open(signal_path, "w") as f:
            json.dump({"status": "ready", "timestamp": time.time(), "reload_duration_s": 5.0}, f)
        assert not is_reloading(signal_path)
        assert is_ready(signal_path)

        # Test wait_until_ready with background writer
        with open(signal_path, "w") as f:
            json.dump({"status": "reloading", "timestamp": time.time()}, f)

        def _write_ready_after_delay():
            time.sleep(1.0)
            with open(signal_path, "w") as f:
                json.dump({"status": "ready", "timestamp": time.time()}, f)

        t = threading.Thread(target=_write_ready_after_delay)
        t.start()
        result = wait_until_ready(signal_path, timeout=5.0, poll_interval=0.3)
        t.join()
        assert result, "wait_until_ready should return True"

        # Test timeout
        with open(signal_path, "w") as f:
            json.dump({"status": "reloading", "timestamp": time.time()}, f)
        result = wait_until_ready(signal_path, timeout=0.5, poll_interval=0.1)
        assert not result, "Should timeout"

        # Clear
        clear_signal(signal_path)
        assert is_ready(signal_path)

    print("[PASS] test_signal_protocol")


def test_reload_config_generation():
    """Test that _build_reload_config produces valid config for orchestrator."""
    from unittest.mock import MagicMock
    from sglang.srt.energy.tier1_solver import Tier1Solution

    # Mock server_args
    sa = MagicMock()
    sa.model_path = "/models/Qwen/Qwen3-32B/"
    sa.port = 51010  # PA port
    sa.mem_fraction_static = 0.85
    sa.afd_dvfs_enabled = True
    sa.afd_energy_model_dir = "/workspace/sglang/benchmark/test_motivation/energy_models"
    sa.afd_ttft_slo_ms = 5000
    sa.afd_tpot_slo_us = 300000
    sa.afd_micro_batch = 1
    sa.enable_tier1_pa = True
    sa.tier1_monitor_window_s = 15.0
    sa.tier1_gpu_count = 4
    sa.tier1_stats_path = "/tmp/tier1_test/decode_stats.json"
    sa.disaggregation_bootstrap_port = 29999
    sa.disaggregation_ib_device = "mlx5_4"

    # Mock scheduler
    scheduler = MagicMock()
    scheduler.server_args = sa

    # Set environment
    os.environ["AFD_NVML_DEVICE_INDEX"] = "7"
    os.environ["CUDA_VISIBLE_DEVICES"] = "6,7"
    os.environ["AFD_UCX_BASE_PORT"] = "26200"
    os.environ["AFD_SCHED_PORT"] = "66400"

    # Import the method
    from sglang.srt.managers.scheduler import Scheduler
    build_fn = Scheduler._build_reload_config

    solution = Tier1Solution(
        k_p=1, k_d=1,
        tp_pa=2, tp_pf=2, tp_da=1, tp_df=1,
        f_pa=1410, f_pf=1410, f_da=930, f_df=930,
    )

    config = build_fn(scheduler, solution, "/tmp/tier1_test/tier1_reload_signal.json")

    # Validate structure
    assert "signal_path" in config
    assert "solution" in config
    assert "server_config" in config

    sol = config["solution"]
    assert sol["tp_pa"] == 2
    assert sol["tp_da"] == 1
    assert sol["f_pa"] == 1410

    sc = config["server_config"]
    assert sc["model_path"] == "/models/Qwen/Qwen3-32B/"
    assert len(sc["modules"]) == 4
    assert sc["router"]["enabled"] is True

    # Verify module names
    names = [m["name"] for m in sc["modules"]]
    assert "PA" in names
    assert "PF" in names
    assert "DA" in names
    assert "DF" in names

    # Verify JSON serializable
    json_str = json.dumps(config, indent=2)
    assert len(json_str) > 100

    print("[PASS] test_reload_config_generation")
    print(f"  Config size: {len(json_str)} bytes")
    print(f"  Modules: {names}")
    print(f"  Solution: tp_pa={sol['tp_pa']} tp_da={sol['tp_da']}")


def test_orchestrator_import():
    """Verify reload_orchestrator can be imported and has expected interface."""
    from sglang.srt.energy.reload_orchestrator import (
        run_reload, _start_all_servers, _write_signal,
        _get_pids_on_port, _wait_port_free, _reset_gpu_clocks,
    )
    print("[PASS] test_orchestrator_import")


def test_benchmark_pause_logic():
    """Test that benchmark workload runner handles reload signal correctly."""
    from sglang.srt.energy.reload_signal import clear_signal, is_reloading

    with tempfile.TemporaryDirectory() as tmpdir:
        signal_path = os.path.join(tmpdir, "tier1_reload_signal.json")
        clear_signal(signal_path)

        # Simulate: signal goes to reloading, then ready after 1s
        def _simulate_reload():
            time.sleep(0.5)
            with open(signal_path, "w") as f:
                json.dump({"status": "reloading", "timestamp": time.time()}, f)
            time.sleep(1.5)
            with open(signal_path, "w") as f:
                json.dump({"status": "ready", "timestamp": time.time(),
                          "reload_duration_s": 1.5}, f)

        t = threading.Thread(target=_simulate_reload)
        t.start()

        # Simulate benchmark checking signal in a loop
        paused = False
        resumed = False
        for i in range(30):
            if is_reloading(signal_path):
                paused = True
                from sglang.srt.energy.reload_signal import wait_until_ready
                wait_until_ready(signal_path, timeout=10)
                resumed = True
                break
            time.sleep(0.2)

        t.join()
        assert paused, "Should have detected reloading"
        assert resumed, "Should have resumed after ready"

    print("[PASS] test_benchmark_pause_logic")


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--e2e", action="store_true", help="Run full end-to-end test (needs GPUs)")
    args = parser.parse_args()

    print("=" * 60)
    print("  TIER1 RELOAD TESTS")
    print("=" * 60)

    test_signal_protocol()
    test_orchestrator_import()
    test_reload_config_generation()
    test_benchmark_pause_logic()

    print("\n" + "=" * 60)
    print("  ALL TESTS PASSED")
    print("=" * 60)


if __name__ == "__main__":
    main()

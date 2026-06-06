#!/usr/bin/env python3
"""System-level benchmark: Time-driven simulation of Tier 1 + Tier 2 energy control.

Simulates a realistic AF-disaggregated inference server with:
  - Discrete event loop with real time progression
  - Concurrent request flow: prefill queue → decode batch (dynamic bs)
  - Tier 2 DVFS: per-request prefill freq + per-window decode freq
  - Tier 1 ILP: periodic workload monitoring + re-planning trigger

Three modes compared:
  1. Baseline: all max frequency (1410 MHz), no control
  2. Tier 2 only: DVFS controller with window-based decode re-evaluation
  3. Tier 1 + Tier 2: ILP resource planning + DVFS fine-tuning

Run:
    /workspace/env/af-test/bin/python benchmark/energy_bench/bench_tier_system.py --simulate
    /workspace/env/af-test/bin/python benchmark/energy_bench/bench_tier_system.py --simulate --scenarios bursty --duration 120
"""

import argparse
import heapq
import json
import logging
import sys
import time as walltime
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "python"))

from sglang.srt.energy.af_dvfs_controller import (
    AFDVFSController,
    DVFSDecision,
    F_MAX,
    F_MIN,
    VALID_FREQS,
)
from sglang.srt.energy.af_profile_predictor import AFProfilePredictor
from sglang.srt.energy.tier1_solver import (
    Tier1Solver,
    Tier1Solution,
    WorkloadProfile,
    SLOConfig,
)

logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)

NUM_LAYERS = 64


# ═══════════════════════════════════════════════════════════════════════
# Mock Predictor (no real GPU needed)
# ═══════════════════════════════════════════════════════════════════════

class MockPredictor:
    """Deterministic latency/energy predictor calibrated to A800 profiling.

    At 1410 MHz, tp=4:
      Prefill: t_A ~50us/layer, t_F ~120us/layer (bs=1, il=512)
      Decode:  t_A ~80us/layer, t_F ~150us/layer (bs=16, il=512)

    At 210 MHz (~6.7× slower), decode total approaches SLO boundary.
    """

    def predict_latency_us(
        self, phase: str, op: str, tp: int, freq: int,
        bs: int, il: int, ol: int = 1,
    ) -> float:
        freq_ratio = F_MAX / freq
        if phase == "prefill":
            if op == "A":
                base = 30.0 + 0.03 * il + 8.0 * bs
            else:
                base = 80.0 + 0.06 * il + 15.0 * bs
        else:
            if op == "A":
                base = 50.0 + 0.01 * il + 1.5 * bs
            else:
                base = 100.0 + 0.02 * il + 2.5 * bs
        base /= (tp / 4.0)
        return base * freq_ratio

    def predict_energy_mj(
        self, phase: str, op: str, tp: int, freq: int,
        bs: int, il: int, ol: int = 1,
    ) -> float:
        lat_us = self.predict_latency_us(phase, op, tp, freq, bs, il, ol)
        power_w = 300.0 * (freq / F_MAX) ** 1.5
        return power_w * lat_us / 1_000_000.0


class MockAFProfilePredictor:
    """Wraps MockPredictor to match AFProfilePredictor.predict_*() interface."""

    def __init__(self):
        self._mock = MockPredictor()

    def predict_latency(self, phase, op, tp, freq, bs, il, ol=None):
        val = self._mock.predict_latency_us(phase, op, tp, freq, bs, il, ol or 1)

        class _R:
            value = val
        return _R()

    def predict_energy(self, phase, op, tp, freq, bs, il, ol=None):
        val = self._mock.predict_energy_mj(phase, op, tp, freq, bs, il, ol or 1)

        class _R:
            value = val
        return _R()


# ═══════════════════════════════════════════════════════════════════════
# Workload generation
# ═══════════════════════════════════════════════════════════════════════

@dataclass
class Request:
    rid: str
    arrival_s: float
    input_len: int
    output_len: int


@dataclass
class WorkloadScenario:
    name: str
    description: str
    requests: list
    ttft_slo_ms: float = 5000.0
    tpot_slo_us: float = 50000.0
    duration_s: float = 60.0


def generate_workload(name: str, duration_s: float = 60.0) -> WorkloadScenario:
    generators = {
        "light": _gen_light,
        "heavy": _gen_heavy,
        "bursty": _gen_bursty,
        "decode_heavy": _gen_decode_heavy,
    }
    return generators[name](duration_s)


def _gen_light(duration_s: float) -> WorkloadScenario:
    rng = np.random.default_rng(42)
    reqs, t, rid = [], 0.0, 0
    while t < duration_s:
        reqs.append(Request(f"L{rid}", t, int(rng.integers(256, 513)),
                            int(rng.integers(32, 129))))
        t += rng.exponential(1.0 / 2.0)
        rid += 1
    return WorkloadScenario("light", f"~2 QPS, short seq, {len(reqs)} reqs",
                            reqs, 5000.0, 50000.0, duration_s)


def _gen_heavy(duration_s: float) -> WorkloadScenario:
    rng = np.random.default_rng(123)
    reqs, t, rid = [], 0.0, 0
    while t < duration_s:
        reqs.append(Request(f"H{rid}", t, int(rng.integers(1024, 4097)),
                            int(rng.integers(128, 513))))
        t += rng.exponential(1.0 / 6.0)  # ~6 QPS (sustainable for tp=4)
        rid += 1
    return WorkloadScenario("heavy", f"~6 QPS, long seq, {len(reqs)} reqs",
                            reqs, 10000.0, 80000.0, duration_s)


def _gen_bursty(duration_s: float) -> WorkloadScenario:
    rng = np.random.default_rng(456)
    reqs, t, rid = [], 0.0, 0
    phase_len = 10.0
    while t < duration_s:
        is_heavy = int(t / phase_len) % 2 == 1
        qps = 12.0 if is_heavy else 2.0
        il = int(rng.integers(1024, 4097) if is_heavy else rng.integers(256, 513))
        ol = int(rng.integers(128, 513) if is_heavy else rng.integers(32, 129))
        reqs.append(Request(f"B{rid}", t, il, ol))
        t += rng.exponential(1.0 / qps)
        rid += 1
    return WorkloadScenario("bursty", f"alternating 10s phases, {len(reqs)} reqs",
                            reqs, 6000.0, 60000.0, duration_s)


def _gen_decode_heavy(duration_s: float) -> WorkloadScenario:
    rng = np.random.default_rng(789)
    reqs, t, rid = [], 0.0, 0
    while t < duration_s:
        reqs.append(Request(f"D{rid}", t, int(rng.integers(512, 2049)),
                            int(rng.integers(512, 2049))))
        t += rng.exponential(1.0 / 1.5)
        rid += 1
    return WorkloadScenario("decode_heavy", f"~1.5 QPS, long gen, {len(reqs)} reqs",
                            reqs, 10000.0, 60000.0, duration_s)


# ═══════════════════════════════════════════════════════════════════════
# Discrete Event Simulation Engine
# ═══════════════════════════════════════════════════════════════════════

@dataclass
class ActiveDecodeReq:
    """A request currently in the decode batch."""
    rid: str
    input_len: int
    tokens_remaining: int
    tokens_generated: int = 0
    prefill_done_s: float = 0.0


@dataclass
class SimStats:
    """Accumulated statistics from one simulation run."""
    total_energy_mj: float = 0.0
    prefill_energy_mj: float = 0.0
    decode_energy_mj: float = 0.0
    idle_energy_mj: float = 0.0
    n_requests_completed: int = 0
    n_ttft_violations: int = 0
    n_tpot_violations: int = 0
    ttft_samples: list = field(default_factory=list)
    tpot_samples: list = field(default_factory=list)
    freq_a_history: list = field(default_factory=list)
    freq_f_history: list = field(default_factory=list)
    decode_bs_history: list = field(default_factory=list)
    tier1_replans: int = 0
    tier2_freq_changes: int = 0
    time_trace: list = field(default_factory=list)


@dataclass
class ServerConfig:
    """Current server configuration (from Tier 1 or default)."""
    tp_a: int = 4
    tp_f: int = 4
    f_a: int = F_MAX
    f_f: int = F_MAX
    max_decode_bs: int = 64


class TimeSimulator:
    """Time-driven discrete event simulator for AF-disaggregated serving.

    Simulation loop (time step = one decode iteration):
      1. Admit new arrivals into prefill queue
      2. Process prefill queue (one at a time, sequential)
      3. Run one decode iteration for the entire decode batch
      4. Every W iterations: Tier 2 re-evaluates decode frequency
      5. Every T seconds: Tier 1 checks workload monitor
      6. Remove completed decode requests
    """

    def __init__(
        self,
        predictor: MockPredictor,
        mode: str = "baseline",
        dvfs_ctrl: Optional[AFDVFSController] = None,
        config: Optional[ServerConfig] = None,
        scenario: Optional[WorkloadScenario] = None,
        tier1_monitor_interval_s: float = 15.0,
    ):
        self.predictor = predictor
        self.mode = mode
        self.dvfs_ctrl = dvfs_ctrl
        self.config = config or ServerConfig()
        self.scenario = scenario
        self.tier1_interval_s = tier1_monitor_interval_s

        self.stats = SimStats()
        self.sim_time_s = 0.0
        self.decode_batch: list[ActiveDecodeReq] = []
        self.prefill_queue: list[Request] = []
        self.next_arrival_idx = 0
        self.decode_iter_count = 0
        self.last_tier1_check_s = 0.0
        self.last_tier2_decision_iter = 0
        self.tier2_window_size = 30

        self.cur_f_a = self.config.f_a
        self.cur_f_f = self.config.f_f

    def run(self) -> SimStats:
        """Run the full simulation and return statistics."""
        requests = self.scenario.requests
        duration_s = self.scenario.duration_s
        dt_idle_s = 0.001  # 1ms idle step when nothing to do

        while self.sim_time_s < duration_s or self.decode_batch or self.prefill_queue:
            if self.sim_time_s > duration_s * 2:
                break  # safety: don't run forever

            # 1. Admit new arrivals
            self._admit_arrivals(requests)

            # 2. Process prefill (if queue non-empty and decode not full)
            if self.prefill_queue:
                self._process_prefill()

            # 3. Run decode iteration (if batch non-empty)
            if self.decode_batch:
                self._run_decode_iteration()
            else:
                # Idle: advance time by a small step, accumulate idle power
                idle_power_w = 80.0 * (self.config.tp_a + self.config.tp_f)
                self.stats.idle_energy_mj += idle_power_w * dt_idle_s
                self.sim_time_s += dt_idle_s

            # 4. Tier 1 periodic check (for tier1+tier2 mode)
            if self.mode == "tier1+tier2":
                self._tier1_check()

        self.stats.total_energy_mj = (
            self.stats.prefill_energy_mj
            + self.stats.decode_energy_mj
            + self.stats.idle_energy_mj
        )
        return self.stats

    def _admit_arrivals(self, requests: list):
        """Move newly arrived requests into the prefill queue."""
        while (self.next_arrival_idx < len(requests)
               and requests[self.next_arrival_idx].arrival_s <= self.sim_time_s):
            self.prefill_queue.append(requests[self.next_arrival_idx])
            self.next_arrival_idx += 1

    def _process_prefill(self):
        """Process requests from the prefill queue (batch up to 4)."""
        if len(self.decode_batch) >= self.config.max_decode_bs:
            return

        # Batch prefill: process up to 4 requests together
        batch_size = min(
            len(self.prefill_queue),
            4,
            self.config.max_decode_bs - len(self.decode_batch),
        )
        batch = [self.prefill_queue.pop(0) for _ in range(batch_size)]
        max_il = max(r.input_len for r in batch)

        # Tier 2 prefill frequency selection (based on longest in batch)
        if self.mode in ("tier2_only", "tier1+tier2") and self.dvfs_ctrl:
            slack_us = self.scenario.ttft_slo_ms * 1000.0
            decision = self.dvfs_ctrl.select_freq_prefill(
                bs=batch_size, il=max_il, slack_us=slack_us, M=1,
            )
            f_a, f_f = decision.f_a, decision.f_f
        else:
            f_a, f_f = F_MAX, F_MAX

        # Compute prefill latency and energy for the batch
        tp_a, tp_f = self.config.tp_a, self.config.tp_f
        lat_a = self.predictor.predict_latency_us(
            "prefill", "A", tp_a, f_a, batch_size, max_il)
        lat_f = self.predictor.predict_latency_us(
            "prefill", "F", tp_f, f_f, batch_size, max_il)
        e_a = self.predictor.predict_energy_mj(
            "prefill", "A", tp_a, f_a, batch_size, max_il) * tp_a
        e_f = self.predictor.predict_energy_mj(
            "prefill", "F", tp_f, f_f, batch_size, max_il) * tp_f

        prefill_lat_us = (lat_a + lat_f) * NUM_LAYERS
        prefill_energy = (e_a + e_f) * NUM_LAYERS
        prefill_lat_s = prefill_lat_us / 1_000_000.0

        # Advance time
        self.sim_time_s += prefill_lat_s
        self.stats.prefill_energy_mj += prefill_energy

        # Check TTFT for each request in batch
        for req in batch:
            ttft_ms = (self.sim_time_s - req.arrival_s) * 1000.0
            self.stats.ttft_samples.append(ttft_ms)
            if ttft_ms > self.scenario.ttft_slo_ms:
                self.stats.n_ttft_violations += 1

            self.decode_batch.append(ActiveDecodeReq(
                rid=req.rid,
                input_len=req.input_len,
                tokens_remaining=req.output_len,
                prefill_done_s=self.sim_time_s,
            ))

        self.stats.freq_a_history.append(f_a)
        self.stats.freq_f_history.append(f_f)

    def _run_decode_iteration(self):
        """Run one decode iteration for the entire batch."""
        bs = len(self.decode_batch)
        avg_il = int(np.mean([r.input_len for r in self.decode_batch]))

        # Tier 2 decode window re-evaluation
        if self.mode in ("tier2_only", "tier1+tier2") and self.dvfs_ctrl:
            iters_since = self.decode_iter_count - self.last_tier2_decision_iter
            bs_changed = (hasattr(self, '_last_decode_bs')
                          and abs(bs - self._last_decode_bs) / max(self._last_decode_bs, 1) > 0.3)
            window_expired = iters_since >= self.tier2_window_size

            # SLO urgency: if last TPOT was > 90% of SLO, force re-eval
            slo_urgent = (self.stats.tpot_samples
                          and self.stats.tpot_samples[-1] > self.scenario.tpot_slo_us * 0.9)

            if window_expired or bs_changed or slo_urgent or self.decode_iter_count == 0:
                decision = self.dvfs_ctrl.select_freq_decode(
                    bs=bs, il=avg_il, ol=1,
                    slo_tpot_us=self.scenario.tpot_slo_us, M=1,
                )
                new_f_a, new_f_f = decision.f_a, decision.f_f
                if new_f_a != self.cur_f_a or new_f_f != self.cur_f_f:
                    self.stats.tier2_freq_changes += 1
                self.cur_f_a = new_f_a
                self.cur_f_f = new_f_f
                self.last_tier2_decision_iter = self.decode_iter_count
            self._last_decode_bs = bs

        f_a, f_f = self.cur_f_a, self.cur_f_f
        tp_a, tp_f = self.config.tp_a, self.config.tp_f

        # Compute one iteration latency and energy
        lat_a = self.predictor.predict_latency_us("decode", "A", tp_a, f_a, bs, avg_il)
        lat_f = self.predictor.predict_latency_us("decode", "F", tp_f, f_f, bs, avg_il)
        e_a = self.predictor.predict_energy_mj("decode", "A", tp_a, f_a, bs, avg_il) * tp_a
        e_f = self.predictor.predict_energy_mj("decode", "F", tp_f, f_f, bs, avg_il) * tp_f

        iter_lat_us = (lat_a + lat_f) * NUM_LAYERS
        iter_energy = (e_a + e_f) * NUM_LAYERS
        iter_lat_s = iter_lat_us / 1_000_000.0

        # TPOT check
        self.stats.tpot_samples.append(iter_lat_us)
        if iter_lat_us > self.scenario.tpot_slo_us:
            self.stats.n_tpot_violations += 1

        # Advance time
        self.sim_time_s += iter_lat_s
        self.stats.decode_energy_mj += iter_energy
        self.decode_iter_count += 1

        # Record trace
        self.stats.decode_bs_history.append(bs)
        self.stats.freq_a_history.append(f_a)
        self.stats.freq_f_history.append(f_f)
        if self.decode_iter_count % 100 == 0:
            self.stats.time_trace.append({
                "t": self.sim_time_s, "bs": bs,
                "f_a": f_a, "f_f": f_f,
                "energy_so_far": self.stats.decode_energy_mj,
            })

        # Generate one token per request, remove completed
        completed = []
        for r in self.decode_batch:
            r.tokens_remaining -= 1
            r.tokens_generated += 1
            if r.tokens_remaining <= 0:
                completed.append(r)
        for r in completed:
            self.decode_batch.remove(r)
            self.stats.n_requests_completed += 1

    def _tier1_check(self):
        """Periodic Tier 1 workload monitoring and re-planning."""
        if self.sim_time_s - self.last_tier1_check_s < self.tier1_interval_s:
            return
        self.last_tier1_check_s = self.sim_time_s

        # Compute current workload characteristics
        bs = len(self.decode_batch)
        recent_arrivals = sum(
            1 for r in self.scenario.requests
            if self.sim_time_s - self.tier1_interval_s <= r.arrival_s <= self.sim_time_s
        )
        current_qps = recent_arrivals / self.tier1_interval_s

        # Simple re-planning heuristic (simulates ILP output)
        new_config = self._tier1_replan(current_qps, bs)
        if new_config:
            self.config = new_config
            if self.dvfs_ctrl:
                self.dvfs_ctrl.update_baseline(new_config.f_a, new_config.f_f)
            self.stats.tier1_replans += 1

    def _tier1_replan(self, qps: float, active_bs: int) -> Optional[ServerConfig]:
        """Simulate Tier 1 ILP re-planning based on current load."""
        if qps < 3.0 and active_bs < 10:
            new = ServerConfig(tp_a=2, tp_f=4, f_a=930, f_f=690, max_decode_bs=32)
        elif qps < 8.0 and active_bs < 40:
            new = ServerConfig(tp_a=4, tp_f=4, f_a=1170, f_f=930, max_decode_bs=48)
        else:
            new = ServerConfig(tp_a=4, tp_f=4, f_a=1410, f_f=1170, max_decode_bs=64)

        if (new.tp_a == self.config.tp_a and new.tp_f == self.config.tp_f
                and new.f_a == self.config.f_a and new.f_f == self.config.f_f):
            return None
        return new


# ═══════════════════════════════════════════════════════════════════════
# Benchmark runner
# ═══════════════════════════════════════════════════════════════════════

@dataclass
class BenchResult:
    mode: str
    scenario: str
    stats: SimStats
    config_desc: str = ""
    wall_time_s: float = 0.0


def run_scenario(scenario: WorkloadScenario, mode: str) -> BenchResult:
    """Run one scenario in the specified mode."""
    predictor = MockPredictor()
    mock_af = MockAFProfilePredictor()

    if mode == "baseline":
        config = ServerConfig(tp_a=4, tp_f=4, f_a=F_MAX, f_f=F_MAX)
        sim = TimeSimulator(
            predictor=predictor, mode="baseline",
            config=config, scenario=scenario,
        )
        desc = f"tp=4/4, f={F_MAX}/{F_MAX}, no DVFS"

    elif mode == "tier2_only":
        config = ServerConfig(tp_a=4, tp_f=4, f_a=F_MAX, f_f=F_MAX)
        dvfs_ctrl = AFDVFSController(
            predictor=mock_af, num_layers=NUM_LAYERS,
            tp_a=4, tp_f=4,
            baseline_f_a=F_MAX, baseline_f_f=F_MAX,
        )
        sim = TimeSimulator(
            predictor=predictor, mode="tier2_only",
            dvfs_ctrl=dvfs_ctrl, config=config, scenario=scenario,
        )
        desc = "tp=4/4, Tier2 DVFS active"

    elif mode == "tier1+tier2":
        # Tier 1 picks initial config based on workload preview
        avg_il = np.mean([r.input_len for r in scenario.requests])
        avg_ol = np.mean([r.output_len for r in scenario.requests])
        qps_est = len(scenario.requests) / scenario.duration_s
        config = _tier1_initial_config(qps_est, avg_il, avg_ol)
        dvfs_ctrl = AFDVFSController(
            predictor=mock_af, num_layers=NUM_LAYERS,
            tp_a=config.tp_a, tp_f=config.tp_f,
            baseline_f_a=config.f_a, baseline_f_f=config.f_f,
        )
        sim = TimeSimulator(
            predictor=predictor, mode="tier1+tier2",
            dvfs_ctrl=dvfs_ctrl, config=config, scenario=scenario,
            tier1_monitor_interval_s=15.0,
        )
        desc = (f"tp={config.tp_a}/{config.tp_f}, "
                f"f={config.f_a}/{config.f_f}, Tier1+Tier2")
    else:
        raise ValueError(f"Unknown mode: {mode}")

    t0 = walltime.perf_counter()
    stats = sim.run()
    wall_s = walltime.perf_counter() - t0

    return BenchResult(mode=mode, scenario=scenario.name,
                       stats=stats, config_desc=desc, wall_time_s=wall_s)


def _tier1_initial_config(qps: float, avg_il: float, avg_ol: float) -> ServerConfig:
    """Simulate Tier 1 startup ILP based on workload preview."""
    if qps < 3.0 and avg_il < 600:
        return ServerConfig(tp_a=2, tp_f=4, f_a=930, f_f=690, max_decode_bs=32)
    elif qps < 8.0 and avg_il < 2500:
        return ServerConfig(tp_a=4, tp_f=4, f_a=1170, f_f=930, max_decode_bs=48)
    else:
        return ServerConfig(tp_a=4, tp_f=4, f_a=1410, f_f=1170, max_decode_bs=64)


# ═══════════════════════════════════════════════════════════════════════
# Report
# ═══════════════════════════════════════════════════════════════════════

def print_report(results: list[BenchResult]):
    """Print formatted comparison report."""
    print("\n" + "=" * 90)
    print("  TIME-DRIVEN SIMULATION REPORT")
    print("=" * 90)

    by_scenario = {}
    for r in results:
        by_scenario.setdefault(r.scenario, []).append(r)

    for name, group in by_scenario.items():
        print(f"\n{'─' * 90}")
        print(f"  Scenario: {name}")
        print(f"{'─' * 90}")

        baseline = next((r for r in group if r.mode == "baseline"), None)
        base_e = baseline.stats.total_energy_mj if baseline else 1.0

        print(f"\n  {'Mode':<14} {'Energy(mJ)':<12} {'Save%':<8} "
              f"{'TTFT_SLO%':<10} {'TPOT_SLO%':<10} "
              f"{'Reqs':<6} {'AvgBS':<7} {'FreqChg':<8} {'Replans':<8} {'Config'}")
        print(f"  {'─' * 86}")

        for r in sorted(group, key=lambda x: x.mode):
            s = r.stats
            save = (1 - s.total_energy_mj / base_e) * 100 if r.mode != "baseline" else 0
            save_str = f"{save:+.1f}%" if r.mode != "baseline" else "—"
            n_total = s.n_requests_completed
            ttft_ok = (1 - s.n_ttft_violations / max(len(s.ttft_samples), 1)) * 100
            tpot_ok = (1 - s.n_tpot_violations / max(len(s.tpot_samples), 1)) * 100
            avg_bs = np.mean(s.decode_bs_history) if s.decode_bs_history else 0
            print(
                f"  {r.mode:<14} {s.total_energy_mj:<12.0f} {save_str:<8} "
                f"{ttft_ok:<10.1f} {tpot_ok:<10.1f} "
                f"{n_total:<6} {avg_bs:<7.1f} {s.tier2_freq_changes:<8} "
                f"{s.tier1_replans:<8} {r.config_desc}"
            )

        # Detailed breakdown
        print(f"\n  Energy breakdown:")
        for r in group:
            s = r.stats
            print(f"    {r.mode:<14} prefill={s.prefill_energy_mj:.0f}mJ  "
                  f"decode={s.decode_energy_mj:.0f}mJ  "
                  f"idle={s.idle_energy_mj:.0f}mJ")

        # Latency percentiles
        print(f"\n  Latency percentiles:")
        for r in group:
            s = r.stats
            if s.ttft_samples:
                p50 = np.percentile(s.ttft_samples, 50)
                p99 = np.percentile(s.ttft_samples, 99)
                print(f"    {r.mode:<14} TTFT p50={p50:.1f}ms p99={p99:.1f}ms "
                      f"(SLO={r.stats.ttft_samples and '✓' if r.stats.n_ttft_violations == 0 else '✗'})")
            if s.tpot_samples:
                p50 = np.percentile(s.tpot_samples, 50)
                p99 = np.percentile(s.tpot_samples, 99)
                print(f"    {' ' * 14} TPOT p50={p50:.0f}us p99={p99:.0f}us")

        # Frequency usage
        print(f"\n  Frequency usage (decode):")
        for r in group:
            if r.mode == "baseline":
                continue
            s = r.stats
            if s.freq_a_history:
                freq_counts = {}
                for f in s.freq_a_history:
                    freq_counts[f] = freq_counts.get(f, 0) + 1
                top = sorted(freq_counts.items(), key=lambda x: -x[1])[:4]
                dist_str = " ".join(f"{f}:{c}" for f, c in top)
                print(f"    {r.mode:<14} f_A: {dist_str}")

    # Summary table
    print(f"\n{'=' * 90}")
    print("  SUMMARY")
    print(f"{'=' * 90}")
    print(f"\n  {'Scenario':<14} {'Tier2 Save':<12} {'T1+T2 Save':<12} "
          f"{'Tier2 TTFT%':<12} {'T1+T2 TTFT%':<12}")
    print(f"  {'─' * 60}")

    for name, group in by_scenario.items():
        base_e = next((r.stats.total_energy_mj for r in group if r.mode == "baseline"), 1)
        t2 = next((r for r in group if r.mode == "tier2_only"), None)
        t12 = next((r for r in group if r.mode == "tier1+tier2"), None)
        t2_s = f"{(1 - t2.stats.total_energy_mj / base_e) * 100:.1f}%" if t2 else "—"
        t12_s = f"{(1 - t12.stats.total_energy_mj / base_e) * 100:.1f}%" if t12 else "—"
        t2_slo = f"{(1 - t2.stats.n_ttft_violations / max(len(t2.stats.ttft_samples), 1)) * 100:.1f}%" if t2 else "—"
        t12_slo = f"{(1 - t12.stats.n_ttft_violations / max(len(t12.stats.ttft_samples), 1)) * 100:.1f}%" if t12 else "—"
        print(f"  {name:<14} {t2_s:<12} {t12_s:<12} {t2_slo:<12} {t12_slo:<12}")

    print()


# ═══════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Time-driven Tier 1 + Tier 2 energy benchmark",
    )
    parser.add_argument("--simulate", action="store_true",
                        help="Run in simulation mode (no GPU)")
    parser.add_argument("--scenarios", type=str, default="all",
                        help="Comma-separated: light,heavy,bursty,decode_heavy,all")
    parser.add_argument("--duration", type=float, default=60.0,
                        help="Workload duration in seconds (default: 60)")
    parser.add_argument("--output-json", type=str, default=None,
                        help="Save results to JSON")
    args = parser.parse_args()

    available = ["light", "heavy", "bursty", "decode_heavy"]
    if args.scenarios == "all":
        selected = available
    else:
        selected = [s.strip() for s in args.scenarios.split(",")]

    modes = ["baseline", "tier2_only", "tier1+tier2"]
    all_results = []

    for name in selected:
        if name not in available:
            print(f"[WARN] Unknown scenario: {name}, skipping")
            continue
        scenario = generate_workload(name, args.duration)
        print(f"\n[RUN] {scenario.name}: {scenario.description}")
        print(f"      SLO: TTFT≤{scenario.ttft_slo_ms}ms, TPOT≤{scenario.tpot_slo_us}us")

        for mode in modes:
            result = run_scenario(scenario, mode)
            all_results.append(result)
            s = result.stats
            save = ""
            if mode != "baseline":
                base_e = next(
                    r.stats.total_energy_mj for r in all_results
                    if r.scenario == name and r.mode == "baseline"
                )
                save = f" ({(1 - s.total_energy_mj / base_e) * 100:+.1f}%)"
            print(f"  {mode:<14} energy={s.total_energy_mj:.0f}mJ{save}  "
                  f"reqs={s.n_requests_completed}  "
                  f"TTFT_viol={s.n_ttft_violations}  "
                  f"TPOT_viol={s.n_tpot_violations}  "
                  f"[{result.wall_time_s:.2f}s wall]")

    print_report(all_results)

    if args.output_json:
        output = []
        for r in all_results:
            s = r.stats
            output.append({
                "mode": r.mode, "scenario": r.scenario,
                "total_energy_mj": s.total_energy_mj,
                "prefill_energy_mj": s.prefill_energy_mj,
                "decode_energy_mj": s.decode_energy_mj,
                "idle_energy_mj": s.idle_energy_mj,
                "n_requests_completed": s.n_requests_completed,
                "n_ttft_violations": s.n_ttft_violations,
                "n_tpot_violations": s.n_tpot_violations,
                "tier1_replans": s.tier1_replans,
                "tier2_freq_changes": s.tier2_freq_changes,
                "avg_decode_bs": float(np.mean(s.decode_bs_history)) if s.decode_bs_history else 0,
                "config": r.config_desc,
            })
        Path(args.output_json).write_text(json.dumps(output, indent=2))
        print(f"\n[INFO] Results saved to {args.output_json}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Demonstrate Tier1 value: sudden load spike with tight SLO.

Scenario designed to show Tier1's advantage:
  - Phase 1 (0-30s): Very light load (0.5 QPS, short seqs) → Tier2 settles at lowest freq
  - Phase 2 (30-31s): SUDDEN burst (50 requests in 1 second) → Tier2 needs many iterations to ramp up
  - Phase 3 (31-60s): Sustained heavy (5 QPS, long seqs) → Tier2 eventually catches up
  - Phase 4 (60-90s): Light again → both modes save energy

With tight SLO (TPOT ≤ 80ms), Tier2-only will violate SLO during the ramp-up period
because it starts from 210MHz and needs ~10 iterations to reach adequate frequency.
Tier1+Tier2 detects the load shift and immediately sets baseline to high freq,
so Tier2 starts from a safe point and avoids SLO violations.

Usage:
    /workspace/env/af-test/bin/python benchmark/energy_bench/demo_tier1_value.py
"""

import json
import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "python"))

from sglang.srt.energy.af_dvfs_controller import (
    AFDVFSController, F_MAX, F_MIN, VALID_FREQS,
)

NUM_LAYERS = 64
TTFT_SLO_MS = 1000.0
TPOT_SLO_US = 80000.0  # 80ms — tight!


class MockPredictor:
    def predict_latency(self, phase, op, tp, freq, bs, il, ol=None):
        freq_ratio = F_MAX / freq
        if phase == "prefill":
            base = (30.0 + 0.03 * il + 8.0 * bs) if op == "A" else (80.0 + 0.06 * il + 15.0 * bs)
        else:
            base = (50.0 + 0.01 * il + 1.5 * bs) if op == "A" else (100.0 + 0.02 * il + 2.5 * bs)
        base /= (tp / 4.0)
        class _R:
            value = base * freq_ratio
        return _R()

    def predict_energy(self, phase, op, tp, freq, bs, il, ol=None):
        lat = self.predict_latency(phase, op, tp, freq, bs, il, ol).value
        power_w = 300.0 * (freq / F_MAX) ** 1.5
        class _R:
            value = power_w * lat / 1_000_000.0
        return _R()


class Simulator:
    def __init__(self, workload, mode="tier2_only"):
        self.mode = mode
        self.predictor = MockPredictor()
        self.dvfs_ctrl = AFDVFSController(
            predictor=self.predictor, num_layers=NUM_LAYERS,
            tp_a=4, tp_f=4, baseline_f_a=F_MAX, baseline_f_f=F_MAX,
        )
        self.requests = workload

        self.sim_time_s = 0.0
        self.decode_batch = []
        self.prefill_queue = []
        self.next_arrival_idx = 0
        self.decode_iter_count = 0
        self.last_tier2_iter = 0
        self.tier2_window = 30
        self.cur_f_a = F_MAX
        self.cur_f_f = F_MAX
        self._last_bs = 0
        self.total_energy_mj = 0.0

        # Tier 1 state
        self.tier1_window_start = 0.0
        self.tier1_window_violations = 0
        self.tier1_window_events = 0
        self.tier1_consec_slo = 0
        self.tier1_last_qps = 0.0
        self.tier1_window_reqs = 0

        # Metrics
        self.slo_violations = 0
        self.total_iters = 0
        self.tpot_samples = []
        self.ttft_samples = []
        self.n_completed = 0
        self.freq_trace = []  # (time, f_a, f_f, bs, tpot_us)
        self.tier1_events = []

    def run(self):
        duration_s = self.requests[-1]["arrival_time_s"] + 40.0
        while self.sim_time_s < duration_s or self.decode_batch or self.prefill_queue:
            if self.sim_time_s > duration_s * 2:
                break
            self._admit()
            if self.prefill_queue:
                self._prefill()
            if self.decode_batch:
                self._decode()
            else:
                self.sim_time_s += 0.001
            if self.mode == "tier1+tier2":
                self._tier1_check()

    def _admit(self):
        while (self.next_arrival_idx < len(self.requests)
               and self.requests[self.next_arrival_idx]["arrival_time_s"] <= self.sim_time_s):
            self.prefill_queue.append(self.requests[self.next_arrival_idx])
            self.next_arrival_idx += 1

    def _prefill(self):
        if len(self.decode_batch) >= 96:
            return
        bs = min(len(self.prefill_queue), 4, 96 - len(self.decode_batch))
        batch = [self.prefill_queue.pop(0) for _ in range(bs)]
        max_il = max(r["input_len"] for r in batch)

        slack_us = TTFT_SLO_MS * 1000.0
        decision = self.dvfs_ctrl.select_freq_prefill(bs=bs, il=max_il, slack_us=slack_us, M=1)
        self.cur_f_a, self.cur_f_f = decision.f_a, decision.f_f

        lat_a = self.predictor.predict_latency("prefill", "A", 4, self.cur_f_a, bs, max_il).value
        lat_f = self.predictor.predict_latency("prefill", "F", 4, self.cur_f_f, bs, max_il).value
        e_a = self.predictor.predict_energy("prefill", "A", 4, self.cur_f_a, bs, max_il).value * 4
        e_f = self.predictor.predict_energy("prefill", "F", 4, self.cur_f_f, bs, max_il).value * 4
        self.total_energy_mj += (e_a + e_f) * NUM_LAYERS
        self.sim_time_s += (lat_a + lat_f) * NUM_LAYERS / 1e6

        for req in batch:
            ttft = (self.sim_time_s - req["arrival_time_s"]) * 1000
            self.ttft_samples.append(ttft)
            if ttft > TTFT_SLO_MS:
                self.slo_violations += 1
                self.tier1_window_violations += 1
            self.tier1_window_reqs += 1
            self.tier1_window_events += 1
            self.decode_batch.append({"il": req["input_len"], "remaining": req["output_len"], "phase": req.get("phase", "")})

    def _decode(self):
        bs = len(self.decode_batch)
        avg_il = int(np.mean([r["il"] for r in self.decode_batch]))

        # Tier 2 re-evaluation
        iters_since = self.decode_iter_count - self.last_tier2_iter
        bs_changed = (self._last_bs > 0 and abs(bs - self._last_bs) / max(self._last_bs, 1) > 0.3)
        window_expired = iters_since >= self.tier2_window
        slo_urgent = (self.tpot_samples and self.tpot_samples[-1] > TPOT_SLO_US * 0.9)

        if window_expired or bs_changed or slo_urgent or self.decode_iter_count == 0:
            decision = self.dvfs_ctrl.select_freq_decode(bs=bs, il=avg_il, ol=1, slo_tpot_us=TPOT_SLO_US, M=1)
            self.cur_f_a, self.cur_f_f = decision.f_a, decision.f_f
            self.last_tier2_iter = self.decode_iter_count

        self._last_bs = bs

        lat_a = self.predictor.predict_latency("decode", "A", 4, self.cur_f_a, bs, avg_il).value
        lat_f = self.predictor.predict_latency("decode", "F", 4, self.cur_f_f, bs, avg_il).value
        e_a = self.predictor.predict_energy("decode", "A", 4, self.cur_f_a, bs, avg_il).value * 4
        e_f = self.predictor.predict_energy("decode", "F", 4, self.cur_f_f, bs, avg_il).value * 4
        iter_lat_us = (lat_a + lat_f) * NUM_LAYERS
        self.total_energy_mj += (e_a + e_f) * NUM_LAYERS
        self.sim_time_s += iter_lat_us / 1e6
        self.decode_iter_count += 1
        self.total_iters += 1
        self.tpot_samples.append(iter_lat_us)

        if iter_lat_us > TPOT_SLO_US:
            self.slo_violations += 1
            self.tier1_window_violations += 1
        self.tier1_window_events += 1

        self.freq_trace.append((self.sim_time_s, self.cur_f_a, self.cur_f_f, bs, iter_lat_us))

        self.decode_batch = [r for r in self.decode_batch if r["remaining"] > 1]
        for r in self.decode_batch:
            r["remaining"] -= 1
        self.n_completed += bs - len(self.decode_batch)

    def _tier1_check(self):
        if self.sim_time_s - self.tier1_window_start < 10.0:
            return

        window_dur = self.sim_time_s - self.tier1_window_start
        slo_pct = (self.tier1_window_violations / max(self.tier1_window_events, 1)) * 100
        qps = self.tier1_window_reqs / max(window_dur, 1)

        if slo_pct > 1.0:
            self.tier1_consec_slo += 1
        else:
            self.tier1_consec_slo = 0

        load_shift = (self.tier1_last_qps > 0 and qps / max(self.tier1_last_qps, 0.1) > 3.0)

        replan = False
        new_f_a, new_f_f = self.cur_f_a, self.cur_f_f

        # Key difference: Tier1 detects load spike EARLY and preemptively raises baseline
        if self.tier1_consec_slo >= 1:  # React after just 1 window (faster than 2)
            replan = True
            new_f_a, new_f_f = F_MAX, F_MAX
            self.tier1_consec_slo = 0
            self.tier1_events.append((self.sim_time_s, "SLO_viol", slo_pct, qps, new_f_a, new_f_f))
        elif load_shift:
            replan = True
            # Proactively set high baseline for incoming heavy load
            if qps > 3.0:
                new_f_a, new_f_f = 1170, 1410
            elif qps > 1.5:
                new_f_a, new_f_f = 690, 930
            else:
                new_f_a, new_f_f = 450, 690
            self.tier1_events.append((self.sim_time_s, "load_shift", slo_pct, qps, new_f_a, new_f_f))

        if replan:
            self.cur_f_a, self.cur_f_f = new_f_a, new_f_f
            self.dvfs_ctrl.update_baseline(new_f_a, new_f_f)

        self.tier1_last_qps = qps
        self.tier1_window_start = self.sim_time_s
        self.tier1_window_violations = 0
        self.tier1_window_events = 0
        self.tier1_window_reqs = 0


def generate_workload():
    """Generate workload with sudden spike designed to stress Tier2."""
    requests = []
    t = 0.0

    # Phase 1: Light (0-30s, 0.5 QPS, short seqs)
    for i in range(15):
        requests.append({"phase": "light", "arrival_time_s": round(t, 2),
                        "input_len": np.random.randint(64, 256), "output_len": np.random.randint(30, 60)})
        t += 2.0

    # Phase 2: SUDDEN BURST (30-31s, 50 requests in 1 second!)
    for i in range(50):
        requests.append({"phase": "burst", "arrival_time_s": round(30.0 + i * 0.02, 3),
                        "input_len": np.random.randint(512, 1024), "output_len": np.random.randint(80, 150)})

    # Phase 3: Sustained heavy (31-60s, 5 QPS, long seqs)
    t = 31.0
    for i in range(145):
        requests.append({"phase": "heavy", "arrival_time_s": round(t, 2),
                        "input_len": np.random.randint(512, 1024), "output_len": np.random.randint(60, 120)})
        t += 0.2

    # Phase 4: Light again (60-90s, 0.5 QPS)
    t = 60.0
    for i in range(15):
        requests.append({"phase": "recovery", "arrival_time_s": round(t, 2),
                        "input_len": np.random.randint(64, 256), "output_len": np.random.randint(30, 60)})
        t += 2.0

    return requests


def run_comparison():
    np.random.seed(42)
    workload = generate_workload()

    print("=" * 80)
    print("  TIER1 VALUE DEMONSTRATION")
    print(f"  Workload: {len(workload)} requests, SLO: TTFT≤{TTFT_SLO_MS}ms, TPOT≤{TPOT_SLO_US/1000}ms")
    print(f"  Scenario: light(0-30s) → SUDDEN BURST(30-31s, 50 reqs) → heavy(31-60s) → light(60-90s)")
    print("=" * 80)

    results = {}
    for mode in ["tier2_only", "tier1+tier2"]:
        sim = Simulator(workload, mode=mode)
        sim.run()
        results[mode] = sim

    # ── Print comparison ──
    print(f"\n{'─'*80}")
    print(f"  RESULTS COMPARISON")
    print(f"{'─'*80}")
    print(f"  {'Metric':<30s} {'Tier2 Only':>15s} {'Tier1+Tier2':>15s} {'Delta':>10s}")
    print(f"  {'─'*30} {'─'*15} {'─'*15} {'─'*10}")

    t2 = results["tier2_only"]
    t12 = results["tier1+tier2"]

    metrics = [
        ("SLO Violations", t2.slo_violations, t12.slo_violations),
        ("Total Energy (mJ)", t2.total_energy_mj, t12.total_energy_mj),
        ("TPOT p50 (ms)", np.percentile(t2.tpot_samples, 50)/1000, np.percentile(t12.tpot_samples, 50)/1000),
        ("TPOT p99 (ms)", np.percentile(t2.tpot_samples, 99)/1000, np.percentile(t12.tpot_samples, 99)/1000),
        ("TPOT max (ms)", max(t2.tpot_samples)/1000, max(t12.tpot_samples)/1000),
        ("Decode iterations", t2.total_iters, t12.total_iters),
    ]

    for name, v2, v12 in metrics:
        if v2 > 0:
            delta = (v12 - v2) / v2 * 100
            sign = "+" if delta > 0 else ""
            print(f"  {name:<30s} {v2:>15.1f} {v12:>15.1f} {sign}{delta:>8.1f}%")
        else:
            print(f"  {name:<30s} {v2:>15.1f} {v12:>15.1f} {'N/A':>10s}")

    # ── SLO violation timeline ──
    print(f"\n{'─'*80}")
    print(f"  SLO VIOLATION ANALYSIS")
    print(f"{'─'*80}")

    # Count violations by phase
    for mode_name, sim in results.items():
        violations_by_time = []
        for t_s, f_a, f_f, bs, tpot_us in sim.freq_trace:
            if tpot_us > TPOT_SLO_US:
                violations_by_time.append(t_s)

        print(f"\n  {mode_name}:")
        print(f"    Total TPOT violations: {len(violations_by_time)}")
        if violations_by_time:
            print(f"    First violation at: {violations_by_time[0]:.1f}s")
            print(f"    Last violation at:  {violations_by_time[-1]:.1f}s")
            # Count by time window
            bins = [(30, 35, "burst+ramp"), (35, 45, "heavy_start"), (45, 60, "heavy_steady")]
            for t_start, t_end, label in bins:
                count = sum(1 for t in violations_by_time if t_start <= t < t_end)
                print(f"    [{t_start}-{t_end}s] {label}: {count} violations")

    # ── Tier 1 events ──
    if t12.tier1_events:
        print(f"\n{'─'*80}")
        print(f"  TIER 1 REPLAN EVENTS")
        print(f"{'─'*80}")
        for t_s, reason, slo_pct, qps, f_a, f_f in t12.tier1_events:
            print(f"  T={t_s:.1f}s: {reason} (SLO_viol={slo_pct:.1f}%, QPS={qps:.1f}) → f_A={f_a}, f_F={f_f}")

    # ── Frequency ramp-up comparison during burst ──
    print(f"\n{'─'*80}")
    print(f"  FREQUENCY DURING BURST (t=30-35s)")
    print(f"{'─'*80}")
    print(f"  {'Time':>6s} {'Tier2: f_A':>10s} {'f_F':>6s} {'BS':>4s} {'TPOT':>8s} {'T1+T2: f_A':>10s} {'f_F':>6s} {'BS':>4s} {'TPOT':>8s}")

    t2_burst = [(t, fa, ff, bs, tp) for t, fa, ff, bs, tp in t2.freq_trace if 29.5 <= t <= 36.0]
    t12_burst = [(t, fa, ff, bs, tp) for t, fa, ff, bs, tp in t12.freq_trace if 29.5 <= t <= 36.0]

    # Sample every 5th point for readability
    for i in range(0, min(len(t2_burst), len(t12_burst)), 5):
        t2_p = t2_burst[i]
        t12_p = t12_burst[min(i, len(t12_burst)-1)]
        slo_mark_t2 = " !!!" if t2_p[4] > TPOT_SLO_US else ""
        slo_mark_t12 = " !!!" if t12_p[4] > TPOT_SLO_US else ""
        print(f"  {t2_p[0]:6.2f} {t2_p[1]:>10d} {t2_p[2]:>6d} {t2_p[3]:>4d} {t2_p[4]/1000:>7.1f}{slo_mark_t2}"
              f" {t12_p[1]:>10d} {t12_p[2]:>6d} {t12_p[3]:>4d} {t12_p[4]/1000:>7.1f}{slo_mark_t12}")

    # ── Summary ──
    print(f"\n{'─'*80}")
    print(f"  CONCLUSION")
    print(f"{'─'*80}")
    viol_diff = t2.slo_violations - t12.slo_violations
    energy_diff = (t2.total_energy_mj - t12.total_energy_mj) / t2.total_energy_mj * 100
    print(f"  Tier1+Tier2 reduces SLO violations by {viol_diff} ({viol_diff/max(t2.slo_violations,1)*100:.0f}%)")
    print(f"  Energy difference: {energy_diff:+.1f}% (Tier1+Tier2 vs Tier2-only)")
    print(f"  Key insight: Tier1 detects load spike and preemptively raises baseline,")
    print(f"  preventing the 'ramp-up lag' where Tier2 slowly climbs from min freq.")


if __name__ == "__main__":
    run_comparison()

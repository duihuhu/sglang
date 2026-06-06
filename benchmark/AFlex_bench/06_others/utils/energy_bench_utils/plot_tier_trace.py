#!/usr/bin/env python3
"""Visualize Tier1+Tier2 decision trace with detailed plots.

Generates a multi-panel figure showing:
  1. Workload arrival rate (QPS) over time with phase labels
  2. Decode batch size over time
  3. Tier 2 frequency decisions (f_A and f_F) with trigger annotations
  4. TPOT vs SLO over time
  5. Tier 1 monitoring windows with replan markers
  6. Cumulative energy consumption comparison

Usage:
    /workspace/env/af-test/bin/python benchmark/energy_bench/plot_tier_trace.py
"""

import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "python"))

from sglang.srt.energy.af_dvfs_controller import (
    AFDVFSController, F_MAX, F_MIN, VALID_FREQS,
)

NUM_LAYERS = 64
TTFT_SLO_MS = 2000.0
TPOT_SLO_US = 120000.0


class MockAFProfilePredictor:
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


class TraceCollector:
    """Run simulation and collect time-series data for plotting."""

    def __init__(self, workload_path: str, mode: str = "tier1+tier2"):
        self.mode = mode
        self.predictor = MockAFProfilePredictor()
        self.dvfs_ctrl = AFDVFSController(
            predictor=self.predictor, num_layers=NUM_LAYERS,
            tp_a=4, tp_f=4, baseline_f_a=F_MAX, baseline_f_f=F_MAX,
        ) if mode != "baseline" else None

        with open(workload_path) as f:
            self.requests = [json.loads(line) for line in f]

        # Time series data
        self.ts_time = []
        self.ts_bs = []
        self.ts_f_a = []
        self.ts_f_f = []
        self.ts_tpot_us = []
        self.ts_energy_cumul = []
        self.ts_phase = []

        # Tier 2 decisions
        self.t2_times = []
        self.t2_triggers = []
        self.t2_f_a = []
        self.t2_f_f = []

        # Tier 1 windows
        self.t1_times = []
        self.t1_qps = []
        self.t1_slo_pct = []
        self.t1_replan = []
        self.t1_new_f_a = []
        self.t1_new_f_f = []

        # State
        self.sim_time_s = 0.0
        self.decode_batch = []
        self.prefill_queue = []
        self.next_arrival_idx = 0
        self.decode_iter_count = 0
        self.last_tier2_decision_iter = 0
        self.tier2_window_size = 30
        self.cur_f_a = F_MAX if mode == "baseline" else F_MAX
        self.cur_f_f = F_MAX if mode == "baseline" else F_MAX
        self._last_decode_bs = 0
        self.total_energy_mj = 0.0

        # Tier 1
        self.tier1_window_start = 0.0
        self.tier1_window_reqs = 0
        self.tier1_window_violations = 0
        self.tier1_window_tpots = []
        self.tier1_last_qps = 0.0
        self.tier1_consec_slo = 0
        self.tier1_consec_load = 0

    def run(self):
        duration_s = self.requests[-1]["arrival_time_s"] + 30.0
        while self.sim_time_s < duration_s or self.decode_batch or self.prefill_queue:
            if self.sim_time_s > duration_s * 2:
                break
            self._admit_arrivals()
            if self.prefill_queue:
                self._process_prefill()
            if self.decode_batch:
                self._run_decode_iteration()
            else:
                self.sim_time_s += 0.001
            if self.mode == "tier1+tier2":
                self._tier1_check()

    def _admit_arrivals(self):
        while (self.next_arrival_idx < len(self.requests)
               and self.requests[self.next_arrival_idx]["arrival_time_s"] <= self.sim_time_s):
            self.prefill_queue.append(self.requests[self.next_arrival_idx])
            self.next_arrival_idx += 1

    def _process_prefill(self):
        if len(self.decode_batch) >= 64:
            return
        batch_size = min(len(self.prefill_queue), 4, 64 - len(self.decode_batch))
        batch = [self.prefill_queue.pop(0) for _ in range(batch_size)]
        max_il = max(r["input_len"] for r in batch)

        if self.dvfs_ctrl and self.mode != "baseline":
            slack_us = TTFT_SLO_MS * 1000.0
            decision = self.dvfs_ctrl.select_freq_prefill(bs=batch_size, il=max_il, slack_us=slack_us, M=1)
            f_a, f_f = decision.f_a, decision.f_f
            self.cur_f_a, self.cur_f_f = f_a, f_f
        else:
            f_a, f_f = self.cur_f_a, self.cur_f_f

        lat_a = self.predictor.predict_latency("prefill", "A", 4, f_a, batch_size, max_il).value
        lat_f = self.predictor.predict_latency("prefill", "F", 4, f_f, batch_size, max_il).value
        e_a = self.predictor.predict_energy("prefill", "A", 4, f_a, batch_size, max_il).value * 4
        e_f = self.predictor.predict_energy("prefill", "F", 4, f_f, batch_size, max_il).value * 4
        prefill_energy = (e_a + e_f) * NUM_LAYERS
        self.total_energy_mj += prefill_energy
        self.sim_time_s += (lat_a + lat_f) * NUM_LAYERS / 1_000_000.0

        for req in batch:
            self.decode_batch.append({"il": req["input_len"], "remaining": req["output_len"], "phase": req.get("phase", "")})
            self.tier1_window_reqs += 1

    def _run_decode_iteration(self):
        bs = len(self.decode_batch)
        avg_il = int(np.mean([r["il"] for r in self.decode_batch]))
        phase = self.decode_batch[0]["phase"] if self.decode_batch else ""

        # Tier 2 decode
        if self.dvfs_ctrl and self.mode != "baseline":
            iters_since = self.decode_iter_count - self.last_tier2_decision_iter
            bs_changed = (self._last_decode_bs > 0 and abs(bs - self._last_decode_bs) / max(self._last_decode_bs, 1) > 0.3)
            window_expired = iters_since >= self.tier2_window_size
            slo_urgent = (self.ts_tpot_us and self.ts_tpot_us[-1] > TPOT_SLO_US * 0.9)

            trigger = None
            if window_expired: trigger = "window"
            elif bs_changed: trigger = "bs_chg"
            elif slo_urgent: trigger = "slo_urg"
            elif self.decode_iter_count == 0: trigger = "first"

            if trigger:
                decision = self.dvfs_ctrl.select_freq_decode(bs=bs, il=avg_il, ol=1, slo_tpot_us=TPOT_SLO_US, M=1)
                self.cur_f_a, self.cur_f_f = decision.f_a, decision.f_f
                self.last_tier2_decision_iter = self.decode_iter_count
                self.t2_times.append(self.sim_time_s)
                self.t2_triggers.append(trigger)
                self.t2_f_a.append(decision.f_a)
                self.t2_f_f.append(decision.f_f)

        self._last_decode_bs = bs

        lat_a = self.predictor.predict_latency("decode", "A", 4, self.cur_f_a, bs, avg_il).value
        lat_f = self.predictor.predict_latency("decode", "F", 4, self.cur_f_f, bs, avg_il).value
        e_a = self.predictor.predict_energy("decode", "A", 4, self.cur_f_a, bs, avg_il).value * 4
        e_f = self.predictor.predict_energy("decode", "F", 4, self.cur_f_f, bs, avg_il).value * 4
        iter_lat_us = (lat_a + lat_f) * NUM_LAYERS
        iter_energy = (e_a + e_f) * NUM_LAYERS

        if iter_lat_us > TPOT_SLO_US:
            self.tier1_window_violations += 1

        self.total_energy_mj += iter_energy
        self.sim_time_s += iter_lat_us / 1_000_000.0
        self.decode_iter_count += 1
        self.tier1_window_tpots.append(iter_lat_us)

        # Record time series
        self.ts_time.append(self.sim_time_s)
        self.ts_bs.append(bs)
        self.ts_f_a.append(self.cur_f_a)
        self.ts_f_f.append(self.cur_f_f)
        self.ts_tpot_us.append(iter_lat_us)
        self.ts_energy_cumul.append(self.total_energy_mj)
        self.ts_phase.append(phase)

        # Remove completed
        self.decode_batch = [r for r in self.decode_batch if r["remaining"] > 1]
        for r in self.decode_batch:
            r["remaining"] -= 1

    def _tier1_check(self):
        if self.sim_time_s - self.tier1_window_start < 15.0:
            return

        window_dur = self.sim_time_s - self.tier1_window_start
        total_events = self.tier1_window_reqs + len(self.tier1_window_tpots)
        slo_viol_pct = (self.tier1_window_violations / max(total_events, 1)) * 100
        qps = self.tier1_window_reqs / max(window_dur, 1)

        # Check triggers
        if slo_viol_pct > 1.0:
            self.tier1_consec_slo += 1
        else:
            self.tier1_consec_slo = 0

        load_shifted = (self.tier1_last_qps > 0 and abs(qps - self.tier1_last_qps) / max(self.tier1_last_qps, 0.1) > 0.5)
        if load_shifted:
            self.tier1_consec_load += 1
        else:
            self.tier1_consec_load = 0

        replan = False
        new_f_a, new_f_f = self.cur_f_a, self.cur_f_f
        if self.tier1_consec_slo >= 2:
            replan = True
            new_f_a, new_f_f = F_MAX, F_MAX
            self.tier1_consec_slo = 0
        elif self.tier1_consec_load >= 2:
            replan = True
            if qps < 2.0: new_f_a, new_f_f = 450, 690
            elif qps < 4.0: new_f_a, new_f_f = 690, 930
            else: new_f_a, new_f_f = 1410, 1410
            self.tier1_consec_load = 0

        self.t1_times.append(self.sim_time_s)
        self.t1_qps.append(qps)
        self.t1_slo_pct.append(slo_viol_pct)
        self.t1_replan.append(replan)
        self.t1_new_f_a.append(new_f_a if replan else 0)
        self.t1_new_f_f.append(new_f_f if replan else 0)

        if replan:
            self.cur_f_a, self.cur_f_f = new_f_a, new_f_f
            if self.dvfs_ctrl:
                self.dvfs_ctrl.update_baseline(new_f_a, new_f_f)

        self.tier1_last_qps = qps
        self.tier1_window_start = self.sim_time_s
        self.tier1_window_reqs = 0
        self.tier1_window_violations = 0
        self.tier1_window_tpots = []


def plot_trace(tier12: TraceCollector, baseline: TraceCollector, output_path: str):
    """Generate multi-panel visualization."""
    fig, axes = plt.subplots(6, 1, figsize=(16, 20), sharex=True)
    fig.suptitle("AFlex Tier1+Tier2 Execution Trace (workload_varying.jsonl)\n"
                 f"SLO: TTFT≤{TTFT_SLO_MS:.0f}ms, TPOT≤{TPOT_SLO_US/1000:.0f}ms",
                 fontsize=14, fontweight='bold')

    t = np.array(tier12.ts_time)
    t_base = np.array(baseline.ts_time)

    # Phase background colors
    phase_colors = {"light": "#E8F5E9", "heavy": "#FFEBEE", "medium": "#FFF3E0",
                    "burst_peak": "#FCE4EC", "burst_quiet": "#E3F2FD"}

    def add_phase_bg(ax):
        phases_seen = set()
        if not tier12.ts_phase:
            return
        i = 0
        while i < len(tier12.ts_phase):
            phase = tier12.ts_phase[i]
            if not phase:
                i += 1
                continue
            start_t = tier12.ts_time[i]
            j = i
            while j < len(tier12.ts_phase) and tier12.ts_phase[j] == phase:
                j += 1
            end_t = tier12.ts_time[min(j, len(tier12.ts_time)-1)]
            color = phase_colors.get(phase, "#F5F5F5")
            if phase not in phases_seen:
                ax.axvspan(start_t, end_t, alpha=0.3, color=color, label=phase)
                phases_seen.add(phase)
            else:
                ax.axvspan(start_t, end_t, alpha=0.3, color=color)
            i = j

    # ── Panel 1: Workload QPS ──
    ax = axes[0]
    add_phase_bg(ax)
    # Compute QPS from request arrivals
    arrivals = [r["arrival_time_s"] for r in tier12.requests]
    qps_times = np.arange(0, max(arrivals)+1, 1.0)
    qps_vals = []
    for t_start in qps_times:
        count = sum(1 for a in arrivals if t_start <= a < t_start + 5.0)
        qps_vals.append(count / 5.0)
    ax.bar(qps_times, qps_vals, width=1.0, color='steelblue', alpha=0.7)
    ax.set_ylabel("QPS (5s window)")
    ax.set_title("Panel 1: Workload Arrival Rate")
    ax.legend(loc='upper right', fontsize=8)
    ax.set_ylim(0, max(qps_vals) * 1.2)

    # ── Panel 2: Decode Batch Size ──
    ax = axes[1]
    add_phase_bg(ax)
    ax.plot(t, tier12.ts_bs, color='darkorange', linewidth=0.8, alpha=0.8)
    ax.axhline(64, color='red', linestyle='--', alpha=0.5, label='max_bs=64')
    ax.set_ylabel("Decode BS")
    ax.set_title("Panel 2: Decode Batch Size")
    ax.legend(loc='upper right', fontsize=8)

    # ── Panel 3: Tier 2 Frequency Decisions ──
    ax = axes[2]
    add_phase_bg(ax)
    ax.step(t, tier12.ts_f_a, where='post', color='blue', linewidth=1.2, label='f_A (Attn)')
    ax.step(t, tier12.ts_f_f, where='post', color='red', linewidth=1.2, label='f_F (FFN)')
    ax.axhline(F_MAX, color='gray', linestyle=':', alpha=0.4, label=f'F_MAX={F_MAX}')
    ax.axhline(F_MIN, color='gray', linestyle=':', alpha=0.4)

    # Mark Tier 1 replans
    for i, (tt, replan) in enumerate(zip(tier12.t1_times, tier12.t1_replan)):
        if replan:
            ax.axvline(tt, color='green', linewidth=2, linestyle='--', alpha=0.8)
            ax.annotate(f'T1: →{tier12.t1_new_f_a[i]}/{tier12.t1_new_f_f[i]}',
                       xy=(tt, F_MAX-50), fontsize=7, color='green', fontweight='bold',
                       rotation=90, va='top')

    # Mark trigger types
    trigger_colors = {"slo_urg": "red", "bs_chg": "orange", "window": "gray", "first": "black"}
    for tt, trig in zip(tier12.t2_times, tier12.t2_triggers):
        if trig == "slo_urg":
            ax.axvline(tt, color='red', linewidth=0.3, alpha=0.3)

    ax.set_ylabel("Frequency (MHz)")
    ax.set_title("Panel 3: Tier 2 Frequency (f_A, f_F) + Tier 1 Replan Events")
    ax.set_yticks(VALID_FREQS)
    ax.legend(loc='upper right', fontsize=8)

    # ── Panel 4: TPOT vs SLO ──
    ax = axes[3]
    add_phase_bg(ax)
    tpot_ms = np.array(tier12.ts_tpot_us) / 1000.0
    tpot_base_ms = np.array(baseline.ts_tpot_us) / 1000.0
    ax.plot(t, tpot_ms, color='purple', linewidth=0.6, alpha=0.7, label='Tier1+Tier2')
    ax.plot(t_base[:len(tpot_base_ms)], tpot_base_ms, color='gray', linewidth=0.4, alpha=0.5, label='Baseline')
    ax.axhline(TPOT_SLO_US/1000, color='red', linestyle='--', linewidth=1.5, label=f'SLO={TPOT_SLO_US/1000:.0f}ms')
    ax.axhline(TPOT_SLO_US*0.9/1000, color='orange', linestyle=':', alpha=0.6, label='90% SLO (urgent)')
    ax.set_ylabel("TPOT (ms)")
    ax.set_title("Panel 4: Per-Iteration TPOT vs SLO")
    ax.legend(loc='upper right', fontsize=8)

    # ── Panel 5: Tier 1 Monitoring ──
    ax = axes[4]
    ax2 = ax.twinx()
    bars = ax.bar(tier12.t1_times, tier12.t1_slo_pct, width=3.0, color='salmon', alpha=0.7, label='SLO viol %')
    ax.axhline(1.0, color='red', linestyle='--', alpha=0.6, label='Threshold (1%)')
    ax2.plot(tier12.t1_times, tier12.t1_qps, 'b-o', markersize=4, label='QPS')

    for i, (tt, replan) in enumerate(zip(tier12.t1_times, tier12.t1_replan)):
        if replan:
            ax.axvline(tt, color='green', linewidth=2.5, linestyle='--', alpha=0.8)
            ax.annotate('REPLAN', xy=(tt, max(tier12.t1_slo_pct)*0.9),
                       fontsize=8, color='green', fontweight='bold', ha='center')

    ax.set_ylabel("SLO Violation %", color='red')
    ax2.set_ylabel("QPS", color='blue')
    ax.set_title("Panel 5: Tier 1 Monitoring Windows (every 15s)")
    ax.legend(loc='upper left', fontsize=8)
    ax2.legend(loc='upper right', fontsize=8)

    # ── Panel 6: Cumulative Energy ──
    ax = axes[5]
    add_phase_bg(ax)
    ax.plot(t, np.array(tier12.ts_energy_cumul)/1000, color='green', linewidth=1.5, label='Tier1+Tier2')
    ax.plot(t_base[:len(baseline.ts_energy_cumul)], np.array(baseline.ts_energy_cumul)/1000,
            color='gray', linewidth=1.5, linestyle='--', label='Baseline (max freq)')
    ax.set_ylabel("Cumulative Energy (J)")
    ax.set_xlabel("Time (s)")
    ax.set_title("Panel 6: Cumulative Energy Consumption")
    ax.legend(loc='upper left', fontsize=8)

    # Final energy saving annotation
    if tier12.ts_energy_cumul and baseline.ts_energy_cumul:
        e_t12 = tier12.ts_energy_cumul[-1]
        e_base = baseline.ts_energy_cumul[-1]
        saving = (e_base - e_t12) / e_base * 100
        ax.annotate(f'Energy saving: {saving:.1f}%',
                   xy=(t[-1]*0.7, e_base/1000*0.5), fontsize=11, color='green', fontweight='bold')

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--workload", type=str,
                        default=str(Path(__file__).parent / "workloads" / "workload_varying.jsonl"))
    parser.add_argument("--output", type=str, default=str(Path(__file__).parent / "results" / "tier_trace.png"))
    args = parser.parse_args()

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)

    print("Running Tier1+Tier2 simulation...")
    tier12 = TraceCollector(args.workload, mode="tier1+tier2")
    tier12.run()
    print(f"  Done: {tier12.decode_iter_count} decode iters, energy={tier12.total_energy_mj:.0f}mJ")

    print("Running Baseline simulation...")
    baseline = TraceCollector(args.workload, mode="baseline")
    baseline.run()
    print(f"  Done: {baseline.decode_iter_count} decode iters, energy={baseline.total_energy_mj:.0f}mJ")

    print("Generating plot...")
    plot_trace(tier12, baseline, args.output)

    # Print summary
    saving = (baseline.total_energy_mj - tier12.total_energy_mj) / baseline.total_energy_mj * 100
    print(f"\n{'='*60}")
    print(f"  SUMMARY")
    print(f"{'='*60}")
    print(f"  Baseline energy:   {baseline.total_energy_mj/1000:.1f} J")
    print(f"  Tier1+Tier2 energy:{tier12.total_energy_mj/1000:.1f} J")
    print(f"  Energy saving:     {saving:.1f}%")
    print(f"  Tier 1 replans:    {sum(tier12.t1_replan)}")
    print(f"  Tier 2 decisions:  {len(tier12.t2_times)}")


if __name__ == "__main__":
    main()

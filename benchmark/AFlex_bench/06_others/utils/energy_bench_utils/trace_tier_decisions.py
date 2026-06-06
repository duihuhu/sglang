#!/usr/bin/env python3
"""Detailed Tier1+Tier2 decision trace — shows every frequency decision with context.

Runs the time-driven simulation on workload_varying.jsonl and prints a detailed
log of every Tier 2 frequency decision and every Tier 1 monitoring window,
including the runtime state that triggered each decision.

Usage:
    /workspace/env/af-test/bin/python benchmark/energy_bench/trace_tier_decisions.py
    /workspace/env/af-test/bin/python benchmark/energy_bench/trace_tier_decisions.py --workload workloads/workload_steady.jsonl
"""

import json
import sys
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
    REEVAL_NONE,
    REEVAL_WINDOW_EXPIRED,
    REEVAL_BS_CHANGE,
    REEVAL_SLO_URGENT,
)
from sglang.srt.energy.af_profile_predictor import AFProfilePredictor

NUM_LAYERS = 64
TTFT_SLO_MS = 2000.0
TPOT_SLO_US = 120000.0


# ═══════════════════════════════════════════════════════════════════════
# Mock Predictor
# ═══════════════════════════════════════════════════════════════════════

class MockAFProfilePredictor:
    """Deterministic predictor calibrated to A800 profiling."""

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


# ═══════════════════════════════════════════════════════════════════════
# Tier 1 Monitor (simplified)
# ═══════════════════════════════════════════════════════════════════════

@dataclass
class Tier1MonitorState:
    window_start_s: float = 0.0
    window_reqs: int = 0
    window_slo_violations: int = 0
    window_tpot_samples: list = field(default_factory=list)
    window_ttft_samples: list = field(default_factory=list)
    consecutive_slo_windows: int = 0
    consecutive_load_shift_windows: int = 0
    last_qps: float = 0.0
    last_avg_il: float = 0.0


@dataclass
class Tier1Decision:
    time_s: float
    reason: str
    old_f_a: int
    old_f_f: int
    new_f_a: int
    new_f_f: int
    qps: float
    avg_il: float
    slo_violation_pct: float
    active_bs: int


# ═══════════════════════════════════════════════════════════════════════
# Tier 2 Decision Log
# ═══════════════════════════════════════════════════════════════════════

@dataclass
class Tier2Decision:
    time_s: float
    phase_type: str  # "prefill" or "decode"
    trigger: str     # "window_expired", "bs_change", "slo_urgent", "new_batch"
    bs: int
    avg_il: int
    old_f_a: int
    old_f_f: int
    new_f_a: int
    new_f_f: int
    switched: bool
    tpot_us: float
    slo_pct: float   # current TPOT / SLO
    energy_mj: float
    latency_us: float


# ═══════════════════════════════════════════════════════════════════════
# Simulation with detailed tracing
# ═══════════════════════════════════════════════════════════════════════

class TracingSimulator:
    def __init__(self, workload_path: str, mode: str = "tier1+tier2"):
        self.mode = mode
        self.predictor = MockAFProfilePredictor()
        self.dvfs_ctrl = AFDVFSController(
            predictor=self.predictor, num_layers=NUM_LAYERS,
            tp_a=4, tp_f=4,
            baseline_f_a=F_MAX, baseline_f_f=F_MAX,
        )

        # Load workload
        with open(workload_path) as f:
            self.requests = [json.loads(line) for line in f]

        # State
        self.sim_time_s = 0.0
        self.decode_batch = []  # list of (rid, il, tokens_remaining, tokens_generated)
        self.prefill_queue = []
        self.next_arrival_idx = 0
        self.decode_iter_count = 0
        self.last_tier2_decision_iter = 0
        self.tier2_window_size = 30
        self.cur_f_a = F_MAX
        self.cur_f_f = F_MAX
        self._last_decode_bs = 0

        # Tier 1 state
        self.tier1_state = Tier1MonitorState()
        self.tier1_interval_s = 15.0

        # Logs
        self.tier2_decisions: list[Tier2Decision] = []
        self.tier1_decisions: list[Tier1Decision] = []
        self.tier1_windows: list[dict] = []

        # Stats
        self.total_energy_mj = 0.0
        self.tpot_samples = []
        self.ttft_samples = []
        self.n_completed = 0
        self.n_slo_violations = 0

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

        # Tier 2 prefill decision
        old_f_a, old_f_f = self.cur_f_a, self.cur_f_f
        slack_us = TTFT_SLO_MS * 1000.0
        decision = self.dvfs_ctrl.select_freq_prefill(
            bs=batch_size, il=max_il, slack_us=slack_us, M=1,
        )
        new_f_a, new_f_f = decision.f_a, decision.f_f
        switched = (new_f_a != self.cur_f_a or new_f_f != self.cur_f_f)
        self.cur_f_a, self.cur_f_f = new_f_a, new_f_f

        # Compute latency
        lat_a = self.predictor.predict_latency("prefill", "A", 4, new_f_a, batch_size, max_il).value
        lat_f = self.predictor.predict_latency("prefill", "F", 4, new_f_f, batch_size, max_il).value
        e_a = self.predictor.predict_energy("prefill", "A", 4, new_f_a, batch_size, max_il).value * 4
        e_f = self.predictor.predict_energy("prefill", "F", 4, new_f_f, batch_size, max_il).value * 4
        prefill_lat_us = (lat_a + lat_f) * NUM_LAYERS
        prefill_energy = (e_a + e_f) * NUM_LAYERS
        prefill_lat_s = prefill_lat_us / 1_000_000.0

        self.tier2_decisions.append(Tier2Decision(
            time_s=self.sim_time_s,
            phase_type="prefill",
            trigger="new_batch",
            bs=batch_size,
            avg_il=max_il,
            old_f_a=old_f_a, old_f_f=old_f_f,
            new_f_a=new_f_a, new_f_f=new_f_f,
            switched=switched,
            tpot_us=0,
            slo_pct=prefill_lat_us / (TTFT_SLO_MS * 1000) * 100,
            energy_mj=prefill_energy,
            latency_us=prefill_lat_us,
        ))

        self.sim_time_s += prefill_lat_s
        self.total_energy_mj += prefill_energy

        for req in batch:
            ttft_ms = (self.sim_time_s - req["arrival_time_s"]) * 1000
            self.ttft_samples.append(ttft_ms)
            self.tier1_state.window_ttft_samples.append(ttft_ms)
            self.tier1_state.window_reqs += 1
            if ttft_ms > TTFT_SLO_MS:
                self.n_slo_violations += 1
                self.tier1_state.window_slo_violations += 1
            self.decode_batch.append({
                "rid": f"r{self.n_completed + len(self.decode_batch)}",
                "il": req["input_len"],
                "remaining": req["output_len"],
                "generated": 0,
            })

    def _run_decode_iteration(self):
        bs = len(self.decode_batch)
        avg_il = int(np.mean([r["il"] for r in self.decode_batch]))

        # Tier 2 decode re-evaluation
        old_f_a, old_f_f = self.cur_f_a, self.cur_f_f
        iters_since = self.decode_iter_count - self.last_tier2_decision_iter
        bs_changed = (self._last_decode_bs > 0
                      and abs(bs - self._last_decode_bs) / max(self._last_decode_bs, 1) > 0.3)
        window_expired = iters_since >= self.tier2_window_size
        slo_urgent = (self.tpot_samples
                      and self.tpot_samples[-1] > TPOT_SLO_US * 0.9)

        trigger = None
        if window_expired:
            trigger = "window_expired"
        elif bs_changed:
            trigger = "bs_change"
        elif slo_urgent:
            trigger = "slo_urgent"
        elif self.decode_iter_count == 0:
            trigger = "first_iter"

        if trigger:
            decision = self.dvfs_ctrl.select_freq_decode(
                bs=bs, il=avg_il, ol=1,
                slo_tpot_us=TPOT_SLO_US, M=1,
            )
            new_f_a, new_f_f = decision.f_a, decision.f_f
            switched = (new_f_a != self.cur_f_a or new_f_f != self.cur_f_f)
            self.cur_f_a, self.cur_f_f = new_f_a, new_f_f
            self.last_tier2_decision_iter = self.decode_iter_count

            # Compute this iteration's latency for the log
            lat_a = self.predictor.predict_latency("decode", "A", 4, new_f_a, bs, avg_il).value
            lat_f = self.predictor.predict_latency("decode", "F", 4, new_f_f, bs, avg_il).value
            iter_lat_us = (lat_a + lat_f) * NUM_LAYERS

            self.tier2_decisions.append(Tier2Decision(
                time_s=self.sim_time_s,
                phase_type="decode",
                trigger=trigger,
                bs=bs,
                avg_il=avg_il,
                old_f_a=old_f_a, old_f_f=old_f_f,
                new_f_a=new_f_a, new_f_f=new_f_f,
                switched=switched,
                tpot_us=self.tpot_samples[-1] if self.tpot_samples else 0,
                slo_pct=iter_lat_us / TPOT_SLO_US * 100,
                energy_mj=decision.energy_mj,
                latency_us=decision.latency_us,
            ))

        self._last_decode_bs = bs

        # Compute iteration
        lat_a = self.predictor.predict_latency("decode", "A", 4, self.cur_f_a, bs, avg_il).value
        lat_f = self.predictor.predict_latency("decode", "F", 4, self.cur_f_f, bs, avg_il).value
        e_a = self.predictor.predict_energy("decode", "A", 4, self.cur_f_a, bs, avg_il).value * 4
        e_f = self.predictor.predict_energy("decode", "F", 4, self.cur_f_f, bs, avg_il).value * 4
        iter_lat_us = (lat_a + lat_f) * NUM_LAYERS
        iter_energy = (e_a + e_f) * NUM_LAYERS
        iter_lat_s = iter_lat_us / 1_000_000.0

        self.tpot_samples.append(iter_lat_us)
        self.tier1_state.window_tpot_samples.append(iter_lat_us)
        if iter_lat_us > TPOT_SLO_US:
            self.n_slo_violations += 1
            self.tier1_state.window_slo_violations += 1

        self.sim_time_s += iter_lat_s
        self.total_energy_mj += iter_energy
        self.decode_iter_count += 1

        # Remove completed
        completed = [r for r in self.decode_batch if r["remaining"] <= 1]
        for r in completed:
            self.decode_batch.remove(r)
            self.n_completed += 1
        for r in self.decode_batch:
            r["remaining"] -= 1
            r["generated"] += 1

    def _tier1_check(self):
        if self.sim_time_s - self.tier1_state.window_start_s < self.tier1_interval_s:
            return

        st = self.tier1_state
        window_duration = self.sim_time_s - st.window_start_s
        total_events = st.window_reqs + len(st.window_tpot_samples)
        slo_viol_pct = (st.window_slo_violations / max(total_events, 1)) * 100

        # Compute current QPS
        qps = st.window_reqs / max(window_duration, 1)
        avg_il = 0
        if self.decode_batch:
            avg_il = int(np.mean([r["il"] for r in self.decode_batch]))

        # Tier 1 re-planning logic
        old_f_a, old_f_f = self.cur_f_a, self.cur_f_f
        reason = None
        new_f_a, new_f_f = old_f_a, old_f_f

        # Check triggers
        if slo_viol_pct > 1.0:
            st.consecutive_slo_windows += 1
        else:
            st.consecutive_slo_windows = 0

        load_shifted = (st.last_qps > 0 and abs(qps - st.last_qps) / max(st.last_qps, 0.1) > 0.5)
        if load_shifted:
            st.consecutive_load_shift_windows += 1
        else:
            st.consecutive_load_shift_windows = 0

        if st.consecutive_slo_windows >= 2:
            reason = "SLO_violation_rate > 1% (2 consecutive windows)"
            new_f_a, new_f_f = F_MAX, F_MAX
            st.consecutive_slo_windows = 0
        elif st.consecutive_load_shift_windows >= 2:
            reason = f"load_shift (QPS {st.last_qps:.1f}→{qps:.1f})"
            # Adapt frequency based on new load
            if qps < 2.0:
                new_f_a, new_f_f = 450, 690
            elif qps < 4.0:
                new_f_a, new_f_f = 690, 930
            elif qps < 6.0:
                new_f_a, new_f_f = 930, 1170
            else:
                new_f_a, new_f_f = 1410, 1410
            st.consecutive_load_shift_windows = 0

        # Record window
        self.tier1_windows.append({
            "time_s": self.sim_time_s,
            "qps": round(qps, 2),
            "avg_il": avg_il,
            "active_bs": len(self.decode_batch),
            "slo_viol_pct": round(slo_viol_pct, 2),
            "tpot_p50_us": round(float(np.percentile(st.window_tpot_samples, 50)), 0) if st.window_tpot_samples else 0,
            "tpot_p99_us": round(float(np.percentile(st.window_tpot_samples, 99)), 0) if st.window_tpot_samples else 0,
            "cur_f_a": self.cur_f_a,
            "cur_f_f": self.cur_f_f,
            "replan": reason is not None,
            "reason": reason or "",
        })

        if reason:
            self.tier1_decisions.append(Tier1Decision(
                time_s=self.sim_time_s,
                reason=reason,
                old_f_a=old_f_a, old_f_f=old_f_f,
                new_f_a=new_f_a, new_f_f=new_f_f,
                qps=qps,
                avg_il=avg_il,
                slo_violation_pct=slo_viol_pct,
                active_bs=len(self.decode_batch),
            ))
            # Apply new baseline
            self.cur_f_a, self.cur_f_f = new_f_a, new_f_f
            self.dvfs_ctrl.update_baseline(new_f_a, new_f_f)

        st.last_qps = qps
        st.last_avg_il = avg_il
        # Reset window
        st.window_start_s = self.sim_time_s
        st.window_reqs = 0
        st.window_slo_violations = 0
        st.window_tpot_samples = []
        st.window_ttft_samples = []


# ═══════════════════════════════════════════════════════════════════════
# Output
# ═══════════════════════════════════════════════════════════════════════

def print_trace(sim: TracingSimulator):
    print("=" * 100)
    print(f"  TIER 1 + TIER 2 DECISION TRACE")
    print(f"  Workload: {len(sim.requests)} requests, mode={sim.mode}")
    print(f"  SLO: TTFT≤{TTFT_SLO_MS}ms, TPOT≤{TPOT_SLO_US}us")
    print("=" * 100)

    # ── Tier 1 Monitoring Windows ──
    print(f"\n{'─'*100}")
    print(f"  TIER 1 MONITORING WINDOWS (every {sim.tier1_interval_s}s)")
    print(f"{'─'*100}")
    print(f"  {'Time':>6s} {'QPS':>5s} {'AvgIL':>6s} {'BS':>4s} {'SLO%':>6s} "
          f"{'TPOT_p50':>9s} {'TPOT_p99':>9s} {'f_A':>5s} {'f_F':>5s} {'Replan':>7s} {'Reason'}")
    print(f"  {'─'*6} {'─'*5} {'─'*6} {'─'*4} {'─'*6} {'─'*9} {'─'*9} {'─'*5} {'─'*5} {'─'*7} {'─'*30}")

    for w in sim.tier1_windows:
        replan_mark = ">>> YES" if w["replan"] else "no"
        print(f"  {w['time_s']:6.1f} {w['qps']:5.1f} {w['avg_il']:6d} {w['active_bs']:4d} "
              f"{w['slo_viol_pct']:6.2f} {w['tpot_p50_us']:9.0f} {w['tpot_p99_us']:9.0f} "
              f"{w['cur_f_a']:5d} {w['cur_f_f']:5d} {replan_mark:>7s} {w['reason']}")

    # ── Tier 1 Re-planning Decisions ──
    print(f"\n{'─'*100}")
    print(f"  TIER 1 RE-PLANNING DECISIONS ({len(sim.tier1_decisions)} total)")
    print(f"{'─'*100}")
    if sim.tier1_decisions:
        for d in sim.tier1_decisions:
            print(f"\n  [T={d.time_s:.1f}s] TIER 1 REPLAN")
            print(f"    Reason:     {d.reason}")
            print(f"    State:      QPS={d.qps:.1f}, avg_il={d.avg_il}, active_bs={d.active_bs}, SLO_viol={d.slo_violation_pct:.1f}%")
            print(f"    Freq change: f_A={d.old_f_a}→{d.new_f_a} MHz, f_F={d.old_f_f}→{d.new_f_f} MHz")
            print(f"    Action:     Update baseline + apply new freq to all 4 pools")
    else:
        print("  (No Tier 1 re-planning triggered — workload within bounds)")

    # ── Tier 2 Decisions (grouped by phase) ──
    print(f"\n{'─'*100}")
    print(f"  TIER 2 FREQUENCY DECISIONS ({len(sim.tier2_decisions)} total)")
    print(f"{'─'*100}")

    # Group by time ranges for readability
    prefill_decisions = [d for d in sim.tier2_decisions if d.phase_type == "prefill"]
    decode_decisions = [d for d in sim.tier2_decisions if d.phase_type == "decode"]

    print(f"\n  Prefill decisions: {len(prefill_decisions)}")
    print(f"  {'Time':>6s} {'BS':>3s} {'IL':>5s} {'f_A':>5s}→{'f_A':>5s} {'f_F':>5s}→{'f_F':>5s} "
          f"{'Switch':>6s} {'Lat%SLO':>8s} {'E(mJ)':>8s}")
    print(f"  {'─'*6} {'─'*3} {'─'*5} {'─'*11} {'─'*11} {'─'*6} {'─'*8} {'─'*8}")
    for d in prefill_decisions[:30]:  # show first 30
        print(f"  {d.time_s:6.1f} {d.bs:3d} {d.avg_il:5d} "
              f"{d.old_f_a:5d}→{d.new_f_a:5d} {d.old_f_f:5d}→{d.new_f_f:5d} "
              f"{'YES' if d.switched else 'no':>6s} {d.slo_pct:7.1f}% {d.energy_mj:8.1f}")
    if len(prefill_decisions) > 30:
        print(f"  ... ({len(prefill_decisions) - 30} more)")

    print(f"\n  Decode decisions: {len(decode_decisions)}")
    print(f"  {'Time':>6s} {'Trigger':>14s} {'BS':>3s} {'IL':>5s} "
          f"{'f_A':>5s}→{'f_A':>5s} {'f_F':>5s}→{'f_F':>5s} "
          f"{'Switch':>6s} {'TPOT%SLO':>9s} {'E(mJ)':>8s} {'Lat(us)':>8s}")
    print(f"  {'─'*6} {'─'*14} {'─'*3} {'─'*5} {'─'*11} {'─'*11} {'─'*6} {'─'*9} {'─'*8} {'─'*8}")
    for d in decode_decisions:
        print(f"  {d.time_s:6.1f} {d.trigger:>14s} {d.bs:3d} {d.avg_il:5d} "
              f"{d.old_f_a:5d}→{d.new_f_a:5d} {d.old_f_f:5d}→{d.new_f_f:5d} "
              f"{'YES' if d.switched else 'no':>6s} {d.slo_pct:8.1f}% {d.energy_mj:8.1f} {d.latency_us:8.0f}")

    # ── Summary ──
    print(f"\n{'─'*100}")
    print(f"  SUMMARY")
    print(f"{'─'*100}")
    print(f"  Total energy:       {sim.total_energy_mj:.0f} mJ")
    print(f"  Requests completed: {sim.n_completed}")
    print(f"  SLO violations:     {sim.n_slo_violations}")
    print(f"  Tier 1 replans:     {len(sim.tier1_decisions)}")
    print(f"  Tier 2 freq changes:{len([d for d in sim.tier2_decisions if d.switched])}")
    print(f"  Decode iterations:  {sim.decode_iter_count}")
    if sim.tpot_samples:
        print(f"  TPOT p50/p99:       {np.percentile(sim.tpot_samples, 50):.0f} / {np.percentile(sim.tpot_samples, 99):.0f} us")
    if sim.ttft_samples:
        print(f"  TTFT p50/p99:       {np.percentile(sim.ttft_samples, 50):.1f} / {np.percentile(sim.ttft_samples, 99):.1f} ms")

    # Frequency distribution
    print(f"\n  Frequency distribution (decode):")
    f_a_counts = {}
    f_f_counts = {}
    for d in decode_decisions:
        f_a_counts[d.new_f_a] = f_a_counts.get(d.new_f_a, 0) + 1
        f_f_counts[d.new_f_f] = f_f_counts.get(d.new_f_f, 0) + 1
    print(f"    f_A: {dict(sorted(f_a_counts.items()))}")
    print(f"    f_F: {dict(sorted(f_f_counts.items()))}")


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Trace Tier1+Tier2 decisions")
    parser.add_argument("--workload", type=str,
                        default=str(Path(__file__).parent / "workloads" / "workload_varying.jsonl"))
    parser.add_argument("--mode", type=str, default="tier1+tier2",
                        choices=["tier2_only", "tier1+tier2"])
    args = parser.parse_args()

    sim = TracingSimulator(args.workload, mode=args.mode)
    sim.run()
    print_trace(sim)


if __name__ == "__main__":
    main()

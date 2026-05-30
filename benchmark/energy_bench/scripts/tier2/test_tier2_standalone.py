#!/usr/bin/env python3
"""Standalone Tier2 DVFS controller simulation — demonstrates frequency adaptation.

This script directly exercises the AFDVFSController without launching a full
server. It simulates varying workload conditions and shows how Tier2 selects
frequencies to minimize energy while respecting SLO constraints.

This is useful for:
  - Validating DVFS logic without GPU servers
  - Visualizing frequency decisions under different loads
  - Generating data for paper figures

Usage:
    python test_tier2_standalone.py
    python test_tier2_standalone.py --slo-tpot-us 100000
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "python"))

from sglang.srt.energy.af_dvfs_controller import (
    AFDVFSController,
    REEVAL_BS_CHANGE,
    REEVAL_SLO_URGENT,
    REEVAL_WINDOW_EXPIRED,
)
from sglang.srt.energy.af_profile_predictor import AFProfilePredictor

ENERGY_MODEL_DIR = "/workspace/sglang/benchmark/test_motivation/energy_models"


def simulate_workload_phases(ctrl: AFDVFSController, slo_tpot_us: float):
    """Simulate multi-phase workload and record DVFS decisions."""
    print("=" * 70)
    print("  Tier2 DVFS Controller Simulation")
    print(f"  SLO TPOT: {slo_tpot_us:.0f} us ({slo_tpot_us/1000:.1f} ms)")
    print("=" * 70)

    phases = [
        # (name, iterations, bs, il, ol, description)
        ("Light Load",    60, 4,   128, 64,   "Low batch size, short seqs → expect freq reduction"),
        ("Heavy Load",    60, 64,  1024, 256,  "High batch size, long seqs → expect max freq"),
        ("Medium Load",   60, 16,  512, 128,   "Moderate load → expect intermediate freq"),
        ("Burst Peak",    20, 128, 512, 128,   "Sudden spike → SLO_URGENT trigger"),
        ("Post-Burst",    40, 4,   128, 64,    "Load drops → freq should decrease"),
    ]

    all_decisions = []
    iteration = 0

    for phase_name, n_iters, bs, il, ol, desc in phases:
        print(f"\n{'─'*70}")
        print(f"  Phase: {phase_name} ({desc})")
        print(f"  bs={bs}, il={il}, ol={ol}, iterations={n_iters}")
        print(f"{'─'*70}")

        phase_decisions = []
        for i in range(n_iters):
            iteration += 1
            ctrl.tick_decode_iteration()

            # Simulate measured TPOT (higher for heavier load)
            simulated_tpot = ctrl._layer_latency("decode", ctrl._decode_state.cur_f_a,
                                                  ctrl._decode_state.cur_f_f,
                                                  bs, il, ol, M=1) * ctrl.num_layers

            reason = ctrl.should_reevaluate_decode(
                current_bs=bs,
                current_tpot_us=simulated_tpot,
                slo_tpot_us=slo_tpot_us,
            )

            if reason != 0:
                decision = ctrl.select_freq_decode(
                    bs=bs, il=il, ol=ol,
                    slo_tpot_us=slo_tpot_us, M=1,
                    reeval_reason=reason,
                )
                reason_name = {1: "WINDOW", 2: "BS_CHG", 3: "SLO_URG"}[reason]
                phase_decisions.append({
                    "iter": iteration,
                    "reason": reason_name,
                    "f_a": decision.f_a,
                    "f_f": decision.f_f,
                    "energy_mj": decision.energy_mj,
                    "latency_us": decision.latency_us,
                    "switched": decision.switched,
                })

                if decision.switched:
                    print(f"    iter {iteration:4d}: [{reason_name:7s}] "
                          f"f_a={decision.f_a:4d} f_f={decision.f_f:4d} "
                          f"E={decision.energy_mj:.1f}mJ lat={decision.latency_us:.0f}us "
                          f"{'↑↑ SWITCHED' if decision.switched else ''}")

        if phase_decisions:
            final = phase_decisions[-1]
            print(f"  → Final freq: f_a={final['f_a']} MHz, f_f={final['f_f']} MHz")
            print(f"    Energy: {final['energy_mj']:.1f} mJ/layer, "
                  f"Latency: {final['latency_us']:.0f} us")
        else:
            st = ctrl._decode_state
            print(f"  → No re-evaluation triggered (holding f_a={st.cur_f_a}, f_f={st.cur_f_f})")

        all_decisions.extend(phase_decisions)

    # Summary
    print(f"\n{'='*70}")
    print("  SUMMARY")
    print(f"{'='*70}")
    print(f"  Total iterations: {iteration}")
    print(f"  Total decisions:  {len(all_decisions)}")
    print(f"  Switch-ups:       {ctrl._stats_switch_up}")
    print(f"  Switch-downs:     {ctrl._stats_switch_down}")
    print(f"  Fallbacks:        {ctrl._stats_fallback}")

    # Frequency distribution
    freq_a_counts: dict[int, int] = {}
    freq_f_counts: dict[int, int] = {}
    for d in all_decisions:
        freq_a_counts[d["f_a"]] = freq_a_counts.get(d["f_a"], 0) + 1
        freq_f_counts[d["f_f"]] = freq_f_counts.get(d["f_f"], 0) + 1

    print("\n  Frequency selection distribution:")
    print(f"    f_a: {dict(sorted(freq_a_counts.items()))}")
    print(f"    f_f: {dict(sorted(freq_f_counts.items()))}")

    # Energy comparison: all-max vs DVFS
    max_energy_total = 0
    dvfs_energy_total = 0
    for phase_name, n_iters, bs, il, ol, _ in phases:
        try:
            e_max = ctrl._layer_energy("decode", 1410, 1410, bs, il, ol) * ctrl.num_layers * n_iters
            max_energy_total += e_max
        except Exception:
            pass

    for d in all_decisions:
        dvfs_energy_total += d["energy_mj"]

    if max_energy_total > 0 and dvfs_energy_total > 0:
        saving_pct = (max_energy_total - dvfs_energy_total) / max_energy_total * 100
        print(f"\n  Energy comparison (decision points only):")
        print(f"    Max-freq energy:  {max_energy_total:.1f} mJ")
        print(f"    DVFS energy:      {dvfs_energy_total:.1f} mJ")
        print(f"    Saving:           {saving_pct:.1f}%")

    return all_decisions


def test_prefill_dvfs(ctrl: AFDVFSController):
    """Test prefill DVFS with different slack budgets."""
    print(f"\n{'='*70}")
    print("  Prefill DVFS Test (per-request frequency selection)")
    print(f"{'='*70}")

    test_cases = [
        # (bs, il, slack_us, description)
        (1,  128,  500000, "Single short request, generous slack"),
        (1,  128,  50000,  "Single short request, tight slack"),
        (4,  1024, 200000, "Batch of 4, medium sequences"),
        (4,  1024, 50000,  "Batch of 4, tight deadline"),
        (8,  2048, 500000, "Large batch, long sequences, generous"),
        (8,  2048, 100000, "Large batch, long sequences, tight"),
    ]

    print(f"\n  {'BS':<4s} {'IL':<6s} {'Slack(ms)':<10s} {'f_a':<6s} {'f_f':<6s} {'E(mJ)':<10s} {'Lat(ms)':<10s} Description")
    print(f"  {'─'*4} {'─'*6} {'─'*10} {'─'*6} {'─'*6} {'─'*10} {'─'*10} {'─'*30}")

    for bs, il, slack_us, desc in test_cases:
        try:
            decision = ctrl.select_freq_prefill(bs=bs, il=il, slack_us=slack_us, M=1)
            print(f"  {bs:<4d} {il:<6d} {slack_us/1000:<10.0f} "
                  f"{decision.f_a:<6d} {decision.f_f:<6d} "
                  f"{decision.energy_mj:<10.1f} {decision.latency_us/1000:<10.2f} {desc}")
        except Exception as e:
            print(f"  {bs:<4d} {il:<6d} {slack_us/1000:<10.0f} {'ERROR':>6s} — {e}")


def main():
    parser = argparse.ArgumentParser(description="Tier2 DVFS standalone simulation")
    parser.add_argument("--energy-model-dir", type=str, default=ENERGY_MODEL_DIR)
    parser.add_argument("--num-layers", type=int, default=28,
                        help="Number of transformer layers (Qwen3-0.6B=28)")
    parser.add_argument("--slo-tpot-us", type=float, default=300000,
                        help="TPOT SLO in microseconds")
    parser.add_argument("--output", type=str, default=None,
                        help="Save decisions to JSON file")
    args = parser.parse_args()

    print("Loading energy prediction models...")
    predictor = AFProfilePredictor(args.energy_model_dir)

    ctrl = AFDVFSController(
        predictor=predictor,
        num_layers=args.num_layers,
        tp_a=1,
        tp_f=1,
        t_comm_us=0.0,
    )

    # Test prefill DVFS
    test_prefill_dvfs(ctrl)

    # Test decode DVFS with varying workload
    decisions = simulate_workload_phases(ctrl, slo_tpot_us=args.slo_tpot_us)

    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            json.dump(decisions, f, indent=2)
        print(f"\nDecisions saved to {out_path}")


if __name__ == "__main__":
    main()

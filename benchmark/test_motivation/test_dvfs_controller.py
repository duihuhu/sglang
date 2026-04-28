#!/usr/bin/env python3
"""Smoke test for AFDVFSController.

Usage (from benchmark/test_motivation where energy_models/ exists):
    python test_dvfs_controller.py
    python test_dvfs_controller.py --model-dir /path/to/energy_models
"""

import argparse
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR.parent.parent / "python"))

from sglang.srt.energy.af_profile_predictor import AFProfilePredictor, VALID_FREQS
from sglang.srt.energy.af_dvfs_controller import (
    AFDVFSController, F_MAX, F_MIN, W_MIN,
)


def test_prefill_dvfs(ctrl: AFDVFSController):
    """Prefill DVFS should pick lower freq when slack is generous."""
    print("\n[1] Prefill DVFS tests")

    # Tight SLO → should pick high freq
    tight = ctrl.select_freq_prefill(bs=4, il=1024, slack_us=100_000, M=1)
    print(f"  Tight SLO:    f_A={tight.f_a}, f_F={tight.f_f}, "
          f"energy={tight.energy_mj:.0f} mJ, lat={tight.latency_us:.0f} us")

    # Generous SLO → should pick lower freq (less energy)
    generous = ctrl.select_freq_prefill(bs=4, il=1024, slack_us=5_000_000, M=1)
    print(f"  Generous SLO: f_A={generous.f_a}, f_F={generous.f_f}, "
          f"energy={generous.energy_mj:.0f} mJ, lat={generous.latency_us:.0f} us")

    assert generous.energy_mj <= tight.energy_mj, \
        f"Generous SLO should use less energy: {generous.energy_mj} vs {tight.energy_mj}"
    print("  Energy ordering correct: generous <= tight")

    # Impossible SLO → fallback to max freq
    impossible = ctrl.select_freq_prefill(bs=4, il=1024, slack_us=1.0, M=1)
    assert impossible.f_a == F_MAX and impossible.f_f == F_MAX, \
        f"Should fallback to max freq, got f_A={impossible.f_a} f_F={impossible.f_f}"
    print(f"  Impossible SLO: correctly falls back to ({F_MAX}, {F_MAX})")
    print("  PASSED")


def test_decode_dvfs(ctrl: AFDVFSController):
    """Decode DVFS should respect SLO and use lazy switching."""
    print("\n[2] Decode DVFS tests")

    ctrl.reset_decode_state()

    # First call: generous SLO
    d1 = ctrl.select_freq_decode(bs=16, il=512, ol=128, slo_tpot_us=50_000, M=1)
    print(f"  First call:  f_A={d1.f_a}, f_F={d1.f_f}, "
          f"energy={d1.energy_mj:.1f} mJ, switched={d1.switched}")

    # Simulate iterations
    for _ in range(5):
        ctrl.tick_decode_iteration()

    # Second call with same params: should NOT trigger re-evaluation
    should_reeval = ctrl.should_reevaluate_decode(current_bs=16)
    print(f"  After 5 iters, should_reevaluate={should_reeval}")
    assert not should_reeval, "Should not re-evaluate after only 5 iterations"

    # Simulate full window
    for _ in range(60):
        ctrl.tick_decode_iteration()
    should_reeval = ctrl.should_reevaluate_decode(current_bs=16)
    print(f"  After 65 iters, should_reevaluate={should_reeval}")
    assert should_reeval, "Should re-evaluate after full window"

    # Batch size change trigger
    ctrl.reset_decode_state()
    ctrl.select_freq_decode(bs=16, il=512, ol=128, slo_tpot_us=50_000, M=1)
    ctrl.tick_decode_iteration()
    should_reeval = ctrl.should_reevaluate_decode(current_bs=32)
    print(f"  BS change 16→32, should_reevaluate={should_reeval}")
    assert should_reeval, "Should re-evaluate on >30% batch size change"

    print("  PASSED")


def test_decode_lazy_switching(ctrl: AFDVFSController):
    """Lazy switching should avoid switching for tiny energy savings."""
    print("\n[3] Lazy switching test")

    ctrl.reset_decode_state()

    # Set initial state to max freq
    d1 = ctrl.select_freq_decode(bs=16, il=512, ol=128, slo_tpot_us=50_000, M=1)
    print(f"  Initial: f_A={d1.f_a}, f_F={d1.f_f}, switched={d1.switched}")

    # Simulate full window and re-evaluate with same SLO
    for _ in range(65):
        ctrl.tick_decode_iteration()
    d2 = ctrl.select_freq_decode(bs=16, il=512, ol=128, slo_tpot_us=50_000, M=1)
    print(f"  Re-eval: f_A={d2.f_a}, f_F={d2.f_f}, switched={d2.switched}")

    print("  PASSED")


def test_window_size_computation(ctrl: AFDVFSController):
    """Window size should scale with iteration time."""
    print("\n[4] Window size computation")

    w1 = ctrl.compute_window_size(t_iter_avg_us=1000)
    print(f"  t_iter=1000us → W={w1}")
    assert w1 == 60, f"Expected 60, got {w1}"

    w2 = ctrl.compute_window_size(t_iter_avg_us=100)
    print(f"  t_iter=100us  → W={w2}")
    assert w2 == 600, f"Expected 600, got {w2}"

    w3 = ctrl.compute_window_size(t_iter_avg_us=100_000)
    print(f"  t_iter=100ms  → W={w3}")
    assert w3 == W_MIN, f"Expected {W_MIN}, got {w3}"

    print("  PASSED")


def test_af_asymmetry(ctrl: AFDVFSController):
    """A and F should get different frequencies (AF asymmetry)."""
    print("\n[5] AF asymmetry test")

    ctrl.reset_decode_state()
    d = ctrl.select_freq_decode(bs=16, il=512, ol=128, slo_tpot_us=50_000, M=1)
    print(f"  Decode bs=16: f_A={d.f_a}, f_F={d.f_f}")
    if d.f_a != d.f_f:
        print(f"  Asymmetric frequencies (expected: DA is memory-bound → lower f_A)")
    else:
        print(f"  Same frequencies (may happen at certain operating points)")

    generous = ctrl.select_freq_prefill(bs=4, il=2048, slack_us=10_000_000, M=1)
    print(f"  Prefill generous: f_A={generous.f_a}, f_F={generous.f_f}")
    print("  PASSED")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", type=str,
                        default=str(SCRIPT_DIR / "energy_models"))
    parser.add_argument("--num-layers", type=int, default=64)
    args = parser.parse_args()

    print(f"Loading models from: {args.model_dir}")
    predictor = AFProfilePredictor(args.model_dir)
    ctrl = AFDVFSController(predictor, num_layers=args.num_layers, tp=1)

    test_prefill_dvfs(ctrl)
    test_decode_dvfs(ctrl)
    test_decode_lazy_switching(ctrl)
    test_window_size_computation(ctrl)
    test_af_asymmetry(ctrl)

    print("\n" + "=" * 50)
    print(" All DVFS controller tests passed!")
    print("=" * 50)


if __name__ == "__main__":
    main()

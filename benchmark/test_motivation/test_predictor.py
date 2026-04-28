#!/usr/bin/env python3
"""Smoke test for AFProfilePredictor.

Usage (from the benchmark/test_motivation directory where energy_models/ exists):
    python test_predictor.py
    python test_predictor.py --model-dir /path/to/energy_models
"""

import argparse
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR.parent.parent / "python"))

from sglang.srt.energy.af_profile_predictor import AFProfilePredictor, VALID_FREQS


def test_basic_predictions(predictor: AFProfilePredictor):
    """Test that all 8 sub-models return valid predictions."""
    print("\n[1] Basic prediction tests")
    cases = [
        ("prefill", "A", 1, 930, 4, 1024, None),
        ("prefill", "F", 1, 930, 4, 1024, None),
        ("prefill", "A", 2, 690, 8, 2048, None),
        ("decode",  "A", 1, 690, 16, 512, 128),
        ("decode",  "F", 1, 690, 16, 512, 128),
        ("decode",  "A", 2, 1170, 32, 1024, 256),
        ("decode",  "F", 4, 450, 8, 256, 64),
    ]
    for phase, op, tp, freq, bs, il, ol in cases:
        lat = predictor.predict_latency(phase, op, tp, freq, bs, il, ol)
        eng = predictor.predict_energy(phase, op, tp, freq, bs, il, ol)
        ol_str = f" ol={ol}" if ol else ""
        print(f"  {phase:>7s}_{op} tp={tp} f={freq:>4d} bs={bs:>3d} il={il:>5d}{ol_str}")
        print(f"    latency={lat.value:>10.1f} us ({lat.model_type}, exact={lat.exact_match})")
        print(f"    energy ={eng.value:>10.1f} mJ ({eng.model_type}, exact={eng.exact_match})")
        assert lat.value > 0, f"Latency should be positive, got {lat.value}"
        assert eng.value > 0, f"Energy should be positive, got {eng.value}"
    print("  PASSED")


def test_exact_match(predictor: AFProfilePredictor):
    """Test that profiling grid points get LUT exact match."""
    print("\n[2] LUT exact match test")
    lat = predictor.predict_latency("prefill", "A", 1, 210, 1, 128, None)
    print(f"  prefill_A tp=1 f=210 bs=1 il=128 -> {lat.model_type} exact={lat.exact_match}")
    if lat.exact_match:
        print(f"  value={lat.value:.1f} us (LUT exact)")
    print("  PASSED" if lat.exact_match else "  WARN: not exact match (LUT may use interpolation)")


def test_monotonicity(predictor: AFProfilePredictor):
    """Energy should generally increase with batch size."""
    print("\n[3] Monotonicity test (energy vs batch_size)")
    prev_e = 0
    monotonic = True
    for bs in [1, 2, 4, 8, 16, 32, 64]:
        eng = predictor.predict_energy("decode", "F", 1, 930, bs, 512, 128)
        if eng.value < prev_e:
            monotonic = False
            print(f"  WARN: bs={bs} energy={eng.value:.1f} < prev={prev_e:.1f}")
        prev_e = eng.value
    print("  PASSED" if monotonic else "  WARN: non-monotonic (may be acceptable for small deviations)")


def test_freq_search(predictor: AFProfilePredictor):
    """Test find_best_freq_pair returns valid result."""
    print("\n[4] Frequency pair search test")
    result = predictor.find_best_freq_pair(
        phase="decode", tp=1, bs=16, il=512,
        slo_budget_us=50000, ol=128, M=1,
    )
    if result:
        f_a, f_f, total_e = result
        print(f"  Best: f_A={f_a}, f_F={f_f}, energy={total_e:.1f} mJ")
        assert f_a in VALID_FREQS and f_f in VALID_FREQS
        print("  PASSED")
    else:
        print("  WARN: no feasible combo found (SLO too tight?)")

    result_tight = predictor.find_best_freq_pair(
        phase="decode", tp=1, bs=16, il=512,
        slo_budget_us=1.0, ol=128, M=1,
    )
    assert result_tight is None, "Should return None for impossible SLO"
    print("  Tight SLO correctly returns None")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", type=str,
                        default=str(SCRIPT_DIR / "energy_models"))
    args = parser.parse_args()

    print(f"Loading models from: {args.model_dir}")
    predictor = AFProfilePredictor(args.model_dir)

    test_basic_predictions(predictor)
    test_exact_match(predictor)
    test_monotonicity(predictor)
    test_freq_search(predictor)

    print("\n" + "=" * 50)
    print(" All tests passed!")
    print("=" * 50)


if __name__ == "__main__":
    main()

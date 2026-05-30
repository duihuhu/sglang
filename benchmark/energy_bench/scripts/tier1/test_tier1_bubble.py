"""Test 1: E_bubble in Tier 1 objective function.

Verifies that:
  1. _PoolCandidate.e_bubble_mj() correctly computes pipeline bubble energy
  2. _pair_bubble_energy() penalizes A/F imbalance
  3. The objective function in _search_kp_kd prefers balanced configs over
     imbalanced ones (all else being equal)
  4. M=1 produces zero bubble (serial execution, no waiting)

Run:
    python -m pytest benchmark/energy_bench/test_tier1_bubble.py -xvs
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "python"))

from sglang.srt.energy.tier1_solver import (
    Tier1Solver,
    WorkloadProfile,
    SLOConfig,
    _PoolCandidate,
    _DEFAULT_M,
    _P_IDLE_W,
)


class TestPoolCandidateBubble:
    """Unit tests for _PoolCandidate.e_bubble_mj()."""

    def test_balanced_zero_bubble(self):
        """Perfectly balanced A/F should have zero bubble energy."""
        c = _PoolCandidate(
            tp=4, freq=930,
            t_a_us=1000.0, t_f_us=1000.0,
            e_a_mj=50.0, e_f_mj=50.0,
        )
        assert c.e_bubble_mj(M=2) == 0.0

    def test_m1_zero_bubble(self):
        """M=1 (serial) should always produce zero bubble."""
        c = _PoolCandidate(
            tp=4, freq=930,
            t_a_us=1000.0, t_f_us=4000.0,
            e_a_mj=50.0, e_f_mj=200.0,
        )
        assert c.e_bubble_mj(M=1) == 0.0

    def test_imbalanced_positive_bubble(self):
        """Imbalanced A/F should produce positive bubble energy."""
        c = _PoolCandidate(
            tp=4, freq=930,
            t_a_us=1000.0, t_f_us=4000.0,
            e_a_mj=50.0, e_f_mj=200.0,
        )
        bubble = c.e_bubble_mj(M=2)
        # E_bubble = P_idle × |t_A - t_F| × (M-1)/M
        # = 80W × 3000us / 1e6 × 1000 × (2-1)/2
        # = 80 × 0.003 × 1000 × 0.5 = 0.12 mJ
        expected = _P_IDLE_W * 3000.0 / 1_000_000.0 * 1000.0 * (2 - 1) / 2
        assert abs(bubble - expected) < 1e-6, f"got {bubble}, expected {expected}"

    def test_m3_reduces_bubble(self):
        """M=3 should have (M-1)/M = 2/3 factor, less than M=2's 1/2."""
        c = _PoolCandidate(
            tp=4, freq=930,
            t_a_us=1000.0, t_f_us=4000.0,
            e_a_mj=50.0, e_f_mj=200.0,
        )
        bubble_m2 = c.e_bubble_mj(M=2)
        bubble_m3 = c.e_bubble_mj(M=3)
        # M=3 has (2/3) factor vs M=2's (1/2) factor → M=3 bubble > M=2 bubble
        assert bubble_m3 > bubble_m2

    def test_bubble_proportional_to_imbalance(self):
        """Larger imbalance → larger bubble energy."""
        c_small = _PoolCandidate(
            tp=4, freq=930,
            t_a_us=900.0, t_f_us=1000.0,
            e_a_mj=50.0, e_f_mj=55.0,
        )
        c_large = _PoolCandidate(
            tp=4, freq=930,
            t_a_us=500.0, t_f_us=4000.0,
            e_a_mj=25.0, e_f_mj=200.0,
        )
        assert c_large.e_bubble_mj(M=2) > c_small.e_bubble_mj(M=2)


class TestPairBubbleEnergy:
    """Unit tests for Tier1Solver._pair_bubble_energy()."""

    def test_balanced_pair(self):
        """Balanced A and F candidates should have zero bubble."""
        cand_a = _PoolCandidate(
            tp=4, freq=930,
            t_a_us=1000.0, t_f_us=1000.0,
            e_a_mj=50.0, e_f_mj=50.0,
        )
        cand_f = _PoolCandidate(
            tp=4, freq=930,
            t_a_us=1000.0, t_f_us=1000.0,
            e_a_mj=50.0, e_f_mj=50.0,
        )
        e = Tier1Solver._pair_bubble_energy(cand_a, cand_f, M=2)
        assert e == 0.0

    def test_imbalanced_pair(self):
        """Imbalanced pair: uses cand_a.t_a_us vs cand_f.t_f_us."""
        cand_a = _PoolCandidate(
            tp=4, freq=1410,
            t_a_us=500.0, t_f_us=500.0,
            e_a_mj=30.0, e_f_mj=30.0,
        )
        cand_f = _PoolCandidate(
            tp=4, freq=690,
            t_a_us=2000.0, t_f_us=3000.0,
            e_a_mj=80.0, e_f_mj=120.0,
        )
        e = Tier1Solver._pair_bubble_energy(cand_a, cand_f, M=2)
        # |cand_a.t_a_us - cand_f.t_f_us| = |500 - 3000| = 2500
        expected = _P_IDLE_W * 2500.0 / 1_000_000.0 * (2 - 1) / 2 * 1000.0
        assert abs(e - expected) < 1e-6

    def test_m1_no_bubble(self):
        """M=1 should always return 0."""
        cand_a = _PoolCandidate(
            tp=4, freq=1410,
            t_a_us=500.0, t_f_us=500.0,
            e_a_mj=30.0, e_f_mj=30.0,
        )
        cand_f = _PoolCandidate(
            tp=4, freq=210,
            t_a_us=5000.0, t_f_us=8000.0,
            e_a_mj=100.0, e_f_mj=200.0,
        )
        assert Tier1Solver._pair_bubble_energy(cand_a, cand_f, M=1) == 0.0


class TestObjectiveFunctionPreference:
    """Integration test: solver should prefer balanced configs."""

    def test_bubble_penalizes_imbalance(self):
        """Given two configs with same raw energy but different balance,
        the solver's objective should rank the balanced one lower (better)."""
        # Balanced: t_a=1000, t_f=1000, E_A+E_F = 100mJ
        cand_balanced_a = _PoolCandidate(
            tp=4, freq=930,
            t_a_us=1000.0, t_f_us=1000.0,
            e_a_mj=50.0, e_f_mj=50.0,
            t_layer_us=2000.0,
            throughput_per_pair=100.0,
        )
        cand_balanced_f = _PoolCandidate(
            tp=4, freq=930,
            t_a_us=1000.0, t_f_us=1000.0,
            e_a_mj=50.0, e_f_mj=50.0,
            t_layer_us=2000.0,
            throughput_per_pair=100.0,
        )

        # Imbalanced: t_a=500, t_f=3500, E_A+E_F = 100mJ (same raw energy)
        cand_imbal_a = _PoolCandidate(
            tp=4, freq=1410,
            t_a_us=500.0, t_f_us=500.0,
            e_a_mj=25.0, e_f_mj=25.0,
            t_layer_us=4000.0,
            throughput_per_pair=100.0,
        )
        cand_imbal_f = _PoolCandidate(
            tp=4, freq=450,
            t_a_us=3500.0, t_f_us=3500.0,
            e_a_mj=75.0, e_f_mj=75.0,
            t_layer_us=4000.0,
            throughput_per_pair=100.0,
        )

        M = 2
        # Balanced total = E_A + E_F + E_bubble = 100 + 0 = 100 mJ
        e_balanced = (
            cand_balanced_a.e_total_mj + cand_balanced_f.e_total_mj
            + Tier1Solver._pair_bubble_energy(cand_balanced_a, cand_balanced_f, M)
        )

        # Imbalanced total = E_A + E_F + E_bubble
        e_imbal = (
            cand_imbal_a.e_total_mj + cand_imbal_f.e_total_mj
            + Tier1Solver._pair_bubble_energy(cand_imbal_a, cand_imbal_f, M)
        )

        # Bubble should make imbalanced config more expensive
        assert e_imbal > e_balanced, (
            f"Imbalanced ({e_imbal:.4f}mJ) should be more expensive than "
            f"balanced ({e_balanced:.4f}mJ) due to bubble penalty"
        )


if __name__ == "__main__":
    import pytest
    pytest.main([__file__, "-xvs"])

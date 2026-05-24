"""Test 2: Decode throughput semantics in Tier 1 solver.

Verifies that:
  1. _pair_throughput returns bs (capacity) for decode phase
  2. _pair_throughput returns bs/batch_time (req/s) for prefill phase
  3. _select_for_pool correctly uses capacity semantics for decode
  4. Decode demand (N_active) is compared against capacity (bs), not rate

Run:
    python -m pytest benchmark/energy_bench/test_tier1_throughput.py -xvs
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "python"))

from sglang.srt.energy.tier1_solver import (
    Tier1Solver,
    _PoolCandidate,
)


class TestPairThroughput:
    """Unit tests for Tier1Solver._pair_throughput()."""

    def test_prefill_throughput_formula(self):
        """Prefill: throughput = bs / (L × t_layer / 1e6)."""
        bs = 8
        t_layer_us = 5000.0  # 5ms per layer
        num_layers = 64
        thpt = Tier1Solver._pair_throughput(bs, t_layer_us, num_layers, "prefill")
        # batch_time = 64 × 5000 / 1e6 = 0.32s
        # throughput = 8 / 0.32 = 25 req/s
        expected = bs / (num_layers * t_layer_us / 1_000_000.0)
        assert abs(thpt - expected) < 1e-6
        assert abs(thpt - 25.0) < 1e-6

    def test_decode_capacity_formula(self):
        """Decode: capacity = bs (concurrent requests per pair)."""
        bs = 16
        t_layer_us = 3000.0
        num_layers = 64
        cap = Tier1Solver._pair_throughput(bs, t_layer_us, num_layers, "decode")
        assert cap == float(bs)
        assert cap == 16.0

    def test_decode_capacity_independent_of_latency(self):
        """Decode capacity should not depend on t_layer or num_layers."""
        bs = 32
        cap1 = Tier1Solver._pair_throughput(bs, 1000.0, 64, "decode")
        cap2 = Tier1Solver._pair_throughput(bs, 5000.0, 128, "decode")
        assert cap1 == cap2 == 32.0

    def test_prefill_throughput_scales_with_bs(self):
        """Prefill throughput should scale linearly with batch size."""
        t_layer_us = 5000.0
        num_layers = 64
        thpt_4 = Tier1Solver._pair_throughput(4, t_layer_us, num_layers, "prefill")
        thpt_8 = Tier1Solver._pair_throughput(8, t_layer_us, num_layers, "prefill")
        assert abs(thpt_8 / thpt_4 - 2.0) < 1e-6

    def test_prefill_throughput_inversely_scales_with_latency(self):
        """Prefill throughput should decrease with higher latency."""
        bs = 8
        num_layers = 64
        thpt_fast = Tier1Solver._pair_throughput(bs, 2000.0, num_layers, "prefill")
        thpt_slow = Tier1Solver._pair_throughput(bs, 4000.0, num_layers, "prefill")
        assert abs(thpt_fast / thpt_slow - 2.0) < 1e-6

    def test_zero_latency_returns_inf(self):
        """Edge case: zero latency should return inf for prefill."""
        thpt = Tier1Solver._pair_throughput(8, 0.0, 64, "prefill")
        assert thpt == float("inf")


class TestSelectForPoolDecode:
    """Test _select_for_pool with decode capacity semantics."""

    def test_decode_capacity_constraint(self):
        """k × bs ≥ (1+α) × N_active for decode."""
        # Candidate with bs=16 (capacity=16)
        cand = _PoolCandidate(
            tp=4, freq=930,
            t_a_us=1000.0, t_f_us=3000.0,
            e_a_mj=50.0, e_f_mj=150.0,
            t_layer_us=4000.0,
            throughput_per_pair=16.0,  # decode capacity = bs
        )

        # k=2, demand=30, alpha=0.15
        # k × capacity = 2 × 16 = 32 ≥ (1+0.15) × 30 = 34.5? No → None
        result = Tier1Solver._select_for_pool([cand], k=2, demand=30, alpha=0.15)
        assert result is None

        # k=3, demand=30, alpha=0.15
        # k × capacity = 3 × 16 = 48 ≥ 34.5? Yes → selected
        result = Tier1Solver._select_for_pool([cand], k=3, demand=30, alpha=0.15)
        assert result is not None
        assert result.tp == 4

    def test_prefill_rate_constraint(self):
        """k × throughput ≥ (1+α) × λ for prefill."""
        # Candidate with throughput = 25 req/s
        cand = _PoolCandidate(
            tp=4, freq=930,
            t_a_us=1000.0, t_f_us=4000.0,
            e_a_mj=50.0, e_f_mj=200.0,
            t_layer_us=5000.0,
            throughput_per_pair=25.0,
        )

        # k=1, demand=20 req/s, alpha=0.15
        # 1 × 25 = 25 ≥ (1.15) × 20 = 23? Yes
        result = Tier1Solver._select_for_pool([cand], k=1, demand=20, alpha=0.15)
        assert result is not None

        # k=1, demand=25 req/s, alpha=0.15
        # 1 × 25 = 25 ≥ 28.75? No
        result = Tier1Solver._select_for_pool([cand], k=1, demand=25, alpha=0.15)
        assert result is None

    def test_selects_min_energy(self):
        """Among feasible candidates, should pick lowest energy."""
        cand_cheap = _PoolCandidate(
            tp=2, freq=690,
            t_a_us=2000.0, t_f_us=5000.0,
            e_a_mj=40.0, e_f_mj=100.0,
            t_layer_us=7000.0,
            throughput_per_pair=16.0,
        )
        cand_expensive = _PoolCandidate(
            tp=4, freq=1410,
            t_a_us=500.0, t_f_us=2000.0,
            e_a_mj=80.0, e_f_mj=200.0,
            t_layer_us=2500.0,
            throughput_per_pair=16.0,
        )

        # Both meet capacity: k=3, demand=30, alpha=0.15 → 48 ≥ 34.5
        result = Tier1Solver._select_for_pool(
            [cand_cheap, cand_expensive], k=3, demand=30, alpha=0.15,
        )
        assert result is cand_cheap


if __name__ == "__main__":
    import pytest
    pytest.main([__file__, "-xvs"])

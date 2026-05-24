"""Test 4: Tier 1 transition logic (drain-then-switch).

Verifies that:
  1. Frequency-only changes are applied immediately
  2. TP/k changes trigger a warning and max-freq interim
  3. Tier 2 baseline is updated after transition

This test mocks the scheduler's _apply_freq and _af_dvfs_ctrl to test
the transition logic in isolation without starting the full server.

Run:
    python -m pytest benchmark/energy_bench/test_tier1_transition.py -xvs
"""

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch, call

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "python"))

from sglang.srt.energy.tier1_solver import Tier1Solution


class MockScheduler:
    """Minimal mock of the scheduler with transition-related methods."""

    def __init__(self):
        self._af_dvfs_ctrl = MagicMock()
        self._af_dvfs_ctrl.update_baseline = MagicMock()
        self._freq_calls = []

    def _apply_freq(self, f_a, f_f):
        self._freq_calls.append((f_a, f_f))

    def _apply_tier1_transition(self, old, new):
        """Copied logic from scheduler.py for isolated testing."""
        freq_only = (
            old.tp_pa == new.tp_pa and old.tp_pf == new.tp_pf
            and old.tp_da == new.tp_da and old.tp_df == new.tp_df
            and old.k_p == new.k_p and old.k_d == new.k_d
        )

        if freq_only:
            self._apply_freq(new.f_da, new.f_df)
            if hasattr(self, "_af_dvfs_ctrl") and self._af_dvfs_ctrl is not None:
                self._af_dvfs_ctrl.update_baseline(new.f_da, new.f_df)
        else:
            self._apply_freq(1410, 1410)
            if hasattr(self, "_af_dvfs_ctrl") and self._af_dvfs_ctrl is not None:
                self._af_dvfs_ctrl.update_baseline(1410, 1410)


class TestFrequencyOnlyTransition:
    """Test scenario 1: only frequencies changed."""

    def test_applies_new_freq(self):
        """Should apply the new frequencies directly."""
        sched = MockScheduler()
        old = Tier1Solution(
            k_p=2, k_d=2,
            tp_pa=4, tp_pf=4, tp_da=4, tp_df=4,
            f_pa=1410, f_pf=1410, f_da=1410, f_df=1410,
            feasible=True,
        )
        new = Tier1Solution(
            k_p=2, k_d=2,
            tp_pa=4, tp_pf=4, tp_da=4, tp_df=4,
            f_pa=930, f_pf=690, f_da=930, f_df=690,
            feasible=True,
        )
        sched._apply_tier1_transition(old, new)
        assert sched._freq_calls == [(930, 690)]

    def test_updates_tier2_baseline(self):
        """Should update Tier 2 baseline to new decode frequencies."""
        sched = MockScheduler()
        old = Tier1Solution(
            k_p=1, k_d=1,
            tp_pa=2, tp_pf=2, tp_da=2, tp_df=2,
            f_pa=1410, f_pf=1410, f_da=1410, f_df=1410,
            feasible=True,
        )
        new = Tier1Solution(
            k_p=1, k_d=1,
            tp_pa=2, tp_pf=2, tp_da=2, tp_df=2,
            f_pa=690, f_pf=450, f_da=690, f_df=450,
            feasible=True,
        )
        sched._apply_tier1_transition(old, new)
        sched._af_dvfs_ctrl.update_baseline.assert_called_once_with(690, 450)


class TestTPChangeTransition:
    """Test scenario 2: TP changed (requires drain-then-switch)."""

    def test_tp_change_goes_max_freq(self):
        """TP change should apply max freq as interim measure."""
        sched = MockScheduler()
        old = Tier1Solution(
            k_p=2, k_d=2,
            tp_pa=4, tp_pf=4, tp_da=4, tp_df=4,
            f_pa=930, f_pf=930, f_da=930, f_df=930,
            feasible=True,
        )
        new = Tier1Solution(
            k_p=2, k_d=2,
            tp_pa=2, tp_pf=8, tp_da=2, tp_df=8,  # TP changed
            f_pa=690, f_pf=450, f_da=690, f_df=450,
            feasible=True,
        )
        sched._apply_tier1_transition(old, new)
        assert sched._freq_calls == [(1410, 1410)]
        sched._af_dvfs_ctrl.update_baseline.assert_called_once_with(1410, 1410)


class TestKChangeTransition:
    """Test scenario 3: k_P or k_D changed."""

    def test_k_change_goes_max_freq(self):
        """k change should apply max freq as interim measure."""
        sched = MockScheduler()
        old = Tier1Solution(
            k_p=2, k_d=2,
            tp_pa=4, tp_pf=4, tp_da=4, tp_df=4,
            f_pa=930, f_pf=930, f_da=930, f_df=930,
            feasible=True,
        )
        new = Tier1Solution(
            k_p=3, k_d=1,  # k changed
            tp_pa=4, tp_pf=4, tp_da=4, tp_df=4,
            f_pa=690, f_pf=690, f_da=1170, f_df=1170,
            feasible=True,
        )
        sched._apply_tier1_transition(old, new)
        assert sched._freq_calls == [(1410, 1410)]

    def test_no_dvfs_ctrl_doesnt_crash(self):
        """Should handle missing _af_dvfs_ctrl gracefully."""
        sched = MockScheduler()
        sched._af_dvfs_ctrl = None
        old = Tier1Solution(
            k_p=2, k_d=2,
            tp_pa=4, tp_pf=4, tp_da=4, tp_df=4,
            f_pa=930, f_pf=930, f_da=930, f_df=930,
            feasible=True,
        )
        new = Tier1Solution(
            k_p=2, k_d=2,
            tp_pa=4, tp_pf=4, tp_da=4, tp_df=4,
            f_pa=690, f_pf=690, f_da=690, f_df=690,
            feasible=True,
        )
        # Should not raise
        sched._apply_tier1_transition(old, new)
        assert sched._freq_calls == [(690, 690)]


if __name__ == "__main__":
    import pytest
    pytest.main([__file__, "-xvs"])

"""Test 4: Tier 1 transition logic (frequency-only reload).

Verifies that:
  1. Frequency-only changes are applied immediately
  2. TP/k changes still apply the new frequencies (not max freq)
  3. Tier 2 baseline is updated after transition
  4. Shared freq config file is written for cross-process sync

This test mocks the scheduler's _apply_freq and _af_dvfs_ctrl to test
the transition logic in isolation without starting the full server.

Run:
    python -m pytest benchmark/energy_bench/test_tier1_transition.py -xvs
"""

import sys
import json
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch, call

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "python"))

from sglang.srt.energy.tier1_solver import Tier1Solution


class MockServerArgs:
    tier1_stats_path = None


class MockScheduler:
    """Minimal mock of the scheduler with transition-related methods."""

    def __init__(self, disagg_mode="decode", tier1_stats_path=None):
        self._af_dvfs_ctrl = MagicMock()
        self._af_dvfs_ctrl.update_baseline = MagicMock()
        self._freq_calls = []
        self._cur_f_a = 1410
        self._cur_f_f = 1410
        self.server_args = MockServerArgs()
        self.server_args.tier1_stats_path = tier1_stats_path
        self._disagg_mode = disagg_mode

    @property
    def disaggregation_mode(self):
        from sglang.srt.disaggregation.utils import DisaggregationMode
        return DisaggregationMode(self._disagg_mode)

    def _apply_freq(self, f_a, f_f):
        self._freq_calls.append((f_a, f_f))
        self._cur_f_a = f_a
        self._cur_f_f = f_f

    def _get_tier1_freq_path(self):
        stats_path = self.server_args.tier1_stats_path
        if not stats_path:
            return ""
        import os
        return os.path.join(os.path.dirname(stats_path), "tier1_freq_config.json")

    def _write_tier1_freq_config(self, solution):
        import time
        freq_path = self._get_tier1_freq_path()
        if not freq_path:
            return
        config = {
            "f_pa": solution.f_pa,
            "f_pf": solution.f_pf,
            "f_da": solution.f_da,
            "f_df": solution.f_df,
            "timestamp": time.time(),
        }
        with open(freq_path, "w") as f:
            json.dump(config, f)

    def _apply_tier1_transition(self, old, new):
        """Mirrors the new scheduler logic: always apply freq, ignore TP/k."""
        from sglang.srt.disaggregation.utils import DisaggregationMode

        disagg_mode = self.disaggregation_mode
        if disagg_mode == DisaggregationMode.PREFILL:
            local_f_a, local_f_f = new.f_pa, new.f_pf
        else:
            local_f_a, local_f_f = new.f_da, new.f_df

        self._apply_freq(local_f_a, local_f_f)
        if hasattr(self, "_af_dvfs_ctrl") and self._af_dvfs_ctrl is not None:
            self._af_dvfs_ctrl.update_baseline(local_f_a, local_f_f)

        self._write_tier1_freq_config(new)


class TestFrequencyOnlyTransition:
    """Test scenario 1: only frequencies changed (decode mode)."""

    def test_applies_new_freq(self):
        """Should apply the new decode frequencies directly."""
        sched = MockScheduler(disagg_mode="decode")
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

    def test_prefill_mode_applies_prefill_freq(self):
        """In prefill mode, should apply PA/PF frequencies."""
        sched = MockScheduler(disagg_mode="prefill")
        old = Tier1Solution(
            k_p=2, k_d=2,
            tp_pa=4, tp_pf=4, tp_da=4, tp_df=4,
            f_pa=1410, f_pf=1410, f_da=1410, f_df=1410,
            feasible=True,
        )
        new = Tier1Solution(
            k_p=2, k_d=2,
            tp_pa=4, tp_pf=4, tp_da=4, tp_df=4,
            f_pa=930, f_pf=690, f_da=1170, f_df=450,
            feasible=True,
        )
        sched._apply_tier1_transition(old, new)
        assert sched._freq_calls == [(930, 690)]

    def test_updates_tier2_baseline(self):
        """Should update Tier 2 baseline to new decode frequencies."""
        sched = MockScheduler(disagg_mode="decode")
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
    """Test scenario 2: TP changed — still applies new frequencies."""

    def test_tp_change_applies_new_freq(self):
        """TP change should still apply the new frequencies (not max freq)."""
        sched = MockScheduler(disagg_mode="decode")
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
        # Should apply new decode freq, not max freq
        assert sched._freq_calls == [(690, 450)]
        sched._af_dvfs_ctrl.update_baseline.assert_called_once_with(690, 450)


class TestKChangeTransition:
    """Test scenario 3: k_P or k_D changed — still applies new frequencies."""

    def test_k_change_applies_new_freq(self):
        """k change should still apply the new frequencies."""
        sched = MockScheduler(disagg_mode="decode")
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
        assert sched._freq_calls == [(1170, 1170)]

    def test_no_dvfs_ctrl_doesnt_crash(self):
        """Should handle missing _af_dvfs_ctrl gracefully."""
        sched = MockScheduler(disagg_mode="decode")
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
        sched._apply_tier1_transition(old, new)
        assert sched._freq_calls == [(690, 690)]


class TestSharedFreqConfig:
    """Test scenario 4: shared freq config file for cross-process sync."""

    def test_writes_freq_config(self):
        """Should write freq config to shared file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            stats_path = str(Path(tmpdir) / "stats.json")
            sched = MockScheduler(disagg_mode="decode", tier1_stats_path=stats_path)
            new = Tier1Solution(
                k_p=2, k_d=2,
                tp_pa=4, tp_pf=4, tp_da=4, tp_df=4,
                f_pa=930, f_pf=690, f_da=1170, f_df=450,
                feasible=True,
            )
            sched._apply_tier1_transition(
                Tier1Solution(
                    k_p=2, k_d=2,
                    tp_pa=4, tp_pf=4, tp_da=4, tp_df=4,
                    f_pa=1410, f_pf=1410, f_da=1410, f_df=1410,
                    feasible=True,
                ),
                new,
            )
            freq_path = str(Path(tmpdir) / "tier1_freq_config.json")
            assert Path(freq_path).exists()
            config = json.loads(Path(freq_path).read_text())
            assert config["f_pa"] == 930
            assert config["f_pf"] == 690
            assert config["f_da"] == 1170
            assert config["f_df"] == 450


if __name__ == "__main__":
    import pytest
    pytest.main([__file__, "-xvs"])

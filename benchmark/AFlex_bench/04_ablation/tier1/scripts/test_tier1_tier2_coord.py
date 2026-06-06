"""Test 3: Tier 1 ↔ Tier 2 coordination.

Verifies that:
  1. AFDVFSController accepts baseline frequencies from Tier 1
  2. update_baseline() correctly updates the baseline
  3. is_energy_saving property reflects whether Tier 2 is below baseline
  4. WorkloadMetricsCollector.build_window() passes is_tier2_energy_saving
  5. WorkloadMonitor suppresses A/F imbalance when Tier 2 is saving energy

Run:
    python -m pytest benchmark/energy_bench/test_tier1_tier2_coord.py -xvs
"""

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "python"))

from sglang.srt.energy.af_dvfs_controller import (
    AFDVFSController,
    DVFSDecision,
    F_MAX,
    REEVAL_NONE,
    REEVAL_WINDOW_EXPIRED,
)
from sglang.srt.energy.workload_monitor import (
    MonitoringWindow,
    WorkloadMonitor,
)
from sglang.srt.energy.workload_collector import WorkloadMetricsCollector


def _make_mock_predictor():
    """Create a mock predictor that returns deterministic values."""
    predictor = MagicMock()

    def mock_latency(phase, op, tp, freq, bs, il, ol=None):
        result = MagicMock()
        base = 1000.0 if op == "A" else 3000.0
        result.value = base * (1410.0 / freq)
        return result

    def mock_energy(phase, op, tp, freq, bs, il, ol=None):
        result = MagicMock()
        base = 50.0 if op == "A" else 150.0
        result.value = base * (freq / 1410.0)
        return result

    predictor.predict_latency = mock_latency
    predictor.predict_energy = mock_energy
    return predictor


class TestBaselineFrequency:
    """Test baseline frequency initialization and update."""

    def test_default_baseline_is_max(self):
        """Without explicit baseline, should default to F_MAX."""
        ctrl = AFDVFSController(
            predictor=_make_mock_predictor(),
            num_layers=64, tp_a=4,
        )
        assert ctrl._baseline_f_a == F_MAX
        assert ctrl._baseline_f_f == F_MAX

    def test_custom_baseline(self):
        """Explicit baseline frequencies should be stored."""
        ctrl = AFDVFSController(
            predictor=_make_mock_predictor(),
            num_layers=64, tp_a=4,
            baseline_f_a=930, baseline_f_f=690,
        )
        assert ctrl._baseline_f_a == 930
        assert ctrl._baseline_f_f == 690

    def test_decode_state_starts_at_baseline(self):
        """Decode window state should start at baseline frequency."""
        ctrl = AFDVFSController(
            predictor=_make_mock_predictor(),
            num_layers=64, tp_a=4,
            baseline_f_a=930, baseline_f_f=690,
        )
        assert ctrl._decode_state.cur_f_a == 930
        assert ctrl._decode_state.cur_f_f == 690

    def test_update_baseline(self):
        """update_baseline() should change the stored baseline."""
        ctrl = AFDVFSController(
            predictor=_make_mock_predictor(),
            num_layers=64, tp_a=4,
        )
        ctrl.update_baseline(690, 450)
        assert ctrl._baseline_f_a == 690
        assert ctrl._baseline_f_f == 450

    def test_reset_decode_uses_baseline(self):
        """After reset, decode state should return to baseline."""
        ctrl = AFDVFSController(
            predictor=_make_mock_predictor(),
            num_layers=64, tp_a=4,
            baseline_f_a=930, baseline_f_f=690,
        )
        # Simulate a frequency change
        ctrl._decode_state.cur_f_a = 1410
        ctrl._decode_state.cur_f_f = 1410
        # Reset
        ctrl.reset_decode_state()
        assert ctrl._decode_state.cur_f_a == 930
        assert ctrl._decode_state.cur_f_f == 690


class TestIsEnergySaving:
    """Test is_energy_saving property."""

    def test_at_baseline_not_saving(self):
        """Running at baseline = not saving energy."""
        ctrl = AFDVFSController(
            predictor=_make_mock_predictor(),
            num_layers=64, tp_a=4,
            baseline_f_a=1410, baseline_f_f=1410,
        )
        assert ctrl.is_energy_saving is False

    def test_below_baseline_is_saving(self):
        """Running below baseline = saving energy."""
        ctrl = AFDVFSController(
            predictor=_make_mock_predictor(),
            num_layers=64, tp_a=4,
            baseline_f_a=1410, baseline_f_f=1410,
        )
        ctrl._decode_state.cur_f_a = 930
        assert ctrl.is_energy_saving is True

    def test_above_baseline_not_saving(self):
        """Running at or above baseline = not saving."""
        ctrl = AFDVFSController(
            predictor=_make_mock_predictor(),
            num_layers=64, tp_a=4,
            baseline_f_a=930, baseline_f_f=690,
        )
        ctrl._decode_state.cur_f_a = 1410
        ctrl._decode_state.cur_f_f = 1410
        assert ctrl.is_energy_saving is False

    def test_one_below_is_saving(self):
        """If either f_a or f_f is below baseline, it's saving."""
        ctrl = AFDVFSController(
            predictor=_make_mock_predictor(),
            num_layers=64, tp_a=4,
            baseline_f_a=1410, baseline_f_f=1410,
        )
        ctrl._decode_state.cur_f_a = 1410
        ctrl._decode_state.cur_f_f = 930
        assert ctrl.is_energy_saving is True


class TestCollectorPassesEnergySaving:
    """Test that WorkloadMetricsCollector passes is_tier2_energy_saving."""

    def test_build_window_with_flag(self):
        """build_window(is_tier2_energy_saving=True) should set the field."""
        collector = WorkloadMetricsCollector(
            ttft_slo_ms=5000.0,
            tpot_slo_us=50000.0,
            window_s=0.001,  # very short window so it triggers immediately
            enable_nvml_polling=False,
        )
        # Add enough dummy data so build_window doesn't return None
        from sglang.srt.energy.workload_collector import _ReqRecord
        collector._reqs = [
            _ReqRecord(rid=f"r{i}", ttft_us=1000.0, il=512)
            for i in range(15)
        ]

        window = collector.build_window(is_tier2_energy_saving=True)
        assert window is not None
        assert window.is_tier2_energy_saving is True

    def test_build_window_default_false(self):
        """Default is_tier2_energy_saving should be False."""
        collector = WorkloadMetricsCollector(
            ttft_slo_ms=5000.0,
            tpot_slo_us=50000.0,
            window_s=0.001,
            enable_nvml_polling=False,
        )
        from sglang.srt.energy.workload_collector import _ReqRecord
        collector._reqs = [
            _ReqRecord(rid=f"r{i}", ttft_us=1000.0, il=512)
            for i in range(15)
        ]

        window = collector.build_window(is_tier2_energy_saving=False)
        assert window is not None
        assert window.is_tier2_energy_saving is False


class TestMonitorSuppression:
    """Test that WorkloadMonitor suppresses A/F imbalance during energy saving."""

    def test_af_imbalance_suppressed_when_saving(self):
        """A/F imbalance should NOT trigger replan when Tier 2 is saving energy."""
        mon = WorkloadMonitor(window_s=10.0, consecutive_threshold=2)

        # Two windows with A/F imbalance but Tier 2 is saving energy
        for _ in range(3):
            mon.record_window(MonitoringWindow(
                slo_violation_rate=0.0,  # SLO is fine
                a_util=0.8, f_util=0.3,  # big imbalance
                p_util=0.5, d_util=0.5,
                is_tier2_energy_saving=True,
            ))

        should, reasons = mon.should_replan()
        # Should NOT trigger because Tier 2 is intentionally saving
        assert "A/F utilization imbalance" not in " ".join(reasons)

    def test_af_imbalance_triggers_when_not_saving(self):
        """A/F imbalance SHOULD trigger replan when Tier 2 is NOT saving."""
        mon = WorkloadMonitor(window_s=10.0, consecutive_threshold=2)

        for _ in range(3):
            mon.record_window(MonitoringWindow(
                slo_violation_rate=0.0,
                a_util=0.8, f_util=0.3,
                p_util=0.5, d_util=0.5,
                is_tier2_energy_saving=False,
            ))

        should, reasons = mon.should_replan()
        assert should is True
        assert any("imbalance" in r for r in reasons)

    def test_af_imbalance_not_suppressed_when_slo_violated(self):
        """Even if saving energy, SLO violation + imbalance should trigger."""
        mon = WorkloadMonitor(window_s=10.0, consecutive_threshold=2)

        for _ in range(3):
            mon.record_window(MonitoringWindow(
                slo_violation_rate=0.05,  # 5% violation
                a_util=0.8, f_util=0.3,
                p_util=0.5, d_util=0.5,
                is_tier2_energy_saving=True,
            ))

        should, reasons = mon.should_replan()
        # SLO violation should trigger regardless
        assert should is True
        assert any("SLO" in r for r in reasons)


if __name__ == "__main__":
    import pytest
    pytest.main([__file__, "-xvs"])

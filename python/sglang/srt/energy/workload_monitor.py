"""B-3: Workload Monitor — runtime metric tracking and Tier-1 re-planning trigger.

Collects per-window statistics from the running scheduler and detects when
the current Tier-1 configuration should be re-planned.  Four metrics are
checked each window, any of which can trigger re-planning after two
consecutive violations (hysteresis to avoid flapping).

Metrics
  1. SLO violation rate  > 1%   → config can't keep up with load
  2. |A_util - F_util|   > 0.2  → A/F pairing ratio is off (unless caused by
     Tier-2 intentional down-clocking while SLO is satisfied)
  3. |P_util - D_util|   > 0.2  → P/D split ratio is off
  4. KL(curr_dist || ref) > thr → workload distribution has shifted

Integration
  The scheduler calls record_window() after each monitoring interval (~10-30 s).
  When should_replan() returns True, the scheduler invokes Tier1Solver.solve()
  and initiates a drain-then-switch transition.

Usage
    mon = WorkloadMonitor(window_s=30.0)
    # ... each window ...
    mon.record_window(MonitoringWindow(
        slo_violation_rate=0.005,
        a_util=0.7, f_util=0.5,
        p_util=0.3, d_util=0.6,
        load_distribution={"il_128": 0.3, "il_512": 0.3, ...},
        is_tier2_energy_saving=False,
    ))
    trigger, reason = mon.should_replan()
    if trigger:
        new_config = tier1_solver.solve(...)
"""

from __future__ import annotations

import logging
import math
import time
from collections import Counter
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger(__name__)

# ── Thresholds ──────────────────────────────────────────────────────────

_SLO_VIOLATION_THRESHOLD = 0.01       # 1%
_AF_UTIL_IMBALANCE_THRESHOLD = 0.2
_PD_UTIL_IMBALANCE_THRESHOLD = 0.2
_KL_DIVERGENCE_THRESHOLD = 2.0

_DEFAULT_WINDOW_S = 30.0
_CONSECUTIVE_WINDOWS = 2


# ═══════════════════════════════════════════════════════════════════════
# Data classes
# ═══════════════════════════════════════════════════════════════════════

@dataclass
class MonitoringWindow:
    """One monitoring window's statistics.

    All utilization values are in [0, 1].

    Args:
        slo_violation_rate: Fraction of requests that violated SLO this window.
        a_util: Average Attention GPU utilization this window.
        f_util: Average FFN GPU utilization this window.
        p_util: Average Prefill pipeline utilization this window.
        d_util: Average Decode pipeline utilization this window.
        load_distribution: Dict mapping bucket label → fraction of requests.
            e.g. {"il_128": 0.3, "il_512": 0.3, "il_1024": 0.2, ...}
        is_tier2_energy_saving: True if Tier 2 is currently running below
            baseline frequency for energy saving (suppresses A/F imbalance
            false positives).
        active_requests: Number of concurrently active decode requests.
        tpot_p99_us: P99 TPOT this window (microseconds).
        ttft_p99_ms: P99 TTFT this window (milliseconds).
    """
    slo_violation_rate: float = 0.0
    a_util: float = 0.0
    f_util: float = 0.0
    p_util: float = 0.0
    d_util: float = 0.0
    load_distribution: dict[str, float] = field(default_factory=dict)
    is_tier2_energy_saving: bool = False
    active_requests: int = 0
    tpot_p99_us: float = 0.0
    ttft_p99_ms: float = 0.0


# ═══════════════════════════════════════════════════════════════════════
# Monitor
# ═══════════════════════════════════════════════════════════════════════

class WorkloadMonitor:
    """Sliding-window metric tracker with hysteresis for Tier-1 re-planning.

    Args:
        window_s: Duration of each monitoring window in seconds (default 30).
        history_size: Maximum number of windows to retain.
        consecutive_threshold: Number of consecutive violating windows required
            to trigger re-planning (default 2).
    """

    def __init__(
        self,
        window_s: float = _DEFAULT_WINDOW_S,
        history_size: int = 32,
        consecutive_threshold: int = _CONSECUTIVE_WINDOWS,
    ):
        self.window_s = window_s
        self.history_size = history_size
        self.consecutive_threshold = consecutive_threshold

        self._history: list[MonitoringWindow] = []
        self._reference_dist: Optional[dict[str, float]] = None
        self._anomaly_counters: dict[str, int] = {
            "slo": 0, "af_util": 0, "pd_util": 0, "kl": 0,
        }
        self._start_time = time.monotonic()

    # ── Public API ──────────────────────────────────────────────────────

    def record_window(self, stats: MonitoringWindow):
        """Record one completed monitoring window.

        Call this from the scheduler's monitoring loop (every ~10-30 s).
        """
        self._history.append(stats)
        # Trim
        if len(self._history) > self.history_size:
            self._history = self._history[-self.history_size:]

        # Store first window's distribution as reference if not set
        if self._reference_dist is None and stats.load_distribution:
            self._reference_dist = dict(stats.load_distribution)

        # Update anomaly counters
        self._update_counters(stats)
        logger.debug(
            "WorkloadMonitor: window #%d slo_viol=%.3f a_util=%.2f f_util=%.2f "
            "p_util=%.2f d_util=%.2f active=%d",
            len(self._history),
            stats.slo_violation_rate, stats.a_util, stats.f_util,
            stats.p_util, stats.d_util, stats.active_requests,
        )

    def should_replan(self) -> tuple[bool, list[str]]:
        """Check whether Tier-1 re-planning should be triggered.

        Returns:
            (should_replan, reasons) where reasons is a list of triggering
            metric names, empty if no re-planning is needed.
        """
        if len(self._history) < self.consecutive_threshold:
            return False, []

        reasons = []
        for metric, count in self._anomaly_counters.items():
            if count >= self.consecutive_threshold:
                reasons.append(self._metric_label(metric))

        return len(reasons) > 0, reasons

    def reset_reference(self, new_dist: dict[str, float]):
        """Update the reference load distribution (call after Tier-1 re-planning)."""
        self._reference_dist = dict(new_dist)
        self._anomaly_counters = {k: 0 for k in self._anomaly_counters}

    # ── Internal ─────────────────────────────────────────────────────────

    def _update_counters(self, stats: MonitoringWindow):
        """Increment or reset each anomaly counter based on the latest window."""
        recent = self._history[-self.consecutive_threshold:]

        # metric_1: SLO violation rate
        if all(w.slo_violation_rate > _SLO_VIOLATION_THRESHOLD for w in recent):
            self._anomaly_counters["slo"] = self.consecutive_threshold
        else:
            self._anomaly_counters["slo"] = 0

        # metric_2: A/F utilization imbalance
        # Suppress if Tier 2 is intentionally saving energy (low util from down-clocking)
        if stats.is_tier2_energy_saving and stats.slo_violation_rate <= _SLO_VIOLATION_THRESHOLD:
            self._anomaly_counters["af_util"] = 0
        elif all(abs(w.a_util - w.f_util) > _AF_UTIL_IMBALANCE_THRESHOLD for w in recent):
            self._anomaly_counters["af_util"] = self.consecutive_threshold
        else:
            self._anomaly_counters["af_util"] = 0

        # metric_3: P/D utilization imbalance
        if all(abs(w.p_util - w.d_util) > _PD_UTIL_IMBALANCE_THRESHOLD for w in recent):
            self._anomaly_counters["pd_util"] = self.consecutive_threshold
        else:
            self._anomaly_counters["pd_util"] = 0

        # metric_4: KL divergence of load distribution
        if self._reference_dist and stats.load_distribution:
            kl = self._kl_divergence(stats.load_distribution, self._reference_dist)
            # We check the current window only (instantaneous), but confirm
            # with the counter that previous windows also had high divergence.
            if kl > _KL_DIVERGENCE_THRESHOLD:
                self._anomaly_counters["kl"] += 1
            else:
                self._anomaly_counters["kl"] = 0
            self._anomaly_counters["kl"] = min(
                self._anomaly_counters["kl"], self.consecutive_threshold,
            )

    @staticmethod
    def _kl_divergence(
        p: dict[str, float], q: dict[str, float],
    ) -> float:
        """Symmetric KL divergence between two discrete distributions.

        Uses JSD-like smoothing: adds a small epsilon to avoid log(0).
        Returns the average of KL(P||Q) and KL(Q||P).
        """
        all_keys = sorted(set(p) | set(q))
        eps = 1e-6
        p_vec = [p.get(k, eps) for k in all_keys]
        q_vec = [q.get(k, eps) for k in all_keys]

        # Normalize
        p_sum = sum(p_vec)
        q_sum = sum(q_vec)
        p_norm = [v / p_sum for v in p_vec]
        q_norm = [v / q_sum for v in q_vec]

        kl_pq = sum(pp * math.log(pp / max(qq, eps)) for pp, qq in zip(p_norm, q_norm))
        kl_qp = sum(qq * math.log(qq / max(pp, eps)) for pp, qq in zip(p_norm, q_norm))
        return (kl_pq + kl_qp) / 2

    @staticmethod
    def _metric_label(metric: str) -> str:
        return {
            "slo": "SLO violation rate > 1%",
            "af_util": "A/F utilization imbalance > 0.2",
            "pd_util": "P/D utilization imbalance > 0.2",
            "kl": "load distribution shift (KL > threshold)",
        }.get(metric, metric)

    # ── Convenience builders ─────────────────────────────────────────────

    def build_distribution_from_bins(
        self,
        bins: dict[str, int],
    ) -> dict[str, float]:
        """Convert count-per-bucket to a normalized distribution.

        Args:
            bins: Mapping of bucket_name → request_count this window.
        """
        total = sum(bins.values())
        if total == 0:
            return {}
        return {k: v / total for k, v in bins.items()}

    @property
    def latest_window(self) -> Optional[MonitoringWindow]:
        return self._history[-1] if self._history else None

    @property
    def uptime_s(self) -> float:
        return time.monotonic() - self._start_time

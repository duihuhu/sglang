"""A-2: AF DVFS Controller — per-iteration frequency selection for AF-disaggregated inference.

Implements two algorithms from the system design:
  - Prefill: per-request DVFS (select f_A, f_F once before forward pass)
  - Decode:  per-window DVFS (re-evaluate every W iterations, with lazy switching)

Usage:
    ctrl = AFDVFSController(predictor, num_layers=64, tp=1)
    f_a, f_f = ctrl.select_freq_prefill(bs=4, il=1024, slack_us=200000, M=1)
    f_a, f_f = ctrl.select_freq_decode(bs=16, il=512, ol=128, slo_tpot_us=50000, M=1)
"""

import logging
import time
from dataclasses import dataclass, field
from typing import Optional

from sglang.srt.energy.af_profile_predictor import AFProfilePredictor, VALID_FREQS

logger = logging.getLogger(__name__)

F_MAX = max(VALID_FREQS)
F_MIN = min(VALID_FREQS)

T_SWITCH_US = 6000.0
E_SWITCH_MJ = 1800.0
F_STEP_MIN = 240
W_MIN = 10


@dataclass
class DVFSDecision:
    f_a: int
    f_f: int
    energy_mj: float
    latency_us: float
    switched: bool = False


@dataclass
class DecodeWindowState:
    """Tracks state for decode per-window DVFS."""
    cur_f_a: int = F_MAX
    cur_f_f: int = F_MAX
    iters_since_decision: int = 0
    last_bs: int = 0
    window_size: int = 60
    last_decision_time: float = 0.0

SLO_URGENCY_RATIO = 0.9

REEVAL_NONE = 0
REEVAL_WINDOW_EXPIRED = 1
REEVAL_BS_CHANGE = 2
REEVAL_SLO_URGENT = 3


class AFDVFSController:
    """Tier 2 DVFS controller for AF-disaggregated serving.

    Args:
        predictor: AFProfilePredictor instance with loaded models.
        num_layers: Number of transformer layers (for prefill budget calc).
        tp_a: Tensor parallelism degree for attention.
        tp_f: Tensor parallelism degree for FFN.
        t_comm_us: Inter-node communication latency per layer (microseconds).
        freqs: Candidate GPU frequencies in MHz.
    """

    def __init__(
        self,
        predictor: AFProfilePredictor,
        num_layers: int = 64,
        tp_a: int = 1,
        tp_f: Optional[int] = None,
        t_comm_us: float = 0.0,
        freqs: Optional[list[int]] = None,
    ):
        self.predictor = predictor
        self.num_layers = num_layers
        self.tp_a = tp_a
        self.tp_f = tp_f if tp_f is not None else tp_a
        self.t_comm_us = t_comm_us
        self.freqs = freqs or VALID_FREQS
        self._decode_state = DecodeWindowState()
        self._stats_switch_up = 0
        self._stats_switch_down = 0
        self._stats_fallback = 0
        self._precompute_freq_pairs()

    def _precompute_freq_pairs(self):
        """Pre-sort frequency pairs by expected energy (low to high)."""
        self._freq_pairs = [
            (fa, ff) for fa in self.freqs for ff in self.freqs
        ]

    def _layer_latency(self, phase: str, f_a: int, f_f: int,
                       bs: int, il: int, ol: Optional[int], M: int) -> float:
        """Compute single-layer latency under pipeline model."""
        lat_a = self.predictor.predict_latency(phase, "A", self.tp_a, f_a, bs, il, ol).value
        lat_f = self.predictor.predict_latency(phase, "F", self.tp_f, f_f, bs, il, ol).value
        if M > 1:
            return max(lat_a, lat_f) + self.t_comm_us / M
        return lat_a + lat_f + self.t_comm_us

    def _layer_energy(self, phase: str, f_a: int, f_f: int,
                      bs: int, il: int, ol: Optional[int]) -> float:
        """Compute single-layer energy (mJ)."""
        e_a = self.predictor.predict_energy(phase, "A", self.tp_a, f_a, bs, il, ol).value
        e_f = self.predictor.predict_energy(phase, "F", self.tp_f, f_f, bs, il, ol).value
        return e_a + e_f

    # ── Prefill: per-request DVFS ──────────────────────────────────

    def select_freq_prefill(
        self, bs: int, il: int, slack_us: float, M: int = 1,
        remaining_layers: Optional[int] = None,
    ) -> DVFSDecision:
        """Select (f_A, f_F) for a prefill batch.

        Args:
            bs: Batch size.
            il: Input sequence length.
            slack_us: Time budget = min(deadline - elapsed) across batch (us).
            M: Number of microbatches in pipeline.
            remaining_layers: Layers left to process (default: all).

        Returns:
            DVFSDecision with chosen frequencies and predicted metrics.
        """
        if remaining_layers is None:
            remaining_layers = self.num_layers

        best: Optional[DVFSDecision] = None

        for f_a, f_f in self._freq_pairs:
            try:
                t_layer = self._layer_latency("prefill", f_a, f_f, bs, il, None, M)
            except (RuntimeError, ValueError):
                continue

            total_lat = t_layer * remaining_layers
            if total_lat > slack_us:
                continue

            try:
                e_layer = self._layer_energy("prefill", f_a, f_f, bs, il, None)
            except (RuntimeError, ValueError):
                continue

            total_e = e_layer * remaining_layers
            if best is None or total_e < best.energy_mj:
                best = DVFSDecision(
                    f_a=f_a, f_f=f_f,
                    energy_mj=total_e, latency_us=total_lat,
                )

        if best is None:
            self._stats_fallback += 1
            logger.warning("Prefill DVFS: no feasible combo, fallback to max freq")
            t_layer = self._layer_latency("prefill", F_MAX, F_MAX, bs, il, None, M)
            e_layer = self._layer_energy("prefill", F_MAX, F_MAX, bs, il, None)
            best = DVFSDecision(
                f_a=F_MAX, f_f=F_MAX,
                energy_mj=e_layer * remaining_layers,
                latency_us=t_layer * remaining_layers,
            )

        logger.debug(
            "Prefill DVFS: bs=%d il=%d → f_a=%d f_f=%d "
            "lat=%.0fus energy=%.1fmJ",
            bs, il, best.f_a, best.f_f, best.latency_us, best.energy_mj,
        )
        return best

    # ── Decode: per-window DVFS ────────────────────────────────────

    def should_reevaluate_decode(
        self, current_bs: int,
        current_tpot_us: float = 0.0,
        slo_tpot_us: float = 0.0,
    ) -> int:
        """Check if decode frequency should be re-evaluated.

        Returns:
            REEVAL_NONE (0) if no re-evaluation needed, otherwise one of
            REEVAL_WINDOW_EXPIRED, REEVAL_BS_CHANGE, REEVAL_SLO_URGENT.
        """
        st = self._decode_state
        if st.iters_since_decision >= st.window_size:
            return REEVAL_WINDOW_EXPIRED
        if st.last_bs > 0 and abs(current_bs - st.last_bs) / st.last_bs > 0.3:
            return REEVAL_BS_CHANGE
        if (current_tpot_us > 0 and slo_tpot_us > 0
                and current_tpot_us > slo_tpot_us * SLO_URGENCY_RATIO):
            return REEVAL_SLO_URGENT
        return REEVAL_NONE

    def select_freq_decode(
        self, bs: int, il: int, ol: int,
        slo_tpot_us: float, M: int = 1,
        reeval_reason: int = REEVAL_WINDOW_EXPIRED,
    ) -> DVFSDecision:
        """Select (f_A, f_F) for a decode window.

        Args:
            bs: Current batch size.
            il: Representative input length for the batch.
            ol: Representative output length for the batch.
            slo_tpot_us: TPOT SLO budget per iteration (us).
            M: Number of microbatches.

        Returns:
            DVFSDecision with chosen frequencies.
        """
        st = self._decode_state

        if reeval_reason == REEVAL_WINDOW_EXPIRED:
            w_remaining = st.window_size
        else:
            w_remaining = st.window_size - st.iters_since_decision

        # Sort candidates by energy (ascending) for early exit
        candidates = []
        for f_a, f_f in self._freq_pairs:
            try:
                e = self._layer_energy("decode", f_a, f_f, bs, il, ol)
                candidates.append((e, f_a, f_f))
            except (RuntimeError, ValueError):
                continue
        candidates.sort()

        for e, f_a, f_f in candidates:
            try:
                t_layer = self._layer_latency("decode", f_a, f_f, bs, il, ol, M)
            except (RuntimeError, ValueError):
                continue

            if t_layer * self.num_layers > slo_tpot_us:
                continue

            switched = self._should_switch(
                f_a, f_f, st.cur_f_a, st.cur_f_f,
                bs, il, ol, w_remaining,
            )

            if switched:
                old_avg = (st.cur_f_a + st.cur_f_f) / 2
                new_avg = (f_a + f_f) / 2
                if new_avg > old_avg:
                    self._stats_switch_up += 1
                else:
                    self._stats_switch_down += 1

            self._update_decode_state(f_a, f_f, bs, switched)
            total_lat = t_layer * self.num_layers
            total_e = e * self.num_layers
            logger.debug(
                "Decode DVFS: bs=%d il=%d ol=%d → f_a=%d f_f=%d "
                "switched=%s lat=%.0fus energy=%.1fmJ "
                "(up=%d down=%d fallback=%d)",
                bs, il, ol, f_a, f_f, switched, total_lat, total_e,
                self._stats_switch_up, self._stats_switch_down,
                self._stats_fallback,
            )
            return DVFSDecision(
                f_a=f_a, f_f=f_f,
                energy_mj=total_e, latency_us=total_lat,
                switched=switched,
            )

        # Fallback: max frequency
        self._stats_fallback += 1
        logger.warning("Decode DVFS: no feasible combo, fallback to max freq")
        self._stats_switch_up += 1
        self._update_decode_state(F_MAX, F_MAX, bs, switched=True)
        t_layer = self._layer_latency("decode", F_MAX, F_MAX, bs, il, ol, M)
        e = self._layer_energy("decode", F_MAX, F_MAX, bs, il, ol)
        return DVFSDecision(
            f_a=F_MAX, f_f=F_MAX,
            energy_mj=e * self.num_layers,
            latency_us=t_layer * self.num_layers,
            switched=True,
        )

    def _should_switch(
        self, f_a_new: int, f_f_new: int,
        f_a_cur: int, f_f_cur: int,
        bs: int, il: int, ol: int,
        w_remaining: int,
    ) -> bool:
        """Lazy switching: only switch if energy savings justify the cost."""
        if f_a_new == f_a_cur and f_f_new == f_f_cur:
            return False

        freq_delta = abs(f_a_new - f_a_cur) + abs(f_f_new - f_f_cur)
        if freq_delta < F_STEP_MIN:
            return False

        try:
            e_cur = self._layer_energy("decode", f_a_cur, f_f_cur, bs, il, ol)
            e_new = self._layer_energy("decode", f_a_new, f_f_new, bs, il, ol)
        except (RuntimeError, ValueError):
            return True

        savings = (e_cur - e_new) * self.num_layers * max(w_remaining, 1)
        return savings > E_SWITCH_MJ

    def _update_decode_state(self, f_a: int, f_f: int, bs: int, switched: bool):
        st = self._decode_state
        if switched:
            st.cur_f_a = f_a
            st.cur_f_f = f_f
        st.last_bs = bs
        st.iters_since_decision = 0
        st.last_decision_time = time.monotonic()

    def tick_decode_iteration(self):
        """Call after each decode iteration to advance the window counter."""
        self._decode_state.iters_since_decision += 1

    def reset_decode_state(self):
        """Reset decode window state (e.g. when batch changes completely)."""
        self._decode_state = DecodeWindowState()

    # ── Utility ────────────────────────────────────────────────────

    def compute_window_size(self, t_iter_avg_us: float) -> int:
        """Compute decode decision window size based on avg iteration time."""
        if t_iter_avg_us <= 0:
            return W_MIN
        w = max(W_MIN, int(10 * T_SWITCH_US / t_iter_avg_us + 0.5))
        self._decode_state.window_size = w
        return w

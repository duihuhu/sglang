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
    # Feedback: consecutive slo_urgent triggers without freq change
    consec_urgent_no_change: int = 0
    # Hold: after forced step-up, hold freq for this many iterations
    hold_iters_remaining: int = 0

SLO_URGENCY_RATIO = 0.9
URGENT_FORCE_STEP_THRESHOLD = 3  # force step up after N consecutive urgent-no-change
URGENT_HOLD_ITERS = 30  # hold forced freq for N iterations before allowing re-eval

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
        t_drain_us: float = 18000.0,  # pipeline drain overhead per iteration (us)
        freqs: Optional[list[int]] = None,
        baseline_f_a: Optional[int] = None,
        baseline_f_f: Optional[int] = None,
        feedback_enabled: bool = False,
        feedback_threshold: int = 3,
        feedback_hold: int = 30,
        online_calibration: bool = False,
        calibration_ema: float = 0.2,
    ):
        self.predictor = predictor
        self.num_layers = num_layers
        self.tp_a = tp_a
        self.tp_f = tp_f if tp_f is not None else tp_a
        self.t_comm_us = t_comm_us
        self.t_drain_us = t_drain_us
        self.freqs = freqs or VALID_FREQS
        self._baseline_f_a = baseline_f_a or F_MAX
        self._baseline_f_f = baseline_f_f or F_MAX
        self._decode_state = DecodeWindowState(
            cur_f_a=self._baseline_f_a,
            cur_f_f=self._baseline_f_f,
        )
        self._stats_switch_up = 0
        self._stats_switch_down = 0
        self._stats_fallback = 0

        # Tier2 feedback: force step-up on repeated SLO urgent
        self._feedback_enabled = feedback_enabled
        self._feedback_threshold = feedback_threshold
        self._feedback_hold = feedback_hold

        # Tier2 online calibration: correct predictor bias
        # Separate factors for pipelined (M>1) vs serial (M=1) execution
        self._calibration_enabled = online_calibration
        self._calibration_ema = calibration_ema
        self._calibration_factor = 1.0  # multiplier for M>1 (pipelined)
        self._calibration_factor_serial = 1.0  # multiplier for M=1 (serial)

        self._precompute_freq_pairs()

    def _precompute_freq_pairs(self):
        """Pre-sort frequency pairs by expected energy (low to high)."""
        self._freq_pairs = [
            (fa, ff) for fa in self.freqs for ff in self.freqs
        ]

    def _next_freq_up(self, current_freq: int) -> int:
        """Return the next higher frequency, or F_MAX if already at max."""
        for f in self.freqs:
            if f > current_freq:
                return f
        return F_MAX

    def update_calibration(self, observed_tpot_us: float, predicted_tpot_us: float,
                           M: int = 2):
        """Update online calibration factor using observed vs predicted TPOT.

        Maintains separate calibration factors for pipelined (M>1) and serial
        (M=1) execution modes, since latency models differ significantly.

        Args:
            observed_tpot_us: Actual measured TPOT (microseconds).
            predicted_tpot_us: Predictor's estimate for the same (bs, il, freq).
            M: Effective micro-batch count used for this iteration.
        """
        if not self._calibration_enabled:
            return
        if predicted_tpot_us <= 0 or observed_tpot_us <= 0:
            return

        ratio = observed_tpot_us / predicted_tpot_us
        ratio = max(0.5, min(ratio, 3.0))

        alpha = self._calibration_ema
        if M > 1:
            self._calibration_factor = (1.0 - alpha) * self._calibration_factor + alpha * ratio
        else:
            self._calibration_factor_serial = (1.0 - alpha) * self._calibration_factor_serial + alpha * ratio

    @property
    def calibration_factor(self) -> float:
        """Current calibration factor for pipelined mode (M>1)."""
        return self._calibration_factor

    def get_calibration_factor(self, M: int = 2) -> float:
        """Get calibration factor appropriate for the given M."""
        if M > 1:
            return self._calibration_factor
        return self._calibration_factor_serial

    def _layer_latency(self, phase: str, f_a: int, f_f: int,
                       bs: int, il: int, ol: Optional[int], M: int) -> float:
        """Compute single-layer latency under pipeline model."""
        lat_a = self.predictor.predict_latency(phase, "A", self.tp_a, f_a, bs, il, ol).value
        lat_f = self.predictor.predict_latency(phase, "F", self.tp_f, f_f, bs, il, ol).value
        if M > 1:
            return max(lat_a, lat_f) + self.t_comm_us / M
        return lat_a + lat_f + self.t_comm_us

    def _iteration_latency(self, phase: str, f_a: int, f_f: int,
                           bs: int, il: int, ol: Optional[int], M: int) -> float:
        """Compute full iteration latency = pipeline + drain overhead.

        Model:
          M > 1: iteration_time = layer_latency * num_layers + t_drain_us
          M = 1: iteration_time = layer_latency * num_layers (no pipeline drain)
        Where:
          - layer_latency: per-layer compute (pipeline overlap for M>1)
          - t_drain_us: IPC sync + pipeline drain (only applies when M>1)
        """
        t_layer = self._layer_latency(phase, f_a, f_f, bs, il, ol, M)
        drain = self.t_drain_us if M > 1 else 0.0
        return t_layer * self.num_layers + drain

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
        lif: float = 1.0,
    ) -> DVFSDecision:
        """Select (f_A, f_F) for a decode window.

        Args:
            bs: Current batch size.
            il: Representative input length for the batch.
            ol: Representative output length for the batch.
            slo_tpot_us: TPOT SLO budget per iteration (us).
            M: Number of microbatches.
            lif: Load Imbalance Factor from expert routing (1.0 = uniform).

        Returns:
            DVFSDecision with chosen frequencies.
        """
        st = self._decode_state

        # If in hold mode after forced step-up, keep current freq
        # but count urgent triggers — if still urgent after hold, step up again
        if st.hold_iters_remaining > 0 and reeval_reason == REEVAL_SLO_URGENT:
            st.consec_urgent_no_change += 1
            return DVFSDecision(
                f_a=st.cur_f_a, f_f=st.cur_f_f,
                energy_mj=0, latency_us=0, switched=False,
            )
        elif st.hold_iters_remaining > 0:
            # Non-urgent re-eval during hold: allow (window expired or bs change)
            pass

        if reeval_reason == REEVAL_WINDOW_EXPIRED:
            w_remaining = st.window_size
        else:
            w_remaining = st.window_size - st.iters_since_decision

        # ─── V2 Coupled Model Path ───────────────────────────────────────
        # If the coupled iteration-level model is available, use it directly
        # for more accurate latency/energy predictions (accounts for IPC,
        # pipeline drain, and micro-batch overlap).
        if (self.predictor is not None
                and self.predictor.has_coupled_model):
            return self._select_freq_decode_coupled(
                bs, il, ol, slo_tpot_us, M, reeval_reason, st, w_remaining, lif)

        # ─── V1 Formula-based Path (fallback) ────────────────────────────
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

            # Use full iteration latency (pipeline + drain) for SLO check
            drain = self.t_drain_us if M > 1 else 0.0
            t_iter = t_layer * self.num_layers + drain
            calib = self.get_calibration_factor(M)
            if (t_iter * calib) > slo_tpot_us:
                continue

            # Feedback: if slo_urgent triggered repeatedly but predictor keeps
            # choosing a freq that doesn't resolve the SLO violation,
            # force step up one notch from current freq.
            if self._feedback_enabled and reeval_reason == REEVAL_SLO_URGENT:
                if f_a <= st.cur_f_a and f_f <= st.cur_f_f:
                    # Predictor wants same or lower freq despite SLO pressure
                    st.consec_urgent_no_change += 1
                else:
                    st.consec_urgent_no_change = 0

                if st.consec_urgent_no_change >= self._feedback_threshold:
                    # Force step up from CURRENT freq (not predictor's choice)
                    st.consec_urgent_no_change = 0
                    f_a_up = self._next_freq_up(st.cur_f_a)
                    f_f_up = self._next_freq_up(st.cur_f_f)
                    logger.info(
                        "Decode DVFS feedback: %d consecutive urgent-no-change, "
                        "forcing step up f_a=%d→%d f_f=%d→%d",
                        self._feedback_threshold, st.cur_f_a, f_a_up, st.cur_f_f, f_f_up,
                    )
                    self._stats_switch_up += 1
                    self._update_decode_state(f_a_up, f_f_up, bs, switched=True)
                    # Hold the higher freq for N iterations before allowing re-eval
                    st.iters_since_decision = 0
                    st.hold_iters_remaining = self._feedback_hold
                    t_layer_up = self._layer_latency("decode", f_a_up, f_f_up, bs, il, ol, M)
                    e_up = self._layer_energy("decode", f_a_up, f_f_up, bs, il, ol)
                    drain_up = self.t_drain_us if M > 1 else 0.0
                    return DVFSDecision(
                        f_a=f_a_up, f_f=f_f_up,
                        energy_mj=e_up * self.num_layers,
                        latency_us=t_layer_up * self.num_layers + drain_up,
                        switched=True,
                    )
            elif reeval_reason != REEVAL_SLO_URGENT:
                st.consec_urgent_no_change = 0

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
            total_lat = t_layer * self.num_layers + drain
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
        drain = self.t_drain_us if M > 1 else 0.0
        return DVFSDecision(
            f_a=F_MAX, f_f=F_MAX,
            energy_mj=e * self.num_layers,
            latency_us=t_layer * self.num_layers + drain,
            switched=True,
        )

    def _select_freq_decode_coupled(
        self, bs: int, il: int, ol: int,
        slo_tpot_us: float, M: int,
        reeval_reason: int, st, w_remaining: int,
        lif: float = 1.0,
    ) -> DVFSDecision:
        """Select decode freq using V2 coupled iteration-level model.

        The coupled model predicts iteration latency and energy directly,
        accounting for IPC communication, pipeline drain, and micro-batch
        overlap — unlike the V1 formula (max(A,F)*N + drain).
        """
        calib = self.get_calibration_factor(M)

        # Build candidates sorted by energy
        candidates = []
        for f_a, f_f in self._freq_pairs:
            energy = self.predictor.predict_iteration_energy(M, f_a, f_f, bs, il, tp_a=self.tp_a, tp_f=self.tp_f, lif=lif)
            if energy is not None:
                total_e = energy[0] + energy[1]
                candidates.append((total_e, f_a, f_f))
        candidates.sort()

        for total_e, f_a, f_f in candidates:
            lat = self.predictor.predict_iteration_latency(M, f_a, f_f, bs, il, tp_a=self.tp_a, tp_f=self.tp_f, lif=lif)
            if lat is None:
                continue
            if (lat * calib) > slo_tpot_us:
                continue

            # Feedback logic (same as V1)
            if self._feedback_enabled and reeval_reason == REEVAL_SLO_URGENT:
                if f_a <= st.cur_f_a and f_f <= st.cur_f_f:
                    st.consec_urgent_no_change += 1
                else:
                    st.consec_urgent_no_change = 0
                if st.consec_urgent_no_change >= self._feedback_threshold:
                    st.consec_urgent_no_change = 0
                    f_a_up = self._next_freq_up(st.cur_f_a)
                    f_f_up = self._next_freq_up(st.cur_f_f)
                    self._stats_switch_up += 1
                    self._update_decode_state(f_a_up, f_f_up, bs, switched=True)
                    st.iters_since_decision = 0
                    st.hold_iters_remaining = self._feedback_hold
                    lat_up = self.predictor.predict_iteration_latency(
                        M, f_a_up, f_f_up, bs, il, tp_a=self.tp_a, tp_f=self.tp_f, lif=lif) or slo_tpot_us
                    e_up = self.predictor.predict_iteration_energy(
                        M, f_a_up, f_f_up, bs, il, tp_a=self.tp_a, tp_f=self.tp_f, lif=lif)
                    e_up_total = (e_up[0] + e_up[1]) if e_up else 0
                    return DVFSDecision(
                        f_a=f_a_up, f_f=f_f_up,
                        energy_mj=e_up_total, latency_us=lat_up,
                        switched=True,
                    )
            elif reeval_reason != REEVAL_SLO_URGENT:
                st.consec_urgent_no_change = 0

            switched = self._should_switch(
                f_a, f_f, st.cur_f_a, st.cur_f_f, bs, il, ol, w_remaining)
            if switched:
                old_avg = (st.cur_f_a + st.cur_f_f) / 2
                new_avg = (f_a + f_f) / 2
                if new_avg > old_avg:
                    self._stats_switch_up += 1
                else:
                    self._stats_switch_down += 1

            self._update_decode_state(f_a, f_f, bs, switched)
            return DVFSDecision(
                f_a=f_a, f_f=f_f,
                energy_mj=total_e, latency_us=lat,
                switched=switched,
            )

        # Fallback: max frequency
        self._stats_fallback += 1
        self._stats_switch_up += 1
        self._update_decode_state(F_MAX, F_MAX, bs, switched=True)
        lat_max = self.predictor.predict_iteration_latency(
            M, F_MAX, F_MAX, bs, il, tp_a=self.tp_a, tp_f=self.tp_f, lif=lif) or slo_tpot_us
        e_max = self.predictor.predict_iteration_energy(
            M, F_MAX, F_MAX, bs, il, tp_a=self.tp_a, tp_f=self.tp_f, lif=lif)
        e_max_total = (e_max[0] + e_max[1]) if e_max else 0
        return DVFSDecision(
            f_a=F_MAX, f_f=F_MAX,
            energy_mj=e_max_total, latency_us=lat_max,
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
        if self._decode_state.hold_iters_remaining > 0:
            self._decode_state.hold_iters_remaining -= 1

    def reset_decode_state(self):
        """Reset decode window state (e.g. when batch changes completely)."""
        self._decode_state = DecodeWindowState(
            cur_f_a=self._baseline_f_a,
            cur_f_f=self._baseline_f_f,
        )

    def update_baseline(self, f_a: int, f_f: int):
        """Update baseline frequencies from Tier 1 re-planning output.

        Called when Tier 1 produces a new solution with updated baseline
        frequencies. Tier 2 will use these as the starting point for
        subsequent decode windows.
        """
        self._baseline_f_a = f_a
        self._baseline_f_f = f_f
        logger.info("DVFS baseline updated: f_a=%d f_f=%d", f_a, f_f)

    @property
    def is_energy_saving(self) -> bool:
        """True if Tier 2 is currently running below baseline frequency."""
        st = self._decode_state
        return (st.cur_f_a < self._baseline_f_a
                or st.cur_f_f < self._baseline_f_f)

    # ── Utility ────────────────────────────────────────────────────

    def compute_window_size(self, t_iter_avg_us: float) -> int:
        """Compute decode decision window size based on avg iteration time."""
        if t_iter_avg_us <= 0:
            return W_MIN
        w = max(W_MIN, int(10 * T_SWITCH_US / t_iter_avg_us + 0.5))
        self._decode_state.window_size = w
        return w

"""Unified single-knob DVFS controller for PD / Native baselines.

This is the "Tier 2 frequency-scaling layer" for the two baselines compared
against the PDAF+Tier design:

  - PD+Tier (cf. BiScale): each 1P1D instance runs Prefill on one GPU and
    Decode on another. A single GPU runs the *full* transformer layer
    (Attention + FFN sequentially), so there is only one frequency knob per
    instance (unlike AFD which has independent f_A / f_F).
  - Native+Tier (cf. DynamoLLM): each single-GPU TP=1 instance runs the full
    model. Per-batch frequency selection picks the lowest-energy frequency
    that still meets the SLO.

Because our TTFT metric already excludes queue wait (pure processing time =
prefill_finished - prefill_run_batch_start), the BiScale prefill MPC (which
exists only to model how frequency changes the queue evolution) collapses to
a single-step search: frequency only affects the current batch's execution
latency. Both baselines therefore share this one controller.

Latency / energy are reused from the AF profile predictor by summing the A and
F per-layer values at a common frequency f (f_A = f_F = f), which models a
single GPU executing Attention then FFN serially within each layer.
"""

import logging
import time
from dataclasses import dataclass
from typing import Optional

from sglang.srt.energy.af_profile_predictor import AFProfilePredictor, VALID_FREQS

logger = logging.getLogger(__name__)

F_MAX = max(VALID_FREQS)
F_MIN = min(VALID_FREQS)

T_SWITCH_US = 6000.0
E_SWITCH_MJ = 1800.0
F_STEP_MIN = 240
W_MIN = 10

# KV-cache utilization above which decode forces max frequency to drain
# requests faster and avoid OOM / retract (BiScale 4.4.2, design direction A).
KV_UTIL_FORCE_MAX = 0.85

REEVAL_NONE = 0
REEVAL_WINDOW_EXPIRED = 1
REEVAL_BS_CHANGE = 2
REEVAL_SLO_URGENT = 3

SLO_URGENCY_RATIO = 0.7


@dataclass
class UnifiedDVFSDecision:
    f: int
    energy_mj: float
    latency_us: float
    switched: bool = False
    forced_max: bool = False


@dataclass
class _DecodeState:
    cur_f: int = F_MAX
    iters_since_decision: int = 0
    last_bs: int = 0
    window_size: int = 60


class UnifiedDVFSController:
    """Single-knob per-iteration DVFS controller for PD / Native instances.

    Args:
        predictor: AFProfilePredictor with loaded profile models.
        num_layers: Number of transformer layers (Qwen3-32B = 64).
        tp: Tensor-parallel degree of the instance (1 for single-card DP).
        freqs: Candidate SM clock frequencies (MHz).
    """

    def __init__(
        self,
        predictor: AFProfilePredictor,
        num_layers: int = 64,
        tp: int = 1,
        freqs: Optional[list[int]] = None,
        baseline_f: Optional[int] = None,
    ):
        self.predictor = predictor
        self.num_layers = num_layers
        self.tp = tp
        self.freqs = sorted(freqs or VALID_FREQS)
        self._baseline_f = baseline_f or F_MAX
        self._decode_state = _DecodeState(cur_f=self._baseline_f)
        self._cur_f = self._baseline_f

        self._stats_switch_up = 0
        self._stats_switch_down = 0
        self._stats_fallback = 0
        self._stats_kv_forced = 0

    # ── Full-layer latency / energy (A + F at common freq f) ──────────

    def _layer_latency(self, phase: str, f: int, bs: int, il: int,
                       ol: Optional[int]) -> float:
        """Single full-layer latency (us): Attention then FFN, both at freq f."""
        lat_a = self.predictor.predict_latency(phase, "A", self.tp, f, bs, il, ol).value
        lat_f = self.predictor.predict_latency(phase, "F", self.tp, f, bs, il, ol).value
        return lat_a + lat_f

    def _layer_energy(self, phase: str, f: int, bs: int, il: int,
                      ol: Optional[int]) -> float:
        """Single full-layer energy (mJ): Attention + FFN, both at freq f."""
        e_a = self.predictor.predict_energy(phase, "A", self.tp, f, bs, il, ol).value
        e_f = self.predictor.predict_energy(phase, "F", self.tp, f, bs, il, ol).value
        return e_a + e_f

    # ── Prefill: per-batch single-step search ─────────────────────────

    def select_freq_prefill(
        self, bs: int, il: int, slack_us: float,
        remaining_layers: Optional[int] = None,
        floor_freq: Optional[int] = None,
    ) -> UnifiedDVFSDecision:
        """Pick the lowest-energy frequency whose prefill latency fits slack.

        slack_us is the tightest TTFT budget (pure processing time, queue wait
        already excluded) across the batch. Searches all candidate freqs.

        Args:
            floor_freq: If set, skip frequencies below this value. Used to
                ensure the GPU does not drop below a frequency needed by
                interleaved decode batches (decode-aware prefill).
        """
        if remaining_layers is None:
            remaining_layers = self.num_layers

        best: Optional[UnifiedDVFSDecision] = None
        for f in self.freqs:
            if floor_freq is not None and f < floor_freq:
                continue
            try:
                t_layer = self._layer_latency("prefill", f, bs, il, None)
            except (RuntimeError, ValueError):
                continue
            total_lat = t_layer * remaining_layers
            if total_lat > slack_us:
                continue
            try:
                e_layer = self._layer_energy("prefill", f, bs, il, None)
            except (RuntimeError, ValueError):
                continue
            total_e = e_layer * remaining_layers
            if best is None or total_e < best.energy_mj:
                best = UnifiedDVFSDecision(
                    f=f, energy_mj=total_e, latency_us=total_lat)

        if best is None:
            self._stats_fallback += 1
            t_layer = self._layer_latency("prefill", F_MAX, bs, il, None)
            e_layer = self._layer_energy("prefill", F_MAX, bs, il, None)
            best = UnifiedDVFSDecision(
                f=F_MAX,
                energy_mj=e_layer * remaining_layers,
                latency_us=t_layer * remaining_layers,
            )

        best.switched = best.f != self._cur_f
        if best.switched:
            self._cur_f = best.f
        logger.debug("Prefill DVFS: bs=%d il=%d slack=%.0fus -> f=%d "
                     "lat=%.0fus e=%.1fmJ",
                     bs, il, slack_us, best.f, best.latency_us, best.energy_mj)
        return best

    # ── Decode: windowed per-iteration search ─────────────────────────

    def should_reevaluate_decode(
        self, current_bs: int,
        current_tpot_us: float = 0.0,
        slo_tpot_us: float = 0.0,
    ) -> int:
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
        slo_tpot_us: float,
        reeval_reason: int = REEVAL_WINDOW_EXPIRED,
        kv_util: float = 0.0,
    ) -> UnifiedDVFSDecision:
        """Pick lowest-energy frequency whose decode iteration meets TPOT SLO.

        If KV-cache utilization is high, force max frequency to drain requests
        faster and avoid OOM/retract regardless of energy.

        Uses a conservative safety margin (80% of SLO) to account for
        prefill chunk blocking in continuous batching (chunked prefill).
        """
        st = self._decode_state
        # Conservative: only use 80% of the TPOT budget for the iteration itself
        effective_slo = slo_tpot_us * 0.8

        # KV-util OOM guard: override energy-optimal choice with max freq.
        if kv_util > KV_UTIL_FORCE_MAX:
            self._stats_kv_forced += 1
            switched = self._should_switch(F_MAX, st.cur_f, bs, il, ol,
                                           st.window_size, force=True)
            self._update_decode_state(F_MAX, bs, switched)
            try:
                t_iter = self._layer_latency("decode", F_MAX, bs, il, ol) * self.num_layers
                e = self._layer_energy("decode", F_MAX, bs, il, ol) * self.num_layers
            except (RuntimeError, ValueError):
                t_iter, e = 0.0, 0.0
            return UnifiedDVFSDecision(f=F_MAX, energy_mj=e, latency_us=t_iter,
                                       switched=switched, forced_max=True)

        if reeval_reason == REEVAL_WINDOW_EXPIRED:
            w_remaining = st.window_size
        else:
            w_remaining = max(st.window_size - st.iters_since_decision, 1)

        candidates = []
        for f in self.freqs:
            try:
                e = self._layer_energy("decode", f, bs, il, ol)
                candidates.append((e, f))
            except (RuntimeError, ValueError):
                continue
        candidates.sort()

        for e_layer, f in candidates:
            try:
                t_layer = self._layer_latency("decode", f, bs, il, ol)
            except (RuntimeError, ValueError):
                continue
            t_iter = t_layer * self.num_layers
            if t_iter > effective_slo:
                continue
            switched = self._should_switch(f, st.cur_f, bs, il, ol, w_remaining)
            if switched:
                if f > st.cur_f:
                    self._stats_switch_up += 1
                else:
                    self._stats_switch_down += 1
            self._update_decode_state(f, bs, switched)
            return UnifiedDVFSDecision(
                f=f, energy_mj=e_layer * self.num_layers,
                latency_us=t_iter, switched=switched)

        # Fallback: no freq meets SLO -> max frequency.
        self._stats_fallback += 1
        self._stats_switch_up += 1
        switched = st.cur_f != F_MAX
        self._update_decode_state(F_MAX, bs, switched)
        try:
            t_iter = self._layer_latency("decode", F_MAX, bs, il, ol) * self.num_layers
            e = self._layer_energy("decode", F_MAX, bs, il, ol) * self.num_layers
        except (RuntimeError, ValueError):
            t_iter, e = 0.0, 0.0
        return UnifiedDVFSDecision(f=F_MAX, energy_mj=e, latency_us=t_iter,
                                   switched=switched)

    def _should_switch(self, f_new: int, f_cur: int, bs: int, il: int,
                       ol: int, w_remaining: int, force: bool = False) -> bool:
        """Lazy switching: only switch if energy savings justify the cost."""
        if f_new == f_cur:
            return False
        if force:
            return True
        if abs(f_new - f_cur) < F_STEP_MIN:
            return False
        try:
            e_cur = self._layer_energy("decode", f_cur, bs, il, ol)
            e_new = self._layer_energy("decode", f_new, bs, il, ol)
        except (RuntimeError, ValueError):
            return True
        savings = (e_cur - e_new) * self.num_layers * max(w_remaining, 1)
        return savings > E_SWITCH_MJ

    def _update_decode_state(self, f: int, bs: int, switched: bool):
        st = self._decode_state
        if switched:
            st.cur_f = f
            self._cur_f = f
        st.last_bs = bs
        st.iters_since_decision = 0

    def tick_decode_iteration(self):
        self._decode_state.iters_since_decision += 1

    def compute_window_size(self, t_iter_avg_us: float) -> int:
        if t_iter_avg_us <= 0:
            return W_MIN
        w = max(W_MIN, int(10 * T_SWITCH_US / t_iter_avg_us + 0.5))
        self._decode_state.window_size = w
        return w

    # ── Decode-aware floor for prefill frequency selection ────────────

    def compute_decode_floor_freq(
        self, decode_bs: int, decode_il: int, decode_ol: int,
        slo_tpot_us: float,
        prefill_bs: int = 1,
        prefill_il: int = 8192,
    ) -> int:
        """Find minimum prefill frequency that limits TPOT degradation.

        In Native mode with chunked prefill, decode requests are blocked for
        the entire duration of a prefill chunk. We cannot make chunk_time <
        TPOT_SLO (chunks are inherently longer), but we can limit the
        degradation relative to max frequency.

        Strategy: allow at most 10% latency increase over the baseline (max
        freq) chunk time. This is very conservative to ensure TPOT SLO
        compliance: even a small increase in prefill chunk time directly
        blocks decode iterations and causes TPOT violations.
        """
        try:
            t_baseline = self._layer_latency(
                "prefill", F_MAX, prefill_bs, prefill_il, None) * self.num_layers
        except (RuntimeError, ValueError):
            return F_MAX

        max_allowed = t_baseline * 1.1

        for f in self.freqs:
            try:
                t_chunk = self._layer_latency(
                    "prefill", f, prefill_bs, prefill_il, None) * self.num_layers
            except (RuntimeError, ValueError):
                continue
            if t_chunk <= max_allowed:
                return f
        return F_MAX

    @property
    def cur_freq(self) -> int:
        return self._cur_f

    @property
    def is_energy_saving(self) -> bool:
        return self._cur_f < self._baseline_f

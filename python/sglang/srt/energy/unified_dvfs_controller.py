"""Unified single-knob DVFS controller for PD / Native baselines.

This is the "Tier 2 frequency-scaling layer" for the two baselines compared
against the PDAF+Tier design:

  - PD+Tier (cf. BiScale): each 1P1D instance runs Prefill on one GPU and
    Decode on another. A single GPU runs the *full* transformer layer
    (Attention + FFN sequentially), so there is only one frequency knob per
    instance (unlike AFD which has independent f_A / f_F).
  - Native+Tier (cf. DynamoLLM): each single-GPU TP=1 instance runs the full
    model. Per-batch frequency selection picks the lowest-energy (default) or
    lowest-frequency frequency that still meets the SLO.

With ``policy="unified"`` (DynamoLLM / legacy PD tier), prefill uses a
single-step min-energy or min-freq search. With ``policy="biscale"`` (BiScale
Tier 2), prefill uses MPC + Algorithm 1 (greedy horizon search, min average
power under TTFT including queue wait) and decode uses ascending min-frequency
under TBT (BiScale §4.4.1–4.4.2).

Latency / energy are reused from the AF profile predictor by summing the A and
F per-layer values at a common frequency f (f_A = f_F = f), which models a
single GPU executing Attention then FFN serially within each layer.
"""

import itertools
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

# KV-cache utilization above which decode forces max frequency to drain
# requests faster and avoid OOM / retract (BiScale 4.4.2, design direction A).
KV_UTIL_FORCE_MAX = 0.85

REEVAL_NONE = 0
REEVAL_WINDOW_EXPIRED = 1
REEVAL_BS_CHANGE = 2
REEVAL_SLO_URGENT = 3

SLO_URGENCY_RATIO = 0.7

# Adaptive margin parameters for load-aware SLO budgeting
# Base margin is TP-aware: fewer instances (higher TP) get tighter margins
# because each instance handles more load with less queuing headroom.
# For TP>=8: decode keeps max freq (prefill-only DVFS) since continuous batching
# prefill blocking makes decode DVFS unstable with so few instances.
_TP_BASE_MARGINS = {1: 0.80, 2: 0.75, 4: 0.55, 8: 0.45}
DECODE_DVFS_DISABLED_TP = 8  # TP >= this -> no decode freq reduction
MIN_SAFETY_MARGIN = 0.30
TPOT_HISTORY_LEN = 5
TREND_TIGHTEN_FACTOR = 0.1  # margin penalty per consecutive TPOT increase
EMERGENCY_BRAKE_RATIO = 0.65  # force max freq if observed TPOT > this * SLO

# BiScale Tier 2 (paper §4.4)
MPC_HORIZON_K = 8
BISCALE_LATENCY_MARGIN = 0.95  # 5% SLO headroom (§4.6)


@dataclass
class PrefillBatchSpec:
    """One projected prefill batch in the MPC horizon."""
    bs: int
    max_il: int
    wait_us: list[float] = field(default_factory=list)


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
    # TPOT trend tracking for adaptive margin
    tpot_history: list = None
    consecutive_increases: int = 0
    # Online calibration: ratio of observed/predicted latency (EMA)
    calibration_factor: float = 1.0

    def __post_init__(self):
        if self.tpot_history is None:
            self.tpot_history = []


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
        objective: str = "energy",
        policy: str = "unified",
    ):
        self.predictor = predictor
        self.num_layers = num_layers
        self.tp = tp
        self.freqs = sorted(freqs or VALID_FREQS)
        self._freqs_desc = sorted(self.freqs, reverse=True)
        self._baseline_f = baseline_f or F_MAX
        self._objective = objective if objective in ("energy", "freq") else "energy"
        self._policy = policy if policy in ("unified", "biscale") else "unified"
        self._decode_state = _DecodeState(cur_f=self._baseline_f)
        self._cur_f = self._baseline_f
        # TP-aware base safety margin
        self._base_margin = _TP_BASE_MARGINS.get(tp, 0.80 / (tp ** 0.3))

        self._stats_switch_up = 0
        self._stats_switch_down = 0
        self._stats_fallback = 0
        self._stats_kv_forced = 0
        self._last_decode_predicted_us = 0.0

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

    def _batch_total_latency(self, phase: str, f: int, bs: int, il: int,
                             ol: Optional[int] = None) -> float:
        return self._layer_latency(phase, f, bs, il, ol) * self.num_layers

    def _batch_total_energy(self, phase: str, f: int, bs: int, il: int,
                            ol: Optional[int] = None) -> float:
        return self._layer_energy(phase, f, bs, il, ol) * self.num_layers

    # ── BiScale prefill MPC (Algorithm 1) ─────────────────────────────

    def _biscale_meets_ttft(
        self, freqs: list[int], batches: list[PrefillBatchSpec], slo_us: float,
    ) -> bool:
        effective_slo = slo_us * BISCALE_LATENCY_MARGIN
        t = 0.0
        for f, batch in zip(freqs, batches):
            if batch.bs <= 0:
                continue
            try:
                lat = self._batch_total_latency("prefill", f, batch.bs, batch.max_il)
            except (RuntimeError, ValueError):
                return False
            t_end = t + lat
            waits = batch.wait_us if batch.wait_us else [0.0] * batch.bs
            for wait_us in waits:
                if wait_us + t_end > effective_slo:
                    return False
            t = t_end
        return True

    def _biscale_avg_power_score(
        self, freqs: list[int], batches: list[PrefillBatchSpec],
    ) -> float:
        total_lat = 0.0
        total_e = 0.0
        for f, batch in zip(freqs, batches):
            if batch.bs <= 0:
                continue
            try:
                lat = self._batch_total_latency("prefill", f, batch.bs, batch.max_il)
                e = self._batch_total_energy("prefill", f, batch.bs, batch.max_il)
            except (RuntimeError, ValueError):
                return float("inf")
            total_lat += lat
            total_e += e
        if total_lat <= 0:
            return float("inf")
        return total_e / total_lat

    def _mutate_freq_assignments(
        self, freqs_opt: list[int], from_f: int, to_options: list[int],
    ) -> list[list[int]]:
        indices = [i for i, f in enumerate(freqs_opt) if f == from_f]
        if not indices:
            return []
        mutations: list[list[int]] = []
        for combo in itertools.product(to_options, repeat=len(indices)):
            new = list(freqs_opt)
            for idx, new_f in zip(indices, combo):
                new[idx] = new_f
            if new != list(freqs_opt):
                mutations.append(new)
        return mutations

    def _algorithm1_greedy(
        self, batches: list[PrefillBatchSpec], slo_us: float,
    ) -> list[int]:
        """BiScale Algorithm 1: greedy frequency selection over MPC horizon."""
        k = len(batches)
        if k == 0:
            return [F_MAX]
        freqs_avail = self._freqs_desc
        n = len(freqs_avail)
        freqs_opt = [freqs_avail[0]] * k
        if n < 2:
            return freqs_opt

        for s in range(1, n - 1):
            from_f = freqs_avail[s - 1]
            to_opts = [freqs_avail[s - 1], freqs_avail[s], freqs_avail[s + 1]]
            candidates: list[list[int]] = []
            for mutated in self._mutate_freq_assignments(freqs_opt, from_f, to_opts):
                if self._biscale_meets_ttft(mutated, batches, slo_us):
                    candidates.append(mutated)
            if not candidates:
                break
            freqs_opt = min(
                candidates,
                key=lambda c: self._biscale_avg_power_score(c, batches),
            )
        return freqs_opt

    def select_freq_prefill_biscale(
        self,
        batches: list[PrefillBatchSpec],
        slo_ttft_us: float,
        slack_us: float,
        bs: int,
        il: int,
        remaining_layers: Optional[int] = None,
    ) -> UnifiedDVFSDecision:
        """BiScale prefill: MPC horizon + Algorithm 1; apply freqs[0] now."""
        if remaining_layers is None:
            remaining_layers = self.num_layers
        if not batches:
            batches = [PrefillBatchSpec(bs=bs, max_il=il, wait_us=[0.0] * max(bs, 1))]

        horizon = batches[:MPC_HORIZON_K]
        freqs = self._algorithm1_greedy(horizon, slo_ttft_us)
        f = freqs[0]

        try:
            t_layer = self._layer_latency("prefill", f, bs, il, None)
            e_layer = self._layer_energy("prefill", f, bs, il, None)
        except (RuntimeError, ValueError):
            f = F_MAX
            t_layer = self._layer_latency("prefill", f, bs, il, None)
            e_layer = self._layer_energy("prefill", f, bs, il, None)

        total_lat = t_layer * remaining_layers
        if total_lat > slack_us * BISCALE_LATENCY_MARGIN:
            f = F_MAX
            t_layer = self._layer_latency("prefill", f, bs, il, None)
            e_layer = self._layer_energy("prefill", f, bs, il, None)
            total_lat = t_layer * remaining_layers

        best = UnifiedDVFSDecision(
            f=f,
            energy_mj=e_layer * remaining_layers,
            latency_us=total_lat,
        )
        best.switched = best.f != self._cur_f
        if best.switched:
            self._cur_f = best.f
        logger.debug(
            "BiScale prefill MPC: horizon=%d -> f=%d lat=%.0fus",
            len(horizon), best.f, best.latency_us,
        )
        return best

    # ── Prefill: per-batch single-step search ─────────────────────────

    def select_freq_prefill(
        self, bs: int, il: int, slack_us: float,
        remaining_layers: Optional[int] = None,
        floor_freq: Optional[int] = None,
        horizon_batches: Optional[list[PrefillBatchSpec]] = None,
        slo_ttft_us: Optional[float] = None,
    ) -> UnifiedDVFSDecision:
        """Pick frequency for prefill under SLO.

        policy=biscale: MPC + Algorithm 1 (min avg power under TTFT w/ queue).
        policy=unified + objective=energy: lowest-energy among SLO-feasible freqs.
        policy=unified + objective=freq: lowest frequency among SLO-feasible freqs.
        """
        if self._policy == "biscale":
            slo = slo_ttft_us if slo_ttft_us is not None else slack_us
            return self.select_freq_prefill_biscale(
                batches=horizon_batches or [],
                slo_ttft_us=slo,
                slack_us=slack_us,
                bs=bs,
                il=il,
                remaining_layers=remaining_layers,
            )

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
            cand = UnifiedDVFSDecision(
                f=f, energy_mj=total_e, latency_us=total_lat)
            if best is None:
                best = cand
            elif self._objective == "freq":
                if cand.f < best.f:
                    best = cand
            elif cand.energy_mj < best.energy_mj:
                best = cand

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
        # Trigger re-evaluation if TPOT approaches the TP-aware margin threshold
        # This ensures high-TP configs react faster to rising latency
        urgency_threshold = slo_tpot_us * min(SLO_URGENCY_RATIO, self._base_margin)
        if (current_tpot_us > 0 and slo_tpot_us > 0
                and current_tpot_us > urgency_threshold):
            return REEVAL_SLO_URGENT
        return REEVAL_NONE

    def select_freq_decode_biscale(
        self, bs: int, il: int, ol: int, slo_tpot_us: float,
        kv_util: float = 0.0, observed_tpot_us: float = 0.0,
    ) -> UnifiedDVFSDecision:
        """BiScale decode: ascending min-freq under TBT + KV-util guard (§4.4.2)."""
        st = self._decode_state
        effective_slo = slo_tpot_us * BISCALE_LATENCY_MARGIN

        if kv_util > KV_UTIL_FORCE_MAX:
            self._stats_kv_forced += 1
            return self._decode_force_max(bs, il, ol, st)

        if (observed_tpot_us > 0 and self._last_decode_predicted_us > 0
                and observed_tpot_us > self._last_decode_predicted_us * 1.05):
            logger.info(
                "BiScale decode under-prediction: obs=%.0fus pred=%.0fus -> F_MAX",
                observed_tpot_us, self._last_decode_predicted_us,
            )
            return self._decode_force_max(bs, il, ol, st)

        for f in self.freqs:
            try:
                t_iter = self._batch_total_latency("decode", f, bs, il, ol)
                e = self._batch_total_energy("decode", f, bs, il, ol)
            except (RuntimeError, ValueError):
                continue
            if t_iter > effective_slo:
                continue
            self._last_decode_predicted_us = t_iter
            switched = f != st.cur_f
            if switched:
                if f > st.cur_f:
                    self._stats_switch_up += 1
                else:
                    self._stats_switch_down += 1
                st.cur_f = f
                self._cur_f = f
            st.last_bs = bs
            st.iters_since_decision = 0
            logger.debug(
                "BiScale decode: bs=%d -> f=%d lat=%.0fus", bs, f, t_iter,
            )
            return UnifiedDVFSDecision(
                f=f, energy_mj=e, latency_us=t_iter, switched=switched,
            )

        self._stats_fallback += 1
        return self._decode_force_max(bs, il, ol, st)

    def _decode_force_max(
        self, bs: int, il: int, ol: int, st: _DecodeState,
    ) -> UnifiedDVFSDecision:
        switched = st.cur_f != F_MAX
        if switched:
            self._stats_switch_up += 1
            st.cur_f = F_MAX
            self._cur_f = F_MAX
        st.last_bs = bs
        st.iters_since_decision = 0
        st.consecutive_increases = 0
        try:
            t_iter = self._batch_total_latency("decode", F_MAX, bs, il, ol)
            e = self._batch_total_energy("decode", F_MAX, bs, il, ol)
        except (RuntimeError, ValueError):
            t_iter, e = 0.0, 0.0
        self._last_decode_predicted_us = t_iter
        return UnifiedDVFSDecision(
            f=F_MAX, energy_mj=e, latency_us=t_iter,
            switched=switched, forced_max=True,
        )

    def select_freq_decode(
        self, bs: int, il: int, ol: int,
        slo_tpot_us: float,
        reeval_reason: int = REEVAL_WINDOW_EXPIRED,
        kv_util: float = 0.0,
        observed_tpot_us: float = 0.0,
    ) -> UnifiedDVFSDecision:
        """Pick frequency for decode under TPOT SLO.

        policy=biscale: ascending min-freq under TBT (BiScale §4.4.2).
        policy=unified + objective=energy: lowest-energy among SLO-feasible freqs.
        policy=unified + objective=freq: lowest frequency among SLO-feasible freqs.
        """
        if self._policy == "biscale":
            return self.select_freq_decode_biscale(
                bs=bs, il=il, ol=ol, slo_tpot_us=slo_tpot_us,
                kv_util=kv_util, observed_tpot_us=observed_tpot_us,
            )

        st = self._decode_state

        # High-TP fast path: for TP >= threshold, only allow decode DVFS
        # when batch is small enough (low load = safe to reduce frequency).
        # At high load (large batch), skip decode DVFS to avoid instability.
        if self.tp >= DECODE_DVFS_DISABLED_TP and bs > 64:
            try:
                t_iter = self._layer_latency("decode", F_MAX, bs, il, ol) * self.num_layers
                e = self._layer_energy("decode", F_MAX, bs, il, ol) * self.num_layers
            except (RuntimeError, ValueError):
                t_iter, e = 0.0, 0.0
            switched = st.cur_f != F_MAX
            if switched:
                self._update_decode_state(F_MAX, bs, switched)
            return UnifiedDVFSDecision(f=F_MAX, energy_mj=e, latency_us=t_iter,
                                       switched=switched, forced_max=True)

        # Track TPOT trend for adaptive margin
        if observed_tpot_us > 0:
            st.tpot_history.append(observed_tpot_us)
            if len(st.tpot_history) > TPOT_HISTORY_LEN:
                st.tpot_history = st.tpot_history[-TPOT_HISTORY_LEN:]
            if len(st.tpot_history) >= 2:
                if st.tpot_history[-1] > st.tpot_history[-2] * 1.02:
                    st.consecutive_increases += 1
                elif st.tpot_history[-1] < st.tpot_history[-2] * 0.95:
                    st.consecutive_increases = max(0, st.consecutive_increases - 2)
                else:
                    st.consecutive_increases = max(0, st.consecutive_increases - 1)
            else:
                st.consecutive_increases = 0

        # Adaptive safety margin: TP-aware base + tighten when TPOT trending up
        trend_penalty = st.consecutive_increases * TREND_TIGHTEN_FACTOR
        margin = max(MIN_SAFETY_MARGIN, self._base_margin - trend_penalty)
        effective_slo = slo_tpot_us * margin

        # Emergency brake: if observed TPOT already exceeds threshold, force max
        # freq immediately to break the positive feedback loop
        if observed_tpot_us > slo_tpot_us * EMERGENCY_BRAKE_RATIO:
            logger.info("DVFS emergency brake: observed_tpot=%.0fus > %.0f%% SLO "
                        "(%.0fus), forcing F_MAX", observed_tpot_us,
                        EMERGENCY_BRAKE_RATIO * 100, slo_tpot_us)
            switched = st.cur_f != F_MAX
            self._update_decode_state(F_MAX, bs, switched)
            st.consecutive_increases = 0
            try:
                t_iter = self._layer_latency("decode", F_MAX, bs, il, ol) * self.num_layers
                e = self._layer_energy("decode", F_MAX, bs, il, ol) * self.num_layers
            except (RuntimeError, ValueError):
                t_iter, e = 0.0, 0.0
            return UnifiedDVFSDecision(f=F_MAX, energy_mj=e, latency_us=t_iter,
                                       switched=switched, forced_max=True)

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

        # Load-factor ceiling: compute iter latency at max freq as baseline.
        # If already using a large fraction of SLO at max freq, limit how much
        # we can reduce frequency (queue is already under pressure).
        try:
            t_iter_maxfreq = (self._layer_latency("decode", F_MAX, bs, il, ol)
                              * self.num_layers)
        except (RuntimeError, ValueError):
            t_iter_maxfreq = 0.0

        # Online calibration: update calibration_factor from observed vs predicted
        if observed_tpot_us > 0 and t_iter_maxfreq > 0 and st.cur_f == F_MAX:
            # Only calibrate when at max freq (direct comparison is cleanest)
            ratio = observed_tpot_us / t_iter_maxfreq
            alpha = 0.3  # EMA smoothing
            st.calibration_factor = alpha * ratio + (1 - alpha) * st.calibration_factor
            st.calibration_factor = max(1.0, min(st.calibration_factor, 4.0))
        elif observed_tpot_us > 0 and t_iter_maxfreq > 0:
            # At non-max freq, only increase calibration if observed >> expected
            predicted_at_cur = t_iter_maxfreq * (F_MAX / max(st.cur_f, 1))
            if observed_tpot_us > predicted_at_cur * 1.3:
                st.calibration_factor = min(4.0, st.calibration_factor * 1.1)

        cal = st.calibration_factor

        load_factor = (t_iter_maxfreq * cal) / slo_tpot_us if slo_tpot_us > 0 else 0.0
        # At high load factor, further tighten the effective SLO ceiling
        if load_factor > 0.15:
            load_ceiling = slo_tpot_us * min(0.7, (1.0 - load_factor) * 1.2)
            effective_slo = min(effective_slo, load_ceiling)

        candidates = []
        for f in self.freqs:
            try:
                e = self._layer_energy("decode", f, bs, il, ol)
                candidates.append((f, e))
            except (RuntimeError, ValueError):
                continue
        if self._objective == "freq":
            candidates.sort(key=lambda x: x[0])
        else:
            candidates.sort(key=lambda x: x[1])

        for f, e_layer in candidates:
            try:
                t_layer = self._layer_latency("decode", f, bs, il, ol)
            except (RuntimeError, ValueError):
                continue
            # Apply calibration factor to model prediction
            t_iter = t_layer * self.num_layers * cal
            if t_iter > effective_slo:
                continue
            switched = self._should_switch(f, st.cur_f, bs, il, ol, w_remaining)
            if switched:
                if f > st.cur_f:
                    self._stats_switch_up += 1
                else:
                    self._stats_switch_down += 1
            self._update_decode_state(f, bs, switched)
            logger.debug("Decode DVFS: bs=%d margin=%.2f trend=%d -> f=%d "
                         "lat=%.0fus e=%.1fmJ",
                         bs, margin, st.consecutive_increases, f, t_iter,
                         e_layer * self.num_layers)
            return UnifiedDVFSDecision(
                f=f, energy_mj=e_layer * self.num_layers,
                latency_us=t_iter, switched=switched)

        # Fallback: no freq meets SLO -> max frequency.
        self._stats_fallback += 1
        self._stats_switch_up += 1
        switched = st.cur_f != F_MAX
        self._update_decode_state(F_MAX, bs, switched)
        st.consecutive_increases = 0  # reset trend after forcing max
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

    def notify_freq_override(self, f: int):
        """Notify that hardware freq was changed externally (e.g. by prefill).

        This keeps _decode_state.cur_f in sync with real hardware so that
        subsequent decode decisions correctly detect whether a switch is needed.
        """
        self._decode_state.cur_f = f
        self._cur_f = f

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
        the entire duration of a prefill chunk. The chunk runs once per TPOT
        cycle at most, so the constraint is:

            chunk_time(f) + decode_iteration(F_MAX) <= TPOT_SLO

        This allows aggressive prefill frequency reduction while ensuring
        decode iterations still meet the SLO. Falls back to the old 10%
        relative threshold if TPOT_SLO is not provided or too tight.
        """
        try:
            t_baseline = self._layer_latency(
                "prefill", F_MAX, prefill_bs, prefill_il, None) * self.num_layers
        except (RuntimeError, ValueError):
            return F_MAX

        # Compute decode iteration time at max frequency
        try:
            t_decode_fmax = self._layer_latency(
                "decode", F_MAX, decode_bs, decode_il, decode_ol) * self.num_layers
        except (RuntimeError, ValueError):
            t_decode_fmax = 0.0

        # Primary constraint: chunk + decode_iter <= TPOT_SLO * safety_margin
        # Use tighter margin for high-TP (fewer instances = more contention)
        if self.tp >= 8:
            tp_margin = 0.7
        elif self.tp >= 4:
            tp_margin = 0.8
        else:
            tp_margin = 0.9
        if slo_tpot_us > 0 and t_decode_fmax > 0:
            max_chunk = slo_tpot_us * tp_margin - t_decode_fmax
        else:
            max_chunk = t_baseline * 1.1  # fallback to old 10% rule

        # Additional constraint for high-TP: limit prefill chunk slowdown factor.
        # At high TP, prefill chunks block decode for their entire duration.
        # Use load-adaptive limit: generous at low load, tight at high load.
        if self.tp >= 8:
            # With tight TPOT SLO, let the SLO constraint itself limit prefill
            load_ratio = min(1.0, decode_bs / 16.0) if decode_bs > 0 else 0.0
            slowdown_limit = 2.0 - load_ratio * 0.5  # generous: 2.0x -> 1.5x
            max_chunk = min(max_chunk, t_baseline * slowdown_limit)
        elif self.tp >= 4:
            max_chunk = min(max_chunk, t_baseline * 1.4)

        # Ensure we don't go below the baseline (would mean SLO is too tight)
        if max_chunk < t_baseline:
            return F_MAX

        for f in self.freqs:
            try:
                t_chunk = self._layer_latency(
                    "prefill", f, prefill_bs, prefill_il, None) * self.num_layers
            except (RuntimeError, ValueError):
                continue
            if t_chunk <= max_chunk:
                return f
        return F_MAX

    @property
    def cur_freq(self) -> int:
        return self._cur_f

    @property
    def is_energy_saving(self) -> bool:
        return self._cur_f < self._baseline_f

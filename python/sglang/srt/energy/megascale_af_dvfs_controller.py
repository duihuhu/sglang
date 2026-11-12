"""Joint P+D DVFS policy for AF-only MegaScale experiments.

Unlike PD/PDAF policies, a MegaScale AF replica has one Attention GPU pool and
one FFN GPU pool shared by prefill and decode. This controller therefore emits
one (f_A, f_F) pair after jointly evaluating PA+DA and PF+DF work.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from sglang.srt.energy.af_dvfs_controller import DVFSDecision, F_MAX


@dataclass(frozen=True)
class PhaseWorkload:
    bs: int
    il: int
    ol: int = 1


@dataclass(frozen=True)
class JointDecisionDetail:
    prefill_latency_us: float
    decode_latency_us: float
    effective_tpot_us: float
    prefill_energy_mj: float
    decode_energy_mj: float
    decode_iterations: float


class MegaScaleAFDVFSController:
    """Select shared A/F clocks for mixed prefill and decode work.

    Predictions are anchored to the observed TPOT and to max-frequency prefill
    latency. This avoids treating prefill slack as free capacity: any slowdown
    of PA/PF is charged directly to resident decode requests.
    """

    def __init__(
        self,
        predictor,
        num_layers: int,
        tp_a: int,
        tp_f: int,
        t_comm_us: float,
        t_drain_us: float = 18000.0,
        freqs: Optional[list[int]] = None,
        frequency_floor_mhz: int = 690,
        tpot_safety_factor: float = 0.8,
        prefill_safety_factor: float = 0.8,
        interference_factor: float = 1.0,
        coupled_frequency: bool = False,
    ):
        self.predictor = predictor
        self.num_layers = num_layers
        self.tp_a = tp_a
        self.tp_f = tp_f
        self.t_comm_us = t_comm_us
        self.t_drain_us = t_drain_us
        self.freqs = freqs or [210, 450, 690, 930, 1170, 1410]
        self.frequency_floor_mhz = frequency_floor_mhz
        self.tpot_safety_factor = tpot_safety_factor
        self.prefill_safety_factor = prefill_safety_factor
        self.interference_factor = interference_factor
        self.coupled_frequency = coupled_frequency
        self.cur_f_a = F_MAX
        self.cur_f_f = F_MAX
        self.last_detail: Optional[JointDecisionDetail] = None

    def _phase_metrics(
        self, phase: str, work: PhaseWorkload, f_a: int, f_f: int,
        micro_batch: int, cache: Optional[dict] = None,
    ) -> tuple[float, float]:
        ol = None if phase == "prefill" else max(work.ol, 1)
        cache = cache if cache is not None else {}

        def predict(metric: str, op: str, tp: int, freq: int) -> float:
            key = (metric, phase, op, tp, freq, work.bs, work.il, ol)
            if key not in cache:
                function = (self.predictor.predict_latency
                            if metric == "latency" else self.predictor.predict_energy)
                cache[key] = function(phase, op, tp, freq, work.bs, work.il, ol).value
            return cache[key]

        lat_a = predict("latency", "A", self.tp_a, f_a)
        lat_f = predict("latency", "F", self.tp_f, f_f)
        energy_a = predict("energy", "A", self.tp_a, f_a)
        energy_f = predict("energy", "F", self.tp_f, f_f)
        if micro_batch > 1:
            layer_latency = max(lat_a, lat_f) + self.t_comm_us / micro_batch
            drain = self.t_drain_us
        else:
            layer_latency = lat_a + lat_f + self.t_comm_us
            drain = 0.0
        return (
            layer_latency * self.num_layers + drain,
            (energy_a + energy_f) * self.num_layers,
        )

    def select_joint(
        self,
        *,
        prefill: Optional[PhaseWorkload],
        decode: Optional[PhaseWorkload],
        ttft_slack_us: float,
        tpot_slo_us: float,
        micro_batch: int,
        observed_tpot_us: float = 0.0,
        prefill_interval_us: float = 0.0,
    ) -> DVFSDecision:
        if prefill is None and decode is None:
            raise ValueError("joint DVFS requires prefill and/or decode work")

        prediction_cache: dict = {}
        p_max_lat = p_max_energy = 0.0
        if prefill is not None:
            p_max_lat, p_max_energy = self._phase_metrics(
                "prefill", prefill, F_MAX, F_MAX, micro_batch, prediction_cache
            )
        d_cur_lat = d_cur_energy = 0.0
        if decode is not None:
            d_cur_lat, d_cur_energy = self._phase_metrics(
                "decode", decode, self.cur_f_a, self.cur_f_f, micro_batch,
                prediction_cache,
            )

        base_tpot = observed_tpot_us if observed_tpot_us > 0 else d_cur_lat
        decode_iterations = 1.0
        if prefill is not None and decode is not None and prefill_interval_us > 0:
            decode_iterations = max(prefill_interval_us / max(base_tpot, 1.0), 1.0)
        elif prefill is not None and decode is not None:
            decode_iterations = 1.0

        best = None
        best_detail = None
        ttft_budget = max(ttft_slack_us * self.prefill_safety_factor, 0.0)
        tpot_budget = max(tpot_slo_us * self.tpot_safety_factor, 0.0)
        for f_a in self.freqs:
            if f_a < self.frequency_floor_mhz:
                continue
            for f_f in self.freqs:
                if f_f < self.frequency_floor_mhz:
                    continue
                if self.coupled_frequency and f_f != f_a:
                    continue
                try:
                    p_lat = p_energy = 0.0
                    if prefill is not None:
                        p_lat, p_energy = self._phase_metrics(
                            "prefill", prefill, f_a, f_f, micro_batch,
                            prediction_cache,
                        )
                        if p_lat > ttft_budget:
                            continue
                    d_lat = d_energy = 0.0
                    effective_tpot = 0.0
                    if decode is not None:
                        d_lat, d_energy = self._phase_metrics(
                            "decode", decode, f_a, f_f, micro_batch,
                            prediction_cache,
                        )
                        decode_scale = d_lat / max(d_cur_lat, 1.0)
                        effective_tpot = base_tpot * decode_scale
                        if prefill is not None:
                            effective_tpot += max(p_lat - p_max_lat, 0.0) * self.interference_factor
                        if effective_tpot > tpot_budget:
                            continue
                except (RuntimeError, ValueError):
                    continue

                total_energy = p_energy + decode_iterations * d_energy
                detail = JointDecisionDetail(
                    prefill_latency_us=p_lat,
                    decode_latency_us=d_lat,
                    effective_tpot_us=effective_tpot,
                    prefill_energy_mj=p_energy,
                    decode_energy_mj=d_energy,
                    decode_iterations=decode_iterations,
                )
                if best is None or total_energy < best[0]:
                    best = (total_energy, f_a, f_f, max(p_lat, d_lat))
                    best_detail = detail

        if best is None:
            f_a = f_f = F_MAX
            p_lat = p_energy = d_lat = d_energy = 0.0
            if prefill is not None:
                p_lat, p_energy = self._phase_metrics(
                    "prefill", prefill, f_a, f_f, micro_batch,
                    prediction_cache,
                )
            if decode is not None:
                d_lat, d_energy = self._phase_metrics(
                    "decode", decode, f_a, f_f, micro_batch,
                    prediction_cache,
                )
            best = (p_energy + decode_iterations * d_energy, f_a, f_f, max(p_lat, d_lat))
            best_detail = JointDecisionDetail(
                p_lat, d_lat, base_tpot, p_energy, d_energy, decode_iterations
            )

        total_energy, f_a, f_f, latency = best
        switched = f_a != self.cur_f_a or f_f != self.cur_f_f
        self.cur_f_a, self.cur_f_f = f_a, f_f
        self.last_detail = best_detail
        return DVFSDecision(
            f_a=f_a,
            f_f=f_f,
            energy_mj=total_energy,
            latency_us=latency,
            switched=switched,
        )

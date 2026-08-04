"""Tier 1 Solver V2 — SLO-only energy minimization with idle-aware modeling.

Key improvements over v1:
  - Per-GPU-group Pareto (prevents high-GPU configs from dominating low-GPU ones)
  - Removes throughput constraint (k * bp ≤ (1+alpha)*demand)
  - Adds idle energy from P/D pipeline imbalance
  - Adds optional architectural constraints: k_P=1, tp_A+tp_F ≤ max_gpu_per_pair

Algorithm sketch
  1. Enumerate all feasible (tp, freq) configs for each of the 4 pools
     (PA, PF, DA, DF), applying hard pruning (SLO, memory, balance).
  2. Pareto-filter per GPU count (tp_A, tp_F), preserving GPU-efficient configs.
  3. For each viable (tp_P, tp_D) and (k_P, k_D), compute min-energy + idle,
     pick global minimum under GPU budget.

Input
  - G: total GPU budget
  - workload: WorkloadProfile
  - slo: SLOConfig
  - profile_table: ProfileTable instance

Output
  Tier1Solution with k_P, k_D, tp_*, f_*, total_energy, gpu_used.
"""
from __future__ import annotations

import logging
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Optional

from sglang.srt.energy.profile_table import ProfileTable

logger = logging.getLogger(__name__)

_VALID_TP = [1, 2, 4, 8]
_VALID_FREQS = [210, 450, 690, 930, 1170, 1410]
_BETA_BALANCE = 0.8
_BYTES_PER_ELEMENT = 2
_DEFAULT_M = 2
_P_IDLE_W = 80.0

# ── Data classes (unchanged from v1) ──

@dataclass
class WorkloadProfile:
    lambda_prefill: float = 10.0
    n_active_decode: int = 32
    il_rep_p: int = 1024
    bs_avg_p: int = 8
    il_rep_d: int = 512
    ol_rep_d: int = 256
    bs_avg_d: int = 16
    mean_ol: Optional[float] = None
    decode_token_rate_demand: Optional[float] = None

@dataclass
class SLOConfig:
    ttft_ms: float = 500.0
    tpot_ms: float = 50.0

@dataclass
class _OpCandidate:
    tp: int; freq: int; t_us: float; e_mj: float

@dataclass
class _AFPairCandidate:
    tp_a: int; freq_a: int; tp_f: int; freq_f: int
    t_a_us: float; t_f_us: float; e_a_mj: float; e_f_mj: float
    t_pair_us: float = 0.0
    e_pair_mj: float = 0.0
    throughput_per_pair: float = 0.0

    def __post_init__(self):
        self.e_pair_mj = self.e_a_mj + self.e_f_mj

@dataclass
class Tier1Solution:
    k_p: int = 1; k_d: int = 1
    tp_pa: int = 1; f_pa: int = 1410
    tp_pf: int = 1; f_pf: int = 1410
    tp_da: int = 1; f_da: int = 1410
    tp_df: int = 1; f_df: int = 1410
    total_energy_mj_per_layer: float = 0.0
    gpu_used: int = 0
    feasible: bool = False
    e_idle_mj: float = 0.0
    predicted_prefill_rho: float = 0.0
    predicted_decode_capacity_tps: float = 0.0
    predicted_decode_demand_tps: float = 0.0
    predicted_decode_rho: float = 0.0
    predicted_decode_headroom_tps: float = 0.0
    predicted_tpot_ms: float = 0.0
    predicted_n_active_decode: float = 0.0
    predicted_energy_per_token: float = float("inf")
    predicted_completion_s: float = float("inf")
    predicted_ttft_ms: float = 0.0
    predicted_memory_headroom_gb: float = 0.0
    system_energy_components: dict = field(default_factory=dict)
    solution_source: str = "infeasible"
    constraint_violations: list[str] = field(default_factory=list)
    objective_valid: bool = False
    meta: dict = field(default_factory=dict)
    infeasibility_diagnostics: dict = field(default_factory=dict)
    _gpu_avail: int = field(default=0, init=False, repr=False)

    def to_dict(self) -> dict:
        return {
            "k_p": self.k_p, "k_d": self.k_d,
            "tp_pa": self.tp_pa, "tp_pf": self.tp_pf,
            "tp_da": self.tp_da, "tp_df": self.tp_df,
            "f_pa": self.f_pa, "f_pf": self.f_pf,
            "f_da": self.f_da, "f_df": self.f_df,
            "total_energy_mj_per_layer": round(self.total_energy_mj_per_layer, 4),
            "e_idle_mj": round(self.e_idle_mj, 4),
            "gpu_used": self.gpu_used,
            "feasible": self.feasible,
            "predicted_prefill_rho": round(self.predicted_prefill_rho, 6),
            "predicted_decode_capacity_tps": round(self.predicted_decode_capacity_tps, 4),
            "predicted_decode_demand_tps": round(self.predicted_decode_demand_tps, 4),
            "predicted_decode_rho": round(self.predicted_decode_rho, 6),
            "predicted_decode_headroom_tps": round(self.predicted_decode_headroom_tps, 4),
            "predicted_tpot_ms": round(self.predicted_tpot_ms, 4),
            "predicted_n_active_decode": round(self.predicted_n_active_decode, 4),
            "predicted_energy_per_token": round(self.predicted_energy_per_token, 6),
            "predicted_completion_s": round(self.predicted_completion_s, 6),
            "predicted_ttft_ms": round(self.predicted_ttft_ms, 4),
            "predicted_memory_headroom_gb": round(self.predicted_memory_headroom_gb, 4),
            "system_energy_components": self.system_energy_components,
            "solution_source": self.solution_source,
            "constraint_violations": self.constraint_violations,
            "objective_valid": self.objective_valid,
            "meta": self.meta,
            "infeasibility_diagnostics": self.infeasibility_diagnostics,
        }

    def format_cli(self) -> str:
        if not self.feasible:
            return "Tier1Solution: INFEASIBLE"
        return (
            f"Tier1Solution: k_P={self.k_p} k_D={self.k_d} | "
            f"PA tp={self.tp_pa} f={self.f_pa}MHz | PF tp={self.tp_pf} f={self.f_pf}MHz | "
            f"DA tp={self.tp_da} f={self.f_da}MHz | DF tp={self.tp_df} f={self.f_df}MHz | "
            f"E={self.total_energy_mj_per_layer:.0f}mJ (idle={self.e_idle_mj:.0f}mJ) | "
            f"GPU={self.gpu_used}/{self.gpu_used + self._gpu_avail}"
        )


class Tier1Solver:
    """SLO-only energy minimizing Tier-1 solver with idle-aware modeling.

    Args:
        profile_table: ProfileTable instance.
        num_layers: Number of transformer layers.
        num_kv_heads: Number of KV attention heads.
        head_dim: Head dimension.
        hidden_size: Model hidden dimension.
        gpu_mem_gb: GPU memory per device.
        attn_weight_gb: Attention weight per GPU at tp=1.
        ffn_weight_gb: FFN weight per GPU at tp=1.
        activation_buffer_gb: Per-GPU activation memory headroom.
        max_gpu_per_pair: Max GPUs per AF pair (≤8 for same-machine AF).
        k_p_fixed: If set, fix k_P to this value (1 for current AFD arch).
        idle_power_w: Idle power per GPU (W) for P/D imbalance calculation.
    """

    def __init__(
        self,
        profile_table: ProfileTable,
        num_layers: int = 64,
        num_kv_heads: int = 8,
        head_dim: int = 128,
        hidden_size: int = 5120,
        gpu_mem_gb: float = 80.0,
        attn_weight_gb: float = 10.0,
        ffn_weight_gb: float = 50.0,
        activation_buffer_gb: float = 4.0,
        max_gpu_per_pair: int = 8,
        k_p_fixed: Optional[int] = 1,
        idle_power_w: float = 80.0,
        pd_disaggregated: bool = True,
        capacity_margin: float = 0.15,
        prefill_rho_max: float = 0.9,
        route_skew_factor: float = 1.0,
    ):
        self.pt = profile_table
        self.num_layers = num_layers
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.hidden_size = hidden_size
        self.gpu_mem_gb = gpu_mem_gb
        self.attn_weight_gb = attn_weight_gb
        self.ffn_weight_gb = ffn_weight_gb
        self.activation_buffer_gb = activation_buffer_gb
        self.max_gpu_per_pair = max_gpu_per_pair
        self.k_p_fixed = k_p_fixed
        self.idle_power_w = idle_power_w
        self.pd_disaggregated = pd_disaggregated
        if capacity_margin < 0 or not 0 < prefill_rho_max < 1 or route_skew_factor < 1:
            raise ValueError("invalid Tier1 capacity configuration")
        self.capacity_margin = capacity_margin
        self.prefill_rho_max = prefill_rho_max
        self.route_skew_factor = route_skew_factor

    # ── Public API ──

    def solve(
        self, G: int, workload: WorkloadProfile, slo: SLOConfig,
        beta: float = _BETA_BALANCE,
    ) -> Tier1Solution:
        """Run the Tier-1 solver (V2: SLO-only, idle-aware).

        Prefill pairs are enumerated per bs = B_total/k_P, since each
        prefill pair's batch size scales inversely with k_P.
        """
        self._slo_tpot_ms = slo.tpot_ms  # stash for queueing check
        self._workload_lambda = workload.lambda_prefill  # stash for idle energy
        logger.info(
            "Tier1SolverV2.solve: G=%d λ=%.1f N_active=%d TTFT=%.0fms TPOT=%.0fms",
            G, workload.lambda_prefill, workload.n_active_decode,
            slo.ttft_ms, slo.tpot_ms,
        )

        # Decode: bs is fixed (per-iteration batch)
        pairs_d = self._enumerate_pairs("decode", workload.bs_avg_d,
                                         workload.il_rep_d, slo.tpot_ms, beta, workload, workload.ol_rep_d, G=G)
        if not pairs_d:
            return self._infeasible_solution(G, {"decode_enumeration": 1})
        pairs_d = self._pareto_by_gpu_group(pairs_d)

        # Prefill: enumerate for each possible bs = B_total/k_P
        k_p_range = [self.k_p_fixed] if self.k_p_fixed else list(range(1, G // 2 + 1))
        k_p_range = [k for k in k_p_range if k <= G // 2]

        pairs_p_by_kp: dict[int, list] = {}
        _bs_seen: dict[int, list] = {}  # cache: bs → pareto pairs
        for k_p in k_p_range:
            bs = max(1, workload.bs_avg_p // k_p)
            if bs in _bs_seen:
                pairs_p_by_kp[k_p] = _bs_seen[bs]
            else:
                pp = self._enumerate_pairs("prefill", bs, workload.il_rep_p,
                                           slo.ttft_ms, beta, workload)
                if pp:
                    pareto = self._pareto_by_gpu_group(pp)
                    _bs_seen[bs] = pareto
                    pairs_p_by_kp[k_p] = pareto

        total_candidates = sum(len(v) for v in pairs_p_by_kp.values())
        logger.info("Phase 1+2: prefill=%d candidates across k_P=%s, decode=%d",
                    total_candidates, sorted(pairs_p_by_kp.keys()), len(pairs_d))

        if not pairs_p_by_kp:
            return self._infeasible_solution(G, {"prefill_enumeration": 1})

        # V2: Energy-only search with idle term
        best = self._search_energy_only(G, pairs_p_by_kp, pairs_d, workload, k_p_range)

        if best is None:
            logger.warning("Tier1SolverV2: no feasible assignment found")
            return self._infeasible_solution(G, getattr(self, "_last_infeasibility", {}))

        return best

    # ── Candidate enumeration (unchanged from v1) ──

    def _enumerate_op_candidates(self, phase, op, bs, il, workload, ol=None, G=16):
        candidates = []
        for tp in _VALID_TP:
            if tp > self.max_gpu_per_pair:  # V2: enforce per-pair GPU limit
                continue
            if op == "A" and phase == "prefill" and not self._check_attn_memory(
                phase, tp, bs, il, workload, k_d=1
            ):
                continue
            if op == "F" and not self._check_ffn_memory(tp):
                continue
            for freq in _VALID_FREQS:
                try:
                    m = self.pt.query_metrics(phase, tp, freq, bs, il, ol)
                except (RuntimeError, ValueError):
                    continue
                t = m.t_a_us if op == "A" else m.t_f_us
                e = m.e_a_mj if op == "A" else m.e_f_mj
                candidates.append(_OpCandidate(tp=tp, freq=freq, t_us=t, e_mj=e))
        return candidates

    def _enumerate_pairs(self, phase, bs, il, slo_ms, beta, workload, ol=None, G=16):
        t_comm_us = self.pt.get_comm_us(bs, il if phase == "prefill" else 1)
        slo_per_layer_us = (slo_ms * 1000.0) / self.num_layers

        cands_a = self._enumerate_op_candidates(phase, "A", bs, il, workload, ol, G=G)
        cands_f = self._enumerate_op_candidates(phase, "F", bs, il, workload, ol, G=G)

        pairs = []
        for ca in cands_a:
            for cf in cands_f:
                # V2: skip cross-machine pairs
                if ca.tp + cf.tp > self.max_gpu_per_pair:
                    continue
                # V2.3: For decode, enforce homogeneous TP (tp_a == tp_f)
                # because heterogeneous TP in AFD decode incurs ~500us/layer
                # sync overhead that profile data does not capture.
                if phase == "decode" and ca.tp != cf.tp:
                    continue
                t_pair = ca.t_us + cf.t_us + t_comm_us
                # V2.5: Add AFD IPC overhead for decode (not captured in profile).
                # Empirical: ~300us base + ~150us per GPU in the pair.
                # TP1+1=2GPU -> +600us, TP2+2=4GPU -> +900us, TP4+4=8GPU -> +1500us
                if phase == "decode":
                    afd_ipc_us = 300.0 + 150.0 * (ca.tp + cf.tp)
                    t_pair += afd_ipc_us
                if t_pair > slo_per_layer_us:
                    continue
                t_max = max(ca.t_us, cf.t_us)
                if t_max > 0 and abs(ca.t_us - cf.t_us) > beta * t_max:
                    continue
                thpt = self._pair_throughput(bs, t_pair, self.num_layers, phase)
                pairs.append(_AFPairCandidate(
                    tp_a=ca.tp, freq_a=ca.freq, tp_f=cf.tp, freq_f=cf.freq,
                    t_a_us=ca.t_us, t_f_us=cf.t_us,
                    e_a_mj=ca.e_mj, e_f_mj=cf.e_mj,
                    t_pair_us=t_pair, throughput_per_pair=thpt,
                ))
        return pairs

    @staticmethod
    def _pair_throughput(bs, t_layer_us, num_layers, phase):
        if phase == "decode":
            return float(bs)
        batch_time_s = num_layers * t_layer_us / 1_000_000.0
        if batch_time_s <= 0:
            return float("inf")
        return bs / batch_time_s

    # ── Memory constraints (unchanged) ──

    def _check_attn_memory(self, phase, tp, bs, il, workload, k_d=1):
        """Check attention weights + KV cache fit in GPU memory.

        For decode phase, k_d specifies how many decode replicas share the
        total n_active_decode requests (each holds n_active/k_d KV entries).
        """
        w_attn_gb = self.attn_weight_gb / tp
        if phase == "prefill":
            kv_cache_gb = (2 * self.num_layers * (self.num_kv_heads / tp)
                           * self.head_dim * bs * il * _BYTES_PER_ELEMENT) / (1024 ** 3)
        else:
            n_per_pair = self.route_skew_factor * workload.n_active_decode / k_d
            avg_seq_len = il + workload.ol_rep_d / 2
            kv_cache_gb = (2 * self.num_layers * (self.num_kv_heads / tp)
                           * self.head_dim * n_per_pair * avg_seq_len * _BYTES_PER_ELEMENT) / (1024 ** 3)
        total_gb = w_attn_gb + kv_cache_gb + self.activation_buffer_gb
        return total_gb <= self.gpu_mem_gb

    def _decode_memory_headroom_gb(self, workload, tp, k_d, n_active=None):
        active = workload.n_active_decode if n_active is None else n_active
        n_per_pair = self.route_skew_factor * active / k_d
        avg_seq_len = workload.il_rep_d + workload.ol_rep_d / 2
        kv_gb = (2 * self.num_layers * (self.num_kv_heads / tp) * self.head_dim
                 * n_per_pair * avg_seq_len * _BYTES_PER_ELEMENT) / (1024 ** 3)
        used_gb = self.attn_weight_gb / tp + kv_gb + self.activation_buffer_gb
        return max(0.0, self.gpu_mem_gb - used_gb)

    def _check_ffn_memory(self, tp):
        w_ffn_gb = self.ffn_weight_gb / tp
        total_gb = w_ffn_gb + self.activation_buffer_gb
        return total_gb <= self.gpu_mem_gb

    # ── V2: Per-GPU-group Pareto ──

    @staticmethod
    def _pareto_by_gpu_group(pairs):
        """每个 (tp_a, tp_f) 独立 Pareto，避免不同 GPU 数的 pair 互相支配。"""
        groups = defaultdict(list)
        for p in pairs:
            groups[(p.tp_a, p.tp_f)].append(p)
        result = []
        for key, group in groups.items():
            # Keep every energy/latency non-dominated point. Equal-latency
            # points are ordered by energy so a dominated frequency is not kept.
            sorted_g = sorted(group, key=lambda p: (p.t_pair_us, p.e_pair_mj, p.freq_a, p.freq_f))
            min_energy = float("inf")
            for p in sorted_g:
                if p.e_pair_mj < min_energy:
                    result.append(p)
                    min_energy = p.e_pair_mj
        return result

    # ── V2: Idle energy ──

    def _compute_idle_energy(self, t_p, t_d, gpu_p, gpu_d, ol_avg, bt, bd):
        """Compute idle energy per layer (mJ).

        In PD-disaggregated mode, P and D run independently. Idle energy
        comes from prefill GPU under-utilization (queueing gaps):
          - Each P replica processes bs requests in t_p us per layer.
          - Total P throughput = k_p * bs / (num_layers * t_p) req/s.
          - Utilization ρ_p = λ / throughput_p.
          - P idle fraction = (1 - ρ_p), bounded to [0, 1].
          - Idle energy = gpu_p * P_idle * (1 - ρ_p) * t_p per layer.

        For non-PD mode, uses the original pipeline imbalance model.
        """
        if self.pd_disaggregated:
            # Prefill queueing idle energy
            # t_p is per-layer time (us) for the prefill pair
            # bt = k_p * bs_p (total prefill batch size)
            # gpu_p = k_p * gpu_per_p_replica
            k_p = bt // max(1, bt // max(1, gpu_p // 2)) if gpu_p > 0 else 1
            # Reconstruct: throughput = bt / (num_layers * t_p / 1e6) req/s
            t_batch_s = self.num_layers * t_p / 1e6  # full batch time in seconds
            if t_batch_s > 0:
                throughput_p = bt / t_batch_s  # req/s total across all P replicas
            else:
                return 0.0
            lam = self._workload_lambda  # stored from solve()
            rho_p = min(lam / throughput_p, 1.0) if throughput_p > 0 else 1.0
            idle_frac = 1.0 - rho_p
            # Idle energy per layer: all P GPUs idle for (1-ρ) fraction of t_p
            return gpu_p * self.idle_power_w * idle_frac * t_p / 1e6 * 1000  # mJ

        decode_iters = bt * ol_avg / bd  # decode iterations per prefill batch
        t_d_effective = t_d * decode_iters

        if t_p >= t_d_effective:
            idle_us = t_p - t_d_effective
            idle_gpus = gpu_d  # decode GPUs idle
        else:
            idle_us = t_d_effective - t_p
            idle_gpus = gpu_p  # prefill GPUs idle

        return idle_gpus * self.idle_power_w * idle_us / 1e6 * 1000  # mJ

    # ── V2.1: Queueing-aware decode SLO check ──

    def _decode_diagnostics(self, workload, best_d, k_d, slo_ms=None):
        """Candidate-specific fixed-point decode capacity diagnostics."""
        import math
        mean_ol = workload.mean_ol if workload.mean_ol is not None else workload.ol_rep_d
        demand = (workload.decode_token_rate_demand if workload.decode_token_rate_demand
                  is not None else workload.lambda_prefill * mean_ol)
        n_active = max(1.0, float(workload.n_active_decode))
        for _ in range(6):
            busy_batch = max(1, math.ceil(self.route_skew_factor * n_active / k_d))
            tpot_ms = self.num_layers * self._scale_t_layer(
                best_d.t_pair_us, busy_batch, workload.bs_avg_d) / 1000.0
            updated = max(1.0, demand * tpot_ms / 1000.0)
            if abs(updated - n_active) < 0.01:
                n_active = updated
                break
            n_active = 0.5 * (n_active + updated)
        busy_batch = max(1, math.ceil(self.route_skew_factor * n_active / k_d))
        tpot_ms = self.num_layers * self._scale_t_layer(
            best_d.t_pair_us, busy_batch, workload.bs_avg_d) / 1000.0
        # A replica can fill up to the profiled effective batch at this
        # latency; use the larger current/profiling batch to avoid treating a
        # lightly loaded instantaneous batch as its maximum service rate.
        effective_batch = max(busy_batch, workload.bs_avg_d)
        capacity = ((k_d / self.route_skew_factor) * effective_batch
                    / (tpot_ms / 1000.0))
        required = (1 + self.capacity_margin) * demand
        limit = getattr(self, "_slo_tpot_ms", float("inf")) if slo_ms is None else slo_ms
        return {"feasible": tpot_ms <= limit and capacity >= required,
                "tpot_ms": tpot_ms, "capacity_tps": capacity,
                "demand_tps": demand, "rho": demand / capacity,
                "headroom_tps": capacity - required, "n_active": n_active,
                "busy_batch": busy_batch, "effective_batch": effective_batch}

    def _check_decode_queueing_slo(self, workload, slo, best_d, k_d):
        d = self._decode_diagnostics(workload, best_d, k_d, slo.tpot_ms)
        return d["feasible"], d["tpot_ms"]

    @staticmethod
    def _scale_t_layer(t_base_us, bs_actual, bs_profile):
        """Scale per-layer decode time based on batch size.

        Empirical from A100 profiling (Qwen3-32B decode):
          - bs 1~64: nearly flat (±5%)
          - bs 64~128: ~10-20% growth
          - bs 128~256: ~50-80% growth (memory-bandwidth saturated)

        Uses piecewise linear model calibrated from profile data.
        """
        if bs_actual <= 64:
            return t_base_us
        elif bs_actual <= 128:
            return t_base_us * (1.0 + 0.3 * (bs_actual - 64) / 64.0)
        else:
            return t_base_us * (1.3 + 0.7 * (bs_actual - 128) / 128.0)

    # ── V2: Energy-only search ──

    def _search_energy_only(self, G, pairs_p_by_kp, pairs_d, workload, k_p_range, M=_DEFAULT_M):
        """Select min-energy (compute + idle) config under GPU budget.

        Args:
            pairs_p_by_kp: dict mapping k_p → list of _AFPairCandidate (prefill).
                Each k_P has its own candidates because bs = B_total/k_P.
            pairs_d: list of decode _AFPairCandidate.
        """
        # Group decode by GPU count
        # Do not collapse a GPU group to its minimum-energy point: a faster,
        # higher-energy Pareto point can be the only candidate satisfying TPOT.
        d_by_gpu = defaultdict(list)
        for d in self._pareto_by_gpu_group(pairs_d):
            d_by_gpu[d.tp_a + d.tp_f].append(d)
        for group in d_by_gpu.values():
            group.sort(key=lambda d: (d.e_pair_mj, d.t_pair_us))

        best_sol = None
        best_e = float("inf")
        best_rank = None
        max_k_d = G // 2
        n_checked = 0
        n_pruned_queue = 0
        rejected = defaultdict(int)
        closest = None
        minimum_required_gpus = None

        for k_p, pairs_p in pairs_p_by_kp.items():
            bs_p = max(1, workload.bs_avg_p // k_p)
            # Group prefill by GPU — keep ALL candidates per group (sorted by energy)
            p_groups: dict[int, list] = defaultdict(list)
            for p in pairs_p:
                p_groups[p.tp_a + p.tp_f].append(p)
            for g in p_groups:
                p_groups[g].sort(key=lambda x: x.e_pair_mj)

            for gpu_p, p_list in p_groups.items():
                best_p = None
                rho_p = float("inf")
                for p in p_list:
                    service_s = self.num_layers * p.t_pair_us / 1e6
                    rho = (workload.lambda_prefill / k_p) * service_s / bs_p
                    if rho <= self.prefill_rho_max:
                        best_p, rho_p = p, rho
                        break
                if best_p is None:
                    continue

                for gpu_d, d_list in d_by_gpu.items():
                    for k_d in range(1, max_k_d + 1):
                      for best_d in d_list:
                        used = k_p * gpu_p + k_d * gpu_d
                        if used > G:
                            rejected["gpu_budget"] += 1
                            minimum_required_gpus = used if minimum_required_gpus is None else min(minimum_required_gpus, used)
                            continue
                        n_checked += 1

                        # Validate each (candidate, k_d) before objective ranking.
                        diag = self._decode_diagnostics(workload, best_d, k_d)
                        original_active = workload.n_active_decode
                        workload.n_active_decode = diag["n_active"]
                        try:
                            memory_ok = self._check_attn_memory(
                                "decode", best_d.tp_a, diag["busy_batch"],
                                workload.il_rep_d, workload, k_d=k_d)
                        finally:
                            workload.n_active_decode = original_active
                        if not memory_ok:
                            rejected["decode_memory"] += 1
                            continue
                        if diag["tpot_ms"] > self._slo_tpot_ms:
                            rejected["decode_tpot"] += 1
                        if diag["capacity_tps"] < (1 + self.capacity_margin) * diag["demand_tps"]:
                            rejected["decode_capacity"] += 1
                        if not diag["feasible"]:
                            n_pruned_queue += 1
                            gap = max(0.0, diag["tpot_ms"] / self._slo_tpot_ms - 1.0) + max(0.0, -diag["headroom_tps"] / max(diag["demand_tps"], 1.0))
                            candidate = {"gap": gap, "gpu_used": used, "k_d": k_d, "tp_da": best_d.tp_a, "tp_df": best_d.tp_f, "tpot_ms": diag["tpot_ms"], "capacity_tps": diag["capacity_tps"], "required_capacity_tps": (1 + self.capacity_margin) * diag["demand_tps"]}
                            if closest is None or (gap, used) < (closest["gap"], closest["gpu_used"]): closest = candidate
                            continue

                        e_idle = self._compute_idle_energy(
                            best_p.t_pair_us, best_d.t_pair_us,
                            k_p * gpu_p, k_d * gpu_d,
                            workload.ol_rep_d,
                            k_p * bs_p,                # total prefill batch
                            workload.bs_avg_d * k_d,   # total decode batch
                        )
                        e_bubble_p = self._pair_bubble_energy(best_p, M)
                        e_bubble_d = self._pair_bubble_energy(best_d, M)
                        # Complete-work energy: dynamic compute plus wall-clock
                        # static energy. The latter makes race-to-idle explicit.
                        mean_ol = workload.mean_ol if workload.mean_ol is not None else workload.ol_rep_d
                        completion_s = max(self.num_layers * best_p.t_pair_us / 1e6,
                                           mean_ol * diag["tpot_ms"] / 1000.0)
                        output_tokens = max(mean_ol, 1.0)
                        active_mj = self.num_layers * (
                            best_p.e_pair_mj / max(bs_p, 1)
                            + mean_ol * best_d.e_pair_mj / max(diag["effective_batch"], 1))
                        bubble_comm_mj = self.num_layers * (
                            e_bubble_p / max(bs_p, 1)
                            + mean_ol * e_bubble_d / max(diag["effective_batch"], 1))
                        static_mj = used * self.idle_power_w * completion_s * 1000.0
                        idle_mj = e_idle * self.num_layers / max(bs_p, 1)
                        system_mj = active_mj + bubble_comm_mj + static_mj + idle_mj
                        energy_per_token = system_mj / output_tokens
                        capacity_headroom = diag["capacity_tps"] - diag["demand_tps"]
                        rank = (-capacity_headroom, used, gpu_p + gpu_d, completion_s)
                        if best_rank is None or energy_per_token < best_e * 0.95:
                            better = True
                        elif energy_per_token <= best_e * 1.05:
                            better = rank < best_rank
                        else:
                            better = False
                        if better:
                            best_e = energy_per_token
                            best_rank = rank
                            s = Tier1Solution(
                                k_p=k_p, k_d=k_d,
                                tp_pa=best_p.tp_a, tp_pf=best_p.tp_f,
                                tp_da=best_d.tp_a, tp_df=best_d.tp_f,
                                f_pa=best_p.freq_a, f_pf=best_p.freq_f,
                                f_da=best_d.freq_a, f_df=best_d.freq_f,
                                total_energy_mj_per_layer=(active_mj + bubble_comm_mj + idle_mj) / self.num_layers,
                                e_idle_mj=e_idle, gpu_used=used, feasible=True,
                                predicted_energy_per_token=energy_per_token,
                                predicted_completion_s=completion_s,
                                predicted_ttft_ms=self.num_layers * best_p.t_pair_us / 1000.0,
                                predicted_memory_headroom_gb=self._decode_memory_headroom_gb(workload, best_d.tp_a, k_d, diag["n_active"]),
                                system_energy_components={"active_compute_mj": round(active_mj, 4), "idle_static_mj": round(static_mj + idle_mj, 4), "communication_bubble_mj": round(bubble_comm_mj, 4), "total_mj": round(system_mj, 4)},
                                solution_source="optimal", objective_valid=True,
                                predicted_prefill_rho=rho_p,
                                predicted_decode_capacity_tps=diag["capacity_tps"],
                                predicted_decode_demand_tps=diag["demand_tps"],
                                predicted_decode_rho=diag["rho"],
                                predicted_decode_headroom_tps=diag["headroom_tps"],
                                predicted_tpot_ms=diag["tpot_ms"],
                                predicted_n_active_decode=diag["n_active"],
                                meta={"decode_busy_batch": diag["busy_batch"],
                                      "capacity_margin": self.capacity_margin,
                                      "prefill_rho_max": self.prefill_rho_max,
                                      "route_skew_factor": self.route_skew_factor},
                            )
                            s._gpu_avail = G - used
                            best_sol = s

        if n_pruned_queue > 0:
            logger.info("Queueing SLO pruned %d/%d candidates (%.0f%%)", n_pruned_queue, n_checked, 100.0 * n_pruned_queue / n_checked if n_checked else 0)
        self._last_infeasibility = {"rejected_counts": dict(rejected), "checked_candidates": n_checked, "closest_candidate": closest, "minimum_required_gpus": minimum_required_gpus}
        return best_sol

    @staticmethod
    def _pair_bubble_energy(pair, M):
        if M <= 1:
            return 0.0
        t_wait_us = abs(pair.t_a_us - pair.t_f_us)
        e_bubble_j = _P_IDLE_W * t_wait_us / 1_000_000.0 * (M - 1) / M
        return e_bubble_j * 1000.0

    # ── V2.4: Max-throughput mode ──

    def solve_max_throughput(
        self, G: int, workload: WorkloadProfile, slo: SLOConfig,
        beta: float = _BETA_BALANCE,
    ) -> Tier1Solution:
        """Maximize system throughput under SLO constraints (ignore energy).

        All frequencies forced to max (1410MHz) — true MegaScale mode.
        Forces GPU usage == G (use all available GPUs).
        Picks the config that maximizes total decode throughput while
        prefill approximately keeps up with arrival rate.
        """
        self._slo_tpot_ms = slo.tpot_ms
        self._workload_lambda = workload.lambda_prefill
        max_freq = max(_VALID_FREQS)
        logger.info(
            "Tier1SolverV2.solve_max_throughput: G=%d λ=%.1f TTFT=%.0fms TPOT=%.0fms freq=%d",
            G, workload.lambda_prefill, slo.ttft_ms, slo.tpot_ms, max_freq,
        )

        # Enumerate with relaxed SLO, then filter to max freq only
        pairs_d_all = self._enumerate_pairs("decode", workload.bs_avg_d,
                                            workload.il_rep_d, slo.tpot_ms, beta, workload, workload.ol_rep_d, G=G)
        pairs_d = [p for p in pairs_d_all if p.freq_a == max_freq and p.freq_f == max_freq]
        if not pairs_d:
            pairs_d = pairs_d_all

        k_p_range = [self.k_p_fixed] if self.k_p_fixed else list(range(1, G // 2 + 1))
        k_p_range = [k for k in k_p_range if k <= G // 2]

        pairs_p_by_kp: dict[int, list] = {}
        _bs_seen: dict[int, list] = {}
        for k_p in k_p_range:
            bs = max(1, workload.bs_avg_p // k_p)
            if bs in _bs_seen:
                pairs_p_by_kp[k_p] = _bs_seen[bs]
            else:
                pp_all = self._enumerate_pairs("prefill", bs, workload.il_rep_p,
                                              slo.ttft_ms, beta, workload)
                pp = [p for p in pp_all if p.freq_a == max_freq and p.freq_f == max_freq]
                if not pp:
                    pp = pp_all
                if pp:
                    _bs_seen[bs] = pp
                    pairs_p_by_kp[k_p] = pp

        if not pairs_p_by_kp:
            return self._infeasible_solution(G, {"prefill_enumeration": 1})

        best = self._search_max_throughput(G, pairs_p_by_kp, pairs_d, workload, k_p_range)
        if best is None:
            logger.warning("Tier1SolverV2.solve_max_throughput: no feasible config")
            return self._infeasible_solution(G, getattr(self, "_last_infeasibility", {"max_throughput_search": 1}))
        return best

    def _search_max_throughput(self, G, pairs_p_by_kp, pairs_d, workload, k_p_range):
        """Select max-throughput config that uses exactly G GPUs.

        All frequencies already filtered to max (1410MHz) before this call.
        Maximizes total decode throughput = k_d * bs_per_replica / t_iter.
        Enforces used == G (use all available GPUs).

        Tie-breaking: prefer smaller decode TP (lower TPOT) with more k_d.
        """
        d_by_gpu = {}
        for d in pairs_d:
            g = d.tp_a + d.tp_f
            if g not in d_by_gpu or d.t_pair_us < d_by_gpu[g].t_pair_us:
                d_by_gpu[g] = d

        best_sol = None
        best_thpt = 0.0
        best_tpot = float("inf")
        best_used = 0
        max_k_d = G // 2

        for k_p, pairs_p in pairs_p_by_kp.items():
            bs_p = max(1, workload.bs_avg_p // k_p)
            from collections import defaultdict
            p_groups: dict[int, list] = defaultdict(list)
            for p in pairs_p:
                p_groups[p.tp_a + p.tp_f].append(p)
            for g in p_groups:
                p_groups[g].sort(key=lambda x: x.t_pair_us)

            for gpu_p, p_list in p_groups.items():
                best_p = None
                for p in p_list:
                    t_service = self.num_layers * p.t_pair_us / 1e6
                    if t_service <= 0:
                        best_p = p
                        break
                    lambda_per_inst = workload.lambda_prefill / k_p
                    rho = lambda_per_inst * t_service / bs_p
                    if rho <= self.prefill_rho_max:
                        best_p = p
                        break
                if best_p is None:
                    continue

                for gpu_d, best_d in d_by_gpu.items():
                    for k_d in range(1, max_k_d + 1):
                        used = k_p * gpu_p + k_d * gpu_d
                        if used > G:
                            continue

                        # Exact memory check with actual k_d
                        if not self._check_attn_memory(
                            "decode", best_d.tp_a, workload.bs_avg_d,
                            workload.il_rep_d, workload, k_d=k_d,
                        ):
                            continue

                        q_ok, tpot_est = self._check_decode_queueing_slo(
                            workload, SLOConfig(tpot_ms=self._slo_tpot_ms),
                            best_d, k_d)
                        if not q_ok:
                            continue

                        bs_per_d = max(1, int(workload.n_active_decode / k_d))
                        t_iter_s = self.num_layers * self._scale_t_layer(
                            best_d.t_pair_us, bs_per_d, workload.bs_avg_d) / 1e6
                        if t_iter_s <= 0:
                            continue
                        thpt_d = k_d * bs_per_d / t_iter_s
                        tpot_ms = t_iter_s * 1000.0

                        # Prefer: higher throughput first, then lower TPOT
                        better = False
                        if thpt_d > best_thpt:
                            better = True
                        elif thpt_d == best_thpt and (tpot_ms, used, k_d, gpu_d) < (best_tpot, best_used, best_sol.k_d if best_sol else float("inf"), (best_sol.tp_da + best_sol.tp_df) if best_sol else float("inf")):
                            better = True

                        if better:
                            best_thpt = thpt_d
                            best_tpot = tpot_ms
                            best_used = used
                            diag = self._decode_diagnostics(workload, best_d, k_d)
                            rho_p = (workload.lambda_prefill / k_p) * (
                                self.num_layers * best_p.t_pair_us / 1e6) / bs_p
                            mean_ol = workload.mean_ol if workload.mean_ol is not None else workload.ol_rep_d
                            completion_s = max(self.num_layers * best_p.t_pair_us / 1e6,
                                               mean_ol * diag["tpot_ms"] / 1000.0)
                            s = Tier1Solution(
                                k_p=k_p, k_d=k_d,
                                tp_pa=best_p.tp_a, tp_pf=best_p.tp_f,
                                tp_da=best_d.tp_a, tp_df=best_d.tp_f,
                                f_pa=best_p.freq_a, f_pf=best_p.freq_f,
                                f_da=best_d.freq_a, f_df=best_d.freq_f,
                                total_energy_mj_per_layer=0, e_idle_mj=0,
                                gpu_used=used, feasible=True,
                                predicted_prefill_rho=rho_p,
                                predicted_decode_capacity_tps=diag["capacity_tps"],
                                predicted_decode_demand_tps=diag["demand_tps"],
                                predicted_decode_rho=diag["rho"],
                                predicted_decode_headroom_tps=diag["headroom_tps"],
                                predicted_tpot_ms=diag["tpot_ms"],
                                predicted_n_active_decode=diag["n_active"],
                                predicted_completion_s=completion_s,
                                predicted_ttft_ms=self.num_layers * best_p.t_pair_us / 1000.0,
                                predicted_memory_headroom_gb=self._decode_memory_headroom_gb(
                                    workload, best_d.tp_a, k_d, diag["n_active"]),
                                solution_source="max_throughput", objective_valid=True,
                                meta={"capacity_margin": self.capacity_margin,
                                      "prefill_rho_max": self.prefill_rho_max,
                                      "route_skew_factor": self.route_skew_factor},
                            )
                            s._gpu_avail = G - used
                            best_sol = s

        if best_sol:
            logger.info("Max-throughput: %.1f tok/s TPOT~%.1fms | %s",
                        best_thpt, best_tpot, best_sol.format_cli())
        return best_sol

    def _infeasible_solution(self, G, diagnostics):
        payload = {"rejected_counts": {}, "checked_candidates": 0, "closest_candidate": None, "minimum_required_gpus": None}
        payload.update(diagnostics or {})
        return Tier1Solution(feasible=False, gpu_used=0, solution_source="infeasible",
                             constraint_violations=["no_solver_feasible_assignment"],
                             objective_valid=False, infeasibility_diagnostics=payload,
                             meta={"gpu_budget": G})

    # ── Warm start ──

    def warm_start(self, G, workload, slo) -> Tier1Solution:
        sol = self.solve(G, workload, slo)
        if sol.feasible:
            sol.solution_source = "optimal"
            return sol

        logger.warning("Tier1SolverV2.warm_start: no feasible ILP solution, "
                       "returning SLO-only energy-optimal fallback")
        # Fallback: re-solve without k_P constraint, pick min-energy at any k_P
        best = self._fallback_energy_optimal(G, workload, slo)
        if best is None:
            # Absolute fallback
            return Tier1Solution(
                k_p=1, k_d=3,
                tp_pa=4, tp_pf=4, tp_da=1, tp_df=1,
                f_pa=1410, f_pf=1410, f_da=1410, f_df=1410,
                gpu_used=14, feasible=False, solution_source="hardcoded",
                constraint_violations=["no_solver_feasible_assignment"], objective_valid=False,
            )
        best.solution_source = "relaxed_kp"
        return best

    def _fallback_energy_optimal(self, G, workload, slo):
        """Find lowest-energy config meeting SLO, ignoring throughput."""
        self._slo_tpot_ms = slo.tpot_ms
        self._workload_lambda = workload.lambda_prefill
        beta = _BETA_BALANCE
        pairs_d = self._enumerate_pairs("decode", workload.bs_avg_d,
                                         workload.il_rep_d, slo.tpot_ms, beta, workload, workload.ol_rep_d, G=G)
        if not pairs_d:
            return None
        pairs_d = self._pareto_by_gpu_group(pairs_d)

        # Enumerate prefill for each k_p (relaxed: any k_p)
        k_p_range = range(1, G // 2 + 1)
        pairs_p_by_kp: dict[int, list] = {}
        _bs_seen: dict[int, list] = {}
        for k_p in k_p_range:
            bs = max(1, workload.bs_avg_p // k_p)
            if bs in _bs_seen:
                pairs_p_by_kp[k_p] = _bs_seen[bs]
            else:
                pp = self._enumerate_pairs("prefill", bs, workload.il_rep_p,
                                           slo.ttft_ms, beta, workload)
                if pp:
                    pareto = self._pareto_by_gpu_group(pp)
                    _bs_seen[bs] = pareto
                    pairs_p_by_kp[k_p] = pareto

        if not pairs_p_by_kp:
            return None

        return self._search_energy_only(G, pairs_p_by_kp, pairs_d, workload,
                                        list(k_p_range))

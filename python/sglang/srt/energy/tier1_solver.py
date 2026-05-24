"""B-2: Tier 1 ILP Solver — joint resource planning for AF-disaggregated serving.

Solves for the optimal 4-pool configuration given GPU budget, workload profile,
and SLO constraints.  Uses enumeration over a small pruned discrete search space
(500–5000 candidates) — no external ILP solver dependency.

Algorithm sketch
  1. Enumerate all feasible (tp, freq) configs for each of the 4 pools
     (PA, PF, DA, DF), applying hard pruning (SLO, memory, balance).
  2. Pareto-filter each pool's candidate list.
  3. For each viable (k_P, k_D) pair, select the min-energy config per pool
     that meets the throughput constraint.
  4. Return the global minimum-energy assignment.

Input
  - G: total GPU budget
  - workload: WorkloadProfile (arrival rate, length distributions, etc.)
  - slo_config: SLOConfig (TTFT and TPOT budgets)
  - profile_table: ProfileTable instance

Output
  Tier1Solution with k_P, k_D, tp_*, f_*, total_energy, gpu_used.

Usage
    from sglang.srt.energy.profile_table import ProfileTable
    from sglang.srt.energy.tier1_solver import Tier1Solver, WorkloadProfile, SLOConfig

    pt = ProfileTable(...)
    solver = Tier1Solver(pt, num_layers=64, num_kv_heads=8, head_dim=128,
                         hidden_size=5120, gpu_mem_gb=80)
    wl = WorkloadProfile(lambda_prefill=10.0, n_active_decode=32,
                         il_p50=512, il_p90=2048,
                         ol_p50=128, ol_p90=512,
                         bs_avg_p=8, bs_avg_d=16)
    slo = SLOConfig(ttft_ms=500.0, tpot_ms=50.0)
    sol = solver.solve(G=16, workload=wl, slo=slo)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

from sglang.srt.energy.profile_table import ProfileTable

logger = logging.getLogger(__name__)

# ── Constants ───────────────────────────────────────────────────────────

_VALID_TP = [1, 2, 4, 8]
_VALID_FREQS = [210, 450, 690, 930, 1170, 1410]
_BETA_BALANCE = 0.8       # hard-prune when |t_A - t_F| > 0.8 × max(t_A, t_F)
_BYTES_PER_ELEMENT = 2    # bfloat16
_DEFAULT_M = 2            # default microbatch count for pipeline bubble estimation
_P_IDLE_W = 80.0          # idle power (W) for a GPU waiting in pipeline bubble


# ═══════════════════════════════════════════════════════════════════════
# Input data classes
# ═══════════════════════════════════════════════════════════════════════

@dataclass
class WorkloadProfile:
    """Aggregate workload characteristics for a Tier-1 planning window.

    The rep (representative) fields are the values used for profile-table
    lookups during optimization.  P90 is recommended for conservatism.
    """
    # Request rates
    lambda_prefill: float = 10.0    # prefill arrival rate (req/s)
    n_active_decode: int = 32       # steady-state active decode requests

    # Prefill representative workload
    il_rep_p: int = 1024            # representative input length
    bs_avg_p: int = 8               # expected prefill batch size

    # Decode representative workload
    il_rep_d: int = 512             # representative input length
    ol_rep_d: int = 256             # representative output length
    bs_avg_d: int = 16              # expected decode batch size


@dataclass
class SLOConfig:
    """Service-level objective budgets."""
    ttft_ms: float = 500.0          # time-to-first-token
    tpot_ms: float = 50.0           # time-per-output-token


# ═══════════════════════════════════════════════════════════════════════
# Internal data classes
# ═══════════════════════════════════════════════════════════════════════

@dataclass
class _PoolCandidate:
    """One viable (tp, freq) configuration for a single pool."""
    tp: int
    freq: int
    t_a_us: float
    t_f_us: float
    e_a_mj: float
    e_f_mj: float
    t_layer_us: float = 0.0       # M=1: t_A + t_F + t_comm
    e_total_mj: float = 0.0        # E_A + E_F per layer
    throughput_per_pair: float = 0.0  # req/s for one pipeline pair

    def __post_init__(self):
        self.e_total_mj = self.e_a_mj + self.e_f_mj

    def e_bubble_mj(self, M: int = _DEFAULT_M) -> float:
        """Pipeline bubble energy: idle power × wait time × (M-1)/M.

        The faster side waits for the slower side. During the wait, the idle
        GPU consumes P_idle at its own frequency.
        """
        if M <= 1:
            return 0.0
        t_wait_us = abs(self.t_a_us - self.t_f_us)
        return _P_IDLE_W * t_wait_us / 1_000_000.0 * 1000.0 * (M - 1) / M


@dataclass
class Tier1Solution:
    """Output of the Tier-1 ILP solver."""
    # Pool counts
    k_p: int = 1
    k_d: int = 1

    # Prefill-Attention
    tp_pa: int = 1
    f_pa: int = 1410

    # Prefill-FFN
    tp_pf: int = 1
    f_pf: int = 1410

    # Decode-Attention
    tp_da: int = 1
    f_da: int = 1410

    # Decode-FFN
    tp_df: int = 1
    f_df: int = 1410

    # Metrics
    total_energy_mj_per_layer: float = 0.0
    gpu_used: int = 0
    feasible: bool = False
    # Internal: set after creation by _search_kp_kd
    _gpu_avail: int = field(default=0, init=False, repr=False)

    def to_dict(self) -> dict:
        return {
            "k_p": self.k_p, "k_d": self.k_d,
            "tp_pa": self.tp_pa, "tp_pf": self.tp_pf,
            "tp_da": self.tp_da, "tp_df": self.tp_df,
            "f_pa": self.f_pa, "f_pf": self.f_pf,
            "f_da": self.f_da, "f_df": self.f_df,
            "total_energy_mj_per_layer": round(self.total_energy_mj_per_layer, 4),
            "gpu_used": self.gpu_used,
            "feasible": self.feasible,
        }

    def format_cli(self) -> str:
        """Human-readable one-line summary."""
        if not self.feasible:
            return "Tier1Solution: INFEASIBLE — no config meets all constraints"
        return (
            f"Tier1Solution: k_P={self.k_p} k_D={self.k_d} | "
            f"PA tp={self.tp_pa} f={self.f_pa}MHz | "
            f"PF tp={self.tp_pf} f={self.f_pf}MHz | "
            f"DA tp={self.tp_da} f={self.f_da}MHz | "
            f"DF tp={self.tp_df} f={self.f_df}MHz | "
            f"E/layer={self.total_energy_mj_per_layer:.2f}mJ | "
            f"GPU={self.gpu_used}/{self.gpu_used + self._gpu_avail} "
            f"(used/total incl. avail)"
        )


# ═══════════════════════════════════════════════════════════════════════
# Solver
# ═══════════════════════════════════════════════════════════════════════

class Tier1Solver:
    """Joint ILP solver for P/D + A/F resource planning.

    Args:
        profile_table: ProfileTable with loaded data and predictor.
        num_layers: Number of transformer layers (L).
        num_kv_heads: Number of KV attention heads (for memory calc).
        head_dim: Dimension of each attention head.
        hidden_size: Model hidden dimension.
        gpu_mem_gb: Available GPU memory per device (default 80 GB for A800).
        attn_weight_gb: Approximate attention weight per GPU at tp=1.
        ffn_weight_gb: Approximate FFN weight per GPU at tp=1.
        activation_buffer_gb: Per-GPU activation memory headroom.
    """

    def __init__(
        self,
        profile_table: ProfileTable,
        num_layers: int = 64,
        num_kv_heads: int = 8,
        head_dim: int = 128,
        hidden_size: int = 5120,
        gpu_mem_gb: float = 80.0,
        attn_weight_gb: float = 7.0,
        ffn_weight_gb: float = 59.0,
        activation_buffer_gb: float = 4.0,
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

    # ── Public API ──────────────────────────────────────────────────────

    def solve(
        self,
        G: int,
        workload: WorkloadProfile,
        slo: SLOConfig,
        alpha: float = 0.15,
        beta: float = _BETA_BALANCE,
    ) -> Tier1Solution:
        """Run the Tier-1 solver.

        Args:
            G: Total GPU budget.
            workload: Workload profile for the planning window.
            slo: SLO budgets.
            alpha: Capacity margin (0.15 = 15% headroom).
            beta: A/F balance pruning threshold (0.8).

        Returns:
            Tier1Solution with the optimal configuration, or feasible=False.
        """
        logger.info(
            "Tier1Solver.solve: G=%d λ=%.1f N_active=%d α=%.2f",
            G, workload.lambda_prefill, workload.n_active_decode, alpha,
        )

        # ── Phase 1: enumerate candidates per pool ─────────────────
        cand_pa = self._enumerate_pool(
            "prefill", "A", workload.bs_avg_p, workload.il_rep_p,
            slo.ttft_ms, beta, workload,
        )
        cand_pf = self._enumerate_pool(
            "prefill", "F", workload.bs_avg_p, workload.il_rep_p,
            slo.ttft_ms, beta, workload,
        )
        cand_da = self._enumerate_pool(
            "decode", "A", workload.bs_avg_d, workload.il_rep_d,
            slo.tpot_ms, beta, workload, workload.ol_rep_d,
        )
        cand_df = self._enumerate_pool(
            "decode", "F", workload.bs_avg_d, workload.il_rep_d,
            slo.tpot_ms, beta, workload, workload.ol_rep_d,
        )

        logger.info(
            "Candidates after pruning: PA=%d PF=%d DA=%d DF=%d",
            len(cand_pa), len(cand_pf), len(cand_da), len(cand_df),
        )
        if not all([cand_pa, cand_pf, cand_da, cand_df]):
            logger.warning("Tier1Solver: at least one pool has zero candidates")
            return Tier1Solution(feasible=False)

        # ── Phase 2: Pareto filter per pool ────────────────────────
        cand_pa = self._pareto_filter(cand_pa)
        cand_pf = self._pareto_filter(cand_pf)
        cand_da = self._pareto_filter(cand_da)
        cand_df = self._pareto_filter(cand_df)
        logger.info(
            "After Pareto: PA=%d PF=%d DA=%d DF=%d",
            len(cand_pa), len(cand_pf), len(cand_da), len(cand_df),
        )

        # ── Phase 3: enumerate (k_P, k_D) ──────────────────────────
        best = self._search_kp_kd(
            G, cand_pa, cand_pf, cand_da, cand_df,
            workload, alpha,
        )

        if best is None:
            logger.warning("Tier1Solver: no feasible (k_P,k_D) assignment found")
            return Tier1Solution(feasible=False)

        return best

    # ── Candidate enumeration ────────────────────────────────────────────

    def _enumerate_pool(
        self,
        phase: str,
        op: str,
        bs: int,
        il: int,
        slo_ms: float,
        beta: float,
        workload: WorkloadProfile,
        ol: Optional[int] = None,
    ) -> list[_PoolCandidate]:
        """Enumerate all feasible (tp, freq) for one pool, with hard pruning."""
        candidates: list[_PoolCandidate] = []
        t_comm_us = self.pt.get_comm_us(bs, il if phase == "prefill" else 1)
        slo_per_layer_us = (slo_ms * 1000.0) / self.num_layers

        for tp in _VALID_TP:
            # Memory pruning
            if op == "A" and not self._check_attn_memory(phase, tp, bs, il, workload):
                continue
            if op == "F" and not self._check_ffn_memory(tp):
                continue

            for freq in _VALID_FREQS:
                try:
                    m = self.pt.query_metrics(phase, tp, freq, bs, il, ol)
                except (RuntimeError, ValueError):
                    continue

                t_layer = m.t_a_us + m.t_f_us + t_comm_us

                # SLO pruning (M=1 conservative)
                if t_layer > slo_per_layer_us:
                    continue

                # A/F balance pruning
                t_max = max(m.t_a_us, m.t_f_us)
                if t_max > 0 and abs(m.t_a_us - m.t_f_us) > beta * t_max:
                    continue

                thpt = self._pair_throughput(bs, t_layer, self.num_layers, phase)
                candidates.append(_PoolCandidate(
                    tp=tp, freq=freq,
                    t_a_us=m.t_a_us, t_f_us=m.t_f_us,
                    e_a_mj=m.e_a_mj, e_f_mj=m.e_f_mj,
                    t_layer_us=t_layer,
                    throughput_per_pair=thpt,
                ))

        return candidates

    @staticmethod
    def _pair_throughput(
        bs: int, t_layer_us: float, num_layers: int,
        phase: str = "prefill",
    ) -> float:
        """Estimated throughput/capacity for one pipeline pair.

        For prefill: throughput = bs / (L × t_layer) in req/s.
            One batch of bs requests takes L × t_layer seconds.
        For decode: capacity = bs (concurrent requests per pair).
            Each iteration serves the entire batch; the constraint is
            k_D × bs ≥ (1+α) × N_active. We return bs directly so that
            _select_for_pool can compare k × capacity vs demand.
        """
        if phase == "decode":
            return float(bs)
        batch_time_s = num_layers * t_layer_us / 1_000_000.0
        if batch_time_s <= 0:
            return float("inf")
        return bs / batch_time_s

    # ── Memory constraints ───────────────────────────────────────────────

    def _check_attn_memory(
        self, phase: str, tp: int, bs: int, il: int,
        workload: WorkloadProfile,
    ) -> bool:
        """Check whether attention weights + KV cache fit in GPU memory."""
        # Attention weights (Q/K/V/O projections + output)
        w_attn_gb = self.attn_weight_gb / tp

        if phase == "prefill":
            # KV cache is transient in prefill (forwarded to decode in PD sep)
            kv_cache_gb = (
                2 * self.num_layers * (self.num_kv_heads / tp)
                * self.head_dim * bs * il * _BYTES_PER_ELEMENT
            ) / (1024 ** 3)
        else:
            # Decode: steady-state KV cache for N_active concurrent requests
            n_per_pair = workload.n_active_decode  # conservative: all on one pair
            avg_seq_len = il + workload.ol_rep_d / 2  # average total sequence
            kv_cache_gb = (
                2 * self.num_layers * (self.num_kv_heads / tp)
                * self.head_dim * n_per_pair * avg_seq_len * _BYTES_PER_ELEMENT
            ) / (1024 ** 3)

        total_gb = w_attn_gb + kv_cache_gb + self.activation_buffer_gb
        return total_gb <= self.gpu_mem_gb

    def _check_ffn_memory(self, tp: int) -> bool:
        """Check whether FFN weights + activation buffer fit in GPU memory."""
        w_ffn_gb = self.ffn_weight_gb / tp
        total_gb = w_ffn_gb + self.activation_buffer_gb
        return total_gb <= self.gpu_mem_gb

    # ── Pareto filtering ─────────────────────────────────────────────────

    @staticmethod
    def _pareto_filter(candidates: list[_PoolCandidate]) -> list[_PoolCandidate]:
        """Keep only non-dominated configs (lower latency AND lower energy is better).

        A config dominates B if it has ≤ latency AND ≤ energy, with at least
        one strictly better.
        """
        if len(candidates) <= 1:
            return candidates

        # Sort by latency ascending
        sorted_c = sorted(candidates, key=lambda c: c.t_layer_us)
        pareto: list[_PoolCandidate] = []
        min_energy = float("inf")

        for c in sorted_c:
            if c.e_total_mj < min_energy:
                pareto.append(c)
                min_energy = c.e_total_mj

        return pareto

    # ── (k_P, k_D) search ───────────────────────────────────────────────

    def _search_kp_kd(
        self,
        G: int,
        cand_pa: list[_PoolCandidate],
        cand_pf: list[_PoolCandidate],
        cand_da: list[_PoolCandidate],
        cand_df: list[_PoolCandidate],
        workload: WorkloadProfile,
        alpha: float,
        M: int = _DEFAULT_M,
    ) -> Optional[Tier1Solution]:
        """Enumerate viable (k_P, k_D) and select min-energy assignment."""
        best_solution: Optional[Tier1Solution] = None
        best_energy = float("inf")

        max_k = G // 2  # each pair needs at least 2 GPUs (tp_A=1 + tp_F=1)

        for k_p in range(1, max_k + 1):
            for k_d in range(1, max_k + 1):
                # Quick resource check — need at least min feasible TP per pair
                min_tp_p = min(c.tp for c in cand_pa) + min(c.tp for c in cand_pf)
                min_tp_d = min(c.tp for c in cand_da) + min(c.tp for c in cand_df)
                if k_p * min_tp_p + k_d * min_tp_d > G:
                    continue

                # Select best config for each pool given (k_p, k_d)
                sel_pa = self._select_for_pool(
                    cand_pa, k_p, workload.lambda_prefill, alpha,
                )
                sel_pf = self._select_for_pool(
                    cand_pf, k_p, workload.lambda_prefill, alpha,
                )
                sel_da = self._select_for_pool(
                    cand_da, k_d, workload.n_active_decode, alpha,
                )
                sel_df = self._select_for_pool(
                    cand_df, k_d, workload.n_active_decode, alpha,
                )

                if sel_pa is None or sel_pf is None or sel_da is None or sel_df is None:
                    continue

                gpu_used = (
                    k_p * (sel_pa.tp + sel_pf.tp)
                    + k_d * (sel_da.tp + sel_df.tp)
                )
                if gpu_used > G:
                    continue

                # Total energy per layer (weighted by pair count)
                # Includes E_bubble: idle power during pipeline stall
                e_bubble_p = self._pair_bubble_energy(sel_pa, sel_pf, M)
                e_bubble_d = self._pair_bubble_energy(sel_da, sel_df, M)
                e_total = (
                    k_p * (sel_pa.e_total_mj + sel_pf.e_total_mj + e_bubble_p)
                    + k_d * (sel_da.e_total_mj + sel_df.e_total_mj + e_bubble_d)
                )

                if e_total < best_energy:
                    best_energy = e_total
                    sol = Tier1Solution(
                        k_p=k_p, k_d=k_d,
                        tp_pa=sel_pa.tp, tp_pf=sel_pf.tp,
                        tp_da=sel_da.tp, tp_df=sel_df.tp,
                        f_pa=sel_pa.freq, f_pf=sel_pf.freq,
                        f_da=sel_da.freq, f_df=sel_df.freq,
                        total_energy_mj_per_layer=e_total,
                        gpu_used=gpu_used,
                        feasible=True,
                    )
                    sol._gpu_avail = G - gpu_used
                    best_solution = sol

        return best_solution

    @staticmethod
    def _pair_bubble_energy(
        cand_a: _PoolCandidate, cand_f: _PoolCandidate, M: int,
    ) -> float:
        """Compute pipeline bubble energy for an A/F pair.

        E_bubble = P_idle × |t_A - t_F| × (M-1)/M  (mJ)

        The faster side idles while waiting for the slower side to finish.
        This penalizes configurations with large A/F latency imbalance.
        """
        if M <= 1:
            return 0.0
        t_wait_us = abs(cand_a.t_a_us - cand_f.t_f_us)
        e_bubble_j = _P_IDLE_W * t_wait_us / 1_000_000.0 * (M - 1) / M
        return e_bubble_j * 1000.0  # convert J to mJ

    @staticmethod
    def _select_for_pool(
        candidates: list[_PoolCandidate],
        k: int,
        demand: float,
        alpha: float,
    ) -> Optional[_PoolCandidate]:
        """Pick the min-energy candidate whose throughput × k meets demand.

        Args:
            candidates: Pareto-filtered pool candidates, sorted by latency.
            k: Number of pipeline pairs sharing the load.
            demand: Request rate (λ or N_active).
            alpha: Capacity margin.

        Returns:
            The lowest-energy candidate that satisfies the throughput constraint,
            or None if no candidate can meet demand even with k pairs.
        """
        best: Optional[_PoolCandidate] = None
        for c in candidates:
            if k * c.throughput_per_pair >= (1 + alpha) * demand:
                if best is None or c.e_total_mj < best.e_total_mj:
                    best = c
        return best

    # ── Convenience ──────────────────────────────────────────────────────

    def warm_start(
        self,
        G: int,
        workload: WorkloadProfile,
        slo: SLOConfig,
    ) -> Tier1Solution:
        """Same as solve() but always returns a solution (max-freq fallback).

        If no feasible config exists, returns max-frequency configuration
        with minimal TP as a safe-start baseline.
        """
        sol = self.solve(G, workload, slo)
        if sol.feasible:
            return sol

        logger.warning("Tier1Solver.warm_start: no feasible ILP solution, "
                       "returning max-freq fallback")
        return Tier1Solution(
            k_p=1, k_d=1,
            tp_pa=1, tp_pf=1, tp_da=1, tp_df=1,
            f_pa=1410, f_pf=1410, f_da=1410, f_df=1410,
            gpu_used=4, feasible=True,
        )

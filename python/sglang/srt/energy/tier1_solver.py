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
class _OpCandidate:
    """One viable (tp, freq) for a single operator (A or F)."""
    tp: int
    freq: int
    t_us: float       # latency of this operator only
    e_mj: float       # energy of this operator only


@dataclass
class _AFPairCandidate:
    """A joint (tp_A, freq_A, tp_F, freq_F) AF-pair candidate.

    This is the correct optimization unit: one pair runs Attention on
    tp_A GPUs at freq_A, and FFN on tp_F GPUs at freq_F.
    """
    tp_a: int
    freq_a: int
    tp_f: int
    freq_f: int
    t_a_us: float
    t_f_us: float
    e_a_mj: float
    e_f_mj: float
    t_pair_us: float = 0.0         # t_A + t_F + t_comm (M=1 layer time)
    e_pair_mj: float = 0.0         # E_A + E_F (correct: no double-count)
    throughput_per_pair: float = 0.0

    def __post_init__(self):
        self.e_pair_mj = self.e_a_mj + self.e_f_mj


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

        # ── Phase 1: enumerate AF-pair candidates ──────────────────
        pairs_p = self._enumerate_pairs(
            "prefill", workload.bs_avg_p, workload.il_rep_p,
            slo.ttft_ms, beta, workload,
        )
        pairs_d = self._enumerate_pairs(
            "decode", workload.bs_avg_d, workload.il_rep_d,
            slo.tpot_ms, beta, workload, workload.ol_rep_d,
        )

        logger.info(
            "AF-pair candidates after pruning: prefill=%d decode=%d",
            len(pairs_p), len(pairs_d),
        )
        if not pairs_p or not pairs_d:
            logger.warning(
                "Tier1Solver: no feasible AF-pair for at least one phase"
            )
            return Tier1Solution(feasible=False)

        # ── Phase 2: Pareto filter per phase ─────────────────────
        pairs_p = self._pareto_filter_pairs(pairs_p)
        pairs_d = self._pareto_filter_pairs(pairs_d)
        logger.info(
            "After Pareto: prefill=%d decode=%d",
            len(pairs_p), len(pairs_d),
        )

        # ── Phase 3: enumerate (k_P, k_D) ───────────────────────
        best = self._search_kp_kd(G, pairs_p, pairs_d, workload, alpha)

        if best is None:
            logger.warning("Tier1Solver: no feasible (k_P,k_D) assignment found")
            return Tier1Solution(feasible=False)

        return best

    # ── Candidate enumeration ────────────────────────────────────────────

    def _enumerate_op_candidates(
        self,
        phase: str,
        op: str,
        bs: int,
        il: int,
        workload: WorkloadProfile,
        ol: Optional[int] = None,
    ) -> list[_OpCandidate]:
        """Enumerate all feasible (tp, freq) for one operator (A or F)."""
        candidates: list[_OpCandidate] = []
        for tp in _VALID_TP:
            if op == "A" and not self._check_attn_memory(
                phase, tp, bs, il, workload
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
                candidates.append(_OpCandidate(
                    tp=tp, freq=freq, t_us=t, e_mj=e,
                ))
        return candidates

    def _enumerate_pairs(
        self,
        phase: str,
        bs: int,
        il: int,
        slo_ms: float,
        beta: float,
        workload: WorkloadProfile,
        ol: Optional[int] = None,
    ) -> list[_AFPairCandidate]:
        """Enumerate all feasible AF-pair candidates with joint pruning.

        Searches over (tp_A, freq_A) × (tp_F, freq_F), applying:
          - Memory pruning per operator
          - Joint SLO pruning: t_A(tp_A,f_A) + t_F(tp_F,f_F) + comm
          - A/F balance pruning on the real pair
        """
        t_comm_us = self.pt.get_comm_us(
            bs, il if phase == "prefill" else 1
        )
        slo_per_layer_us = (slo_ms * 1000.0) / self.num_layers

        cands_a = self._enumerate_op_candidates(
            phase, "A", bs, il, workload, ol
        )
        cands_f = self._enumerate_op_candidates(
            phase, "F", bs, il, workload, ol
        )

        pairs: list[_AFPairCandidate] = []
        for ca in cands_a:
            for cf in cands_f:
                t_pair = ca.t_us + cf.t_us + t_comm_us
                if t_pair > slo_per_layer_us:
                    continue
                t_max = max(ca.t_us, cf.t_us)
                if t_max > 0 and abs(ca.t_us - cf.t_us) > beta * t_max:
                    continue
                thpt = self._pair_throughput(
                    bs, t_pair, self.num_layers, phase
                )
                pairs.append(_AFPairCandidate(
                    tp_a=ca.tp, freq_a=ca.freq,
                    tp_f=cf.tp, freq_f=cf.freq,
                    t_a_us=ca.t_us, t_f_us=cf.t_us,
                    e_a_mj=ca.e_mj, e_f_mj=cf.e_mj,
                    t_pair_us=t_pair,
                    throughput_per_pair=thpt,
                ))
        return pairs

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
    def _pareto_filter_pairs(
        pairs: list[_AFPairCandidate],
    ) -> list[_AFPairCandidate]:
        """Keep only non-dominated AF-pair configs.

        A pair dominates another if it has ≤ latency AND ≤ energy,
        with at least one strictly better.
        """
        if len(pairs) <= 1:
            return pairs
        sorted_p = sorted(pairs, key=lambda p: p.t_pair_us)
        pareto: list[_AFPairCandidate] = []
        min_energy = float("inf")
        for p in sorted_p:
            if p.e_pair_mj < min_energy:
                pareto.append(p)
                min_energy = p.e_pair_mj
        return pareto

    # ── (k_P, k_D) search ───────────────────────────────────────────────

    def _search_kp_kd(
        self,
        G: int,
        pairs_p: list[_AFPairCandidate],
        pairs_d: list[_AFPairCandidate],
        workload: WorkloadProfile,
        alpha: float,
        M: int = _DEFAULT_M,
    ) -> Optional[Tier1Solution]:
        """Enumerate viable (k_P, k_D) and select min-energy assignment."""
        best_solution: Optional[Tier1Solution] = None
        best_energy = float("inf")

        max_k = G // 2

        for k_p in range(1, max_k + 1):
            for k_d in range(1, max_k + 1):
                sel_p = self._select_pair(
                    pairs_p, k_p, workload.lambda_prefill, alpha,
                )
                sel_d = self._select_pair(
                    pairs_d, k_d, workload.n_active_decode, alpha,
                )
                if sel_p is None or sel_d is None:
                    continue

                gpu_per_p = sel_p.tp_a + sel_p.tp_f
                gpu_per_d = sel_d.tp_a + sel_d.tp_f
                gpu_used = k_p * gpu_per_p + k_d * gpu_per_d
                if gpu_used > G:
                    continue

                e_bubble_p = self._pair_bubble_energy(sel_p, M)
                e_bubble_d = self._pair_bubble_energy(sel_d, M)
                e_total = (
                    k_p * (sel_p.e_pair_mj + e_bubble_p)
                    + k_d * (sel_d.e_pair_mj + e_bubble_d)
                )

                if e_total < best_energy:
                    best_energy = e_total
                    sol = Tier1Solution(
                        k_p=k_p, k_d=k_d,
                        tp_pa=sel_p.tp_a, tp_pf=sel_p.tp_f,
                        tp_da=sel_d.tp_a, tp_df=sel_d.tp_f,
                        f_pa=sel_p.freq_a, f_pf=sel_p.freq_f,
                        f_da=sel_d.freq_a, f_df=sel_d.freq_f,
                        total_energy_mj_per_layer=e_total,
                        gpu_used=gpu_used,
                        feasible=True,
                    )
                    sol._gpu_avail = G - gpu_used
                    best_solution = sol

        return best_solution

    @staticmethod
    def _pair_bubble_energy(
        pair: _AFPairCandidate, M: int,
    ) -> float:
        """Compute pipeline bubble energy for an AF pair.

        E_bubble = P_idle * |t_A - t_F| * (M-1)/M  (mJ)
        """
        if M <= 1:
            return 0.0
        t_wait_us = abs(pair.t_a_us - pair.t_f_us)
        e_bubble_j = _P_IDLE_W * t_wait_us / 1_000_000.0 * (M - 1) / M
        return e_bubble_j * 1000.0

    @staticmethod
    def _select_pair(
        pairs: list[_AFPairCandidate],
        k: int,
        demand: float,
        alpha: float,
    ) -> Optional[_AFPairCandidate]:
        """Pick the min-energy AF-pair whose throughput * k meets demand.

        The throughput is the real pipeline-pair throughput, determined by
        the bottleneck side (max(t_A, t_F) + comm), not each pool
        independently.
        """
        best: Optional[_AFPairCandidate] = None
        for p in pairs:
            if k * p.throughput_per_pair >= (1 + alpha) * demand:
                if best is None or p.e_pair_mj < best.e_pair_mj:
                    best = p
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

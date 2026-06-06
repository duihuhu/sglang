#!/usr/bin/env python3
"""Real GPU benchmark: Tier 1 + Tier 2 energy evaluation on actual hardware.

Uses GPU 4,5 with Qwen3-32B to measure real inference energy under different
DVFS configurations. Tests the full control loop:
  1. Baseline: max freq (1410 MHz)
  2. Tier 1: ILP-planned baseline frequencies
  3. Tier 2: Tier 1 baseline + per-batch DVFS refinement

Measures:
  - Per-layer A/F latency at different frequencies
  - Total energy consumption (NVML hardware counter)
  - SLO satisfaction (TTFT, TPOT)
  - Frequency switching overhead
  - Idle power at each frequency

Requires:
  - GPU 4,5 available (A800-SXM4-80GB)
  - libdvfs_ctrl.so built
  - Qwen3-32B model at /models/Qwen/Qwen3-32B/

Run:
    CUDA_VISIBLE_DEVICES=4,5 /workspace/env/sglang-tier/bin/python \
        benchmark/energy_bench/bench_real_gpu.py

    # Quick mode (fewer iterations):
    CUDA_VISIBLE_DEVICES=4,5 /workspace/env/sglang-tier/bin/python \
        benchmark/energy_bench/bench_real_gpu.py --quick

    # Custom SLO:
    CUDA_VISIBLE_DEVICES=4,5 /workspace/env/sglang-tier/bin/python \
        benchmark/energy_bench/bench_real_gpu.py --quick --tpot-slo 100000 --ttft-slo 5000
"""

import argparse
import json
import logging
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "python"))
os.environ["DVFS_CTRL_LIB"] = str(
    Path(__file__).resolve().parents[2]
    / "benchmark/test_motivation/hucc/dvfs/libdvfs_ctrl.so"
)

from sglang.srt.layers.dvfs import DVFSController
from sglang.srt.energy.af_dvfs_controller import AFDVFSController, F_MAX
from sglang.srt.energy.tier1_solver import (
    Tier1Solver, Tier1Solution, WorkloadProfile, SLOConfig,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

MODEL_PATH = "/models/Qwen/Qwen3-32B/"
TEST_FREQS = [210, 450, 690, 930, 1170, 1410]
GPU_INDICES = [4, 5]

# Qwen3-32B: 64 layers, 8 KV heads, 128 head_dim, 5120 hidden
NUM_LAYERS = 64
NUM_KV_HEADS = 8
HEAD_DIM = 128
HIDDEN_SIZE = 5120

# Default SLOs — set looser than max-freq latency so DVFS has slack to exploit.
# Qwen3-32B decode at max freq takes ~84ms/step at bs=1, so TPOT_SLO must be >84ms.
DEFAULT_TTFT_SLO_MS = 5000.0    # 5s
DEFAULT_TPOT_SLO_US = 120000.0  # 120ms (gives ~30% slack vs 84ms at max freq)


# ═══════════════════════════════════════════════════════════════════════════
# Workload definitions
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class BenchRequest:
    input_ids: torch.Tensor
    output_len: int
    input_len: int


def generate_requests(
    tokenizer, n_requests: int, il_range=(256, 1024), ol_range=(32, 128),
    seed: int = 42,
) -> list[BenchRequest]:
    rng = np.random.default_rng(seed)
    requests = []
    for _ in range(n_requests):
        il = int(rng.integers(*il_range))
        ol = int(rng.integers(*ol_range))
        input_ids = torch.randint(100, 30000, (il,), dtype=torch.long)
        requests.append(BenchRequest(input_ids=input_ids, output_len=ol, input_len=il))
    return requests


# ═══════════════════════════════════════════════════════════════════════════
# Energy measurement utilities
# ═══════════════════════════════════════════════════════════════════════════

class EnergyMeter:
    """Measures GPU energy consumption using NVML hardware counters."""

    def __init__(self, gpu_indices: list[int]):
        self.controllers = {i: DVFSController(device_index=i) for i in gpu_indices}
        self._start_energy = {}
        self._start_time = 0.0

    def start(self):
        torch.cuda.synchronize()
        self._start_time = time.perf_counter()
        self._start_energy = {
            i: ctrl.get_energy_mj() for i, ctrl in self.controllers.items()
        }

    def stop(self) -> dict:
        torch.cuda.synchronize()
        elapsed_s = time.perf_counter() - self._start_time
        result = {"elapsed_s": elapsed_s, "per_gpu": {}}
        total_mj = 0.0
        for i, ctrl in self.controllers.items():
            e_end = ctrl.get_energy_mj()
            e_mj = e_end - self._start_energy[i]
            avg_power = e_mj / (elapsed_s * 1000) if elapsed_s > 0 else 0
            result["per_gpu"][i] = {"energy_mj": e_mj, "avg_power_w": avg_power}
            total_mj += e_mj
        result["total_energy_mj"] = total_mj
        result["total_avg_power_w"] = total_mj / (elapsed_s * 1000) if elapsed_s > 0 else 0
        return result

    def lock_freq(self, freq_mhz: int):
        for ctrl in self.controllers.values():
            ctrl.lock_sm_clock(freq_mhz)
        time.sleep(0.05)

    def reset_freq(self):
        for ctrl in self.controllers.values():
            ctrl.unlock_sm_clock()


# ═══════════════════════════════════════════════════════════════════════════
# Model loading
# ═══════════════════════════════════════════════════════════════════════════

def load_model(model_path: str):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    logger.info(f"Loading tokenizer from {model_path}")
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

    logger.info(f"Loading model from {model_path}")
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    model.eval()
    logger.info("Model loaded successfully")
    return model, tokenizer


# ═══════════════════════════════════════════════════════════════════════════
# Benchmark functions
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class FreqTestResult:
    freq_mhz: int
    phase: str          # "prefill" or "decode"
    bs: int
    seq_len: int
    latency_ms: float   # prefill: ms/forward; decode: ms/step (bs tokens)
    energy_mj: float    # prefill: mJ/forward; decode: mJ/step
    power_w: float
    throughput_tok_s: float


def bench_prefill_at_freq(
    model, freq: int, meter: EnergyMeter,
    bs: int = 1, seq_len: int = 512, n_warmup: int = 3, n_iter: int = 10,
) -> FreqTestResult:
    """Benchmark prefill at a specific frequency."""
    device = next(model.parameters()).device
    input_ids = torch.randint(100, 30000, (bs, seq_len), device=device)

    meter.lock_freq(freq)

    for _ in range(n_warmup):
        with torch.no_grad():
            model(input_ids, use_cache=False)

    meter.start()
    for _ in range(n_iter):
        with torch.no_grad():
            model(input_ids, use_cache=False)
    result = meter.stop()

    latency_ms = result["elapsed_s"] * 1000 / n_iter
    energy_per_iter = result["total_energy_mj"] / n_iter
    tokens = bs * seq_len * n_iter
    throughput = tokens / result["elapsed_s"]

    meter.reset_freq()
    return FreqTestResult(
        freq_mhz=freq, phase="prefill", bs=bs, seq_len=seq_len,
        latency_ms=latency_ms, energy_mj=energy_per_iter,
        power_w=result["total_avg_power_w"], throughput_tok_s=throughput,
    )


def bench_decode_at_freq(
    model, freq: int, meter: EnergyMeter,
    bs: int = 16, seq_len: int = 512, n_tokens: int = 32,
    n_warmup: int = 2, n_iter: int = 5,
) -> FreqTestResult:
    """Benchmark decode at a specific frequency.

    Each outer iteration does a fresh prefill to reset KV cache, then
    generates n_tokens. This ensures all measured steps have the same
    context length (unlike the previous version where KV cache accumulated).
    """
    device = next(model.parameters()).device
    input_ids = torch.randint(100, 30000, (bs, seq_len), device=device)

    meter.lock_freq(freq)

    # Warmup: one full cycle
    with torch.no_grad():
        outputs = model(input_ids, use_cache=True)
        pkv = outputs.past_key_values
    next_tok = outputs.logits[:, -1:, :].argmax(dim=-1)
    for _ in range(n_warmup):
        with torch.no_grad():
            out = model(next_tok, past_key_values=pkv, use_cache=True)
            next_tok = out.logits[:, -1:, :].argmax(dim=-1)
            pkv = out.past_key_values

    meter.start()
    for _ in range(n_iter):
        # Fresh prefill each iteration → same context length every time
        with torch.no_grad():
            outputs = model(input_ids, use_cache=True)
            pkv = outputs.past_key_values
        next_tok = outputs.logits[:, -1:, :].argmax(dim=-1)
        for _ in range(n_tokens):
            with torch.no_grad():
                out = model(next_tok, past_key_values=pkv, use_cache=True)
                next_tok = out.logits[:, -1:, :].argmax(dim=-1)
                pkv = out.past_key_values
    result = meter.stop()

    total_steps = n_tokens * n_iter
    latency_per_step_ms = result["elapsed_s"] * 1000 / total_steps
    energy_per_step = result["total_energy_mj"] / total_steps
    total_tokens = bs * n_tokens * n_iter
    throughput = total_tokens / result["elapsed_s"]

    meter.reset_freq()
    return FreqTestResult(
        freq_mhz=freq, phase="decode", bs=bs, seq_len=seq_len,
        latency_ms=latency_per_step_ms, energy_mj=energy_per_step,
        power_w=result["total_avg_power_w"], throughput_tok_s=throughput,
    )


def bench_freq_switching_overhead(meter: EnergyMeter, n_iter: int = 20) -> dict:
    """Measure frequency switching latency via lock_sm_clock()."""
    results = {}
    ctrl = list(meter.controllers.values())[0]
    for target_freq in TEST_FREQS:
        ctrl.lock_sm_clock(F_MAX)
        time.sleep(0.01)
        latencies = []
        for _ in range(n_iter):
            t0 = time.perf_counter_ns()
            ctrl.lock_sm_clock(target_freq)
            t1 = time.perf_counter_ns()
            latencies.append((t1 - t0) / 1000.0)
            ctrl.lock_sm_clock(F_MAX)
            time.sleep(0.005)
        results[target_freq] = {
            "mean_us": np.mean(latencies),
            "p99_us": np.percentile(latencies, 99),
        }
    ctrl.unlock_sm_clock()
    return results


# ═══════════════════════════════════════════════════════════════════════════
# Real-data predictor (LUT from fresh profiling)
# ═══════════════════════════════════════════════════════════════════════════

_A_SPLIT_PREFILL = 0.30   # attention fraction of total layer time (prefill)
_A_SPLIT_DECODE = 0.35    # attention fraction of total layer time (decode)

# WARNING: A/F split ratios are estimates, not measured.  In a real AFD
# deployment these come from separate A-side and F-side profiling.  Here we
# split the unified-model measurement using ratios derived from Qwen3-32B's
# arithmetic intensity breakdown.  Changing these ratios changes DVFS decisions.


def _make_real_predictor(prefill_results, decode_results, num_layers=NUM_LAYERS):
    """Build a LUT-based predictor from freshly-collected profiling data."""

    class RealDataPredictor:
        def __init__(self, prefill_data, decode_data):
            self._prefill = {}   # (freq, bs, il) → (t_layer_us, e_layer_mj)
            for r in prefill_data:
                key = (r.freq_mhz, r.bs, r.seq_len)
                self._prefill[key] = (
                    r.latency_ms * 1000 / num_layers,
                    r.energy_mj / num_layers,
                )
            self._decode = {}    # (freq, bs) → (t_step_us, e_step_mj)
            for r in decode_data:
                key = (r.freq_mhz, r.bs)
                self._decode[key] = (
                    r.latency_ms * 1000 / num_layers,
                    r.energy_mj / num_layers,
                )

        def _nearest_prefill(self, freq, bs, il):
            key = (freq, bs, il)
            if key in self._prefill:
                return key
            return min(self._prefill.keys(),
                       key=lambda k: abs(k[2] - il) + abs(k[1] - bs) + abs(k[0] - freq))

        def _nearest_decode(self, freq, bs):
            key = (freq, bs)
            if key in self._decode:
                return key
            return min(self._decode.keys(),
                       key=lambda k: abs(k[1] - bs) + abs(k[0] - freq))

        def predict_latency(self, phase, op, tp, freq, bs, il, ol=None):
            if phase == "prefill":
                key = self._nearest_prefill(freq, bs, il)
                lat_us = self._prefill[key][0]
            else:
                key = self._nearest_decode(freq, bs)
                lat_us = self._decode[key][0]
            # Split into A/F components
            split = _A_SPLIT_PREFILL if phase == "prefill" else _A_SPLIT_DECODE
            if op == "A":
                lat_us *= split
            else:
                lat_us *= (1.0 - split)

            class _R:
                value = lat_us
            return _R()

        def predict_energy(self, phase, op, tp, freq, bs, il, ol=None):
            if phase == "prefill":
                key = self._nearest_prefill(freq, bs, il)
                e_mj = self._prefill[key][1]
            else:
                key = self._nearest_decode(freq, bs)
                e_mj = self._decode[key][1]
            split = _A_SPLIT_PREFILL if phase == "prefill" else _A_SPLIT_DECODE
            if op == "A":
                e_mj *= split
            else:
                e_mj *= (1.0 - split)

            class _R:
                value = e_mj
            return _R()

    return RealDataPredictor(prefill_results, decode_results)


# ═══════════════════════════════════════════════════════════════════════════
# End-to-end workload runner
# ═══════════════════════════════════════════════════════════════════════════

def run_workload(
    model, meter: EnergyMeter,
    dvfs_ctrl: Optional[AFDVFSController],
    requests: list[BenchRequest],
    ttft_slo_ms: float,
    tpot_slo_us: float,
    mode: str,                # "max", "tier1", "tier2"
    tier1_freqs: Optional[dict] = None,  # {"prefill": f_mhz, "decode": f_mhz}
) -> dict:
    """Run a workload under one control mode and measure energy + SLO.

    Modes:
      max    — lock to F_MAX for everything
      tier1  — lock to tier1_freqs (static baseline from ILP)
      tier2  — tier1 baseline + per-batch DVFS refinement
    """
    device = next(model.parameters()).device
    e_start = {i: c.get_energy_mj() for i, c in meter.controllers.items()}
    t_start = time.perf_counter()

    n_ttft_viol = 0
    n_tpot_viol = 0
    ttft_samples = []
    tpot_samples = []
    freq_history = []
    switch_count = 0

    for req in requests:
        input_ids = req.input_ids.unsqueeze(0).to(device)
        max_decode_tokens = min(req.output_len, 32)

        # ── Prefill ──────────────────────────────────────────────────
        if mode == "max":
            meter.lock_freq(F_MAX)
        elif mode == "tier1" and tier1_freqs:
            meter.lock_freq(tier1_freqs["prefill"])
        elif mode == "tier2" and dvfs_ctrl:
            # Use select_freq_prefill with actual SLO slack
            decision = dvfs_ctrl.select_freq_prefill(
                bs=1, il=req.input_len,
                slack_us=ttft_slo_ms * 1000, M=2,
            )
            if decision.f_a != F_MAX or decision.f_f != F_MAX:
                meter.lock_freq(decision.f_a)
                freq_history.append(("prefill", decision.f_a, decision.f_f))
                switch_count += 1
            else:
                meter.lock_freq(F_MAX)
        else:
            meter.lock_freq(F_MAX)

        torch.cuda.synchronize()
        t_pf = time.perf_counter()
        with torch.no_grad():
            outputs = model(input_ids, use_cache=True)
            pkv = outputs.past_key_values
        torch.cuda.synchronize()
        ttft_ms = (time.perf_counter() - t_pf) * 1000
        ttft_samples.append(ttft_ms)
        if ttft_ms > ttft_slo_ms:
            n_ttft_viol += 1

        # ── Decode ───────────────────────────────────────────────────
        if mode == "max":
            pass  # already at F_MAX
        elif mode == "tier1" and tier1_freqs:
            meter.lock_freq(tier1_freqs["decode"])
        elif mode == "tier2" and dvfs_ctrl:
            # Use proper decode window semantics
            dvfs_ctrl.tick_decode_iteration()
            # Compute avg iteration time from previous samples
            avg_tpot = np.mean(tpot_samples[-5:]) if len(tpot_samples) >= 5 else 0.0
            if avg_tpot > 0:
                dvfs_ctrl.compute_window_size(avg_tpot)
            reason = dvfs_ctrl.should_reevaluate_decode(
                1, current_tpot_us=avg_tpot, slo_tpot_us=tpot_slo_us,
            )
            if reason:
                decision = dvfs_ctrl.select_freq_decode(
                    bs=1, il=req.input_len, ol=req.output_len,
                    slo_tpot_us=tpot_slo_us, M=2, reeval_reason=reason,
                )
                if decision.switched:
                    meter.lock_freq(decision.f_a)
                    freq_history.append(("decode", decision.f_a, decision.f_f))
                    switch_count += 1

        next_tok = outputs.logits[:, -1:, :].argmax(dim=-1)
        for _ in range(max_decode_tokens):
            torch.cuda.synchronize()
            t_tok = time.perf_counter()
            with torch.no_grad():
                out = model(next_tok, past_key_values=pkv, use_cache=True)
                next_tok = out.logits[:, -1:, :].argmax(dim=-1)
                pkv = out.past_key_values
            torch.cuda.synchronize()
            tpot_us = (time.perf_counter() - t_tok) * 1_000_000
            tpot_samples.append(tpot_us)
            if tpot_us > tpot_slo_us:
                n_tpot_viol += 1

    meter.reset_freq()
    t_end = time.perf_counter()
    elapsed = t_end - t_start

    total_energy = sum(
        c.get_energy_mj() - e_start[i] for i, c in meter.controllers.items()
    )

    return {
        "mode": mode,
        "total_energy_mj": total_energy,
        "elapsed_s": elapsed,
        "n_requests": len(requests),
        "ttft_p50_ms": float(np.percentile(ttft_samples, 50)) if ttft_samples else 0,
        "ttft_p99_ms": float(np.percentile(ttft_samples, 99)) if ttft_samples else 0,
        "tpot_p50_us": float(np.percentile(tpot_samples, 50)) if tpot_samples else 0,
        "tpot_p99_us": float(np.percentile(tpot_samples, 99)) if tpot_samples else 0,
        "n_ttft_viol": n_ttft_viol,
        "n_tpot_viol": n_tpot_viol,
        "n_total_steps": len(tpot_samples),
        "slo_satisfaction_pct": (1 - n_tpot_viol / max(len(tpot_samples), 1)) * 100,
        "avg_power_w": total_energy / (elapsed * 1000),
        "n_freq_switches": switch_count,
    }


# ═══════════════════════════════════════════════════════════════════════════
# Tier 1 ILP solver demo
# ═══════════════════════════════════════════════════════════════════════════

def run_tier1_demo(
    model, meter: EnergyMeter,
    results: dict,
    ttft_slo_ms: float, tpot_slo_us: float,
    quick: bool,
):
    """Solve Tier 1 ILP using pre-recorded profile data, print solution.

    Uses the existing ProfileTable (trained on offline profiling data)
    rather than the fresh benchmark data — this matches how Tier 1 works
    in production (plans from pre-recorded profiles before deployment).
    """
    from sglang.srt.energy.profile_table import ProfileTable

    logger.info("\n" + "=" * 60)
    logger.info("Test 5: Tier 1 ILP Solver (pre-recorded profile data)")
    logger.info("=" * 60)

    prefill_path = str(
        Path(__file__).resolve().parents[2]
        / "benchmark/test_motivation/hucc/paper/prefill_data_v1.txt"
    )
    decode_path = str(
        Path(__file__).resolve().parents[2]
        / "benchmark/test_motivation/hucc/paper/decode_data_v1.txt"
    )
    model_dir = str(
        Path(__file__).resolve().parents[2]
        / "benchmark/test_motivation/energy_models"
    )

    if not Path(prefill_path).exists():
        logger.warning("Profile data not found at %s, skipping Tier 1", prefill_path)
        return None

    pt = ProfileTable(
        prefill_path=prefill_path,
        decode_path=decode_path,
        energy_model_dir=model_dir,
    )

    solver = Tier1Solver(
        profile_table=pt,
        num_layers=NUM_LAYERS,
        num_kv_heads=NUM_KV_HEADS,
        head_dim=HEAD_DIM,
        hidden_size=HIDDEN_SIZE,
        gpu_mem_gb=80.0,
    )

    # Build workload profile from args
    wl = WorkloadProfile(
        lambda_prefill=2.0 if quick else 5.0,
        n_active_decode=4 if quick else 8,
        il_rep_p=1024, bs_avg_p=4,
        il_rep_d=512, ol_rep_d=256, bs_avg_d=8,
    )
    slo = SLOConfig(ttft_ms=ttft_slo_ms, tpot_ms=tpot_slo_us / 1000.0)

    print(f"\n  Workload: λ={wl.lambda_prefill} req/s, N_active={wl.n_active_decode}")
    print(f"  SLO: TTFT={slo.ttft_ms}ms, TPOT={slo.tpot_ms}ms")
    print(f"  GPU budget: 8")

    sol = solver.solve(G=8, workload=wl, slo=slo)

    if not sol.feasible:
        logger.warning("Tier 1: infeasible — no config meets all constraints")
        sol = solver.warm_start(G=8, workload=wl, slo=slo)
        print(f"  → WARM START fallback (max freq, min TP)")

    print(f"\n  Solution:")
    print(f"  {'Pool':<8} {'tp':<6} {'freq(MHz)':<12}")
    print(f"  {'─' * 26}")
    print(f"  {'PA':<8} {sol.tp_pa:<6} {sol.f_pa:<12}")
    print(f"  {'PF':<8} {sol.tp_pf:<6} {sol.f_pf:<12}")
    print(f"  {'DA':<8} {sol.tp_da:<6} {sol.f_da:<12}")
    print(f"  {'DF':<8} {sol.tp_df:<6} {sol.f_df:<12}")
    print(f"  k_P={sol.k_p}  k_D={sol.k_d}  E/layer={sol.total_energy_mj_per_layer:.2f}mJ  "
          f"GPU={sol.gpu_used}/8")

    results["tier1_solution"] = sol.to_dict()

    # Build frequency map for the unified-model benchmark:
    # Use max(A, F) per phase to be conservative.
    tier1_freqs = {
        "prefill": max(sol.f_pa, sol.f_pf),
        "decode": max(sol.f_da, sol.f_df),
    }
    print(f"\n  Unified-model mapping: prefill={tier1_freqs['prefill']}MHz "
          f"decode={tier1_freqs['decode']}MHz "
          f"(max of A/F per phase)")
    return tier1_freqs, sol.f_da, sol.f_df


# ═══════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Real GPU energy benchmark")
    parser.add_argument("--quick", action="store_true",
                        help="Quick mode: fewer iterations")
    parser.add_argument("--model-path", type=str, default=MODEL_PATH)
    parser.add_argument("--output-json", type=str, default=None)
    parser.add_argument("--skip-model", action="store_true",
                        help="Skip model loading, only test DVFS overhead")
    parser.add_argument("--ttft-slo", type=float, default=DEFAULT_TTFT_SLO_MS,
                        help=f"TTFT SLO in ms (default: {DEFAULT_TTFT_SLO_MS})")
    parser.add_argument("--tpot-slo", type=float, default=DEFAULT_TPOT_SLO_US,
                        help=f"TPOT SLO in us (default: {DEFAULT_TPOT_SLO_US})")
    parser.add_argument("--gpus", type=str, default="4,5",
                        help="GPU indices to use (default: 4,5)")
    args = parser.parse_args()

    gpu_indices = [int(x.strip()) for x in args.gpus.split(",")]
    meter = EnergyMeter(gpu_indices)
    results = {}

    try:
        _run_all_tests(meter, args, results)
    except KeyboardInterrupt:
        logger.warning("Interrupted by user")
    except Exception as e:
        logger.error("Benchmark failed: %s", e, exc_info=True)
    finally:
        meter.reset_freq()
        logger.info("GPU frequencies reset to default")

    if args.output_json and results:
        Path(args.output_json).write_text(json.dumps(results, indent=2, default=str))
        logger.info("Results saved to %s", args.output_json)


def _run_all_tests(meter: EnergyMeter, args, results: dict):
    ttft_slo = args.ttft_slo
    tpot_slo = args.tpot_slo

    # ── Test 1: Frequency switching overhead ──────────────────────
    logger.info("=" * 60)
    logger.info("Test 1: Frequency switching overhead")
    logger.info("=" * 60)
    switch_results = bench_freq_switching_overhead(
        meter, n_iter=10 if args.quick else 20,
    )
    results["freq_switch_overhead"] = switch_results
    print("\n  Freq switch overhead (1410 → target):")
    print(f"  {'Target':<10} {'Mean(us)':<12} {'P99(us)'}")
    for freq, data in sorted(switch_results.items()):
        print(f"  {freq:<10} {data['mean_us']:<12.1f} {data['p99_us']:.1f}")

    # ── Test 2: Idle power per frequency ──────────────────────────
    logger.info("\n" + "=" * 60)
    logger.info("Test 2: Idle power at each frequency")
    logger.info("=" * 60)
    idle_results = {}
    for freq in TEST_FREQS:
        meter.lock_freq(freq)
        time.sleep(0.5)
        meter.start()
        time.sleep(2.0 if not args.quick else 1.0)
        r = meter.stop()
        idle_results[freq] = r["total_avg_power_w"]
    meter.reset_freq()
    results["idle_power"] = idle_results
    print("\n  Idle power per frequency:")
    for freq, power in sorted(idle_results.items()):
        print(f"  {freq} MHz: {power:.1f} W")

    if args.skip_model:
        logger.info("Skipping model tests (--skip-model)")
        return

    # ── Load model ────────────────────────────────────────────────
    model, tokenizer = load_model(args.model_path)

    # ── Test 3: Prefill profiling ─────────────────────────────────
    logger.info("\n" + "=" * 60)
    logger.info("Test 3: Prefill profiling at different frequencies")
    logger.info("=" * 60)
    prefill_results = []
    n_iter = 3 if args.quick else 8
    for freq in TEST_FREQS:
        for bs in [1, 4]:
            for seq_len in [512, 1024]:
                r = bench_prefill_at_freq(model, freq, meter,
                                          bs=bs, seq_len=seq_len,
                                          n_warmup=2, n_iter=n_iter)
                prefill_results.append(r)
                print(f"  prefill freq={freq} bs={bs} seq={seq_len}: "
                      f"lat={r.latency_ms:.1f}ms energy={r.energy_mj:.0f}mJ "
                      f"power={r.power_w:.0f}W thpt={r.throughput_tok_s:.0f}tok/s")
    results["prefill"] = [vars(r) for r in prefill_results]

    # ── Test 4: Decode profiling ──────────────────────────────────
    logger.info("\n" + "=" * 60)
    logger.info("Test 4: Decode profiling at different frequencies")
    logger.info("=" * 60)
    decode_results = []
    n_iter_d = 2 if args.quick else 5
    for freq in TEST_FREQS:
        for bs in [1, 8]:
            r = bench_decode_at_freq(model, freq, meter,
                                     bs=bs, seq_len=512, n_tokens=16,
                                     n_warmup=2, n_iter=n_iter_d)
            decode_results.append(r)
            print(f"  decode freq={freq} bs={bs}: "
                  f"lat={r.latency_ms:.2f}ms/step energy={r.energy_mj:.0f}mJ/step "
                  f"power={r.power_w:.0f}W thpt={r.throughput_tok_s:.0f}tok/s")
    results["decode"] = [vars(r) for r in decode_results]

    # ── Test 5: Tier 1 ILP solver ─────────────────────────────────
    tier1_out = run_tier1_demo(
        model, meter, results,
        ttft_slo_ms=ttft_slo, tpot_slo_us=tpot_slo, quick=args.quick,
    )
    if tier1_out is None:
        tier1_freqs = None
        tier1_baseline_f_a = F_MAX
        tier1_baseline_f_f = F_MAX
    else:
        tier1_freqs, tier1_baseline_f_a, tier1_baseline_f_f = tier1_out

    # ── Test 6: E2E 3-way comparison ──────────────────────────────
    logger.info("\n" + "=" * 60)
    logger.info("Test 6: End-to-end workload comparison")
    logger.info("=" * 60)
    n_reqs = 5 if args.quick else 12
    requests = generate_requests(tokenizer, n_reqs,
                                 il_range=(256, 512), ol_range=(32, 64))

    # 6a. Baseline: max frequency
    logger.info("  Running baseline (1410 MHz)...")
    r_max = run_workload(
        model, meter, None, requests,
        ttft_slo_ms=ttft_slo, tpot_slo_us=tpot_slo,
        mode="max",
    )
    results["e2e_max"] = r_max

    # 6b. Tier 1: ILP baseline frequencies (static)
    if tier1_freqs:
        logger.info("  Running Tier 1 (ILP baseline: prefill=%d decode=%d)...",
                    tier1_freqs["prefill"], tier1_freqs["decode"])
        r_tier1 = run_workload(
            model, meter, None, requests,
            ttft_slo_ms=ttft_slo, tpot_slo_us=tpot_slo,
            mode="tier1", tier1_freqs=tier1_freqs,
        )
        results["e2e_tier1"] = r_tier1
    else:
        r_tier1 = None

    # 6c. Tier 2: Tier 1 baseline + per-batch DVFS refinement
    if tier1_freqs:
        logger.info("  Running Tier 2 (ILP baseline + DVFS refinement)...")
        mock_predictor = _make_real_predictor(prefill_results, decode_results)
        dvfs_ctrl = AFDVFSController(
            predictor=mock_predictor, num_layers=NUM_LAYERS,
            tp_a=2, tp_f=2,
            baseline_f_a=tier1_baseline_f_a,
            baseline_f_f=tier1_baseline_f_f,
        )
        r_tier2 = run_workload(
            model, meter, dvfs_ctrl, requests,
            ttft_slo_ms=ttft_slo, tpot_slo_us=tpot_slo,
            mode="tier2", tier1_freqs=tier1_freqs,
        )
        results["e2e_tier2"] = r_tier2
    else:
        # Fallback: Tier 2 from max-freq baseline
        logger.info("  Running Tier 2 (from max-freq baseline)...")
        mock_predictor = _make_real_predictor(prefill_results, decode_results)
        dvfs_ctrl = AFDVFSController(
            predictor=mock_predictor, num_layers=NUM_LAYERS,
            tp_a=2, tp_f=2,
            baseline_f_a=F_MAX, baseline_f_f=F_MAX,
        )
        r_tier2 = run_workload(
            model, meter, dvfs_ctrl, requests,
            ttft_slo_ms=ttft_slo, tpot_slo_us=tpot_slo,
            mode="tier2", tier1_freqs=None,
        )
        results["e2e_tier2"] = r_tier2

    # ── Summary ───────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  SUMMARY")
    print("=" * 60)
    print(f"  SLO budget: TTFT={ttft_slo}ms, TPOT={tpot_slo}us")
    print()
    print(f"  {'Mode':<16} {'Energy(mJ)':<14} {'vsMax':<10} {'Time(s)':<10} "
          f"{'TPOT_p99':<12} {'SLO%':<8} {'Switches'}")
    print(f"  {'─' * 70}")
    print(f"  {'max':<16} {r_max['total_energy_mj']:<14.0f} {'—':<10} "
          f"{r_max['elapsed_s']:<10.1f} "
          f"{r_max['tpot_p99_us']:<12.0f} "
          f"{r_max['slo_satisfaction_pct']:<8.1f} "
          f"{r_max['n_freq_switches']}")

    if r_tier1:
        saving1 = (1 - r_tier1['total_energy_mj'] / r_max['total_energy_mj']) * 100
        print(f"  {'tier1':<16} {r_tier1['total_energy_mj']:<14.0f} "
              f"{saving1:+.1f}%{'':5} "
              f"{r_tier1['elapsed_s']:<10.1f} "
              f"{r_tier1['tpot_p99_us']:<12.0f} "
              f"{r_tier1['slo_satisfaction_pct']:<8.1f} "
              f"{r_tier1['n_freq_switches']}")

    saving2 = (1 - r_tier2['total_energy_mj'] / r_max['total_energy_mj']) * 100
    print(f"  {'tier2':<16} {r_tier2['total_energy_mj']:<14.0f} "
          f"{saving2:+.1f}%{'':5} "
          f"{r_tier2['elapsed_s']:<10.1f} "
          f"{r_tier2['tpot_p99_us']:<12.0f} "
          f"{r_tier2['slo_satisfaction_pct']:<8.1f} "
          f"{r_tier2['n_freq_switches']}")

    print()
    if r_tier1 and r_tier2['total_energy_mj'] < r_tier1['total_energy_mj']:
        dvfs_delta = (1 - r_tier2['total_energy_mj'] / r_tier1['total_energy_mj']) * 100
        print(f"  Tier 2 DVFS saves an additional {dvfs_delta:.1f}% vs Tier 1 alone "
              f"(per-batch refinement on top of ILP baseline).")
    elif r_tier1:
        print(f"  Tier 2 did not find additional savings vs Tier 1 baseline "
              f"(no slack to exploit).")


if __name__ == "__main__":
    main()

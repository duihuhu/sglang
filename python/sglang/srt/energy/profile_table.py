"""B-1: AF Profile Table — data layer for Tier 1 ILP and Tier 2 DVFS.

Loads raw profiling data and trained energy/latency models, providing a unified
query API.  Prefers exact LUT matches over model predictions — exact matches
carry zero approximation error.

Data sources
  - prefill_data_v1.txt  (906 rows)
  - decode_data_v1.txt   (7530 rows)
  - energy_models/*.pkl  (LUT + LinearReg, trained by energy_model.py)
  - af_comm_*.txt        (optional; embedded defaults from NVLink P2P measurements)

Usage
    pt = ProfileTable(
        prefill_path="benchmark/test_motivation/hucc/paper/prefill_data_v1.txt",
        decode_path="benchmark/test_motivation/hucc/paper/decode_data_v1.txt",
        energy_model_dir="benchmark/test_motivation/energy_models",
    )
    t_a, t_f = pt.query_latency("prefill", tp=4, freq=930, bs=4, il=1024)
    e_a, e_f = pt.query_energy("decode", tp=2, freq=690, bs=16, il=512, ol=128)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from sglang.srt.energy.af_profile_predictor import (
    AFProfilePredictor,
    VALID_FREQS,
)

logger = logging.getLogger(__name__)

# ── Communication overhead defaults ────────────────────────────────────
# Measured on A800-80GB SXM, NVLink P2P, tensor = (bs, seq_len, H=5120) bf16.
# From system_design_unified.md Section 1.
# Key insight: decode (seq_len=1) comm is ~37-42 us regardless of bs,
# prefill comm grows with seq_len × bs.

_AF_COMM_DECODE_US = 40.0        # seq_len=1, typical decode
_AF_COMM_PREFILL_FALLBACK_US = 283.0   # seq_len=1024, bs=4 — mid-range representative


# ═══════════════════════════════════════════════════════════════════════
# Data loading helpers
# ═══════════════════════════════════════════════════════════════════════

def _load_profile_file(path: str) -> pd.DataFrame:
    """Load a tab-separated profile file, skipping the comment/header preamble.

    Files have this structure:
        [P] Prefill                            # line 0 — comment
        tp\tinput_len\t...\tF_energy_mj        # line 1 — header
        1\t128\t1\t210\t1\t1031.16\t...        # line 2+ — data
    """
    with open(path) as f:
        lines = f.readlines()

    # Find the header line (starts with "tp\t")
    header_idx = None
    for i, line in enumerate(lines):
        if line.startswith("tp\t"):
            header_idx = i
            break
    if header_idx is None:
        raise ValueError(f"No header line found in {path}")

    from io import StringIO
    df = pd.read_csv(StringIO("".join(lines[header_idx:])), sep="\t")
    df.columns = df.columns.str.strip()
    return df


# ═══════════════════════════════════════════════════════════════════════
# Profile table
# ═══════════════════════════════════════════════════════════════════════

@dataclass
class  LayerMetrics:
    """Per-layer latency and energy for one operator configuration."""
    t_a_us: float
    t_f_us: float
    e_a_mj: float
    e_f_mj: float
    source: str = ""  # "exact", "LUT", "GBDT", "LinearReg"


class ProfileTable:
    """Unified query layer over raw profile data + trained prediction models.

    Args:
        prefill_path: Path to prefill_data_v1.txt.
        decode_path:  Path to decode_data_v1.txt.
        energy_model_dir: Directory containing *.pkl model files.
        comm_prefill_path: Optional path to af_comm_p2p.txt or e2e.
    """

    def __init__(
        self,
        prefill_path: str,
        decode_path: str,
        energy_model_dir: str,
        comm_prefill_path: Optional[str] = None,
    ):
        # ── Raw data ────────────────────────────────────────────────
        self._df_p = _load_profile_file(prefill_path)
        self._df_d = _load_profile_file(decode_path)
        logger.info(
            "ProfileTable: loaded %d prefill + %d decode rows",
            len(self._df_p), len(self._df_d),
        )

        # ── Energy/latency predictor (LUT + GBDT/LinearReg fallback) ─
        self.predictor = AFProfilePredictor(energy_model_dir)

        # ── Communication overhead ───────────────────────────────────
        self._comm_prefill: dict[tuple, float] = {}
        if comm_prefill_path and Path(comm_prefill_path).exists():
            self._load_comm_data(comm_prefill_path)
        self._comm_decode_us = _AF_COMM_DECODE_US
        self._comm_prefill_fallback_us = _AF_COMM_PREFILL_FALLBACK_US

        # ── Pre-built index for exact lookups ────────────────────────
        self._index_p: dict[tuple, tuple] = {}  # (tp,freq,bs,il) → (A,F,A_energy_mj,F_energy_mj)
        self._index_d: dict[tuple, tuple] = {}  # (tp,freq,bs,il,ol) → (A,F,A_energy_mj,F_energy_mj)
        self._build_exact_index()

        self.freqs = VALID_FREQS
        self.valid_tp = [1, 2, 4, 8]

    # ── Index ────────────────────────────────────────────────────────────

    def _build_exact_index(self):
        """Pre-compute lookup dictionaries for O(1) exact-match queries."""
        for _, row in self._df_p.iterrows():
            key = (int(row["tp"]), int(row["gpu_clock"]),
                   int(row["batch_size"]), int(row["input_len"]))
            self._index_p[key] = (
                float(row["A"]), float(row["F"]),
                float(row["A_energy_mj"]), float(row["F_energy_mj"]),
            )
        for _, row in self._df_d.iterrows():
            key = (int(row["tp"]), int(row["gpu_clock"]),
                   int(row["batch_size"]), int(row["input_len"]),
                   int(row["output_len"]))
            self._index_d[key] = (
                float(row["A"]), float(row["F"]),
                float(row["A_energy_mj"]), float(row["F_energy_mj"]),
            )

    # ── Communication overhead ───────────────────────────────────────────

    def _load_comm_data(self, path: str):
        """Load af_comm measurement data (mode / n_gpus / batch_size / seq_len / data_bytes / latency_us / bandwidth_gbps)."""
        df = pd.read_csv(path, sep="\t", skiprows=1)
        for _, row in df.iterrows():
            key = (int(row["batch_size"]), int(row["seq_len"]))
            self._comm_prefill[key] = float(row["latency_us"])
        logger.info("ProfileTable: loaded %d comm overhead entries", len(self._comm_prefill))

    def get_comm_us(self, bs: int, seq_len: int) -> float:
        """Return AF communication overhead (microseconds) for a given tensor shape.

        For decode (seq_len=1) the overhead is roughly constant (~40 us).
        For prefill it scales with tensor size; falls back to the mid-range
        representative (bs=4, seq_len=1024, ~283 us) when no exact data exists.
        """
        if seq_len == 1:
            return self._comm_decode_us
        key = (bs, seq_len)
        if key in self._comm_prefill:
            return self._comm_prefill[key]
        return self._comm_prefill_fallback_us

    # ── Query API ────────────────────────────────────────────────────────

    def query_latency(
        self, phase: str, tp: int, freq: int,
        bs: int, il: int, ol: Optional[int] = None,
    ) -> tuple[float, float]:
        """Return (t_A_us, t_F_us) for one layer.  Prefers exact match, falls back to predictor."""
        exact = self._try_exact(phase, tp, freq, bs, il, ol)
        if exact is not None:
            return exact[0], exact[1]

        pa = self.predictor.predict_latency(phase, "A", tp, freq, bs, il, ol)
        pf = self.predictor.predict_latency(phase, "F", tp, freq, bs, il, ol)
        return pa.value, pf.value

    def query_energy(
        self, phase: str, tp: int, freq: int,
        bs: int, il: int, ol: Optional[int] = None,
    ) -> tuple[float, float]:
        """Return (E_A_mJ, E_F_mJ) for one layer.  Prefers exact match, falls back to predictor."""
        exact = self._try_exact(phase, tp, freq, bs, il, ol)
        if exact is not None:
            return exact[2], exact[3]

        pa = self.predictor.predict_energy(phase, "A", tp, freq, bs, il, ol)
        pf = self.predictor.predict_energy(phase, "F", tp, freq, bs, il, ol)
        return pa.value, pf.value

    def query_metrics(
        self, phase: str, tp: int, freq: int,
        bs: int, il: int, ol: Optional[int] = None,
    ) -> LayerMetrics:
        """Return full LayerMetrics (t_A, t_F, E_A, E_F) in one call."""
        exact = self._try_exact(phase, tp, freq, bs, il, ol)
        if exact is not None:
            return LayerMetrics(
                t_a_us=exact[0], t_f_us=exact[1],
                e_a_mj=exact[2], e_f_mj=exact[3],
                source="exact",
            )

        la = self.predictor.predict_latency(phase, "A", tp, freq, bs, il, ol)
        lf = self.predictor.predict_latency(phase, "F", tp, freq, bs, il, ol)
        ea = self.predictor.predict_energy(phase, "A", tp, freq, bs, il, ol)
        ef = self.predictor.predict_energy(phase, "F", tp, freq, bs, il, ol)
        return LayerMetrics(
            t_a_us=la.value, t_f_us=lf.value,
            e_a_mj=ea.value, e_f_mj=ef.value,
            source=la.model_type,
        )

    def _try_exact(
        self, phase: str, tp: int, freq: int,
        bs: int, il: int, ol: Optional[int] = None,
    ) -> Optional[tuple]:
        """Return (A, F, A_energy_mj, F_energy_mj) if an exact match exists."""
        if phase == "prefill":
            return self._index_p.get((tp, freq, bs, il))
        return self._index_d.get((tp, freq, bs, il, ol))

    # ── Batch query for ILP ──────────────────────────────────────────────

    def enumerate_configs(
        self, phase: str, tp: int, bs: int, il: int,
        ol: Optional[int] = None,
        freqs: Optional[list[int]] = None,
    ) -> list[dict]:
        """Enumerate all (freq) metrics for a given (phase, tp, workload).

        Returns a list of dicts with keys: freq, t_a_us, t_f_us, e_a_mj, e_f_mj.
        Faster than calling query_metrics() individually because we can batch
        predictor calls.
        """
        if freqs is None:
            freqs = self.freqs
        results = []
        for f in freqs:
            m = self.query_metrics(phase, tp, f, bs, il, ol)
            results.append({
                "freq": f,
                "t_a_us": m.t_a_us, "t_f_us": m.t_f_us,
                "e_a_mj": m.e_a_mj, "e_f_mj": m.e_f_mj,
                "source": m.source,
            })
        return results

    def get_available_tp_values(self, phase: str) -> list[int]:
        """Return TP values present in the profile data for the given phase."""
        df = self._df_p if phase == "prefill" else self._df_d
        return sorted(df["tp"].unique().tolist())

    def get_bs_range(self, phase: str, tp: int) -> tuple[int, int]:
        """Return (min_bs, max_bs) available in raw data for (phase, tp)."""
        df = self._df_p if phase == "prefill" else self._df_d
        sub = df[df["tp"] == tp]
        return int(sub["batch_size"].min()), int(sub["batch_size"].max())

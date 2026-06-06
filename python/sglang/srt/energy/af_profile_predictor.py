"""A-1: AF Profile Predictor — energy/latency prediction for AF-disaggregated inference.

Loads pre-trained models (from energy_model.py) and provides a simple scalar API:
    predictor.predict_latency("prefill", "A", tp=1, freq=930, bs=4, il=1024)
    predictor.predict_energy("decode", "F", tp=2, freq=690, bs=16, il=512, ol=128)

Strategy: LUT exact match first, fallback to GBDT (energy + decode latency) / LinearReg (prefill latency).
"""

import logging
import pickle
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd


class _ModelUnpickler(pickle.Unpickler):
    """Resolve classes pickled under __main__ or energy_model modules."""

    _class_cache: dict = {}

    def find_class(self, module: str, name: str):
        # V2 models (from energy_model_v2.py) — check first to avoid V1 conflict
        if module in ("__main__", "energy_model_v2") and name in ("GBDTModel", "LookupModel"):
            v2_key = f"v2_{name}"
            if v2_key not in self._class_cache:
                import importlib.util
                em_path = (
                    Path(__file__).resolve().parents[4]
                    / "benchmark" / "test_motivation" / "energy_model_v2.py"
                )
                spec = importlib.util.spec_from_file_location("_energy_model_v2", em_path)
                mod = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(mod)
                self._class_cache["v2_GBDTModel"] = getattr(mod, "GBDTModel")
                self._class_cache["v2_LookupModel"] = getattr(mod, "LookupModel")
            return self._class_cache[v2_key]
        # V1 models (from energy_model.py)
        if module == "__main__" and name in (
            "LookupTableModel", "LinearRegressionModel",
        ):
            if name not in self._class_cache:
                import importlib.util
                em_path = (
                    Path(__file__).resolve().parents[4]
                    / "benchmark" / "test_motivation" / "energy_model.py"
                )
                spec = importlib.util.spec_from_file_location("_energy_model", em_path)
                mod = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(mod)
                for cn in ("LookupTableModel", "LinearRegressionModel", "GBDTModel"):
                    self._class_cache[cn] = getattr(mod, cn)
            return self._class_cache[name]
        return super().find_class(module, name)

logger = logging.getLogger(__name__)

VALID_FREQS = [210, 450, 690, 930, 1170, 1410]

_LABEL_MAP = {
    ("prefill", "A", "energy"): "Prefill_A",
    ("prefill", "F", "energy"): "Prefill_F",
    ("decode",  "A", "energy"): "Decode_A",
    ("decode",  "F", "energy"): "Decode_F",
    ("prefill", "A", "latency"): "Prefill_A_lat",
    ("prefill", "F", "latency"): "Prefill_F_lat",
    ("decode",  "A", "latency"): "Decode_A_lat",
    ("decode",  "F", "latency"): "Decode_F_lat",
}

_BEST_MODEL_TYPE = {
    "Prefill_A":     "GBDT",
    "Prefill_F":     "GBDT",
    "Decode_A":      "GBDT",
    "Decode_F":      "GBDT",
    "Prefill_A_lat": "LinearReg",
    "Prefill_F_lat": "LinearReg",
    "Decode_A_lat":  "GBDT",
    "Decode_F_lat":  "GBDT",
}

# V2 coupled decode pipeline models
_V2_LABELS = [
    "Decode_iter_lat",
    "Decode_iter_energy_A",
    "Decode_iter_energy_F",
]


@dataclass
class PredictionResult:
    value: float
    model_type: str  # "LUT", "GBDT", or "LinearReg"
    exact_match: bool


class AFProfilePredictor:
    """Runtime predictor for AF-disaggregated energy and latency.

    Args:
        model_dir: Directory containing *.pkl model files from energy_model.py.
    """

    def __init__(self, model_dir: str):
        self.model_dir = Path(model_dir)
        self._models: dict[str, dict[str, object]] = {}
        self._load_models()

    def _load_models(self):
        """Load all available pkl models (V1 + V2)."""
        loaded = 0
        # V1 models (independent A/F per-layer)
        for label in _LABEL_MAP.values():
            self._models[label] = {}
            for mtype in ("LUT", "LinearReg", "GBDT"):
                pkl_path = self.model_dir / f"{label}_{mtype}.pkl"
                if pkl_path.exists():
                    with open(pkl_path, "rb") as f:
                        self._models[label][mtype] = _ModelUnpickler(f).load()
                    loaded += 1

        # V2 models (coupled decode pipeline)
        self._v2_models: dict[str, object] = {}
        for label in _V2_LABELS:
            for mtype in ("GBDT", "LUT"):
                pkl_path = self.model_dir / f"{label}_{mtype}.pkl"
                if pkl_path.exists():
                    with open(pkl_path, "rb") as f:
                        self._v2_models[f"{label}_{mtype}"] = _ModelUnpickler(f).load()
                    loaded += 1
        self._v2_available = any(
            f"Decode_iter_lat_{t}" in self._v2_models for t in ("GBDT", "LUT")
        )
        logger.info(
            "AFProfilePredictor: loaded %d models from %s (v2_coupled=%s)",
            loaded, self.model_dir, self._v2_available)

    def _build_df(self, phase: str, tp: int, freq: int,
                  bs: int, il: int, ol: Optional[int]) -> pd.DataFrame:
        """Build a single-row DataFrame matching the model's expected input."""
        if phase == "prefill":
            return pd.DataFrame([{
                "tp": tp, "gpu_clock": freq,
                "input_len": il, "batch_size": bs,
            }])
        else:
            assert ol is not None, "output_len required for decode prediction"
            return pd.DataFrame([{
                "tp": tp, "gpu_clock": freq,
                "input_len": il, "output_len": ol, "batch_size": bs,
            }])

    def _predict_single(self, label: str, phase: str, tp: int,
                        freq: int, bs: int, il: int,
                        ol: Optional[int]) -> PredictionResult:
        """Core prediction: try LUT exact match, then best model fallback."""
        models = self._models.get(label, {})
        if not models:
            raise ValueError(f"No models loaded for {label}")

        df = self._build_df(phase, tp, freq, bs, il, ol)

        lut = models.get("LUT")
        if lut is not None:
            pred = lut.predict(df)
            if not np.isnan(pred[0]):
                dist = self._lut_min_distance(lut, df)
                if dist < 1e-6:
                    return PredictionResult(float(pred[0]), "LUT", exact_match=True)

        best_type = _BEST_MODEL_TYPE[label]
        model = models.get(best_type)
        if model is not None:
            pred = model.predict(df)
            val = pred[0] if not np.isnan(pred[0]) else None
            if val is not None:
                return PredictionResult(float(val), best_type, exact_match=False)

        for fallback_type in ("LUT", "LinearReg", "GBDT"):
            if fallback_type == best_type:
                continue
            model = models.get(fallback_type)
            if model is not None:
                pred = model.predict(df)
                if not np.isnan(pred[0]):
                    return PredictionResult(float(pred[0]), fallback_type, exact_match=False)

        raise RuntimeError(f"All models failed for {label} with tp={tp} freq={freq} bs={bs} il={il} ol={ol}")

    @staticmethod
    def _lut_min_distance(lut, df: pd.DataFrame) -> float:
        """Check if the query point is an exact match in the LUT."""
        tp_val = int(df["tp"].iloc[0])
        if tp_val not in lut.models:
            return float("inf")
        tree, _values, mins, scale = lut.models[tp_val]
        point = df[lut.feature_cols].values.astype(float)
        normed = (point - mins) / scale
        dist, _ = tree.query(normed, k=1)
        return float(dist[0]) if np.ndim(dist) > 0 else float(dist)

    def predict_latency(self, phase: str, op: str, tp: int, freq: int,
                        bs: int, il: int, ol: Optional[int] = None) -> PredictionResult:
        """Predict per-layer latency (microseconds) for one operator.

        Args:
            phase: "prefill" or "decode"
            op: "A" (attention) or "F" (FFN)
            tp: tensor parallelism degree
            freq: GPU clock frequency in MHz
            bs: batch size
            il: input sequence length
            ol: output sequence length (required for decode)
        """
        phase = phase.lower()
        op = op.upper()
        label = _LABEL_MAP[(phase, op, "latency")]
        return self._predict_single(label, phase, tp, freq, bs, il, ol)

    def predict_energy(self, phase: str, op: str, tp: int, freq: int,
                       bs: int, il: int, ol: Optional[int] = None) -> PredictionResult:
        """Predict per-layer energy (millijoules) for one operator.

        Args: same as predict_latency.
        """
        phase = phase.lower()
        op = op.upper()
        label = _LABEL_MAP[(phase, op, "energy")]
        return self._predict_single(label, phase, tp, freq, bs, il, ol)

    def find_best_freq_pair(
        self, phase: str, tp: int, bs: int, il: int,
        slo_budget_us: float, ol: Optional[int] = None,
        freqs: Optional[list[int]] = None, M: int = 1,
        t_comm_us: float = 0.0,
        num_layers: int = 1,
    ) -> Optional[tuple[int, int, float]]:
        """Search all (f_A, f_F) combos for minimum energy under SLO.

        Args:
            slo_budget_us: Total SLO budget (e.g. TTFT or TPOT) in microseconds.
            num_layers: Number of transformer layers. The per-layer latency is
                multiplied by this to compare against slo_budget_us.

        Returns (f_A, f_F, total_energy_mj) or None if no combo meets SLO.
        """
        if freqs is None:
            freqs = VALID_FREQS

        best = None
        for f_a in freqs:
            for f_f in freqs:
                try:
                    lat_a = self.predict_latency(phase, "A", tp, f_a, bs, il, ol).value
                    lat_f = self.predict_latency(phase, "F", tp, f_f, bs, il, ol).value
                except (RuntimeError, ValueError):
                    continue

                if M > 1:
                    t_layer = max(lat_a, lat_f) + t_comm_us / M
                else:
                    t_layer = lat_a + lat_f + t_comm_us

                if t_layer * num_layers > slo_budget_us:
                    continue

                try:
                    e_a = self.predict_energy(phase, "A", tp, f_a, bs, il, ol).value
                    e_f = self.predict_energy(phase, "F", tp, f_f, bs, il, ol).value
                except (RuntimeError, ValueError):
                    continue

                total_e = (e_a + e_f) * num_layers
                if best is None or total_e < best[2]:
                    best = (f_a, f_f, total_e)

        return best

    # ─── V2 Coupled Decode Pipeline Interface ─────────────────────────────

    @property
    def has_coupled_model(self) -> bool:
        """Whether V2 coupled decode pipeline models are available."""
        return self._v2_available

    def _v2_predict(self, label: str, features: np.ndarray) -> Optional[float]:
        """Predict using V2 coupled model (GBDT preferred, LUT fallback)."""
        for mtype in ("GBDT", "LUT"):
            key = f"{label}_{mtype}"
            model = self._v2_models.get(key)
            if model is not None:
                pred = model.predict(features)
                val = pred[0] if hasattr(pred, '__len__') else float(pred)
                if not np.isnan(val) and val > 0:
                    return float(val)
        return None

    def predict_iteration_latency(
        self, M: int, f_a: int, f_f: int,
        bs: int, il: int
    ) -> Optional[float]:
        """Predict end-to-end decode iteration latency (us) using coupled model.

        This accounts for IPC communication, pipeline drain, and micro-batch
        overlap — effects that the independent per-layer model cannot capture.

        Returns iteration latency in microseconds, or None if model unavailable.
        """
        if not self._v2_available:
            return None
        features = np.array([[M, f_a, f_f, il, bs]], dtype=float)
        return self._v2_predict("Decode_iter_lat", features)

    def predict_iteration_energy(
        self, M: int, f_a: int, f_f: int,
        bs: int, il: int
    ) -> Optional[tuple[float, float]]:
        """Predict per-iteration energy (DA_mJ, DF_mJ) using coupled model.

        Returns (da_energy_mj, df_energy_mj) or None if model unavailable.
        """
        if not self._v2_available:
            return None
        features = np.array([[M, f_a, f_f, il, bs]], dtype=float)
        da_e = self._v2_predict("Decode_iter_energy_A", features)
        df_e = self._v2_predict("Decode_iter_energy_F", features)
        if da_e is None or df_e is None:
            return None
        return (da_e, df_e)

    def find_best_freq_pair_coupled(
        self, M: int, bs: int, il: int,
        slo_budget_us: float,
        freqs: Optional[list[int]] = None,
    ) -> Optional[tuple[int, int, float, float]]:
        """Find minimum-energy (f_A, f_F) pair under SLO using coupled model.

        Unlike find_best_freq_pair which uses per-layer independent models,
        this uses the V2 iteration-level coupled model that accounts for
        pipeline overhead and A/F interaction.

        Returns (f_A, f_F, iter_lat_us, total_energy_mj) or None.
        """
        if not self._v2_available:
            return None
        if freqs is None:
            freqs = VALID_FREQS

        best = None
        for f_a in freqs:
            for f_f in freqs:
                lat = self.predict_iteration_latency(M, f_a, f_f, bs, il)
                if lat is None or lat > slo_budget_us:
                    continue
                energy = self.predict_iteration_energy(M, f_a, f_f, bs, il)
                if energy is None:
                    continue
                total_e = energy[0] + energy[1]
                if best is None or total_e < best[3]:
                    best = (f_a, f_f, lat, total_e)

        return best

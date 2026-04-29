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
    """Resolve classes pickled under __main__ to energy_model module."""

    _class_cache: dict = {}

    def find_class(self, module: str, name: str):
        if module == "__main__" and name in (
            "LookupTableModel", "LinearRegressionModel", "GBDTModel",
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
        """Load all available pkl models."""
        loaded = 0
        for label in _LABEL_MAP.values():
            self._models[label] = {}
            for mtype in ("LUT", "LinearReg", "GBDT"):
                pkl_path = self.model_dir / f"{label}_{mtype}.pkl"
                if pkl_path.exists():
                    with open(pkl_path, "rb") as f:
                        self._models[label][mtype] = _ModelUnpickler(f).load()
                    loaded += 1
        logger.info("AFProfilePredictor: loaded %d models from %s", loaded, self.model_dir)

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

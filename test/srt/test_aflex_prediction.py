"""CPU-only model portability and profiling schema tests."""

import io
import pickle
import subprocess
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]


def bootstrap():
    # Avoid optional serving dependencies in top-level sglang.__init__.
    for name, path in (
        ("sglang", ROOT / "python/sglang"),
        ("sglang.srt", ROOT / "python/sglang/srt"),
        ("sglang.srt.energy", ROOT / "python/sglang/srt/energy"),
    ):
        if name not in sys.modules:
            package = types.ModuleType(name)
            package.__path__ = [str(path)]
            sys.modules[name] = package


bootstrap()
from sglang.srt.energy.af_profile_predictor import AFProfilePredictor, _ModelUnpickler
from sglang.srt.energy.prediction import energy_model as v1
from sglang.srt.energy.prediction import energy_model_v2 as v2
from sglang.srt.energy.prediction import energy_model_v3 as v3
from sglang.srt.energy.profiling.normalize import load_profile, normalize_profile

# Serving test utilities initialize CUDA on some installations; keep this suite CPU-only.
CustomTestCase = unittest.TestCase


class TestAFlexPrediction(CustomTestCase):
    def frame(self):
        return pd.DataFrame(
            {
                "tp": [1] * 12,
                "gpu_clock": np.arange(12) * 50 + 210,
                "input_len": [128] * 12,
                "batch_size": [1] * 12,
                "A_energy_mj": np.arange(12) + 2.0,
            }
        )

    def test_small_fit_pickle(self):
        df = self.frame()
        for cls in (v1.LookupTableModel, v1.LinearRegressionModel):
            model = cls("test")
            model.fit(df, ["gpu_clock", "input_len", "batch_size"], "A_energy_mj")
            loaded = pickle.loads(pickle.dumps(model))
            self.assertIn("sglang.srt.energy.prediction", type(loaded).__module__)
            np.testing.assert_allclose(loaded.predict(df), model.predict(df))
            self.assertTrue(np.isfinite(loaded.predict(df.iloc[[4, 8]])).all())
        for mod in (v2, v3):
            model = mod.LookupModel("test", ["x"], k=1)
            x = np.arange(6.0).reshape(-1, 1)
            model.fit(x, np.arange(6.0) + 1)
            np.testing.assert_allclose(
                pickle.loads(pickle.dumps(model)).predict(x), np.arange(6.0) + 1
            )

    def legacy(self, model, name, module="__main__"):
        legacy = type(name, (), {})
        legacy.__module__ = module
        obj = legacy()
        obj.__dict__.update(model.__dict__)
        mod = sys.modules.get(module) or types.ModuleType(module)
        with patch.dict(sys.modules, {module: mod}), patch.object(
            mod, name, legacy, create=True
        ):
            data = pickle.dumps(obj)
        return _ModelUnpickler(io.BytesIO(data)).load()

    def test_legacy_gbdt_state_dispatch(self):
        for mod, features in (
            (v1, ["gpu_clock"]),
            (v2, ["M", "f_A", "f_F", "input_len", "batch_size"]),
            (v3, ["tp_a", "tp_f", "M", "f_A", "f_F", "input_len", "batch_size"]),
        ):
            model = (
                mod.GBDTModel("test") if mod is v1 else mod.GBDTModel("test", features)
            )
            model.feature_cols = features
            if mod is v1:
                model.all_features = ["tp"] + features
            self.assertIs(type(self.legacy(model, "GBDTModel")), mod.GBDTModel)
            old_module = (
                "energy_model"
                if mod is v1
                else "energy_model_" + ("v2" if mod is v2 else "v3")
            )
            self.assertIs(
                type(self.legacy(model, "GBDTModel", old_module)), mod.GBDTModel
            )

    def test_gbdt_small_cpu_fit(self):
        try:
            import lightgbm as lgb
        except ImportError:
            self.skipTest("lightgbm is optional")
        train = lgb.train

        def cpu_train(params, data, num_boost_round):
            return train(
                dict(params, num_threads=1),
                data,
                num_boost_round=min(3, num_boost_round),
            )

        df = self.frame()
        with patch.object(lgb, "train", side_effect=cpu_train):
            model = v1.GBDTModel("cpu")
            model.fit(df, ["gpu_clock"], "A_energy_mj")
            self.assertTrue(
                np.isfinite(pickle.loads(pickle.dumps(model)).predict(df)).all()
            )
            for mod in (v2, v3):
                model = mod.GBDTModel("cpu", ["x"])
                x = np.arange(12.0).reshape(-1, 1)
                model.fit(x, np.arange(12.0) + 1, n_rounds=3)
                self.assertTrue(
                    np.isfinite(pickle.loads(pickle.dumps(model)).predict(x)).all()
                )

    def test_v1_training_cli_and_stable_pickles(self):
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            df = self.frame()
            df["F_energy_mj"] = df.A_energy_mj * 2
            df["A"] = df.A_energy_mj * 10
            df["F"] = df.A_energy_mj * 20
            prefill = directory / "prefill.tsv"
            decode = directory / "decode.tsv"
            df.to_csv(prefill, sep="\t", index=False)
            df["output_len"] = 64
            df.to_csv(decode, sep="\t", index=False)
            output = directory / "models"
            argv = [
                "train",
                "--prefill",
                str(prefill),
                "--decode",
                str(decode),
                "--output-dir",
                str(output),
                "--folds",
                "2",
            ]
            # Exercise complete CLI and its LUT/linear fallback without optional GBDT.
            with patch.dict(sys.modules, {"lightgbm": None}), patch.object(
                sys, "argv", argv
            ), patch("sys.stdout", new=io.StringIO()):
                import runpy

                runpy.run_module(
                    "sglang.srt.energy.prediction.train", run_name="__main__"
                )
            self.assertEqual(len(list(output.glob("*.pkl"))), 16)
            self.assertTrue((output / "cv_mape_report.tsv").exists())
            with open(output / "Prefill_A_LUT.pkl", "rb") as stream:
                self.assertIs(type(pickle.load(stream)), v1.LookupTableModel)
            prediction = AFProfilePredictor(str(output)).predict_latency(
                "prefill", "A", 1, 210, 1, 128
            )
            self.assertTrue(prediction.exact_match)
            self.assertAlmostEqual(prediction.value, 20.0)

    def test_normalization_no_scaling(self):
        for phase, prefix in (("prefill", "P"), ("decode", "D")):
            raw = {
                "tp": [4],
                "input_len": [128],
                "gpu_clock": [930],
                "batch_size": [2],
                f"{prefix}_A_lat": [1234.5],
                f"{prefix}_F_lat": [25.0],
                f"{prefix}_A_energy": [7.25],
                f"{prefix}_F_energy": [1.5],
            }
            if phase == "decode":
                raw["output_len"] = [64]
            normalized = normalize_profile(pd.DataFrame(raw), phase)
            self.assertEqual(normalized.A.iloc[0], 1234.5)
            self.assertEqual(normalized.A_energy_mj.iloc[0], 7.25)
            with tempfile.TemporaryDirectory() as directory:
                path = Path(directory) / "raw.tsv"
                path.write_text(
                    "# preamble\n" + pd.DataFrame(raw).to_csv(sep="\t", index=False)
                )
                pd.testing.assert_frame_equal(load_profile(path, phase), normalized)

    def test_v2_runtime_feature_shape(self):
        with tempfile.TemporaryDirectory() as directory:
            features = ["M", "f_A", "f_F", "input_len", "batch_size"]
            model = v2.LookupModel("test", features, k=1)
            model.fit(np.array([[2, 930, 690, 128, 4.0]]), np.array([23.0]))
            for label in (
                "Decode_iter_lat",
                "Decode_iter_energy_A",
                "Decode_iter_energy_F",
            ):
                with open(Path(directory) / f"{label}_LUT.pkl", "wb") as stream:
                    pickle.dump(model, stream)
            predictor = AFProfilePredictor(directory)
            self.assertEqual(
                predictor.predict_iteration_latency(2, 930, 690, 4, 128), 23
            )
            self.assertEqual(
                predictor.predict_iteration_energy(2, 930, 690, 4, 128), (23, 23)
            )

    def test_cli_help_without_runtime(self):
        code = (
            "import sys,types,runpy; from pathlib import Path; root=Path(sys.argv[1]); "
        )
        code += "[(sys.modules.setdefault(n,types.ModuleType(n)), setattr(sys.modules[n],'__path__',[str(root/p)])) for n,p in [('sglang','python/sglang'),('sglang.srt','python/sglang/srt'),('sglang.srt.energy','python/sglang/srt/energy')]]; "
        code += "module=sys.argv[2]; sys.argv=[module,'--help']; runpy.run_module(module,run_name='__main__')"
        for suffix in (
            "prediction.train",
            "prediction.energy_model_v2",
            "prediction.energy_model_v3",
            "profiling.bench_prefill_af",
            "profiling.bench_decode_af",
            "profiling.normalize",
        ):
            result = subprocess.run(
                [sys.executable, "-c", code, str(ROOT), "sglang.srt.energy." + suffix],
                capture_output=True,
                text=True,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn("usage:", result.stdout)


if __name__ == "__main__":
    unittest.main()

"""CPU-only launcher asset checks; no GPU, server, or remote calls are made."""

import importlib.util
import sys
import tempfile
import types
import unittest
from contextlib import ExitStack
from pathlib import Path
from unittest.mock import Mock, patch

try:
    from sglang.test.test_utils import CustomTestCase
except ImportError:
    CustomTestCase = unittest.TestCase


ROOT = Path(__file__).resolve().parents[2]
LAUNCHER = ROOT / "python/sglang/srt/energy/af_launcher.py"


def load_launcher():
    # Direct loading avoids importing optional top-level serving dependencies.
    spec = importlib.util.spec_from_file_location("_aflex_launcher_assets", LAUNCHER)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class TestAFlexLauncherAssets(CustomTestCase):
    def setUp(self):
        self.launcher = load_launcher()
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.models = self.root / "external-models"
        self.models.mkdir()
        self.prefill = self.root / "external-prefill.txt"
        self.decode = self.root / "external-decode.txt"
        self.prefill.touch()
        self.decode.touch()

    def config(self):
        return {
            "model": {"path": "example/external-model"},
            "afd": {"dvfs_enabled": False, "energy_model_dir": str(self.models)},
            "tier1": {
                "enable_tier1_pa": False,
                "prefill_data_path": str(self.prefill),
                "decode_data_path": str(self.decode),
            },
            "logs": {"dir": str(self.root / "logs")},
        }

    def assert_rejected_before_side_effects(self, cfg, key, start=False):
        with ExitStack() as stack:
            guards = []
            for name in (
                "_set_gpu_locked_clock",
                "_run_preflight_solve",
                "_check_port_ready",
                "_build_server_cmd",
                "_build_router_cmd",
            ):
                guards.append(
                    stack.enter_context(
                        patch.object(
                            self.launcher, name, side_effect=AssertionError(name)
                        )
                    )
                )
            guards.append(
                stack.enter_context(
                    patch.object(
                        self.launcher.subprocess,
                        "Popen",
                        side_effect=AssertionError("unexpected process launch"),
                    )
                )
            )
            with self.assertRaisesRegex(ValueError, key):
                self.launcher._validate_external_assets(cfg, start)
            with self.assertLogs("af_launcher", level="ERROR") as logs:
                self.assertEqual(self.launcher.launch_all(cfg, start), 1)
            self.assertIn(key, " ".join(logs.output))
            for guard in guards:
                guard.assert_not_called()
        self.assertFalse((self.root / "logs").exists())

    def test_missing_model_path_fails_before_side_effects(self):
        for value in (None, "", "absent-key", "absent-section"):
            with self.subTest(value=value):
                cfg = self.config()
                if value == "absent-section":
                    del cfg["model"]
                elif value == "absent-key":
                    del cfg["model"]["path"]
                else:
                    cfg["model"]["path"] = value
                self.assert_rejected_before_side_effects(cfg, "model.path")

    def test_missing_energy_models_for_each_consumer(self):
        for consumer in ("dvfs", "tier1", "preflight"):
            for value in (None, "", str(self.root / "missing"), str(self.prefill)):
                with self.subTest(consumer=consumer, value=value):
                    cfg = self.config()
                    cfg["afd"]["dvfs_enabled"] = consumer == "dvfs"
                    cfg["tier1"]["enable_tier1_pa"] = consumer == "tier1"
                    cfg["afd"]["energy_model_dir"] = value
                    self.assert_rejected_before_side_effects(
                        cfg, "afd.energy_model_dir", consumer == "preflight"
                    )

    def test_missing_profile_files_for_each_tier1_mode(self):
        for start in (False, True):
            for key in ("prefill_data_path", "decode_data_path"):
                for value in (None, "", str(self.root / "missing"), str(self.models)):
                    with self.subTest(start=start, key=key, value=value):
                        cfg = self.config()
                        cfg["tier1"]["enable_tier1_pa"] = not start
                        cfg["tier1"][key] = value
                        self.assert_rejected_before_side_effects(
                            cfg, "tier1." + key, start
                        )

    def test_absent_asset_keys_do_not_use_local_fallbacks(self):
        local = self.root / "bundled"
        local.mkdir()
        (local / "energy_models").mkdir()
        (local / "prefill_data_v1.txt").touch()
        (local / "decode_data_v1.txt").touch()
        with patch.object(self.launcher, "_HERE", local):
            for section, key in (
                ("afd", "energy_model_dir"),
                ("tier1", "prefill_data_path"),
                ("tier1", "decode_data_path"),
            ):
                with self.subTest(section=section, key=key):
                    cfg = self.config()
                    cfg["tier1"]["enable_tier1_pa"] = True
                    del cfg[section][key]
                    self.assert_rejected_before_side_effects(cfg, section + "." + key)

    def test_valid_external_assets_are_accepted(self):
        for dvfs, tier1, start in (
            (True, False, False),
            (False, True, False),
            (False, False, True),
            (True, True, True),
        ):
            with self.subTest(dvfs=dvfs, tier1=tier1, start=start):
                cfg = self.config()
                cfg["afd"]["dvfs_enabled"] = dvfs
                cfg["tier1"]["enable_tier1_pa"] = tier1
                self.assertIsNone(self.launcher._validate_external_assets(cfg, start))

    def test_minimal_dvfs_off_needs_no_energy_assets(self):
        for cfg in (
            {"model": {"path": "example/external-model"}},
            {
                "model": {"path": str(self.root / "model")},
                "afd": {"dvfs_enabled": False},
                "tier1": {},
            },
        ):
            with self.subTest(cfg=cfg), patch.object(
                self.launcher.Path, "is_dir", side_effect=AssertionError("asset lookup")
            ), patch.object(
                self.launcher.Path,
                "is_file",
                side_effect=AssertionError("asset lookup"),
            ):
                self.assertIsNone(self.launcher._validate_external_assets(cfg, False))

    def test_dvfs_alone_does_not_require_profiles(self):
        cfg = self.config()
        cfg["afd"]["dvfs_enabled"] = True
        del cfg["tier1"]
        self.assertIsNone(self.launcher._validate_external_assets(cfg, False))

    def test_preflight_uses_configured_external_paths_without_fallback(self):
        cfg = self.config()
        profile_module = types.ModuleType("sglang.srt.energy.profile_table")
        profile_module.ProfileTable = Mock()
        solver_module = types.ModuleType("sglang.srt.energy.tier1_solver")
        solver_module.Tier1Solver = Mock()
        solver_module.SLOConfig = Mock()
        solver_module.WorkloadProfile = Mock()
        solution = solver_module.Tier1Solver.return_value.solve.return_value
        solution.feasible = True
        solution.gpu_used = 4
        solution._gpu_avail = 0
        solution.to_dict.return_value = {"feasible": True}
        # Exact leaf-module stubs avoid importing sglang or solver dependencies.
        with patch.dict(
            sys.modules,
            {
                profile_module.__name__: profile_module,
                solver_module.__name__: solver_module,
            },
        ):
            result = self.launcher._run_preflight_solve(cfg, self.root)
        self.assertEqual(result, str(self.root / "tier1_initial_solution.json"))
        profile_module.ProfileTable.assert_called_once_with(
            prefill_path=str(self.prefill),
            decode_path=str(self.decode),
            energy_model_dir=str(self.models),
        )
        solver_module.Tier1Solver.return_value.warm_start.assert_not_called()


if __name__ == "__main__":
    unittest.main()

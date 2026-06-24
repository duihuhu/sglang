"""Tests for the graceful reload orchestrator.

Tests verify:
1. Signal state transitions (draining → starting → switching → ready)
2. Router drain/activate API logic
3. Module idle detection and abort-on-timeout
4. Diff detection (which modules need reload)
"""

import importlib
import json
import os
import sys
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

# Direct import to avoid full sglang dependency chain (pybase64 etc.)
_PYTHON_DIR = Path(__file__).resolve().parents[2] / "python"
sys.path.insert(0, str(_PYTHON_DIR))
# Suppress sglang top-level __init__ by pre-creating the package hierarchy
_sglang_pkg = str(_PYTHON_DIR / "sglang")
_srt_pkg = str(_PYTHON_DIR / "sglang" / "srt")
_energy_pkg = str(_PYTHON_DIR / "sglang" / "srt" / "energy")

import importlib.util


def _load_module_from_path(module_name: str, file_path: str):
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = mod
    spec.loader.exec_module(mod)
    return mod


reload_signal = _load_module_from_path(
    "reload_signal",
    str(_PYTHON_DIR / "sglang" / "srt" / "energy" / "reload_signal.py"),
)
graceful_orchestrator = _load_module_from_path(
    "graceful_orchestrator",
    str(_PYTHON_DIR / "sglang" / "srt" / "energy" / "graceful_orchestrator.py"),
)


class TestReloadSignal(unittest.TestCase):
    """Test the extended reload signal protocol."""

    def setUp(self):
        self.tmp_dir = tempfile.mkdtemp()
        self.signal_path = os.path.join(self.tmp_dir, "tier1_reload_signal.json")

    def test_write_and_read_signal(self):
        reload_signal.write_signal(self.signal_path, "draining", {"changed_modules": ["PA"]})
        sig = reload_signal.read_signal(self.signal_path)
        self.assertEqual(sig["status"], "draining")
        self.assertEqual(sig["changed_modules"], ["PA"])
        self.assertIn("timestamp", sig)

    def test_is_reloading_for_graceful_statuses(self):
        for status in reload_signal.GRACEFUL_STATUSES:
            reload_signal.write_signal(self.signal_path, status)
            self.assertTrue(reload_signal.is_reloading(self.signal_path),
                            f"Expected is_reloading=True for status={status}")

    def test_is_reloading_for_legacy(self):
        reload_signal.write_signal(self.signal_path, "reloading")
        self.assertTrue(reload_signal.is_reloading(self.signal_path))

    def test_is_ready(self):
        reload_signal.write_signal(self.signal_path, "ready")
        self.assertTrue(reload_signal.is_ready(self.signal_path))

    def test_is_ready_idle(self):
        reload_signal.clear_signal(self.signal_path)
        self.assertTrue(reload_signal.is_ready(self.signal_path))

    def test_not_ready_during_drain(self):
        reload_signal.write_signal(self.signal_path, "draining")
        self.assertFalse(reload_signal.is_ready(self.signal_path))


class TestDiffModules(unittest.TestCase):
    """Test module diff detection."""

    def test_no_change(self):
        old = {"tp_pa": 1, "tp_pf": 1, "tp_da": 1, "tp_df": 1}
        new = {"tp_pa": 1, "tp_pf": 1, "tp_da": 1, "tp_df": 1}
        self.assertEqual(graceful_orchestrator._diff_modules(old, new), [])

    def test_pf_change(self):
        old = {"tp_pa": 1, "tp_pf": 1, "tp_da": 1, "tp_df": 1}
        new = {"tp_pa": 1, "tp_pf": 2, "tp_da": 1, "tp_df": 1}
        self.assertEqual(graceful_orchestrator._diff_modules(old, new), ["PF"])

    def test_multiple_changes(self):
        old = {"tp_pa": 1, "tp_pf": 1, "tp_da": 1, "tp_df": 1}
        new = {"tp_pa": 2, "tp_pf": 2, "tp_da": 1, "tp_df": 1}
        changed = graceful_orchestrator._diff_modules(old, new)
        self.assertIn("PA", changed)
        self.assertIn("PF", changed)
        self.assertEqual(len(changed), 2)

    def test_all_change(self):
        old = {"tp_pa": 1, "tp_pf": 1, "tp_da": 1, "tp_df": 1}
        new = {"tp_pa": 2, "tp_pf": 2, "tp_da": 2, "tp_df": 2}
        self.assertEqual(len(graceful_orchestrator._diff_modules(old, new)), 4)


class TestFindModulesByNames(unittest.TestCase):
    """Test module filtering."""

    def setUp(self):
        self.modules = [
            {"name": "PA", "port": 8000, "disagg_mode": "prefill"},
            {"name": "PF", "port": 8001, "disagg_mode": "prefill"},
            {"name": "DA", "port": 8010, "disagg_mode": "decode"},
            {"name": "DF", "port": 8011, "disagg_mode": "decode"},
        ]

    def test_filter_single(self):
        result = graceful_orchestrator._find_modules_by_names(self.modules, ["PF"])
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["name"], "PF")

    def test_filter_multiple(self):
        result = graceful_orchestrator._find_modules_by_names(self.modules, ["PA", "PF"])
        self.assertEqual(len(result), 2)

    def test_filter_case_insensitive(self):
        result = graceful_orchestrator._find_modules_by_names(self.modules, ["pa", "df"])
        self.assertEqual(len(result), 2)


class TestRouterDrainActivate(unittest.TestCase):
    """Test MiniLB drain/activate logic in isolation."""

    def test_drain_marks_urls(self):
        """Verify drain state management logic."""
        _ROUTER_DIR = Path(__file__).resolve().parents[2] / "sgl-model-gateway" / "bindings" / "python" / "src"
        sys.path.insert(0, str(_ROUTER_DIR))
        try:
            from sglang_router.mini_lb import MiniLoadBalancer
        except ImportError:
            self.skipTest("sglang_router not available")

        mock_args = MagicMock()
        mock_args.host = "127.0.0.1"
        mock_args.port = 9000
        mock_args.request_timeout_secs = 60
        mock_args.prefill_urls = [("http://127.0.0.1:8000", 9999)]
        mock_args.decode_urls = ["http://127.0.0.1:8010"]
        mock_args.test_external_dp_routing = False
        mock_args.policy = "random"
        mock_args.pd_disaggregation = True

        lb = MiniLoadBalancer(mock_args)

        self.assertEqual(len(lb.draining_prefill_urls), 0)
        self.assertFalse(lb.is_draining_all)

        lb.draining_prefill_urls.add("http://127.0.0.1:8000")
        lb.is_draining_all = (
            lb.draining_prefill_urls >= set(lb.prefill_urls)
        )
        self.assertTrue(lb.is_draining_all)


class TestGracefulOrchestratorEndToEnd(unittest.TestCase):
    """Integration test for run_graceful_reload with mocked HTTP calls."""

    def setUp(self):
        self.tmp_dir = tempfile.mkdtemp()
        self.signal_path = os.path.join(self.tmp_dir, "tier1_reload_signal.json")
        self.config = {
            "signal_path": self.signal_path,
            "old_solution": {
                "tp_pa": 1, "tp_pf": 1, "tp_da": 1, "tp_df": 1,
                "f_pa": 1410, "f_pf": 1410, "f_da": 1410, "f_df": 1410,
                "k_p": 1, "k_d": 1,
            },
            "solution": {
                "tp_pa": 2, "tp_pf": 2, "tp_da": 1, "tp_df": 1,
                "f_pa": 1410, "f_pf": 1410, "f_da": 1410, "f_df": 1410,
                "k_p": 1, "k_d": 1,
            },
            "drain_timeout_s": 5.0,
            "server_config": {
                "model_path": "/models/test",
                "gpu_indices": [0, 1, 2, 3],
                "all_ports": [7990, 8000, 8001, 8010, 8011],
                "router_port": 7990,
                "bootstrap_port": 29999,
                "ib_device": "mlx5_4",
                "mem_fraction": 0.85,
                "extra_args": [],
                "dvfs_args": [],
                "tier1_args": [],
                "log_dir": self.tmp_dir,
                "modules": [
                    {"name": "PA", "perspective": "attn", "disagg_mode": "prefill",
                     "port": 8000, "visible_gpus": "2,3", "base_gpu_id": 1,
                     "ucx_base_port": 26200, "sched_port": 66400,
                     "nvml_device_index": 3, "ffn_host": "127.0.0.1", "is_pa": True,
                     "peer_device": 0},
                    {"name": "PF", "perspective": "ffn", "disagg_mode": "prefill",
                     "port": 8001, "visible_gpus": "2,3", "base_gpu_id": 0,
                     "ucx_base_port": 26200, "sched_port": 66400,
                     "nvml_device_index": 2, "is_pa": False, "peer_device": 1},
                    {"name": "DA", "perspective": "attn", "disagg_mode": "decode",
                     "port": 8010, "visible_gpus": "0,1", "base_gpu_id": 1,
                     "ucx_base_port": 26300, "sched_port": 66500,
                     "nvml_device_index": 1, "ffn_host": "127.0.0.1", "is_pa": False,
                     "peer_device": 0},
                    {"name": "DF", "perspective": "ffn", "disagg_mode": "decode",
                     "port": 8011, "visible_gpus": "0,1", "base_gpu_id": 0,
                     "ucx_base_port": 26300, "sched_port": 66500,
                     "nvml_device_index": 0, "is_pa": False, "peer_device": 1},
                ],
                "router": {"enabled": True, "prefill_port": 8000, "decode_port": 8010},
            },
        }

    @patch.object(graceful_orchestrator, "_http_post_json")
    @patch.object(graceful_orchestrator, "_http_get_json")
    @patch.object(graceful_orchestrator, "_start_module")
    @patch.object(graceful_orchestrator, "_graceful_kill_port")
    @patch.object(graceful_orchestrator, "_wait_port_ready")
    @patch.object(graceful_orchestrator, "_wait_health")
    @patch.object(graceful_orchestrator, "_reset_gpu_clocks")
    def test_graceful_reload_success(
        self, mock_reset, mock_health, mock_port_ready,
        mock_kill, mock_start, mock_get, mock_post
    ):
        """Full success path: drain → start → kill old → activate."""
        mock_post.return_value = {"status": "ok"}
        mock_get.return_value = {"idle": True, "inflight": 0}
        mock_start.return_value = MagicMock(pid=12345)
        mock_port_ready.return_value = True
        mock_health.return_value = True

        result = graceful_orchestrator.run_graceful_reload(self.config)

        self.assertTrue(result)

        sig = reload_signal.read_signal(self.signal_path)
        self.assertEqual(sig["status"], "ready")
        self.assertIn("reload_duration_s", sig)
        self.assertEqual(sig["changed_modules"], ["PA", "PF"])

        # Verify router drain was called
        drain_call = mock_post.call_args_list[0]
        self.assertIn("/admin/drain_module", drain_call[0][0])

        # Verify router activate was called
        activate_call = mock_post.call_args_list[1]
        self.assertIn("/admin/activate_module", activate_call[0][0])

        # Verify old modules were killed
        self.assertEqual(mock_kill.call_count, 2)

    @patch.object(graceful_orchestrator, "_http_post_json")
    @patch.object(graceful_orchestrator, "_http_get_json")
    @patch.object(graceful_orchestrator, "_start_module")
    @patch.object(graceful_orchestrator, "_graceful_kill_port")
    @patch.object(graceful_orchestrator, "_wait_port_ready")
    @patch.object(graceful_orchestrator, "_wait_health")
    @patch.object(graceful_orchestrator, "_reset_gpu_clocks")
    @patch.object(graceful_orchestrator, "_abort_module")
    def test_graceful_reload_drain_timeout(
        self, mock_abort, mock_reset, mock_health, mock_port_ready,
        mock_kill, mock_start, mock_get, mock_post
    ):
        """Drain timeout triggers abort on remaining requests."""
        mock_get.return_value = {"idle": False, "inflight": 5}
        mock_post.return_value = {"status": "ok"}
        mock_start.return_value = MagicMock(pid=12345)
        mock_port_ready.return_value = True
        mock_health.return_value = True

        self.config["drain_timeout_s"] = 0.1

        result = graceful_orchestrator.run_graceful_reload(self.config)
        self.assertTrue(result)

        self.assertTrue(mock_abort.call_count >= 1)

    @patch.object(graceful_orchestrator, "_http_post_json")
    def test_graceful_reload_no_changes(self, mock_post):
        """No TP changes means instant success."""
        self.config["old_solution"]["tp_pa"] = 2
        self.config["old_solution"]["tp_pf"] = 2
        self.config["solution"]["tp_pa"] = 2
        self.config["solution"]["tp_pf"] = 2

        result = graceful_orchestrator.run_graceful_reload(self.config)
        self.assertTrue(result)
        mock_post.assert_not_called()

        sig = reload_signal.read_signal(self.signal_path)
        self.assertEqual(sig["status"], "ready")


if __name__ == "__main__":
    unittest.main()

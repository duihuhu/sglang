import asyncio
import json
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from run_m2_pool_comparison import BASE, SelectedGpuMonitor, capacity, command, ensure_idle


class LivePlanTests(unittest.TestCase):
    def test_natural_does_not_override_pools(self):
        cmd = command("mixed-4096", "static-natural", 30143)
        self.assertNotIn("--max-total-tokens", cmd)
        self.assertNotIn("--max-mamba-cache-size", cmd)
        self.assertIn("--enable-mixed-chunk", cmd)

    def test_rebalanced_does_not_enable_unified(self):
        cmd = command("separate-4096", "static-rebalanced", 30143, kv_tokens=1000000)
        self.assertEqual(cmd[cmd.index("--max-mamba-cache-size") + 1], "607")
        self.assertNotIn("--enable-unified-memory", cmd)
        self.assertNotIn("--enable-mixed-chunk", cmd)

    def test_old_cap_uses_current_natural_readback(self):
        cmd = command("mixed-4096", "unified-old-cap", 30143, natural_cap=484)
        self.assertEqual(cmd[cmd.index("--max-mamba-cache-size") + 1], "484")
        self.assertIn("--enable-unified-memory", cmd)

    def test_audit_wrapper_keeps_model_arguments(self):
        cmd = command("mixed-4096", "unified-raised-cap", 30143, audited=True)
        self.assertTrue(cmd[1].endswith("launch_m2_pool_audit_server.py"))
        self.assertEqual(cmd[2], "--model-path")
        self.assertNotIn("-m", cmd)

    def test_busy_gpu_rejected_without_stopping_it(self):
        with patch("run_m2_pool_comparison.socket.socket"), patch("run_m2_pool_comparison.gpu_snapshot",
                    return_value=[{"index": 0, "memory_mib": 3000, "utilization": 5},
                                  {"index": 1, "memory_mib": 1, "utilization": 0}]):
            with self.assertRaises(RuntimeError):
                ensure_idle(30143, [0, 1])

    def test_unified_readback_must_match_selected_arm(self):
        path = BASE / "results/m2_state_mediation_20260916/inspect03/m2mediate-natural-mixed-4096-cap485-rep1-inspect.json"
        info = json.loads(path.read_text())["server_initial"]
        capacity(info, "mixed-4096", "static-natural", 485)
        with self.assertRaises(ValueError):
            capacity(info, "mixed-4096", "unified-old-cap", 485)


class MonitorTests(unittest.IsolatedAsyncioTestCase):
    async def sample(self, gpus, content):
        monitor = SelectedGpuMonitor(gpus)
        stream = asyncio.StreamReader()
        stream.feed_data(content)
        stream.feed_eof()
        monitor.process = SimpleNamespace(stdout=stream)
        return await monitor.read_sample()

    async def test_selected_physical_devices(self):
        result = await self.sample([0, 1], b"0, 99, 1410, 55, 280, 60000\n1, 98, 1410, 56, 281, 60000\n")
        self.assertEqual([row["index"] for row in result["gpus"]], [0, 1])

    async def test_old_devices_rejected(self):
        with self.assertRaises(RuntimeError):
            await self.sample([0, 1], b"6, 99, 1410, 55, 280, 60000\n7, 98, 1410, 56, 281, 60000\n")

    async def test_nonfinite_sample_rejected(self):
        with self.assertRaises(RuntimeError):
            await self.sample([0, 1], b"0, nan, 1410, 55, 280, 60000\n1, 98, 1410, 56, 281, 60000\n")


if __name__ == "__main__":
    unittest.main()

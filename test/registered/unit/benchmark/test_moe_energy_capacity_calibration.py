"""CPU-only tests for the Qwen3 A/F capacity calibration planner."""

import importlib.util
import sys
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO_ROOT / "python"))

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=2, suite="base-a-test-cpu")

SCRIPT = (
    REPO_ROOT
    / "benchmark/moe-energy/profiling/scripts/run_capacity_calibration.py"
)


def load_calibrator():
    name = "moe_energy_capacity_calibration"
    spec = importlib.util.spec_from_file_location(name, SCRIPT)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


cal = load_calibrator()


class CapacityCalibrationTest(CustomTestCase):
    def axis(self, *, size=2, length=64):
        return cal.Axis("prefill", "A", "A", "attn_tp", size, length)

    def test_start_is_next_power_after_axis_specific_baseline(self):
        axis = self.axis()
        other = cal.Axis("prefill", "F-TP", "F", "moe_tp", 2, 64)
        baseline = {
            (axis, 1, 210),
            (axis, 8, 1410),
            (other, 64, 1410),
        }
        tasks, starts, censored = cal.build_initial_plan(
            [axis],
            baseline,
            baseline,
            {},
            set(),
            {"prefill": 4096, "decode": 16384},
        )
        self.assertEqual(starts, [(axis, 8, 16)])
        self.assertEqual(tasks, [cal.Task(axis, 16, 1410, "probe")])
        self.assertEqual(censored, [])

    def test_successful_probe_generates_backfill_and_next_probe(self):
        task = cal.Task(self.axis(), 128, 1410, "probe")
        followups, terminal = cal.advance_probe(
            task,
            succeeded=True,
            cap=4096,
            known_freqs={1410, 690},
        )
        self.assertIsNone(terminal)
        self.assertEqual(
            {(item.kind, item.batch, item.freq) for item in followups},
            {
                ("backfill", 128, 210),
                ("backfill", 128, 450),
                ("backfill", 128, 930),
                ("backfill", 128, 1170),
                ("probe", 256, 1410),
            },
        )

    def test_failed_probe_stops_chain(self):
        axis = self.axis()
        followups, terminal = cal.advance_probe(
            cal.Task(axis, 128, 1410, "probe"),
            succeeded=False,
            cap=4096,
            known_freqs=set(),
        )
        self.assertEqual(followups, [])
        self.assertEqual(terminal, "boundary")

        baseline = {(axis, 64, 1410)}
        successes = baseline | {(axis, 128, 1410)}
        tasks, _, _ = cal.build_initial_plan(
            [axis],
            baseline,
            successes,
            {axis.key: {"status": "timeout", "boundary": True}},
            set(),
            {"prefill": 4096, "decode": 16384},
        )
        self.assertEqual(
            {(task.kind, task.batch, task.freq) for task in tasks},
            {
                ("backfill", 128, 210),
                ("backfill", 128, 450),
                ("backfill", 128, 690),
                ("backfill", 128, 930),
                ("backfill", 128, 1170),
            },
        )

    def test_success_at_host_cap_is_censored(self):
        followups, terminal = cal.advance_probe(
            cal.Task(self.axis(), 4096, 1410, "probe"),
            succeeded=True,
            cap=4096,
            known_freqs={1410},
        )
        self.assertEqual(terminal, "censored")
        self.assertTrue(followups)
        self.assertTrue(all(item.kind == "backfill" for item in followups))

    def test_gpu_packing_fills_node_capacity_by_world_size(self):
        nodes = [
            cal.Node("node1", "local"),
            cal.Node("node3", "host3"),
            cal.Node("node4", "host4"),
        ]
        for size, task_count, expected_per_node in (
            (8, 3, 1),
            (4, 6, 2),
            (2, 12, 4),
        ):
            tasks = [
                cal.Task(self.axis(size=size, length=64 + index), 1, 1410, "probe")
                for index in range(task_count)
            ]
            packed = cal.pack_once(tasks, nodes)
            self.assertEqual(len(packed), task_count)
            counts = {
                node.name: sum(item_node == node for _, item_node, _ in packed)
                for node in nodes
            }
            self.assertEqual(set(counts.values()), {expected_per_node})
            self.assertEqual(sum(len(gpus) for _, _, gpus in packed), 24)


if __name__ == "__main__":
    unittest.main()

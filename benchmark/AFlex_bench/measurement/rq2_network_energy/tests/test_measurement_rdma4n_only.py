import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts")]
spec = importlib.util.spec_from_file_location("r4", ROOT / "scripts/run_measurement_rdma4n_only.py")
r4 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(r4)
from aflex_benchmark.config import load_bundle
from aflex_benchmark.runner import expand_queue


class RDMA4NOnlyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.bundle = load_bundle(ROOT / "configs", ROOT / "configs/measurement_rdma_qps4_qwen3_30b_a3b_4gpu.json")
        cls.items = {(x["point"]["architecture"], x["workload"]): x for x in expand_queue(cls.bundle) if x["point"]["metadata"]["topology_id"] == "rdma4n"}

    def test_three_lanes_use_distinct_local_hcas(self):
        self.assertEqual({lane: value["nic"] for lane, value in r4.LANES.items()}, {0: "mlx5_0", 2: "mlx5_1", 6: "mlx5_5"})
        self.assertEqual(len({value["nic"] for value in r4.LANES.values()}), 3)
        for lane, value in r4.LANES.items():
            self.assertTrue(all(gpus == [lane] for gpus in value["gpus"].values()))

    def test_architecture_roles_and_symmetric_startup(self):
        expected = {
            "native": {"NATIVE": {"node1", "node2", "node3", "node4"}},
            "pd": {"P": {"node1", "node2"}, "D": {"node3", "node4"}},
            "af": {"F": {"node1", "node2"}, "A": {"node3", "node4"}},
        }
        for arch in r4.ARCHITECTURES:
            item, _ = r4.configure(self.items[arch, r4.WORKLOADS[0]], 1, 1, 0, [], 1)
            plan = r4.core.validate_plan_shape(self.bundle, item)
            r4.validate_symmetric_stages(plan)
            self.assertEqual(plan.used_gpu_map(), {f"node{i}": [0] for i in range(1, 5)})
            actual = {}
            for process in plan.processes:
                if process.gpus:
                    actual.setdefault(process.role, set()).add(process.node)
            self.assertEqual(actual, expected[arch])

    def test_repeat_run_ids_and_artifacts_are_independent(self):
        ids = []
        paths = []
        for repeat in (1, 2, 3):
            item, _ = r4.configure(self.items["native", r4.WORKLOADS[0]], repeat, repeat, 0, [], 1)
            ids.append(item["run_id"])
            paths.append(f"repeat{repeat}/attempt1/{item['run_id']}")
        self.assertEqual(len(set(ids)), 3)
        self.assertEqual(len(set(paths)), 3)
        self.assertTrue(all(f"repeat{i}" in paths[i - 1] for i in (1, 2, 3)))

    def test_formal_runner_requires_atomic_canary_gate(self):
        with tempfile.TemporaryDirectory() as directory:
            results = Path(directory)
            with self.assertRaisesRegex(RuntimeError, "gate missing"):
                r4.require_canary_gate(results)
            (results / r4.GATE_NAME).write_text(json.dumps({"status": "fail"}))
            with self.assertRaisesRegex(RuntimeError, "not pass"):
                r4.require_canary_gate(results)

    def test_exactly_54_logical_keys(self):
        keys = {(repeat, point["point"]["id"], workload) for repeat in (1, 2, 3) for (_, workload), point in self.items.items()}
        self.assertEqual(len(keys), 54)


if __name__ == "__main__":
    unittest.main()

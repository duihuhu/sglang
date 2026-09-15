import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from aflex_benchmark.config import load_bundle
from aflex_benchmark.deploy import build_plan
from aflex_benchmark.runner import execute_item, expand_queue

CONFIG = "pd_cuda_ipc_qwen3_30b_a3b_same_node_smoke.json"


class PDCudaIpcSmokeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.bundle = load_bundle(ROOT / "configs", ROOT / "configs" / CONFIG)

    def test_schema_queue_and_shared_trace_snapshot(self):
        import jsonschema

        matrix = self.bundle["matrix"]
        jsonschema.validate(matrix, json.loads((ROOT / "schemas/matrix.schema.json").read_text()))
        self.assertEqual(len(matrix["points"]), 2)
        self.assertEqual(len(expand_queue(self.bundle, smoke=True)), 4)
        self.assertEqual(matrix["metadata"]["expected_runs"], 4)
        trace = ROOT / "configs/shared_stage_4req.jsonl"
        self.assertEqual(len([line for line in trace.read_text().splitlines() if line]), 4)
        for point in matrix["points"]:
            self.assertEqual(point["workload_path"], str(trace))
            self.assertEqual(point["qps"], [1, 2])
            self.assertEqual(point["expected_link"], "nvlink")
            self.assertEqual(point["metadata"]["transport"], "cuda_ipc")

    def test_deployment_snapshots_use_same_host_equal_tp_and_visible_mapping(self):
        expected = [(1, [0], [1], 57200), (2, [0, 1], [2, 3], 57300)]
        model = self.bundle["models"]["models"]["moe_qwen3_30b_a3b"]["path"]
        for point, (tp, pgpus, dgpus, base) in zip(self.bundle["matrix"]["points"], expected):
            with self.subTest(point=point["id"]):
                plan = build_plan(self.bundle["cluster"], model, point)
                plan.validate()
                p, d, router = plan.processes
                self.assertEqual((p.role, d.role, router.role), ("P", "D", "PD_ROUTER"))
                self.assertEqual((p.host, d.host, router.host), (p.host, p.host, p.host))
                self.assertEqual((p.gpus, d.gpus), (pgpus, dgpus))
                self.assertEqual(plan.used_gpu_map(), {p.node: pgpus + dgpus})
                self.assertEqual((p.port, d.port, router.port), (base, base + 10, base + 20))
                self.assertEqual((p.nccl_port, d.nccl_port), (base + 1, base + 11))
                self.assertEqual(p.bootstrap_port, base + 2)
                self.assertEqual((p.startup_stage, d.startup_stage, router.startup_stage), (0, 1, 2))
                for process in (p, d, router):
                    self.assertEqual(process.metadata["transport"], "cuda_ipc")
                    self.assertEqual(process.metadata["expected_link"], "nvlink")
                    self.assertEqual(process.metadata["tp"], tp)
                for process in (p, d):
                    self.assertIn("--disaggregation-transfer-backend cuda_ipc", process.command)
                    self.assertIn(f"--disaggregation-bootstrap-port {base + 2}", process.command)
                    self.assertIn(f"--tp {tp}", process.command)
                visible = ",".join(map(str, pgpus + dgpus))
                self.assertIn(f"CUDA_VISIBLE_DEVICES={visible}", p.command)
                self.assertIn(f"CUDA_VISIBLE_DEVICES={visible}", d.command)
                self.assertIn("--base-gpu-id 0", p.command)
                self.assertIn(f"--base-gpu-id {tp}", d.command)
                self.assertIn(f"--prefill http://{p.host}:{base} {base + 2}", router.command)
                self.assertIn(f"--decode http://{d.host}:{base + 10}", router.command)

    def test_dry_run_never_executes_or_writes_results(self):
        queue = expand_queue(self.bundle, smoke=True)
        self.assertEqual({item["state"] for item in queue}, {"ready"})
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / "must-not-exist"
            for item in queue:
                self.assertEqual(execute_item(item, self.bundle, target, dry_run=True)["status"], "dry-run")
            self.assertFalse(target.exists())


if __name__ == "__main__":
    unittest.main()

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

CONFIG = "af_mooncake_rdma_qwen3_30b_a3b_cross_node_smoke.json"


class AFMooncakeRDMASmokeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.bundle = load_bundle(ROOT / "configs", ROOT / "configs" / CONFIG)

    def test_schema_queue_and_shared_trace_snapshot(self):
        import jsonschema

        matrix = self.bundle["matrix"]
        jsonschema.validate(matrix, json.loads((ROOT / "schemas/matrix.schema.json").read_text()))
        self.assertEqual(len(matrix["points"]), 3)
        self.assertEqual(len(expand_queue(self.bundle, smoke=True)), 6)
        self.assertEqual(matrix["metadata"]["expected_runs"], 6)
        self.assertIn("sync", matrix["metadata"]["run_prerequisite"])
        point = matrix["points"][0]
        trace = ROOT / "configs/shared_stage_4req.jsonl"
        self.assertEqual(len([line for line in trace.read_text().splitlines() if line]), 4)
        self.assertEqual(point["workload_path"], str(trace))
        self.assertEqual(point["workloads"], ["shared_stage_4req"])
        self.assertEqual(point["qps"], [1, 2])
        self.assertEqual(point["expected_link"], "rdma")
        self.assertEqual(point["afd_backend"], "mooncake")
        self.assertTrue(point["metadata"]["requires_code_sync"])

    def test_deployment_snapshot_uses_node3_a_node4_f_and_mooncake_ports(self):
        point = self.bundle["matrix"]["points"][0]
        model = self.bundle["models"]["models"][point["model"]]["path"]
        plan = build_plan(self.bundle["cluster"], model, point)
        plan.validate()
        f, a = (next(process for process in plan.processes if process.role == role)
                for role in ("F", "A"))
        self.assertEqual((a.node, a.gpus, f.node, f.gpus), ("node3", [0], "node4", [0]))
        self.assertEqual(plan.used_gpu_map(), {"node4": [0], "node3": [0]})
        self.assertEqual((f.port, a.port), (57400, 57410))
        self.assertEqual((f.startup_stage, a.startup_stage), (0, 0))
        self.assertEqual(f.internal_ports, [57402, 57440])
        self.assertEqual(a.internal_ports, [57430])
        for process, peer in ((f, a.host), (a, f.host)):
            self.assertIn("--afd-comm-backend mooncake", process.command)
            self.assertIn("--afd-micro-batch 1", process.command)
            self.assertIn("--afd-attn-tp 1", process.command)
            self.assertIn("--afd-ffn-tp 1", process.command)
            self.assertIn("CUDA_VISIBLE_DEVICES=0", process.command)
            self.assertIn("--base-gpu-id 0", process.command)
            self.assertIn("SGLANG_MOONCAKE_TRANSPORT=rdma", process.command)
            self.assertIn("MOONCAKE_IB_DEVICE=mlx5_0", process.command)
            self.assertIn("MOONCAKE_USE_CUDA_IPC=0", process.command)
            self.assertIn("GLOO_SOCKET_IFNAME=bond0", process.command)
            self.assertIn("NCCL_SOCKET_IFNAME=bond0", process.command)
            self.assertIn(f"SGLANG_HOST_IP={process.host}", process.command)
            self.assertIn("AFD_MOONCAKE_ATTN_CONTROL_PORT=57440", process.command)
            self.assertIn("AFD_MOONCAKE_FFN_CONTROL_PORT=57430", process.command)
            self.assertIn(f"AFD_SCHED_HOST={f.host}", process.command)
            self.assertIn(f"AFD_MOONCAKE_PEER_HOST={peer}", process.command)
            self.assertEqual(process.metadata["expected_link"], "rdma")
            self.assertEqual(process.metadata["transport"], "mooncake")
        self.assertEqual(plan.pre_actions, [])


    def test_tp2_local_snapshot(self):
        point = self.bundle["matrix"]["points"][1]
        model = self.bundle["models"]["models"][point["model"]]["path"]
        plan = build_plan(self.bundle["cluster"], model, point)
        plan.validate()
        f, a = (next(process for process in plan.processes if process.role == role)
                for role in ("F", "A"))
        self.assertEqual((a.node, a.gpus, f.node, f.gpus),
                         ("node3", [0, 1], "node4", [0, 1]))
        self.assertEqual((f.port, a.port), (57500, 57510))
        self.assertEqual((f.startup_stage, a.startup_stage), (0, 0))
        self.assertEqual((f.health_path, a.health_path),
                         ("/get_model_info", "/get_model_info"))
        self.assertNotIn("--nnodes", f.command + a.command)
        self.assertIn("CUDA_VISIBLE_DEVICES=0,1", f.command)
        self.assertIn("CUDA_VISIBLE_DEVICES=0,1", a.command)
        self.assertEqual(f.internal_ports, [57505, 57540, 57541])
        self.assertEqual(a.internal_ports, [57530, 57531])
        for process, peer in ((f, a.host), (a, f.host)):
            self.assertIn("--tp 2", process.command)
            self.assertIn("--afd-comm-backend mooncake", process.command)
            self.assertIn("--afd-attn-tp 2", process.command)
            self.assertIn("--afd-ffn-tp 2", process.command)
            self.assertIn(f"AFD_MOONCAKE_PEER_HOST={peer}", process.command)
            self.assertIn("GLOO_SOCKET_IFNAME=bond0", process.command)
            self.assertIn("NCCL_SOCKET_IFNAME=bond0", process.command)
            self.assertIn(f"SGLANG_HOST_IP={process.host}", process.command)
            self.assertEqual(process.metadata["global_tp"], 2)
            self.assertEqual(process.metadata["component_nodes"], 1)
            self.assertEqual(process.metadata["node_rank"], 0)

    def test_tp2_global_snapshot_has_rank_mapped_peers_and_rank0_readiness(self):
        point = self.bundle["matrix"]["points"][2]
        model = self.bundle["models"]["models"][point["model"]]["path"]
        plan = build_plan(self.bundle["cluster"], model, point)
        plan.validate()
        self.assertEqual(plan.used_gpu_map(), {
            "node1": [0], "node2": [0], "node3": [0], "node4": [0]
        })
        by_role = {role: sorted(
            (process for process in plan.processes if process.role == role),
            key=lambda process: process.metadata["node_rank"])
            for role in ("A", "F")}
        self.assertEqual([process.node for process in by_role["A"]], ["node1", "node2"])
        self.assertEqual([process.node for process in by_role["F"]], ["node3", "node4"])
        self.assertEqual([process.startup_stage for process in plan.processes], [0, 0, 0, 0])
        self.assertEqual([process.health_path for process in by_role["A"]],
                         ["/get_model_info", None])
        self.assertEqual([process.health_path for process in by_role["F"]],
                         ["/get_model_info", None])
        expected_peers = {
            "node1": "10.252.129.34", "node2": "10.252.129.33",
            "node3": "10.252.129.36", "node4": "10.252.129.35",
        }
        for role, specs in by_role.items():
            dist = "10.252.129.36:57612" if role == "A" else "10.252.129.34:57602"
            for rank, process in enumerate(specs):
                self.assertIn("--nnodes 2", process.command)
                self.assertIn(f"--node-rank {rank}", process.command)
                self.assertIn(f"--dist-init-addr {dist}", process.command)
                self.assertIn("GLOO_SOCKET_IFNAME=bond0", process.command)
                self.assertIn("NCCL_SOCKET_IFNAME=bond0", process.command)
                self.assertIn(f"SGLANG_HOST_IP={process.host}", process.command)
                self.assertIn(f"AFD_MOONCAKE_PEER_HOST={expected_peers[process.node]}",
                              process.command)
                self.assertEqual(process.metadata["global_tp"], 2)
                self.assertEqual(process.metadata["component_nodes"], 2)
                self.assertEqual(process.metadata["readiness_rank"], 0)
                control = 57640 + rank if role == "F" else 57630 + rank
                expected_internal = (
                    ([57602 if role == "F" else 57612] if rank == 0 else [])
                    + ([57605] if role == "F" and rank == 0 else [])
                    + [control]
                )
                self.assertEqual(process.internal_ports, expected_internal)
        self.assertEqual(plan.endpoint, "http://10.252.129.36:57610")

    def test_dry_run_never_executes_or_writes_results(self):
        queue = expand_queue(self.bundle, smoke=True)
        self.assertEqual({item["state"] for item in queue}, {"ready"})
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / "must-not-exist"
            for item in queue:
                result = execute_item(item, self.bundle, target, dry_run=True)
                self.assertEqual(result["status"], "dry-run")
            self.assertFalse(target.exists())


if __name__ == "__main__":
    unittest.main()

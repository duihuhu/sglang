import importlib.util
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

SPEC = importlib.util.spec_from_file_location(
    "run_comm_ablation", ROOT / "scripts/run_comm_ablation.py"
)
ORCHESTRATOR = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(ORCHESTRATOR)

MATRIX = "matrix_workloads_qwen3_30b_a3b_4gpu_qps2.json"
SMOKE_MATRIX = "matrix_workloads_qwen3_30b_a3b_4gpu_smoke.json"


class MatrixWorkloadsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.bundle = load_bundle(ROOT / "configs", ROOT / "configs" / MATRIX)

    def test_schema_and_matrix_cardinality(self):
        import jsonschema

        matrix = self.bundle["matrix"]
        jsonschema.validate(
            matrix, json.loads((ROOT / "schemas/matrix.schema.json").read_text())
        )
        self.assertEqual(len(matrix["points"]), 12)
        self.assertEqual(len(expand_queue(self.bundle)), 72)
        self.assertEqual(len(ORCHESTRATOR.build_runs(self.bundle, "qps2", 3)), 216)
        self.assertEqual(matrix["metadata"]["expected_runs"], 216)

    def test_independent_smoke_matrix_preserves_deployment_and_plans_12_runs(self):
        smoke = load_bundle(ROOT / "configs", ROOT / "configs" / SMOKE_MATRIX)
        source_points = self.bundle["matrix"]["points"]
        smoke_points = smoke["matrix"]["points"]
        workload_path = ROOT / "configs/shared_stage_4req.jsonl"

        self.assertEqual(len(smoke_points), 12)
        self.assertEqual(len(ORCHESTRATOR.build_runs(smoke, "smoke", 3)), 12)
        self.assertEqual(len(ORCHESTRATOR.build_runs(smoke, "all", 3)), 12)
        self.assertEqual(smoke["matrix"]["metadata"]["expected_runs"], 12)
        self.assertEqual(
            len(
                [
                    line
                    for line in workload_path.read_text().splitlines()
                    if line.strip()
                ]
            ),
            4,
        )
        deployment_keys = (
            "model",
            "architecture",
            "recipe",
            "nodes",
            "gpus_total",
            "expected_link",
            "port_base",
            "skip_warmup",
            "multinode_preflight_complete",
        )
        optional_deployment_keys = (
            "pd_tp",
            "afd_backend",
            "af_tp",
            "attention_placements",
            "ffn_placements",
            "semantic_comm_ledger",
            "expected_internal_link",
            "component_tp",
        )
        for source, point in zip(source_points, smoke_points):
            with self.subTest(point=point["id"]):
                self.assertEqual(
                    point["id"], source["id"].removesuffix("-qps2") + "-smoke"
                )
                self.assertEqual(
                    {key: point[key] for key in deployment_keys},
                    {key: source[key] for key in deployment_keys},
                )
                self.assertEqual(
                    {key: point.get(key) for key in optional_deployment_keys},
                    {key: source.get(key) for key in optional_deployment_keys},
                )
                self.assertEqual(
                    point["metadata"]["topology_id"], source["metadata"]["topology_id"]
                )
                self.assertEqual(
                    point["metadata"]["ablation_id"],
                    source["metadata"]["comm_ablation_id"],
                )
                self.assertEqual(point["metadata"]["phase"], "smoke")
                self.assertEqual(
                    (
                        point["workloads"],
                        point["qps"],
                        point["max_inflight"],
                        Path(point["workload_path"]),
                        point["smoke"],
                    ),
                    (["comm_smoke"], [2], 4, workload_path, True),
                )
                source_plan = build_plan(
                    self.bundle["cluster"],
                    self.bundle["models"]["models"][source["model"]]["path"],
                    source,
                )
                smoke_plan = build_plan(
                    smoke["cluster"],
                    smoke["models"]["models"][point["model"]]["path"],
                    point,
                )
                self.assertEqual(
                    [
                        (p.role, p.node, p.gpus, p.port, p.command, p.metadata)
                        for p in smoke_plan.processes
                    ],
                    [
                        (p.role, p.node, p.gpus, p.port, p.command, p.metadata)
                        for p in source_plan.processes
                    ],
                )

    def test_topology_cross_product_and_unique_ports(self):
        points = self.bundle["matrix"]["points"]
        self.assertEqual(
            {(p["architecture"], p["metadata"]["topology_id"]) for p in points},
            {
                (a, t)
                for a in ("native", "pd", "af")
                for t in ("pcie1n", "rdma2n", "rdma4n", "nvlink1n")
            },
        )
        self.assertEqual(len({p["port_base"] for p in points}), 12)
        self.assertTrue(
            all(
                p["gpus_total"] == 4 and len(p["workloads"]) == 6 and p["qps"] == [2]
                for p in points
            )
        )

    def test_all_plans_compile_to_exact_four_unique_gpus(self):
        for point in self.bundle["matrix"]["points"]:
            with self.subTest(point=point["id"]):
                plan = build_plan(
                    self.bundle["cluster"],
                    self.bundle["models"]["models"][point["model"]]["path"],
                    point,
                )
                plan.validate()
                allocated = [(p.node, gpu) for p in plan.processes for gpu in p.gpus]
                self.assertEqual(len(allocated), 4)
                self.assertEqual(len(set(allocated)), 4)
                self.assertEqual(sum(len(v) for v in plan.used_gpu_map().values()), 4)

    def test_measurement_explicit_gpu_nic_placements(self):
        bundle = load_bundle(
            ROOT / "configs",
            ROOT / "configs/measurement_rdma_qps4_qwen3_30b_a3b_4gpu.json",
        )
        model = bundle["models"]["models"]["moe_qwen3_30b_a3b"]["path"]
        for point in bundle["matrix"]["points"]:
            plan = build_plan(bundle["cluster"], model, point)
            expected = {"node1": [2, 3], "node2": [2, 3]} if point["nodes"] == 2 else {f"node{i}": [2] for i in range(1, 5)}
            self.assertEqual(plan.used_gpu_map(), expected)
            for process in (item for item in plan.processes if item.gpus):
                self.assertIn("CUDA_VISIBLE_DEVICES=2" if len(process.gpus) == 1 else "CUDA_VISIBLE_DEVICES=2,3", process.command)
                self.assertIn("mlx5_1", process.command)

    def test_global_components_have_parameterized_rank_zero_readiness(self):
        for point in self.bundle["matrix"]["points"]:
            if point["nodes"] == 1:
                continue
            plan = build_plan(
                self.bundle["cluster"],
                self.bundle["models"]["models"][point["model"]]["path"],
                point,
            )
            components = {}
            for process in plan.processes:
                if not process.gpus:
                    continue
                components.setdefault(process.role, []).append(process)
            for specs in components.values():
                ranks = sorted(p.metadata["node_rank"] for p in specs)
                self.assertEqual(ranks, list(range(len(specs))))
                self.assertEqual(sum(p.health_path is not None for p in specs), 1)
                self.assertIsNotNone(
                    next(p for p in specs if p.metadata["node_rank"] == 0).health_path
                )

    def test_global_components_pin_collectives_and_host_to_roce_ipv4(self):
        nodes = {node["name"]: node for node in self.bundle["cluster"]["nodes"]}
        for point in self.bundle["matrix"]["points"]:
            plan = build_plan(
                self.bundle["cluster"],
                self.bundle["models"]["models"][point["model"]]["path"],
                point,
            )
            for process in (p for p in plan.processes if p.gpus):
                node = nodes[process.node]
                self.assertIn(f"GLOO_SOCKET_IFNAME={node['roce']}", process.command)
                self.assertIn(f"NCCL_SOCKET_IFNAME={node['roce']}", process.command)
                self.assertIn(f"SGLANG_HOST_IP={node['host']}", process.command)

    def test_verified_backend_snapshots_and_dry_run(self):
        points = {
            (p["architecture"], p["metadata"]["topology_id"]): p
            for p in self.bundle["matrix"]["points"]
        }
        model = self.bundle["models"]["models"]["moe_qwen3_30b_a3b"]["path"]

        pd_snapshots = {
            "pcie1n": {
                "P": [("node1", "0,1", 0)],
                "D": [("node1", "2,3", 0)],
            },
            "rdma2n": {
                "P": [("node1", "0,1", 0)],
                "D": [("node2", "0,1", 0)],
            },
            "rdma4n": {
                "P": [("node1", "0", 0), ("node2", "0", 0)],
                "D": [("node3", "0", 0), ("node4", "0", 0)],
            },
            "nvlink1n": {
                "P": [("node1", "0,1,2,3", 0)],
                "D": [("node1", "0,1,2,3", 2)],
            },
        }
        for topology_id, expected_roles in pd_snapshots.items():
            with self.subTest(architecture="pd", topology_id=topology_id):
                pd = build_plan(
                    self.bundle["cluster"], model, points[("pd", topology_id)]
                )
                for role, expected in expected_roles.items():
                    processes = sorted(
                        (x for x in pd.processes if x.role == role),
                        key=lambda x: x.metadata["node_rank"],
                    )
                    self.assertEqual(len(processes), len(expected))
                    for process, (node, visible, base_gpu_id) in zip(
                        processes, expected
                    ):
                        self.assertEqual(process.node, node)
                        self.assertIn(
                            f"CUDA_VISIBLE_DEVICES={visible}", process.command
                        )
                        self.assertIn(f"--base-gpu-id {base_gpu_id}", process.command)

        pd_nvlink = build_plan(
            self.bundle["cluster"], model, points[("pd", "nvlink1n")]
        )
        self.assertTrue(
            all(
                "--disaggregation-transfer-backend cuda_ipc" in process.command
                for process in pd_nvlink.processes
                if process.gpus
            )
        )

        pd_pcie = build_plan(self.bundle["cluster"], model, points[("pd", "pcie1n")])
        for process in (x for x in pd_pcie.processes if x.gpus):
            self.assertIn("unset MC_FORCE_HCA MC_FORCE_MNNVL", process.command)
            self.assertIn("SGLANG_MOONCAKE_CUSTOM_MEM_POOL; export", process.command)
            self.assertIn("MC_FORCE_TCP=1", process.command)
            self.assertIn("MC_LOG_LEVEL=INFO", process.command)
            self.assertIn("MOONCAKE_USE_CUDA_IPC=0", process.command)
            self.assertIn("MOONCAKE_PROTOCOL=tcp", process.command)
            self.assertIn("SGLANG_MOONCAKE_TRANSPORT=tcp", process.command)
            self.assertEqual(process.metadata["expected_internal_link"], "nvlink")
            self.assertEqual(process.metadata["component_tp"], 2)
            self.assertEqual(process.metadata["comm_ledger_backend"], "mooncake_tcp")

        af_pcie = build_plan(self.bundle["cluster"], model, points[("af", "pcie1n")])
        self.assertTrue(
            all(
                "--afd-comm-backend zmq" in x.command
                for x in af_pcie.processes
                if x.gpus
            )
        )
        for process in (x for x in af_pcie.processes if x.gpus):
            self.assertEqual(process.metadata["expected_internal_link"], "nvlink")
            self.assertEqual(process.metadata["component_tp"], 2)
            self.assertEqual(process.metadata["comm_ledger_backend"], "af_zmq")
        af_nvlink = build_plan(
            self.bundle["cluster"], model, points[("af", "nvlink1n")]
        )
        self.assertTrue(
            all(
                "--afd-comm-backend ipc_cpp" in x.command
                for x in af_nvlink.processes
                if x.gpus
            )
        )
        self.assertTrue(
            all(
                "CUDA_VISIBLE_DEVICES=0,1,2,3" in x.command
                for x in af_nvlink.processes
                if x.gpus
            )
        )

        local = build_plan(self.bundle["cluster"], model, points[("af", "rdma2n")])
        self.assertEqual(local.used_gpu_map(), {"node1": [0, 1], "node2": [0, 1]})
        self.assertTrue(
            all("--afd-comm-backend mooncake" in x.command for x in local.processes)
        )
        global_plan = build_plan(
            self.bundle["cluster"], model, points[("af", "rdma4n")]
        )
        self.assertEqual(
            global_plan.used_gpu_map(),
            {"node1": [0], "node2": [0], "node3": [0], "node4": [0]},
        )
        for role in ("A", "F"):
            ranks = sorted(
                (x for x in global_plan.processes if x.role == role),
                key=lambda x: x.metadata["node_rank"],
            )
            self.assertEqual([x.metadata["node_rank"] for x in ranks], [0, 1])
            self.assertIn("--nnodes 2", ranks[0].command)

        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / "must-not-exist"
            for item in expand_queue(self.bundle):
                self.assertEqual(
                    execute_item(item, self.bundle, target, dry_run=True)["status"],
                    "dry-run",
                )
            self.assertFalse(target.exists())

    def test_af_mooncake_timeout_derives_from_request_timeout_with_floor(self):
        points = {
            p["metadata"]["topology_id"]: p
            for p in self.bundle["matrix"]["points"]
            if p["architecture"] == "af" and p.get("afd_backend") == "mooncake"
        }
        model = self.bundle["models"]["models"]["moe_qwen3_30b_a3b"]["path"]
        for topology_id, point in points.items():
            with self.subTest(topology_id=topology_id):
                plan = build_plan(self.bundle["cluster"], model, point)
                self.assertTrue(
                    all(
                        "AFD_MOONCAKE_TIMEOUT_MS=300000" in process.command
                        for process in plan.processes
                    )
                )

        point = json.loads(json.dumps(next(iter(points.values()))))
        point["request_timeout_s"] = 3600
        plan = build_plan(self.bundle["cluster"], model, point)
        self.assertTrue(
            all(
                "AFD_MOONCAKE_TIMEOUT_MS=3660000" in process.command
                for process in plan.processes
            )
        )

        point["afd_mooncake_timeout_ms"] = 1234567
        plan = build_plan(self.bundle["cluster"], model, point)
        self.assertTrue(
            all(
                "AFD_MOONCAKE_TIMEOUT_MS=1234567" in process.command
                for process in plan.processes
            )
        )

    def test_fixed_qps2_workloads_are_seeded_64_request_snapshots(self):
        for workload in self.bundle["matrix"]["points"][0]["workloads"]:
            rows = [
                json.loads(line)
                for line in (ROOT / "data" / "workloads" / f"{workload}_qps2.jsonl")
                .read_text()
                .splitlines()
            ]
            self.assertEqual(len(rows), 64)
            self.assertTrue(all(row["source"] == workload for row in rows))
            self.assertTrue(
                all(
                    b["arrival_time_s"] > a["arrival_time_s"]
                    for a, b in zip(rows, rows[1:])
                )
            )


if __name__ == "__main__":
    unittest.main()

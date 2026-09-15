import json, shlex, subprocess, sys, tempfile, threading, time
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import unittest
from unittest.mock import patch
from aflex_benchmark.config import (ConfigError, load_bundle, validate_cluster,
                                     validate_matrix, validate_workloads)
from aflex_benchmark.deploy import build_plan
from aflex_benchmark.deploy.base import (DeploymentHandle, DeploymentPlan, ProcessSpec,
                                             RemoteExecutor, RoutingPolicy, execute_lifecycle)
from aflex_benchmark.runner import (expand_queue, execute_item, filter_queue_by_point_ids,
                                    normalize_requests, request_endpoint,
                                    resolve_workload_path)
from aflex_benchmark.analysis import aggregate
from aflex_benchmark.stats import summarize_requests
from aflex_benchmark.workloads import generate

def bundle(name):
    return load_bundle(ROOT / "configs", ROOT / "configs" / name)


def _capture_lifecycle(plan, executor, result, error):
    try:
        result.append(execute_lifecycle(
            plan, executor, lambda _: 9, health_timeout_s=37))
    except BaseException as exc:
        error.append(exc)

class BenchmarkSuiteTests(unittest.TestCase):
    def test_rq5_complete_matrix(self):
        points = bundle("rq5_32gpu.json")["matrix"]["points"]
        formal = [p for p in points if not p.get("experimental")]
        self.assertEqual({p["model"] for p in formal}, {"dense_qwen3_32b", "moe_qwen3_30b_a3b"})
        self.assertEqual({p["architecture"] for p in formal}, {"native", "pd", "af", "pdaf"})
        self.assertTrue(all(p["nodes"] == 4 and p["qps"] == [1,4,8] and len(p["workloads"]) == 8 for p in formal))
        self.assertTrue(all(p.get("experimental") for p in points if p.get("parallelism") == "ep"))

    def test_qps_cap_rejects_matrix_and_workload_values_above_16(self):
        with self.assertRaisesRegex(ConfigError, "must not exceed 16"):
            validate_matrix({
                "rq": "RQ5",
                "points": [{
                    "id": "too-fast", "architecture": "native",
                    "nodes": 1, "qps": [16.01],
                }],
            })
        with self.assertRaisesRegex(ConfigError, "must not exceed 16"):
            validate_workloads({"max_qps": 16, "qps": [1, 17]})

    def test_formal_qps_matrices_respect_cap(self):
        workloads = bundle("rq5_32gpu.json")["workloads"]
        self.assertEqual(workloads["max_qps"], 16)
        self.assertLessEqual(max(workloads["qps"]), 16)
        for name in ("rq5_32gpu.json", "rq6_scaling.json"):
            points = bundle(name)["matrix"]["points"]
            self.assertTrue(all(max(point["qps"]) <= 16 for point in points))

    def test_multinode_af_variants_require_preflight(self):
        for name in ("rq5_32gpu.json", "rq6_scaling.json"):
            points = bundle(name)["matrix"]["points"]
            risky = [p for p in points if p["architecture"] in {"af", "pdaf"} and p["nodes"] > 1]
            self.assertTrue(risky)
            self.assertTrue(all(p.get("requires_preflight") and p.get("blocked_reason") for p in risky))

    def test_smoke_matrix_builds_all_eight_8gpu_points(self):
        b = bundle("smoke.json")
        points = b["matrix"]["points"]
        expected_models = {"dense_qwen3_32b", "moe_qwen3_30b_a3b"}
        expected_architectures = {"native", "pd", "af", "pdaf"}
        self.assertEqual(len(points), 8)
        self.assertEqual(
            {(p["model"], p["architecture"]) for p in points},
            {(model, architecture) for model in expected_models
             for architecture in expected_architectures},
        )
        for point in points:
            with self.subTest(point=point["id"]):
                self.assertEqual(point["nodes"], 1)
                self.assertIs(point["smoke"], True)
                self.assertEqual(point["workloads"], ["fixed_short"])
                self.assertEqual(point["qps"], [1])
                plan = build_plan(
                    b["cluster"],
                    b["models"]["models"][point["model"]]["path"],
                    point,
                )
                plan.validate()
                self.assertEqual(plan.routing_policy, RoutingPolicy.SINGLE)
                self.assertEqual(plan.endpoints, [plan.endpoint])
                allocated = [(process.node, gpu) for process in plan.processes
                             for gpu in process.gpus]
                self.assertEqual(len(allocated), 8)
                self.assertEqual(len(set(allocated)), 8)

    def _legacy_32_plan(self, architecture):
        b = bundle("smoke_32gpu.json")
        point = next(p for p in b["matrix"]["points"] if p["architecture"] == architecture)
        return b, point, build_plan(b["cluster"], b["models"]["models"][point["model"]]["path"], point)

    def test_32gpu_smoke_builds_and_conserves_all_gpus(self):
        b = bundle("smoke_32gpu.json")
        expected = {"native":"legacy_native_tp", "pd":"legacy_pd_dual",
                    "af":"legacy_af_profile_replicas", "pdaf":"legacy_pdaf_3p1d"}
        self.assertEqual({p["architecture"]: p["recipe"] for p in b["matrix"]["points"]}, expected)
        states = {item["point"]["architecture"]: item["state"]
                  for item in expand_queue(b, smoke=True)}
        self.assertEqual(states, {
            "native": "ready", "pd": "ready",
            "af": "requires_preflight", "pdaf": "ready",
        })
        for point in b["matrix"]["points"]:
            with self.subTest(point=point["id"]):
                plan = build_plan(b["cluster"], b["models"]["models"][point["model"]]["path"], point)
                allocated = [(p.node,g) for p in plan.processes for g in p.gpus]
                self.assertEqual((len(allocated),len(set(allocated))), (32,32))
                ports=[]
                for p in plan.processes:
                    ports += [(p.host,p.port)]
                    ports += [(p.host,x) for x in ([p.bootstrap_port] if p.bootstrap_port else [])]
                    ports += [(p.host,x) for x in ([p.nccl_port] if p.nccl_port else [])]
                    ports += [(p.host,x) for x in p.internal_ports]
                self.assertEqual(len(ports),len(set(ports)))
                servers=[p for p in plan.processes if "sglang.launch_server" in p.command]
                self.assertTrue(all("/workspace/moe-tier/python" in p.command for p in servers))
                self.assertTrue(all("--mem-fraction-static 0.85" in p.command and "--max-running-requests 512" in p.command for p in servers))


    def test_32gpu_af_experimental_requires_explicit_unlock(self):
        b = bundle("smoke_32gpu.json")
        af = next(p for p in b["matrix"]["points"] if p["architecture"] == "af")
        self.assertIs(af["experimental"], True)
        self.assertIs(af["requires_preflight"], True)
        self.assertTrue(af["blocked_reason"])
        self.assertEqual(len(af["artifact_paths"]), 3)
        unlocked = {item["point"]["architecture"]: item["state"]
                    for item in expand_queue(
                        b, smoke=True, allow_experimental=True)}
        self.assertTrue(all(state == "ready" for state in unlocked.values()))

    def test_qps8_tp8_32gpu_resource_snapshots(self):
        b = bundle("qps8_tp8_32gpu.json")
        points = {point["architecture"]: point for point in b["matrix"]["points"]}
        self.assertEqual(set(points), {"native", "pd", "af", "pdaf"})
        self.assertTrue(all(point["workloads"] == ["fixed_tp8_qps8"]
                            and point["qps"] == [8]
                            and point["warmup_parallel"]
                            and point["wait_after_warmup_s"] == 3
                            and point["energy_scope"] == "cluster_32gpu"
                            and "max_inflight" not in point
                            for point in points.values()))
        expected_roles = {
            "native": ["NATIVE"] * 4 + ["NATIVE_ROUTER"],
            "pd": ["P", "P", "D", "D", "PD_ROUTER"],
            "af": ["AF_COORDINATOR", "F", "F", "A", "A"],
            "pdaf": ["PF", "DF", "PA", "DA", "PDAF_ROUTER"],
        }
        for architecture, point in points.items():
            plan = build_plan(b["cluster"], b["models"]["models"][point["model"]]["path"],
                              point, run_tag="qps8-snapshot")
            self.assertEqual([process.role for process in plan.processes],
                             expected_roles[architecture])
            used = [(process.node, gpu) for process in plan.processes for gpu in process.gpus]
            self.assertEqual(len(used), 32)
            self.assertEqual(len(used), len(set(used)))
            self.assertTrue(all("qps8-snapshot" in process.log_path
                                for process in plan.processes))
            ports = []
            for process in plan.processes:
                ports += [(process.host, process.port)]
                ports += [(process.host, value) for value in
                          (process.bootstrap_port, process.nccl_port) if value is not None]
                ports += [(process.host, value) for value in process.internal_ports]
            self.assertEqual(len(ports), len(set(ports)))
        self.assertEqual(points["native"]["recipe"], "legacy_native_tp")
        self.assertEqual(points["pd"]["recipe"], "pd_exact_2p2d_tp8")
        self.assertEqual(points["pdaf"]["recipe"], "pdaf_cross_zmq_tp8")
        af = points["af"]
        self.assertEqual(
            (af["multinode_preflight_complete"], af["preflight_artifact"]),
            (True, "results/shared_af_stage8/89582474f5692ab9/"),
        )
        self.assertNotIn("requires_preflight", af)
        self.assertNotIn("blocked_reason", af)
        self.assertNotIn("experimental", af)
        af_queue = [item for item in expand_queue(b)
                    if item["point"]["architecture"] == "af"]
        self.assertTrue(af_queue)
        self.assertTrue(all(item["state"] == "ready" for item in af_queue))

    def test_qps8_af_shared_bipartite_pool_snapshot(self):
        b = bundle("qps8_tp8_32gpu.json")
        point = next(p for p in b["matrix"]["points"] if p["architecture"] == "af")
        self.assertEqual(point["recipe"], "shared_bipartite_pool")
        self.assertTrue(point["multinode_preflight_complete"])
        self.assertEqual(
            point["preflight_artifact"],
            "results/shared_af_stage8/89582474f5692ab9/",
        )
        self.assertNotIn("requires_preflight", point)
        self.assertNotIn("blocked_reason", point)
        self.assertNotIn("experimental", point)
        self.assertTrue(all(item["state"] == "ready" for item in expand_queue(b)
                            if item["point"]["architecture"] == "af"))
        model = b["models"]["models"][point["model"]]["path"]
        plan = build_plan(b["cluster"], model, point, run_tag="shared-pool-test")
        plan.validate()

        coordinator, f0, f1, a0, a1 = plan.processes
        self.assertEqual(
            [(p.role, p.node, p.startup_stage) for p in plan.processes],
            [("AF_COORDINATOR", "node1", 0), ("F", "node2", 1),
             ("F", "node4", 1), ("A", "node1", 2), ("A", "node3", 2)],
        )
        self.assertEqual([len(p.gpus) for p in (f0, f1, a0, a1)], [8] * 4)
        used = [(p.node, gpu) for p in plan.processes for gpu in p.gpus]
        self.assertEqual((len(used), len(set(used))), (32, 32))
        self.assertIsNone(coordinator.health_path)
        self.assertIsNotNone(coordinator.ready_command)
        manifest_coordinator = plan.manifest()["processes"][0]
        self.assertIn(
            "export PYTHONPATH=/workspace/moe-tier/python:${PYTHONPATH:-};",
            manifest_coordinator["ready_command"],
        )
        self.assertIn(
            "from sglang.srt.managers.afd_pool_coordinator import ",
            manifest_coordinator["ready_command"],
        )
        self.assertIn("AFDPoolCoordinatorClient",
                      manifest_coordinator["ready_command"])
        self.assertEqual(point["pf_capacity"], 64)
        self.assertIn("--pf F0:64 --pf F1:64", coordinator.command)
        expected_static_pfs = [
            {"instance_id": "F0", "capacity": 64},
            {"instance_id": "F1", "capacity": 64},
        ]
        self.assertEqual(coordinator.metadata["static_pfs"], expected_static_pfs)
        self.assertEqual(manifest_coordinator["metadata"]["static_pfs"],
                         expected_static_pfs)
        self.assertEqual(plan.manifest()["afd_pool_coordinator"]["static_pfs"],
                         expected_static_pfs)
        self.assertNotIn("qwen3_sha256", coordinator.command)
        self.assertEqual(plan.routing_policy.value, "client_round_robin")
        self.assertEqual(len(plan.endpoints), 2)
        self.assertTrue(plan.runtime_options["warmup_parallel"])

        forbidden = ("--disaggregation-mode", "--pd-disaggregation",
                     "--disaggregation-transfer-backend")
        for server in (f0, f1, a0, a1):
            self.assertTrue(all(flag in server.command for flag in (
                "--tp 8", "--afd-attn-tp 8", "--afd-ffn-tp 8",
                "--afd-comm-backend zmq", "--afd-micro-batch 1",
                "--afd-shared-pool", "--afd-instance-id",
                "--afd-coordinator-endpoint", "--afd-shared-peer-specs",
            )))
            self.assertIn("AFD_CROSS_NODE_EXPERIMENTAL=1", server.command)
            self.assertIn("AFD_LOCAL_TP=8", server.command)
            self.assertIn("AFD_ZMQ_TIMEOUT_MS=300000", server.command)
            self.assertIn("AFD_ZMQ_HANDSHAKE_TIMEOUT_MS=300000", server.command)
            self.assertTrue(all(flag not in server.command for flag in forbidden))
            token = server.command.split("--afd-shared-peer-specs ", 1)[1].split(" >>", 1)[0]
            peers = json.loads(shlex.split(token)[0])
            self.assertEqual(len(peers), 2)
            expected_prefix = "F" if server.role == "A" else "A"
            self.assertEqual({peer["peer_id"] for peer in peers},
                             {f"{expected_prefix}0", f"{expected_prefix}1"})
            if server.role == "A":
                self.assertTrue(all("control_endpoint" in peer for peer in peers))

        edges = plan.manifest()["afd_pool_edges"]
        self.assertEqual(len(edges), 4)
        self.assertEqual({edge["edge_id"] for edge in edges},
                         {"A0-F0", "A0-F1", "A1-F0", "A1-F1"})
        edge_ports = []
        for edge in edges:
            for key in ("ffn_base_port", "attn_base_port",
                        "ffn_handshake_base_port", "attn_handshake_base_port"):
                edge_ports.extend(range(edge[key] + 1, edge[key] + 9))
        self.assertEqual(len(edge_ports), len(set(edge_ports)))
        listener_ports = [port for p in plan.processes for port in p.internal_ports]
        self.assertEqual(len(listener_ports), len(set(listener_ports)))
        for edge in edges:
            fproc = f0 if edge["pf_id"] == "F0" else f1
            aproc = a0 if edge["pa_id"] == "A0" else a1
            self.assertTrue(set(range(edge["attn_base_port"] + 1,
                                          edge["attn_base_port"] + 9)).issubset(fproc.internal_ports))
            self.assertTrue(set(range(edge["ffn_base_port"] + 1,
                                          edge["ffn_base_port"] + 9)).issubset(aproc.internal_ports))

        derived = build_plan(
            b["cluster"], model, dict(point, health_timeout_s=360),
            run_tag="shared-pool-derived-timeout",
        )
        derived_commands = [process.command for process in derived.processes
                            if process.role in ("A", "F")]
        self.assertTrue(all("AFD_ZMQ_TIMEOUT_MS=360000" in command
                            for command in derived_commands))
        self.assertTrue(all("AFD_ZMQ_HANDSHAKE_TIMEOUT_MS=360000" in command
                            for command in derived_commands))

    def test_node34_a8f8_tp8_smoke_then_qps8_snapshot(self):
        import jsonschema

        config_name = "node34_af_a8f8_tp8_smoke_qps8.json"
        config_path = ROOT / "configs" / config_name
        matrix = json.loads(config_path.read_text())
        schema = json.loads((ROOT / "schemas/matrix.schema.json").read_text())
        jsonschema.validate(matrix, schema)

        b = bundle(config_name)
        points = b["matrix"]["points"]
        self.assertEqual(
            [point["id"] for point in points],
            [
                "node34-af-a8f8-tp8-smoke-shared-stage-4req-qps1",
                "node34-af-a8f8-tp8-fixed-qps8",
            ],
        )
        self.assertEqual(
            [(point["workloads"], point["qps"], point["pf_capacity"])
             for point in points],
            [(["shared_stage_4req"], [1], 1),
             (["fixed_tp8_qps8"], [8], 64)],
        )
        self.assertEqual([point["priority"] for point in points], [0, 1])

        all_ports = []
        for index, point in enumerate(points):
            with self.subTest(point=point["id"]):
                self.assertEqual(
                    (point["nodes"], point["recipe"], point["coordinator_node"],
                     point["energy_scope"]),
                    (2, "shared_bipartite_pool", "node3", "plan_used_gpus"),
                )
                self.assertEqual(
                    (point["health_timeout_s"], point["warmup_timeout_s"],
                     point["warmup_curl_timeout_s"], point["request_timeout_s"]),
                    (300, 300, 180, 180),
                )
                model = b["models"]["models"][point["model"]]["path"]
                plan = build_plan(b["cluster"], model, point,
                                  run_tag=f"node34-a8f8-{index}")
                plan.validate()
                self.assertEqual(
                    [(process.role, process.node, process.gpus)
                     for process in plan.processes],
                    [("AF_COORDINATOR", "node3", []),
                     ("F", "node4", list(range(8))),
                     ("A", "node3", list(range(8)))],
                )
                self.assertEqual(plan.used_gpu_map(),
                                 {"node3": list(range(8)),
                                  "node4": list(range(8))})
                self.assertEqual(sum(map(len, plan.used_gpu_map().values())), 16)
                self.assertEqual(plan.endpoints, ["http://10.252.129.34:" +
                                                  str(point["afd_port_base"] - 500)])
                self.assertTrue(plan.endpoint.startswith("http://10.252.129.34:"))
                ports = [(process.host, port) for process in plan.processes
                         for port in ([process.port] + process.internal_ports +
                                      [value for value in
                                       (process.nccl_port, process.bootstrap_port)
                                       if value is not None])]
                self.assertEqual(len(ports), len(set(ports)))
                all_ports.extend(ports)

        self.assertEqual(len(all_ports), len(set(all_ports)))
        self.assertEqual([point["afd_coordinator_port"] for point in points],
                         [19108, 19109])
        queue = expand_queue(b)
        self.assertEqual([item["point"]["id"] for item in queue],
                         [point["id"] for point in points])
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / "must-not-exist"
            for item in queue:
                result = execute_item(item, b, target, dry_run=True)
                self.assertEqual(result["status"], "dry-run")
            self.assertFalse(target.exists())

    def test_node34_a8f8_tp1_shared_pool_snapshot(self):
        import jsonschema

        config_name = "node34_af_a8f8_tp1_shared_smoke_qps8.json"
        matrix = json.loads((ROOT / "configs" / config_name).read_text())
        schema = json.loads((ROOT / "schemas/matrix.schema.json").read_text())
        jsonschema.validate(matrix, schema)

        b = bundle(config_name)
        self.assertEqual(
            b["cluster_source"],
            str(ROOT / "configs/cluster_operator_test.json"),
        )
        points = b["matrix"]["points"]
        self.assertEqual(
            [(p["workloads"], p["qps"], p["pf_capacity"]) for p in points],
            [(["shared_stage_4req"], [1], 1),
             (["fixed_tp8_qps8"], [8], 64)],
        )
        self.assertEqual([p["afd_port_base"] for p in points], [24000, 40000])
        self.assertEqual([p["afd_coordinator_port"] for p in points],
                         [19208, 19209])

        for index, point in enumerate(points):
            with self.subTest(point=point["id"]):
                self.assertEqual(
                    (point["nodes"], point["tp"], point["recipe"],
                     point["coordinator_node"], point["energy_scope"]),
                    (2, 1, "shared_bipartite_pool", "node3",
                     "plan_used_gpus"),
                )
                self.assertEqual(
                    (point["health_timeout_s"], point["warmup_timeout_s"],
                     point["warmup_curl_timeout_s"],
                     point["request_timeout_s"], point["warmup_parallel"],
                     point["wait_after_warmup_s"]),
                    (300, 300, 180, 180, True, 3),
                )
                self.assertTrue(point["experimental"])
                self.assertTrue(point["requires_preflight"])
                self.assertTrue(point["blocked_reason"])
                self.assertEqual(
                    [(item["id"], item["node"], item["gpus"], item["tp"])
                     for item in point["attention_instances"]],
                    [(f"A{i}", "node3", [i], 1) for i in range(8)],
                )
                self.assertEqual(
                    [(item["id"], item["node"], item["gpus"], item["tp"])
                     for item in point["ffn_instances"]],
                    [(f"F{i}", "node4", [i], 1) for i in range(8)],
                )

                model = b["models"]["models"][point["model"]]["path"]
                plan = build_plan(b["cluster"], model, point,
                                  run_tag=f"node34-a8f8-tp1-{index}")
                plan.validate()
                self.assertEqual(len(plan.processes), 17)
                self.assertEqual(
                    [(p.role, p.node, p.gpus) for p in plan.processes],
                    [("AF_COORDINATOR", "node3", [])]
                    + [("F", "node4", [i]) for i in range(8)]
                    + [("A", "node3", [i]) for i in range(8)],
                )
                self.assertEqual(plan.used_gpu_map(),
                                 {"node3": list(range(8)),
                                  "node4": list(range(8))})
                self.assertEqual(len(plan.endpoints), 8)
                self.assertEqual(plan.routing_policy,
                                 RoutingPolicy.CLIENT_ROUND_ROBIN)

                servers = [p for p in plan.processes if p.role in ("A", "F")]
                self.assertTrue(all("--tp 1" in p.command and
                                    "--afd-attn-tp 1" in p.command and
                                    "--afd-ffn-tp 1" in p.command
                                    for p in servers))
                # The runtime compatibility wrapper still exports 8 globally;
                # explicit TP-aware server flags above remain authoritative.
                self.assertTrue(all("AFD_LOCAL_TP=8" in p.command
                                    for p in servers))

                edges = plan.manifest()["afd_pool_edges"]
                self.assertEqual(len(edges), 64)
                self.assertEqual(
                    {edge["edge_id"] for edge in edges},
                    {f"A{a}-F{f}" for a in range(8) for f in range(8)},
                )
                self.assertTrue(all(len(p.metadata["peer_ids"]) == 8
                                    for p in servers))

                ports = [(p.host, port) for p in plan.processes
                         for port in ([p.port] + p.internal_ports
                                      + [value for value in
                                         (p.nccl_port, p.bootstrap_port)
                                         if value is not None])]
                self.assertEqual(len(ports), len(set(ports)))
                self.assertTrue(all(1 <= port <= 65535 for _, port in ports))
                edge_ports = []
                for edge in edges:
                    for key in ("ffn_base_port", "attn_base_port",
                                "ffn_handshake_base_port",
                                "attn_handshake_base_port"):
                        edge_ports.append((edge["pa_host"] if key.startswith("ffn")
                                           else edge["pf_host"], edge[key] + 1))
                self.assertEqual(len(edge_ports), len(set(edge_ports)))
                self.assertTrue(all(1 <= port <= 65535
                                    for _, port in edge_ports))

        workload = resolve_workload_path(
            b["workloads"]["generated_dir"], "fixed_tp8_qps8", 8)
        rows = [json.loads(line) for line in workload.read_text().splitlines()
                if line.strip()]
        self.assertEqual(len(rows), 64)
        self.assertTrue(all((row["input_len"], row["output_len"]) == (128, 64)
                            for row in rows))

        queue = expand_queue(b)
        self.assertEqual(len(queue), 2)
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / "must-not-exist"
            for item in queue:
                result = execute_item(item, b, target, dry_run=True)
                self.assertEqual(result["status"], "dry-run")
            self.assertFalse(target.exists())

    def test_node34_8xa1f1_fixed_qps8_comparison_snapshot(self):
        config_name = "node34_af_8xa1f1_fixed_qps8.json"
        config_path = ROOT / "configs" / config_name
        matrix = json.loads(config_path.read_text())
        validate_matrix(matrix)

        b = bundle(config_name)
        self.assertEqual(b["cluster_source"], str(ROOT / "configs" /
                                                  "cluster_operator_node34.json"))
        self.assertEqual(sum(len(node["gpus"]) for node in b["cluster"]["nodes"]),
                         16)
        point = b["matrix"]["points"][0]
        self.assertEqual(
            (point["id"], point["recipe"], point["pair_tp"],
             point["replicas_per_node"], point["topology"]),
            ("node34-af-8xa1f1-fixed-qps8", "af_node34_a1f1_pool", 1, 4,
             "per_node_replicas"),
        )
        self.assertEqual((point["workloads"], point["qps"]),
                         (["fixed_tp8_qps8"], [8]))
        self.assertEqual(
            (point["request_timeout_s"], point["health_timeout_s"],
             point["warmup_timeout_s"], point["warmup_curl_timeout_s"],
             point["warmup_parallel"], point["wait_after_warmup_s"],
             point["skip_warmup"], point["energy_scope"]),
            (180, 300, 300, 180, True, 3, False, "plan_used_gpus"),
        )
        self.assertTrue(point["multinode_preflight_complete"])
        for absent in ("experimental", "requires_preflight", "blocked_reason",
                       "max_inflight"):
            self.assertNotIn(absent, point)

        workload = resolve_workload_path(
            b["workloads"]["generated_dir"], point["workloads"][0],
            point["qps"][0],
        )
        self.assertEqual(workload,
                         ROOT / "data/workloads/fixed_tp8_qps8.jsonl")
        rows = [json.loads(line) for line in workload.read_text().splitlines()
                if line.strip()]
        self.assertEqual(len(rows), 64)
        self.assertTrue(all((row["input_len"], row["output_len"],
                             row["timeout_s"]) == (128, 64, 180)
                            for row in rows))

        model = b["models"]["models"][point["model"]]["path"]
        plan = build_plan(b["cluster"], model, point,
                          run_tag="node34-8xa1f1-qps8-snapshot")
        plan.validate()
        self.assertEqual(len(plan.processes), 16)
        self.assertEqual([process.role for process in plan.processes],
                         ["F", "A"] * 8)
        self.assertEqual(len(plan.endpoints), 8)
        self.assertEqual(plan.routing_policy,
                         RoutingPolicy.CLIENT_ROUND_ROBIN)
        allocated = [(process.node, gpu) for process in plan.processes
                     for gpu in process.gpus]
        self.assertEqual((len(allocated), len(set(allocated))), (16, 16))
        replicas_by_node = {}
        for process in plan.processes:
            replicas_by_node.setdefault(process.node, set()).add(
                process.metadata["replica"])
        self.assertEqual({node: len(replicas) for node, replicas in
                          replicas_by_node.items()}, {"node3": 4, "node4": 4})

        queue = expand_queue(b)
        self.assertEqual(len(queue), 1)
        self.assertEqual(queue[0]["state"], "ready")
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / "must-not-exist"
            result = execute_item(queue[0], b, target, dry_run=True)
            self.assertEqual(result["status"], "dry-run")
            self.assertFalse(target.exists())

    def test_moe_qwen3_30b_a3b_16gpu_af_six_point_snapshot(self):
        import jsonschema

        config_name = "node34_moe_qwen3_30b_a3b_af_smoke_qps8.json"
        matrix = json.loads((ROOT / "configs" / config_name).read_text())
        schema = json.loads((ROOT / "schemas/matrix.schema.json").read_text())
        jsonschema.validate(matrix, schema)
        validate_matrix(matrix)

        b = bundle(config_name)
        points = b["matrix"]["points"]
        expected_ids = [
            "node34-moe-qwen3-30b-a3b-8xa1f1-smoke-shared-stage-4req-qps1",
            "node34-moe-qwen3-30b-a3b-8xa1f1-fixed-qps8",
            "node34-moe-qwen3-30b-a3b-a8f8-tp8-smoke-shared-stage-4req-qps1",
            "node34-moe-qwen3-30b-a3b-a8f8-tp8-fixed-qps8",
            "node34-moe-qwen3-30b-a3b-a8f8-tp1-smoke-shared-stage-4req-qps1",
            "node34-moe-qwen3-30b-a3b-a8f8-tp1-fixed-qps8",
        ]
        self.assertEqual([point["id"] for point in points], expected_ids)
        self.assertEqual(sum(len(node["gpus"]) for node in b["cluster"]["nodes"]), 16)
        self.assertTrue(all(point["model"] == "moe_qwen3_30b_a3b" for point in points))
        self.assertEqual(
            [(p["workloads"], p["qps"]) for p in points],
            [(["shared_stage_4req"], [1]), (["fixed_tp8_qps8"], [8])] * 3,
        )

        expected_shape = [
            (16, 0, 8), (16, 0, 8),
            (3, 1, 1), (3, 1, 1),
            (17, 64, 8), (17, 64, 8),
        ]
        for index, (point, shape) in enumerate(zip(points, expected_shape)):
            with self.subTest(point=point["id"]):
                self.assertEqual(
                    (point["health_timeout_s"], point["warmup_timeout_s"],
                     point["warmup_curl_timeout_s"], point["request_timeout_s"],
                     point["warmup_parallel"], point["wait_after_warmup_s"],
                     point["energy_scope"]),
                    (300, 300, 180, 180, True, 3, "plan_used_gpus"),
                )
                if point["recipe"] == "af_node34_a1f1_pool":
                    for absent in ("experimental", "requires_preflight", "blocked_reason"):
                        self.assertNotIn(absent, point)
                else:
                    self.assertTrue(point["experimental"])
                    self.assertTrue(point["requires_preflight"])
                    self.assertEqual(point["pf_capacity"], 1 if index % 2 == 0 else 64)

                model = b["models"]["models"][point["model"]]["path"]
                plan = build_plan(b["cluster"], model, point, run_tag=f"moe-six-{index}")
                plan.validate()
                manifest = plan.manifest()
                self.assertEqual(
                    (len(plan.processes), len(manifest.get("afd_pool_edges", [])),
                     len(plan.endpoints)), shape,
                )
                self.assertEqual(sum(map(len, plan.used_gpu_map().values())), 16)
                ports = [(process.host, port) for process in plan.processes
                         for port in ([process.port] + process.internal_ports
                                      + [value for value in
                                         (process.nccl_port, process.bootstrap_port)
                                         if value is not None])]
                self.assertEqual(len(ports), len(set(ports)))
                self.assertTrue(all(1024 < port < 65535 for _, port in ports))
                edges = manifest.get("afd_pool_edges", [])
                edge_listener_ports = []
                for edge in edges:
                    for key in ("ffn_base_port", "attn_base_port",
                                "ffn_handshake_base_port", "attn_handshake_base_port"):
                        host = edge["pa_host"] if key.startswith("ffn") else edge["pf_host"]
                        edge_listener_ports.extend((host, edge[key] + rank)
                                                   for rank in range(1, point["tp"] + 1))
                self.assertEqual(len(edge_listener_ports), len(set(edge_listener_ports)))
                self.assertTrue(all(1024 < port < 65535
                                    for _, port in edge_listener_ports))
                if point.get("tp") == 1:
                    self.assertLess(max(port for _, port in edge_listener_ports), 65535)

        smoke_rows = [json.loads(line) for line in
                      (ROOT / "configs/shared_stage_4req.jsonl").read_text().splitlines()
                      if line.strip()]
        self.assertEqual(len(smoke_rows), 4)
        self.assertTrue(all((row["input_len"], row["output_len"]) == (128, 4)
                            for row in smoke_rows))
        qps8_rows = [json.loads(line) for line in resolve_workload_path(
            b["workloads"]["generated_dir"], "fixed_tp8_qps8", 8
        ).read_text().splitlines() if line.strip()]
        self.assertEqual(len(qps8_rows), 64)
        self.assertTrue(all((row["input_len"], row["output_len"]) == (128, 64)
                            for row in qps8_rows))

        queue = expand_queue(b)
        self.assertEqual([item["point"]["id"] for item in queue], expected_ids)
        self.assertEqual([item["state"] for item in queue],
                         ["ready", "ready"] + ["requires_preflight"] * 4)
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / "must-not-exist"
            for item in queue:
                self.assertEqual(execute_item(item, b, target, dry_run=True)["status"],
                                 "dry-run")
            self.assertFalse(target.exists())

    def test_shared_pool_pf_capacity_defaults_and_instance_override(self):
        b = bundle("shared_af_stage4_2a2f_4gpu.json")
        point = next(p for p in b["matrix"]["points"]
                     if p["recipe"] == "shared_bipartite_pool")
        model = b["models"]["models"][point["model"]]["path"]

        default_plan = build_plan(b["cluster"], model, point,
                                  run_tag="shared-pool-default-capacity")
        default_coordinator = default_plan.processes[0]
        self.assertIn("--pf F0:1 --pf F1:1", default_coordinator.command)
        self.assertEqual(
            default_plan.manifest()["afd_pool_coordinator"]["static_pfs"],
            [{"instance_id": "F0", "capacity": 1},
             {"instance_id": "F1", "capacity": 1}],
        )

        overridden = dict(point, pf_capacity=7)
        overridden["ffn_instances"] = [dict(instance)
                                       for instance in point["ffn_instances"]]
        overridden["ffn_instances"][1]["capacity"] = 11
        override_plan = build_plan(b["cluster"], model, overridden,
                                   run_tag="shared-pool-override-capacity")
        override_coordinator = override_plan.processes[0]
        expected = [{"instance_id": "F0", "capacity": 7},
                    {"instance_id": "F1", "capacity": 11}]
        self.assertIn("--pf F0:7 --pf F1:11", override_coordinator.command)
        self.assertEqual(override_coordinator.metadata["static_pfs"], expected)
        self.assertEqual(
            override_plan.manifest()["afd_pool_coordinator"]["static_pfs"], expected)

    def test_shared_pool_pf_capacity_validation(self):
        valid = {
            "rq": "RQ5", "points": [{
                "id": "af-capacity", "architecture": "af", "nodes": 1,
                "qps": [1], "pf_capacity": 1,
                "ffn_instances": [{"capacity": 2}],
            }],
        }
        validate_matrix(valid)
        for field, value in (("pf_capacity", 0), ("pf_capacity", True)):
            invalid = json.loads(json.dumps(valid))
            invalid["points"][0][field] = value
            with self.assertRaisesRegex(ConfigError, "pf_capacity must be a positive integer"):
                validate_matrix(invalid)
        invalid = json.loads(json.dumps(valid))
        invalid["points"][0]["ffn_instances"][0]["capacity"] = 0
        with self.assertRaisesRegex(
                ConfigError, r"ffn_instances\[0\]\.capacity must be a positive integer"):
            validate_matrix(invalid)

    def test_shared_pool_staged_configs_compile_and_scope_resources(self):
        configs = [
            "shared_af_stage1_2gpu.json", "shared_af_stage2_1a2f_3gpu.json",
            "shared_af_stage3_2a1f_3gpu.json", "shared_af_stage4_2a2f_4gpu.json",
            "shared_af_stage5_cross_tp1_4gpu.json",
            "shared_af_stage6_cross_tp2_8gpu.json",
            "shared_af_stage7_cross_tp4_16gpu.json",
            "shared_af_stage8_cross_tp8_32gpu.json", "qps8_tp8_32gpu.json",
        ]
        expected_gpus = [2, 3, 3, 4, 4, 8, 16, 32, 32]
        expected_edges = [1, 2, 2, 4, 4, 4, 4, 4, 4]
        for index, name in enumerate(configs):
            b = bundle(name)
            point = next(p for p in b["matrix"]["points"]
                         if p["recipe"] == "shared_bipartite_pool")
            model = b["models"]["models"][point["model"]]["path"]
            plan = build_plan(b["cluster"], model, point, run_tag=f"stage-{index + 1}")
            plan.validate()
            server_commands = [process["command"] for process in plan.manifest()["processes"]
                               if process["role"] in ("A", "F")]
            self.assertTrue(server_commands)
            self.assertTrue(all("AFD_ZMQ_TIMEOUT_MS=300000" in command
                                for command in server_commands))
            self.assertTrue(all("AFD_ZMQ_HANDSHAKE_TIMEOUT_MS=300000" in command
                                for command in server_commands))
            self.assertEqual(sum(map(len, plan.used_gpu_map().values())), expected_gpus[index])
            self.assertEqual(len(plan.manifest()["afd_pool_edges"]), expected_edges[index])
            capacities = plan.manifest()["afd_pool_coordinator"]["static_pfs"]
            expected_capacity = 64 if name == "qps8_tp8_32gpu.json" else 1
            self.assertTrue(all(pf["capacity"] == expected_capacity
                                for pf in capacities))
            self.assertEqual(len(plan.endpoints), len(point["attention_instances"]))
            allocated = {(item["node"], gpu)
                         for field in ("attention_instances", "ffn_instances")
                         for item in point[field] for gpu in item["gpus"]}
            scoped = {(node, gpu) for node, gpus in plan.used_gpu_map().items() for gpu in gpus}
            self.assertEqual(scoped, allocated)
            ports = [(proc.host, port) for proc in plan.processes
                     for port in ([proc.port] + proc.internal_ports +
                                  [x for x in (proc.nccl_port, proc.bootstrap_port) if x is not None])]
            self.assertEqual(len(ports), len(set(ports)))
            edges = plan.manifest()["afd_pool_edges"]
            expected = {f"{a['id']}-{f['id']}" for a in point["attention_instances"]
                        for f in point["ffn_instances"]}
            self.assertEqual({edge["edge_id"] for edge in edges}, expected)

        first = bundle(configs[0])
        queue = expand_queue(first)
        self.assertEqual(queue[0]["state"], "ready")
        for name in configs[1:7]:
            self.assertEqual(expand_queue(bundle(name))[0]["state"], "requires_preflight")
            self.assertEqual(expand_queue(bundle(name), allow_experimental=True)[0]["state"], "ready")

    def test_read_cluster_energy_accepts_exact_gpu_scope(self):
        from aflex_benchmark.collect.energy import read_cluster_energy
        class Result:
            stdout = '{"1": 10}\n'
        class Executor:
            def __init__(self): self.calls = []
            def run(self, host, command, check=False):
                self.calls.append((host, command)); return Result()
        b = bundle("shared_af_stage1_2gpu.json")
        executor = Executor()
        value = read_cluster_energy(executor, b["cluster"], 4, {"node1": [1]})
        self.assertEqual(value, {"node1": {"1": 10}})
        self.assertEqual(len(executor.calls), 1)
        self.assertIn("idx=[1]", shlex.split(executor.calls[0][1])[-1])

    def test_qps8_pd_and_pdaf_role_placement_flags_and_stages(self):
        b = bundle("qps8_tp8_32gpu.json")
        nodes = b["cluster"]["nodes"]
        model = b["models"]["models"]["dense_qwen3_32b"]["path"]
        pd_point = next(p for p in b["matrix"]["points"] if p["architecture"] == "pd")
        pd = build_plan(b["cluster"], model, pd_point)
        self.assertEqual([(p.role, p.node, p.startup_stage) for p in pd.processes], [
            ("P", nodes[0]["name"], 0), ("P", nodes[1]["name"], 0),
            ("D", nodes[2]["name"], 1), ("D", nodes[3]["name"], 1),
            ("PD_ROUTER", nodes[0]["name"], 2),
        ])
        self.assertTrue(all("--tp 8" in p.command for p in pd.processes if p.role in {"P", "D"}))
        self.assertEqual(len([token for token in pd.processes[-1].command.split()
                              if token == "--decode"]), 2)

        point = next(p for p in b["matrix"]["points"] if p["architecture"] == "pdaf")
        plan = build_plan(b["cluster"], model, point)
        self.assertEqual([(p.role, p.node, p.startup_stage) for p in plan.processes], [
            ("PF", nodes[1]["name"], 0), ("DF", nodes[3]["name"], 0),
            ("PA", nodes[0]["name"], 1), ("DA", nodes[2]["name"], 1),
            ("PDAF_ROUTER", nodes[0]["name"], 2),
        ])
        servers = plan.processes[:-1]
        self.assertTrue(all("--tp 8" in p.command and "--afd-comm-backend zmq" in p.command
                            for p in servers))
        self.assertTrue(all("AFD_ZMQ_SHARDING=0" in p.command
                            and "AFD_ZMQ_PEER_HOST=" in p.command
                            and "AFD_SCHED_HOST=" in p.command
                            and p.ready_log_patterns == ["AFD ZMQ handshake ready"]
                            and p.ready_timeout_s == 600
                            for p in servers))
        self.assertNotIn("--afd-dvfs-enabled", " ".join(p.command for p in servers))
        self.assertTrue(all("--lock-gpu-clocks=1410,1410" in command
                            for _, command in plan.pre_actions))
        self.assertEqual(len(plan.pre_actions), 32)

    def test_workload_path_prefers_exact_name(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            exact = root / "fixed_tp8_qps8.jsonl"
            exact.write_text("exact\n")
            (root / "fixed_tp8_qps8_qps8.jsonl").write_text("fallback\n")
            self.assertEqual(
                resolve_workload_path(root, "fixed_tp8_qps8", 8), exact
            )

    def test_workload_path_falls_back_to_qps_suffix(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            fallback = root / "fixed_short_qps4.jsonl"
            fallback.write_text("fallback\n")
            self.assertEqual(
                resolve_workload_path(root, "fixed_short", 4), fallback
            )

    def test_workload_path_missing_lists_candidates(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with self.assertRaises(FileNotFoundError) as raised:
                resolve_workload_path(root, "missing", 8)
            message = str(raised.exception)
            self.assertIn(str(root / "missing.jsonl"), message)
            self.assertIn(str(root / "missing_qps8.jsonl"), message)

    def test_fixed_tp8_qps8_workload_and_index(self):
        rows = [json.loads(line) for line in (
            ROOT / "data/workloads/fixed_tp8_qps8.jsonl"
        ).read_text().splitlines() if line.strip()]
        self.assertEqual(len(rows), 64)
        self.assertTrue(all((row["input_len"], row["output_len"], row["timeout_s"])
                            == (128, 64, 180) for row in rows))
        self.assertTrue(all(rows[index]["arrival_time_s"] < rows[index + 1]["arrival_time_s"]
                            for index in range(len(rows) - 1)))
        index = json.loads((ROOT / "data/workloads/index.json").read_text())
        entry = next(item for item in index["entries"]
                     if item["name"] == "fixed_tp8_qps8" and item["qps"] == 8)
        self.assertEqual((entry["requests"], entry["seed"], entry["timeout_s"]),
                         (64, 20260823, 180))

    def test_command_health_supports_non_http_process(self):
        calls = []
        executor = RemoteExecutor(emit=lambda _: None)
        executor._startup_status = lambda spec: subprocess.CompletedProcess([], 0, "", "")
        executor.run = lambda host, cmd, **kwargs: (
            calls.append(cmd) or subprocess.CompletedProcess([], 0, "", ""))
        spec = ProcessSpec(
            "AF_COORDINATOR", "node", "host", [], 19090, "launch",
            health_path=None, ready_command="python3 -c 'check_status()'",
        )
        executor.wait_health(spec, timeout_s=1, interval_s=0)
        self.assertTrue(any("check_status" in command for command in calls))
        self.assertFalse(any("curl" in command for command in calls))

    def test_wait_log_patterns_and_lifecycle_marker_order(self):
        calls = []
        executor = RemoteExecutor(emit=lambda _: None)
        executor._startup_status = lambda spec: subprocess.CompletedProcess([], 0, "", "")
        executor.run = lambda host, cmd, **kwargs: (
            calls.append(cmd) or subprocess.CompletedProcess([], 0, "", ""))
        spec = ProcessSpec("PF", "node", "host", [0], 41000, "cmd",
                           ready_log_patterns=["listener ready", "handshake ready"],
                           ready_timeout_s=17)
        executor.wait_log_patterns(spec, interval_s=0)
        self.assertIn("listener ready", calls[0])
        self.assertIn("handshake ready", calls[0])

        events = []
        class FakeExecutor:
            dry_run = False
            def deep_cleanup(self, plan): pass
            def preclean_plan(self, plan): pass
            def verify_ports_free(self, plan): pass
            def verify_gpu_compute_apps_zero(self, plan): pass
            def launch(self, process): pass
            def verify_alive(self, process): events.append("alive")
            def wait_health(self, process, timeout): events.append("health")
            def wait_log_patterns(self, process): events.append("marker")
            def warmup(self, endpoint, requests, **kwargs): pass
            def collect_logs(self, plan, output): pass
            def cleanup_plan(self, plan): pass
        plan = DeploymentPlan("native", [spec], "http://host:41000")
        execute_lifecycle(plan, FakeExecutor(), lambda _: events.append("body"))
        self.assertEqual(events, ["alive", "health", "marker", "body"])

    def test_legacy_native_snapshot(self):
        _,_,plan=self._legacy_32_plan("native")
        workers=[p for p in plan.processes if p.role=="NATIVE"]
        self.assertEqual(len(workers),4)
        self.assertEqual([p.gpus for p in workers],[list(range(8))]*4)
        self.assertTrue(all("--tp 8" in p.command and p.nccl_port for p in workers))
        self.assertEqual(plan.processes[-1].role,"NATIVE_ROUTER")
        self.assertIn("--policy round_robin",plan.processes[-1].command)
        self.assertEqual({p.startup_stage for p in workers}, {0})
        self.assertEqual(plan.processes[-1].startup_stage, 1)

    def test_legacy_pd_dual_snapshot(self):
        _,_,plan=self._legacy_32_plan("pd")
        self.assertEqual(len([p for p in plan.processes if p.role=="P"]),4)
        self.assertEqual(len([p for p in plan.processes if p.role=="D"]),8)
        self.assertEqual(len([p for p in plan.processes if p.role=="PD_SUBROUTER"]),2)
        self.assertEqual(plan.processes[-1].role,"PD_TOP_ROUTER")
        self.assertTrue(all("--tp 4" in p.command for p in plan.processes if p.role=="P"))
        self.assertTrue(all("--tp 2" in p.command for p in plan.processes if p.role=="D"))
        joined=" ".join(p.command for p in plan.processes)
        self.assertIn("--disaggregation-ib-device mlx5_0",joined)
        self.assertEqual({p.metadata["cluster"] for p in plan.processes if p.role in {"P","D"}},{0,1})
        self.assertEqual({p.startup_stage for p in plan.processes if p.role == "P"}, {0})
        self.assertEqual({p.startup_stage for p in plan.processes if p.role == "D"}, {1})
        self.assertEqual({p.startup_stage for p in plan.processes if p.role == "PD_SUBROUTER"}, {2})
        self.assertEqual(plan.processes[-1].startup_stage, 3)

    def test_legacy_af_profile_snapshot(self):
        _, point, plan = self._legacy_32_plan("af")
        self.assertEqual(point["pair_tp"], 1)
        self.assertEqual(point["replicas_per_node"], 4)
        self.assertEqual([p.role for p in plan.processes], ["F", "A"] * 16)
        self.assertEqual(len(plan.endpoints), 16)
        self.assertEqual(plan.routing_policy, RoutingPolicy.CLIENT_ROUND_ROBIN)
        joined = " ".join(p.command for p in plan.processes)
        self.assertNotIn("--disaggregation-mode", joined)
        self.assertNotIn("--disaggregation-transfer-backend", joined)
        self.assertNotIn("--pd-disaggregation", joined)
        servers = [p for p in plan.processes if p.role in {"F", "A"}]
        self.assertTrue(all("--tp 1" in p.command for p in servers))
        self.assertTrue(all("--afd-ipc-per-rank" not in p.command for p in servers))
        self.assertEqual({p.startup_stage for p in servers if p.role == "F"}, {0})
        self.assertEqual({p.startup_stage for p in servers if p.role == "A"}, {1})
        self.assertNotIn("--tp 2", joined)
        self.assertNotIn("SGLANG_LAYER_PROFILE", joined)
        by_replica = {}
        for process in servers:
            by_replica.setdefault(process.metadata["replica"], []).append(process)
        self.assertEqual(set(by_replica), set(range(16)))
        for rid, pair in by_replica.items():
            self.assertEqual(len(pair), 2)
            f, a = pair
            base = (rid % 4) * 2
            expected_channel = 400 + rid * 10
            self.assertEqual(f.gpus, [base])
            self.assertEqual(a.gpus, [base + 1])
            self.assertIn(f"CUDA_VISIBLE_DEVICES={base},{base + 1}", f.command)
            self.assertIn(f"CUDA_VISIBLE_DEVICES={base},{base + 1}", a.command)
            self.assertIn("--base-gpu-id 0", f.command)
            self.assertIn("--base-gpu-id 1", a.command)
            self.assertIn("AFD_IPC_PEER_OFFSET=1", f.command)
            self.assertIn("AFD_IPC_PEER_OFFSET=-1", a.command)
            self.assertIn(f"AFD_NVML_DEVICE_INDICES={base} AFD_NVML_DEVICE_INDEX={base}", f.command)
            self.assertIn(f"AFD_NVML_DEVICE_INDICES={base + 1} AFD_NVML_DEVICE_INDEX={base + 1}", a.command)
            self.assertEqual({p.metadata["channel_base"] for p in pair}, {expected_channel})
            self.assertEqual({p.metadata["channel_strategy"] for p in pair}, {"shared_tp0"})
            self.assertEqual({p.metadata["channel_id"] for p in pair}, {None})
            self.assertEqual({tuple(p.metadata["channels"]) for p in pair}, {()})
            sched_ports = {
                part.split("=", 1)[1]
                for p in pair for part in p.command.split()
                if part.startswith("AFD_SCHED_PORT=")
            }
            self.assertEqual(len(sched_ports), 1)
            self.assertEqual(len(f.internal_ports), 2)
            self.assertEqual(len(a.internal_ports), 1)
        expected_hosts = [node["host"] for node in bundle("smoke_32gpu.json")["cluster"]["nodes"]]
        self.assertEqual(
            plan.endpoints,
            [f"http://{expected_hosts[rid // 4]}:{40101 + rid * 10}" for rid in range(16)],
        )

    def test_legacy_af_tp2_requires_experimental(self):
        b, point, _ = self._legacy_32_plan("af")
        tp2 = dict(point, pair_tp=2, replicas_per_node=2,
                   attention_gpus=2, ffn_gpus=2)
        tp2.pop("experimental")
        with self.assertRaisesRegex(ValueError, "pair_tp=2 must be experimental"):
            build_plan(b["cluster"], b["models"]["models"][point["model"]]["path"], tp2)
        tp2["experimental"] = True
        plan = build_plan(
            b["cluster"], b["models"]["models"][point["model"]]["path"], tp2
        )
        self.assertEqual(len(plan.endpoints), 8)
        self.assertTrue(all("--tp 2" in p.command for p in plan.processes))

    def test_legacy_af_per_rank_channel_strategy(self):
        b, point, _ = self._legacy_32_plan("af")
        point = dict(point, ipc_channel_strategy="per_rank")
        plan = build_plan(
            b["cluster"], b["models"]["models"][point["model"]]["path"], point
        )
        servers = [p for p in plan.processes if p.role in {"F", "A"}]
        self.assertTrue(all("--afd-ipc-per-rank" in p.command for p in servers))
        by_replica = {}
        for process in servers:
            by_replica.setdefault(process.metadata["replica"], []).append(process)
        for rid, pair in by_replica.items():
            expected = 400 + rid * 10
            self.assertEqual({p.metadata["channel_strategy"] for p in pair},
                             {"per_rank"})
            self.assertEqual({p.metadata["channel_id"] for p in pair}, {expected})
            self.assertEqual({tuple(p.metadata["channels"]) for p in pair},
                             {(expected, expected + 1)})
            sched_ports = {
                part.split("=", 1)[1]
                for p in pair for part in p.command.split()
                if part.startswith("AFD_SCHED_PORT=")
            }
            self.assertEqual(len(sched_ports), 2)

    def test_client_round_robin_is_deterministic_by_request_index(self):
        handle = DeploymentHandle(
            ("http://a:1", "http://b:2", "http://c:3"),
            RoutingPolicy.CLIENT_ROUND_ROBIN,
        )
        self.assertEqual(
            [request_endpoint(handle, index) for index in range(8)],
            ["http://a:1", "http://b:2", "http://c:3", "http://a:1",
             "http://b:2", "http://c:3", "http://a:1", "http://b:2"],
        )
        plan = DeploymentPlan("af", [], list(handle.endpoints),
                              routing_policy=handle.routing_policy)
        self.assertEqual(request_endpoint(plan, 4), "http://b:2")
        self.assertEqual(request_endpoint("http://router:9", 99), "http://router:9")

    def test_legacy_pdaf_3p1d_snapshot(self):
        _,_,plan=self._legacy_32_plan("pdaf")
        for node in ("node1","node2","node3","node4"):
            servers=[p for p in plan.processes if p.node==node and p.role in {"PF","PA","DF","DA"}]
            self.assertEqual([p.role for p in servers],["PF","PA"]*3+["DF","DA"])
            self.assertEqual(len([p for p in plan.processes if p.node==node and p.role=="PDAF_SUBROUTER"]),3)
            self.assertEqual(len([p for p in plan.processes if p.node==node and p.role=="PDAF_NODE_ROUTER"]),1)
        self.assertEqual(plan.processes[-1].role,"PDAF_TOP_ROUTER")
        joined=" ".join(p.command for p in plan.processes)
        self.assertNotIn("AFLEX_DVFS",joined); self.assertNotIn("--afd-dvfs-enabled",joined)
        self.assertIn("--afd-disagg-interleave-poll",joined)
        self.assertIn("--num-reserved-decode-tokens 512",joined)
        self.assertNotIn("SGLANG_LAYER_PROFILE", joined)
        servers = [p for p in plan.processes if p.role in {"PF", "PA", "DF", "DA"}]
        self.assertTrue(all("--afd-ipc-per-rank" not in p.command for p in servers))
        expected_stages = {
            ("prefill", 0, "PF"): 0,
            ("prefill", 0, "PA"): 1,
            ("prefill", 1, "PF"): 2,
            ("prefill", 1, "PA"): 3,
            ("prefill", 2, "PF"): 4,
            ("prefill", 2, "PA"): 5,
            ("decode", 0, "DF"): 6,
            ("decode", 0, "DA"): 7,
        }
        self.assertTrue(all(
            p.startup_stage == expected_stages[
                (p.metadata["phase"], p.metadata["pair"], p.role)
            ]
            for p in servers
        ))
        for stage in range(8):
            staged = [p for p in servers if p.startup_stage == stage]
            self.assertEqual(len(staged), 4)
            self.assertEqual({p.node for p in staged}, {"node1", "node2", "node3", "node4"})
        self.assertEqual({p.startup_stage for p in plan.processes if p.role == "PDAF_SUBROUTER"}, {8})
        self.assertEqual({p.startup_stage for p in plan.processes if p.role == "PDAF_NODE_ROUTER"}, {9})
        self.assertEqual(plan.processes[-1].startup_stage, 10)
        by_pair = {}
        for process in servers:
            key = (process.metadata["node_index"], process.metadata["phase"],
                   process.metadata["pair"])
            by_pair.setdefault(key, []).append(process)
        self.assertEqual(len(by_pair), 16)
        channel_bases = []
        for pair in by_pair.values():
            self.assertEqual(len(pair), 2)
            bases = {p.metadata["channel_base"] for p in pair}
            self.assertEqual(len(bases), 1)
            channel_base = bases.pop()
            self.assertTrue(all(f"AFD_IPC_CHANNEL_BASE={channel_base}" in p.command
                                for p in pair))
            channel_bases.append(channel_base)
        self.assertEqual(len(channel_bases), len(set(channel_bases)))

    def test_unknown_multinode_af_topology_stays_blocked(self):
        for name in ("rq5_32gpu.json", "rq6_scaling.json"):
            b = bundle(name)
            risky = [item for item in expand_queue(b) if item["point"]["architecture"] in {"af", "pdaf"} and item["point"]["nodes"] > 1]
            self.assertTrue(risky)
            self.assertTrue(all(item["state"] == "requires_preflight" for item in risky))

    def test_smoke_resource_shapes(self):
        points = bundle("smoke.json")["matrix"]["points"]
        for point in points:
            with self.subTest(point=point["id"]):
                architecture = point["architecture"]
                if architecture == "native":
                    self.assertEqual((point["tp"], point["replicas"]), (8, 1))
                elif architecture == "pd":
                    self.assertEqual(
                        (point["prefill_replicas"], point["prefill_tp"],
                         point["decode_replicas"], point["decode_tp"]),
                        (1, 4, 1, 4),
                    )
                elif architecture == "af":
                    self.assertEqual(
                        (point["ffn_gpus"], point["attention_gpus"]), (4, 4)
                    )
                else:
                    self.assertEqual(
                        (point["pf_gpus"], point["pa_gpus"],
                         point["df_gpus"], point["da_gpus"]),
                        (2, 2, 2, 2),
                    )

    def test_af_only_ipc_smoke_has_f_before_a_and_no_pd(self):
        b = bundle("smoke.json")
        points = [p for p in b["matrix"]["points"] if p["architecture"] == "af"]
        self.assertEqual(len(points), 2)
        for point in points:
            with self.subTest(point=point["id"]):
                self.assertNotIn("disaggregation_mode", point)
                self.assertFalse(point.get("pd_router"))
                plan = build_plan(
                    b["cluster"],
                    b["models"]["models"][point["model"]]["path"],
                    point,
                )
                joined = " ".join(x.command for x in plan.processes)
                self.assertEqual([x.role for x in plan.processes], ["F", "A"])
                self.assertNotIn("--disaggregation-mode", joined)
                self.assertNotIn("--disaggregation-transfer-backend", joined)
                self.assertNotIn("--pd-disaggregation", joined)
                self.assertNotIn("sglang_router", joined)
                self.assertIn("--afd-comm-backend ipc_cpp", joined)
                self.assertIn("AFD_IPC_PEER_OFFSET=", joined)

    def test_pd_router_bootstrap_and_start_order(self):
        b = bundle("rq6_scaling.json")
        point = next(p for p in b["matrix"]["points"] if p["architecture"] == "pd" and p["nodes"] == 1)
        plan = build_plan(b["cluster"], b["models"]["models"][point["model"]]["path"], point)
        roles = [p.role for p in plan.processes]
        self.assertLess(max(i for i,r in enumerate(roles) if r == "P"), min(i for i,r in enumerate(roles) if r == "D"))
        self.assertEqual(roles[-1], "PD_ROUTER")
        prefills = [p for p in plan.processes if p.role == "P"]
        router = plan.processes[-1].command
        for process in prefills:
            self.assertIn(f"--prefill http://{process.host}:{process.port} {process.bootstrap_port}", router)
            self.assertIn(f"--disaggregation-bootstrap-port {process.bootstrap_port}", process.command)
        decodes = [p for p in plan.processes if p.role == "D"]
        for index, process in enumerate(decodes):
            expected = prefills[index % len(prefills)].bootstrap_port
            self.assertIn(f"--disaggregation-bootstrap-port {expected}", process.command)

    def test_pdaf_bootstrap_peer_pairing_and_order(self):
        b = bundle("rq6_scaling.json")
        point = next(p for p in b["matrix"]["points"] if p["architecture"] == "pdaf" and p["nodes"] == 1)
        plan = build_plan(b["cluster"], b["models"]["models"][point["model"]]["path"], point)
        self.assertEqual([p.role for p in plan.processes], ["PF", "PA", "DF", "DA", "PDAF_ROUTER"])
        joined = " ".join(p.command for p in plan.processes)
        self.assertIn("AFD_IPC_PEER_OFFSET=", joined)
        self.assertIn("--disaggregation-bootstrap-port", joined)
        pa = next(p for p in plan.processes if p.role == "PA")
        self.assertIn(f"--prefill http://{pa.host}:{pa.port} {pa.bootstrap_port}", plan.processes[-1].command)

    def test_all_architectures_disable_flashinfer_version_check(self):
        b = bundle("smoke.json")
        for point in b["matrix"]["points"]:
            with self.subTest(point=point["id"]):
                plan = build_plan(
                    b["cluster"],
                    b["models"]["models"][point["model"]]["path"],
                    point,
                )
                servers = [
                    process for process in plan.processes
                    if "sglang.launch_server" in process.command
                ]
                self.assertTrue(servers)
                self.assertTrue(all(
                    "FLASHINFER_DISABLE_VERSION_CHECK=1" in process.command
                    for process in servers
                ))

    def test_health_polling_is_quiet_and_reports_last_failure(self):
        emitted = []

        class FailingExecutor(RemoteExecutor):
            def __init__(self):
                super().__init__(emit=emitted.append)
                self.quiet_values = []

            def run(self, host, cmd, check=True, quiet=False):
                self.quiet_values.append(quiet)
                return __import__("subprocess").CompletedProcess(
                    ["curl"], 7, "", "connection refused"
                )

        executor = FailingExecutor()
        executor._startup_status = lambda spec: subprocess.CompletedProcess([], 0, "", "")
        executor.verify_port_owner = lambda spec: None
        spec = ProcessSpec("native", "node", "host", [0], 30000, "cmd")
        with self.assertRaisesRegex(TimeoutError, "connection refused"):
            executor.wait_health(spec, timeout_s=0.001, interval_s=0.01)
        self.assertTrue(executor.quiet_values)
        self.assertTrue(all(executor.quiet_values))
        self.assertEqual(len(emitted), 1)
        self.assertIn("HEALTH FAILED", emitted[0])
        self.assertIn("last returncode=7", emitted[0])

    def test_warmup_retries_until_third_attempt_succeeds(self):
        results = [
            subprocess.CompletedProcess([], 1, "workers=0", ""),
            subprocess.CompletedProcess([], 1, "workers=0", ""),
            subprocess.CompletedProcess([], 0, "ok", ""),
        ]
        calls = []
        executor = RemoteExecutor(emit=lambda _: None)

        def run(host, cmd, check=True, quiet=False, timeout=None):
            calls.append((host, cmd, check, quiet, timeout))
            return results.pop(0)

        executor.run = run
        executor.warmup(
            "http://router:30000", timeout_s=1, interval_s=0
        )
        self.assertEqual(len(calls), 3)
        self.assertTrue(all(call[0] == "router" for call in calls))
        self.assertTrue(all(call[2:4] == (False, True) for call in calls))
        self.assertTrue(all(call[4] == 130 for call in calls))
        self.assertIn('test "$status" = 200', calls[0][1])

    def test_warmup_payload_uses_configured_input_length(self):
        calls = []
        executor = RemoteExecutor(emit=lambda _: None)
        executor.run = lambda host, cmd, **kwargs: (
            calls.append(cmd) or subprocess.CompletedProcess([], 0, "ok", "")
        )
        executor.warmup("http://router:30000", input_len=3)
        self.assertIn('"input_ids": [1000, 1000, 1000]', calls[0])
        self.assertNotIn('"input_ids": [1000, 1000, 1000, 1000]', calls[0])

    def test_warmup_command_uses_runtime_timeouts_and_token_limit(self):
        calls = []
        executor = RemoteExecutor(emit=lambda _: None)
        executor.run = lambda host, cmd, **kwargs: (
            calls.append((cmd, kwargs.get("timeout"))) or
            subprocess.CompletedProcess([], 0, "ok", "")
        )
        executor.warmup("http://router:30000", curl_timeout_s=20,
                        max_new_tokens=2)
        self.assertIn("--max-time 20", calls[0][0])
        self.assertIn('"max_new_tokens": 2', calls[0][0])
        self.assertEqual(calls[0][1], 30)

    def test_verify_ports_free_fails_immediately_with_host_and_port(self):
        executor = RemoteExecutor(emit=lambda _: None)
        executor.run = lambda *args, **kwargs: subprocess.CompletedProcess(
            [], 1, "port 41000 busy: LISTEN pid=99", "")
        plan = DeploymentPlan("native", [
            ProcessSpec("native", "n", "host-a", [0], 41000, "cmd")
        ], "http://host-a:41000")
        with self.assertRaisesRegex(RuntimeError, "host-a.*41000"):
            executor.verify_ports_free(plan)

    def test_wait_health_fails_immediately_on_bind_fatal(self):
        executor = RemoteExecutor(emit=lambda _: None)
        executor._startup_status = lambda spec: subprocess.CompletedProcess(
            [], 22, "fatal startup log: Address already in use", "")
        executor.run = lambda *args, **kwargs: self.fail(
            "health curl must not run after a fatal bind log")
        spec = ProcessSpec("F", "node", "host", [0], 41000, "cmd")
        with self.assertRaisesRegex(RuntimeError, "Address already in use"):
            executor.wait_health(spec, timeout_s=90, interval_s=0)

    def test_remote_run_accepts_per_call_timeout(self):
        executor = RemoteExecutor(timeout=60, emit=lambda _: None)
        completed = subprocess.CompletedProcess([], 0, "", "")
        with patch(
            "aflex_benchmark.deploy.base.subprocess.run",
            return_value=completed,
        ) as run:
            self.assertIs(executor.run("host", "true", timeout=150), completed)
        self.assertEqual(run.call_args.kwargs["timeout"], 150)

    def test_warmup_retries_timeout_expired_and_reports_final_timeout(self):
        emitted = []
        calls = []
        executor = RemoteExecutor(emit=emitted.append)

        def run(host, cmd, check=True, quiet=False, timeout=None):
            calls.append(timeout)
            raise subprocess.TimeoutExpired(
                ["curl"], timeout, output=b"partial response"
            )

        executor.run = run
        with patch(
            "aflex_benchmark.deploy.base.time.monotonic",
            side_effect=[0, 0, 181],
        ), patch("aflex_benchmark.deploy.base.time.sleep"), self.assertRaisesRegex(
            TimeoutError, "outer timeout=130.0s"
        ) as raised:
            executor.warmup("http://router:30000")
        self.assertEqual(calls, [130.0, 130.0])
        self.assertIn("partial response", str(raised.exception))
        self.assertEqual(len(emitted), 1)
        self.assertIn("WARMUP FAILED", emitted[0])

    def test_warmup_persistent_failure_reports_last_output(self):
        emitted = []
        outputs = iter(["workers=0", "workers still unavailable"])
        executor = RemoteExecutor(emit=emitted.append)

        def run(host, cmd, check=True, quiet=False, timeout=None):
            return subprocess.CompletedProcess([], 1, next(outputs), "")

        executor.run = run
        with patch(
            "aflex_benchmark.deploy.base.time.monotonic",
            side_effect=[0, 0, 181],
        ), patch("aflex_benchmark.deploy.base.time.sleep"), self.assertRaisesRegex(
            TimeoutError, "workers still unavailable"
        ) as raised:
            executor.warmup("http://router:30000")
        self.assertIn("last returncode=1", str(raised.exception))
        self.assertEqual(len(emitted), 1)
        self.assertIn("WARMUP FAILED", emitted[0])

    def test_native_multiple_replicas_have_router(self):
        b = bundle("rq6_scaling.json")
        point = next(p for p in b["matrix"]["points"] if p["architecture"] == "native" and p["nodes"] == 1)
        plan = build_plan(b["cluster"], b["models"]["models"][point["model"]]["path"], point)
        self.assertEqual(plan.processes[-1].role, "NATIVE_ROUTER")
        self.assertIn("--worker-urls", plan.processes[-1].command)

    def test_lifecycle_health_warmup_cleanup_and_unlock(self):
        events = []
        class FakeExecutor:
            dry_run = False
            def deep_cleanup(self, p): events.append(("deep", p.architecture))
            def preclean_plan(self, p): events.append(("preclean", p.architecture))
            def verify_ports_free(self, p): events.append(("ports-free", p.architecture))
            def verify_gpu_compute_apps_zero(self, p): events.append(("gpu-zero", p.architecture))
            def launch(self, p): events.append(("launch", p.role))
            def verify_alive(self, p): events.append(("alive", p.role))
            def wait_health(self, p, timeout): events.append(("health", p.role))
            def warmup(self, endpoint, requests, **kwargs): events.append(("warmup", endpoint))
            def collect_logs(self, plan, output): events.append(("logs", str(output)))
            def cleanup_plan(self, plan): events.extend(("cleanup", x[0]) for x in plan.cleanup); events.extend(("unlock", x[0]) for x in plan.unlock)
        plan = DeploymentPlan("native", [ProcessSpec("native", "n", "h", [0], 1, "cmd")], "http://h:1", [("h","clean")], [("h","unlock")])
        with tempfile.TemporaryDirectory() as directory:
            result = execute_lifecycle(plan, FakeExecutor(), lambda endpoint: events.append(("body", endpoint)) or 7, Path(directory))
            self.assertEqual(result, 7)
            self.assertTrue((Path(directory) / "deployment_manifest.json").exists())
        self.assertEqual([x[0] for x in events], ["deep","preclean","ports-free","gpu-zero","launch","alive","health","warmup","body","logs","deep","cleanup","unlock","gpu-zero"])

    def test_lifecycle_skip_warmup_runs_only_body_request(self):
        events = []
        class FakeExecutor:
            dry_run = False
            def deep_cleanup(self, plan): pass
            def preclean_plan(self, plan): pass
            def verify_ports_free(self, plan): pass
            def verify_gpu_compute_apps_zero(self, plan): pass
            def launch(self, process): pass
            def verify_alive(self, process): pass
            def wait_health(self, process, timeout): events.append(("health", timeout))
            def warmup(self, endpoint, requests, **kwargs): events.append(("warmup", endpoint))
            def collect_logs(self, plan, output): pass
            def cleanup_plan(self, plan): pass
        process = ProcessSpec(
            "native", "n", "h", [0], 1, "cmd",
            metadata={"activation_warmup": True},
        )
        plan = DeploymentPlan("native", [process], "http://h:1")
        result = execute_lifecycle(
            plan, FakeExecutor(),
            lambda endpoint: events.append(("body", endpoint)) or 7,
            runtime_options={"skip_warmup": True, "health_timeout_s": 90},
        )
        self.assertEqual(result, 7)
        self.assertEqual(events, [("health", 90.0), ("body", "http://h:1")])

    def test_lifecycle_stage_barrier_and_same_stage_parallelism(self):
        events = []
        lock = threading.Lock()
        stage_zero_started = threading.Event()
        release_stage_zero = threading.Event()
        stage_zero_health_started = threading.Event()
        release_stage_zero_health = threading.Event()

        class MockExecutor:
            dry_run = False
            def deep_cleanup(self, plan): pass
            def preclean_plan(self, plan): pass
            def verify_ports_free(self, plan): pass
            def verify_gpu_compute_apps_zero(self, plan): pass
            def launch(self, process):
                with lock:
                    events.append(("launch", process.role))
                    if process.startup_stage == 0 and sum(
                            event[0] == "launch" and event[1] in {"F0", "F1"}
                            for event in events) == 2:
                        stage_zero_started.set()
                if process.startup_stage == 0:
                    release_stage_zero.wait(1)
            def verify_alive(self, process):
                with lock: events.append(("alive", process.role))
            def wait_health(self, process, timeout):
                with lock:
                    events.append(("health", process.role, timeout))
                    if process.startup_stage == 0 and sum(
                            event[0] == "health" and event[1] in {"F0", "F1"}
                            for event in events) == 2:
                        stage_zero_health_started.set()
                if process.startup_stage == 0:
                    release_stage_zero_health.wait(1)
            def warmup(self, endpoint, requests, **kwargs): pass
            def collect_logs(self, plan, output): pass
            def cleanup_plan(self, plan): pass

        processes = [
            ProcessSpec("F0", "n", "h", [0], 1, "cmd", startup_stage=0),
            ProcessSpec("A", "n", "h", [2], 3, "cmd", startup_stage=1),
            ProcessSpec("F1", "n", "h", [1], 2, "cmd", startup_stage=0),
        ]
        plan = DeploymentPlan("native", processes, "http://h:3")
        result = []
        error = []
        thread = threading.Thread(target=lambda: _capture_lifecycle(
            plan, MockExecutor(), result, error))
        thread.start()
        self.assertTrue(stage_zero_started.wait(1), "same-stage launches were serialized")
        self.assertNotIn(("launch", "A"), events)
        release_stage_zero.set()
        self.assertTrue(stage_zero_health_started.wait(1),
                        "same-stage health waits were serialized")
        self.assertNotIn(("launch", "A"), events)
        release_stage_zero_health.set()
        thread.join(2)
        self.assertFalse(thread.is_alive())
        self.assertEqual(error, [])
        self.assertEqual(result, [9])
        a_launch = events.index(("launch", "A"))
        for role in ("F0", "F1"):
            self.assertLess(events.index(("health", role, 37)), a_launch)

    def test_lifecycle_stage_failure_cancels_and_cleans_up(self):
        events = []
        blocker_started = threading.Event()
        release = threading.Event()

        class MockExecutor:
            dry_run = False
            def deep_cleanup(self, plan): events.append("deep")
            def preclean_plan(self, plan): events.append("preclean")
            def verify_ports_free(self, plan): events.append("ports-free")
            def verify_gpu_compute_apps_zero(self, plan): events.append("gpu-zero")
            def launch(self, process):
                events.append(f"launch-{process.role}")
                if process.role == "slow":
                    blocker_started.set()
                    release.wait(1)
                else:
                    blocker_started.wait(1)
                    release.set()
                    raise RuntimeError("launch boom")
            def verify_alive(self, process): events.append(f"alive-{process.role}")
            def wait_health(self, process, timeout): events.append(f"health-{process.role}")
            def warmup(self, endpoint, requests, **kwargs): pass
            def collect_logs(self, plan, output): events.append("logs")
            def cleanup_plan(self, plan): events.append("cleanup")

        plan = DeploymentPlan("native", [
            ProcessSpec("slow", "n", "h", [0], 1, "cmd"),
            ProcessSpec("bad", "n", "h", [1], 2, "cmd"),
            ProcessSpec("later", "n", "h", [2], 3, "cmd", startup_stage=1),
        ], "http://h:3")
        with tempfile.TemporaryDirectory() as directory, self.assertRaisesRegex(
                RuntimeError, "launch boom"):
            execute_lifecycle(plan, MockExecutor(), lambda _: None,
                              Path(directory))
        self.assertNotIn("launch-later", events)
        self.assertNotIn("alive-slow", events)
        self.assertEqual(events[-3:], ["deep", "cleanup", "gpu-zero"])

    def test_lifecycle_warms_all_client_round_robin_endpoints(self):
        warmed = []
        body_targets = []

        class FakeExecutor:
            dry_run = False
            def deep_cleanup(self, plan): pass
            def preclean_plan(self, plan): pass
            def verify_ports_free(self, plan): pass
            def verify_gpu_compute_apps_zero(self, plan): pass
            def launch(self, process): pass
            def verify_alive(self, process): pass
            def wait_health(self, process, timeout): pass
            def warmup(self, endpoint, requests, **kwargs): warmed.append((endpoint, requests))
            def collect_logs(self, plan, output): pass
            def cleanup_plan(self, plan): pass

        processes = [
            ProcessSpec("F", "n", "h", [0], 1, "cmd"),
            ProcessSpec("A", "n", "h", [1], 2, "cmd"),
        ]
        endpoints = ["http://h:2", "http://h:3"]
        plan = DeploymentPlan("af", processes, endpoints,
                              routing_policy=RoutingPolicy.CLIENT_ROUND_ROBIN)
        result = execute_lifecycle(
            plan, FakeExecutor(),
            lambda target: body_targets.append(target) or 11,
            warmup_requests=1,
        )
        self.assertEqual(result, 11)
        self.assertEqual(warmed, [(endpoint, 1) for endpoint in endpoints])
        self.assertEqual(len(body_targets), 1)
        self.assertIsInstance(body_targets[0], DeploymentHandle)
        self.assertEqual(list(body_targets[0].endpoints), endpoints)

    def test_lifecycle_parallel_warmup_barrier_sleep_and_energy_boundary(self):
        events = []
        lock = threading.Lock()
        both_started = threading.Event()
        release = threading.Event()

        class FakeExecutor:
            dry_run = False
            def deep_cleanup(self, plan): pass
            def preclean_plan(self, plan): pass
            def verify_ports_free(self, plan): pass
            def verify_gpu_compute_apps_zero(self, plan): pass
            def launch(self, process): pass
            def verify_alive(self, process): pass
            def wait_health(self, process, timeout): pass
            def warmup(self, endpoint, requests, **kwargs):
                with lock:
                    events.append(("warmup-start", endpoint, kwargs))
                    if sum(event[0] == "warmup-start" for event in events) == 2:
                        both_started.set()
                release.wait(1)
                with lock:
                    events.append(("warmup-done", endpoint))
            def collect_logs(self, plan, output): pass
            def cleanup_plan(self, plan): pass

        endpoints = ["http://h:2", "http://h:3"]
        plan = DeploymentPlan(
            "af",
            [ProcessSpec("F", "n", "h", [0], 1, "cmd"),
             ProcessSpec("A", "n", "h", [1], 2, "cmd")],
            endpoints,
            routing_policy=RoutingPolicy.CLIENT_ROUND_ROBIN,
        )
        result = []
        error = []
        def run():
            try:
                result.append(execute_lifecycle(
                    plan, FakeExecutor(),
                    lambda _: events.append(("energy-body",)) or 17,
                    runtime_options={
                        "warmup_parallel": True,
                        "warmup_input_len": 12,
                        "warmup_max_new_tokens": 2,
                        "wait_after_warmup_s": 3,
                    },
                ))
            except BaseException as exc:
                error.append(exc)

        with patch("aflex_benchmark.deploy.base.time.sleep",
                   side_effect=lambda seconds: events.append(("sleep", seconds))):
            thread = threading.Thread(target=run)
            thread.start()
            self.assertTrue(both_started.wait(1), "endpoint warmups were serialized")
            self.assertNotIn(("energy-body",), events)
            release.set()
            thread.join(2)
        self.assertFalse(thread.is_alive())
        self.assertEqual(error, [])
        self.assertEqual(result, [17])
        starts = [event for event in events if event[0] == "warmup-start"]
        self.assertEqual({event[1] for event in starts}, set(endpoints))
        self.assertTrue(all(event[2]["input_len"] == 12 for event in starts))
        sleep_index = events.index(("sleep", 3.0))
        body_index = events.index(("energy-body",))
        self.assertTrue(all(events.index(("warmup-done", endpoint)) < sleep_index
                            for endpoint in endpoints))
        self.assertLess(sleep_index, body_index)

    def test_lifecycle_cleans_up_after_health_failure(self):
        events = []
        class FakeExecutor:
            dry_run = False
            def deep_cleanup(self,p): events.append("deep")
            def preclean_plan(self,p): events.append("preclean")
            def verify_ports_free(self,p): events.append("ports-free")
            def verify_gpu_compute_apps_zero(self,p): events.append("gpu-zero")
            def launch(self,p): events.append("launch")
            def verify_alive(self,p): events.append("alive")
            def wait_health(self,p,t): raise RuntimeError("not ready")
            def warmup(self,e,r,**kwargs): events.append("warmup")
            def collect_logs(self,p,o): events.append("logs")
            def cleanup_plan(self,p): events.extend(["cleanup","unlock"])
        plan = DeploymentPlan("native", [ProcessSpec("native","n","h",[0],1,"cmd")], "http://h:1")
        with self.assertRaises(RuntimeError):
            execute_lifecycle(plan, FakeExecutor(), lambda _: None)
        self.assertEqual(events, ["deep","preclean","ports-free","gpu-zero","launch","alive","deep","cleanup","unlock","gpu-zero"])

    def test_cluster_override_and_smoke_operator_test_default(self):
        smoke = bundle("smoke_32gpu.json")
        self.assertEqual(smoke["cluster"]["container"], "operator_test")
        self.assertEqual(smoke["cluster"]["container_root"], "/workspace/sglang")
        self.assertIs(smoke["cluster"]["host_network"], True)
        overridden = load_bundle(
            ROOT / "configs", ROOT / "configs" / "smoke_32gpu.json",
            ROOT / "configs" / "cluster.json",
        )
        self.assertEqual(overridden["cluster"]["container"], "moe-energy")

    def test_cleanup_targets_process_groups_with_term_then_kill(self):
        b = bundle("smoke_32gpu.json")
        point = b["matrix"]["points"][0]
        plan = build_plan(b["cluster"], b["models"]["models"][point["model"]]["path"], point)
        command = " ".join(cmd for _, cmd in plan.cleanup)
        self.assertIn("kill -TERM -- -$pid", command)
        self.assertIn("kill -0 -- -$pid", command)
        self.assertIn("kill -KILL -- -$pid", command)
        self.assertLess(command.index("kill -TERM"), command.index("kill -KILL"))

    def test_preclean_includes_http_and_bootstrap_ports(self):
        commands = []
        executor = RemoteExecutor(emit=lambda _: None)
        executor.run = lambda host, cmd, check=True, quiet=False: (
            commands.append((host, cmd)) or subprocess.CompletedProcess([], 0, "", "")
        )
        plan = DeploymentPlan("pd", [
            ProcessSpec("P", "n", "h", [0], 41000, "cmd", bootstrap_port=41001),
            ProcessSpec("D", "n", "h", [1], 41002, "cmd"),
        ], "http://h:41002")
        executor.preclean_plan(plan)
        self.assertEqual(len(commands), 1)
        self.assertIn("rm -rf -- /tmp/aflex_bench/dryrun", commands[0][1])
        self.assertNotIn("/tmp/aflex_bench/run-one", commands[0][1])
        for port in (41000, 41001, 41002):
            self.assertIn(str(port), commands[0][1])
        self.assertIn("fuser ${p}/tcp", commands[0][1])

    def test_host_network_preclean_runs_host_before_container_for_plan_ports(self):
        calls = []
        executor = RemoteExecutor(emit=lambda _: None)
        completed = lambda: subprocess.CompletedProcess([], 0, "", "")
        executor.host_run = lambda host, cmd, check=True, quiet=False, timeout=None: (
            calls.append(("host", host, cmd)) or completed()
        )
        executor.run = lambda host, cmd, check=True, quiet=False: (
            calls.append(("container", host, cmd)) or completed()
        )
        plan = DeploymentPlan("pdaf", [
            ProcessSpec("PF", "n1", "h1", [0], 48000, "cmd",
                        bootstrap_port=48001, nccl_port=48002,
                        internal_ports=[48003, 48004]),
            ProcessSpec("ROUTER", "n1", "h1", [], 48005, "cmd",
                        internal_ports=[48006]),
            ProcessSpec("DA", "n2", "h2", [0], 48100, "cmd",
                        nccl_port=48101, internal_ports=[48102]),
        ], "http://h1:48005", cluster={"host_network": True})

        executor.preclean_plan(plan)

        self.assertEqual([(kind, host) for kind, host, _ in calls], [
            ("host", "h1"), ("container", "h1"),
            ("host", "h2"), ("container", "h2"),
        ])
        expected = {"h1": {48000, 48001, 48002, 48003, 48004, 48005, 48006},
                    "h2": {48100, 48101, 48102}}
        for kind, host, command in calls:
            self.assertIn("kill -TERM", command)
            self.assertIn("kill -KILL", command)
            self.assertLess(command.index("kill -TERM"), command.index("kill -KILL"))
            for port in expected[host]:
                self.assertIn(str(port), command)
            for other_host, ports in expected.items():
                if other_host != host:
                    for port in ports:
                        self.assertNotIn(str(port), command)

    def test_preclean_skips_host_run_when_host_network_is_false(self):
        calls = []
        executor = RemoteExecutor(emit=lambda _: None)
        executor.host_run = lambda *args, **kwargs: calls.append("host")
        executor.run = lambda host, cmd, check=True, quiet=False: (
            calls.append("container") or subprocess.CompletedProcess([], 0, "", "")
        )
        plan = DeploymentPlan(
            "native", [ProcessSpec("native", "n", "h", [0], 41000, "cmd")],
            "http://h:41000", cluster={"host_network": False},
        )

        executor.preclean_plan(plan)

        self.assertEqual(calls, ["container"])

    def test_liveness_is_checked_before_health(self):
        events = []
        class FakeExecutor:
            dry_run = False
            def deep_cleanup(self, p): events.append("deep")
            def preclean_plan(self, p): events.append("preclean")
            def verify_ports_free(self, p): events.append("ports-free")
            def verify_gpu_compute_apps_zero(self, p): events.append("gpu-zero")
            def launch(self, p): events.append("launch")
            def verify_alive(self, p): events.append("alive"); raise RuntimeError("dead")
            def wait_health(self, p, t): events.append("health")
            def warmup(self, e, r, **kwargs): events.append("warmup")
            def collect_logs(self, p, o): events.append("logs")
            def cleanup_plan(self, p): events.append("cleanup")
        plan = DeploymentPlan("native", [ProcessSpec("native", "n", "h", [0], 1, "cmd")], "http://h:1")
        with self.assertRaisesRegex(RuntimeError, "dead"):
            execute_lifecycle(plan, FakeExecutor(), lambda _: None)
        self.assertEqual(events, ["deep", "preclean", "ports-free", "gpu-zero", "launch", "alive", "deep", "cleanup", "gpu-zero"])

    def test_manifest_records_selected_cluster(self):
        b = bundle("smoke_32gpu.json")
        point = b["matrix"]["points"][0]
        plan = build_plan(b["cluster"], b["models"]["models"][point["model"]]["path"], point)
        manifest = plan.manifest()
        self.assertEqual(manifest["container"], "operator_test")
        self.assertEqual(manifest["cluster"]["container_root"], "/workspace/sglang")
        self.assertIs(manifest["cluster"]["host_network"], True)
        self.assertEqual(len(manifest["cluster"]["nodes"]), 4)
        self.assertEqual(manifest["endpoints"], plan.endpoints)
        self.assertEqual(manifest["routing_policy"], "single")
        self.assertIn("startup_stage", manifest["processes"][0])

    def test_point_filter_accepts_repeated_and_comma_separated_ids(self):
        queue = expand_queue(bundle("smoke_32gpu.json"), smoke=True)
        selected = filter_queue_by_point_ids(queue, [
            "smoke-32gpu-qwen3-native-legacy-tp8,smoke-32gpu-qwen3-pd-legacy-dual",
            "smoke-32gpu-qwen3-native-legacy-tp8",
        ])
        self.assertEqual(
            [item["point"]["id"] for item in selected],
            ["smoke-32gpu-qwen3-native-legacy-tp8", "smoke-32gpu-qwen3-pd-legacy-dual"],
        )
        with self.assertRaisesRegex(ValueError, "not present in expanded queue"):
            filter_queue_by_point_ids(queue, ["missing-point"])

    def test_deep_cleanup_runs_on_all_participating_hosts_only_when_configured(self):
        calls = []
        executor = RemoteExecutor(emit=lambda _: None)
        executor.run = lambda host, cmd, check=True, quiet=False: (
            calls.append((host, cmd)) or subprocess.CompletedProcess([], 0, "", "")
        )
        processes = [
            ProcessSpec("P", "n1", "h1", [0], 1, "cmd"),
            ProcessSpec("D", "n2", "h2", [0], 2, "cmd"),
            ProcessSpec("ROUTER", "n1", "h1", [], 3, "cmd"),
        ]
        plan = DeploymentPlan("pd", processes, "http://h1:3", cluster={"host_deep_cleanup_cmd": "cleanup-host"})
        executor.host_run = lambda host, cmd, check=True, quiet=False, timeout=None: (
            calls.append((host, cmd)) or subprocess.CompletedProcess([], 0, "", ""))
        executor.deep_cleanup(plan)
        self.assertEqual([host for host, _ in calls], ["h1", "h2"])
        self.assertTrue(all(cmd == "cleanup-host" for _, cmd in calls))
        calls.clear()
        plan.cluster = {}
        executor.deep_cleanup(plan)
        self.assertEqual(calls, [])

    def test_deep_cleanup_non_strict_reports_failure_and_continues(self):
        messages = []
        executor = RemoteExecutor(emit=messages.append)
        executor.host_run = lambda *args, **kwargs: subprocess.CompletedProcess(
            [], 255, "", ""
        )
        plan = DeploymentPlan(
            "af", [ProcessSpec("F", "n1", "h1", [0], 1, "cmd")],
            "http://h1:1",
            cluster={"host_deep_cleanup_cmd": "cleanup-host",
                     "host_deep_cleanup_strict": False},
        )
        executor.deep_cleanup(plan)
        self.assertEqual(messages, [
            "BEST-EFFORT host deep cleanup failed on h1 (rc=255)"
        ])

    def test_operator_cleanup_is_best_effort_without_host_router_pkill(self):
        cluster = json.loads((ROOT / "configs" / "cluster_operator_test.json").read_text())
        validate_cluster(cluster)
        command = cluster["host_deep_cleanup_cmd"]
        self.assertIs(cluster["host_deep_cleanup_strict"], False)
        self.assertNotIn("pkill", command)
        self.assertTrue(command.endswith("; exit 0"))
        self.assertIn("nvidia-smi --query-compute-apps=pid", command)
        self.assertIn("cleanup_node.sh", command)

    def test_cluster_rejects_non_boolean_cleanup_strictness(self):
        cluster = json.loads((ROOT / "configs" / "cluster_operator_test.json").read_text())
        cluster["host_deep_cleanup_strict"] = "false"
        with self.assertRaisesRegex(ConfigError, "must be a boolean"):
            validate_cluster(cluster)

    def test_gpu_zero_guard_checks_only_planned_gpus(self):
        calls = []
        executor = RemoteExecutor(emit=lambda _: None)
        def run(host, cmd, check=True, quiet=False):
            calls.append((host, cmd, check, quiet))
            return subprocess.CompletedProcess([], 1 if host == "worker-host" else 0,
                                               "GPU 0 busy: 1234\n", "")
        executor.host_run = run
        plan = DeploymentPlan("native", [
            ProcessSpec("native", "worker", "worker-host", [0], 1, "cmd"),
            ProcessSpec("NATIVE_ROUTER", "router", "router-host", [], 2, "cmd"),
        ], "http://router-host:2")
        with self.assertRaisesRegex(RuntimeError, "worker-host"):
            executor.verify_gpu_compute_apps_zero(plan)
        self.assertEqual([call[0] for call in calls], ["worker-host"])
        self.assertIn("nvidia-smi -i $gpu", calls[0][1])
        self.assertIn("for gpu in 0", calls[0][1])

    def test_router_http_cannot_collide_with_bootstrap_on_same_host(self):
        plan = DeploymentPlan("pd", [
            ProcessSpec("P", "n1", "same-host", [0], 41000, "cmd", bootstrap_port=41006),
            ProcessSpec("PD_ROUTER", "n1", "same-host", [], 41006, "cmd"),
        ], "http://same-host:41006")
        with self.assertRaisesRegex(ValueError, "port collision"):
            plan.validate()

    def test_operator_python_source_is_prepended_to_all_commands(self):
        source = "/workspace/moe-tier/python"
        for name in ("smoke_af_minimal.json", "smoke_32gpu.json"):
            b = bundle(name)
            self.assertEqual(b["cluster"]["python_source"], source)
            for point in b["matrix"]["points"]:
                with self.subTest(config=name, point=point["id"]):
                    plan = build_plan(
                        b["cluster"],
                        b["models"]["models"][point["model"]]["path"],
                        point,
                    )
                    expected = f"export PYTHONPATH={source}:${{PYTHONPATH:-}}; "
                    self.assertTrue(plan.processes)
                    self.assertTrue(all(
                        process.command.startswith(expected)
                        for process in plan.processes
                    ))
                    self.assertEqual(plan.manifest()["cluster"]["python_source"], source)

    def test_af_decode4_debug_smoke_compiles_with_safe_env(self):
        b = bundle("smoke_af_decode4_node3.json")
        queue = expand_queue(b, smoke=True)
        self.assertEqual(len(queue), 1)
        item = queue[0]
        point = item["point"]
        self.assertEqual((item["workload"], item["qps"]), ("fixed_af_decode4", 1))
        self.assertEqual(point["extra_env"], {
            "AFD_NULL_DEBUG": "1", "CUDA_LAUNCH_BLOCKING": "1"
        })
        self.assertIs(point["skip_warmup"], True)
        self.assertEqual(point["health_timeout_s"], 90)
        self.assertEqual(point["request_timeout_s"], 30)
        workload = ROOT / "data/workloads/fixed_af_decode4_qps1.jsonl"
        rows = [json.loads(line) for line in workload.read_text().splitlines()]
        self.assertEqual(len(rows), 1)
        self.assertEqual((rows[0]["input_len"], rows[0]["output_len"]), (128, 4))

        plan = build_plan(
            b["cluster"], b["models"]["models"][point["model"]]["path"], point
        )
        self.assertEqual([process.role for process in plan.processes], ["F", "A"])
        for process in plan.processes:
            self.assertIn("AFD_NULL_DEBUG=1", process.command)
            self.assertIn("CUDA_LAUNCH_BLOCKING=1", process.command)

    def test_af_extra_env_rejects_non_dict_and_unknown_variables(self):
        b = bundle("smoke_af_decode4_node3.json")
        model = b["models"]["models"]["dense_qwen3_32b"]["path"]
        point = dict(b["matrix"]["points"][0])
        point["extra_env"] = "AFD_NULL_DEBUG=1"
        with self.assertRaisesRegex(ValueError, "must be a dictionary"):
            build_plan(b["cluster"], model, point)
        point["extra_env"] = {"LD_PRELOAD": "/tmp/not-allowed.so"}
        with self.assertRaisesRegex(ValueError, "unsupported variables: LD_PRELOAD"):
            build_plan(b["cluster"], model, point)

    def test_minimal_af_smoke_compiles_without_execution(self):
        b = bundle("smoke_af_minimal.json")
        self.assertNotIn("fixed_af_minimal", b["workloads"]["classes"])
        self.assertIn("fixed_af_minimal", b["workloads"]["indexed_classes"])
        queue = expand_queue(b, smoke=True)
        self.assertEqual(len(queue), 1)
        item = queue[0]
        self.assertEqual((item["workload"], item["qps"]),
                         ("fixed_af_minimal", 1))
        self.assertEqual(item["point"]["model"], "dense_qwen3_32b")
        self.assertEqual(
            {key: item["point"][key] for key in (
                "skip_warmup", "health_timeout_s", "warmup_timeout_s",
                "warmup_curl_timeout_s", "request_timeout_s",
                "warmup_max_new_tokens",
            )},
            {"skip_warmup": True, "health_timeout_s": 90,
             "warmup_timeout_s": 30, "warmup_curl_timeout_s": 20,
             "request_timeout_s": 60,
             "warmup_max_new_tokens": 2},
        )
        self.assertIs(item["point"]["debug_fast_fail"], True)
        self.assertIn("dense Qwen3", b["matrix"]["description"])
        plan = build_plan(
            b["cluster"],
            b["models"]["models"][item["point"]["model"]]["path"],
            item["point"],
        )
        self.assertEqual([process.role for process in plan.processes], ["F", "A"])
        self.assertEqual([process.gpus for process in plan.processes], [[0], [1]])
        self.assertTrue(all("--tp 1" in process.command for process in plan.processes))
        self.assertTrue(all("--afd-ipc-per-rank" not in process.command
                            for process in plan.processes))
        self.assertEqual(
            {part for process in plan.processes for part in process.command.split()
             if part.startswith("AFD_SCHED_PORT=")},
            {"AFD_SCHED_PORT=68400"},
        )
        self.assertEqual(
            {part for process in plan.processes for part in process.command.split()
             if part.startswith("AFD_IPC_CHANNEL_BASE=")},
            {"AFD_IPC_CHANNEL_BASE=400"},
        )
        self.assertEqual(
            {process.metadata["channel_strategy"] for process in plan.processes},
            {"shared_tp0"},
        )
        expected_source = "export PYTHONPATH=/workspace/moe-tier/python:${PYTHONPATH:-}; "
        self.assertTrue(all(process.command.startswith(expected_source)
                            for process in plan.processes))
        rows = [json.loads(line) for line in (
            ROOT / "data/workloads/fixed_af_minimal_qps1.jsonl"
        ).read_text().splitlines() if line.strip()]
        self.assertEqual(len(rows), 1)
        self.assertEqual((rows[0]["input_len"], rows[0]["output_len"]),
                         (128, 2))
        self.assertEqual(rows[0]["timeout_s"], 60)
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / "does-not-exist"
            result = execute_item(item, b, target, dry_run=True)
            self.assertEqual(result["status"], "dry-run")
            self.assertFalse(target.exists())

    def test_dry_run_execute_item_has_no_side_effect(self):
        b = bundle("smoke.json")
        item = expand_queue(b, smoke=True)[0]
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / "does-not-exist"
            result = execute_item(item, b, target, dry_run=True)
            self.assertEqual(result["status"], "dry-run")
            self.assertFalse(target.exists())


    def test_requires_preflight_execute_item_never_executes(self):
        b = bundle("smoke_32gpu.json")
        item = next(item for item in expand_queue(b, smoke=True)
                    if item["point"]["architecture"] == "af")
        with tempfile.TemporaryDirectory() as directory, \
                patch("aflex_benchmark.runner.build_plan") as build_plan:
            result = execute_item(item, b, Path(directory), dry_run=False)
            self.assertEqual(result["status"], "requires_preflight")
            build_plan.assert_not_called()

    def test_ep_requires_experimental(self):
        with self.assertRaises(ConfigError):
            validate_matrix({"rq":"RQ5","points":[{"architecture":"native","parallelism":"ep","nodes":1,"qps":[1]}]})

    def test_generate_all_25_variants(self):
        with tempfile.TemporaryDirectory() as directory:
            tmp = Path(directory)
            entries = generate(tmp, count=3, trace_dir=tmp/"missing")
            self.assertEqual(len(entries), 25)
            self.assertTrue(all((tmp/e["path"]).exists() for e in entries))
            light = next(e for e in entries
                         if e["name"] == "fixed_qps16_light")
            self.assertEqual((light["qps"], light["requests"]), (16, 3))

    def test_analysis_reports_blocked_statuses(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            statuses = ("complete", "requires_preflight", "blocked", "failed")
            for index, status in enumerate(statuses):
                run = root / str(index)
                run.mkdir()
                summary = {"status": status}
                if status == "complete":
                    summary["point"] = {
                        "rq": "SMOKE", "architecture": "native",
                        "parallelism": "tp", "nodes": 4,
                    }
                (run / "summary.json").write_text(json.dumps(summary))
            report = aggregate(root)
            self.assertEqual(report["blocked"], 2)
            self.assertEqual(report["status_counts"], {
                "blocked": 1, "complete": 1, "failed": 1,
                "requires_preflight": 1,
            })

    def test_request_summary_statuses(self):
        success = {"success": True, "sent_offset_s": 0,
                   "completed_offset_s": 1}
        failure = {"success": False, "sent_offset_s": 0,
                   "completed_offset_s": 1}
        self.assertEqual(summarize_requests([])["status"], "failed")
        self.assertEqual(summarize_requests([failure])["status"], "failed")
        self.assertEqual(
            summarize_requests([success, failure])["status"], "partial"
        )
        self.assertEqual(summarize_requests([success])["status"], "complete")

    def test_repeated_run_id_replaces_only_its_artifact_directory(self):
        class FakePlan:
            runtime_options = {}

        def fake_lifecycle(plan, executor, body, **kwargs):
            return body("http://router:30000")

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            workload = root / "work_qps1.jsonl"
            workload.write_text(json.dumps({"request_id": "request-1", "arrival_time_s": 0}) + "\n")
            item = {"run_id": "same-run", "state": "ready",
                    "point": {"nodes": 1}, "model": {"path": "model"},
                    "workload": "work", "qps": 1}
            b = {"cluster": {"container": "container"},
                 "workloads": {"generated_dir": str(root)}}
            results = root / "results"
            other = results / "other-run"
            other.mkdir(parents=True)
            (other / "keep.txt").write_text("keep")
            patches = (
                patch("aflex_benchmark.runner.build_plan", return_value=FakePlan()),
                patch("aflex_benchmark.runner.RemoteExecutor"),
                patch("aflex_benchmark.runner.execute_lifecycle", side_effect=fake_lifecycle),
                patch("aflex_benchmark.runner.collect_system", return_value={}),
                patch("aflex_benchmark.runner.read_cluster_energy", return_value={}),
                patch("aflex_benchmark.runner.read_gpu_uuids", return_value={}),
                patch("aflex_benchmark.runner.energy_delta", return_value={}),
                patch("aflex_benchmark.runner.stream_request",
                      return_value={"request_id": "request-1", "success": True,
                                    "sent_offset_s": 0, "completed_offset_s": 1}),
            )
            with patches[0], patches[1], patches[2], patches[3], patches[4], patches[5], patches[6], patches[7]:
                execute_item(item, b, results)
                (results / "same-run" / "stale.txt").write_text("stale")
                execute_item(item, b, results)

            rows = [json.loads(line) for line in
                    (results / "same-run" / "requests.jsonl").read_text().splitlines()]
            self.assertEqual([row["request_id"] for row in rows], ["request-1"])
            self.assertFalse((results / "same-run" / "stale.txt").exists())
            self.assertEqual((other / "keep.txt").read_text(), "keep")

    def test_execute_item_rejects_run_id_path_escape(self):
        with tempfile.TemporaryDirectory() as directory:
            item = {"run_id": "../other", "state": "ready"}
            with self.assertRaisesRegex(ValueError, "single safe path component"):
                execute_item(item, {}, Path(directory))

    def test_execute_item_preserves_request_summary_status(self):
        class FakePlan:
            runtime_options = {}
        with tempfile.TemporaryDirectory() as directory, patch(
            "aflex_benchmark.runner.build_plan", return_value=FakePlan()
        ), patch("aflex_benchmark.runner.RemoteExecutor"), patch(
            "aflex_benchmark.runner.execute_lifecycle",
            return_value={"status": "partial", "requests_success": 1,
                          "requests_failed": 1},
        ):
            root = Path(directory)
            (root / "work_qps1.jsonl").write_text(
                json.dumps({"request_id": "unused"}) + "\n"
            )
            item = {"run_id": "run", "state": "ready",
                    "point": {"nodes": 1}, "model": {"path": "model"},
                    "workload": "work", "qps": 1}
            b = {"cluster": {"container": "container"},
                 "workloads": {"generated_dir": str(root)}}
            summary = execute_item(item, b, root / "results")
        self.assertEqual(summary["status"], "partial")

    def test_request_energy_and_throughput_fields(self):
        rows = [{"success":True,"input_tokens":5,"completion_tokens":3,"sent_offset_s":0,
                 "completed_offset_s":1,"ttft_client_ms":10,"ttft_server_ms":8,
                 "tpot_ms":5,"e2e_ms":20,"itl_ms":[4,6]}]
        summary = summarize_requests(rows, {"total_j":12})
        self.assertEqual(summary["input_throughput_tokens_s"], 5)
        self.assertEqual(summary["output_throughput_tokens_s"], 3)
        self.assertEqual(summary["all_throughput_tokens_s"], 8)
        self.assertEqual(summary["achieved_qps"], 1)
        self.assertEqual(summary["energy_per_output_token_j"], 4)
        self.assertEqual(summary["average_cluster_power_w"], 12)
        self.assertAlmostEqual(summary["qps_per_w"], 1/12)


    def _node34_plan(self, architecture):
        b = bundle("smoke_16gpu_node34.json")
        point = next(p for p in b["matrix"]["points"]
                     if p["architecture"] == architecture)
        model = b["models"]["models"][point["model"]]["path"]
        return b, point, build_plan(b["cluster"], model, point)

    def test_node34_cluster_is_strictly_isolated(self):
        b = bundle("smoke_16gpu_node34.json")
        cluster = b["cluster"]
        self.assertEqual([n["host"] for n in cluster["nodes"]],
                         ["10.252.129.34", "10.252.129.33"])
        self.assertEqual(sum(len(n["gpus"]) for n in cluster["nodes"]), 16)
        self.assertEqual(cluster["container"], "operator_test")
        self.assertIs(cluster["host_network"], True)
        self.assertEqual(cluster["python_source"], "/workspace/moe-tier/python")
        serialized = json.dumps(cluster)
        self.assertNotIn("10.252.129.35", serialized)
        self.assertNotIn("10.252.129.36", serialized)

    def test_node34_16gpu_recipe_snapshots(self):
        expected_recipes = {
            "native": "legacy_native_pair_tp8",
            "pd": "legacy_pd_xnode_16",
            "af": "af_node34_a1f1_pool",
            "pdaf": "legacy_pdaf_xnode_tp4",
        }
        b = bundle("smoke_16gpu_node34.json")
        self.assertEqual({p["architecture"]: p["recipe"]
                          for p in b["matrix"]["points"]}, expected_recipes)
        states = {item["point"]["architecture"]: item["state"]
                  for item in expand_queue(b, smoke=True)}
        self.assertEqual(states, {"native": "ready", "pd": "ready",
                                  "af": "ready", "pdaf": "ready"})
        unlocked = {item["point"]["architecture"]: item["state"]
                    for item in expand_queue(b, smoke=True,
                                             allow_experimental=True)}
        self.assertEqual(unlocked["af"], "ready")
        points = {point["architecture"]: point
                  for point in b["matrix"]["points"]}
        self.assertEqual(
            (points["af"]["skip_warmup"],
             points["af"]["request_timeout_s"],
             points["af"]["multinode_preflight_complete"],
             points["af"]["preflight_artifact"]),
            (True, 90, True,
             "results/node34_af_eight_replicas/a07927517bec675e/"),
        )
        self.assertEqual(
            (points["pdaf"]["skip_warmup"],
             points["pdaf"]["request_timeout_s"]),
            (False, 120),
        )
        for architecture in expected_recipes:
            with self.subTest(architecture=architecture):
                _, point, plan = self._node34_plan(architecture)
                allocated = [(process.node, gpu) for process in plan.processes
                             for gpu in process.gpus]
                self.assertEqual((len(allocated), len(set(allocated))), (16, 16))
                self.assertEqual(set(process.host for process in plan.processes),
                                 {"10.252.129.34", "10.252.129.33"})
                manifest = json.dumps(plan.manifest())
                self.assertNotIn("10.252.129.35", manifest)
                self.assertNotIn("10.252.129.36", manifest)
                self.assertEqual(point["workloads"], ["fixed_short"])
                self.assertEqual(point["qps"], [1])

        _, _, native = self._node34_plan("native")
        self.assertEqual([p.role for p in native.processes],
                         ["NATIVE", "NATIVE", "NATIVE_ROUTER"])
        self.assertEqual([p.gpus for p in native.processes[:2]],
                         [list(range(8)), list(range(8))])
        self.assertTrue(all("--tp 8" in p.command for p in native.processes[:2]))
        self.assertEqual(native.processes[-1].host, "10.252.129.34")
        self.assertIn("--policy round_robin", native.processes[-1].command)

        _, _, pd = self._node34_plan("pd")
        self.assertEqual([p.role for p in pd.processes],
                         ["P", "P", "D", "D", "D", "D", "PD_ROUTER"])
        self.assertTrue(all(p.host == "10.252.129.34" and "--tp 4" in p.command
                            for p in pd.processes if p.role == "P"))
        self.assertTrue(all(p.host == "10.252.129.33" and "--tp 2" in p.command
                            for p in pd.processes if p.role == "D"))
        self.assertEqual([p.startup_stage for p in pd.processes],
                         [0, 0, 1, 1, 1, 1, 2])
        joined = " ".join(p.command for p in pd.processes)
        for flag in ("--disaggregation-transfer-backend mooncake",
                     "--disaggregation-bootstrap-port", "--disaggregation-ib-device",
                     "--nccl-port"):
            self.assertIn(flag, joined)
        self.assertEqual(pd.processes[-1].host, "10.252.129.34")

        _, _, af = self._node34_plan("af")
        self.assertEqual([p.role for p in af.processes], ["F", "A"] * 8)
        self.assertEqual(len(af.endpoints), 8)
        self.assertEqual(af.routing_policy, RoutingPolicy.CLIENT_ROUND_ROBIN)
        self.assertEqual([p.gpus for p in af.processes],
                         [[i] for _ in range(2) for i in range(8)])

        _, _, pdaf = self._node34_plan("pdaf")
        self.assertEqual([p.role for p in pdaf.processes],
                         ["PF", "PA", "DF", "DA", "PDAF_ROUTER"])
        self.assertEqual([p.startup_stage for p in pdaf.processes], [0, 1, 1, 2, 3])
        by_role = {p.role: p for p in pdaf.processes}
        self.assertEqual(by_role["PF"].gpus, [1, 3, 5, 7])
        self.assertEqual(by_role["PA"].gpus, [0, 2, 4, 6])
        self.assertEqual(by_role["DF"].gpus, [1, 3, 5, 7])
        self.assertEqual(by_role["DA"].gpus, [0, 2, 4, 6])
        joined = " ".join(p.command for p in pdaf.processes)
        for flag in ("--tp 4", "--gpu-id-step 2", "--base-gpu-id 1",
                     "--base-gpu-id 0", "AFD_IPC_PEER_OFFSET=-1",
                     "AFD_IPC_PEER_OFFSET=1", "--disaggregation-ib-device mlx5_bond_0",
                     "--afd-disagg-interleave-poll", "--num-reserved-decode-tokens 512"):
            self.assertIn(flag, joined)
        self.assertNotIn("--afd-dvfs-enabled", joined)
        self.assertNotIn("AFLEX_DVFS", joined)
        self.assertEqual(pdaf.processes[-1].host, "10.252.129.34")

    def test_node34_fixed_short_uses_point_request_timeout(self):
        b = bundle("smoke_16gpu_node34.json")
        item = next(item for item in expand_queue(
            b, smoke=True, allow_experimental=True
        ) if item["point"]["architecture"] == "pdaf")

        class FakePlan:
            runtime_options = {}

        def fake_lifecycle(plan, executor, body, **kwargs):
            body("http://router:30000")
            return {"status": "complete"}

        with tempfile.TemporaryDirectory() as directory, patch(
            "aflex_benchmark.runner.build_plan", return_value=FakePlan()
        ), patch("aflex_benchmark.runner.RemoteExecutor"), patch(
            "aflex_benchmark.runner.execute_lifecycle", side_effect=fake_lifecycle
        ), patch("aflex_benchmark.runner.collect_system", return_value={}), patch(
            "aflex_benchmark.runner.read_cluster_energy", return_value={}
        ), patch("aflex_benchmark.runner.energy_delta", return_value={}), patch(
            "aflex_benchmark.runner.stream_request",
            return_value={"success": True, "sent_offset_s": 0,
                          "completed_offset_s": 1},
        ) as request:
            execute_item(item, b, Path(directory))

        self.assertTrue(request.call_args_list)
        self.assertTrue(all(call.args[3] == 120
                            for call in request.call_args_list))


    def test_node34_qps16_light_workload_shape_and_rate(self):
        path = ROOT / "data/workloads/fixed_qps16_light_qps16.jsonl"
        rows = [json.loads(line) for line in path.read_text().splitlines()
                if line.strip()]
        self.assertEqual(len(rows), 64)
        self.assertTrue(all(
            (row["input_len"], row["output_len"], row["timeout_s"])
            == (128, 64, 60) for row in rows
        ))
        arrivals = [row["arrival_time_s"] for row in rows]
        self.assertTrue(all(b > a for a, b in zip(arrivals, arrivals[1:])))
        self.assertGreater(arrivals[0], 0, "open-loop schedule must not burst at t=0")
        realized_qps = len(rows) / arrivals[-1]
        self.assertAlmostEqual(realized_qps, 16, delta=4)
        self.assertGreater(len(set(round(b - a, 6)
                                   for a, b in zip(arrivals, arrivals[1:]))), 20)

    def test_node34_qps16_matrix_four_points_ready(self):
        b = bundle("bench_16gpu_node34_qps16.json")
        points = b["matrix"]["points"]
        self.assertEqual(len(points), 4)
        self.assertEqual({p["architecture"] for p in points},
                         {"native", "pd", "af", "pdaf"})
        self.assertTrue(all(p["workloads"] == ["fixed_qps16_light"]
                            and p["qps"] == [16]
                            and p["max_inflight"] == 16 for p in points))
        self.assertEqual({item["state"] for item in expand_queue(b, smoke=True)},
                         {"ready"})
        by_arch = {p["architecture"]: p for p in points}
        self.assertEqual(by_arch["native"]["request_timeout_s"], 60)
        self.assertEqual(by_arch["pd"]["request_timeout_s"], 60)
        self.assertEqual(
            (by_arch["af"]["skip_warmup"],
             by_arch["af"]["request_timeout_s"],
             by_arch["af"]["multinode_preflight_complete"]),
            (True, 60, True),
        )
        self.assertEqual(
            (by_arch["pdaf"]["skip_warmup"],
             by_arch["pdaf"]["request_timeout_s"]),
            (False, 90),
        )
        for point in points:
            plan = build_plan(b["cluster"],
                              b["models"]["models"][point["model"]]["path"],
                              point)
            plan.validate()
            allocated = [(process.node, gpu) for process in plan.processes
                         for gpu in process.gpus]
            self.assertEqual((len(allocated), len(set(allocated))), (16, 16))

    def test_runner_max_inflight_caps_fake_stream_at_16(self):
        b = bundle("bench_16gpu_node34_qps16.json")
        item = next(item for item in expand_queue(b, smoke=True)
                    if item["point"]["architecture"] == "native")
        active = 0
        maximum = 0
        lock = threading.Lock()

        class FakePlan:
            runtime_options = {}

        def fake_stream(endpoint, request, start, timeout):
            nonlocal active, maximum
            with lock:
                active += 1
                maximum = max(maximum, active)
            time.sleep(0.02)
            with lock:
                active -= 1
            scheduled = request["arrival_time_s"]
            return {"request_id": request["request_id"], "success": True,
                    "scheduled_arrival_s": scheduled,
                    "sent_offset_s": scheduled, "arrival_lag_ms": 0,
                    "completed_offset_s": scheduled + 0.02}

        def fake_lifecycle(plan, executor, body, **kwargs):
            return body("http://router:30000")

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            workload = root / "fixed_qps16_light_qps16.jsonl"
            workload.write_text("".join(
                json.dumps({"request_id": f"fake-{index}",
                            "arrival_time_s": 0, "input_len": 128,
                            "output_len": 64, "timeout_s": 60}) + "\n"
                for index in range(64)
            ))
            b["workloads"]["generated_dir"] = str(root)
            with patch(
            "aflex_benchmark.runner.build_plan", return_value=FakePlan()
        ), patch("aflex_benchmark.runner.RemoteExecutor"), patch(
            "aflex_benchmark.runner.execute_lifecycle", side_effect=fake_lifecycle
        ), patch("aflex_benchmark.runner.collect_system", return_value={}), patch(
            "aflex_benchmark.runner.read_cluster_energy", return_value={}
        ), patch("aflex_benchmark.runner.energy_delta", return_value={}), patch(
            "aflex_benchmark.runner.stream_request", side_effect=fake_stream
        ):
                summary = execute_item(item, b, root / "results")
        self.assertEqual(summary["status"], "complete")
        self.assertLessEqual(maximum, 16)
        self.assertEqual(maximum, 16)

    def test_node34_recipes_accept_arbitrary_node_names(self):
        b = bundle("smoke_16gpu_node34.json")
        cluster = json.loads(json.dumps(b["cluster"]))
        cluster["nodes"][0]["name"] = "prefill-owner"
        cluster["nodes"][1]["name"] = "decode-worker"
        for point in b["matrix"]["points"]:
            with self.subTest(recipe=point["recipe"]):
                model = b["models"]["models"][point["model"]]["path"]
                plan = build_plan(cluster, model, point)
                allocated = [(p.node, gpu) for p in plan.processes for gpu in p.gpus]
                self.assertEqual((len(allocated), len(set(allocated))), (16, 16))

    def test_node34_minimal_af_smoke_snapshot(self):
        b = bundle("smoke_af_minimal_node34.json")
        point = b["matrix"]["points"][0]
        plan = build_plan(b["cluster"], b["models"]["models"][point["model"]]["path"], point)
        self.assertEqual([p.role for p in plan.processes], ["F", "A"])
        self.assertTrue(all(p.host == "10.252.129.34" for p in plan.processes))
        self.assertEqual([(p.node, p.gpus) for p in plan.processes],
                         [("node3", [0]), ("node3", [1])])
        self.assertEqual(point["workloads"], ["fixed_af_minimal"])
        self.assertEqual(
            (point["skip_warmup"], point["health_timeout_s"],
             point["request_timeout_s"]),
            (True, 90, 60),
        )

    def test_node34_cli_dry_run_never_mentions_old_nodes(self):
        command = [sys.executable, str(ROOT / "scripts" / "run_benchmark.py"),
                   "--matrix", str(ROOT / "configs" / "smoke_16gpu_node34.json"),
                   "--smoke", "--dry-run"]
        result = subprocess.run(command, cwd=ROOT, text=True, capture_output=True,
                                check=True)
        output = result.stdout + result.stderr
        self.assertNotIn("10.252.129.35", output)
        self.assertNotIn("10.252.129.36", output)
        self.assertNotIn("node1", output)
        self.assertNotIn("node2", output)

    def test_af_replica_staircase_compiles_without_execution(self):
        cases = (
            ("smoke_af_four_replicas_node3.json", "fixed_af_4replicas", 4,
             1, {"10.252.129.34"}),
            ("smoke_af_eight_replicas_node34.json", "fixed_af_8replicas", 8,
             2, {"10.252.129.34", "10.252.129.33"}),
        )
        expected_source = (
            "export PYTHONPATH=/workspace/moe-tier/python:${PYTHONPATH:-}; "
        )
        for config, workload, replicas, nodes, hosts in cases:
            with self.subTest(config=config):
                b = bundle(config)
                point = b["matrix"]["points"][0]
                self.assertIs(point.get("experimental", False), False)
                if replicas == 8:
                    self.assertTrue(point["multinode_preflight_complete"])
                    self.assertEqual(
                        point["preflight_artifact"],
                        "results/node34_af_eight_replicas/a07927517bec675e/",
                    )
                self.assertIs(point["debug_fast_fail"], True)
                self.assertIs(point["skip_warmup"], True)
                self.assertEqual(point["request_timeout_s"], 60)
                self.assertEqual((point["nodes"], point["workloads"],
                                  point["qps"]),
                                 (nodes, [workload], [1]))
                plan = build_plan(
                    b["cluster"],
                    b["models"]["models"][point["model"]]["path"],
                    point,
                )
                plan.validate()
                self.assertEqual(len(plan.processes), replicas * 2)
                self.assertEqual(len(plan.endpoints), replicas)
                self.assertEqual(plan.routing_policy,
                                 RoutingPolicy.CLIENT_ROUND_ROBIN)
                allocated = [(process.node, gpu) for process in plan.processes
                             for gpu in process.gpus]
                self.assertEqual((len(allocated), len(set(allocated))),
                                 (replicas * 2, replicas * 2))
                self.assertEqual({process.host for process in plan.processes},
                                 hosts)
                self.assertTrue(all(process.command.startswith(expected_source)
                                    for process in plan.processes))
                rows = [json.loads(line) for line in (
                    ROOT / "data/workloads" / f"{workload}_qps1.jsonl"
                ).read_text().splitlines() if line.strip()]
                self.assertEqual(len(rows), replicas)
                self.assertTrue(all(
                    (row["input_len"], row["output_len"], row["timeout_s"])
                    == (128, 2, 60) for row in rows
                ))
                queue = expand_queue(b, smoke=True)
                self.assertEqual(len(queue), 1)
                with tempfile.TemporaryDirectory() as directory:
                    target = Path(directory) / "does-not-exist"
                    result = execute_item(queue[0], b, target, dry_run=True)
                    self.assertEqual(result["status"], "dry-run")
                    self.assertFalse(target.exists())

    def test_af_replica_staircase_preflight_order(self):
        four = expand_queue(
            bundle("smoke_af_four_replicas_node3.json"), smoke=True
        )[0]
        eight = expand_queue(
            bundle("smoke_af_eight_replicas_node34.json"), smoke=True
        )[0]
        self.assertEqual(four["state"], "ready")
        self.assertEqual(eight["state"], "ready")
        self.assertTrue(eight["point"]["multinode_preflight_complete"])
        self.assertEqual(
            eight["point"]["preflight_artifact"],
            "results/node34_af_eight_replicas/a07927517bec675e/",
        )
        self.assertLess(four["point"]["priority"],
                        eight["point"]["priority"])

    def test_legacy_tier1_code_qps16_snapshot(self):
        b = bundle("reproduce_tier1_code_qps16_node34.json")
        point = b["matrix"]["points"][0]
        plan = build_plan(
            b["cluster"], b["models"]["models"][point["model"]]["path"], point
        )
        plan.validate()
        self.assertEqual(point["workloads"], ["historical_code"])
        self.assertNotIn("max_inflight", point)
        self.assertEqual(point["request_timeout_s"], 180)
        workload = Path(point["workload_path"])
        self.assertTrue(workload.is_absolute())
        self.assertEqual(sum(1 for line in workload.read_text().splitlines()
                             if line.strip()), 800)

        servers = [p for p in plan.processes if p.role in {"PF", "PA", "DF", "DA"}]
        self.assertEqual([p.role for p in servers], ["PF", "PA"] * 7 + ["DF", "DA"])
        self.assertEqual(
            [(p.node, p.role, p.gpus) for p in servers],
            [("node3", "PF", [2]), ("node3", "PA", [3]),
             ("node3", "PF", [4]), ("node3", "PA", [5]),
             ("node3", "PF", [6]), ("node3", "PA", [7]),
             ("node4", "PF", [0]), ("node4", "PA", [1]),
             ("node4", "PF", [2]), ("node4", "PA", [3]),
             ("node4", "PF", [4]), ("node4", "PA", [5]),
             ("node4", "PF", [6]), ("node4", "PA", [7]),
             ("node3", "DF", [0]), ("node3", "DA", [1])],
        )
        allocated = [(p.node, gpu) for p in servers for gpu in p.gpus]
        self.assertEqual((len(allocated), len(set(allocated))), (16, 16))
        self.assertEqual({p.startup_stage for p in servers if p.role == "PF"}, {0})
        self.assertEqual({p.startup_stage for p in servers if p.role in {"PA", "DF"}}, {1})
        self.assertEqual({p.startup_stage for p in servers if p.role == "DA"}, {2})

        routers = [p for p in plan.processes if p.role == "PDAF_SUBROUTER"]
        self.assertEqual(len(routers), 7)
        self.assertEqual([p.port for p in routers], list(range(45000, 45007)))
        self.assertEqual(plan.endpoints,
                         [f"http://10.252.129.34:{p}" for p in range(45000, 45007)])
        self.assertEqual(plan.routing_policy, RoutingPolicy.CLIENT_ROUND_ROBIN)
        self.assertNotIn("TOP_ROUTER", " ".join(p.role for p in plan.processes))
        self.assertTrue(all("--decode http://10.252.129.34:43020" in p.command
                            for p in routers))

        joined = " ".join(p.command for p in servers)
        for flag in (
            "--afd-comm-backend ipc_cpp", "--afd-micro-batch 1",
            "--max-running-requests 64", "--afd-attn-tp 1", "--afd-ffn-tp 1",
            "--disaggregation-transfer-backend mooncake",
            "--disaggregation-ib-device /tmp/ib_scal_map.json",
            "--afd-dvfs-enabled", "Qwen3-32B/models_v1",
            "--afd-dvfs-decode-compositional", "--afd-dvfs-idle-lock",
            "AFD_IPC_PEER_DEVICE=", "AFD_UCX_TLS=rc,tcp,cuda_copy,cuda_ipc",
        ):
            self.assertIn(flag, joined)
        self.assertTrue(all("--mem-fraction-static 0.75" in p.command
                            for p in servers if p.role in {"PF", "PA"}))
        self.assertTrue(all("--mem-fraction-static 0.88" in p.command
                            for p in servers if p.role in {"DF", "DA"}))
        self.assertTrue(all("CUDA_VISIBLE_DEVICES=" in p.command for p in servers))
        manifest = json.dumps(plan.manifest())
        self.assertNotIn("10.252.129.35", manifest)
        self.assertNotIn("10.252.129.36", manifest)
        self.assertNotIn("node1", manifest)
        self.assertNotIn("node2", manifest)
        self.assertEqual(len(plan.pre_actions), 18)
        self.assertEqual(sum("ib_scal_map.json" in cmd for _, cmd in plan.pre_actions), 2)
        locks = [(host, cmd) for host, cmd in plan.pre_actions if "--lock-gpu-clocks" in cmd]
        self.assertEqual(len(locks), 16)
        self.assertEqual(sum("1410,1410" in cmd for _, cmd in locks), 14)
        self.assertEqual(sum("930,930" in cmd for _, cmd in locks), 2)
        meta = plan.cluster["legacy_tier1"]
        self.assertEqual((meta["k_p"], meta["k_d"]), (7, 1))
        self.assertEqual(set(meta["freq_maps"]), {"10.252.129.34", "10.252.129.33"})

    def test_legacy_tier1_runner_prefers_absolute_workload_without_semaphore(self):
        b = bundle("reproduce_tier1_code_qps16_node34.json")
        item = expand_queue(b, smoke=True)[0]
        self.assertEqual(item["state"], "ready")
        point = item["point"]
        self.assertNotIn("max_inflight", point)
        self.assertEqual(Path(point["workload_path"]).name,
                         "macro_code_qps16.jsonl")

    def test_historical_requests_are_normalized_without_mutating_trace(self):
        historical = [
            {"arrival_time_s": 0.1, "input_len": 128, "output_len": 32},
            {"arrival_time_s": 0.2, "input_len": 256, "output_len": 64,
             "source": "legacy"},
            {"request_id": "kept-id", "arrival_time_s": 0.3,
             "input_len": 512, "output_len": 16, "timeout_s": 45},
        ]
        original = json.loads(json.dumps(historical))

        normalized = normalize_requests(
            historical, "historical_code", 16, timeout_s=180
        )

        self.assertEqual(historical, original)
        self.assertEqual(
            [row["request_id"] for row in normalized],
            ["historical_code-16-00000", "historical_code-16-00001", "kept-id"],
        )
        self.assertEqual([row["timeout_s"] for row in normalized], [180, 180, 45])
        self.assertEqual(normalized[1]["source"], "legacy")

    def test_historical_request_normalization_is_stable(self):
        requests = [{"input_len": 128, "output_len": 32},
                    {"request_id": "existing", "input_len": 64,
                     "output_len": 8}]
        first = normalize_requests(requests, "historical_code", 16, 180)
        second = normalize_requests(requests, "historical_code", 16, 180)
        self.assertEqual(first, second)
        self.assertEqual(second[1]["request_id"], "existing")

    def test_execute_item_distinguishes_deployment_and_client_failures(self):
        class FakePlan:
            runtime_options = {}

        item = {"run_id": "run", "state": "ready", "point": {"nodes": 1},
                "model": {"path": "model"}, "workload": "work", "qps": 1}
        b = {"cluster": {"container": "container"},
             "workloads": {"generated_dir": "unused"}}
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "work_qps1.jsonl").write_text(
                json.dumps({"input_len": 1, "output_len": 1}) + "\n"
            )
            b["workloads"]["generated_dir"] = str(root)
            with patch("aflex_benchmark.runner.build_plan", return_value=FakePlan()), patch(
                "aflex_benchmark.runner.RemoteExecutor"
            ), patch("aflex_benchmark.runner.execute_lifecycle",
                     side_effect=RuntimeError("launch failed")):
                deployment = execute_item(item, b, root / "deployment")

            def fail_in_body(plan, executor, body, **kwargs):
                return body("http://router:30000")

            with patch("aflex_benchmark.runner.build_plan", return_value=FakePlan()), patch(
                "aflex_benchmark.runner.RemoteExecutor"
            ), patch("aflex_benchmark.runner.execute_lifecycle",
                     side_effect=fail_in_body), patch(
                "aflex_benchmark.runner.collect_system", return_value={}
            ), patch("aflex_benchmark.runner.read_cluster_energy", return_value={}), patch(
                "aflex_benchmark.runner.stream_request",
                side_effect=KeyError("unsupported_client_field"),
            ):
                client = execute_item(item, b, root / "client")

        self.assertEqual(deployment["failure_stage"], "deployment")
        self.assertEqual(client["failure_stage"], "client_compatibility")

    def _smoke_32gpu_light_bundle(self):
        return bundle("smoke_32gpu_light.json")

    def test_32gpu_light_four_points_are_ready_without_old_blocks(self):
        b = self._smoke_32gpu_light_bundle()
        points = b["matrix"]["points"]
        self.assertEqual(len(points), 4)
        self.assertEqual({p["architecture"] for p in points},
                         {"native", "pd", "af", "pdaf"})
        self.assertEqual({item["state"] for item in expand_queue(b, smoke=True)},
                         {"ready"})
        for point in points:
            for old_field in ("experimental", "requires_preflight",
                              "blocked", "blocked_reason", "artifact_paths"):
                self.assertNotIn(old_field, point)
        self.assertNotIn("extra_env",
                         next(p for p in points if p["architecture"] == "af"))

    def test_32gpu_light_uses_operator_cluster_and_all_32_gpus(self):
        b = self._smoke_32gpu_light_bundle()
        self.assertEqual(b["cluster"]["container"], "operator_test")
        self.assertEqual(len(b["cluster"]["nodes"]), 4)
        for point in b["matrix"]["points"]:
            with self.subTest(point=point["id"]):
                plan = build_plan(
                    b["cluster"], b["models"]["models"][point["model"]]["path"],
                    point,
                )
                plan.validate()
                allocated = [(process.node, gpu) for process in plan.processes
                             for gpu in process.gpus]
                self.assertEqual((len(allocated), len(set(allocated))), (32, 32))

    def test_32gpu_light_shared_workload_has_eight_poisson_requests(self):
        b = self._smoke_32gpu_light_bundle()
        self.assertIn("fixed_smoke32_light", b["workloads"]["indexed_classes"])
        rows = [json.loads(line) for line in (
            ROOT / "data/workloads/fixed_smoke32_light_qps1.jsonl"
        ).read_text().splitlines() if line.strip()]
        self.assertEqual(len(rows), 8)
        self.assertTrue(all((row["input_len"], row["output_len"],
                             row["timeout_s"]) == (128, 4, 60)
                            for row in rows))
        arrivals = [row["arrival_time_s"] for row in rows]
        self.assertTrue(all(b > a for a, b in zip(arrivals, arrivals[1:])))
        intervals = [arrivals[0]] + [b - a for a, b in zip(arrivals, arrivals[1:])]
        self.assertGreater(len({round(value, 6) for value in intervals}), 1)
        index = json.loads((ROOT / "data/workloads/index.json").read_text())
        entry = next(e for e in index["entries"]
                     if e["name"] == "fixed_smoke32_light" and e["qps"] == 1)
        self.assertEqual(entry["requests"], 8)

    def test_32gpu_light_runtime_options_and_dry_run_compile(self):
        b = self._smoke_32gpu_light_bundle()
        expected_ids = [
            "smoke-32gpu-light-qwen3-native-legacy-tp8",
            "smoke-32gpu-light-qwen3-pd-legacy-dual",
            "smoke-32gpu-light-qwen3-af-legacy-16xa1f1",
            "smoke-32gpu-light-qwen3-pdaf-legacy-3p1d",
        ]
        points = b["matrix"]["points"]
        self.assertEqual([p["id"] for p in points], expected_ids)
        expected_recipes = {
            "native": "legacy_native_tp", "pd": "legacy_pd_dual",
            "af": "legacy_af_profile_replicas", "pdaf": "legacy_pdaf_3p1d",
        }
        self.assertEqual({p["architecture"]: p["recipe"] for p in points},
                         expected_recipes)
        for point in points:
            self.assertEqual((point["workloads"], point["qps"],
                              point["health_timeout_s"],
                              point["request_timeout_s"], point["max_inflight"]),
                             (["fixed_smoke32_light"], [1], 600, 60, 8))
            self.assertEqual(point["skip_warmup"], point["architecture"] == "af")
        queue = expand_queue(b, smoke=True)
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / "does-not-exist"
            for item in queue:
                result = execute_item(item, b, target, dry_run=True)
                self.assertEqual(result["status"], "dry-run")
            self.assertFalse(target.exists())

    def test_32gpu_light_warm_af_config_is_ready_and_dry_run_only(self):
        b = bundle("smoke_32gpu_light_warm.json")
        self.assertEqual(len(b["cluster"]["nodes"]), 4)
        queue = expand_queue(b, smoke=True)
        self.assertEqual(len(queue), 1)
        item = queue[0]
        point = item["point"]
        self.assertEqual(item["state"], "ready")
        self.assertEqual(point["id"],
                         "smoke-32gpu-light-warm-qwen3-af-legacy-16xa1f1")
        self.assertEqual((point["architecture"], point["recipe"], point["workloads"]),
                         ("af", "legacy_af_profile_replicas",
                          ["fixed_smoke32_light"]))
        self.assertEqual(
            {key: point[key] for key in (
                "skip_warmup", "warmup_parallel", "warmup_input_len",
                "warmup_max_new_tokens", "warmup_timeout_s",
                "warmup_curl_timeout_s", "wait_after_warmup_s",
                "request_timeout_s", "max_inflight",
            )},
            {"skip_warmup": False, "warmup_parallel": True,
             "warmup_input_len": 8, "warmup_max_new_tokens": 2,
             "warmup_timeout_s": 120, "warmup_curl_timeout_s": 90,
             "wait_after_warmup_s": 3, "request_timeout_s": 60,
             "max_inflight": 8},
        )
        plan = build_plan(
            b["cluster"], b["models"]["models"][point["model"]]["path"], point
        )
        plan.validate()
        allocated = [(process.node, gpu) for process in plan.processes
                     for gpu in process.gpus]
        self.assertEqual((len(allocated), len(set(allocated))), (32, 32))
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / "must-not-exist"
            result = execute_item(item, b, target, dry_run=True)
            self.assertEqual(result["status"], "dry-run")
            self.assertFalse(target.exists())

    def test_comm_ablation_matrix_recipes_compile_without_gpu_execution(self):
        b = bundle("comm_ablation_qwen3_30b_a3b_2gpu_qps2.json")
        points = b["matrix"]["points"]
        import jsonschema
        jsonschema.validate(
            b["matrix"], json.loads((ROOT / "schemas/matrix.schema.json").read_text())
        )
        self.assertEqual(len(points), 9)
        self.assertEqual(
            {(p["architecture"], p["expected_link"]) for p in points},
            {(a, link) for a in ("native", "pd", "af")
             for link in ("nvlink", "pcie_host_staged", "rdma")},
        )
        self.assertEqual(len({p["id"] for p in points}), 9)
        self.assertEqual(len({p["port_base"] for p in points}), 9)
        for point in points:
            with self.subTest(point=point["id"]):
                self.assertEqual((point["model"], point["qps"], point["gpus_total"]),
                                 ("moe_qwen3_30b_a3b", [2], 2))
                plan = build_plan(b["cluster"], b["models"]["models"][point["model"]]["path"], point)
                plan.validate()
                allocated = [(process.node, gpu) for process in plan.processes
                             for gpu in process.gpus]
                self.assertEqual((len(allocated), len(set(allocated))), (2, 2))
                server_processes = [process for process in plan.processes if process.gpus]
                self.assertTrue(all(process.metadata["expected_link"] == point["expected_link"]
                                    for process in server_processes))
                self.assertEqual(plan.used_gpu_map(),
                    {"node3": [0], "node4": [0]} if point["expected_link"] == "rdma"
                    else {"node3": [0, 1]})
        queue = expand_queue(b, smoke=True)
        self.assertEqual(len(queue), 9)
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / "must-not-exist"
            for item in queue:
                self.assertEqual(execute_item(item, b, target, dry_run=True)["status"], "dry-run")
            self.assertFalse(target.exists())

    def test_comm_ablation_smoke_preserves_deployment_snapshot_and_is_9_ready(self):
        source = bundle("comm_ablation_qwen3_30b_a3b_2gpu_qps2.json")
        smoke = bundle("comm_ablation_qwen3_30b_a3b_2gpu_smoke.json")
        source_points = source["matrix"]["points"]
        smoke_points = smoke["matrix"]["points"]
        self.assertEqual(len(smoke_points), 9)
        workload_path = ROOT / "configs/shared_stage_4req.jsonl"
        self.assertEqual(len([line for line in workload_path.read_text().splitlines()
                              if line.strip()]), 4)
        for source_point, smoke_point in zip(source_points, smoke_points):
            with self.subTest(point=smoke_point["id"]):
                self.assertEqual(smoke_point["id"], source_point["id"] + "-smoke")
                self.assertEqual(
                    {key: smoke_point[key] for key in (
                        "model", "architecture", "recipe", "nodes", "gpus_total",
                        "expected_link", "port_base",
                    )},
                    {key: source_point[key] for key in (
                        "model", "architecture", "recipe", "nodes", "gpus_total",
                        "expected_link", "port_base",
                    )},
                )
                self.assertEqual(
                    (smoke_point["workloads"], smoke_point["qps"],
                     smoke_point["max_inflight"], smoke_point["request_timeout_s"],
                     Path(smoke_point["workload_path"])),
                    (["comm_smoke"], [2], 4, 180, workload_path),
                )
                source_plan = build_plan(
                    source["cluster"],
                    source["models"]["models"][source_point["model"]]["path"],
                    source_point,
                )
                smoke_plan = build_plan(
                    smoke["cluster"],
                    smoke["models"]["models"][smoke_point["model"]]["path"],
                    smoke_point,
                )
                self.assertEqual(
                    [(p.role, p.node, p.gpus, p.port, p.command, p.metadata)
                     for p in smoke_plan.processes],
                    [(p.role, p.node, p.gpus, p.port, p.command, p.metadata)
                     for p in source_plan.processes],
                )
                self.assertTrue(smoke_point["expected_link"])
        queue = expand_queue(smoke, smoke=True)
        self.assertEqual(len(queue), 9)
        self.assertEqual({item["state"] for item in queue}, {"ready"})
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / "must-not-exist"
            for item in queue:
                result = execute_item(item, smoke, target, dry_run=True)
                self.assertEqual(result["status"], "dry-run")
            self.assertFalse(target.exists())

    def test_comm_ablation_workload_is_64x_128_64_poisson_qps2(self):
        rows = [json.loads(line) for line in (
            ROOT / "data/workloads/fixed_comm_ablation_qps2.jsonl"
        ).read_text().splitlines() if line.strip()]
        self.assertEqual(len(rows), 64)
        self.assertTrue(all((row["input_len"], row["output_len"], row["source"])
                            == (128, 64, "fixed_comm_ablation") for row in rows))
        arrivals = [row["arrival_time_s"] for row in rows]
        self.assertTrue(all(right > left for left, right in zip(arrivals, arrivals[1:])))
        intervals = [arrivals[0]] + [right - left for left, right in zip(arrivals, arrivals[1:])]
        self.assertGreater(len({round(value, 6) for value in intervals}), 1)
        self.assertAlmostEqual(len(rows) / arrivals[-1], 2.0, delta=0.75)

    def test_comm_ablation_transport_flags_are_explicit(self):
        b = bundle("comm_ablation_qwen3_30b_a3b_2gpu_qps2.json")
        plans = {}
        for point in b["matrix"]["points"]:
            plans[(point["architecture"], point["expected_link"])] = build_plan(
                b["cluster"], b["models"]["models"][point["model"]]["path"], point)
        native_nv = " ".join(p.command for p in plans[("native", "nvlink")].processes)
        native_pcie = " ".join(p.command for p in plans[("native", "pcie_host_staged")].processes)
        native_rdma = " ".join(p.command for p in plans[("native", "rdma")].processes)
        self.assertIn("NCCL_P2P_DISABLE=0", native_nv)
        self.assertIn("NCCL_NVLS_ENABLE=0", native_nv)
        self.assertIn("--disable-custom-all-reduce", native_nv)
        self.assertIn("NCCL_P2P_DISABLE=1", native_pcie)
        self.assertIn("NCCL_SHM_DISABLE=0", native_pcie)
        self.assertIn("NCCL_NVLS_ENABLE=0", native_pcie)
        self.assertIn("--disable-custom-all-reduce", native_pcie)
        self.assertIn("NCCL_IB_DISABLE=0", native_rdma)
        self.assertIn("--nnodes 2", native_rdma)
        for command in (native_nv, native_pcie, native_rdma):
            self.assertIn("NCCL_DEBUG=INFO", command)
            self.assertIn("NCCL_DEBUG_SUBSYS=INIT,GRAPH,P2P,SHM,NET", command)
            self.assertIn("NCCL_NVLS_ENABLE=0", command)
            self.assertIn(">> /tmp/aflex_bench/dryrun/", command)
            self.assertIn("2>&1", command)
            self.assertNotIn("NCCL_DEBUG_FILE", command)
        for link, protocol in (("nvlink", "tcp"), ("pcie_host_staged", "tcp"), ("rdma", "rdma")):
            plan = plans[("pd", link)]
            joined = " ".join(p.command for p in plan.processes)
            self.assertIn("--disaggregation-transfer-backend mooncake", joined)
            self.assertIn(f"MOONCAKE_PROTOCOL={protocol}", joined)
            expected_transport = (
                "mooncake_nvlink" if link == "nvlink" else f"mooncake_{protocol}"
            )
            self.assertTrue(
                all(
                    p.metadata["transport"] == expected_transport
                    for p in plan.processes
                )
            )
        pd_nvlink = " ".join(p.command for p in plans[("pd", "nvlink")].processes)
        self.assertIn("SGLANG_MOONCAKE_CUSTOM_MEM_POOL=NVLINK", pd_nvlink)
        self.assertIn("MC_FORCE_MNNVL=true", pd_nvlink)
        self.assertIn("--log-level debug", pd_nvlink)
        self.assertIn("unset MC_FORCE_TCP MOONCAKE_USE_CUDA_IPC", pd_nvlink)
        self.assertNotIn("MOONCAKE_USE_CUDA_IPC=1", pd_nvlink)
        pd_pcie_plan = plans[("pd", "pcie_host_staged")]
        pd_pcie = " ".join(p.command for p in pd_pcie_plan.processes)
        for process in (p for p in pd_pcie_plan.processes if p.role in {"P", "D"}):
            self.assertIn("MC_FORCE_TCP=1", process.command)
            self.assertIn("MC_LOG_LEVEL=INFO", process.command)
            self.assertIn("MOONCAKE_USE_CUDA_IPC=0", process.command)
            self.assertIn(
                "unset MC_FORCE_HCA MC_FORCE_MNNVL MC_INTRANODE_NVLINK "
                "MC_INTRA_NVLINK SGLANG_MOONCAKE_CUSTOM_MEM_POOL",
                process.command,
            )
        self.assertNotIn("MC_FORCE_MNNVL=true", pd_pcie)
        self.assertTrue(
            all(p.metadata["transport"] == "mooncake_tcp" for p in pd_pcie_plan.processes)
        )
        pd_rdma = " ".join(p.command for p in plans[("pd", "rdma")].processes)
        self.assertIn("MOONCAKE_USE_CUDA_IPC=0", pd_rdma)
        self.assertIn(
            "unset SGLANG_MOONCAKE_CUSTOM_MEM_POOL MC_FORCE_MNNVL MC_FORCE_TCP",
            pd_rdma,
        )
        self.assertNotIn("MC_FORCE_MNNVL=true", pd_rdma)
        for link, backend in (("nvlink", "ipc_cpp"), ("pcie_host_staged", "zmq"), ("rdma", "ucx")):
            plan = plans[("af", link)]
            joined = " ".join(p.command for p in plan.processes)
            self.assertIn(f"--afd-comm-backend {backend}", joined)
            f, a = (next(p for p in plan.processes if p.role == role)
                    for role in ("F", "A"))
            if link == "rdma":
                self.assertEqual((f.gpus, a.gpus), ([0], [0]))
                self.assertIn("CUDA_VISIBLE_DEVICES=0", f.command)
                self.assertIn("CUDA_VISIBLE_DEVICES=0", a.command)
                self.assertIn("--base-gpu-id 0", f.command)
                self.assertIn("--base-gpu-id 0", a.command)
                self.assertNotIn("AFD_IPC_PEER_OFFSET", joined)
            else:
                self.assertEqual((f.gpus, a.gpus), ([0], [1]))
                self.assertIn("CUDA_VISIBLE_DEVICES=0,1", f.command)
                self.assertIn("CUDA_VISIBLE_DEVICES=0,1", a.command)
                self.assertIn("--base-gpu-id 0", f.command)
                self.assertIn("--base-gpu-id 1", a.command)
                if link == "nvlink":
                    self.assertIn("AFD_IPC_PEER_OFFSET=1", f.command)
                    self.assertIn("AFD_IPC_PEER_OFFSET=-1", a.command)
        af_pcie_plan = plans[("af", "pcie_host_staged")]
        af_pcie_commands = " ".join(p.command for p in af_pcie_plan.processes)
        for env in (
            "AFD_CROSS_NODE_EXPERIMENTAL=1",
            "AFD_LOCAL_TP=1",
            "AFD_ZMQ_TIMEOUT_MS=300000",
            "AFD_ZMQ_HANDSHAKE_TIMEOUT_MS=300000",
            "AFD_ZMQ_PEER_HOST=127.0.0.1",
            "AFD_ZMQ_HOST_STAGING=1",
        ):
            self.assertTrue(
                all(env in process.command for process in af_pcie_plan.processes), env
            )
        self.assertEqual(
            {
                process.role: process.internal_ports
                for process in af_pcie_plan.processes
            },
            {"F": [54102, 54141, 54161], "A": [54131, 54151]},
        )
        self.assertIn("AFD_FFN_BASE_PORT=54130", af_pcie_commands)
        self.assertIn("AFD_ATTN_BASE_PORT=54140", af_pcie_commands)
        self.assertIn("AFD_ZMQ_FFN_HANDSHAKE_BASE_PORT=54150", af_pcie_commands)
        self.assertIn("AFD_ZMQ_ATTN_HANDSHAKE_BASE_PORT=54160", af_pcie_commands)
        af_rdma_plan = plans[("af", "rdma")]
        af_rdma_commands = " ".join(p.command for p in af_rdma_plan.processes)
        for env in (
            "UCX_TLS=rc,tcp,cuda_copy",
            "AFD_UCX_TLS=rc,tcp,cuda_copy",
            "UCX_SOCKADDR_TLS_PRIORITY=rdmacm,tcp",
            "UCX_LOG_LEVEL=info",
            "UCX_PROTO_INFO=y",
            "AFD_UCX_GPU_DIRECT=1",
        ):
            self.assertTrue(
                all(env in process.command for process in af_rdma_plan.processes), env
            )
        self.assertNotIn("cuda_ipc", af_rdma_commands)
        self.assertIn("UCX_NET_DEVICES=mlx5_0:1", af_rdma_commands)
        self.assertEqual(af_rdma_commands.count("UCX_NET_DEVICES=mlx5_0:1"), 2)
        loader_env = (
            "LD_LIBRARY_PATH=/usr/local/lib/python3.10/dist-packages/libucx/lib:"
            "/usr/local/lib:${LD_LIBRARY_PATH:-}"
        )
        self.assertTrue(
            all("export SGLANG_DISABLE_REQUEST_LOGGING=true " in p.command
                and loader_env in p.command for p in af_rdma_plan.processes)
        )
        self.assertNotIn("'${LD_LIBRARY_PATH:-}'", af_rdma_commands)
        self.assertEqual(
            af_rdma_plan.pre_actions,
            [
                ("10.252.129.33", f"export {loader_env}; /usr/bin/python3 -c 'import ucp'"),
                ("10.252.129.34", f"export {loader_env}; /usr/bin/python3 -c 'import ucp'"),
            ],
        )
        for link in ("nvlink", "pcie_host_staged"):
            self.assertEqual(plans[("af", link)].pre_actions, [])

    def test_schemas_are_json(self):
        for path in (ROOT / "schemas").glob("*.json"):
            self.assertTrue(json.loads(path.read_text())["$schema"])


class RunScopedLifecycleTests(unittest.TestCase):
    def test_two_runs_do_not_share_remote_paths(self):
        bundle = globals()["bundle"]("smoke.json")
        point = bundle["matrix"]["points"][0]
        model = bundle["models"]["models"][point["model"]]["path"]
        first = build_plan(bundle["cluster"], model, point, run_tag="run-one")
        second = build_plan(bundle["cluster"], model, point, run_tag="run-two")
        self.assertTrue(all("/tmp/aflex_bench/run-one/" in p.log_path for p in first.processes))
        self.assertTrue(all("/tmp/aflex_bench/run-two/" in p.pid_path for p in second.processes))
        self.assertTrue(set(p.log_path for p in first.processes).isdisjoint(
            p.log_path for p in second.processes))

    def test_manifest_and_commands_include_run_paths_and_fingerprint(self):
        bundle = globals()["bundle"]("smoke.json")
        point = bundle["matrix"]["points"][0]
        model = bundle["models"]["models"][point["model"]]["path"]
        plan = build_plan(bundle["cluster"], model, point, run_tag="fingerprint-test")
        manifest = plan.manifest()
        self.assertEqual(manifest["run_tag"], "fingerprint-test")
        for process in plan.processes:
            self.assertEqual(manifest["processes"][plan.processes.index(process)]["log_path"], process.log_path)
            self.assertIn("AFLEX_SOURCE sglang_file", process.command)
            self.assertIn("qwen3_sha256", process.command)
            self.assertIn("qwen3_mtime", process.command)
            self.assertIn("qwen3_marker", process.command)
            self.assertIn(">>", process.command)
            self.assertNotIn(f"/tmp/aflex_{process.port}.log", process.command)

    def test_collect_logs_reads_only_spec_path(self):
        spec = ProcessSpec("native", "node", "host", [0], 30000, "cmd",
                           log_path="/tmp/aflex_bench/current/node_native_30000.log",
                           pid_path="/tmp/aflex_bench/current/node_native_30000.pid")
        plan = DeploymentPlan("native", [spec], "http://host:30000", run_tag="current")
        commands = []
        executor = RemoteExecutor(dry_run=False)
        executor.run = lambda host, cmd, **kwargs: (
            commands.append(cmd) or subprocess.CompletedProcess([], 0, "new-log", ""))
        with tempfile.TemporaryDirectory() as directory:
            executor.collect_logs(plan, Path(directory))
            self.assertEqual((Path(directory) / "00_node_native_30000.log").read_text(), "new-log")
        self.assertIn(spec.log_path, commands[0])
        self.assertNotIn("/tmp/aflex_30000.log", commands[0])


if __name__ == "__main__":
    unittest.main()

import json
import sys
import tempfile
import threading
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from aflex_benchmark.config import load_bundle
from aflex_benchmark.sweep import (_run_point, generate_poisson_workload,
                                   point_wall_limit, run_af_qps_sweep,
                                   sweep_run_tag)


def bundle():
    return load_bundle(ROOT / "configs",
                       ROOT / "configs/sweep_af_qps1_16_node34.json")


class AFQpsSweepTests(unittest.TestCase):
    def test_sweep_run_tags_are_unique_and_path_safe(self):
        point = {"id": "af-sweep"}
        first = sweep_run_tag(point)
        second = sweep_run_tag(point)
        self.assertNotEqual(first, second)
        self.assertRegex(first, r"^sweep-[0-9T]+-[0-9a-f]{8}$")

    def test_config_compiles_as_eight_replica_pool(self):
        b = bundle()
        point = b["matrix"]["points"][0]
        self.assertEqual(point["recipe"], "af_node34_a1f1_pool")
        self.assertEqual(point["qps"], list(range(1, 17)))
        self.assertEqual((point["requests_per_qps"], point["input_len"],
                          point["output_len"], point["request_timeout_s"]),
                         (32, 128, 64, 45))
        self.assertTrue(point["skip_warmup"])
        self.assertEqual(point["sweep_max_inflight"], 16)
        self.assertEqual(point["point_wall_cap_s"], 90)

    def test_poisson_workload_is_independent_deterministic_and_has_rate(self):
        qps4 = generate_poisson_workload(4, count=2000, seed=7)
        self.assertEqual(qps4, generate_poisson_workload(4, count=2000, seed=7))
        self.assertNotEqual([r["arrival_time_s"] for r in qps4],
                            [r["arrival_time_s"] for r in
                             generate_poisson_workload(5, count=2000, seed=7)])
        self.assertTrue(all((r["input_len"], r["output_len"], r["timeout_s"])
                            == (128, 64, 45) for r in qps4))
        realized = len(qps4) / qps4[-1]["arrival_time_s"]
        self.assertAlmostEqual(realized, 4, delta=0.25)

    def test_deploys_once_increases_qps_and_stops_at_boundary(self):
        b = bundle()
        lifecycle_calls = []
        point_calls = []
        probes = []

        def lifecycle(plan, executor, body, **kwargs):
            lifecycle_calls.append(plan)
            return body(plan.handle)

        def point_runner(target, qps, requests, point_dir, executor, bundle,
                         point, wall_timeout):
            point_calls.append((qps, wall_timeout))
            complete = qps < 4
            return {"status": "complete" if complete else "partial",
                    "requests_total": 32, "requests_success": 32 if complete else 31,
                    "requests_failed": 0 if complete else 1,
                    "achieved_qps": qps, "arrival_lag_ms": {"p99": 0},
                    "wall_time_s": 1}

        def probe(plan, executor):
            probes.append(True)
            return {"all_endpoints_ok": True, "all_processes_alive": True,
                    "fatal_log": False}

        with tempfile.TemporaryDirectory() as directory:
            result = run_af_qps_sweep(
                b, Path(directory), 1, 8, executor=object(), lifecycle=lifecycle,
                point_runner=point_runner, probe=probe)
            saved = json.loads((Path(directory) / "sweep_summary.json").read_text())
        self.assertEqual(len(lifecycle_calls), 1)
        self.assertEqual([qps for qps, _ in point_calls], [1, 2, 3, 4])
        self.assertTrue(all(60 <= wall <= 90 for _, wall in point_calls))
        self.assertEqual(len(probes), 4)
        self.assertEqual((result["last_stable_qps"], result["first_failed_qps"]),
                         (3, 4))
        self.assertEqual(saved["stop_reason"], "partial_request_failure")

    def test_continue_on_partial_only_stops_for_hard_health_failure(self):
        b = bundle()
        calls = []
        def lifecycle(plan, executor, body, **kwargs):
            return body(plan.handle)
        def point_runner(target, qps, requests, point_dir, executor, bundle,
                         point, wall_timeout):
            calls.append(qps)
            return {"status": "partial", "requests_total": 32,
                    "requests_success": 31, "requests_failed": 1}
        def probe(plan, executor):
            qps = calls[-1]
            return {"all_endpoints_ok": qps < 3, "all_processes_alive": True,
                    "fatal_log": False}
        with tempfile.TemporaryDirectory() as directory:
            result = run_af_qps_sweep(
                b, Path(directory), 1, 6, True, executor=object(),
                lifecycle=lifecycle, point_runner=point_runner, probe=probe)
        self.assertEqual(calls, [1, 2, 3])
        self.assertEqual(result["first_failed_qps"], 1)
        self.assertEqual(result["stop_reason"], "endpoint_probe_failed")

    def test_wall_limit_covers_arrivals_and_request_timeout_with_cap(self):
        requests = [{"arrival_time_s": 10}, {"arrival_time_s": 20}]
        self.assertEqual(point_wall_limit(requests, 45, 90), 70)
        self.assertEqual(point_wall_limit([{"arrival_time_s": 1}], 45, 90), 60)
        self.assertEqual(point_wall_limit([{"arrival_time_s": 80}], 45, 90), 90)

    def test_point_collects_energy_per_qps_and_caps_concurrency(self):
        requests = [{"request_id": str(i), "arrival_time_s": 0,
                     "input_len": 128, "output_len": 64, "timeout_s": 45}
                    for i in range(20)]
        active = maximum = 0
        lock = threading.Lock()
        energy_reads = []
        def fake_stream(endpoint, request, start, timeout):
            nonlocal active, maximum
            with lock:
                active += 1
                maximum = max(maximum, active)
            time.sleep(0.01)
            with lock:
                active -= 1
            return {"request_id": request["request_id"], "success": True,
                    "input_tokens": 128, "completion_tokens": 64,
                    "sent_offset_s": 0, "completed_offset_s": 0.01,
                    "arrival_lag_ms": 0, "itl_ms": []}
        def fake_energy(*args):
            energy_reads.append(True)
            return {"node": {"0": len(energy_reads) * 1000}}
        b = {"cluster": {"nodes": []}}
        point = {"nodes": 2, "sweep_max_inflight": 16}
        target = SimpleNamespace(endpoint_for_request=lambda index: "http://a")
        with tempfile.TemporaryDirectory() as directory, patch(
            "aflex_benchmark.sweep.collect_system", return_value=[]
        ) as system, patch(
            "aflex_benchmark.sweep.read_cluster_energy", side_effect=fake_energy
        ), patch("aflex_benchmark.sweep.stream_request", side_effect=fake_stream):
            summary = _run_point(target, 1, requests, Path(directory), object(),
                                 b, point, wall_timeout_s=2)
        self.assertEqual(len(energy_reads), 2)
        self.assertEqual(system.call_count, 2)
        self.assertLessEqual(maximum, 16)
        self.assertGreater(maximum, 1)
        self.assertEqual(summary["max_inflight"], 16)
        self.assertEqual(summary["offered_qps"], 1)
        self.assertEqual(summary["arrival_span"], 0)
        self.assertEqual(summary["wall_limit"], 2)


if __name__ == "__main__":
    unittest.main()

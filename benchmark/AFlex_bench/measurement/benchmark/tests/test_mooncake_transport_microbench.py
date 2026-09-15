import importlib.util
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "bench_mooncake_transport.py"
spec = importlib.util.spec_from_file_location("bench_mooncake_transport", SCRIPT)
bench = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bench)


class MooncakeTransportMicrobenchTests(unittest.TestCase):
    def test_transport_environment_is_explicit_and_disables_cuda_ipc(self):
        nvlink = bench.transport_environment("nvlink")
        self.assertEqual(nvlink["MC_FORCE_MNNVL"], "true")
        self.assertEqual(nvlink["MOONCAKE_USE_CUDA_IPC"], "0")
        self.assertEqual(bench.transport_environment("tcp")["MC_FORCE_TCP"], "1")
        self.assertEqual(bench.transport_environment("rdma")["MC_FORCE_MNNVL"], "")

    def test_counter_validation_is_fail_closed(self):
        unavailable = {
            kind: {"available": False, "delta": {}} for kind in ("nvlink", "rdma")
        }
        for transport in bench.TRANSPORTS:
            self.assertFalse(bench.validate_counters(transport, unavailable)["ok"])
        active = {
            "nvlink": {
                "available": True,
                "delta": {"tx_bytes": bench.COUNTER_NOISE_BYTES + 1},
            },
            "rdma": {"available": True, "delta": {"tx_bytes": 0}},
        }
        self.assertTrue(bench.validate_counters("nvlink", active)["ok"])
        self.assertFalse(bench.validate_counters("rdma", active)["ok"])

    def test_container_worker_command_uses_container_path_and_gpu(self):
        args = bench.parser().parse_args(
            ["--transport", "nvlink", "--container", "operator_test"]
        )
        command, env = bench.build_worker_command(args, "receiver")
        self.assertEqual(
            command[:7],
            [
                "docker",
                "exec",
                "-e",
                "CUDA_VISIBLE_DEVICES=1",
                "operator_test",
                "python3",
                bench.CONTAINER_SCRIPT,
            ],
        )
        self.assertEqual(command[7:9], ["receiver", "--worker"])
        self.assertNotIn("CUDA_VISIBLE_DEVICES", env)
        self.assertNotIn("--container", command)

    def test_host_worker_command_uses_current_script_and_environment(self):
        args = bench.parser().parse_args(["--transport", "tcp", "--src-gpu", "3"])
        command, env = bench.build_worker_command(args, "sender")
        self.assertEqual(command[:3], [sys.executable, str(SCRIPT), "sender"])
        self.assertEqual(command[3], "--worker")
        self.assertEqual(env["CUDA_VISIBLE_DEVICES"], "3")
        self.assertNotIn("docker", command)

    def test_worker_rejects_recursive_container(self):
        with patch.object(
            sys,
            "argv",
            [
                str(SCRIPT),
                "sender",
                "--worker",
                "--transport",
                "tcp",
                "--container",
                "operator_test",
            ],
        ):
            self.assertEqual(bench.main(), 1)

    def test_launcher_writes_failure_json_without_gpu(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "result.json"
            with (
                patch.object(bench, "launch", side_effect=RuntimeError("no gpu")),
                patch.object(
                    sys,
                    "argv",
                    [str(SCRIPT), "--transport", "nvlink", "--output", str(output)],
                ),
            ):
                self.assertEqual(bench.main(), 1)
            result = json.loads(output.read_text())
            self.assertEqual(result["status"], "fail")
            self.assertIn("no gpu", result["error"])

    def test_cli_rejects_unknown_transport(self):
        proc = subprocess.run(
            [sys.executable, str(SCRIPT), "--transport", "cuda_ipc"],
            text=True,
            capture_output=True,
            check=False,
        )
        self.assertNotEqual(proc.returncode, 0)
        self.assertIn("invalid choice", proc.stderr)


if __name__ == "__main__":
    unittest.main(verbosity=2)

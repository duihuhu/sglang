import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from aflex_benchmark.collect.link import (
    LINK_COUNTER_NOISE_BYTES,
    build_link_telemetry,
    collect_link_snapshot,
    finish_link_telemetry,
    parse_comm_ledgers,
    parse_dmon_pcie_rates,
    parse_nvlink_counters,
    parse_pcie_rates,
    parse_rdma_counters,
    start_link_telemetry,
    validate_link,
)
from aflex_benchmark.deploy.base import DeploymentPlan, ProcessSpec


class LinkTelemetryParserTests(unittest.TestCase):
    def test_comm_ledger_uses_last_cumulative_marker_per_process(self):
        def marker(pid, backend, calls, d2h, h2d):
            return "AFLEX_COMM_LEDGER " + json.dumps(
                {
                    "schema_version": 1,
                    "pid": pid,
                    "process_id": f"node:{pid}",
                    "backend": backend,
                    "calls": calls,
                    "tx_calls": calls,
                    "rx_calls": 0,
                    "logical_tx_bytes": d2h,
                    "logical_rx_bytes": 0,
                    "expected_d2h_bytes": d2h,
                    "expected_h2d_bytes": h2d,
                }
            )

        logs = "\n".join(
            [
                marker(10, "af_zmq", 1, 100, 100),
                marker(11, "af_zmq", 2, 300, 300),
                marker(10, "af_zmq", 4, 400, 400),
                marker(10, "af_zmq", 2, 200, 200),
            ]
        )
        ledger = parse_comm_ledgers(logs)
        self.assertEqual(ledger["marker"], "AFLEX_COMM_LEDGER")
        self.assertEqual(ledger["calls"], 6)
        self.assertEqual(ledger["expected_d2h_bytes"], 700)
        self.assertEqual(len(ledger["processes"]), 2)

    def test_comm_ledger_ignores_malformed_markers(self):
        ledger = parse_comm_ledgers(
            'AFLEX_COMM_LEDGER not-json\nAFLEX_COMM_LEDGER {"pid":"bad"}'
        )
        self.assertEqual(ledger["calls"], 0)
        self.assertEqual(ledger["processes"], [])

    def test_parsers_accept_driver_and_sysfs_snapshots(self):
        nvlink = parse_nvlink_counters(
            "GPU 0: Link 0: Rx: 2 MiB\nGPU 0: Link 0: Tx: 3 MiB\n"
        )
        self.assertEqual(nvlink, {"rx_bytes": 2 * 1024**2, "tx_bytes": 3 * 1024**2})
        self.assertEqual(
            parse_pcie_rates("0, 12, 34\n1, 1, 2\n"),
            {"rx_bytes_s": 13 * 1024, "tx_bytes_s": 36 * 1024},
        )
        self.assertEqual(
            parse_rdma_counters("port_rcv_data 10\nport_xmit_data 20\n"),
            {"rx_bytes": 40.0, "tx_bytes": 80.0},
        )

    def test_dmon_parser_uses_full_window_and_aggregates_gpu_peaks(self):
        output = """# gpu   rxpci   txpci
# Idx    MB/s    MB/s
0          0       0
1          2       3
0          7       5
1          1       9
"""
        self.assertEqual(
            parse_dmon_pcie_rates(output),
            {"peak_rx_bytes_s": 7 * 1024, "peak_tx_bytes_s": 9 * 1024},
        )

    def test_dmon_lifecycle_starts_only_for_unsupported_and_cleans_up(self):
        commands = []

        class FakeExecutor:
            def run(self, host, command, **kwargs):
                commands.append((host, command, kwargs))
                output = (
                    "# gpu rxpci txpci\n0 0 0\n0 11 13\n"
                    if "kill -TERM" in command
                    else ""
                )
                return subprocess.CompletedProcess([], 0, output, "")

        plan = DeploymentPlan(
            "native",
            [ProcessSpec("NATIVE", "n0", "host0", [0], 30000, "serve")],
            "http://host0:30000",
        )
        before = {
            "nodes": [
                {
                    "node": "n0",
                    "host": "host0",
                    "gpus": "0",
                    "pcie": {"available": False, "values": {}},
                },
                {
                    "node": "n1",
                    "host": "host1",
                    "gpus": "1",
                    "pcie": {"available": True, "values": {"rx_bytes_s": 1}},
                },
            ]
        }
        executor = FakeExecutor()
        handles = start_link_telemetry(executor, plan, before)
        self.assertEqual([handle["node"] for handle in handles], ["n0"])
        self.assertIn("nvidia-smi dmon -i 0 -s t -d 1", commands[0][1])
        result = finish_link_telemetry(executor, handles)
        self.assertEqual(result["n0"]["values"]["peak_rx_bytes_s"], 11 * 1024)
        self.assertIn("kill -TERM", commands[1][1])
        self.assertIn("rm -f --", commands[1][1])

    def test_dmon_window_makes_unsupported_query_available(self):
        snapshot = {
            "nodes": [
                {
                    "node": "n0",
                    "pcie": {"available": False, "values": {}},
                    "nvlink": {"available": True, "values": {"rx_bytes": 0}},
                    "rdma": {"available": True, "values": {"rx_bytes": 0}},
                }
            ]
        }
        telemetry = build_link_telemetry(
            snapshot,
            snapshot,
            {
                "n0": {
                    "available": True,
                    "values": {
                        "peak_rx_bytes_s": 4096,
                        "peak_tx_bytes_s": 2048,
                    },
                }
            },
        )
        pcie = telemetry["nodes"][0]["pcie"]
        self.assertEqual(pcie["source"], "dmon_window")
        self.assertTrue(pcie["available"])
        result = validate_link(
            {"architecture": "native", "expected_link": "pcie"},
            telemetry,
            "NCCL INFO NET/Socket : Using [0]eth0\n"
            "NCCL INFO Channel 00/0 : 0[0] -> 1[1] via SHM/direct/direct",
        )
        self.assertEqual(result["status"], "pass")

    def test_native_pcie_only_actual_channel_is_contradictory(self):
        telemetry = {
            "nodes": [
                {
                    "pcie": {
                        "available": True,
                        "peak_rx_bytes_s": 1,
                        "peak_tx_bytes_s": 0,
                    },
                    "nvlink": {"available": True, "delta": {"rx_bytes": 0}},
                    "rdma": {"available": True, "delta": {"rx_bytes": 0}},
                }
            ]
        }
        initialization = (
            "NCCL INFO NET/Socket : Using [0]eth0\n"
            "NCCL INFO Channel 00/0 : 0[0] -> 1[1] via SHM/direct/direct"
        )
        self.assertEqual(
            validate_link(
                {"architecture": "native", "expected_link": "pcie"},
                telemetry,
                initialization,
            )["status"],
            "pass",
        )
        for channel in ("NET/Socket", "P2P/CUMEM/read", "NVLS"):
            with self.subTest(channel=channel):
                result = validate_link(
                    {"architecture": "native", "expected_link": "pcie"},
                    telemetry,
                    initialization + f"\nNCCL INFO Channel 01 : via {channel}",
                )
                self.assertEqual(result["status"], "invalid_link")
                self.assertTrue(result["paths"][0]["contradictory_evidence"])

    def test_delta_snapshot_preserves_unsupported(self):
        before = {
            "nodes": [
                {
                    "node": "n0",
                    "pcie": {
                        "available": True,
                        "values": {"rx_bytes_s": 5, "tx_bytes_s": 6},
                    },
                    "nvlink": {"available": False, "values": {}},
                    "rdma": {
                        "available": True,
                        "values": {"rx_bytes": 100, "tx_bytes": 200},
                    },
                }
            ]
        }
        after = {
            "nodes": [
                {
                    "node": "n0",
                    "pcie": {
                        "available": True,
                        "values": {"rx_bytes_s": 7, "tx_bytes_s": 8},
                    },
                    "nvlink": {"available": False, "values": {}},
                    "rdma": {
                        "available": True,
                        "values": {"rx_bytes": 160, "tx_bytes": 280},
                    },
                }
            ]
        }
        result = build_link_telemetry(before, after)["nodes"][0]
        self.assertFalse(result["nvlink"]["available"])
        self.assertEqual(result["rdma"]["delta"], {"rx_bytes": 60.0, "tx_bytes": 80.0})
        self.assertEqual(result["pcie"]["peak_tx_bytes_s"], 8)

    def test_validation_requires_log_counter_and_no_contradiction(self):
        telemetry = {
            "nodes": [
                {
                    "pcie": {
                        "available": True,
                        "peak_rx_bytes_s": 100,
                        "peak_tx_bytes_s": 20,
                    },
                    "nvlink": {"available": False, "delta": {}},
                    "rdma": {"available": False, "delta": {}},
                }
            ]
        }
        passed = validate_link(
            {"architecture": "native", "expected_link": "pcie"},
            telemetry,
            "NCCL INFO Channel 00 : 0[0] -> 1[1] via SHM/direct/direct",
        )
        self.assertEqual(passed["status"], "pass")
        unsupported = validate_link(
            {"architecture": "native", "expected_link": "nvlink"},
            telemetry,
            "NCCL INFO Channel 00 : P2P/IPC",
        )
        self.assertEqual(unsupported["status"], "invalid_link")
        self.assertEqual(
            unsupported["paths"][0]["reason"], "hardware_counter_unsupported"
        )
        missing_log = validate_link(
            {"architecture": "native", "expected_link": "pcie"}, telemetry, ""
        )
        self.assertEqual(
            missing_log["paths"][0]["reason"], "missing_backend_log_evidence"
        )

    def test_native_patterns_match_nccl_2_27_transport_lines(self):
        cases = (
            (
                "nvlink",
                (
                    "node3:1234:5678 [0] NCCL INFO NET/Socket : Using [0]eth0\n"
                    "node3:1234:5678 [0] NCCL INFO Channel 00/0 : "
                    "0[0] -> 1[1] via P2P/CUMEM/read"
                ),
                "nvlink",
            ),
            (
                "pcie_host_staged",
                (
                    "node3:1234:5678 [0] NCCL INFO Channel 00/0 : "
                    "0[0] -> 1[1] via SHM/direct/direct"
                ),
                "pcie",
            ),
            (
                "rdma",
                (
                    "node3:1234:5678 [0] NCCL INFO NET/Socket : Using [0]eth0\n"
                    "node3:1234:5678 [0] NCCL INFO NET/IB : "
                    "Using [0]mlx5_0:1/RoCE"
                ),
                "rdma",
            ),
        )
        for expected_link, log, active_counter in cases:
            telemetry = {
                "nodes": [
                    {
                        "pcie": {
                            "available": active_counter == "pcie",
                            "peak_rx_bytes_s": 1024 if active_counter == "pcie" else 0,
                            "peak_tx_bytes_s": 2048 if active_counter == "pcie" else 0,
                        },
                        "nvlink": {
                            "available": active_counter == "nvlink",
                            "delta": {
                                "rx_bytes": (LINK_COUNTER_NOISE_BYTES + 1)
                                if active_counter == "nvlink"
                                else 0
                            },
                        },
                        "rdma": {
                            "available": active_counter == "rdma",
                            "delta": {
                                "rx_bytes": (LINK_COUNTER_NOISE_BYTES + 1)
                                if active_counter == "rdma"
                                else 0
                            },
                        },
                    }
                ]
            }
            with self.subTest(expected_link=expected_link):
                result = validate_link(
                    {"architecture": "native", "expected_link": expected_link},
                    telemetry,
                    log,
                )
                self.assertEqual(result["status"], "pass")
                self.assertTrue(result["paths"][0]["backend_log"])
                self.assertFalse(result["paths"][0]["contradictory_evidence"])

    def test_native_rdma_and_pd_mooncake_evidence(self):
        telemetry = {
            "nodes": [
                {
                    "pcie": {
                        "available": False,
                        "peak_rx_bytes_s": 0,
                        "peak_tx_bytes_s": 0,
                    },
                    "nvlink": {"available": False, "delta": {}},
                    "rdma": {
                        "available": True,
                        "delta": {
                            "rx_bytes": LINK_COUNTER_NOISE_BYTES + 1,
                            "tx_bytes": 0,
                        },
                    },
                }
            ]
        }
        native = validate_link(
            {"architecture": "native", "expected_link": "rdma"},
            telemetry,
            "NCCL INFO NET/IB : Using mlx5_0",
        )
        self.assertEqual(native["status"], "pass")
        pd = validate_link(
            {"architecture": "pd", "expected_link": "rdma"},
            telemetry,
            "Mooncake transfer engine selected RDMA verbs",
        )
        self.assertEqual(pd["status"], "pass")
        contradicted = validate_link(
            {"architecture": "pd", "expected_link": "rdma"},
            telemetry,
            "Mooncake transfer engine selected RDMA verbs; UCX transport tcp",
        )
        self.assertEqual(contradicted["status"], "invalid_link")
        self.assertEqual(
            contradicted["paths"][0]["reason"], "contradictory_negative_evidence"
        )

    def test_pd_nvlink_matches_custom_pool_log_and_rejects_rdma_data(self):
        custom_pool_log = (
            "DEBUG sglang.srt.disaggregation.mooncake.utils: "
            "Initialized custom memory pool: NVLINK on device cuda:0\n"
            "Mooncake transfer engine protocol tcp"
        )

        def telemetry(rdma_bytes=0):
            return {
                "nodes": [
                    {
                        "pcie": {
                            "available": False,
                            "peak_rx_bytes_s": 0,
                            "peak_tx_bytes_s": 0,
                        },
                        "nvlink": {
                            "available": True,
                            "delta": {"rx_bytes": LINK_COUNTER_NOISE_BYTES + 1},
                        },
                        "rdma": {
                            "available": True,
                            "delta": {"rx_bytes": rdma_bytes},
                        },
                    }
                ]
            }

        passed = validate_link(
            {"architecture": "pd", "expected_link": "nvlink"},
            telemetry(),
            custom_pool_log,
        )
        self.assertEqual(passed["status"], "pass")
        self.assertTrue(passed["paths"][0]["backend_log"])
        self.assertFalse(passed["paths"][0]["contradictory_evidence"])

        rdma_transfer = validate_link(
            {"architecture": "pd", "expected_link": "nvlink"},
            telemetry(LINK_COUNTER_NOISE_BYTES + 1),
            custom_pool_log,
        )
        self.assertEqual(rdma_transfer["status"], "invalid_link")
        self.assertEqual(
            rdma_transfer["paths"][0]["reason"],
            "contradictory_negative_evidence",
        )

    def test_pd_nvlink_accepts_cuda_ipc_backend_marker_and_rejects_rdma_data(self):
        def telemetry(rdma_bytes=0):
            return {
                "nodes": [
                    {
                        "pcie": {"available": False},
                        "nvlink": {
                            "available": True,
                            "delta": {"rx_bytes": LINK_COUNTER_NOISE_BYTES + 1},
                        },
                        "rdma": {
                            "available": True,
                            "delta": {"rx_bytes": rdma_bytes},
                        },
                    }
                ]
            }

        marker = "CUDA IPC KV backend ready hostname=node0 gpu_id=0 tp_rank=0 tp_size=1"
        passed = validate_link(
            {"architecture": "pd", "expected_link": "nvlink"},
            telemetry(),
            marker,
        )
        self.assertEqual(passed["status"], "pass")
        self.assertTrue(passed["paths"][0]["backend_log"])

        contradicted = validate_link(
            {"architecture": "pd", "expected_link": "nvlink"},
            telemetry(LINK_COUNTER_NOISE_BYTES + 1),
            marker,
        )
        self.assertEqual(contradicted["status"], "invalid_link")
        self.assertEqual(
            contradicted["paths"][0]["reason"],
            "contradictory_negative_evidence",
        )

    def test_pd_nvlink_rejects_legacy_cuda_ipc_marker(self):
        telemetry = {
            "nodes": [
                {
                    "pcie": {"available": False},
                    "nvlink": {
                        "available": True,
                        "delta": {"rx_bytes": LINK_COUNTER_NOISE_BYTES + 1},
                    },
                    "rdma": {"available": True, "delta": {"rx_bytes": 0}},
                }
            ]
        }
        result = validate_link(
            {"architecture": "pd", "expected_link": "nvlink"},
            telemetry,
            "MOONCAKE_USE_CUDA_IPC=1; Mooncake transfer engine selected NVLink",
        )
        self.assertEqual(result["status"], "invalid_link")
        self.assertEqual(result["paths"][0]["reason"], "missing_backend_log_evidence")

    def test_pd_host_staged_requires_active_tcp_and_pcie_counter(self):
        def telemetry(rdma_bytes=4608, pcie_rate=4096, nvlink_bytes=0):
            return {
                "nodes": [
                    {
                        "pcie": {
                            "available": True,
                            "peak_rx_bytes_s": pcie_rate,
                            "peak_tx_bytes_s": 0,
                        },
                        "nvlink": {
                            "available": True,
                            "delta": {"rx_bytes": nvlink_bytes},
                        },
                        "rdma": {
                            "available": True,
                            "delta": {"rx_bytes": rdma_bytes},
                        },
                    }
                ]
            }

        actual_tcp_log = (
            "installTransport, type=rdma\n"
            "MC_FORCE_TCP is set, using TCP transport only\n"
            "TcpTransport: listen on port 12345"
        )
        passed = validate_link(
            {"architecture": "pd", "expected_link": "pcie_host_staged"},
            telemetry(),
            actual_tcp_log,
        )
        self.assertEqual(passed["status"], "pass")
        self.assertTrue(passed["paths"][0]["backend_log"])
        self.assertFalse(passed["paths"][0]["contradictory_evidence"])

        significant_rdma = validate_link(
            {"architecture": "pd", "expected_link": "pcie_host_staged"},
            telemetry(rdma_bytes=LINK_COUNTER_NOISE_BYTES + 1),
            actual_tcp_log,
        )
        self.assertEqual(significant_rdma["status"], "invalid_link")
        self.assertEqual(
            significant_rdma["paths"][0]["reason"],
            "contradictory_negative_evidence",
        )

        no_pcie = validate_link(
            {"architecture": "pd", "expected_link": "pcie_host_staged"},
            telemetry(pcie_rate=0),
            actual_tcp_log,
        )
        self.assertEqual(no_pcie["status"], "invalid_link")
        self.assertEqual(
            no_pcie["paths"][0]["reason"], "missing_positive_counter_evidence"
        )

        vague_tcp = validate_link(
            {"architecture": "pd", "expected_link": "pcie_host_staged"},
            telemetry(),
            "Mooncake transfer engine protocol tcp",
        )
        self.assertEqual(vague_tcp["status"], "invalid_link")
        self.assertEqual(
            vague_tcp["paths"][0]["reason"], "missing_backend_log_evidence"
        )

    def test_nvlink_and_rdma_counter_noise_boundaries(self):
        for kind, log in (
            ("nvlink", "NCCL INFO Channel 00 : P2P/IPC"),
            ("rdma", "NCCL INFO NET/IB : Using mlx5_0"),
        ):
            for counter_value, expected_status, background_noise in (
                (0, "invalid_link", False),
                (2304, "invalid_link", True),
                (LINK_COUNTER_NOISE_BYTES, "invalid_link", True),
                (LINK_COUNTER_NOISE_BYTES + 1, "pass", False),
            ):
                telemetry = {
                    "nodes": [
                        {
                            "pcie": {
                                "available": False,
                                "peak_rx_bytes_s": 0,
                                "peak_tx_bytes_s": 0,
                            },
                            "nvlink": {
                                "available": kind == "nvlink",
                                "delta": {
                                    "rx_bytes": counter_value if kind == "nvlink" else 0
                                },
                            },
                            "rdma": {
                                "available": kind == "rdma",
                                "delta": {
                                    "rx_bytes": counter_value if kind == "rdma" else 0
                                },
                            },
                        }
                    ]
                }
                with self.subTest(kind=kind, counter_value=counter_value):
                    result = validate_link(
                        {"architecture": "native", "expected_link": kind},
                        telemetry,
                        log,
                    )
                    validation = result["paths"][0]
                    self.assertEqual(result["status"], expected_status)
                    self.assertEqual(validation["counter_value"], counter_value)
                    self.assertEqual(
                        validation["noise_threshold"], LINK_COUNTER_NOISE_BYTES
                    )
                    self.assertEqual(validation["background_noise"], background_noise)

    def test_native_pcie_ignores_subthreshold_rdma_noise(self):
        telemetry = {
            "nodes": [
                {
                    "pcie": {
                        "available": True,
                        "peak_rx_bytes_s": 1,
                        "peak_tx_bytes_s": 0,
                    },
                    "nvlink": {"available": True, "delta": {"rx_bytes": 0}},
                    "rdma": {"available": True, "delta": {"rx_bytes": 4608}},
                }
            ]
        }
        result = validate_link(
            {"architecture": "native", "expected_link": "pcie"},
            telemetry,
            "NCCL INFO Channel 00 : via SHM/direct/direct",
        )
        validation = result["paths"][0]
        self.assertEqual(result["status"], "pass")
        self.assertEqual(validation["counter_value"], 1)
        self.assertEqual(validation["noise_threshold"], 0)
        self.assertFalse(validation["background_noise"])
        self.assertFalse(validation["contradictory_evidence"])

    def test_af_cppipc_actual_logs_validate_nvlink_and_ipc_cpp(self):
        logs = """[CppIPC ATTN] connected to peer 0
[CppIPC FFN] accepted peer 1
[CppIPC] handshake complete
"""
        telemetry = {
            "nodes": [
                {
                    "pcie": {
                        "available": True,
                        "peak_rx_bytes_s": 1,
                        "peak_tx_bytes_s": 0,
                    },
                    "nvlink": {
                        "available": True,
                        "delta": {"rx_bytes": LINK_COUNTER_NOISE_BYTES + 1},
                    },
                    "rdma": {"available": True, "delta": {"rx_bytes": 0}},
                }
            ]
        }

        nvlink = validate_link(
            {"architecture": "af", "expected_link": "nvlink"}, telemetry, logs
        )["paths"][0]
        self.assertGreater(nvlink["counter_value"], nvlink["noise_threshold"])
        self.assertTrue(nvlink["backend_log"])
        self.assertTrue(nvlink["valid"])

        ipc_cpp = validate_link(
            {"architecture": "af", "expected_link": "ipc_cpp"}, telemetry, logs
        )["paths"][0]
        self.assertGreater(ipc_cpp["counter_value"], ipc_cpp["noise_threshold"])
        self.assertTrue(ipc_cpp["backend_log"])
        self.assertTrue(ipc_cpp["valid"])

    def test_af_ipc_marker_aliases_match_nvlink_and_ipc_cpp(self):
        telemetry = {
            "nodes": [
                {
                    "pcie": {
                        "available": True,
                        "peak_rx_bytes_s": 1,
                        "peak_tx_bytes_s": 0,
                    },
                    "nvlink": {
                        "available": True,
                        "delta": {"rx_bytes": LINK_COUNTER_NOISE_BYTES + 1},
                    },
                    "rdma": {"available": True, "delta": {"rx_bytes": 0}},
                }
            ]
        }
        for marker in ("[CppIPC] handshake complete", "afd_ipc_cpp", "AFD IPC"):
            with self.subTest(marker=marker):
                for path in ("nvlink", "ipc_cpp"):
                    validation = validate_link(
                        {"architecture": "af", "expected_link": path},
                        telemetry,
                        marker,
                    )["paths"][0]
                    self.assertTrue(validation["backend_log"])
                    self.assertTrue(validation["valid"])

    def test_af_cppipc_rejects_conflicting_transport_logs(self):
        telemetry = {
            "nodes": [
                {
                    "pcie": {
                        "available": True,
                        "peak_rx_bytes_s": 1,
                        "peak_tx_bytes_s": 0,
                    },
                    "nvlink": {
                        "available": True,
                        "delta": {"rx_bytes": LINK_COUNTER_NOISE_BYTES + 1},
                    },
                    "rdma": {"available": True, "delta": {"rx_bytes": 0}},
                }
            ]
        }
        cppipc = "[CppIPC] handshake complete"
        for conflicting_log in (
            "AFD ZMQ handshake ready",
            "UCX transport tcp ready",
            "UCX transport rdma ready",
        ):
            with self.subTest(conflicting_log=conflicting_log):
                logs = f"{cppipc}\n{conflicting_log}"
                for path in ("nvlink", "ipc_cpp"):
                    validation = validate_link(
                        {"architecture": "af", "expected_link": path},
                        telemetry,
                        logs,
                    )["paths"][0]
                    self.assertTrue(validation["backend_log"])
                    self.assertTrue(validation["contradictory_evidence"])
                    self.assertFalse(validation["valid"])
                    self.assertEqual(
                        validation["reason"], "contradictory_negative_evidence"
                    )

    def test_af_pcie_host_staged_requires_zmq_handshake_and_rejects_other_links(self):
        def telemetry(*, pcie_rate=9, nvlink_bytes=0, rdma_bytes=0):
            return {
                "nodes": [
                    {
                        "pcie": {
                            "available": True,
                            "peak_rx_bytes_s": pcie_rate,
                            "peak_tx_bytes_s": 0,
                        },
                        "nvlink": {
                            "available": True,
                            "delta": {"rx_bytes": nvlink_bytes},
                        },
                        "rdma": {
                            "available": True,
                            "delta": {"rx_bytes": rdma_bytes},
                        },
                    }
                ]
            }

        marker = "AFD ZMQ handshake ready: perspective=AFD_PERSPECTIVE_ATTN rank=0"
        passed = validate_link(
            {"architecture": "af", "expected_link": "pcie_host_staged"},
            telemetry(),
            marker,
        )["paths"][0]
        self.assertTrue(passed["backend_log"])
        self.assertTrue(passed["positive_counter"])
        self.assertFalse(passed["contradictory_evidence"])
        self.assertTrue(passed["valid"])

        for logs in ("ZMQ communicator configured", "AFD IPC ready", "UCX ready"):
            with self.subTest(logs=logs):
                result = validate_link(
                    {"architecture": "af", "expected_link": "pcie_host_staged"},
                    telemetry(),
                    logs,
                )["paths"][0]
                self.assertFalse(result["backend_log"])
                self.assertEqual(result["reason"], "missing_backend_log_evidence")

        for counters in (
            {"nvlink_bytes": LINK_COUNTER_NOISE_BYTES + 1},
            {"rdma_bytes": LINK_COUNTER_NOISE_BYTES + 1},
        ):
            with self.subTest(counters=counters):
                result = validate_link(
                    {"architecture": "af", "expected_link": "pcie_host_staged"},
                    telemetry(**counters),
                    marker,
                )["paths"][0]
                self.assertTrue(result["contradictory_evidence"])
                self.assertFalse(result["valid"])

    def test_af_pcie_semantic_ledger_allows_zero_dmon_without_physical_claim(self):
        telemetry = {
            "nodes": [
                {
                    "pcie": {
                        "available": True,
                        "peak_rx_bytes_s": 0,
                        "peak_tx_bytes_s": 0,
                        "source": "dmon_window",
                    },
                    "nvlink": {"available": True, "delta": {"rx_bytes": 0}},
                    "rdma": {"available": True, "delta": {"rx_bytes": 0}},
                }
            ]
        }
        point = {
            "architecture": "af",
            "expected_link": "pcie_host_staged",
        }
        marker = (
            "AFLEX_COMM_LEDGER "
            '{"schema_version":1,"pid":7,"process_id":"n0:7",'
            '"backend":"af_zmq","calls":2,"tx_calls":1,"rx_calls":1,'
            '"logical_tx_bytes":4096,"logical_rx_bytes":4096,'
            '"expected_d2h_bytes":4096,"expected_h2d_bytes":4096}'
        )
        result = validate_link(point, telemetry, marker)
        self.assertEqual(result["schema_version"], 2)
        self.assertEqual(result["status"], "verified_semantic")
        self.assertEqual(result["validation_mode"], "semantic")
        self.assertFalse(result["physical_verified"])
        self.assertTrue(result["paths"][0]["semantic_valid"])
        self.assertFalse(result["paths"][0]["positive_counter"])

        contradicted = validate_link(
            point,
            {
                "nodes": [
                    {
                        **telemetry["nodes"][0],
                        "nvlink": {
                            "available": True,
                            "delta": {"rx_bytes": LINK_COUNTER_NOISE_BYTES + 1},
                        },
                    }
                ]
            },
            marker,
        )
        self.assertEqual(contradicted["status"], "invalid_link")

    def test_pd_tcp_actual_ledger_allows_semantic_validation(self):
        telemetry = {
            "nodes": [
                {
                    "pcie": {
                        "available": True,
                        "peak_rx_bytes_s": 0,
                        "peak_tx_bytes_s": 0,
                    },
                    "nvlink": {"available": True, "delta": {"rx_bytes": 0}},
                    "rdma": {"available": True, "delta": {"rx_bytes": 0}},
                }
            ]
        }
        logs = (
            'AFLEX_COMM_LEDGER {"schema_version":1,"pid":8,'
            '"process_id":"n1:8","backend":"mooncake_tcp","calls":1,'
            '"tx_calls":1,"rx_calls":0,"logical_tx_bytes":8192,'
            '"logical_rx_bytes":0,"expected_d2h_bytes":8192,'
            '"expected_h2d_bytes":8192}'
        )
        result = validate_link(
            {"architecture": "pd", "expected_link": "pcie_host_staged"},
            telemetry,
            logs,
        )
        self.assertEqual(result["status"], "verified_semantic")
        self.assertEqual(result["semantic_comm_ledger"]["logical_tx_bytes"], 8192)

    def test_component_tp2_nvlink_is_internal_for_semantic_host_staging(self):
        telemetry = {
            "nodes": [
                {
                    "pcie": {
                        "available": True,
                        "peak_rx_bytes_s": 0,
                        "peak_tx_bytes_s": 0,
                    },
                    "nvlink": {
                        "available": True,
                        "delta": {"rx_bytes": LINK_COUNTER_NOISE_BYTES + 1},
                    },
                    "rdma": {"available": True, "delta": {"rx_bytes": 0}},
                }
            ]
        }
        ledgers = {
            "af": "af_zmq",
            "pd": "mooncake_tcp",
        }
        for architecture, backend in ledgers.items():
            with self.subTest(architecture=architecture):
                logs = (
                    'AFLEX_COMM_LEDGER {"schema_version":1,"pid":9,'
                    f'"process_id":"n1:9","backend":"{backend}","calls":1,'
                    '"tx_calls":1,"rx_calls":0,"logical_tx_bytes":8192,'
                    '"logical_rx_bytes":0,"expected_d2h_bytes":8192,'
                    '"expected_h2d_bytes":8192}'
                )
                result = validate_link(
                    {
                        "architecture": architecture,
                        "expected_link": "pcie_host_staged",
                        "expected_internal_link": "nvlink",
                        "component_tp": 2,
                    },
                    telemetry,
                    logs,
                )
                self.assertEqual(result["status"], "verified_semantic")
                self.assertFalse(result["paths"][0]["contradictory_evidence"])
                self.assertEqual(
                    result["internal_link_evidence"],
                    {
                        "expected_internal_link": "nvlink",
                        "component_tp": 2,
                        "nvlink_counter_value": LINK_COUNTER_NOISE_BYTES + 1,
                        "nvlink_active": True,
                        "accepted_as_internal": True,
                    },
                )

    def test_internal_nvlink_never_masks_rdma_boundary_evidence(self):
        telemetry = {
            "nodes": [
                {
                    "pcie": {
                        "available": True,
                        "peak_rx_bytes_s": 0,
                        "peak_tx_bytes_s": 0,
                    },
                    "nvlink": {
                        "available": True,
                        "delta": {"rx_bytes": LINK_COUNTER_NOISE_BYTES + 1},
                    },
                    "rdma": {
                        "available": True,
                        "delta": {"rx_bytes": LINK_COUNTER_NOISE_BYTES + 1},
                    },
                }
            ]
        }
        logs = (
            'AFLEX_COMM_LEDGER {"schema_version":1,"pid":10,'
            '"process_id":"n1:10","backend":"mooncake_tcp","calls":1,'
            '"tx_calls":1,"rx_calls":0,"logical_tx_bytes":8192,'
            '"logical_rx_bytes":0,"expected_d2h_bytes":8192,'
            '"expected_h2d_bytes":8192}'
        )
        result = validate_link(
            {
                "architecture": "pd",
                "expected_link": "pcie_host_staged",
                "component_tp": 2,
            },
            telemetry,
            logs,
        )
        self.assertEqual(result["status"], "invalid_link")
        self.assertTrue(result["paths"][0]["contradictory_evidence"])
        self.assertFalse(result["internal_link_evidence"]["accepted_as_internal"])

    def test_pd_cuda_ipc_completed_request_allows_semantic_when_counter_unavailable(self):
        point = {
            "architecture": "pd",
            "recipe": "pd_cuda_ipc_same_node",
            "expected_link": "nvlink",
        }
        unavailable = {
            "nodes": [
                {
                    "pcie": {"available": False, "peak_rx_bytes_s": 0, "peak_tx_bytes_s": 0},
                    "nvlink": {"available": False, "delta": {}},
                    "rdma": {"available": True, "delta": {"rx_bytes": 0}},
                }
            ]
        }
        logs = (
            "CUDA IPC KV backend ready hostname=node gpu_id=0 tp_rank=0 tp_size=4\n"
            "CUDA IPC KV transfer complete bytes=29360128 room=7"
        )
        result = validate_link(
            point, unavailable, logs, request_succeeded=True
        )
        self.assertEqual(result["status"], "verified_semantic")
        self.assertFalse(result["physical_verified"])
        self.assertTrue(result["cuda_ipc_semantic_evidence"]["accepted"])
        self.assertTrue(result["paths"][0]["semantic_valid"])

    def test_pd_cuda_ipc_semantic_fallback_requires_all_specific_evidence(self):
        base_point = {
            "architecture": "pd",
            "recipe": "pd_cuda_ipc_same_node",
            "expected_link": "nvlink",
        }
        unavailable = {
            "nodes": [
                {
                    "pcie": {"available": False, "peak_rx_bytes_s": 0, "peak_tx_bytes_s": 0},
                    "nvlink": {"available": False, "delta": {}},
                    "rdma": {"available": True, "delta": {"rx_bytes": 0}},
                }
            ]
        }
        complete = (
            "CUDA IPC KV backend ready hostname=node gpu_id=0 tp_rank=0 tp_size=4\n"
            "CUDA IPC KV transfer complete bytes=4096 room=7"
        )
        cases = [
            (dict(base_point, recipe="pd_rdma"), complete, True),
            (base_point, "CUDA IPC KV backend ready", True),
            (base_point, complete, False),
            (dict(base_point, expected_link="rdma"), complete, True),
        ]
        for point, logs, succeeded in cases:
            with self.subTest(point=point, logs=logs, succeeded=succeeded):
                result = validate_link(
                    point, unavailable, logs, request_succeeded=succeeded
                )
                self.assertEqual(result["status"], "invalid_link")
                self.assertFalse(result["cuda_ipc_semantic_evidence"]["accepted"])

    def test_native_shm_remains_physical_pass(self):
        telemetry = {
            "nodes": [
                {
                    "pcie": {
                        "available": True,
                        "peak_rx_bytes_s": 1,
                        "peak_tx_bytes_s": 0,
                    },
                    "nvlink": {"available": True, "delta": {"rx_bytes": 0}},
                    "rdma": {"available": True, "delta": {"rx_bytes": 0}},
                }
            ]
        }
        result = validate_link(
            {"architecture": "native", "expected_link": "pcie_host_staged"},
            telemetry,
            "NCCL INFO Channel 00 : via SHM/direct/direct",
        )
        self.assertEqual(result["status"], "pass")
        self.assertEqual(result["validation_mode"], "physical")
        self.assertTrue(result["physical_verified"])

    def test_af_backends_require_physical_counter(self):
        telemetry = {
            "nodes": [
                {
                    "pcie": {
                        "available": True,
                        "peak_rx_bytes_s": 9,
                        "peak_tx_bytes_s": 8,
                    },
                    "nvlink": {"available": False, "delta": {}},
                    "rdma": {"available": False, "delta": {}},
                }
            ]
        }
        for marker in (
            "afd-comm-backend ipc_cpp; AFD IPC ready",
            "AFD ZMQ handshake ready",
        ):
            result = validate_link(
                {"architecture": "af", "expected_link": "pcie"}, telemetry, marker
            )
            self.assertEqual(result["status"], "pass")


class LinkTelemetryDryRunSnapshotTests(unittest.TestCase):
    def test_collector_command_snapshot_uses_read_only_probes(self):
        commands = []

        class FakeExecutor:
            def run(self, host, command, **kwargs):
                commands.append(("container", host, command, kwargs))
                output = (
                    "0, 1, 2\n"
                    if "pcie.rx_util" in command
                    else "GPU 0 Rx 3 MiB\nGPU 0 Tx 4 MiB\n"
                )
                return subprocess.CompletedProcess([], 0, output, "")

            def host_run(self, host, command, **kwargs):
                commands.append(("host", host, command, kwargs))
                return subprocess.CompletedProcess(
                    [], 0, "port_rcv_data 10\nport_xmit_data 20\n", ""
                )

        plan = DeploymentPlan(
            "native",
            [ProcessSpec("NATIVE", "n0", "host0", [0, 1], 30000, "serve")],
            "http://host0:30000",
            cluster={"nodes": [{"name": "n0", "host": "host0", "gpus": [0, 1]}]},
        )
        snapshot = collect_link_snapshot(FakeExecutor(), plan)
        self.assertEqual(snapshot["nodes"][0]["gpus"], "0,1")
        rendered = "\n".join(command for _, _, command, _ in commands)
        self.assertIn("pcie.rx_util,pcie.tx_util", rendered)
        self.assertIn("nvidia-smi nvlink", rendered)
        self.assertIn("/sys/class/infiniband", rendered)
        self.assertNotIn("sglang serve", rendered)
        self.assertTrue(
            all(
                kwargs == {"check": False, "quiet": True}
                for _, _, _, kwargs in commands
            )
        )

    def test_artifact_snapshot_is_json_serializable(self):
        telemetry = build_link_telemetry({"nodes": []}, {"nodes": []})
        validation = validate_link(
            {"architecture": "af", "expected_link": "ucx"}, telemetry, "UCX ready"
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "link_telemetry.json").write_text(json.dumps(telemetry, indent=2))
            (root / "link_validation.json").write_text(json.dumps(validation, indent=2))
            self.assertEqual(
                json.loads((root / "link_validation.json").read_text())["status"],
                "invalid_link",
            )


if __name__ == "__main__":
    unittest.main(verbosity=2)

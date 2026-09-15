import os
import sys
import types
import unittest
from typing import ClassVar
from unittest.mock import patch

from sglang.srt.distributed.device_communicators.mooncake_transfer_engine import (
    MOONCAKE_TRANSPORT_ENV,
    MooncakeTransferEngine,
    configure_mooncake_transport_env,
    resolve_mooncake_transport,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=1, suite="stage-a-cpu-only")


class FakeTransferEngine:
    instances: ClassVar[list] = []

    def __init__(self):
        self.initialize_args = None
        self.__class__.instances.append(self)

    def initialize(self, *args):
        self.initialize_args = args
        return 0

    def get_rpc_port(self):
        return 12345

    def transfer_sync_write(self, *args):
        return 0

    def batch_transfer_sync_write(self, *args):
        return 0


class MooncakeTransportTests(CustomTestCase):
    def setUp(self):
        FakeTransferEngine.instances.clear()

    def _make_engine(self, transport=None, env=None):
        module = types.ModuleType("mooncake.engine")
        module.TransferEngine = FakeTransferEngine
        package = types.ModuleType("mooncake")
        package.engine = module
        clean = {
            MOONCAKE_TRANSPORT_ENV: "",
            "MC_FORCE_MNNVL": "",
            "MC_FORCE_TCP": "",
        }
        clean.update(env or {})
        with (
            patch.dict(os.environ, clean, clear=False),
            patch.dict(sys.modules, {"mooncake": package, "mooncake.engine": module}),
        ):
            engine = MooncakeTransferEngine("127.0.0.1", transport=transport)
            flags = {k: os.environ.get(k) for k in clean}
        return engine, FakeTransferEngine.instances[-1].initialize_args, flags

    def test_tcp_success_records_semantic_ledger(self):
        engine, _, _ = self._make_engine("tcp")
        with patch(
            "sglang.srt.distributed.device_communicators.mooncake_transfer_engine.record_comm"
        ) as record:
            self.assertEqual(engine.transfer_sync("peer", 1, 2, 4096), 0)
            record.assert_called_once_with(
                "mooncake_tcp",
                logical_tx_bytes=4096,
                expected_d2h_bytes=4096,
                expected_h2d_bytes=4096,
                tx_calls=1,
            )

    def test_tcp_batch_success_records_total_semantic_ledger(self):
        engine, _, _ = self._make_engine("tcp")
        with patch(
            "sglang.srt.distributed.device_communicators.mooncake_transfer_engine.record_comm"
        ) as record:
            self.assertEqual(
                engine.batch_transfer_sync("peer", [1, 2], [3, 4], [100, 200]), 0
            )
            self.assertEqual(record.call_args.kwargs["logical_tx_bytes"], 300)
            self.assertEqual(record.call_args.kwargs["expected_h2d_bytes"], 300)

    def test_non_tcp_and_failed_transfers_do_not_record_tcp_ledger(self):
        engine, _, _ = self._make_engine("rdma")
        with patch(
            "sglang.srt.distributed.device_communicators.mooncake_transfer_engine.record_comm"
        ) as record:
            engine.transfer_sync("peer", 1, 2, 16)
            record.assert_not_called()
            engine.transport = "tcp"
            engine.engine.transfer_sync_write = lambda *args: -1
            engine.transfer_sync("peer", 1, 2, 16)
            record.assert_not_called()

    def test_default_remains_rdma(self):
        engine, args, _ = self._make_engine()
        self.assertEqual(engine.transport, "rdma")
        self.assertEqual(args[2], "rdma")

    def test_explicit_transports_configure_python_api_and_selectors(self):
        cases = {
            "rdma": ("rdma", None, None),
            "tcp": ("tcp", None, "1"),
            "nvlink": ("rdma", "true", None),
        }
        for transport, (protocol, mnnvl, tcp) in cases.items():
            with self.subTest(transport=transport):
                engine, args, flags = self._make_engine(transport)
                self.assertEqual(engine.transport, transport)
                self.assertEqual(args[2], protocol)
                self.assertEqual(flags["MC_FORCE_MNNVL"], mnnvl)
                self.assertEqual(flags["MC_FORCE_TCP"], tcp)

    def test_sglang_and_legacy_env_select_transport(self):
        with patch.dict(os.environ, {MOONCAKE_TRANSPORT_ENV: "tcp"}, clear=True):
            self.assertEqual(resolve_mooncake_transport(), "tcp")
        with patch.dict(os.environ, {"MC_FORCE_MNNVL": "true"}, clear=True):
            self.assertEqual(resolve_mooncake_transport(), "nvlink")

    def test_invalid_transport_fails_before_engine_initialization(self):
        with self.assertRaisesRegex(ValueError, "Unsupported Mooncake transport"):
            resolve_mooncake_transport("cuda_ipc")

    def test_selector_helper_is_mutually_exclusive(self):
        with patch.dict(
            os.environ, {"MC_FORCE_MNNVL": "true", "MC_FORCE_TCP": "1"}, clear=True
        ):
            configure_mooncake_transport_env("rdma")
            self.assertNotIn("MC_FORCE_MNNVL", os.environ)
            self.assertNotIn("MC_FORCE_TCP", os.environ)


if __name__ == "__main__":
    unittest.main(verbosity=3)

import dataclasses
import importlib.util
import socket
import sys
import types
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[4]


def load_conn_without_sglang_dependencies():
    base_conn = types.ModuleType("sglang.srt.disaggregation.base.conn")
    class KVArgs: pass
    class KVPoll:
        Failed, Bootstrapping, WaitingForInput, Transferring, Success = range(5)
    base_conn.KVArgs, base_conn.KVPoll = KVArgs, KVPoll

    common_conn = types.ModuleType("sglang.srt.disaggregation.common.conn")
    for name in ("CommonKVBootstrapServer", "CommonKVManager", "CommonKVReceiver", "CommonKVSender"):
        setattr(common_conn, name, type(name, (), {}))

    common_utils = types.ModuleType("sglang.srt.disaggregation.common.utils")
    def group_concurrent_contiguous(src, dst):
        src_groups, dst_groups = [], []
        start = 0
        for i in range(1, len(src) + 1):
            if i == len(src) or src[i] != src[i - 1] + 1 or dst[i] != dst[i - 1] + 1:
                src_groups.append(src[start:i])
                dst_groups.append(dst[start:i])
                start = i
        return src_groups, dst_groups
    common_utils.group_concurrent_contiguous = group_concurrent_contiguous

    disagg_utils = types.ModuleType("sglang.srt.disaggregation.utils")
    class DisaggregationMode: pass
    disagg_utils.DisaggregationMode = DisaggregationMode
    server_args = types.ModuleType("sglang.srt.server_args")
    server_args.ServerArgs = type("ServerArgs", (), {})
    stubs = {
        "sglang.srt.disaggregation.base.conn": base_conn,
        "sglang.srt.disaggregation.common.conn": common_conn,
        "sglang.srt.disaggregation.common.utils": common_utils,
        "sglang.srt.disaggregation.utils": disagg_utils,
        "sglang.srt.server_args": server_args,
    }
    old = {name: sys.modules.get(name) for name in stubs}
    sys.modules.update(stubs)
    try:
        spec = importlib.util.spec_from_file_location(
            "cuda_ipc_conn_under_test",
            ROOT / "python/sglang/srt/disaggregation/cuda_ipc/conn.py",
        )
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        return module
    finally:
        for name, value in old.items():
            if value is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = value


conn = load_conn_without_sglang_dependencies()


class MockRuntime:
    def __init__(self):
        self.opened, self.closed, self.copies = [], [], []
    def open(self, handle, device):
        ptr = 10_000 + len(self.opened) * 1_000
        self.opened.append((handle, device, ptr))
        return ptr
    def close(self, ptr): self.closed.append(ptr)
    def copy(self, dst, dst_device, src, src_device, length):
        self.copies.append((dst, dst_device, src, src_device, length))
    def synchronize(self, device): self.synced = device
    def can_access_peer(self, src_device, dst_device): return True
    def driver_version(self): return 12090


def registration(tp_rank=1, tp_size=2, hostname=None):
    shared_handle = b"a" * conn.IPC_HANDLE_BYTES
    return conn.CudaIpcRegistration(
        hostname=hostname or socket.gethostname(), endpoint="127.0.0.1",
        driver_version=12090, gpu_id=3, tp_rank=tp_rank, tp_size=tp_size,
        kv_handles=(shared_handle, shared_handle),
        kv_offsets=(128, 8192),
        kv_allocation_sizes=(16384, 16384),
        kv_lens=(4096, 4096), kv_item_lens=(64, 64), aux_ptrs=(32123,),
    )

class TestCudaIpcProtocol(unittest.TestCase):
    def test_registration_roundtrip(self):
        reg = registration()
        self.assertEqual(conn.decode_registration(conn.encode_registration(reg)), reg)

    def test_extension_runtime_accepts_new_tuple_and_legacy_bytes(self):
        runtime = object.__new__(conn.ExtensionCudaIpcRuntime)
        runtime._ipc = types.SimpleNamespace(
            export_handle=lambda ptr, length, device: (b"h" * 64, 128, 4096)
        )
        self.assertEqual(runtime.export(1, 2, 3), (b"h" * 64, 128, 4096))
        runtime._ipc.export_handle = lambda ptr, length, device: b"l" * 64
        self.assertEqual(runtime.export(1, 2, 3), b"l" * 64)

    def test_tp2_identity_mapping(self):
        registration().validate_for(socket.gethostname(), 12090, 1, 2)
        with self.assertRaisesRegex(RuntimeError, "identity TP rank mapping"):
            registration(tp_rank=0).validate_for(socket.gethostname(), 12090, 1, 2)
        with self.assertRaisesRegex(RuntimeError, "equal prefill/decode"):
            registration(tp_size=1).validate_for(socket.gethostname(), 12090, 1, 2)

    def test_fail_fast_host_and_driver(self):
        with self.assertRaisesRegex(RuntimeError, "same host"):
            registration(hostname="remote").validate_for(socket.gethostname(), 12090, 1, 2)
        with self.assertRaisesRegex(RuntimeError, "driver"):
            registration().validate_for(socket.gethostname(), 12080, 1, 2)

    def test_mock_copy_layout_and_lifecycle(self):
        runtime = MockRuntime()
        mgr = object.__new__(conn.CudaIpcKVManager)
        mgr.runtime, mgr.attn_tp_size, mgr.local_hostname = runtime, 2, socket.gethostname()
        mgr.opened_remote_bases = {}
        mgr.is_mla_backend, mgr.opened_remote_ptrs, mgr.remote_registrations = False, {}, {}
        mgr.kv_args = conn.CudaIpcKVArgs()
        mgr.kv_args.engine_rank, mgr.kv_args.gpu_id = 1, 1
        mgr.kv_args.kv_data_ptrs, mgr.kv_args.kv_item_lens = [1000, 2000], [64, 64]
        mgr.get_mha_kv_ptrs_with_pp = lambda src, dst: ([src[0]], [src[1]], [dst[0]], [dst[1]], 1)
        reg = registration()
        with self.assertLogs(conn.logger, level="INFO") as logs:
            mgr.add_registration(reg)
        self.assertIn(
            "CUDA IPC KV registration complete remote_gpu_id=3 handles=2",
            logs.output[0],
        )
        self.assertEqual(list(mgr.remote_registrations), [f"{socket.gethostname()}:3"])
        copied_bytes = mgr.copy_kv(
            reg,
            np.array([2, 3, 7], np.int32),
            np.array([5, 6, 9], np.int32),
        )
        self.assertEqual(copied_bytes, 384)
        self.assertEqual(len(runtime.copies), 4)
        self.assertEqual(len(runtime.opened), 1)
        self.assertEqual(runtime.copies[0], (10448, 3, 1128, 1, 128))
        mgr.close()
        self.assertEqual(runtime.closed, [10000])

    def test_rejects_invalid_layout_handle_and_bounds(self):
        reg = registration()
        cases = [
            (dataclasses.replace(reg, kv_offsets=(0,)), "allocation layout"),
            (dataclasses.replace(reg, kv_handles=(b"bad", reg.kv_handles[1])), "handle size"),
            (dataclasses.replace(reg, kv_offsets=(128, 15000)), "allocation bounds"),
        ]
        for invalid, message in cases:
            with self.subTest(message=message), self.assertRaisesRegex(RuntimeError, message):
                invalid.validate_for(socket.gethostname(), 12090, 1, 2)

    def test_sender_accumulates_bytes_and_logs_only_boundaries(self):
        reg = registration()
        key = f"{reg.hostname}:{reg.gpu_id}"

        class Manager:
            def __init__(self):
                self.transfer_infos = {
                    7: {key: (np.array([5, 6, 9], np.int32), 4)}
                }
                self.remote_registrations = {key: reg}
                self.copy_sizes = iter((128, 64))
                self.statuses = []

            def copy_kv(self, reg, src, dst):
                return next(self.copy_sizes)

            def update_status(self, room, status):
                self.statuses.append((room, status))

            def record_failure(self, room, error):
                raise AssertionError(error)

        mgr = Manager()
        sender = object.__new__(conn.CudaIpcKVSender)
        sender.kv_mgr = mgr
        sender.bootstrap_room = 7
        sender.curr_idx = 0
        sender.num_kv_indices = 3
        sender.conclude_state = None
        sender.transferred_bytes = 0
        sender.transfer_started = False
        sender._notify_done = lambda *args: None

        with self.assertLogs(conn.logger, level="INFO") as logs:
            sender.send(np.array([1, 2], np.int32))
            sender.send(np.array([3], np.int32))

        messages = "\n".join(logs.output)
        self.assertEqual(messages.count("CUDA IPC KV transfer started"), 1)
        self.assertEqual(messages.count("CUDA IPC KV transfer complete"), 1)
        self.assertIn("CUDA IPC KV transfer complete bytes=192 room=7", messages)
        self.assertEqual(mgr.statuses, [(7, conn.KVPoll.Success)])


if __name__ == "__main__": unittest.main()

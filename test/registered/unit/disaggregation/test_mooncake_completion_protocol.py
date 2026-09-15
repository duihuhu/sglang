import importlib.util
import sys
import types
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]


def load_mooncake_conn_without_sglang_dependencies():
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
    common_utils.FastQueue = type("FastQueue", (), {})

    disagg_utils = types.ModuleType("sglang.srt.disaggregation.utils")
    class DisaggregationMode: pass
    disagg_utils.DisaggregationMode = DisaggregationMode
    disagg_utils.filter_kv_indices_for_cp_rank = lambda *args: args[1:3]
    server_args = types.ModuleType("sglang.srt.server_args")
    server_args.ServerArgs = type("ServerArgs", (), {})
    stubs = {
        "sglang.srt.disaggregation.base.conn": base_conn,
        "sglang.srt.disaggregation.common.conn": common_conn,
        "sglang.srt.disaggregation.common.utils": common_utils,
        "sglang.srt.disaggregation.utils": disagg_utils,
        "sglang.srt.server_args": server_args,
        "sglang.srt.distributed.parallel_state": types.SimpleNamespace(
            get_mooncake_transfer_engine=lambda: None
        ),
        "sglang.srt.environ": types.SimpleNamespace(envs=types.SimpleNamespace()),
        "sglang.srt.utils.network": types.SimpleNamespace(
            NetworkAddress=type("NetworkAddress", (), {})
        ),
        "sglang.srt.disaggregation.mooncake.utils": types.SimpleNamespace(
            check_mooncake_custom_mem_pool_enabled=lambda: (False, None)
        ),
    }
    old = {name: sys.modules.get(name) for name in stubs}
    sys.modules.update(stubs)
    try:
        spec = importlib.util.spec_from_file_location(
            "mooncake_conn_under_test",
            ROOT / "python/sglang/srt/disaggregation/mooncake/conn.py",
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



conn = load_mooncake_conn_without_sglang_dependencies()


class TestMooncakeCompletionProtocol(unittest.TestCase):
    def test_tp2_independent_prefill_managers_complete_one_local_target(self):
        # Each TP2 prefill rank owns one destination even if bootstrap metadata is 2.
        for _prefill_rank in range(2):
            self.assertTrue(conn.MooncakeKVManager._local_transfer_is_complete(True, 1, 1))

    def test_tp1_and_multi_destination_same_rank(self):
        self.assertTrue(conn.MooncakeKVManager._local_transfer_is_complete(True, 1, 1))
        self.assertFalse(conn.MooncakeKVManager._local_transfer_is_complete(True, 1, 2))
        self.assertTrue(conn.MooncakeKVManager._local_transfer_is_complete(True, 2, 2))
        self.assertFalse(conn.MooncakeKVManager._local_transfer_is_complete(False, 2, 2))

    def test_decode_aggregates_unique_prefill_ranks(self):
        mgr = object.__new__(conn.MooncakeKVManager)
        room = 17
        mgr.request_status = {room: conn.KVPoll.WaitingForInput}
        mgr.prefill_response_tracker = {room: set()}
        mgr.required_prefill_response_num_table = {room: 2}
        updates = []
        mgr.update_status = lambda got_room, status: updates.append((got_room, status))
        mgr.record_failure = lambda *_args: self.fail("unexpected failure")

        mgr._handle_decode_transfer_status(room, conn.KVPoll.Success, 0)
        mgr._handle_decode_transfer_status(room, conn.KVPoll.Success, 0)
        self.assertEqual(updates, [])
        mgr._handle_decode_transfer_status(room, conn.KVPoll.Success, 1)
        mgr._handle_decode_transfer_status(room, conn.KVPoll.Success, 1)
        self.assertEqual(updates, [(room, conn.KVPoll.Success)])


if __name__ == "__main__":
    unittest.main()

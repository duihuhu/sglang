import importlib.util
import sys
import types
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[4]


def load_common_conn_without_sglang_dependencies():
    base_conn = types.ModuleType("sglang.srt.disaggregation.base.conn")

    class BaseKVManager:
        pass

    class BaseKVReceiver:
        pass

    class BaseKVSender:
        pass

    class BaseKVBootstrapServer:
        pass

    class KVArgs:
        pass

    class KVPoll:
        Failed, Bootstrapping, WaitingForInput, Transferring, Success = range(5)

    for name, value in locals().copy().items():
        if name in {
            "BaseKVManager",
            "BaseKVReceiver",
            "BaseKVSender",
            "BaseKVBootstrapServer",
            "KVArgs",
            "KVPoll",
        }:
            setattr(base_conn, name, value)

    disagg_utils = types.ModuleType("sglang.srt.disaggregation.utils")
    disagg_utils.DisaggregationMode = type("DisaggregationMode", (), {})
    dp_attention = types.ModuleType("sglang.srt.layers.dp_attention")
    for name in (
        "get_attention_cp_rank",
        "get_attention_dp_rank",
        "get_attention_tp_rank",
    ):
        setattr(dp_attention, name, lambda: 0)
    for name in (
        "get_attention_cp_size",
        "get_attention_dp_size",
        "get_attention_tp_size",
    ):
        setattr(dp_attention, name, lambda: 1)

    class _Env:
        def get(self):
            return 60

    envs = types.SimpleNamespace(
        SGLANG_DISAGGREGATION_ALL_CP_RANKS_TRANSFER=_Env(),
        SGLANG_DISAGGREGATION_BOOTSTRAP_TIMEOUT=_Env(),
        SGLANG_DISAGGREGATION_HEARTBEAT_INTERVAL=_Env(),
        SGLANG_DISAGGREGATION_HEARTBEAT_MAX_FAILURE=_Env(),
        SGLANG_DISAGGREGATION_WAITING_TIMEOUT=_Env(),
        SGLANG_DISAGGREGATION_BOOTSTRAP_ENTRY_CLEANUP_INTERVAL=_Env(),
    )
    network = types.ModuleType("sglang.srt.utils.network")
    network.NetworkAddress = type("NetworkAddress", (), {})
    network.get_local_ip_auto = lambda: "127.0.0.1"
    network.get_zmq_socket_on_host = lambda *args, **kwargs: (0, None)

    stubs = {
        "sglang.srt.disaggregation.base.conn": base_conn,
        "sglang.srt.disaggregation.utils": disagg_utils,
        "sglang.srt.distributed": types.SimpleNamespace(get_pp_group=lambda: None),
        "sglang.srt.environ": types.SimpleNamespace(envs=envs),
        "sglang.srt.layers.dp_attention": dp_attention,
        "sglang.srt.server_args": types.SimpleNamespace(ServerArgs=type("ServerArgs", (), {})),
        "sglang.srt.utils.network": network,
    }
    old = {name: sys.modules.get(name) for name in stubs}
    sys.modules.update(stubs)
    try:
        spec = importlib.util.spec_from_file_location(
            "common_conn_under_test",
            ROOT / "python/sglang/srt/disaggregation/common/conn.py",
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


conn = load_common_conn_without_sglang_dependencies()


class _Manager:
    def __init__(self):
        self.request_status = {}
        self.failure_records = {}
        self.prefill_info_table = {
            "prefill:8998": conn.PrefillServerInfo(
                attn_tp_size=2,
                attn_cp_size=1,
                dp_size=1,
                pp_size=1,
                page_size=1,
                kv_cache_dtype="auto",
                follow_bootstrap_room=True,
                generation=3,
                target_tp_rank=0,
                target_tp_ranks=[0],
                target_cp_ranks=[0],
                target_pp_ranks=[0],
                required_dst_info_num=1,
                required_prefill_response_num=1,
            )
        }
        self.required_prefill_response_num_table = {}
        self.connection_pool = {}
        self.is_mla_backend = False

    def update_status(self, room, status):
        self.request_status[room] = status

    def record_failure(self, room, reason):
        self.failure_records[room] = reason


class _Receiver(conn.CommonKVReceiver):
    registrations = []
    route_fetches = 0

    def _replay_registration_on_cached_route(self):
        return True

    def _get_bootstrap_info_from_server(self, *args):
        type(self).route_fetches += 1
        return {"rank_ip": "127.0.0.1", "rank_port": 41000}

    def _register_kv_args(self):
        type(self).registrations.append(
            (self.bootstrap_room, tuple(info["rank_port"] for info in self.bootstrap_infos))
        )


class TestCommonReceiverRegistrationLifecycle(unittest.TestCase):
    def setUp(self):
        _Receiver.registrations = []
        _Receiver.route_fetches = 0

    def test_each_room_replays_registration_when_endpoint_cache_is_reused(self):
        mgr = _Manager()
        first = _Receiver(mgr, "prefill:8998", bootstrap_room=101, prefill_dp_rank=0)
        second = _Receiver(mgr, "prefill:8998", bootstrap_room=102, prefill_dp_rank=0)

        self.assertEqual(first.bootstrap_infos, second.bootstrap_infos)
        self.assertEqual(_Receiver.route_fetches, 1)
        self.assertEqual(
            _Receiver.registrations,
            [(101, (41000,)), (102, (41000,))],
        )
        self.assertEqual(mgr.request_status[101], conn.KVPoll.Bootstrapping)
        self.assertEqual(mgr.request_status[102], conn.KVPoll.Bootstrapping)


if __name__ == "__main__":
    unittest.main()

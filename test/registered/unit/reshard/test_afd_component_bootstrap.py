import asyncio
import json
import unittest
from sglang.srt.disaggregation.common.conn import CommonKVBootstrapServer
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=1, suite="stage-a-cpu-only")


class _PutRequest:
    def __init__(self, payload):
        self.payload = payload

    async def json(self):
        return self.payload


class _GetRequest:
    query = {
        "prefill_dp_rank": "-1",
        "prefill_cp_rank": "-1",
        "target_tp_rank": "-1",
        "target_pp_rank": "-1",
    }


def _payload(generation, tp_size, tp_rank):
    return {
        "generation": generation,
        "attn_tp_size": tp_size,
        "attn_tp_rank": tp_rank,
        "attn_cp_size": 1,
        "attn_cp_rank": 0,
        "attn_dp_size": 1,
        "attn_dp_rank": 0,
        "pp_size": 1,
        "pp_rank": 0,
        "system_dp_size": 1,
        "system_dp_rank": 0,
        "rank_ip": "127.0.0.1",
        "rank_port": 41000 + generation * 10 + tp_rank,
        "page_size": 1,
        "kv_cache_dtype": "auto",
        "load_balance_method": "follow_bootstrap_room",
    }


def _server():
    server = object.__new__(CommonKVBootstrapServer)
    server.lock = asyncio.Lock()
    server.attn_tp_size = None
    server.attn_cp_size = None
    server.dp_size = None
    server.pp_size = None
    server.page_size = None
    server.kv_cache_dtype = None
    server.follow_bootstrap_room = None
    server.prefill_port_table = {}
    server._registered_count = 0
    server.generation = -1
    server._pending_generation = None
    server._pending_metadata = None
    server._pending_table = {}
    server._pending_keys = set()
    return server


def _json(response):
    return json.loads(response.body.decode())


class TestAFDComponentBootstrapTopology(CustomTestCase):
    def test_atomic_tp2_tp4_tp1_generations(self):
        async def scenario():
            server = _server()

            for rank in range(2):
                response = await server._handle_route_put(
                    _PutRequest(_payload(0, 2, rank))
                )
                self.assertEqual(response.status, 200)
            self.assertEqual(
                _json(await server._handle_route_get(_GetRequest()))["attn_tp_size"], 2
            )

            for rank in range(3):
                response = await server._handle_route_put(
                    _PutRequest(_payload(1, 4, rank))
                )
                self.assertFalse(_json(response)["published"])
                route = _json(await server._handle_route_get(_GetRequest()))
                self.assertEqual((route["generation"], route["attn_tp_size"]), (0, 2))
            response = await server._handle_route_put(_PutRequest(_payload(1, 4, 3)))
            self.assertTrue(_json(response)["published"])
            route = _json(await server._handle_route_get(_GetRequest()))
            self.assertEqual((route["generation"], route["attn_tp_size"]), (1, 4))
            self.assertEqual(set(server.prefill_port_table[0][0]), {0, 1, 2, 3})
            self.assertEqual(server._registered_count, 4)

            response = await server._handle_route_put(_PutRequest(_payload(2, 1, 0)))
            self.assertTrue(_json(response)["published"])
            route = _json(await server._handle_route_get(_GetRequest()))
            self.assertEqual((route["generation"], route["attn_tp_size"]), (2, 1))
            self.assertEqual(set(server.prefill_port_table[0][0]), {0})
            self.assertEqual(server._registered_count, 1)

            stale = await server._handle_route_put(_PutRequest(_payload(1, 4, 0)))
            self.assertEqual(stale.status, 409)
            self.assertEqual(server.generation, 2)

        asyncio.run(scenario())


if __name__ == "__main__":
    unittest.main(verbosity=2)

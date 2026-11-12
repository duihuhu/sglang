import ast
import os
import threading
import unittest
from pathlib import Path
from unittest.mock import patch

import torch


from sglang.srt.layers.afd import (
    BroadcastTensorCommunicator,
    ShardedZMQTensorCommunicator,
    _reassemble_tensor_shards,
    _shard_tensor_for_rank,
    _zmq_sharding_enabled,
    _zmq_double_buffer_enabled,
    _ZMQBufferSlot,
)
from sglang.test.test_utils import CustomTestCase


class TestAfdCrossNodeExperimental(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        cls.repo = Path(__file__).resolve().parents[4]

    def test_helpers_default_switch_to_zero(self):
        for relative in (
            "python/sglang/srt/layers/afd.py",
            "python/sglang/srt/layers/rdma_comm.py",
        ):
            tree = ast.parse((self.repo / relative).read_text())
            helper = next(
                node for node in tree.body
                if isinstance(node, ast.FunctionDef)
                and node.name == "_cross_node_experimental_enabled"
            )
            call = next(node for node in ast.walk(helper) if isinstance(node, ast.Call))
            self.assertEqual(call.args[0].value, "AFD_CROSS_NODE_EXPERIMENTAL")
            self.assertEqual(call.args[1].value, "0")
            comparison = next(
                node for node in ast.walk(helper) if isinstance(node, ast.Compare)
            )
            self.assertEqual(comparison.comparators[0].value, "1")

    def test_experimental_zmq_has_real_ready_handshake(self):
        afd_source = (self.repo / "python/sglang/srt/layers/afd.py").read_text()
        self.assertIn("self._get_pull_socket()", afd_source)
        self.assertIn("self._perform_ready_handshake()", afd_source)
        self.assertIn("AFD ZMQ handshake ready", afd_source)
        self.assertIn("zmq.IMMEDIATE", afd_source)

        launcher_source = (
            self.repo
            / "benchmark/AFlex_bench/multi_node/scalibility/run_aflex_32g_cross_af.py"
        ).read_text()
        self.assertIn('("AFD ZMQ handshake ready",)', launcher_source)
        self.assertNotIn("AFD ZMQ ready after HTTP health", launcher_source)

    def test_zmq_handshake_ports_are_valid_and_separate(self):
        source = (self.repo / "python/sglang/srt/layers/afd.py").read_text()
        self.assertIn('AFD_ZMQ_FFN_HANDSHAKE_BASE_PORT", "60000"', source)
        self.assertIn('AFD_ZMQ_ATTN_HANDSHAKE_BASE_PORT", "61000"', source)

    def test_32_gpu_launcher_explicitly_enables_switch(self):
        launcher = self.repo / "benchmark/AFlex_bench/multi_node/scalibility/run_aflex_32g_cross_af.py"
        source = launcher.read_text()
        self.assertIn('"AFD_CROSS_NODE_EXPERIMENTAL=1"', source)
        self.assertIn('env = [', source)

    def test_zmq_sharding_requires_master_switch(self):
        for master, sharding, expected in (("0", "0", False), ("0", "1", False), ("1", "0", False), ("1", "1", True)):
            with patch.dict(os.environ, {"AFD_CROSS_NODE_EXPERIMENTAL": master, "AFD_ZMQ_SHARDING": sharding}, clear=False):
                self.assertEqual(_zmq_sharding_enabled(), expected)

    def test_zmq_double_buffer_default_isolated(self):
        cases = (("0", "0", "1", False), ("1", "0", "1", False),
                 ("0", "1", "1", False), ("1", "1", "0", False),
                 ("1", "1", "1", True))
        for master, sharding, double_buffer, expected in cases:
            with patch.dict(os.environ, {"AFD_CROSS_NODE_EXPERIMENTAL": master,
                                         "AFD_ZMQ_SHARDING": sharding,
                                         "AFD_ZMQ_DOUBLE_BUFFER": double_buffer}):
                self.assertEqual(_zmq_double_buffer_enabled(), expected)

    def test_double_buffer_slot_alternation_and_fence(self):
        slots = [_ZMQBufferSlot(), _ZMQBufferSlot()]
        seen = []
        for sequence in range(4):
            slot = slots[sequence % 2]
            slot.acquire(sequence)
            seen.append((sequence, slots.index(slot), slot.sequence))
            slot.free.set()
        self.assertEqual(seen, [(0, 0, 0), (1, 1, 1), (2, 0, 2), (3, 1, 3)])

    def test_double_buffer_slot_resize_and_reuse(self):
        buffers = [torch.zeros(8, dtype=torch.uint8), torch.zeros(16, dtype=torch.uint8)]
        with patch("torch.empty", side_effect=buffers):
            slot = _ZMQBufferSlot()
            first = slot.resize(8)
            self.assertIs(slot.resize(4), first)
            grown = slot.resize(16)
            self.assertIsNot(grown, first)
            self.assertEqual(slot.capacity, 16)

    def test_launcher_double_buffer_contract(self):
        source = (self.repo / "benchmark/AFlex_bench/multi_node/scalibility/run_aflex_32g_cross_af.py").read_text()
        self.assertIn('"--zmq-double-buffer"', source)
        self.assertIn('AFD_ZMQ_DOUBLE_BUFFER={int(ZMQ_DOUBLE_BUFFER)}', source)
        self.assertIn('--zmq-double-buffer requires --zmq-sharding', source)
        self.assertIn('"zmq_double_buffer":ZMQ_DOUBLE_BUFFER', source)

    def test_shard_padding_and_reassemble_bf16(self):
        full = torch.arange(15, dtype=torch.float32).reshape(5, 3).to(torch.bfloat16)
        shards = [_shard_tensor_for_rank(full, rank, 4)[0] for rank in range(4)]
        self.assertEqual([tuple(x.shape) for x in shards], [(2, 3)] * 4)
        self.assertTrue(torch.equal(_reassemble_tensor_shards(shards, 5), full))

    def test_num_tokens_less_than_tp(self):
        full = torch.arange(6, dtype=torch.float32).reshape(2, 3)
        shards = [_shard_tensor_for_rank(full, rank, 4)[0] for rank in range(4)]
        self.assertTrue(torch.equal(_reassemble_tensor_shards(shards, 2), full))

    def test_selection_types_are_distinct(self):
        self.assertFalse(
            issubclass(ShardedZMQTensorCommunicator, BroadcastTensorCommunicator)
        )

    def test_ports_use_local_rank(self):
        from sglang.srt.layers.afd import AFDPerspective, ZMQSimpleTensorCommunicator

        comm = ZMQSimpleTensorCommunicator.__new__(ZMQSimpleTensorCommunicator)
        comm.start_lport = 47000
        comm.start_dport = 47100
        comm._afd_perspective = AFDPerspective.AFD_PERSPECTIVE_ATTN
        with patch.dict(os.environ, {"LOCAL_RANK": "3"}, clear=False):
            self.assertEqual(comm._get_lport(), 47004)
            self.assertEqual(comm._get_dport(), 47104)
            self.assertEqual(comm._get_handshake_lport(), 60004)
            self.assertEqual(comm._get_handshake_dport(), 61004)

    def test_launcher_injects_sharding_and_waits_all_ranks(self):
        source = (self.repo / "benchmark/AFlex_bench/multi_node/scalibility/run_aflex_32g_cross_af.py").read_text()
        self.assertIn('f"AFD_ZMQ_SHARDING={int(ZMQ_SHARDING)}"', source)
        self.assertIn('ready_ranks >= set(range(TP))', source)
        self.assertIn('"--zmq-sharding"', source)

    def test_peek_async_communicator_cache_paths_are_side_effect_free(self):
        import sglang.srt.layers.afd as afd

        legacy = object()
        override = object()
        experimental = object()
        pooled = object()
        perspective = afd.AFDPerspective.AFD_PERSPECTIVE_FFN
        server_args = type(
            "Args",
            (),
            {"afd_multi_pf_continuation": False, "afd_pf_group_id": 3},
        )()

        with patch.object(afd, "get_global_server_args", return_value=server_args), \
                patch.object(afd, "get_afd_perspective", return_value=perspective), \
                patch.object(afd, "get_tensor_communicator") as tensor_constructor, \
                patch.object(afd, "get_afd_peer_channel_pool") as pool_constructor:
            with patch.object(afd, "_per_mb_async_override", override):
                self.assertIs(afd.peek_async_communicator(), override)

            with patch.object(afd, "_per_mb_async_override", None), \
                    patch.object(afd, "_async_communicator", legacy), \
                    patch.dict(os.environ, {"AFD_CROSS_NODE_EXPERIMENTAL": "0"}):
                self.assertIs(afd.peek_async_communicator(), legacy)
                self.assertIs(afd.get_existing_async_communicator(), legacy)

            with patch.object(afd, "_per_mb_async_override", None), \
                    patch.dict(os.environ, {"AFD_CROSS_NODE_EXPERIMENTAL": "1"}), \
                    patch.dict(
                        afd._experimental_async_communicators,
                        {perspective: experimental},
                        clear=True,
                    ):
                self.assertIs(afd.peek_async_communicator(), experimental)

            server_args.afd_multi_pf_continuation = True
            pool = {3: type("Channel", (), {"async_comm": pooled})()}
            with patch.object(afd, "_per_mb_async_override", None), \
                    patch.object(afd, "_afd_peer_channel_pool", pool):
                self.assertIs(afd.peek_async_communicator(), pooled)

            tensor_constructor.assert_not_called()
            pool_constructor.assert_not_called()

    def test_peek_async_communicator_returns_none_without_construction(self):
        import sglang.srt.layers.afd as afd

        perspective = afd.AFDPerspective.AFD_PERSPECTIVE_ATTN
        server_args = type(
            "Args",
            (),
            {"afd_multi_pf_continuation": False, "afd_pf_group_id": 0},
        )()
        with patch.object(afd, "get_global_server_args", return_value=server_args), \
                patch.object(afd, "get_afd_perspective", return_value=perspective), \
                patch.object(afd, "_per_mb_async_override", None), \
                patch.object(afd, "_async_communicator", None), \
                patch.dict(os.environ, {"AFD_CROSS_NODE_EXPERIMENTAL": "0"}), \
                patch.object(afd, "get_tensor_communicator") as tensor_constructor:
            self.assertIsNone(afd.peek_async_communicator())
            tensor_constructor.assert_not_called()

        server_args.afd_multi_pf_continuation = True
        with patch.object(afd, "get_global_server_args", return_value=server_args), \
                patch.object(afd, "_per_mb_async_override", None), \
                patch.object(afd, "_afd_peer_channel_pool", None), \
                patch.object(afd, "get_afd_peer_channel_pool") as pool_constructor:
            self.assertIsNone(afd.peek_async_communicator())
            pool_constructor.assert_not_called()

        server_args.afd_multi_pf_continuation = False
        with patch.object(afd, "get_global_server_args", return_value=server_args), \
                patch.object(afd, "get_afd_perspective", return_value=perspective), \
                patch.object(afd, "_per_mb_async_override", None), \
                patch.dict(os.environ, {"AFD_CROSS_NODE_EXPERIMENTAL": "1"}), \
                patch.dict(afd._experimental_async_communicators, {}, clear=True), \
                patch.object(afd, "get_tensor_communicator") as tensor_constructor:
            self.assertIsNone(afd.peek_async_communicator())
            tensor_constructor.assert_not_called()

    def test_experimental_communicator_singleton_all_entrypoints(self):
        import sglang.srt.layers.afd as afd

        perspective = afd.AFDPerspective.AFD_PERSPECTIVE_ATTN
        created = []

        class FakeCommunicator:
            def close(self):
                pass

        def factory():
            comm = FakeCommunicator()
            created.append(comm)
            return comm

        afd.reset_afd_communicators()
        server_args = type("Args", (), {"afd_multi_pf_continuation": False})()
        with patch.dict(os.environ, {"AFD_CROSS_NODE_EXPERIMENTAL": "1"}), \
                patch.object(afd, "get_global_server_args", return_value=server_args), \
                patch.object(afd, "get_afd_perspective", return_value=perspective), \
                patch.object(afd, "_create_tensor_communicator", side_effect=factory):
            tensor_a = afd.get_tensor_communicator()
            tensor_b = afd.get_tensor_communicator()
            async_a = afd.get_async_communicator()
            async_b = afd.get_async_communicator()
        self.assertIs(tensor_a, tensor_b)
        self.assertIs(async_a, async_b)
        self.assertIs(async_a.inner, tensor_a)
        self.assertEqual(len(created), 1)
        afd.reset_afd_communicators()

    def test_experimental_communicator_singleton_is_thread_safe(self):
        import sglang.srt.layers.afd as afd

        perspective = afd.AFDPerspective.AFD_PERSPECTIVE_ATTN
        created = []
        barrier = threading.Barrier(8)

        class FakeCommunicator:
            def close(self):
                pass

        def factory():
            created.append(FakeCommunicator())
            return created[-1]

        def get_one(results):
            barrier.wait()
            results.append(afd.get_tensor_communicator())

        afd.reset_afd_communicators()
        results = []
        with patch.dict(os.environ, {"AFD_CROSS_NODE_EXPERIMENTAL": "1"}), \
                patch.object(afd, "get_afd_perspective", return_value=perspective), \
                patch.object(afd, "_create_tensor_communicator", side_effect=factory):
            threads = [threading.Thread(target=get_one, args=(results,)) for _ in range(8)]
            for thread in threads:
                thread.start()
            for thread in threads:
                thread.join()
        self.assertEqual(len(created), 1)
        self.assertEqual(len({id(comm) for comm in results}), 1)
        afd.reset_afd_communicators()

    def test_reset_closes_experimental_communicator(self):
        import sglang.srt.layers.afd as afd

        perspective = afd.AFDPerspective.AFD_PERSPECTIVE_FFN
        closed = []

        class FakeCommunicator:
            def close(self):
                closed.append(self)

        afd.reset_afd_communicators()
        server_args = type("Args", (), {"afd_multi_pf_continuation": False})()
        with patch.dict(os.environ, {"AFD_CROSS_NODE_EXPERIMENTAL": "1"}), \
                patch.object(afd, "get_global_server_args", return_value=server_args), \
                patch.object(afd, "get_afd_perspective", return_value=perspective), \
                patch.object(afd, "_create_tensor_communicator", return_value=FakeCommunicator()):
            comm = afd.get_tensor_communicator()
            afd.get_async_communicator()
            afd.reset_afd_communicators()
        self.assertEqual(closed, [comm])


if __name__ == "__main__":
    unittest.main(verbosity=3)

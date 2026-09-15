import os
import unittest
from unittest.mock import Mock, patch

from sglang.srt.layers.afd import ZMQSimpleTensorCommunicator
from sglang.srt.layers.afd_type import AFDPerspective
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=1, suite="stage-a-cpu-only")


class TestAFDSharedZMQ(CustomTestCase):
    def test_stream_ordered_interface_delegates_to_fifo_transfer(self):
        comm = ZMQSimpleTensorCommunicator.__new__(ZMQSimpleTensorCommunicator)
        tensor = object()
        received = object()
        comm.send_tensor = Mock()
        comm.recv_tensor = Mock(return_value=received)

        self.assertIsNone(comm.send_stream_ordered(tensor))
        self.assertIs(comm.recv_stream_ordered(), received)
        comm.send_tensor.assert_called_once_with(tensor)
        comm.recv_tensor.assert_called_once_with()

    def test_explicit_ports_and_listener_direction(self):
        with patch("sglang.srt.layers.afd.zmq.Context"):
            attn = ZMQSimpleTensorCommunicator(
                AFDPerspective.AFD_PERSPECTIVE_ATTN,
                peer_host="ffn-host",
                ffn_base_port=40000,
                attn_base_port=41000,
                ffn_handshake_base_port=42000,
                attn_handshake_base_port=43000,
                timeout_ms=17,
            )
            self.assertEqual(attn._get_lport(), 40001)  # receives F->A
            self.assertEqual(attn._get_dport(), 41001)  # sends A->F
            self.assertEqual(attn.peer_host, "ffn-host")
            self.assertEqual(attn.socket_timeout_ms, 17)
            ffn = ZMQSimpleTensorCommunicator(
                AFDPerspective.AFD_PERSPECTIVE_FFN,
                peer_host="attn-host",
                ffn_base_port=40000,
                attn_base_port=41000,
                ffn_handshake_base_port=42000,
                attn_handshake_base_port=43000,
            )
            self.assertEqual(ffn._get_lport(), 41001)  # receives A->F
            self.assertEqual(ffn._get_dport(), 40001)  # sends F->A

    def test_environment_backwards_compatibility(self):
        env = {
            "AFD_ZMQ_PEER_HOST": "legacy-host",
            "AFD_FFN_BASE_PORT": "50100",
            "AFD_ATTN_BASE_PORT": "50200",
            "AFD_ZMQ_TIMEOUT_MS": "123",
        }
        with patch.dict(os.environ, env), patch("sglang.srt.layers.afd.zmq.Context"):
            comm = ZMQSimpleTensorCommunicator(AFDPerspective.AFD_PERSPECTIVE_ATTN)
            self.assertEqual(comm.peer_host, "legacy-host")
            self.assertEqual(comm._get_lport(), 50101)
            self.assertEqual(comm._get_dport(), 50201)
            self.assertEqual(comm.socket_timeout_ms, 123)


if __name__ == "__main__":
    unittest.main(verbosity=3)

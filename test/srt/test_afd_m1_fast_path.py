"""CPU-only regression tests for the M=1 AFD IPC fast path."""

import os
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.srt.layers import afd
from sglang.test.test_utils import CustomTestCase


class _FakeLayerCommunicator:
    def __init__(self, layer_id, calls):
        self.layer_id = layer_id
        self.calls = calls

    def prepare_attn(self, hidden_states, residual, forward_batch):
        self.calls.append(("prepare_attn", self.layer_id))
        return hidden_states, residual

    def prepare_mlp(self, hidden_states, residual, forward_batch):
        self.calls.append(("prepare_mlp", self.layer_id))
        return hidden_states, residual

    def postprocess_layer(self, hidden_states, residual, forward_batch):
        self.calls.append(("postprocess", self.layer_id))
        return hidden_states, residual


class _FakeLayer:
    def __init__(self, layer_id, calls):
        self.layer_id = layer_id
        self.calls = calls
        inner = _FakeLayerCommunicator(layer_id, calls)
        self.layer_communicator = SimpleNamespace(layer_communicator=inner)

    def _run_attn(self, positions, hidden_states, forward_batch):
        self.calls.append(("attn", self.layer_id))
        return hidden_states + 1

    def _run_mlp(self, hidden_states, forward_batch):
        self.calls.append(("mlp", self.layer_id))
        return hidden_states + 1


class _FakeIpc:
    def __init__(self):
        self.calls = []
        self.last_sent = None

    def send_tensor(self, tensor):
        self.calls.append("send")
        self.last_sent = tensor

    def recv_tensor(self):
        self.calls.append("recv")
        return self.last_sent if self.last_sent is not None else torch.zeros(1, 1)

    def send_tensor_gpu_only(self, tensor):
        self.calls.append("send_gpu")
        self.last_sent = tensor

    def recv_tensor_gpu_only(self):
        self.calls.append("recv_gpu")
        return self.last_sent if self.last_sent is not None else torch.zeros(1, 1)


class _FakeAsyncCommunicator:
    _RING_SIZE = 3

    def __init__(self, inner):
        self.inner = inner

    def drain_recvs(self):
        pass

    def drain_sends(self):
        pass


class TestAfdM1IpcFastPath(CustomTestCase):
    def _run(self, is_attn, gpu_only, experimental=True):
        layer_calls = []
        layers = [_FakeLayer(0, layer_calls), _FakeLayer(1, layer_calls)]
        ipc = _FakeIpc()
        async_comm = _FakeAsyncCommunicator(ipc)
        forward_batch = SimpleNamespace(afd_children=None)
        inputs = [{
            "hidden_states": torch.zeros(1, 1),
            "residual": None,
            "positions": torch.zeros(1, dtype=torch.long),
            "forward_batch": forward_batch,
        }]
        server_args = SimpleNamespace(
            afd_async_schedule=False,
            afd_async_pipeline=False,
        )
        env = {
            "AFD_CROSS_NODE_EXPERIMENTAL": "1" if experimental else "0",
            "AFD_GPU_ONLY_IPC": "1" if gpu_only else "0",
            "AFD_FUSED_PIPELINE": "0",
            "SGLANG_LAYER_PROFILE": "0",
            "AFD_DETAILED_TIMING": "0",
        }
        with (
            patch.dict(os.environ, env),
            patch.object(afd, "model_forward_afd_split_inputs", return_value=inputs),
            patch.object(afd, "get_async_communicator", return_value=async_comm),
            patch.object(afd, "get_global_server_args", return_value=server_args),
            patch.object(afd, "afd_is_attn", return_value=is_attn),
            patch.object(afd, "afd_is_ffn", return_value=not is_attn),
        ):
            afd.model_forward_afd(
                layers,
                inputs[0]["positions"],
                forward_batch,
                inputs[0]["hidden_states"],
                None,
                None,
            )
        return layer_calls, ipc.calls

    def test_default_switch_preserves_legacy_layers_and_gpu_ipc(self):
        with patch.dict(os.environ, {}, clear=True):
            self.assertFalse(afd._cross_node_experimental_enabled())
        layer_calls, ipc_calls = self._run(
            is_attn=True, gpu_only=True, experimental=False
        )
        self.assertEqual([i for op, i in layer_calls if op == "attn"], [1])
        self.assertEqual(ipc_calls, ["send_gpu", "recv_gpu"])

    def test_attn_executes_layer_zero_and_warms_up_metadata(self):
        layer_calls, ipc_calls = self._run(is_attn=True, gpu_only=True)
        self.assertIn(("attn", 0), layer_calls)
        self.assertEqual(ipc_calls, ["send", "recv", "send_gpu", "recv_gpu"])

    def test_ffn_executes_layer_zero_and_warms_up_metadata(self):
        layer_calls, ipc_calls = self._run(is_attn=False, gpu_only=True)
        self.assertIn(("mlp", 0), layer_calls)
        self.assertEqual(ipc_calls, ["recv", "send", "recv_gpu", "send_gpu"])

    def test_normal_ipc_executes_all_layers(self):
        attn_calls, attn_ipc_calls = self._run(is_attn=True, gpu_only=False)
        ffn_calls, ffn_ipc_calls = self._run(is_attn=False, gpu_only=False)
        self.assertEqual([i for op, i in attn_calls if op == "attn"], [0, 1])
        self.assertEqual([i for op, i in ffn_calls if op == "mlp"], [0, 1])
        self.assertEqual(attn_ipc_calls, ["send", "recv", "send", "recv"])
        self.assertEqual(ffn_ipc_calls, ["recv", "send", "recv", "send"])


if __name__ == "__main__":
    unittest.main(verbosity=2)

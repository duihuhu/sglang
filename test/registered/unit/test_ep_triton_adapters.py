"""Unit tests for dedicated EP dispatcher to Triton adapters."""

import unittest
from unittest.mock import patch

import torch

from sglang.srt.layers.moe.moe_runner.base import MoeRunnerConfig
from sglang.srt.layers.moe.moe_runner.triton import (
    TritonMoeQuantInfo,
    TritonRunnerOutput,
    _map_global_expert_ids_to_local,
    post_permute_triton_to_deepep_normal,
    post_permute_triton_to_flashinfer,
    pre_permute_deepep_normal_to_triton,
    pre_permute_flashinfer_to_triton,
)
from sglang.srt.layers.moe.token_dispatcher.deepep import DeepEPNormalDispatchOutput
from sglang.srt.layers.moe.token_dispatcher.flashinfer import FlashinferDispatchOutput
from sglang.srt.layers.moe.topk import StandardTopKOutput
from sglang.srt.server_args import ServerArgs, set_global_server_args_for_scheduler
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=5, suite="stage-b-test-small-1-gpu")


class TestDedicatedEPToTritonAdapters(CustomTestCase):
    def setUp(self):
        set_global_server_args_for_scheduler(
            ServerArgs(model_path="dummy", skip_tokenizer_init=True)
        )
        if not torch.cuda.is_available():
            self.skipTest("CUDA is required for Triton adapter tests")
        self.device = torch.device("cuda")
        self.config = MoeRunnerConfig(
            num_experts=8,
            num_local_experts=2,
            hidden_size=4,
            intermediate_size_per_partition=8,
            top_k=3,
            params_dtype=torch.bfloat16,
            no_combine=False,
        )
        self.quant = TritonMoeQuantInfo(
            w13_weight=torch.empty(2, 16, 4, dtype=torch.bfloat16, device=self.device),
            w2_weight=torch.empty(2, 4, 8, dtype=torch.bfloat16, device=self.device),
        )

    @patch(
        "sglang.srt.distributed.parallel_state.get_moe_expert_parallel_rank",
        return_value=1,
    )
    def test_global_ids_map_to_local_ids(self, _):
        ids = torch.tensor(
            [[1, 2, 3], [4, -1, 2]], dtype=torch.int32, device=self.device
        )
        weights = torch.tensor([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]], device=self.device)
        local_ids, local_weights = _map_global_expert_ids_to_local(
            ids, weights, self.config
        )
        torch.testing.assert_close(
            local_ids,
            torch.tensor(
                [[-1, 0, 1], [-1, -1, 0]], dtype=torch.int32, device=self.device
            ),
        )
        torch.testing.assert_close(
            local_weights,
            torch.tensor([[0.0, 0.2, 0.3], [0.0, 0.0, 0.6]], device=self.device),
        )

    @patch(
        "sglang.srt.distributed.parallel_state.get_moe_expert_parallel_rank",
        return_value=0,
    )
    def test_flashinfer_adapter_preserves_fixed_slots_and_dummy(self, _):
        hidden = torch.zeros(4, 4, dtype=torch.bfloat16, device=self.device)
        ids = torch.tensor(
            [[0, 1, 6], [-1, -1, -1], [2, 3, 7], [1, 4, 5]],
            dtype=torch.int32,
            device=self.device,
        )
        weights = torch.ones(4, 3, dtype=torch.float32, device=self.device)
        dispatch = FlashinferDispatchOutput(
            hidden_states=hidden,
            hidden_states_scale=None,
            topk_output=StandardTopKOutput(weights, ids, None),
            moe_output=None,
        )
        state = {}
        runner_input = pre_permute_flashinfer_to_triton(
            dispatch, self.quant, self.config, state
        )
        self.assertEqual(runner_input.hidden_states.shape, hidden.shape)
        self.assertTrue(torch.all(runner_input.topk_ids[1] == -1))
        self.assertTrue(torch.all(runner_input.topk_weights[1] == 0))
        combine = post_permute_triton_to_flashinfer(
            TritonRunnerOutput(hidden_states=torch.zeros_like(hidden)),
            self.quant,
            self.config,
            state,
        )
        self.assertEqual(combine.hidden_states.shape, hidden.shape)

    def test_flashinfer_adapter_rejects_quantized_dispatch(self):
        dispatch = FlashinferDispatchOutput(
            hidden_states=torch.zeros(1, 4, dtype=torch.bfloat16, device=self.device),
            hidden_states_scale=torch.ones(1, 1, device=self.device),
            topk_output=StandardTopKOutput(
                torch.ones(1, 3, device=self.device),
                torch.zeros(1, 3, dtype=torch.int32, device=self.device),
                None,
            ),
            moe_output=None,
        )
        with self.assertRaisesRegex(ValueError, "BF16/FP16"):
            pre_permute_flashinfer_to_triton(dispatch, self.quant, self.config, {})

    @patch(
        "sglang.srt.distributed.parallel_state.get_moe_expert_parallel_rank",
        return_value=0,
    )
    def test_deepep_normal_adapter_maps_global_ids(self, _):
        hidden = torch.randn(3, 4, dtype=torch.bfloat16, device=self.device)
        ids = torch.tensor(
            [[0, 1, 2], [3, 4, 5], [6, 7, -1]],
            dtype=torch.int32,
            device=self.device,
        )
        weights = torch.tensor(
            [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6], [0.7, 0.8, 0.9]],
            device=self.device,
        )
        dispatch = DeepEPNormalDispatchOutput(
            hidden_states=hidden,
            hidden_states_scale=None,
            topk_ids=ids,
            topk_weights=weights,
            num_recv_tokens_per_expert=[1, 1],
        )
        state = {}
        runner_input = pre_permute_deepep_normal_to_triton(
            dispatch, self.quant, self.config, state
        )
        torch.testing.assert_close(
            runner_input.topk_ids,
            torch.tensor(
                [[0, 1, -1], [-1, -1, -1], [-1, -1, -1]],
                dtype=torch.int32,
                device=self.device,
            ),
        )
        torch.testing.assert_close(
            runner_input.topk_weights,
            torch.tensor(
                [[0.1, 0.2, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
                device=self.device,
            ),
        )
        combine = post_permute_triton_to_deepep_normal(
            TritonRunnerOutput(hidden_states=torch.zeros_like(hidden)),
            self.quant,
            self.config,
            state,
        )
        self.assertEqual(combine.topk_ids.shape, ids.shape)

    def test_deepep_normal_adapter_rejects_fp8_dispatch(self):
        dispatch = DeepEPNormalDispatchOutput(
            hidden_states=torch.zeros(1, 4, dtype=torch.bfloat16, device=self.device),
            hidden_states_scale=torch.ones(1, 1, device=self.device),
            topk_ids=torch.zeros(1, 3, dtype=torch.int32, device=self.device),
            topk_weights=torch.ones(1, 3, device=self.device),
            num_recv_tokens_per_expert=[0, 0],
        )
        with self.assertRaisesRegex(ValueError, "BF16/FP16"):
            pre_permute_deepep_normal_to_triton(dispatch, self.quant, self.config, {})


if __name__ == "__main__":
    unittest.main(verbosity=3)

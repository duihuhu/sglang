"""Manual tests for Ampere EP (FlashInfer MoeAlltoAll + Triton) on SM80+."""

import unittest

import torch

from sglang.srt.distributed import init_distributed_environment
from sglang.srt.distributed.parallel_state import (
    get_tp_group,
    initialize_model_parallel,
)
from sglang.srt.layers.dp_attention import set_dp_buffer_len
from sglang.srt.layers.moe.token_dispatcher.ampere_ep import AmpereEPDispatcher
from sglang.srt.layers.moe.utils import initialize_moe_config
from sglang.srt.server_args import ServerArgs, set_global_server_args_for_scheduler
from sglang.test.test_utils import CustomTestCase


class TestAmpereEPDispatcher(CustomTestCase):

    @classmethod
    def setUpClass(cls):
        server_args = ServerArgs(model_path="dummy")
        server_args.moe_runner_backend = "triton"
        server_args.moe_a2a_backend = "ampere_ep"
        set_global_server_args_for_scheduler(server_args)
        initialize_moe_config(server_args)

        init_distributed_environment(
            world_size=-1,
            rank=-1,
            local_rank=-1,
            backend="nccl",
        )
        world_size = torch.distributed.get_world_size()
        rank = torch.distributed.get_rank()
        device = torch.device(f"cuda:{rank % torch.cuda.device_count()}")
        torch.cuda.set_device(device)
        initialize_model_parallel(
            tensor_model_parallel_size=world_size, expert_model_parallel_size=world_size
        )

    @classmethod
    def tearDownClass(cls):
        if torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()

    def create_dispatcher(
        self, router_topk=2, num_experts=8, num_local_experts=4, hidden_size=128
    ):
        return AmpereEPDispatcher(
            group=get_tp_group().device_group,
            router_topk=router_topk,
            num_experts=num_experts,
            num_local_experts=num_local_experts,
            hidden_size=hidden_size,
            params_dtype=torch.bfloat16,
        )

    def test_dispatch_basic(self):
        num_tokens = 16
        hidden_size = 128
        router_topk = 1
        world_size = torch.distributed.get_world_size()
        rank = torch.distributed.get_rank()
        num_experts = world_size
        num_local_experts = 1

        set_dp_buffer_len(
            global_dp_buffer_len=num_tokens * world_size,
            local_dp_buffer_len=num_tokens,
            dp_max_padding=True,
            global_num_tokens=None,
        )

        hidden_states = torch.full(
            (num_tokens, hidden_size), 100.0 + rank, dtype=torch.bfloat16, device="cuda"
        )
        target_expert = (rank + 1) % world_size
        topk_ids = torch.full(
            (num_tokens, router_topk), target_expert, dtype=torch.int32, device="cuda"
        )
        topk_weights = torch.ones(
            (num_tokens, router_topk), dtype=torch.float32, device="cuda"
        )

        from sglang.srt.layers.moe.topk import StandardTopKOutput

        topk_output = StandardTopKOutput(
            topk_weights=topk_weights, topk_ids=topk_ids, router_logits=None
        )

        torch.distributed.barrier()
        dispatcher = self.create_dispatcher(
            router_topk=router_topk,
            num_experts=num_experts,
            num_local_experts=num_local_experts,
            hidden_size=hidden_size,
        )
        dispatcher.set_quant_config({"input_global_scale": None})

        dispatch_output = dispatcher.dispatch(hidden_states, topk_output)
        received_hidden_states = dispatch_output.hidden_states
        self.assertEqual(dispatch_output.hidden_states_scale, None)
        self.assertEqual(
            received_hidden_states.shape[0],
            num_tokens * world_size,
        )

        expected_source_rank = (rank - 1 + world_size) % world_size
        self.assertTrue(
            torch.all(
                received_hidden_states[
                    expected_source_rank
                    * num_tokens : (expected_source_rank + 1)
                    * num_tokens
                ]
                == 100.0 + expected_source_rank
            )
        )


if __name__ == "__main__":
    unittest.main()

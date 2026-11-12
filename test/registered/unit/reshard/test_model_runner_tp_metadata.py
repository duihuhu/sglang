import unittest
from types import SimpleNamespace

import torch

from sglang.srt.model_executor.model_runner import ModelRunner
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=1, suite="stage-a-cpu-only")


class _Attention(torch.nn.Module):
    def __init__(self, tp):
        super().__init__()
        self.total_num_heads = 8
        self.total_num_kv_heads = 4
        self.head_dim = 2
        self.num_heads = 8 // tp
        self.num_kv_heads = max(1, 4 // tp)
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim
        self.attn = torch.nn.Module()
        self.attn.tp_q_head_num = self.num_heads
        self.attn.tp_k_head_num = self.num_kv_heads
        self.attn.tp_v_head_num = self.num_kv_heads


class _QKVLinear(torch.nn.Module):
    def __init__(self, tp):
        super().__init__()
        self.tp_size = tp
        self.tp_rank = 0
        self.total_num_heads = 8
        self.total_num_kv_heads = 4
        self.head_size = 2
        self.v_head_size = 2
        self.num_heads = 8 // tp
        self.num_kv_heads = max(1, 4 // tp)
        self.num_kv_head_replicas = tp // 4 if tp >= 4 else 1
        self.q_proj_shard_size = self.num_heads * self.head_size
        self.kv_proj_shard_size = self.num_kv_heads * self.head_size
        self.v_proj_shard_size = self.num_kv_heads * self.v_head_size
        self.output_sizes = [8 * 2, 4 * 2, 4 * 2]
        self.output_size = sum(self.output_sizes)
        self.output_partition_sizes = [size // tp for size in self.output_sizes]
        self.output_size_per_partition = sum(self.output_partition_sizes)


class _RowLinear(torch.nn.Module):
    def __init__(self, tp):
        super().__init__()
        self.tp_size = tp
        self.tp_rank = 0
        self.input_size = 16
        self.output_size = 8
        self.input_size_per_partition = self.input_size // tp
        self.reduce_results = True


class _MetadataModel(torch.nn.Module):
    def __init__(self, tp):
        super().__init__()
        self.attention = _Attention(tp)
        self.qkv = _QKVLinear(tp)
        self.row = _RowLinear(tp)


def _runner(tp):
    runner = object.__new__(ModelRunner)
    runner.model = _MetadataModel(tp)
    runner.model_config = SimpleNamespace(get_num_kv_heads=lambda size: max(1, 4 // size))
    runner.token_to_kv_pool = SimpleNamespace(
        head_num=max(1, 4 // tp), head_dim=2, row_dim=max(1, 4 // tp) * 2
    )
    runner.tp_rank = 0
    runner.tp_size = tp
    runner._pre_reshard_tp = tp
    return runner


class TestModelRunnerTPMetadata(CustomTestCase):
    def _assert_tp_metadata(self, old_tp, new_tp):
        runner = _runner(old_tp)
        runner._update_model_tp_metadata(new_tp)

        attention = runner.model.attention
        self.assertEqual(attention.num_heads, 8 // new_tp)
        self.assertEqual(attention.num_kv_heads, max(1, 4 // new_tp))
        self.assertEqual(attention.q_size, attention.num_heads * 2)
        self.assertEqual(attention.kv_size, attention.num_kv_heads * 2)
        self.assertEqual(attention.attn.tp_q_head_num, attention.num_heads)
        self.assertEqual(attention.attn.tp_k_head_num, attention.num_kv_heads)

        qkv = runner.model.qkv
        self.assertEqual(qkv.tp_size, new_tp)
        self.assertEqual(qkv.q_proj_shard_size, (8 // new_tp) * 2)
        self.assertEqual(qkv.kv_proj_shard_size, max(1, 4 // new_tp) * 2)
        self.assertEqual(qkv.output_sizes, [16, 8, 8])
        self.assertEqual(
            qkv.output_partition_sizes, [16 // new_tp, 8 // new_tp, 8 // new_tp]
        )
        self.assertEqual(qkv.output_size_per_partition, 32 // new_tp)

        self.assertEqual(runner.model.row.input_size_per_partition, 16 // new_tp)
        self.assertEqual(runner.token_to_kv_pool.head_num, max(1, 4 // new_tp))
        self.assertEqual(runner.token_to_kv_pool.row_dim, max(1, 4 // new_tp) * 2)

    def test_tp4_to_tp1_recomputes_full_dimensions(self):
        self._assert_tp_metadata(4, 1)

    def test_tp2_to_tp4_recomputes_partition_dimensions(self):
        self._assert_tp_metadata(2, 4)

    def test_component_demote_uses_component_standby_flag(self):
        runner = object.__new__(ModelRunner)
        runner.model = torch.nn.Linear(2, 2)
        runner.device = "cpu"
        runner.is_afd_component_standby_rank = False
        runner._free_inplace_reshard_kv_pools = lambda: None

        runner.afd_component_demote_rank("attn")

        self.assertIsNone(runner.model)
        self.assertTrue(runner.is_afd_component_standby_rank)
        self.assertFalse(hasattr(runner, "is_inplace_standby_rank"))


if __name__ == "__main__":
    unittest.main(verbosity=3)

import unittest
from types import SimpleNamespace

import torch

from sglang.srt.layers.afd_type import AFDPerspective
from sglang.srt.mem_cache.allocator import AFDFFNNoKVPoolAllocator
from sglang.srt.mem_cache.memory_pool import AFDFFNReqToTokenPool
from sglang.srt.model_executor.model_runner import ModelRunner
from sglang.srt.model_executor.model_runner_kv_cache_mixin import (
    ModelRunnerKVCacheMixin,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=2, suite="stage-a-cpu-only")


class _FakeFFNRunner(ModelRunnerKVCacheMixin):
    def __init__(self):
        self.server_args = SimpleNamespace(
            afd_perspective=AFDPerspective.AFD_PERSPECTIVE_FFN,
            max_total_tokens=None,
            max_prefill_tokens=64,
            max_running_requests=8,
            mem_fraction_static=0.72,
            enable_memory_saver=False,
            kv_cache_dtype="auto",
        )
        self.model_config = SimpleNamespace(context_len=128)
        self.model = SimpleNamespace(quant_config=None)
        self.dp_size = 1
        self.device = "cpu"
        self.dtype = torch.bfloat16
        self.page_size = 1
        self.is_hybrid_swa = False
        self.kv_cache_dtype = torch.float16
        self.req_to_token_pool = None
        self.token_to_kv_pool = None
        self.token_to_kv_pool_allocator = None
        self.tp_size = 2
        self.metadata_updates = []

    def profile_max_num_token(self, *args, **kwargs):
        raise AssertionError("AFD FFN must not profile KV capacity")

    def _update_model_tp_metadata(self, new_tp):
        self.metadata_updates.append(new_tp)

    configure_kv_cache_dtype = ModelRunner.configure_kv_cache_dtype


class TestAFDFFNNoKVRuntime(CustomTestCase):
    def test_init_skips_profile_and_physical_kv(self):
        runner = _FakeFFNRunner()
        runner.init_memory_pool(pre_model_load_memory=0)

        self.assertEqual(runner.max_total_num_tokens, 8 * 128)
        self.assertEqual(runner.max_running_requests, 8)
        self.assertIsInstance(runner.req_to_token_pool, AFDFFNReqToTokenPool)
        self.assertEqual(runner.req_to_token_pool.req_to_token.shape, (8, 132))
        self.assertEqual(
            runner.req_to_token_pool.req_to_token.untyped_storage().nbytes(), 4
        )
        self.assertIsNone(runner.token_to_kv_pool)
        self.assertIsInstance(
            runner.token_to_kv_pool_allocator, AFDFFNNoKVPoolAllocator
        )
        self.assertIsNone(runner.token_to_kv_pool_allocator.get_kvcache())

    def test_virtual_allocator_preserves_scheduler_accounting(self):
        allocator = AFDFFNNoKVPoolAllocator(
            size=16, page_size=1, dtype=torch.float16, device="cpu"
        )
        locations = allocator.alloc(6)
        self.assertEqual(locations.tolist(), [0] * 6)
        self.assertEqual(allocator.available_size(), 10)
        allocator.free(locations[:2])
        self.assertEqual(allocator.available_size(), 12)

    def test_ffn_reshard_rebuild_stays_no_kv(self):
        runner = _FakeFFNRunner()
        runner.init_memory_pool(pre_model_load_memory=0)
        old_allocator = runner.token_to_kv_pool_allocator

        config = runner.rebuild_memory_pool_after_inplace_reshard()

        self.assertEqual(config.max_total_num_tokens, 8 * 128)
        self.assertIsNone(runner.token_to_kv_pool)
        self.assertIsNot(runner.token_to_kv_pool_allocator, old_allocator)
        self.assertEqual(runner.metadata_updates, [2])

    def test_ffn_reshard_rebuild_configures_missing_kv_cache_dtype(self):
        runner = _FakeFFNRunner()
        del runner.kv_cache_dtype

        config = runner.rebuild_memory_pool_after_inplace_reshard()

        self.assertEqual(config.max_total_num_tokens, 8 * 128)
        self.assertEqual(runner.kv_cache_dtype, torch.bfloat16)
        self.assertEqual(runner.token_to_kv_pool_allocator.dtype, torch.bfloat16)
        self.assertIsNone(runner.token_to_kv_pool)
        self.assertIsNone(runner.token_to_kv_pool_allocator.get_kvcache())

    def test_attn_perspective_is_not_no_kv(self):
        runner = _FakeFFNRunner()
        runner.server_args.afd_perspective = AFDPerspective.AFD_PERSPECTIVE_ATTN
        self.assertFalse(runner._is_afd_ffn_no_kv())


if __name__ == "__main__":
    unittest.main(verbosity=3)

"""Integration test for the Central I/O MHA HostKVCache adapter."""

from __future__ import annotations

import multiprocessing as mp
import os
import time
import unittest


def _agent(socket_path: str, pool_bytes: int) -> None:
    from sglang.srt.mem_cache.central_io import CentralIOAgent

    CentralIOAgent(socket_path, pool_bytes).serve_forever()


def _model(socket_path: str, results) -> None:
    try:
        import torch

        from sglang.srt.mem_cache.memory_pool import MHATokenToKVPool
        from sglang.srt.mem_cache.memory_pool_host import CentralIOMHATokenToKVPoolHost

        torch.cuda.set_device(0)
        pool = MHATokenToKVPool(
            size=128,
            page_size=16,
            dtype=torch.float16,
            head_num=2,
            head_dim=16,
            layer_num=2,
            device="cuda:0",
            enable_memory_saver=False,
        )
        host = CentralIOMHATokenToKVPoolHost(
            pool,
            host_to_device_ratio=2.0,
            host_size=0,
            page_size=16,
            layout="page_first",
            socket_path=socket_path,
            model_id="adapter-test-model",
        )
        host_indices = host.alloc(16)
        if host_indices is None:
            raise AssertionError("Central I/O adapter failed to allocate host slots")
        device_indices = torch.arange(32, 48, dtype=torch.int64, device="cuda:0")
        expected = []
        for ordinal, tensor in enumerate(pool.k_buffer + pool.v_buffer):
            values = torch.arange(tensor.numel(), dtype=torch.float16, device="cuda:0").view_as(tensor)
            tensor.copy_(values + (ordinal + 1) * 100)
            expected.append(tensor[device_indices].cpu())
        torch.cuda.synchronize()

        host.backup_from_device_all_layer(pool, host_indices, device_indices, "kernel")
        for tensor in pool.k_buffer + pool.v_buffer:
            tensor[device_indices] = 0
        torch.cuda.synchronize()
        for layer_id in range(pool.layer_num):
            host.load_to_device_per_layer(pool, host_indices, device_indices, layer_id, "kernel")
        torch.cuda.synchronize()
        for ordinal, tensor in enumerate(pool.k_buffer + pool.v_buffer):
            if not torch.equal(tensor[device_indices].cpu(), expected[ordinal]):
                raise AssertionError(f"restored tensor {ordinal} differs from its backed-up value")
        host.free(host_indices)
        if host.available_size() != host.size:
            raise AssertionError("Central I/O host slots leaked after free")
        host.close()
        results.put((True, "adapter roundtrip and slot lifecycle verified"))
    except Exception as error:
        results.put((False, repr(error)))


class CentralIOHostKVCacheTest(unittest.TestCase):
    def test_real_host_kv_cache_roundtrip_through_agent(self):
        import torch

        if not torch.cuda.is_available():
            self.skipTest("CUDA is required")
        context = mp.get_context("spawn")
        socket_path = f"/tmp/sglang-central-io-test-{os.getpid()}.sock"
        try:
            os.unlink(socket_path)
        except FileNotFoundError:
            pass
        agent = context.Process(target=_agent, args=(socket_path, 1 << 20))
        agent.start()
        deadline = time.time() + 60
        while not os.path.exists(socket_path):
            if not agent.is_alive():
                self.fail(f"Central I/O agent exited: {agent.exitcode}")
            if time.time() > deadline:
                self.fail("Central I/O agent did not create its socket")
            time.sleep(0.05)
        results = context.Queue()
        model = context.Process(target=_model, args=(socket_path, results))
        model.start()
        try:
            ok, detail = results.get(timeout=300)
            model.join(timeout=60)
            self.assertEqual(model.exitcode, 0, "model process failed")
            self.assertTrue(ok, detail)
        finally:
            if model.is_alive():
                model.terminate()
                model.join(timeout=10)
            if agent.is_alive():
                agent.terminate()
                agent.join(timeout=10)
            try:
                os.unlink(socket_path)
            except FileNotFoundError:
                pass


if __name__ == "__main__":
    unittest.main()

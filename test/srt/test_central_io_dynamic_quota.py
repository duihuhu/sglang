"""Three-instance Central I/O quota-transfer integration test.

The test uses real SGLang MHA KV pools and one agent-owned pinned host pool.
It transfers only owner-free segments: live donor slots remain non-transferable.
"""

from __future__ import annotations

import multiprocessing as mp
import os
import time
import unittest


def _agent(socket_path: str, pool_bytes: int) -> None:
    from sglang.srt.mem_cache.central_io import CentralIOAgent

    CentralIOAgent(socket_path, pool_bytes).serve_forever()


def _make_host(socket_path: str, model_id: str):
    import torch

    from sglang.srt.mem_cache.memory_pool import MHATokenToKVPool
    from sglang.srt.mem_cache.memory_pool_host import CentralIOMHATokenToKVPoolHost

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
        model_id=model_id,
        initial_capacity=192,
        segment_tokens=16,
    )
    return pool, host


def _model(socket_path: str, results) -> None:
    try:
        import torch

        torch.cuda.set_device(0)
        hot_pool, hot = _make_host(socket_path, "hot")
        _, cold_b = _make_host(socket_path, "cold-b")
        _, cold_c = _make_host(socket_path, "cold-c")
        try:
            if [host.active_size for host in (hot, cold_b, cold_c)] != [192, 192, 192]:
                raise AssertionError("unexpected initial quotas")
            hot.clear()
            if hot.available_size() != 192 or hot.quota_status()["reserved"] != 0:
                raise AssertionError("HostKVCache.clear re-exposed inactive Central I/O slots")

            # A donor with live host KV cannot silently surrender its quota.
            live_cold_slots = cold_b.alloc(192)
            try:
                cold_b.resize_quota(160)
            except RuntimeError:
                pass
            else:
                raise AssertionError("Central I/O shrank a donor with live KV slots")
            cold_b.free(live_cold_slots)
            released_status = cold_b.quota_status()
            if released_status["reserved"] != 0:
                raise AssertionError(f"donor release did not reach agent: {released_status}")
            if released_status["segments"] != 12:
                raise AssertionError(f"donor segment state changed after rejected shrink: {released_status}")

            # Two cold instances return only free 16-token segments.  The hot
            # instance then consumes exactly those newly released pages.
            started = time.perf_counter()
            cold_b.resize_quota(160)
            cold_c.resize_quota(160)
            hot.resize_quota(256)
            rebalance_ms = (time.perf_counter() - started) * 1000
            if [host.active_size for host in (hot, cold_b, cold_c)] != [256, 160, 160]:
                raise AssertionError("quota transfer did not conserve capacity")
            if hot.quota_status()["reserved"] != 0:
                raise AssertionError("hot model unexpectedly has live slots before allocation")

            # Consume the original 192 slots so the next allocation must come
            # from the quota obtained from the two cold instances.
            old_slots = hot.alloc(192)
            new_slots = hot.alloc(16)
            if old_slots is None or new_slots is None or int(new_slots.min()) < 192:
                raise AssertionError("grown quota was not exposed through HostKVCache.alloc")

            device_indices = torch.arange(32, 48, dtype=torch.int64, device="cuda:0")
            expected = []
            for ordinal, tensor in enumerate(hot_pool.k_buffer + hot_pool.v_buffer):
                values = torch.arange(tensor.numel(), dtype=torch.float16, device="cuda:0").view_as(tensor)
                tensor.copy_(values + (ordinal + 1) * 100)
                expected.append(tensor[device_indices].cpu())
            torch.cuda.synchronize()

            hot.backup_from_device_all_layer(hot_pool, new_slots, device_indices, "kernel")
            for tensor in hot_pool.k_buffer + hot_pool.v_buffer:
                tensor[device_indices] = 0
            torch.cuda.synchronize()
            for layer_id in range(hot_pool.layer_num):
                hot.load_to_device_per_layer(
                    hot_pool, new_slots, device_indices, layer_id, "kernel"
                )
            torch.cuda.synchronize()
            for ordinal, tensor in enumerate(hot_pool.k_buffer + hot_pool.v_buffer):
                if not torch.equal(tensor[device_indices].cpu(), expected[ordinal]):
                    raise AssertionError(f"grown-quota restore differs in tensor {ordinal}")

            hot.free(new_slots)
            hot.free(old_slots)
            results.put(
                (
                    True,
                    "192/192/192 -> 256/160/160 quota transfer plus grown-range DMA "
                    f"verified; owner-free rebalance={rebalance_ms:.3f}ms",
                )
            )
        finally:
            for host in (hot, cold_b, cold_c):
                host.close()
    except Exception as error:
        results.put((False, repr(error)))


class CentralIODynamicQuotaTest(unittest.TestCase):
    def test_owner_free_quota_transfers_between_three_instances(self):
        import torch

        if not torch.cuda.is_available():
            self.skipTest("CUDA is required")
        context = mp.get_context("spawn")
        socket_path = f"/tmp/sglang-central-io-dynamic-{os.getpid()}.sock"
        try:
            os.unlink(socket_path)
        except FileNotFoundError:
            pass
        agent = context.Process(target=_agent, args=(socket_path, 1 << 20))
        agent.start()
        model = None
        try:
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
            ok, detail = results.get(timeout=300)
            model.join(timeout=60)
            self.assertEqual(model.exitcode, 0, "model process failed")
            self.assertTrue(ok, detail)
        finally:
            if model is not None and model.is_alive():
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

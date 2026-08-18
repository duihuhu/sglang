"""Integration test for the Central I/O MHA HostKVCache adapter."""

from __future__ import annotations

import multiprocessing as mp
import os
import time
import unittest


def _agent(socket_path: str, pool_bytes: int) -> None:
    from sglang.srt.mem_cache.central_io import CentralIOAgent

    agent = CentralIOAgent(socket_path, pool_bytes)
    # The benchmark runner warms this operator before publishing the socket.
    # Keep the integration test on the same lifecycle so the first measured
    # restore cannot compile CUDA while a model is already issuing requests.
    agent.warmup_restore_operator()
    agent.serve_forever()


def _model(socket_path: str, results) -> None:
    try:
        import torch
        from types import SimpleNamespace

        from sglang.srt.mem_cache.hiradix_cache import HiRadixCache
        from sglang.srt.mem_cache.live_page_reclaim import PersistentLeafReclaimIndex
        from sglang.srt.mem_cache.memory_pool import MHATokenToKVPool
        from sglang.srt.mem_cache.memory_pool_host import CentralIOMHATokenToKVPoolHost
        from sglang.srt.mem_cache.radix_cache import RadixKey, TreeNode

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
        initial_active_size = host.active_size
        host_indices = host.alloc(16)
        if host_indices is None:
            raise AssertionError("Central I/O adapter failed to allocate host slots")
        after_alloc = host.local_residency_status()
        if after_alloc["live_pages"] != 1:
            raise AssertionError(f"local residency missed allocation: {after_alloc}")
        if after_alloc["clean_pages"] + after_alloc["live_pages"] != after_alloc["effective_pages"]:
            raise AssertionError(f"local residency violated page accounting: {after_alloc}")
        device_indices = torch.arange(32, 48, dtype=torch.int64, device="cuda:0")
        expected = []
        for ordinal, tensor in enumerate(pool.k_buffer + pool.v_buffer):
            values = torch.arange(tensor.numel(), dtype=torch.float16, device="cuda:0").view_as(tensor)
            tensor.copy_(values + (ordinal + 1) * 100)
            expected.append(tensor[device_indices].cpu())
        torch.cuda.synchronize()

        host.backup_from_device_all_layer(pool, host_indices, device_indices, "kernel")
        after_backup = host.local_residency_status()
        # A single transfer only measures DMA copy time.  It must not be
        # misreported as sustained HBM-to-host ingress until another backup
        # establishes an arrival interval.
        if after_backup["backup_ingress_pages_per_s"] != 0:
            raise AssertionError(
                "one backup was incorrectly treated as a sustained ingress rate: "
                f"{after_backup}"
            )
        time.sleep(0.01)
        host.backup_from_device_all_layer(pool, host_indices, device_indices, "kernel")
        after_backup = host.local_residency_status()
        if after_backup["backup_ingress_pages_per_s"] <= 0:
            raise AssertionError(
                f"local residency missed sustained HBM backup ingress: {after_backup}"
            )
        host.record_retention_loss(64)
        after_loss = host.local_residency_status()
        if after_loss["retention_debt_tokens"] != 64:
            raise AssertionError(f"local residency missed reclaim-loss feedback: {after_loss}")
        for tensor in pool.k_buffer + pool.v_buffer:
            tensor[device_indices] = 0
        torch.cuda.synchronize()
        for layer_id in range(pool.layer_num):
            host.load_to_device_per_layer(pool, host_indices, device_indices, layer_id, "kernel")
        torch.cuda.synchronize()
        for ordinal, tensor in enumerate(pool.k_buffer + pool.v_buffer):
            if not torch.equal(tensor[device_indices].cpu(), expected[ordinal]):
                raise AssertionError(f"restored tensor {ordinal} differs from its backed-up value")

        # Fill the small real host quota, then let HiRadix's local maintainer
        # delete a safe host-only leaf. It must make pages reusable by this
        # model without changing the Central I/O lease capacity.
        filler = host.alloc(initial_active_size - len(host_indices))
        if filler is None:
            raise AssertionError("could not fill the adapter's initial host quota")
        all_host_indices = torch.cat((host_indices, filler))
        cache = HiRadixCache.__new__(HiRadixCache)
        cache.page_size = host.page_size
        cache.root_node = TreeNode(id=0)
        cache.root_node.key = RadixKey([])
        leaf = TreeNode(id=1)
        leaf.key = RadixKey([1])
        leaf.parent = cache.root_node
        leaf.host_value = all_host_indices
        cache.root_node.children[(1,)] = leaf
        cache.evictable_host_leaves = set()
        cache.get_child_key_fn = lambda key: (key.token_ids[0],)
        cache._record_remove_event = lambda node: None
        cache.cache_controller = SimpleNamespace(evict_host=host.free)
        cache.token_to_kv_pool_host = host
        cache._central_live_reclaim_queue = PersistentLeafReclaimIndex(
            page_size=host.page_size
        )
        cache._central_live_reclaim_nodes = {}
        cache._update_host_leaf_status(leaf)
        before_local = host.local_residency_status()
        if before_local["clean_pages"] != 0:
            raise AssertionError(f"local maintainer fixture is not full: {before_local}")
        local_result = cache._maintain_local_residency()
        if local_result is None or local_result["delete_host_leaf_node_ids"] != [leaf.id]:
            raise AssertionError(f"local host-only reclaim did not delete the safe leaf: {local_result}")
        after_local = host.local_residency_status()
        if after_local["effective_pages"] != initial_active_size // host.page_size:
            raise AssertionError(f"local reclaim changed effective quota: {after_local}")
        if after_local["clean_pages"] != after_local["effective_pages"]:
            raise AssertionError(f"local reclaim did not return pages to this allocator: {after_local}")
        if after_local["ready_latency_p95_s"] <= 0:
            raise AssertionError(
                "local maintainer did not feed its observed reclaim latency back "
                f"into watermarks: {after_local}"
            )

        # Exercise the full dynamic page lifecycle through the real adapter:
        # fence a live page, let the HostKV allocator release it, detach and
        # scrub it in the agent, then grow the same model back to its initial
        # quota. The existing GPU KV pool stays valid throughout.
        host_indices = host.alloc(16)
        if host_indices is None:
            raise AssertionError("could not allocate a page after local reclaim")
        page_range = [(int(host_indices[0]) // host.page_size, 1)]
        host.begin_live_page_drain(page_range)
        host.free(host_indices)
        after_free = host.local_residency_status()
        if after_free["live_pages"] != 0:
            raise AssertionError(f"local residency missed free: {after_free}")
        reclaim = host.commit_live_page_reclaim(page_range)
        after_commit = host.local_residency_status()
        expected_effective_pages = initial_active_size // host.page_size - 1
        if after_commit["effective_pages"] != expected_effective_pages:
            raise AssertionError(
                "local residency retained stale effective quota after reclaim: "
                f"{after_commit}"
            )
        deadline = time.time() + 30
        while True:
            status = host.reclaim_status(reclaim["reclaim_id"])
            if status["state"] == "ready":
                break
            if status["state"] == "failed" or time.time() > deadline:
                raise AssertionError(f"Central I/O page scrub did not complete: {status}")
            time.sleep(0.01)
        host.resize_quota(initial_active_size)
        if host.available_size() != initial_active_size:
            raise AssertionError("Central I/O host page lifecycle leaked slots")
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
        old_lease_mode = os.environ.get("SGLANG_CENTRAL_IO_LEASE_MODE")
        old_logical_max = os.environ.get("SGLANG_CENTRAL_IO_LOGICAL_MAX_GIB")
        old_initial = os.environ.get("SGLANG_CENTRAL_IO_INITIAL_GIB")
        os.environ["SGLANG_CENTRAL_IO_LEASE_MODE"] = "page"
        # The agent owns physical host memory.  A model may therefore keep a
        # small startup quota yet expose a larger logical ceiling for later
        # Central-I/O grow, without changing its GPU KV pool.
        os.environ["SGLANG_CENTRAL_IO_LOGICAL_MAX_GIB"] = "0.0005"
        os.environ["SGLANG_CENTRAL_IO_INITIAL_GIB"] = "0.00005"
        try:
            os.unlink(socket_path)
        except FileNotFoundError:
            pass
        agent = context.Process(target=_agent, args=(socket_path, 1 << 20))
        agent.start()
        # Initial CUDA extension compilation precedes socket publication by
        # design. It is not serving time, but a cold test environment may
        # legitimately need longer than the normal model-start budget.
        deadline = time.time() + 300
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
            if old_lease_mode is None:
                del os.environ["SGLANG_CENTRAL_IO_LEASE_MODE"]
            else:
                os.environ["SGLANG_CENTRAL_IO_LEASE_MODE"] = old_lease_mode
            if old_logical_max is None:
                del os.environ["SGLANG_CENTRAL_IO_LOGICAL_MAX_GIB"]
            else:
                os.environ["SGLANG_CENTRAL_IO_LOGICAL_MAX_GIB"] = old_logical_max
            if old_initial is None:
                del os.environ["SGLANG_CENTRAL_IO_INITIAL_GIB"]
            else:
                os.environ["SGLANG_CENTRAL_IO_INITIAL_GIB"] = old_initial


if __name__ == "__main__":
    unittest.main()

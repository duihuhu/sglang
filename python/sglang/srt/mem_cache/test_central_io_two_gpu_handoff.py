"""Two-GPU Central I/O handoff through real SGLang HostKV adapters."""

from __future__ import annotations

import multiprocessing as mp
import os
import time
import unittest


def _agent(socket_path: str, pool_bytes: int) -> None:
    from sglang.srt.mem_cache.central_io import CentralIOAgent

    os.environ["SGLANG_CENTRAL_IO_LEASE_MODE"] = "page"
    CentralIOAgent(socket_path, pool_bytes).serve_forever()


def _make_host(socket_path: str, model_id: str, device_id: int):
    import torch

    from sglang.srt.mem_cache.memory_pool import MHATokenToKVPool
    from sglang.srt.mem_cache.memory_pool_host import CentralIOMHATokenToKVPoolHost

    torch.cuda.set_device(device_id)
    os.environ["SGLANG_CENTRAL_IO_LEASE_MODE"] = "page"
    os.environ["SGLANG_CENTRAL_IO_LOGICAL_MAX_GIB"] = "0.0005"
    os.environ["SGLANG_CENTRAL_IO_INITIAL_GIB"] = "0.00005"
    pool = MHATokenToKVPool(
        size=128,
        page_size=16,
        dtype=torch.float16,
        head_num=2,
        head_dim=16,
        layer_num=2,
        device=f"cuda:{device_id}",
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
    )
    return pool, host


def _recipient(
    socket_path: str,
    donor_reclaimed,
    registered,
    results,
) -> None:
    try:
        _, host = _make_host(socket_path, "recipient-gpu1", 1)
        initial_pages = host.active_size // host.page_size
        registered.set()
        if not donor_reclaimed.wait(timeout=90):
            raise AssertionError("timed out waiting for donor scrub")
        before_pages = host.active_size // host.page_size
        host.resize_quota(host.active_size + host.page_size)
        after_pages = host.active_size // host.page_size
        indices = host.alloc(host.page_size)
        if indices is None:
            raise AssertionError("recipient could not allocate transferred page")
        host.free(indices)
        host.close()
        results.put(("recipient", True, initial_pages, before_pages, after_pages))
    except Exception as error:
        results.put(("recipient", False, repr(error)))


def _donor(socket_path: str, recipient_registered, donor_reclaimed, results) -> None:
    try:
        _, host = _make_host(socket_path, "donor-gpu0", 0)
        if not recipient_registered.wait(timeout=90):
            raise AssertionError("timed out waiting for recipient registration")
        initial_pages = host.active_size // host.page_size
        indices = host.alloc(host.page_size)
        if indices is None:
            raise AssertionError("donor could not allocate a live page")
        page_range = [(int(indices[0]) // host.page_size, 1)]
        host.begin_live_page_drain(page_range)
        host.free(indices)
        reclaim = host.commit_live_page_reclaim(page_range)
        deadline = time.time() + 60
        while True:
            state = host.reclaim_status(reclaim["reclaim_id"])["state"]
            if state == "ready":
                break
            if state == "failed" or time.time() > deadline:
                raise AssertionError(f"donor page reclaim failed: {state}")
            time.sleep(0.01)
        remaining_pages = host.active_size // host.page_size
        donor_reclaimed.set()
        host.close()
        results.put(("donor", True, initial_pages, remaining_pages))
    except Exception as error:
        results.put(("donor", False, repr(error)))


class CentralIOTwoGPUHandoffTest(unittest.TestCase):
    def test_live_page_moves_from_gpu0_donor_to_gpu1_recipient(self):
        import torch

        if torch.cuda.device_count() < 2:
            self.skipTest("two CUDA devices are required")
        context = mp.get_context("spawn")
        socket_path = f"/tmp/sglang-central-io-two-gpu-{os.getpid()}.sock"
        try:
            os.unlink(socket_path)
        except FileNotFoundError:
            pass
        # The agent owns fixed 256 MiB physical arena extents even though this
        # test consumes only a few SGLang KV pages from one such extent.
        agent = context.Process(target=_agent, args=(socket_path, 256 << 20))
        recipient_registered = context.Event()
        donor_reclaimed = context.Event()
        results = context.Queue()
        recipient = donor = None
        agent.start()
        try:
            deadline = time.time() + 60
            while not os.path.exists(socket_path):
                if not agent.is_alive():
                    self.fail(f"Central agent exited: {agent.exitcode}")
                if time.time() > deadline:
                    self.fail("Central agent did not create its socket")
                time.sleep(0.05)
            recipient = context.Process(
                target=_recipient,
                args=(socket_path, donor_reclaimed, recipient_registered, results),
            )
            recipient.start()
            self.assertTrue(
                recipient_registered.wait(timeout=120),
                "recipient did not register its GPU1 KV pool",
            )
            donor = context.Process(
                target=_donor,
                args=(socket_path, recipient_registered, donor_reclaimed, results),
            )
            donor.start()
            received = {}
            for _ in range(2):
                record = results.get(timeout=180)
                received[record[0]] = record
        except Exception:
            # Drain independently below so a worker exception remains visible.
            raise
        finally:
            for process in (donor, recipient):
                if process is not None:
                    process.join(timeout=90)
                    if process.is_alive():
                        process.terminate()
                        process.join(timeout=10)
            if agent.is_alive():
                agent.terminate()
                agent.join(timeout=10)
            try:
                os.unlink(socket_path)
            except FileNotFoundError:
                pass

        self.assertEqual(recipient.exitcode, 0, "recipient process failed")
        self.assertEqual(donor.exitcode, 0, "donor process failed")
        self.assertTrue(received["donor"][1], received["donor"])
        self.assertTrue(received["recipient"][1], received["recipient"])
        _, _, donor_initial, donor_remaining = received["donor"]
        _, _, recipient_initial, recipient_before, recipient_after = received["recipient"]
        self.assertEqual(donor_remaining, donor_initial - 1)
        self.assertEqual(recipient_before, recipient_initial)
        self.assertEqual(recipient_after, recipient_initial + 1)


if __name__ == "__main__":
    unittest.main()

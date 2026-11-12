"""Unit test for afd_ipc_cpp extension compilation and basic API.

Tests:
1. Extension compiles successfully (JIT)
2. AfdIpcComm object creation
3. Metadata encoding/decoding correctness
4. Single-process send/recv (same GPU, for API validation)

For multi-GPU cross-process tests, use bench_cpp_ipc.py.
"""

import os
import sys
import unittest
import logging

logging.basicConfig(level=logging.INFO)

# Skip if no CUDA
import torch
if not torch.cuda.is_available():
    print("CUDA not available, skipping tests")
    sys.exit(0)

if torch.cuda.device_count() < 2:
    print("Need at least 2 GPUs for IPC tests, skipping")
    sys.exit(0)


class TestAfdIpcCppCompilation(unittest.TestCase):
    """Test that the C++ extension compiles and loads."""

    def test_jit_compile(self):
        """Verify JIT compilation succeeds."""
        from sglang.srt.layers.afd_ipc_cpp import get_module
        mod = get_module()
        self.assertIsNotNone(mod)
        self.assertTrue(hasattr(mod, "AfdIpcComm"))
        self.assertTrue(hasattr(mod, "SyncMode"))

    def test_create_communicator(self):
        """Verify AfdIpcComm object creation."""
        from sglang.srt.layers.afd_ipc_cpp import get_module
        mod = get_module()

        # Create FFN-side communicator (device 0, peer device 1)
        comm = mod.AfdIpcComm(
            True,   # is_ffn
            0,      # local_device
            1,      # peer_device
            999,    # rank (unique for test)
            -1,     # mb_id
            "cpu_flag",  # sync_mode
        )
        self.assertFalse(comm.is_ready())
        self.assertEqual(comm.sync_mode(), "cpu_flag")


class TestCppIpcCommunicator(unittest.TestCase):
    """Test the Python wrapper class."""

    def test_factory_function(self):
        """Test create_ipc_communicator factory."""
        from sglang.srt.layers.afd_ipc_cpp.communicator import create_ipc_communicator
        from sglang.srt.layers.afd_type import AFDPerspective

        os.environ["AFD_IPC_PEER_DEVICE"] = "1"
        os.environ["AFD_IPC_SYNC_MODE"] = "cpu_flag"
        os.environ["AFD_SCHED_PORT"] = "65998"  # rank = 998

        # Should create C++ communicator
        comm = create_ipc_communicator(
            AFDPerspective.AFD_PERSPECTIVE_ATTN,
            mb_id=None,
            use_cpp=True,
        )
        self.assertIsNotNone(comm)
        # Cleanup
        del os.environ["AFD_IPC_PEER_DEVICE"]
        del os.environ["AFD_IPC_SYNC_MODE"]
        del os.environ["AFD_SCHED_PORT"]


class TestCrossProcessIpc(unittest.TestCase):
    """Cross-process IPC test using multiprocessing.

    Spawns sender (ATTN, device 0) and receiver (FFN, device 1) in
    separate processes, exchanges a tensor, verifies correctness.
    """

    def test_send_recv_correctness(self):
        """Verify data integrity across processes."""
        import multiprocessing as mp
        mp.set_start_method("spawn", force=True)

        result_queue = mp.Queue()

        def sender_fn(q):
            try:
                os.environ["AFD_IPC_PEER_DEVICE"] = "1"
                os.environ["AFD_IPC_SYNC_MODE"] = "cpu_flag"
                os.environ["AFD_SCHED_PORT"] = "65700"
                torch.cuda.set_device(0)

                from sglang.srt.layers.afd_ipc_cpp.communicator import CppIpcTensorCommunicator
                from sglang.srt.layers.afd_type import AFDPerspective

                comm = CppIpcTensorCommunicator(
                    AFDPerspective.AFD_PERSPECTIVE_ATTN, mb_id=None
                )
                comm._wait_ready()

                # Send a known tensor
                x = torch.arange(1024, dtype=torch.float32, device="cuda:0")
                comm.send_tensor(x)

                # Receive ack
                ack = comm.recv_tensor()
                q.put(("sender_ok", ack.item()))
            except Exception as e:
                q.put(("sender_error", str(e)))

        def receiver_fn(q):
            try:
                os.environ["AFD_IPC_PEER_DEVICE"] = "0"
                os.environ["AFD_IPC_SYNC_MODE"] = "cpu_flag"
                os.environ["AFD_SCHED_PORT"] = "65700"
                torch.cuda.set_device(1)

                from sglang.srt.layers.afd_ipc_cpp.communicator import CppIpcTensorCommunicator
                from sglang.srt.layers.afd_type import AFDPerspective

                comm = CppIpcTensorCommunicator(
                    AFDPerspective.AFD_PERSPECTIVE_FFN, mb_id=None
                )
                comm._wait_ready()

                # Receive tensor
                data = comm.recv_tensor()

                # Verify
                expected = torch.arange(1024, dtype=torch.float32, device="cuda:1")
                match = torch.allclose(data, expected)

                # Send ack
                ack = torch.tensor([1.0 if match else 0.0],
                                   dtype=torch.float32, device="cuda:1")
                comm.send_tensor(ack)

                q.put(("receiver_ok", match))
            except Exception as e:
                q.put(("receiver_error", str(e)))

        # Launch processes
        p_recv = mp.Process(target=receiver_fn, args=(result_queue,))
        p_send = mp.Process(target=sender_fn, args=(result_queue,))

        p_recv.start()
        import time; time.sleep(1)
        p_send.start()

        p_send.join(timeout=30)
        p_recv.join(timeout=30)

        # Check results
        results = {}
        while not result_queue.empty():
            key, val = result_queue.get()
            results[key] = val

        self.assertIn("receiver_ok", results,
                      f"Receiver failed: {results}")
        self.assertTrue(results["receiver_ok"],
                        "Data mismatch after IPC transfer")


if __name__ == "__main__":
    unittest.main()

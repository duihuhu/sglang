"""Unit tests for the data-driven AFD scheduler (afd_async_sched).

These are pure unit tests with no GPU / no real server — we substitute
fake channels and fake decoder layers so the test verifies *only* the
driver's scheduling logic:

* Each mb chain advances independently.
* The driver picks the smallest-next_layer ready chain.
* When all chains stall on recv, the driver blocks on the recv-done queue
  and resumes whichever mb arrives first (proves out-of-order recv works).
* Per-mb hidden_states is threaded correctly across A→F→A→F steps.
"""

import os
import threading
import time
import unittest
from typing import Optional

# Avoid pulling CUDA-heavy server bits.  We only need numpy-style tensors
# for the driver's plumbing — so we mock torch with a tiny stub if torch
# is unavailable.  But torch IS available in the sglang env; just import it.
import torch

# Pre-import the afd module so the per-step import cost in the fake
# layer does not race with the feeder thread's polling deadline.
from sglang.srt.layers import afd as _afd  # noqa: F401
from sglang.srt.layers.afd_async_sched import AsyncMbDriver, _State
from sglang.srt.layers.afd_type import AFDPerspective


class _FakeAsyncComm:
    """Minimal stand-in for AsyncTensorCommunicator inside a PerMbChannel.

    Tracks send order and lets the test trigger recv completion for a
    specific mb at will.
    """

    def __init__(self, mb_id: int):
        self.mb_id = mb_id
        self.sent: list = []
        self.recv_started = 0
        self.recv_completed = 0
        self._recv_event = threading.Event()
        self._next_recv_value: Optional[torch.Tensor] = None

    def send_async(self, x):
        self.sent.append(x)

    def recv_start(self):
        self.recv_started += 1
        self._recv_event.clear()

    def recv_wait(self):
        self._recv_event.wait(timeout=5.0)
        if not self._recv_event.is_set():
            raise RuntimeError("FakeAsyncComm: recv timed out")
        self.recv_completed += 1
        v = self._next_recv_value
        self._next_recv_value = None
        return v

    # Test API
    def deliver(self, value: torch.Tensor):
        self._next_recv_value = value
        self._recv_event.set()

    def drain_sends(self):
        pass

    def drain_recvs(self):
        pass


class _FakePerMbChannel:
    """Stand-in for PerMbChannel with a controllable recv watcher.

    Supports the test pattern "schedule a delivery for mb X with priority
    P" by buffering pending deliveries and firing them in priority order
    once the driver issues ``recv_start``.
    """

    def __init__(self, mb_id: int):
        self.mb_id = mb_id
        self.async_comm = _FakeAsyncComm(mb_id)
        self._on_ready = None
        self._lock = threading.Lock()
        self._queued: list = []  # list of pending tensors to deliver

    def recv_start(self, on_ready):
        with self._lock:
            self.async_comm.recv_start()
            self._on_ready = on_ready
            if self._queued:
                value = self._queued.pop(0)
                cb = self._on_ready
                self._on_ready = None
                fire = True
            else:
                fire = False
        if fire:
            self.async_comm.deliver(value)
            cb(self.mb_id)

    def signal_ready(self, value: torch.Tensor):
        """Test helper: simulate "bytes have landed" + queue the GPU value
        that recv_wait will return. If the driver has not yet issued
        ``recv_start``, the signal is buffered and fires as soon as it
        does."""
        with self._lock:
            if self._on_ready is None:
                # Buffer until the driver issues recv_start.
                self._queued.append(value)
                return
            cb = self._on_ready
            self._on_ready = None
        self.async_comm.deliver(value)
        cb(self.mb_id)

    def drain(self):
        pass


class _FakeChannelSet:
    def __init__(self, num_mb: int):
        self.num_mb = num_mb
        self.channels = [_FakePerMbChannel(i) for i in range(num_mb)]

    def __getitem__(self, mb_id: int) -> _FakePerMbChannel:
        return self.channels[mb_id]

    def __len__(self) -> int:
        return self.num_mb

    def drain(self):
        for c in self.channels:
            c.drain()


class _FakeLayer:
    """Stand-in decoder layer.

    Records every (perspective, stage, layer, mb) call along with the
    hidden tensor's "id" (an int payload we encode into the tensor's
    only element so we can verify state propagation).
    """

    def __init__(self, layer_id: int, log: list, perspective: AFDPerspective):
        self.layer_id = layer_id
        self.log = log
        self.perspective = perspective

    def _bump(self, hs: torch.Tensor) -> torch.Tensor:
        # +1 to encode "this layer was applied"
        return hs + 1

    def forward_afd_A(self, positions, hidden_states, forward_batch, residual):
        from sglang.srt.layers import afd as _afd
        comm = _afd.get_async_communicator()
        is_attn = self.perspective == AFDPerspective.AFD_PERSPECTIVE_ATTN

        if is_attn:
            # DA A-step: compute attn locally, then send to DF.
            new_hs = self._bump(hidden_states)
            comm.send_async(new_hs)
            self.log.append(("A", "DA", self.layer_id, "send",
                             new_hs.item()))
            return new_hs, residual
        else:
            # DF A-step: recv from DA (input is placeholder).
            recv_hs = comm.recv_wait()
            self.log.append(("A", "DF", self.layer_id, "recv",
                             recv_hs.item()))
            return recv_hs, residual

    def forward_afd_F(self, hidden_states, forward_batch, residual):
        from sglang.srt.layers import afd as _afd
        comm = _afd.get_async_communicator()
        is_attn = self.perspective == AFDPerspective.AFD_PERSPECTIVE_ATTN

        if is_attn:
            # DA F-step: recv from DF.
            recv_hs = comm.recv_wait()
            self.log.append(("F", "DA", self.layer_id, "recv",
                             recv_hs.item()))
            return recv_hs, residual
        else:
            # DF F-step: compute MLP, send back to DA.
            new_hs = self._bump(hidden_states) + 1000  # marker
            comm.send_async(new_hs)
            self.log.append(("F", "DF", self.layer_id, "send",
                             new_hs.item()))
            return new_hs, residual


class TestAsyncMbDriverDA(unittest.TestCase):
    """Driver behaviour from the Attn (DA) side."""

    def setUp(self):
        self.num_layers = 2
        self.m_stage = 3
        self.log: list = []
        self.layers = [
            _FakeLayer(l, self.log, AFDPerspective.AFD_PERSPECTIVE_ATTN)
            for l in range(self.num_layers)
        ]
        self.channels = _FakeChannelSet(self.m_stage)
        # Seed each mb's initial hidden_states so we can track propagation.
        # mb i starts with tensor([100 + i]).
        self.input_arrs = [
            {
                "positions": None,
                "forward_batch": None,
                "hidden_states": torch.tensor([100.0 + i]),
                "residual": None,
            }
            for i in range(self.m_stage)
        ]

    def test_da_initial_state_is_ready(self):
        driver = AsyncMbDriver(
            perspective=AFDPerspective.AFD_PERSPECTIVE_ATTN,
            layers=self.layers,
            num_layers=self.num_layers,
            m_stage=self.m_stage,
            channels=self.channels,
            input_arrs=self.input_arrs,
        )
        # All 3 mbs are READY_TO_COMPUTE at the start (DA's first step is A,
        # which doesn't need recv).
        for mb in driver.mbs:
            self.assertEqual(mb.state, _State.READY_TO_COMPUTE)
            self.assertEqual(mb.next_stage, "A")
            self.assertEqual(mb.next_layer, 0)

    def test_da_out_of_order_recv(self):
        """Drive a 3-mb 2-layer schedule and prove a slower mb does NOT
        block faster mbs. We make recv for mb0 always arrive *last* and
        verify the timeline order shows mb1/mb2 finishing earlier."""
        driver = AsyncMbDriver(
            perspective=AFDPerspective.AFD_PERSPECTIVE_ATTN,
            layers=self.layers,
            num_layers=self.num_layers,
            m_stage=self.m_stage,
            channels=self.channels,
            input_arrs=self.input_arrs,
            detailed_timeline=[],
        )

        # Background feeder: each "round" corresponds to one F-step per
        # layer.  We wait until the driver has issued recv_start on every
        # mb, then deliver in reverse mb order.  The buffered-delivery
        # support in _FakePerMbChannel handles the case where the feeder
        # races ahead of the driver.
        def feeder():
            for f_round in range(self.num_layers):
                deadline = time.time() + 5.0
                while time.time() < deadline:
                    if all(
                        self.channels[m]._on_ready is not None
                        for m in range(self.m_stage)
                    ):
                        break
                    time.sleep(0.001)
                else:
                    return  # timeout, let main thread report failure
                for mb_id in [2, 1, 0]:
                    self.channels[mb_id].signal_ready(
                        torch.tensor([200.0 + mb_id + f_round * 10])
                    )
                    # Tiny wait so the driver picks up this completion
                    # and dispatches before the next signal arrives.
                    time.sleep(0.005)

        t = threading.Thread(target=feeder, daemon=True)
        t.start()
        driver.run()
        t.join(timeout=5.0)

        # All 3 chains finished.
        self.assertEqual(
            sum(1 for mb in driver.mbs if mb.state == _State.DONE),
            self.m_stage,
        )

        # Each mb sent its A-stage tensor in layer order.
        for mb_id in range(self.m_stage):
            sent = self.channels[mb_id].async_comm.sent
            self.assertEqual(
                len(sent),
                self.num_layers,
                f"mb {mb_id}: send count mismatch ({len(sent)} != "
                f"{self.num_layers})",
            )

        # Verify the picked order in the timeline alternates layers and
        # respects readiness — specifically, we expect a pattern where
        # all A(0)s execute first (no recv blocked), then F(0)s land
        # in reverse order, then A(1)s in reverse order, etc.
        tl = driver.detailed_timeline
        a0_steps = [e for e in tl
                    if e["stage"] == "AFD_FORWARD_STAGE_A" and e["layer"] == 0]
        self.assertEqual(len(a0_steps), self.m_stage)
        first_f0_idx = next(
            i for i, e in enumerate(tl)
            if e["stage"] == "AFD_FORWARD_STAGE_F" and e["layer"] == 0
        )
        a0_max_idx = max(
            i for i, e in enumerate(tl)
            if e["stage"] == "AFD_FORWARD_STAGE_A" and e["layer"] == 0
        )
        self.assertLess(a0_max_idx, first_f0_idx)
        # F-steps appear in mb-reverse order (mb2, mb1, mb0) for layer 0.
        f0_mbs = [e["mb"] for e in tl
                  if e["stage"] == "AFD_FORWARD_STAGE_F" and e["layer"] == 0]
        self.assertEqual(f0_mbs, [2, 1, 0],
                         f"Expected reverse F order, got {f0_mbs}")


class TestAsyncMbDriverDF(unittest.TestCase):
    """Driver behaviour from the FFN (DF) side."""

    def setUp(self):
        self.num_layers = 2
        self.m_stage = 2
        self.log: list = []
        self.layers = [
            _FakeLayer(l, self.log, AFDPerspective.AFD_PERSPECTIVE_FFN)
            for l in range(self.num_layers)
        ]
        self.channels = _FakeChannelSet(self.m_stage)
        self.input_arrs = [
            {
                "positions": None,
                "forward_batch": None,
                "hidden_states": torch.tensor([0.0]),
                "residual": None,
            }
            for _ in range(self.m_stage)
        ]

    def test_df_starts_in_wait_recv(self):
        driver = AsyncMbDriver(
            perspective=AFDPerspective.AFD_PERSPECTIVE_FFN,
            layers=self.layers,
            num_layers=self.num_layers,
            m_stage=self.m_stage,
            channels=self.channels,
            input_arrs=self.input_arrs,
        )
        # DF's first step is A, which needs recv → all mbs in WAIT_RECV.
        for mb in driver.mbs:
            self.assertEqual(mb.state, _State.WAIT_RECV)

    def test_df_completes_with_in_order_arrivals(self):
        driver = AsyncMbDriver(
            perspective=AFDPerspective.AFD_PERSPECTIVE_FFN,
            layers=self.layers,
            num_layers=self.num_layers,
            m_stage=self.m_stage,
            channels=self.channels,
            input_arrs=self.input_arrs,
            detailed_timeline=[],
        )

        def feeder():
            for layer in range(self.num_layers):
                # DF needs M recvs at the start of each layer (the A-step).
                t_start = time.time()
                while time.time() - t_start < 5.0:
                    if all(
                        self.channels[m]._on_ready is not None
                        for m in range(self.m_stage)
                    ):
                        break
                    time.sleep(0.001)
                for mb_id in range(self.m_stage):
                    self.channels[mb_id].signal_ready(
                        torch.tensor([300.0 + mb_id + layer * 10])
                    )
                    time.sleep(0.002)

        t = threading.Thread(target=feeder, daemon=True)
        t.start()
        driver.run()
        t.join(timeout=5.0)

        self.assertEqual(
            sum(1 for mb in driver.mbs if mb.state == _State.DONE),
            self.m_stage,
        )
        # Each mb sent F-stage output for each layer.
        for mb_id in range(self.m_stage):
            self.assertEqual(
                len(self.channels[mb_id].async_comm.sent),
                self.num_layers,
            )


if __name__ == "__main__":
    unittest.main()

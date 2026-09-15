import unittest
from types import SimpleNamespace
from unittest.mock import Mock

from sglang.srt.managers.scheduler_afd_mixin import SchedulerAFDMixin
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=1, suite="stage-a-cpu-only")


class _Req:
    def __init__(self, finished_reason=None, *, to_finish=None, stream=False):
        self.rid = "rid-1"
        self.finished_reason = finished_reason
        self.to_finish = to_finish
        self.stream = stream
        self.afd_pa_instance_id = "pa-1"
        self.afd_pf_instance_id = "pf-1"
        self.afd_lease_id = "lease-1"
        self.afd_pair_epoch = 3

    def finished(self):
        return self.finished_reason is not None


class TestAFDSharedDispatchCompletion(CustomTestCase):
    def setUp(self):
        self.client = SimpleNamespace(
            complete_dispatch=Mock(return_value=True),
            release=Mock(return_value=True),
        )
        self.scheduler = SimpleNamespace(
            _afd_shared_client=self.client,
            server_args=SimpleNamespace(afd_instance_id="pa-1"),
        )

    @staticmethod
    def make_batch(req):
        return SimpleNamespace(
            reqs=[req],
            afd_dispatch_identity="pa-1:3:7",
            afd_lease_id="lease-1",
            afd_lease_rid="rid-1",
        )

    def complete(self, req):
        SchedulerAFDMixin.afd_complete_shared_dispatch(
            self.scheduler, self.make_batch(req)
        )

    def test_live_streaming_request_retains_lease(self):
        req = _Req(stream=True)

        self.complete(req)

        self.client.complete_dispatch.assert_called_once_with(
            "pa-1:3:7", "lease-1"
        )
        self.client.release.assert_not_called()
        self.assertEqual(req.afd_lease_id, "lease-1")

    def test_pending_abort_retains_lease_until_finish_is_committed(self):
        req = _Req(to_finish=object())

        self.complete(req)

        self.client.release.assert_not_called()
        self.assertEqual(req.afd_lease_id, "lease-1")

    def test_terminal_reasons_release_lease(self):
        for reason_name in ("length", "eos_or_stop", "abort"):
            with self.subTest(reason=reason_name):
                self.client.release.reset_mock()
                req = _Req(finished_reason=object(), stream=True)

                self.complete(req)

                self.client.release.assert_called_once_with("rid-1", "pa-1")
                self.assertIsNone(req.afd_pa_instance_id)
                self.assertIsNone(req.afd_pf_instance_id)
                self.assertIsNone(req.afd_lease_id)
                self.assertEqual(req.afd_pair_epoch, 0)


if __name__ == "__main__":
    unittest.main(verbosity=3)

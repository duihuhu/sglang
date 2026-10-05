import unittest
from types import SimpleNamespace
from unittest.mock import Mock

from verify_m2_slot_admission import method, SCHEDULER
from m2_gpu_health import health_errors


class HardwareHealthTests(unittest.TestCase):
    def test_missing_samples_rejected(self):
        self.assertTrue(health_errors([]))

    def test_sustained_loaded_low_clock_rejected(self):
        sample = {"gpus": [{"index": i, "utilization": 100, "sm_mhz": 210} for i in (6, 7)]}
        self.assertTrue(health_errors([sample] * 3))
        self.assertFalse(health_errors([sample] * 2))

    def test_idle_low_clocks_allowed(self):
        sample = {"gpus": [{"index": i, "utilization": 0, "sm_mhz": 210} for i in (6, 7)]}
        self.assertFalse(health_errors([sample] * 10))

    def test_normal_load_allowed(self):
        sample = {"gpus": [{"index": i, "utilization": 100, "sm_mhz": 1140} for i in (6, 7)]}
        self.assertFalse(health_errors([sample] * 10))


class SlotGateControlTests(unittest.TestCase):
    def setUp(self):
        self.namespace = {
            "ENABLE_TENANT_RESOURCE_TRACKING": True,
            "logging": Mock(), "logger": Mock(),
            "SetInternalStateReqOutput": SimpleNamespace,
        }
        self.update = method(SCHEDULER, "Scheduler", "set_internal_state", self.namespace)
        self.scheduler = SimpleNamespace(
            spec_algorithm=SimpleNamespace(is_none=lambda: True),
            metrics_reporter=SimpleNamespace(), is_fully_idle=lambda: True,
        )

    def call(self, code):
        return self.update(self.scheduler, SimpleNamespace(server_args={"tenant_slot_gate_code": code}))

    def test_requires_experiment_opt_in(self):
        self.namespace["ENABLE_TENANT_RESOURCE_TRACKING"] = False
        self.assertFalse(self.call(0).updated)
        self.assertFalse(hasattr(self.scheduler, "_tenant_legacy_slot_gate"))

    def test_busy_scheduler_does_not_switch(self):
        self.scheduler.is_fully_idle = lambda: False
        self.assertFalse(self.call(0).updated)
        self.assertFalse(hasattr(self.scheduler, "_tenant_legacy_slot_gate"))

    def test_only_two_gate_values_allowed(self):
        self.assertFalse(self.call(2).updated)
        self.assertFalse(hasattr(self.scheduler, "_tenant_legacy_slot_gate"))

    def test_idle_experiment_can_switch_both_ways(self):
        self.assertTrue(self.call(0).updated)
        self.assertTrue(self.scheduler._tenant_legacy_slot_gate)
        self.assertTrue(self.call(1).updated)
        self.assertFalse(self.scheduler._tenant_legacy_slot_gate)

class SlotGateTests(unittest.TestCase):
    def setUp(self):
        self.parallel = SimpleNamespace(pp_max_micro_batch_size=8)
        namespace = {"get_parallel": lambda: self.parallel}
        self.gate = method(SCHEDULER, "Scheduler", "_prefill_req_limit_reached", namespace)
        self.limit = method(SCHEDULER, "Scheduler", "get_num_allocatable_reqs", namespace)
        self.free = 5
        self.scheduler = SimpleNamespace(req_to_token_pool=SimpleNamespace(
            available_size=lambda: self.free))
        self.scheduler.get_num_allocatable_reqs = lambda running: self.limit(self.scheduler, running)

    def test_one_continuation_does_not_spend_a_second_slot(self):
        self.assertFalse(self.gate(self.scheduler, 2, 5, 1, False))
        self.scheduler._tenant_legacy_slot_gate = True
        self.assertTrue(self.gate(self.scheduler, 2, 5, 1, False))

    def test_runtime_cap_still_counts_continuations(self):
        self.free = 100
        self.assertTrue(self.gate(self.scheduler, 2, 6, 1, False))

    def test_new_slot_exhaustion_still_blocks(self):
        self.free = 4
        self.assertTrue(self.gate(self.scheduler, 2, 5, 1, False))

    def test_existing_candidate_does_not_need_free_slot(self):
        self.free = 4
        self.assertFalse(self.gate(self.scheduler, 2, 5, 1, True))

    def test_no_reuse_is_identical_to_legacy(self):
        for running in range(9):
            for free in range(9):
                self.free = free
                for selected in range(9):
                    self.scheduler._tenant_legacy_slot_gate = False
                    actual = self.gate(self.scheduler, running, selected, 0, False)
                    self.scheduler._tenant_legacy_slot_gate = True
                    self.assertEqual(actual, self.gate(self.scheduler, running, selected, 0, False))

    def test_exhaustive_distinct_constraints(self):
        for running in range(9):
            for free in range(9):
                self.free = free
                for selected in range(9):
                    for reused in range(selected + 1):
                        for candidate_reuses in (True, False):
                            expected = (running + selected + 1 > 8 or
                                        selected - reused + int(not candidate_reuses) > free)
                            self.assertEqual(expected, self.gate(
                                self.scheduler, running, selected, reused, candidate_reuses))

    def test_closed_form_admission_limits_and_deficit_bound(self):
        for running in range(9):
            headroom = 8 - running
            for reused in range(headroom + 1):
                for free in range(9):
                    self.free = free
                    limits = []
                    for legacy in (True, False):
                        self.scheduler._tenant_legacy_slot_gate = legacy
                        fresh = 0
                        while not self.gate(self.scheduler, running, reused + fresh, reused, False):
                            fresh += 1
                        limits.append(fresh)
                    expected = [max(0, min(headroom, free) - reused),
                                max(0, min(headroom - reused, free))]
                    self.assertEqual(limits, expected)
                    self.assertTrue(0 <= limits[1] - limits[0] <= reused)


if __name__ == "__main__":
    unittest.main()

import copy
import unittest

from analyze_m2_hybrid_factors import bootstrap_mean, lifecycle, metrics


def specimen():
    return {
        "experiment_clock_origin_s": 100,
        "experiment": {"family": "E1"},
        "requests": [{"rid": "a", "group": "activation", "submitted_s": .1,
                      "scheduled_arrival_s": .1, "finished_s": 2.2,
                      "prompt_tokens": 4096, "completion_tokens": 16}],
        "summary": {"groups": {"activation": {"p95_jct_s": 2.1}}},
        "scheduler_batch_trace": {"admission_blocks": []},
        "scheduler_resource_trace": {
            "event_clock_origin_s": 99,
            "system": {"peak_state_slots": 1, "peak_kv_tokens": 8192,
                       "state_slot_seconds": 1.9},
            "events": [
                {"observed_s": 1.2, "added": [{"rid": "a"}], "removed": [],
                 "state_slots": 1, "active_requests": 1, "kv_tokens": 4096},
                {"observed_s": 2, "added": [], "removed": [],
                 "state_slots": 1, "active_requests": 1, "kv_tokens": 8192},
                {"observed_s": 3.1, "added": [], "removed": ["a"],
                 "state_slots": 0, "active_requests": 0, "kv_tokens": 0},
            ],
        },
    }


class AnalysisTests(unittest.TestCase):
    def test_clock_alignment_and_queue_not_jct(self):
        value = metrics(specimen())
        self.assertAlmostEqual(value["activation_queue_p95"], .1)
        self.assertAlmostEqual(value["activation_active_mean"], 1.9)
        self.assertEqual(value["kv_progress_without_new_state_events"], 1)

    def test_missing_release_rejected(self):
        item = specimen()
        item["scheduler_resource_trace"]["events"].pop()
        with self.assertRaisesRegex(ValueError, "missing"):
            lifecycle(item)

    def test_reactivation_rejected(self):
        item = specimen()
        item["scheduler_resource_trace"]["events"] += copy.deepcopy(
            item["scheduler_resource_trace"]["events"])
        with self.assertRaisesRegex(ValueError, "reactivation"):
            lifecycle(item)

    def test_negative_queue_rejected(self):
        item = specimen()
        item["experiment_clock_origin_s"] = 101
        with self.assertRaisesRegex(ValueError, "clocks"):
            metrics(item)

    def test_zero_difference_control(self):
        self.assertEqual(bootstrap_mean([0] * 5), {"mean": 0, "ci95": [0, 0], "n": 5})

    def test_actual_request_slot_gate_and_allocator_disagree_on_new_demand(self):
        from verify_m2_slot_admission import witness
        value = witness()
        self.assertTrue(value["actual_gate_blocks_next_candidate"])
        self.assertTrue(value["actual_allocator_accepts_continuation_plus_five_new"])
        self.assertEqual(value["new_slots_consumed"], 5)


if __name__ == "__main__":
    unittest.main()

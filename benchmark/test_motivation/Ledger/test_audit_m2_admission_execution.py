import unittest

from audit_m2_admission_execution import window_metrics


def fixture(events):
    return {"experiment": {"arrival": 10}, "experiment_clock_origin_s": 100,
            "requests": [{"rid": "h", "group": "holder"}, {"rid": "c", "group": "candidate"}],
            "scheduler_batch_trace": {"event_clock_origin_s": 105, "events": events,
                                      "admission_blocks": [{"observed_s": 6, "reason": "request_limit"},
                                                           {"observed_s": 30, "reason": "prefill_no_token"}]}}


def batch(timestamp, *members):
    return {"launch_s": timestamp, "mode": "extend", "members": list(members)}


HOLDER = {"rid": "h", "phase": "decode", "work_tokens": 1}
CANDIDATE = {"rid": "c", "phase": "prefill", "work_tokens": 4096}


class WindowTests(unittest.TestCase):
    def test_clock_alignment_and_window_filter(self):
        result = window_metrics(fixture([batch(4, CANDIDATE), batch(6, CANDIDATE),
                                         batch(8, HOLDER), batch(10, HOLDER), batch(40, CANDIDATE)]), 16)
        self.assertEqual(result["candidate_prefill_batches_before_last_holder_release"], 1)
        self.assertEqual(result["candidate_prefill_tokens_before_last_holder_release"], 4096)
        self.assertEqual(result["holder_decode_launches_before_last_holder_release"], 2)
        self.assertEqual(result["first_holder_decode_launch_after_arrival_s"], 3)
        self.assertEqual(result["max_interval_without_holder_decode_launch_s"], 3)
        self.assertEqual(result["admission_blocks"], {"request_limit": 1})

    def test_mixed_is_not_a_decode_exclusion(self):
        result = window_metrics(fixture([batch(6, CANDIDATE, HOLDER)]), 16)
        self.assertEqual(result["candidate_prefill_with_holder_decode_batches"], 1)
        self.assertEqual(result["holder_decode_launches_before_last_holder_release"], 1)

    def test_invalid_window_and_missing_trace(self):
        item = fixture([])
        self.assertEqual(window_metrics(item, 9), {"valid_window": False})
        item["scheduler_batch_trace"]["dropped_events"] = 1
        with self.assertRaises(ValueError):
            window_metrics(item, 16)

    def test_no_holder_launch_interval_includes_boundaries(self):
        result = window_metrics(fixture([batch(6, CANDIDATE)]), 16)
        self.assertIsNone(result["first_holder_decode_launch_after_arrival_s"])
        self.assertEqual(result["max_interval_without_holder_decode_launch_s"], 6)


if __name__ == "__main__":
    unittest.main()

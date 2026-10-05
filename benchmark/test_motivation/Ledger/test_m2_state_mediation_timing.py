import unittest

from audit_m2_state_mediation_timing import launch_gaps


def item(times):
    return {"scheduler_batch_trace": {"event_clock_origin_s": 100,
            "events": [{"launch_s": t, "recorded_s": t + .1, "mode": "decode", "batch_size": 128} for t in times]},
            "gpu_health_samples": [{"observed_s": 100.5, "gpus": [{"utilization": 20}, {"utilization": 40}]}]}


class TimingTests(unittest.TestCase):
    def test_gap_and_absolute_sample_alignment(self):
        result = launch_gaps(item([0, 2, 2.1]))
        self.assertEqual(result["max_launch_interval_s"], 2)
        self.assertEqual(len(result["intervals_over_1s"]), 1)
        self.assertAlmostEqual(result["intervals_over_1s"][0]["record_to_next_launch_s"], 1.9)
        self.assertEqual(result["intervals_over_1s"][0]["mean_utilization_during_gap"], 30)

    def test_nonmonotonic_is_rejected(self):
        with self.assertRaises(ValueError):
            launch_gaps(item([2, 1]))

    def test_no_large_gap_and_no_fabricated_empty_value(self):
        self.assertEqual(launch_gaps(item([0, .5]))["intervals_over_1s"], [])
        self.assertIsNone(launch_gaps(item([2, 4]))["intervals_over_1s"][0]["mean_utilization_during_gap"])


if __name__ == "__main__":
    unittest.main()

import unittest

from analyze_m2_mixed_near import select_baseline
from run_m2_mixed_near import ORIGINAL_SESSIONS, original, session_plan


class MixedNearTests(unittest.TestCase):
    def test_subset_keeps_original_order_repetitions_and_policies(self):
        original_plan = ORIGINAL_SESSIONS("measure")
        plan = session_plan("measure")
        cells = [(p, profile, phase, s["rep"]) for s in plan for p, profile, phase in s["cases"]]
        self.assertEqual(len(cells), 16)
        self.assertTrue(all(phase == "near" for _, _, phase, _ in cells))
        self.assertEqual(sum(original.PROFILES[p][0] + 64 for _, p, _, _ in cells), 8192)
        for a, b in zip(original_plan, plan):
            self.assertEqual(b["cases"], [c for c in a["cases"] if c[2] == "near"])
        self.assertEqual(original.session_plan, ORIGINAL_SESSIONS)

    def test_no_background_calibration_override(self):
        with self.assertRaises(ValueError):
            session_plan("calibrate")

    def test_near_selection_does_not_require_or_read_far_or_heldout(self):
        index = {}
        for rep in original.REPS:
            for policy, mean in (("native607", 10), ("cap512", 8), ("slots96", 9)):
                index[policy, "train512", rep] = {
                    "experiment": {"policy": policy, "rep": rep}, "fingerprint": "paired",
                    "scene_errors": [], "snapshot": {"alive": 512, "median_remaining": 32},
                    "control": {"native_slot_delays": 32},
                    "metrics": {"blocks": {"request_limit": 1}, "mean_jct_s": mean,
                        "p95_jct_s": 12, "makespan_s": 15, "state_slot_seconds": 100,
                        "candidate": {"mean_jct_s": 10, "mean_ttft_s": 2, "p95_ttft_s": 3}}}
        self.assertEqual(select_baseline(index)["policy"], "cap512")
        index["cap512", "hold384", 81] = {"invalid": "not read"}
        self.assertEqual(select_baseline(index)["policy"], "cap512")
        index["cap512", "train512", 82]["snapshot"]["alive"] = 511
        self.assertEqual(select_baseline(index)["policy"], "slots96")
        index["slots96", "train512", 81]["control"]["native_slot_delays"] = 0
        self.assertEqual(select_baseline(index)["policy"], "native607")


if __name__ == "__main__":
    unittest.main()

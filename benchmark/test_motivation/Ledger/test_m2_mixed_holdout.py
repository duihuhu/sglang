import unittest
from unittest.mock import patch

from analyze_m2_mixed_holdout import scene_matched, select_baseline, triggered
from run_m2_mixed_holdout import POLICIES, PROFILES, RAW_BYTES, cap_for, command, session_plan, specs_for, validate
from run_m2_hybrid_factors import fingerprint


class MixedHoldoutTests(unittest.TestCase):
    def test_commands_share_physical_pool_and_mixed_mode(self):
        cmd = command("native607", 30144)
        self.assertIn("--enable-mixed-chunk", cmd)
        self.assertEqual(cmd, command("cap512", 30144))
        self.assertEqual(cmd, command("decode32", 30144))
        self.assertEqual(command("slots96", 30144), cmd + ["--min-free-slots-delay", "96"])
        self.assertEqual(cap_for("cap512"), 512)
        with self.assertRaises(ValueError):
            command("invalid", 30144)

    def test_frozen_matrix_and_reversed_sessions(self):
        plan = session_plan("measure")
        cells = [(p, profile, phase, s["rep"]) for s in plan for p, profile, phase in s["cases"]]
        expected = {(p, profile, phase, rep) for p in POLICIES for profile in PROFILES
                    for phase in ("far", "near") for rep in (81, 82)}
        self.assertEqual(len(cells), 32)
        self.assertEqual(set(cells), expected)
        for a, b in zip(plan[:3], reversed(plan[3:])):
            self.assertEqual(a["cases"], list(reversed(b["cases"])))
        self.assertEqual(sum(PROFILES[p][0] + 64 for _, p, _, _ in cells), 16384)

    def test_payload_and_arrivals_are_policy_independent(self):
        a = specs_for("native", "train512", 81, 10)
        b = specs_for("diagnostic", "train512", 81, 10)
        self.assertEqual(fingerprint(a), fingerprint(b))
        self.assertEqual(len(a), 576)
        held = specs_for("held", "hold384", 81, 10)
        self.assertEqual(len(held), 448)
        self.assertEqual(len(held[0].input_ids), 1024)
        self.assertEqual(len(specs_for("background", "hold384", 71, None)), 384)

    def test_validate_checks_mixed_and_physical_budget(self):
        info = {"internal_states": [{"min_free_slots_delay": 96, "enable_prefill_delayer": False,
            "m2_admission": {"diagnostic_limit": 0, "shared_pool_bytes": RAW_BYTES}}]}
        with patch("run_m2_mixed_holdout.capacity", return_value=(607, 781725, 607, 607)) as call:
            validate(info, "slots96")
            self.assertEqual(call.call_args.args[1], "mixed-4096")
        info["internal_states"][0]["m2_admission"]["shared_pool_bytes"] += 1
        with self.assertRaises(ValueError):
            validate(info, "slots96")

    def test_scene_matching_is_not_just_near_label(self):
        a = {"scene_errors": [], "snapshot": {"alive": 512, "median_remaining": 32}}
        b = {"scene_errors": [], "snapshot": {"alive": 512, "median_remaining": 40}}
        self.assertTrue(scene_matched(a, b))
        b["snapshot"]["median_remaining"] = 41
        self.assertFalse(scene_matched(a, b))
        b["snapshot"]["median_remaining"] = 32
        b["snapshot"]["alive"] = 511
        self.assertFalse(scene_matched(a, b))

    def test_untriggered_controls_are_not_positive(self):
        row = {"experiment": {"policy": "cap512"}, "control": {}, "metrics": {"blocks": {}}}
        self.assertFalse(triggered(row))
        row["metrics"]["blocks"]["request_limit"] = 1
        self.assertTrue(triggered(row))

    def test_selection_uses_training_only_and_requires_both_phases(self):
        index = {}
        for phase in ("far", "near"):
            for rep in (81, 82):
                for policy, mean in (("native607", 10), ("cap512", 8), ("slots96", 9)):
                    index[policy, "train512", phase, rep] = {
                        "experiment": {"policy": policy, "rep": rep}, "fingerprint": "paired",
                        "scene_errors": [], "snapshot": {"alive": 512, "median_remaining": 32},
                        "control": {"native_slot_delays": 32},
                        "metrics": {"blocks": {"request_limit": 1}, "mean_jct_s": mean,
                            "p95_jct_s": 12, "makespan_s": 15, "state_slot_seconds": 100,
                            "candidate": {"mean_jct_s": 10, "mean_ttft_s": 2, "p95_ttft_s": 3}}}
        self.assertEqual(select_baseline(index)["policy"], "cap512")
        index["cap512", "hold384", "near", 81] = {"invalid": "must not be read"}
        self.assertEqual(select_baseline(index)["policy"], "cap512")
        index["cap512", "train512", "far", 82]["metrics"]["makespan_s"] = 16
        self.assertEqual(select_baseline(index)["policy"], "slots96")
        index["slots96", "train512", "near", 81]["control"]["native_slot_delays"] = 0
        self.assertEqual(select_baseline(index)["policy"], "native607")


if __name__ == "__main__":
    unittest.main()

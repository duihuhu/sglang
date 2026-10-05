import unittest
from unittest.mock import patch

from analyze_m2_native_counterexample import constrained, contrast
from run_m2_native_counterexample import RAW_BYTES, cap_for, command, session_plan, validate


class NativeCalibrationTests(unittest.TestCase):
    def test_cap_is_not_a_physical_pool_change(self):
        self.assertEqual(command("native607", 30144), command("cap512", 30144))
        self.assertEqual(cap_for("native607"), 607)
        self.assertEqual(cap_for("cap512"), 512)
        self.assertEqual(command("slots96", 30144), command("native607", 30144) + ["--min-free-slots-delay", "96"])
        with self.assertRaises(ValueError):
            command("unknown", 30144)

    def test_exact_counterbalanced_matrix(self):
        plan = session_plan()
        cells = [(p, phase, s["rep"]) for s in plan for p, phase in s["cases"]]
        expected = {(p, phase, r) for p in ("native607", "cap512", "slots96") for phase in ("far", "near") for r in (61, 62)}
        self.assertEqual(len(cells), 12)
        self.assertEqual(set(cells), expected)
        self.assertEqual(plan[0]["cases"], list(reversed(plan[3]["cases"])))
        self.assertEqual(plan[1]["cases"], list(reversed(plan[2]["cases"])))

    def test_validate_keeps_actual_runtime_cap(self):
        info = {"internal_states": [{"min_free_slots_delay": None, "enable_prefill_delayer": False,
            "m2_admission": {"diagnostic_limit": 0, "shared_pool_bytes": RAW_BYTES}}]}
        with patch("run_m2_native_counterexample.capacity", return_value=(607, 781725, 607, 512)) as check:
            self.assertEqual(validate(info, "cap512")[-1], 512)
            self.assertEqual(check.call_args.args[-1], 512)
        info["internal_states"][0]["m2_admission"]["shared_pool_bytes"] += 1
        with self.assertRaises(ValueError):
            validate(info, "native607")

    def test_same_resource_not_enough_when_bootstrap_changes(self):
        info = {"internal_states": [{"min_free_slots_delay": 96, "enable_prefill_delayer": False,
            "m2_admission": {"diagnostic_limit": 0, "shared_pool_bytes": RAW_BYTES}}]}
        with patch("run_m2_native_counterexample.capacity", return_value=(512, 781725, 512, 512)):
            with self.assertRaises(ValueError):
                validate(info, "slots96")
        self.assertEqual(info["internal_states"][0]["min_free_slots_delay"], 96)

    def test_ttft_cost_is_not_hidden_by_primary_gate(self):
        a = dict(mean_jct_s=10, p95_jct_s=12, makespan_s=15, state_slot_seconds=100,
                 candidate=dict(mean_jct_s=10, mean_ttft_s=2, p95_ttft_s=3))
        b = {**a, "mean_jct_s": 5, "candidate": dict(mean_jct_s=11, mean_ttft_s=4, p95_ttft_s=5)}
        gain = contrast(a, b)
        self.assertTrue(constrained(gain))
        self.assertEqual(gain["candidate_mean_ttft_s"], -100)
        self.assertFalse(constrained({**gain, "makespan_s": -5.01}))


if __name__ == "__main__":
    unittest.main()

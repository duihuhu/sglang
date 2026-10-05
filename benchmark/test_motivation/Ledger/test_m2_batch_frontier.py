import unittest
from unittest.mock import patch
import socket

from analyze_m2_batch_frontier import dominates, finite, paired_effect
from confirm_m2_batch_frontier import evaluate


class AnalysisTests(unittest.TestCase):
    def test_gap_reduction_is_not_dominance(self):
        baseline = dict(split_p95=10, packed_p95=2, makespan=12)
        slower_neighbor = dict(split_p95=10, packed_p95=10, makespan=12)
        self.assertFalse(dominates(slower_neighbor, baseline))
        self.assertTrue(dominates(baseline, slower_neighbor))

    def test_tradeoff_is_not_dominance(self):
        left = dict(split_p95=5, packed_p95=12, makespan=12)
        right = dict(split_p95=10, packed_p95=8, makespan=14)
        self.assertFalse(dominates(left, right))
        self.assertFalse(dominates(right, left))

    def test_equal_points_do_not_dominate(self):
        point = dict(split_p95=5, packed_p95=12, makespan=12)
        self.assertFalse(dominates(point, point))

    def test_paired_improvement_sign_and_eligibility(self):
        result = paired_effect([10] * 5, [8] * 5, draws=100)
        self.assertAlmostEqual(result["mean_improvement"], 0.2)
        self.assertTrue(result["claim_eligible_n"])
        self.assertAlmostEqual(result["ci95"][0], 0.2)
        self.assertFalse(paired_effect([10], [8], draws=10)["claim_eligible_n"])

    def test_missing_not_zero(self):
        self.assertEqual(finite({"a": float("nan"), "b": [float("inf"), 0]}),
                         {"a": None, "b": [None, 0]})

    def test_unpaired_rejected(self):
        with self.assertRaises(ValueError):
            paired_effect([10, 20], [8])

    def test_confirmation_requires_benefit_without_large_other_loss(self):
        selection = dict(baseline="base", candidate="candidate", last_screening_rep=1,
                         primary_endpoint="split_p95", min_primary_improvement=0.1,
                         noninferiority_margin=0.05)
        rows = [dict(stage="confirm", factor=16, config=config, runs=[
            dict(rep=r, split_p95=short, packed_p95=long, makespan=20)
            for r in range(10, 15)])
            for config, short, long in [("base", 10, 10), ("candidate", 8, 10)]]
        self.assertTrue(evaluate({"rows": rows}, selection, 16)["native_control_benefit_gate_passed"])
        for run in rows[1]["runs"]:
            run["packed_p95"] = 12
        self.assertFalse(evaluate({"rows": rows}, selection, 16)["native_control_benefit_gate_passed"])

    def test_confirmation_cannot_reuse_screen(self):
        selection = dict(baseline="base", candidate="candidate", last_screening_rep=1,
                         primary_endpoint="split_p95", min_primary_improvement=0.1,
                         noninferiority_margin=0.05)
        rows = [dict(stage="confirm", factor=16, config=config, runs=[
            dict(rep=r, split_p95=10, packed_p95=10, makespan=20) for r in range(1, 6)])
            for config in ["base", "candidate"]]
        with self.assertRaises(ValueError):
            evaluate({"rows": rows}, selection, 16)


class RunnerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            import run_m2_batch_frontier
        except ModuleNotFoundError as error:
            if error.name == "aiohttp":
                raise unittest.SkipTest("aiohttp required; runner tests also run in serving container")
            raise
        cls.runner = run_m2_batch_frontier

    def test_paired_input_identity_and_unique_salts(self):
        m = self.runner
        a = m.case_args("a", 1, "mixed", "vtc", 16, 8, 1.5)
        b = m.case_args("b", 1, "mixed", "instant", 16, 8, 1.5)
        sa, sb = m.paired_specs(a, 1), m.paired_specs(b, 1)
        self.assertEqual(m.payload_fingerprint(sa), m.payload_fingerprint(sb))
        self.assertNotEqual(sa[0].cache_salt, sb[0].cache_salt)
        self.assertEqual(len(sa), 136)

    def test_isolation_preserves_offsets_and_inputs(self):
        m = self.runner
        mixed = m.paired_specs(m.case_args("m", 2, "mixed", "vtc", 16, 8, 1.5), 2)
        isolated = m.paired_specs(m.case_args("i", 2, "split-only", "fcfs", 16, 8, 1.5), 2)
        selected = [s for s in mixed if s.group == "split"]
        self.assertEqual(m.payload_fingerprint(selected), m.payload_fingerprint(isolated))

    def test_graph_and_capacity_are_identical_between_modes(self):
        m = self.runner
        for mode in m.MODES:
            command = m.launch_command(mode, 30126)
            for flag, value in [("--max-running-requests", "96"),
                                ("--max-mamba-cache-size", "480"),
                                ("--cuda-graph-max-bs-decode", "96")]:
                self.assertEqual(command[command.index(flag) + 1], value)
            self.assertIn("--disable-prefill-cuda-graph", command)

    def test_confirmation_warms_every_composition_and_cap(self):
        cases = self.runner.warmup_cases("confirm", [48, 96], 8)
        self.assertEqual(len(cases), 6)
        self.assertEqual({case[0] for case in cases}, {48, 96})
        self.assertEqual({case[1] for case in cases},
                         {"mixed", "split-only", "packed-only"})
        self.assertTrue(all(case[3] == 8 for case in cases))
        self.assertEqual(self.runner.warmup_cases("screen", [9, 96], 8),
                         [(96, "mixed", "vtc", 1)])

    def test_live_listener_is_not_treated_as_available(self):
        with socket.socket() as listener:
            listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            listener.bind(("127.0.0.1", 0))
            listener.listen()
            with patch.object(self.runner, "gpu_snapshot", return_value="6,u,g,1,81920,0"):
                with self.assertRaises(OSError):
                    self.runner.ensure_available(listener.getsockname()[1])


if __name__ == "__main__":
    unittest.main()

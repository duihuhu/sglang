import unittest

from analyze_m2_pool_comparison import paired_contrasts


def row(pool, cap, rep, value=100, p95=100, fingerprint="paired"):
    return {"experiment": {"pool": pool, "mode": "mixed-4096", "cap": cap, "rep": rep},
            "payload_fingerprint": fingerprint,
            "metrics": {"mean_jct_s": value, "p95_jct_s": p95, "makespan_s": 100,
                        "mean_queue_s": 100, "state_slot_seconds": 100}}


class ContrastTests(unittest.TestCase):
    def test_gain_direction(self):
        rows = [r for rep in (31, 32) for r in (row("static-natural", 485, rep), row("unified-raised-cap", 607, rep, 90))]
        result = paired_contrasts(rows)[0]
        self.assertEqual(result["comparison"], "unified-and-cap")
        self.assertEqual(result["mean_gain_percent"]["mean_jct_s"], 10)
        self.assertTrue(result["all_repeats_constrained_benefit"])

    def test_tail_regression_violates_constraint(self):
        rows = [r for rep in (31, 32) for r in (row("static-natural", 485, rep), row("unified-raised-cap", 607, rep, 90, 110))]
        self.assertFalse(paired_contrasts(rows)[0]["all_repeats_constrained_benefit"])

    def test_single_repeat_not_labeled_both(self):
        self.assertFalse(paired_contrasts([row("static-natural", 485, 31), row("unified-raised-cap", 607, 31, 90)])[0]["all_repeats_constrained_benefit"])

    def test_unpaired_workload_rejected(self):
        with self.assertRaises(ValueError):
            paired_contrasts([row("static-natural", 485, 31), row("unified-raised-cap", 607, 31, fingerprint="different")])

    def test_duplicate_cell_rejected(self):
        with self.assertRaises(ValueError):
            paired_contrasts([row("static-natural", 485, 31), row("static-natural", 485, 31)])


if __name__ == "__main__":
    unittest.main()

import unittest

from analyze_m2_state_mediation import contrasts


def rows():
    result = []
    for rep in (11, 12):
        for pool, cap, separate, mixed in (("natural", 485, 10., 15.), ("wide", 485, 10.1, 15.2), ("wide", 607, 9., 12.)):
            for mode, value in (("separate-4096", separate), ("mixed-4096", mixed)):
                result.append({"experiment": {"pool": pool, "cap": cap, "mode": mode, "rep": rep},
                               "metrics": {key: value for key in ("all_mean_jct_s", "mean_queue_s", "mean_ttft_s", "makespan_s", "state_slot_seconds")}})
    return result


class ContrastTests(unittest.TestCase):
    def test_positive_interaction_is_tighter_cap_amplification(self):
        result = contrasts(rows())
        self.assertEqual(len(result), 10)
        self.assertAlmostEqual(result[0]["delta_tight_minus_relaxed"], 2.1)

    def test_pool_effect_is_reported_separately(self):
        value = contrasts(rows())[0]["by_mode"]["mixed-4096"]
        self.assertAlmostEqual(value["pool_change_B_minus_A"], .2)
        self.assertAlmostEqual(value["cap_relaxation_B_minus_C"], 3.2)

    def test_negative_interaction_is_not_clamped_or_relabelled(self):
        values = rows()
        for row in values:
            if row["experiment"]["cap"] == 607 and row["experiment"]["mode"] == "mixed-4096":
                row["metrics"]["all_mean_jct_s"] = 20.
        self.assertLess(contrasts(values)[0]["delta_tight_minus_relaxed"], 0)

    def test_missing_cell_is_rejected(self):
        with self.assertRaises(KeyError):
            contrasts(rows()[:-1])

    def test_duplicate_cell_is_rejected(self):
        values = rows()
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            contrasts(values + values[:1])


if __name__ == "__main__":
    unittest.main()

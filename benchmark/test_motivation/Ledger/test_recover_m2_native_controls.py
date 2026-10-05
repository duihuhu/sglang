import copy
import unittest

from recover_m2_native_controls import check_adopted, verify_gpu_identity


class RecoveryTests(unittest.TestCase):
    def test_exact_rep51_matrix_required(self):
        rows = [{"experiment": {"policy": p, "profile": profile, "phase": phase, "rep": 51}}
                for p in ("slots64", "delayer32") for profile in ("state", "kv") for phase in ("near", "far")]
        missing = check_adopted(rows)
        self.assertEqual(len(missing), 8)
        self.assertTrue(all(cell[-1] == 52 for cell in missing))
        for invalid in (rows[:-1], rows+[copy.deepcopy(rows[0])]):
            with self.assertRaises(ValueError):
                check_adopted(invalid)
        rows[0]["experiment"]["rep"] = 52
        with self.assertRaises(ValueError):
            check_adopted(rows)

    def test_physical_identity_on_all_idle_snapshots(self):
        expected = [{"index": 4, "uuid": "a"}, {"index": 5, "uuid": "b"}]
        verify_gpu_identity([expected, list(reversed(expected))], expected)
        changed = copy.deepcopy(expected)
        changed[0]["uuid"] = "replacement"
        with self.assertRaises(ValueError):
            verify_gpu_identity([expected, changed], expected)


if __name__ == "__main__":
    unittest.main()

"""Small arithmetic tests locally; native CPU/meta tests in the serving venv."""

import importlib.util
import os
import unittest

from preflight_m2_pool_comparison import bootstrap_plan, native_request_cap, unique_storage_bytes


class BudgetTests(unittest.TestCase):
    def setUp(self):
        self.rows = bootstrap_plan(21708800, 8064, 1048576, (485, 1458643, 485))

    def test_three_primary_arms_and_stale_cap_control(self):
        self.assertEqual([r["bootstrap_state_cap"] for r in self.rows], [485, 607, 607, 485])

    def test_static_budget_matches_to_one_token(self):
        self.assertLessEqual(self.rows[1]["main_budget_delta_bytes"], 0)
        self.assertGreater(self.rows[1]["main_budget_delta_bytes"], -8064)

    def test_unified_does_not_add_static_dummy_state(self):
        self.assertEqual(self.rows[1]["main_allocation_bytes"] - self.rows[2]["main_allocation_bytes"], 21708800)

    def test_both_layouts_charge_mla_padding(self):
        for row in self.rows:
            self.assertEqual(row["main_allocation_bytes"] - row["main_cache_bytes"], 8064)

    def test_insufficient_budget_is_rejected(self):
        with self.assertRaises(ValueError):
            bootstrap_plan(21708800, 8064, 1048576, (485, 1458643, 485), target=1500)

    def test_bad_reference_is_rejected(self):
        with self.assertRaises(ValueError):
            bootstrap_plan(21708800, 8064, 1048576, (485, 1458643, 400))

    def test_nonpositive_geometry_is_rejected(self):
        with self.assertRaises(ValueError):
            bootstrap_plan(0, 8064, 1048576, (485, 1458643, 485))


@unittest.skipUnless(importlib.util.find_spec("torch"), "torch is required for native CPU/meta tests")
class NativeTests(unittest.TestCase):
    def setUp(self):
        if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
            self.fail("Hide CUDA before running native CPU/meta tests")
        import torch
        self.torch = torch

    def test_unified_switch_alone_does_not_raise_cap(self):
        for unified in (False, True):
            self.assertEqual(native_request_cap(485, 1458643, 1048576, unified), 485)

    def test_explicit_request_flag_cannot_bypass_old_state_cap(self):
        self.assertEqual(native_request_cap(485, 1458643, 1048576, True, requested=607), 485)

    def test_raised_bootstrap_cap(self):
        self.assertEqual(native_request_cap(607, 1000000, 1048576, True), 607)

    def test_smaller_explicit_request_cap_is_preserved(self):
        self.assertEqual(native_request_cap(607, 1000000, 1048576, True, requested=485), 485)

    def test_token_constraint_still_applies(self):
        self.assertEqual(native_request_cap(607, 400, 1048576, True), 200)

    def test_shared_views_are_not_counted_twice(self):
        for device in ("cpu", "meta"):
            raw = self.torch.empty(128, dtype=self.torch.uint8, device=device)
            other = self.torch.empty_like(raw)
            result = unique_storage_bytes({"raw": raw, "view": raw[4:32], "cast": raw.view(self.torch.int32), "other": other})
            self.assertEqual(result["unique_bytes"], 256)
            self.assertEqual(result["storage_count"], 2)


if __name__ == "__main__":
    unittest.main()

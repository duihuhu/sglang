import copy
import unittest

from prepare_m2_pool_budget import ARMS, GIB, budget_plan, frozen_sessions, refine_plan


def sample_rows():
    rows = {}
    for arm in ARMS:
        raised = arm in ("static-rebalanced", "unified-raised-cap")
        extra = 512 << 20 if raised else 0
        rows[arm] = {"memory": [{"allocated": 74 * GIB + extra, "cuda_used": 76 * GIB + extra}] * 2,
                     "kv_tokens": 1130211 if raised else 1458643,
                     "capacity": [607 if raised else 485, 1000000, 607 if raised else 485, 485],
                     "config": {"kv_tokens": 800000 if raised else 1190000},
                     "log_allocation_retry_warning_count": 0}
    return rows


class BudgetPlanTests(unittest.TestCase):
    def test_reduction_uses_memory_not_latency(self):
        rows = sample_rows()
        sessions, targets = budget_plan(rows)
        self.assertEqual(targets, {"allocated": 72 * GIB, "cuda_used": 74 * GIB})
        self.assertEqual(len(sessions), 4)
        for session in sessions:
            self.assertLess(session["kv_tokens"], rows[session["arm"]]["kv_tokens"])
            self.assertGreaterEqual(session["kv_tokens"], 2 * 607 * 640)

    def test_rejects_insufficient_kv_headroom(self):
        rows = sample_rows()
        rows["static-rebalanced"]["kv_tokens"] = 400000
        with self.assertRaises(ValueError):
            budget_plan(rows)

    def test_unmatched_memory_blocks_freeze(self):
        with self.assertRaises(ValueError):
            frozen_sessions(sample_rows())

    def test_counterbalanced_matrix(self):
        rows = sample_rows()
        for row in rows.values():
            row["memory"] = copy.deepcopy(rows["static-natural"]["memory"])
        sessions = frozen_sessions(rows)
        self.assertEqual(len(sessions), 16)
        self.assertEqual(sum(len(s["caps"]) for s in sessions), 24)
        for a, b in zip(sessions[:8], reversed(sessions[8:])):
            self.assertEqual((a["arm"], a["mode"]), (b["arm"], b["mode"]))
            self.assertEqual(a["caps"], list(reversed(b["caps"])))

    def test_recovered_allocation_retry_is_not_request_failure(self):
        rows = sample_rows()
        for row in rows.values():
            row["memory"] = copy.deepcopy(rows["static-natural"]["memory"])
        rows["unified-raised-cap"]["log_allocation_retry_warning_count"] = 1
        self.assertEqual(len(frozen_sessions(rows)), 16)

    def test_refinement_only_reduces_high_memory_conditions(self):
        rows = sample_rows()
        for row in rows.values():
            row["memory"] = copy.deepcopy(rows["static-natural"]["memory"])
        rows["static-natural"]["memory"][0]["allocated"] += 90 << 20
        sessions, targets = refine_plan(rows)
        self.assertEqual([s["arm"] for s in sessions], ["static-natural"])
        self.assertLess(sessions[0]["kv_tokens"], rows["static-natural"]["kv_tokens"])


if __name__ == "__main__":
    unittest.main()

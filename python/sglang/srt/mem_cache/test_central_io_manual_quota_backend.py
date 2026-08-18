"""Tests for binding manual quota intents to Central I/O control targets."""

from __future__ import annotations

import unittest

from sglang.srt.mem_cache.central_io_manual_quota_backend import (
    CentralIOTargetQuotaBackend,
)


class _Control:
    def __init__(self):
        self.targets = []
        self.models = {
            "hot": {
                "capacity": 30,
                "page_size": 1,
                "token_bytes": 10,
                "pending_capacity": 0,
                "pending_release_capacity": 0,
                "residency": {"floor_pages": 10},
            },
            "cold": {
                "capacity": 40,
                "page_size": 1,
                "token_bytes": 10,
                "pending_capacity": 0,
                "pending_release_capacity": 0,
                "residency": {
                    "floor_pages": 12,
                    "clean_pages": 20,
                    "ready_pages": 6,
                    "high_pages": 8,
                    "unresolved_clean_pages": 0,
                },
            },
        }

    def global_status(self):
        return {
            "global_free_bytes": 300,
            "models": {model_id: dict(state) for model_id, state in self.models.items()},
        }

    def set_quota_targets(self, targets):
        self.targets.append(dict(targets))
        for model_id, target in targets.items():
            self.models[model_id]["capacity"] = target
        return {"targets": targets, "target_set_ns": 123}


class _StepControl(_Control):
    def __init__(self, capacities):
        super().__init__()
        self.capacities = list(capacities)

    def global_status(self):
        if self.capacities:
            next_capacity = self.capacities.pop(0)
            for model_id, capacity in next_capacity.items():
                self.models[model_id]["capacity"] = capacity
        return super().global_status()

    def set_quota_targets(self, targets):
        self.targets.append(dict(targets))
        return {"targets": targets, "target_set_ns": 123}


class CentralIOTargetQuotaBackendTest(unittest.TestCase):
    def test_snapshot_converts_central_io_slots_to_pages_without_policy_metrics(self):
        backend = CentralIOTargetQuotaBackend(_Control(), wait_interval_s=0)

        snapshot = backend.snapshot()

        self.assertEqual(snapshot["global_pool_pages"], 100)
        self.assertEqual(snapshot["models"]["hot"], {"effective_pages": 30, "floor_pages": 10})
        self.assertEqual(snapshot["models"]["cold"], {"effective_pages": 40, "floor_pages": 12})

    def test_transfer_pages_publishes_atomic_targets_and_waits_for_effective_capacity(self):
        control = _Control()
        backend = CentralIOTargetQuotaBackend(control, wait_interval_s=0)

        result = backend.transfer_pages("cold", "hot", 7, "operator-safe-offer")

        self.assertEqual(control.targets, [{"cold": 33, "hot": 37}])
        self.assertEqual(result["donor_effective_pages"], 33)
        self.assertEqual(result["recipient_effective_pages"], 37)

    def test_grow_and_shrink_publish_only_explicit_targets(self):
        control = _Control()
        backend = CentralIOTargetQuotaBackend(control, wait_interval_s=0)

        grow = backend.grow_from_reserve("hot", 5)
        shrink = backend.shrink_to_reserve("cold", 3)

        self.assertEqual(control.targets, [{"hot": 35}, {"cold": 37}])
        self.assertEqual(grow["effective_pages"], 35)
        self.assertEqual(shrink["effective_pages"], 37)

    def test_begin_grow_publishes_final_target_once_and_waits_boundaries(self):
        control = _StepControl([{"hot": 30}, {"hot": 34}, {"hot": 37}, {"hot": 42}])
        backend = CentralIOTargetQuotaBackend(control, wait_interval_s=0)

        plan = backend.begin_grow_from_reserve("hot", 12)
        first = backend.wait_grow_boundary("hot", 37)
        full = backend.wait_grow_boundary("hot", 42)

        self.assertEqual(control.targets, [{"hot": 42}])
        self.assertEqual(plan["start_effective_pages"], 30)
        self.assertEqual(plan["target_effective_pages"], 42)
        self.assertEqual(first["effective_pages"], 37)
        self.assertEqual(full["effective_pages"], 42)

    def test_begin_transfer_publishes_atomic_final_target_once_and_waits_boundaries(self):
        control = _StepControl(
            [
                {"cold": 40, "hot": 30},
                {"cold": 36, "hot": 34},
                {"cold": 34, "hot": 36},
                {"cold": 27, "hot": 43},
            ]
        )
        backend = CentralIOTargetQuotaBackend(control, wait_interval_s=0)

        plan = backend.begin_transfer_pages(
            "cold", "hot", 13, "operator-safe-offer-13"
        )
        first = backend.wait_transfer_boundary("cold", 34, "hot", 36)
        full = backend.wait_transfer_boundary("cold", 27, "hot", 43)

        self.assertEqual(control.targets, [{"cold": 27, "hot": 43}])
        self.assertEqual(plan["donor_target_effective_pages"], 27)
        self.assertEqual(plan["recipient_target_effective_pages"], 43)
        self.assertEqual(first["recipient_effective_pages"], 36)
        self.assertEqual(full["donor_effective_pages"], 27)

    def test_donation_offer_reads_allocator_health_without_publishing_targets(self):
        control = _Control()
        backend = CentralIOTargetQuotaBackend(control, wait_interval_s=0)

        offer = backend.donation_offer("cold")

        self.assertEqual(offer["immediate_pages"], 12)
        self.assertEqual(offer["ready_pages"], 6)
        self.assertEqual(offer["total_pages"], 18)
        self.assertEqual(control.targets, [])


if __name__ == "__main__":
    unittest.main()

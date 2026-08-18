"""CPU checks for LatticeKV's pressure-triggered global quota decisions."""

from __future__ import annotations

import unittest

from sglang.srt.mem_cache.pressure_scheduler import (
    InstancePressure,
    PressureScheduler,
)


def instance(
    model_id: str,
    *,
    effective: int,
    floor: int,
    clean: int,
    ready: int,
    minimum: int,
    low: int,
    high: int,
    debt: int = 0,
    shortfall: int = 0,
    ready_score: float = 0.0,
) -> InstancePressure:
    return InstancePressure(
        model_id=model_id,
        effective_bytes=effective,
        floor_bytes=floor,
        clean_bytes=clean,
        ready_bytes=ready,
        min_bytes=minimum,
        low_bytes=low,
        high_bytes=high,
        retention_debt_bytes=debt,
        unresolved_clean_bytes=shortfall,
        ready_reclaim_score=ready_score,
    )


class PressureSchedulerTest(unittest.TestCase):
    def test_retention_debt_moves_only_safe_donor_supply_to_hot_instance(self):
        scheduler = PressureScheduler(max_transfer_bytes=500)
        hot = instance(
            "hot", effective=2_000, floor=1_000, clean=100, ready=0,
            minimum=200, low=400, high=600, debt=800,
        )
        cold = instance(
            "cold", effective=2_000, floor=1_000, clean=1_100, ready=500,
            minimum=200, low=300, high=400,
        )

        decision = scheduler.decide([hot, cold], global_free_bytes=0)

        self.assertEqual(decision.target_bytes["hot"], 2_500)
        self.assertEqual(decision.target_bytes["cold"], 1_500)
        self.assertEqual(decision.transfers[0].source_model_id, "cold")
        self.assertEqual(decision.transfers[0].destination_model_id, "hot")
        self.assertEqual(decision.transfers[0].bytes, 500)

    def test_low_clean_space_without_retention_debt_stays_local(self):
        scheduler = PressureScheduler(max_transfer_bytes=500)
        hot = instance(
            "hot", effective=2_000, floor=1_000, clean=100, ready=0,
            minimum=200, low=400, high=600,
        )
        cold = instance(
            "cold", effective=2_000, floor=1_000, clean=1_100, ready=0,
            minimum=200, low=300, high=400,
        )

        decision = scheduler.decide([hot, cold], global_free_bytes=0)

        self.assertEqual(decision.target_bytes, {"hot": 2_000, "cold": 2_000})
        self.assertEqual(decision.transfers, [])

    def test_recent_retention_debt_prevents_an_instance_from_being_a_donor(self):
        scheduler = PressureScheduler(max_transfer_bytes=500)
        hot = instance(
            "hot", effective=2_000, floor=1_000, clean=100, ready=0,
            minimum=200, low=400, high=600, debt=800,
        )
        recovering = instance(
            "recovering", effective=2_000, floor=1_000, clean=1_500, ready=0,
            minimum=200, low=300, high=400, debt=1,
        )

        decision = scheduler.decide([hot, recovering], global_free_bytes=0)

        self.assertEqual(decision.target_bytes["hot"], 2_000)
        self.assertEqual(decision.target_bytes["recovering"], 2_000)
        self.assertEqual(decision.unmet_bytes["hot"], 500)

    def test_ready_donors_are_ranked_by_observed_reclaim_loss(self):
        scheduler = PressureScheduler(max_transfer_bytes=500)
        hot = instance(
            "hot", effective=2_000, floor=1_000, clean=100, ready=0,
            minimum=200, low=400, high=600, debt=700,
        )
        expensive = instance(
            "expensive", effective=2_000, floor=1_000, clean=400, ready=500,
            minimum=200, low=300, high=400, ready_score=500,
        )
        cheap = instance(
            "cheap", effective=2_000, floor=1_000, clean=400, ready=500,
            minimum=200, low=300, high=400, ready_score=5,
        )

        decision = scheduler.decide([hot, expensive, cheap], global_free_bytes=0)

        self.assertEqual(decision.transfers[0].source_model_id, "cheap")
        self.assertEqual(decision.transfers[0].readiness, "ready")

    def test_unresolved_backup_shortfall_can_request_growth_before_revisit_loss(self):
        scheduler = PressureScheduler(max_transfer_bytes=500)
        hot = instance(
            "hot", effective=2_000, floor=1_000, clean=100, ready=0,
            minimum=200, low=400, high=600, shortfall=350,
        )
        cold = instance(
            "cold", effective=2_000, floor=1_000, clean=1_100, ready=0,
            minimum=200, low=300, high=400,
        )

        decision = scheduler.decide([hot, cold], global_free_bytes=0)

        # The immediate backup deficit is 350B, but reaching this instance's
        # high watermark requires 500B.  One grow should restore circulation,
        # not merely make the current allocation barely succeed.
        self.assertEqual(decision.target_bytes["hot"], 2_500)
        self.assertEqual(decision.target_bytes["cold"], 1_500)

    def test_ready_candidates_do_not_reduce_clean_capacity_growth_need(self):
        scheduler = PressureScheduler(max_transfer_bytes=500)
        hot = instance(
            "hot", effective=2_000, floor=1_000, clean=100, ready=500,
            minimum=200, low=400, high=600, shortfall=350,
        )
        cold = instance(
            "cold", effective=2_000, floor=1_000, clean=1_100, ready=0,
            minimum=200, low=300, high=400,
        )

        decision = scheduler.decide([hot, cold], global_free_bytes=0)

        # Ready candidates still contain host KV. They cannot absorb the next
        # HBM backup, so hot needs the full clean gap to reach its high mark.
        self.assertEqual(decision.target_bytes["hot"], 2_500)
        self.assertEqual(decision.target_bytes["cold"], 1_500)

    def test_retention_debt_with_healthy_clean_headroom_stays_local(self):
        scheduler = PressureScheduler(
            max_transfer_bytes=500,
            growth_quantum_bytes=200,
        )
        hot = instance(
            "hot", effective=2_000, floor=1_000, clean=700, ready=0,
            minimum=200, low=400, high=600, debt=1,
        )
        cold = instance(
            "cold", effective=2_000, floor=1_000, clean=1_100, ready=0,
            minimum=200, low=300, high=400,
        )

        decision = scheduler.decide([hot, cold], global_free_bytes=0)

        # A past reclaim loss protects the instance from donation, but clean
        # host capacity is already above high. Growing here would reproduce
        # the V8 failure mode: every fresh loss consumes another quota quantum
        # even though local circulation is currently healthy.
        self.assertEqual(decision.target_bytes["hot"], 2_000)
        self.assertEqual(decision.target_bytes["cold"], 2_000)

    def test_retention_debt_below_low_requests_one_bounded_quota_quantum(self):
        scheduler = PressureScheduler(
            max_transfer_bytes=500,
            growth_quantum_bytes=200,
        )
        hot = instance(
            "hot", effective=2_000, floor=1_000, clean=300, ready=0,
            minimum=200, low=400, high=600, debt=1,
        )
        cold = instance(
            "cold", effective=2_000, floor=1_000, clean=1_100, ready=0,
            minimum=200, low=300, high=400,
        )

        decision = scheduler.decide([hot, cold], global_free_bytes=0)

        # The one quantum is a floor, not a cap: recovering from 300 to the
        # 600-byte high watermark needs 300 bytes in this concrete state.
        self.assertEqual(decision.target_bytes["hot"], 2_300)
        self.assertEqual(decision.target_bytes["cold"], 1_700)


if __name__ == "__main__":
    unittest.main()

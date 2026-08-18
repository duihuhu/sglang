"""CPU-only checks for LatticeKV local host-residency control."""

from __future__ import annotations

import unittest

from sglang.srt.mem_cache.local_residency import (
    LocalAction,
    LocalResidencyState,
)


class LocalResidencyStateTest(unittest.TestCase):
    def make_state(self) -> LocalResidencyState:
        return LocalResidencyState(
            effective_pages=100,
            floor_pages=20,
            large_backup_batch_pages=8,
            backup_ingress_pages_per_s=4.0,
            ready_latency_p95_s=1.0,
            maintenance_batch_pages=12,
            maintenance_cycle_s=2.0,
            retention_debt_cooldown_s=10.0,
        )

    def test_low_watermark_starts_and_high_watermark_stops_preparation(self):
        state = self.make_state()
        self.assertEqual(state.watermarks.min_pages, 8)
        self.assertEqual(state.watermarks.low_pages, 20)
        self.assertEqual(state.watermarks.high_pages, 28)

        state.observe_pages(clean_pages=19, ready_pages=0, live_pages=81)
        self.assertEqual(state.action(now_s=0.0), LocalAction.PREPARE)

        state.observe_pages(clean_pages=24, ready_pages=0, live_pages=76)
        self.assertEqual(state.action(now_s=1.0), LocalAction.PREPARE)

        state.observe_pages(clean_pages=28, ready_pages=0, live_pages=72)
        self.assertEqual(state.action(now_s=2.0), LocalAction.IDLE)

    def test_below_min_requires_emergency_preparation(self):
        state = self.make_state()
        state.observe_pages(clean_pages=7, ready_pages=0, live_pages=93)
        self.assertEqual(state.action(now_s=0.0), LocalAction.EMERGENCY)

    def test_ready_pages_do_not_hide_a_clean_page_emergency(self):
        state = self.make_state()
        state.observe_pages(clean_pages=7, ready_pages=21, live_pages=93)

        self.assertEqual(state.action(now_s=0.0), LocalAction.EMERGENCY)

    def test_backup_observation_updates_watermarks_from_hbm_ingress(self):
        state = self.make_state()
        state.observe_backup(pages=24, elapsed_s=2.0)

        self.assertEqual(state.backup_ingress_pages_per_s, 12.0)
        self.assertEqual(state.watermarks.min_pages, 24)
        self.assertEqual(state.watermarks.low_pages, 36)
        self.assertEqual(state.watermarks.high_pages, 60)

    def test_recent_backup_burst_keeps_admission_slack_only_within_its_window(self):
        state = LocalResidencyState(
            effective_pages=200,
            floor_pages=20,
            large_backup_batch_pages=8,
            backup_ingress_pages_per_s=4.0,
            ready_latency_p95_s=1.0,
            maintenance_batch_pages=12,
            maintenance_cycle_s=2.0,
            retention_debt_cooldown_s=10.0,
            ingress_window_s=5.0,
        )
        state.observe_backup(pages=40, elapsed_s=1.0, now_s=10.0)

        self.assertEqual(state.watermarks_at(now_s=14.9).min_pages, 40)
        self.assertEqual(state.watermarks_at(now_s=15.0).min_pages, 8)

    def test_large_backup_batch_sets_min_even_when_its_average_rate_is_small(self):
        state = self.make_state()
        state.observe_backup(pages=24, elapsed_s=20.0, now_s=10.0)

        self.assertEqual(state.watermarks_at(now_s=10.0).min_pages, 24)

    def test_observed_reclaim_latency_expands_backup_admission_minimum(self):
        state = self.make_state()
        state.observe_reclaim_latency(elapsed_s=3.0)

        # Four incoming pages per second may arrive while one prepared page
        # becomes clean.  The minimum must cover that real delay rather than
        # retain the zero-cost/default estimate.
        self.assertEqual(state.ready_latency_p95_s, 3.0)
        self.assertEqual(state.watermarks.min_pages, 12)

    def test_donor_only_offers_capacity_above_its_recovery_target(self):
        state = self.make_state()
        state.observe_pages(clean_pages=40, ready_pages=0, live_pages=60)

        offer = state.offer_donation(requested_pages=50, now_s=0.0)

        self.assertEqual(offer.immediate_pages, 12)
        self.assertEqual(offer.ready_pages, 0)
        self.assertEqual(offer.low_loss_pages, 0)

    def test_donor_reports_prepared_pages_as_ready_soon_not_immediate(self):
        state = self.make_state()
        state.observe_pages(clean_pages=20, ready_pages=20, live_pages=80)

        offer = state.offer_donation(requested_pages=50, now_s=0.0)

        self.assertEqual(offer.immediate_pages, 0)
        self.assertEqual(offer.ready_pages, 12)
        self.assertEqual(offer.low_loss_pages, 0)

    def test_donor_never_offers_pages_below_its_configured_floor(self):
        state = LocalResidencyState(
            effective_pages=100,
            floor_pages=90,
            large_backup_batch_pages=8,
            backup_ingress_pages_per_s=4.0,
            ready_latency_p95_s=1.0,
            maintenance_batch_pages=12,
            maintenance_cycle_s=2.0,
            retention_debt_cooldown_s=10.0,
        )
        state.observe_pages(clean_pages=100, ready_pages=0, live_pages=0)

        offer = state.offer_donation(requested_pages=50, now_s=0.0)

        self.assertEqual(offer.immediate_pages, 10)
        self.assertEqual(offer.total_pages, 10)

    def test_recent_reclaim_loss_blocks_donation_until_cooldown_expires(self):
        state = self.make_state()
        state.observe_pages(clean_pages=40, ready_pages=0, live_pages=60)
        state.record_retention_loss(reprefill_tokens=4096, now_s=0.0)

        self.assertEqual(state.offer_donation(requested_pages=12, now_s=1.0).total_pages, 0)
        self.assertEqual(state.offer_donation(requested_pages=12, now_s=11.0).total_pages, 12)

    def test_status_refresh_expires_old_retention_debt_without_a_donor_request(self):
        state = self.make_state()
        state.record_retention_loss(reprefill_tokens=4096, now_s=0.0)

        self.assertEqual(state.active_retention_debt_tokens(now_s=1.0), 4096)
        self.assertEqual(state.active_retention_debt_tokens(now_s=10.0), 0)

    def test_unresolved_admission_shortfall_blocks_donation_without_waiting_for_miss(self):
        state = self.make_state()
        state.observe_pages(clean_pages=40, ready_pages=0, live_pages=60)
        state.record_clean_shortfall(pages=6)

        self.assertEqual(state.unresolved_clean_pages, 6)
        self.assertEqual(state.offer_donation(requested_pages=12, now_s=0.0).total_pages, 0)
        state.record_clean_shortfall(pages=0)
        self.assertEqual(state.offer_donation(requested_pages=12, now_s=0.0).total_pages, 12)

    def test_intent_does_not_change_effective_quota_until_ack(self):
        state = self.make_state()
        transfer = state.begin_intent(target_pages=120)

        self.assertEqual(state.effective_pages, 100)
        state.mark_pages_assigned(transfer.transfer_id, pages=20)
        self.assertEqual(state.effective_pages, 100)
        state.acknowledge_recipient(transfer.transfer_id, pages=20)
        self.assertEqual(state.effective_pages, 120)


if __name__ == "__main__":
    unittest.main()

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
        self.assertEqual(state.watermarks.min_pages, 12)
        self.assertEqual(state.watermarks.low_pages, 24)
        self.assertEqual(state.watermarks.high_pages, 36)

        state.observe_pages(clean_pages=19, ready_pages=0, live_pages=81)
        self.assertEqual(state.action(now_s=0.0), LocalAction.PREPARE)

        state.observe_pages(clean_pages=24, ready_pages=0, live_pages=76)
        self.assertEqual(state.action(now_s=1.0), LocalAction.PREPARE)

        state.observe_pages(clean_pages=36, ready_pages=0, live_pages=64)
        self.assertEqual(state.action(now_s=2.0), LocalAction.IDLE)

    def test_below_min_requires_emergency_preparation(self):
        state = self.make_state()
        state.observe_pages(clean_pages=7, ready_pages=0, live_pages=93)
        self.assertEqual(state.action(now_s=0.0), LocalAction.EMERGENCY)

    def test_ready_pages_do_not_hide_a_clean_page_emergency(self):
        state = self.make_state()
        state.observe_pages(clean_pages=7, ready_pages=21, live_pages=93)

        self.assertEqual(state.action(now_s=0.0), LocalAction.EMERGENCY)

    def test_ready_candidates_do_not_stop_low_watermark_cleaning(self):
        state = self.make_state()
        # These pages have passed value checks, but they still hold KV and
        # cannot receive the next HBM backup until the maintainer deletes
        # their owners.  They must not make the local action look healthy.
        state.observe_pages(clean_pages=12, ready_pages=30, live_pages=88)

        self.assertEqual(state.action(now_s=0.0), LocalAction.PREPARE)

    def test_backup_observation_updates_watermarks_from_hbm_ingress(self):
        state = self.make_state()
        state.observe_backup(pages=24, elapsed_s=0.001, now_s=0.0)
        state.observe_backup(pages=24, elapsed_s=0.001, now_s=2.0)

        self.assertEqual(state.backup_ingress_pages_per_s, 12.0)
        self.assertEqual(state.watermarks.min_pages, 36)
        # ``low`` keeps one observed maintenance cycle before ``min`` and
        # ``high`` keeps one further cycle after recovery.
        self.assertEqual(state.watermarks.low_pages, 60)
        self.assertEqual(state.watermarks.high_pages, 84)

    def test_backup_copy_bandwidth_is_not_mistaken_for_host_ingress_rate(self):
        state = LocalResidencyState(
            effective_pages=30_000,
            floor_pages=20,
            large_backup_batch_pages=8,
            backup_ingress_pages_per_s=4.0,
            ready_latency_p95_s=1.0,
            maintenance_batch_pages=12,
            maintenance_cycle_s=2.0,
            retention_debt_cooldown_s=10.0,
            ingress_window_s=30.0,
        )

        # The first 8K-page GPU->host copy can complete quickly.  That says
        # nothing about how frequently independent HBM eviction batches will
        # arrive, so it must not inflate low/high watermarks to the quota cap.
        state.observe_backup(pages=8_192, elapsed_s=0.050, now_s=10.0)
        self.assertEqual(state.backup_ingress_pages_per_s, 4.0)
        self.assertEqual(state.watermarks.min_pages, 8_196)
        self.assertEqual(state.watermarks.high_pages, 8_220)

        # A second batch ten seconds later establishes the actual sustained
        # host ingress: 8,192 pages / 10s, not 8,192 / 50ms.
        state.observe_backup(pages=8_192, elapsed_s=0.010, now_s=20.0)
        self.assertEqual(state.backup_ingress_pages_per_s, 819.2)
        self.assertEqual(state.watermarks.low_pages, 10_651)
        self.assertEqual(state.watermarks.high_pages, 12_290)

    def test_recent_backup_batch_is_retained_longer_than_the_ingress_rate(self):
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

        self.assertEqual(state.watermarks_at(now_s=14.9).min_pages, 44)
        # The 5s ingress window can forget an old rate burst, but one normal
        # 40-page HBM backup remains an admission requirement for the longer
        # batch window.  Otherwise the next node-sized backup would force a
        # synchronous eviction even though average ingress is quiet.
        self.assertEqual(state.watermarks_at(now_s=15.0).min_pages, 44)
        self.assertEqual(state.watermarks_at(now_s=69.9).min_pages, 44)
        self.assertEqual(state.watermarks_at(now_s=70.0).min_pages, 12)

    def test_large_backup_batch_sets_min_even_when_its_average_rate_is_small(self):
        state = self.make_state()
        state.observe_backup(pages=24, elapsed_s=20.0, now_s=10.0)

        self.assertEqual(state.watermarks_at(now_s=10.0).min_pages, 28)

    def test_observed_reclaim_latency_expands_backup_admission_minimum(self):
        state = self.make_state()
        state.observe_reclaim_latency(elapsed_s=3.0)

        # Four incoming pages per second may arrive while one prepared page
        # becomes clean.  The minimum must cover that real delay rather than
        # retain the zero-cost/default estimate.
        self.assertEqual(state.ready_latency_p95_s, 3.0)
        self.assertEqual(state.watermarks.min_pages, 20)

    def test_hard_minimum_covers_next_backup_while_ready_page_converts(self):
        state = self.make_state()
        state.observe_reclaim_latency(elapsed_s=3.0)

        # A page already entering admission can be followed by another normal
        # HBM backup before the prepared page becomes clean.  Reserving only
        # the larger of the two terms permits the next discrete backup to
        # cross ``min`` and force an avoidable emergency path.
        self.assertEqual(state.watermarks.min_pages, 20)

    def test_reclaim_throughput_deficit_requests_global_help_only_below_low(self):
        state = LocalResidencyState(
            effective_pages=200,
            floor_pages=20,
            large_backup_batch_pages=8,
            backup_ingress_pages_per_s=0.0,
            ready_latency_p95_s=1.0,
            maintenance_batch_pages=12,
            maintenance_cycle_s=2.0,
            retention_debt_cooldown_s=10.0,
        )
        # Four completed waves establish a conservative 4 pages/s local
        # reclaim rate.  Two backup observations establish 12 pages/s ingress.
        for now_s in range(4):
            state.observe_reclaim_result(pages=4, elapsed_s=1.0, now_s=float(now_s))
        state.observe_backup(pages=12, elapsed_s=1.0, now_s=4.0)
        state.observe_backup(pages=12, elapsed_s=1.0, now_s=5.0)

        state.observe_pages(clean_pages=48, ready_pages=0, live_pages=152)
        self.assertEqual(state.turnover_deficit_pages(now_s=5.0), 0)

        state.observe_pages(clean_pages=20, ready_pages=0, live_pages=180)
        self.assertEqual(state.backup_ingress_pages_per_s, 12.0)
        self.assertEqual(state.safe_reclaim_pages_per_s(now_s=5.0), 4.0)
        self.assertEqual(state.turnover_deficit_pages(now_s=5.0), 16)

    def test_retention_debt_and_turnover_deficit_stop_proactive_reclaim_above_min(self):
        state = self.make_state()
        # Four completed waves establish that value-safe reclamation can only
        # produce one page/s, while this instance is receiving four pages/s.
        # A quick revisit then proves that further proactive deletion is
        # already damaging reuse.  The instance is below ``low`` but still
        # above ``min``: it must report a quota deficit instead of letting
        # every subsequent backup spend a fresh maintenance budget reclaiming
        # more history.
        for now_s in range(4):
            state.observe_reclaim_result(pages=1, elapsed_s=1.0, now_s=float(now_s))
        state.observe_pages(clean_pages=16, ready_pages=0, live_pages=84)
        state.record_retention_loss(reprefill_tokens=4096, now_s=4.0)

        self.assertGreater(state.turnover_deficit_pages(now_s=4.0), 0)
        self.assertEqual(state.action(now_s=4.0), "quota_deficit")

    def test_retention_debt_stops_proactive_reclaim_before_throughput_deficit(self):
        state = self.make_state()
        # A fast revisit is already direct evidence that proactive reclaim is
        # harming reuse.  The local reclaim-rate estimator may still be empty
        # (or briefly optimistic), so a throughput deficit is not a safe
        # prerequisite for stopping ordinary reclaim above the hard minimum.
        state.observe_pages(clean_pages=16, ready_pages=0, live_pages=84)
        state.record_retention_loss(reprefill_tokens=4096, now_s=1.0)

        self.assertEqual(state.turnover_deficit_pages(now_s=1.0), 0)
        self.assertEqual(state.action(now_s=1.0), "quota_deficit")

    def test_donor_only_offers_capacity_above_its_recovery_target(self):
        state = self.make_state()
        state.observe_pages(clean_pages=40, ready_pages=0, live_pages=60)

        offer = state.offer_donation(requested_pages=50, now_s=0.0)

        self.assertEqual(offer.immediate_pages, 4)
        self.assertEqual(offer.ready_pages, 0)
        self.assertEqual(offer.low_loss_pages, 0)

    def test_donor_reports_prepared_pages_as_ready_soon_not_immediate(self):
        state = self.make_state()
        state.observe_pages(clean_pages=20, ready_pages=20, live_pages=80)

        offer = state.offer_donation(requested_pages=50, now_s=0.0)

        self.assertEqual(offer.immediate_pages, 0)
        self.assertEqual(offer.ready_pages, 4)
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
        self.assertEqual(state.offer_donation(requested_pages=12, now_s=25.0).total_pages, 4)

    def test_recent_reclaim_loss_limits_proactive_recovery_to_low_watermark(self):
        state = self.make_state()
        # Start a normal recovery wave below low, then observe that deleting
        # a prefix caused a fast re-prefill.  With 24 clean pages, ordinary
        # recovery would keep deleting until high=36; loss-aware recovery must
        # stop because low=24 is already safe for normal admission.
        state.observe_pages(clean_pages=19, ready_pages=0, live_pages=81)
        self.assertEqual(state.action(now_s=0.0), LocalAction.PREPARE)
        state.record_retention_loss(reprefill_tokens=4096, now_s=1.0)
        state.observe_pages(clean_pages=24, ready_pages=0, live_pages=76)

        self.assertTrue(state.value_constrained(now_s=2.0))
        self.assertEqual(state.maintenance_target_pages(now_s=2.0), 24)
        self.assertEqual(state.action(now_s=2.0), LocalAction.IDLE)

    def test_safety_only_policy_keeps_high_recovery_target_after_reclaim_loss(self):
        state = LocalResidencyState(
            effective_pages=100,
            floor_pages=20,
            large_backup_batch_pages=8,
            backup_ingress_pages_per_s=4.0,
            ready_latency_p95_s=1.0,
            maintenance_batch_pages=12,
            maintenance_cycle_s=2.0,
            retention_debt_cooldown_s=10.0,
            loss_aware_maintenance=False,
        )
        state.record_retention_loss(reprefill_tokens=4096, now_s=1.0)

        self.assertFalse(state.value_constrained(now_s=2.0))
        self.assertEqual(state.maintenance_target_pages(now_s=2.0), 36)

    def test_status_refresh_expires_old_retention_debt_without_a_donor_request(self):
        state = self.make_state()
        state.record_retention_loss(reprefill_tokens=4096, now_s=0.0)

        self.assertEqual(state.active_retention_debt_tokens(now_s=1.0), 4096)
        self.assertEqual(state.feedback_horizon_s(now_s=10.0), 25.0)
        self.assertEqual(state.active_retention_debt_tokens(now_s=25.0), 0)

    def test_reclaim_feedback_horizon_tracks_host_turnover_not_client_rps(self):
        state = self.make_state()

        # 100 pages / 4 backup pages/s means one host turnover takes 25s.
        self.assertEqual(state.feedback_horizon_s(now_s=0.0), 25.0)
        state.observe_backup(pages=40, elapsed_s=0.001, now_s=0.0)
        state.observe_backup(pages=40, elapsed_s=0.001, now_s=1.0)

        # A sustained 40-page/s backup burst shortens the relevant feedback
        # window to 2.5s, bounded by the 5s minimum.
        self.assertEqual(state.feedback_horizon_s(now_s=1.0), 5.0)

    def test_unresolved_admission_shortfall_blocks_donation_without_waiting_for_miss(self):
        state = self.make_state()
        state.observe_pages(clean_pages=40, ready_pages=0, live_pages=60)
        state.record_clean_shortfall(pages=6, now_s=0.0)

        self.assertEqual(state.unresolved_clean_pages, 6)
        self.assertEqual(state.offer_donation(requested_pages=12, now_s=0.0).total_pages, 0)
        self.assertEqual(state.offer_donation(requested_pages=12, now_s=10.0).total_pages, 4)

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

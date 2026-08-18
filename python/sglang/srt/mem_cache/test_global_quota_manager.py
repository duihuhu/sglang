"""Policy-free Global Quota Manager ledger tests."""

from __future__ import annotations

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from sglang.srt.mem_cache.global_quota_manager import GlobalQuotaManager


class GlobalQuotaManagerTest(unittest.TestCase):
    def manager(self) -> GlobalQuotaManager:
        manager = GlobalQuotaManager(global_pool_pages=100, clock_ns=lambda: 1_000_000_000)
        manager.register_instance("hot", floor_pages=10, effective_pages=30)
        manager.register_instance("cold", floor_pages=10, effective_pages=20)
        return manager

    def test_grow_from_reserve_changes_effective_only_after_allocator_ack(self):
        manager = self.manager()

        intent = manager.submit_intent("grow", recipient="hot", pages=10)

        self.assertEqual(manager.effective_pages("hot"), 30)
        self.assertEqual(manager.reserve_pages, 40)
        self.assertEqual(manager.pending_ack_pages("hot"), 10)

        manager.ack_recipient_allocator(intent.intent_id)

        self.assertEqual(manager.effective_pages("hot"), 40)
        self.assertEqual(manager.reserve_pages, 40)
        self.assertEqual(manager.pending_ack_pages("hot"), 0)
        self.assertEqual(manager.audit_kinds(intent.intent_id), [
            "intent_received",
            "reserve_allocated",
            "allocator_ack",
            "effective_quota_changed",
        ])
        manager.assert_conserved()

    def test_transfer_requires_explicit_donor_and_records_full_causal_chain(self):
        manager = self.manager()

        intent = manager.submit_intent(
            "transfer",
            donor="cold",
            recipient="hot",
            pages=8,
            offer_id="operator-offer-7",
        )
        manager.mark_donor_detached(intent.intent_id)
        manager.mark_scrub_complete(intent.intent_id)

        self.assertEqual(manager.effective_pages("cold"), 12)
        self.assertEqual(manager.effective_pages("hot"), 30)
        self.assertEqual(manager.pending_ack_pages("hot"), 8)

        manager.ack_recipient_allocator(intent.intent_id)

        self.assertEqual(manager.effective_pages("hot"), 38)
        self.assertEqual(manager.audit_kinds(intent.intent_id), [
            "intent_received",
            "offer_accepted",
            "donor_detached",
            "scrub_completed",
            "recipient_allocator_install_ready",
            "allocator_ack",
            "effective_quota_changed",
        ])
        manager.assert_conserved()

    def test_shrink_respects_floor_and_returns_pages_to_reserve_after_scrub(self):
        manager = self.manager()

        with self.assertRaisesRegex(ValueError, "floor"):
            manager.submit_intent("shrink", donor="cold", pages=11)

        intent = manager.submit_intent("shrink", donor="cold", pages=10)
        manager.mark_donor_detached(intent.intent_id)
        self.assertEqual(manager.effective_pages("cold"), 10)
        self.assertEqual(manager.scrub_pending_pages("cold"), 10)

        manager.mark_scrub_complete(intent.intent_id)

        self.assertEqual(manager.reserve_pages, 60)
        self.assertEqual(manager.scrub_pending_pages("cold"), 0)
        self.assertEqual(manager.audit_kinds(intent.intent_id), [
            "intent_received",
            "donor_detached",
            "scrub_completed",
            "reserve_recredited",
        ])
        manager.assert_conserved()

    def test_policy_metrics_are_rejected_at_intent_boundary(self):
        manager = self.manager()

        for forbidden in ("D", "TTFT", "RPS", "cache_ratio", "profile"):
            with self.subTest(forbidden=forbidden):
                with self.assertRaisesRegex(ValueError, "policy-free"):
                    manager.submit_intent(
                        "grow",
                        recipient="hot",
                        pages=1,
                        policy_context={forbidden: 1},
                    )

    def test_first_actual_backup_on_new_page_records_t_usable_once(self):
        tick = {"now": 1_000_000_000}
        manager = GlobalQuotaManager(
            global_pool_pages=64,
            clock_ns=lambda: tick["now"],
        )
        manager.register_instance("hot", floor_pages=8, effective_pages=16)

        intent = manager.submit_intent("grow", recipient="hot", pages=4)
        tick["now"] += 250_000_000
        manager.ack_recipient_allocator(intent.intent_id)
        tick["now"] += 750_000_000

        first = manager.record_actual_backup("hot", pages=1)
        second = manager.record_actual_backup("hot", pages=1)

        self.assertEqual(first["intent_id"], intent.intent_id)
        self.assertEqual(first["t_usable_ns"], 1_000_000_000)
        self.assertIsNone(second)
        self.assertIn("first_actual_backup_on_new_page", manager.audit_kinds(intent.intent_id))
        manager.assert_conserved()

    def test_conflicting_intent_is_rejected_until_ack_or_completion(self):
        manager = self.manager()
        first = manager.submit_intent("grow", recipient="hot", pages=4)

        with self.assertRaisesRegex(RuntimeError, "in-flight"):
            manager.submit_intent("shrink", donor="hot", pages=1)

        manager.ack_recipient_allocator(first.intent_id)
        manager.submit_intent("shrink", donor="hot", pages=1)

    def test_audit_jsonl_is_append_only_and_records_timeout_without_losing_pages(self):
        with TemporaryDirectory() as tmp:
            audit_path = Path(tmp) / "quota_audit.jsonl"
            manager = GlobalQuotaManager(
                global_pool_pages=32,
                clock_ns=lambda: 1,
                audit_path=audit_path,
            )
            manager.register_instance("a", floor_pages=4, effective_pages=8)

            intent = manager.submit_intent("grow", recipient="a", pages=4)
            manager.mark_timeout(intent.intent_id, reason="allocator did not ack")

            records = audit_path.read_text().strip().splitlines()
            self.assertEqual(len(records), 3)
            self.assertIn('"kind":"intent_received"', records[0])
            self.assertIn('"kind":"reserve_allocated"', records[1])
            self.assertIn('"kind":"intent_timeout"', records[2])
            self.assertEqual(manager.effective_pages("a"), 8)
            self.assertEqual(manager.pending_ack_pages("a"), 4)
            manager.assert_conserved()


if __name__ == "__main__":
    unittest.main()

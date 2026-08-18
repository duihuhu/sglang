"""Policy-free manual quota intent runner tests."""

from __future__ import annotations

import json
from pathlib import Path
import sys
import unittest
from tempfile import TemporaryDirectory

sys.path.insert(0, str(Path(__file__).resolve().parent))

from manual_quota_intent_runner import ManualQuotaIntentRunner
from manual_quota_intent_runner import PressureLogBackupObserver


class _Backend:
    def __init__(self):
        self.models = {
            "hot": {"effective_pages": 30, "floor_pages": 10},
            "cold": {"effective_pages": 40, "floor_pages": 10},
        }
        self.global_pool_pages = 100
        self.calls = []

    def snapshot(self):
        return {
            "global_pool_pages": self.global_pool_pages,
            "models": {
                model_id: dict(state) for model_id, state in self.models.items()
            },
        }

    def grow_from_reserve(self, model_id: str, pages: int):
        self.calls.append(("grow_from_reserve", model_id, pages))
        self.models[model_id]["effective_pages"] += pages
        return {"effective_pages": self.models[model_id]["effective_pages"]}

    def shrink_to_reserve(self, model_id: str, pages: int):
        self.calls.append(("shrink_to_reserve", model_id, pages))
        self.models[model_id]["effective_pages"] -= pages
        return {"effective_pages": self.models[model_id]["effective_pages"]}

    def transfer_pages(self, donor: str, recipient: str, pages: int, offer_id: str):
        self.calls.append(("transfer_pages", donor, recipient, pages, offer_id))
        self.models[donor]["effective_pages"] -= pages
        self.models[recipient]["effective_pages"] += pages
        return {
            "donor_effective_pages": self.models[donor]["effective_pages"],
            "recipient_effective_pages": self.models[recipient]["effective_pages"],
        }

    def donation_offer(self, model_id: str):
        return {"model_id": model_id, "total_pages": 0}


class _AsyncBackend(_Backend):
    def __init__(self):
        super().__init__()
        self.boundaries = []

    def begin_grow_from_reserve(self, model_id: str, pages: int):
        current = self.models[model_id]["effective_pages"]
        target = current + pages
        self.calls.append(("begin_grow_from_reserve", model_id, pages, target))
        self.models[model_id]["target_effective_pages"] = target
        return {
            "model_id": model_id,
            "start_effective_pages": current,
            "target_effective_pages": target,
        }

    def begin_transfer_pages(
        self, donor: str, recipient: str, pages: int, offer_id: str
    ):
        donor_current = self.models[donor]["effective_pages"]
        recipient_current = self.models[recipient]["effective_pages"]
        donor_target = donor_current - pages
        recipient_target = recipient_current + pages
        self.calls.append(
            (
                "begin_transfer_pages",
                donor,
                recipient,
                pages,
                offer_id,
                donor_target,
                recipient_target,
            )
        )
        self.models[donor]["target_effective_pages"] = donor_target
        self.models[recipient]["target_effective_pages"] = recipient_target
        return {
            "donor_start_effective_pages": donor_current,
            "recipient_start_effective_pages": recipient_current,
            "donor_target_effective_pages": donor_target,
            "recipient_target_effective_pages": recipient_target,
        }

    def wait_grow_boundary(self, model_id: str, target_effective_pages: int):
        self.calls.append(("wait_grow_boundary", model_id, target_effective_pages))
        self.models[model_id]["effective_pages"] = target_effective_pages
        self.boundaries.append((model_id, target_effective_pages))
        return {"effective_pages": target_effective_pages}

    def wait_transfer_boundary(
        self,
        donor: str,
        donor_effective_pages: int,
        recipient: str,
        recipient_effective_pages: int,
    ):
        self.calls.append(
            (
                "wait_transfer_boundary",
                donor,
                donor_effective_pages,
                recipient,
                recipient_effective_pages,
            )
        )
        self.models[donor]["effective_pages"] = donor_effective_pages
        self.models[recipient]["effective_pages"] = recipient_effective_pages
        self.boundaries.append((donor, donor_effective_pages, recipient, recipient_effective_pages))
        return {
            "donor_effective_pages": donor_effective_pages,
            "recipient_effective_pages": recipient_effective_pages,
        }


class _OfferBackend(_AsyncBackend):
    def donation_offer(self, model_id: str):
        return {"model_id": model_id, "total_pages": 7}


class _BackupObserver:
    def __init__(self):
        self.calls = []

    def first_actual_backup(self, model_id: str, pages: int):
        self.calls.append((model_id, pages))
        return {"pages": pages}


class _FailingBackupObserver:
    def first_actual_backup(self, model_id: str, pages: int):
        raise AssertionError("backup observation must not block tranche dispatch")


class ManualQuotaIntentRunnerTest(unittest.TestCase):
    def test_jsonl_grow_runs_in_tranches_and_records_first_usable_backup(self):
        backend = _Backend()
        observer = _BackupObserver()

        with TemporaryDirectory() as tmp:
            intent_path = Path(tmp) / "intents.jsonl"
            result_path = Path(tmp) / "results.jsonl"
            audit_path = Path(tmp) / "audit.jsonl"
            intent_path.write_text(
                json.dumps(
                    {
                        "intent_id": "manual-grow-1",
                        "op": "grow",
                        "recipient": "hot",
                        "pages": 12,
                        "tranche_pages": 5,
                    }
                )
                + "\n"
            )
            runner = ManualQuotaIntentRunner(
                backend,
                backup_observer=observer,
                audit_path=audit_path,
                clock_ns=lambda: 1_000,
            )

            records = runner.run_jsonl(intent_path, result_path=result_path)

            self.assertEqual(
                backend.calls,
                [
                    ("grow_from_reserve", "hot", 5),
                    ("grow_from_reserve", "hot", 5),
                    ("grow_from_reserve", "hot", 2),
                ],
            )
            self.assertEqual(backend.models["hot"]["effective_pages"], 42)
            self.assertEqual([item["pages"] for item in records], [5, 5, 2])
            self.assertEqual([item["external_intent_id"] for item in records], ["manual-grow-1"] * 3)
            self.assertEqual(observer.calls, [("hot", 5), ("hot", 5), ("hot", 2)])
            self.assertIn('"kind":"first_actual_backup_on_new_page"', audit_path.read_text())
            self.assertEqual(len(result_path.read_text().strip().splitlines()), 3)

    def test_deferred_backup_observation_dispatches_all_tranches_without_waiting(self):
        backend = _Backend()

        with TemporaryDirectory() as tmp:
            intent_path = Path(tmp) / "intents.jsonl"
            intent_path.write_text(
                json.dumps(
                    {
                        "intent_id": "manual-grow-fast",
                        "op": "grow",
                        "recipient": "hot",
                        "pages": 11,
                        "tranche_pages": 5,
                    }
                )
                + "\n"
            )
            runner = ManualQuotaIntentRunner(
                backend,
                backup_observer=_FailingBackupObserver(),
                wait_for_backup_observation=False,
                clock_ns=lambda: 10,
            )

            records = runner.run_jsonl(intent_path)

            self.assertEqual(
                backend.calls,
                [
                    ("grow_from_reserve", "hot", 5),
                    ("grow_from_reserve", "hot", 5),
                    ("grow_from_reserve", "hot", 1),
                ],
            )
            self.assertEqual([item["first_actual_backup"] for item in records], [None, None, None])
            self.assertEqual([item["t_ack_ns"] for item in records], [0, 0, 0])

    def test_async_overlap_grow_publishes_one_final_target_and_acks_boundaries(self):
        backend = _AsyncBackend()
        ticks = iter(range(100, 10_000, 10))

        with TemporaryDirectory() as tmp:
            intent_path = Path(tmp) / "intents.jsonl"
            intent_path.write_text(
                json.dumps(
                    {
                        "intent_id": "manual-grow-overlap",
                        "op": "grow",
                        "recipient": "hot",
                        "pages": 12,
                        "initial_tranche_pages": 7,
                        "tranche_pages": 5,
                        "async_overlap": True,
                    }
                )
                + "\n"
            )
            runner = ManualQuotaIntentRunner(
                backend,
                wait_for_backup_observation=False,
                clock_ns=lambda: next(ticks),
            )

            records = runner.run_jsonl(intent_path)

            self.assertEqual(
                backend.calls,
                [
                    ("begin_grow_from_reserve", "hot", 12, 42),
                    ("wait_grow_boundary", "hot", 37),
                    ("wait_grow_boundary", "hot", 42),
                ],
            )
            self.assertEqual([item["pages"] for item in records], [7, 5])
            self.assertEqual([item["tranche_target_effective_pages"] for item in records], [37, 42])
            self.assertTrue(records[0]["initial_tranche"])
            self.assertEqual(records[0]["async_plan"]["target_effective_pages"], 42)
            self.assertEqual(records[-1]["async_intent_done"], True)
            self.assertLess(records[0]["t_ack_ns"], records[-1]["t_ack_ns"])

    def test_async_overlap_transfer_publishes_one_atomic_final_target(self):
        backend = _AsyncBackend()

        with TemporaryDirectory() as tmp:
            intent_path = Path(tmp) / "intents.jsonl"
            intent_path.write_text(
                json.dumps(
                    {
                        "intent_id": "manual-transfer-overlap",
                        "op": "transfer",
                        "donor": "cold",
                        "recipient": "hot",
                        "pages": 13,
                        "initial_tranche_pages": 6,
                        "tranche_pages": 4,
                        "offer_id": "operator-safe-offer-13",
                        "async_overlap": True,
                    }
                )
                + "\n"
            )
            runner = ManualQuotaIntentRunner(
                backend,
                wait_for_backup_observation=False,
                clock_ns=lambda: 1_000,
            )

            records = runner.run_jsonl(intent_path)

            self.assertEqual(
                backend.calls,
                [
                    (
                        "begin_transfer_pages",
                        "cold",
                        "hot",
                        13,
                        "operator-safe-offer-13",
                        27,
                        43,
                    ),
                    ("wait_transfer_boundary", "cold", 34, "hot", 36),
                    ("wait_transfer_boundary", "cold", 30, "hot", 40),
                    ("wait_transfer_boundary", "cold", 27, "hot", 43),
                ],
            )
            self.assertEqual([item["pages"] for item in records], [6, 4, 3])
            self.assertEqual([item["tranche_target_effective_pages"] for item in records], [36, 40, 43])
            self.assertEqual([item["donor_target_effective_pages"] for item in records], [34, 30, 27])
            self.assertEqual(records[-1]["async_intent_done"], True)

    def test_required_donor_offer_fails_before_publishing_transfer_target(self):
        backend = _OfferBackend()
        runner = ManualQuotaIntentRunner(
            backend,
            wait_for_backup_observation=False,
        )

        with self.assertRaisesRegex(ValueError, "donor offer below explicit requirement"):
            runner.apply_external_intent(
                {
                    "intent_id": "manual-transfer-offer-gate",
                    "op": "transfer",
                    "donor": "cold",
                    "recipient": "hot",
                    "pages": 9,
                    "tranche_pages": 9,
                    "offer_id": "operator-safe-offer",
                    "async_overlap": True,
                    "require_donor_offer_pages": 9,
                }
            )

        self.assertNotIn("begin_transfer_pages", [call[0] for call in backend.calls])

    def test_transfer_requires_offer_and_does_not_read_policy_context(self):
        backend = _Backend()
        with TemporaryDirectory() as tmp:
            intent_path = Path(tmp) / "intents.jsonl"
            result_path = Path(tmp) / "results.jsonl"
            intent_path.write_text(
                json.dumps(
                    {
                        "intent_id": "manual-transfer-1",
                        "op": "transfer",
                        "donor": "cold",
                        "recipient": "hot",
                        "pages": 9,
                        "tranche_pages": 4,
                        "offer_id": "operator-safe-offer-9",
                    }
                )
                + "\n"
            )

            runner = ManualQuotaIntentRunner(backend, clock_ns=lambda: 2_000)
            records = runner.run_jsonl(intent_path, result_path=result_path)

            self.assertEqual(
                backend.calls,
                [
                    ("transfer_pages", "cold", "hot", 4, "operator-safe-offer-9"),
                    ("transfer_pages", "cold", "hot", 4, "operator-safe-offer-9"),
                    ("transfer_pages", "cold", "hot", 1, "operator-safe-offer-9"),
                ],
            )
            self.assertEqual(backend.models["cold"]["effective_pages"], 31)
            self.assertEqual(backend.models["hot"]["effective_pages"], 39)
            self.assertEqual([item["manager_action"] for item in records], ["transfer"] * 3)

    def test_live_transaction_window_metadata_is_recorded_for_policy_iteration(self):
        backend = _Backend()
        runner = ManualQuotaIntentRunner(
            backend,
            wait_for_backup_observation=False,
            clock_ns=lambda: 4_000,
        )

        records = runner.apply_external_intent(
            {
                "intent_id": "window-2-grow",
                "op": "grow",
                "recipient": "hot",
                "pages": 3,
                "phase": "live_transaction_window",
                "decision_window_id": "adaptive-window-2",
            },
            line_number=7,
        )

        self.assertEqual(len(records), 1)
        self.assertEqual(records[0]["phase"], "live_transaction_window")
        self.assertEqual(records[0]["decision_window_id"], "adaptive-window-2")
        self.assertEqual(records[0]["line_number"], 7)

    def test_startup_correction_is_not_a_live_quota_intent(self):
        backend = _Backend()
        runner = ManualQuotaIntentRunner(
            backend,
            wait_for_backup_observation=False,
        )

        with self.assertRaisesRegex(ValueError, "startup correction"):
            runner.apply_external_intent(
                {
                    "intent_id": "bad-startup-correction",
                    "op": "grow",
                    "recipient": "hot",
                    "pages": 1,
                    "phase": "startup_correction",
                }
            )

        self.assertEqual(backend.calls, [])

    def test_policy_fields_are_rejected_before_backend_calls(self):
        backend = _Backend()
        with TemporaryDirectory() as tmp:
            intent_path = Path(tmp) / "intents.jsonl"
            intent_path.write_text(
                json.dumps(
                    {
                        "intent_id": "bad-policy",
                        "op": "grow",
                        "recipient": "hot",
                        "pages": 1,
                        "D": 12,
                    }
                )
                + "\n"
            )
            runner = ManualQuotaIntentRunner(backend, clock_ns=lambda: 3_000)

            with self.assertRaisesRegex(ValueError, "policy-free"):
                runner.run_jsonl(intent_path)

            self.assertEqual(backend.calls, [])

    def test_pressure_log_observer_ignores_old_backup_and_returns_new_backup(self):
        with TemporaryDirectory() as tmp:
            log = Path(tmp) / "pressure.jsonl"
            log.write_text('{"backup_slots":8,"page_size_tokens":1}\n')

            observer = PressureLogBackupObserver(
                {"hot": log},
                wait_timeout_s=0.01,
                wait_interval_s=0,
            )
            log.write_text(
                log.read_text()
                + '{"backup_slots":0,"page_size_tokens":1}\n'
                + '{"backup_slots":3,"page_size_tokens":1}\n'
            )

            self.assertEqual(
                observer.first_actual_backup("hot", pages=2),
                {"backup_slots": 3, "backup_pages": 3},
            )


if __name__ == "__main__":
    unittest.main()

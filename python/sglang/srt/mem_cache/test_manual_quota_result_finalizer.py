from __future__ import annotations

import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from manual_quota_result_finalizer import finalize_manual_results


class ManualQuotaResultFinalizerTest(unittest.TestCase):
    def test_backfills_first_backup_and_t_usable_after_effective_quota(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            result = root / "manual_results.jsonl"
            audit = root / "manual_audit.jsonl"
            pressure = root / "pressure.jsonl"
            output = root / "finalized.jsonl"
            result.write_text(
                json.dumps(
                    {
                        "manager_intent_id": 7,
                        "external_intent_id": "w-1",
                        "first_actual_backup": None,
                    }
                )
                + "\n"
            )
            audit.write_text(
                "\n".join(
                    json.dumps(row)
                    for row in [
                        {
                            "intent_id": 7,
                            "kind": "intent_received",
                            "time_ns": 1_000_000_000,
                            "pages": 4,
                        },
                        {
                            "intent_id": 7,
                            "kind": "allocator_ack",
                            "model_id": "hot",
                            "time_ns": 1_100_000_000,
                            "pages": 4,
                        },
                        {
                            "intent_id": 7,
                            "kind": "effective_quota_changed",
                            "model_id": "hot",
                            "time_ns": 1_120_000_000,
                            "pages": 104,
                        },
                    ]
                )
                + "\n"
            )
            pressure.write_text(
                "\n".join(
                    json.dumps(row)
                    for row in [
                        {
                            "monotonic_time_s": 1.11,
                            "active_quota_pages": 104,
                            "backup_slots": 9,
                            "page_size_tokens": 1,
                        },
                        {
                            "monotonic_time_s": 1.25,
                            "active_quota_pages": 104,
                            "backup_slots": 3,
                            "page_size_tokens": 1,
                            "alloc_pages": 3,
                            "live_pages": 33,
                        },
                    ]
                )
                + "\n"
            )

            rows = finalize_manual_results(
                result_jsonl=result,
                audit_jsonl=audit,
                pressure_logs={"hot": pressure},
                output_jsonl=output,
            )

            self.assertEqual(len(rows), 1)
            self.assertAlmostEqual(rows[0]["intent_to_ack_s"], 0.1)
            self.assertAlmostEqual(rows[0]["t_usable_s"], 0.25)
            self.assertAlmostEqual(
                rows[0]["first_actual_backup"]["effective_to_first_backup_s"],
                0.13,
            )
            self.assertEqual(
                rows[0]["first_actual_backup"]["observer"]["backup_slots"], 3
            )
            self.assertIn('"t_usable_s":0.25', output.read_text())


if __name__ == "__main__":
    unittest.main()

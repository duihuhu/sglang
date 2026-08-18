from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

from build_d_window_observations import build_window


def write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")


class BuildDWindowObservationsTest(unittest.TestCase):
    def test_window_uses_retention_loss_as_d_and_revisit_rows_as_opportunity(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            a_client = root / "a_client.jsonl"
            a_pressure = root / "a_pressure.jsonl"
            b_client = root / "b_client.jsonl"
            b_pressure = root / "b_pressure.jsonl"
            write_jsonl(
                a_client,
                [
                    {"main_request_id": 0, "sub_request_id": 0, "scheduled_time_s": 5, "wall_dispatch_time_s": 105, "prompt_tokens": 100, "cached_tokens": 0, "ttft_s": 0.1, "success": True},
                    {"main_request_id": 0, "sub_request_id": 1, "scheduled_time_s": 12, "wall_dispatch_time_s": 112, "prompt_tokens": 120, "cached_tokens": 30, "ttft_s": 0.3, "success": True},
                    {"main_request_id": 1, "sub_request_id": 0, "scheduled_time_s": 8, "wall_dispatch_time_s": 108, "prompt_tokens": 250, "cached_tokens": 0, "ttft_s": 0.1, "success": True},
                    {"main_request_id": 1, "sub_request_id": 1, "scheduled_time_s": 18, "wall_dispatch_time_s": 118, "prompt_tokens": 300, "cached_tokens": 120, "ttft_s": 0.5, "success": True},
                    {"main_request_id": 2, "sub_request_id": 1, "scheduled_time_s": 40, "wall_dispatch_time_s": 140, "prompt_tokens": 900, "cached_tokens": 10, "ttft_s": 2.0, "success": True},
                ],
            )
            write_jsonl(
                a_pressure,
                [
                    {"retention_loss_tokens": 40, "admission_shortfall_pages": 0, "host_evict_slots": 2, "wall_time_s": 112},
                    {"retention_loss_tokens": 60, "admission_shortfall_pages": 3, "host_evict_slots": 0, "wall_time_s": 118},
                    {"retention_loss_tokens": 700, "admission_shortfall_pages": 9, "host_evict_slots": 11, "wall_time_s": 140},
                ],
            )
            write_jsonl(
                b_client,
                [
                    {"main_request_id": 0, "sub_request_id": 0, "scheduled_time_s": 10, "prompt_tokens": 100, "cached_tokens": 0, "ttft_s": 0.1, "success": True},
                ],
            )
            write_jsonl(b_pressure, [{"retention_loss_tokens": 0, "admission_shortfall_pages": 0}])

            window = build_window(
                decision_window_id="w1",
                window_start_s=10,
                window_end_s=20,
                models={
                    "A": {"client_jsonl": a_client, "pressure_jsonl": a_pressure},
                    "B": {"client_jsonl": b_client, "pressure_jsonl": b_pressure},
                },
            )

            self.assertEqual(window["decision_window_id"], "w1")
            self.assertEqual(window["reports"]["A"]["confirmed_reuse_loss_tokens"], 100)
            self.assertEqual(window["reports"]["A"]["raw_retention_loss_tokens"], 100)
            self.assertEqual(window["reports"]["A"]["revisit_opportunity_tokens"], 420)
            self.assertEqual(window["reports"]["A"]["revisit_cache_match_tokens"], 150)
            self.assertEqual(window["reports"]["A"]["revisit_cache_miss_tokens"], 270)
            self.assertEqual(window["reports"]["A"]["revisit_incremental_tokens"], 70)
            self.assertEqual(window["reports"]["A"]["revisit_seen_cache_miss_tokens"], 200)
            self.assertEqual(window["reports"]["A"]["d_coverage_of_seen_miss"], 0.5)
            self.assertEqual(window["reports"]["A"]["reprefill_tokens"], 100)
            self.assertEqual(window["reports"]["A"]["admission_shortfall_pages"], 3)
            self.assertEqual(window["reports"]["A"]["host_evicted_tokens"], 2)
            self.assertEqual(window["reports"]["A"]["conditional_ttft_ms_p50"], 300.0)
            self.assertEqual(window["reports"]["B"]["confirmed_reuse_loss_tokens"], 0)
            self.assertEqual(window["reports"]["B"]["revisit_opportunity_tokens"], 0)

    def test_window_confirmed_loss_is_capped_to_seen_revisit_miss(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            client = root / "client.jsonl"
            pressure = root / "pressure.jsonl"
            write_jsonl(
                client,
                [
                    {"main_request_id": 0, "sub_request_id": 0, "scheduled_time_s": 0, "wall_dispatch_time_s": 100, "prompt_tokens": 100, "cached_tokens": 0, "ttft_s": 0.1, "success": True},
                    {"main_request_id": 0, "sub_request_id": 1, "scheduled_time_s": 10, "wall_dispatch_time_s": 110, "prompt_tokens": 120, "cached_tokens": 70, "ttft_s": 0.2, "success": True},
                ],
            )
            write_jsonl(
                pressure,
                [
                    {"retention_loss_tokens": 200, "admission_shortfall_pages": 0, "host_evict_slots": 0, "wall_time_s": 110},
                ],
            )

            window = build_window(
                decision_window_id="w1",
                window_start_s=0,
                window_end_s=20,
                models={"A": {"client_jsonl": client, "pressure_jsonl": pressure}},
            )

            report = window["reports"]["A"]
            self.assertEqual(report["raw_retention_loss_tokens"], 200)
            self.assertEqual(report["revisit_seen_cache_miss_tokens"], 30)
            self.assertEqual(report["confirmed_reuse_loss_tokens"], 30)
            self.assertEqual(report["d_coverage_of_seen_miss"], 1.0)


if __name__ == "__main__":
    unittest.main()

"""Control-plane integration checks for the pressure scheduler tick."""

from __future__ import annotations

import unittest

from sglang.srt.mem_cache.pressure_scheduler_runtime import PressureSchedulerRuntime


class _Control:
    def __init__(self, snapshot):
        self.snapshot = snapshot
        self.published = []

    def global_status(self):
        return self.snapshot

    def set_quota_targets(self, targets):
        self.published.append(targets)
        return {"targets": targets}


def model(capacity, *, clean, ready, high, debt=0, shortfall=0):
    return {
        "capacity": capacity,
        "pending_capacity": 0,
        "page_size": 10,
        "token_bytes": 1,
        "residency": {
            "effective_pages": capacity // 10,
            "floor_pages": 100,
            "clean_pages": clean // 10,
            "ready_pages": ready // 10,
            "live_pages": (capacity - clean) // 10,
            "min_pages": 20,
            "low_pages": 40,
            "high_pages": high // 10,
            "retention_debt_tokens": debt,
            "unresolved_clean_pages": shortfall,
            "action": "prepare",
        },
    }


class PressureSchedulerRuntimeTest(unittest.TestCase):
    def test_tick_publishes_page_aligned_intents_without_claiming_effective_growth(self):
        control = _Control(
            {
                "global_free_bytes": 0,
                "models": {
                    "hot": model(2_000, clean=100, ready=0, high=600, debt=800),
                    "cold": model(2_000, clean=1_100, ready=500, high=400),
                },
            }
        )
        runtime = PressureSchedulerRuntime(control, max_transfer_bytes=500)

        result = runtime.tick()

        self.assertEqual(control.published, [{"hot": 2_500, "cold": 1_500}])
        self.assertEqual(result["published_targets"], {"hot": 2_500, "cold": 1_500})
        self.assertEqual(result["effective_targets"], {"hot": 2_000, "cold": 2_000})

    def test_tick_skips_models_without_a_local_health_report(self):
        control = _Control(
            {
                "global_free_bytes": 0,
                "models": {
                    "hot": {
                        "capacity": 2_000,
                        "pending_capacity": 0,
                        "page_size": 10,
                        "token_bytes": 1,
                        "residency": None,
                    }
                },
            }
        )
        runtime = PressureSchedulerRuntime(control, max_transfer_bytes=500)

        result = runtime.tick()

        self.assertEqual(control.published, [])
        self.assertEqual(result["published_targets"], {})
        self.assertEqual(result["missing_reports"], ["hot"])

    def test_tick_revokes_an_unfinished_donor_shrink_after_retention_debt(self):
        control = _Control(
            {
                "global_free_bytes": 0,
                "models": {
                    "cold": {
                        **model(2_000, clean=1_100, ready=0, high=400, debt=1),
                        "quota_target": 1_500,
                    }
                },
            }
        )
        runtime = PressureSchedulerRuntime(control, max_transfer_bytes=500)

        result = runtime.tick()

        self.assertEqual(control.published, [{"cold": 2_000}])
        self.assertEqual(result["published_targets"], {"cold": 2_000})


if __name__ == "__main__":
    unittest.main()

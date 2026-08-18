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
        return {"targets": targets, "target_set_ns": 123456}


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
            "retention_debt_epoch": 1 if debt else 0,
            "unresolved_clean_pages": shortfall,
            "feedback_horizon_s": 20.0,
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
        self.assertEqual(result["local_health"]["hot"]["retention_debt_tokens"], 800)
        self.assertEqual(result["local_health"]["cold"]["ready_pages"], 50)
        self.assertEqual(result["quota_transitions"]["hot"]["effective_slots"], 2_000)
        self.assertEqual(result["published_target_set_ns"], 123456)

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

    def test_tick_waits_for_a_donor_local_remove_before_recomputing_intents(self):
        control = _Control(
            {
                "global_free_bytes": 0,
                "models": {
                    "cold": {
                        **model(2_000, clean=1_100, ready=0, high=400),
                        "pending_release_capacity": 500,
                    }
                },
            }
        )
        runtime = PressureSchedulerRuntime(control, max_transfer_bytes=500)

        result = runtime.tick()

        self.assertEqual(result["pending_models"], ["cold"])
        self.assertEqual(result["local_health"], {})
        self.assertEqual(control.published, [])

    def test_tick_waits_for_a_recipient_allocator_ack_before_recomputing(self):
        control = _Control(
            {
                "global_free_bytes": 0,
                "models": {
                    "hot": {
                        **model(2_000, clean=100, ready=0, high=600, debt=800),
                        "quota_target": 2_500,
                    },
                    "cold": model(2_000, clean=1_100, ready=0, high=400),
                },
            }
        )
        runtime = PressureSchedulerRuntime(control, max_transfer_bytes=500)

        result = runtime.tick()

        # Repeated decisions must not progressively drain cold while the hot
        # allocator is still consuming the previously published target.
        self.assertEqual(result["pending_models"], ["hot"])
        self.assertEqual(control.published, [])

    def test_tick_uses_configured_growth_quantum_for_retention_debt(self):
        control = _Control(
            {
                "global_free_bytes": 0,
                "models": {
                    "hot": model(2_000, clean=300, ready=0, high=450, debt=1),
                    "cold": model(2_000, clean=1_100, ready=0, high=400),
                },
            }
        )
        runtime = PressureSchedulerRuntime(
            control,
            max_transfer_bytes=500,
            growth_quantum_bytes=200,
        )

        result = runtime.tick()

        # Retention debt becomes a growth signal only after clean capacity has
        # dropped below low. The configured quantum is larger than the 150B
        # local gap, so it determines this bounded trial.
        self.assertEqual(control.published, [{"hot": 2_200, "cold": 1_800}])
        self.assertEqual(result["published_targets"], {"hot": 2_200, "cold": 1_800})

    def test_one_retention_loss_epoch_cannot_repeat_growth_after_ack(self):
        snapshot = {
            "global_free_bytes": 0,
            "models": {
                "hot": model(2_000, clean=100, ready=0, high=600, debt=800),
                "cold": model(2_000, clean=1_100, ready=0, high=400),
            },
        }
        control = _Control(snapshot)
        runtime = PressureSchedulerRuntime(control, max_transfer_bytes=500)

        runtime.tick()
        self.assertEqual(control.published, [{"hot": 2_500, "cold": 1_500}])

        # The first transfer has become effective, but there has been no new
        # re-prefill loss; the same epoch must not drain cold again.
        snapshot["models"]["hot"]["capacity"] = 2_500
        snapshot["models"]["hot"]["residency"]["effective_pages"] = 250
        snapshot["models"]["cold"]["capacity"] = 1_500
        snapshot["models"]["cold"]["residency"]["effective_pages"] = 150
        runtime.tick()
        self.assertEqual(control.published, [{"hot": 2_500, "cold": 1_500}])

    def test_new_retention_epoch_allows_one_more_floor_bounded_transfer_after_ack(self):
        """Repeated pressure may transfer again, but only after a new loss epoch."""
        snapshot = {
            "global_free_bytes": 0,
            "models": {
                "hot": model(2_000, clean=100, ready=0, high=600, debt=800),
                "cold": model(2_000, clean=1_100, ready=0, high=400),
            },
        }
        control = _Control(snapshot)
        runtime = PressureSchedulerRuntime(control, max_transfer_bytes=500)

        runtime.tick()
        self.assertEqual(control.published, [{"hot": 2_500, "cold": 1_500}])

        # First handoff is effective. A fresh reclamation-loss epoch then
        # makes one further bounded request legal; cold must not go below its
        # 1,000-byte floor.
        snapshot["models"]["hot"] = model(
            2_500, clean=100, ready=0, high=600, debt=900
        )
        snapshot["models"]["hot"]["residency"]["retention_debt_epoch"] = 2
        snapshot["models"]["cold"] = model(1_500, clean=900, ready=0, high=400)
        runtime.tick()

        self.assertEqual(
            control.published,
            [{"hot": 2_500, "cold": 1_500}, {"hot": 3_000, "cold": 1_000}],
        )


if __name__ == "__main__":
    unittest.main()

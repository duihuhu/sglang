"""Control-plane adapter checks for D-driven shadow/apply planning."""

from __future__ import annotations

from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parent))

from d_marginal_trial_scheduler import InitialDReallocationPolicy
from d_marginal_trial_runtime import DMarginalTrialRuntime


class _Control:
    def __init__(self, snapshot):
        self.snapshot = snapshot
        self.published = []

    def global_status(self):
        return self.snapshot

    def set_quota_targets(self, targets):
        self.published.append(dict(targets))
        return {"targets": targets, "target_set_ns": 123456789}


def model(
    capacity,
    *,
    floor=500,
    clean=800,
    ready=0,
    high=600,
    shortfall=0,
    pending_capacity=0,
    pending_release_capacity=0,
    quota_target=None,
):
    return {
        "capacity": capacity,
        "pending_capacity": pending_capacity,
        "pending_release_capacity": pending_release_capacity,
        "quota_target": quota_target,
        "page_size": 1,
        "token_bytes": 10,
        "residency": {
            "effective_pages": capacity,
            "floor_pages": floor,
            "clean_pages": clean,
            "ready_pages": ready,
            "live_pages": capacity - clean,
            "high_pages": high,
            "unresolved_clean_pages": shortfall,
            "retention_debt_tokens": 999,
        },
    }


class DMarginalTrialRuntimeTest(unittest.TestCase):
    def test_shadow_replay_outputs_candidates_plan_and_outcome_schema_without_publish(self):
        control = _Control(
            {
                "global_free_bytes": 0,
                "models": {
                    "hot": model(1_000, clean=200, high=600),
                    "cold": model(1_000, clean=900, ready=100, high=600),
                },
            }
        )
        runtime = DMarginalTrialRuntime(
            control,
            trial_quantum_pages=100,
            min_revisit_opportunity_tokens=1_000,
            min_confirmed_loss_tokens=100,
            min_severity=0.05,
            min_volume_loss_tokens_per_s=10.0,
        )

        result = runtime.tick(
            {
                "hot": {
                    "window_start_s": 0.0,
                    "window_end_s": 5.0,
                    "confirmed_reuse_loss_tokens": 500,
                    "revisit_opportunity_tokens": 2_000,
                    "reprefill_tokens": 900,
                },
                "cold": {
                    "window_start_s": 0.0,
                    "window_end_s": 5.0,
                    "confirmed_reuse_loss_tokens": 0,
                    "revisit_opportunity_tokens": 10_000,
                },
            },
            mode="shadow",
            trial_id="shadow-1",
        )

        self.assertEqual(control.published, [])
        self.assertEqual(result["mode"], "shadow")
        self.assertEqual(result["published_targets"], {})
        self.assertEqual(result["plan"]["targets"], {"hot": 1_100, "cold": 900})
        self.assertEqual(result["plan"]["transactions"][0]["source_model_id"], "cold")
        self.assertEqual(result["signals"]["hot"]["severity"], 0.25)
        self.assertEqual(result["signals"]["hot"]["volume_loss_tokens_per_s"], 100.0)
        self.assertEqual(result["outcome_record"]["trial_id"], "shadow-1")
        self.assertIsNone(result["outcome_record"]["intent_id"])
        self.assertIsNone(result["outcome_record"]["effective_after_ack"])

    def test_apply_mode_publishes_plan_but_does_not_record_effective_before_ack(self):
        control = _Control(
            {
                "global_free_bytes": 0,
                "models": {
                    "hot": model(1_000, clean=200, high=600),
                    "cold": model(1_000, clean=900, high=600),
                },
            }
        )
        runtime = DMarginalTrialRuntime(
            control,
            trial_quantum_pages=100,
            min_revisit_opportunity_tokens=1_000,
            min_confirmed_loss_tokens=100,
            min_severity=0.05,
            min_volume_loss_tokens_per_s=10.0,
            clock_ns=lambda: 10,
        )

        result = runtime.tick(
            {
                "hot": {
                    "window_start_s": 0.0,
                    "window_end_s": 5.0,
                    "confirmed_reuse_loss_tokens": 500,
                    "revisit_opportunity_tokens": 2_000,
                },
                "cold": {
                    "window_start_s": 0.0,
                    "window_end_s": 5.0,
                    "confirmed_reuse_loss_tokens": 0,
                    "revisit_opportunity_tokens": 10_000,
                },
            },
            mode="apply",
            trial_id="apply-1",
        )

        self.assertEqual(control.published, [{"hot": 1_100, "cold": 900}])
        self.assertEqual(result["published_targets"], {"hot": 1_100, "cold": 900})
        self.assertEqual(result["outcome_record"]["intent_id"], "target_set:123456789")
        self.assertIsNone(result["outcome_record"]["effective_after_ack"])

        ack_record = runtime.record_allocator_ack(
            "apply-1",
            effective_pages={"hot": 1_100, "cold": 900},
            timestamp_ns=30,
        )
        self.assertEqual(ack_record["effective_after_ack"], {"hot": 1_100, "cold": 900})

    def test_tick_skips_missing_observation_and_pending_models(self):
        control = _Control(
            {
                "global_free_bytes": 0,
                "models": {
                    "hot": model(1_000, clean=200, high=600),
                    "cold": model(1_000, clean=900, high=600),
                    "pending": model(1_000, clean=900, high=600, pending_capacity=10),
                },
            }
        )
        runtime = DMarginalTrialRuntime(
            control,
            trial_quantum_pages=100,
            min_revisit_opportunity_tokens=1_000,
            min_confirmed_loss_tokens=100,
            min_severity=0.05,
            min_volume_loss_tokens_per_s=10.0,
        )

        result = runtime.tick(
            {
                "hot": {
                    "window_start_s": 0.0,
                    "window_end_s": 5.0,
                    "confirmed_reuse_loss_tokens": 500,
                    "revisit_opportunity_tokens": 2_000,
                },
                "pending": {
                    "window_start_s": 0.0,
                    "window_end_s": 5.0,
                    "confirmed_reuse_loss_tokens": 0,
                    "revisit_opportunity_tokens": 10_000,
                },
            },
            mode="shadow",
            trial_id="shadow-2",
        )

        self.assertEqual(result["missing_observations"], ["cold"])
        self.assertEqual(result["pending_models"], ["pending"])
        self.assertEqual(control.published, [])
        self.assertEqual(result["plan"]["unmet_pages"], {"hot": 100})

    def test_shadow_runtime_accepts_grow_pulled_reallocation_policy_without_shrink_candidates(self):
        control = _Control(
            {
                "global_free_bytes": 0,
                "models": {
                    "hot": model(1_000, clean=200, high=600),
                    "cold": model(1_000, clean=900, high=600),
                },
            }
        )
        runtime = DMarginalTrialRuntime(
            control,
            trial_quantum_pages=100,
            min_revisit_opportunity_tokens=1_000,
            min_confirmed_loss_tokens=100,
            min_severity=0.05,
            min_volume_loss_tokens_per_s=10.0,
            trial_policy=InitialDReallocationPolicy(
                grow_quantum_pages=100,
                shrink_quantum_pages=40,
                cold_revisit_opportunity_tokens=5_000,
                cold_max_confirmed_loss_tokens=0,
            ),
        )

        result = runtime.tick(
            {
                "hot": {
                    "window_start_s": 0.0,
                    "window_end_s": 5.0,
                    "confirmed_reuse_loss_tokens": 500,
                    "revisit_opportunity_tokens": 2_000,
                },
                "cold": {
                    "window_start_s": 0.0,
                    "window_end_s": 5.0,
                    "confirmed_reuse_loss_tokens": 0,
                    "revisit_opportunity_tokens": 10_000,
                },
            },
            mode="shadow",
            trial_id="shadow-reallocation",
        )

        self.assertEqual(control.published, [])
        self.assertEqual(
            [(item["model_id"], item["action"], item["pages"])
             for item in result["candidates"]],
            [("hot", "grow", 100)],
        )
        self.assertEqual(result["plan"]["targets"], {"hot": 1_100, "cold": 900})
        self.assertEqual(
            [(item["source_model_id"], item["destination_model_id"], item["pages"])
             for item in result["plan"]["transactions"]],
            [("cold", "hot", 100)],
        )
        self.assertEqual(result["plan"]["unmet_pages"], {})


if __name__ == "__main__":
    unittest.main()

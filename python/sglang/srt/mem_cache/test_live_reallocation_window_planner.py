"""CPU tests for fixed-window live reallocation planning."""

from __future__ import annotations

from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parent))

from live_reallocation_window_planner import plan_windows


def model(capacity=1_000, *, floor=500, clean=900, high=600, ready=0):
    return {
        "capacity": capacity,
        "pending_capacity": 0,
        "pending_release_capacity": 0,
        "quota_target": None,
        "page_size": 1,
        "token_bytes": 1,
        "residency": {
            "effective_pages": capacity,
            "floor_pages": floor,
            "clean_pages": clean,
            "high_pages": high,
            "ready_pages": ready,
            "live_pages": capacity - clean,
            "unresolved_clean_pages": 0,
        },
    }


def report(d_raw, opportunity=20_000, duration=10.0):
    return {
        "window_start_s": 0.0,
        "window_end_s": duration,
        "confirmed_reuse_loss_tokens": d_raw,
        "revisit_opportunity_tokens": opportunity,
        "reprefill_tokens": d_raw // 2,
        "revisit_cache_match_tokens": max(0, opportunity - d_raw),
    }


class LiveReallocationWindowPlannerTest(unittest.TestCase):
    def test_high_urgency_window_outputs_grow_pulled_live_transfer_intents(self):
        snapshot = {
            "global_free_bytes": 0,
            "models": {
                "A": model(),
                "B": model(),
                "C": model(),
            },
        }
        windows = [
            {
                "decision_window_id": "a-hot",
                "reports": {
                    "A": report(8_000),
                    "B": report(0, opportunity=50_000),
                    "C": report(0, opportunity=50_000),
                },
            },
            {
                "decision_window_id": "no-hot",
                "reports": {
                    "A": report(0, opportunity=50_000),
                    "B": report(0, opportunity=50_000),
                    "C": report(0, opportunity=50_000),
                },
            },
        ]

        proofs, intents = plan_windows(
            snapshot=snapshot,
            windows=windows,
            base_grow_pages=100,
            base_shrink_pages=40,
            min_revisit_opportunity_tokens=1_000,
            min_confirmed_loss_tokens=100,
            min_severity=0.05,
            min_volume_loss_tokens_per_s=10.0,
            donor_max_severity=0.0,
            cold_revisit_opportunity_tokens=5_000,
            cold_max_confirmed_loss_tokens=0,
            high_urgency_multiplier=4,
            medium_urgency_multiplier=2,
            min_grow_pages=0,
            min_shrink_pages=0,
            grow_effective_ratio=0.0,
            shrink_effective_ratio=0.0,
        )

        self.assertEqual(len(proofs), 2)
        self.assertEqual(
            [
                (
                    item["donor"],
                    item["recipient"],
                    item["pages"],
                    item["phase"],
                    item["decision_window_id"],
                )
                for item in intents
            ],
            [
                ("B", "A", 300, "live_transaction_window", "a-hot"),
                ("C", "A", 100, "live_transaction_window", "a-hot"),
            ],
        )
        self.assertEqual(
            [(item["model_id"], item["action"], item["pages"])
             for item in proofs[0]["shadow"]["candidates"]],
            [("A", "grow", 400)],
        )
        self.assertEqual(proofs[0]["shadow"]["plan"]["unmet_pages"], {})
        self.assertEqual(proofs[1]["generated_intents"], [])

    def test_impact_sized_args_raise_window_quantum_above_tiny_base_pages(self):
        snapshot = {
            "global_free_bytes": 0,
            "models": {
                "A": model(capacity=200_000, floor=100_000, clean=180_000, high=1_000),
                "B": model(capacity=200_000, floor=100_000, clean=180_000, high=1_000),
                "C": model(capacity=200_000, floor=100_000, clean=180_000, high=1_000),
            },
        }
        windows = [
            {
                "decision_window_id": "a-hot",
                "reports": {
                    "A": report(8_000),
                    "B": report(0, opportunity=50_000),
                    "C": report(0, opportunity=50_000),
                },
            },
        ]

        proofs, intents = plan_windows(
            snapshot=snapshot,
            windows=windows,
            base_grow_pages=100,
            base_shrink_pages=40,
            min_revisit_opportunity_tokens=1_000,
            min_confirmed_loss_tokens=100,
            min_severity=0.05,
            min_volume_loss_tokens_per_s=10.0,
            donor_max_severity=0.0,
            cold_revisit_opportunity_tokens=5_000,
            cold_max_confirmed_loss_tokens=0,
            high_urgency_multiplier=4,
            medium_urgency_multiplier=2,
            min_grow_pages=5_000,
            min_shrink_pages=5_000,
            grow_effective_ratio=0.025,
            shrink_effective_ratio=0.025,
        )

        self.assertEqual(
            [(item["donor"], item["recipient"], item["pages"]) for item in intents],
            [("B", "A", 20_000)],
        )
        self.assertEqual(proofs[0]["shadow"]["plan"]["unmet_pages"], {})
        self.assertEqual(
            proofs[0]["shadow"]["plan"]["targets"],
            {"A": 220_000, "B": 180_000, "C": 200_000},
        )


if __name__ == "__main__":
    unittest.main()

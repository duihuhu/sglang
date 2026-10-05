"""Check M1 plot semantics against the existing compact confirmation data."""

import hashlib
import json
import unittest
from pathlib import Path

from plot_m1_evidence_series import PARTITION_CELLS, RESOURCE_CELLS, series_data


class M1FigureTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.path = Path(__file__).parent / "results/vtc_state_tax_20260826/analysis.json"
        cls.source = json.loads(cls.path.read_text(encoding="utf-8"))
        cls.data = series_data(cls.path)

    def test_source_fingerprint(self):
        self.assertEqual(self.data["source_sha256"], hashlib.sha256(self.path.read_bytes()).hexdigest())

    def test_equal_budget_and_state_tax_have_distinct_denominators(self):
        packed, split = self.data["opening"]["rows"]
        self.assertEqual(packed["offered_token_charge"], 17408)
        self.assertEqual(split["offered_token_charge"], 17408)
        self.assertEqual(split["input_per_request"] * split["requests_per_wave"], 16384)
        key = "fixed_state_per_initial_resident_token_relative_to_packed"
        self.assertEqual(packed[key], 1)
        self.assertEqual(split[key], 16)

    def test_peaks_normalized_by_their_own_pools(self):
        self.assertEqual(tuple(r["cell"] for r in self.data["resources"]), RESOURCE_CELLS)
        for row in self.data["resources"]:
            metric = self.source["decisions"]["cells"][row["cell"]]["token"]
            for tenant in row["tenants"]:
                name = tenant["tenant"]
                self.assertEqual(tenant["peak_state_slots"], metric[f"{name}_peak_state"])
                self.assertEqual(tenant["peak_state_fraction"], metric[f"{name}_peak_state"] / row["state_capacity"])
                self.assertEqual(tenant["peak_kv_fraction"], metric[f"{name}_peak_kv"] / row["kv_capacity"])

    def test_partition_scan_controls_and_latency_units(self):
        self.assertEqual(tuple(r["cell"] for r in self.data["partitions"]), PARTITION_CELLS)
        for row in self.data["partitions"]:
            self.assertEqual(row["state_capacity"], 48)
            self.assertEqual(row["kv_capacity"], 196608)
            self.assertEqual(row["wave_interval_s"], 1.5)
            self.assertEqual(row["split_factor"] * row["split_input_per_request"], 16384)
            self.assertEqual(row["split_factor"] * row["split_output_per_request"], 512)
            metric = self.source["decisions"]["cells"][row["cell"]]["token"]
            self.assertEqual(row["split_p95_slowdown_ratio"], 1 + metric["split_slowdown"])
            self.assertEqual(row["packed_queue_p95_s"], metric["packed_queue_p95"])

    def test_mean_absolute_gap_is_not_recomputed_from_means(self):
        control = self.data["partitions"][0]
        mean_gap = 100 * self.source["decisions"]["cells"]["f1-base"]["token"]["gap"]
        difference_of_means = 100 * abs(control["split_p95_slowdown_ratio"] - control["packed_p95_slowdown_ratio"])
        self.assertEqual(control["mean_absolute_slowdown_gap_pp"], mean_gap)
        self.assertNotAlmostEqual(mean_gap, difference_of_means)


if __name__ == "__main__":
    unittest.main()

import unittest

from audit_m2_capacity_evidence import KV_BYTES, RAW_BYTES, STATE_BYTES, audit_item, logical_bound


class CapacityEvidenceTests(unittest.TestCase):
    def test_old_workload_does_not_force_logical_capacity_shortfall(self):
        result = logical_bound([(512, 512, 128), (64, 8192, 64)])
        self.assertEqual(result["requests"], 576)
        self.assertEqual(result["joint_body_upper_bytes"], 19407568896)
        self.assertEqual(result["logical_upper_headroom_bytes"], 73503104)
        self.assertFalse(result["joint_body_upper_exceeds_budget"])

    def test_new_probe_can_change_joint_bound_without_hitting_request_count(self):
        result = logical_bound([(512, 512, 128), (64, 8192, 64), (24, 512, 64)])
        self.assertEqual(result["requests"], 600)
        self.assertTrue(result["all_requests_below_request_cap"])
        self.assertTrue(result["kv_body_alone_fits"])
        self.assertTrue(result["joint_body_upper_exceeds_budget"])

    def test_invalid_shapes(self):
        for shapes in ([], [(0, 512, 64)], [(1, 512)], [(1, 512, -1)]):
            with self.assertRaises(ValueError):
                logical_bound(shapes)

    def test_joint_peak_uses_same_event_not_sum_of_independent_peaks(self):
        item = {"experiment": {}, "requests": [dict(success=True, cached_tokens=0, num_retractions=0,
                input_len=8192, output_len=128) for _ in range(2)],
                "server_initial": {"internal_states": [{"page_size": 1, "m2_admission": {"shared_pool_bytes": RAW_BYTES}}]},
                "scheduler_resource_trace": {"events": [
                    dict(active_requests=2, state_slots=2, kv_tokens=100),
                    dict(active_requests=1, state_slots=1, kv_tokens=8000)],
                    "system": {"current_active_requests": 0}},
                "scheduler_batch_trace": {"admission_blocks": [{"reason": "request_limit"}]}}
        result = audit_item(item)
        self.assertEqual(result["peak_joint_live_body_bytes"], STATE_BYTES + 8000 * KV_BYTES)
        self.assertEqual(result["no_token_events"], 0)
        item["scheduler_batch_trace"]["admission_blocks"] += [{"reason": "prefill_no_token"}]
        self.assertEqual(audit_item(item)["no_token_events"], 1)
        item["scheduler_batch_trace"]["dropped_admission_blocks"] = 1
        with self.assertRaises(ValueError):
            audit_item(item)


if __name__ == "__main__":
    unittest.main()

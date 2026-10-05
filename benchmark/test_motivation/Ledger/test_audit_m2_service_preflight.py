import copy
import unittest

from audit_m2_service_preflight import inspect


def artifact():
    key, rid = "hybrid-tenant:test:A", "a"
    row = {"rid": rid, "tenant": key}
    events = [
        {"kind": "queue_counter", "members": [row], "counters_before": {},
         "counters_after": {key: 0}, "lift_delta": 0},
        {"kind": "queue_inserted", "members": [row], "inserted": True, "waiting": {key: 1}},
        {"kind": "admission", "members": [{**row, "first_input_charge": True, "input_charge": 4,
                                              "has_request_slot": False}],
         "counters_before": {key: 0}, "counters_after": {key: 4}, "waiting": {}},
        {"kind": "resource_observation", "source_event": {"added": [row], "removed": []}},
        {"kind": "batch_launch", "batch_id": 1,
         "members": [{**row, "phase": "prefill", "work_tokens": 4, "prefix_tokens": 0}]},
        {"kind": "output_charge", "members": [{**row, "previous_output": 0, "current_output": 1,
                                                 "output_charge": 2}],
         "counters_before": {key: 4}, "counters_after": {key: 6}},
        {"kind": "batch_result", "batch_id": 1, "members": [row], "finished_rids": [rid]},
        {"kind": "resource_observation", "source_event": {"added": [], "removed": [rid]}},
    ]
    for i, event in enumerate(events, 1):
        event.update(seq=i, observed_s=float(i))
    return {"experiment": {"run_id": "test"}, "client_clock": {"boot_id": "boot", "implementation": "clock"},
            "requests": [{"rid": rid, "input_len": 4, "completion_tokens": 1}],
            "service_trace": {"enabled": True, "events": events, "boot_id": "boot", "clock": "clock",
                              "run_ids": ["test"], "dropped_events": 0, "truncated_fields": 0}}


class AuditTests(unittest.TestCase):
    def test_complete_conservation(self):
        audit = inspect(artifact())
        self.assertEqual(audit["errors"], [])
        self.assertEqual(audit["both_waiting_seconds"], 0)

    def test_missing_clock_drop_or_sequence_is_rejected(self):
        for key, value in (("boot_id", "other"), ("dropped_events", 1), ("truncated_fields", 1)):
            item = artifact()
            item["service_trace"][key] = value
            self.assertTrue(inspect(item)["errors"])
        item = artifact()
        item["service_trace"]["events"][0]["seq"] = 9
        self.assertIn("non-contiguous sequence", inspect(item)["errors"])

    def test_unexplained_charge_and_duplicate_input_rejected(self):
        item = artifact()
        item["service_trace"]["events"][2]["members"][0]["input_charge"] = 8
        self.assertIn("counter change not explained by charges/lift", inspect(item)["errors"])
        self.assertIn("request charge/work conservation failed", inspect(item)["errors"])

    def test_missing_batch_or_resource_release_rejected(self):
        for index in (6, 7):
            item = artifact()
            item["service_trace"]["events"].pop(index)
            self.assertTrue(inspect(item)["errors"])

    def test_waiting_not_inferred_from_running_or_client_requests(self):
        item = artifact()
        events = item["service_trace"]["events"]
        events[1]["waiting"]["hybrid-tenant:test:B"] = 1
        self.assertEqual(inspect(item)["both_waiting_seconds"], 1)
        events[2]["waiting"] = {"hybrid-tenant:test:A": 1}
        self.assertEqual(inspect(item)["both_waiting_seconds"], 1)


if __name__ == "__main__":
    unittest.main()

import unittest

from analyze_m2_service_preflight import process
from test_audit_m2_service_preflight import artifact


class SummaryTests(unittest.TestCase):
    def test_charge_execution_lead_and_completed_prefill(self):
        item = artifact()
        item["experiment"].update(observer="on", target="packed", neighbor="alone")
        item.update(fingerprints={}, integrity_errors=[], experiment_clock_origin_s=100,
                    summary={"groups": {"A": {"p95_jct_s": 7}}},
                    scheduler_resource_trace={"system": {}, "tenants": {}, "events": [], "event_clock_origin_s": 100})
        item["service_trace"]["event_clock_origin_s"] = 100
        item["requests"][0].update(group="A", finished_s=8, output_len=1, token_events=[(7, 1)],
                                   submitted_s=0, scheduled_arrival_s=0)
        result = process(item)
        self.assertEqual(result["audit"]["errors"], [])
        self.assertEqual(result["max_charge_ahead_of_completed_work"], {"A": 4})
        last = result["counter_execution_curves"][-1]["tenants"]["A"]
        self.assertEqual(last["charge_without_lift"], last["executed_weighted_tokens"])
        self.assertEqual(last["executed_weighted_tokens"], 6)
        self.assertEqual(last["outstanding"], 0)
        self.assertEqual(result["output50_s"], {"A": 7})


if __name__ == "__main__":
    unittest.main()

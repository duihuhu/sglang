import copy
import unittest

from analyze_m2_slot_batching import extra_metrics


def specimen():
    return {
        "experiment": {"family": "E2"}, "experiment_clock_origin_s": 100,
        "requests": [
            {"rid": "s", "group": "short", "latency_s": 2, "ttft_s": .5,
             "prompt_tokens": 512, "completion_tokens": 64,
             "submitted_s": 0, "scheduled_arrival_s": 0, "finished_s": 2},
            {"rid": "l", "group": "long", "latency_s": 3, "ttft_s": 1,
             "prompt_tokens": 8192, "completion_tokens": 64,
             "submitted_s": 0, "scheduled_arrival_s": 0, "finished_s": 3},
        ],
        "summary": {"groups": {"short": {"p95_jct_s": 2}, "long": {"p95_jct_s": 3}}},
        "scheduler_resource_trace": {
            "event_clock_origin_s": 100,
            "system": {"peak_state_slots": 2, "peak_kv_tokens": 1024, "state_slot_seconds": 4.8},
            "events": [
                {"observed_s": .1, "added": [{"rid": "s"}, {"rid": "l"}], "removed": [],
                 "state_slots": 2, "active_requests": 2, "kv_tokens": 1024},
                {"observed_s": 2, "added": [], "removed": ["s"],
                 "state_slots": 1, "active_requests": 1, "kv_tokens": 512},
                {"observed_s": 3, "added": [], "removed": ["l"],
                 "state_slots": 0, "active_requests": 0, "kv_tokens": 0},
            ],
        },
        "scheduler_batch_trace": {
            "admission_blocks": [{"reason": "request_limit", "candidate_rid": "next",
                "selected_requests": 5, "selected_reusing_request_slots": 1,
                "runtime_cap": 8, "running_requests": 2, "request_slots_free": 5}],
            "events": [{"mode": "mixed", "extend_tokens": 512, "decoding_requests": 1},
                       {"mode": "decode", "extend_tokens": 512, "decoding_requests": 1}],
        },
    }


class MetricsTests(unittest.TestCase):
    def test_exact_slot_witness(self):
        self.assertEqual(extra_metrics(specimen())["reuse_slot_gate_witnesses"], 1)

    def test_runtime_cap_denial_is_not_slot_witness(self):
        item = specimen()
        item["scheduler_batch_trace"]["admission_blocks"][0]["runtime_cap"] = 7
        self.assertEqual(extra_metrics(item)["reuse_slot_gate_witnesses"], 0)

    def test_pool_exhaustion_is_not_slot_witness(self):
        item = specimen()
        item["scheduler_batch_trace"]["admission_blocks"][0]["request_slots_free"] = 4
        self.assertEqual(extra_metrics(item)["reuse_slot_gate_witnesses"], 0)

    def test_decode_with_stale_fields_is_not_mixed_prefill(self):
        value = extra_metrics(specimen())
        self.assertEqual(value["mixed_batches"], 1)
        self.assertEqual(value["prefill_batches"], 1)

    def test_admission_episode_and_tpot(self):
        value = extra_metrics(specimen())
        self.assertEqual(value["short_first_episode_admitted"], 1)
        self.assertEqual(value["long_first_episode_admitted"], 1)
        self.assertAlmostEqual(value["short_tpot_mean"], 1.5 / 63)
        self.assertEqual(value["short_max_jct"], 2)

    def test_late_activation_excluded_from_first_episode(self):
        item = copy.deepcopy(specimen())
        events = item["scheduler_resource_trace"]["events"]
        events[0]["added"] = [{"rid": "s"}]
        events[1]["added"] = [{"rid": "l"}]
        self.assertEqual(extra_metrics(item)["long_first_episode_admitted"], 0)


if __name__ == "__main__":
    unittest.main()

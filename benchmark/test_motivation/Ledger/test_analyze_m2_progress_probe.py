import unittest

from analyze_m2_progress_probe import intersect_intervals, metrics, overlap_duration, union_intervals


def artifact():
    events = []
    for t, rid, phase in ((0, "a", "prefill"), (1, "b", "prefill"), (2, "a", "decode"), (3, "b", "decode")):
        events.append({"launch_s": t, "recorded_s": t + .5, "members": [{
            "rid": rid, "phase": phase, "work_tokens": 2 if phase == "prefill" else 1}]})
    requests = [{"rid": rid, "group": "group", "submitted_s": 0., "scheduled_arrival_s": 0.,
                 "finished_s": finish, "latency_s": finish, "ttft_s": 1.,
                 "prompt_tokens": 2, "completion_tokens": 2} for rid, finish in (("a", 3.), ("b", 4.))]
    resource = {"event_clock_origin_s": 0., "events": [
        {"observed_s": 0., "added": [{"rid": "a"}], "removed": [], "state_slots": 1},
        {"observed_s": 1., "added": [{"rid": "b"}], "removed": [], "state_slots": 2},
        {"observed_s": 3., "added": [], "removed": ["a"], "state_slots": 1},
        {"observed_s": 4., "added": [], "removed": ["b"], "state_slots": 0}], "system": {
            "peak_active_requests": 2, "peak_kv_tokens": 4,
            "state_slot_seconds": 6., "kv_token_seconds": 12.}}
    return {"experiment_clock_origin_s": 0., "resolved_capacity": [2, 4, 2, 2],
            "requests": requests, "scheduler_resource_trace": resource,
            "scheduler_batch_trace": {"event_clock_origin_s": 0., "events": events, "admission_blocks": []}}


class AnalysisTests(unittest.TestCase):
    def test_interval_union_avoids_double_counting_waiters(self):
        windows = union_intervals([(0, 3), (1, 5), (7, 8), (6, 6)])
        self.assertEqual(windows, [(0, 5), (7, 8)])
        self.assertEqual(overlap_duration(2, 7.5, windows), 3.5)

    def test_intersection_requires_simultaneous_pressure(self):
        self.assertEqual(intersect_intervals([(0, 3), (5, 8)], [(2, 6), (7, 9)]),
                         [(2, 3), (5, 6), (7, 8)])

    def test_pending_before_full_pool_is_not_full_pool_blocking(self):
        result, _ = metrics(artifact())
        self.assertEqual(result["state_full_with_pending_seconds"], 0.)
        self.assertEqual(result["decode_ready_unselected_state_full_with_pending_slot_seconds_proxy"], 0.)

    def test_full_pool_gap_is_subset_of_pending_gap(self):
        item = artifact()
        item["requests"].append({**item["requests"][1], "rid": "c"})
        item["scheduler_resource_trace"]["events"][2]["added"] = [{"rid": "c"}]
        item["scheduler_resource_trace"]["events"][2]["state_slots"] = 2
        item["scheduler_resource_trace"]["events"][3]["removed"].append("c")
        item["scheduler_batch_trace"]["events"][3]["members"].append({
            "rid": "c", "phase": "prefill", "work_tokens": 2})
        result, _ = metrics(item)
        self.assertEqual(result["state_full_with_pending_seconds"], 2.)
        self.assertEqual(result["decode_ready_unselected_state_full_with_pending_slot_seconds_proxy"], 2.)

    def test_participation_gaps_are_not_queue_wait(self):
        result, detail = metrics(artifact())
        self.assertEqual(result["unselected_slot_seconds_proxy"], 2.)
        self.assertEqual(result["decode_ready_unselected_slot_seconds_proxy"], 2.)
        self.assertEqual(result["decode_ready_unselected_with_pending_slot_seconds_proxy"], 0.)
        self.assertEqual(result["mean_queue_s"], .5)
        self.assertEqual(result["mean_first_prefill_fraction"], 1.)
        self.assertEqual(result["decode_work_minus_delivered_histogram"], {0: 2})

    def test_prefill_execution_not_classified_as_decode_ready_gap(self):
        item = artifact()
        item["scheduler_batch_trace"]["events"][0]["recorded_s"] = 2.5
        result, _ = metrics(item)
        self.assertEqual(result["unselected_slot_seconds_proxy"], 2.)
        self.assertEqual(result["decode_ready_unselected_slot_seconds_proxy"], 1.)

    def test_mixed_batch_includes_both_holders(self):
        item = artifact()
        item["scheduler_batch_trace"]["events"][1]["members"].append({
            "rid": "a", "phase": "decode", "work_tokens": 1})
        result, _ = metrics(item)
        self.assertEqual(result["decode_ready_unselected_slot_seconds_proxy"], 1.)

    def test_rejects_nonmonotonic_launches(self):
        item = artifact()
        item["scheduler_batch_trace"]["events"][2]["launch_s"] = .1
        with self.assertRaises(ValueError):
            metrics(item)


if __name__ == "__main__":
    unittest.main()

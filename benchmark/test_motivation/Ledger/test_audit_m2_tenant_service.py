import copy
import json
from pathlib import Path
import tempfile
import unittest

from audit_m2_tenant_service import (
    audit, common_client_window, digest, features, inspect_item, milestones,
    output_events, output_window, read_verified,
)


def request(group="packed", rid="p", events=None):
    return dict(rid=rid, group=group, success=True, error=None, cached_tokens=0,
                num_retractions=0, input_len=10, prompt_tokens=10, output_len=4,
                completion_tokens=4, submitted_s=0.0, finished_s=6.0,
                first_token_s=1.0, token_events=events or [(1.0, 1), (3.0, 2), (5.0, 4)])


def artifact(factor=16, rep=10, cap=48, composition="mixed"):
    groups = ("packed", "split") if composition == "mixed" else (composition.removesuffix("-only"),)
    requests, tenants = [], {}
    for group in groups:
        count = 8 if group == "packed" else 8 * factor
        prompt, output = (16384, 512) if group == "packed" else (16384 // factor, 512 // factor)
        for index in range(count):
            req = request(group, f"{group}-{index}")
            req.update(input_len=prompt, prompt_tokens=prompt, output_len=output,
                       completion_tokens=output, token_events=[(1.0, 1), (5.0, output)])
            requests.append(req)
        tenants[group] = dict(active_request_seconds=count, state_slot_seconds=3 * count,
                              kv_token_seconds=count * prompt)
    state = dict(max_mamba_cache_size=480, max_total_tokens=196608,
                 pp_max_micro_batch_size=cap, chunked_prefill_size=4096,
                 schedule_policy="tenant-vtc" if composition == "mixed" else "fcfs")
    return dict(
        experiment=dict(stage="confirm", batch_mode="separate-4096", split_factor=factor,
                        repetition=rep, admission_cap=cap, composition=composition),
        requests=requests, integrity_errors=[], payload_fingerprint=f"{factor}/{rep}/{composition}",
        server_initial={"internal_states": [state]},
        server_final={"internal_states": [copy.deepcopy(state)]},
        scheduler_resource_trace={"system": {"current_active_requests": 0}, "tenants": tenants},
        scheduler_batch_trace={"event_count": 1, "events": [{"waiting_requests": 1}]},
        summary={"groups": {group: {"p95_jct_s": 6.0} for group in groups}},
        scheduler_vtc_trace=dict(weights=dict(request=0, input=1, input_quadratic=0,
                                             output=2, input_output=0, output_quadratic=0),
                                 observation_count=1, current_counter_gap=0, max_counter_gap=10,
                                 counters=dict.fromkeys(groups, 139264), queue_waits={}),
    )


class TenantServiceAuditTests(unittest.TestCase):
    def test_cumulative_counts_become_deltas_not_double_counted(self):
        events = output_events([request()])
        self.assertEqual([e[2] for e in events], [1, 1, 2])
        self.assertEqual(milestones(events, "packed", 4), {"25": 1.0, "50": 3.0, "75": 5.0, "100": 5.0})

    def test_invalid_stream_and_request_rejected(self):
        for field, value in [("success", False), ("num_retractions", 1), ("cached_tokens", 1),
                             ("completion_tokens", 3), ("first_token_s", 2.0),
                             ("token_events", [(1.0, 2), (3.0, 1), (5.0, 4)]),
                             ("token_events", [(1.0, 1), (0.5, 2), (5.0, 4)]),
                             ("token_events", [(1.0, 1), (3.0, 2)]),
                             ("token_events", [(1.0, 1), (7.0, 4)]),
                             ("submitted_s", float("nan"))]:
            item = request()
            item[field] = value
            with self.subTest(field=field, value=value), self.assertRaises(ValueError):
                output_events([item])

    def test_duplicate_request_rejected(self):
        with self.assertRaises(ValueError):
            output_events([request(), request()])

    def test_common_window_uses_latest_actual_submit_and_earliest_tenant_finish(self):
        a, b = request(), request("split", "s")
        b["submitted_s"], b["finished_s"] = 2.0, 8.0
        c, d = copy.deepcopy(a), copy.deepcopy(b)
        c["submitted_s"], c["finished_s"] = 2.5, 5.0
        self.assertEqual(common_client_window([{"requests": [a, b]}, {"requests": [c, d]}]), (2.5, 5.0))
        c["submitted_s"] = 8.0
        self.assertIsNone(common_client_window([{"requests": [a, b]}, {"requests": [c, d]}]))

    def test_window_subtracts_preexisting_output_and_coalesces_ties(self):
        events = [(0.0, "split", 3), (1.0, "packed", 1),
                  (2.0, "packed", 1), (2.0, "split", 1)]
        value = output_window(events, 1.0, 3.0, dict(packed=4, split=4))
        self.assertEqual(value["initial_delivered_output"], dict(packed=1, split=3))
        self.assertEqual(value["max_abs_incremental_output_fraction_gap"], 0)
        self.assertEqual(value["time_mean_abs_incremental_output_fraction_gap"], 0)

    def test_window_gap_and_area_are_output_only(self):
        events = [(2.0, "split", 2), (3.0, "packed", 1)]
        value = output_window(events, 1.0, 5.0, dict(packed=4, split=4))
        self.assertEqual(value["max_abs_incremental_output_fraction_gap"], 0.5)
        self.assertEqual(value["end_split_minus_packed_output_fraction"], 0.25)
        self.assertEqual(value["time_mean_abs_incremental_output_fraction_gap"], 0.25)
        self.assertIn("NOT", value["scope"])

    def test_missing_trace_is_not_inferred_from_summaries(self):
        item = dict(requests=[request()], scheduler_vtc_trace={"counters": {"packed": 10}},
                    scheduler_resource_trace={"tenants": {"packed": {}}},
                    scheduler_batch_trace={"events": [{"waiting_requests": 2, "tenant_counts": {"packed": 1}}]})
        observed = features(item)
        self.assertTrue(observed["vtc_counter_summary"])
        self.assertFalse(observed["vtc_counter_events"])
        self.assertFalse(observed["per_tenant_waiting_snapshots"])
        self.assertFalse(observed["resource_events"])
        self.assertFalse(observed["per_request_batch_work"])
        self.assertFalse(observed["client_absolute_origin"])

    def test_hash_mismatch_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "artifact.json"
            raw = json.dumps({"example": 1}).encode()
            path.write_bytes(raw)
            self.assertEqual(read_verified(path, digest(raw)), {"example": 1})
            with self.assertRaises(ValueError):
                read_verified(path, "0" * 64)

    def test_full_item_preserves_extra_state_buffers(self):
        row, events = inspect_item(artifact())
        self.assertEqual(row["request_count"], 136)
        for group in ("packed", "split"):
            self.assertEqual(row["tenants"][group]["completed_token_charge"], 139264)
            self.assertEqual(row["tenants"][group]["state_slots_per_active_request_time"], 3)
            self.assertEqual(sum(n for _, g, n in events if g == group), 4096)

    def test_dropped_trace_and_unreleased_resources_rejected(self):
        for field, key in [("scheduler_batch_trace", "dropped_events"),
                           ("scheduler_batch_trace", "dropped_admission_blocks"),
                           ("scheduler_resource_trace", "dropped_events")]:
            item = artifact()
            item[field][key] = 1
            with self.subTest(key=key), self.assertRaises(ValueError):
                inspect_item(item)
        item = artifact()
        item["scheduler_resource_trace"]["system"]["current_active_requests"] = 1
        with self.assertRaises(ValueError):
            inspect_item(item)

    def test_config_drift_and_changed_cost_rejected(self):
        item = artifact()
        item["server_final"]["internal_states"][0]["disable_radix_cache"] = True
        with self.assertRaises(ValueError):
            inspect_item(item)
        item = artifact()
        item["scheduler_vtc_trace"]["weights"]["request"] = 1
        with self.assertRaises(ValueError):
            inspect_item(item)

    def test_complete_matrix_and_missing_artifact_guard(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            hashes = {}
            for factor in (16, 32):
                directory = root / f"confirm_f{factor}"
                directory.mkdir()
                for rep in range(10, 15):
                    for cap in (48, 96):
                        for composition in ("mixed", "packed-only", "split-only"):
                            path = directory / f"{rep}-{cap}-{composition}.json"
                            raw = json.dumps(artifact(factor, rep, cap, composition)).encode()
                            path.write_bytes(raw)
                            hashes[str(path)] = digest(raw)
            index = root / "index.json"
            index.write_text(json.dumps({"source_hashes": hashes}))
            result = audit(index)
            self.assertEqual(result["counts"], dict(verified_artifacts=60, requests=8000,
                                                   mixed_runs=20, isolation_runs=40))
            self.assertEqual(len(result["paired_client_windows"]), 10)
            self.assertEqual(result["feature_coverage_mixed_runs"]["vtc_counter_events"], 0)
            for row in result["mixed_rows"]:
                self.assertEqual(row["tenants"]["split"]["matched_p95_slowdown"], 1)
            hashes.pop(next(iter(hashes)))
            index.write_text(json.dumps({"source_hashes": hashes}))
            with self.assertRaises(ValueError):
                audit(index)


if __name__ == "__main__":
    unittest.main()

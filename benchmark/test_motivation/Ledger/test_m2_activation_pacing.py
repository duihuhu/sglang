import ast
import copy
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock

from verify_m2_slot_admission import SCHEDULER, method


class ActivationGateTests(unittest.TestCase):
    def setUp(self):
        self.gate = method(SCHEDULER, "Scheduler", "_tenant_activation_limit_reached", {})
        self.scheduler = SimpleNamespace()

    def test_default_is_disabled(self):
        self.assertFalse(self.gate(self.scheduler, 100, 0, False))

    def test_zero_disables(self):
        self.scheduler._tenant_new_activation_limit = 0
        self.assertFalse(self.gate(self.scheduler, 100, 0, False))

    def test_continuations_do_not_spend_fresh_allowance(self):
        self.scheduler._tenant_new_activation_limit = 4
        self.assertFalse(self.gate(self.scheduler, 4, 1, False))
        self.assertTrue(self.gate(self.scheduler, 5, 1, False))

    def test_existing_candidate_not_throttled(self):
        self.scheduler._tenant_new_activation_limit = 1
        self.assertFalse(self.gate(self.scheduler, 5, 1, True))

    def test_empty_system_can_always_start(self):
        for limit in (0, 1, 4):
            self.scheduler._tenant_new_activation_limit = limit
            self.assertFalse(self.gate(self.scheduler, 0, 0, False))

    def test_actual_candidate_loop_stops_without_marking_capacity_full(self):
        tree = ast.parse(SCHEDULER.read_text())
        owner = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "Scheduler")
        function = next(n for n in owner.body if isinstance(n, ast.FunctionDef) and n.name == "_get_new_batch_prefill_raw")
        loop = next(n for n in function.body if isinstance(n, ast.For)
                    and ast.unparse(n.iter) == "self.waiting_queue")
        requests = [SimpleNamespace(req_pool_idx=None, rid=f"new{i}", init_next_round_input=Mock()) for i in range(4)]
        adder = SimpleNamespace(can_run_list=[SimpleNamespace(req_pool_idx=3)])
        def add(req, **kwargs):
            adder.can_run_list.append(req)
            return "continue"
        adder.add_one_req = add
        scheduler = SimpleNamespace(
            _tenant_new_activation_limit=1, enable_lora=False, waiting_queue=requests,
            disaggregation_mode="null", enable_hicache_storage=False,
            enable_priority_preemption=False, tree_cache=object(), chunked_req=None,
            truncation_align_size=None, _record_tenant_admission_block=Mock(),
            _prefill_req_limit_reached=lambda *args: False,
        )
        scheduler._tenant_activation_limit_reached = lambda *args: self.gate(scheduler, *args)
        running = SimpleNamespace(reqs=[object()], batch_is_full=False)
        scope = {"self": scheduler, "adder": adder, "reused_req_slots": 1,
                 "running_batch": running, "DisaggregationMode": SimpleNamespace(PREFILL="prefill"),
                 "AddReqResult": SimpleNamespace(CONTINUE="continue")}
        exec(compile(ast.Module(body=[loop], type_ignores=[]), str(SCHEDULER), "exec"), scope)
        self.assertEqual(adder.can_run_list[1:], requests[:1])
        self.assertFalse(running.batch_is_full)
        requests[0].init_next_round_input.assert_called_once()
        for req in requests[1:]:
            req.init_next_round_input.assert_not_called()
        self.assertEqual(scheduler._record_tenant_admission_block.call_args.args[0], "activation_pacing")


class ActivationControlTests(unittest.TestCase):
    def setUp(self):
        self.namespace = {"ENABLE_TENANT_RESOURCE_TRACKING": True, "logging": Mock(),
                          "logger": Mock(), "SetInternalStateReqOutput": SimpleNamespace}
        self.update = method(SCHEDULER, "Scheduler", "set_internal_state", self.namespace)
        self.scheduler = SimpleNamespace(max_running_requests=8, is_fully_idle=lambda: True,
                                         spec_algorithm=SimpleNamespace(is_none=lambda: True),
                                         metrics_reporter=SimpleNamespace())

    def call(self, limit):
        return self.update(self.scheduler, SimpleNamespace(server_args={"tenant_new_activation_limit": limit}))

    def test_requires_opt_in(self):
        self.namespace["ENABLE_TENANT_RESOURCE_TRACKING"] = False
        self.assertFalse(self.call(1).updated)
        self.assertFalse(hasattr(self.scheduler, "_tenant_new_activation_limit"))

    def test_busy_scheduler_rejected_without_mutation(self):
        self.scheduler.is_fully_idle = lambda: False
        self.assertFalse(self.call(1).updated)
        self.assertFalse(hasattr(self.scheduler, "_tenant_new_activation_limit"))

    def test_invalid_limits_rejected(self):
        for limit in (-1, 9, 1.5, "1", None, True, float("nan")):
            with self.subTest(limit=limit):
                self.assertFalse(self.call(limit).updated)
                self.assertFalse(hasattr(self.scheduler, "_tenant_new_activation_limit"))

    def test_enable_and_disable(self):
        self.assertTrue(self.call(4).updated)
        self.assertEqual(self.scheduler._tenant_new_activation_limit, 4)
        self.assertTrue(self.call(0).updated)
        self.assertEqual(self.scheduler._tenant_new_activation_limit, 0)


try:
    from run_m2_activation_pacing import build_specs, command, fingerprint, integrity_errors, validate_info
    from analyze_m2_activation_pacing import analyze, metrics, nearest_rank, paired_summary
    HAS_RUNNER = True
except ModuleNotFoundError as error:
    if error.name != "aiohttp":
        raise
    HAS_RUNNER = False


def readback(mode="mixed-1024", limit=0):
    return {"internal_states": [{
        "tp_size": 2, "pp_size": 1, "tenant_slot_gate_code": 1, "tenant_new_activation_limit": limit,
        "schedule_policy": "tenant-vtc", "tenant_vtc_request_weight": 0,
        "tenant_vtc_input_weight": 1, "tenant_vtc_output_weight": 2,
        "tenant_vtc_input_quadratic_weight": 0, "tenant_vtc_input_output_weight": 0,
        "tenant_vtc_output_quadratic_weight": 0, "disable_radix_cache": True,
        "chunked_prefill_size": 1024, "enable_mixed_chunk": True,
        "enable_prefill_delayer": False, "min_free_slots_delay": None, "prefill_max_requests": None,
        "max_mamba_cache_size": 8, "max_total_tokens": 131072,
        "pp_max_micro_batch_size": 8, "effective_max_running_requests_per_dp": 8,
        "cuda_graph_config": {phase: {"backend": "disabled"} for phase in ("prefill", "decode")},
        "tenant_resource_trace": {"event_trace_enabled": True},
    }]}


@unittest.skipUnless(HAS_RUNNER, "aiohttp required for experiment runner imports")
class WorkloadAndReadbackTests(unittest.TestCase):
    def specs(self, run="a", order="short-first", waves=4):
        return build_specs(run, 1, order, waves, 2, 16, 4, 64)

    def test_payloads_identical_across_interventions(self):
        self.assertEqual(fingerprint(self.specs("a")), fingerprint(self.specs("b")))
        self.assertNotEqual(self.specs("a")[0].rid, self.specs("b")[0].rid)

    def test_order_changes_arrival_but_not_contents(self):
        a, b = self.specs(), self.specs(order="long-first")
        self.assertNotEqual(fingerprint(a), fingerprint(b))
        self.assertEqual(fingerprint(a, arrivals=False), fingerprint(b, arrivals=False))

    def test_waves_have_distinct_ids_and_no_output_oracle(self):
        specs = self.specs()
        self.assertEqual(len(specs), 80)
        self.assertEqual(len({s.rid for s in specs}), 80)
        self.assertEqual(len({s.cache_salt for s in specs}), 80)
        self.assertEqual({s.output_len for s in specs}, {64})
        self.assertEqual(sum(s.group == "short" for s in specs), 64)

    def test_invalid_workload_rejected(self):
        with self.assertRaises(ValueError):
            build_specs("a", 1, "bad", 1, 2, 16, 4, 64)
        with self.assertRaises(ValueError):
            build_specs("a", 1, "short-first", 1, .01, 16, 4, 64)

    def test_natural_regime_does_not_force_resource_cap(self):
        args = command("mixed-1024", "natural", 30129)
        for flag in ("--max-mamba-cache-size", "--max-total-tokens", "--max-running-requests"):
            self.assertNotIn(flag, args)
        self.assertEqual(args[args.index("--tp-size") + 1], "2")

    def test_single_dp_readback_is_expected_for_tp2(self):
        self.assertEqual(validate_info(readback(), "mixed-1024", "state8", 0), (8, 131072, 8, 8))

    def test_old_gate_or_other_pacing_rejected(self):
        for key, value in (("tenant_slot_gate_code", 0), ("tenant_new_activation_limit", 1),
                           ("prefill_max_requests", 4), ("min_free_slots_delay", 2), ("tp_size", 1)):
            data = readback()
            data["internal_states"][0][key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                validate_info(data, "mixed-1024", "state8", 0)

    def test_missing_readback_key_rejected(self):
        data = readback()
        del data["internal_states"][0]["prefill_max_requests"]
        with self.assertRaises(ValueError):
            validate_info(data, "mixed-1024", "state8", 0)

    def test_natural_readback_requires_resolved_positive_capacities(self):
        data = readback()
        data["internal_states"][0]["max_mamba_cache_size"] = None
        with self.assertRaises(ValueError):
            validate_info(data, "mixed-1024", "natural", 0)


@unittest.skipUnless(HAS_RUNNER, "aiohttp required for experiment runner imports")
class AnalysisTests(unittest.TestCase):
    def specimen(self, limit=0):
        from test_analyze_m2_slot_batching import specimen
        item = specimen()
        item["experiment"] = {"run_id": "m2p-test", "stage": "screen", "mode": "mixed-1024",
                              "regime": "state8", "activation_limit": limit, "order": "short-first",
                              "rep": 1, "session": 0, "waves": 1, "interval_s": 2,
                              "short_per_wave": 1, "long_per_wave": 1, "output_tokens": 64}
        item["integrity_errors"] = []
        item["resolved_capacity"] = [8, 131072, 8, 8]
        item["server_initial"] = readback(limit=limit)
        item["server_final"] = readback(limit=limit)
        item["payload_fingerprint"] = "test-payload"
        item["scheduler_vtc_trace"] = {"max_counter_gap": 100}
        item["gpu_health_samples"] = [{"gpus": [
            {"index": i, "utilization": 100, "sm_mhz": 1400} for i in (6, 7)]}]
        item["scheduler_resource_trace"]["system"].update(
            peak_active_requests=2, current_active_requests=0, kv_token_seconds=1000)
        item["scheduler_batch_trace"]["admission_blocks"] = []
        for r in item["requests"]:
            r.update(input_len=r["prompt_tokens"], output_len=64, success=True,
                     cached_tokens=0, num_retractions=0, token_events=[(.5, 1), (2, 64)])
        return item

    def test_small_sample_p95_is_nearest_rank_not_interpolation(self):
        self.assertEqual(nearest_rank([1] * 15 + [12]), 12)

    def test_mismatched_pairs_rejected(self):
        with self.assertRaises(ValueError):
            paired_summary({1: {"x": 1}}, {2: {"x": 1}})

    def test_zero_baseline_is_not_infinite_improvement(self):
        value = paired_summary({1: {"events": 0}}, {1: {"events": 4}})
        self.assertIsNone(value["events"]["mean_reduction_fraction"])
        self.assertEqual(value["events"]["mean_difference"], 4)

    def test_effect_is_paired_not_ratio_of_means(self):
        value = paired_summary({1: {"latency": 10}, 2: {"latency": 20}},
                               {1: {"latency": 5}, 2: {"latency": 18}})
        self.assertAlmostEqual(value["latency"]["mean_reduction_fraction"], .3)

    def test_metrics_include_both_groups_and_all_request_mean(self):
        item = self.specimen()
        self.assertEqual(integrity_errors(item, 2, .05), [])
        value = metrics(item)
        self.assertEqual(value["all_mean_jct"], 2.5)
        self.assertEqual(value["short_mean_jct"], 2)
        self.assertEqual(value["long_mean_jct"], 3)
        self.assertAlmostEqual(value["short_mean_queue"], .1)
        self.assertAlmostEqual(value["short_mean_active"], 1.9)

    def test_missing_and_dropped_traces_rejected(self):
        item = self.specimen()
        item["scheduler_resource_trace"]["dropped_events"] = 1
        item["scheduler_batch_trace"]["dropped_admission_blocks"] = 1
        item["scheduler_vtc_trace"] = None
        self.assertEqual(integrity_errors(item, 2, .05), ["VTC trace", "batch trace", "resource trace"])

    def test_excessive_lateness_rejected(self):
        item = self.specimen()
        item["requests"][0]["submitted_s"] = .1
        self.assertIn("arrival lateness", integrity_errors(item, 2, .05))

    def test_old_slot_counterexample_rejected(self):
        from test_analyze_m2_slot_batching import specimen
        item = self.specimen()
        item["scheduler_batch_trace"]["admission_blocks"] = specimen()["scheduler_batch_trace"]["admission_blocks"]
        self.assertIn("legacy slot defect reappeared", integrity_errors(item, 2, .05))

    def test_pacing_has_its_own_count_and_is_not_a_capacity_failure(self):
        item = self.specimen(limit=1)
        item["scheduler_batch_trace"]["admission_blocks"] = [{
            "reason": "activation_pacing", "candidate_rid": "next", "new_activation_limit": 1,
            "selected_requests": 2, "selected_reusing_request_slots": 1, "running_requests": 1}]
        self.assertEqual(integrity_errors(item, 2, .05), [])
        value = metrics(item)
        self.assertEqual(value["blocks_activation_pacing"], 1)
        self.assertEqual(value["blocks_request_limit"], 0)
        self.assertEqual(value["pacing_with_continuation"], 1)
        item["experiment"]["activation_limit"] = 0
        self.assertIn("pacing count mismatch", integrity_errors(item, 2, .05))

    def test_end_to_end_synthetic_matrix_and_unpaired_rejection(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            args = {"stage": "screen", "modes": ["mixed-1024"], "regime": "state8",
                    "sessions": 1, "reps_per_session": 1, "rep_start": 1, "waves": 1,
                    "short_per_wave": 1, "long_per_wave": 1, "output_tokens": 64,
                    "interval": 2, "max_lateness": .05}
            manifest = {"args": args, "source_hashes": {}, "planned_measured_requests": 12}
            (root / "manifest.json").write_text(json.dumps(manifest))
            (root / "completed.json").write_text(json.dumps({"completed": True, "source_hashes": {}}))
            for limit in (0, 1, 4):
                for order in ("short-first", "long-first"):
                    item = self.specimen(limit)
                    item["experiment"].update(order=order, run_id=f"m2p-{limit}-{order}")
                    (root / f"m2p-{limit}-{order}.json").write_text(json.dumps(item))
            data = analyze([root])
            self.assertTrue(data["complete_planned_matrix"])
            self.assertEqual(len(data["paired_effects"]), 4)
            changed = root / "m2p-1-short-first.json"
            item = json.loads(changed.read_text())
            item["payload_fingerprint"] = "changed"
            changed.write_text(json.dumps(item))
            with self.assertRaises(ValueError):
                analyze([root])


if __name__ == "__main__":
    unittest.main()

"""CPU-only hook contracts, using actual VTC method bodies from the repository."""

import ast
from collections import Counter, defaultdict, deque
import copy
from pathlib import Path
import time
from types import SimpleNamespace as NS
import unittest

from m2_service_observer import ServiceRecorder, cpu_number, install_observer
from m2_service_workload import fingerprints, workload


ROOT = Path(__file__).resolve().parents[3]


def policy_class():
    source = ast.parse((ROOT / "python/sglang/srt/managers/schedule_policy.py").read_text())
    cls = next(n for n in source.body if isinstance(n, ast.ClassDef) and n.name == "SchedulePolicy")
    names = {"_tenant", "on_request_queued", "_sort_by_tenant_vtc", "on_requests_admitted",
             "on_batch_processed", "_vtc_admission_cost", "_vtc_output_cost", "_observe_vtc_counters"}
    cls.body = [n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name in names]
    ns = dict(Counter=Counter, defaultdict=defaultdict, deque=deque, time=time,
              CacheAgnosticPolicy=NS(TENANT_VTC="vtc", TENANT_VECTOR_DEBT="vector", TENANT_LIFECYCLE_COST="life"))
    tree = ast.Module(body=[ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0), cls], type_ignores=[])
    exec(compile(ast.fix_missing_locations(tree), "actual_vtc_methods", "exec"), ns)
    return ns["SchedulePolicy"]


def new_policy(cls):
    p = cls()
    p.policy, p.tenant_counters, p.last_client_left_queue = "vtc", defaultdict(float), None
    p._vtc_queued_at, p._vtc_output_lengths, p._vtc_input_charged_rids = {}, {}, set()
    p._vtc_peak_waiting_requests, p._vtc_queue_waits = defaultdict(int), defaultdict(list)
    p._vtc_current_counter_gap = p._vtc_max_counter_gap = p._vtc_counter_observations = 0
    p.vtc_request_weight = p.vtc_input_quadratic_weight = p.vtc_input_output_weight = p.vtc_output_quadratic_weight = 0
    p.vtc_input_weight, p.vtc_output_weight = 1, 2
    return p


class Req:
    def __init__(self, rid="a", key="A", length=4):
        self.rid, self.extra_key = rid, f"hybrid-tenant:test:{key}"
        self.origin_input_ids, self.output_ids = [1] * length, []
        self.req_pool_idx = self.mamba_pool_idx = None
        self.done = False

    def finished(self):
        return self.done


def fake_classes():
    class Adder:
        def __init__(self):
            self.can_run_list = []
            self.rem_total_tokens = self.cur_rem_tokens = self.rem_chunk_tokens = 100

        def add_one_req(self, req, **kwargs):
            self.can_run_list.append(req)
            return NS(name="OTHER")  # Selected AND stopped, not a rejection.

        def add_chunked_req(self, req):
            self.can_run_list.append(req)
            return req

    class Scheduler:
        def __init__(self, policy):
            self.policy = policy
            self.schedule_policy = "tenant-vtc"
            self.ps = NS(pp_rank=0, attn_tp_rank=0, attn_cp_rank=0)
            self.waiting_queue, self.chunked_req, self.forward_ct = [], None, 0
            self.req_to_token_pool = NS(available_size=lambda: 100, mamba_allocator=NS(available_size=lambda: 100))
            self.token_to_kv_pool_allocator = NS(available_size=lambda: 1000)
            self.tenant_resource_tracker = NS(events=[], started_at=time.perf_counter())

        def _add_request_to_queue(self, req, is_retracted=False):
            self.policy.on_request_queued(req, self.waiting_queue, is_retracted=is_retracted)
            self.waiting_queue.append(req)
            return "queued"

        def _get_new_batch_prefill_raw(self, prefill_delayer_single_pass, running_batch):
            self.policy._sort_by_tenant_vtc(self.waiting_queue)
            adder = Adder()
            if self.chunked_req is not None:
                adder.add_chunked_req(self.chunked_req)
                self.chunked_req = None
            if self.waiting_queue:
                adder.add_one_req(self.waiting_queue[0])
            self.waiting_queue = [r for r in self.waiting_queue if r not in adder.can_run_list]
            self.policy.on_requests_admitted(adder.can_run_list, self.waiting_queue)
            return NS(reqs=adder.can_run_list), running_batch

        def _record_tenant_admission_block(self, reason, running_bs, req=None, adder=None):
            return reason

        def run_batch(self, batch, **kwargs):
            self.forward_ct += 1
            batch.forward_iter = self.forward_ct
            return batch

        def process_batch_result(self, batch, result):
            self.policy.on_batch_processed(batch)
            return result

        def observe_tenant_resources(self, *batches):
            return 5

        def get_internal_state(self, req):
            return NS(internal_state={})

    return Scheduler, Adder


class ObserverTests(unittest.TestCase):
    def setUp(self):
        self.Policy = policy_class()
        self.Scheduler, self.Adder = fake_classes()
        self.undo = install_observer(self.Scheduler, self.Policy, self.Adder, enabled=True)
        self.s = self.Scheduler(new_policy(self.Policy))

    def tearDown(self):
        self.undo()

    def trace(self):
        return self.s.get_internal_state(None).internal_state["m2_service_trace"]

    def test_disabled_does_not_patch(self):
        self.undo()
        original = self.Policy.on_request_queued
        install_observer(self.Scheduler, self.Policy, self.Adder)
        self.assertIs(self.Policy.on_request_queued, original)

    def test_queue_lift_is_separate_from_work(self):
        self.s.policy.tenant_counters["hybrid-tenant:test:A"] = 100
        a, b = Req(), Req("b", "B")
        self.assertEqual(self.s._add_request_to_queue(a), "queued")
        self.s._add_request_to_queue(b)
        events = [e for e in self.trace()["events"] if e["kind"] == "queue_counter"]
        self.assertEqual(events[-1]["lift_delta"], 100)
        self.assertEqual(events[-1]["lift_reason"], "active_waiting_floor")
        self.assertEqual(events[-1]["waiting_before"], {a.extra_key: 1})

    def test_actual_policy_order_and_counter_unchanged(self):
        expected = new_policy(self.Policy)
        reqs = [Req("a1"), Req("a2"), Req("b", "B", 8)]
        waiting = []
        for req in reqs:
            expected.on_request_queued(req, waiting)
            waiting.append(req)
            self.s._add_request_to_queue(req)
        expected._sort_by_tenant_vtc(waiting)
        self.s.policy._sort_by_tenant_vtc(self.s.waiting_queue)
        self.assertEqual([r.rid for r in waiting], ["a1", "b", "a2"])
        self.assertEqual(self.s.waiting_queue, waiting)
        self.assertEqual(dict(expected.tenant_counters), dict(self.s.policy.tenant_counters))
        event = self.trace()["events"][-1]
        self.assertEqual([r["shadow_charge"] for r in event["candidates"]], [4, 8, 4])

    def test_chunk_reuse_is_not_charged_twice_and_other_can_be_selected(self):
        req = Req()
        self.s._add_request_to_queue(req)
        running = NS(reqs=[], batch_is_full=False)
        self.s._get_new_batch_prefill_raw(None, running)
        req.req_pool_idx = req.mamba_pool_idx = object()
        self.s.chunked_req = req
        self.s._get_new_batch_prefill_raw(None, running)
        admissions = [e for e in self.trace()["events"] if e["kind"] == "admission"]
        self.assertEqual([e["members"][0]["input_charge"] for e in admissions], [4, 0])
        candidates = [e for e in self.trace()["events"] if e["kind"] == "candidate"]
        self.assertTrue(candidates[0]["selected"])
        self.assertEqual(candidates[0]["result"], "OTHER")
        self.assertTrue(candidates[1]["member"]["has_request_slot"])
        self.assertTrue(candidates[1]["continued_chunk"])

    def test_output_increment_and_batch_identity(self):
        req = Req()
        self.s._add_request_to_queue(req)
        self.s.policy.on_requests_admitted([req], [])
        batch = NS(reqs=[req], forward_iter=42)
        req.output_ids.extend([7, 8])
        self.assertEqual(self.s.process_batch_result(batch, "result"), "result")
        event = [e for e in self.trace()["events"] if e["kind"] == "output_charge"][-1]
        self.assertEqual(event["batch_id"], 42)
        self.assertEqual(event["members"][0]["output_charge"], 4)
        self.assertEqual(event["counters_after"][req.extra_key], 8)

    def test_launch_work_is_cpu_only(self):
        req = Req()
        batch = NS(reqs=[req], forward_mode=NS(name="EXTEND"), extend_lens=[4], prefix_lens=[0])
        self.assertIs(self.s.run_batch(batch), batch)
        event = self.trace()["events"][0]
        self.assertEqual(event["members"][0]["work_tokens"], 4)
        self.assertEqual(event["batch_id"], batch.forward_iter)

    def test_reset_follows_new_policy_instance(self):
        self.s._add_request_to_queue(Req())
        self.assertGreater(len(self.trace()["events"]), 0)
        self.s.policy = new_policy(self.Policy)
        self.assertEqual(self.trace()["events"], [])

    def test_primary_rank_only(self):
        self.s.ps.attn_tp_rank = 1
        self.s._add_request_to_queue(Req())
        self.assertEqual(self.trace(), {"enabled": False})

    def test_bounded_trace_and_monotonic_sequence(self):
        recorder = ServiceRecorder(self.s, 2)
        for _ in range(3):
            recorder.emit("test")
        self.assertEqual(recorder.dropped, 1)
        self.assertEqual([e["seq"] for e in recorder.events], [1, 2])
        self.assertLessEqual(recorder.events[0]["observed_s"], recorder.events[1]["observed_s"])

    def test_device_scalar_never_converted(self):
        class DeviceScalar:
            def __int__(self):
                raise AssertionError("device conversion")
            def __float__(self):
                raise AssertionError("device conversion")
        self.assertIsNone(cpu_number(DeviceScalar()))
        self.assertIsNone(cpu_number(float("inf")))


class WorkloadTests(unittest.TestCase):
    def test_target_is_exactly_fixed_across_neighbors_sessions_and_labels(self):
        for target in ("packed", "split"):
            reference = fingerprints(workload("ref", target, "alone"))["A"]
            for neighbor in ("alone", "packed", "split"):
                for swap in (False, True):
                    self.assertEqual(reference, fingerprints(workload("new", target, neighbor, swap=swap))["A"])

    def test_neighbor_canonical_token_stream_and_budget_unchanged(self):
        packed = fingerprints(workload("x", "packed", "packed"))["B"]
        split = fingerprints(workload("y", "packed", "split"))["B"]
        for key in ("canonical_input", "input_tokens", "output_tokens"):
            self.assertEqual(packed[key], split[key])
        self.assertNotEqual(packed["request_shape_arrivals"], split["request_shape_arrivals"])

    def test_order_does_not_regenerate_payload_and_subset_is_exact(self):
        a = workload("x", "split", "packed", first="A")
        b = workload("x", "split", "packed", first="B")
        self.assertEqual(fingerprints(a)["A"]["canonical_input"], fingerprints(b)["A"]["canonical_input"])
        self.assertEqual([r for r in a if r.group == "A"], workload("x", "split", "alone"))
        self.assertNotEqual(fingerprints(a)["A"]["request_shape_arrivals"], fingerprints(b)["A"]["request_shape_arrivals"])


if __name__ == "__main__":
    unittest.main()

import importlib.util
import os
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[3]
spec = importlib.util.spec_from_file_location("m2_tracker_under_test", ROOT /
    "python/sglang/srt/managers/scheduler_components/tenant_resource_tracker.py")
tracker_module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = tracker_module
spec.loader.exec_module(tracker_module)


def request(rid, kv):
    return SimpleNamespace(rid=rid, extra_key="hybrid-tenant:test:t", mamba_pool_idx=1,
                           mamba_ping_pong_track_buffer=None,
                           kv=SimpleNamespace(kv_allocated_len=kv, swa_evicted_seqlen=0))


class EventTests(unittest.TestCase):
    def test_activation_continuation_and_release(self):
        t = tracker_module.TenantResourceTracker(capture_events=True)
        t.started_at = t.last_observed_at = 10
        a = request("a", 4096)
        t.observe([a, a], now=11)
        a.kv.kv_allocated_len = 8192
        t.observe([a], now=12)
        t.observe([], now=13)
        events = t.snapshot()["events"]
        self.assertEqual([e["state_slots"] for e in events], [1, 1, 0])
        self.assertEqual([len(e["added"]) for e in events], [1, 0, 0])
        self.assertEqual(events[-1]["removed"], ["a"])
        self.assertEqual(t.snapshot()["system"]["state_slot_seconds"], 2)

    def test_trace_is_bounded_and_reports_drops(self):
        t = tracker_module.TenantResourceTracker(capture_events=True, max_events=1)
        t.observe([request("a", 1)])
        t.observe([])
        self.assertEqual(len(t.events), 1)
        self.assertEqual(t.dropped_events, 1)

    def test_disabled_trace_preserves_accounting(self):
        t = tracker_module.TenantResourceTracker(capture_events=False)
        t.observe([request("a", 4096)])
        self.assertEqual(t.events, [])
        self.assertEqual(t.snapshot()["system"]["peak_state_slots"], 1)

    def test_admission_trace_opt_in_and_bound(self):
        with patch.dict(os.environ, {"SGLANG_TENANT_EVENT_TRACE": "0"}):
            t = tracker_module.TenantBatchTracker(max_events=1)
            t.record_admission_block({"reason": "request_limit"})
            self.assertEqual(t.admission_blocks, [])
        with patch.dict(os.environ, {"SGLANG_TENANT_EVENT_TRACE": "1"}):
            t = tracker_module.TenantBatchTracker(max_events=1)
            t.record_admission_block({"reason": "request_limit"})
            t.record_admission_block({"reason": "prefill_no_token"})
            self.assertEqual(t.snapshot()["dropped_admission_blocks"], 1)


class WorkloadTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            import run_m2_hybrid_factors
        except ModuleNotFoundError as error:
            if error.name == "aiohttp":
                raise unittest.SkipTest("runner dependencies tested in serving container")
            raise
        cls.m = run_m2_hybrid_factors

    def test_order_changes_arrivals_not_contents(self):
        a = self.m.build_specs("a", "E2", "short-first", 10)
        b = self.m.build_specs("b", "E2", "long-first", 10)
        self.assertEqual(self.m.fingerprint(a, arrivals=False), self.m.fingerprint(b, arrivals=False))
        self.assertNotEqual(self.m.fingerprint(a), self.m.fingerprint(b))
        self.assertEqual(len(a), 20)

    def test_equal_prefill_not_equal_decode_work(self):
        for n in (1, 4, 16, 32):
            specs = self.m.build_specs("a", "E1", f"n{n}", 10)
            self.assertEqual(sum(len(s.input_ids) for s in specs), 16384)
            self.assertEqual(sum(s.output_len for s in specs), 128 * n)

    def test_lifecycle_probe_is_identical(self):
        a = self.m.build_specs("a", "E3", "o32", 10)
        b = self.m.build_specs("b", "E3", "o256", 10)
        self.assertEqual(self.m.fingerprint(a[-1:]), self.m.fingerprint(b[-1:]))
        self.assertEqual(a[-1].arrival_offset_s, 3)

    def test_no_graph_or_prefix_confound(self):
        for regime in self.m.REGIMES:
            cmd = self.m.command(regime, 30127)
            for flag in ("--cuda-graph-backend-decode", "--cuda-graph-backend-prefill"):
                self.assertEqual(cmd[cmd.index(flag) + 1], "disabled")
            self.assertIn("--disable-radix-cache", cmd)


if __name__ == "__main__":
    unittest.main()

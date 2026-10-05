import importlib.util
from pathlib import Path
from types import SimpleNamespace
import sys
import unittest
from unittest.mock import patch


PATH = Path(__file__).resolve().parents[3] / "python/sglang/srt/managers/scheduler_components/tenant_resource_tracker.py"
SPEC = importlib.util.spec_from_file_location("m2_progress_tracker", PATH)
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)

try:
    from run_m2_progress_probe import progress_errors, specs_for, validate_natural_info
except ModuleNotFoundError:
    specs_for = None


def request(rid, tenant="hybrid-tenant:probe:short"):
    return SimpleNamespace(rid=rid, extra_key=tenant)


class ProgressTests(unittest.TestCase):
    def batch(self, mode="MIXED"):
        reqs = [request("new"), request("existing"), request("decode")]
        return SimpleNamespace(launch_ts=20., forward_mode=SimpleNamespace(name=mode),
                               reqs=reqs, extend_lens=[512, 1024, 1],
                               prefix_lens=[0, 4096, 512], extend_num_tokens=1537,
                               decoding_reqs=[reqs[-1]])

    def tracker(self):
        with patch.dict("os.environ", {"SGLANG_TENANT_EVENT_TRACE": "1",
                                      "SGLANG_TENANT_PROGRESS_TRACE": "1"}):
            tracker = MODULE.TenantBatchTracker()
        tracker.started_at = 10.
        return tracker

    def test_mixed_work_and_continuation(self):
        tracker = self.tracker()
        batch = self.batch()
        tracker.record(batch, SimpleNamespace(num_correct_drafts=0), waiting_requests=5)
        event = tracker.events[0]
        self.assertEqual([m["work_tokens"] for m in event["members"]], [512, 1024, 1])
        self.assertEqual([m["phase"] for m in event["members"]], ["prefill", "prefill", "decode"])
        self.assertEqual([m["prefix_tokens"] for m in event["members"]], [0, 4096, None])
        self.assertEqual(event["launch_s"], 10.)
        batch.extend_lens[0] = 99
        self.assertEqual(event["members"][0]["work_tokens"], 512)

    def test_default_off_preserves_compact_trace(self):
        with patch.dict("os.environ", {"SGLANG_TENANT_PROGRESS_TRACE": "0"}):
            tracker = MODULE.TenantBatchTracker()
        tracker.record(self.batch(), SimpleNamespace(), waiting_requests=0)
        self.assertNotIn("members", tracker.events[0])

    def test_progress_requires_event_opt_in(self):
        with patch.dict("os.environ", {"SGLANG_TENANT_PROGRESS_TRACE": "1",
                                      "SGLANG_TENANT_EVENT_TRACE": "0"}):
            tracker = MODULE.TenantBatchTracker()
        self.assertFalse(tracker.capture_progress)

    def test_decode_ignores_stale_extend_fields(self):
        tracker = self.tracker()
        batch = self.batch("DECODE")
        batch.decoding_reqs = None
        tracker.record(batch, SimpleNamespace(), waiting_requests=0)
        self.assertEqual([m["work_tokens"] for m in tracker.events[0]["members"]], [1, 1, 1])

    def test_ignores_unlabeled_requests(self):
        tracker = self.tracker()
        batch = self.batch()
        batch.reqs[0].extra_key = None
        tracker.record(batch, SimpleNamespace(), waiting_requests=0)
        self.assertEqual(len(tracker.events[0]["members"]), 2)

    def test_missing_work_is_explicit_not_fabricated(self):
        tracker = self.tracker()
        batch = self.batch()
        batch.extend_lens = None
        tracker.record(batch, SimpleNamespace(), waiting_requests=0)
        self.assertIsNone(tracker.events[0]["members"][0]["work_tokens"])


@unittest.skipIf(specs_for is None, "Replay dependencies unavailable locally")
class ReplayTests(unittest.TestCase):
    def test_resolved_capacity_does_not_modify_config_readback(self):
        info = {"internal_states": [{"max_total_tokens": None, "memory_usage": {"token_capacity": 123456}}]}
        with patch("run_m2_progress_probe.validate_activation_info", return_value=(485, 123456, 485, 485)) as validate:
            self.assertEqual(validate_natural_info(info, "mixed-4096")[1], 123456)
            self.assertEqual(validate.call_args.args[0]["internal_states"][0]["max_total_tokens"], 123456)
        self.assertIsNone(info["internal_states"][0]["max_total_tokens"])

    def test_manual_capacity_rejected(self):
        with self.assertRaises(ValueError):
            validate_natural_info({"internal_states": [{"max_total_tokens": 100}]}, "mixed-4096")

    def test_shapes_have_equal_input_not_equal_output(self):
        counts = []
        for shape in ("short", "medium", "long", "mixed"):
            specs = specs_for("run", shape, 1, 100)
            self.assertEqual(sum(len(s.input_ids) for s in specs), 131072)
            counts.append(len(specs))
        self.assertEqual(sum(counts) * 4, 1888)

    def test_same_payload_across_modes(self):
        a = specs_for("mode-a", "mixed", 2, 100)
        b = specs_for("mode-b", "mixed", 2, 100)
        self.assertEqual([(s.input_ids, s.output_len, s.arrival_offset_s) for s in a],
                         [(s.input_ids, s.output_len, s.arrival_offset_s) for s in b])

    def test_prefill_validation_rejects_repeated_prefix(self):
        item = {"requests": [{"rid": "r", "prompt_tokens": 1024}], "scheduler_batch_trace": {
            "progress_trace_enabled": True, "events": [{"members": [{
                "rid": "r", "work_tokens": 512, "prefix_tokens": 0, "phase": "prefill"}]}] * 2}}
        self.assertIn("prefill gap or repeated work", progress_errors(item))

    def test_prefill_validation_accepts_complete_progress(self):
        item = {"requests": [{"rid": "r", "prompt_tokens": 1024}], "scheduler_batch_trace": {
            "progress_trace_enabled": True, "events": [{"members": [{
                "rid": "r", "work_tokens": 512, "prefix_tokens": start, "phase": "prefill"}]}
                for start in (0, 512)]}}
        self.assertEqual(progress_errors(item), [])


if __name__ == "__main__":
    unittest.main()

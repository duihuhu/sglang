import copy
import unittest
from unittest.mock import patch

from analyze_m2_unified_admission import control_delays, metrics, paired, screen_status


def row(policy, rep, jct=10, p95=20, span=21, errors=None, delay=32):
    return {"experiment": {"policy": policy, "profile": "state", "phase": "near", "rep": rep},
            "fingerprint": f"payload-{rep}", "scene_errors": errors or [],
            "control": {"delayed": delay, "native_slot_delays": 0, "native_prefill_reasons": {}},
            "metrics": {"mean_jct_s": jct, "p95_jct_s": p95, "makespan_s": span, "state_slot_seconds": 100}}


class PairedTests(unittest.TestCase):
    def test_two_reps_required(self):
        rows = [row("native", 51), row("decode32", 51, 9)]
        self.assertFalse(paired(rows)[0]["two_rep_screen_positive"])
        rows += [row("native", 52), row("decode32", 52, 9)]
        self.assertTrue(paired(rows)[0]["two_rep_screen_positive"])

    def test_tail_tradeoff_not_net_gain(self):
        rows = [r for rep in (51, 52) for r in (row("native", rep), row("decode32", rep, 9, p95=22))]
        self.assertFalse(paired(rows)[0]["two_rep_screen_positive"])

    def test_invalid_scene_kept_but_not_passed(self):
        rows = [r for rep in (51, 52) for r in (row("native", rep), row("decode32", rep, 9, errors=["wrong phase"]))]
        result = paired(rows)[0]
        self.assertFalse(result["two_rep_screen_positive"])
        self.assertEqual(len(result["samples"]), 2)

    def test_untriggered_diagnostic_not_gain(self):
        rows = [r for rep in (51, 52) for r in (row("native", rep), row("decode32", rep, 9, delay=0))]
        self.assertFalse(paired(rows)[0]["two_rep_screen_positive"])

    def test_payload_and_duplicate_protection(self):
        a, b = row("native", 51), row("decode32", 51)
        b["fingerprint"] = "other"
        with self.assertRaises(ValueError):
            paired([a, b])
        with self.assertRaises(ValueError):
            paired([a, copy.deepcopy(a)])

    def test_exact_missing_session(self):
        from recover_m2_unified_admission import check_missing
        rows = [{"experiment": {"policy": p, "profile": profile, "phase": phase, "rep": rep}}
                for p, rep in (("native", 51), ("decode32", 51), ("decode32", 52))
                for profile in ("state", "kv") for phase in ("far", "near")]
        self.assertEqual(len(check_missing(rows)), 4)
        with self.assertRaises(ValueError):
            check_missing(rows[:-1])
        with self.assertRaises(ValueError):
            check_missing(rows+[copy.deepcopy(rows[0])])

    def test_status_distinguishes_missing_invalid_and_negative(self):
        rows = [row("native", 51), row("decode32", 51, 9)]
        self.assertEqual(screen_status(paired(rows)[0]), "PENDING: incomplete pairs")
        rows += [row("native", 52), row("decode32", 52, 9)]
        self.assertEqual(screen_status(paired(rows)[0]), "SCREEN POSITIVE")
        rows[-1]["metrics"]["mean_jct_s"] = 11
        self.assertEqual(screen_status(paired(rows)[0]), "NO CONSISTENT CONSTRAINED GAIN")
        rows[-1]["control"]["delayed"] = 0
        self.assertEqual(screen_status(paired(rows)[0]), "NOT TRIGGERED")
        rows[-1]["scene_errors"] = ["wrong phase"]
        self.assertEqual(screen_status(paired(rows)[0]), "INVALID SCENE")

    def test_ttft_kept_separate_from_queue_and_jct(self):
        item = {"experiment": {"arrival": None}, "requests": [
            {"rid": "h", "group": "holder", "latency_s": 10, "ttft_s": 3,
             "submitted_s": 0, "scheduled_arrival_s": 0, "finished_s": 10},
            {"rid": "c", "group": "candidate", "latency_s": 20, "ttft_s": 7,
             "submitted_s": 0, "scheduled_arrival_s": 0, "finished_s": 20}],
            "scheduler_resource_trace": {"system": {"peak_active_requests": 2, "state_slot_seconds": 25}},
            "scheduler_batch_trace": {"admission_blocks": []}}
        with patch("analyze_m2_unified_admission.lifecycle", return_value={"h": (1, 10), "c": (4, 20)}):
            result = metrics(item)
        self.assertEqual(result["mean_ttft_s"], 5)
        self.assertEqual(result["p95_ttft_s"], 7)
        self.assertEqual(result["candidate"]["mean_queue_s"], 4)
        self.assertEqual(result["candidate"]["mean_ttft_s"], 7)
        self.assertEqual(result["candidate"]["mean_jct_s"], 20)

    def test_native_delays_not_diagnostic_counter(self):
        own = row("slots64", 51, delay=0)
        own["control"]["native_slot_delays"] = 7
        self.assertEqual(control_delays(own), 7)
        own["experiment"]["policy"] = "delayer32"
        own["control"]["native_prefill_reasons"] = {"no_wait": 12, "delay": 4}
        self.assertEqual(control_delays(own), 4)
        self.assertEqual(control_delays(row("decode32", 51)), 32)


if __name__ == "__main__":
    unittest.main()

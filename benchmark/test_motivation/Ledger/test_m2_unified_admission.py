"""Guard the new screening workload, scene labels and one-shot intervention."""

import ast
from pathlib import Path
import unittest

from run_m2_unified_admission import command, fingerprint, scene_errors, specs_for


def decision_function():
    path = Path(__file__).with_name("launch_m2_unified_admission.py")
    source = ast.parse(path.read_text())
    function = next(n for n in source.body if isinstance(n, ast.FunctionDef) and n.name == "decision")
    namespace = {}
    exec(compile(ast.Module(body=[function], type_ignores=[]), str(path), "exec"), namespace)
    return namespace["decision"]


class WorkloadTests(unittest.TestCase):
    def test_calibration_counts(self):
        for profile, count, length in (("state", 512, 512), ("kv", 96, 16384)):
            specs = specs_for("a", profile, 41)
            self.assertEqual(len(specs), count)
            self.assertTrue(all(len(s.input_ids) == length and s.output_len == 128 for s in specs))

    def test_candidate_budget_below_request_limit(self):
        for profile in ("state", "kv"):
            specs = specs_for("a", profile, 51, 15.)
            candidates = [s for s in specs if s.group == "candidate"]
            self.assertEqual(len(candidates), 64)
            self.assertLess(len(specs), 607)
            self.assertEqual(candidates[0].arrival_offset_s, 15.)
            self.assertAlmostEqual(candidates[-1].arrival_offset_s, 15.063)
            self.assertTrue(all(s.tenant_key == "shared" for s in specs))

    def test_fingerprints_ignore_policy_id(self):
        a = fingerprint(specs_for("native", "state", 51, 15.))
        b = fingerprint(specs_for("decode32", "state", 51, 15.))
        self.assertEqual(a, b)
        self.assertNotEqual(a, fingerprint(specs_for("native", "state", 52, 15.)))
        self.assertNotEqual(a, fingerprint(specs_for("native", "state", 51, 16.)))

    def test_same_physical_pool_across_policies(self):
        for policy in ("native", "decode32", "slots64", "delayer32"):
            cmd = command(policy, 30144)
            self.assertEqual(cmd[cmd.index("--max-mamba-cache-size")+1], "607")
            self.assertEqual(cmd[cmd.index("--max-total-tokens")+1], "781725")
            self.assertIn("--enable-unified-memory", cmd)
            self.assertIn("launch_m2_unified_admission.py", cmd[1])
        self.assertIn("--min-free-slots-delay", command("slots64", 30144))
        self.assertIn("--enable-prefill-delayer", command("delayer32", 30144))


class DecisionTests(unittest.TestCase):
    def setUp(self):
        self.decide = decision_function()
        self.state = {"started": False, "finished": False, "delayed": 0}
        self.kw = dict(has_candidate=True, holders=10, chunked=False, limit=32)

    def test_disabled_policy(self):
        self.assertFalse(self.decide(self.state, **{**self.kw, "limit": 0}))
        self.assertEqual(self.state["delayed"], 0)

    def test_finite_one_shot(self):
        self.assertTrue(all(self.decide(self.state, **self.kw) for _ in range(32)))
        self.assertFalse(self.decide(self.state, **self.kw))
        self.assertFalse(self.decide(self.state, **self.kw))
        self.assertEqual(self.state["delayed"], 32)

    def test_continuing_chunk_not_delayed(self):
        self.assertFalse(self.decide(self.state, **{**self.kw, "chunked": True}))
        self.assertEqual(self.state["delayed"], 0)

    def test_no_idle_delay(self):
        self.assertFalse(self.decide(self.state, **{**self.kw, "holders": 0}))
        self.assertFalse(self.decide(self.state, **{**self.kw, "has_candidate": False}))

    def test_stop_after_holders_release(self):
        self.assertTrue(self.decide(self.state, **self.kw))
        self.assertFalse(self.decide(self.state, **{**self.kw, "holders": 0}))
        self.assertFalse(self.decide(self.state, **self.kw))


class SceneTests(unittest.TestCase):
    def test_valid_and_invalid_labels(self):
        snap = dict(holders=100, alive=95, prefilled=100, median_remaining=32, state_fraction=.8)
        self.assertFalse(scene_errors(snap, "state", "near"))
        self.assertTrue(scene_errors(snap, "state", "far"))
        self.assertTrue(scene_errors(snap, "kv", "near"))
        self.assertTrue(scene_errors({**snap, "alive": 79}, "state", "near"))
        self.assertTrue(scene_errors({**snap, "prefilled": 94}, "state", "near"))
        self.assertFalse(scene_errors({**snap, "state_fraction": .2, "median_remaining": 112}, "kv", "far"))


if __name__ == "__main__":
    unittest.main()

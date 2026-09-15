import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from aflex_benchmark.semantics import (
    QUESTIONS,
    SEMANTIC_AF_COORDINATOR_PORT,
    SEMANTIC_AF_PORT_BASE,
    build_semantic_plans,
    expected_match,
    load_reused_baseline,
    normalize_text,
    parse_generate_response,
    port_conflict_snapshot,
    summarize,
)


class SemanticValidationTests(unittest.TestCase):
    def test_response_parser_handles_sglang_and_variants(self):
        parsed = parse_generate_response({"text": "Paris", "meta_info": {
            "output_ids": [1, 2], "completion_tokens": 2}})
        self.assertEqual((parsed["text"], parsed["output_ids"], parsed["output_token_count"]),
                         ("Paris", [1, 2], 2))
        self.assertEqual(parse_generate_response({"choices": [{"text": "Mars"}]})["text"], "Mars")
        self.assertEqual(parse_generate_response({"outputs": [{"generated_text": "H2O"}]})["text"], "H2O")
        with self.assertRaisesRegex(ValueError, "generated text"):
            parse_generate_response({"meta_info": {}})

    def test_normalization_and_expected_aliases(self):
        self.assertEqual(normalize_text("  The PACIFIC—Ocean! "), "the pacific ocean")
        examples = ["5.", "Paris", "H₂O", "Mars", "seven", "The Pacific Ocean.",
                    "George Orwell", "3 sides"]
        for question, answer in zip(QUESTIONS, examples):
            with self.subTest(question=question["id"]):
                self.assertTrue(expected_match(answer, question))
        self.assertFalse(expected_match("London", QUESTIONS[1]))

    def test_summary_keeps_semantics_and_token_equality_separate(self):
        baseline = [{"question_id": "arithmetic", "api_success": True,
                     "semantic_pass": True, "normalized_text": "5", "output_ids": [5]}]
        af = [{"question_id": "arithmetic", "api_success": True,
               "semantic_pass": True, "normalized_text": "five", "output_ids": [9]}]
        result = summarize({"baseline": baseline, "af": af})
        self.assertTrue(result["systems"]["af"]["all_semantic_pass"])
        self.assertFalse(result["comparisons"][0]["normalized_text_exact"])
        self.assertFalse(result["comparisons"][0]["token_ids_exact"])

    def test_plans_cover_native_workers_and_all_af_attention_endpoints(self):
        _, plans = build_semantic_plans(ROOT, ("baseline", "af"))
        self.assertEqual(len(plans["baseline"].endpoints), 2)
        self.assertEqual(len(plans["af"].endpoints), 8)
        self.assertEqual(plans["baseline"].routing_policy.value, "client_round_robin")
        self.assertEqual(plans["af"].routing_policy.value, "client_round_robin")
        self.assertEqual(plans["af"].runtime_options["warmup_parallel"], True)
        self.assertEqual([plans["af"].handle.endpoint_for_request(i)
                          for i in range(8)], plans["af"].endpoints)


    def test_semantic_af_has_isolated_legal_unique_port_plan(self):
        _, plans = build_semantic_plans(ROOT, ("af",), run_tag="semantic-test")
        plan = plans["af"]
        snapshot = port_conflict_snapshot(plan)
        all_ports = [port for ports in snapshot["ports_by_host"].values() for port in ports]
        edges = plan.runtime_options["afd_pool_edges"]
        self.assertEqual(SEMANTIC_AF_PORT_BASE, 12000)
        self.assertEqual(next(process.port for process in plan.processes
                              if process.role == "AF_COORDINATOR"),
                         SEMANTIC_AF_COORDINATOR_PORT)
        self.assertEqual(len(edges), 64)
        self.assertEqual(max(edge["attn_handshake_base_port"] + 1 for edge in edges), 37501)
        self.assertGreaterEqual(min(all_ports), SEMANTIC_AF_PORT_BASE - 800)
        self.assertLessEqual(max(all_ports), 37501)
        self.assertTrue(snapshot["unique"])
        self.assertFalse(snapshot["collisions"])
        self.assertEqual(set(snapshot["free_check_commands"]),
                         {process.host for process in plan.processes})
        self.assertTrue(all("ss -ltnpH" in command
                            for command in snapshot["free_check_commands"].values()))
        self.assertTrue(all("semantic-test-af" in process.log_path
                            for process in plan.processes))

    def test_reuse_baseline_accepts_file_and_directory(self):
        source = ROOT / "results/semantic_validation_20260824/semantic_results.json"
        rows_from_file = load_reused_baseline(source, expected_model="/models/Qwen3-32B/")
        rows_from_directory = load_reused_baseline(source.parent)
        self.assertEqual([row["question_id"] for row in rows_from_file],
                         [question["id"] for question in QUESTIONS])
        self.assertEqual(rows_from_file, rows_from_directory)
        summary = summarize({"baseline": rows_from_file, "af": rows_from_file})
        self.assertEqual(summary["determinism"]["paired_questions"], 8)
        self.assertTrue(summary["systems"]["baseline"]["all_semantic_pass"])

    def test_reuse_baseline_rejects_bad_schema(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "semantic_results.json"
            for payload, message in [
                ({"schema_version": 2}, "schema_version 1"),
                ({"schema_version": 1, "model": "/models/Qwen3-32B/",
                  "sampling_params": {"temperature": 1}, "systems": {"baseline": []}},
                 "sampling_params"),
            ]:
                with self.subTest(message=message):
                    path.write_text(json.dumps(payload))
                    with self.assertRaisesRegex(ValueError, message):
                        load_reused_baseline(path)

    def test_dry_run_has_no_results_or_remote_side_effects(self):
        script = ROOT / "scripts/validate_af_semantics.py"
        with tempfile.TemporaryDirectory() as directory:
            results = Path(directory) / "must-not-exist"
            completed = subprocess.run(
                [sys.executable, str(script), "--results", str(results)],
                text=True, capture_output=True, check=True,
            )
            plan = json.loads(completed.stdout)
            self.assertTrue(plan["no_side_effects"])
            self.assertFalse(results.exists())
            af = plan["plans"]["af"]
            self.assertEqual(next(process["port"] for process in af["processes"]
                                  if process["role"] == "AF_COORDINATOR"), 19308)
            self.assertTrue(plan["port_conflict_snapshots"]["af"]["unique"])

    def test_cli_reuse_requires_af_system(self):
        script = ROOT / "scripts/validate_af_semantics.py"
        completed = subprocess.run(
            [sys.executable, str(script), "--reuse-baseline", str(ROOT / "results")],
            text=True, capture_output=True, check=False,
        )
        self.assertEqual(completed.returncode, 2)
        self.assertIn("requires --system af", completed.stderr)



if __name__ == "__main__":
    unittest.main(verbosity=2)

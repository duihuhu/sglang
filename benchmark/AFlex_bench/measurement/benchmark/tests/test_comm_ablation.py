import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "run_comm_ablation", ROOT / "scripts/run_comm_ablation.py")
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def make_bundle() -> dict:
    points = []
    for index in range(3):
        key = f"link-{index}"
        common = {"model": "model", "nodes": 1, "architecture": "native",
                  "workloads": ["short"], "metadata": {"comm_ablation_id": key}}
        points.append({**common, "id": f"{key}-smoke", "qps": [1], "smoke": True,
                       "metadata": {**common["metadata"],
                                    "comm_ablation_phase": "smoke"}})
        points.append({**common, "id": f"{key}-qps2-a", "qps": [2],
                       "metadata": {**common["metadata"],
                                    "comm_ablation_phase": "qps2"}})
        points.append({**common, "id": f"{key}-qps2-b", "qps": [2],
                       "metadata": {**common["metadata"],
                                    "comm_ablation_phase": "qps2"}})
    return {"matrix": {"points": points},
            "models": {"models": {"model": {"path": "/model"}}},
            "workloads": {"classes": ["short"], "qps": [1, 2]},
            "cluster": {}}


class CommAblationTests(unittest.TestCase):
    def test_build_runs_filters_metadata_and_repeats_only_qps2(self):
        runs = MODULE.build_runs(make_bundle(), "all", 3)
        self.assertEqual(len(runs), 3 + 6 * 3)
        self.assertEqual([row["phase"] for row in runs[:3]], ["smoke"] * 3)
        self.assertEqual([row["repeat"] for row in runs[3:6]], [0, 1, 2])
        self.assertEqual(len({row["run_id"] for row in runs}), len(runs))

    def test_matrix_honors_declared_cardinality_and_requires_point_phase(self):
        bundle = make_bundle()
        bundle["matrix"]["metadata"] = {"expected_points": 9}
        bundle["matrix"]["points"].pop()
        with self.assertRaisesRegex(ValueError, "exactly 9"):
            MODULE.validate_matrix(bundle)
        bundle = make_bundle()
        del bundle["matrix"]["points"][0]["metadata"]["comm_ablation_phase"]
        with self.assertRaisesRegex(ValueError, "declare exactly one"):
            MODULE.validate_matrix(bundle)

    def test_matrix_without_declared_cardinality_accepts_generic_point_count(self):
        bundle = make_bundle()
        bundle["matrix"]["points"].extend(
            {**bundle["matrix"]["points"][index],
             "id": f"generic-extra-{index}"}
            for index in range(3)
        )
        self.assertEqual(len(MODULE.validate_matrix(bundle)), 12)

    def test_plan_writes_manifest_and_status_without_execution(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = MODULE.orchestrate(
                make_bundle(), root / "matrix.json", root / "results",
                "smoke", 3, False)
            status = json.loads((root / "results/status.json").read_text())
            saved = json.loads((root / "results/run_manifest.json").read_text())
        self.assertEqual(len(manifest["runs"]), 3)
        self.assertEqual(saved["execution"], "strictly_serial")
        self.assertEqual(status["status"], "planned")
        self.assertEqual(status["counts"], {"pending": 3})

    def test_invalid_link_skips_formal_repeats_and_continues(self):
        calls = []
        def execute(item, bundle, results):
            calls.append((item["point"]["id"], item["repeat"]))
            if item["point"]["id"] == "link-1-smoke":
                return {"status": "failed", "error": "invalid_link: RDMA unavailable"}
            return {"status": "complete", "requests_success": 1,
                    "completion_tokens": 4}
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = MODULE.orchestrate(
                make_bundle(), root / "matrix.json", root / "results",
                "all", 2, True, executor=execute)
            status = json.loads((root / "results/status.json").read_text())
        link1_qps = [row for row in manifest["runs"]
                     if row["ablation_id"] == "link-1" and row["phase"] == "qps2"]
        self.assertTrue(all(row["status"] == "skipped_invalid_link"
                            for row in link1_qps))
        self.assertFalse(any(point.startswith("link-1-qps2") for point, _ in calls))
        self.assertTrue(any(point.startswith("link-2-qps2") for point, _ in calls))
        self.assertEqual(status["status"], "complete")

    def test_failed_point_is_recorded_and_later_runs_continue(self):
        calls = []
        def execute(item, bundle, results):
            calls.append(item["point"]["id"])
            if item["point"]["id"] == "link-0-smoke":
                raise RuntimeError("synthetic failure")
            return {"status": "complete", "requests_success": 1,
                    "completion_tokens": 1}
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = MODULE.orchestrate(
                make_bundle(), root / "matrix.json", root / "results",
                "smoke", 3, True, executor=execute)
        self.assertEqual(manifest["runs"][0]["status"], "failed")
        self.assertEqual(len(calls), 3)

    def test_smoke_requires_success_and_complete_tokens(self):
        self.assertEqual(MODULE.classify_result(
            {"status": "complete", "requests_success": 1,
             "completion_tokens": 0}, "smoke")[0], "failed")
        self.assertEqual(MODULE.classify_result(
            {"status": "complete", "requests_success": 1,
             "completion_tokens": 1}, "smoke")[0], "complete")

    def test_resume_runs_only_pending_entries(self):
        calls = []
        def execute(item, bundle, results):
            calls.append(item["run_id"])
            return {"status": "complete", "requests_success": 1,
                    "completion_tokens": 1}
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            results = root / "results"
            MODULE.orchestrate(make_bundle(), root / "matrix.json", results,
                               "smoke", 3, False)
            manifest_path = results / "run_manifest.json"
            manifest = json.loads(manifest_path.read_text())
            manifest["runs"][0]["status"] = "complete"
            manifest_path.write_text(json.dumps(manifest))
            resumed = MODULE.orchestrate(
                make_bundle(), root / "matrix.json", results, "smoke", 3,
                True, resume=True, executor=execute)
        self.assertEqual(len(calls), 2)
        self.assertTrue(all(row["status"] == "complete" for row in resumed["runs"]))


if __name__ == "__main__":
    unittest.main()

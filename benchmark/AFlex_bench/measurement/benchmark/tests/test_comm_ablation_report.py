import json
import tempfile
import unittest
from pathlib import Path

from aflex_benchmark.comm_ablation_report import (
    EXPECTED_VALID_POINTS,
    build_report,
    render_markdown,
    write_report,
)


def write_run(root: Path, point: str, repeat: int, value: float = 1.0,
              *, status: str = "complete", link_status: str = "pass") -> None:
    run = root / point / f"repeat{repeat}" / f"run{repeat}"
    run.mkdir(parents=True, exist_ok=True)
    architecture, link = point.split("_", 1)
    summary = {
        "status": status, "run_id": f"run{repeat}", "achieved_qps": value,
        "output_throughput_tokens_s": value * 10,
        "energy": {"total_j": value * 100},
        "energy_per_output_token_j": value * 2,
        "average_cluster_power_w": value * 3,
        "metrics": {name: {p: value * multiplier for p in ("p50", "p90", "p99")}
                    for name, multiplier in (("ttft_client_ms", 4),
                                             ("tpot_ms", 5), ("e2e_ms", 6))},
        "point": {"architecture": architecture, "expected_link": link},
    }
    validation = {"status": link_status, "paths": [{"path": link,
                   "counter_value": value * 7, "backend_log": True}]}
    (run / "summary.json").write_text(json.dumps(summary))
    (run / "link_validation.json").write_text(json.dumps(validation))


def write_invalid(root: Path, directory: str, architecture: str, link: str) -> None:
    run = root / directory / "run"
    run.mkdir(parents=True, exist_ok=True)
    validation = {"status": "invalid_link", "reason": "path_validation_failed",
                  "paths": [{"path": link, "reason": "missing_positive_counter_evidence",
                             "backend_log": True, "counter_supported": True,
                             "counter_value": 0, "noise_threshold": 1}]}
    (run / "summary.json").write_text(json.dumps({"status": "invalid_link",
        "point": {"architecture": architecture}, "link_validation": validation}))
    (run / "link_validation.json").write_text(json.dumps(validation))


class CommAblationReportTests(unittest.TestCase):
    def make_inputs(self, root: Path) -> tuple[Path, Path]:
        qps2, smoke = root / "qps2", root / "smoke"
        for point in EXPECTED_VALID_POINTS:
            for repeat, value in enumerate((1.0, 2.0, 3.0), 1):
                write_run(qps2, point, repeat, value)
        write_invalid(smoke, "pd_nvlink_custom_pool", "pd", "nvlink")
        write_invalid(qps2, "pd_pcie", "pd", "pcie_host_staged")
        write_invalid(qps2, "af_pcie", "af", "pcie_host_staged")
        write_invalid(smoke, "af_rdma_nic_bound", "af", "rdma")
        return qps2, smoke

    def test_aggregates_three_valid_repeats_and_sample_std(self):
        with tempfile.TemporaryDirectory() as directory:
            report = build_report(*self.make_inputs(Path(directory)))
        native_pcie = next(row for row in report["aggregates"]
                           if row["architecture"] == "native" and row["link"] == "pcie")
        self.assertEqual(native_pcie["runs"], 3)
        self.assertEqual(native_pcie["metrics"]["qps"], {"mean": 2.0, "std": 1.0})
        self.assertEqual(native_pcie["metrics"]["ttft_ms_p99"]["mean"], 8.0)
        self.assertEqual(native_pcie["metrics"]["counter_pcie"]["mean"], 14.0)

    def test_filters_nonpassing_repeat_and_checks_completeness(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            qps2, smoke = self.make_inputs(root)
            write_run(qps2, "native_pcie", 2, 99, link_status="invalid_link")
            with self.assertRaisesRegex(ValueError, "native_pcie: 2/3"):
                build_report(qps2, smoke)
            partial = build_report(qps2, smoke, strict_repeats=False)
        self.assertFalse(partial["repeat_check"]["all_complete"])
        group = next(row for row in partial["aggregates"] if row["link"] == "pcie"
                     and row["architecture"] == "native")
        self.assertEqual(group["runs"], 2)

    def test_relative_values_require_same_architecture_nvlink(self):
        with tempfile.TemporaryDirectory() as directory:
            report = build_report(*self.make_inputs(Path(directory)))
        native_pcie = next(row for row in report["aggregates"]
                           if row["architecture"] == "native" and row["link"] == "pcie")
        pd_rdma = next(row for row in report["aggregates"]
                       if row["architecture"] == "pd")
        self.assertEqual(native_pcie["relative_to_architecture_nvlink"]["qps"], 1.0)
        self.assertIsNone(pd_rdma["relative_to_architecture_nvlink"])

    def test_invalid_evidence_and_outputs(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            report = build_report(*self.make_inputs(root))
            json_path, markdown_path = write_report(report, root / "report")
            saved = json.loads(json_path.read_text())
            markdown = markdown_path.read_text()
        self.assertEqual(len(saved["invalid_points"]), 4)
        self.assertEqual(saved["invalid_points"][0]["paths"][0]["counter"], 0)
        self.assertIn("missing_positive_counter_evidence", markdown)
        self.assertIn("native / nvlink", render_markdown(report))


if __name__ == "__main__":
    unittest.main()

#!/usr/bin/env python3
"""Small CPU-only self-test for the qwen3-af-v2 compatibility exporter."""
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from export_compat_matrix import ExportError, export_matrices


class ExportCompatMatrixTest(unittest.TestCase):
    def row(self, component: str, mode: str, *, phase: str = "prefill", seed: int = 7) -> dict:
        world = 2
        return {
            "schema_version": "qwen3-af-v2", "status": "ok", "phase": phase,
            "component": component, "parallel_mode": mode, "world_size": world,
            "attn_tp": world, "moe_tp": world if mode == "moe_tp" else 1,
            "moe_ep": world if mode == "moe_ep" else 1,
            "length": 128, "batch": 4, "freq_mhz": 690,
            "latency_us": 10.0 if component == "A" else (20.0 if mode == "moe_tp" else 30.0),
            "energy_total_mj": 1.0 if component == "A" else (2.0 if mode == "moe_tp" else 3.0),
            "request_seed": seed, "hidden_seed": seed + 1,
        }

    def write_rows(self, root: Path, rows: list[dict]) -> None:
        paths = [root / "node1" / "a.jsonl", root / "node3" / "f.jsonl"]
        split = (rows[::2], rows[1::2])
        for path, subset in zip(paths, split):
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("".join(json.dumps(row) + "\n" for row in subset))

    def test_tp_and_ep_tables(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "raw"
            out = Path(tmp) / "compat"
            self.write_rows(root, [
                self.row("A", "attn_tp"), self.row("F", "moe_tp"), self.row("F", "moe_ep"),
                self.row("A", "attn_tp", phase="decode"),
                self.row("F", "moe_tp", phase="decode"),
                self.row("F", "moe_ep", phase="decode"),
            ])
            summary = export_matrices(root, out, strict=True)
            self.assertEqual(summary.complete_rows, 4)
            tp = (out / "prefill_test_matrix_tp2.txt").read_text().splitlines()
            ep = (out / "prefill_test_matrix_ep2.txt").read_text().splitlines()
            self.assertEqual(tp[0], "tp\tinput_len\tgpu_clock\tbatch_size\tP_A_lat\tP_F_lat\tP_A_energy\tP_F_energy")
            self.assertEqual(ep[0], "ep\tinput_len\tgpu_clock\tbatch_size\tP_A_lat\tP_F_lat\tP_A_energy\tP_F_energy")
            self.assertEqual(tp[1], "2\t128\t690\t4\t10.0\t20.0\t1.0\t2.0")
            self.assertEqual(ep[1], "2\t128\t690\t4\t10.0\t30.0\t1.0\t3.0")
            decode_tp = (out / "decode_test_matrix_tp2.txt").read_text().splitlines()
            self.assertEqual(decode_tp[0], "tp\tcontext_len\tgpu_clock\tbatch_size\tD_A_lat\tD_F_lat\tD_A_energy\tD_F_energy")

    def test_seed_mismatch_is_always_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "raw"
            self.write_rows(root, [self.row("A", "attn_tp"), self.row("F", "moe_tp", seed=99)])
            with self.assertRaisesRegex(ExportError, "rejected"):
                export_matrices(root, Path(tmp) / "compat")


if __name__ == "__main__":
    unittest.main()

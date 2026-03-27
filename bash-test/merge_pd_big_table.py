#!/usr/bin/env python3
"""
Merge all per-run `processed/nvtx_PD_combined_stats.csv` under --work-dir into one long-format CSV
compatible with `pivot_pd_ops_wide.py` (same schema as batch_pd_nvtx_test.py full run).

This is a thin wrapper around the same logic as:
  python bash-test/batch_pd_nvtx_test.py --only-merge ...

Defaults: keep processed dirs (unlike batch_pd_nvtx_test --only-merge without --keep-processed).

Usage (from repo root):
  python bash-test/merge_pd_big_table.py
  python bash-test/merge_pd_big_table.py --work-dir bash-test/pd_batch_work \\
    --output-lens 64,256,512 --final-csv bash-test/pd_latency_big_table.csv

Then pivot:
  python bash-test/pivot_pd_ops_wide.py --csv bash-test/pd_latency_big_table.csv
"""
from __future__ import annotations

import argparse
import os
import sys

# Import merge helpers from batch script (same aggregation rules).
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if _SCRIPT_DIR not in sys.path:
    sys.path.insert(0, _SCRIPT_DIR)

from batch_pd_nvtx_test import _merge_existing_results  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Merge per-run nvtx_PD_combined_stats.csv into pd_latency_big_table format."
    )
    parser.add_argument(
        "--work-dir",
        type=str,
        default="bash-test/pd_batch_work",
        help="Directory containing tp*_in*_clk*/processed/nvtx_PD_combined_stats.csv",
    )
    parser.add_argument(
        "--output-lens",
        type=str,
        default="64,256,512",
        help="Comma-separated output lengths (must match your batch --output-lens).",
    )
    parser.add_argument(
        "--final-csv",
        type=str,
        default="bash-test/pd_latency_big_table.csv",
        help="Output long-format CSV path.",
    )
    parser.add_argument(
        "--delete-processed-after-merge",
        action="store_true",
        help=(
            "Remove each tp*_in*_clk* run dir after writing the big table "
            "(default: keep dirs; same as batch_pd_nvtx_test.py --only-merge without --keep-processed)."
        ),
    )
    args = parser.parse_args()

    repo = os.getcwd()
    work_dir = os.path.abspath(os.path.join(repo, args.work_dir))
    final_csv = os.path.abspath(os.path.join(repo, args.final_csv))
    output_lens = [int(x) for x in args.output_lens.split(",") if x.strip()]

    if not output_lens:
        print("[error] --output-lens must list at least one integer", file=sys.stderr)
        sys.exit(1)

    keep_processed = not args.delete_processed_after_merge

    out_path = _merge_existing_results(
        work_dir=work_dir,
        output_lens=output_lens,
        final_csv_path=final_csv,
        keep_processed=keep_processed,
    )
    print(f"[saved] {out_path}")
    print(f"[info] output_lens used: {output_lens}")
    try:
        rel_final = os.path.relpath(final_csv, repo)
    except ValueError:
        rel_final = final_csv
    print(f"[info] next: python bash-test/pivot_pd_ops_wide.py --csv {rel_final}")


if __name__ == "__main__":
    main()

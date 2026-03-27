#!/bin/bash
# Remove all generated intermediate logs from PD batch profiling.
cd "$(dirname "$0")"

rm -rf pd_batch_work pd_batch_work_sm
rm -rf pd_batch_logs pd_batch_logs_sm
rm -rf pd_batch_work_ut_* pd_batch_logs_ut_* pd_batch_work_ut_both pd_batch_logs_ut_both
rm -f nvtx_gpu_proj_trace*.csv nvtx_PD_combined_stats.csv
rm -rf pivot_pd_ops_out pivot_pd_ops_out_drop
rm -f pd_latency_big_table_ut_*.csv pd_latency_big_table_ut_both.csv
rm -f pd_ops_wide.csv pd_ops_D_wide.csv pd_ops_P_wide.csv
rm -f sglang.out.sqlite sglang.out.sqlite-*
rm -f test.log
rm -f test_ttft.log test_op.log test_both.log

echo "Cleaned intermediate logs."

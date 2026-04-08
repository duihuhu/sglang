#!/bin/bash
# Remove all generated intermediate logs from PD batch profiling.
cd "$(dirname "$0")"

rm -rf pd_batch_work pd_batch_work_ut_both
rm -rf pd_batch_logs pd_batch_logs_ut_both
rm pd_latency_big_table_ut_both.csv
rm test_both.log
rm -rf pivot_pd_ops_out

echo "Cleaned intermediate logs."

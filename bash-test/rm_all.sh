#!/bin/bash
# Remove all generated intermediate logs from PD batch profiling.
cd "$(dirname "$0")"
rm -rf pd_batch_work*
rm -rf pd_batch_logs*
rm pd_latency_big_table_ut_both.csv
rm test_both.log
rm -rf pivot_pd_ops_out*
rm -rf *.csv
rm -rf pd_batch_work_p_small
rm -rf pd_batch_logs_p_small
rm -rf pd_latency_big_table_p_small.csv
rm -rf *.log

echo "Cleaned intermediate logs."

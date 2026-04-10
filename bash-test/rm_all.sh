#!/bin/bash
# Remove all generated intermediate logs from PD batch profiling.
cd "$(dirname "$0")"
rm -rf pd_batch_work*
rm -rf pd_batch_logs*
rm pd_latency_big_table_ut_both.csv
rm test_both.log
rm -rf pivot_pd_ops_out*
rm -rf *.csv

echo "Cleaned intermediate logs."

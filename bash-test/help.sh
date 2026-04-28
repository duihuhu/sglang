
echo 'export ANTHROPIC_AUTH_TOKEN="sk-Ebz8qMvfWYi8ZzMD7XvBooIMzebJCbto8p3LrtXjjvTzMmO3"' >> ~/.bashrc
echo 'export ANTHROPIC_BASE_URL="https://api.aipaibox.com/"' >> ~/.bashrc
source ~/.bashrc


nohup bash bash-test/test.sh > bash-test/test_pd_driver.log 2>&1 &

nohup bash bash-test/test_small.sh > bash-test/test_pd_driver_small.log 2>&1 &

python3 bash-test/convert_pd_csv_to_big_table.py \
  --work-dir bash-test/pd_batch_work_p_small \
  --bench-stage P \
  --output-lens 1 \
  --final-csv bash-test/pd_latency_big_table_ut_both.csv
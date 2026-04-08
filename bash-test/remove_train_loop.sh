#!/usr/bin/env bash
# 循环 SIGKILL 命令行匹配 "train" 的进程。
# 注意：pkill -f train 会匹配到本脚本路径里的 "train"（kill_train_loop），
#       因此用 pgrep + 跳过当前 shell 的 $$，避免执行一次就把自己杀掉。
# Ctrl+C 退出。

INTERVAL="${INTERVAL:-1}"

while true; do
  mapfile -t _pids < <(pgrep -f train 2>/dev/null || true)
  for pid in "${_pids[@]}"; do
    [[ -z "$pid" ]] && continue
    [[ "$pid" -eq $$ ]] && continue
    kill -9 "$pid" 2>/dev/null || true
  done
  sleep "$INTERVAL"
done

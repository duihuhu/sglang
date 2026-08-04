# 32GPU 四节点 Node Scalability 测试进展

> 最后更新：2026-07-27 20:49（换设备交接）

## 集群

| 节点 | IP | hostname |
|------|-----|----------|
| node1 | 10.252.129.36 | d1n41a11g01（当前操作机） |
| node2 | 10.252.129.35 | d1n41a29g01 |
| node3 | 10.252.129.34 | d1n41a16g02 |
| node4 | 10.252.129.33 | d1n41a16g03 |

容器：`operator_test`（`--network host` / `--privileged` / `--gpus all`）  
代码挂载：宿主机 `/mnt/workspace/lt` → 容器 `/workspace`

## 测试配置

- **脚本**：`scripts/run_4node_scalability_benchmark.py`
- **方案**：sglang, dynamollm, distserve, biscale, aflex（跳过 MegaScale）
- **数据集 / QPS**：`code,conv` × `qps=32`
- **Workload**：`macro/data/workloads/macro_{code,conv}_qps32.jsonl`
- **锁频**：SGLang/DistServe 固定 1410MHz（`dvfs.py`）；DynamoLLM/BiScale/AFlex 用实例内 DVFS

### 部署口径

| 方案 | 部署 |
|------|------|
| sglang / dynamollm | 16×TP2，4 节点各 4 实例，round-robin router `:48000` |
| distserve / biscale | 2×[2P(TP4)+4D(TP2)]：(n1,n2) + (n3,n4)，顶层 router |
| aflex | 4×每节点 3P+1D（ipc_cpp），顶层 round-robin router |

---

## 数据路径

| 类型 | 路径（容器内） |
|------|----------------|
| **汇总结果** | `data/32gpu_six_schemes.json` |
| 结果备份 | `data/32gpu_six_schemes.json.bak.20260727_184226` |
| 主控日志 | `logs/run_4node_*.log` |
| 实例部署日志 | `logs/32g_four_node/{scheme}_*.log`（**各节点本地**，decode 日志在 node2/node4 需 ssh 查看） |

宿主机等价路径：`/mnt/workspace/lt/sglang/benchmark/AFlex_bench/multi_node/more_test/Ablation/node_scalibility/...`

---

## 结果状态（qps32）

| 方案 | code_qps32 | conv_qps32 | 备注 |
|------|-----------|-----------|------|
| **sglang** | ✅ PASS (486 tok/s, 164 mJ/tok) | ✅ PASS (1826 tok/s, 296 mJ/tok) | 完成 |
| **dynamollm** | ✅ PASS (454 tok/s, 158 mJ/tok) | ✅ PASS (1594 tok/s, 301 mJ/tok) | **已重跑**（修正锁频顺序） |
| **distserve** | ✅ PASS (474 tok/s, 137 mJ/tok) | 🔄 部署中 | code 在 remain3 跑通；conv 正在跑 |
| **biscale** | ❌ FAILED（旧 dvfs 路径错误） | ❌ FAILED | 待跑 |
| **aflex** | ❌ FAILED（旧 dvfs 路径错误） | ✅ PASS（旧数据，未 force 重跑） | 待跑 code |

JSON `meta.updated_at`: `2026-07-27T20:42:01`

---

## 当前运行中的任务

```
PID:  1445061
日志: logs/run_4node_remain3_20260727_204100.log
命令: python3 -u scripts/run_4node_scalability_benchmark.py \
        --force --schemes distserve,biscale,aflex \
        --datasets code,conv --qps 32
```

**20:47:50** 起正在部署 **distserve | conv | qps32**（distserve code 已于 20:47:40 完成并写入 JSON）。

查看进度：
```bash
tail -f logs/run_4node_remain3_20260727_204100.log
pgrep -af run_4node_scalability
```

---

## 续跑命令

进程若中断，按未完成项续跑（已 PASS 的会自动跳过，除非加 `--force`）：

```bash
# 续跑剩余 scheme（推荐）
nohup python3 -u scripts/run_4node_scalability_benchmark.py \
  --schemes distserve,biscale,aflex \
  --datasets code,conv --qps 32 \
  > logs/run_4node_continue_$(date +%Y%m%d_%H%M%S).log 2>&1 &

# 强制重跑某一方案
nohup python3 -u scripts/run_4node_scalability_benchmark.py \
  --force --schemes biscale,aflex \
  --datasets code,conv --qps 32 \
  > logs/run_4node_force_$(date +%Y%m%d_%H%M%S).log 2>&1 &
```

跑前清理四节点（宿主机级）：
```bash
for ip in 10.252.129.36 10.252.129.35 10.252.129.34 10.252.129.33; do
  ssh $ip "bash -s" <<'EOF'
for pid in $(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null); do kill -9 $pid; done
pgrep -f 'sglang.launch_server' | xargs -r kill -9
pgrep -f 'sglang::scheduler' | xargs -r kill -9
pgrep -f 'launch_router' | xargs -r kill -9
docker exec operator_test bash -lc 'bash /workspace/sglang/benchmark/AFlex_bench/multi_node/scripts/cleanup_node.sh'
EOF
done
```

---

## 脚本已修复的问题（`run_4node_scalability_benchmark.py`）

1. **node1 清理误报 `remaining=-1`**：状态输出改到 `docker exec cleanup` 之前；SSH 输出丢失时用独立 `nvidia-smi` 校验。
2. **TP2 端口冲突（node4 GPU67 / 53350）**：每实例启动前 `kill_ports`；`wait_health_with_log` 快速失败；失败重试。
3. **DynamoLLM 锁频顺序错误**：部署前不锁频 → 16 实例 healthy 后统一锁 1410MHz baseline（对齐 macro benchmark）。
4. **DistServe PD 部署**：
   - Prefill/Decode 启动前端口预清理（含 bootstrap/nccl）。
   - Sub-router 端口：instance0→48001，instance1→48002。
   - **Prometheus 端口冲突**：默认 29000 导致同节点 sub-router + top-router 崩溃；已为每个 router 分配独立 `--prometheus-port`（29100+offset）。

同步代码到 node2–4：
```bash
for ip in 10.252.129.35 10.252.129.34 10.252.129.33; do
  rsync -az /mnt/workspace/lt/sglang/benchmark/AFlex_bench/ \
    $ip:/mnt/workspace/lt/sglang/benchmark/AFlex_bench/
  rsync -az /mnt/workspace/lt/sglang/python/sglang/ \
    $ip:/mnt/workspace/lt/sglang/python/sglang/
done
```

---

## 历史日志索引

| 日志 | 内容 |
|------|------|
| `run_4node_20260727_192224.log` | 首轮全量（sglang PASS；dynamollm 锁频有误） |
| `run_4node_dynamollm_rerun_20260727_194429.log` | DynamoLLM 修正后重跑 ✅ |
| `run_4node_remain_20260727_202155.log` | distserve 首次续跑（DEPLOY_FAILED） |
| `run_4node_remain2_20260727_203229.log` | 修 PD 端口后（仍 Prometheus 冲突） |
| `run_4node_remain3_20260727_204100.log` | **当前**：distserve code ✅，conv 进行中 |

---

## 待办

- [ ] 等 remain3 跑完：distserve conv → biscale code/conv → aflex code
- [ ] 确认 `32gpu_six_schemes.json` 五项 scheme 的 `code_qps32` / `conv_qps32` 均为 PASS
- [ ] 可选：aflex conv 用 `--force` 重跑（当前为旧 PASS）
- [ ] 画图：`charts/plot_node_scalability_energy.py`（如有）

---

## node2 他人任务说明

node2 曾有 latticekv 容器占 CPU（不占 GPU）。清理时勿误杀他人长期任务，仅清 benchmark 相关端口/GPU 进程。

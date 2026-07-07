# Multi-Node 环境检查记录

检查时间：2026-07-01（从 node1 宿主机 `10.252.129.36` 发起）

四节点概览：

| 节点 | IP | 主机名 | 角色/备注 |
|---|---|---|---|
| node1 | 10.252.129.36 | d1n41a11g01 | 编排入口，Unison 同步源 |
| node2 | 10.252.129.35 | d1n41a29g01 | Unison 同步目标，Decode 侧（16 卡基准） |
| node3 | 10.252.129.34 | d1n41a16g02 | 新节点，共享机器，**未接入同步** |
| node4 | 10.252.129.33 | d1n41a16g03 | 新节点，容器曾停止，**代码过期** |

---

## 1. 各节点状态摘要

| 项 | node1 | node2 | node3 | node4 |
|---|---|---|---|---|
| 容器 `operator_test` | Up | Up | Up | Exited→已手动 start |
| 容器镜像 | `cuda:12.2.0-devel` | `operator_test:latest` | `cuda:12.2.0-devel` | `operator_test:latest` |
| 8× A800 | ✅ | ✅ | ✅（空闲） | ✅ |
| `import sglang`（系统 python） | ✅ | ✅ | ✅ | ❌（仅 venv 可用） |
| mooncake / pynvml | ✅ | ✅ | ✅ | venv 内 ✅ |
| 容器内 docker | 无 | 无 | 无 | 无 |
| 容器内 ssh | ✅ | ✅ | ❌ 未安装 | ✅（无密钥） |
| Qwen3-32B | `/models/Qwen3-32B` | ✅ | ✅ | `/models/Qwen/Qwen3-32B` ⚠️ |
| Mixtral-8x7B | ✅ | ✅ | ❌ | `/models/Mixtral/Mixtral-8x7B` |
| Unison 同步 | 源 | 目标 | ❌ 未接入 | ❌ 未接入 |
| git HEAD | `d6e3b5280` (v15) | `8ea5b10e5` (v14) | 无 `.git` | `8ea5b10e5` (v14) |
| `multi_node/` 目录 | ✅ | ✅ | 过期快照 | ❌ 不存在 |
| workspace 磁盘 | 2% | 1% | **97%** 🔴 | 1% |
| dpser 目录 | 无 | 无 | **1.2T** | 无 |

---

## 2. node1（10.252.129.36）

### 正常项

- 容器 host 网络，`/mnt/workspace/lt` → `/workspace`，`/mnt/data/models` → `/models`
- 可免密 `ssh node2`；容器内可 `ssh node2`
- 宿主机编排依赖齐全：`aiohttp numpy requests pynvml`
- Unison 同步守护进程运行中（日志在 `/tmp/sync_sglang.log`）
- 跨节点 RoCE：`mlx5_bond_0` / `bond0` Up

### 问题

- **根分区 `/dev/sda2` 92%→100% 满**（约剩 8.5G→0），可能导致本地写文件失败、Unison 存档失败
  - `/root/.unison/unison.log` 约 195MB，可清理
- git 比 node2 多 1 个 commit（`d6e3b5280` version15），`.git` 被 Unison 排除，node2 不会自动跟上

---

## 3. node2（10.252.129.35）

### 正常项

- 容器、挂载、sglang/mooncake/GPU 均正常
- Unison 目标端，node1 变更约 **2–5 秒**后可见
- 可反向 SSH 到 node1
- 当前有 Mixtral PDAF 实验进程在跑（`node_scalibility`）

### 问题

- 容器镜像标签与 node1 不同（`operator_test:latest` vs `cuda:12.2.0-devel`），功能上目前一致
- git HEAD 落后 node1 一个 commit（Unison 不同步 `.git`）
- 宿主机缺 `aiohttp`/`numpy`（编排只在 node1 跑则不影响；NVML 可用）
- 曾出现 Unison `No space left on device`（保存 archive 失败），需关注

---

## 4. node3（10.252.129.34）— 问题最多

### 正常项

- SSH 可达，容器 Up，8× GPU，bond0 Up
- Qwen3-32B 模型存在
- 系统 python 可 `import sglang` / mooncake

### 磁盘布局（与 node1/node2 不同）

```
/dev/md0 (RAID10, 两块 3.5T NVMe) 挂载为三个入口：
  /ssd           ← md0 根（可见全部数据）
  /mnt/workspace ← md0[/mnt_workspace] 子目录挂载（只能看到 lt/）
  /mnt/data      ← md0[/mnt_data] 子目录挂载（只能看到 models/）
```

`du /mnt/workspace/lt` 只看到 **1.2T**，但 `df` 显示 **3.2T/3.5T（97%）**，因为其余空间在 `/ssd` 下：

| 路径 | 大小 | 说明 |
|---|---|---|
| `/ssd/openpi/` | ~1.8T | 其他项目（机器人训练） |
| `/mnt/workspace/lt/dpser/sglang_runs/` | ~1.2T | HiCache 多模型 benchmark 产物 |
| `/ssd/Models/`、`LlamaFactory/` 等 | ~150G+ | 其他项目 |

### dpser 为何在 `lt/` 下

- **不是 mount**，是普通目录，**2026-06-29** 由他人实验创建
- 实验脚本硬编码 `/mnt/workspace/lt/dpser/sglang_runs/...`，容器内对应 `/workspace/dpser/...`
- 内容为 sglang HiCache file backend + multimodel elastic/static benchmark 日志与 cache
- node1/node2 **没有** dpser；node3 是共享机器，多人共用

### 其他问题

- **未接入 Unison**：node1 写入后 node3 **不同步**（实测 SYNC_FAIL）
- **无 `.git`**，sglang 代码停留在 **Jun 28** 左右
- 容器内 **无 ssh**；node3 宿主机 **无法** SSH 到 node1/node2（publickey denied）
- 无 Mixtral 模型
- 容器内有 **60+ 僵尸 sglang 进程**（Jun 28 遗留），GPU 当前空闲，建议 `docker restart operator_test`
- `/mnt/workspace/lt` 与 `/ssd/mnt_workspace/lt` 是**同一 inode**（同一份数据，非重复占用）

---

## 5. node4（10.252.129.33）

### 正常项

- SSH 可达；node4 宿主机可 SSH 到 node1/node2
- 磁盘充足：workspace 1%、data 8%，**无 dpser**
- 磁盘布局与 node1/node2 类似（独立 nvme0/nvme1，非 RAID10）
- bond0 Up，8× GPU
- Qwen3-32B + Mixtral-8x7B 均存在

### 问题

- 容器 **`operator_test` 已 Exited(255) 5 天**（2026-06-26 停止），检查时已手动 `docker start`
- **sglang 代码过期**（~Jun 19），git 在 `8ea5b10e5` (version14)，**缺少 `multi_node/`**（version15 才引入）
- **系统 python 无 sglang/mooncake**；需用 venv：
  ```bash
  /workspace/env/sglang-test/bin/python -c "import sglang"
  ```
- **模型路径与 node1 不一致**：

  | node1/node2 | node4 |
  |---|---|
  | `/models/Qwen3-32B` | `/models/Qwen/Qwen3-32B` |
  | `/models/Mixtral-8x7B` | `/models/Mixtral/Mixtral-8x7B` |

  benchmark 脚本写死路径会报错，需 symlink 或改参数。

- 未接入 Unison；无 `sync_sglang.sh`
- 容器内 ssh 无密钥，无法连 node1/node2
- 宿主机缺 `aiohttp`/`numpy`

---

## 6. 代码同步机制（Unison）

配置文件：`/root/.unison/sglang.prf`（node1）

```
root = /mnt/workspace/lt/sglang
root = ssh://root@10.252.129.35//mnt/workspace/lt/sglang
```

守护进程：`/mnt/workspace/lt/sync_sglang.sh`（inotify + 2s debounce + unison）

- **仅 node1 → node2**，node3/node4 不在链路中
- **排除 `.git`**，故 git HEAD 与 working tree 状态各节点可能分叉
- 日志：**`/tmp/sync_sglang.log`**（不是 `/mnt/workspace/lt/sync_sglang.log`）

---

## 7. 待办（纳入新节点前）

### 紧急

- [ ] **node1 根分区清理**（当前 100% 满，影响本地操作与 Unison）
- [ ] **node3 磁盘**：确认 `/mnt/workspace/lt/dpser/sglang_runs`（1.2T）可否删除或迁走；`/ssd/openpi`（1.8T）属其他项目

### node3 可用化

- [ ] 将 node3 加入 Unison profile，或手动 rsync 一次完整 sglang
- [ ] 补 `.git` 或定期从 node1 pull
- [ ] 容器安装 `openssh-client`；配置 node3 宿主机 SSH 密钥
- [ ] `docker restart operator_test` 清理僵尸进程
- [ ] 如需 Mixtral：同步模型到 `/mnt/data/models/`

### node4 可用化

- [ ] `git pull` 到 version15+（含 `multi_node/`），或 rsync 最新 sglang
- [ ] 统一 Python：容器内 `pip install -e python/.` 或脚本改用 venv python
- [ ] 模型路径 symlink：
  ```bash
  ln -s /models/Qwen/Qwen3-32B /models/Qwen3-32B
  ln -s /models/Mixtral/Mixtral-8x7B /models/Mixtral-8x7B
  ```
- [ ] 接入 Unison（可选）；确认容器 restart 后稳定运行
- [ ] 容器内配置 SSH 密钥（若需容器内互访）

### 长期

- [ ] 统一四节点 Docker 镜像标签
- [ ] 明确 node3 共享策略，避免他人实验写入 `lt/dpser`
- [ ] 更新 `README.md` 环境变量：增加 `MN_NODE3_IP` / `MN_NODE4_IP`

---

## 8. 快速自检命令

在 node1 宿主机执行：

```bash
# 同步守护
pgrep -af sync_sglang && tail -3 /tmp/sync_sglang.log

# 四节点 SSH
for ip in 36 35 34 33; do echo -n "10.252.129.$ip: "; ssh -o BatchMode=yes 10.252.129.$ip hostname; done

# 容器健康
for ip in 36 35 34 33; do
  echo "=== $ip ==="
  ssh 10.252.129.$ip 'docker ps --filter name=operator_test --format "{{.Status}}"; docker exec operator_test python3 -c "import sglang; print(sglang.__file__)" 2>&1'
done

# 代码一致性
md5sum /mnt/workspace/lt/sglang/benchmark/AFlex_bench/multi_node/README.md
for ip in 35 34 33; do
  ssh 10.252.129.$ip "md5sum /mnt/workspace/lt/sglang/benchmark/AFlex_bench/multi_node/README.md 2>&1"
done

# 磁盘
for ip in 36 35 34 33; do
  echo "=== $ip ==="
  ssh 10.252.129.$ip 'df -h /mnt/workspace /mnt/data / 2>/dev/null | grep -v Filesystem'
done
```

# Attn(DP|TP) × MoE(TP|EP) 四宫格压测报告

## 环境与 workload

| 项目 | 配置 |
|------|------|
| GPU | 8× A800（并行时各用 4 卡） |
| 模型 | Qwen3-30B-A3B（128 experts, top-k=8, BF16） |
| MoE runner | triton |
| CUDA Graph | 关闭（`--disable-cuda-graph`） |
| 显存 | `--mem-fraction-static 0.85` |
| workload | random-ids, in=32, out=256, seed=42, warmup=0 |

并发阶梯：512 / 1024 / 2048 / 4096 / 6144

## 四种部署配置

| 名称 | Attention | MoE | 启动参数 |
|------|-----------|-----|----------|
| **Attn TP + MoE TP** | TP4 | TP（无 EP） | `--tp-size 4 --dp-size 1 --ep-size 1` |
| **Attn TP + MoE EP** | TP4 | EP4 | `--tp-size 4 --dp-size 1 --ep-size 4` |
| **Attn DP + MoE TP** | DP4 | TP（无 EP） | `--tp-size 4 --dp-size 4 --ep-size 1 --enable-dp-attention` |
| **Attn DP + MoE EP** | DP4 | EP4 | `--tp-size 4 --dp-size 4 --ep-size 4 --enable-dp-attention` |

KV / 调度特征：

| 配置 | KV pool | scheduler 数 | max_running/worker |
|------|---------|-------------|-------------------|
| Attn TP + MoE * | 统一 ~2.28M tokens | **1** | 4096 |
| Attn DP + MoE * | 4× ~551K tokens | **4** | 4096/rank |

## 完整结果表

### 输出吞吐 (tok/s)

| C | Attn TP<br>MoE TP | Attn TP<br>MoE EP | Attn DP<br>MoE TP | Attn DP<br>MoE EP |
|---:|---:|---:|---:|---:|
| 512 | 6,144 | 5,914 | **6,364** | 6,132 |
| 1024 | 9,824 | 9,655 | **11,451** | 11,162 |
| 2048 | 11,546 | 11,452 | 12,865 | **13,009** |
| 4096 | 10,395 | 10,524 | 11,100 | **11,181** |
| 6144 | **10,724** | 10,478 | 10,429 | 10,334 |

### TTFT mean (ms)

| C | Attn TP<br>MoE TP | Attn TP<br>MoE EP | Attn DP<br>MoE TP | Attn DP<br>MoE EP |
|---:|---:|---:|---:|---:|
| 512 | 876 | 922 | **679** | 737 |
| 1024 | 1,631 | 1,804 | 1,731 | **1,689** |
| 2048 | 2,821 | 2,614 | **2,189** | **2,189** |
| 4096 | 4,881 | 4,878 | 5,060 | **5,034** |
| 6144 | **39,427** | **40,719** | **7,413** | **7,522** |

### TTFT p99 (ms)

| C | Attn TP<br>MoE TP | Attn TP<br>MoE EP | Attn DP<br>MoE TP | Attn DP<br>MoE EP |
|---:|---:|---:|---:|---:|
| 512 | 1,085 | 1,154 | **900** | 974 |
| 1024 | **1,786** | 2,208 | 2,525 | 2,503 |
| 2048 | 3,752 | 2,886 | **2,512** | **2,519** |
| 4096 | 5,975 | 5,952 | **5,517** | 5,776 |
| 6144 | **105,699** | **110,038** | **8,153** | **8,762** |

### TPOT mean (ms)

| C | Attn TP<br>MoE TP | Attn TP<br>MoE EP | Attn DP<br>MoE TP | Attn DP<br>MoE EP |
|---:|---:|---:|---:|---:|
| 512 | 79.6 | 82.8 | **77.5** | 80.2 |
| 1024 | 96.8 | 98.1 | **81.6** | 84.2 |
| 2048 | 164.0 | 166.1 | 148.2 | **146.5** |
| 4096 | 368.8 | 365.0 | 342.3 | **340.7** |
| 6144 | **307.5** | 317.3 | 549.1 | 554.3 |

## 因子分解（控制变量）

### 固定 Attn=TP，切换 MoE（TP → EP）

| C | Δ吞吐 (EP−TP) | ΔTTFT p99 | 结论 |
|---:|---:|---:|------|
| 512 | −3.7% | +6% | 基本持平 |
| 1024 | −1.7% | +24% | 基本持平 |
| 2048 | −0.8% | −23% | EP 略优 |
| 4096 | +1.2% | −0.4% | 持平 |
| **6144** | −2.3% | **+4%** | **TTFT 尾部同样崩溃（~110s）** |

→ **MoE EP 不改变 Attn TP 的调度瓶颈。**

### 固定 Attn=DP，切换 MoE（TP → EP）

| C | Δ吞吐 (EP−TP) | ΔTTFT p99 | 结论 |
|---:|---:|---:|------|
| 512 | −3.6% | +8% | 持平 |
| 1024 | −2.5% | −1% | 持平 |
| 2048 | +1.1% | +0.3% | 持平 |
| 4096 | +0.7% | +5% | 持平 |
| **6144** | −0.9% | **+7%** | **TTFT 均健康（~8s）** |

→ **Attn DP 下，MoE EP 仅带来小幅吞吐差异，TTFT 行为几乎相同。**

### 固定 MoE=TP，切换 Attn（TP → DP）

| C | Δ吞吐 | ΔTTFT p99 |
|---:|---:|---:|
| 2048 | +11% | −33% |
| 6144 | −3% | **−92%**（106s → 8s） |

### 固定 MoE=EP，切换 Attn（TP → DP）

| C | Δ吞吐 | ΔTTFT p99 |
|---:|---:|---:|
| 2048 | +14% | −13% |
| 6144 | −1% | **−92%**（110s → 9s） |

→ **C=6144 的 TTFT 改善完全来自 Attn DP，与 MoE 无关。**

## 核心结论

1. **四宫格中，Attn 是调度维度，MoE 是算力维度**
   - Attn TP：单 scheduler，C>4096 时 2048 请求排队，p99 TTFT ~105–110s
   - Attn DP：4 scheduler 并行，C=6144 时 p99 TTFT ~8s

2. **MoE EP 的价值在中高并发吞吐（C=1024–4096）**
   - 在 Attn DP 下，MoE EP 峰值吞吐 C=2048 达 **13,009 tok/s**（四配置最高）
   - 在 Attn TP 下，MoE EP 与 MoE TP 差异 <3%

3. **新发现的第四种组合 Attn DP + MoE TP**
   - 调度行为与 Attn DP + MoE EP 几乎一致（C=6144：p99 8153 vs 8762）
   - 说明 **不需要 MoE EP 也能获得 Attn DP 的调度收益**
   - 但 C=2048 吞吐略低于 MoE EP（12865 vs 13009）

4. **C=6144 时 Attn DP 的 TPOT 更高（~550ms vs ~310ms）**
   - 因每 rank batch 更小；Attn TP 单 rank decode batch 更大、单 token 更快
   - 但 Attn TP 的 TTFT 尾部灾难远超 TPOT 优势

## 数据路径

- 默认输出：`benchmark/moe-energy/benchmark/data/attn_moe_matrix/`
- 脚本：`benchmark/moe-energy/benchmark/scripts/run_attn_moe_matrix_parallel.sh`
- 汇总：`benchmark/moe-energy/benchmark/scripts/summarize_attn_moe_matrix.py`

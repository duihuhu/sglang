# 旧 EP 强制路由分析：balanced vs skewed_rank0

本文档对比旧 EP（`moe_a2a_backend=none`）在两种强制路由下的延迟与能耗：

- `balanced`：token 在 EP rank 间均匀分配。
- `skewed_rank0`：所有 token 的 top-k expert 落在 rank0。

数据来自 `data/EP/`：

- Prefill：`PF-balanced.txt`、`PF-skewed.txt`
- Decode：`DF-balanced.txt`、`DF-skewed.txt`

测试矩阵与缺失点见 [EP-test.md](EP-test.md)。

## 1. 分析方法

不再跨维度求平均或中位数。分析包含四个自变量：

1. EP size（world size）
2. input/context length
3. batch size
4. GPU frequency

每张图只改变其中一个维度，另外三个维度固定。当前采用的代表性基准为：

```text
EP size = 4
length = 512
batch = 32
GPU frequency = 930 MHz
```

例如展示 batch 影响时，仅改变 batch，同时严格固定：

```text
EP size=4, length=512, frequency=930 MHz
```

图中不包含跨 frequency、EP size、length 或 batch 的聚合。balanced 与 skewed 使用完全相同的 `(EP, length, batch, frequency)` 测点直接比较。

以下比值用于描述差异：

```text
latency ratio = skewed latency / balanced latency
energy ratio  = skewed energy / balanced energy
```

- 比值 `> 1`：balanced 更好。
- 比值 `< 1`：skewed 更好。

## 2. batch 维度

固定：`EP=4, length=512, frequency=930 MHz`

![固定其他维度，仅改变 batch](../fig/ep_fixed_vary_batch.png)

### Prefill

- batch=1 时接近持平：延迟比 0.94，能耗比 0.75。
- batch 从 2 增大到 32 时，skewed 的延迟惩罚迅速上升。
- batch=32 时，延迟比为 **2.96×**，能耗比为 **1.77×**。
- batch≥32 后，延迟比约为 2.6~2.9×，能耗比约为 1.5~1.8×。

因此在这个固定切片上，balanced 的延迟改善明显大于能耗改善，变化并不成比例。

### Decode

- batch≤512 时 skewed 延迟约为 balanced 的 0.88~0.92×，能耗也通常更低。
- batch=1024 时发生反转：延迟比为 **1.36×**，能耗比为 **1.09×**，balanced 开始占优。
- 该切片没有 ws2 缺失的大 batch 点参与，不进行跨 EP 聚合。

## 3. EP size 维度

固定：`length=512, batch=32, frequency=930 MHz`

![固定其他维度，仅改变 EP size](../fig/ep_fixed_vary_ws.png)

### Prefill

| EP size | 延迟比 skew/bal | 能耗比 skew/bal |
|---------|-----------------|-----------------|
| 2 | 1.82× | 1.26× |
| 4 | 2.96× | 1.77× |
| 8 | 4.20× | 2.36× |

EP size 越大，skewed 的 rank0 straggler 越严重。延迟差距增长速度大于能耗差距。

### Decode

| EP size | 延迟比 skew/bal | 能耗比 skew/bal |
|---------|-----------------|-----------------|
| 2 | 0.92× | 0.70× |
| 4 | 0.90× | 0.78× |
| 8 | 0.91× | 0.70× |

在固定 `length=512, batch=32, freq=930` 下，三个 EP size 都是 skewed 更快、更省电，且随 EP size 变化不大。

## 4. length 维度

固定：`EP=4, batch=32, frequency=930 MHz`

![固定其他维度，仅改变 length](../fig/ep_fixed_vary_length.png)

### Prefill

- length=64：延迟比 2.05×，能耗比 1.24×。
- length=512：延迟比 2.96×，能耗比 1.77×。
- length=4096：延迟比 2.71×，能耗比 1.65×。

延迟和能耗均随 length 增长，但 skewed 延迟始终比 balanced 高约 2~3×；能耗差距仅约 1.2~1.8×。

### Decode

- 全部 length 上延迟比稳定在 0.84~0.92×。
- 能耗比约为 0.65~0.83×。

在该固定切片上，Decode 对 context length 不敏感，skewed 始终略快且更省电。

## 5. GPU frequency 维度

固定：`EP=4, length=512, batch=32`

![固定其他维度，仅改变 GPU frequency](../fig/ep_fixed_vary_freq.png)

这是此前分析缺失、但必须独立控制的维度。frequency 对延迟和能耗都有显著影响。

### Prefill

| Frequency | 延迟比 skew/bal | 能耗比 skew/bal |
|-----------|-----------------|-----------------|
| 210 MHz | 3.16× | 2.36× |
| 930 MHz | 2.96× | 1.77× |
| 1410 MHz | 2.79× | 1.50× |

- 提高频率显著降低两种路由的延迟。
- 能耗并非单调下降：balanced 和 skewed 均在中高频附近出现能耗低点，1410 MHz 又回升。
- 随频率提高，skewed/balanced 的能耗比从 2.36× 降至 1.50×，说明把不同频率混合聚合会掩盖 routing tradeoff。

### Decode

- 延迟比在各频率下约为 0.88~0.93×。
- 能耗比约为 0.75~0.80×。
- 两种路由的 Decode 能耗均呈现中频较低、最高频回升的趋势。

## 6. 固定切片下的核心发现

1. **Prefill 的 routing imbalance 结论仍成立。**  
   在所有四个固定切片中，除极小 batch 外，balanced 均显著降低延迟和能耗。

2. **延迟改善与能耗改善不成比例。**  
   例如固定 `EP=4, length=512, batch=32, freq=930`：

   ```text
   skew/bal latency ratio = 2.96×
   skew/bal energy ratio  = 1.77×
   ```

   从 skewed 切换到 balanced，相当于延迟降低约 66%，能耗降低约 44%。延迟收益明显更大。

3. **EP size 会放大 Prefill imbalance。**  
   固定其他条件时，延迟比从 EP2 的 1.82× 增至 EP8 的 4.20×。

4. **Decode 的主导因素是 batch。**  
   在当前基准切片中，batch≤512 时 skewed 更优，batch=1024 时 balanced 反超。EP、length、frequency 变化时路由差距相对稳定。

5. **frequency 必须严格控制。**  
   频率会同时改变延迟、绝对能耗和 balanced/skewed 的相对能耗比；不能将不同频率直接混合后推导单一维度规律。

## 7. 图表与重生成

| 文件 | 变化维度 | 固定维度 |
|------|----------|----------|
| `ep_fixed_vary_batch.png` | batch | EP=4, length=512, freq=930 |
| `ep_fixed_vary_ws.png` | EP size | length=512, batch=32, freq=930 |
| `ep_fixed_vary_length.png` | length | EP=4, batch=32, freq=930 |
| `ep_fixed_vary_freq.png` | frequency | EP=4, length=512, batch=32 |

每张图均包含：

- Prefill latency：balanced vs skewed
- Prefill energy：balanced vs skewed
- Decode latency：balanced vs skewed
- Decode energy：balanced vs skewed

重生成：

```bash
cd /mnt/workspace/lt/sglang-source/sglang
python3 benchmark/moe-energy/profiling/scripts/plot_ep_bal_vs_skew.py
```

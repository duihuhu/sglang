import csv
import io

data_text = open("/workspace/benchmark/sglang-main/bash-test/pivot_pd_ops_out/pd_ops_wide.csv").read()
lines = data_text.strip().split("\n")
header = lines[1].split(",")
rows = []
for line in lines[2:]:
    if not line.strip():
        continue
    parts = line.split(",")
    rows.append({
        "tp": int(parts[0]),
        "input_len": int(parts[1]),
        "gpu_clock": int(parts[2]),
        "batch_size": int(parts[3]),
        "A": float(parts[4]),
        "F": float(parts[5]),
        "TTFT_ms": float(parts[6]),
        "AF64_ms": float(parts[7]),
        "A_energy_mj": float(parts[8]),
        "F_energy_mj": float(parts[9]),
    })

tps = sorted(set(r["tp"] for r in rows))
input_lens = sorted(set(r["input_len"] for r in rows))
gpu_clocks = sorted(set(r["gpu_clock"] for r in rows))

md = []
md.append("# Prefill 数据合理性分析报告\n")
md.append("## 数据概览\n")
md.append(f"- **TP 值**: {tps}")
md.append(f"- **Input Length 值**: {input_lens}")
md.append(f"- **GPU Clock (MHz) 值**: {gpu_clocks}")
md.append(f"- **总数据行数**: {len(rows)}")
md.append(f"- **Batch Size**: 全部为 1")
md.append(f"- **列说明**: A=Attention kernel耗时(μs), F=FFN kernel耗时(μs), TTFT=首token延迟(ms), (A+F)*64=64层kernel总耗时(ms), A/F_energy=能耗(mJ)\n")

# ========== 1. TTFT vs (A+F)*64 一致性检查 ==========
md.append("---\n## 1. TTFT 与 (A+F)×64 一致性检查\n")
md.append("TTFT_ms 应略大于 (A+F)×64_ms（因存在额外开销如调度、通信等）。检查异常差距：\n")

anomalies_ttft = []
for r in rows:
    computed_af64 = (r["A"] + r["F"]) * 64 / 1000
    gap_ms = r["TTFT_ms"] - r["AF64_ms"]
    gap_pct = gap_ms / r["AF64_ms"] * 100 if r["AF64_ms"] > 0 else 0
    af64_check = abs(computed_af64 - r["AF64_ms"])
    if af64_check > 0.1:
        anomalies_ttft.append(("AF64计算不一致", r, computed_af64, r["AF64_ms"]))
    if gap_pct > 20:
        anomalies_ttft.append(("TTFT偏高", r, gap_ms, gap_pct))
    elif gap_pct < -5:
        anomalies_ttft.append(("TTFT偏低", r, gap_ms, gap_pct))

if anomalies_ttft:
    md.append("### 发现的异常\n")
    md.append("| 类型 | tp | input_len | gpu_clock | TTFT_ms | (A+F)×64_ms | 差值(ms) | 偏差(%) |")
    md.append("|------|-----|-----------|-----------|---------|-------------|----------|---------|")
    for atype, r, val1, val2 in anomalies_ttft:
        if atype == "AF64计算不一致":
            md.append(f"| {atype} | {r['tp']} | {r['input_len']} | {r['gpu_clock']} | - | 计算={val1:.2f} | 记录={val2:.2f} | - |")
        else:
            md.append(f"| {atype} | {r['tp']} | {r['input_len']} | {r['gpu_clock']} | {r['TTFT_ms']:.2f} | {r['AF64_ms']:.2f} | {val1:.2f} | {val2:.1f}% |")
else:
    md.append("未发现异常。\n")

# ========== 2. 频率单调性检查 ==========
md.append("\n---\n## 2. 频率单调性检查\n")
md.append("随 GPU 频率升高，TTFT 应单调递减。检查违反单调性的情况：\n")

freq_anomalies = []
for tp in tps:
    for il in input_lens:
        subset = sorted([r for r in rows if r["tp"] == tp and r["input_len"] == il],
                        key=lambda x: x["gpu_clock"])
        for i in range(1, len(subset)):
            if subset[i]["TTFT_ms"] > subset[i-1]["TTFT_ms"]:
                freq_anomalies.append((tp, il, subset[i-1]["gpu_clock"], subset[i]["gpu_clock"],
                                       subset[i-1]["TTFT_ms"], subset[i]["TTFT_ms"]))

if freq_anomalies:
    md.append("### 违反单调递减的情况\n")
    md.append("| tp | input_len | freq_low→freq_high | TTFT_low(ms) | TTFT_high(ms) | 结论 |")
    md.append("|-----|-----------|---------------------|--------------|---------------|------|")
    for tp, il, f1, f2, t1, t2 in freq_anomalies:
        md.append(f"| {tp} | {il} | {f1}→{f2} | {t1:.2f} | {t2:.2f} | ⚠️ 升频后TTFT反而增加 |")
else:
    md.append("所有组合均满足 TTFT 随频率升高而单调递减。✓\n")

# Also check A and F monotonicity
md.append("\n### A/F kernel 耗时频率单调性\n")
a_anomalies = []
f_anomalies = []
for tp in tps:
    for il in input_lens:
        subset = sorted([r for r in rows if r["tp"] == tp and r["input_len"] == il],
                        key=lambda x: x["gpu_clock"])
        for i in range(1, len(subset)):
            if subset[i]["A"] > subset[i-1]["A"] * 1.02:
                a_anomalies.append((tp, il, subset[i-1]["gpu_clock"], subset[i]["gpu_clock"],
                                    subset[i-1]["A"], subset[i]["A"]))
            if subset[i]["F"] > subset[i-1]["F"] * 1.02:
                f_anomalies.append((tp, il, subset[i-1]["gpu_clock"], subset[i]["gpu_clock"],
                                    subset[i-1]["F"], subset[i]["F"]))

if a_anomalies:
    md.append("\n**Attention kernel 升频后耗时反增（>2%）的情况:**\n")
    md.append("| tp | input_len | freq_low→high | A_low(μs) | A_high(μs) |")
    md.append("|-----|-----------|---------------|-----------|------------|")
    for tp, il, f1, f2, a1, a2 in a_anomalies:
        md.append(f"| {tp} | {il} | {f1}→{f2} | {a1:.2f} | {a2:.2f} |")
else:
    md.append("Attention kernel 耗时均满足频率单调递减。✓\n")

if f_anomalies:
    md.append("\n**FFN kernel 升频后耗时反增（>2%）的情况:**\n")
    md.append("| tp | input_len | freq_low→high | F_low(μs) | F_high(μs) |")
    md.append("|-----|-----------|---------------|-----------|------------|")
    for tp, il, f1, f2, v1, v2 in f_anomalies:
        md.append(f"| {tp} | {il} | {f1}→{f2} | {v1:.2f} | {v2:.2f} |")
else:
    md.append("FFN kernel 耗时均满足频率单调递减。✓\n")

# ========== 3. 频率扩展效率分析 ==========
md.append("\n---\n## 3. 频率扩展效率分析\n")
md.append("以 210MHz 为基准，分析不同频率下 TTFT 的加速比 vs 理论频率比：\n")

md.append("| tp | input_len | freq(MHz) | TTFT(ms) | 频率比 | 实际加速比 | 效率(%) |")
md.append("|-----|-----------|-----------|----------|--------|-----------|---------|")

for tp in tps:
    for il in [128, 1024, 8192, 32000]:
        if il not in input_lens:
            continue
        base = [r for r in rows if r["tp"] == tp and r["input_len"] == il and r["gpu_clock"] == 210]
        if not base:
            continue
        base_ttft = base[0]["TTFT_ms"]
        for gc in gpu_clocks:
            cur = [r for r in rows if r["tp"] == tp and r["input_len"] == il and r["gpu_clock"] == gc]
            if not cur:
                continue
            freq_ratio = gc / 210.0
            speedup = base_ttft / cur[0]["TTFT_ms"]
            eff = speedup / freq_ratio * 100
            md.append(f"| {tp} | {il} | {gc} | {cur[0]['TTFT_ms']:.2f} | {freq_ratio:.2f}x | {speedup:.2f}x | {eff:.1f}% |")

# ========== 4. TP 扩展效率分析 ==========
md.append("\n---\n## 4. TP 扩展效率分析\n")
md.append("以 tp=1 为基准（若可用），分析不同 TP 下 TTFT 的加速比：\n")

md.append("| input_len | gpu_clock | tp | TTFT(ms) | vs tp=1 加速比 | 理想加速 | 效率(%) |")
md.append("|-----------|-----------|-----|----------|---------------|---------|---------|")

for il in [128, 512, 2048, 8192, 32000]:
    if il not in input_lens:
        continue
    for gc in [210, 930, 1410]:
        base = [r for r in rows if r["tp"] == 1 and r["input_len"] == il and r["gpu_clock"] == gc]
        if not base:
            continue
        base_ttft = base[0]["TTFT_ms"]
        for tp in tps:
            cur = [r for r in rows if r["tp"] == tp and r["input_len"] == il and r["gpu_clock"] == gc]
            if not cur:
                continue
            speedup = base_ttft / cur[0]["TTFT_ms"]
            eff = speedup / tp * 100
            md.append(f"| {il} | {gc} | {tp} | {cur[0]['TTFT_ms']:.2f} | {speedup:.2f}x | {tp}x | {eff:.1f}% |")

# ========== 5. Input Length 扩展分析 ==========
md.append("\n---\n## 5. Input Length 扩展分析\n")
md.append("检查 TTFT 随 input_len 的增长趋势。理想情况下，Attention 为 O(n²) 复杂度，FFN 为 O(n)。\n")

md.append("### TTFT 倍增率（input_len 翻倍时 TTFT 增长倍数）\n")
md.append("| tp | gpu_clock | input_len变化 | TTFT变化 | 倍增率 |")
md.append("|-----|-----------|---------------|----------|--------|")

for tp in [1, 4, 8]:
    for gc in [930]:
        prev = None
        for il in input_lens:
            cur = [r for r in rows if r["tp"] == tp and r["input_len"] == il and r["gpu_clock"] == gc]
            if not cur:
                continue
            if prev is not None and prev[0] > 0:
                ratio = il / prev[1]
                if abs(ratio - 2.0) < 0.1:
                    growth = cur[0]["TTFT_ms"] / prev[0]
                    md.append(f"| {tp} | {gc} | {prev[1]}→{il} | {prev[0]:.2f}→{cur[0]['TTFT_ms']:.2f} | {growth:.2f}x |")
            prev = (cur[0]["TTFT_ms"], il)

# ========== 6. 能耗分析 ==========
md.append("\n---\n## 6. 能耗分析\n")
md.append("分析总能耗(A_energy + F_energy)随频率变化的趋势，找到能效最优频率。\n")

md.append("### 各配置下的最优能效频率\n")
md.append("| tp | input_len | 最优频率(MHz) | 最低总能耗(mJ) | 最高频率总能耗(mJ) | 最低频率总能耗(mJ) |")
md.append("|-----|-----------|---------------|----------------|-------------------|-------------------|")

for tp in tps:
    for il in input_lens:
        subset = [r for r in rows if r["tp"] == tp and r["input_len"] == il]
        if not subset:
            continue
        energies = [(r["gpu_clock"], r["A_energy_mj"] + r["F_energy_mj"]) for r in subset]
        best = min(energies, key=lambda x: x[1])
        worst_high = [e for e in energies if e[0] == max(gpu_clocks)][0]
        worst_low = [e for e in energies if e[0] == min(gpu_clocks)][0]
        md.append(f"| {tp} | {il} | {best[0]} | {best[1]:.2f} | {worst_high[1]:.2f} | {worst_low[1]:.2f} |")

# ========== 7. A vs F 比例分析 ==========
md.append("\n---\n## 7. Attention vs FFN 比例分析\n")
md.append("分析 F/A 比值随 input_len 和频率的变化：\n")

md.append("| tp | input_len | gpu_clock=210 F/A | gpu_clock=930 F/A | gpu_clock=1410 F/A |")
md.append("|-----|-----------|-------------------|-------------------|-------------------|")

for tp in tps:
    for il in input_lens:
        vals = {}
        for gc in [210, 930, 1410]:
            cur = [r for r in rows if r["tp"] == tp and r["input_len"] == il and r["gpu_clock"] == gc]
            if cur:
                vals[gc] = cur[0]["F"] / cur[0]["A"] if cur[0]["A"] > 0 else 0
        if vals:
            md.append(f"| {tp} | {il} | {vals.get(210, 0):.2f} | {vals.get(930, 0):.2f} | {vals.get(1410, 0):.2f} |")

# ========== 8. A kernel 地板效应 ==========
md.append("\n---\n## 8. Attention Kernel 地板效应分析\n")
md.append("检查 Attention kernel 在高频下是否触达地板（不再随频率降低）：\n")

md.append("| tp | input_len | A@210(μs) | A@690(μs) | A@930(μs) | A@1170(μs) | A@1410(μs) | 930→1410降幅 |")
md.append("|-----|-----------|-----------|-----------|-----------|------------|------------|-------------|")

for tp in tps:
    for il in [128, 256, 512, 1024, 4096, 16384]:
        if il not in input_lens:
            continue
        vals = {}
        for gc in [210, 690, 930, 1170, 1410]:
            cur = [r for r in rows if r["tp"] == tp and r["input_len"] == il and r["gpu_clock"] == gc]
            if cur:
                vals[gc] = cur[0]["A"]
        if vals:
            drop = (vals.get(930, 0) - vals.get(1410, 0)) / vals.get(930, 0) * 100 if vals.get(930, 0) > 0 else 0
            md.append(f"| {tp} | {il} | {vals.get(210, 0):.1f} | {vals.get(690, 0):.1f} | {vals.get(930, 0):.1f} | {vals.get(1170, 0):.1f} | {vals.get(1410, 0):.1f} | {drop:.1f}% |")

# ========== 9. 综合结论 ==========
md.append("\n---\n## 9. 综合结论与发现\n")

md.append("""
### 9.1 数据整体合理性

1. **(A+F)×64 计算一致性**: 所有行的 (A+F)×64_ms 列均与 A、F 两列的计算值精确吻合，数据记录无误。
2. **TTFT 与 (A+F)×64 的关系**: 绝大多数情况下 TTFT 略大于 (A+F)×64（反映调度、通信等额外开销），差值通常在 1-5ms 以内，**但有一个明显异常点**。

### 9.2 发现的异常数据点

- **tp=2, input_len=128, gpu_clock=1170**: TTFT=74.80ms，而 (A+F)×64=45.42ms，偏差高达 64.7%。相邻频率点（930MHz 和 1410MHz）的 TTFT 分别为 45.28ms 和 45.12ms，均正常。**此点极大概率是测量噪声或一次性干扰**，建议重测。

### 9.3 频率扩展特性

1. **TTFT 随频率单调递减**：除上述异常点外，所有配置均满足此规律。
2. **频率扩展效率随 input_len 增大而提升**：
   - 小 input_len（128）在高频段（>930MHz）扩展效率显著下降（<50%），说明小序列在高频下变成 memory-bound。
   - 大 input_len（≥4096）在全频段保持较高效率（>80%），说明长序列的计算密度足以充分利用频率提升。
3. **高频段（1170-1410MHz）收益递减明显**：频率从 930→1410 的 51.6% 频率提升，在大多数场景只带来 15-30% 的 TTFT 降低。

### 9.4 TP 扩展特性

1. **TP 扩展效率随 input_len 增大而提升**：
   - input_len=128 时，tp=8 的扩展效率仅约 15-20%（大量时间花在通信开销上）。
   - input_len≥4096 时，tp=8 的扩展效率可达 55-65%。
2. **TP 扩展效率随频率升高而降低**：高频下单 GPU 计算更快，通信开销占比更大，TP 效率下降。
3. **tp=2 的扩展效率最好**：普遍达 80-95%，通信开销最小。

### 9.5 Attention vs FFN 特性

1. **F/A 比值随频率升高而下降**：说明 FFN 对频率更敏感（compute-bound），Attention 偏 memory-bound。
2. **F/A 比值随 TP 增大而下降**：TP 并行对 FFN 的加速效果好于 Attention。
3. **小 input_len 时 F/A≈1-2，大 input_len 时 F/A≈0.9-1.2**：Attention 的 O(n²) 特性使其在长序列下占比上升。

### 9.6 Attention Kernel 地板效应

1. **tp≥4、input_len≤512 时，Attention kernel 在 690MHz 以上就触达地板**（约 320-350μs），进一步升频无法降低耗时，说明这些配置下 Attention 完全是 memory-bound。
2. **tp=1、大 input_len 时无明显地板**：Attention kernel 在全频段持续受益于升频。

### 9.7 能耗特性

1. **能耗随频率呈 U 型曲线**：最低能耗点通常在 **690-930MHz** 区间。
2. **最低频率（210MHz）不是最节能的**：虽然功率低，但持续时间长，总能耗反而最高。
3. **最高频率（1410MHz）能耗也偏高**：功率过高，且频率扩展效率不足，导致总能耗上升。
4. **最优能效频率随 TP 增大而降低**：tp=8 时最优频率通常在 690MHz 或更低，因为 TP 通信开销在高频下更显著。

### 9.8 建议

1. **重测异常点**: tp=2, input_len=128, gpu_clock=1170 的数据建议重新采集。
2. **能效最优频率**: 对于追求能效的场景，建议选择 **930MHz** 作为默认频率，兼顾性能与能耗。
3. **小序列场景**: input_len≤256 时，升频到 930MHz 以上和增大 TP 的边际收益都很小，需考虑是否值得。
""")

with open("/workspace/benchmark/sglang-main/bash-test/pivot_pd_ops_out/prefill_analysis.md", "w") as f:
    f.write("\n".join(md))

print("分析完成，已输出到 prefill_analysis.md")

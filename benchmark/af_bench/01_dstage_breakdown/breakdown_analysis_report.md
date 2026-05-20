# PD+AF M=3 Pipeline Analysis — Qwen3-32B (tp=4×1)

## Test Configuration

| Parameter | Value |
|-----------|-------|
| Model | Qwen3-32B (64 layers, hidden=5120, kv_heads=8) |
| Architecture | PD+AF Disaggregated (4 GPUs: DF/DA/PF/PA) |
| TP per module | 1 (4 GPUs total) |
| Micro-batches (M) | 3 |
| Decode batch size | ~6 concurrent requests |
| Output tokens | 8 per request |
| Interconnect | UCX RDMA (mlx5_4) |
| GPU type | NVIDIA A800-SXM4-80GB |

## Pipeline Schedule

The pipeline interleaves A-stages and F-stages across all 64 layers for each of the 3 micro-batches:

```
attn_stage (DA):  A(0,0) F(0,0) A(1,0) F(1,0) ... A(63,0) F(63,0)  ── mb=0
                  A(0,1) F(0,1) A(1,1) F(1,1) ... A(63,1) F(63,1)  ── mb=1
                  A(0,2) F(0,2) A(1,2) F(1,2) ... A(63,2) F(63,2)  ── mb=2

ffn_stage  (DF):  A(0,0) F(0,0) A(1,0) F(1,0) ... A(63,0) F(63,0)  ── mb=0
                  A(0,1) F(0,1) A(1,1) F(1,1) ... A(63,1) F(63,1)  ── mb=1
                  A(0,2) F(0,2) A(1,2) F(1,2) ... A(63,2) F(63,2)  ── mb=2
```

Total: 64 layers × 3 micro-batches × 2 stages = **384 steps per forward pass**.

## Stage Breakdown

### Per-stage operations

| Node | Stage | Operations | Duration* |
|------|-------|-----------|-----------|
| DA (Attn) | A | `prep_attn` (recv+norm) → `attn` (compute) → `prep_mlp` (send to DF) | 156.6ms |
| DA (Attn) | F | `proxy_mlp` → `postprocess` (recv from DF + norm) | 216.0ms |
| DF (FFN) | A | `prep_attn` (recv from DA) → `proxy_attn` → `prep_mlp` | 225.6ms |
| DF (FFN) | F | `mlp` (compute) → `postprocess` (send to DA) | 150.1ms |

*per forward pass, all 64 layers × 3 micro-batches

## Schedule Divergence: Why M=3 Creates Pipeline Bubbles

**The core problem: DA and DF execute DIFFERENT pipeline schedules, not the same one.**

### DA (attn_stage) — first 18 steps:
```
A(0,0) A(0,1) A(0,2)  ← all 3 micro-batch A-stages for layer 0 FIRST
F(0,0) A(1,0) F(0,1) A(1,1) F(0,2) A(1,2)  ← then interleaved F + next-layer A
F(1,0) A(2,0) F(1,1) A(2,1) F(1,2) A(2,2)  ← pattern: F(l,mb) A(l+1,mb)
```

### DF (ffn_stage) — first 18 steps:
```
A(0,0) F(0,0) A(0,1) F(0,1) A(0,2) F(0,2)  ← ALL 3 MBs of layer 0: A→F→A→F→A→F
A(1,0) F(1,0) A(1,1) F(1,1) A(1,2) F(1,2)  ← then ALL 3 MBs of layer 1
A(2,0) F(2,0) A(2,1) F(2,1) A(2,2) F(2,2)  ← then ALL 3 MBs of layer 2
```

### What this means for data transfers:

**Q: Are data transfers batched across multiple requests?**  
**A: YES.** DA's ring buffer (RING_SIZE=3) sends A(0,0), A(0,1), A(0,2) outputs to DF together for layer 0 — all 3 micro-batches are batched. Each micro-batch contains a subset of the ~6 concurrent requests' tokens.

**But the schedule mismatch means DF can't consume them efficiently:**

1. DA sends A(0,0) → DF can consume it immediately (DF's first step is A(0,0))
2. DA sends A(0,1) → DF is busy with F(0,0), must wait
3. DA sends A(0,2) → DF is busy with A(0,1)→F(0,1), must wait
4. When DA starts F(0,0) (waiting for DF's FFN result), DF might still be computing F(0,1) for a different micro-batch!
5. This creates a **~212ms bubble** on DA and **~219ms bubble** on DF

### Visual proof from actual timing data:

```
DA timeline (first 30 steps, actual durations in ms):
  step  0: A(0,0) 1.14ms  ← sends to DF (micro-batch 0)
  step  1: A(0,1) 1.04ms  ← sends to DF (micro-batch 1)
  step  2: A(0,2) 1.14ms  ← sends to DF (micro-batch 2)
  step  3: F(0,0) 1.21ms  ← WAITS for DF result of mb=0 → but DF is still on F(0,0)!
  step  4: A(1,0) 0.90ms
  step  5: F(0,1) 1.20ms  ← WAITS for DF result of mb=1 → DF might be ahead
  ...

DF timeline (first 30 steps):
  step  0: A(0,0) 0.58ms  ← recv from DA (micro-batch 0)
  step  1: F(0,0) 0.88ms  ← FFN compute + send to DA
  step  2: A(0,1) 1.01ms  ← WAITS for DA result of mb=1 → DA has already sent it!
  step  3: F(0,1) 0.86ms
  step  4: A(0,2) 1.25ms  ← WAITS for DA result of mb=2
  step  5: F(0,2) 0.87ms
  ...
```

The individual per-stage timing is fast (~1ms per layer per stage), but the cumulative misalignment across 64 layers × 3 micro-batches = 384 steps creates massive bubbles because:
- DA's F stages wait for DF to finish, but DF is processing different micro-batches
- DF's A stages wait for DA to send, but DA is processing different layers



### Aggregate Timing (Single Forward Pass)

| Metric | DA (Attn Node) | DF (FFN Node) |
|--------|---------------|---------------|
| **Total time** | 372.6 ms | 375.7 ms |
| **A stage** | 156.6 ms (42%) | 225.6 ms (60%) |
| &nbsp;&nbsp;└ Real compute | 113.8 ms (attn) | 6.2 ms (proxy) |
| &nbsp;&nbsp;└ Wait/transfer | 42.8 ms | 219.4 ms |
| **F stage** | 216.0 ms (58%) | 150.1 ms (40%) |
| &nbsp;&nbsp;└ Real compute | 3.6 ms (proxy) | 129.9 ms (FFN) |
| &nbsp;&nbsp;└ Wait/transfer | 212.4 ms | 20.2 ms |

### Compute vs Wait

| Node | GPU Compute | Wait/Transfer | Utilization |
|------|------------|---------------|-------------|
| DA (GPU 1) | 117.4 ms | 255.2 ms | **32%** |
| DF (GPU 0) | 129.9 ms | 245.8 ms | **35%** |
| **Overall** | 247.3 ms | 501.0 ms | **33%** |

### Pipeline Bottleneck Analysis

```
Forward pass:  ~375ms

DA timeline:
  [──A_stage 157ms──][────────F_stage 216ms (waiting for DF)────────]

DF timeline:
  [────A_stage 226ms (waiting for DA)────][──F_stage 150ms──]
                                            └─ FFN compute 130ms

The serial dependency chain:
  DA computes attn(113.8ms) → sends to DF → DF waits(219.4ms) →
  DF computes FFN(129.9ms) → sends to DA → DA waits(212.4ms) →
  next layer starts...
```

**Key finding:** 67% of the forward pass time is spent waiting for UCX transfers, not computing. Each layer requires 2 data transfers (DA→DF for A-stage output, DF→DA for F-stage output), and the pipeline serializes these transfers.

### Comparison: Native vs PD+AF (tp=4)

| Metric | Native (tp=4) | PD+AF (M=3) | Ratio |
|--------|-------------|-----------|-------|
| Mean TPOT | 44.25 ms | 343.11 ms | **7.8× worse** |
| Per-forward-pass | ~45 ms | ~375 ms | **8.3× worse** |
| GPU Utilization | ~95%+ | 33% | **2.9× worse** |
| Output Throughput | 111.6 tok/s | 27.3 tok/s | **4.1× worse** |

## Generated Visualizations

| File | Description |
|------|-------------|
| `pipeline_bubble_explanation.png` | **Why M=3 creates bubbles** — DA vs DF schedule mismatch with data flow arrows |
| `schedule_divergence.png` | Side-by-side timeline of DA vs DF first 36 steps showing different schedules |
| `per_layer_breakdown.png` | Per-layer A+F duration summed across 3 micro-batches |
| `pipeline_schematic.png` | High-level pipeline flow diagram with stage annotations |
| `pipeline_full_gantt.png` | Full Gantt chart: all 64 layers × 3 micro-batches for DA and DF |
| `pipeline_zoomed_l0_l9.png` | Zoomed view: first 10 layers showing detailed A/F interleaving |
| `pipeline_breakdown_bars.png` | Stacked bar chart: compute vs wait per stage |
| `pipeline_compute_vs_wait.png` | Summary bar: overall compute utilization |
| `pipeline_per_layer_heatmap.png` | Heatmap: per-layer stage duration across micro-batches |

## Data Files

| File | Description |
|------|-------------|
| `parsed_timeline_m3.json` | Parsed per-stage timing data (384 steps, DA + DF) |
| `breakdown_stats_m3.json` | Aggregate CUDA event timing breakdown |
| `analysis_summary.json` | Summary metrics in machine-readable format |
| `raw_logs/da.log` | Raw DA server log with AFD_TIMELINE entries |
| `raw_logs/df.log` | Raw DF server log with AFD_TIMELINE entries |
| `raw_logs/pa.log` | Raw PA server log |
| `raw_logs/pf.log` | Raw PF server log |
| `raw_logs/router.log` | Raw router log |

## Test Script

`test_m3_timeline.py` — Launches PD+AF servers with `AFD_DETAILED_TIMING=1`, sends 6 concurrent requests to trigger M=3 decode, collects per-stage timing data from server logs.

## Conclusion

The M=3 pipeline suffers from poor GPU utilization (~33%) because:
1. **Each layer requires 2 sequential data transfers** across UCX: DA→DF (after A stage) and DF→DA (after F stage).
2. **No overlap between DA and DF compute** — the pipeline alternates A and F stages sequentially within each layer, but the data dependency forces one node to wait while the other computes.
3. **UCX transfer latency dominates** — each transfer is ~2-3ms for small decode tensors, accumulating to ~200ms of total wait time per forward pass.

Potential improvements:
- **Increase micro-batch count** to better pipeline overlapping layers (e.g., start layer L+1's A stage on DA while DF finishes layer L's F stage)
- **Use CUDA IPC instead of UCX** for single-node deployments (GPUs on same node)
- **Fuse attention+FFN for small decode batches** to avoid the disaggregation overhead entirely

# VTC state-tax observation report

Overall decision: **FAIL**

## Integrity

Artifacts 387; requests 34107; failures 0; cached tokens 0; retractions 0; exact resource traces 387/387; VTC traces 216/216; resolved policy/weights 387/387; GPU 6,7 metadata 387/387; equal-token mixed 288/288.

Profile fit: **PASS**; median relative error 0.2%, max 1.6%.

## Frozen selection

Global request-tax label: `token-vtc`.

| cell | frozen per-cell request-tax oracle |
|---|---|
| f1-base | request-tax-r1024 |
| f4-base | token-vtc |
| f16-base | token-vtc |
| f32-base | token-vtc |
| f16-state-relaxed | token-vtc |
| f16-kv-relaxed | token-vtc |
| f16-full | token-vtc |
| f16-fast | token-vtc |
| f16-slow | request-tax-r8192 |

## Confirmation by cell

| cell | token gap | profiled gap | global-tax gap | oracle-tax gap | instant gap | max resource ratio | composition ratio | split/packed queue p95 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| f1-base | 1.2% | 1.1% | 1.2% | 1.1% | 1.2% | 1.07x | 1.00x | 10.584/10.788 |
| f4-base | 757.7% | 670.0% | 751.0% | 752.5% | 391.0% | 3.90x | 3.97x | 11.319/8.331 |
| f16-base | 913.2% | 877.4% | 912.0% | 922.2% | 290.2% | 17.50x | 15.81x | 10.686/8.181 |
| f32-base | 924.6% | 923.1% | 927.7% | 929.9% | 162.2% | 33.69x | 31.62x | 11.048/8.281 |
| f16-state-relaxed | 988.0% | 983.0% | 990.5% | 982.4% | 958.7% | 4.92x | 16.02x | 2.680/0.861 |
| f16-kv-relaxed | 915.0% | 874.1% | 923.3% | 920.5% | 244.4% | 17.49x | 15.81x | 10.699/8.194 |
| f16-full | 983.7% | 982.2% | 986.6% | 984.6% | 957.1% | 5.35x | 16.02x | 2.631/0.809 |
| f16-fast | 686.3% | 649.4% | 688.3% | 688.0% | 85.0% | 17.45x | 15.81x | 14.105/11.598 |
| f16-slow | 26.1% | 30.4% | 26.6% | 29.1% | 14.7% | 16.66x | 15.80x | 2.125/0.001 |

## Standard-VTC resource detail

| cell | split/packed slowdown | peak state split/packed | peak KV split/packed | state-time ratio | KV-time ratio |
|---|---:|---:|---:|---:|---:|
| f1-base | 76.9%/77.5% | 12.0/12.0 | 67582/67581 | 1.07x | 1.07x |
| f4-base | 790.2%/32.5% | 21.0/15.0 | 29566/83876 | 1.02x | 3.90x |
| f16-base | 934.1%/21.0% | 24.0/18.0 | 8448/100166 | 1.11x | 17.50x |
| f32-base | 947.5%/22.9% | 24.0/18.0 | 4224/100163 | 1.07x | 33.69x |
| f16-state-relaxed | 1032.8%/44.8% | 270.0/24.0 | 94845/135086 | 4.92x | 3.22x |
| f16-kv-relaxed | 936.4%/21.4% | 24.0/18.0 | 8448/100166 | 1.11x | 17.49x |
| f16-full | 1028.7%/45.0% | 270.0/24.0 | 94861/135089 | 5.35x | 2.99x |
| f16-fast | 711.4%/25.1% | 24.0/18.0 | 8448/100177 | 1.11x | 17.45x |
| f16-slow | 140.5%/114.4% | 24.0/12.0 | 8448/66751 | 1.06x | 16.66x |

## Preregistered gates

- O1 VTC abstraction mismatch: **PASS**. f16/f32 max resource ratios 17.50x/33.69x; composition ratios 15.81x/31.62x; slowdown gaps 913.2%/924.6%.
- O2 hybrid state specificity: **FAIL**. state-relaxed gap reduction -8.2%; KV-relaxed reduction -0.2%; identical-shape f1 gap 1.2%.
- O3 scalar repair insufficiency: **PASS**. instant-share mean improvement 43.8%, cell-bootstrap 95% CI [21.7%, 65.5%], wins 8/9, mean makespan change -1.2%.


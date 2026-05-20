# Logs

Contains raw server logs from all benchmarks. Logs are large binary/text files and should not be committed to git.

## Subdirectories

| Directory | Contents |
|-----------|----------|
| `bubble_logs/` | Bubble breakdown experiment logs (DA, DF, PA, PF, router) |
| `ipc_breakdown_logs/` | IPC backend breakdown logs (moved to `02_communication/logs/`) |
| `raw_logs/` | Original raw server logs from initial pipeline analysis |
| `throughput_logs/` | All throughput benchmark logs and result JSONs |
| `wire_breakdown_logs/` | Wire-level transfer breakdown logs (moved to `02_communication/logs/`) |

## Log Naming Convention

- `{config}_{server}.log` — e.g., `pdaf_m3_da.log` = M=3 PDAF config, DA server
- Server tags: `da` (decode-attn), `df` (decode-ffn), `pa` (prefill-attn), `pf` (prefill-ffn), `router`
- Result JSONs: `results_{config}.json`

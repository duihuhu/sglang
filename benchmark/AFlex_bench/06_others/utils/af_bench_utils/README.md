# Utilities

Shared utility scripts used across multiple analysis and benchmark modules.

## Scripts

| Script | Purpose |
|--------|---------|
| `parse_logs.py` | Parse `AFD_TIMELINE` log entries from server logs into structured JSON. Used by most analysis scripts as the data preprocessing step. |
| `gen_all_breakdown_charts.py` | Batch-generate all pipeline breakdown charts from parsed timing data. Calls multiple plotting scripts in sequence. |

## Usage

```bash
# Parse raw server logs into JSON
python benchmark/af_bench/utils/parse_logs.py --log <path/to/da.log>

# Generate all breakdown charts
python benchmark/af_bench/utils/gen_all_breakdown_charts.py
```

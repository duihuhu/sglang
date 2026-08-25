# Shared paths for moe-energy benchmark scripts.
MOE_SCRIPTS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MOE_BENCH_ROOT="$(cd "$MOE_SCRIPTS_DIR/.." && pwd)"
MOE_DATA_DIR="${MOE_DATA_DIR:-$MOE_BENCH_ROOT/data}"

#!/usr/bin/env python3
"""Check M=2 low batch size data."""
import json, re, sys
from pathlib import Path

STEP_RE = re.compile(r"steps=(\[.*\])")
path = Path(sys.argv[1])
m = int(sys.argv[2]) if len(sys.argv) > 2 else 2

found_bs = {}
for i, line in enumerate(path.open(errors="replace")):
    if "[AFD_PER_STEP]" not in line:
        continue
    mt = STEP_RE.search(line)
    if not mt:
        continue
    steps = json.loads(mt.group(1))
    l0 = [s for s in steps if s["stage"] == "A" and s["layer_id"] == 0]
    bs = sum(s["batch_size"] for s in l0)
    n_mb = len(l0)
    key = f"M={n_mb},bs={bs}"
    if key not in found_bs:
        found_bs[key] = 0
    found_bs[key] += 1

for k, v in sorted(found_bs.items()):
    print(f"  {k}: {v} forwards")

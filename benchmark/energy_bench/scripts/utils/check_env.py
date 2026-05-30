#!/usr/bin/env python3
"""Quick validation: check all dependencies and configs are correct."""

import json
import sys
from pathlib import Path

PYTHON_PATH = "/workspace/env/sglang-tier/bin/python"
SCRIPT_DIR = Path(__file__).resolve().parent

errors = []

# Check Python env
print("Checking dependencies...")
try:
    import aiohttp
    print("  ✓ aiohttp")
except ImportError:
    errors.append("aiohttp not installed")

try:
    import numpy
    print("  ✓ numpy")
except ImportError:
    errors.append("numpy not installed")

try:
    import pynvml
    pynvml.nvmlInit()
    count = pynvml.nvmlDeviceGetCount()
    pynvml.nvmlShutdown()
    print(f"  ✓ pynvml ({count} GPUs)")
except Exception as e:
    errors.append(f"pynvml: {e}")

# Check energy models
print("\nChecking energy models...")
model_dir = Path("/workspace/sglang/benchmark/test_motivation/energy_models")
if model_dir.exists():
    pkls = list(model_dir.glob("*.pkl"))
    print(f"  ✓ {len(pkls)} model files in {model_dir}")
else:
    errors.append(f"Energy model dir not found: {model_dir}")

# Check sglang import
print("\nChecking sglang imports...")
sys.path.insert(0, str(SCRIPT_DIR.parents[2] / "python"))
try:
    from sglang.srt.energy.af_dvfs_controller import AFDVFSController
    print("  ✓ AFDVFSController")
except ImportError as e:
    errors.append(f"AFDVFSController import: {e}")

try:
    from sglang.srt.energy.af_profile_predictor import AFProfilePredictor
    print("  ✓ AFProfilePredictor")
except ImportError as e:
    errors.append(f"AFProfilePredictor import: {e}")

try:
    from sglang.srt.energy.af_launcher import launch_all
    print("  ✓ af_launcher")
except ImportError as e:
    errors.append(f"af_launcher import: {e}")

# Check configs
print("\nChecking launch configs...")
for name in ("launch_config_dvfs.json", "launch_config_baseline.json"):
    path = SCRIPT_DIR / name
    if path.exists():
        cfg = json.loads(path.read_text())
        model_path = cfg["model"]["path"]
        print(f"  ✓ {name} (model={model_path})")
    else:
        errors.append(f"Config not found: {path}")

# Check model availability
print("\nChecking model...")
try:
    from transformers import AutoConfig
    cfg = AutoConfig.from_pretrained("Qwen/Qwen3-0.6B")
    print(f"  ✓ Qwen/Qwen3-0.6B (layers={cfg.num_hidden_layers}, hidden={cfg.hidden_size})")
except Exception as e:
    errors.append(f"Model check: {e}")

# Summary
print("\n" + "=" * 50)
if errors:
    print(f"FAILED: {len(errors)} error(s)")
    for e in errors:
        print(f"  ✗ {e}")
    sys.exit(1)
else:
    print("ALL CHECKS PASSED ✓")
    print("\nTo run the benchmark:")
    print(f"  cd {SCRIPT_DIR}")
    print(f"  bash run_all.sh")
    print("\nOr step by step:")
    print(f"  {PYTHON_PATH} gen_workload.py --output-dir workloads")
    print(f"  {PYTHON_PATH} test_tier2_standalone.py  # standalone DVFS test")
    print(f"  {PYTHON_PATH} run_tier2_bench.py --workload workloads/workload_varying.jsonl")
    sys.exit(0)

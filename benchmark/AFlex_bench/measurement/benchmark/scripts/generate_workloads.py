#!/usr/bin/env python3
import argparse,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT/"src"))
from aflex_benchmark.config import load_json
from aflex_benchmark.workloads import generate
def main():
 p=argparse.ArgumentParser(); p.add_argument("--config",default=ROOT/"configs/workloads.json",type=Path); a=p.parse_args(); c=load_json(a.config); entries=generate(Path(c["generated_dir"]),c["requests_per_fixed_workload"],tuple(c["qps"]),c["seed"],Path(c["source_trace_dir"])); print(f"generated/indexed {len(entries)} workload variants")
if __name__=="__main__": main()

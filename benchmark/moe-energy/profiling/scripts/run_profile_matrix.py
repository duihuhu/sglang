#!/usr/bin/env python3
"""Serial matrix orchestrator for Qwen3 Attention/MoE profiling."""
import argparse, json, os, shlex, socket, subprocess, sys
from datetime import datetime, timezone
from pathlib import Path
HERE=Path(__file__).resolve().parent; ROOT=HERE.parent; DEFAULT_DATA=ROOT/'data'/'raw_v2'/'node1'

def free_port():
    with socket.socket() as s: s.bind(("",0)); return s.getsockname()[1]

def parse_args():
    p=argparse.ArgumentParser(description="Run isolated A-TP, F-TP and F-EP profiling jobs serially")
    p.add_argument("--model-path",default="/models/Qwen3-30B-A3B"); p.add_argument("--data-dir",type=Path,default=DEFAULT_DATA)
    p.add_argument("--phases",nargs="+",choices=["prefill","decode"],default=["prefill","decode"])
    p.add_argument("--components",nargs="+",choices=["A","F-TP","F-EP"],default=["A","F-TP","F-EP"])
    p.add_argument("--sizes",nargs="+",type=int,choices=[2,4,8],default=[2,4,8]); p.add_argument("--gpus",default="0,1,2,3,4,5,6,7")
    p.add_argument("--quick",action="store_true"); p.add_argument("--dry-run",action="store_true"); p.add_argument("--continue-on-error",action="store_true")
    p.add_argument("--extra-args",default="",help="extra arguments appended to each benchmark command")
    return p.parse_args()

def main():
    a=parse_args(); a.data_dir.mkdir(parents=True,exist_ok=True); (a.data_dir/'logs').mkdir(exist_ok=True); manifest=a.data_dir/'manifest.jsonl'; visible=[x.strip() for x in a.gpus.split(',') if x.strip()]
    jobs=[]
    mapping={"A":("A","attn_tp"),"F-TP":("F","moe_tp"),"F-EP":("F","moe_ep")}
    for phase in a.phases:
        for label in a.components:
            component,mode=mapping[label]
            for size in a.sizes:
                if size>len(visible): raise SystemExit(f"size {size} needs {size} GPUs, only {len(visible)} listed")
                ep=size if mode=="moe_ep" else 1; stem=f"qwen3_{phase}_{mode}_ws{size}"; output=a.data_dir/f"{stem}.jsonl"; log=a.data_dir/'logs'/f"{stem}.log"; port=free_port()
                cmd=[sys.executable,str(HERE/f"bench_{phase}_af.py"),"--model-path",a.model_path,"--component",component,"--parallel-mode",mode,"--tp-size",str(size),"--ep-size",str(ep),"--moe-runner-backend","triton","--moe-a2a-backend","none","--cuda-graph-backend-decode","disabled","--cuda-graph-backend-prefill","disabled","--nccl-port",str(port),"--output",str(output)]
                if a.quick: cmd.append("--quick")
                cmd.extend(shlex.split(a.extra_args)); jobs.append((cmd,output,log,size,phase,mode))
    print(f"planned {len(jobs)} serial jobs")
    for i,(cmd,output,log,size,phase,mode) in enumerate(jobs,1):
        shown=shlex.join(cmd); print(f"[{i}/{len(jobs)}] CUDA_VISIBLE_DEVICES={','.join(visible[:size])} {shown}")
        record={"timestamp":datetime.now(timezone.utc).isoformat(),"phase":phase,"parallel_mode":mode,"world_size":size,"output":str(output),"log":str(log),"command":cmd,"status":"dry-run" if a.dry_run else "running"}
        if a.dry_run: continue
        env=os.environ.copy(); env["CUDA_VISIBLE_DEVICES"]=','.join(visible[:size])
        before_rows = sum(1 for line in output.open() if line.strip()) if output.exists() else 0
        with log.open('a') as f: result=subprocess.run(cmd,env=env,stdout=f,stderr=subprocess.STDOUT)
        after_rows = sum(1 for line in output.open() if line.strip()) if output.exists() else 0
        no_data = result.returncode == 0 and after_rows == before_rows and before_rows == 0
        record["rows_before"] = before_rows; record["rows_after"] = after_rows
        record["returncode"] = result.returncode or (2 if no_data else 0)
        record["status"] = "no-data" if no_data else ("ok" if result.returncode == 0 else "failed")
        with manifest.open('a') as f: f.write(json.dumps(record,sort_keys=True)+'\n'); f.flush(); os.fsync(f.fileno())
        if record["returncode"] and not a.continue_on_error: raise SystemExit(f"job failed ({record['returncode']}); see {log}")
if __name__=='__main__': main()

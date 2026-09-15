#!/usr/bin/env python3
import argparse,json,os,sys
from datetime import datetime,timezone
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/"src"),str(ROOT/"scripts")]
from aflex_benchmark.runner import expand_queue
import run_measurement_rdma2n_4gpu_qps4 as r2
def now():return datetime.now(timezone.utc).isoformat()
def main(argv=None):
 p=argparse.ArgumentParser(description="Three-architecture bound canary gate; default has no side effects")
 p.add_argument("--results",type=Path,default=ROOT/"results/raw/measurement_rdma2n_4gpu_qps4_20260825");p.add_argument("--execute",action="store_true");a=p.parse_args(argv);b=r2.load_bundle_2n()
 if not a.execute:
  try:g=r2.require_gate(a.results,b);print(json.dumps({"mode":"check-only","status":"pass","gate":g},indent=2));return 0
  except Exception as e:print(json.dumps({"mode":"check-only","status":"blocked","blocker":str(e)},indent=2));return 2
 lock=r2.acquire(a.results,"canary");pre=r2.online_preflight(a.results)
 if pre["status"]!="pass":print(json.dumps(pre,indent=2));return 2
 pending=a.results/"canary/canary.pending.json";bases={(x["point"]["architecture"],x["workload"]):x for x in expand_queue(b)};records=[]
 for ordinal,(arch,slot) in enumerate(zip(r2.ARCHITECTURES,("slotA","slotB","slotA")),1):
  job={"repeat":0,"ordinal":ordinal,"pair":"pair34","slot":slot};item,actual,summary,path,ok,reason=r2.execute_once(b,bases[arch,r2.WORKLOADS[0]],a.results,job,[f"canary:{arch}"],1,True);record={"architecture":arch,"pair":"pair34","slot":slot,"run_id":item["run_id"],"artifact":str(path.resolve()),"layout":actual,"status":"pass" if ok else "fail","reason":reason,"finished_at":now()}
  if ok:record["artifact_hash"]=r2.artifact_hash(path)
  records.append(record);r2.atomic_write(pending,{"status":"pending","artifacts":records})
  if not ok:raise RuntimeError(record)
 gate={"schema_version":2,"status":"pass","generated_at":now(),"generator_pid":os.getpid(),"topology":{"pairs":["pair34"],"slots":["slotA","slotB"],"total_gpus_per_job":4},"bindings":r2.hashes(b),"artifacts":records};r2.atomic_write(a.results/r2.GATE_NAME,gate);r2.atomic_write(a.results/"canary/canary.final.json",gate);print(json.dumps(gate,indent=2));return 0
if __name__=="__main__":raise SystemExit(main())

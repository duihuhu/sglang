#!/usr/bin/env python3
from __future__ import annotations
import argparse, copy, json, sys, threading, time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT/"src")); sys.path.insert(0,str(ROOT/"scripts"))
from aflex_benchmark.config import load_bundle
from aflex_benchmark.runner import execute_item, expand_queue, run_id
from run_measurement_rdma_qps4 import valid, write
LANES={
 "lane0":{"nic":"mlx5_0","gpus":{"node1":[0,1],"node2":[0,1],"node3":[0],"node4":[0]}},
 "lane1":{"nic":"mlx5_1","gpus":{"node1":[2,3],"node2":[2,3],"node3":[2],"node4":[2]}},
 "lane2":{"nic":"mlx5_4","gpus":{"node1":[4,5],"node2":[5],"node3":[4],"node4":[4]}},
 "lane3":{"nic":"mlx5_5","gpus":{"node1":[6,7],"node2":[6,7],"node3":[6],"node4":[6]}},
}
EXCLUDED={"node2":[4]}; LOCK=threading.Lock()
def now(): return datetime.now(timezone.utc).isoformat()
def placements(point,lane):
 g=LANES[lane]["gpus"]; topo=point["metadata"]["topology_id"]; arch=point["architecture"]
 if topo=="rdma2n":
  assert len(g["node1"])==2 and len(g["node2"])==2
  if arch=="native": return {"native_placements":[{"node":"node1","gpus":g["node1"]},{"node":"node2","gpus":g["node2"]}]}
  roles=("prefill_placements","decode_placements") if arch=="pd" else ("ffn_placements","attention_placements")
  return {roles[0]:[{"node":"node1","gpus":g["node1"]}],roles[1]:[{"node":"node2","gpus":g["node2"]}]}
 single={node:[g[node][0]] for node in ("node1","node2","node3","node4")}
 if arch=="native": return {"native_placements":[{"node":n,"gpus":single[n]} for n in ("node1","node2","node3","node4")]}
 roles=("prefill_placements","decode_placements") if arch=="pd" else ("ffn_placements","attention_placements")
 return {roles[0]:[{"node":n,"gpus":single[n]} for n in ("node1","node2")],roles[1]:[{"node":n,"gpus":single[n]} for n in ("node3","node4")]}
def configure(base,repeat,ordinal,lane,batch,concurrent):
 item=copy.deepcopy(base); point=item["point"]; point.update(placements(point,lane)); point["port_base"]=20000+ordinal*100
 actual={}
 for field in ("native_placements","prefill_placements","decode_placements","ffn_placements","attention_placements"):
  for placement in point.get(field,[]): actual.setdefault(placement["node"],set()).update(placement["gpus"])
 layout={node:{"gpus":sorted(gpus),"nic":LANES[lane]["nic"]} for node,gpus in actual.items()}
 point["metadata"].update({"parallel_execution":True,"parallel_batch":batch,"concurrent_job_ids":concurrent,"lane":lane,"actual_layout":layout,"isolation":"exclusive physical GPU and per-node RDMA NIC lane; unique ports/tmp dir"})
 item["run_id"]=run_id(point,item["workload"],item["qps"],repeat*100000+ordinal); return item,layout
def energy_scope_valid(run_dir,layout):
 energy=json.loads((run_dir/"energy.json").read_text()); expected={n:sorted(map(str,v["gpus"])) for n,v in layout.items()}; actual={n:sorted(v) for n,v in energy.get("gpu_uuids",{}).items()}
 return actual==expected and "4" not in actual.get("node2",[]) and all(value.startswith("GPU-") for rows in energy.get("gpu_uuids",{}).values() for value in rows.values())
def main():
 ap=argparse.ArgumentParser(); ap.add_argument("--matrix",type=Path,default=ROOT/"configs/measurement_rdma_qps4_qwen3_30b_a3b_4gpu.json"); ap.add_argument("--results",type=Path,default=ROOT/"results/raw/measurement_rdma_qps4_20260825"); ap.add_argument("--max-attempts",type=int,default=3); ap.add_argument("--execute",action="store_true"); a=ap.parse_args()
 bundle=load_bundle(ROOT/"configs",a.matrix,None); queue=expand_queue(bundle); by={(x["point"]["metadata"]["topology_id"],x["point"]["architecture"],x["workload"]):x for x in queue}; pp=a.results/"progress.json"; progress=json.loads(pp.read_text()); rows={(r["repeat"],r["point_id"],r["workload"]):r for r in progress["runs"]}
 # Recover complete fail-closed artifacts not checkpointed before serial scheduler stopped.
 for row in rows.values():
  if row["status"]=="valid": continue
  for summary_path in sorted(a.results.glob("repeat*/attempt*/*/summary.json")):
   summary=json.loads(summary_path.read_text())
   if summary.get("point",{}).get("id")==row["point_id"] and summary.get("workload")==row["workload"]:
    ok,reason=valid(summary,summary_path.parent)
    if ok:
     row["attempts"].append({"attempt":"recovered_serial","run_id":summary["run_id"],"artifact":str(summary_path.parent),"status":"complete","valid":True,"reason":None,"finished_at":now()}); row["status"]="valid"; break
 progress.update({"status":"parallel_running" if a.execute else "parallel_planned","parallel_execution":True,"resource_topology":{"lanes":LANES,"permanently_excluded_gpus":EXCLUDED,"max_safe_concurrency":4,"scheduled_concurrency":3,"reason":"three architecture jobs per comparison group; distinct per-node HCA lanes"},"isolation_policy":{"gpu":"global exclusive lease","nic":"no same-node same-HCA overlap","ports":"unique base per logical run, stride 100","cleanup":"job PID/ports/plan GPU only","fabric_caveat":"distinct HCAs still share the cluster fabric and may have residual cross-traffic"}}); progress["runs"]=list(rows.values()); write(pp,progress)
 ordinal=0
 for repeat in (1,2,3):
  for workload in ["measurement_qa_lpld","measurement_chatbot_lphd","measurement_balanced_mpmd","measurement_rag_hpld","measurement_summary_hphd","measurement_longcontext"]:
   for topo in ("rdma2n","rdma4n"):
    batch=f"r{repeat}-{workload}-{topo}"; lane_pool=["lane0","lane1","lane3"] if topo=="rdma2n" else ["lane0","lane1","lane2","lane3"]; rotation=(repeat+["measurement_qa_lpld","measurement_chatbot_lphd","measurement_balanced_mpmd","measurement_rag_hpld","measurement_summary_hphd","measurement_longcontext"].index(workload))%len(lane_pool); selected=[lane_pool[(rotation+i)%len(lane_pool)] for i in range(3)]; arches=("native","pd","af"); concurrent=[f"{repeat}:{topo}:{arch}:{workload}" for arch in arches]; jobs=[]
    for arch,lane in zip(arches,selected):
     ordinal+=1; base=by[(topo,arch,workload)]; key=(repeat,base["point"]["id"],workload); row=rows.setdefault(key,{"repeat":repeat,"point_id":key[1],"workload":workload,"status":"pending","attempts":[]})
     if row["status"]=="valid": continue
     item,layout=configure(base,repeat,ordinal,lane,batch,concurrent); jobs.append((item,layout,row,ordinal))
    if not a.execute: continue
    def run(job):
     item,layout,row,index=job; attempt=len(row["attempts"])+1; final_attempt=attempt+a.max_attempts-1
     while attempt<=final_attempt:
      current=copy.deepcopy(item); current["point"]["metadata"]["retry_attempt"]=attempt; current["run_id"]=run_id(current["point"],current["workload"],current["qps"],repeat*100000+index*100+attempt)
      root=a.results/f"repeat{repeat}"/f"parallel_attempt{attempt}"; summary=execute_item(current,bundle,root); run_dir=root/current["run_id"]; ok,reason=valid(summary,run_dir)
      if ok and not energy_scope_valid(run_dir,layout): ok,reason=False,"energy_uuid_scope"
      with LOCK:
       row["attempts"].append({"attempt":attempt,"run_id":current["run_id"],"artifact":str(run_dir),"status":summary.get("status"),"valid":ok,"reason":reason,"finished_at":now(),"parallel_batch":batch,"lane":item["point"]["metadata"]["lane"],"layout":layout,"port_base":item["point"]["port_base"],"concurrent_job_ids":concurrent}); row["status"]="valid" if ok else "retry_needed"; progress["runs"]=list(rows.values()); progress["updated_at"]=now(); progress["valid_runs"]=sum(r["status"]=="valid" for r in rows.values()); write(pp,progress)
      if ok:return
      attempt+=1
     raise RuntimeError(f"exhausted retries {item['run_id']}")
    with ThreadPoolExecutor(max_workers=3) as pool:
     futures=[pool.submit(run,job) for job in jobs]
     for future in as_completed(futures): future.result()
 progress.update({"status":"complete","updated_at":now(),"valid_runs":sum(r["status"]=="valid" for r in rows.values())}); progress["runs"]=list(rows.values()); write(pp,progress)
 print(json.dumps({"valid":progress["valid_runs"],"progress":str(pp)},indent=2))
if __name__=="__main__": main()

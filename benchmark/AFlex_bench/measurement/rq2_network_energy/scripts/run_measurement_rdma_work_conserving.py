#!/usr/bin/env python3
from __future__ import annotations
import argparse,copy,fcntl,json,os,sys,threading,time
from concurrent.futures import ThreadPoolExecutor,wait,FIRST_COMPLETED
from datetime import datetime,timezone
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]; sys.path[:0]=[str(ROOT/"src"),str(ROOT/"scripts")]
from aflex_benchmark.config import load_bundle
from aflex_benchmark.deploy import build_plan
from aflex_benchmark.runner import execute_item,expand_queue,run_id
from run_measurement_rdma_qps4 import valid,write
NICS={0:"mlx5_0",1:"mlx5_1",2:"mlx5_4",3:"mlx5_5"}
PAIR_GPUS={
 "pair12":{0:{"node1":[0,1],"node2":[0,1]},1:{"node1":[2,3],"node2":[2,3]},3:{"node1":[6,7],"node2":[6,7]}},
 "pair34":{0:{"node3":[0,1],"node4":[0,1]},1:{"node3":[2,3],"node4":[2,3]},2:{"node3":[4,5],"node4":[4,5]},3:{"node3":[6,7],"node4":[6,7]}},
}
FOUR_GPUS={0:{"node1":[0],"node2":[0],"node3":[0],"node4":[0]},1:{"node1":[2],"node2":[2],"node3":[2],"node4":[2]},2:{"node1":[4],"node2":[5],"node3":[4],"node4":[4]},3:{"node1":[6],"node2":[6],"node3":[6],"node4":[6]}}
EXCLUDED={("node2",4)}; LOCK=threading.Lock()
def now():return datetime.now(timezone.utc).isoformat()
def acquire_scheduler_lock(results):
 results.mkdir(parents=True,exist_ok=True); path=results/"scheduler.lock"; handle=path.open("a+")
 try: fcntl.flock(handle.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
 except BlockingIOError as exc: raise RuntimeError(f"another measurement scheduler holds {path}") from exc
 handle.seek(0); handle.truncate(); handle.write(json.dumps({"pid":os.getpid(),"started_at":now(),"scheduler":"work_conserving"})+"\n"); handle.flush(); os.fsync(handle.fileno()); return handle
def resource(job):
 return ({(node,gpu) for node,gpus in job["layout"].items() for gpu in gpus},{(node,job["nic"]) for node in job["layout"]})
def placements(point,layout):
 nodes=list(layout); arch=point["architecture"]
 if arch=="native":return {"native_placements":[{"node":n,"gpus":layout[n]} for n in nodes]}
 roles=("prefill_placements","decode_placements") if arch=="pd" else ("ffn_placements","attention_placements")
 if len(nodes)==2:return {roles[0]:[{"node":nodes[0],"gpus":layout[nodes[0]]}],roles[1]:[{"node":nodes[1],"gpus":layout[nodes[1]]}]}
 return {roles[0]:[{"node":n,"gpus":layout[n]} for n in nodes[:2]],roles[1]:[{"node":n,"gpus":layout[n]} for n in nodes[2:]]}
def configure(base,repeat,ordinal,slot,corunners):
 item=copy.deepcopy(base); point=item["point"]; layout=slot["layout"]; point.update(placements(point,layout)); point["port_base"]=20000+ordinal*100; nic=NICS[slot["lane"]]; actual={n:{"gpus":g,"nic":nic} for n,g in layout.items()}; point["metadata"].update({"parallel_execution":True,"lane":f"lane{slot['lane']}","node_pair":slot.get("pair"),"actual_layout":actual,"concurrent_job_ids":corunners,"isolation":"exclusive GPU and per-node HCA lane; unique ports/tmp"}); item["run_id"]=run_id(point,item["workload"],item["qps"],repeat*100000+ordinal); return item,actual
def validate_plan_shape(bundle,item):
 point=item["point"]; plan=build_plan(bundle["cluster"],item["model"]["path"],point,run_tag=item["run_id"]); used=plan.used_gpu_map(); topo=point["metadata"]["topology_id"]
 expected_nodes=2 if topo=="rdma2n" else 4; expected_per_node=2 if topo=="rdma2n" else 1
 if len(used)!=expected_nodes or any(len(gpus)!=expected_per_node for gpus in used.values()) or sum(map(len,used.values()))!=4:
  raise ValueError(f"invalid {topo} plan GPU shape: {used}")
 expected={placement["node"] for field in ("native_placements","prefill_placements","decode_placements","ffn_placements","attention_placements") for placement in point.get(field,[])}
 if set(used)!=expected: raise ValueError(f"plan nodes {set(used)} differ from placements {expected}")
 for process in (p for p in plan.processes if p.gpus):
  nic=point["metadata"]["actual_layout"][process.node]["nic"]
  if nic not in process.command: raise ValueError(f"{process.role} missing NIC binding {nic}")
 return plan

def validate_artifact(path,item,actual):
 summary=json.loads((path/"summary.json").read_text()); manifest=json.loads((path/"deployment_manifest.json").read_text()); link=json.loads((path/"link_validation.json").read_text()); requests=[json.loads(x) for x in (path/"requests.jsonl").read_text().splitlines() if x.strip()]
 used={}
 for process in manifest["processes"]:
  if process["gpus"]: used.setdefault(process["node"],set()).update(process["gpus"])
 used={n:sorted(v) for n,v in used.items()}; expected={n:sorted(v["gpus"]) for n,v in actual.items()}; topo=item["point"]["metadata"]["topology_id"]
 return (used==expected and sum(map(len,used.values()))==4 and len(used)==(2 if topo=="rdma2n" else 4) and summary.get("status")=="complete" and summary.get("requests_success")==64 and len(requests)==64 and all(r.get("success") and r.get("completion_tokens")==r.get("expected_completion_tokens") for r in requests) and link.get("status")=="pass" and link.get("physical_verified") and energy_ok(path,actual))
def energy_ok(path,actual):
 e=json.loads((path/"energy.json").read_text()); expected={n:sorted(map(str,v["gpus"])) for n,v in actual.items()}; observed={n:sorted(v) for n,v in e.get("gpu_uuids",{}).items()}; return observed==expected and "4" not in observed.get("node2",[]) and all(v.startswith("GPU-") for x in e.get("gpu_uuids",{}).values() for v in x.values())
def slots(topo):
 if topo=="rdma2n":return [{"topology":topo,"pair":pair,"lane":lane,"layout":layout} for pair,lanes in PAIR_GPUS.items() for lane,layout in lanes.items()]
 return [{"topology":topo,"lane":lane,"layout":layout} for lane,layout in FOUR_GPUS.items()]
def main():
 ap=argparse.ArgumentParser(); ap.add_argument("--matrix",type=Path,default=ROOT/"configs/measurement_rdma_qps4_qwen3_30b_a3b_4gpu.json"); ap.add_argument("--results",type=Path,default=ROOT/"results/raw/measurement_rdma_qps4_20260825"); ap.add_argument("--execute",action="store_true"); ap.add_argument("--max-attempts",type=int,default=3); a=ap.parse_args(); scheduler_lock=acquire_scheduler_lock(a.results); bundle=load_bundle(ROOT/"configs",a.matrix,None); queue=expand_queue(bundle); base={(x["point"]["metadata"]["topology_id"],x["point"]["architecture"],x["workload"]):x for x in queue}; pp=a.results/"progress.json"; progress=json.loads(pp.read_text()); rows={(r["repeat"],r["point_id"],r["workload"]):r for r in progress["runs"]}
 workloads=["measurement_qa_lpld","measurement_chatbot_lphd","measurement_balanced_mpmd","measurement_rag_hpld","measurement_summary_hphd","measurement_longcontext"]; pending=[]; ordinal=0
 for repeat in (1,2,3):
  for wi,workload in enumerate(workloads):
   for topo in ("rdma2n","rdma4n"):
    available=slots(topo); rotation=(repeat+wi)%len(available)
    for ai,arch in enumerate(("native","pd","af")):
     ordinal+=1; b=base[topo,arch,workload]; key=(repeat,b["point"]["id"],workload); row=rows.setdefault(key,{"repeat":repeat,"point_id":key[1],"workload":workload,"status":"pending","attempts":[]})
     if row["status"]=="valid":continue
     slot=available[(rotation+ai)%len(available)]; pending.append({"base":b,"row":row,"repeat":repeat,"ordinal":ordinal,"slot":slot,"logical_id":f"{repeat}:{topo}:{arch}:{workload}"})
 progress.update({"status":"work_conserving_running" if a.execute else "work_conserving_planned","parallel_execution":True,"resource_topology":{"permanently_excluded_gpus":{"node2":[4]},"theoretical_concurrency":{"rdma2n":6,"rdma4n":4},"selected_concurrency":{"rdma2n":6,"rdma4n":4},"rdma2n_pairs":{"pair12":[0,1,3],"pair34":[0,1,2,3]},"rdma4n_lanes":{"lane0":[0,0,0,0],"lane1":[2,2,2,2],"lane2":[4,5,4,4],"lane3":[6,6,6,6]},"gpu_utilization_estimate":{"rdma2n":"24/31 usable GPUs at six concurrent jobs = 77.4%","rdma4n":"16/31 usable GPUs at four concurrent jobs = 51.6%"}},"isolation_policy":{"gpu":"global exclusive lease","nic":"no same-node same-HCA overlap","ports":"unique stride 100","cleanup":"job PID/ports/plan GPU only","fabric_caveat":"separate HCAs still share cluster fabric"}}); progress["runs"]=list(rows.values()); write(pp,progress)
 if not a.execute:return
 def execute(job,corunners):
  row=job["row"]; attempt=len(row["attempts"])+1; final=attempt+a.max_attempts-1
  while attempt<=final:
   item,actual=configure(job["base"],job["repeat"],job["ordinal"],job["slot"],corunners); item["point"]["metadata"].update({"retry_attempt":attempt,"measurement_repeat":job["repeat"]}); item["run_id"]=run_id(item["point"],item["workload"],item["qps"],job["repeat"]*100000+job["ordinal"]*100+attempt); validate_plan_shape(bundle,item); root=a.results/f"repeat{job['repeat']}"/f"wc_attempt{attempt}"; summary=execute_item(item,bundle,root); path=root/item["run_id"]; ok,reason=valid(summary,path)
   if ok and not validate_artifact(path,item,actual):ok,reason=False,"artifact_topology_or_scope"
   with LOCK:
    row["attempts"].append({"attempt":attempt,"run_id":item["run_id"],"artifact":str(path),"status":summary.get("status"),"valid":ok,"reason":reason,"finished_at":now(),"lane":item["point"]["metadata"]["lane"],"node_pair":item["point"]["metadata"]["node_pair"],"layout":actual,"port_base":item["point"]["port_base"],"concurrent_job_ids":corunners}); row["status"]="valid" if ok else "retry_needed"; progress["valid_runs"]=sum(r["status"]=="valid" for r in rows.values()); progress["runs"]=list(rows.values()); progress["updated_at"]=now(); write(pp,progress)
   if ok:return
   attempt+=1
  raise RuntimeError(f"exhausted retries {job['logical_id']}")
 active={}; used_gpu=set(); used_nic=set()
 with ThreadPoolExecutor(max_workers=10) as pool:
  while pending or active:
   launched=False
   for job in list(pending):
    gpus,nics=resource({"layout":job["slot"]["layout"],"nic":NICS[job["slot"]["lane"]]})
    if gpus&used_gpu or nics&used_nic:continue
    corunners=[x["logical_id"] for x,_,_ in active.values()]+[job["logical_id"]]; future=pool.submit(execute,job,corunners); active[future]=(job,gpus,nics); used_gpu|=gpus; used_nic|=nics; pending.remove(job); launched=True
   if not active:raise RuntimeError("resource scheduler deadlock")
   done,_=wait(active,return_when=FIRST_COMPLETED)
   for future in done:
    job,gpus,nics=active.pop(future); used_gpu-=gpus; used_nic-=nics; future.result()
 progress.update({"status":"complete","valid_runs":sum(r["status"]=="valid" for r in rows.values()),"updated_at":now()}); progress["runs"]=list(rows.values()); write(pp,progress); print(json.dumps({"valid":progress["valid_runs"]},indent=2))
if __name__=="__main__":main()

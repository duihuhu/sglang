#!/usr/bin/env python3
from __future__ import annotations
import argparse,copy,fcntl,hashlib,json,os,signal,sys,threading,time
from concurrent.futures import FIRST_COMPLETED,ThreadPoolExecutor,wait
from datetime import datetime,timezone
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/"src"),str(ROOT/"scripts")]
from aflex_benchmark.config import load_bundle
from aflex_benchmark.deploy import build_plan
from aflex_benchmark.deploy.base import RemoteExecutor
from aflex_benchmark.runner import execute_item,expand_queue
ARCHITECTURES=("native","pd","af");WORKLOADS=("measurement_qa_lpld","measurement_chatbot_lphd","measurement_balanced_mpmd","measurement_rag_hpld","measurement_summary_hphd","measurement_longcontext")
PAIRS={"pair34":("node3","node4")};LANES={0:"mlx5_0",2:"mlx5_1",4:"mlx5_4",6:"mlx5_5"};FORMAL_PORT_START=24000;FORMAL_PORT_STRIDE=48;FORMAL_PORT_ATTEMPTS=16;FORMAL_JOB_COUNT=54;DERIVED_PORT_MAX_OFFSET=41;GATE_NAME="rdma2n_qps2_canary_gate.json";STATES={"preflight","canary","formal","paused","complete","failed"};CONFIG=ROOT/"configs/measurement_rdma2n_qps2_qwen3_30b_a3b_2gpu.json";MUTEX=threading.Lock();STOP=threading.Event()
def now():return datetime.now(timezone.utc).isoformat()
def digest(x):return hashlib.sha256((x if isinstance(x,bytes) else json.dumps(x,sort_keys=True,separators=(",",":")).encode())).hexdigest()
def file_hash(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def atomic_write(path,value):
 path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_name(f".{path.name}.{os.getpid()}.{threading.get_ident()}.tmp")
 with tmp.open("w") as f:json.dump(value,f,indent=2,sort_keys=True);f.write("\n");f.flush();os.fsync(f.fileno())
 os.replace(tmp,path);fd=os.open(path.parent,os.O_RDONLY);os.fsync(fd);os.close(fd)
def acquire(results,owner="scheduler"):
 results.mkdir(parents=True,exist_ok=True);f=(results/"scheduler.lock").open("a+")
 try:fcntl.flock(f.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
 except BlockingIOError as e:
  f.seek(0);info=f.read().strip() or "<no metadata>";f.close();raise RuntimeError(f"singleton lock held: {info}") from e
 f.seek(0);f.truncate();json.dump({"pid":os.getpid(),"owner":owner,"started_at":now(),"cmdline":sys.argv},f);f.write("\n");f.flush();os.fsync(f.fileno());return f
def load_bundle_2n():
 b=load_bundle(ROOT/"configs",CONFIG);b["cluster"]["nodes"]=[n for n in b["cluster"]["nodes"] if n["name"] in {"node3","node4"}]
 if {n["name"] for n in b["cluster"]["nodes"]}!={"node3","node4"}:raise ValueError("requires exactly node3/node4")
 return b
def resource_plan(c=4):return {"pairs":PAIRS,"lanes":LANES,"max_concurrency":c,"planned_nodes":["node3","node4"],"excluded_nodes":["node1","node2"],"atomic_lease":["gpu","hca","ports"],"total_gpus_per_job":2}
def hashes(bundle,c=4):return {"config_hash":file_hash(CONFIG),"runtime_manifest_hash":digest({k:bundle["cluster"].get(k) for k in ("container","python_source","host_network")}),"resource_plan_hash":digest(resource_plan(c))}
def resolve_artifact_path(path):
 path=Path(path)
 if path.exists(): return path
 marker="/benchmark/results/"
 text=str(path)
 if marker in text:
  candidate=ROOT/"results/raw"/text.split(marker,1)[1]
  if candidate.exists(): return candidate
 return path

def artifact_hash(path):
 path=resolve_artifact_path(path)
 h=hashlib.sha256()
 for n in ("deployment_manifest.json","summary.json","requests.jsonl","link_validation.json","energy.json"):
  f=Path(path)/n
  if not f.is_file():raise RuntimeError(f"missing canary artifact {f}")
  h.update(n.encode());h.update(f.read_bytes())
 return h.hexdigest()
def layout(pair,lane):return {n:[lane] for n in PAIRS[pair]}
def placement_fields(a,p):
 ns=list(p)
 if a=="native":return {"native_placements":[{"node":n,"gpus":p[n]} for n in ns]}
 rs=("prefill_placements","decode_placements") if a=="pd" else ("ffn_placements","attention_placements")
 return {rs[0]:[{"node":ns[0],"gpus":p[ns[0]]}],rs[1]:[{"node":ns[1],"gpus":p[ns[1]]}]}
def mapping(r,w,a):return "pair34",tuple(LANES)[(w*3+a+r-1)%4]
def logical_key(r,a,w):return f"{r}:{a}:{w}"
def allocate_port_base(ordinal,attempt,canary=False):
 if canary:return 50000+(ordinal-1)*48
 if not 1<=ordinal<=FORMAL_JOB_COUNT:raise ValueError(f"ordinal outside 1..{FORMAL_JOB_COUNT}: {ordinal}")
 if not 1<=attempt<=FORMAL_PORT_ATTEMPTS:raise ValueError(f"attempt outside allocated range 1..{FORMAL_PORT_ATTEMPTS}: {attempt}")
 base=FORMAL_PORT_START+((attempt-1)*FORMAL_JOB_COUNT+ordinal-1)*FORMAL_PORT_STRIDE
 if base+DERIVED_PORT_MAX_OFFSET>65535:raise ValueError(f"derived port exceeds 65535: {base+DERIVED_PORT_MAX_OFFSET}")
 return base
def configure(base,repeat,ordinal,pair,lane,corunners,attempt,canary=False,config_hash=None):
 item=copy.deepcopy(base);p=item["point"];placed=layout(pair,lane);p.update(placement_fields(p["architecture"],placed));p["port_base"]=allocate_port_base(ordinal,attempt,canary);slot=f"{pair}:gpu{lane}:{LANES[lane]}";identity={"repeat":repeat,"attempt":attempt,"point":p["id"],"workload":item["workload"],"qps":item["qps"],"resource_slot":slot,"config_hash":config_hash or file_hash(CONFIG),"canary":canary};item["run_id"]=digest(identity)[:24];actual={n:{"gpus":g,"nic":LANES[lane]} for n,g in placed.items()};p.setdefault("metadata",{}).update({"measurement_repeat":repeat,"retry_attempt":attempt,"logical_key":logical_key(repeat,p["architecture"],item["workload"]),"resource_slot":slot,"lane_gpu":lane,"hca_group":LANES[lane],"actual_layout":actual,"concurrent_job_ids":list(corunners),"startup_barrier":"AFLEX_RUN_ID+role+port+environment+GPU_UUID","run_identity":identity,"canary":canary});return item,actual
def plan_ports(plan):return {(x.host,p) for x in plan.processes for p in (x.port,x.bootstrap_port,x.nccl_port,*x.internal_ports) if p is not None}
def validate_plan_shape(bundle,item):
 p=item["point"];plan=build_plan(bundle["cluster"],item["model"]["path"],p,run_tag=item["run_id"]);used=plan.used_gpu_map();actual=p["metadata"]["actual_layout"]
 if set(used)!={"node3","node4"} or used!={n:v["gpus"] for n,v in actual.items()} or sum(map(len,used.values()))!=2:raise ValueError(f"invalid total-2-GPU plan {used}")
 by={}
 for x in plan.processes:
  if x.node not in {"node3","node4"}:raise ValueError(f"forbidden node {x.node}")
  if not x.gpus:continue
  role=x.metadata.get("component_role",x.role);by.setdefault(role,[]).append(x)
  for token in (LANES[p["metadata"]["lane_gpu"]],f"CUDA_VISIBLE_DEVICES={x.gpus[0]}",f"AFLEX_RUN_ID={item['run_id']}",f"AFLEX_COMPONENT_ROLE={x.role}",f"AFLEX_COMPONENT_PORT={x.port}"):
   if token not in x.command:raise ValueError(f"{x.role} missing {token}")
 expected={"native":{"NATIVE":2},"pd":{"P":1,"D":1},"af":{"F":1,"A":1}}[p["architecture"]]
 if {k:len(v) for k,v in by.items()}!=expected:raise ValueError(f"wrong components {list(by)}")
 tp=2 if p["architecture"]=="native" else 1
 if any(f"--tp {tp}" not in x.command for xs in by.values() for x in xs):raise ValueError("wrong TP")
 if p["architecture"]=="native" and sorted(x.metadata.get("node_rank") for x in by["NATIVE"])!=[0,1]:raise ValueError("Native global ranks invalid")
 return plan
def build_inventory(bundle,c=4):
 q=expand_queue(bundle);base={(x["point"]["architecture"],x["workload"]):x for x in q}
 if len(base)!=18 or any(x["qps"]!=2 for x in q):raise ValueError("expected 18 QPS2 base items")
 jobs=[];ids={};arts={};ports={};ordinal=0
 for repeat in (1,2,3):
  for wi,w in enumerate(WORKLOADS):
   for ai,a in enumerate(ARCHITECTURES):
    ordinal+=1;pair,lane=mapping(repeat,wi,ai);item,actual=configure(base[a,w],repeat,ordinal,pair,lane,[],1);plan=validate_plan_shape(bundle,item);key=logical_key(repeat,a,w);art=f"repeat{repeat}/attempt1/{item['run_id']}"
    if item["run_id"] in ids and ids[item["run_id"]]!=key:raise ValueError("run_id collision")
    if art in arts and arts[art]!=key:raise ValueError("artifact collision")
    ids[item["run_id"]]=key;arts[art]=key;lease=(pair,lane);attempt_ports={}
    for retry_attempt in range(1,FORMAL_PORT_ATTEMPTS+1):
     retry_item,_=configure(base[a,w],repeat,ordinal,pair,lane,[],retry_attempt);retry_plan=validate_plan_shape(bundle,retry_item);ps=plan_ports(retry_plan)
     if any(x in ports for x in ps):raise ValueError("derived port collision")
     ports.update({x:(key,retry_attempt) for x in ps});attempt_ports[str(retry_attempt)]=[{"host":h,"port":p} for h,p in sorted(ps)]
    jobs.append({"logical_id":key,"repeat":repeat,"architecture":a,"workload":w,"ordinal":ordinal,"pair":pair,"lane":lane,"hca":LANES[lane],"run_id_attempt1":item["run_id"],"ports":attempt_ports["1"],"port_leases":attempt_ports,"actual_layout":actual})
 if len(jobs)!=54 or len({x["logical_id"] for x in jobs})!=54:raise ValueError("logical keys must be 54 unique")
 return {"schema_version":2,"generated_at":now(),"resource_plan":resource_plan(c),"hashes":hashes(bundle,c),"jobs":jobs}
def strict_audit(path,item=None,actual=None):
 path=Path(path)
 for n in ("deployment_manifest.json","summary.json"):
  if not (path/n).is_file():return False,f"artifact_missing:{n}"
 try:s=json.loads((path/"summary.json").read_text());m=json.loads((path/"deployment_manifest.json").read_text())
 except Exception as x:return False,f"artifact_parse:{type(x).__name__}"
 if s.get("status")!="complete":
  stage=s.get("failure_stage") or "requests"
  detail=s.get("error") or s.get("status") or "failed"
  return False,f"{stage}:{detail}"
 for n in ("requests.jsonl","link_validation.json","energy.json"):
  if not (path/n).is_file():return False,f"artifact_missing:{n}"
 try:l=json.loads((path/"link_validation.json").read_text());e=json.loads((path/"energy.json").read_text());rs=[json.loads(x) for x in (path/"requests.jsonl").read_text().splitlines() if x.strip()]
 except Exception as x:return False,f"artifact_parse:{type(x).__name__}"
 if (s.get("requests_total"),s.get("requests_success"),s.get("requests_failed"))!=(64,64,0) or len(rs)!=64:return False,"request_summary"
 if any(not r.get("success") or r.get("completion_tokens")!=r.get("expected_completion_tokens") or r.get("prompt_tokens")!=r.get("expected_prompt_tokens") for r in rs):return False,"exact_input_output"
 if l.get("status")!="pass" or not l.get("physical_verified"):return False,"physical_rdma"
 rd=[x for x in l.get("paths",[]) if x.get("path")=="rdma"]
 if not rd or any(not x.get("backend_log") or float(x.get("counter_value") or 0)<=float(x.get("noise_threshold") or 0) for x in rd):return False,"rdma_evidence"
 used={}
 for x in m.get("processes",[]):
  if x.get("gpus"):used.setdefault(x["node"],set()).update(x["gpus"])
 if actual and {n:sorted(v) for n,v in used.items()}!={n:v["gpus"] for n,v in actual.items()}:return False,"manifest_topology"
 uu=e.get("gpu_uuids",{});flat=[u for v in uu.values() for u in v.values()]
 if len(flat)!=2 or len(set(flat))!=2 or any(not str(u).startswith("GPU-") for u in flat):return False,"energy_uuid_scope"
 if actual and (e.get("scope")!={n:v["gpus"] for n,v in actual.items()} or set(uu)!=set(actual)):return False,"energy_scope"
 if not isinstance(e.get("normalized",{}).get("total_j"),(int,float)) or e["normalized"]["total_j"]<=0:return False,"energy"
 return True,None
def classify_failure(reason):
 t=(reason or "").lower()
 if any(x in t for x in ("config","placement","gate","collision","forbidden","wrong components","wrong tp","exact_input_output","manifest_topology")):return "non_retryable"
 if any(x in t for x in ("startup","health timeout","ready log timeout","connection","gpu compute applications remain","address already in use","planned port still occupied","artifact_missing","requests:partial","requests:failed")):return "retryable"
 return "non_retryable"
def require_gate(results,bundle=None,c=4):
 try:g=json.loads((Path(results)/GATE_NAME).read_text())
 except Exception as e:raise RuntimeError("complete canary gate missing or invalid") from e
 if g.get("status")!="pass" or g.get("topology")!={"pairs":["pair34"],"lanes":[0,2,4,6],"total_gpus_per_job":2}:raise RuntimeError("gate topology invalid")
 a=g.get("artifacts",[])
 if len(a)!=3 or {x.get("architecture") for x in a}!=set(ARCHITECTURES) or any(x.get("status")!="pass" for x in a):raise RuntimeError("gate requires three complete passes")
 expected=hashes(bundle or load_bundle_2n(),4)
 if g.get("bindings")!=expected:raise RuntimeError(f"gate bindings changed; rerun canary: expected={expected}, actual={g.get('bindings')}")
 for x in a:
  if artifact_hash(x["artifact"])!=x.get("artifact_hash"):raise RuntimeError(f"gate artifact hash invalid: {x.get('architecture')}")
 return g
def offline_preflight(results,c=4):
 b=load_bundle_2n();inv=build_inventory(b,c);block=[];pp=Path(results)/"progress.json"
 if pp.exists():
  try:p=json.loads(pp.read_text());keys=[logical_key(x["repeat"],x["architecture"],x["workload"]) for x in p.get("runs",[])];valid=[x for x in p.get("runs",[]) if x.get("status")=="valid"]
  except Exception as e:block.append(f"progress unreadable: {e}")
  else:
   if len(keys)!=54 or len(set(keys))!=54:block.append(f"logical keys {len(keys)}/{len(set(keys))}")
   if len(valid)!=len({logical_key(x["repeat"],x["architecture"],x["workload"]) for x in valid}):block.append("duplicate valid keys")
 return {"status":"pass" if not block else "blocked","mode":"offline","ssh_performed":False,"inventory":inv,"blockers":block}
def online_preflight(results,c=4):
 out=offline_preflight(results,c);b=load_bundle_2n();ex=RemoteExecutor(b["cluster"]["container"],False);block=list(out["blockers"])
 for n in b["cluster"]["nodes"]:
  for name,cmd in (("container",f"docker inspect -f '{{{{.State.Running}}}}' {b['cluster']['container']} | grep -Fx true"),("gpus","for g in 0 2 4 6; do test -z \"$(nvidia-smi -i $g --query-compute-apps=pid --format=csv,noheader,nounits | tr -d '[:space:]')\" || exit 1; done"),("topology","nvidia-smi topo -m >/dev/null"),("runtime",f"docker exec {b['cluster']['container']} test -d {b['cluster']['python_source']}")):
   r=ex.host_run(n["host"],cmd,check=False,quiet=True)
   if r.returncode:block.append(f"{n['name']}:{name}:{(r.stderr or r.stdout or 'failed').strip()}")
 return {**out,"status":"pass" if not block else "blocked","mode":"online","ssh_performed":True,"blockers":block}
def recover_progress(results,inv,pause=True):
 pp=Path(results)/"progress.json";old=json.loads(pp.read_text()) if pp.exists() else {};previous={logical_key(x["repeat"],x["architecture"],x["workload"]):x for x in old.get("runs",[])};rows=[];bases={(x["point"]["architecture"],x["workload"]):x for x in expand_queue(load_bundle_2n())}
 for j in inv["jobs"]:
  row=previous.get(j["logical_id"],{"repeat":j["repeat"],"architecture":j["architecture"],"workload":j["workload"],"point_id":bases[j["architecture"],j["workload"]]["point"]["id"],"status":"pending","attempts":[]})
  if row.get("status") not in {"valid","blocked"}:row["status"]="pending"
  rows.append(row)
 valid=sum(x.get("status")=="valid" for x in rows);state="complete" if valid==54 else ("paused" if pause else "formal");p={**old,"schema_version":2,"state":state,"status":state,"expected_valid_runs":54,"valid_runs":valid,"pause_reason":"user_requested_resource_release" if pause and valid<54 else None,"resource_topology":inv["resource_plan"],"hashes":inv["hashes"],"runs":rows,"updated_at":now()};atomic_write(pp,p);return p
def execute_once(bundle,base,results,job,corunners,attempt,canary=False):
 item,actual=configure(base,job["repeat"],job["ordinal"],job["pair"],job["lane"],corunners,attempt,canary,hashes(bundle)["config_hash"]);validate_plan_shape(bundle,item);root=Path(results)/("canary" if canary else f"repeat{job['repeat']}")/f"attempt{attempt}";summary=execute_item(item,bundle,root);path=root/item["run_id"];ok,reason=strict_audit(path,item,actual);return item,actual,summary,path,ok,reason
def run_scheduler(a,b,inv):
 pp=a.results/"progress.json";progress=recover_progress(a.results,inv,False);rows={logical_key(x["repeat"],x["architecture"],x["workload"]):x for x in progress["runs"]};bases={(x["point"]["architecture"],x["workload"]):x for x in expand_queue(b)};jobs=[]
 for j in inv["jobs"]:
  row=rows[j["logical_id"]]
  if row["status"]=="valid" or (row["status"]=="blocked" and not a.retry_blocked):continue
  row["status"]="pending";jobs.append({**j,"base":bases[j["architecture"],j["workload"]]})
 def persist(state="formal",reason=None):
  with MUTEX:progress.update({"runs":list(rows.values()),"valid_runs":sum(x["status"]=="valid" for x in rows.values()),"state":state,"status":state,"pause_reason":reason,"updated_at":now()});atomic_write(pp,progress)
 def worker(j,corunners):
  row=rows[j["logical_id"]];start=max([int(x.get("attempt",0)) for x in row.get("attempts",[])]+[0])+1
  for attempt in range(start,start+a.max_attempts):
   if STOP.is_set():return
   try:item,actual,summary,path,ok,reason=execute_once(b,j["base"],a.results,j,corunners,attempt)
   except Exception as e:item={"run_id":digest({"key":j["logical_id"],"attempt":attempt})[:24]};actual=j["actual_layout"];summary={"status":"failed","failure_stage":"scheduler_wrapper","error":repr(e)};path=a.results/f"repeat{j['repeat']}"/f"attempt{attempt}"/item["run_id"];ok=False;reason=f"scheduler_wrapper:{e}"
   cls=None if ok else classify_failure(reason);row.setdefault("attempts",[]).append({"attempt":attempt,"run_id":item["run_id"],"artifact":str(path),"valid":ok,"reason":reason,"failure_class":cls,"failure_stage":summary.get("failure_stage"),"error":summary.get("error"),"status":summary.get("status"),"pair":j["pair"],"lane_gpu":j["lane"],"hca":j["hca"],"layout":actual,"concurrent_job_ids":corunners,"finished_at":now()});row["status"]="valid" if ok else ("retry_needed" if cls=="retryable" and attempt<start+a.max_attempts-1 else "blocked");persist()
   if ok or cls=="non_retryable":return
   time.sleep(a.backoff_base*2**(attempt-start))
 active={};reservations=[]
 def reservation(j):
  return ((j["pair"],j["lane"],j["hca"]),frozenset((x["host"],x["port"]) for xs in j["port_leases"].values() for x in xs))
 def conflicts(candidate):
  slot,ports=candidate
  return any(slot==other_slot or bool(ports&other_ports) for other_slot,other_ports in reservations)
 with ThreadPoolExecutor(max_workers=a.max_concurrency) as pool:
  while (jobs or active) and not STOP.is_set():
   for j in list(jobs):
    lease=reservation(j)
    if conflicts(lease) or len(active)>=a.max_concurrency:continue
    corunners=sorted([x[0]["logical_id"] for x in active.values()]+[j["logical_id"]]);f=pool.submit(worker,j,corunners);active[f]=(j,lease);reservations.append(lease);jobs.remove(j)
   if not active:break
   done,_=wait(active,return_when=FIRST_COMPLETED)
   for f in done:
    j,lease=active.pop(f);reservations.remove(lease)
    try:f.result()
    except BaseException as e:rows[j["logical_id"]]["status"]="blocked";rows[j["logical_id"]]["scheduler_error"]=repr(e);persist()
  if STOP.is_set():
   for f in active:f.cancel()
 valid=sum(x["status"]=="valid" for x in rows.values());state="complete" if valid==54 else "paused";persist(state,None if state=="complete" else ("signal_requested" if STOP.is_set() else "blocked_or_pending_runs"));return progress
def parse_args(argv=None):
 p=argparse.ArgumentParser(description="2N QPS2 runner; default is local dry-run with no SSH")
 p.add_argument("--results",type=Path,default=ROOT/"results/raw/measurement_rdma2n_qps2_20260825");p.add_argument("--execute",action="store_true");p.add_argument("--resume",action="store_true");p.add_argument("--retry-blocked",action="store_true");p.add_argument("--max-attempts",type=int,default=3,help="additional attempts allowed for each selected key in this resume invocation");p.add_argument("--max-concurrency",type=int,choices=(1,2,3,4),default=4);p.add_argument("--backoff-base",type=float,default=5);p.add_argument("--inventory-json",type=Path);p.add_argument("--offline-preflight",action="store_true");p.add_argument("--preflight-only",action="store_true",help="online read-only checks; requires --execute");p.add_argument("--check-gate",action="store_true");return p.parse_args(argv)
def main(argv=None):
 a=parse_args(argv);b=load_bundle_2n();inv=build_inventory(b,a.max_concurrency)
 if a.inventory_json:atomic_write(a.inventory_json,inv)
 if a.offline_preflight:r=offline_preflight(a.results,a.max_concurrency);print(json.dumps(r,indent=2));return 0 if r["status"]=="pass" else 2
 if a.preflight_only:
  if not a.execute:raise SystemExit("online --preflight-only requires --execute; use --offline-preflight for local checks")
  r=online_preflight(a.results,a.max_concurrency);print(json.dumps(r,indent=2));return 0 if r["status"]=="pass" else 2
 if a.check_gate:
  try:g=require_gate(a.results,b,a.max_concurrency);print(json.dumps({"status":"pass","gate":g},indent=2));return 0
  except Exception as e:print(json.dumps({"status":"blocked","blocker":str(e)},indent=2));return 2
 if not a.execute:p=recover_progress(a.results,inv,True);print(json.dumps({"mode":"dry-run","ssh_performed":False,"state":p["state"],"valid":p["valid_runs"],"expected":54,"inventory":inv},indent=2));return 0
 if not a.resume:raise SystemExit("--execute requires --resume; refusing side effects")
 lock=acquire(a.results,"formal");require_gate(a.results,b,a.max_concurrency);pre=online_preflight(a.results,a.max_concurrency)
 if pre["status"]!="pass":print(json.dumps(pre,indent=2));return 2
 STOP.clear();old={x:signal.getsignal(x) for x in (signal.SIGTERM,signal.SIGINT)}
 for x in old:signal.signal(x,lambda signum,frame:STOP.set())
 try:
  result=run_scheduler(a,b,inv);print(json.dumps({"state":result["state"],"valid":result["valid_runs"],"expected":54},indent=2));return 0 if result["state"]=="complete" else 3
 finally:
  for x,h in old.items():signal.signal(x,h)
if __name__=="__main__":raise SystemExit(main())

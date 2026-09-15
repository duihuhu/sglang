#!/usr/bin/env python3
import json,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]; sys.path[:0]=[str(ROOT/"src"),str(ROOT/"scripts")]
from aflex_benchmark.config import load_bundle
from aflex_benchmark.runner import execute_item,expand_queue,run_id
from run_measurement_rdma_qps4 import valid
import run_measurement_rdma_work_conserving as wc
bundle=load_bundle(ROOT/"configs",ROOT/"configs/measurement_rdma_qps4_qwen3_30b_a3b_4gpu.json"); base={(x["point"]["metadata"]["topology_id"],x["point"]["architecture"],x["workload"]):x for x in expand_queue(bundle)}; out=ROOT/"results/raw/measurement_rdma_qps4_20260825/placement_validation"; out.mkdir(parents=True,exist_ok=True); results=[]
cases=[("pair12",base["rdma2n","native","measurement_qa_lpld"],{"topology":"rdma2n","pair":"pair12","lane":1,"layout":wc.PAIR_GPUS["pair12"][1]}),("pair34",base["rdma2n","native","measurement_qa_lpld"],{"topology":"rdma2n","pair":"pair34","lane":1,"layout":wc.PAIR_GPUS["pair34"][1]}),("rdma4n",base["rdma4n","native","measurement_qa_lpld"],{"topology":"rdma4n","lane":1,"layout":wc.FOUR_GPUS[1]})]
for index,(name,item,slot) in enumerate(cases,1):
 current,actual=wc.configure(item,0,250+index,slot,[]); current["point"]["metadata"].update({"placement_validation":name,"measurement_repeat":0}); current["run_id"]=run_id(current["point"],current["workload"],current["qps"],900000+int(time.time())*10+index); wc.validate_plan_shape(bundle,current); summary=execute_item(current,bundle,out); path=out/current["run_id"]; ok,reason=valid(summary,path); ok=ok and wc.validate_artifact(path,current,actual); results.append({"case":name,"valid":ok,"reason":reason,"artifact":str(path),"layout":actual,"run_id":current["run_id"]}); (out/"placement_validation.json").write_text(json.dumps(results,indent=2)+"\n")
 if not ok: raise RuntimeError(results[-1])
print(json.dumps(results,indent=2))

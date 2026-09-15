#!/usr/bin/env python3
from __future__ import annotations
import hashlib, json, random
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/"data/workloads"
SEED=20260822
SPECS=[("measurement_qa_lpld",128,64),("measurement_chatbot_lphd",128,1024),("measurement_balanced_mpmd",512,256),("measurement_rag_hpld",4096,64),("measurement_summary_hphd",4096,1024),("measurement_longcontext",16384,256)]
def arrivals(count,qps,seed):
 rng=random.Random(seed); current=0.0; values=[]
 for _ in range(count): current+=rng.expovariate(qps); values.append(current)
 return values
def main():
 index_path=OUT/"index.json"; index=json.loads(index_path.read_text()) if index_path.exists() else {"entries": []}; entries=[]
 for name,input_len,output_len in SPECS:
  path=OUT/f"{name}_qps4.jsonl"; values=arrivals(64,4.0,SEED+input_len+output_len+4)
  rows=[{"request_id":f"{name}-4-{i:05d}","arrival_time_s":round(value,6),"input_len":input_len,"output_len":output_len,"source":name} for i,value in enumerate(values)]
  path.write_text("".join(json.dumps(row,sort_keys=True)+"\n" for row in rows))
  entries.append({"name":name,"qps":4,"path":path.name,"requests":64,"sha256":hashlib.sha256(path.read_bytes()).hexdigest(),"provenance":"generated_measurement_fixed_poisson","seed":SEED,"input_len":input_len,"output_len":output_len,"measurement":"RQ2"})
 index_path.write_text(json.dumps({"version":1,"entries":entries},indent=2)+"\n")
 print(json.dumps(entries[-6:],indent=2))
if __name__=="__main__": main()

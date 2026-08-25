import numpy as np,csv,json,torch,argparse
from pathlib import Path
from sglang.srt.eplb.eplb_algorithms.deepseek import rebalance_experts
a=argparse.ArgumentParser();a.add_argument('--trace',type=Path);a.add_argument('--output',type=Path);x=a.parse_args();trace=json.loads(x.trace.read_text());counts=np.zeros((48,128),dtype=np.int64)
for s in trace['samples']:
 z=np.array(s['experts']);start=max(0,s['prompt_tokens']-1);dec=z[start:start+s['completion_tokens']]
 for l in range(48):counts[l]+=np.bincount(dec[:,l].ravel(),minlength=128)
rows=[]
for redundant in (8,16,32,64):
 phy,_,_=rebalance_experts(torch.tensor(counts),128+redundant,1,1,8,False);epr=(128+redundant)//8;loads=np.zeros((48,8),float)
 for l in range(48):
  for e in range(128):
   reps=np.where(phy[l].numpy()==e)[0];share=counts[l,e]/len(reps)
   for pp in reps:loads[l,pp//epr]+=share
 before=np.zeros((48,8),float)
 for l in range(48):
  for e in range(128):before[l,e//16]+=counts[l,e]
 rows.append({'redundant_experts':redundant,'memory_overhead_pct':redundant/128*100,'before_mean_max_over_mean':np.mean(before.max(1)/before.mean(1)),'after_mean_max_over_mean':np.mean(loads.max(1)/loads.mean(1)),'after_p95_max_over_mean':np.quantile(loads.max(1)/loads.mean(1),.95)})
x.output.parent.mkdir(parents=True,exist_ok=True)
with x.output.open('w',newline='') as f:w=csv.DictWriter(f,rows[0],delimiter='\t');w.writeheader();w.writerows(rows)
print(rows)

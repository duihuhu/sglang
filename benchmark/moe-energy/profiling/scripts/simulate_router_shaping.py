#!/usr/bin/env python3
"""Offline quality-budget rank-capacity router shaping on captured routed IDs."""
from __future__ import annotations
import argparse,csv,json,time
from pathlib import Path
import numpy as np
def shape(ids,topc,experts_per_rank,envelope,move_budget):
 # ids [topc] are ordered router candidates. Keep K=8 unique choices; replace tail assignments only.
 chosen=list(ids[:8]);loads=np.bincount(np.array(chosen)//experts_per_rank,minlength=8);target=int(np.ceil(8/8*envelope));moves=0;score_loss=0
 # Per-token local operation cannot know batch loads; caller wraps global greedy. Return candidates.
 return chosen
def run_layer(cands,budget,envelope):
 # cands [tokens, C], initial first8. Greedy over lowest-priority slots (7..0), move only to candidate on underloaded rank.
 T,C=cands.shape;sel=cands[:,:8].copy();loads=np.bincount(sel.ravel()//16,minlength=8);ideal=sel.size/8;upper=ideal*(1+envelope);moves=[];t0=time.perf_counter_ns()
 for tok in range(T):
  if len(moves)>=budget*sel.size:break
  for slot in range(7,-1,-1):
   src=sel[tok,slot]//16
   if loads[src]<=upper:continue
   used=set(sel[tok]);replacement=None;rank=None;candidate_rankpos=None
   for pos,e in enumerate(cands[tok,8:],start=8):
    rr=e//16
    if e not in used and loads[rr]<ideal:
     replacement=e;rank=rr;candidate_rankpos=pos;break
   if replacement is not None:
    old=sel[tok,slot];sel[tok,slot]=replacement;loads[src]-=1;loads[rank]+=1;moves.append((tok,slot,int(old),int(replacement),candidate_rankpos));break
 elapsed=(time.perf_counter_ns()-t0)/1000
 return sel,loads,moves,elapsed
def main():
 a=argparse.ArgumentParser();a.add_argument('--trace',type=Path,required=True);a.add_argument('--output',type=Path,required=True);x=a.parse_args();d=json.loads(x.trace.read_text());rows=[]
 # Existing capture has only Top8. Synthesize Top-C rank alternatives by nearest expert IDs as a lower-bound scheduling study; mark as surrogate.
 for req,s in enumerate(d['samples']):
  arr=np.array(s['experts']);start=max(0,s['prompt_tokens']-1);dec=arr[start:start+s['completion_tokens']]
  for it in range(len(dec)):
   for layer in range(48):
    ids=dec[it,layer];alts=[]
    for e in ids:
     alts.extend([(int(e)+16*j)%128 for j in range(1,8)])
    c=np.array([list(map(int,ids))+alts[:8]])
    before=np.bincount(ids//16,minlength=8)
    for envelope in (.05,.10,.20):
     for budget in (.05,.10,.20):
      sel,after,moves,us=run_layer(c,budget,envelope);rows.append({'request':req,'iteration':it,'layer':layer,'envelope':envelope,'move_budget':budget,'before_max_over_mean':before.max()/before.mean(),'after_max_over_mean':after.max()/after.mean(),'moved_assignments':len(moves),'move_fraction':len(moves)/8,'scheduler_us':us,'candidate_source':'surrogate_rank_alternatives_not_router_logits'})
 x.output.parent.mkdir(parents=True,exist_ok=True)
 with x.output.open('w',newline='') as f:w=csv.DictWriter(f,rows[0],delimiter='\t');w.writeheader();w.writerows(rows)
 print(x.output,len(rows))
if __name__=='__main__':main()

#!/usr/bin/env python3
from __future__ import annotations
import argparse,csv,json,math
from pathlib import Path
import numpy as np
def jsd(a,b):
 a=a/a.sum() if a.sum() else np.ones_like(a)/len(a);b=b/b.sum() if b.sum() else np.ones_like(b)/len(b);m=(a+b)/2
 def kl(x,y):z=x>0;return np.sum(x[z]*np.log2(x[z]/y[z]))
 return .5*kl(a,m)+.5*kl(b,m)
def main():
 a=argparse.ArgumentParser();a.add_argument('--raw',type=Path);a.add_argument('--out',type=Path);x=a.parse_args();x.out.mkdir(parents=True,exist_ok=True);meta=[json.loads(z) for z in (x.raw/'requests.jsonl').open()];arrays={}
 for p in x.raw.glob('*_chunk_*.npz'):
  with np.load(p) as z:
   for k in z.files:arrays[k]=z[k]
 # trivial Qwen3 EP8 layout
 with (x.out/'expert_layout.tsv').open('w',newline='') as f:
  fs=['layer','expert_id','ep_rank','local_expert_id'];w=csv.DictWriter(f,fs,delimiter='\t');w.writeheader();w.writerows({'layer':l,'expert_id':e,'ep_rank':e//16,'local_expert_id':e%16} for l in range(48) for e in range(128))
 layer_rows=[];decode_rows=[];reqsig={};reqrank={}
 for order in ('original','shuffled'):
  ms=sorted((m for m in meta if m['order']==order),key=lambda m:m['submit_index']);ex=np.zeros((48,128),np.int64);px=np.zeros((48,128),np.int64);dx=np.zeros((48,128),np.int64)
  for m in ms:
   ar=arrays[m['key']];prompt_rows=max(0,m['prompt_tokens']-1);pre=ar[:prompt_rows];dec=ar[prompt_rows:prompt_rows+m['completion_tokens']]
   for l in range(48):
    if len(pre):px[l]+=np.bincount(pre[:,l].ravel(),minlength=128)
    if len(dec):dx[l]+=np.bincount(dec[:,l].ravel(),minlength=128)
   # request decode signature layer×rank
   sig=np.zeros((48,128),np.int32);ranks=np.zeros((48,8),np.int32)
   for l in range(48):
    sig[l]=np.bincount(dec[:,l].ravel(),minlength=128);ranks[l]=np.bincount((dec[:,l].ravel()//16),minlength=8)
   reqsig[m['key']]=sig;reqrank[m['key']]=ranks
   for t in range(len(dec)):
    for l in range(48):
     c=np.bincount(dec[t,l]//16,minlength=8);decode_rows.append({'order':order,'submit_index':m['submit_index'],'dataset_index':m['dataset_index'],'decode_iteration':t,'layer':l,**{f'rank{r}_assignments':int(c[r]) for r in range(8)},'active_experts':int(len(np.unique(dec[t,l]))),'hot_rank':int(c.argmax()),'max_over_mean':float(c.max()/c.mean())})
  for l in range(48):
   for phase,c in [('prefill',px[l]),('decode',dx[l])]:
    rc=np.add.reduceat(c,np.arange(0,128,16));p=c/c.sum();rp=rc/rc.sum();ent=lambda z:-np.sum(z[z>0]*np.log2(z[z>0]));layer_rows.append({'order':order,'phase':phase,'layer':l,'total_assignments':int(c.sum()),'active_experts':int(np.count_nonzero(c)),'expert_entropy_bits':float(ent(p)),'expert_entropy_normalized':float(ent(p)/7),'expert_gini':float(np.abs(c[:,None]-c[None,:]).sum()/(2*128*c.sum())),'hot_expert':int(c.argmax()),'hot_expert_fraction':float(c.max()/c.sum()),'rank_entropy_normalized':float(ent(rp)/3),'rank_max_over_mean':float(rc.max()/rc.mean()),'hot_rank':int(rc.argmax())})
 # Request lag similarity, original vs shuffled. Similarity is 1-JSD on layer-aggregated expert and rank histograms.
 pair=[]
 for order in ('original','shuffled'):
  ms=sorted((m for m in meta if m['order']==order),key=lambda m:m['submit_index'])
  for lag in (1,2,4,8,16,32,64):
   vals=[];rvals=[];over=[]
   for i in range(len(ms)-lag):
    aa=reqsig[ms[i]['key']].sum(0);bb=reqsig[ms[i+lag]['key']].sum(0);vals.append(1-jsd(aa.astype(float),bb.astype(float)));ra=reqrank[ms[i]['key']].sum(0);rb=reqrank[ms[i+lag]['key']].sum(0);rvals.append(1-jsd(ra.astype(float),rb.astype(float)));over.append(len(set(np.flatnonzero(aa))&set(np.flatnonzero(bb)))/max(1,len(set(np.flatnonzero(aa))|set(np.flatnonzero(bb)))))
   pair.append({'order':order,'request_lag':lag,'num_pairs':len(vals),'expert_js_similarity_mean':float(np.mean(vals)),'rank_js_similarity_mean':float(np.mean(rvals)),'active_expert_jaccard_mean':float(np.mean(over))})
 for name,rows in [('layer_hotness.tsv',layer_rows),('decode_iteration_rank_load.tsv',decode_rows),('request_lag_similarity.tsv',pair)]:
  with (x.out/name).open('w',newline='') as f:w=csv.DictWriter(f,rows[0],delimiter='\t');w.writeheader();w.writerows(rows)
 # Coactivation matrix per layer across all original decode observations.
 co=np.zeros((48,128,128),np.int32)
 for m in (z for z in meta if z['order']=='original'):
  ar=arrays[m['key']];start=max(0,m['prompt_tokens']-1);dec=ar[start:start+m['completion_tokens']]
  for l in range(48):
   for row in dec[:,l]:co[l][np.ix_(row,row)]+=1
 np.savez_compressed(x.out/'expert_coactivation.npz',coactivation=co)
 print('requests',len(meta),'arrays',len(arrays),'layer_rows',len(layer_rows),'decode_rows',len(decode_rows),'pairs',len(pair))
if __name__=='__main__':main()

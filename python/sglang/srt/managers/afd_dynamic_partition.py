"""Causal AFD partitioning from the previous Decode iteration's real routes."""
from __future__ import annotations
import json,os,tempfile,threading,time
from pathlib import Path
import numpy as np
import torch
_LOCK=threading.Lock();_ROUTES={}
def enabled():return bool(os.environ.get('AFD_DYNAMIC_PARTITION_FILE'))
def record(layer:int,mb:int,topk_ids:torch.Tensor):
 if not enabled() or mb not in (0,1):return
 with _LOCK:_ROUTES[(int(layer),int(mb))]=topk_ids.detach().clone()
def _counts(ids,ep):
 x=ids.to(torch.int64).cpu().numpy();width=128//ep;return np.stack([np.bincount(row//width,minlength=ep)[:ep] for row in x])
def _cost(loads):return float(loads.max(axis=2).sum())
def _balanced(sig):
 n=len(sig);caps=((n+1)//2,n//2);bins=[[],[]];loads=np.zeros((2,*sig.shape[1:]),dtype=np.int64);order=np.argsort(-sig.max(axis=(1,2)),kind='stable')
 for idx in order:
  cand=[]
  for b in (0,1):
   if len(bins[b])>=caps[b]:continue
   trial=loads.copy();trial[b]+=sig[idx];score=(_cost(trial),abs(_cost(trial[0:1])-_cost(trial[1:2])),len(bins[b]),b);cand.append((score,b))
  b=min(cand)[1];bins[b].append(int(idx));loads[b]+=sig[idx]
 return bins,loads
def _skewed(sig):
 n=len(sig);half=n//2;best=None
 for l in range(sig.shape[1]):
  for r in range(sig.shape[2]):
   order=np.argsort(-sig[:,l,r],kind='stable');bins=[order[:half].tolist(),order[half:].tolist()];loads=np.stack([sig[bins[0]].sum(0),sig[bins[1]].sum(0)]);gap=abs(_cost(loads[0:1])-_cost(loads[1:2]));score=(-gap,_cost(loads))
   if best is None or score<best[0]:best=(score,bins,loads)
 return best[1],best[2]
def finalize(req_ids:list[str],split:int,ep:int):
 if not enabled() or len(req_ids)<2:return None
 with _LOCK:
  routes=dict(_ROUTES);_ROUTES.clear()
 if len(routes)<96:return None
 parts=[]
 for mb,(lo,hi) in enumerate(((0,split),(split,len(req_ids)))):
  layers=[]
  for l in range(48):
   ids=routes.get((l,mb))
   if ids is None:return None
   layers.append(_counts(ids,ep))
  parts.append(np.stack(layers,axis=1))
 sig=np.concatenate(parts,axis=0)
 if len(sig)!=len(req_ids):return None
 mode=os.environ.get('AFD_DYNAMIC_PARTITION_MODE','balanced');bins,loads=(_balanced(sig) if mode=='balanced' else _skewed(sig));mapping={req_ids[i]:b for b in (0,1) for i in bins[b]};costs=[_cost(loads[b:b+1]) for b in (0,1)];payload={'schema_version':1,'mode':mode,'generated_at':time.time(),'requests':len(req_ids),'groups':mapping,'critical_costs':costs,'gap_pct':100*abs(costs[0]-costs[1])/max((costs[0]+costs[1])/2,1e-9)};path=Path(os.environ['AFD_DYNAMIC_PARTITION_FILE']);path.parent.mkdir(parents=True,exist_ok=True);fd,tmp=tempfile.mkstemp(prefix=path.name+'.',dir=path.parent)
 with os.fdopen(fd,'w') as f:json.dump(payload,f);f.flush()
 os.replace(tmp,path)
 audit=os.environ.get('AFD_DYNAMIC_PARTITION_AUDIT')
 if audit:
  with open(audit,'a') as f:f.write(json.dumps({k:v for k,v in payload.items() if k!='groups'})+'\n')
 return payload
def load_mapping():
 path=os.environ.get('AFD_DYNAMIC_PARTITION_FILE')
 if not path or not Path(path).exists():return None
 try:return json.loads(Path(path).read_text())
 except (OSError,json.JSONDecodeError):return None

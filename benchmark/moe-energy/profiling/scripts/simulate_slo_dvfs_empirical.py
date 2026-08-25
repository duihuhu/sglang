#!/usr/bin/env python3
import argparse,csv,itertools
from pathlib import Path
import numpy as np
a=argparse.ArgumentParser();a.add_argument('--loads',type=Path);a.add_argument('--output',type=Path);x=a.parse_args();X=np.load(x.loads).sum(axis=0);freqs=np.array([210,690,930,1410]);latrat=np.array([2.2688207396,1.1058176282,1.0387691029,1.0]);powrat=np.array([.5446089555,.6610302387,.7018858277,1.0]);sfs=np.array([1,1.02,1.05,1.1,1.2,1.35,1.5,1.75,2.0]);rows=[]
def base_time(load):return 180+4.0*load
def layer_best(load,sf):
 bt=base_time(load);bound=bt.max()*sf;best=None
 # Enumerate candidate makespan induced by each rank/frequency; each rank selects lowest-power frequency meeting it.
 for target in np.unique((bt[:,None]*latrat).ravel()):
  if target>bound+1e-9:continue
  fi=[]
  for r in range(8):
   ok=np.where(bt[r]*latrat<=target+1e-9)[0]
   if len(ok)==0:break
   fi.append(ok[np.argmin(powrat[ok])])
  if len(fi)<8:continue
  m=max(bt[r]*latrat[fi[r]] for r in range(8));e=m*sum((58+1.2*load[r])*powrat[fi[r]] for r in range(8))/1e6
  if best is None or e<best[0]:best=(e,m,np.array(fi))
 return best
def evaluate_static(loads,fi):
 bt=base_time(loads);t=bt*latrat[fi][None,:];m=t.max(1);p=(58+1.2*loads)*powrat[fi][None,:];return m.sum(),(m*p.sum(1)/1e6).sum()
def static_beam(loads,slo,beam=256):
 states=[(0.0,np.full(8,3,dtype=np.int8))]
 # iterative neighbor expansion, retain energy-best feasible + latency-diverse
 seen=set()
 for _ in range(24):
  cand=[]
  for _,fi in states:
   key=tuple(fi)
   if key not in seen:
    seen.add(key);t,e=evaluate_static(loads,fi)
    if t<=slo+1e-9:cand.append((e,t,fi.copy()))
   for r in range(8):
    if fi[r]>0:
     z=fi.copy();z[r]-=1;k=tuple(z)
     if k not in seen:
      t,e=evaluate_static(loads,z)
      if t<=slo+1e-9:cand.append((e,t,z))
  if not cand:break
  cand.sort(key=lambda q:q[0]);states=[(e,fi) for e,t,fi in cand[:beam]]
 return min(((*evaluate_static(loads,fi),fi) for _,fi in states),key=lambda q:q[1])
for sf in sfs:
 acc={k:[] for k in ('uniform','static_per_rank','layer_oracle')}
 for loads in X:
  bt=base_time(loads);base_m=bt.max(1);base_p=(58+1.2*loads);lmin=base_m.sum();emin=(base_m*base_p.sum(1)/1e6).sum();slo=lmin*sf
  # uniform enumerate
  uv=[]
  for i in range(4):
   fi=np.full(8,i);t,e=evaluate_static(loads,fi)
   if t<=slo+1e-9:uv.append((e,t,fi))
  ue,ut,ufi=min(uv)
  st,se,sfi=static_beam(loads,slo)
  le=lt=0
  for l in range(48):e,m,fi=layer_best(loads[l],sf);le+=e;lt+=m
  for name,t,e in [('uniform',ut,ue),('static_per_rank',st,se),('layer_oracle',lt,le)]:acc[name].append((t,e,lmin,emin))
 for name,v in acc.items():
  z=np.array(v);rows.append({'slo_factor':sf,'strategy':name,'latency_ratio_mean':np.mean(z[:,0]/z[:,2]),'latency_ratio_p99':np.quantile(z[:,0]/z[:,2],.99),'energy_saving_vs_all1410_pct':np.mean(1-z[:,1]/z[:,3])*100,'slo_miss_rate_pct':np.mean(z[:,0]>z[:,2]*sf+1e-9)*100})
x.output.parent.mkdir(parents=True,exist_ok=True)
with x.output.open('w',newline='') as f:w=csv.DictWriter(f,rows[0],delimiter='\t');w.writeheader();w.writerows(rows)
print(rows)

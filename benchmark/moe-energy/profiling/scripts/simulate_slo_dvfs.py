#!/usr/bin/env python3
import argparse,csv,itertools,json
from pathlib import Path
import numpy as np
a=argparse.ArgumentParser();a.add_argument('--loads',type=Path);a.add_argument('--output',type=Path);x=a.parse_args();X=np.load(x.loads).sum(axis=0);freqs=np.array([210,450,690,930,1170,1410]);sfs=np.array([1,1.02,1.05,1.1,1.2,1.35,1.5,1.75,2.0]);F=np.array(list(itertools.product(freqs,repeat=8)),dtype=np.int16);rows=[]
def time(load,f):return 180+5*load*(930/f)**.82
def power(load,f):return 72+1.6*load*(f/930)**.42
def evaluate(loads,fs):
 t=time(loads,fs);m=t.max(-1);e=m*power(loads,fs).sum(-1)/1e6;return m,e
for sf in sfs:
 agg={k:[] for k in ('uniform','static_per_rank','layer_oracle')}
 for loads in X:
  base_m,base_e=evaluate(loads,np.full((48,8),1410));lmin=base_m.sum();emin=base_e.sum();slo=lmin*sf
  # uniform
  uv=[]
  for f in freqs:
   m,e=evaluate(loads,np.full((48,8),f));uv.append((e.sum(),m.sum(),f))
  ue,ut,uf=min((z for z in uv if z[1]<=slo),default=uv[-1]);agg['uniform'].append((ut,ue,lmin,emin))
  # static per rank, chunk vectorized across 6^8 candidates
  best=(1e30,None,None)
  for st in range(0,len(F),32768):
   fc=F[st:st+32768].astype(float);t=time(loads[None,:,:],fc[:,None,:]);m=t.max(-1).sum(-1);e=(t.max(-1)*power(loads[None,:,:],fc[:,None,:]).sum(-1)/1e6).sum(-1);ok=m<=slo
   if ok.any():ii=np.where(ok)[0][np.argmin(e[ok])];
   else:continue
   if e[ii]<best[0]:best=(e[ii],m[ii],fc[ii].copy())
  agg['static_per_rank'].append((best[1],best[0],lmin,emin))
  # layer oracle: per layer choose minimum energy option constrained by allocated proportional slack; exact per-layer bound sf*base layer time.
  lt=le=0
  for l in range(48):
   t=time(loads[l][None,:],F.astype(float));m=t.max(-1);e=m*power(loads[l][None,:],F.astype(float)).sum(-1)/1e6;bound=base_m[l]*sf;ok=m<=bound;ii=np.where(ok)[0][np.argmin(e[ok])];lt+=m[ii];le+=e[ii]
  agg['layer_oracle'].append((lt,le,lmin,emin))
 for name,v in agg.items():
  z=np.array(v);rows.append({'slo_factor':sf,'strategy':name,'latency_ratio_mean':np.mean(z[:,0]/z[:,2]),'latency_ratio_p99':np.quantile(z[:,0]/z[:,2],.99),'energy_saving_vs_all1410_pct':np.mean(1-z[:,1]/z[:,3])*100,'slo_miss_rate_pct':np.mean(z[:,0]>z[:,2]*sf+1e-6)*100})
x.output.parent.mkdir(parents=True,exist_ok=True)
with x.output.open('w',newline='') as f:w=csv.DictWriter(f,rows[0],delimiter='\t');w.writeheader();w.writerows(rows)
print(rows)

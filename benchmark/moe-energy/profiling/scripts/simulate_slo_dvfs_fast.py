#!/usr/bin/env python3
import argparse,csv
from pathlib import Path
import numpy as np
a=argparse.ArgumentParser();a.add_argument('--loads',type=Path);a.add_argument('--output',type=Path);x=a.parse_args();X=np.load(x.loads).sum(axis=0);freqs=np.array([210,450,690,930,1170,1410]);sfs=np.array([1,1.02,1.05,1.1,1.2,1.35,1.5,1.75,2.0])
def tm(load,f):return 180+5*load*(930/f)**.82
def pw(load,f):return 58+1.35*load*(f/930)**.48
def ev(loads,fs):
 fs=np.asarray(fs);fs=np.full(8,float(fs)) if fs.ndim==0 else fs
 t=tm(loads,fs if fs.ndim==2 else fs[None,:]);m=t.max(1);e=m*pw(loads,fs if fs.ndim==2 else fs[None,:]).sum(1)/1e6;return m.sum(),e.sum()
def layer_oracle(loads,sf):
 base=tm(loads,1410);bound=base.max(1)*sf;fs=np.empty((48,8),int)
 for l in range(48):
  for r in range(8):
   valid=freqs[tm(loads[l,r],freqs)<=bound[l]];fs[l,r]=valid.min() if len(valid) else 1410
 return (*ev(loads,fs),fs)
def uniform(loads,slo):
 z=[(*ev(loads,f),f) for f in freqs];return min((q for q in z if q[0]<=slo),key=lambda q:q[1])
def static_cd(loads,slo):
 fs=np.full(8,1410);changed=True
 while changed:
  changed=False
  for r in range(8):
   cur=ev(loads,fs);best=(cur[1],fs[r])
   for f in freqs:
    c=fs.copy();c[r]=f;t,e=ev(loads,c)
    if t<=slo+1e-9 and e<best[0]:best=(e,f)
   if best[1]!=fs[r]:fs[r]=best[1];changed=True
 return (*ev(loads,fs),fs)
rows=[];details=[]
for sf in sfs:
 acc={k:[] for k in ('uniform','static_per_rank','layer_oracle')}
 for it,loads in enumerate(X):
  lmin,emin=ev(loads,np.full(8,1410));slo=lmin*sf;ut,ue,uf=uniform(loads,slo);st,se,sfsx=static_cd(loads,slo);lt,le,lfs=layer_oracle(loads,sf)
  for name,t,e,fsx in [('uniform',ut,ue,np.full(8,uf)),('static_per_rank',st,se,sfsx),('layer_oracle',lt,le,lfs)]:acc[name].append((t,e,lmin,emin));details.append({'iteration':it,'slo_factor':sf,'strategy':name,'latency_ratio':t/lmin,'energy_saving_pct':(1-e/emin)*100,'frequency_vector':','.join(map(str,fsx.tolist())) if fsx.ndim==1 else 'layerwise'})
 for name,v in acc.items():
  z=np.array(v);rows.append({'slo_factor':sf,'strategy':name,'latency_ratio_mean':np.mean(z[:,0]/z[:,2]),'latency_ratio_p99':np.quantile(z[:,0]/z[:,2],.99),'energy_saving_vs_all1410_pct':np.mean(1-z[:,1]/z[:,3])*100,'slo_miss_rate_pct':np.mean(z[:,0]>z[:,2]*sf+1e-9)*100})
x.output.parent.mkdir(parents=True,exist_ok=True)
with x.output.open('w',newline='') as f:w=csv.DictWriter(f,rows[0],delimiter='\t');w.writeheader();w.writerows(rows)
with x.output.with_name('slo_sweep_details.tsv').open('w',newline='') as f:w=csv.DictWriter(f,details[0],delimiter='\t');w.writeheader();w.writerows(details)
print(rows)

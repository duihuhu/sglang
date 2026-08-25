#!/usr/bin/env python3
import argparse,csv,json,os,tempfile
from pathlib import Path
def main():
 p=argparse.ArgumentParser();p.add_argument('--raw-root',type=Path,required=True);p.add_argument('--output-dir',type=Path,required=True);a=p.parse_args();rows={}
 for path in sorted(a.raw_root.glob('node*/*.jsonl')):
  for line in path.open():
   r=json.loads(line);m=r.get('routing',{}).get('forced_mode','')
   if r.get('status')!='ok' or not m.startswith('hot_rank_'):continue
   k=(r['phase'],m,int(r['length']),int(r['batch']),int(r['freq_mhz']));rows.setdefault(k,r)
 a.output_dir.mkdir(parents=True,exist_ok=True);out=a.output_dir/'hot-EP.tsv';fd,tmp=tempfile.mkstemp(dir=a.output_dir,text=True)
 with os.fdopen(fd,'w',newline='') as f:
  fields=['phase','routing','hot_rank_fraction','ep_size','input_or_context_len','batch_size','gpu_clock','latency_us','energy_mj','energy_per_rank_mj','latency_per_rank_us'];w=csv.DictWriter(f,fields,delimiter='\t',lineterminator='\n');w.writeheader()
  for k,r in sorted(rows.items()):
   frac=int(k[1].rsplit('_',1)[1])/1000
   w.writerow({'phase':k[0],'routing':k[1],'hot_rank_fraction':frac,'ep_size':r['moe_ep'],'input_or_context_len':k[2],'batch_size':k[3],'gpu_clock':k[4],'latency_us':r['latency_us'],'energy_mj':r['energy_total_mj'],'energy_per_rank_mj':json.dumps(r['energy_per_rank_mj']),'latency_per_rank_us':json.dumps(r['latency_per_rank_us'])})
 os.replace(tmp,out);print(f'Wrote {out} rows={len(rows)}')
if __name__=='__main__':main()

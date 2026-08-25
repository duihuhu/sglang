#!/usr/bin/env python3
import argparse,csv,json,os,re,tempfile
from pathlib import Path
def main():
 p=argparse.ArgumentParser();p.add_argument('--raw-root',type=Path,required=True);p.add_argument('--output-dir',type=Path,required=True);a=p.parse_args();rows={}
 for path in a.raw_root.glob('node*/*.jsonl'):
  m=re.search(r'_a(\d+)\.jsonl$',path.name);n=int(m.group(1))
  for line in path.open():
   r=json.loads(line);k=(r['phase'],n,int(r['length']),int(r['batch']));rows.setdefault(k,r)
 out=a.output_dir/'hot-expert.tsv';fd,tmp=tempfile.mkstemp(dir=a.output_dir,text=True)
 with os.fdopen(fd,'w',newline='') as f:
  fs=['phase','hot_rank_fraction','active_experts_per_rank','input_or_context_len','batch_size','gpu_clock','latency_us','energy_mj','energy_per_rank_mj','latency_per_rank_us'];w=csv.DictWriter(f,fs,delimiter='\t',lineterminator='\n');w.writeheader()
  for k,r in sorted(rows.items()):w.writerow({'phase':k[0],'hot_rank_fraction':.75,'active_experts_per_rank':k[1],'input_or_context_len':k[2],'batch_size':k[3],'gpu_clock':r['freq_mhz'],'latency_us':r['latency_us'],'energy_mj':r['energy_total_mj'],'energy_per_rank_mj':json.dumps(r['energy_per_rank_mj']),'latency_per_rank_us':json.dumps(r['latency_per_rank_us'])})
 os.replace(tmp,out);print(f'Wrote {out} rows={len(rows)}')
if __name__=='__main__':main()

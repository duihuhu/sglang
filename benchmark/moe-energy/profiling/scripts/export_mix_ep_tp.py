#!/usr/bin/env python3
"""Export mixed EP+TP full-FFN JSONL rows to one TSV."""
import argparse,csv,json,tempfile,os
from pathlib import Path

def main():
 p=argparse.ArgumentParser();p.add_argument('--raw-root',type=Path,required=True);p.add_argument('--output-dir',type=Path,required=True);a=p.parse_args()
 rows={};read=0;dups=0
 for path in sorted(a.raw_root.glob('node*/*.jsonl')):
  for line in path.open():
   if not line.strip():continue
   read+=1;r=json.loads(line)
   if r.get('status')!='ok' or r.get('component')!='F':continue
   mode=r.get('routing',{}).get('forced_mode')
   if mode not in ('balanced','skewed_rank0'):continue
   k=(r['phase'],mode,int(r['world_size']),int(r['moe_ep']),int(r['moe_tp']),int(r['length']),int(r['batch']),int(r['freq_mhz']))
   if k in rows:dups+=1;continue
   rows[k]=r
 a.output_dir.mkdir(parents=True,exist_ok=True);out=a.output_dir/'mix-EP-TP.tsv'
 fields=['phase','routing','world_size','ep_size','moe_tp_size','input_or_context_len','batch_size','gpu_clock','latency_us','energy_mj']
 fd,tmp=tempfile.mkstemp(dir=a.output_dir,prefix='.mix-',text=True)
 with os.fdopen(fd,'w',newline='') as f:
  w=csv.DictWriter(f,fields,delimiter='\t',lineterminator='\n');w.writeheader()
  for k,r in sorted(rows.items()):
   w.writerow(dict(zip(fields,[k[0],k[1],k[2],k[3],k[4],k[5],k[6],k[7],r['latency_us'],r['energy_total_mj']])))
 os.replace(tmp,out);print(f'Wrote {out} rows={len(rows)} read={read} duplicates={dups}')
if __name__=='__main__':main()

#!/usr/bin/env python3
import argparse,csv,json,sys
from pathlib import Path
HERE=Path(__file__).resolve().parent; ROOT=HERE.parents[0]; sys.path.insert(0,str(HERE))
from rq1lib import aggregate,atomic_json,canary_gate,load

def main():
 p=argparse.ArgumentParser(); p.add_argument('--results',type=Path,default=ROOT/'results/default'); a=p.parse_args()
 progress=load(a.results/'progress.json'); groups=aggregate(progress); rows=[]
 for g in groups:
  row={k:g[k] for k in ('model','architecture','workload','qps','repeats')}
  for metric,value in g['metrics'].items(): row[metric+'_mean']=value['mean']; row[metric+'_sample_std']=value['sample_std']
  rows.append(row)
 report={'schema_version':1,'rq':'RQ1','canary_gate':canary_gate(progress),'complete':len(groups)==432 and all(g['repeats']==3 for g in groups),'expected_groups':432,'groups':groups}
 atomic_json(a.results/'rq1_report.json',report)
 if rows:
  fields=sorted(set().union(*(r.keys() for r in rows))); f=(a.results/'rq1_report.csv').open('w',newline=''); w=csv.DictWriter(f,fields); w.writeheader(); w.writerows(rows); f.close()
 lines=['# RQ1 model × dataset × architecture energy report','',f"Complete: {report['complete']}",f"Groups: {len(groups)}/432",f"Canary: {report['canary_gate']['valid']}/18",'', 'JSON/CSV include achieved QPS, input/output token throughput, energy per token/request, tokens/J, power, success rate, and TTFT/TPOT/E2E p90 SLA metrics.']
 (a.results/'rq1_report.md').write_text('\n'.join(lines)+'\n'); print(json.dumps({'groups':len(groups),'complete':report['complete'],'output':str(a.results)},indent=2))
if __name__=='__main__': main()

#!/usr/bin/env python3
import argparse,json
from collections import defaultdict
from pathlib import Path
def main():
 p=argparse.ArgumentParser();p.add_argument('--report',type=Path,default=Path(__file__).resolve().parents[1]/'results/default/rq1_report.json');p.add_argument('--output',type=Path,default=Path(__file__).resolve().parents[1]/'analysis/energy_sla_summary.json');a=p.parse_args();r=json.loads(a.report.read_text());groups=r.get('groups',[]);by=defaultdict(list)
 for g in groups:
  key=(g['model'],g['architecture'],g['workload']);m=g.get('metrics',{});by[key].append({'qps':g['qps'],'repeats':g['repeats'],'metrics':m})
 rows=[]
 for key,points in sorted(by.items()):
  valid=[x for x in points if x['repeats']==3 and x['metrics'].get('success_rate',{}).get('mean',0)>=0.99 and x['metrics'].get('achieved_qps_ratio',{}).get('mean',0)>=0.9]
  rows.append({'model':key[0],'architecture':key[1],'workload':key[2],'sla_valid_qps':[x['qps'] for x in valid],'max_sustainable_qps':max((x['qps'] for x in valid),default=None),'points':points})
 a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps({'schema_version':1,'rq':'RQ1','complete':r.get('complete',False),'groups':rows},indent=2)+'\n');print(a.output)
if __name__=='__main__':main()
